"""Build the requested Adaptive-K progression, confusion and correlation report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "research" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from adaptive_k_optimization_core import load_frozen_data  # noqa: E402
from compute_adaptive_k_full_metrics import policy_metrics  # noqa: E402

WORK = ROOT / "data/work/xgboost_adaptive_k_optimization_20261016"
OUT = WORK / "progression"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def choices_from_csv(path: Path, qid_position: dict[str, int]) -> np.ndarray:
    choices = np.zeros(len(qid_position), dtype=int)
    for row in read_csv(path):
        choices[qid_position[row["qid"]]] = int(row["chosen_k"])
    if np.any(choices == 0):
        raise ValueError(f"Incomplete choices: {path}")
    return choices


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_frozen_data()
    qid_position = {str(row["qid"]): index for index, row in enumerate(data.rows)}

    fixed60 = policy_metrics(data.rows, np.full(len(data.rows), 60, dtype=int))

    feature_rows = read_csv(WORK / "features/ablation.csv")
    feature_results = []
    for row in feature_rows:
        name = row["model"]
        choices = choices_from_csv(WORK / "features" / f"{name}_oof.csv", qid_position)
        feature_results.append({"model": name, **policy_metrics(data.rows, choices)})

    class_order = ["bins2", "bins3", "bins4", "bins5", "ordinal"]
    class_results, confusion_payload = [], {}
    class_labels = {
        "bins2": ["1–10", "11–200"],
        "bins3": ["1–10", "11–70", "71–200"],
        "bins4": ["1–10", "11–20", "21–70", "71–200"],
        "bins5": ["1–10", "11–20", "21–50", "51–100", "101–200"],
        "ordinal": ["1–10", "11–20", "21–50", "51–100", "101–200"],
    }
    for name in class_order:
        path = WORK / "classification" / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        choices = choices_from_csv(WORK / "classification" / f"{name}_oof.csv", qid_position)
        metrics = policy_metrics(data.rows, choices)
        matrix = np.asarray(payload["confusion_matrix"], dtype=int)
        easy_total = int(matrix[0].sum())
        class_results.append({
            "model": name,
            **metrics,
            "easy_total": easy_total,
            "easy_predicted_easy": int(matrix[0, 0]),
            "easy_predicted_non_easy": int(matrix[0, 1:].sum()),
            "easy_predicted_hardest": int(matrix[0, -1]),
            "easy_recall": float(matrix[0, 0] / easy_total),
        })
        confusion_payload[name] = {"labels": class_labels[name], "matrix": matrix.tolist()}

    regression_choices = choices_from_csv(WORK / "features/score_modality_oof.csv", qid_position)
    regression = {"model": "continuous_regression", **policy_metrics(data.rows, regression_choices)}

    # Full feature-level correlation matrix for the current score+modality model.
    dictionary = read_csv(WORK / "features/feature_dictionary.csv")
    selected_families = {"point", "local", "band", "shape", "modality"}
    feature_names = [row["feature"] for row in dictionary if row["family"] in selected_families]
    training = read_csv(WORK / "features/training_data.csv")
    feature_matrix = np.asarray([[float(row[name]) for name in feature_names] for row in training], dtype=float)
    correlation = np.corrcoef(feature_matrix, rowvar=False)
    np.savetxt(OUT / "new_feature_correlation.csv", correlation, delimiter=",", header=",".join(feature_names), comments="")

    pairs = []
    for left in range(len(feature_names)):
        for right in range(left + 1, len(feature_names)):
            pairs.append({"feature_a": feature_names[left], "feature_b": feature_names[right], "correlation": float(correlation[left, right]), "absolute_correlation": float(abs(correlation[left, right]))})
    pairs.sort(key=lambda item: -item["absolute_correlation"])
    with (OUT / "top_correlated_feature_pairs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)

    figure, axis = plt.subplots(figsize=(22, 20), dpi=150)
    image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
    axis.set_title("Correlation matrix — score + modality feature set", fontsize=16)
    axis.set_xticks(np.arange(len(feature_names))); axis.set_yticks(np.arange(len(feature_names)))
    axis.set_xticklabels(feature_names, rotation=90, fontsize=5); axis.set_yticklabels(feature_names, fontsize=5)
    axis.tick_params(length=0)
    figure.colorbar(image, ax=axis, fraction=.025, pad=.02, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(OUT / "new_feature_correlation.png", bbox_inches="tight")
    plt.close(figure)

    labels_json = json.dumps(feature_names, ensure_ascii=False)
    values_json = json.dumps(np.round(correlation, 4).tolist(), ensure_ascii=False, separators=(",", ":"))
    fragment = f"""<div id="adaptive-k-correlation">
  <h2>New score + modality features: Pearson correlation</h2>
  <div class="viz-row text-small text-muted" aria-live="polite"><span id="akc-selection">Select a cell to inspect a feature pair.</span></div>
  <svg role="img" aria-label="Correlation matrix for 75 score and modality features"></svg>
  <div class="viz-row text-small"><span style="color:var(--viz-series-2)">■ negative</span><span style="color:var(--muted-foreground)">■ near zero</span><span style="color:var(--viz-series-1)">■ positive</span></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('adaptive-k-correlation');
  const labels = {labels_json};
  const values = {values_json};
  const svg = d3.select(root.querySelector('svg'));
  const selection = root.querySelector('#akc-selection');
  function draw() {{
    const width = Math.max(320, root.getBoundingClientRect().width);
    const margin = {{top: 82, right: 12, bottom: 128, left: 128}};
    const size = Math.min(width - 4, 860);
    const inner = Math.max(180, size - margin.left - margin.right);
    svg.attr('viewBox', `0 0 ${{size}} ${{size + 40}}`).attr('width', '100%');
    svg.selectAll('*').remove();
    const x = d3.scaleBand().domain(d3.range(labels.length)).range([margin.left, margin.left + inner]);
    const y = d3.scaleBand().domain(d3.range(labels.length)).range([margin.top, margin.top + inner]);
    svg.append('rect').attr('data-chart-frame','').attr('x',margin.left).attr('y',margin.top).attr('width',inner).attr('height',inner).attr('fill','none').attr('stroke','var(--border)');
    const cells = [];
    values.forEach((row,i) => row.forEach((value,j) => cells.push({{i,j,value}})));
    svg.append('g').selectAll('rect').data(cells).join('rect')
      .attr('x', d => x(d.j)).attr('y', d => y(d.i)).attr('width', x.bandwidth()+.15).attr('height', y.bandwidth()+.15)
      .attr('fill', d => Math.abs(d.value) < .05 ? 'var(--muted)' : d.value > 0 ? 'var(--viz-series-1)' : 'var(--viz-series-2)')
      .attr('fill-opacity', d => Math.abs(d.value) < .05 ? .35 : .12 + .88*Math.abs(d.value))
      .attr('tabindex', 0).attr('aria-label', d => `${{labels[d.i]}} and ${{labels[d.j]}}: correlation ${{d.value.toFixed(3)}}`)
      .on('click focus', (_,d) => {{ selection.textContent = `${{labels[d.i]}} × ${{labels[d.j]}} = ${{d.value.toFixed(3)}}`; }});
    const step = width < 600 ? 10 : 5;
    const ticks = d3.range(0, labels.length, step);
    svg.append('g').selectAll('text').data(ticks).join('text').attr('fill','var(--foreground)').attr('font-size',11)
      .attr('x',d=>x(d)+x.bandwidth()/2).attr('y',margin.top-5).attr('text-anchor','start').attr('transform',d=>`rotate(-55,${{x(d)+x.bandwidth()/2}},${{margin.top-5}})`).text(d=>labels[d]);
    svg.append('g').selectAll('text').data(ticks).join('text').attr('fill','var(--foreground)').attr('font-size',11)
      .attr('x',margin.left-5).attr('y',d=>y(d)+y.bandwidth()/2+4).attr('text-anchor','end').text(d=>labels[d]);
    svg.append('text').attr('class','axis-title').attr('data-axis','x').attr('fill','var(--foreground)').attr('font-size',12).attr('x',margin.left+inner/2).attr('y',size+22).attr('text-anchor','middle').text('Feature');
    svg.append('text').attr('class','axis-title').attr('data-axis','y').attr('fill','var(--foreground)').attr('font-size',12).attr('transform',`translate(16,${{margin.top+inner/2}}) rotate(-90)`).attr('text-anchor','middle').text('Feature');
  }}
  draw();
  new ResizeObserver(draw).observe(root);
}})();
</script>
"""
    (OUT / "new-feature-correlation.html").write_text(fragment, encoding="utf-8")

    # Confusion matrices use counts and explicit true/predicted axes.
    figure, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    for axis, name in zip(axes.flat, ["bins2", "bins3", "bins4", "bins5"]):
        item = confusion_payload[name]; matrix = np.asarray(item["matrix"]); labels = item["labels"]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(name); axis.set_xlabel("Predicted class"); axis.set_ylabel("True class")
        axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels=labels)
        threshold = matrix.max() / 2
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > threshold else "black")
    figure.tight_layout(); figure.savefig(OUT / "classification_confusion_matrices.png", bbox_inches="tight"); plt.close(figure)

    payload = {"fixed_k60": fixed60, "feature_ablation": feature_results, "classification_progression": class_results, "regression": regression, "confusion_matrices": confusion_payload, "correlation": {"features": len(feature_names), "max_absolute_off_diagonal": pairs[0]["absolute_correlation"], "pairs_absolute_ge_0_95": int(sum(item["absolute_correlation"] >= .95 for item in pairs)), "top_pairs": pairs[:20]}}
    (OUT / "progression.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report = ["# Adaptive-K model progression", "", "## Fixed K=60", "", f"Recall {pct(fixed60['macro_recall'])}; precision {pct(fixed60['macro_precision'])}; F1 {pct(fixed60['macro_f1'])}; nDCG {100*fixed60['macro_ndcg']:.2f}.", "", "## Feature ablation", "", "| Feature set | Mean K | Recall | Precision | F1 | nDCG |", "|---|---:|---:|---:|---:|---:|"]
    report += [f"| {row['model']} | {row['mean_k']:.2f} | {pct(row['macro_recall'])} | {pct(row['macro_precision'])} | {pct(row['macro_f1'])} | {100*row['macro_ndcg']:.2f} |" for row in feature_results]
    report += ["", "## Classification to regression", "", "| Formulation | Mean K | Recall | Precision | F1 | nDCG | Easy retained | Easy sent non-easy |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    report += [f"| {row['model']} | {row['mean_k']:.2f} | {pct(row['macro_recall'])} | {pct(row['macro_precision'])} | {pct(row['macro_f1'])} | {100*row['macro_ndcg']:.2f} | {row['easy_predicted_easy']}/{row['easy_total']} | {row['easy_predicted_non_easy']}/{row['easy_total']} |" for row in class_results if row["model"] != "ordinal"]
    report += [f"| regression | {regression['mean_k']:.2f} | {pct(regression['macro_recall'])} | {pct(regression['macro_precision'])} | {pct(regression['macro_f1'])} | {100*regression['macro_ndcg']:.2f} | n/a | n/a |", ""]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"fixed_k60": fixed60, "classification": class_results, "regression": regression, "correlation": payload["correlation"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
