"""Build an inline visualization for English/French dense-score separation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "data/work/ohr_replacement_20260920/retrieval/language_score_distribution_analysis/file_query_scores.jsonl"
QUERY_SCORES = ROOT / "data/work/ohr_replacement_20260920/retrieval/per_query_language_dense_distributions/query_scores.jsonl"
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/per_query_language_dense_distributions/language-separation.html"


def main() -> int:
    file_rows = [
        json.loads(line)
        for line in SCORES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    query_rows = [
        json.loads(line)
        for line in QUERY_SCORES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    payload: dict[str, dict[str, list[float]]] = {}
    for dataset in ("physics", "industrial", "ohrbench"):
        payload[dataset] = {"en": [], "fr": [], "gaps": []}
    for row in file_rows:
        payload[str(row["dataset"])][str(row["file_language"])].append(round(float(row["dense_raw"]), 5))
    for row in query_rows:
        payload[str(row["dataset"])] ["gaps"].append(round(float(row["raw_max_gap_fr_minus_en"]), 5))

    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html = f'''<div id="language-separation-viz" class="language-separation-viz">
  <div class="viz-heading">
    <h2>Dense separation by document language</h2>
    <div class="viz-legend" aria-label="Legend">
      <span><i class="legend-swatch english"></i>English files</span>
      <span><i class="legend-swatch french"></i>French files</span>
      <span><i class="legend-swatch gap"></i>French max - English max</span>
    </div>
  </div>
  <div class="viz-grid" id="language-separation-panels"></div>
  <p class="viz-note">Upper plot: ECDF of raw max-page dense scores across all files. Lower plot: per-query gap between the strongest French and English file.</p>
</div>
<style>
#language-separation-viz {{
  color: var(--foreground);
  font-family: Georgia, "Times New Roman", serif;
  width: 100%;
}}
#language-separation-viz .viz-heading {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 18px;
  flex-wrap: wrap;
  margin: 0 0 14px;
}}
#language-separation-viz h2 {{
  font-size: 20px;
  font-weight: 500;
  margin: 0;
}}
#language-separation-viz .viz-legend {{
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  color: var(--muted-foreground);
  font-size: 12px;
}}
#language-separation-viz .viz-legend span {{
  display: inline-flex;
  align-items: center;
  gap: 5px;
}}
#language-separation-viz .legend-swatch {{
  width: 10px;
  height: 10px;
  display: inline-block;
  background: currentColor;
}}
#language-separation-viz .legend-swatch.english {{ color: var(--viz-series-1); }}
#language-separation-viz .legend-swatch.french {{ color: var(--viz-series-2); }}
#language-separation-viz .legend-swatch.gap {{ color: var(--viz-series-3); }}
#language-separation-viz .viz-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}}
#language-separation-viz .dataset-panel {{ min-width: 0; }}
#language-separation-viz svg {{
  display: block;
  width: 100%;
  height: auto;
  overflow: visible;
}}
#language-separation-viz .axis path,
#language-separation-viz .axis line {{ stroke: var(--border); shape-rendering: crispEdges; }}
#language-separation-viz .axis text,
#language-separation-viz .plot-label {{ fill: var(--foreground); font-size: 12px; }}
#language-separation-viz .muted-label {{ fill: var(--muted-foreground); font-size: 11px; }}
#language-separation-viz .grid-line {{ stroke: var(--border); stroke-opacity: .48; shape-rendering: crispEdges; }}
#language-separation-viz .median-line {{ stroke-width: 1.5; stroke-dasharray: 3 3; opacity: .8; }}
#language-separation-viz .ecdf-line {{ fill: none; stroke-width: 2.2; }}
#language-separation-viz .gap-bar {{ opacity: .72; }}
#language-separation-viz .zero-line {{ stroke: var(--foreground); stroke-opacity: .65; stroke-width: 1.2; }}
#language-separation-viz .viz-note {{
  color: var(--muted-foreground);
  font-size: 12px;
  line-height: 1.4;
  margin: 8px 0 0;
}}
@media (max-width: 760px) {{
  #language-separation-viz .viz-grid {{ grid-template-columns: 1fr; }}
}}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("language-separation-viz");
  const host = document.getElementById("language-separation-panels");
  const data = {data_json};
  const datasets = [
    ["physics", "Physics"],
    ["industrial", "Industrial"],
    ["ohrbench", "OHR-Bench"]
  ];
  const colors = {{
    en: getComputedStyle(root).getPropertyValue("--viz-series-1").trim(),
    fr: getComputedStyle(root).getPropertyValue("--viz-series-2").trim(),
    gap: getComputedStyle(root).getPropertyValue("--viz-series-3").trim(),
    foreground: getComputedStyle(root).getPropertyValue("--foreground").trim(),
    muted: getComputedStyle(root).getPropertyValue("--muted-foreground").trim(),
    border: getComputedStyle(root).getPropertyValue("--border").trim()
  }};

  const margin = {{ top: 28, right: 18, bottom: 30, left: 42 }};
  const upperHeight = 150;
  const lowerHeight = 112;
  const gapBetween = 42;
  const totalHeight = margin.top + upperHeight + gapBetween + lowerHeight + margin.bottom;

  function ecdf(values) {{
    const sorted = [...values].sort((a, b) => a - b);
    return sorted.map((value, index) => [value, (index + 1) / sorted.length]);
  }}

  function median(values) {{ return d3.quantile([...values].sort(d3.ascending), .5); }}

  function drawPanel(key, label, panel) {{
    const width = Math.max(320, panel.clientWidth || 360);
    const innerWidth = width - margin.left - margin.right;
    const svg = d3.select(panel).select("svg");
    svg.attr("viewBox", `0 0 ${{width}} ${{totalHeight}}`);
    svg.selectAll("*").remove();

    const upper = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);
    const lower = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top + upperHeight + gapBetween}})`);
    const values = data[key];
    const allScores = [...values.en, ...values.fr];
    const scoreMax = d3.max(allScores) || 1;
    const scoreX = d3.scaleLinear().domain([0, scoreMax * 1.04]).nice().range([0, innerWidth]);
    const ecdfY = d3.scaleLinear().domain([0, 1]).range([upperHeight, 0]);
    const gapAbs = d3.max(values.gaps, value => Math.abs(value)) || 1;
    const gapX = d3.scaleLinear().domain([-gapAbs * 1.08, gapAbs * 1.08]).nice().range([0, innerWidth]);
    const bins = d3.bin().domain(gapX.domain()).thresholds(12)(values.gaps);
    const countMax = d3.max(bins, bin => bin.length) || 1;
    const gapY = d3.scaleLinear().domain([0, countMax]).nice().range([lowerHeight, 0]);

    svg.append("title").text(`${{label}} dense score language separation`);
    svg.append("desc").text(`English and French file score ECDFs plus per-query French-minus-English maximum score gaps for ${{label}}.`);
    svg.append("text").attr("class", "plot-label").attr("x", margin.left).attr("y", 16).text(label);

    upper.append("g").attr("class", "axis").attr("transform", `translate(0,${{upperHeight}})`).call(d3.axisBottom(scoreX).ticks(4).tickFormat(d3.format(".2f")));
    upper.append("g").attr("class", "axis").call(d3.axisLeft(ecdfY).ticks(3).tickFormat(d3.format(".0%")));
    upper.append("text").attr("class", "muted-label").attr("x", innerWidth).attr("y", upperHeight + 27).attr("text-anchor", "end").text("raw dense score");
    upper.append("text").attr("class", "muted-label").attr("transform", "rotate(-90)").attr("x", -4).attr("y", -31).attr("text-anchor", "end").text("ECDF");

    for (const [language, labelText] of [["en", "EN"], ["fr", "FR"]]) {{
      const line = d3.line().x(point => scoreX(point[0])).y(point => ecdfY(point[1]));
      upper.append("path").datum(ecdf(values[language])).attr("class", "ecdf-line").attr("stroke", colors[language]).attr("d", line);
      const med = median(values[language]);
      upper.append("line").attr("class", "median-line").attr("stroke", colors[language]).attr("x1", scoreX(med)).attr("x2", scoreX(med)).attr("y1", 0).attr("y2", upperHeight);
      upper.append("text").attr("class", "muted-label").attr("fill", colors[language]).attr("x", scoreX(med) + 4).attr("y", language === "en" ? 16 : 31).text(`${{labelText}} median ${{med.toFixed(3)}}`);
    }}

    lower.append("g").attr("class", "axis").attr("transform", `translate(0,${{lowerHeight}})`).call(d3.axisBottom(gapX).ticks(5).tickFormat(d3.format("+.2f")));
    lower.append("g").attr("class", "axis").call(d3.axisLeft(gapY).ticks(3).tickFormat(d3.format("d")));
    lower.append("text").attr("class", "muted-label").attr("x", innerWidth).attr("y", lowerHeight + 27).attr("text-anchor", "end").text("French max - English max");
    lower.append("text").attr("class", "muted-label").attr("transform", "rotate(-90)").attr("x", -4).attr("y", -31).attr("text-anchor", "end").text("queries");
    lower.append("line").attr("class", "zero-line").attr("x1", gapX(0)).attr("x2", gapX(0)).attr("y1", 0).attr("y2", lowerHeight);
    lower.selectAll(".gap-bar").data(bins).join("rect")
      .attr("class", "gap-bar")
      .attr("x", bin => gapX(bin.x0) + 1)
      .attr("y", bin => gapY(bin.length))
      .attr("width", bin => Math.max(0, gapX(bin.x1) - gapX(bin.x0) - 2))
      .attr("height", bin => lowerHeight - gapY(bin.length))
      .attr("fill", bin => ((bin.x0 + bin.x1) / 2 >= 0 ? colors.fr : colors.en));
    const positive = values.gaps.filter(value => value > 0).length / values.gaps.length;
    lower.append("text").attr("class", "muted-label").attr("x", innerWidth).attr("y", 12).attr("text-anchor", "end").text(`${{(positive * 100).toFixed(1)}}% French max higher`);
  }}

  for (const [key, label] of datasets) {{
    const panel = document.createElement("div");
    panel.className = "dataset-panel";
    panel.innerHTML = "<svg role=\"img\"></svg>";
    host.appendChild(panel);
    const redraw = () => drawPanel(key, label, panel);
    redraw();
    new ResizeObserver(redraw).observe(panel);
  }}
}})();
</script>'''
    OUTPUT.write_text(html, encoding="utf-8")
    print(OUTPUT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
