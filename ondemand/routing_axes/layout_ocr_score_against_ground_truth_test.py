import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ondemand.routing_axes import layout_ocr_score_against_ground_truth as k

PARAS = ["Torque settings for the main rotor hub are listed below.",
         "Inspect every bolt for corrosion before installation and replace damaged parts.",
         "Record the inspection date in the aircraft logbook after each service."]
MD = "| Part | Torque |\n|---|---|\n| Hub bolt | 45 Nm |\n| Pitch link | 12 Nm |"
HTML = "<table><tr><td>Part</td><td>Torque</td></tr><tr><td>Hub bolt</td><td>45 Nm</td></tr>" \
       "<tr><td>Pitch link</td><td>12 Nm</td></tr></table>"
LATEX = "Senators\n\n\\begin{tabular}{c|l|r}\nRank & \\multicolumn{2}{c|}{Amount} \\\\\n\\hline\n1 & Smith & \\$0 \\\\\n\\end{tabular}"
LATEX_HTML = "<table><tr><td>Rank</td><td colspan=\"2\">Amount</td></tr><tr><td>1</td><td>Smith</td><td>$0</td></tr></table>"

refs = {"vidore_industrial::a#page=0": {"page_id": "vidore_industrial::a#page=0", "group": "vidore_industrial",
                                         "text": "\n\n".join(PARAS[:2] + [MD] + PARAS[2:])},
        "ohrbench::b#page=0": {"page_id": "ohrbench::b#page=0", "group": "ohrbench:scan", "text": LATEX}}


def regions(page, items):
    return [{"page_id": page, "region_idx": i, "order": i, "type": t, "box": [0, 0, 1, 1], "text": x, "seconds": 0.5}
            for i, (t, x) in enumerate(items)]


A, B = "vidore_industrial::a#page=0", "ohrbench::b#page=0"
good_a = [("text", PARAS[0]), ("text", PARAS[1]), ("table", HTML), ("text", PARAS[2])]
good_b = [("text", "Senators"), ("table", LATEX_HTML)]
readers = {
    "perfect": regions(A, good_a) + regions(B, good_b),
    "reversed": regions(A, good_a[::-1]) + regions(B, good_b),
    "looping": regions(A, good_a[:3] + [("text", " ".join(["the bolt torque is forty five newton metres"] * 6))])
               + regions(B, good_b),
}
KEPT = {A: len(good_a), B: len(good_b)}

with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    (folder / "layout.jsonl").write_text("".join(json.dumps({"page_id": p, "seconds_orient": 0.1, "seconds_layout": 0.2,
                                                             "n_kept": KEPT[p]}) + "\n" for p in refs))
    for name, rows in readers.items():
        (folder / f"regions_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    flat = {A: regions(A, [("page", "\n".join(PARAS))]), B: []}
    results, excluded = k.score(folder, refs, flat=flat)
    (folder / "regions_cut.jsonl").write_text("".join(json.dumps(r) + "\n" for r in regions(A, good_a[:2]) + regions(B, good_b)))
    cut, cut_excluded = k.score(folder, refs, flat=flat)

perfect = results["perfect"]["all"]
assert perfect["ned"] < 1e-9 and perfect["precision"] == perfect["recall"] == 1.0, perfect
assert abs(perfect["teds"] - 1.0) < 1e-9 and perfect["n_teds"] == 2, perfect
assert perfect["order"] == 0.0 and perfect["n_order"] == 1 and perfect["loop_pct"] == 0.0, perfect
assert abs(perfect["seconds"] - 1.5) < 1e-9 and abs(perfect["layout_seconds"] - 0.3) < 1e-9, perfect
rev = results["reversed"]["vidore_industrial"]
assert rev["order"] > 0 and rev["recall"] == 1.0 and rev["precision"] == 1.0, rev
loop = results["looping"]
assert loop["vidore_industrial"]["loop_pct"] == 100.0 and not any(excluded.values()), (loop, excluded)
flat_s = results[k.FLAT]
assert flat_s["vidore_industrial"]["teds"] == 0.0 and flat_s["ohrbench:scan"]["empty"] == 1, flat_s
assert cut_excluded["cut"] == [A] and cut_excluded["perfect"] == [], cut_excluded
assert all(cut[r]["all"]["n"] == 1 and "vidore_industrial" not in cut[r] for r in cut), {r: cut[r]["all"]["n"] for r in cut}
assert k.tables_of(LATEX) == k.tables_of(LATEX_HTML) and k.tables_of(MD) == k.tables_of(HTML)
assert k.norm(LATEX) == "senators " + k.norm(LATEX_HTML) == "senators rank amount 1 smith 0", k.norm(LATEX)
assert k.norm(MD) == k.norm(HTML)
OTSL = "<fcel>Rank<fcel>Amount<lcel><nl><fcel>1<fcel>Smith<fcel>$0<nl>"
assert k.tables_of(k.otsl_to_html(OTSL)) == k.tables_of(LATEX_HTML), k.otsl_to_html(OTSL)
SPAN = "<fcel>A<fcel>B<nl><ucel><fcel>C<nl>"
assert k.otsl_to_html(SPAN) == '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'
assert k.otsl_to_html("plain text") == "plain text" and k.otsl_to_html(HTML) == HTML
print(k.table(results))
print("norm / tables / TEDS / order / loop / cut-run page intersection ok")
