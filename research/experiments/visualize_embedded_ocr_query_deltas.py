"""Compare per-query retrieval with no embedded OCR, raw OCR, and filtered OCR."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import html
import json
from math import log2
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ndcg(pages: Sequence[str], gold: Mapping[str, int], k: int) -> float:
    dcg = sum((2.0 ** gold[page] - 1.0) / log2(rank + 1) for rank, page in enumerate(pages[:k], 1) if page in gold)
    ideal = sum((2.0 ** value - 1.0) / log2(rank + 1) for rank, value in enumerate(sorted(gold.values(), reverse=True)[:k], 1))
    return dcg / ideal if ideal else 0.0


def metrics(row: Mapping[str, Any], gold: Mapping[str, int]) -> dict[str, Any]:
    pages = [str(item["page_id"]) for item in row["chunks"]]
    gold_set = set(gold)
    ranks = {page: index for index, page in enumerate(pages, 1)}
    return {
        "recall10": len(set(pages[:10]) & gold_set) / len(gold_set),
        "recall20": len(set(pages[:20]) & gold_set) / len(gold_set),
        "ndcg10": ndcg(pages, gold, 10),
        "ndcg20": ndcg(pages, gold, 20),
        "pages": pages,
        "gold_ranks": {page: ranks.get(page) for page in gold},
        "gold_top10": [page for page in pages[:10] if page in gold_set],
        "gold_top20": [page for page in pages[:20] if page in gold_set],
    }


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--raw-run", type=Path, required=True)
    parser.add_argument("--filtered-run", type=Path, required=True)
    parser.add_argument("--ocr-pages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    queries = {str(row["query_id"]): row for row in read_jsonl(args.queries)}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(args.qrels):
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))
    runs = {
        "baseline": {str(row["query_id"]): row for row in read_jsonl(args.baseline_run)},
        "raw_ocr": {str(row["query_id"]): row for row in read_jsonl(args.raw_run)},
        "filtered_1pct_16k": {str(row["query_id"]): row for row in read_jsonl(args.filtered_run)},
    }
    ocr_pages = {str(row["page_id"]) for row in read_jsonl(args.ocr_pages) if row.get("text", "").strip() and row.get("status") == "ok"}
    qids = sorted(set(queries) & set(qrels) & set(runs["baseline"]) & set(runs["raw_ocr"]) & set(runs["filtered_1pct_16k"]))
    records: list[dict[str, Any]] = []
    for qid in qids:
        gold = qrels[qid]
        base = metrics(runs["baseline"][qid], gold)
        raw = metrics(runs["raw_ocr"][qid], gold)
        filt = metrics(runs["filtered_1pct_16k"][qid], gold)
        raw_new_gold = sorted(set(raw["gold_top20"]) - set(base["gold_top20"]))
        filt_new_gold = sorted(set(filt["gold_top20"]) - set(base["gold_top20"]))
        record = {
            "query_id": qid, "source": queries[qid].get("source", qid.split("::", 1)[0]), "query": queries[qid]["query"],
            "gold_pages": sorted(gold), "gold_count": len(gold),
            "baseline": {key: base[key] for key in ("recall10", "recall20", "ndcg10", "ndcg20", "gold_ranks", "gold_top10", "gold_top20")},
            "raw_ocr": {key: raw[key] for key in ("recall10", "recall20", "ndcg10", "ndcg20", "gold_ranks", "gold_top10", "gold_top20")},
            "filtered": {key: filt[key] for key in ("recall10", "recall20", "ndcg10", "ndcg20", "gold_ranks", "gold_top10", "gold_top20")},
            "delta_raw": {"recall10": raw["recall10"] - base["recall10"], "recall20": raw["recall20"] - base["recall20"], "ndcg10": raw["ndcg10"] - base["ndcg10"], "ndcg20": raw["ndcg20"] - base["ndcg20"]},
            "delta_filtered": {"recall10": filt["recall10"] - base["recall10"], "recall20": filt["recall20"] - base["recall20"], "ndcg10": filt["ndcg10"] - base["ndcg10"], "ndcg20": filt["ndcg20"] - base["ndcg20"]},
            "raw_new_gold_in_top20": raw_new_gold, "filtered_new_gold_in_top20": filt_new_gold,
            "raw_new_gold_with_ocr_text": [page for page in raw_new_gold if page in ocr_pages],
            "filtered_new_gold_with_ocr_text": [page for page in filt_new_gold if page in ocr_pages],
        }
        records.append(record)

    changed_records = [
        row for row in records
        if any(abs(float(row["delta_raw"][metric])) > 0 for metric in ("recall10", "recall20", "ndcg10", "ndcg20"))
        or any(abs(float(row["delta_filtered"][metric])) > 0 for metric in ("recall10", "recall20", "ndcg10", "ndcg20"))
    ]

    def aggregate(key: str) -> dict[str, float]:
        return {metric: statistics.mean(float(row[key][metric]) for row in records) for metric in ("recall10", "recall20", "ndcg10", "ndcg20")}

    def summary(delta_key: str) -> dict[str, Any]:
        values = [row[delta_key][metric] for row in records for metric in ["recall10"]]
        return {"improved": sum(value > 0 for value in values), "unchanged": sum(value == 0 for value in values), "declined": sum(value < 0 for value in values), "mean_delta_recall10": statistics.mean(values), "median_delta_recall10": statistics.median(values)}

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row["source"]].append(row)
    overview = {"queries": len(records), "ocr_pages": len(ocr_pages), "overall": {"baseline": aggregate("baseline"), "raw_ocr": aggregate("raw_ocr"), "filtered": aggregate("filtered")}, "delta_raw": summary("delta_raw"), "delta_filtered": summary("delta_filtered"), "by_source": {source: {"queries": len(items), "baseline": {metric: statistics.mean(r["baseline"][metric] for r in items) for metric in ("recall10", "recall20", "ndcg10", "ndcg20")}, "raw_ocr": {metric: statistics.mean(r["raw_ocr"][metric] for r in items) for metric in ("recall10", "recall20", "ndcg10", "ndcg20")}, "filtered": {metric: statistics.mean(r["filtered"][metric] for r in items) for metric in ("recall10", "recall20", "ndcg10", "ndcg20")}} for source, items in groups.items()}}
    (args.output_dir / "per_query_ocr_delta.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records), encoding="utf-8")
    (args.output_dir / "changed_queries.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in changed_records), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = json.dumps(changed_records, ensure_ascii=False).replace("</", "<\\/")
    overview_payload = json.dumps(overview, ensure_ascii=False).replace("</", "<\\/")
    cards = []
    for row in changed_records:
        cards.append(f'''<tr data-source="{esc(row['source'])}" data-draw="{row['delta_raw']['recall10']:.8f}" data-dfilt="{row['delta_filtered']['recall10']:.8f}"><td><code>{esc(row['query_id'])}</code></td><td>{esc(row['source'])}</td><td class="query">{esc(row['query'])}</td><td>{row['baseline']['recall10']:.2%}</td><td>{row['raw_ocr']['recall10']:.2%}</td><td>{row['filtered']['recall10']:.2%}</td><td class="delta raw">{row['delta_raw']['recall10']:+.2%}</td><td class="delta filt">{row['delta_filtered']['recall10']:+.2%}</td><td>{esc(', '.join(f"{p}:b{row['baseline']['gold_ranks'][p] or '-'},r{row['raw_ocr']['gold_ranks'][p] or '-'},f{row['filtered']['gold_ranks'][p] or '-'}" for p in row['gold_pages']))}</td></tr>''')
    source_options = ''.join(f'<option value="{esc(source)}">{esc(source)}</option>' for source in sorted(groups))
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Embedded OCR query deltas</title>
<style>body{{font:14px system-ui;margin:0;color:#17212b;background:#f4f6f8}}header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccd5df;padding:14px 20px;z-index:2}}h1{{margin:0 0 8px;font-size:21px}}.cards{{display:flex;gap:10px;flex-wrap:wrap}}.stat{{background:#eef3f7;border:1px solid #d2dce5;border-radius:7px;padding:8px 12px}}.controls{{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:10px}}main{{padding:16px}}.chart{{background:#fff;border:1px solid #d5dde5;border-radius:7px;margin-bottom:14px;padding:10px}}svg{{width:100%;height:320px;background:#fafbfd}}table{{width:100%;border-collapse:collapse;background:#fff;font-size:12px}}th,td{{border-bottom:1px solid #e0e5ea;padding:7px;text-align:left;vertical-align:top}}th{{position:sticky;top:138px;background:#edf2f6;cursor:pointer}}td.query{{max-width:430px;min-width:250px}}td.delta{{font-weight:700;white-space:nowrap}}.pos{{color:#087443}}.neg{{color:#a12a2a}}code{{font-size:11px}}.ranks{{font-family:monospace;font-size:10px;max-width:260px;overflow-wrap:anywhere}}</style></head><body>
<header><h1>Embedded-image OCR: changed queries only</h1><div class="cards"><span class="stat">Changed queries: <b>{len(changed_records)}</b> / {len(records)}</span><span class="stat">Raw OCR Recall@10: <b>{overview['overall']['raw_ocr']['recall10']:.2%}</b></span><span class="stat">Filtered Recall@10: <b>{overview['overall']['filtered']['recall10']:.2%}</b></span><span class="stat">Raw wins / ties / losses: <b>{overview['delta_raw']['improved']} / {overview['delta_raw']['unchanged']} / {overview['delta_raw']['declined']}</b></span><span class="stat">Filtered wins / ties / losses: <b>{overview['delta_filtered']['improved']} / {overview['delta_filtered']['unchanged']} / {overview['delta_filtered']['declined']}</b></span></div><div class="controls"><label>Source <select id="source"><option value="all">all</option>{source_options}</select></label><label>Sort <select id="sort"><option value="raw">raw OCR delta</option><option value="filtered">filtered delta</option><option value="qid">query ID</option></select></label><label>Change <select id="change"><option value="all">all</option><option value="up">improved</option><option value="same">unchanged</option><option value="down">declined</option></select></label><span id="shown"></span></div></header>
<main><section class="chart"><b>Recall@10: baseline vs raw OCR</b><svg id="scatter" viewBox="0 0 900 320" role="img"></svg></section><section class="chart"><b>Per-query Recall@10 delta</b><svg id="bars" viewBox="0 0 900 380" role="img"></svg></section><table><thead><tr><th>QID</th><th>Source</th><th>Query</th><th>Base R10</th><th>Raw R10</th><th>Filt R10</th><th>Δ Raw</th><th>Δ Filt</th><th>Gold page ranks (base/raw/filt)</th></tr></thead><tbody>{''.join(cards)}</tbody></table></main>
<script>const DATA={payload};const OVERVIEW={overview_payload};const rows=[...document.querySelectorAll('tbody tr')];const source=document.querySelector('#source'),sorter=document.querySelector('#sort'),change=document.querySelector('#change'),shown=document.querySelector('#shown');function status(r,kind){{const d=kind==='filtered'?r.delta_filtered.recall10:r.delta_raw.recall10;return d>0?'up':d<0?'down':'same'}}function apply(){{const s=source.value,c=change.value,kind=sorter.value;rows.forEach(r=>{{const item=DATA.find(x=>x.query_id===r.cells[0].innerText);r.style.display=(s==='all'||item.source===s)&&(c==='all'||status(item,kind==='filtered'?'filtered':'raw')===c)?'table-row':'none';}});rows.sort((a,b)=>{{const x=DATA.find(x=>x.query_id===a.cells[0].innerText),y=DATA.find(x=>x.query_id===b.cells[0].innerText);if(kind==='qid')return x.query_id.localeCompare(y.query_id);const key=kind==='filtered'?'delta_filtered':'delta_raw';return y[key].recall10-x[key].recall10;}}).forEach(r=>r.parentNode.appendChild(r));shown.textContent=rows.filter(r=>r.style.display!=='none').length+' shown';draw(kind);}}function draw(kind){{const svg=document.querySelector('#scatter'),bar=document.querySelector('#bars');const W=900,H=320,p=42;svg.innerHTML='<line x1="'+p+'" y1="'+(H-p)+'" x2="'+(W-p)+'" y2="'+p+'" stroke="#9aa8b5" stroke-dasharray="4"/><text x="'+p+'" y="18">raw Recall@10</text><text x="'+(W-120)+'" y="'+(H-8)+'">baseline Recall@10</text>';DATA.forEach(d=>{{const x=p+(d.baseline.recall10)*(W-2*p),y=H-p-(d.raw_ocr.recall10)*(H-2*p),col=d.delta_raw.recall10>0?'#087443':d.delta_raw.recall10<0?'#b33a3a':'#647482';svg.innerHTML+='<circle cx="'+x+'" cy="'+y+'" r="4" fill="'+col+'"><title>'+d.query_id+' '+d.delta_raw.recall10.toFixed(3)+'</title></circle>';}});const sorted=[...DATA].sort((a,b)=>(kind==='filtered'?b.delta_filtered.recall10-a.delta_filtered.recall10:b.delta_raw.recall10-a.delta_raw.recall10));const step=(W-2*p)/Math.max(sorted.length-1,1);bar.innerHTML='';sorted.forEach((d,i)=>{{const x=p+i*step,y=H/2-(kind==='filtered'?d.delta_filtered.recall10:d.delta_raw.recall10)*260;const zero=H/2;const color=(kind==='filtered'?d.delta_filtered.recall10:d.delta_raw.recall10)>=0?'#087443':'#b33a3a';bar.innerHTML+='<line x1="'+x+'" y1="'+zero+'" x2="'+x+'" y2="'+y+'" stroke="'+color+'" stroke-width="3"><title>'+d.query_id+'</title></line>';}});}}[source,sorter,change].forEach(x=>x.addEventListener('input',apply));apply();</script></body></html>'''
    (args.output_dir / "index_changed_only.html").write_text(document, encoding="utf-8")
    # Preserve the previous all-query view under an explicit name; the default
    # deliverable is now the compact changed-query view.
    previous = args.output_dir / "index.html"
    if previous.is_file():
        previous.replace(args.output_dir / "index_all_queries.html")
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")
    print(json.dumps({"queries": len(records), "changed_queries": len(changed_records), "summary": str((args.output_dir / "summary.json").resolve()), "html": str((args.output_dir / "index.html").resolve()), "all_html": str((args.output_dir / "index_all_queries.html").resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
