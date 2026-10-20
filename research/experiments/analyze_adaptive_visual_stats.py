"""Build per-query statistics and an inline D3 dashboard for Adaptive-k runs."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "data/work/benchmark_hierarchical_local/bundle_light_dense_bm25_adaptive"
QRELS = ROOT / (
    "data/work/benchmark_kf3_top20_kdl_ocr64_t300-20260917T181644Z-1-001/"
    "benchmark_kf3_top20_kdl_ocr64_t300/light_retry32_eng_v2/qrels_page_level.jsonl"
)
OUT_DIR = ROOT / "data/work/benchmark_hierarchical_local/adaptive_stats"
OUT_JSON = OUT_DIR / "adaptive_stats.json"
OUT_HTML = OUT_DIR / "adaptive_diagnostics.html"

SETTINGS = [
    ("bm25_hier", "BM25 hierarchical", "bm25_adaptive_gap_plus5_raw.jsonl", "#2563eb"),
    ("bm25_global", "BM25 global", "bm25_global_adaptive_gap_plus5_raw.jsonl", "#0f766e"),
    ("dense_hier", "Dense hierarchical", "dense_adaptive_gap_plus5_raw.jsonl", "#c2410c"),
    ("dense_global", "Dense global", "dense_global_adaptive_gap_plus5_raw.jsonl", "#7c3aed"),
]


def load_qrels() -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with QRELS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qrels.setdefault(row["query_id"], set()).add(row["page_id"])
    return qrels


def gap_stats(scores: list[float]) -> tuple[float, int, float]:
    if len(scores) < 2:
        return 0.0, 1, 0.0
    gaps = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    index = max(range(len(gaps)), key=lambda i: gaps[i])
    return float(gaps[index]), index + 1, float(min(gaps))


def load_points(qrels: dict[str, set[str]], filename: str) -> list[dict]:
    points = []
    with (RUN_DIR / filename).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scores = [float(chunk["score"]) for chunk in row.get("chunks", [])]
            gap, gap_rank, min_gap = gap_stats(scores)
            k = int(row.get("kp", len(row.get("returned_pages", []))))
            points.append(
                {
                    "q": row["query_id"],
                    "g": len(qrels[row["query_id"]]),
                    "k": k,
                    "m": round(gap, 9),
                    "l": round(min_gap, 9),
                    "r": gap_rank,
                    "n": len(scores),
                }
            )
    return points


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def make_payload(qrels: dict[str, set[str]]) -> dict:
    settings = []
    for key, label, filename, color in SETTINGS:
        points = load_points(qrels, filename)
        ks = [p["k"] for p in points]
        gaps = [p["m"] for p in points]
        settings.append(
            {
                "key": key,
                "label": label,
                "color": color,
                "points": points,
                "summary": {
                    "queries": len(points),
                    "meanK": round(mean(ks), 3),
                    "medianK": round(median(ks), 3),
                    "minK": min(ks),
                    "maxK": max(ks),
                    "meanGold": round(mean(p["g"] for p in points), 3),
                    "meanGap": round(mean(gaps), 6),
                    "medianGap": round(median(gaps), 6),
                    "p90Gap": round(percentile(gaps, 0.9), 6),
                    "underGold": sum(p["k"] < p["g"] for p in points),
                    "equalGold": sum(p["k"] == p["g"] for p in points),
                    "overGold": sum(p["k"] > p["g"] for p in points),
                },
            }
        )
    return {
        "meta": {
            "queries": len(qrels),
            "goldDefinition": "unique qrel evidence pages per query",
            "gapDefinition": "largest adjacent descending score gap within the returned score prefix",
            "adaptiveBuffer": "gap-based k with retrieve_more=5; raw uncalibrated adaptive outputs",
        },
        "settings": settings,
    }


def make_html(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    escaped_data = html.escape(data, quote=False)
    return f'''<div id="adaptive-k-dashboard" class="adaptive-k-dashboard">
  <style>
    #adaptive-k-dashboard {{
      --ink: #172033; --muted: #5d6b82; --grid: #d9e1ec; --card: #ffffff;
      --canvas: #f5f7fb; --accent: #2563eb; color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      max-width: 1220px; margin: 0 auto; padding: 20px; background: var(--canvas);
      box-sizing: border-box;
    }}
    #adaptive-k-dashboard * {{ box-sizing: border-box; }}
    #adaptive-k-dashboard h2 {{ margin: 0 0 6px; font-size: 23px; letter-spacing: -0.02em; }}
    #adaptive-k-dashboard h3 {{ margin: 0 0 4px; font-size: 16px; }}
    #adaptive-k-dashboard p {{ margin: 0; color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .adaptive-k-header {{ margin-bottom: 14px; }}
    .adaptive-k-note {{ margin-top: 6px !important; }}
    .adaptive-k-card {{ background: var(--card); border: 1px solid #e1e7f0; border-radius: 12px; padding: 15px; margin-top: 14px; box-shadow: 0 2px 9px rgba(20, 35, 60, 0.04); }}
    .adaptive-k-card-head {{ display: flex; justify-content: space-between; gap: 14px; align-items: baseline; margin-bottom: 10px; }}
    .adaptive-k-legend {{ display: flex; flex-wrap: wrap; gap: 9px 15px; color: var(--muted); font-size: 12px; }}
    .adaptive-k-legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
    .adaptive-k-swatch {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
    .adaptive-k-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .adaptive-k-panel {{ border: 1px solid #e7ebf2; border-radius: 9px; padding: 7px; min-width: 0; }}
    .adaptive-k-panel-title {{ font-size: 12px; font-weight: 650; margin: 1px 4px 2px; }}
    .adaptive-k-panel svg {{ width: 100%; height: auto; display: block; }}
    .adaptive-k-summary {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }}
    .adaptive-k-summary th, .adaptive-k-summary td {{ padding: 7px 8px; border-bottom: 1px solid #edf0f5; text-align: right; white-space: nowrap; }}
    .adaptive-k-summary th:first-child, .adaptive-k-summary td:first-child {{ text-align: left; }}
    .adaptive-k-summary th {{ color: var(--muted); font-weight: 600; }}
    .adaptive-k-tooltip {{ position: fixed; pointer-events: none; display: none; background: #172033; color: #fff; padding: 7px 9px; border-radius: 6px; font-size: 12px; line-height: 1.35; z-index: 20; box-shadow: 0 3px 12px rgba(0,0,0,.18); }}
    .axis text {{ fill: var(--muted); font-size: 10px; }} .axis path, .axis line {{ stroke: var(--grid); }}
    .grid line {{ stroke: #edf0f5; }} .zero-line {{ stroke: #8e9bb0; stroke-dasharray: 4 4; }}
    .adaptive-k-caption {{ margin-top: 8px !important; font-size: 12px !important; }}
    @media (max-width: 760px) {{ #adaptive-k-dashboard {{ padding: 12px; }} .adaptive-k-grid {{ grid-template-columns: 1fr; }} .adaptive-k-card-head {{ display: block; }} .adaptive-k-legend {{ margin-top: 8px; }} .adaptive-k-summary {{ display: block; overflow-x: auto; }} }}
  </style>
  <div class="adaptive-k-header">
    <h2>Adaptive-k: per-query diagnostic view</h2>
    <p>220 queries · four raw, uncalibrated adaptive runs · gold count = unique evidence pages in qrels.</p>
    <p class="adaptive-k-note">Score-gap note: max gap is computed from the returned descending score prefix. With retrieve_more=5, the selected gap is normally five ranks before the returned cutoff; this is a diagnostic of the saved run, not a new retrieval pass.</p>
  </div>
  <section class="adaptive-k-card">
    <div class="adaptive-k-card-head"><div><h3>1. Distribution of adaptive TopK</h3><p>Log-scaled x-axis keeps the long BM25 tail visible.</p></div><div id="adaptive-k-legend-1" class="adaptive-k-legend"></div></div>
    <div id="adaptive-k-hist" class="adaptive-k-grid"></div>
    <p class="adaptive-k-caption">Per-query view: each bar is one query in the saved run order; the Y-axis is linear so bar heights represent the actual TopK scale.</p>
    <div id="adaptive-k-query-bars" class="adaptive-k-grid"></div>
  </section>
  <section class="adaptive-k-card">
    <div class="adaptive-k-card-head"><div><h3>2. Predicted TopK versus gold page count</h3><p>Queries are sorted by gold-page count. Faint dots show raw queries; bold lines show an 11-query rolling median. Hover a dot to see the query ID.</p></div><div class="adaptive-k-legend"><span><i class="adaptive-k-swatch" style="background:#64748b"></i>gold pages</span><span><i class="adaptive-k-swatch" style="background:#2563eb"></i>adaptive K</span></div></div>
    <div id="adaptive-k-gold" class="adaptive-k-grid"></div>
  </section>
  <section class="adaptive-k-card">
    <div class="adaptive-k-card-head"><div><h3>3. Per-query score-gap range</h3><p>Each panel follows the saved query order. Only an 11-query moving mean is shown. Left Y-axis = score gap; right Y-axis = gold pages.</p></div><div class="adaptive-k-legend"><span><i class="adaptive-k-swatch" style="background:#2563eb"></i>mean max gap</span><span><i class="adaptive-k-swatch" style="background:#94a3b8"></i>mean min gap</span><span><i class="adaptive-k-swatch" style="background:#ea580c"></i>mean gold pages</span></div></div>
    <div id="adaptive-k-gap" class="adaptive-k-grid"></div>
  </section>
  <section class="adaptive-k-card">
    <div class="adaptive-k-card-head"><div><h3>Per-setting summary</h3><p>Under/equal/over counts compare adaptive K with the query's gold page count.</p></div></div>
    <div id="adaptive-k-table"></div>
  </section>
  <div id="adaptive-k-tooltip" class="adaptive-k-tooltip"></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
  (() => {{
    const DATA = {escaped_data};
    const margin = {{top: 20, right: 18, bottom: 38, left: 48}};
    const tooltip = d3.select('#adaptive-k-tooltip');
    const settings = DATA.settings;
    const color = new Map(settings.map(s => [s.key, s.color]));
    const fmt = d3.format('.3~f');
    const fmtGap = d3.format('.4f');
    const showTip = (event, text) => tooltip.style('display', 'block').style('left', (event.clientX + 12) + 'px').style('top', (event.clientY + 12) + 'px').html(text);
    const hideTip = () => tooltip.style('display', 'none');
    const panel = (parent, setting) => parent.append('div').attr('class', 'adaptive-k-panel').each(function() {{
      d3.select(this).append('div').attr('class', 'adaptive-k-panel-title').text(setting.label);
    }});
    settings.forEach(s => d3.select('#adaptive-k-legend-1').append('span').html(`<i class="adaptive-k-swatch" style="background:${{s.color}}"></i>${{s.label}}`));

    function svgFor(container, width=520, height=280) {{
      const w = Math.max(280, Math.min(width, container.node().getBoundingClientRect().width - 14));
      const h = height; const svg = container.append('svg').attr('viewBox', `0 0 ${{w}} ${{h}}`).attr('role','img');
      return {{svg, w, h, iw: w - margin.left - margin.right, ih: h - margin.top - margin.bottom}};
    }}
    function axes(g, x, y, iw, ih, xLabel, yLabel) {{
      g.append('g').attr('class','grid').attr('transform', `translate(0,${{ih}})`).call(d3.axisBottom(x).tickSize(-ih).tickFormat(''));
      g.append('g').attr('class','grid').call(d3.axisLeft(y).tickSize(-iw).tickFormat(''));
      g.append('g').attr('class','axis').attr('transform', `translate(0,${{ih}})`).call(d3.axisBottom(x));
      g.append('g').attr('class','axis').call(d3.axisLeft(y));
      g.append('text').attr('x', iw/2).attr('y', ih + 34).attr('text-anchor','middle').attr('fill','var(--muted)').attr('font-size',11).text(xLabel);
      g.append('text').attr('transform','rotate(-90)').attr('x', -ih/2).attr('y', -35).attr('text-anchor','middle').attr('fill','var(--muted)').attr('font-size',11).text(yLabel);
    }}

    // 1) Histograms with log-scaled bins.
    const histHost = d3.select('#adaptive-k-hist');
    settings.forEach(s => {{
      const box = panel(histHost, s), {{svg, w, h, iw, ih}} = svgFor(box, 520, 275);
      const g = svg.append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);
      const maxK = d3.max(settings.flatMap(x => x.points.map(p => p.k)));
      const edges = [1,2,4,8,16,32,64,128,256,512,1024,2048].filter(x => x <= Math.max(2, maxK * 1.01));
      if (edges[edges.length-1] < maxK) edges.push(Math.pow(2, Math.ceil(Math.log2(maxK))));
      const bins = d3.bin().domain([1, d3.max(edges)]).thresholds(edges.slice(1))(s.points.map(p => p.k));
      const x = d3.scaleLog().domain([1, d3.max(edges)]).range([0, iw]);
      const y = d3.scaleLinear().domain([0, d3.max(bins, b => b.length) || 1]).nice().range([ih,0]);
      axes(g,x,y,iw,ih,'adaptive TopK (log scale)','queries');
      g.selectAll('rect').data(bins).join('rect').attr('x', b => x(Math.max(1,b.x0)) + 1).attr('y', b => y(b.length)).attr('width', b => Math.max(1, x(Math.max(1,b.x1)) - x(Math.max(1,b.x0)) - 2)).attr('height', b => ih-y(b.length)).attr('fill', s.color).attr('opacity', .72).on('mousemove', (e,b) => showTip(e, `K ${{fmt(b.x0)}}–${{fmt(b.x1)}}<br>${{b.length}} queries`)).on('mouseleave', hideTip);
    }});

    // 1b) One bar per query, preserving the saved run order.
    const queryBarHost = d3.select('#adaptive-k-query-bars');
    const allK = settings.flatMap(x => x.points.map(p => p.k));
    settings.forEach(s => {{
      const box = panel(queryBarHost, s), {{svg, w, h, iw, ih}} = svgFor(box, 520, 285);
      const g = svg.append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);
      const points = s.points.map((p, i) => ({{...p, i}}));
      const x = d3.scaleBand().domain(points.map(p => p.i)).range([0, iw]).paddingInner(.08);
      const y = d3.scaleLinear().domain([0, Math.max(1, d3.max(points, p => p.k))]).nice().range([ih, 0]);
      axes(g, x, y, iw, ih, 'query index', 'adaptive TopK');
      g.selectAll('.axis').filter((d, i) => i === 0).selectAll('.tick').filter((d, i) => i % 25 !== 0).remove();
      g.selectAll('rect').data(points).join('rect')
        .attr('x', p => x(p.i)).attr('y', p => y(p.k))
        .attr('width', Math.max(1, x.bandwidth())).attr('height', p => ih - y(p.k))
        .attr('fill', s.color).attr('opacity', .78)
        .on('mousemove', (e,p) => showTip(e, `${{p.q}}<br>adaptive K=${{p.k}}<br>gold pages=${{p.g}}`))
        .on('mouseleave', hideTip);
    }});

    // 2) Query-wise lines: gold page count vs predicted K.
    const goldHost = d3.select('#adaptive-k-gold');
    const queryOrder = settings[0].points.slice().sort((a,b) => d3.ascending(a.g, b.g) || d3.ascending(a.q, b.q)).map(p => p.q);
    settings.forEach(s => {{
      const box = panel(goldHost, s), {{svg, w, h, iw, ih}} = svgFor(box, 520, 290);
      const g = svg.append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);
      const byQuery = new Map(s.points.map(p => [p.q, p]));
      const points = queryOrder.map((q, i) => ({{...byQuery.get(q), i: i + 1}}));
      const maxK = d3.max(settings.flatMap(x => x.points.map(p => p.k)));
      const x = d3.scaleLinear().domain([1, queryOrder.length]).range([0,iw]);
      const y = d3.scaleLog().domain([1, Math.max(2,maxK)]).range([ih,0]);
      axes(g,x,y,iw,ih,'queries sorted by gold page count','TopK (log scale)');
      const line = d3.line().x(p => x(p.i)).y(p => y(p.value));
      const gold = points.map(p => ({{...p, value: p.g}}));
      const adaptive = points.map(p => ({{...p, value: p.k}}));
      const smooth = values => values.map((p, i) => {{
        const lo = Math.max(0, i - 5), hi = Math.min(values.length, i + 6);
        return {{...p, value: d3.median(values.slice(lo, hi), v => v.value)}};
      }});
      g.append('path').datum(gold).attr('fill','none').attr('stroke','#64748b').attr('stroke-opacity',.18).attr('stroke-width',1).attr('d',line);
      g.append('path').datum(adaptive).attr('fill','none').attr('stroke',s.color).attr('stroke-opacity',.18).attr('stroke-width',1).attr('d',line);
      g.append('path').datum(smooth(gold)).attr('fill','none').attr('stroke','#64748b').attr('stroke-width',2.8).attr('stroke-dasharray','6 4').attr('d',line);
      g.append('path').datum(smooth(adaptive)).attr('fill','none').attr('stroke',s.color).attr('stroke-width',2.8).attr('d',line);
      g.selectAll('.gold-point').data(gold).join('circle').attr('class','gold-point').attr('cx',p => x(p.i)).attr('cy',p => y(p.value)).attr('r',2).attr('fill','#64748b').attr('opacity',.28).on('mousemove',(e,p) => showTip(e, `${{p.q}}<br>gold pages=${{p.g}}<br>adaptive K=${{p.k}}`)).on('mouseleave',hideTip);
      g.selectAll('.adaptive-point').data(adaptive).join('circle').attr('class','adaptive-point').attr('cx',p => x(p.i)).attr('cy',p => y(p.value)).attr('r',2.2).attr('fill',s.color).attr('opacity',.38).on('mousemove',(e,p) => showTip(e, `${{p.q}}<br>adaptive K=${{p.k}}<br>gold pages=${{p.g}}<br>max gap=${{fmtGap(p.m)}}`)).on('mouseleave',hideTip);
    }});

    // 3) Query-wise max/min adjacent score gaps.
    const gapHost = d3.select('#adaptive-k-gap');
    settings.forEach(s => {{
      const box = panel(gapHost, s), {{svg, w, h, iw, ih}} = svgFor(box, 520, 290);
      const g = svg.append('g').attr('transform', `translate(${{margin.left}},${{margin.top}})`);
      const points = s.points.map((p, i) => ({{...p, i}}));
      const x = d3.scaleLinear().domain([0, points.length - 1]).range([0,iw]);
      const maxGap = d3.max(points, p => p.m);
      const y = d3.scaleLinear().domain([0, Math.max(0.001, maxGap)]).nice().range([ih,0]);
      axes(g,x,y,iw,ih,'query index','adjacent score gap');
      const maxGold = d3.max(points, p => p.g);
      const yGold = d3.scaleLinear().domain([0, Math.max(1, maxGold)]).nice().range([ih,0]);
      g.append('g').attr('class','axis').attr('transform', `translate(${{iw}},0)`).call(d3.axisRight(yGold));
      g.append('text').attr('transform','rotate(90)').attr('x',ih/2).attr('y',-iw-40).attr('text-anchor','middle').attr('fill','#ea580c').attr('font-size',11).text('gold pages');
      const maxValues = points.map(p => ({{...p, value: p.m}}));
      const minValues = points.map(p => ({{...p, value: p.l}}));
      const goldValues = points.map(p => ({{...p, value: p.g}}));
      const line = d3.line().x(p => x(p.i)).y(p => y(p.value));
      const goldLine = d3.line().x(p => x(p.i)).y(p => yGold(p.value));
      const movingMean = values => values.map((p, i) => {{
        const lo = Math.max(0, i - 5), hi = Math.min(values.length, i + 6);
        return {{...p, value: d3.mean(values.slice(lo, hi), v => v.value)}};
      }});
      g.append('path').datum(movingMean(minValues)).attr('fill','none').attr('stroke','#94a3b8').attr('stroke-width',2.8).attr('stroke-dasharray','6 4').attr('d',line);
      g.append('path').datum(movingMean(maxValues)).attr('fill','none').attr('stroke',s.color).attr('stroke-width',2.8).attr('d',line);
      g.append('path').datum(movingMean(goldValues)).attr('fill','none').attr('stroke','#ea580c').attr('stroke-width',2.8).attr('stroke-dasharray','3 3').attr('d',goldLine);
    }});

    // Summary table.
    const table = d3.select('#adaptive-k-table').append('table').attr('class','adaptive-k-summary');
    table.append('thead').append('tr').selectAll('th').data(['setting','mean K','median K','range K','mean gold','mean max gap','under / equal / over']).join('th').text(d => d);
    table.append('tbody').selectAll('tr').data(settings).join('tr').html(s => `<td><span class="adaptive-k-swatch" style="background:${{s.color}}"></span> ${{s.label}}</td><td>${{fmt(s.summary.meanK)}}</td><td>${{fmt(s.summary.medianK)}}</td><td>${{s.summary.minK}}–${{s.summary.maxK}}</td><td>${{fmt(s.summary.meanGold)}}</td><td>${{fmtGap(s.summary.meanGap)}}</td><td>${{s.summary.underGold}} / ${{s.summary.equalGold}} / ${{s.summary.overGold}}</td>`);
  }})();
  </script>
</div>'''


def main() -> None:
    qrels = load_qrels()
    payload = make_payload(qrels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_HTML.write_text(make_html(payload), encoding="utf-8")
    print(json.dumps({"json": str(OUT_JSON), "html": str(OUT_HTML), "queries": len(qrels)}, indent=2))
    for setting in payload["settings"]:
        print(setting["label"], json.dumps(setting["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
