import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import fitz
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw

from ondemand.bench import BENCH, doc_of, gold as load_gold, group_of, load_jsonl, queries as load_queries, work
from ondemand.evaluate import as_run, evaluate
from ondemand.text import enfr, real_text

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT / "data/work/ondemand_v2/ef7f11811803/upload/light_ocr_bundle.zip"
RESULTS = next((p for p in (ROOT / "light_ocr_results_comparison_btw_ppocr.zip",
                            ROOT / "light_ocr_results_with_rotation.zip",
                            ROOT / "light_ocr_results.zip") if p.exists()), None)
GATES = {"ppocrv5 gate": "gate_k20_ppocr.json", "tesseract gate": "gate_k20_best.json"}
PAGES = {"gate_k20_ppocr.json": "pages_ocr_ppocr.jsonl", "gate_k20_best.json": "pages_ocr_sparse.jsonl"}
OCR_ENGINES = ("ppocrv5", "ppocrv5_rot", "ppocrv5_doc", "easyocr", "easyocr_rot")
K = 20


def as_text(x):
    if isinstance(x, (list, tuple)):
        return " ".join(as_text(i) for i in x)
    return "" if x is None else str(x)


def tokens(x):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", as_text(x)).casefold())


@st.cache_data(show_spinner=False)
def gate_of(name):
    return json.loads((work("light_prep") / name).read_text())


@st.cache_data(show_spinner=False)
def pages_of(name):
    return {r["page_id"]: r for r in load_jsonl(work("light_prep") / name)}


@st.cache_data(show_spinner=False)
def kdl_text():
    return {r["page_id"]: r["text"] for r in load_jsonl(work("kdl") / "kdl_pages.jsonl")}


@st.cache_data(show_spinner=False)
def labels():
    return load_gold(BENCH), load_queries(BENCH)


@st.cache_data(show_spinner=False)
def per_query(name):
    gate, (gold, queries) = gate_of(name), labels()
    run = {q: as_run(top) for q, top in gate["light_run_top100"].items()}
    _, _, at10 = evaluate(run, gold, 10)
    _, _, at20 = evaluate(run, gold, 20)
    rows = []
    for q, top in gate["gate"].items():
        positives = [u for u, g in gold[q].items() if g > 0]
        ranks = gate["gold_rank"][q]
        hit = [u for u in positives if u in top]
        rows.append({"query_id": q, "group": group_of(q, queries[q]), "query": queries[q]["query"],
                     "gold_pages": len(positives), "in_gate": len(hit),
                     "recall@20": 100 * at20[q]["recall_20"], "ndcg@10": 100 * at10[q]["ndcg_cut_10"],
                     "file_in_gate": any(doc_of(u) in {doc_of(t) for t in top} for u in positives),
                     "best_gold_rank": min([ranks[u] for u in positives if ranks.get(u)], default=None),
                     "missed": [u for u in positives if u not in top]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def page_image(unit, name):
    page = pages_of(name)[unit]
    with fitz.open(BENCH / page["relative_path"]) as doc:
        pixmap = doc[page["page_index"]].get_pixmap(dpi=110)
    return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")


@st.cache_data(show_spinner=False)
def ocr_results():
    texts, boxes, meta = {}, {}, {}
    if RESULTS is None:
        return texts, boxes, meta
    with zipfile.ZipFile(RESULTS) as zf:
        names = set(zf.namelist())
        for engine in OCR_ENGINES:
            if f"light_ocr_{engine}.jsonl" in names:
                texts[engine] = {r["unit"]: r["text"] for r in
                                 map(json.loads, zf.open(f"light_ocr_{engine}.jsonl").read().decode().splitlines())}
            if f"light_ocr_boxes_{engine}.jsonl" in names:
                boxes[engine] = {r["unit"]: r["lines"] for r in
                                 map(json.loads, zf.open(f"light_ocr_boxes_{engine}.jsonl").read().decode().splitlines())}
            if f"meta_{engine}.json" in names:
                meta[engine] = json.loads(zf.open(f"meta_{engine}.json").read())
    return texts, boxes, meta


def stage(row):
    if row["in_gate"] == row["gold_pages"]:
        return "all gold in gate"
    if row["in_gate"] > 0:
        return "some gold in gate"
    if row["file_in_gate"]:
        return "right file, no gold page"
    return "file missed"


st.set_page_config(page_title="Light retrieval failure portal", layout="wide")
mode = st.sidebar.radio("Mode", ("Retrieval failures", "OCR engines"))
available = {label: name for label, name in GATES.items() if (work("light_prep") / name).exists()}
label = st.sidebar.selectbox("Gate", list(available))
name = available[label]
other = next((n for n in available.values() if n != name), None)

if mode == "Retrieval failures":
    gate = gate_of(name)
    table = per_query(name).assign(stage=lambda d: d.apply(stage, axis=1))
    if other:
        before = per_query(other).set_index("query_id")["recall@20"]
        table["delta_recall@20"] = table["recall@20"] - table["query_id"].map(before)

    st.title("Light retrieval failures")
    st.caption(f"{label} · pages `{gate.get('pages_source', '?')}` · k={gate['gate_k']} · "
               f"R@20 {gate['metrics']['all']['light_recall@20']} · NDCG@10 {gate['metrics']['all']['light_ndcg@10']} · "
               f"file recall {gate['metrics']['all']['file_recall@20']}")

    metrics = pd.DataFrame(gate["metrics"]).T.reset_index(names="group")
    st.dataframe(metrics, hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        counts = table.groupby(["group", "stage"]).size().reset_index(name="queries")
        st.plotly_chart(px.bar(counts, x="group", y="queries", color="stage", title="Where each query stands"),
                        width="stretch")
    with right:
        lost = (table.assign(lost=lambda d: (100 - d["recall@20"]) / len(d))
                .groupby("group")["lost"].sum().reset_index())
        st.plotly_chart(px.bar(lost, x="group", y="lost", title="R@20 points lost per group"), width="stretch")

    groups = st.sidebar.multiselect("Groups", sorted(table["group"].unique()), default=sorted(table["group"].unique()))
    stages = st.sidebar.multiselect("Stage", sorted(table["stage"].unique()),
                                    default=[s for s in table["stage"].unique() if s != "all gold in gate"])
    view = table[table["group"].isin(groups) & table["stage"].isin(stages)].sort_values("recall@20")
    st.subheader(f"{len(view)} queries")
    st.dataframe(view.drop(columns=["missed"]).round(1), hide_index=True, width="stretch", height=260)
    if view.empty:
        st.stop()

    chosen = st.selectbox("Query", view["query_id"].tolist(),
                          format_func=lambda q: f"{view.set_index('query_id').loc[q, 'recall@20']:.0f}%  {view.set_index('query_id').loc[q, 'query'][:90]}")
    row = view.set_index("query_id").loc[chosen]
    gold, queries = labels()
    meta = queries[chosen]["metadata"]
    st.markdown(f"**{row['query']}**")
    st.caption(f"answers: {as_text(queries[chosen].get('answers'))} · group {row['group']} · stage {row['stage']} · "
               f"{row['in_gate']}/{row['gold_pages']} gold pages in the gate")
    if meta.get("evidence_context"):
        st.info(as_text(meta["evidence_context"]))

    pages, kdl = pages_of(PAGES[name]), kdl_text()
    query_terms = set(enfr(row["query"]))
    top = gate["gate"][chosen]
    scores = dict(zip(top, gate["gate_scores"][chosen]))
    ranks = gate["gold_rank"][chosen]

    def describe(unit, tag):
        page = pages.get(unit, {})
        text = real_text(page.get("text", ""))
        overlap = sorted(set(enfr(text)) & query_terms)
        return {"page": unit, "what": tag, "rank": ranks.get(unit) if tag.startswith("gold") else (top.index(unit) + 1 if unit in top else None),
                "score": round(scores.get(unit, float("nan")), 4) if unit in scores else None,
                "chars": len(text), "source": page.get("source_path") and page.get("text_source"),
                "query terms on page": len(overlap), "terms": ", ".join(overlap[:12])}

    missed = list(row["missed"])
    found = [u for u in gold[chosen] if gold[chosen][u] > 0 and u in top]
    summary = ([describe(u, "gold missed") for u in missed] + [describe(u, "gold in gate") for u in found]
               + [describe(u, "gate page") for u in top[:5]])
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch", height=260)

    units = missed + found + top[:5]
    unit = st.selectbox("Inspect page", units, format_func=lambda u: f"{'MISSED ' if u in missed else ''}{u}")
    image_column, text_column = st.columns([1, 1])
    with image_column:
        st.image(page_image(unit, PAGES[name]), width="stretch")
    with text_column:
        page = pages.get(unit, {})
        st.caption(f"text_source: {page.get('text_source')} · ocr words {page.get('ocr_word_count')} · "
                   f"ocr confidence {page.get('ocr_mean_confidence')} · rank {ranks.get(unit) or (top.index(unit) + 1 if unit in top else '>200')}")
        gate_text, gold_text = real_text(page.get("text", "")), real_text(kdl.get(unit, ""))
        st.markdown("**query terms present:** " + (", ".join(sorted(set(enfr(gate_text)) & query_terms)) or "none"))
        tab_gate, tab_kdl = st.tabs(["text the gate saw", "KDL text"])
        tab_gate.text(gate_text or "(empty)")
        tab_kdl.text(gold_text or "(empty)")
else:
    texts, boxes, meta = ocr_results()
    if not texts:
        st.error("No OCR results zip found")
        st.stop()
    bundle_pages = [json.loads(line) for line in zipfile.ZipFile(BUNDLE).open("pages.jsonl").read().decode().splitlines()]
    angles = {}
    with zipfile.ZipFile(RESULTS) as zf:
        if "angles_ppocrv5_rot.json" in zf.namelist():
            angles = json.loads(zf.open("angles_ppocrv5_rot.json").read())

    st.title("OCR engines on the 297 weak pages")
    systems = {"tesseract": {p["unit"]: p["tesseract_text"] for p in bundle_pages},
               "kdl": {p["unit"]: p["kdl_text"] for p in bundle_pages}, **texts}
    MAIN = [e for e in ("tesseract", "ppocrv5_doc", "kdl") if e in systems]
    if not st.sidebar.checkbox("show orientation variants", value=False):
        systems = {e: systems[e] for e in MAIN}
    pairs = [(p["unit"], e) for p in bundle_pages for e in p["evidence"]]
    rows = []
    for engine, text in systems.items():
        for unit, e in pairs:
            have, want = set(tokens(text.get(unit, ""))), set(tokens(e["evidence_context"]))
            answers = e["answers"] if isinstance(e["answers"], (list, tuple)) else [e["answers"]]
            flat = " ".join(tokens(text.get(unit, "")))
            rows.append({"engine": engine, "unit": unit, "rotation": angles.get(unit, 0) or 0,
                         "evidence_recall": 100 * len(want & have) / max(len(want), 1),
                         "answer_found": any(tokens(a) and " ".join(tokens(a)) in flat for a in answers)})
    scored = pd.DataFrame(rows)
    scored["orientation"] = scored["rotation"].map(lambda a: "rotated" if a else "upright")
    pivot = (scored.groupby(["engine", "orientation"])["evidence_recall"].mean().unstack().round(1)
             .join(scored.groupby("engine")["evidence_recall"].mean().round(1).rename("all")))
    st.dataframe(pivot.reset_index(), hide_index=True, width="stretch")
    st.plotly_chart(px.bar(scored.groupby(["engine", "orientation"])["evidence_recall"].mean().reset_index(),
                           x="engine", y="evidence_recall", color="orientation", barmode="group"), width="stretch")

    engines = st.sidebar.multiselect("Engines", list(systems), default=list(systems))
    units = sorted({u for u, _ in pairs})
    unit = st.selectbox("Page", units, format_func=lambda u: f"{'rotated ' if angles.get(u) else ''}{u}")
    evidence = [e for p in bundle_pages if p["unit"] == unit for e in p["evidence"]]
    _, all_queries = labels()
    for e in evidence:
        question = all_queries.get(e["query_id"], {}).get("query", "")
        st.info(f"**{question}**\n\nanswers: {as_text(e['answers'])}\n\nevidence: {as_text(e['evidence_context'])}\n\n"
                f"rotation detected: {angles.get(unit, 0) or 0}°")
    want = set().union(*[set(tokens(e["evidence_context"])) for e in evidence]) if evidence else set()
    columns = st.columns(max(len(engines), 1))
    for column, engine in zip(columns, engines):
        with column:
            st.subheader(engine)
            text = systems[engine].get(unit, "")
            lines = boxes.get(engine, {}).get(unit, [])
            if lines:
                image = page_image(unit, "pages_ocr_ppocr.jsonl").copy()
                scale = image.width / max(max(x for l in lines for x, _ in l["poly"]), 1)
                draw = ImageDraw.Draw(image)
                for line in lines:
                    points = [(x * scale, y * scale) for x, y in line["poly"]]
                    hit = bool(set(tokens(line["text"])) & want)
                    draw.line(points + [points[0]], fill=(34, 160, 78) if hit else (200, 60, 60), width=2)
                st.image(image, width="stretch")
            else:
                st.image(page_image(unit, "pages_ocr_ppocr.jsonl"), width="stretch")
            st.markdown("**found:** " + (", ".join(sorted(want & set(tokens(text)))) or "none"))
            st.markdown("**missed:** " + (", ".join(sorted(want - set(tokens(text)))) or "none"))
            st.text(text or "(empty)")
