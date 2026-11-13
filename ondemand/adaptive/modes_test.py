import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ondemand.adaptive.modes import Modes, parse_overrides
from ondemand.adaptive.qa_judge import STRUCTURED_PROMPT, fill, format_references, parse_structured
from ondemand.bench import FLAWED_QIDS, gold as load_gold, load_jsonl, work

HEAVY = ("torch", "transformers", "vllm", "xgboost", "sklearn", "qwen_vl_utils", "colpali_engine", "fitz",
         "tiktoken", "openai")
LEGACY = work("results", "xk_evie_v2_stop10_214")
QIDS = [q for q in sorted(load_gold()) if q not in FLAWED_QIDS][:3]


def blocked(code, allow=()):
    names = tuple(n for n in HEAVY if n not in allow)
    prelude = (f"import sys\nBLOCKED = {names!r}\n"
               "class Block:\n"
               "    def find_spec(self, name, path=None, target=None):\n"
               "        if name.split('.')[0] in BLOCKED:\n"
               "            raise ImportError('blocked ' + name)\n"
               "sys.meta_path.insert(0, Block())\n"
               f"sys.path.insert(0, {str(ROOT)!r})\n")
    done = subprocess.run([sys.executable, "-c", prelude + code], cwd=ROOT, capture_output=True, text=True,
                          timeout=1800)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return done.stdout


def last_json(stdout):
    return json.loads(stdout.strip().splitlines()[-1])


def test_imports_without_heavy_packages():
    blocked("import ondemand.adaptive.pipeline, ondemand.adaptive.qa, ondemand.adaptive.__main__\n"
            "import ondemand.adaptive.selector, ondemand.adaptive.embedders, ondemand.adaptive.retrievers\n"
            "import ondemand.adaptive.services.evie, ondemand.adaptive_retrieval.inference\n"
            "from ondemand.adaptive.chunk_rank import shingles, dedupe, FixedChunker, minmax\n"
            "from ondemand.adaptive.embedding import Embedder\n"
            "from ondemand.adaptive.subdata import Passage\n"
            "from ondemand.adaptive.branches.visual import VisualBranch\n"
            "from ondemand.adaptive.services.base import Pages\n"
            "from ondemand.adaptive.qa import cached_or_refuse, cache_path, judge_guarded, generate, generator_key\n"
            "from ondemand.adaptive.pipeline import load_gate, subdata_for\n"
            "from ondemand.adaptive.registry import names\n"
            "for kind in ('query_prep', 'embedder', 'light_retriever', 'adaptive_selector', 'top3_check',\n"
            "             'scorer', 'qa_reader', 'qa_judge'):\n"
            "    assert names(kind), kind\n"
            "print('ok')\n")


def test_modes():
    modes = Modes("offline", parse_overrides(["qa_reader=online", "qa_judge=online"]))
    assert modes("embedder") == "offline" and modes.online("qa_reader") and modes.online("qa_judge")
    assert Modes("online")("visual_encoder") == "online"
    with pytest.raises(SystemExit):
        Modes("offline", {"nonsense": "online"})
    with pytest.raises(SystemExit):
        Modes("offline", {"embedder": "sometimes"})


def test_judge_prompt():
    filled = fill(STRUCTURED_PROMPT, "Q {answer}?", format_references(["a", "b"]), "A")
    assert filled.endswith("Question:\nQ {answer}?\n\nReference answer(s):\n1. a\n2. b\n\nGenerated answer:\nA")
    assert filled.count('{"answer_type":"yes_no|direct_fact|multi_part|explanatory|calculation|unclear",') == 1
    assert fill(STRUCTURED_PROMPT, "{query}", "{gold}", "{answer}") == STRUCTURED_PROMPT
    reply = json.dumps({"answer_type": "direct_fact", "judgment": "Correct", "required_points": ["x"],
                        "covered_points": ["x"], "missing_points": [], "contradicted_points": [],
                        "material_extra_claims": [], "semantic_equivalence_note": "", "explanation": "ok"})
    assert parse_structured(reply)["judgment"] == "Correct"
    with pytest.raises(ValueError):
        parse_structured(reply.replace('"Correct"', '"Mostly right"'))
    with pytest.raises(ValueError):
        parse_structured(json.dumps({"judgment": "Correct"}))


def test_offline_smoke_new_flow():
    with tempfile.TemporaryDirectory() as tmp:
        out = blocked("import json\nfrom ondemand.adaptive import pipeline\n"
                      f"s = pipeline.run('modes_smoke', limit=3, exclude_flawed=True, out_dir={tmp!r})\n"
                      "print(json.dumps({'mean_k': s['mean_k'], 'n': s['n_queries'], 'fallbacks': s['fallbacks'],\n"
                      "                  'light': s['light']['embedder']['model'], 'branch': s['branch_counts']}))\n",
                      allow=("xgboost", "sklearn"))
        summary = last_json(out)
        records = load_jsonl(Path(tmp) / "per_query.jsonl")
        manifest = json.loads((Path(tmp) / "manifest.json").read_text())
    assert summary["n"] == 3 and summary["fallbacks"] == {} and summary["branch"] == {"visual": 3}
    assert summary["light"] == "qwen/qwen3-embedding-8b"
    assert [r["query_id"] for r in records] == QIDS
    assert all(len(r["branch_pages"]) == 10 and 1 <= r["k"] <= 100 for r in records)
    assert all(r["branch_notes"]["scorer"].startswith("evie:cached") for r in records)
    assert manifest["settings"]["modes"]["visual_encoder"] == "offline"
    assert manifest["settings"]["light"]["signals"] == "lang"
    assert manifest["settings"]["light"]["file_mode"] == "blended_pool"


def test_legacy_te3s_hier_matches_gate():
    out = blocked("import json\nfrom ondemand.adaptive.modes import Modes\n"
                  "from ondemand.adaptive.pipeline import build_light\n"
                  "from ondemand.adaptive.query_prep import NoPrep\n"
                  "from ondemand.adaptive.retrievers import load_gate\n"
                  "from ondemand.bench import BENCH, queries\n"
                  "stages = {'light': 'hier', 'embedder': 'te3s', 'signals': 'all', 'file_mode': 'dense_pool',\n"
                  "          'w_dense': 0.70, 'parent': 0.15}\n"
                  f"qids = {QIDS!r}\n"
                  "light = build_light(stages, Modes('offline'), BENCH, 'pages_ppocrv5.jsonl')\n"
                  "light.prepare(qids)\n"
                  "gate = load_gate('gate_ppocrv5_ungated_xgbK214.json')\n"
                  "q = queries()\n"
                  "same = [light.retrieve(i, NoPrep().prepare(q[i]['query']))[0].pages == gate['light_run_top100'][i]\n"
                  "        for i in qids]\n"
                  "print(json.dumps(same))\n")
    assert last_json(out) == [True, True, True]


@pytest.mark.parametrize("scorer", ["evie_cached:skip_blank", "evie:skip_blank"])
def test_legacy_retrieval_regression(scorer):
    legacy = {r["query_id"]: [p["unit"] for p in r["branch_pages"]] for r in load_jsonl(LEGACY / "per_query.jsonl")}
    with tempfile.TemporaryDirectory() as tmp:
        blocked("from ondemand.adaptive import pipeline\n"
                f"pipeline.run('legacy_check', light='gate:gate_ppocrv5_ungated_xgbK214.json', selector='gate_k',\n"
                f"             branch_args={{'visual': 'visual:{scorer}+ppocrv5+stop:10'}}, limit=3,\n"
                f"             exclude_flawed=True, out_dir={tmp!r})\n")
        records = load_jsonl(Path(tmp) / "per_query.jsonl")
    assert [r["query_id"] for r in records] == QIDS
    assert all([p["unit"] for p in r["branch_pages"]] == legacy[r["query_id"]] for r in records)


def test_legacy_qa_from_cache_only():
    labels = {r["query_id"]: r["judgment"] for r in load_jsonl(LEGACY / "qa.jsonl")}
    with tempfile.TemporaryDirectory() as tmp:
        rows = [r for r in load_jsonl(LEGACY / "per_query.jsonl") if r["query_id"] in QIDS]
        (Path(tmp) / "per_query.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        blocked("from ondemand.adaptive.modes import Modes\nfrom ondemand.adaptive.qa import run\n"
                f"run('legacy_qa', reader='deepseek_v4_flash', judge='legacy_4omini', modes=Modes('offline'),\n"
                f"    out_dir={tmp!r})\n")
        got = {r["query_id"]: r["judgment"] for r in load_jsonl(Path(tmp) / "qa_deepseek_v4_flash__legacy_4omini.jsonl")}
    assert got == {q: labels[q] for q in QIDS}
