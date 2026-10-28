import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ondemand.routing_axes import layout_ocr_score_against_ground_truth

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


OTSL_SRC = "OTSL = re.compile(r\"<(fcel|ecel|lcel|ucel|xcel|nl|ched|rhed|srow)>\")\n\n\n" + inspect.getsource(layout_ocr_score_against_ground_truth.otsl_to_html)

PADDLE_STAGE = r'''
import json
import os
import sys
import time

import fitz
import numpy as np
from PIL import Image

mode, cfg = sys.argv[1], json.load(open(sys.argv[2]))
out = cfg["out"]


def done(path, key):
    return {key(json.loads(l)) for l in open(path)} if os.path.exists(path) else set()


if mode == "orient":
    from paddleocr import DocImgOrientationClassification
    clf = DocImgOrientationClassification(model_name=cfg["orient_model"], device="gpu:0")
    path = f"{out}/orient.jsonl"
    have = done(path, lambda r: r["page_id"])
    with open(path, "a") as f:
        for p in cfg["pages"]:
            if p["page_id"] in have:
                continue
            t0 = time.perf_counter()
            with fitz.open(p["pdf"]) as doc:
                pix = doc.load_page(p["index"]).get_pixmap(dpi=cfg["dpi"], colorspace=fitz.csRGB, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            t1 = time.perf_counter()
            res = next(iter(clf.predict(np.asarray(img)[:, :, ::-1].copy(), batch_size=1)))
            angle, score = int(res["label_names"][0]), float(res["scores"][0])
            t2 = time.perf_counter()
            (img.rotate(angle, expand=True) if angle else img).save(f"{out}/pages/{p['key']}.png")
            f.write(json.dumps({"page_id": p["page_id"], "key": p["key"], "angle": angle, "angle_score": score,
                                "seconds_render": t1 - t0, "seconds_orient": t2 - t1}) + "\n")
            f.flush()

if mode == "ppocr":
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(text_detection_model_name="PP-OCRv5_server_det", text_recognition_model_name="PP-OCRv5_server_rec",
                    use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                    device="gpu:0")
    path = f"{out}/regions_ppocrv5.jsonl"
    have = done(path, lambda r: (r["page_id"], r["region_idx"]))
    started = time.time()
    with open(path, "a") as f:
        for line in open(f"{out}/regions.jsonl"):
            r = json.loads(line)
            if r["dropped"] or (r["page_id"], r["region_idx"]) in have:
                continue
            if time.time() - started > cfg["budget_s"]:
                break
            t0 = time.perf_counter()
            texts = []
            for res in ocr.predict(f"{out}/{r['crop']}"):
                texts.extend(res["rec_texts"])
            f.write(json.dumps({**{k: r[k] for k in ("page_id", "region_idx", "order", "type", "box")},
                                "reader": "ppocrv5", "text": "\n".join(texts), "raw": None,
                                "seconds": time.perf_counter() - t0, "n_tokens": None, "truncated": False},
                               ensure_ascii=False) + "\n")
            f.flush()
'''

HF_STAGE = r'''
import json
import os
import re
import sys
import time

import torch
from PIL import Image

mode, cfg = sys.argv[1], json.load(open(sys.argv[2]))
out = cfg["out"]
dev = "cuda:0"
DTYPES = {"float16": torch.float16, "float32": torch.float32}

__OTSL__


def done(path, key):
    return {key(json.loads(l)) for l in open(path)} if os.path.exists(path) else set()


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


if mode == "layout":
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    proc = AutoImageProcessor.from_pretrained(cfg["layout_model"], revision=cfg["layout_sha"])
    model = AutoModelForObjectDetection.from_pretrained(cfg["layout_model"], revision=cfg["layout_sha"]).to(dev).eval()
    orient = {json.loads(l)["page_id"]: json.loads(l) for l in open(f"{out}/orient.jsonl")}
    have = done(f"{out}/layout.jsonl", lambda r: r["page_id"])
    with open(f"{out}/layout.jsonl", "a") as lf, open(f"{out}/regions.jsonl", "a") as rf:
        for p in cfg["pages"]:
            if p["page_id"] in have or p["page_id"] not in orient:
                continue
            img = Image.open(f"{out}/pages/{p['key']}.png").convert("RGB")
            t0 = sync_time()
            with torch.inference_mode():
                res = proc.post_process_object_detection(model(**proc(images=img, return_tensors="pt").to(dev)),
                                                         target_sizes=[img.size[::-1]])[0]
            t1 = sync_time()
            kept = 0
            for i, (score, label, box) in enumerate(zip(res["scores"], res["labels"], res["boxes"])):
                kind = model.config.id2label[int(label)]
                x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)
                dropped = kind in cfg["drop_types"] or x1 - x0 < 2 or y1 - y0 < 2
                crop = None
                if not dropped:
                    crop = f"crops/{p['key']}_{i:03d}.png"
                    img.crop((x0, y0, x1, y1)).save(f"{out}/{crop}")
                    kept += 1
                rf.write(json.dumps({"page_id": p["page_id"], "region_idx": i, "order": i, "type": kind,
                                     "box": [x0, y0, x1, y1], "score": float(score), "crop": crop,
                                     "dropped": dropped}) + "\n")
            lf.write(json.dumps({**orient[p["page_id"]], "seconds_layout": t1 - t0, "n_regions": len(res["labels"]),
                                 "n_kept": kept, "width": img.width, "height": img.height}) + "\n")
            rf.flush()
            lf.flush()

if mode == "read":
    from transformers import AutoModelForImageTextToText, AutoProcessor
    name = sys.argv[3]
    spec = cfg["readers"][name]

    def load(dtype):
        m = AutoModelForImageTextToText.from_pretrained(spec["model"], revision=spec["sha"], dtype=DTYPES[dtype],
                                                        attn_implementation=cfg["attn"])
        return m.to(dev).eval()

    proc = AutoProcessor.from_pretrained(spec["model"], revision=spec["sha"])
    tok = proc.tokenizer
    keep = set(f"<{t}>" for t in ("fcel", "ecel", "lcel", "ucel", "xcel", "nl", "ched", "rhed", "srow"))
    strip = sorted(set(tok.all_special_tokens) - keep, key=len, reverse=True)
    model = load(cfg["dtype"])

    def kind_of(region_type):
        return "table" if region_type == "table" else "formula" if region_type == "formula" else \
            "seal" if region_type == "seal" and "seal" in spec["prompts"] else "text"

    def read(m, r):
        kind = kind_of(r["type"])
        img = Image.open(f"{out}/{r['crop']}").convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": spec["prompts"][kind]}]}]
        t0 = sync_time()
        inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                          return_tensors="pt").to(dev)
        limit = cfg["max_new"]["table" if kind == "table" else "formula" if kind == "formula" else "text"]
        with torch.inference_mode():
            ids = m.generate(**inputs, max_new_tokens=limit, do_sample=False)
        t1 = sync_time()
        new = ids[0, inputs["input_ids"].shape[1]:]
        raw = tok.decode(new, skip_special_tokens=False)
        for s in strip:
            raw = raw.replace(s, "")
        raw = raw.strip()
        n = int((new != tok.pad_token_id).sum()) if tok.pad_token_id is not None else int(new.numel())
        return kind, raw, t1 - t0, n, n >= limit

    path = f"{out}/regions_{name}.jsonl"
    have = done(path, lambda r: (r["page_id"], r["region_idx"]))
    todo = [json.loads(l) for l in open(f"{out}/regions.jsonl")]
    todo = [r for r in todo if not r["dropped"]]
    started = time.time()
    with open(path, "a") as f:
        for r in todo:
            if (r["page_id"], r["region_idx"]) in have:
                continue
            if time.time() - started > cfg["budget_s"]:
                break
            kind, raw, seconds, n, cut = read(model, r)
            f.write(json.dumps({**{k: r[k] for k in ("page_id", "region_idx", "order", "type", "box")},
                                "reader": name, "prompt": kind, "text": otsl_to_html(raw) if kind == "table" else raw,
                                "raw": raw, "seconds": seconds, "n_tokens": n, "truncated": cut},
                               ensure_ascii=False) + "\n")
            f.flush()
    if cfg["fp32_check"]:
        del model
        torch.cuda.empty_cache()
        ref = load("float32")
        rows = [json.loads(l) for l in open(path)]
        picked = [r for r in rows if r["prompt"] == "text"][:cfg["fp32_check"]] + \
                 [r for r in rows if r["prompt"] == "table"][:cfg["fp32_check"]]
        crops = {(r["page_id"], r["region_idx"]): r for r in todo}
        with open(f"{out}/fp32_{name}.jsonl", "w") as f:
            for r in picked:
                _, raw32, s32, n32, _ = read(ref, crops[(r["page_id"], r["region_idx"])])
                f.write(json.dumps({"page_id": r["page_id"], "region_idx": r["region_idx"], "prompt": r["prompt"],
                                    "fp16": r["raw"], "fp32": raw32, "same": raw32 == r["raw"],
                                    "seconds_fp16": r["seconds"], "seconds_fp32": s32}, ensure_ascii=False) + "\n")
'''.replace("__OTSL__", OTSL_SRC)

md(r'''
# 14 Layout regions + three readers on the QA context pages (Kaggle T4 x2)

Plan `.plans/k14-layout-readers.md`, settings confirmed 2026-10-28. Pages: the 1339 unique top-10 pages of `ungated_evie_m10` on 214 queries (industrial 612, physics 472, OHR 255); smoke: 30 pages (the 9 OHR pages where KDL broke, 10 other OHR, 6 industrial with tables, 5 physics; seed 0). No retrieval change, no QA.

Per page: render 144 dpi -> PP-LCNet orientation (0/90/180/270, rotate only, no deskew) -> PP-DocLayoutV3 regions in its reading order -> image/chart regions dropped, every other region cropped once. All readers read the same crops:
- `paddlevl` PaddleOCR-VL-1.6 (0.9B), prompts `OCR:` / `Table Recognition:` / `Formula Recognition:` / `Seal Recognition:`
- `xiaomi` Xiaomi-OCR-0 (0.8B), prompts from its card (text, table -> OTSL, formula -> LaTeX)
- `ppocrv5` PP-OCRv5 server det + rec on the crop (orientation off: the page is already rotated)

Readers: transformers, fp16 (T4 has no bf16), sdpa, batch 1, greedy, no repetition guard (loops are measured), max new tokens text 1024 / table 2048 / formula 512. OTSL tables -> HTML (`raw` keeps the model output). Smoke also reruns 5 text + 5 table crops per VLM in fp32 (`fp32_<reader>.jsonl`). Timing: `cuda.synchronize` around each call; render, orientation, layout and each reader timed separately.

Environments: Paddle (orientation, PP-OCRv5) in Kaggle's Python; layout + VLM readers in a uv Python 3.13 venv with transformers 5.17.0 / torch 2.14.0 (Xiaomi's verified stack). PaddleOCR-VL runs on GPU 0 and Xiaomi on GPU 1 at the same time.

Input: attach `k3_input` (as notebooks 09-13) and a dataset holding `layout_ocr_pages.json` (repo `kaggle/layout_ocr_pages.json`). To resume a cut run, also attach its `k14_out` zip as a dataset. Settings: GPU T4 x2, Internet ON. Start with `SMOKE = True`.
Output: `k14_out.zip` (folder `k14_out`): `layout.jsonl`, `regions.jsonl`, `regions_{paddlevl,xiaomi,ppocrv5}.jsonl`, `crops/` (needed to resume), `meta.json`; smoke adds `pages/` and `fp32_*.jsonl`.
Local scoring (repo root): `python ondemand/routing_axes/layout_ocr_score_against_ground_truth.py <unzipped>/k14_out`.
''')
md("## 1. Settings")
code(r'''
INPUT_NAME = "pages.json"
PAGES_NAME = "layout_ocr_pages.json"
INPUT_ROOT = "/kaggle/input"
WORK_DIR = "/kaggle/working"
OUT = f"{WORK_DIR}/k14_out"
SMOKE = True
DPI = 144
ORIENT_MODEL = "PP-LCNet_x1_0_doc_ori"
LAYOUT_MODEL = "PaddlePaddle/PP-DocLayoutV3_safetensors"
LAYOUT_SHA = "e4f489dc536556fc1be02b973ba06756e0d4a2ac"
READERS = {
    "paddlevl": {"model": "PaddlePaddle/PaddleOCR-VL-1.6", "sha": "c5630abae1d940eafe0697512a0325494b02ab42",
                 "prompts": {"text": "OCR:", "table": "Table Recognition:", "formula": "Formula Recognition:",
                             "seal": "Seal Recognition:"}},
    "xiaomi": {"model": "SeerRay-Lab/Xiaomi-OCR-0", "sha": "6e75d0cbe7899fea33bbb176febd879eace33429",
               "prompts": {"text": "Extract the text in the image.", "table": "Parse the table in the image into OTSL.",
                           "formula": "Identify the formula in the image and represent it using LATEX format."}},
}
DROP_TYPES = ["image", "chart"]
MAX_NEW = {"text": 1024, "table": 2048, "formula": 512}
DTYPE = "float16"
ATTN = "sdpa"
FP32_CHECK = 5
PADDLE_PIP = ["paddlepaddle-gpu==3.0.0", "-i", "https://www.paddlepaddle.org.cn/packages/stable/cu126/"]
PADDLEOCR_PIP = "paddleocr>=3.0,<4"
HF_PIP = ["transformers==5.17.0", "torch==2.14.0", "torchvision==0.29.0", "accelerate", "pillow", "pymupdf"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
VENV = f"{WORK_DIR}/hfenv"
TIME_BUDGET_H = 11.0
''')
md("## 2. Install (Paddle in Kaggle's Python, transformers stack in a uv venv) and check both GPUs")
code(r'''
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

started = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PADDLE_PIP], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", PADDLEOCR_PIP, "pymupdf"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
if not os.path.exists(f"{VENV}/bin/python"):
    subprocess.run(["uv", "venv", "-q", "--python", "3.13", VENV], check=True)
subprocess.run(["uv", "pip", "install", "-q", "--python", f"{VENV}/bin/python", "--index-url", TORCH_INDEX,
                "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match", *HF_PIP], check=True)
probe = subprocess.run([f"{VENV}/bin/python", "-c", "import torch, transformers; "
                        "print(torch.__version__, transformers.__version__, torch.cuda.device_count(), "
                        "torch.cuda.get_device_name(0), (torch.ones(8, device='cuda') @ torch.ones(8, device='cuda')).item())"],
                       capture_output=True, text=True)
print(probe.stdout, probe.stderr[-2000:])
if probe.returncode:
    raise SystemExit("venv torch cannot use the GPU: change TORCH_INDEX (e.g. .../whl/cu128) and rerun this cell")
TORCH_V, TRANSFORMERS_V, N_GPUS = probe.stdout.split()[0], probe.stdout.split()[1], int(probe.stdout.split()[2])
import paddle
import paddleocr
print("paddle", paddle.__version__, "paddleocr", paddleocr.__version__, "cuda", paddle.device.is_compiled_with_cuda())
''')
code("PADDLE_STAGE = " + repr(PADDLE_STAGE) + "\nHF_STAGE = " + repr(HF_STAGE))
md("## 3. Pages, input and resume")
code(r'''
found = sorted(glob.glob(f"{INPUT_ROOT}/**/{INPUT_NAME}", recursive=True))
found_pages = sorted(glob.glob(f"{INPUT_ROOT}/**/{PAGES_NAME}", recursive=True))
if not found or not found_pages:
    raise SystemExit(f"attach k3_input ({INPUT_NAME}) and the dataset with {PAGES_NAME}")
K3 = os.path.dirname(found[0])
where = {p["page_id"]: p for p in json.load(open(found[0]))}
spec = json.load(open(found_pages[0]))
wanted = spec["smoke"] if SMOKE else spec["pages"]
missing = [p for p in wanted if p not in where]
if missing:
    raise SystemExit(f"{len(missing)} pages not in k3_input, e.g. {missing[:3]}")
pages = [{"page_id": p, "pdf": f"{K3}/pdfs/{where[p]['pdf']}", "index": where[p]["index"],
          "key": hashlib.sha1(p.encode()).hexdigest()[:16]} for p in wanted]
for d in ("pages", "crops"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
prior = [p for p in glob.glob(f"{INPUT_ROOT}/**/k14_out/orient.jsonl", recursive=True) if not p.startswith(OUT)]
if prior and not os.path.exists(f"{OUT}/orient.jsonl"):
    print("resuming from", os.path.dirname(prior[0]))
    shutil.copytree(os.path.dirname(prior[0]), OUT, dirs_exist_ok=True)
cfg = {"out": OUT, "pages": pages, "dpi": DPI, "orient_model": ORIENT_MODEL, "layout_model": LAYOUT_MODEL,
       "layout_sha": LAYOUT_SHA, "readers": READERS, "drop_types": DROP_TYPES, "max_new": MAX_NEW, "dtype": DTYPE,
       "attn": ATTN, "fp32_check": FP32_CHECK if SMOKE else 0}
budget = lambda share: max(60.0, (TIME_BUDGET_H * 3600 - (time.time() - started)) * share)
open(f"{WORK_DIR}/paddle_stage.py", "w").write(PADDLE_STAGE)
open(f"{WORK_DIR}/hf_stage.py", "w").write(HF_STAGE)


def stage(python, script, mode, gpu, *extra, share=1.0, wait=True):
    path = f"{WORK_DIR}/cfg_{mode}_{'_'.join(extra) or 'all'}.json"
    json.dump({**cfg, "budget_s": budget(share)}, open(path, "w"))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    proc = subprocess.Popen([python, f"{WORK_DIR}/{script}", mode, path, *extra], env=env)
    if wait and proc.wait():
        raise SystemExit(f"{mode} {extra} failed ({proc.returncode})")
    return proc


print(len(pages), "pages", "(smoke)" if SMOKE else "")
''')
md("## 4. Render + orientation (Paddle, GPU 0)")
code(r'''
stage(sys.executable, "paddle_stage.py", "orient", 0)
orient = [json.loads(l) for l in open(f"{OUT}/orient.jsonl")]
print(len(orient), "pages; angles", {a: sum(r["angle"] == a for r in orient) for a in (0, 90, 180, 270)})
print("turned:", [(r["page_id"], r["angle"], round(r["angle_score"], 3)) for r in orient if r["angle"]][:20])
''')
md("## 5. Layout (PP-DocLayoutV3, transformers, GPU 0)")
code(r'''
stage(f"{VENV}/bin/python", "hf_stage.py", "layout", 0)
regions = [json.loads(l) for l in open(f"{OUT}/regions.jsonl")]
from collections import Counter
print(len(regions), "regions,", sum(not r["dropped"] for r in regions), "kept;", Counter(r["type"] for r in regions).most_common())


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0, w) * max(0, h)
    return inter / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


by_page = {}
for r in regions:
    if not r["dropped"]:
        by_page.setdefault(r["page_id"], []).append(r["box"])
overlaps = sum(iou(x, y) > 0.5 for bs in by_page.values() for i, x in enumerate(bs) for y in bs[i + 1:])
print("kept box pairs with IoU > 0.5:", overlaps, "on", len(by_page), "pages")
''')
md("## 6. VLM readers (PaddleOCR-VL on GPU 0, Xiaomi-OCR-0 on GPU 1)")
code(r'''
names = list(READERS)
if N_GPUS >= 2:
    procs = [stage(f"{VENV}/bin/python", "hf_stage.py", "read", g, n, share=0.8, wait=False) for g, n in enumerate(names)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"reader exit codes {dict(zip(names, codes))}")
else:
    for k, n in enumerate(names):
        stage(f"{VENV}/bin/python", "hf_stage.py", "read", 0, n, share=0.8 / (len(names) - k))
''')
md("## 7. PP-OCRv5 on the same crops (Paddle, GPU 0)")
code(r'''
stage(sys.executable, "paddle_stage.py", "ppocr", 0)
''')
md("## 8. Summary and save")
code(r'''
kept = {(r["page_id"], r["region_idx"]) for r in regions if not r["dropped"]}
summary = {}
for n in [*READERS, "ppocrv5"]:
    rows = [json.loads(l) for l in open(f"{OUT}/regions_{n}.jsonl")] if os.path.exists(f"{OUT}/regions_{n}.jsonl") else []
    s = [r["seconds"] for r in rows]
    summary[n] = {"regions": len(rows), "left": len(kept) - len(rows), "empty": sum(not r["text"].strip() for r in rows),
                  "truncated": sum(bool(r["truncated"]) for r in rows), "s_per_region": sum(s) / max(1, len(s)),
                  "s_per_page": sum(s) / max(1, len({r["page_id"] for r in rows}))}
    print(n, summary[n])
for n in READERS:
    if os.path.exists(f"{OUT}/fp32_{n}.jsonl"):
        rows = [json.loads(l) for l in open(f"{OUT}/fp32_{n}.jsonl")]
        print(n, "fp16 == fp32 on", sum(r["same"] for r in rows), "of", len(rows), "crops")
layout = [json.loads(l) for l in open(f"{OUT}/layout.jsonl")]
mean = lambda k: sum(r[k] for r in layout) / max(1, len(layout))
print("s/page render", round(mean("seconds_render"), 3), "orient", round(mean("seconds_orient"), 3), "layout", round(mean("seconds_layout"), 3))
json.dump({"smoke": SMOKE, "n_pages": len(pages), "pages_done": len(layout), "dpi": DPI, "orient_model": ORIENT_MODEL,
           "layout_model": LAYOUT_MODEL, "layout_sha": LAYOUT_SHA, "readers": READERS, "drop_types": DROP_TYPES,
           "max_new": MAX_NEW, "dtype": DTYPE, "attn": ATTN, "torch": TORCH_V, "transformers": TRANSFORMERS_V,
           "paddle": paddle.__version__, "paddleocr": paddleocr.__version__, "torch_index": TORCH_INDEX,
           "summary": summary, "n_gpus": N_GPUS, "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                                                      capture_output=True, text=True).stdout.split("\n")[0],
           "hours": round((time.time() - started) / 3600, 2)}, open(f"{OUT}/meta.json", "w"), indent=1)
skip = [] if SMOKE else ["-x", "k14_out/pages/*"]
subprocess.run(["zip", "-qr", f"{WORK_DIR}/k14_out.zip", "k14_out", *skip], cwd=WORK_DIR, check=True)
from IPython.display import FileLink, display
print("k14_out.zip", f"{os.path.getsize(f'{WORK_DIR}/k14_out.zip') / 2**20:.0f} MB")
display(FileLink("k14_out.zip"))
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
