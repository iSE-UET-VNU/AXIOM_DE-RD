"""A real visual retrieval arm -- images as the embedding layer, no torch, no API.

Every previous visual argument in this ledger was a blocker list. This actually
runs one. Three constraints were worked around rather than cited:

  images not downloaded  -> rendered locally from the source PDFs (PyMuPDF)
  torch 2.2.2 / no GPU   -> ONNX Runtime on CPU (torch dropped x86 macOS after
                            2.2.x, so the pin is hardware, not choice)
  API budget exhausted   -> the model is a free HuggingFace download

Honest expectation, stated before the numbers: CLIP ViT-B/32 sees a page at
224x224, where body text is unreadable, and it is English-trained while these
queries are French. It captures layout and figure gist, not text. This measures
whether *any* visual signal is complementary to the text arm, and builds the
pipeline a document-VLM can drop into. It is not a ColPali reproduction.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
norm = lambda m: m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12, None)


def preprocess(path: Path, size: int) -> np.ndarray:
    """CLIP preprocessing: shortest side to `size`, centre crop, normalise."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = size / min(w, h)
    image = image.resize((max(size, int(round(w * scale))), max(size, int(round(h * scale)))),
                         Image.BICUBIC)
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    image = image.crop((left, top, left + size, top + size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return ((array - MEAN) / STD).transpose(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Xenova/clip-vit-base-patch32")
    parser.add_argument("--images", default="data/work/vidore_physics_page_images")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="data/work/vidore_physics_clip.npz")
    args = parser.parse_args()

    model_path = hf_hub_download(args.repo, "onnx/model.onnx")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.repo)

    image_dir = ROOT / args.images
    files = sorted(image_dir.glob("*.png"))
    keys = [f.stem.replace("__", "::") for f in files]
    print(f"{len(files)} page images; encoding with {args.repo} on CPU...", flush=True)

    dummy = tokenizer(["a"], return_tensors="np", padding=True)
    vectors = []
    for start in range(0, len(files), args.batch):
        batch = np.stack([preprocess(f, args.size) for f in files[start:start + args.batch]])
        out = session.run(["image_embeds"], {
            "pixel_values": batch,
            "input_ids": dummy["input_ids"].astype(np.int64),
            "attention_mask": dummy["attention_mask"].astype(np.int64)})[0]
        vectors.append(out)
        if (start // args.batch) % 10 == 0:
            print(f"  {start + len(batch)}/{len(files)}", flush=True)
    image_vectors = norm(np.concatenate(vectors).astype(np.float32))

    from src.evaluation.benchmarks import load
    bench = load("vidore_v3", subset="physics", language="french")
    qrels = bench.qrels()
    questions = [q for q in bench.questions() if qrels.get(q.qid)]
    blank = np.zeros((1, 3, args.size, args.size), dtype=np.float32)
    qvecs = []
    for start in range(0, len(questions), 64):
        chunk = [q.query for q in questions[start:start + 64]]
        tokens = tokenizer(chunk, return_tensors="np", padding=True,
                           truncation=True, max_length=77)
        out = session.run(["text_embeds"], {
            "pixel_values": blank,
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64)})[0]
        qvecs.append(out)
    query_vectors = norm(np.concatenate(qvecs).astype(np.float32))

    target = ROOT / args.out
    np.savez_compressed(target, image_vectors=image_vectors, query_vectors=query_vectors,
                        keys=np.array(keys), qids=np.array([q.qid for q in questions]))
    print(f"wrote {target}  images {image_vectors.shape}  queries {query_vectors.shape}")


if __name__ == "__main__":
    main()
