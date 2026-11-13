import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

import numpy as np

from ...bench import BENCH, work
from .base import JsonlCache, Pages, Service, gpu_info, release_gpu

MODEL = "tencent/EVIE-4.5B"
EVIE_REPO = "https://github.com/Tencent/EVIE.git"
EVIE_PIN = "70b8a2be14de7285970a05ac0f670f768abad966"
TRANSFORMERS = "5.13.1"
DPI, SCORE_DIM, MAX_VISUAL_TOKENS = 144, 2048, 1024
CHECK_MIN_COS = 0.999


def square_rows(batch):
    n = int(batch["image_grid_thw"][0].prod())
    return batch["pixel_values"][0][:n].reshape(n // 4, 4, 3, -1)


def blank_of(batch):
    px = square_rows(batch)
    return (px == px[:, :1, :, :1]).all(3).all(2).all(1).cpu().numpy()


class EVIE(Service):
    name = "evie"
    modes = ("cached", "local")

    def __init__(self, mode="cached", bench=BENCH, evie_dir=None):
        super().__init__(mode, bench)
        self.evie_dir = evie_dir or os.getenv("EVIE_DIR")
        self.pages = Pages(bench)
        self.model = self.processor = None
        self.query_emb = {}
        folder = self.folder()
        self.scores = JsonlCache(folder / "scores.jsonl", lambda r: (r["qid"], r["page_id"]))
        self.encoded = JsonlCache(folder / "pages.jsonl", lambda r: r["page_id"])
        self.matrix = self._matrix()

    def settings(self):
        return {"model": MODEL, "evie_commit": EVIE_PIN, "variant": "v2_skip_blank_squares", "dpi": DPI,
                "score_dim": SCORE_DIM, "max_visual_tokens": MAX_VISUAL_TOKENS, "dtype": "float16", "attn": "sdpa"}

    def _matrix(self):
        folder = work("routing_axes", "k13", bench=self.bench)
        keys = work("colvec", bench=self.bench)
        meta_path, scores_path = folder / "meta_v2_all.json", folder / "scores_v2_all.npy"
        if not (meta_path.exists() and scores_path.exists()):
            self.log.warning("no EVIE V2 corpus matrix in %s; cached mode reads only the live score cache", folder)
            return None
        meta = json.loads(meta_path.read_text())["k13_meta"]
        if meta.get("evie_commit") != EVIE_PIN:
            self.log.warning("the EVIE V2 matrix was built at commit %s, not %s; ignoring it",
                             meta.get("evie_commit"), EVIE_PIN)
            return None
        return {"scores": np.load(scores_path, mmap_mode="r"),
                "col": {k: i for i, k in enumerate(json.loads((keys / "colvec_keys.json").read_text()))},
                "row": {q: i for i, q in enumerate(json.loads((keys / "colvec_qids.json").read_text()))}}

    def load(self):
        if self.loaded or self.mode != "local":
            self.loaded = True
            return
        info = gpu_info()
        if not info["cuda"]:
            raise SystemExit("evie local needs a CUDA GPU; run visual_encoder offline here")
        if not self.evie_dir or not os.path.isdir(os.path.join(self.evie_dir, "colpali")):
            raise SystemExit(f"set EVIE_DIR to a clone of {EVIE_REPO} at {EVIE_PIN} (colab/setup_gpu.sh does this)")
        try:
            found = version("transformers")
        except PackageNotFoundError:
            found = None
        if found != TRANSFORMERS:
            self.log.warning("transformers %s is installed; the cached EVIE V2 scores used %s", found, TRANSFORMERS)
        sys.path.insert(0, os.path.join(self.evie_dir, "colpali"))
        import torch
        from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
        started = perf_counter()
        self.processor = ColQwen3_5Processor.from_pretrained(MODEL, max_num_visual_tokens=MAX_VISUAL_TOKENS)
        self.model = ColQwen3_5.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda:0",
                                                attn_implementation="sdpa").eval()
        self.model.enable_bidirectional_attention()
        seconds = perf_counter() - started
        self.timings().record("evie_model_load", "model", 1, seconds, model=MODEL)
        self.log.info("loaded %s on %s in %.0fs", MODEL, info["name"], seconds)
        self.loaded = True

    def close(self):
        if self.model is not None:
            self.model = self.processor = None
            release_gpu()
        super().close()

    def encode(self, batch):
        import torch
        batch = batch.to(self.model.device)
        self.model.rope_deltas = None
        with torch.no_grad():
            out = self.model(**batch)
        return out[SCORE_DIM][0][batch["attention_mask"][0].bool()]

    def encode_kept(self, batch, keep, early=True):
        import torch
        from transformers.vision_utils import get_vision_bilinear_indices_and_weights, get_vision_position_ids
        model = self.model
        batch = batch.to(model.device)
        ids, am, grid = batch["input_ids"], batch["attention_mask"], batch["image_grid_thw"]
        vis = model.visual
        pv = batch["pixel_values"][0][:int(grid.prod())].type(vis.dtype)
        img = ids[0] == model.config.image_token_id
        keep = torch.as_tensor(keep, device=ids.device)
        with torch.no_grad():
            emb = model.get_input_embeddings()(ids)
            feats = None
            if early:
                rows = keep.repeat_interleave(vis.spatial_merge_unit)
                n = int(rows.sum())
                if n:
                    idx, w = get_vision_bilinear_indices_and_weights(grid, num_grid_per_side=vis.num_grid_per_side,
                                                                     spatial_merge_size=vis.spatial_merge_size)
                    pos = get_vision_position_ids(grid, vis.spatial_merge_size)
                    feats = vis(pv[rows], grid_thw=grid,
                                cu_seqlens=torch.tensor([0, n], dtype=torch.int32, device=ids.device),
                                position_ids=pos[rows], bilinear_indices=idx[:, rows], bilinear_weights=w[:, rows],
                                return_dict=True).pooler_output
            else:
                feats = vis(pv, grid_thw=grid, return_dict=True).pooler_output[keep]
            at = img.nonzero()[:, 0][keep]
            if len(at):
                emb[0, at] = feats.to(emb.dtype)
            tok = ~img
            tok[at] = True
            pos3, _ = model.get_rope_index(ids, mm_token_type_ids=batch["mm_token_type_ids"], image_grid_thw=grid,
                                           attention_mask=am)
            hidden = model.language_model(input_ids=None, inputs_embeds=emb[:, tok], attention_mask=am[:, tok],
                                          position_ids=pos3[:, :, tok], use_cache=False).last_hidden_state
            out = model.custom_text_proj(hidden)[0, :, :SCORE_DIM]
        return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def query_embedding(self, key, query):
        if key not in self.query_emb:
            self.query_emb[key] = self.encode(self.processor.process_queries([query]))
        return self.query_emb[key]

    def encode_page(self, page_id):
        import torch
        started = perf_counter()
        batch = self.processor.process_images([self.pages.image(page_id, DPI)])
        rendered = perf_counter()
        blank = blank_of(batch)
        emb = self.encode_kept(batch, ~blank, early=True)
        torch.cuda.synchronize()
        return emb, {"render_seconds": round(rendered - started, 6),
                     "encode_seconds": round(perf_counter() - rendered, 6), "squares": int(len(blank)),
                     "kept": int((~blank).sum()), "tokens": int(emb.shape[0])}

    def lookup(self, key, page_id):
        row = self.scores.get((key, page_id))
        if row is not None:
            return row["score"]
        if self.matrix and key in self.matrix["row"] and page_id in self.matrix["col"]:
            return float(self.matrix["scores"][self.matrix["row"][key], self.matrix["col"][page_id]])
        return None

    def score(self, key, query, pages):
        need = self.pending([(key, p) for p in pages], lambda k: self.lookup(*k))
        if need and self.mode == "local":
            self.load()
            q = self.query_embedding(key, query)
            for _, page in need:
                emb, notes = self.encode_page(page)
                value = float((q @ emb.to(q.dtype).T).max(dim=1).values.float().sum())
                self.scores.put({"qid": key, "page_id": page, "score": value})
                if page not in self.encoded:
                    self.encoded.put({"page_id": page, **notes})
                self.timings().record("evie_page", "page", 1, notes["render_seconds"] + notes["encode_seconds"],
                                      unit_id=page, **notes)
        found = {p: self.lookup(key, p) for p in pages}
        return {p: s for p, s in found.items() if s is not None}, [p for p, s in found.items() if s is None]

    def page_seconds(self, pages):
        rows = [self.encoded.get(p) for p in pages]
        return round(sum(r["render_seconds"] + r["encode_seconds"] for r in rows if r), 6)

    def verify(self, items, check_pages=5):
        import torch
        self.load()
        pages = sorted({p for _, _, ps in items for p in ps})[:check_pages]
        checks = []
        for page in pages:
            batch = self.processor.process_images([self.pages.image(page, DPI)])
            blank = blank_of(batch)
            ref = self.encode(batch).float()
            got = self.encode_kept(batch, np.ones(len(blank), bool), early=True).float()
            cos = float(torch.nn.functional.cosine_similarity(ref, got, dim=-1).min()) \
                if got.shape == ref.shape else float("nan")
            v2 = self.encode_kept(batch, ~blank, early=True)
            checks.append({"page_id": page, "tokens": int(ref.shape[0]), "keep_all_min_cos": cos,
                           "v2_tokens": int(v2.shape[0]), "blank": int(blank.sum())})
        keep_all_ok = all(c["keep_all_min_cos"] >= CHECK_MIN_COS and c["v2_tokens"] == c["tokens"] - c["blank"]
                          for c in checks)
        live, ref = [], []
        if self.matrix:
            for qid, query, ps in items:
                for page in ps:
                    if qid in self.matrix["row"] and page in self.matrix["col"]:
                        q = self.query_embedding(qid, query)
                        emb, _ = self.encode_page(page)
                        live.append(float((q @ emb.to(q.dtype).T).max(dim=1).values.float().sum()))
                        ref.append(float(self.matrix["scores"][self.matrix["row"][qid], self.matrix["col"][page]]))
        result = {"keep_all": checks, "keep_all_ok": keep_all_ok, "n_pairs": len(live)}
        if len(live) > 1:
            live, ref = np.array(live), np.array(ref)
            result.update({"pearson": float(np.corrcoef(live, ref)[0, 1]),
                           "max_abs_diff": float(np.abs(live - ref).max())})
            result["reproduces"] = keep_all_ok and result["pearson"] >= 0.999
        self.log.info("evie verify: %s", json.dumps(result))
        return result
