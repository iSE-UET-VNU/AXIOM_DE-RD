import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ondemand.bench import work

W = work()
K11 = W / "routing_axes/k11/k11_out"
OUT = W / "routing_axes/k13"
CHECK, MIN_CORR = 100, 0.9999


def keys():
    return json.loads((W / "colvec/colvec_keys.json").read_text())


def k11_pages():
    return json.loads((K11 / "pool_page_ids.json").read_text())


def merge(folder):
    old, new = np.load(K11 / "scores_v2.npy"), np.load(folder / "scores_v2.npy")
    old_col = {p: i for i, p in enumerate(k11_pages())}
    new_col = {p: i for i, p in enumerate(json.loads((folder / "pool_page_ids.json").read_text()))}
    meta = json.loads((folder / "meta_v2.json").read_text())
    if meta["smoke"] or any(meta["left"].values()):
        raise SystemExit(f"smoke or unfinished run (left {meta['left']}); not merging")
    check = sorted(set(old_col) & set(new_col))
    a = old[:, [old_col[p] for p in check]].ravel()
    b = new[:, [new_col[p] for p in check]].ravel()
    corr, diff = float(np.corrcoef(a, b)[0, 1]), float(np.abs(a - b).max())
    print(f"check pages {len(check)}: corr {corr:.6f}, max abs diff {diff:.4f}")
    if len(check) != CHECK or corr < MIN_CORR:
        raise SystemExit("K13 does not reproduce K11 V2 on the check pages; not merging")
    order = keys()
    merged = np.stack([old[:, old_col[p]] if p in old_col else new[:, new_col[p]] for p in order], axis=1)
    if np.isnan(merged).any():
        raise SystemExit(f"{int(np.isnan(merged).any(0).sum())} pages have no V2 score; not merging")
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "scores_v2_all.npy", merged.astype(np.float32))
    (OUT / "meta_v2_all.json").write_text(json.dumps({"columns": "colvec_keys.json", "shape": list(merged.shape),
                                                      "from_k11": len(old_col), "from_k13": len(order) - len(old_col),
                                                      "check_pages": len(check), "check_corr": corr,
                                                      "check_max_abs_diff": diff, "k13_meta": meta}, indent=1))
    print(f"{merged.shape} -> {OUT / 'scores_v2_all.npy'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    merge(ap.parse_args().folder)
