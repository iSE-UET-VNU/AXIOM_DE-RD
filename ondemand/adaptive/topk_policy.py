"""Import an offline Adaptive-K policy into an on-demand retrieval gate.

The XGBoost experiments are deliberately kept out of the live runtime.  A
validated experiment exports one integer K per query; this module validates
that lookup and attaches it to a gate consumed by ``AdaptiveK``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCHEMA_VERSION = 1


def _as_k(value, qid):
    try:
        number = float(value)
        k = int(number)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid K for {qid!r}: {value!r}") from error
    if number != k or k < 1:
        raise ValueError(f"K for {qid!r} must be a positive integer, got {value!r}")
    return k


def predictions_from_csv(path, qid_column="qid", k_column="chosen_k"):
    predictions = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {qid_column, k_column} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, 2):
            qid = (row.get(qid_column) or "").strip()
            if not qid:
                raise ValueError(f"empty qid at {path}:{row_number}")
            if qid in predictions:
                raise ValueError(f"duplicate qid {qid!r} at {path}:{row_number}")
            predictions[qid] = _as_k(row.get(k_column), qid)
    if not predictions:
        raise ValueError(f"{path} has no predictions")
    return predictions


def build_policy(predictions, *, name, source_run=None, metrics=None, notes=None):
    values = list(predictions.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_type": "precomputed_adaptive_k",
        "name": name,
        "n_queries": len(predictions),
        "mean_k": sum(values) / len(values),
        "min_k": min(values),
        "max_k": max(values),
        "predictions": dict(sorted(predictions.items())),
    }
    if source_run:
        payload["source_run"] = source_run
    if metrics:
        payload["metrics"] = metrics
    if notes:
        payload["notes"] = notes
    return payload


def load_policy(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported policy schema: {payload.get('schema_version')!r}")
    if payload.get("policy_type") != "precomputed_adaptive_k":
        raise ValueError(f"unsupported policy type: {payload.get('policy_type')!r}")
    raw = payload.get("predictions")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("policy predictions must be a non-empty object")
    payload["predictions"] = {str(qid): _as_k(k, qid) for qid, k in raw.items()}
    if payload.get("n_queries") != len(payload["predictions"]):
        raise ValueError("policy n_queries does not match its prediction count")
    return payload


def ranking_for(gate, qid):
    """Return the deepest ordered page list available for a query."""
    for field in ("light_run_top200", "light_run_top100", "gate"):
        rows = gate.get(field, {}).get(qid)
        if rows:
            return rows, field
    return [], None


def apply_policy(gate, policy, *, allow_subset=False, allow_cap=False):
    predictions = policy["predictions"]
    gate_qids = set(gate.get("gate", {}))
    policy_qids = set(predictions)
    missing = sorted(gate_qids - policy_qids)
    extra = sorted(policy_qids - gate_qids)
    if extra:
        raise ValueError(f"policy contains {len(extra)} qids absent from gate, e.g. {extra[:3]}")
    if missing and not allow_subset:
        raise ValueError(f"policy misses {len(missing)} gate qids, e.g. {missing[:3]}")

    applied, capped, depths = {}, {}, {}
    for qid, requested_k in predictions.items():
        ranking, field = ranking_for(gate, qid)
        depth = len(ranking)
        if not depth:
            raise ValueError(f"gate has no ranked pages for {qid!r}")
        if requested_k > depth and not allow_cap:
            raise ValueError(
                f"policy requests K={requested_k} for {qid!r}, but gate only has {depth} pages "
                f"in {field}; rebuild a top-200 gate or pass --allow-cap"
            )
        applied[qid] = min(requested_k, depth)
        depths[field] = depths.get(field, 0) + 1
        if requested_k > depth:
            capped[qid] = {"requested_k": requested_k, "available_depth": depth}

    output = dict(gate)
    if allow_subset:
        # Preserve an existing/fallback policy for qids not present in this one.
        combined = dict(gate.get("adaptive_k") or {})
        combined.update(applied)
        applied = combined
    output["adaptive_k"] = applied
    output["adaptive_k_policy"] = {
        key: value for key, value in policy.items() if key != "predictions"
    }
    output["adaptive_k_policy"].update(
        {"ranking_fields": depths, "capped_queries": capped, "allow_subset": allow_subset}
    )
    return output


def _json_object(value, flag):
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise SystemExit(f"{flag} must be a JSON object: {error}") from error
    if not isinstance(parsed, dict):
        raise SystemExit(f"{flag} must be a JSON object")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="convert an OOF prediction CSV to a policy JSON")
    export.add_argument("--predictions", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--name", required=True)
    export.add_argument("--qid-column", default="qid")
    export.add_argument("--k-column", default="chosen_k")
    export.add_argument("--source-run")
    export.add_argument("--metrics", help="JSON object, e.g. '{\"macro_recall\": 0.8523}'")
    export.add_argument("--notes")

    apply = subparsers.add_parser("apply", help="attach a policy JSON to an existing retrieval gate")
    apply.add_argument("--policy", required=True)
    apply.add_argument("--gate", required=True)
    apply.add_argument("--output", required=True)
    apply.add_argument("--allow-subset", action="store_true")
    apply.add_argument("--allow-cap", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "export":
        predictions = predictions_from_csv(args.predictions, args.qid_column, args.k_column)
        policy = build_policy(
            predictions,
            name=args.name,
            source_run=args.source_run,
            metrics=_json_object(args.metrics, "--metrics"),
            notes=args.notes,
        )
        output = policy
    else:
        gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
        output = apply_policy(
            gate,
            load_policy(args.policy),
            allow_subset=args.allow_subset,
            allow_cap=args.allow_cap,
        )

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
