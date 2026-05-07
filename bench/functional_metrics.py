#!/usr/bin/env python3
"""Compute syntactic@k and functional@k per the LAVE paper (Section 4.4).

syntactic@k  — fraction of instances where ≥1 of k outputs validates against
               the JSON Schema (uses `schema` field in JSONL; falls back to
               the `valid` EOS/mask flag if schema is absent).

functional@k — fraction of instances where ≥1 of k outputs exactly matches
               the ground-truth JSON (JSON-Mode-Eval / jsonschema_* IDs only;
               requires `passed_tests` in JSONL or HuggingFace access).
               For jsb datasets (no ground truth), functional = syntactic.

Usage:
  # Single file — reports @1
  python bench/functional_metrics.py results/dp_jsb_medium_s0_t128_gl512.jsonl

  # Multiple files — compare methods side by side (one per line)
  python bench/functional_metrics.py \\
    results/v2_async_ac4_timed_jsb_medium_s0_t128.jsonl \\
    results/dp_jsb_medium_s0_t128_gl512.jsonl \\
    results/lave_timed_jsb_medium_s0_t128_gl512.jsonl

  # Multiple seeds for same method — functional@k
  python bench/functional_metrics.py --k \\
    results/dp_jsb_medium_s0_t128_gl512.jsonl \\
    results/dp_jsb_medium_s1_t128_gl512.jsonl \\
    results/dp_jsb_medium_s2_t128_gl512.jsonl

  # Chunk files from same run (merged automatically)
  python bench/functional_metrics.py \\
    results/dp_jsb_medium_s0_t128.jsonl \\
    results/dp_jsb_medium_s0_t128_off66.jsonl \\
    results/dp_jsb_medium_s0_t128_off132.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    from jsonschema import validators
    from jsonschema.exceptions import ValidationError
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _schema_valid(extracted: Optional[str], schema_str: Optional[str]) -> bool:
    if not _HAS_JSONSCHEMA or not extracted or not schema_str:
        return False
    try:
        inst = json.loads(extracted)
        sch = json.loads(schema_str)
        cls = validators.validator_for(sch)
        cls(sch).validate(inst)
        return True
    except Exception:
        return False


def _exact_match(extracted: Optional[str], reference: Optional[str]) -> bool:
    if not extracted or not reference:
        return False
    try:
        return (
            json.dumps(json.loads(extracted), indent=4)
            == json.dumps(json.loads(reference), indent=4)
        )
    except Exception:
        return False


def _load_ground_truth() -> Optional[dict[str, str]]:
    """Load {jsonschema_N: gt_output} from eth-sri/json-mode-eval-extended."""
    try:
        from datasets import load_dataset as _hf
        ds = _hf("eth-sri/json-mode-eval-extended", split="test")
        gt = {f"jsonschema_{i}": row["output"] for i, row in enumerate(ds)}
        print(f"  Loaded {len(gt)} ground-truth solutions.", file=sys.stderr)
        return gt
    except Exception as e:
        print(f"  Warning: could not load ground truth ({e}); functional will equal syntactic.",
              file=sys.stderr)
        return None


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def merge_deduplicate(paths: list[Path]) -> list[dict]:
    """Merge multiple JSONL files, keeping the first occurrence per instance_id."""
    seen: set[str] = set()
    rows: list[dict] = []
    for p in paths:
        for r in load_rows(p):
            iid = r.get("instance_id", "")
            if iid not in seen:
                seen.add(iid)
                rows.append(r)
    return rows


# ── per-row evaluation ────────────────────────────────────────────────────────

def _row_syntactic(r: dict) -> bool:
    schema = r.get("schema")
    if schema:
        return _schema_valid(r.get("extracted"), schema)
    return bool(r.get("valid", False))


def _row_functional(r: dict, iid: str, is_jsonmode: bool, gt: Optional[dict]) -> bool:
    if is_jsonmode:
        if "passed_tests" in r:
            return bool(r["passed_tests"])
        if gt is not None:
            return _exact_match(r.get("extracted"), gt.get(iid))
        return _row_syntactic(r)  # GT unavailable
    return _row_syntactic(r)


# ── single-method evaluation (k=1) ───────────────────────────────────────────

def evaluate_single(rows: list[dict], gt: Optional[dict], label: str) -> dict:
    if not rows:
        return {}

    is_jsonmode = rows[0].get("instance_id", "").startswith("jsonschema_")
    n = syn_ok = func_ok = diff_ok = 0
    times: list[float] = []

    for r in rows:
        n += 1
        iid = r.get("instance_id", "")
        if r.get("valid"):
            diff_ok += 1
        if _row_syntactic(r):
            syn_ok += 1
        if _row_functional(r, iid, is_jsonmode, gt):
            func_ok += 1
        t = r.get("time_taken")
        if t is not None:
            times.append(float(t))

    has_schema = any(r.get("schema") for r in rows)
    avg_t = sum(times) / len(times) if times else 0.0

    return {
        "label": label,
        "n": n,
        "k": 1,
        "diff_valid_pct": diff_ok / n * 100,
        "syntactic@1": syn_ok / n * 100,
        "functional@1": func_ok / n * 100,
        "avg_time_s": avg_t,
        "is_jsonmode": is_jsonmode,
        "has_schema": has_schema,
        "has_gt": is_jsonmode and gt is not None,
    }


# ── multi-seed evaluation (functional@k) ─────────────────────────────────────

def evaluate_k(files: list[Path], gt: Optional[dict]) -> dict:
    per_instance: dict[str, list[dict]] = defaultdict(list)
    for f in files:
        for r in load_rows(f):
            per_instance[r["instance_id"]].append(r)

    if not per_instance:
        return {}

    first_id = next(iter(per_instance))
    is_jsonmode = first_id.startswith("jsonschema_")
    k = max(len(v) for v in per_instance.values())
    n = len(per_instance)
    syn_ok = func_ok = 0

    for iid, runs in per_instance.items():
        if any(_row_syntactic(r) for r in runs):
            syn_ok += 1
        if any(_row_functional(r, iid, is_jsonmode, gt) for r in runs):
            func_ok += 1

    all_times = [r.get("time_taken", 0) or 0 for runs in per_instance.values() for r in runs]
    avg_t = sum(all_times) / len(all_times) if all_times else 0.0

    return {
        "label": f"{files[0].stem} (@{k} seeds)",
        "n": n,
        "k": k,
        f"syntactic@{k}": syn_ok / n * 100,
        f"functional@{k}": func_ok / n * 100,
        "avg_time_s": avg_t,
        "is_jsonmode": is_jsonmode,
        "has_gt": is_jsonmode and gt is not None,
    }


# ── printing ──────────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    if not results:
        return

    k = results[0].get("k", 1)
    syn_key = f"syntactic@{k}"
    func_key = f"functional@{k}"
    is_jsonmode = results[0].get("is_jsonmode", False)
    has_gt = results[0].get("has_gt", False)

    print(f"\n{'Method':<45} {'n':>5}  {syn_key:>13}  {func_key:>13}  {'avg_time':>9}")
    print("-" * 92)
    for r in results:
        syn = r.get(syn_key, r.get("syntactic@1", 0))
        func = r.get(func_key, r.get("functional@1", 0))
        note = " *" if not r.get("has_schema", True) else ""
        print(f"{r['label']:<45} {r['n']:>5}  {syn:>12.1f}%  {func:>12.1f}%  {r['avg_time_s']:>8.1f}s{note}")

    if any(not r.get("has_schema", True) for r in results):
        print("\n  *  No `schema` field in JSONL — syntactic uses `valid` (EOS/mask) as fallback.")
        print("     Re-run with updated bench scripts to get schema-based syntactic@k.")
    if not has_gt and is_jsonmode:
        print(f"\n  functional@{k} = syntactic@{k} (ground truth not loaded).")
    elif not is_jsonmode:
        print(f"\n  Dataset has no ground truth — functional@{k} = syntactic@{k} (schema validation).")
    print()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", type=Path, help="JSONL result files")
    parser.add_argument(
        "--k", action="store_true",
        help="Treat all files as independent seeds of ONE method (computes functional@k)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all files into one result set (for chunk files from the same run)",
    )
    args = parser.parse_args()

    files = [p for p in args.files if p.exists()]
    missing = [p for p in args.files if not p.exists()]
    if missing:
        for p in missing:
            print(f"Warning: {p} not found, skipping.", file=sys.stderr)
    if not files:
        sys.exit("No valid input files.")

    # Detect dataset type
    sample = load_rows(files[0])
    is_jsonmode = bool(sample and sample[0].get("instance_id", "").startswith("jsonschema_"))
    gt = _load_ground_truth() if is_jsonmode else None

    dataset_label = "JSON-Mode-Eval (272)" if is_jsonmode else "JSONSchemaBench"
    func_def = "exact match with ground truth" if is_jsonmode else "schema validation (no ground truth)"
    print(f"\nDataset : {dataset_label}")
    print(f"Functional: {func_def}")

    if args.k:
        result = evaluate_k(files, gt)
        print_table([result])
    elif args.merge:
        rows = merge_deduplicate(files)
        label = files[0].stem
        result = evaluate_single(rows, gt, label)
        print_table([result])
    else:
        results = []
        for f in files:
            rows = merge_deduplicate([f])
            result = evaluate_single(rows, gt, label=f.stem)
            if result:
                results.append(result)
        print_table(results)


if __name__ == "__main__":
    main()
