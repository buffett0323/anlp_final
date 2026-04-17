#!/usr/bin/env bash
# Run all benchmarks and print comparison metrics.
#
# Usage:
#   bash run_all_benchmarks.sh [dataset] [seed] [steps] [limit]
#
# Defaults: dataset=jsb_medium, seed=0, steps=128, limit=586
#
# Each method is run in chunks of CHUNK_SIZE instances (offset-based sharding).
# After all runs complete, compare_results.py aggregates and prints the summary.

set -euo pipefail

DATASET="${1:-jsb_medium}"
SEED="${2:-0}"
STEPS="${3:-128}"
LIMIT="${4:-586}"
CHUNK_SIZE=66   # matches existing shard size in results/

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "================================================="
echo " Benchmark runner"
echo "  Dataset : $DATASET"
echo "  Seed    : $SEED"
echo "  Steps   : $STEPS"
echo "  Limit   : $LIMIT"
echo "  Chunks  : $CHUNK_SIZE instances each"
echo "================================================="
echo ""

# ── helper: run a method in chunks ────────────────────────────────────────────
run_chunked() {
    local script="$1"
    local method_label="$2"
    shift 2
    local extra_args=("$@")   # extra positional args appended after standard ones

    echo "-------------------------------------------------"
    echo " Running: $method_label"
    echo "-------------------------------------------------"

    local offset=0
    while [ "$offset" -lt "$LIMIT" ]; do
        local chunk=$(( LIMIT - offset < CHUNK_SIZE ? LIMIT - offset : CHUNK_SIZE ))
        echo "  → offset=$offset  chunk=$chunk"
        python "$script" \
            "$SEED" \
            "$chunk" \
            "$DATASET" \
            "$STEPS" \
            "$offset" \
            "${extra_args[@]+"${extra_args[@]}"}"
        offset=$(( offset + CHUNK_SIZE ))
    done
    echo ""
}

# ── 1. LAVE ───────────────────────────────────────────────────────────────────
run_chunked bench/run_lave_timed.py "LAVE"

# ── 2. Dgrammar (v2 async) ────────────────────────────────────────────────────
# argv: seed limit dataset steps offset block_ar(=1) method(=dgrammar_v2_async)
run_chunked bench/run_dgrammar_timed.py "Dgrammar (v2 async)" 1 "dgrammar_v2_async"

# ── 3. DPGrammar ──────────────────────────────────────────────────────────────
# argv: seed limit dataset steps offset block_ar(=1) method(=dp)
run_chunked bench/run_dgrammar_timed.py "DPGrammar" 1 "dp"

# ── 4. Aggregate and print results ────────────────────────────────────────────
echo "================================================="
echo " Aggregating results → results/comparison.md"
echo "================================================="
python bench/compare_results.py

echo ""
echo "================================================="
echo " Per-file metrics (Validity / Med / Mean / P95 / Eff.const% / Resamples)"
echo "================================================="

# Print per-method timed metrics using jsonschemabench_metrics.py style aggregation.
# We use a small inline Python snippet that mirrors jsonschemabench_metrics but
# also reports median, P95, eff. constraint %, and mean resamples from the JSONL.
python - <<'PYEOF'
import json, math, sys
from pathlib import Path
from collections import defaultdict
import re

RESULTS_DIR = Path("results")

def base_name(stem):
    return re.sub(r"_off\d+$", "", stem)

def pct(q, vals):
    vals = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not vals:
        return float("nan")
    k = (len(vals) - 1) * q
    f, c = int(k), min(int(k) + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)

METHOD_LABELS = {
    "lave":              "LAVE",
    "dgrammar_v2_async": "Dgrammar",
    "dgrammar_dp":       "DPGrammar",
}
METHOD_ORDER = ["lave", "dgrammar_v2_async", "dgrammar_dp"]

# Load & merge shards
groups = defaultdict(dict)
for path in sorted(RESULTS_DIR.glob("*.jsonl")):
    base = base_name(path.stem)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        iid = rec.get("instance_id")
        if iid:
            groups[base][iid] = rec

def sort_key(base):
    recs = list(groups[base].values())
    m = next(iter(recs), {}).get("method", base)
    try:
        return (METHOD_ORDER.index(m), base)
    except ValueError:
        return (len(METHOD_ORDER), base)

sorted_bases = sorted(groups.keys(), key=sort_key)

# Header
print(f"\n{'Method':<22} {'N':>5} {'Valid%':>7} {'Med.t(s)':>9} {'Mean.t(s)':>10} {'P95.t(s)':>9} {'Eff.const%':>11} {'MeanResamples':>14}")
print("-" * 95)

for base in sorted_bases:
    recs = list(groups[base].values())
    method = next(iter(recs), {}).get("method", base)
    label = METHOD_LABELS.get(method, method)

    n = len(recs)
    valid = sum(1 for r in recs if r.get("valid"))
    times = [r["time_taken"] for r in recs if "time_taken" in r]
    resamples = [r["resamples"] for r in recs if "resamples" in r]
    eff_pcts = [r.get("timing", {}).get("effective_constraint_pct")
                for r in recs if r.get("timing", {}).get("effective_constraint_pct") is not None]

    med_t  = pct(0.50, times)
    mean_t = sum(times) / len(times) if times else float("nan")
    p95_t  = pct(0.95, times)
    med_eff = pct(0.50, eff_pcts) if eff_pcts else float("nan")
    mean_res = sum(resamples) / len(resamples) if resamples else float("nan")

    def f(v, fmt):
        return fmt.format(v) if not math.isnan(v) else "—"

    print(
        f"{label:<22} {n:>5} {valid/n*100:>6.1f}%"
        f" {f(med_t, '{:>9.2f}')}"
        f" {f(mean_t, '{:>10.2f}')}"
        f" {f(p95_t, '{:>9.2f}')}"
        f" {f(med_eff, '{:>10.2f}%')}"
        f" {f(mean_res, '{:>14.2f}')}"
    )

print()
PYEOF
