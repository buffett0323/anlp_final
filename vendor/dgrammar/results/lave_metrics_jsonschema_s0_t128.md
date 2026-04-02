# LAVE improvement-direction benchmark — performance summary

**Source files** (10 instances each, same `instance_id` ordering per run):

- **`lave_timed_jsonschema_s0_t128.jsonl`** — baseline **LAVE** (`run_lave_timed.py`, `block_length=32`, `change_logits=False`)
- `lave_dir1_timed_jsonschema_s0_t128.jsonl`
- `lave_dir2_timed_jsonschema_s0_t128.jsonl`
- `lave_dir3_timed_jsonschema_s0_t128.jsonl`
- `lave_dir4_timed_jsonschema_s0_t128.jsonl`
- `lave_combined_timed_jsonschema_s0_t128.jsonl`

**Setup:** dataset `jsonschema` (`eth-sri/json-mode-eval-extended` test split), seed `0`, denoising steps `T=128`, generation length 256 tokens (per runner). **Baseline** matches the same schedule family as **dir2–dir4** / **combined** (`block_length=32`). **dir1** uses `block_length=128` (not comparable to baseline on wall time / constraint mix).

---

## End-to-end outcome

| Experiment | Instances | Valid | Valid rate | Timeouts |
|------------|-----------|-------|------------|----------|
| baseline (LAVE) | 10 | 10 | 100% | 0 |
| dir1 | 10 | 10 | 100% | 0 |
| dir2 | 10 | 10 | 100% | 0 |
| dir3 | 10 | 10 | 100% | 0 |
| dir4 | 10 | 10 | 100% | 0 |
| combined | 10 | 10 | 100% | 0 |

---

## Wall-clock time (seconds)

| Experiment | Mean | Median | Std | Min | Max |
|------------|------|--------|-----|-----|-----|
| baseline (LAVE) | 7.17 | 6.46 | 3.66 | 3.62 | 13.23 |
| dir1 | 12.76 | 8.34 | 9.26 | 4.21 | 30.35 |
| dir2 | 8.55 | 7.61 | 4.41 | 4.24 | 16.02 |
| dir3 | 8.34 | 7.99 | 3.92 | 4.26 | 14.44 |
| dir4 | 8.19 | 7.69 | 3.83 | 4.34 | 14.45 |
| combined | 8.41 | 7.77 | 4.28 | 4.22 | 15.59 |

---

## Resamples (retries)

| Experiment | Mean | Median | Std | Min | Max |
|------------|------|--------|-----|-----|-----|
| baseline (LAVE) | 0.20 | 0.00 | 0.63 | 0 | 2 |
| dir1 | 1.50 | 0.00 | 3.17 | 0 | 8 |
| dir2 | 0.20 | 0.00 | 0.63 | 0 | 2 |
| dir3 | 0.20 | 0.00 | 0.63 | 0 | 2 |
| dir4 | 0.10 | 0.00 | 0.32 | 0 | 1 |
| combined | 0.10 | 0.00 | 0.32 | 0 | 1 |

---

## Instrumented timing (means over instances)

Values below are **means** of per-instance aggregates. `constraint_pct` is the share of (constraint-related ms) in (constraint + forward) as reported in the JSONL.

| Experiment | Forward total (ms) | Total constraint (ms) | Constraint % | Per-token wall (ms/token) | Per-token constraint (ms/token) |
|------------|----------------------|-------------------------|----------------|-----------------------------|----------------------------------|
| baseline (LAVE) | 2083.5 | 48.8 | 1.34 | 27.99 | 0.19 |
| dir1 | 3074.0 | 2991.9 | 19.54 | 49.85 | 11.69 |
| dir2 | 3064.3 | 71.5 | 1.36 | 33.38 | 0.28 |
| dir3 | 3072.1 | 61.9 | 1.22 | 32.57 | 0.24 |
| dir4 | 3043.9 | 63.9 | 1.26 | 32.00 | 0.25 |
| combined | 2967.1 | 67.2 | 1.32 | 32.86 | 0.26 |

**Subcomponents (means, ms):** dir1 is dominated by **compute_mask** when present (mean **2938.6** ms). Baseline and dir2–combined cluster around **~39–55** ms mean `compute_mask_total_ms` on this slice; baseline mean forward time is lower than the improved runs (**2084** vs **~2967–3074** ms), consistent with faster wall-clock on baseline.

---

## Notes

1. **dir1 vs baseline / dir2–combined:** dir1 uses `block_length=128`; baseline and dir2–combined use `block_length=32` (different masking schedule).
2. **Per-token columns** use fixed `gen_length=256` in the runner for `per_token_total_ms` / `per_token_constraint_ms` in the logs.
3. Metrics are computed from the six JSONL files in `dgrammar/results/`; regenerate this file if you re-run experiments or change instance counts.
