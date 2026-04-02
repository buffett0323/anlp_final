# LAVE on JSONSchemaBench (`Github_hard`) — baseline vs `combined`

**Source files**

- `lave_timed_jsb_hard_s0_t128.jsonl` — baseline LAVE (`run_lave_timed.py`)
- `lave_combined_timed_jsb_hard_s0_t128.jsonl` — improved **`combined`** (`run_lave_improved_timed.py`: dir2 + dir3 + dir4; `block_length=32`, `change_logits=True`, `top_n_beam=50`, `random_n_beam=0`, mask cache on)

**Setup:** dataset `jsb_hard` → Hugging Face `epfl-dlab/JSONSchemaBench` / `Github_hard`, seed `0`, denoising steps `T=128`, **10 instances** (same ordering in both runs), generation length 256 tokens.

---

## End-to-end outcome (all 10 rows)

| Run | Instances | `valid=true` | Matcher error | Timeout | Valid rate |
|-----|-----------|--------------|-----------------|----------|------------|
| Baseline (`lave`) | 10 | 3 | 6 | 1 | 30% |
| Improved (`lave_combined`) | 10 | 3 | 6 | 1 | 30% |

**Same pattern in both files:** the six instances `o10293`, `o10296`, `o10346`, `o10347`, `o10421`, `o10515` fail immediately with **`Matcher is in error state`** (llguidance matcher after `Checker` init). `o10495` hits the **120s** wall-clock timeout in both runs. The three completed instances are **`o10499`**, **`o1051`**, **`o1052`**.

---

## Metrics on **successful** runs only (`valid=true`, `n=3` each)

Statistics below are **means** (with min / max / median where useful) over the three instances that finished without error or timeout.

### Wall-clock time (seconds)

| Run | Mean | Median | Min | Max | Std |
|-----|------|--------|-----|-----|-----|
| Baseline | 65.75 | 65.30 | 58.39 | 73.55 | 7.59 |
| Combined | 64.91 | 69.22 | 48.38 | 77.12 | 14.85 |

### Resamples (retries)

| Run | Mean | Median | Min | Max | Std |
|-----|------|--------|-----|-----|-----|
| Baseline | 58 | 31 | 27 | 116 | 50.27 |
| Combined | 165 | 81 | 32 | 381 | 188.95 |

### Instrumented timing (means over the 3 successes)

| Run | Forward total (ms) | Constraint % | Per-token wall (ms/token) | Sum of `cache_hits` (3 rows) |
|-----|----------------------|--------------|----------------------------|------------------------------|
| Baseline | 6876 | 28.84 | 256.82 | 0 |
| Combined | 6799 | 7.78 | 253.54 | 53 |

On this slice, **`combined`** reports much **lower constraint share** of (constraint + forward) time and **non-zero cache hits** on `o10499` (dir3 cache), but **higher** mean **resamples** (driven by `o1051`: 381 vs 116).

---

## Notes

1. **Matcher errors:** Six failures are **infrastructure** (matcher enters error state under llguidance), not a difference between baseline and improved configs; both runs see the same counts.
2. **Timeout:** One instance (`o10495`) hits the **120s** `instance_timeout` in both runs.
3. **Per-token columns** in JSONL assume **256** generated tokens for `per_token_total_ms` / `per_token_constraint_ms`.
4. **Qualitative output** on `o1051` / `o1052` differs between runs (e.g. truncated or malformed JSON in `extracted`); validity flags can still be `true` if EOS/EOT heuristics pass—inspect `extracted` for semantic quality if needed.
