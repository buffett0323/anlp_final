# Dgrammar & DPGrammar: Constrained Decoding for Diffusion LLMs



Jeng-Yue Liu, Wilson Zheng, Haoling Pu

Language Technologies Institute, Carnegie Mellon University





## Abstract

LAVE-style **stochastic lookahead–verify** reports 99% validity on easy JSON benchmarks but **does not transfer** to harder schemas. On **JSONSchemaBench**-style `**jsb_medium`**, lookahead-verify reaches **78%** validity on matched instances, with **large lookahead draw counts** and **timeouts**. **Dgrammar** replaces random suffix sampling with **deterministic frontier masking**, AIMD batching, and async mask overlap (**84.9%** validity in Table 1, **no timeouts**). **DPGrammar** adds a **Viterbi DP** over violated spans, reaching **95.3%** validity with much lower resample churn (Table 1).

---

## Problem

**LAVE** accepts a frontier token only if ≥1 of *K* **random** full completions is grammar-valid. Under **joint** constraints (e.g. enum casing **and** a fixed-length hex pattern), valid random suffixes are astronomically rare → false rejections, many forwards per token, and **120 s** timeouts.

**Dgrammar** enforces the schema only at the **first unresolved** position (`compute_mask` / `LLMatcher`), carries **incremental** automaton state, uses **AIMD** batch commits, **violator remask**, and **async** mask prep overlapping the GPU forward—**no** stochastic completions at the frontier.

**DPGrammar** keeps the same denoising schedule; on reject, it runs `**dp_fix_prefix`**: Viterbi over the span [c,s) with **top-*K*** logits and grammar DFA states, then falls back to Dgrammar’s greedy remask if no feasible path exists.

---

## Main results

All methods on the **same** **511** instances after **75** schemas are dropped (**llguidance** compile failures). **LLaDA-8B-Instruct**, **T=128** diffusion steps, block **32**, temperature **0.2** (details in the paper Appendix).


|                               | LAVE      | Dgrammar  | DPGrammar     |
| ----------------------------- | --------- | --------- | ------------- |
| **Skipped (grammar-invalid)** | 75        | 75        | 75            |
| **Evaluated (n)**             | 511       | 511       | 511           |
| **Valid**                     | 397 / 511 | 434 / 511 | **487 / 511** |
| **Validity**                  | 77.7%     | 84.9%     | **95.3%**     |
| **Timeouts (>120 s)**         | 68        | 0         | 0             |
| **Mean resamples†**           | 233.87    | 32.43     | **2.11**      |
| **Mean wall time (s)**        | 41.12     | **13.38** | 16.14         |
| **Median wall time (s)**      | 27.34     | **13.65** | 15.61         |
| **P95 wall time (s)**         | 120.00    | **20.93** | 29.36         |
| **Max wall time (s)**         | 120.05    | **33.45** | 88.17         |


† **LAVE:** lookahead verification draws. **Dgrammar / DPGrammar:** token re-mask events (Table 1 footnote in `final.tex`).

### Metrics

- **Schema validity:** benchmark `valid` after extraction.  
- **Wall time:** end-to-end driver time per instance.  
- **Constraint / effective constraint %:** driver-reported grammar and mask share of wall time (see paper §Experimental design; cross-method comparison is nuanced—`[docs/experiment_comparison.md](docs/experiment_comparison.md)`).

---

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11

uv pip install -e .
# llguidance for token-level grammar checking
uv pip install llguidance>=1.6
# For v1 (rustformlang backend)
cd vendor/constrained-diffusion/rustformlang_bindings && maturin develop --release

export HF_TOKEN= #your hugging face token
```

For LLaDA: `uv pip install -r requirements-llada.txt`.

**Bench stack:** `llguidance` ≥ 1.6, `vendor/dgrammar` on `PYTHONPATH`, plus constrained-diffusion / CD4dLLM-style layout per `[vendor/dgrammar/README.md](vendor/dgrammar/README.md)`.

---

## Running benchmarks (Modal)

From `**vendor/dgrammar`** with the [Modal](https://modal.com) CLI installed and authenticated. GPU jobs use the image defined in each script (A100); results go to the `**dgrammar-results`** volume unless you change the script. Merge JSONL shards on `**instance_id**` as in **§Data** of `latex/latex/final.tex`.

Paper-style sharding uses `**--total 586`** and `**--chunks 9`** (66 instances per chunk, offsets 0, 66, …, 528)—same idea as `vendor/dgrammar/run_all_benchmarks.sh`. For a smoke test, use e.g. `--total 66 --chunks 1`.

### LAVE (`bench/modal_lave_bench.py`)

```bash
modal run bench/modal_lave_bench.py \
  --seed 0 --total 586 --steps 128 --chunks 9 \
  --dataset jsb_medium --instance-timeout 120
```

### Dgrammar (`bench/modal_dgrammar_bench.py`, `--method dgrammar`)

```bash
modal run bench/modal_dgrammar_bench.py \
  --seed 0 --total 586 --steps 128 --chunks 9 \
  --block-ar 1 --dataset jsb_medium --method dgrammar
```

### DPGrammar (same Modal app, `--method dp`)

```bash
modal run bench/modal_dgrammar_bench.py \
  --seed 0 --total 586 --steps 128 --chunks 9 \
  --block-ar 1 --dataset jsb_medium --method dp
```

`modal_lave_improved_bench.py` is for ablation directions, not the main Table 1 **DPGrammar** line. For non-Modal runs, call `bench/run_lave_timed.py` / `bench/run_dgrammar_timed.py` with the same argv layout as `run_all_benchmarks.sh`.

---

## Poster

You can view the final project poster here:

[📄 Final Poster (PDF)](docs/Final_Poster.pdf)

---

## References

LAVE (Zhang et al., ACM 2026); DINGO (Suresh et al., NeurIPS 2025); constrained diffusion / CD4dLLM; **llguidance** (Microsoft, 2025); STATIC (Su et al., 2026); **JSONSchemaBench** (Geng et al., 2025).

---

## Limitations

See **§Limitations** in `final.tex`: single split and backbone; validity ≠ semantic quality; DPGrammar **top-*K*** and `find_constraint_end` heuristic; fixed T, temperature, and timeout not fully tuned.