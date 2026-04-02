# Dgrammar vs LAVE on JSB Medium Test

Dataset: epfl-dlab/JSONSchemaBench, Github_medium, test split (586)
Model: LLaDA-8B-Instruct, A100, T=128, seed=0

## Head-to-head

| Metric | LAVE | Dgrammar |
|---|---|---|
| Skipped (grammar invalid) | 75 | 75 |
| Results | 511 | 511 |
| Valid | 396/511 (77.5%) | 434/511 (84.9%) |
| Timeouts (>120s) | 73 | 0 |
| Mean time | 27.4s | 24.2s |
| Median time | 21.9s | 20.6s |
| P95 time | 73.4s | 51.7s |
| Max time | 113.2s | 96.7s |

## Dgrammar details

| Metric | Value |
|---|---|
| Eff constraint % mean | 53.1% |
| Eff constraint % median | 50.6% |
| mask_compute mean | 13.23ms |
| grammar_check mean | 0.15ms |
| mask_wait mean | 0.05ms |
| mask_time_saved mean | 1034ms |
| AC fallback | 134/511 (26.2%) |

## vs original jsonschema (272)

| Metric | jsonschema LAVE | jsonschema Dgrammar | jsb_medium LAVE | jsb_medium Dgrammar |
|---|---|---|---|---|
| Valid | 98.9% | 100.0% | 77.5% | 84.9% |
| Mean time | 7.6s | 8.2s | 27.4s | 24.2s |
| P95 time | 23.6s | 9.6s | 73.4s | 51.7s |
| Timeouts | 0 | 0 | 73 | 0 |
