# LAVE on JSB Medium Test

Dataset: epfl-dlab/JSONSchemaBench, Github_medium, test split
Model: LLaDA-8B-Instruct, A100, T=128, seed=0, timeout=120s/instance

## Results (n=586)

| | Count | % |
|---|---|---|
| Grammar invalid (skipped) | 75 | 12.8% |
| Results produced | 511 | 87.2% |
| Valid | 396 | 77.5% of 511 |
| Invalid (non-timeout) | 42 | 8.2% of 511 |
| Timeout (>120s) | 73 | 14.3% of 511 |

## Timing

| Metric | Value |
|---|---|
| Mean | 27.4s |
| Median | 21.9s |
| P95 | 73.4s |
| Max | 113.2s |

## Constraint Overhead

| Metric | Value |
|---|---|
| Constraint % mean | 13.2% |
| Constraint % median | 3.5% |
| Constraint % P95 | 51.0% |
| compute_mask mean (when >0) | 24.7ms |

## vs Original jsonschema (272)

| Metric | jsonschema | jsb_medium_test |
|---|---|---|
| Valid | 98.9% | 77.5% |
| Mean time | 7.6s | 27.4s |
| P95 time | 23.6s | 73.4s |
| Constraint % mean | 8.5% | 13.2% |
