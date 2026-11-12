# CAGE-4 Reward SOTA Comparison

- `status`: `completed`
- `superiority_claim_ready`: `True`
- `claim_boundary`: CybSAGE beat every executed CAGE-4 baseline with positive bootstrap confidence intervals.

## Baseline Comparisons

| baseline | cybsage_mean | baseline_mean | mean_difference | ci95_low | ci95_high | sign_test_p_value | cybsage_wins | paired_runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| restore_priority | -59.3 | -604.567 | 545.267 | 514.464 | 579.802 | 1.86265e-09 | 30 | 30 |
| random | -59.3 | -526.333 | 467.033 | 407.395 | 531.037 | 1.86265e-09 | 30 | 30 |
| remove_priority | -59.3 | -112.833 | 53.5333 | 21.8 | 88.3333 | 0.200488 | 19 | 30 |
| monitor | -59.3 | -100.3 | 41 | 17.4992 | 64.8675 | 0.200488 | 19 | 30 |
| sleep | -59.3 | -100.3 | 41 | 17.4992 | 64.8675 | 0.200488 | 19 | 30 |
| decoy_priority | -59.3 | -97.7333 | 38.4333 | 14.5667 | 62.47 | 0.0427739 | 21 | 30 |
