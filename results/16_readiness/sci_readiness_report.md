# SCI Readiness Report

## Claim Gate

- Status: `completed`
- Paper claim level: `SCI-ready superiority claim against executed CAGE-4 baselines`
- Superiority claim ready: `True`
- Minimum paired runs observed: `30`
- Recommended minimum paired runs: `30`
- Claim boundary: CybSAGE beat every executed baseline with positive paired bootstrap confidence and enough paired runs.

## CAGE-4 Effect Sizes

| baseline | paired_runs | cybsage_mean | baseline_mean | mean_difference | ci95_low | ci95_high | win_rate | tie_rate | cohen_d_paired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| restore_priority | 30 | -59.3 | -604.567 | 545.267 | 514.464 | 579.802 | 1.000 | 0.000 | 5.79965 |
| random | 30 | -59.3 | -526.333 | 467.033 | 407.395 | 531.037 | 1.000 | 0.000 | 2.58034 |
| remove_priority | 30 | -59.3 | -112.833 | 53.5333 | 21.8 | 88.3333 | 0.633 | 0.000 | 0.566718 |
| monitor | 30 | -59.3 | -100.3 | 41 | 17.4992 | 64.8675 | 0.633 | 0.000 | 0.610229 |
| sleep | 30 | -59.3 | -100.3 | 41 | 17.4992 | 64.8675 | 0.633 | 0.000 | 0.610229 |
| decoy_priority | 30 | -59.3 | -97.7333 | 38.4333 | 14.5667 | 62.47 | 0.700 | 0.000 | 0.578717 |

## Profile Ablation

| rank | profile | total_reward | delta_from_best | selection_priority |
| ---: | --- | ---: | ---: | ---: |
| 1 | alert_response_decoy | -73.8667 | 0 | 7.5 |
| 2 | staged_decoy_sweep | -81.7 | -7.83333 | 6.8 |
| 3 | adaptive_decoy_monitor | -84.0333 | -10.1667 | 6.2 |
| 3 | decoy_context_high | -84.0333 | -10.1667 | 5.2 |
| 3 | decoy_context_low | -84.0333 | -10.1667 | 5 |
| 3 | decoy_context_mid | -84.0333 | -10.1667 | 5.5 |
| 3 | decoy_guarded | -84.0333 | -10.1667 | 3 |
| 3 | decoy_remove_pressure | -84.0333 | -10.1667 | 4 |
| 9 | monitor_alert_remove | -87.1333 | -13.2667 | 7 |
| 10 | decoy_alert_remove | -89.5 | -15.6333 | 7.3 |
| 11 | deception | -96.5 | -22.6333 | 1 |
| 12 | rotating_decoy | -96.9 | -23.0333 | 5.8 |
| 13 | balanced | -107.033 | -33.1667 | 1 |
| 13 | containment | -107.033 | -33.1667 | 1 |
| 15 | adaptive_decoy_repair | -124.967 | -51.1 | 6 |
| 16 | resilience | -494.567 | -420.7 | 1 |

## Required Next Evidence For SCI Q1

- Run more paired CAGE-4 episodes and seeds before making a Q1-level superiority claim.
- Include both finite-state and random-select red-agent settings.
- Keep the claim bounded if CybSAGE ties the strongest baseline instead of beating it.
- Use this report as the paper claim gate, not notebook completion alone.
