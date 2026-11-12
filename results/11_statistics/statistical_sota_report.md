# CybSAGE-MARL Statistical and SOTA Readiness Report

## Current Verified Pipeline Metrics

- `trajectory_rows`: `6234`
- `mean_reward`: `0.3996791786974655`
- `policy_val_accuracy`: `0.8757016840417001`
- `policy_val_macro_f1`: `0.5799318678000763`
- `explanations`: `100`
- `explanations_with_evidence`: `91`
- `evidence_coverage`: `0.91`

## Internal Baseline/Ablation Ranking

| method | method_role | runs | accuracy_mean | accuracy_std | balanced_accuracy_mean | balanced_accuracy_std | macro_f1_mean | macro_f1_std | weighted_f1_mean | weighted_f1_std | decision_utility_mean | decision_utility_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rule_based_oracle | synthetic_oracle_upper_bound_not_sota | 5 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
| cybsage_full | proposed_learned_model | 5 | 0.871051 | 0.00590382 | 0.585046 | 0.0200871 | 0.581415 | 0.033442 | 0.841263 | 0.00567206 | 0.845261 | 0.00708459 |
| no_posterior | learned_ablation | 5 | 0.863031 | 0.0112898 | 0.565005 | 0.0304539 | 0.554326 | 0.0507941 | 0.830884 | 0.016824 | 0.835638 | 0.0135477 |
| no_ssl | learned_ablation | 5 | 0.865597 | 0.00979859 | 0.558924 | 0.0043934 | 0.538675 | 0.00389768 | 0.830331 | 0.00944396 | 0.838717 | 0.0117583 |
| state_only | learned_ablation | 5 | 0.860946 | 0.00604377 | 0.55279 | 0.00455051 | 0.531469 | 0.0042952 | 0.825741 | 0.00628923 | 0.833136 | 0.00725252 |
| random_action | trivial_baseline | 5 | 0.2 | 0.0136398 | 0.207627 | 0.023467 | 0.169031 | 0.0121876 | 0.231847 | 0.0165529 | 0.04 | 0.0163678 |
| majority_action | trivial_baseline | 5 | 0.428228 | 0 | 0.2 | 0 | 0.119933 | 0 | 0.256792 | 0 | 0.313873 | 0 |

## Paired Statistical Tests

| primary_method | baseline_method | metric | paired_runs | mean_difference | std_difference | cohens_dz | bootstrap_ci95_low | bootstrap_ci95_high | sign_test_p_value | primary_wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cybsage_full | majority_action | macro_f1 | 5 | 0.461482 | 0.033442 | 13.7995 | 0.43403 | 0.490138 | 0.0625 | 5 |
| cybsage_full | majority_action | accuracy | 5 | 0.442823 | 0.00590382 | 75.0061 | 0.438172 | 0.447314 | 0.0625 | 5 |
| cybsage_full | majority_action | decision_utility | 5 | 0.531387 | 0.00708459 | 75.0061 | 0.525806 | 0.536776 | 0.0625 | 5 |
| cybsage_full | no_posterior | macro_f1 | 5 | 0.0270882 | 0.0633807 | 0.427389 | -0.0248993 | 0.0751845 | 0.375 | 4 |
| cybsage_full | no_posterior | accuracy | 5 | 0.00801925 | 0.00926561 | 0.865485 | 0.00096231 | 0.0150762 | 0.375 | 4 |
| cybsage_full | no_posterior | decision_utility | 5 | 0.0096231 | 0.0111187 | 0.865485 | 0.00115477 | 0.0180914 | 0.375 | 4 |
| cybsage_full | no_ssl | macro_f1 | 5 | 0.0427392 | 0.0315675 | 1.3539 | 0.0177781 | 0.0694386 | 0.0625 | 5 |
| cybsage_full | no_ssl | accuracy | 5 | 0.00545309 | 0.00609146 | 0.895202 | 0.00096231 | 0.00994387 | 0.375 | 4 |
| cybsage_full | no_ssl | decision_utility | 5 | 0.0065437 | 0.00730975 | 0.895202 | 0.00115477 | 0.0119326 | 0.375 | 4 |
| cybsage_full | random_action | macro_f1 | 5 | 0.412383 | 0.0426584 | 9.66711 | 0.381093 | 0.447123 | 0.0625 | 5 |
| cybsage_full | random_action | accuracy | 5 | 0.671051 | 0.0133274 | 50.3512 | 0.660626 | 0.681155 | 0.0625 | 5 |
| cybsage_full | random_action | decision_utility | 5 | 0.805261 | 0.0159929 | 50.3512 | 0.792751 | 0.817386 | 0.0625 | 5 |
| cybsage_full | rule_based_oracle | macro_f1 | 5 | -0.418585 | 0.033442 | -12.5168 | -0.446037 | -0.38993 | 0.0625 | 0 |
| cybsage_full | rule_based_oracle | accuracy | 5 | -0.128949 | 0.00590382 | -21.8417 | -0.133601 | -0.124459 | 0.0625 | 0 |
| cybsage_full | rule_based_oracle | decision_utility | 5 | -0.154739 | 0.00708459 | -21.8417 | -0.160321 | -0.14935 | 0.0625 | 0 |
| cybsage_full | state_only | macro_f1 | 5 | 0.049946 | 0.0348506 | 1.43315 | 0.0227599 | 0.079782 | 0.0625 | 5 |
| cybsage_full | state_only | accuracy | 5 | 0.0101043 | 0.00536751 | 1.88249 | 0.00609463 | 0.014595 | 0.0625 | 5 |
| cybsage_full | state_only | decision_utility | 5 | 0.0121251 | 0.00644101 | 1.88249 | 0.00731355 | 0.017514 | 0.0625 | 5 |

## SOTA Comparability Matrix

| paper | year | environment_or_dataset | method_family | primary_metrics | directly_comparable_now | reason |
| --- | --- | --- | --- | --- | --- | --- |
| Posterior-Conditioned Multi-Agent Reinforcement Learning for Adaptive Cyber Defense | 2026 | CybORG CAGE-4 | Posterior-conditioned MARL | Mission reward, adaptation under mixed/switching attack regimes | False | Requires real CAGE-4 reward-based multi-agent rollout; current CybSAGE run is preliminary trajectory classification/evidence evaluation. |
| Trident: How to Break Deep Reinforcement Learning Cyber Defenses | 2026 | CAGE-4 + CyberWheel | Agentic LLM/RLVR red-team benchmark | Blue-agent cumulative episodic reward under adaptive generated attacks | False | Compares adaptive red-agent attack strength against trained blue defenders; not the same task as current blue-policy action prediction. |
| Automated Cyber Defense with Generalizable Graph-based Reinforcement Learning Agents | 2025 | Yawning Titan, CAGE-2, CAGE-4 | Graph-based RL | Reward/generalization on unseen graph topologies | False | Needs matched CAGE-4/graph generalization protocol and reward metrics. |
| Large Language Models are Autonomous Cyber Defenders | 2025 | CybORG CAGE-4 | LLM + RL defender teams | CAGE-4 reward, runtime, red-agent strategy robustness | False | Needs live CAGE-4 defender rollout and LLM/RL team baselines. |
| Learning to Communicate in Multi-Agent Reinforcement Learning for Autonomous Cyber Defence | 2025 | CybORG scenarios | DIAL communication MARL vs QMIX | Mean return across small/large network games | False | Needs the same communication-game setup or a mapped communication ablation. |
| Design and evaluation of an Autonomous Cyber Defence agent using DRL and an augmented LLM | 2025 | Cyber gym + CTI/MITRE/CVE knowledge graphs | DRL + augmented LLM/RAG + CTI KG | Mitigation/recovery, CVE-to-MITRE precision/recall/macro-F1, RAG quality | False | Current macro-F1 is policy-action prediction, not CVE-to-MITRE mapping or RAG QA faithfulness. |

## Interpretation Boundary

These outputs support preliminary and ablation-level claims for the current implementation. They do not establish superiority over CAGE-4 SOTA papers until real CAGE-4 rollouts, matched baselines, multiple seeds, and reward/adaptation metrics are executed under the same protocol.
