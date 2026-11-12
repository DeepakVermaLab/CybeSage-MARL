# Pipeline

The implementation follows the proposal loop:

```text
trajectory -> self-supervised representation -> attack-regime posterior
-> defender policy -> action -> feedback -> belief update -> explanation
```

## Stage 1: Audit

Verifies CybORG, MITRE ATT&CK, and NVD manifests/counts from local disk.

## Stage 2: Trajectories

Creates trajectory JSONL files with a common schema. The current runnable path uses a deterministic cyber-defense simulator for CPU smoke testing. The CybORG adapter checks the downloaded CAGE-4 repository and is the integration point for scenario-specific CAGE-4 rollouts.

## Stage 3: CTI Graph

Builds compact MITRE/NVD evidence tables and a graph edge list for action explanations.

## Stage 4: SSL

Trains a masked reconstruction encoder over trajectory observations and saves embeddings.

## Stage 5: Posterior Policy

Trains a posterior-conditioned policy baseline over trajectory states. This is the CPU baseline before adding full IPPO/MAPPO.

## Stage 6: Explanations

Generates policy-grounded, evidence-grounded action explanations.

## Stage 7: Evaluation

Computes defense, adaptation, representation, and explanation metrics plus ablations.

## Stage 8: SOTA Ablation Suite

Runs multi-seed internal baselines and ablations:

- `cybsage_full`: state + attack posterior + SSL embedding.
- `no_ssl`: state + attack posterior only.
- `no_posterior`: state + SSL embedding only.
- `state_only`: raw trajectory state only.
- `rule_based_oracle`: synthetic-environment rule policy sanity baseline.
- `majority_action`: majority-class baseline.
- `random_action`: random-action baseline.

Outputs are saved under `outputs/10_ablation_sota/`.

## Stage 9: Statistical and SOTA Analysis

Aggregates ablation results into paper-ready tables, paired seed-wise tests, bootstrap confidence intervals, sign-test p-values, effect sizes, and a SOTA comparability matrix. This stage explicitly marks current SOTA superiority as not ready until matched real CAGE-4 reward-based experiments are run.

Outputs are saved under `outputs/11_statistical_analysis/`.

## Stage 10: CAGE-4 Dependency Check

Checks whether the downloaded CAGE-4 repository and required runtime packages can import. This stage must pass before real reward rollouts can run.

Outputs are saved under `outputs/12_cage4_dependency_check/`.

## Stage 11: CAGE-4 Baseline Rollouts

Runs real CybORG/CAGE-4 reward rollouts for matched blue-agent baselines:

- `sleep`
- `random`
- `monitor`
- `restore_priority`
- `remove_priority`
- `decoy_priority`

Outputs are saved under `outputs/13_cage4_baseline_rollouts/`.

## Stage 12: CybSAGE CAGE-4 Policy Rollouts

Tunes lightweight CybSAGE action profiles on training seeds, selects the best profile, and evaluates it on held-out CAGE-4 seeds.

Outputs are saved under `outputs/14_cybsage_cage4_policy/`.

## Stage 13: SOTA Reward Comparison

Runs paired reward comparison between CybSAGE and all executed CAGE-4 baselines. This is the superiority gate. A SOTA claim is allowed only if CybSAGE beats every baseline with positive bootstrap confidence intervals.

Outputs are saved under `outputs/15_sota_reward_comparison/`.

## Stage 14: SCI Readiness Pack

Builds paper-facing evidence from the CAGE-4 reward outputs:

- paired effect sizes,
- bootstrap confidence intervals,
- win/tie/loss rates,
- CybSAGE profile ablation/ranking,
- SCI claim-readiness summary,
- Markdown report for manuscript drafting.

This stage adds a minimum paired-run gate. The default gate requires at least 30 paired runs before a superiority claim can be marked SCI-ready.

Outputs are saved under `outputs/16_sci_readiness_pack/`.

## Stage 15: Expanded CAGE-4 SCI Experiment

The notebook `17_expanded_cage4_sci_experiment.ipynb` reruns stages 11-14 with a larger CAGE-4 budget:

- 20 seeds,
- 3 episodes per seed,
- 100 environment steps,
- at least 30 held-out paired CybSAGE-vs-baseline comparisons.

Use this notebook for the paper-grade evidence run after the quick stages have passed.
