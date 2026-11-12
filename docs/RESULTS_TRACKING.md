# Results Tracking

The implementation now stores required and extra artifacts for every stage.

## Required Outputs

- `outputs/01_audit/`: dataset audit JSON/CSV and manifest.
- `outputs/02_eda/`: dataset inventory, MITRE/NVD counts, trajectory distributions when available.
- `outputs/03_trajectories/`: generated trajectory JSONL and trajectory summary.
- `outputs/04_preprocessing/`: feature matrices, row metadata, split counts, feature statistics.
- `outputs/05_cti_graph/`: MITRE/NVD evidence tables and graph edges.
- `outputs/06_ssl/`: SSL model, embeddings, training log, summary.
- `outputs/07_posterior_policy/`: policy model, policy training log, dataset index, summary.
- `outputs/08_explanations/`: action explanations, explanation CSV/JSON, summary.
- `outputs/09_evaluation/`: evaluation summary CSV/JSON.
- `outputs/10_ablation_sota/`: seed-wise baselines, ablations, and aggregate ranking.
- `outputs/11_statistical_analysis/`: paired tests, SOTA matrix, and report.
- `outputs/12_cage4_dependency_check/`: CAGE-4 import/dependency readiness.
- `outputs/13_cage4_baseline_rollouts/`: real CAGE-4 baseline reward rollouts.
- `outputs/14_cybsage_cage4_policy/`: CybSAGE CAGE-4 reward rollouts and profile tuning.
- `outputs/15_sota_reward_comparison/`: paired reward comparison and superiority gate.
- `outputs/16_sci_readiness_pack/`: CAGE-4 effect sizes, profile ablation table, and SCI claim-readiness report.

Each stage writes `stage_manifest.json`, which marks artifacts as:

- `required: true`: needed for paper results, reruns, or downstream stages.
- `required: false`: useful diagnostic/extra output, not mandatory for paper claims.

## Not Stored in Git

Large/generated outputs are ignored by `.gitignore`:

- `outputs/`
- `logs/`
- `artifacts/`
- `checkpoints/`
- model files such as `*.pt`
- arrays such as `*.npy`

Use `scripts/sync_outputs.py` to copy outputs to the configured Google Drive folder.
