# Curated Experiment Evidence

This directory contains lightweight summaries copied from the completed local output tree for repository review and manuscript traceability.

## Included evidence

- SSL and posterior-policy training summaries and epoch logs.
- Prototype explanation coverage.
- Ablation and paired statistical summaries.
- CAGE-4 baseline and selected-profile reward summaries.
- Paired effect sizes, profile ablation, and the bounded readiness report.

## Excluded evidence

Complete step traces, raw trajectories, model checkpoints, arrays, embeddings, and datasets are intentionally excluded from Git. They remain in the local/Drive artifact stores and can be regenerated through the notebooks and stage runner.

## Claim boundary

The real CAGE-4 evidence concerns a tuned rule/profile policy compared with six executed heuristic baselines. The prototype SSL, posterior-policy, explanation, and ablation artifacts are separate synthetic-pipeline results. These two evidence classes must not be presented as one end-to-end learned MARL experiment.
