# Repository Map

This document explains which files belong in the private research repository and which artifacts remain outside Git.

## Tracked research assets

| Path | Purpose |
|---|---|
| `configs/` | Project and dataset-path configuration |
| `docs/` | Pipeline, artifact policy, and repository guidance |
| `notebooks/` | Stage-wise notebooks with retained printed outputs |
| `scripts/` | Stage orchestration, validation, export, and synchronization tools |
| `src/cybsage_marl/` | Reusable implementation modules |
| `tests/` | Unit tests for configuration and synthetic-environment behavior |
| `results/` | Curated CSV, JSON, and Markdown evidence selected from completed runs |
| `logs/` | Small human-readable operational logs |
| `deployment/` | Source and generated bundle for the interactive research console |

## Local-only generated assets

The following remain outside Git because they are large, reproducible, environment-specific, or contain complete experiment traces:

- `outputs/`
- `artifacts/`
- `checkpoints/`
- model files (`*.pt`, `*.pth`, `*.ckpt`)
- arrays and embeddings (`*.npy`, `*.npz`, `*.pkl`)
- datasets and external CybORG checkouts
- virtual environments and notebook checkpoints

The complete local output tree remains the authoritative source for regeneration. The tracked `results/` directory is a compact publication-facing evidence layer, not a replacement for the full artifacts.

## Public simulation boundary

The separate `CybeSage-MARL-Simulation` repository contains only the static browser experience, its exported evidence bundle, screenshots, and GitHub Pages workflow. It excludes the private training code, notebooks, raw outputs, datasets, Drive identifiers, and machine-specific paths.
