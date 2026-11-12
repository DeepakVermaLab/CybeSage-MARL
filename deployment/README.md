# CybSAGE-MARL Research Console

Static, publication-oriented web interface over retained CybSAGE implementation artifacts.

## Evidence represented

- Recorded CAGE-4 multi-agent step traces for 30 held-out episodes.
- Six executed baseline summaries and paired bootstrap confidence intervals.
- Synthetic-trajectory prototype ablation and explanation records.
- Retained model/embedding sizes and recorded wall-clock episode runtime.
- Explicit indication that energy was not instrumented.

The console does not run model inference in the browser and does not conflate the recorded CAGE-4 profile trace with the separate prototype explanation records.

## Refresh the data bundle

Run from the implementation repository root:

```bash
python deployment/scripts/export_artifacts.py
```

## Local preview

```bash
python -m http.server 8765 --directory deployment/dist
```

Open `http://127.0.0.1:8765`.

## GitHub Pages

The static site lives in `deployment/dist`. The included workflow at `.github/workflows/pages.yml` publishes this directory when manually dispatched or when the deployment files change on `main`.

Before public deployment, confirm repository visibility, Pages permissions, and whether project data should be public.

## Paper screenshots

Paper-ready screenshots are stored in `deployment/screenshots`. The screenshot manifest records viewport and view parameters.
