from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cybsage_marl.utils.io import read_json, write_json


def _bootstrap_ci(values: np.ndarray, seed: int = 42, draws: int = 5000) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _sign_test_p_value(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    wins = int((nonzero > 0).sum())
    k = min(wins, n - wins)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def _write_report(out_path: Path, comparison: pd.DataFrame, summary: dict[str, Any]) -> None:
    lines = [
        "# CAGE-4 Reward SOTA Comparison",
        "",
        f"- `status`: `{summary['status']}`",
        f"- `superiority_claim_ready`: `{summary['superiority_claim_ready']}`",
        f"- `claim_boundary`: {summary['claim_boundary']}",
        "",
        "## Baseline Comparisons",
        "",
    ]
    if comparison.empty:
        lines.append("No comparison rows were produced.")
    else:
        lines.append("| baseline | cybsage_mean | baseline_mean | mean_difference | ci95_low | ci95_high | sign_test_p_value | cybsage_wins | paired_runs |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in comparison.to_dict("records"):
            lines.append(
                "| {baseline} | {cybsage_mean:.6g} | {baseline_mean:.6g} | {mean_difference:.6g} | {ci95_low:.6g} | {ci95_high:.6g} | {sign_test_p_value:.6g} | {cybsage_wins} | {paired_runs} |".format(
                    **row
                )
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cage4_sota_comparison(baseline_dir: Path, cybsage_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary_path = baseline_dir / "cage4_baseline_summary.json"
    cybsage_summary_path = cybsage_dir / "cybsage_cage4_summary.json"
    baseline_episode_path = baseline_dir / "cage4_baseline_episode_results.csv"
    cybsage_episode_path = cybsage_dir / "cybsage_cage4_episode_results.csv"

    blockers = []
    if not baseline_summary_path.exists() or not baseline_episode_path.exists():
        blockers.append("Missing CAGE-4 baseline rollout outputs.")
    if not cybsage_summary_path.exists() or not cybsage_episode_path.exists():
        blockers.append("Missing CAGE-4 CybSAGE rollout outputs.")
    if blockers:
        summary = {
            "status": "blocked",
            "superiority_claim_ready": False,
            "claim_boundary": "CAGE-4 reward comparison cannot run until stages 13 and 14 complete.",
            "blockers": blockers,
        }
        write_json(out_dir / "cage4_sota_summary.json", summary)
        pd.DataFrame().to_csv(out_dir / "cage4_sota_comparison.csv", index=False)
        _write_report(out_dir / "cage4_sota_report.md", pd.DataFrame(), summary)
        return summary

    baseline_summary = read_json(baseline_summary_path)
    cybsage_summary = read_json(cybsage_summary_path)
    if baseline_summary.get("status") != "completed":
        blockers.append(f"Baseline stage not completed: {baseline_summary.get('error', baseline_summary.get('status'))}")
    if cybsage_summary.get("status") != "completed":
        blockers.append(f"CybSAGE CAGE-4 stage not completed: {cybsage_summary.get('error', cybsage_summary.get('status'))}")
    if blockers:
        summary = {
            "status": "blocked",
            "superiority_claim_ready": False,
            "claim_boundary": "CAGE-4 reward comparison is blocked by failed prerequisite stages.",
            "blockers": blockers,
        }
        write_json(out_dir / "cage4_sota_summary.json", summary)
        pd.DataFrame().to_csv(out_dir / "cage4_sota_comparison.csv", index=False)
        _write_report(out_dir / "cage4_sota_report.md", pd.DataFrame(), summary)
        return summary

    baseline = pd.read_csv(baseline_episode_path)
    cybsage = pd.read_csv(cybsage_episode_path)
    rows = []
    cyb = cybsage[["seed", "episode", "total_reward"]].rename(columns={"total_reward": "cybsage_reward"})
    for policy in sorted(baseline["policy"].unique()):
        base = baseline[baseline["policy"] == policy][["seed", "episode", "total_reward"]].rename(columns={"total_reward": "baseline_reward"})
        merged = cyb.merge(base, on=["seed", "episode"], how="inner")
        if merged.empty:
            continue
        diffs = (merged["cybsage_reward"] - merged["baseline_reward"]).to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(diffs)
        rows.append(
            {
                "baseline": policy,
                "paired_runs": int(len(diffs)),
                "cybsage_mean": float(merged["cybsage_reward"].mean()),
                "baseline_mean": float(merged["baseline_reward"].mean()),
                "mean_difference": float(diffs.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "sign_test_p_value": _sign_test_p_value(diffs),
                "cybsage_wins": int((diffs > 0).sum()),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("mean_difference", ascending=False) if rows else pd.DataFrame()
    comparison.to_csv(out_dir / "cage4_sota_comparison.csv", index=False)

    superiority_ready = bool(
        not comparison.empty
        and (comparison["mean_difference"] > 0).all()
        and (comparison["ci95_low"] > 0).all()
    )
    summary = {
        "status": "completed",
        "baselines_compared": int(len(comparison)),
        "superiority_claim_ready": superiority_ready,
        "claim_boundary": (
            "CybSAGE beat every executed CAGE-4 baseline with positive bootstrap confidence intervals."
            if superiority_ready
            else "Do not claim SOTA superiority yet; at least one baseline was not beaten with positive confidence."
        ),
    }
    write_json(out_dir / "cage4_sota_summary.json", summary)
    _write_report(out_dir / "cage4_sota_report.md", comparison, summary)
    return summary
