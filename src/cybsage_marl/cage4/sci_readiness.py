from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from cybsage_marl.utils.io import read_json, write_json


def _bootstrap_ci(values: np.ndarray, seed: int = 42, draws: int = 5000) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _cohen_d(values: np.ndarray) -> float:
    if values.size < 2:
        return math.nan
    std = float(values.std(ddof=1))
    if std == 0.0:
        return math.inf if float(values.mean()) > 0.0 else 0.0
    return float(values.mean() / std)


def _holm_bonferroni(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda idx: p_values[idx])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(p_values[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted


def _load_episode_tables(baseline_dir: Path, cybsage_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = baseline_dir / "cage4_baseline_episode_results.csv"
    cybsage_path = cybsage_dir / "cybsage_cage4_episode_results.csv"
    if not baseline_path.exists() or not cybsage_path.exists():
        missing = [str(p) for p in [baseline_path, cybsage_path] if not p.exists()]
        raise RuntimeError(f"Missing CAGE-4 episode table(s): {missing}")
    try:
        baseline = pd.read_csv(baseline_path)
        cybsage = pd.read_csv(cybsage_path)
    except EmptyDataError as exc:
        raise RuntimeError(
            "CAGE-4 episode table exists but is empty. Rerun stages 13 and 14 before stage 16."
        ) from exc
    if baseline.empty or cybsage.empty:
        raise RuntimeError("CAGE-4 episode table has no rows. Rerun stages 13 and 14 before stage 16.")
    return baseline, cybsage


def _effect_table(baseline: pd.DataFrame, cybsage: pd.DataFrame) -> pd.DataFrame:
    cyb = cybsage[["seed", "episode", "total_reward"]].rename(columns={"total_reward": "cybsage_reward"})
    rows: list[dict[str, Any]] = []
    for policy in sorted(baseline["policy"].unique()):
        base = baseline[baseline["policy"] == policy][["seed", "episode", "total_reward"]].rename(
            columns={"total_reward": "baseline_reward"}
        )
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
                "median_difference": float(np.median(diffs)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "win_rate": float((diffs > 0).mean()),
                "tie_rate": float((diffs == 0).mean()),
                "loss_rate": float((diffs < 0).mean()),
                "cohen_d_paired": _cohen_d(diffs),
            }
        )
    effect = pd.DataFrame(rows)
    if effect.empty:
        return effect
    effect["raw_superiority"] = (effect["mean_difference"] > 0) & (effect["ci95_low"] > 0)
    return effect.sort_values(["mean_difference", "win_rate"], ascending=[False, False])


def _ablation_table(cybsage_dir: Path) -> pd.DataFrame:
    path = cybsage_dir / "cybsage_profile_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if table.empty or "total_reward" not in table.columns:
        return table
    best = float(table["total_reward"].max())
    table["delta_from_best"] = table["total_reward"] - best
    table["rank"] = table["total_reward"].rank(ascending=False, method="min").astype(int)
    return table.sort_values(["rank", "profile"])


def _write_report(out_path: Path, summary: dict[str, Any], effect: pd.DataFrame, ablation: pd.DataFrame) -> None:
    lines = [
        "# SCI Readiness Report",
        "",
        "## Claim Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Paper claim level: `{summary['paper_claim_level']}`",
        f"- Superiority claim ready: `{summary['superiority_claim_ready']}`",
        f"- Minimum paired runs observed: `{summary['min_paired_runs']}`",
        f"- Recommended minimum paired runs: `{summary['recommended_min_paired_runs']}`",
        f"- Claim boundary: {summary['claim_boundary']}",
        "",
        "## CAGE-4 Effect Sizes",
        "",
    ]
    if effect.empty:
        lines.append("No paired effect table was produced.")
    else:
        lines.append("| baseline | paired_runs | cybsage_mean | baseline_mean | mean_difference | ci95_low | ci95_high | win_rate | tie_rate | cohen_d_paired |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in effect.to_dict("records"):
            lines.append(
                "| {baseline} | {paired_runs} | {cybsage_mean:.6g} | {baseline_mean:.6g} | {mean_difference:.6g} | {ci95_low:.6g} | {ci95_high:.6g} | {win_rate:.3f} | {tie_rate:.3f} | {cohen_d_paired:.6g} |".format(
                    **row
                )
            )
    lines.extend(["", "## Profile Ablation", ""])
    if ablation.empty:
        lines.append("No profile ablation table was found.")
    else:
        cols = [col for col in ["rank", "profile", "total_reward", "delta_from_best", "selection_priority"] if col in ablation.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---" if col == "profile" else "---:" for col in cols]) + " |")
        for row in ablation[cols].to_dict("records"):
            rendered = []
            for col in cols:
                val = row[col]
                rendered.append(f"{val:.6g}" if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(rendered) + " |")
    lines.extend(
        [
            "",
            "## Required Next Evidence For SCI Q1",
            "",
            "- Run more paired CAGE-4 episodes and seeds before making a Q1-level superiority claim.",
            "- Include both finite-state and random-select red-agent settings.",
            "- Keep the claim bounded if CybSAGE ties the strongest baseline instead of beating it.",
            "- Use this report as the paper claim gate, not notebook completion alone.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sci_readiness_pack(
    baseline_dir: Path,
    cybsage_dir: Path,
    sota_dir: Path,
    out_dir: Path,
    recommended_min_paired_runs: int = 30,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        baseline, cybsage = _load_episode_tables(baseline_dir, cybsage_dir)
    except RuntimeError as exc:
        summary = {
            "status": "blocked",
            "paper_claim_level": "blocked",
            "superiority_claim_ready": False,
            "stage15_superiority_claim_ready": False,
            "baselines_compared": 0,
            "min_paired_runs": 0,
            "recommended_min_paired_runs": recommended_min_paired_runs,
            "all_positive_ci": False,
            "claim_boundary": str(exc),
        }
        pd.DataFrame().to_csv(out_dir / "cage4_effect_sizes.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "cage4_profile_ablation_table.csv", index=False)
        write_json(out_dir / "sci_readiness_summary.json", summary)
        _write_report(out_dir / "sci_readiness_report.md", summary, pd.DataFrame(), pd.DataFrame())
        return summary
    effect = _effect_table(baseline, cybsage)
    ablation = _ablation_table(cybsage_dir)

    effect_path = out_dir / "cage4_effect_sizes.csv"
    ablation_path = out_dir / "cage4_profile_ablation_table.csv"
    effect.to_csv(effect_path, index=False)
    ablation.to_csv(ablation_path, index=False)

    sota_summary_path = sota_dir / "cage4_sota_summary.json"
    sota_summary = read_json(sota_summary_path) if sota_summary_path.exists() else {}
    min_paired_runs = int(effect["paired_runs"].min()) if not effect.empty else 0
    all_positive_ci = bool(not effect.empty and effect["raw_superiority"].all())
    enough_runs = min_paired_runs >= recommended_min_paired_runs
    superiority_ready = bool(all_positive_ci and enough_runs)

    if superiority_ready:
        paper_claim_level = "SCI-ready superiority claim against executed CAGE-4 baselines"
        claim_boundary = "CybSAGE beat every executed baseline with positive paired bootstrap confidence and enough paired runs."
    elif all_positive_ci:
        paper_claim_level = "preliminary superiority; needs larger episode/seed budget"
        claim_boundary = "CybSAGE beat every executed baseline, but paired-run count is below the SCI-readiness gate."
    elif not effect.empty and (effect["mean_difference"] >= 0).all():
        paper_claim_level = "competitive/tie with strongest baseline"
        claim_boundary = "CybSAGE is competitive but at least one baseline is tied or lacks positive confidence."
    else:
        paper_claim_level = "prototype evidence only"
        claim_boundary = "CybSAGE does not yet beat every executed CAGE-4 baseline."

    summary = {
        "status": "completed",
        "paper_claim_level": paper_claim_level,
        "superiority_claim_ready": superiority_ready,
        "stage15_superiority_claim_ready": bool(sota_summary.get("superiority_claim_ready", False)),
        "baselines_compared": int(len(effect)),
        "min_paired_runs": min_paired_runs,
        "recommended_min_paired_runs": recommended_min_paired_runs,
        "all_positive_ci": all_positive_ci,
        "claim_boundary": claim_boundary,
    }
    write_json(out_dir / "sci_readiness_summary.json", summary)
    _write_report(out_dir / "sci_readiness_report.md", summary, effect, ablation)
    return summary
