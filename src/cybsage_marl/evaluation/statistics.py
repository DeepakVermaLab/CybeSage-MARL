from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cybsage_marl.utils.io import read_json, write_json


SOTA_COMPARISON_ROWS = [
    {
        "paper": "Posterior-Conditioned Multi-Agent Reinforcement Learning for Adaptive Cyber Defense",
        "year": 2026,
        "environment_or_dataset": "CybORG CAGE-4",
        "method_family": "Posterior-conditioned MARL",
        "primary_metrics": "Mission reward, adaptation under mixed/switching attack regimes",
        "directly_comparable_now": False,
        "reason": "Requires real CAGE-4 reward-based multi-agent rollout; current CybSAGE run is preliminary trajectory classification/evidence evaluation.",
    },
    {
        "paper": "Trident: How to Break Deep Reinforcement Learning Cyber Defenses",
        "year": 2026,
        "environment_or_dataset": "CAGE-4 + CyberWheel",
        "method_family": "Agentic LLM/RLVR red-team benchmark",
        "primary_metrics": "Blue-agent cumulative episodic reward under adaptive generated attacks",
        "directly_comparable_now": False,
        "reason": "Compares adaptive red-agent attack strength against trained blue defenders; not the same task as current blue-policy action prediction.",
    },
    {
        "paper": "Automated Cyber Defense with Generalizable Graph-based Reinforcement Learning Agents",
        "year": 2025,
        "environment_or_dataset": "Yawning Titan, CAGE-2, CAGE-4",
        "method_family": "Graph-based RL",
        "primary_metrics": "Reward/generalization on unseen graph topologies",
        "directly_comparable_now": False,
        "reason": "Needs matched CAGE-4/graph generalization protocol and reward metrics.",
    },
    {
        "paper": "Large Language Models are Autonomous Cyber Defenders",
        "year": 2025,
        "environment_or_dataset": "CybORG CAGE-4",
        "method_family": "LLM + RL defender teams",
        "primary_metrics": "CAGE-4 reward, runtime, red-agent strategy robustness",
        "directly_comparable_now": False,
        "reason": "Needs live CAGE-4 defender rollout and LLM/RL team baselines.",
    },
    {
        "paper": "Learning to Communicate in Multi-Agent Reinforcement Learning for Autonomous Cyber Defence",
        "year": 2025,
        "environment_or_dataset": "CybORG scenarios",
        "method_family": "DIAL communication MARL vs QMIX",
        "primary_metrics": "Mean return across small/large network games",
        "directly_comparable_now": False,
        "reason": "Needs the same communication-game setup or a mapped communication ablation.",
    },
    {
        "paper": "Design and evaluation of an Autonomous Cyber Defence agent using DRL and an augmented LLM",
        "year": 2025,
        "environment_or_dataset": "Cyber gym + CTI/MITRE/CVE knowledge graphs",
        "method_family": "DRL + augmented LLM/RAG + CTI KG",
        "primary_metrics": "Mitigation/recovery, CVE-to-MITRE precision/recall/macro-F1, RAG quality",
        "directly_comparable_now": False,
        "reason": "Current macro-F1 is policy-action prediction, not CVE-to-MITRE mapping or RAG QA faithfulness.",
    },
]


def _sign_test_p_value(diffs: np.ndarray) -> float:
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    wins = int((nonzero > 0).sum())
    k = min(wins, n - wins)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def _bootstrap_ci(diffs: np.ndarray, seed: int = 42, draws: int = 5000) -> tuple[float, float]:
    if len(diffs) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    samples = rng.choice(diffs, size=(draws, len(diffs)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _paired_statistics(results: pd.DataFrame, primary_method: str = "cybsage_full") -> pd.DataFrame:
    rows = []
    primary = results[results["method"] == primary_method][["seed", "macro_f1", "accuracy", "decision_utility"]]
    for method in sorted(set(results["method"]) - {primary_method}):
        other = results[results["method"] == method][["seed", "macro_f1", "accuracy", "decision_utility"]]
        merged = primary.merge(other, on="seed", suffixes=("_primary", "_baseline"))
        if merged.empty:
            continue
        for metric in ["macro_f1", "accuracy", "decision_utility"]:
            diffs = (merged[f"{metric}_primary"] - merged[f"{metric}_baseline"]).to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_ci(diffs)
            rows.append(
                {
                    "primary_method": primary_method,
                    "baseline_method": method,
                    "metric": metric,
                    "paired_runs": int(len(diffs)),
                    "mean_difference": float(diffs.mean()),
                    "std_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                    "cohens_dz": float(diffs.mean() / diffs.std(ddof=1)) if len(diffs) > 1 and diffs.std(ddof=1) > 0 else math.nan,
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "sign_test_p_value": _sign_test_p_value(diffs),
                    "primary_wins": int((diffs > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _read_optional_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.exists() else {}


def _resolve_ablation_dir(ablation_dir: Path) -> Path:
    seed_results_path = ablation_dir / "ablation_seed_results.csv"
    summary_path = ablation_dir / "ablation_summary.csv"
    if seed_results_path.exists() and summary_path.exists():
        return ablation_dir

    # Some Drive syncs may contain stage folders at the repository root, e.g.
    # IMPLEMENTATION/10_ablation_sota, instead of IMPLEMENTATION/outputs/10_ablation_sota.
    if len(ablation_dir.parents) >= 2:
        root_level_dir = ablation_dir.parents[1] / ablation_dir.name
        if (root_level_dir / "ablation_seed_results.csv").exists() and (root_level_dir / "ablation_summary.csv").exists():
            return root_level_dir

    return ablation_dir


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            display[col] = display[col].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, sep, *body])


def _write_markdown_report(
    out_path: Path,
    evaluation_summary: dict[str, Any],
    ablation_summary: pd.DataFrame,
    stats: pd.DataFrame,
    sota: pd.DataFrame,
) -> None:
    lines = [
        "# CybSAGE-MARL Statistical and SOTA Readiness Report",
        "",
        "## Current Verified Pipeline Metrics",
        "",
    ]
    if evaluation_summary:
        for key, value in evaluation_summary.items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- No `outputs/09_evaluation/evaluation_summary.json` file was found.")

    lines.extend(
        [
            "",
            "## Internal Baseline/Ablation Ranking",
            "",
            _markdown_table(ablation_summary) if not ablation_summary.empty else "No ablation summary available.",
            "",
            "## Paired Statistical Tests",
            "",
            _markdown_table(stats) if not stats.empty else "No paired statistics available.",
            "",
            "## SOTA Comparability Matrix",
            "",
            _markdown_table(sota),
            "",
            "## Interpretation Boundary",
            "",
            "These outputs support preliminary and ablation-level claims for the current implementation. "
            "They do not establish superiority over CAGE-4 SOTA papers until real CAGE-4 rollouts, matched baselines, "
            "multiple seeds, and reward/adaptation metrics are executed under the same protocol.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_statistical_analysis(
    evaluation_dir: Path,
    ablation_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation_summary = _read_optional_json(evaluation_dir / "evaluation_summary.json")

    ablation_dir = _resolve_ablation_dir(ablation_dir)
    seed_results_path = ablation_dir / "ablation_seed_results.csv"
    summary_path = ablation_dir / "ablation_summary.csv"
    if not seed_results_path.exists() or not summary_path.exists():
        raise RuntimeError(
            f"Missing ablation inputs. Expected {seed_results_path} and {summary_path}. "
            "Run stage `ablation` before `statistical`."
        )

    seed_results = pd.read_csv(seed_results_path)
    ablation_summary = pd.read_csv(summary_path)
    stats = _paired_statistics(seed_results)
    sota = pd.DataFrame(SOTA_COMPARISON_ROWS)

    stats.to_csv(out_dir / "paired_statistical_tests.csv", index=False)
    sota.to_csv(out_dir / "sota_comparability_matrix.csv", index=False)
    ablation_summary.to_csv(out_dir / "paper_ablation_table.csv", index=False)
    _write_markdown_report(out_dir / "statistical_sota_report.md", evaluation_summary, ablation_summary, stats, sota)

    summary = {
        "ablation_methods": int(ablation_summary["method"].nunique()) if "method" in ablation_summary else 0,
        "paired_tests": int(len(stats)),
        "sota_papers_tracked": int(len(sota)),
        "direct_sota_comparison_ready": bool(sota["directly_comparable_now"].all()),
        "claim_boundary": "Preliminary internal ablation only; not yet CAGE-4 SOTA superiority.",
    }
    write_json(out_dir / "statistical_analysis_summary.json", summary)
    return summary
