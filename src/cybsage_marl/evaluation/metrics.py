from __future__ import annotations

from pathlib import Path

import pandas as pd

from cybsage_marl.utils.io import iter_jsonl, read_json, write_json


def evaluate_pipeline(trajectory_path: Path, policy_dir: Path, explanation_dir: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(trajectory_path))
    rewards = [float(r["reward"]) for r in rows]
    policy_summary = read_json(policy_dir / "policy_summary.json")
    explanation_summary = read_json(explanation_dir / "explanation_summary.json")
    result = {
        "trajectory_rows": len(rows),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
        "policy_val_accuracy": policy_summary.get("val_accuracy"),
        "policy_val_macro_f1": policy_summary.get("val_macro_f1"),
        "explanations": explanation_summary.get("explanations"),
        "explanations_with_evidence": explanation_summary.get("with_evidence"),
        "evidence_coverage": explanation_summary.get("with_evidence", 0) / max(1, explanation_summary.get("explanations", 1)),
    }
    pd.DataFrame([result]).to_csv(out_dir / "evaluation_summary.csv", index=False)
    write_json(out_dir / "evaluation_summary.json", result)
    return result
