from __future__ import annotations

from pathlib import Path

import pandas as pd

from cybsage_marl.cti.knowledge_graph import ACTION_TO_TACTIC
from cybsage_marl.utils.io import iter_jsonl, write_json


def generate_explanations(trajectory_path: Path, cti_dir: Path, out_dir: Path, limit: int = 100) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    techniques_path = cti_dir / "mitre_techniques.csv"
    techniques = pd.read_csv(techniques_path) if techniques_path.exists() else pd.DataFrame()
    explanations = []
    for idx, row in enumerate(iter_jsonl(trajectory_path)):
        if idx >= limit:
            break
        action = row["action"]
        tactic = ACTION_TO_TACTIC.get(action, "unknown")
        evidence = []
        if not techniques.empty and "kill_chain_phases" in techniques.columns:
            subset = techniques[techniques["kill_chain_phases"].fillna("").str.contains(tactic, case=False, regex=False)].head(3)
            evidence = subset[["name", "kill_chain_phases"]].to_dict("records")
        explanations.append(
            {
                "episode_id": row["episode_id"],
                "step": row["step"],
                "action": action,
                "regime": row["regime"],
                "explanation": (
                    f"Selected {action} while the inferred attacker regime was {row['regime']}. "
                    f"The action is grounded in the {tactic} context and the observed trajectory state."
                ),
                "evidence": evidence,
                "evidence_count": len(evidence),
            }
        )
    pd.DataFrame(explanations).to_csv(out_dir / "action_explanations.csv", index=False)
    write_json(out_dir / "action_explanations.json", explanations)
    summary = {"explanations": len(explanations), "with_evidence": sum(1 for e in explanations if e["evidence_count"] > 0)}
    write_json(out_dir / "explanation_summary.json", summary)
    return summary
