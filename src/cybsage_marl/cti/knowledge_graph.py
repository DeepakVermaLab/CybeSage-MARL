from __future__ import annotations

from pathlib import Path

import pandas as pd

from cybsage_marl.cti.mitre import load_mitre_tables
from cybsage_marl.cti.nvd import nvd_to_table


ACTION_TO_TACTIC = {
    "monitor": "reconnaissance",
    "analyze": "discovery",
    "restore": "impact",
    "isolate": "lateral-movement",
    "deploy_decoy": "defense-evasion",
}


def build_compact_graph(mitre_file: Path, nvd_dir: Path, out_dir: Path, nvd_limit: int = 5000) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = load_mitre_tables(mitre_file)
    nvd = nvd_to_table(nvd_dir, limit=nvd_limit)

    techniques = tables["techniques"]
    tactics = tables["tactics"]
    action_edges = pd.DataFrame(
        [{"source": action, "target": tactic, "edge_type": "defensive-action-context"} for action, tactic in ACTION_TO_TACTIC.items()]
    )
    cve_edges = pd.DataFrame(
        [
            {"source": row.cve_id, "target": "vulnerability-grounding", "edge_type": "evidence-for"}
            for row in nvd.itertuples()
            if row.cve_id
        ]
    )
    edge_list = pd.concat([action_edges, cve_edges], ignore_index=True)

    paths = {
        "techniques": out_dir / "mitre_techniques.csv",
        "tactics": out_dir / "mitre_tactics.csv",
        "nvd_sample": out_dir / "nvd_cve_sample.csv",
        "edges": out_dir / "cti_graph_edges.csv",
    }
    techniques.to_csv(paths["techniques"], index=False)
    tactics.to_csv(paths["tactics"], index=False)
    nvd.to_csv(paths["nvd_sample"], index=False)
    edge_list.to_csv(paths["edges"], index=False)
    return paths
