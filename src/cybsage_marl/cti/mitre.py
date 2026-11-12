from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_mitre_tables(mitre_file: Path) -> dict[str, pd.DataFrame]:
    data = json.loads(mitre_file.read_text(encoding="utf-8"))
    objects = data.get("objects", [])
    techniques = []
    tactics = []
    relationships = []
    for obj in objects:
        typ = obj.get("type")
        if typ == "attack-pattern" and not obj.get("revoked", False):
            techniques.append(
                {
                    "id": obj.get("id"),
                    "name": obj.get("name"),
                    "description": obj.get("description", ""),
                    "kill_chain_phases": ",".join(
                        p.get("phase_name", "") for p in obj.get("kill_chain_phases", [])
                    ),
                }
            )
        elif typ == "x-mitre-tactic":
            tactics.append(
                {
                    "id": obj.get("id"),
                    "name": obj.get("name"),
                    "shortname": obj.get("x_mitre_shortname"),
                    "description": obj.get("description", ""),
                }
            )
        elif typ == "relationship":
            relationships.append(
                {
                    "id": obj.get("id"),
                    "relationship_type": obj.get("relationship_type"),
                    "source_ref": obj.get("source_ref"),
                    "target_ref": obj.get("target_ref"),
                }
            )
    return {
        "techniques": pd.DataFrame(techniques),
        "tactics": pd.DataFrame(tactics),
        "relationships": pd.DataFrame(relationships),
    }
