from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from cybsage_marl.config import ProjectConfig
from cybsage_marl.utils.io import write_json


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["exists"] = True
    return payload


def audit_cyborg(cfg: ProjectConfig) -> dict[str, Any]:
    repo = cfg.cyborg_dir / "cage-challenge-4"
    manifest = _safe_read_json(cfg.cyborg_dir / "download_manifest.json")
    result: dict[str, Any] = {
        "dataset": "CybORG_CAGE_Challenge_4",
        "folder": str(cfg.cyborg_dir),
        "repo": str(repo),
        "repo_exists": repo.exists(),
        "manifest": manifest,
    }
    if repo.exists():
        try:
            local_head = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception as exc:
            local_head = f"ERROR: {exc}"
        result["local_git_head"] = local_head
    return result


def audit_mitre(cfg: ProjectConfig) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset": "MITRE_ATTACK_Enterprise_STIX",
        "file": str(cfg.mitre_file),
        "exists": cfg.mitre_file.exists(),
    }
    if not cfg.mitre_file.exists():
        return result
    data = json.loads(cfg.mitre_file.read_text(encoding="utf-8"))
    objects = data.get("objects", [])
    result.update(
        {
            "bytes": cfg.mitre_file.stat().st_size,
            "objects": len(objects),
            "attack_patterns": sum(1 for o in objects if o.get("type") == "attack-pattern"),
            "tactics": sum(1 for o in objects if o.get("type") == "x-mitre-tactic"),
        }
    )
    return result


def audit_nvd(cfg: ProjectConfig) -> dict[str, Any]:
    manifest_path = cfg.nvd_dir / "download_manifest.json"
    manifest = _safe_read_json(manifest_path)
    result: dict[str, Any] = {
        "dataset": "NVD_CVE_API_2_0",
        "folder": str(cfg.nvd_dir),
        "manifest": manifest,
    }
    pages_dir = cfg.nvd_dir / "pages"
    if pages_dir.exists():
        result["page_files_on_disk"] = len(list(pages_dir.glob("nvd_cve_start_*.json")))
    full_json = cfg.nvd_dir / "nvd_cve_records_full.json"
    if full_json.exists():
        result["full_json_bytes"] = full_json.stat().st_size
    return result


def audit_all(cfg: ProjectConfig, out_dir: Path | None = None) -> dict[str, Any]:
    summary = {
        "cyborg": audit_cyborg(cfg),
        "mitre": audit_mitre(cfg),
        "nvd": audit_nvd(cfg),
    }
    if out_dir:
        write_json(out_dir / "dataset_audit.json", summary)
        rows = []
        for name, payload in summary.items():
            row = {"dataset": name}
            row.update({k: v for k, v in payload.items() if not isinstance(v, (dict, list))})
            rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "dataset_audit.csv", index=False)
    return summary
