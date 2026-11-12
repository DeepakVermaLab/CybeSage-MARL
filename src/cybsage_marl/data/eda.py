from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cybsage_marl.config import ProjectConfig
from cybsage_marl.cti.mitre import load_mitre_tables
from cybsage_marl.utils.io import iter_jsonl, read_json, write_json
from cybsage_marl.utils.tracking import write_stage_manifest


def _dataset_inventory(cfg: ProjectConfig) -> pd.DataFrame:
    rows = []
    for name, path in {
        "CybORG_CAGE_Challenge_4": cfg.cyborg_dir,
        "MITRE_ATTACK_Enterprise_STIX": cfg.mitre_file,
        "NVD_CVE_API_2_0": cfg.nvd_dir,
    }.items():
        rows.append(
            {
                "dataset": name,
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file(),
                "is_dir": path.is_dir(),
                "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            }
        )
    return pd.DataFrame(rows)


def _trajectory_eda(trajectory_path: Path, out_dir: Path) -> dict[str, Any]:
    if not trajectory_path.exists():
        return {"trajectory_exists": False}
    rows = list(iter_jsonl(trajectory_path))
    frame = pd.DataFrame(rows)
    metrics: dict[str, Any] = {
        "trajectory_exists": True,
        "trajectory_rows": int(len(frame)),
        "episodes": int(frame["episode_id"].nunique()) if "episode_id" in frame else 0,
        "mean_reward": float(frame["reward"].mean()) if "reward" in frame and not frame.empty else 0.0,
    }
    if "regime" in frame:
        regime_counts = frame["regime"].value_counts().rename_axis("regime").reset_index(name="count")
        regime_counts.to_csv(out_dir / "trajectory_regime_counts.csv", index=False)
    if "action" in frame:
        action_counts = frame["action"].value_counts().rename_axis("action").reset_index(name="count")
        action_counts.to_csv(out_dir / "trajectory_action_counts.csv", index=False)
    return metrics


def run_eda(cfg: ProjectConfig, out_dir: Path, trajectory_path: Path | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = _dataset_inventory(cfg)
    inventory.to_csv(out_dir / "dataset_inventory.csv", index=False)

    mitre_tables = load_mitre_tables(cfg.mitre_file)
    mitre_summary = {
        "techniques": int(len(mitre_tables["techniques"])),
        "tactics": int(len(mitre_tables["tactics"])),
        "relationships": int(len(mitre_tables["relationships"])),
    }

    nvd_manifest_path = cfg.nvd_dir / "download_manifest.json"
    nvd_manifest = read_json(nvd_manifest_path) if nvd_manifest_path.exists() else {}
    nvd_summary = {
        "records_downloaded": nvd_manifest.get("records_downloaded"),
        "page_files": nvd_manifest.get("page_files"),
        "bad_page_files": nvd_manifest.get("bad_page_files"),
        "is_full_download": nvd_manifest.get("is_full_download"),
    }

    trajectory_metrics = _trajectory_eda(trajectory_path, out_dir) if trajectory_path else {"trajectory_exists": False}
    summary = {
        "dataset_inventory_rows": int(len(inventory)),
        "mitre": mitre_summary,
        "nvd": nvd_summary,
        "trajectories": trajectory_metrics,
    }
    write_json(out_dir / "eda_summary.json", summary)
    write_stage_manifest(
        out_dir,
        "eda",
        [
            (out_dir / "dataset_inventory.csv", True, "Dataset path inventory and existence flags"),
            (out_dir / "eda_summary.json", True, "EDA summary for datasets and generated trajectories"),
            (out_dir / "trajectory_regime_counts.csv", False, "Regime distribution when trajectory file exists"),
            (out_dir / "trajectory_action_counts.csv", False, "Action distribution when trajectory file exists"),
        ],
        metrics=summary,
    )
    return summary
