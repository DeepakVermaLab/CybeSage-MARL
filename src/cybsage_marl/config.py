from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    local_dataset_root: Path
    cyborg_dir: Path
    mitre_file: Path
    nvd_dir: Path
    output_root: Path
    log_root: Path
    drive_output_folder_id: str
    rclone_remote: str
    seed: int = 42
    device: str = "auto"

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def output_dir(self, stage: str) -> Path:
        path = self.repo_root / self.output_root / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log_dir(self, stage: str) -> Path:
        path = self.repo_root / self.log_root / stage
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: str | Path | None = None) -> ProjectConfig:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(path) if path else repo_root / "configs" / "project.json"
    data: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))

    output_root = Path(os.environ.get("CYBSAGE_OUTPUT_ROOT", data["output_root"]))
    log_root = Path(os.environ.get("CYBSAGE_LOG_ROOT", data["log_root"]))
    dataset_root = _resolve_dataset_root(Path(data["local_dataset_root"]))

    return ProjectConfig(
        project_name=data["project_name"],
        local_dataset_root=dataset_root,
        cyborg_dir=dataset_root / "CybORG_CAGE_Challenge_4",
        mitre_file=dataset_root / "MITRE_ATTACK_Enterprise_STIX" / "enterprise-attack.json",
        nvd_dir=dataset_root / "NVD_CVE_API_2_0",
        output_root=output_root,
        log_root=log_root,
        drive_output_folder_id=data["drive_output_folder_id"],
        rclone_remote=data["rclone_remote"],
        seed=int(os.environ.get("CYBSAGE_SEED", data.get("seed", 42))),
        device=os.environ.get("CYBSAGE_DEVICE", data.get("device", "auto")),
    )


def _resolve_dataset_root(configured_root: Path) -> Path:
    candidates = []
    env_root = os.environ.get("CYBSAGE_DATASET_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            configured_root,
            Path("/content/drive/MyDrive/projects/Dataset/Security"),
            Path("/content/drive/MyDrive/Datasets/Security"),
            Path("/content/drive/MyDrive/Security"),
            Path.home() / "Datasets/Security",
        ]
    )
    for candidate in candidates:
        if _looks_like_dataset_root(candidate):
            return candidate
    return configured_root


def _looks_like_dataset_root(path: Path) -> bool:
    return (
        path.exists()
        and (path / "CybORG_CAGE_Challenge_4").exists()
        and (path / "MITRE_ATTACK_Enterprise_STIX").exists()
        and (path / "NVD_CVE_API_2_0").exists()
    )
