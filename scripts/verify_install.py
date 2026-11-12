#!/usr/bin/env python3
from __future__ import annotations

from cybsage_marl.config import load_config
from cybsage_marl.data.synthetic_env import generate_trajectories
from cybsage_marl.utils.device import resolve_device


def main() -> None:
    cfg = load_config()
    assert cfg.project_name == "CybSAGE-MARL"
    assert cfg.local_dataset_root.exists(), cfg.local_dataset_root
    assert cfg.cyborg_dir.exists(), cfg.cyborg_dir
    assert cfg.mitre_file.exists(), cfg.mitre_file
    assert cfg.nvd_dir.exists(), cfg.nvd_dir
    rows = generate_trajectories(episodes=2, seed=cfg.seed, max_steps=4)
    assert rows and "observation" in rows[0]
    print("verify_install OK")
    print("device:", resolve_device(cfg.device))
    print("dataset_root:", cfg.local_dataset_root)
    print("drive_output_folder_id:", cfg.drive_output_folder_id)


if __name__ == "__main__":
    main()
