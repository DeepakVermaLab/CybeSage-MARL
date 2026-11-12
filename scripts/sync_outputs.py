#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from cybsage_marl.config import load_config


def main() -> None:
    cfg = load_config()
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is not available in PATH.")
    source = cfg.repo_root / cfg.output_root
    if not source.exists():
        raise FileNotFoundError(f"Output folder not found: {source}")
    remote_target = f"{cfg.rclone_remote}IMPLEMENTATION/{cfg.output_root.as_posix()}"
    log_file = cfg.repo_root / cfg.log_root / "drive_sync.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rclone",
        "copy",
        str(source),
        remote_target,
        "--drive-root-folder-id",
        cfg.drive_output_folder_id,
        "--create-empty-src-dirs",
        "--exclude",
        "._**",
        "--exclude",
        ".DS_Store",
        "--progress",
        "--log-file",
        str(log_file),
        "--log-level",
        "INFO",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Synced outputs to Drive folder ID:", cfg.drive_output_folder_id, "under", cfg.output_root.as_posix())


if __name__ == "__main__":
    main()
