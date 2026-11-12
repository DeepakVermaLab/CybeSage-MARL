#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from cybsage_marl.config import load_config


EXCLUDES = [
    ".git/**",
    "outputs/**",
    "logs/**",
    "artifacts/**",
    "checkpoints/**",
    "__pycache__/**",
    "._**",
    ".DS_Store",
    ".ipynb_checkpoints/**",
]


def main() -> None:
    cfg = load_config()
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is not available in PATH.")
    remote = f"{cfg.rclone_remote}IMPLEMENTATION"
    cmd = [
        "rclone",
        "copy",
        str(cfg.repo_root),
        remote,
        "--drive-root-folder-id",
        cfg.drive_output_folder_id,
        "--create-empty-src-dirs",
        "--progress",
    ]
    for pattern in EXCLUDES:
        cmd.extend(["--exclude", pattern])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Uploaded implementation source to Drive folder ID:", cfg.drive_output_folder_id)
    print("Expected Colab path after mounting Drive depends on your Drive layout.")
    print("Set REPO_ROOT_OVERRIDE in notebooks to the mounted IMPLEMENTATION folder path.")


if __name__ == "__main__":
    main()
