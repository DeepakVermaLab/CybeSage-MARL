from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cybsage_marl.utils.io import write_json


def file_record(path: Path, required: bool, description: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "name": path.name,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "required": required,
        "description": description,
    }


def write_stage_manifest(
    out_dir: Path,
    stage: str,
    artifacts: list[tuple[Path, bool, str]],
    metrics: dict[str, Any] | None = None,
    notes: str | None = None,
) -> Path:
    payload = {
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics or {},
        "notes": notes or "",
        "artifacts": [file_record(path, required, description) for path, required, description in artifacts],
    }
    return write_json(out_dir / "stage_manifest.json", payload)
