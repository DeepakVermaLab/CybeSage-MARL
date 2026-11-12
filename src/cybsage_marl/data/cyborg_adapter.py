from __future__ import annotations

import sys
from pathlib import Path


def add_cyborg_to_path(cyborg_repo: Path) -> bool:
    if not cyborg_repo.exists():
        return False
    repo = str(cyborg_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return True


def cyborg_import_status(cyborg_repo: Path) -> dict[str, object]:
    added = add_cyborg_to_path(cyborg_repo)
    result: dict[str, object] = {"repo": str(cyborg_repo), "path_added": added}
    try:
        import CybORG  # type: ignore

        result["import_ok"] = True
        result["module_file"] = getattr(CybORG, "__file__", None)
    except Exception as exc:
        result["import_ok"] = False
        result["error"] = str(exc)
    return result
