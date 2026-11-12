from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from cybsage_marl.data.cyborg_adapter import add_cyborg_to_path
from cybsage_marl.cage4.compat import install_ray_rllib_stub
from cybsage_marl.utils.io import write_json


REQUIRED_IMPORTS = [
    ("gym", "gym"),
    ("gymnasium", "gymnasium"),
    ("CybORG", "CybORG"),
]

OPTIONAL_IMPORTS = [
    ("ray", "ray", "Only required for RLlib training examples; CPU rollouts use a lightweight compatibility shim."),
    ("pettingzoo", "pettingzoo", "Only required for PettingZoo API tests/examples, not for stages 13-15 CPU rollouts."),
]


def check_cage4_dependencies(cyborg_repo: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    add_cyborg_to_path(cyborg_repo)
    rows: list[dict[str, Any]] = []
    for package, import_name in REQUIRED_IMPORTS:
        row: dict[str, Any] = {"package": package, "import_name": import_name, "required_for_cpu_rollouts": True}
        try:
            module = importlib.import_module(import_name)
            row["ok"] = True
            row["version"] = getattr(module, "__version__", None)
            row["file"] = getattr(module, "__file__", None)
            row["error"] = ""
        except Exception as exc:
            row["ok"] = False
            row["version"] = None
            row["file"] = None
            row["error"] = str(exc)
        rows.append(row)

    for package, import_name, note in OPTIONAL_IMPORTS:
        row = {"package": package, "import_name": import_name, "required_for_cpu_rollouts": False, "note": note}
        try:
            module = importlib.import_module(import_name)
            row["ok"] = True
            row["version"] = getattr(module, "__version__", None)
            row["file"] = getattr(module, "__file__", None)
            row["error"] = ""
        except Exception as exc:
            row["ok"] = False
            row["version"] = None
            row["file"] = None
            row["error"] = str(exc)
        rows.append(row)

    wrapper_import_ok = False
    wrapper_import_error = ""
    ray_stub_used = False
    try:
        ray_stub_used = install_ray_rllib_stub()
        from CybORG.Agents.Wrappers import BlueEnterpriseWrapper  # type: ignore

        wrapper_import_ok = BlueEnterpriseWrapper is not None
    except Exception as exc:
        wrapper_import_error = str(exc)

    cyborg_repo_ok = cyborg_repo.exists() and (cyborg_repo / "CybORG").exists()
    required_imports_ok = all(bool(row["ok"]) for row in rows if row["required_for_cpu_rollouts"])
    ready = bool(cyborg_repo_ok and required_imports_ok and wrapper_import_ok)
    install_hint = (
        "For RLlib/PettingZoo experiments only, run: python -m pip install -r "
        f"{cyborg_repo / 'Requirements.txt'}. Stages 13-15 need ready_for_cage4_rollouts=true."
    )
    summary = {
        "cyborg_repo": str(cyborg_repo),
        "cyborg_repo_ok": cyborg_repo_ok,
        "python": sys.executable,
        "ready_for_cage4_rollouts": ready,
        "missing_required_packages": [
            row["package"] for row in rows if row["required_for_cpu_rollouts"] and not row["ok"]
        ],
        "missing_optional_packages": [
            row["package"] for row in rows if not row["required_for_cpu_rollouts"] and not row["ok"]
        ],
        "wrapper_import_ok": wrapper_import_ok,
        "wrapper_import_error": wrapper_import_error,
        "ray_rllib_stub_used": ray_stub_used,
        "install_hint": "" if ready else install_hint,
    }
    pd.DataFrame(rows).to_csv(out_dir / "dependency_status.csv", index=False)
    write_json(out_dir / "dependency_summary.json", summary)
    return summary
