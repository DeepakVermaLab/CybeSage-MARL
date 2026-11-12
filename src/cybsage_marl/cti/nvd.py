from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd


def iter_nvd_records(nvd_dir: Path, limit: int | None = None) -> Iterator[dict]:
    count = 0
    pages_dir = nvd_dir / "pages"
    for path in sorted(pages_dir.glob("nvd_cve_start_*.json")):
        if path.name.startswith("._"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("vulnerabilities", []):
            yield row
            count += 1
            if limit is not None and count >= limit:
                return


def nvd_to_table(nvd_dir: Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    for item in iter_nvd_records(nvd_dir, limit=limit):
        cve = item.get("cve", {})
        descriptions = cve.get("descriptions", [])
        english = next((d.get("value", "") for d in descriptions if d.get("lang") == "en"), "")
        metrics = cve.get("metrics", {})
        rows.append(
            {
                "cve_id": cve.get("id"),
                "published": cve.get("published"),
                "last_modified": cve.get("lastModified"),
                "description": english,
                "has_cvss_v31": "cvssMetricV31" in metrics,
                "has_cvss_v40": "cvssMetricV40" in metrics,
            }
        )
    return pd.DataFrame(rows)
