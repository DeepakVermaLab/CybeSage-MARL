from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cybsage_marl.utils.io import iter_jsonl, write_json
from cybsage_marl.utils.tracking import write_stage_manifest


def run_preprocessing(trajectory_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(iter_jsonl(trajectory_path))
    if not rows:
        raise RuntimeError(f"No trajectory rows found in {trajectory_path}")

    observations = np.asarray([row["observation"] for row in rows], dtype=np.float32)
    next_observations = np.asarray([row["next_observation"] for row in rows], dtype=np.float32)
    meta = pd.DataFrame(
        [
            {
                "episode_id": row["episode_id"],
                "step": row["step"],
                "regime": row["regime"],
                "action": row["action"],
                "optimal_action": row["optimal_action"],
                "reward": row["reward"],
                "done": row["done"],
            }
            for row in rows
        ]
    )
    meta["split"] = "train"
    unique_episodes = sorted(meta["episode_id"].unique())
    val_cut = max(1, int(0.8 * len(unique_episodes)))
    val_episodes = set(unique_episodes[val_cut:])
    meta.loc[meta["episode_id"].isin(val_episodes), "split"] = "validation"

    np.save(out_dir / "observations.npy", observations)
    np.save(out_dir / "next_observations.npy", next_observations)
    meta.to_csv(out_dir / "trajectory_metadata.csv", index=False)
    feature_stats = pd.DataFrame(
        {
            "feature_index": list(range(observations.shape[1])),
            "mean": observations.mean(axis=0),
            "std": observations.std(axis=0),
            "min": observations.min(axis=0),
            "max": observations.max(axis=0),
        }
    )
    feature_stats.to_csv(out_dir / "feature_statistics.csv", index=False)
    split_counts = meta.groupby("split").size().rename("rows").reset_index()
    split_counts.to_csv(out_dir / "split_counts.csv", index=False)

    summary: dict[str, Any] = {
        "rows": int(len(meta)),
        "episodes": int(meta["episode_id"].nunique()),
        "features": int(observations.shape[1]),
        "train_rows": int((meta["split"] == "train").sum()),
        "validation_rows": int((meta["split"] == "validation").sum()),
        "mean_reward": float(meta["reward"].mean()),
    }
    write_json(out_dir / "preprocessing_summary.json", summary)
    write_stage_manifest(
        out_dir,
        "preprocess",
        [
            (out_dir / "observations.npy", True, "Observation matrix used by SSL and policy stages"),
            (out_dir / "next_observations.npy", True, "Next-observation matrix for transition analysis"),
            (out_dir / "trajectory_metadata.csv", True, "Trajectory row metadata, split labels, actions, rewards"),
            (out_dir / "feature_statistics.csv", True, "Preprocessing feature descriptive statistics"),
            (out_dir / "split_counts.csv", True, "Train/validation split counts"),
            (out_dir / "preprocessing_summary.json", True, "Preprocessing summary metrics"),
        ],
        metrics=summary,
    )
    return summary
