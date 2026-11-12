from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cybsage_marl.data.synthetic_env import ACTIONS, REGIMES, SimpleCyberDefenseEnv
from cybsage_marl.marl.policy import PosteriorConditionedPolicy
from cybsage_marl.marl.train import _posterior_vector
from cybsage_marl.utils.io import iter_jsonl, write_json


METHODS = [
    "cybsage_full",
    "no_ssl",
    "no_posterior",
    "state_only",
    "rule_based_oracle",
    "majority_action",
    "random_action",
]

METHOD_ROLES = {
    "cybsage_full": "proposed_learned_model",
    "no_ssl": "learned_ablation",
    "no_posterior": "learned_ablation",
    "state_only": "learned_ablation",
    "rule_based_oracle": "synthetic_oracle_upper_bound_not_sota",
    "majority_action": "trivial_baseline",
    "random_action": "trivial_baseline",
}


@dataclass(frozen=True)
class FeatureBundle:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame


def _load_embeddings(path: Path | None, expected_rows: int) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    embeddings = np.load(path)
    if embeddings.shape[0] != expected_rows:
        raise RuntimeError(
            f"Embedding row mismatch: {path} has {embeddings.shape[0]} rows, "
            f"but trajectory file has {expected_rows} rows."
        )
    return embeddings.astype(np.float32)


def _build_feature_bundle(
    trajectory_path: Path,
    embedding_path: Path | None,
    method: str,
) -> FeatureBundle:
    rows = list(iter_jsonl(trajectory_path))
    if not rows:
        raise RuntimeError(f"No trajectory rows found in {trajectory_path}")
    embeddings = _load_embeddings(embedding_path, len(rows))

    features: list[list[float]] = []
    labels: list[int] = []
    meta: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        obs = list(map(float, row["observation"]))
        uncertainty = float(obs[4]) if len(obs) > 4 else 0.3
        posterior = _posterior_vector(row["regime"], uncertainty)
        emb = embeddings[idx].tolist() if embeddings is not None else []

        if method == "cybsage_full":
            feat = obs + posterior + emb
        elif method == "no_ssl":
            feat = obs + posterior
        elif method == "no_posterior":
            feat = obs + emb
        elif method == "state_only":
            feat = obs
        else:
            raise ValueError(f"{method} is not a trainable feature method")

        features.append(feat)
        labels.append(ACTIONS.index(row["optimal_action"]))
        meta.append(
            {
                "episode_id": int(row["episode_id"]),
                "step": int(row["step"]),
                "regime": row["regime"],
                "action": row["action"],
                "optimal_action": row["optimal_action"],
                "reward": float(row["reward"]),
            }
        )
    return FeatureBundle(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), pd.DataFrame(meta))


def _utility_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = y_true == y_pred
    return float(np.where(correct, 1.0, -0.2).mean())


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "decision_utility": _utility_from_predictions(y_true, y_pred),
    }


def _train_neural_policy(bundle: FeatureBundle, seed: int, epochs: int, batch_size: int, device: str) -> dict[str, Any]:
    try:
        x_train, x_val, y_train, y_val = train_test_split(
            bundle.x,
            bundle.y,
            test_size=0.2,
            random_state=seed,
            stratify=bundle.y,
        )
    except ValueError:
        x_train, x_val, y_train, y_val = train_test_split(
            bundle.x,
            bundle.y,
            test_size=0.2,
            random_state=seed,
        )

    torch.manual_seed(seed)
    model = PosteriorConditionedPolicy(input_dim=bundle.x.shape[1], num_actions=len(ACTIONS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    last_loss = math.nan
    for _ in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses)) if losses else math.nan

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_val).to(device)).argmax(dim=1).detach().cpu().numpy()
    result = _macro_metrics(y_val, pred)
    result.update(
        {
            "train_loss": last_loss,
            "validation_rows": int(len(y_val)),
            "input_dim": int(bundle.x.shape[1]),
        }
    )
    return result


def _rule_based_predictions(bundle: FeatureBundle) -> np.ndarray:
    predictions = []
    for idx, row in bundle.meta.iterrows():
        env = SimpleCyberDefenseEnv(seed=0)
        # Use the same rule policy as the CPU simulator to provide an upper-bound
        # sanity baseline for the current synthetic trajectory schema.
        state = SimpleNamespace(
            regime=row["regime"],
            alerts=int(10 * float(bundle.x[idx, 2])) if bundle.x.shape[1] >= 3 else 0,
            compromised=int(10 * float(bundle.x[idx, 0])) if bundle.x.shape[1] >= 1 else 0,
            service_down=int(10 * float(bundle.x[idx, 1])) if bundle.x.shape[1] >= 2 else 0,
            decoys=int(5 * float(bundle.x[idx, 3])) if bundle.x.shape[1] >= 4 else 0,
        )
        predictions.append(ACTIONS.index(env.best_action(state)))
    return np.asarray(predictions, dtype=np.int64)


def _evaluate_non_trainable(bundle: FeatureBundle, method: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    try:
        _, x_val, _, y_val, _, meta_val = train_test_split(
            bundle.x,
            bundle.y,
            bundle.meta,
            test_size=0.2,
            random_state=seed,
            stratify=bundle.y,
        )
    except ValueError:
        _, x_val, _, y_val, _, meta_val = train_test_split(
            bundle.x,
            bundle.y,
            bundle.meta,
            test_size=0.2,
            random_state=seed,
        )
    val_bundle = FeatureBundle(x_val, y_val, meta_val.reset_index(drop=True))
    if method == "rule_based_oracle":
        pred = _rule_based_predictions(val_bundle)
    elif method == "majority_action":
        values, counts = np.unique(bundle.y, return_counts=True)
        pred = np.full_like(y_val, values[np.argmax(counts)])
    elif method == "random_action":
        pred = rng.integers(0, len(ACTIONS), size=len(y_val), dtype=np.int64)
    else:
        raise ValueError(f"Unknown non-trainable method: {method}")
    result = _macro_metrics(y_val, pred)
    result.update({"train_loss": math.nan, "validation_rows": int(len(y_val)), "input_dim": 0})
    return result


def _aggregate(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "decision_utility"]
    rows = []
    for method, group in results.groupby("method"):
        row: dict[str, Any] = {"method": method, "method_role": METHOD_ROLES.get(method, "unknown"), "runs": int(len(group))}
        for metric in metric_cols:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["macro_f1_mean", "accuracy_mean"], ascending=False).reset_index(drop=True)


def run_ablation_suite(
    trajectory_path: Path,
    embedding_path: Path | None,
    out_dir: Path,
    seeds: list[int],
    epochs: int = 5,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not seeds:
        raise ValueError("At least one seed is required for ablation.")

    rows: list[dict[str, Any]] = []
    bundles: dict[str, FeatureBundle] = {}
    for method in METHODS:
        if method in {"cybsage_full", "no_ssl", "no_posterior", "state_only"}:
            bundle = _build_feature_bundle(trajectory_path, embedding_path, method)
            bundles[method] = bundle
            for seed in seeds:
                result = _train_neural_policy(bundle, seed, epochs, batch_size, device)
                rows.append({"method": method, "seed": seed, "epochs": epochs, **result})
        else:
            base_bundle = bundles.get("state_only") or _build_feature_bundle(trajectory_path, embedding_path, "state_only")
            for seed in seeds:
                result = _evaluate_non_trainable(base_bundle, method, seed)
                rows.append({"method": method, "seed": seed, "epochs": 0, **result})

    results = pd.DataFrame(rows)
    aggregate = _aggregate(results)
    results.to_csv(out_dir / "ablation_seed_results.csv", index=False)
    aggregate.to_csv(out_dir / "ablation_summary.csv", index=False)

    best = aggregate.iloc[0].to_dict() if not aggregate.empty else {}
    summary = {
        "trajectory_rows": int(_build_feature_bundle(trajectory_path, embedding_path, "state_only").x.shape[0]),
        "methods": METHODS,
        "seeds": seeds,
        "epochs": epochs,
        "best_method": best.get("method"),
        "best_macro_f1_mean": best.get("macro_f1_mean"),
        "best_accuracy_mean": best.get("accuracy_mean"),
    }
    write_json(out_dir / "ablation_summary.json", summary)
    return summary
