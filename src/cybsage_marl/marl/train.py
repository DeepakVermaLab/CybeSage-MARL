from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cybsage_marl.data.synthetic_env import ACTIONS, REGIMES
from cybsage_marl.marl.policy import PosteriorConditionedPolicy
from cybsage_marl.utils.io import iter_jsonl, write_json


def _posterior_vector(regime: str, uncertainty: float) -> list[float]:
    probs = np.full(len(REGIMES), uncertainty / max(1, len(REGIMES) - 1), dtype=np.float32)
    probs[REGIMES.index(regime)] = 1.0 - uncertainty
    probs = probs / probs.sum()
    return probs.tolist()


def build_policy_dataset(trajectory_path: Path, embedding_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = list(iter_jsonl(trajectory_path))
    embeddings = None
    if embedding_path and embedding_path.exists():
        embeddings = np.load(embedding_path)
    features = []
    labels = []
    meta = []
    for idx, row in enumerate(rows):
        obs = list(map(float, row["observation"]))
        uncertainty = float(obs[4]) if len(obs) > 4 else 0.3
        posterior = _posterior_vector(row["regime"], uncertainty)
        emb = embeddings[idx].tolist() if embeddings is not None else []
        features.append(obs + posterior + emb)
        labels.append(ACTIONS.index(row["optimal_action"]))
        meta.append({"episode_id": row["episode_id"], "step": row["step"], "regime": row["regime"], "action": row["action"], "optimal_action": row["optimal_action"]})
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), pd.DataFrame(meta)


def train_policy(
    trajectory_path: Path,
    embedding_path: Path | None,
    out_dir: Path,
    epochs: int = 5,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, meta = build_policy_dataset(trajectory_path, embedding_path)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    model = PosteriorConditionedPolicy(input_dim=x.shape[1], num_actions=len(ACTIONS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
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
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_val).to(device))
            pred = logits.argmax(dim=1).detach().cpu().numpy()
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_accuracy": float(accuracy_score(y_val, pred)),
                "val_macro_f1": float(f1_score(y_val, pred, average="macro")),
            }
        )
    torch.save(model.state_dict(), out_dir / "posterior_policy.pt")
    pd.DataFrame(history).to_csv(out_dir / "policy_training_log.csv", index=False)
    meta.to_csv(out_dir / "policy_dataset_index.csv", index=False)
    summary = {"rows": int(x.shape[0]), "input_dim": int(x.shape[1]), "num_actions": len(ACTIONS), "epochs": epochs, **history[-1]}
    write_json(out_dir / "policy_summary.json", summary)
    return summary
