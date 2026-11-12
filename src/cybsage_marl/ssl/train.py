from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cybsage_marl.ssl.model import TrajectoryEncoder
from cybsage_marl.utils.io import iter_jsonl, write_json


def load_observations(trajectory_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    rows = list(iter_jsonl(trajectory_path))
    x = np.asarray([r["observation"] for r in rows], dtype=np.float32)
    meta = pd.DataFrame([{k: v for k, v in r.items() if k not in {"observation", "next_observation"}} for r in rows])
    return x, meta


def train_ssl_encoder(
    trajectory_path: Path,
    out_dir: Path,
    epochs: int = 5,
    batch_size: int = 128,
    embedding_dim: int = 32,
    device: str = "cpu",
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    x, meta = load_observations(trajectory_path)
    train_x, val_x = train_test_split(x, test_size=0.2, random_state=42)
    model = TrajectoryEncoder(input_dim=x.shape[1], embedding_dim=embedding_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(train_x)), batch_size=batch_size, shuffle=True)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            mask = (torch.rand_like(batch) > 0.25).float()
            corrupted = batch * mask
            _, recon = model(corrupted)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val = torch.tensor(val_x).to(device)
            _, recon = model(val)
            val_loss = float(loss_fn(recon, val).detach().cpu())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})

    torch.save(model.state_dict(), out_dir / "ssl_encoder.pt")
    model.eval()
    with torch.no_grad():
        embeddings = model.encoder(torch.tensor(x).to(device)).detach().cpu().numpy()
    np.save(out_dir / "trajectory_embeddings.npy", embeddings)
    meta.to_csv(out_dir / "trajectory_embedding_index.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "ssl_training_log.csv", index=False)
    summary = {"rows": int(x.shape[0]), "input_dim": int(x.shape[1]), "embedding_dim": embedding_dim, "epochs": epochs}
    write_json(out_dir / "ssl_summary.json", summary)
    return summary
