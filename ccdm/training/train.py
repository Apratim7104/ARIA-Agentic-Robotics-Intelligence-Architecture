"""
ccdm/training/train.py

Base training pipeline for CCDM v3.0 on synthetic and benchmark dynamical datasets.
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from ccdm.models.ccdm import CCDM
from ccdm.data.dataset import CCDMDataset, generate_synthetic_dataset
from ccdm.utils.normalization import StandardScaler
from ccdm.training.losses import HeteroscedasticNLLLoss


def train_synthetic(
    output_dir: str = "experiments",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    num_samples: int = 5000
):
    os.makedirs(output_dir, exist_ok=True)
    sys_cfg = {
        "state_dim": 6,
        "action_dim": 4,
        "context_dim": 5,
        "physical_dim": 3,
        "residual_dim": 6
    }

    print("[1/4] Generating Synthetic Trajectory Dataset...")
    states, actions, contexts, phys_ctx, residuals = generate_synthetic_dataset(
        num_samples=num_samples,
        state_dim=sys_cfg["state_dim"],
        action_dim=sys_cfg["action_dim"],
        context_dim=sys_cfg["context_dim"],
        physical_dim=sys_cfg["physical_dim"]
    )

    state_scaler = StandardScaler()
    state_scaler.fit(states)
    states_norm = state_scaler.transform(states)

    res_scaler = StandardScaler()
    res_scaler.fit(residuals)
    residuals_norm = res_scaler.transform(residuals)

    scalers_dict = {
        "state_scaler": state_scaler.to_dict(),
        "res_scaler": res_scaler.to_dict(),
        "temperature_scale": 1.0
    }
    with open(os.path.join(output_dir, "scalers.json"), "w") as f:
        json.dump(scalers_dict, f, indent=2)

    dataset = CCDMDataset(states_norm, actions, contexts, phys_ctx, residuals_norm)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print("[2/4] Initializing Uncertainty-Aware CCDM Network...")
    model = CCDM(
        state_dim=sys_cfg["state_dim"],
        action_dim=sys_cfg["action_dim"],
        context_dim=sys_cfg["context_dim"],
        physical_dim=sys_cfg["physical_dim"],
        residual_dim=sys_cfg["residual_dim"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = HeteroscedasticNLLLoss()

    print(f"[3/4] Training CCDM Model (NLL Loss) for {epochs} Epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pred_mean, pred_log_var = model(
                batch["state"], batch["action"], batch["context"], batch["physical_context"]
            )
            loss = criterion(pred_mean, pred_log_var, batch["residual_target"])
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch["state"].size(0)

        train_loss /= train_size

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    pred_mean, pred_log_var = model(
                        batch["state"], batch["action"], batch["context"], batch["physical_context"]
                    )
                    loss = criterion(pred_mean, pred_log_var, batch["residual_target"])
                    val_loss += loss.item() * batch["state"].size(0)
            val_loss /= val_size
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train NLL: {train_loss:.6f} | Val NLL: {val_loss:.6f}")

    weights_path = os.path.join(output_dir, "ccdm_film.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"[4/4] Model successfully trained and saved to {weights_path}")
    return model, scalers_dict


if __name__ == "__main__":
    train_synthetic(epochs=10)

