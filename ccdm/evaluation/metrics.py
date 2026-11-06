import numpy as np
import torch
from typing import Dict, Tuple

def compute_residual_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error on residual vectors."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_next_state_rmse(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error on predicted next state vectors."""
    return float(np.sqrt(np.mean((x_true - x_pred) ** 2)))

def evaluate_model_performance(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader
) -> Dict[str, float]:
    """Evaluates trained CCDM network on a dataset split."""
    model.eval()
    all_true_r = []
    all_pred_r = []

    with torch.no_grad():
        for batch in test_loader:
            pred_r, _ = model(
                batch["state"],
                batch["action"],
                batch["context"],
                batch["physical_context"]
            )
            all_true_r.append(batch["residual_target"].numpy())
            all_pred_r.append(pred_r.numpy())

    y_true = np.vstack(all_true_r)
    y_pred = np.vstack(all_pred_r)

    rmse_res = compute_residual_rmse(y_true, y_pred)
    mae_res = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "residual_rmse": round(rmse_res, 6),
        "residual_mae": round(mae_res, 6)
    }

