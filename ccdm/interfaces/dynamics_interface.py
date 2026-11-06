"""
ccdm_dynamics/interfaces/dynamics_interface.py

Controller-Agnostic Interface for downstream Model Predictive Controllers (MPC).
Provides physical residual corrections (r_hat) and predictive uncertainty (sigma_hat).
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, Optional

try:
    from ccdm_dynamics.models.ccdm import CCDM
    from ccdm_dynamics.utils.normalization import StandardScaler
except ImportError:
    from ccdm.models.ccdm import CCDM
    from ccdm.utils.normalization import StandardScaler


class CCDMInferenceInterface:
    """
    High-speed inference wrapper for CCDM v3.0 model.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        model: Optional[CCDM] = None
    ):
        if config is None:
            config = {
                "system": {
                    "state_dim": 6,
                    "action_dim": 4,
                    "context_dim": 5,
                    "physical_dim": 3,
                    "residual_dim": 6
                }
            }
        self.config = config
        sys_cfg = config["system"]

        # Auto-discover weights if not explicitly provided
        if model is None and model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for candidate in [
                os.path.join(base_dir, "checkpoints", "ccdm_film.pth"),
                os.path.join("experiments", "ccdm_film.pth")
            ]:
                if os.path.exists(candidate):
                    model_path = candidate
                    break

        if scaler_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for candidate in [
                os.path.join(base_dir, "checkpoints", "scalers.json"),
                os.path.join("experiments", "scalers.json")
            ]:
                if os.path.exists(candidate):
                    scaler_path = candidate
                    break

        if model is not None:
            self.model = model
        else:
            self.model = CCDM(
                state_dim=sys_cfg["state_dim"],
                action_dim=sys_cfg["action_dim"],
                context_dim=sys_cfg["context_dim"],
                physical_dim=sys_cfg["physical_dim"],
                residual_dim=sys_cfg["residual_dim"]
            )
            if model_path is not None and os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        self.model.eval()

        self.state_scaler = StandardScaler()
        self.res_scaler = StandardScaler()
        self.temp_scale = 1.0

        if scaler_path is not None and os.path.exists(scaler_path):
            with open(scaler_path, "r") as f:
                scalers_dict = json.load(f)
            self.state_scaler.mean = np.array(scalers_dict["state_scaler"]["mean"], dtype=np.float32)
            self.state_scaler.std = np.array(scalers_dict["state_scaler"]["std"], dtype=np.float32)
            self.res_scaler.mean = np.array(scalers_dict["res_scaler"]["mean"], dtype=np.float32)
            self.res_scaler.std = np.array(scalers_dict["res_scaler"]["std"], dtype=np.float32)
            self.temp_scale = scalers_dict.get("temperature_scale", 1.0)
            if isinstance(self.temp_scale, list):
                self.temp_scale = np.array(self.temp_scale, dtype=np.float32)
        else:
            # Default identity normalization
            self.state_scaler.mean = np.zeros(sys_cfg["state_dim"], dtype=np.float32)
            self.state_scaler.std = np.ones(sys_cfg["state_dim"], dtype=np.float32)
            self.res_scaler.mean = np.zeros(sys_cfg["residual_dim"], dtype=np.float32)
            self.res_scaler.std = np.ones(sys_cfg["residual_dim"], dtype=np.float32)

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
        context: np.ndarray,
        physical_context: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Exposes prediction interface returning physical residual mean and calibrated uncertainty vector.
        """
        state_norm = self.state_scaler.transform(state)

        t_state = torch.tensor(state_norm, dtype=torch.float32)
        if t_state.ndim == 1:
            t_state = t_state.unsqueeze(0)

        t_action = torch.tensor(action, dtype=torch.float32)
        if t_action.ndim == 1:
            t_action = t_action.unsqueeze(0)

        t_context = torch.tensor(context, dtype=torch.float32)
        if t_context.ndim == 1:
            t_context = t_context.unsqueeze(0)

        t_phys = None
        if physical_context is not None:
            t_phys = torch.tensor(physical_context, dtype=torch.float32)
            if t_phys.ndim == 1:
                t_phys = t_phys.unsqueeze(0)

        with torch.no_grad():
            r_norm_mean, log_var = self.model(t_state, t_action, t_context, t_phys)
            r_norm_mean = r_norm_mean.squeeze(0).cpu().numpy()
            log_var = torch.clamp(log_var, min=-10.0, max=5.0).squeeze(0).cpu().numpy()

        # Un-normalize residual prediction to physical units
        r_hat = self.res_scaler.inverse_transform(r_norm_mean)

        # Calculate standard deviation in physical units with conformal temperature scale
        sigma_hat = np.sqrt(np.exp(log_var)) * self.res_scaler.std * self.temp_scale

        return {
            "residual": r_hat,
            "uncertainty": sigma_hat
        }
