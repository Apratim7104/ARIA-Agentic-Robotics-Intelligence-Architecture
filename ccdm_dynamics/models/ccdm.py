import torch
import torch.nn as nn
from typing import Optional, Tuple

from .state_encoder import StateActionEncoder
from .context_encoder import ContextEncoder
from .physical_encoder import PhysicalEncoder
from .fusion_network import FiLMLayer, FusionNetwork
from .residual_head import ResidualHead
from .uncertainty_head import UncertaintyHead


class CCDM(nn.Module):
    """
    Context-Conditioned Dynamics Model (Version 3.0 - FiLM Modulated)
    Applies affine context conditioning (gamma, beta) to state-action representations.
    Outputs:
        r_mean: residual transition mean (residual_dim)
        log_var: heteroscedastic log-variance s = log(sigma^2) (residual_dim)
    """
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 4,
        context_dim: int = 5,
        physical_dim: int = 3,
        residual_dim: int = 6
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        self.physical_dim = physical_dim
        self.residual_dim = residual_dim

        self.state_action_enc = StateActionEncoder(state_dim + action_dim, 64)
        self.context_enc = ContextEncoder(context_dim, 32)
        self.physical_enc = PhysicalEncoder(physical_dim, 32)

        phys_out_dim = 32 if physical_dim > 0 else 0
        base_feature_dim = 64 + phys_out_dim

        # FiLM Generator
        self.film_layer = FiLMLayer(feature_dim=base_feature_dim, context_emb_dim=32)

        self.fusion_net = nn.Sequential(
            nn.Linear(base_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Dual Output Heads
        self.residual_head = nn.Linear(128, residual_dim)
        self.uncertainty_head = nn.Linear(128, residual_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        physical_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for CCDM v3.0:
            e_xu = state_action_enc([x, u])   (64-d)
            e_c  = context_enc(c)              (32-d)
            e_p  = physical_enc(p)             (32-d)
            h_phys = [e_xu, e_p]              (96-d)
            h_film = FiLM(h_phys, e_c)        (96-d)
            h = fusion_net(h_film)            (128-d)
            r_mean = residual_head(h)         (6-d)
            log_var = uncertainty_head(h)     (6-d)
        """
        e_xu = self.state_action_enc(torch.cat([state, action], dim=-1))
        e_c = self.context_enc(context)

        features = [e_xu]
        if physical_context is not None and self.physical_enc.net is not None:
            e_p = self.physical_enc(physical_context)
            if e_p is not None:
                features.append(e_p)

        h_phys = torch.cat(features, dim=-1)

        # Apply FiLM Affine Modulation
        h_film = self.film_layer(h_phys, e_c)

        h = self.fusion_net(h_film)

        r_mean = self.residual_head(h)
        log_var = self.uncertainty_head(h)

        return r_mean, log_var

