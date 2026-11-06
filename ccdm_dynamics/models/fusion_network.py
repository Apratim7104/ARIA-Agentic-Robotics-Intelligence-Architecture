import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    Applies context-conditioned affine modulation:
        h_out = gamma(c_emb) * h_in + beta(c_emb)
    """
    def __init__(self, feature_dim: int, context_emb_dim: int):
        super().__init__()
        self.gamma_gen = nn.Linear(context_emb_dim, feature_dim)
        self.beta_gen = nn.Linear(context_emb_dim, feature_dim)

    def forward(self, h: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_gen(c_emb)
        beta = self.beta_gen(c_emb)
        return gamma * h + beta

class FusionNetwork(nn.Module):
    """
    Trunk MLP fusing modulated physical and context representations.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)

