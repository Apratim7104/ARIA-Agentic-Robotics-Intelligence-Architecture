import torch
import torch.nn as nn

class StateActionEncoder(nn.Module):
    """
    State-action encoder MLP mapping (state_dim + action_dim) to latent representation (default 64-d).
    """
    def __init__(self, in_dim: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x_u: torch.Tensor) -> torch.Tensor:
        return self.net(x_u)

