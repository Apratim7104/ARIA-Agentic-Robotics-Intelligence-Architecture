import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    """
    Context encoder MLP mapping 5-d operational vector to 32-d embedding.
    """
    def __init__(self, in_dim: int, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)

