import torch
import torch.nn as nn

class UncertaintyHead(nn.Module):
    """
    Linear prediction head outputting 6-d heteroscedastic log-variance.
    """
    def __init__(self, in_dim: int = 128, residual_dim: int = 6):
        super().__init__()
        self.head = nn.Linear(in_dim, residual_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

