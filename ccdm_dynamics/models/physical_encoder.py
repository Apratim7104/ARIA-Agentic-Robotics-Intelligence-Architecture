import torch
import torch.nn as nn
from typing import Optional

class PhysicalEncoder(nn.Module):
    """
    Physical context encoder MLP mapping 3-d physical parameters to 32-d embedding.
    """
    def __init__(self, in_dim: int, out_dim: int = 32):
        super().__init__()
        if in_dim == 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim)
            )

    def forward(self, p: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.net is None or p is None:
            return None
        return self.net(p)

