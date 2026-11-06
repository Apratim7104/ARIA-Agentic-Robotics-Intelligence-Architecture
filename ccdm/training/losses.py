import torch
import torch.nn as nn

class HeteroscedasticNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss for residual mean and variance prediction.
    L(r, r_hat, s) = 0.5 * [ (r - r_hat)^2 * exp(-s) + s ]
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_log_var: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        # Clamp log variance to prevent numerical explosion/collapse
        log_var = torch.clamp(pred_log_var, min=-10.0, max=5.0)
        inv_var = torch.exp(-log_var)

        squared_error = (target - pred_mean) ** 2
        loss = 0.5 * (squared_error * inv_var + log_var)
        return torch.mean(loss)

