import numpy as np
from typing import Tuple

class SyntheticDynamicalEnvironment:
    """
    Generates x_{t+1}^{true} and nominal x_{t+1}^{nom} to create residual targets r_t.
    """
    def __init__(self, state_dim: int = 6, action_dim: int = 4, dt: float = 0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dt = dt

    def nominal_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Nominal linear kinematic point-mass model."""
        x_next = np.zeros_like(x)
        x_next[:3] = x[:3] + x[3:] * self.dt
        x_next[3:] = x[3:] + u[:3] * self.dt
        return x_next

    def true_step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        c: np.ndarray,
        p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        True dynamics with context-dependent unmodeled drag, mass shift, and disturbances.
        """
        x_nom = self.nominal_step(x, u)

        payload_mass = float(p[0]) if len(p) > 0 else 1.0
        drag_coeff = float(c[0]) * 0.5  # Fragility/Context affects operating limits
        disturbance = float(p[2]) if len(p) > 2 else 0.0

        # Nonlinear residual forces
        acceleration_residual = -drag_coeff * (x[3:] ** 2) / max(payload_mass, 0.05) + np.array([disturbance, 0.0, 0.0])
        r_true = np.zeros_like(x)
        r_true[3:] = acceleration_residual * (self.dt ** 2)

        x_true = x_nom + r_true
        return x_true, r_true

