import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict

from .simulator import SyntheticDynamicalEnvironment


class CCDMDataset(Dataset):
    """PyTorch Dataset wrapper for Context-Conditioned Dynamics samples."""
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        contexts: np.ndarray,
        physical_contexts: np.ndarray,
        residual_targets: np.ndarray
    ):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        self.contexts = torch.tensor(contexts, dtype=torch.float32)
        self.physical_contexts = torch.tensor(physical_contexts, dtype=torch.float32)
        self.residual_targets = torch.tensor(residual_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "context": self.contexts[idx],
            "physical_context": self.physical_contexts[idx],
            "residual_target": self.residual_targets[idx]
        }


def generate_synthetic_dataset(
    num_samples: int = 10000,
    state_dim: int = 6,
    action_dim: int = 4,
    context_dim: int = 5,
    physical_dim: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic trajectories with randomized physical properties and context."""
    env = SyntheticDynamicalEnvironment(state_dim=state_dim, action_dim=action_dim)

    states = np.random.uniform(-2.0, 2.0, size=(num_samples, state_dim)).astype(np.float32)
    actions = np.random.uniform(-1.0, 1.0, size=(num_samples, action_dim)).astype(np.float32)
    contexts = np.random.uniform(0.0, 1.0, size=(num_samples, context_dim)).astype(np.float32)
    physical_contexts = np.random.uniform(0.1, 2.0, size=(num_samples, physical_dim)).astype(np.float32)

    residual_targets = np.zeros((num_samples, state_dim), dtype=np.float32)

    for i in range(num_samples):
        _, r_true = env.true_step(states[i], actions[i], contexts[i], physical_contexts[i])
        residual_targets[i] = r_true

    return states, actions, contexts, physical_contexts, residual_targets

