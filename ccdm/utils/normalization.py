import numpy as np
from typing import Dict, Any

class StandardScaler:
    """
    Calculates and applies Z-score normalization for dataset tensors:
        x_norm = (x - mean) / std
    Includes epsilon=1e-8 to prevent division by zero, and JSON export/import.
    """
    def __init__(self):
        self.mean: np.ndarray = None
        self.std: np.ndarray = None

    def fit(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return data
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return data
        return (data * self.std) + self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else []
        }

    def from_dict(self, d: Dict[str, Any]):
        self.mean = np.array(d["mean"], dtype=np.float32)
        self.std = np.array(d["std"], dtype=np.float32)

