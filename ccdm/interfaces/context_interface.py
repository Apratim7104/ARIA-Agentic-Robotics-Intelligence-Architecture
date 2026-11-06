"""
ccdm/interfaces/context_interface.py

Context vector validation, clamping, and encoding for CCDM v3.0.
Standardizes multi-agent context dictionary into 5-d operational context vector:
c_t = [fragility, speed_priority, safety_priority, task_code, crowd_density]
"""

import numpy as np
from typing import Dict, Any, Union, Sequence


# Default operational context parameter ranges
TASK_CODE_MAP = {
    "reach": 0.0,
    "pick": 0.0,
    "pick_and_place": 0.0,
    "lift": 0.0,
    "can": 1.0,
    "square": 2.0,
    "peg_insertion": 2.0,
    "drawer": 2.0
}


class ContextInterface:
    """
    Manages operational context vectors for CCDM conditioning.
    """
    def __init__(self, context_dim: int = 5):
        self.context_dim = context_dim

    def encode_context(
        self,
        context_input: Union[Dict[str, Any], Sequence[float], np.ndarray]
    ) -> np.ndarray:
        """
        Encodes diverse context representations into standard 5-d vector:
            [fragility, speed_priority, safety_priority, task_code, crowd_density]
        Clamps elements to valid numerical bounds.
        """
        if isinstance(context_input, (list, tuple, np.ndarray)):
            arr = np.asarray(context_input, dtype=np.float32).flatten()
            if len(arr) >= self.context_dim:
                c_vec = arr[:self.context_dim].copy()
            else:
                c_vec = np.zeros(self.context_dim, dtype=np.float32)
                c_vec[:len(arr)] = arr
        elif isinstance(context_input, dict):
            # Parse semantic fields
            fragility = float(context_input.get("fragility", 0.5))
            speed_pri = float(context_input.get("speed_priority", 0.5))
            safety_pri = float(context_input.get("safety_priority", 0.8))

            task_raw = context_input.get("task_code", 0.0)
            if isinstance(task_raw, str):
                task_code = TASK_CODE_MAP.get(task_raw.lower(), 0.0)
            else:
                task_code = float(task_raw)

            crowd = float(context_input.get("crowd_density", context_input.get("obstacle_density", 0.0)))

            c_vec = np.array([fragility, speed_pri, safety_pri, task_code, crowd], dtype=np.float32)
        else:
            c_vec = np.array([0.5, 0.5, 0.8, 0.0, 0.0], dtype=np.float32)

        # Clamping
        c_vec[0] = np.clip(c_vec[0], 0.0, 1.0) # fragility
        c_vec[1] = np.clip(c_vec[1], 0.0, 1.0) # speed_priority
        c_vec[2] = np.clip(c_vec[2], 0.0, 1.0) # safety_priority
        c_vec[3] = np.clip(c_vec[3], 0.0, 5.0) # task_code
        c_vec[4] = np.clip(c_vec[4], 0.0, 1.0) # crowd_density

        return c_vec

