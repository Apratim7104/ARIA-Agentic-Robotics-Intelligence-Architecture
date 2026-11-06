"""
ccdm/data/robomimic_adapter.py

Adapter converting Robomimic HDF5 demonstration logs (Franka Panda Arm)
into CCDM v3.0 supervised training tensors (x_t, u_t, c_t, p_t, r_t).
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional

try:
    import h5py
except ImportError:
    h5py = None


class RobomimicDatasetAdapter:
    """
    Parses Robomimic HDF5 files into CCDM residual dynamics samples.
    """

    def __init__(self, dt: float = 0.05):
        """
        Args:
            dt: Robomimic simulation time step (20 Hz controller rate = 0.05s).
        """
        self.dt = dt

    def parse_hdf5_file(
        self,
        hdf5_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts all demonstration trajectories from a Robomimic HDF5 dataset file.
        """
        if h5py is None:
            raise ImportError("h5py is required to parse Robomimic HDF5 files. Install via pip install h5py.")

        all_states, all_actions, all_contexts, all_phys, all_residuals = [], [], [], [], []

        # Operational Context Vector c_t [5-d] for Pick-and-Place / Lift task:
        # [fragility, speed_priority, safety_priority, task_code, crowd_density]
        c_t = np.array([0.7, 0.4, 0.95, 0.0, 0.0], dtype=np.float32)

        with h5py.File(hdf5_path, "r") as f:
            demos = f["data"]
            demo_keys = list(demos.keys())
            print(f" Parsing {len(demo_keys)} trajectories from Robomimic HDF5 dataset...")

            for ep_key in demo_keys:
                ep = demos[ep_key]
                obs = ep["obs"]

                # Extract End-Effector 3D Position (T, 3)
                eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
                T = len(eef_pos)
                if T < 2:
                    continue

                # Extract Linear Velocity or compute causal finite difference fallback
                if "robot0_eef_vel_lin" in obs:
                    eef_vel = np.array(obs["robot0_eef_vel_lin"], dtype=np.float32)
                else:
                    eef_vel = np.zeros_like(eef_pos)
                    eef_vel[1:] = (eef_pos[1:] - eef_pos[:-1]) / self.dt
                    eef_vel[0] = eef_vel[1]

                # Extract Actions (T, 7) -> slice first 4 dims [vx_cmd, vy_cmd, vz_cmd, yaw_cmd]
                raw_actions = np.array(ep["actions"], dtype=np.float32)
                actions = raw_actions[:, :4]

                # Extract Object Position (T, 3) if available for relative distance computation
                if "object" in obs:
                    obj_pos = np.array(obs["object"][:, :3], dtype=np.float32)
                else:
                    obj_pos = np.zeros_like(eef_pos)

                for t in range(T - 1):
                    # 1. State x_t [6-d]: [x, y, z, vx, vy, vz]
                    x_curr = np.concatenate([eef_pos[t], eef_vel[t]]).astype(np.float32)

                    # 2. Action u_t [4-d]
                    u_curr = actions[t].astype(np.float32)

                    # 3. Target relative distance d_rel = ||eef_pos - obj_pos||
                    d_rel = float(np.linalg.norm(eef_pos[t] - obj_pos[t]))

                    # 4. Physical Context Vector p_t [3-d]: [mass, friction, target_dist]
                    p_t = np.array([0.25, 0.40, d_rel], dtype=np.float32)

                    # 5. Observed Next State x_{t+1}^{true} [6-d]
                    x_true_next = np.concatenate([eef_pos[t + 1], eef_vel[t + 1]]).astype(np.float32)

                    # 6. Nominal Kinematic Prediction x_{t+1}^{nom} [6-d]
                    x_nom_next = x_curr.copy()
                    x_nom_next[:3] = x_curr[:3] + x_curr[3:6] * self.dt
                    x_nom_next[3:6] = x_curr[3:6] + u_curr[:3] * self.dt

                    # 7. Residual Target r_t = x_{t+1}^{true} - x_{t+1}^{nom}
                    r_t = x_true_next - x_nom_next

                    all_states.append(x_curr)
                    all_actions.append(u_curr)
                    all_contexts.append(c_t)
                    all_phys.append(p_t)
                    all_residuals.append(r_t)

        return (
            np.array(all_states, dtype=np.float32),
            np.array(all_actions, dtype=np.float32),
            np.array(all_contexts, dtype=np.float32),
            np.array(all_phys, dtype=np.float32),
            np.array(all_residuals, dtype=np.float32)
        )

