# Context-Conditioned Dynamics Model (CCDM v3.0)

Dedicated **Layer 3** module of the 4-layer autonomous robotic manipulation stack:

$$\text{Layer 1: Multi-Agent VLM/LLM} \longrightarrow \text{Layer 2: Spatial Reasoning} \longrightarrow \mathbf{\text{Layer 3: CCDM Dynamics NN}} \longrightarrow \text{Layer 4: NMPC Controller}$$

---

## 1. Overview & Physics Formulation

The **Context-Conditioned Dynamics Model (CCDM v3.0)** bridges high-level task semantics and geometric spatial perception with low-level predictive control. It models unmodeled contact physics, payload shifts, surface friction, and aerodynamic/inertial disturbances via residual dynamics learning:

$$x_{t+1}^{\text{true}} = f_{\text{nom}}(x_t, u_t) + r_t(x_t, u_t, c_t, p_t)$$

where:
* $f_{\text{nom}}$ is the nominal kinematic transition.
* $r_t \in \mathbb{R}^6$ is the residual transition error modeled by the neural network.
* $c_t \in [0, 1]^5$ is the operational context vector (`[fragility, speed_priority, safety_priority, task_code, crowd_density]`).
* $p_t \in \mathbb{R}^3$ is the physical parameter context (`[mass, friction, relative_distance]`).
* $\sigma_t \in \mathbb{R}^6$ is the calibrated heteroscedastic standard deviation vector.

---

## 2. 3 Inputs $\longrightarrow$ 2 Outputs Conditioning Architecture

```
                   ┌────────────────────────────────────────────────────────┐
                   │  Input 1: Context (Multi-Agent Pipeline)               │
                   │  c_t = [fragility, speed_pri, safety_pri, task, crowd] │
                   └──────────────────────────┬─────────────────────────────┘
                                              │
┌──────────────────────────────────────────┐  │  ┌─────────────────────────────────────────┐
│ Input 2: States (Spatial Reasoning Layer)│  │  │ Input 3: Physical Scene (Simulator Cam) │
│ • p_ee, v_ee, p_obj, p_obj_0             ├──┼──┤ • Depth map (metric Z-buffer)           │
│ • d_rel = ||p_ee - p_obj||, Δ_rel        │  │  │ • Table safety barrier height           │
│ • Range rate dd_rel/dt                   │  │  │ • Payload mass (kg), friction (μ)       │
└──────────────────────────────────────────┘  │  └─────────────────────────────────────────┘
                                              ▼
                                 ╔════════════════════════════╗
                                 ║     CCDM v3.0 Network      ║
                                 ║  • State-Action Enc (64-d) ║
                                 ║  • Context Enc (32-d)      ║
                                 ║  • Physical Enc (32-d)     ║
                                 ║  • FiLM Affine Modulation  ║
                                 ║  • Trunk MLP (128-d)       ║
                                 ║  • Dual Prediction Heads   ║
                                 ╚════════════════════════════╝
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
       ┌─────────────────────────────┐                 ┌─────────────────────────────┐
       │ Output 1: Control Layer     │                 │ Output 2: NMPC Controller   │
       │ Physical Constraints        │                 │ Dynamic Cost Weights        │
       │ • v_max(d_rel, fragility)   │                 │ • Q_pos (terminal 5x boost) │
       │ • a_max <= μ*g (anti-slip)  │                 │ • Q_vel (overshoot damping) │
       │ • 2*σ chance margin (95.4%) │                 │ • S_slew (anti-jerk penalty)│
       │ • Table barrier non-penetrat│                 │ • R_action, Q_term, W_slack │
       └─────────────────────────────┘                 └─────────────────────────────┘
```

---

## 3. Directory Layout

```text
ccdm_dynamics/
├── config/
│   ├── model_config.yaml           # System dimensions [6, 4, 5, 3, 6], batch size, learning rate
│   └── dataset_config.yaml         # Synthetic environment bounds and noise parameters
├── models/
│   ├── ccdm.py                     # Primary FiLM-modulated CCDM model with dual output heads
│   ├── state_encoder.py            # 10-d -> 64-d state-action MLP
│   ├── context_encoder.py          # 5-d -> 32-d operational context MLP
│   ├── physical_encoder.py         # 3-d -> 32-d physical parameter MLP
│   ├── fusion_network.py           # FiLM affine modulation layer and 128-d trunk
│   ├── residual_head.py            # 128-d -> 6-d residual mean head
│   └── uncertainty_head.py         # 128-d -> 6-d heteroscedastic log-variance head
├── interfaces/
│   ├── controller_adaptation.py    # Bridge computing dynamic physical constraints & NMPC weights
│   ├── dynamics_interface.py       # Production CCDMInferenceInterface for real-time solvers
│   ├── spatial_grounding.py        # System-agnostic spatial scene grounding & relative distance
│   └── context_interface.py        # Operational context validation and bound clamping
├── training/
│   ├── losses.py                   # HeteroscedasticNLLLoss with numerical clamping [-10, 5]
│   └── train.py                    # Base training loop and conformal calibration
├── data/
│   ├── simulator.py                # Analytic kinematic point-mass dynamical simulator
│   ├── dataset.py                  # Synthetic trajectory generator & PyTorch Dataset
│   └── robomimic_adapter.py        # Franka Panda HDF5 manipulation demonstration parser
├── utils/
│   ├── normalization.py            # StandardScaler with z-score transform & JSON persistence
│   ├── reproducibility.py          # Deterministic random seed management
│   └── logging.py                  # Structured console and file logger
├── evaluation/
│   └── metrics.py                  # Mathematical implementations of RMSE, MAE, and RVR
├── checkpoints/
│   ├── ccdm_film.pth               # Pretrained weights checkpoint
│   └── scalers.json                # Fitted Z-score scalers & conformal temperature scale
└── README.md
```

---

## 4. Connection with Spatial Reasoning Layer

The Spatial Reasoning Layer (`spatial_layer`) connects directly to `ccdm_dynamics` through the master orchestrator `IntegratedSpatialCCDMPipeline` in [`spatial_layer/integrated_pipeline.py`](../spatial_layer/integrated_pipeline.py):

```python
from spatial_layer.integrated_pipeline import IntegratedSpatialCCDMPipeline

# Instantiate integrated pipeline with live MuJoCo model and data
pipeline = IntegratedSpatialCCDMPipeline(model, data)

# 1. Ingest Multi-Agent JSON message
task_plan = pipeline.set_task_from_multi_agent(multi_agent_json)

# 2. Step pipeline at every timestep
output = pipeline.step()

# Access continuous relative distance & locked initial pose
print("Initial object pose:", output.initial_object_location)
print("Continuous distance to grasp:", output.relative_distance_to_grasp)

# Access dynamic constraints and cost function weights ready for NMPC
print("NMPC Velocity limit:", output.physical_constraints.velocity_max)
print("NMPC Q_pos weights:", output.nmpc_weights.Q_pos)
```

