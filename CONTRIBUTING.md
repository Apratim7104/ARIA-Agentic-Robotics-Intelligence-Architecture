# Contributing to ARIA

Thank you for your interest in contributing to the **ARIA: Autonomous Robotic Manipulation System**! We welcome community contributions to all layers of the stack.

---

## Code of Conduct

Please be respectful and constructive in all discussions, issues, and code reviews.

---

## Getting Started

1. **Fork the Repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/ARIA.git
   cd ARIA
   ```
3. **Create a development branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up the virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   pip install -e .
   ```

---

## Architectural Guidelines

The ARIA architecture follows a strict 4-layer hierarchical design:

1. **Layer 1: Multi-Agent Orchestrator (`multi_agent_orchestrator/`)**:
   - Handles natural language prompts, intent classification, YOLOv8/VLM perception, SAM2 masks, and 6-DOF grasp planning.
   - Emits a clean, serialized `robot_handoff` payload to Layer 2.

2. **Layer 2: Spatial Reasoning Layer (`spatial_layer/`)**:
   - Synchronizes with MuJoCo simulation and depth cameras.
   - Ingests the `robot_handoff` payload, locks the initial target location, and continuously tracks relative distance $d_{\text{rel}}$ and line-of-sight range rate $\dot{d}_{\text{rel}}$.
   - Assembles the 30-dimensional spatial feature vector.

3. **Layer 3: Context-Conditioned Dynamics Model (`ccdm_dynamics/`)**:
   - Conditioned on context $c_t$, spatial state, and physical scene.
   - Computes dynamic physical constraints (velocity corridors, slip limits, table safety barrier) and dynamic NMPC cost function weights.

4. **Layer 4: Franka 7-DOF NMPC Controller (`nmpc_controller.py`)**:
   - CasADi + IPOPT multiple shooting formulation at 100 Hz.
   - Generates joint torque commands $\tau_{\text{apply}}$ for execution on Franka Emika Panda in MuJoCo physics.

> **Rule:** Never bypass layer boundaries. Any communication between Layer 1 and Layer 4 must flow through the designated interfaces in Layer 2 and Layer 3.

---

## Testing & Quality Assurance

Before submitting a Pull Request, ensure that all unit and integration tests pass:

```bash
pytest tests/
```

To run end-to-end deployment verification:
```bash
python verify_deployment.py --prompt "Pick up the red cube and place it on the target spot"
```

---

## Submitting Pull Requests

1. Commit your changes with clear, semantic commit messages (e.g. `feat: add tactile slip feedback to CCDM`, `fix: resolve camera pitch deprojection bug`).
2. Push your branch to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```
3. Open a Pull Request against the `main` branch of the official repository.
4. Ensure all CI checks pass.
