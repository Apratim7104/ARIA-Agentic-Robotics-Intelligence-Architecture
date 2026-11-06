"""
scripts/run_integrated_demo.py

Interactive / standalone demonstration of the integrated:
1. Multi-Agent System Pipeline (JSON input ingestion)
2. Spatial Reasoning Layer (2D scene deprojection, initial location locking & continuous relative distance tracking)
3. CCDM Neural Network (Layer 3: 3 inputs -> dynamic physical constraints & NMPC weights)

Ready for downstream integration with Layer 4 NMPC controller.
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_layer.integrated_pipeline import (
    IntegratedSpatialCCDMPipeline,
    IntegratedPipelineOutput
)
from spatial_layer.multi_agent_interface import MultiAgentInterface

SCENE_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "franka_scene.xml")


def main():
    parser = argparse.ArgumentParser(description="ARIA Integrated Autonomous Manipulation Demo")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Carefully pick up the red cube from the table and place it on the target spot",
        help="High-level natural language prompt for the Multi-Agent System"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of control steps to simulate"
    )
    args = parser.parse_args()

    print("=" * 85)
    print(" INTEGRATED AUTONOMOUS MANIPULATION PIPELINE DEMO ")
    print(" Multi-Agent Orchestrator --> Spatial Reasoning Layer --> CCDM Dynamic Constraints & NMPC Weights")
    print("=" * 85)

    # -------------------------------------------------------------
    # 1. INITIALIZE SIMULATOR & INTEGRATED PIPELINE
    # -------------------------------------------------------------
    print("\n[Step 1] Initializing MuJoCo Franka Emika Panda Simulation & Spatial-CCDM Pipeline...")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # Reset arm to home keyframe
    home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    mujoco.mj_step(model, data)

    pipeline = IntegratedSpatialCCDMPipeline(model, data)

    # -------------------------------------------------------------
    # 2. RUN MULTI-AGENT ORCHESTRATOR ON USER PROMPT
    # -------------------------------------------------------------
    print(f"\n[Step 2] Executing Multi-Agent Orchestrator on User Prompt:")
    print(f" Prompt: \"{args.prompt}\"")
    agent_result = pipeline.run_multi_agent_layer(user_prompt=args.prompt)
    task_plan = pipeline.current_task

    print(f" Route Decided       : {(agent_result.get('intent') or {}).get('route')}")
    print(f" Task ID Registered  : {task_plan.task_id}")
    print(f" Target Object Class : '{task_plan.pick_target.cls}', Mass: {task_plan.pick_target.mass} kg")
    print(f" Operational Context : {task_plan.operational_context}")

    # -------------------------------------------------------------
    # 3. RUN PIPELINE ACROSS TIMESTEPS
    # -------------------------------------------------------------
    print("\n[Step 3] Running Integrated Pipeline Across Control Steps (Simulating Arm Approach)...")
    print("-" * 115)
    header = (
        f"{'Step':<5} | {'Phase':<16} | {'EE Pos (m)':<22} | {'Target Pos (m)':<22} | "
        f"{'d_grasp (m)':<11} | {'v_max (m/s)':<11} | {'Q_pos (x,y,z)':<18}"
    )
    print(header)
    print("-" * 115)

    num_steps = args.steps
    initial_obj_pos = None

    for step_idx in range(1, num_steps + 1):
        # Step simulation physics
        mujoco.mj_step(model, data)

        # Move the simulated arm gradually towards target block to demonstrate phase transitions
        if step_idx > 5:
            target_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")]
            # Interpolate end-effector towards target
            alpha = min(1.0, (step_idx - 5) * 0.045)
            # Simulated PD joint displacement
            data.ctrl[1] = -0.3 + alpha * 0.60
            data.ctrl[3] = -2.2 + alpha * 0.50
            data.ctrl[5] = 1.9 - alpha * 0.20

        # Execute one integrated pipeline step
        output: IntegratedPipelineOutput = pipeline.step(force_depth_refresh=(step_idx == 1))

        if initial_obj_pos is None:
            initial_obj_pos = output.initial_object_location.copy()

        ee_str = f"[{output.end_effector_position[0]:.2f}, {output.end_effector_position[1]:.2f}, {output.end_effector_position[2]:.2f}]"
        tgt_str = f"[{output.current_object_location[0]:.2f}, {output.current_object_location[1]:.2f}, {output.current_object_location[2]:.2f}]"
        q_pos_str = f"[{output.nmpc_weights.Q_pos[0]:.0f}, {output.nmpc_weights.Q_pos[1]:.0f}, {output.nmpc_weights.Q_pos[2]:.0f}]"
        v_max_str = f"{output.physical_constraints.velocity_max[0]:.3f}"

        print(
            f"{output.timestep:<5} | {output.phase:<16} | {ee_str:<22} | {tgt_str:<22} | "
            f"{output.relative_distance_to_grasp:<11.4f} | {v_max_str:<11} | {q_pos_str:<18}"
        )

        time.sleep(0.02)

    print("-" * 115)
    print("\n[Step 4] Final Pipeline Diagnostics & NMPC Readiness Summary:")
    print(f" - Locked Initial Object Location : {initial_obj_pos}")
    print(f" - Current Target Object Location : {output.current_object_location}")
    print(f" - Distance from Grasp to Target  : {output.relative_distance_to_grasp:.4f} m")
    print(f" - Relative Range Rate (dd/dt)    : {output.relative_range_rate:.4f} m/s")
    print(f" - Distance from Object to Place  : {output.relative_distance_to_place:.4f} m")
    print(f" - Active Task Phase              : {output.phase}")
    print("\n Dynamic Physical Constraints for NMPC Controller:")
    print(f"   * Velocity Bounds [vx, vy, vz, yaw] : min={output.physical_constraints.velocity_min}, max={output.physical_constraints.velocity_max}")
    print(f"   * Acceleration Limits [ax, ay, az]  : min={output.physical_constraints.acceleration_min}, max={output.physical_constraints.acceleration_max}")
    print(f"   * 2-Sigma Chance Margins            : {output.physical_constraints.chance_margins} m")
    print(f"   * Table Safety Barrier Height       : {output.physical_constraints.table_barrier_z:.4f} m")
    print(f"   * Safe Obstacle Clearance           : {output.physical_constraints.safe_obstacle_clearance:.4f} m")
    print("\n Dynamic NMPC Cost Function Weights:")
    print(f"   * Q_pos (Position Tracking Weight)  : {output.nmpc_weights.Q_pos}")
    print(f"   * Q_vel (Velocity Damping Weight)   : {output.nmpc_weights.Q_vel}")
    print(f"   * R_action (Control Effort Penalty) : {output.nmpc_weights.R_action}")
    print(f"   * S_slew (Anti-Jerk Rate Penalty)   : {output.nmpc_weights.S_slew}")
    print(f"   * Q_terminal (Terminal Gain)        : {output.nmpc_weights.Q_terminal}")
    print(f"   * W_slack (Slack Violation Penalty) : {output.nmpc_weights.W_slack:.0f}")
    print(f"\n Pipeline Latency Breakdown:")
    for k, v in output.timings_ms.items():
        print(f"   * {k:<25} : {v:.3f} ms")

    print("\n" + "=" * 85)
    print(" INTEGRATED PIPELINE IS READY FOR DOWNSTREAM NMPC CONTROLLER DEPLOYMENT ")
    print("=" * 85)


if __name__ == "__main__":
    main()

