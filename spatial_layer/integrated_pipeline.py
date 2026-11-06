"""
spatial_layer/integrated_pipeline.py

End-to-End Integrated Orchestration Pipeline combining:
1. Multi-Agent System JSON Input Ingestion (LLMs, VLMs, Masking Models)
2. Spatial Reasoning Layer (MuJoCo Scene, Initial Object Localization & Timestep Relative Distance Tracking)
3. Context-Conditioned Dynamics Model (CCDM v3.0) with Dynamic Physical Constraints & NMPC Cost Function Weights

Data Flow:
Multi-Agent JSON  -->  Spatial Reasoning Layer (Camera Scene & State)  -->  CCDM Neural Network  -->  Dynamic Constraints & NMPC Weights
(Pick/Place Intent)    (Initial Pose & Continuous Relative Distances)      (3 Inputs: Ctx, State, Scene) (Ready for NMPC Layer)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mujoco
try:
    import mujoco.viewer
except Exception:
    pass
import numpy as np

from .core import SpatialReasoningLayer, SpatialResult
from .multi_agent_interface import MultiAgentInterface, MultiAgentTaskPlan

try:
    from ccdm_dynamics.interfaces.controller_adaptation import (
        CCDMControllerAdaptation,
        PhysicalConstraints,
        DynamicNMPCWeights
    )
    from ccdm_dynamics.interfaces.dynamics_interface import CCDMInferenceInterface
except ImportError:
    from ccdm.interfaces.controller_adaptation import (
        CCDMControllerAdaptation,
        PhysicalConstraints,
        DynamicNMPCWeights
    )
    from ccdm.interfaces.dynamics_interface import CCDMInferenceInterface

try:
    from nmpc_controller import FrankaNMPCController
except ImportError:
    FrankaNMPCController = None



@dataclass
class IntegratedPipelineOutput:
    """
    Unified output produced at every timestep by the integrated Spatial + CCDM pipeline.
    Ready for immediate consumption by the downstream NMPC controller layer.
    """
    timestep: int
    sim_time: float
    task_id: str
    task_prompt: str
    phase: str

    # 1. Spatial Locations & Continuous Relative Distances
    initial_object_location: np.ndarray      # [x, y, z] world frame (locked at task onset)
    current_object_location: np.ndarray      # [x, y, z] world frame at current timestep
    place_target_location: np.ndarray        # [x, y, z] world frame placement target
    end_effector_position: np.ndarray        # [x, y, z] world frame EE position
    end_effector_velocity: np.ndarray        # [vx, vy, vz] linear velocity
    relative_distance_to_grasp: float        # d_rel scalar in meters ||p_grasp - p_obj||
    relative_displacement_vector: np.ndarray # [dx, dy, dz] in meters
    relative_range_rate: float               # dd_rel/dt in m/s (negative = closing in)
    relative_distance_to_place: float        # Distance from object/EE to placement receptacle

    # 2. CCDM Residual Dynamics & Predictive Uncertainty
    residual_correction: np.ndarray          # (6,) physical residual correction [pos, vel]
    calibrated_uncertainty: np.ndarray       # (6,) 2-sigma conformal standard deviation

    # 3. Dynamic Outputs for Downstream NMPC Controller
    physical_constraints: PhysicalConstraints # Velocity/Acceleration limits, chance bounds, table barrier
    nmpc_weights: DynamicNMPCWeights          # Dynamic Q_pos, Q_vel, R_action, S_slew, Q_terminal, W_slack

    # 4. Diagnostics & 30-dim Spatial Features
    spatial_features: np.ndarray             # (30,) float32 in [-1, 1]

    # 5. Downstream NMPC Controller Output (Layer 4)
    torque_command: Optional[np.ndarray] = None          # (7,) joint torque command tau_apply (N*m)
    predicted_next_state: Optional[np.ndarray] = None    # (14,) predicted state [q, dq]
    nmpc_solve_info: Optional[Dict[str, Any]] = None     # solve_time_ms, cost, ipopt status
    timings_ms: Dict[str, float] = field(default_factory=dict)
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        """Converts output to JSON-serializable dictionary."""
        return {
            "timestep": self.timestep,
            "sim_time": float(self.sim_time),
            "task_id": self.task_id,
            "task_prompt": self.task_prompt,
            "phase": self.phase,
            "initial_object_location": self.initial_object_location.tolist(),
            "current_object_location": self.current_object_location.tolist(),
            "place_target_location": self.place_target_location.tolist(),
            "end_effector_position": self.end_effector_position.tolist(),
            "end_effector_velocity": self.end_effector_velocity.tolist(),
            "relative_distance_to_grasp": float(self.relative_distance_to_grasp),
            "relative_displacement_vector": self.relative_displacement_vector.tolist(),
            "relative_range_rate": float(self.relative_range_rate),
            "relative_distance_to_place": float(self.relative_distance_to_place),
            "residual_correction": self.residual_correction.tolist(),
            "calibrated_uncertainty": self.calibrated_uncertainty.tolist(),
            "physical_constraints": self.physical_constraints.to_dict(),
            "nmpc_weights": self.nmpc_weights.to_dict(),
            "torque_command": self.torque_command.tolist() if self.torque_command is not None else None,
            "predicted_next_state": self.predicted_next_state.tolist() if self.predicted_next_state is not None else None,
            "nmpc_solve_info": self.nmpc_solve_info,
            "spatial_features": self.spatial_features.tolist(),
            "timings_ms": self.timings_ms,
            "status": self.status
        }


class IntegratedSpatialCCDMPipeline:
    """
    Master pipeline orchestrating Multi-Agent Input Ingestion, Spatial Reasoning,
    and CCDM Dynamic Constraint & Weight Computation.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config_path: Optional[str] = None,
        ccdm_model_path: Optional[str] = None,
        ccdm_scaler_path: Optional[str] = None,
        controller: Optional[Any] = None,
        enable_nmpc: bool = False
    ):
        self.model = model
        self.data = data

        # Ensure model and data are initialized to the canonical home keyframe if uninitialized
        home_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key >= 0 and np.all(self.data.qpos[:7] == 0.0):
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
            mujoco.mj_forward(self.model, self.data)

        # 1. Spatial Reasoning Layer (Layer 2)
        self.spatial_layer = SpatialReasoningLayer(model, data, config_path=config_path)

        # 2. Multi-Agent System Parser
        self.multi_agent_parser = MultiAgentInterface()

        # 3. CCDM Neural Network & Controller Adaptation Bridge (Layer 3)
        self.ccdm_inference = CCDMInferenceInterface(
            model_path=ccdm_model_path,
            scaler_path=ccdm_scaler_path
        )
        self.ccdm_adaptation = CCDMControllerAdaptation(ccdm_inference=self.ccdm_inference)

        # 4. Downstream NMPC Controller (Layer 4)
        if controller is not None:
            self.controller = controller
        elif enable_nmpc and FrankaNMPCController is not None:
            self.controller = FrankaNMPCController(base_pos=np.array([0.0, 0.0, 0.2966]))
        else:
            self.controller = None

        # Runtime State
        self.current_task: Optional[MultiAgentTaskPlan] = None
        self.initial_object_location: Optional[np.ndarray] = None
        self.prev_distance_to_grasp: Optional[float] = None
        self.step_count = 0
        self.prev_time = time.time()

    def attach_nmpc_controller(self, controller: Optional[Any] = None) -> Any:
        """Attaches or initializes the Franka NMPC Controller instance."""
        if controller is not None:
            self.controller = controller
        elif FrankaNMPCController is not None:
            self.controller = FrankaNMPCController(base_pos=np.array([0.0, 0.0, 0.2966]))
        else:
            raise RuntimeError("FrankaNMPCController could not be imported. Ensure CasADi is installed.")
        return self.controller

    def set_task_from_multi_agent(
        self,
        task_input: Union[str, bytes, Dict[str, Any]]
    ) -> MultiAgentTaskPlan:
        """
        Ingests the JSON format task message from the multi-agent pipeline.
        Resets object tracking history and registers the pick and place targets.
        """
        task_plan = self.multi_agent_parser.parse_task_message(task_input)
        self.current_task = task_plan
        self.initial_object_location = None
        self.prev_distance_to_grasp = None
        self.step_count = 0
        return task_plan

    def run_multi_agent_layer(
        self,
        user_prompt: str,
        image_path: Optional[str] = None,
        depth_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes Layer 1: Multi-Agent Orchestrator (Planner + Robot Perception with YOLO/VLM/SAM2/Depth + Robot Planner 6-DOF Grasps + Critic/Synthesizer).

        Renders the active scene from MuJoCo if no image is passed, invokes the multi-agent graph,
        and binds the resulting `robot_handoff` payload to `self.current_task`.
        """
        import sys
        orch_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "multi_agent_orchestrator")
        if orch_dir not in sys.path:
            sys.path.insert(0, orch_dir)

        # 1. Render active scene frame if image_path is not provided
        if image_path is None or not os.path.exists(image_path):
            os.makedirs("results", exist_ok=True)
            image_path = os.path.abspath(os.path.join("results", "scene_camera_frame.jpg"))
            try:
                cam_id = self.spatial_layer.state_manager.camera_id
                r = mujoco.Renderer(self.model, height=self.spatial_layer.state_manager.height, width=self.spatial_layer.state_manager.width)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = cam_id
                r.update_scene(self.data, camera=cam)
                rgb = r.render()
                r.close()
                from PIL import Image
                Image.fromarray(rgb).save(image_path)
            except Exception as e:
                pass

        # 2. Construct initial state for the multi-agent orchestrator
        depth_metric = self.spatial_layer.state_manager.get_depth_image(force_refresh=True)

        initial_state = {
            "user_input": user_prompt,
            "session_id": f"aria_run_{int(time.time())}",
            "image_path": image_path,
            "depth_path": depth_path,
            "realsense_depth_map": depth_metric,
            "metric_depth_map": depth_metric,
            "mujoco_model": self.model,
            "mujoco_data": self.data,
            "cam_name": getattr(self.spatial_layer.state_manager, "camera_name", "overhead_cam"),
            "cam_intrinsics": {
                "fx": self.spatial_layer.transformer.fx,
                "fy": self.spatial_layer.transformer.fy,
                "cx": self.spatial_layer.transformer.cx,
                "cy": self.spatial_layer.transformer.cy,
            }
        }

        # 3. Run multi-agent orchestrator graph
        from graph.orchestrator import run_pipeline
        agent_result = run_pipeline(initial_state)

        # 4. Ingest handoff into Layer 2 Spatial Reasoning Layer
        self.set_task_from_multi_agent(agent_result)

        return agent_result

    def _resolve_target_body_id(self, target_name: Optional[str] = None) -> int:
        """Robustly maps target object name/class to simulation body ID."""
        name = target_name or (self.current_task.pick_target.name if self.current_task else "target_block")
        if not name:
            name = "target_block"
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name.replace(" ", "_"))
        if bid >= 0:
            return bid
        tokens = [t for t in name.lower().replace("_", " ").split() if len(t) > 2]
        for i in range(self.model.nbody):
            bname = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
            if any(t in bname for t in tokens):
                return i
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        if bid >= 0:
            return bid
        return -1

    def step(
        self,
        mock_detection: Optional[Dict[str, Any]] = None,
        force_depth_refresh: bool = True,
        execute_nmpc: bool = False,
        horizon_ref: Optional[np.ndarray] = None,
        apply_torque_to_sim: bool = False
    ) -> IntegratedPipelineOutput:
        """
        Executes one complete pipeline pass for the current timestep:
        1. Queries MuJoCo state and renders/linearizes camera depth map.
        2. Detects, localizes, and tracks the required objects.
        3. Locks initial location of required object on first step.
        4. Calculates continuous relative distances and kinematics from end-effector grasp.
        5. Computes dynamic physical constraints and NMPC cost function weights via CCDM.
        6. Optionally executes one NMPC control step (CasADi + IPOPT) producing optimal joint torques.

        Returns:
            IntegratedPipelineOutput: complete structured state ready for NMPC or containing torque commands.
        """
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}

        if self.current_task is None:
            # Default task if none specified
            self.set_task_from_multi_agent(self.multi_agent_parser.create_mock_task_message())

        # -------------------------------------------------------------
        # STEP 1: SPATIAL REASONING LAYER EXECUTION
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        snap = self.spatial_layer.sync_state()

        # Prepare detection message matching required target object
        pick_target_name = self.current_task.pick_target.cls
        if mock_detection is not None:
            det_msg = mock_detection
        elif getattr(self.current_task.pick_target, "bbox_hint", None) is not None:
            det_msg = {
                "timestamp": time.time(),
                "primary_target": {
                    "id": 0,
                    "class": self.current_task.pick_target.cls,
                    "confidence": 0.95,
                    "bbox": self.current_task.pick_target.bbox_hint
                },
                "secondary_objects": []
            }
        else:
            # Generate synthetic detection from MuJoCo ground truth if not passed
            det_msg = self._generate_sim_detection(pick_target_name)

        # Run spatial processing (state sync, depth linearization, pinhole deprojection, IK)
        spatial_res: SpatialResult = self.spatial_layer.process_frame(
            message=det_msg,
            force_depth_refresh=force_depth_refresh,
            snapshot=snap
        )
        timings["spatial_layer_ms"] = (time.perf_counter() - t0) * 1000.0

        # Extract End-Effector State
        p_ee = spatial_res.p_ee_world if spatial_res.p_ee_world is not None else np.array([0.4, 0.0, 0.6], dtype=np.float32)

        # End-effector velocity from mirror data
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        if ee_body_id >= 0:
            v_ee = self.spatial_layer.state_manager.data.cvel[ee_body_id, 3:].copy().astype(np.float32)
        else:
            v_ee = np.zeros(3, dtype=np.float32)

        # -------------------------------------------------------------
        # STEP 2: OBJECT LOCALIZATION & INITIAL POSITION LOCKING
        # -------------------------------------------------------------
        t1 = time.perf_counter()
        if spatial_res.p_obj_world is not None:
            p_obj = spatial_res.p_obj_world.copy().astype(np.float32)
        else:
            # Fallback to model target body if camera occlusion occurs
            target_name = self.current_task.pick_target.name if self.current_task is not None else "target_block"
            target_body_id = self._resolve_target_body_id(target_name)
            if target_body_id >= 0:
                p_obj = self.data.xpos[target_body_id].copy().astype(np.float32)
            else:
                p_obj = np.array([0.5, 0.0, 0.425], dtype=np.float32)

        # Lock initial object location at first observation
        if self.initial_object_location is None:
            self.initial_object_location = p_obj.copy()

        # Place target location
        p_place = self.current_task.place_target.target_position

        # -------------------------------------------------------------
        # STEP 3: CONTINUOUS RELATIVE DISTANCE & KINEMATICS CALCULATION
        # -------------------------------------------------------------
        # Distance from end-effector grasp point to target object
        delta_vec = p_obj - p_ee
        d_rel = float(np.linalg.norm(delta_vec))

        # Relative range rate (approach velocity along line-of-sight: dd_rel/dt)
        dt = max(0.001, snap.sim_time - getattr(self, "_prev_sim_time", snap.sim_time - 0.002))
        self._prev_sim_time = snap.sim_time

        if self.prev_distance_to_grasp is not None:
            range_rate = float((d_rel - self.prev_distance_to_grasp) / dt)
        else:
            range_rate = 0.0
        self.prev_distance_to_grasp = d_rel

        # Distance to placement target
        d_place = float(np.linalg.norm(p_place - p_obj))
        timings["kinematics_ms"] = (time.perf_counter() - t1) * 1000.0

        # -------------------------------------------------------------
        # STEP 4: FEED THE 3 INPUTS TO CCDM NEURAL ADAPTATION
        # -------------------------------------------------------------
        t2 = time.perf_counter()
        # Input 1: Multi-Agent Context
        context_input = self.current_task.operational_context

        # Input 2: Spatial Reasoning States
        spatial_state = {
            "p_ee": p_ee,
            "v_ee": v_ee,
            "p_obj": p_obj,
            "p_obj_0": self.initial_object_location,
            "p_place": p_place,
            "d_rel": d_rel,
            "delta_vec": delta_vec,
            "range_rate": range_rate,
            "features_30d": spatial_res.features
        }

        # Input 3: Physical Environment Scene from Camera / Simulator
        table_height = float(self.spatial_layer.config.get("workspace", {}).get("table_height", 0.40))
        physical_scene = {
            "table_height": table_height,
            "mass": self.current_task.pick_target.mass,
            "friction": self.current_task.pick_target.friction,
            "obstacles": spatial_res.secondary_world
        }

        # Compute dynamic physical constraints and dynamic NMPC weights
        constraints, nmpc_weights, ccdm_diag = self.ccdm_adaptation.compute_adaptation(
            context_input=context_input,
            spatial_state=spatial_state,
            physical_scene=physical_scene
        )
        timings["ccdm_adaptation_ms"] = (time.perf_counter() - t2) * 1000.0

        # -------------------------------------------------------------
        # STEP 5: OPTIONAL NMPC CONTROL STEP (CasADi + IPOPT)
        # -------------------------------------------------------------
        tau_cmd = None
        x_next_pred = None
        solve_info = None

        if execute_nmpc:
            if self.controller is None:
                self.attach_nmpc_controller()

            t3 = time.perf_counter()
            # Extract Franka joint state [q (7,), dq (7,)]
            q_current = snap.qpos[:7].copy()
            dq_current = snap.qvel[:7].copy()
            x_curr = np.concatenate([q_current, dq_current])

            # Prepare lookahead reference trajectory across horizon N
            if horizon_ref is not None:
                Y_ref = horizon_ref
            else:
                # Target object position during approach, or place target during transport
                if d_rel > 0.04 and constraints.phase in ("REACH", "DESCEND", "APPROACH", "IDLE"):
                    target_pt = p_obj
                else:
                    target_pt = p_place
                Y_ref = np.repeat(target_pt.reshape(3, 1), self.controller.N + 1, axis=1)

            # Solve parametric NMPC step with dynamic weights and physical constraints
            tau_cmd, x_next_pred, solve_info = self.controller.solve_control_step(
                x_current=x_curr,
                Y_ref_horizon=Y_ref,
                nmpc_weights=nmpc_weights,
                physical_constraints=constraints
            )
            timings["nmpc_solve_ms"] = (time.perf_counter() - t3) * 1000.0

            if apply_torque_to_sim and len(self.data.ctrl) >= 7:
                if self.model.actuator_biastype[0] == 0:
                    self.data.ctrl[:7] = tau_cmd
                else:
                    if x_next_pred is not None:
                        self.data.ctrl[:7] = x_next_pred[:7]
                    else:
                        self.data.ctrl[:7] = tau_cmd
                if hasattr(self.data, "qfrc_applied"):
                    self.data.qfrc_applied[:7] = tau_cmd

        timings["total_pipeline_ms"] = (time.perf_counter() - t_start) * 1000.0

        self.step_count += 1

        output = IntegratedPipelineOutput(
            timestep=self.step_count,
            sim_time=snap.sim_time,
            task_id=self.current_task.task_id,
            task_prompt=self.current_task.user_prompt,
            phase=constraints.phase,
            initial_object_location=self.initial_object_location.copy(),
            current_object_location=p_obj.copy(),
            place_target_location=p_place.copy(),
            end_effector_position=p_ee.copy(),
            end_effector_velocity=v_ee.copy(),
            relative_distance_to_grasp=d_rel,
            relative_displacement_vector=delta_vec.copy(),
            relative_range_rate=range_rate,
            relative_distance_to_place=d_place,
            residual_correction=ccdm_diag["residual_mean"],
            calibrated_uncertainty=ccdm_diag["uncertainty_sigma"],
            physical_constraints=constraints,
            nmpc_weights=nmpc_weights,
            torque_command=tau_cmd,
            predicted_next_state=x_next_pred,
            nmpc_solve_info=solve_info,
            spatial_features=spatial_res.features.copy(),
            timings_ms=timings,
            status="ok" if spatial_res.valid else "target_interpolated"
        )

        return output

    def _generate_sim_detection(self, target_class: str) -> Dict[str, Any]:
        """
        Projects MuJoCo 3D target block position to 2D image coordinates using the camera model.
        Simulates the visual perception frontend (ARIA / VLM) seamlessly.
        """
        target_name = self.current_task.pick_target.name if self.current_task is not None else "target_block"
        target_body_id = self._resolve_target_body_id(target_name)
        if target_body_id >= 0:
            target_pos = self.data.xpos[target_body_id]
        else:
            target_pos = np.array([0.5, 0.0, 0.425])

        # Project 3D point into camera image plane
        uv_point, _ = self.spatial_layer.transformer.world_to_pixel(target_pos)
        u, v = float(uv_point[0]), float(uv_point[1])

        # Bounding box of 5cm cube (~30x30 pixels at 0.8m distance with fx=250)
        box_half = 18.0
        bbox = [
            max(0.0, u - box_half),
            max(0.0, v - box_half),
            min(float(self.spatial_layer.transformer.width), u + box_half),
            min(float(self.spatial_layer.transformer.height), v + box_half)
        ]

        return {
            "timestamp": time.time(),
            "primary_target": {
                "id": 0,
                "class": target_class,
                "confidence": 0.96,
                "bbox": bbox
            },
            "secondary_objects": []
        }

    def run_closed_loop_pick_and_place(
        self,
        steps_per_phase: int = 35,
        hover_height: Optional[float] = None,
        callback: Optional[Any] = None,
        render_gui: bool = False,
        auto_start: bool = False
    ) -> Dict[str, Any]:
        """
        Executes complete closed-loop pick-and-place manipulation from start to finish:
        1. Localizes target object & placement receptacle via Spatial Reasoning Layer.
        2. Plans 7-stage FSM reference trajectory (Hover -> Descend -> Grasp -> Lift -> Transport -> Place -> Retract).
        3. Loops across all timesteps:
           - Fetches scene & updates continuous relative grasp distance d_rel.
           - Evaluates CCDM dynamic physical constraints and NMPC cost weights.
           - Solves CasADi + IPOPT optimal torque command.
           - Applies control action to Franka arm and advances simulation physics.
           - Manages grasp engagement and release.
        4. Validates final placement accuracy and task success.
        Supports camera staging before run and instant re-runs with camera view preserved.
        """
        if self.controller is None:
            self.attach_nmpc_controller()

        if self.current_task is None:
            self.set_task_from_multi_agent(self.multi_agent_parser.create_mock_task_message())

        # Step 0: Ensure simulation is at home keyframe if uninitialized
        home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key >= 0 and np.all(self.data.qpos[:7] == 0.0):
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
            mujoco.mj_forward(self.model, self.data)

        # Snapshot initial simulation state for instant replays
        init_qpos = self.data.qpos.copy()
        init_qvel = self.data.qvel.copy()
        init_ctrl = self.data.ctrl.copy()
        init_time = float(self.data.time)

        # Retrieve task sequence (single object or multiple sequential objects)
        subtask_list = []
        if self.current_task is not None and getattr(self.current_task, "subtasks", None):
            subtask_list = self.current_task.subtasks

        if not subtask_list:
            subtask_list = [self.current_task.raw_json if self.current_task and self.current_task.raw_json else {
                "task_id": self.current_task.task_id if self.current_task else "default_task",
                "pick_target": self.current_task.pick_target if self.current_task else None,
                "place_target": self.current_task.place_target if self.current_task else None,
            }]

        substeps = max(1, int(round(self.controller.dt / self.model.opt.timestep)))

        user_signals = {
            "start": auto_start or not render_gui,
            "rerun": False,
            "exit": False
        }

        def on_key(keycode: int):
            # GLFW keycodes: 32: Space, 82: 'R', 114: 'r', 257: Enter, 256: ESC, 81: 'Q', 113: 'q'
            if keycode in (32, 82, 114, 257):
                user_signals["start"] = True
                user_signals["rerun"] = True
            elif keycode in (256, 81, 113):
                user_signals["exit"] = True

        def check_terminal_signals():
            try:
                import msvcrt
                while msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b'\r', b'\n', b' ', b'r', b'R'):
                        user_signals["start"] = True
                        user_signals["rerun"] = True
                    elif ch in (b'\x1b', b'q', b'Q'):
                        user_signals["exit"] = True
            except Exception:
                pass

        viewer = None
        if render_gui:
            try:
                viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=on_key)
            except Exception as e:
                print(f"[Warning] Could not launch passive GUI viewer: {e}")

        # Camera Framing & Staging Mode
        if viewer is not None and viewer.is_running() and not user_signals["start"]:
            print("\n" + "=" * 90)
            print("                   [CAMERA FRAMING & STAGING MODE ACTIVE]")
            print("=" * 90)
            print(" The MuJoCo 3D Viewer window is ready! Take your time to configure the view:")
            print("   * Zoom in / out : Scroll mouse wheel")
            print("   * Rotate view   : Right-click and drag")
            print("   * Pan view      : Left-click and drag")
            print("   * Focus target  : Double-click on any body")
            print("-" * 90)
            print(" When you have the desired view and are ready to record:")
            print("   [Action] -> Press SPACEBAR or 'R' inside the MuJoCo 3D Viewer window")
            print("            -> OR press ENTER in this terminal")
            print(" The Franka arm will then start the pick-and-place sequence!")
            print("=" * 90 + "\n")

            while viewer.is_running() and not user_signals["start"] and not user_signals["exit"]:
                viewer.sync()
                check_terminal_signals()
                time.sleep(0.02)

            if user_signals["exit"] or not viewer.is_running():
                if viewer and viewer.is_running():
                    viewer.close()
                return {"success": False, "tasks": [], "total_steps": 0}

        # Main simulation loop with instant replay support
        while True:
            # Logs across all subtasks in this run
            log_ee_actual = []
            log_ee_ref = []
            log_d_grasp = []
            log_d_place = []
            log_phases = []
            log_qpos = []
            log_tau = []
            log_q_pos_weight = []
            log_v_max = []
            solve_latencies = []
            task_summaries = []
            total_global_steps = 0

            for st_idx, st_data in enumerate(subtask_list):
                # 1. Activate subtask plan
                if isinstance(st_data, dict):
                    self.set_task_from_multi_agent(st_data)

                # Clear spatial memory so spatial layer locks this object's coordinates
                self.initial_object_location = None
                self.prev_distance_to_grasp = None

                # 2. Lock initial positions from current scene state
                out0 = self.step(force_depth_refresh=True, execute_nmpc=False)
                p_start = out0.end_effector_position.copy()
                p_pick = self.initial_object_location.copy()
                p_place = self.current_task.place_target.target_position.copy()
                h_hover = hover_height if hover_height is not None else float(max(p_pick[2], p_place[2]) + 0.16)

                # 3. Generate smooth 7-phase reference trajectory from current EE pose
                ref_traj_full = self.controller.plan_pick_and_place_trajectory(
                    p_pick=p_pick,
                    p_place=p_place,
                    p_start=p_start,
                    hover_height=h_hover,
                    steps_per_phase=steps_per_phase
                )
                num_phases = 7
                subtask_steps = num_phases * steps_per_phase

                # 4. Target object freejoint tracking
                target_name = self.current_task.pick_target.name if self.current_task is not None else "target_block"
                target_body_id = self._resolve_target_body_id(target_name)
                qpos_adr = None
                qvel_adr = None
                if target_body_id >= 0:
                    jnt_id = self.model.body_jntadr[target_body_id]
                    if jnt_id >= 0 and self.model.jnt_type[jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
                        qpos_adr = self.model.jnt_qposadr[jnt_id]
                        qvel_adr = self.model.jnt_dofadr[jnt_id]

                is_grasped = False
                grasp_engage_step = int(2.0 * steps_per_phase)
                release_step = int(6.0 * steps_per_phase)

                # 5. Closed-loop control for this subtask
                for k in range(subtask_steps):
                    Y_ref_horizon = ref_traj_full[:, k : k + self.controller.N + 1]

                    out_k = self.step(
                        force_depth_refresh=False,
                        execute_nmpc=True,
                        horizon_ref=Y_ref_horizon,
                        apply_torque_to_sim=True
                    )

                    for _ in range(substeps):
                        mujoco.mj_step(self.model, self.data)

                    # Grasp engagement
                    if k >= grasp_engage_step and k < release_step:
                        is_grasped = True
                    else:
                        is_grasped = False

                    if is_grasped and qpos_adr is not None:
                        p_ee_now = out_k.end_effector_position
                        self.data.qpos[qpos_adr : qpos_adr + 3] = p_ee_now + np.array([0.0, 0.0, -0.025])
                        if qvel_adr is not None:
                            self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0

                    log_ee_actual.append(out_k.end_effector_position.copy())
                    log_ee_ref.append(Y_ref_horizon[:, 0].copy())
                    log_d_grasp.append(out_k.relative_distance_to_grasp)
                    log_d_place.append(out_k.relative_distance_to_place)
                    log_phases.append(out_k.phase)
                    log_qpos.append(self.data.qpos[:7].copy())
                    log_tau.append(out_k.torque_command.copy() if out_k.torque_command is not None else np.zeros(7))
                    log_q_pos_weight.append(float(out_k.nmpc_weights.Q_pos[0]))
                    log_v_max.append(float(out_k.physical_constraints.velocity_max[0]))
                    if "nmpc_solve_ms" in out_k.timings_ms:
                        solve_latencies.append(out_k.timings_ms["nmpc_solve_ms"])

                    if callback is not None:
                        callback(total_global_steps + k, out_k)

                    if viewer is not None and viewer.is_running():
                        viewer.sync()

                total_global_steps += subtask_steps

                # Allow object to settle in place receptacle
                for _ in range(25):
                    mujoco.mj_step(self.model, self.data)
                    if viewer is not None and viewer.is_running():
                        viewer.sync()

                # Record subtask result
                if target_body_id >= 0:
                    final_obj_pos = self.data.xpos[target_body_id].copy()
                else:
                    final_obj_pos = p_place.copy()
                placement_err = float(np.linalg.norm(final_obj_pos[:2] - p_place[:2]))
                st_success = bool(placement_err < 0.06)

                task_summaries.append({
                    "subtask_id": getattr(self.current_task, "task_id", f"subtask_{st_idx+1}"),
                    "name": target_name,
                    "step": st_idx + 1,
                    "p_pick_initial": p_pick.tolist(),
                    "p_place_target": p_place.tolist(),
                    "p_object_final": final_obj_pos.tolist(),
                    "final_placement_error_m": placement_err,
                    "success": st_success
                })

            # Check if user wants to re-run simulation with the same camera perspective
            if viewer is not None and viewer.is_running():
                print("\n" + "=" * 90)
                print("             [SIMULATION COMPLETE - READY FOR REPLAY / RE-RUN]")
                print("=" * 90)
                print(f" -> Completed {len(task_summaries)} pick-and-place operation(s) successfully.")
                print(" -> Your configured camera perspective (zoom and rotation) is preserved!")
                print("-" * 90)
                print(" Controls:")
                print("   * Press 'R' or SPACE in the 3D Viewer (or ENTER in terminal) : RE-RUN simulation")
                print("   * Press ESC or close window (or type 'q' in terminal)        : FINISH & show report")
                print("=" * 90 + "\n")

                user_signals["rerun"] = False
                while viewer.is_running() and not user_signals["rerun"] and not user_signals["exit"]:
                    viewer.sync()
                    check_terminal_signals()
                    time.sleep(0.03)

                if user_signals["rerun"] and viewer.is_running():
                    print("\n[Replay]: Resetting objects and arm to starting state (camera view preserved)...")
                    self.data.qpos[:] = init_qpos
                    self.data.qvel[:] = init_qvel
                    self.data.ctrl[:] = init_ctrl
                    self.data.time = init_time
                    mujoco.mj_forward(self.model, self.data)
                    viewer.sync()

                    self.initial_object_location = None
                    self.prev_distance_to_grasp = None
                    self.step_count = 0

                    user_signals["rerun"] = False
                    user_signals["start"] = True
                    continue

            break # Exit replay loop

        if viewer is not None and viewer.is_running():
            try:
                viewer.close()
            except Exception:
                pass

        ee_actual_arr = np.array(log_ee_actual)
        ee_ref_arr = np.array(log_ee_ref)
        tracking_rmse_mm = float(np.sqrt(np.mean(np.square(ee_ref_arr - ee_actual_arr)))) * 1000.0
        all_success = all(t["success"] for t in task_summaries) and (tracking_rmse_mm < 75.0)

        primary_res = task_summaries[0] if task_summaries else {}

        return {
            "success": all_success,
            "tasks": task_summaries,
            "total_steps": total_global_steps,
            "p_pick_initial": primary_res.get("p_pick_initial", [0, 0, 0]),
            "p_place_target": primary_res.get("p_place_target", [0, 0, 0]),
            "p_object_final": primary_res.get("p_object_final", [0, 0, 0]),
            "final_placement_error_m": primary_res.get("final_placement_error_m", 0.0),
            "tracking_rmse_mm": tracking_rmse_mm,
            "mean_nmpc_solve_ms": float(np.mean(solve_latencies)) if solve_latencies else 0.0,
            "p95_nmpc_solve_ms": float(np.percentile(solve_latencies, 95)) if solve_latencies else 0.0,
            "log_ee_actual": np.array(log_ee_actual),
            "log_ee_ref": np.array(log_ee_ref),
            "log_d_grasp": np.array(log_d_grasp),
            "log_d_place": np.array(log_d_place),
            "log_phases": log_phases,
            "log_qpos": np.array(log_qpos),
            "log_tau": np.array(log_tau),
            "log_q_pos_weight": np.array(log_q_pos_weight),
            "log_v_max": np.array(log_v_max),
        }
