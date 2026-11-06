# state/schema.py
from __future__ import annotations
from typing import TypedDict, Optional, List, Dict, Any

# ══════════════════════════════════════════════════════════════════════════════
# ROUTE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
class Route:
    WEB = "web"
    EXCEL = "excel"
    IMAGE = "image"
    VIDEO = "video"
    VIDEO_FOOD = "video_food"
    ROBOT_PICKUP = "robot_pickup"
    ALL = {WEB, EXCEL, IMAGE, VIDEO, VIDEO_FOOD, ROBOT_PICKUP}

# ══════════════════════════════════════════════════════════════════════════════
# AGENT NAME CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
class AgentName:
    PLANNER = "planner"
    VISION = "vision"
    NUTRITION = "nutrition"
    RESEARCH = "research"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"
    ROBOT_PERC = "robot_perception"
    ROBOT_PLAN = "robot_planner"
    # ── FIX: MPC constant added — used by planner.py _fallback_intent()
    # and PLANNER_SYSTEM prompt. MPC Controller itself is external to the
    # graph — this constant is only used as a label in plans/subtasks.
    MPC = "mpc_controller"

# ══════════════════════════════════════════════════════════════════════════════
# SUB-TYPED DICTS — general
# ══════════════════════════════════════════════════════════════════════════════
class AgentMessage(TypedDict, total=False):
    role: str
    content: str
    timestamp: str

class PlannerIntent(TypedDict, total=False):
    route: str
    agents_needed: List[str]
    reasoning: str
    requires_file: bool
    file_type: str
    confidence: int
    target_object: str # NEW — object to grasp/manipulate (ROBOT_PICKUP routes); "" otherwise

class SubTask(TypedDict, total=False):
    task_id: str
    description: str
    agent: str

# ══════════════════════════════════════════════════════════════════════════════
# CRITIC FEEDBACK TYPED DICT
# ══════════════════════════════════════════════════════════════════════════════
class CriticFeedback(TypedDict, total=False):
    approved: bool
    score: int
    issues: List[str]
    suggestions: List[str]
    retry_reason: str
    reviewed_at_step: int

# ══════════════════════════════════════════════════════════════════════════════
# ROBOT-SPECIFIC TYPED DICTS
# ══════════════════════════════════════════════════════════════════════════════
class DetectedObject(TypedDict, total=False):
    """
    One object detected by the robot perception pipeline.

    Fields
    ------
    label : class name — from YOLO's fixed COCO vocabulary, or the
            raw target_object string when sourced from the VLM
            open-vocabulary fallback (e.g. "red mug")
    confidence : detector confidence in [0, 1]
    bbox_xyxy : bounding box [x1, y1, x2, y2] in pixels
    mask : SAM2 binary mask — (H, W) uint8 numpy array, or None
    depth_m : metric depth at object centroid in metres
    centroid_3d : [X, Y, Z] in camera frame (metres), back-projected
                  using intrinsics sourced live from MuJoCo at runtime
    centroid_src : "sam2" if centroid came from SAM2 mask moments,
                   "bbox" if it fell back to the detector's bounding-box centre
    detector_src : "yolo" if this detection came from YOLOv8's fixed 80-class
                   vocabulary, "vlm" if it came from the open-vocabulary
                   Qwen2.5-VL fallback (llm/vlm_client.py) — only triggered
                   when YOLO fails to find a match for state["target_object"].
                   Mirrors the existing centroid_src pattern for traceability.
    """
    label: str
    confidence: float
    bbox_xyxy: List[float]
    mask: Any
    depth_m: float
    centroid_3d: List[float]
    centroid_src: str
    detector_src: str # NEW — "yolo" or "vlm", mirrors the existing centroid_src pattern

class GraspPlan(TypedDict, total=False):
    """
    Internal grasp plan produced by robot_planner_agent.
    Poses are stored as raw [x, y, z, roll, pitch, yaw] lists here.
    The conversion to named dicts happens inside _build_robot_handoff()
    in robot_planner_agent.py — that is the MPC Controller interface boundary.
    """
    target_object: str
    grasp_pose: List[float]
    pre_grasp_pose: List[float]
    post_grasp_pose: List[float]
    gripper_width: float
    confidence: float
    reasoning: str

class TrajectoryPoint(TypedDict, total=False):
    t: float
    q: List[float]
    dq: List[float]
    ddq: List[float]

class MPCResult(TypedDict, total=False):
    success: bool
    trajectory: List[TrajectoryPoint]
    solve_time_ms: float
    cost: float
    error_message: str

# ══════════════════════════════════════════════════════════════════════════════
# CAMERA INTRINSICS TYPED DICT
# ══════════════════════════════════════════════════════════════════════════════
class CameraIntrinsics(TypedDict, total=False):
    """
    Camera intrinsic parameters.
    In simulation : populated live from MuJoCo via
                    get_camera_intrinsics_from_mujoco() — MuJoCo XML is the
                    single source of truth; FX/FY/CX/CY are never hardcoded.
    Sim-to-Real : populated from RealSense factory calibration, e.g.:
                  {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0}
    """
    fx: float
    fy: float
    cx: float
    cy: float
    K: Any # np.ndarray (3, 3) float64

# ══════════════════════════════════════════════════════════════════════════════
# MASTER ORCHESTRATOR STATE
# ══════════════════════════════════════════════════════════════════════════════
class OrchestratorState(TypedDict, total=False):
    """
    Shared state dict that flows through every LangGraph node.

    Sections
    --------
    Core — query, paths, session, messages, control flags
    Planner — intent, plan, subtasks, critic feedback
    Vision / Food — food detection and nutrition data
    Robot Perception — detected objects, SAM2 masks, depth map, scene description
    MuJoCo Runtime — live MjModel / MjData objects + depth buffer
    Camera — intrinsics resolved live from MuJoCo (no hardcoding)
    Robot Planner — grasp plan, target object, robot_handoff payload
    MPC Controller — reserved for future wiring (external, not a graph node)
    Output — final answer and analysis
    """
    # ── Core ──────────────────────────────────────────────────────────────────
    # ── FIX: user_input is the canonical key used by main.py and run_pipeline()
    # user_query is kept as a secondary alias for backward compatibility with
    # any legacy code that still writes it — all agents now read user_input first.
    user_input: str # canonical key (set by main.py)
    user_query: str # legacy alias — do not use in new code
    session_id: str # ── FIX: added — required by run_pipeline()
    video_path: Optional[str]
    image_path: Optional[str]
    health_profile: Optional[Dict[str, Any]]
    messages: List[AgentMessage]
    current_step: int
    retry_count: int
    needs_retry: bool
    tool: str

    # ── Planner ───────────────────────────────────────────────────────────────
    intent: Optional[PlannerIntent]
    plan: Optional[str]
    subtasks: Optional[List[SubTask]]
    critic_feedback: Optional[CriticFeedback] # typed — was Dict[str, Any]

    # ── Vision / Food ─────────────────────────────────────────────────────────
    detected_foods: Optional[str]
    nutrition_data: Optional[str]
    research_data: Optional[str]
    analysis: Optional[str] # written by nutrition_agent_node

    # ── Robot Perception ──────────────────────────────────────────────────────
    detected_objects: Optional[List[DetectedObject]]
    scene_description: Optional[str]
    masks: Optional[Any]
    depth_map: Optional[Any]

    # ── MuJoCo Runtime objects ────────────────────────────────────────────────
    mujoco_model: Optional[Any]
    mujoco_data: Optional[Any]
    mujoco_depth_buffer: Optional[Any]

    # ── Real hardware depth (sim-to-real) ─────────────────────────────────────
    realsense_depth_map: Optional[Any]

    # ── Camera intrinsics ─────────────────────────────────────────────────────
    cam_intrinsics: Optional[CameraIntrinsics]
    cam_fovy: Optional[float]

    # ── Robot Planner ─────────────────────────────────────────────────────────
    target_object: Optional[str]
    grasp_plan: Optional[GraspPlan]
    robot_handoff: Optional[Dict[str, Any]]

    # ── MPC Controller (reserved — external to graph) ─────────────────────────
    # mpc_result: Optional[MPCResult] # uncomment when MPC is wired in
    # execution_status: Optional[str] # uncomment when MPC is wired in

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer: Optional[str]

# ══════════════════════════════════════════════════════════════════════════════
# LEGACY ALIAS
# Keeps all existing agents that import AgentState working without changes.
# ══════════════════════════════════════════════════════════════════════════════
AgentState = OrchestratorState

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def add_message(
    state: OrchestratorState,
    role: str,
    content: str,
) -> List[AgentMessage]:
    """
    Append a timestamped message to state['messages'] and return the new list.

    Args:
        state : current OrchestratorState
        role : sender identifier — use AgentName constants
        content : message body string

    Returns:
        New list of AgentMessage dicts (does not mutate state in place).
    """
    import datetime
    msgs: List[AgentMessage] = list(state.get("messages") or [])
    msgs.append({
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })
    return msgs

def create_initial_state(
    user_input: str, # ── FIX: renamed from user_query
    video_path: str = "",
    image_path: str = "",
    target_object: str = "",
    health_profile: dict = None,
    mujoco_model: Any = None,
    mujoco_data: Any = None,
    session_id: str = "", # ── FIX: added session_id parameter
) -> OrchestratorState:
    """
    Factory — create a clean initial OrchestratorState.

    Args:
        user_input : the user's request string (canonical key)
        video_path : path to input video file (food pipeline)
        image_path : path to RGB frame for robot perception
        target_object : object label the robot should pick up
        health_profile : user dietary / health constraints dict
        mujoco_model : mujoco.MjModel instance
        mujoco_data : mujoco.MjData instance
        session_id : unique session identifier (auto-generated if empty)

    Returns:
        OrchestratorState with all fields initialised to safe defaults.

    Example:
        import mujoco
        model = mujoco.MjModel.from_xml_path("franka_panda_scene.xml")
        data = mujoco.MjData(model)
        state = create_initial_state(
            user_input = "Pick up the red cube",
            image_path = "frames/frame_001.png",
            target_object = "red_cube",
            mujoco_model = model,
            mujoco_data = data,
        )
    """
    import uuid
    return OrchestratorState(
        user_input = user_input,
        user_query = user_input, # legacy alias — keep in sync
        session_id = session_id or str(uuid.uuid4()),
        video_path = video_path or None,
        image_path = image_path or None,
        health_profile = health_profile or {},
        target_object = target_object or None,
        mujoco_model = mujoco_model,
        mujoco_data = mujoco_data,
        messages = [],
        current_step = 0,
        retry_count = 0,
        needs_retry = False,
        tool = Route.WEB,
        detected_objects = None,
        masks = None,
        depth_map = None,
        grasp_plan = None,
        robot_handoff = None,
        cam_intrinsics = None,
        cam_fovy = None,
        analysis = None,
        final_answer = None,
    )
