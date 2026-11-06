# orchestrator.py
# ─────────────────────────────────────────────────────────────────────────────
# ARIA Orchestrator — LangGraph-based multi-agent pipeline
# Handles: web research, image/video analysis, AND robot pickup (SAM2 pipeline)
#
# NOTE: The MPC Controller is intentionally kept OUTSIDE this multi-agent
# system. The robot branch terminates at robot_planner, which emits a clean
# `robot_handoff` payload (6-DOF grasp poses + selected items). The external
# MPC Controller reads that payload independently once the full pipeline is
# validated on datasets and wired to simulation / hardware.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
from typing import Any, Dict

try:
    from langgraph.graph import END, StateGraph
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    END = "__end__"

# ── Schema ────────────────────────────────────────────────────────────────────
from state.schema import OrchestratorState, Route

# ── Agents ────────────────────────────────────────────────────────────────────
from agents.planner import planner_node
from agents.vision_agent import vision_agent_node
from agents.research_agent import research_agent_node
from agents.nutrition_agent import nutrition_agent_node # ── NEW
from agents.synthesis_agent import synthesis_agent_node
from agents.critic import critic_node

# ── Robot Pipeline Agents ─────────────────────────────────────────────────────
# mpc_controller is intentionally NOT imported here.
# It lives outside the multi-agent graph and will be wired in separately
# once the pipeline is validated end-to-end on datasets.
from agents.robot_perception_agent import robot_perception_node
from agents.robot_planner_agent import robot_planner_node

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def route_after_planner(state: OrchestratorState) -> str:
    """
    Reads the route decided by the planner and directs the graph
    to the correct next node.

    Routes:
    Route.IMAGE → vision_agent
    Route.VIDEO → vision_agent
    Route.VIDEO_FOOD → vision_agent (then nutrition_agent before synthesizer)
    Route.ROBOT_PICKUP → robot_perception (YOLO + SAM2 + Depth)
    Route.WEB (default)→ research_agent
    Route.EXCEL → research_agent
    """
    intent = state.get("intent") or {}
    route = intent.get("route", Route.WEB)
    logger.info("[Orchestrator] Planner chose route: %s", route)
    if route in (Route.IMAGE, Route.VIDEO, Route.VIDEO_FOOD):
        return "vision_agent"
    if route == Route.ROBOT_PICKUP:
        return "robot_perception" # ← SAM2 pipeline entry point
    return "research_agent" # default: web / excel research

def route_after_vision(state: OrchestratorState) -> str:
    """
    After the vision agent runs, check whether this is a VIDEO_FOOD route.
    VIDEO_FOOD → nutrition_agent (dietary analysis on top of food detection)
    All others → synthesizer (direct synthesis)
    """
    intent = state.get("intent") or {}
    route = intent.get("route", Route.WEB)
    if route == Route.VIDEO_FOOD:
        logger.info("[Orchestrator] VIDEO_FOOD route — forwarding to nutrition_agent.")
        return "nutrition_agent"
    return "synthesizer"

def route_after_critic(state: OrchestratorState) -> str:
    """
    After the critic evaluates the synthesizer output:
    - If quality is acceptable → END
    - If quality needs revision → back to synthesizer (one retry)
    """
    critique = state.get("critic_feedback") or {}
    if not critique.get("approved", True):
        logger.info("[Orchestrator] Critic rejected output — looping back to synthesizer.")
        return "synthesizer"
    return END

# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class FallbackCompiledGraph:
    """Direct execution runner when LangGraph library is not installed."""
    def invoke(self, state: Dict[str, Any]) -> OrchestratorState:
        s = dict(state)
        # 1. Planner
        p_up = planner_node(s)
        s.update(p_up)

        # 2. Routing after planner
        next_node = route_after_planner(s)
        if next_node == "robot_perception":
            rp_up = robot_perception_node(s)
            s.update(rp_up)
            rplan_up = robot_planner_node(s)
            s.update(rplan_up)
        elif next_node == "vision_agent":
            v_up = vision_agent_node(s)
            s.update(v_up)
            if route_after_vision(s) == "nutrition_agent":
                n_up = nutrition_agent_node(s)
                s.update(n_up)
        else: # research_agent
            res_up = research_agent_node(s)
            s.update(res_up)

        # 3. Synthesizer
        synth_up = synthesis_agent_node(s)
        s.update(synth_up)

        # 4. Critic loop
        crit_up = critic_node(s)
        s.update(crit_up)
        if route_after_critic(s) == "synthesizer":
            synth_up = synthesis_agent_node(s)
            s.update(synth_up)
            crit_up = critic_node(s)
            s.update(crit_up)

        return s

def build_graph() -> Any:
    """
    Constructs and compiles the full ARIA LangGraph.

    Graph topology:
    ┌──────────┐
    │ planner │
    └────┬─────┘
         │ route_after_planner()
         ├──────────────────────────► vision_agent
         │                             │ route_after_vision()
         │                             ├──────────────────► nutrition_agent ──┐
         │                             │                    (VIDEO_FOOD only) │
         │                             └──────────────────► synthesizer ◄─────┘
         │                               (IMAGE / VIDEO)    ▲
         ├──────────────────────────► research_agent ───────┤
         │                                                  │
         └──────────────────────────► robot_perception      │
                                       │                    │
                                      robot_planner         │
                                  (emits robot_handoff) ────┘
                                       │
                                     critic
                                       │ route_after_critic()
                                       ├──► synthesizer (revision loop)
                                       └──► END

    External MPC Controller (wired in separately — NOT part of this graph):
    Reads : state["robot_handoff"] (6-DOF poses + selected items)
    Runs : CasADi / OSQP trajectory optimisation
    Sends : Joint waypoints to robot hardware / simulator
    """
    if not HAS_LANGGRAPH:
        return FallbackCompiledGraph()

    g = StateGraph(OrchestratorState)

    # ── Register all nodes ────────────────────────────────────────────────────
    g.add_node("planner", planner_node)
    g.add_node("vision_agent", vision_agent_node)
    g.add_node("research_agent", research_agent_node)
    g.add_node("nutrition_agent", nutrition_agent_node) # ── NEW
    g.add_node("synthesizer", synthesis_agent_node)
    g.add_node("critic", critic_node)

    # Robot pipeline nodes (perception + reasoning only — no controller)
    g.add_node("robot_perception", robot_perception_node) # YOLO + SAM2 + Depth
    g.add_node("robot_planner", robot_planner_node) # LLM → 6-DOF grasp poses

    # ── Entry point ───────────────────────────────────────────────────────────
    g.set_entry_point("planner")

    # ── Conditional routing after planner ─────────────────────────────────────
    g.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "vision_agent": "vision_agent",
            "research_agent": "research_agent",
            "robot_perception": "robot_perception",
        },
    )

    # ── Vision routing — VIDEO_FOOD forks to nutrition, others go direct ───────
    # ── FIX: vision_agent no longer has a direct edge to synthesizer.
    # route_after_vision() decides the next hop based on the route.
    g.add_conditional_edges(
        "vision_agent",
        route_after_vision,
        {
            "nutrition_agent": "nutrition_agent", # VIDEO_FOOD route
            "synthesizer": "synthesizer", # IMAGE / VIDEO routes
        },
    )

    # ── Nutrition agent always flows to synthesizer ────────────────────────────
    g.add_edge("nutrition_agent", "synthesizer")

    # ── Research pipeline ──────────────────────────────────────────────────────
    g.add_edge("research_agent", "synthesizer")

    # ── Robot pipeline edges ───────────────────────────────────────────────────
    # robot_planner is the terminal node of the robot branch inside this graph.
    # It writes a clean `robot_handoff` dict into state, then passes control
    # to the synthesizer which summarises the pickup plan for the user.
    # The external MPC Controller picks up `robot_handoff` after graph exit.
    g.add_edge("robot_perception", "robot_planner") # SAM2 masks → grasp plan
    g.add_edge("robot_planner", "synthesizer") # grasp plan → summary

    # ── Critic loop ───────────────────────────────────────────────────────────
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "synthesizer": "synthesizer", # revision loop
            END: END,
        },
    )

    return g.compile()

# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON — compile once, reuse everywhere
# ─────────────────────────────────────────────────────────────────────────────
graph = build_graph()

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(initial_state: Dict[str, Any]) -> OrchestratorState:
    """
    Runs the full ARIA multi-agent pipeline from an initial state dict.
    Args:
        initial_state: Must contain at minimum:
        {
            "user_input": str, # raw user query
            "session_id": str, # unique session identifier
            "image_path": str | None, # path to image/video if applicable
            "depth_path": str | None, # path to depth map (robot pickup only)
        }
    Returns:
        Final OrchestratorState after the graph completes.
        For VIDEO_FOOD routes the state will additionally contain:
        state["analysis"] — structured nutrition recommendations
        produced by nutrition_agent_node.
        For ROBOT_PICKUP routes the state will additionally contain:
        state["robot_handoff"] — controller-ready payload:
        {
            "selected_items": [...], # detected + chosen objects
            "grasp_poses": [...], # 6-DOF poses in base_link frame
            "masks": ..., # SAM2 mask references
            "depth_map": ..., # aligned depth map reference
            "timestamp": str,
        }
        This payload is consumed by the external MPC Controller
        independently of this graph.
    """
    logger.info(
        "[Orchestrator] Starting pipeline. Session: %s",
        initial_state.get("session_id", "unknown"),
    )
    final_state = graph.invoke(initial_state)
    logger.info(
        "[Orchestrator] Pipeline complete. Route taken: %s",
        (final_state.get("intent") or {}).get("route", "unknown"),
    )
    return final_state
