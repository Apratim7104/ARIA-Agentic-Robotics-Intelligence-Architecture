# agents/planner.py
"""
Planner Agent
=============
Routes the user's request to the correct downstream agent(s) by calling
the LLM and parsing a structured JSON intent object.

State keys READ:
 user_input : str — raw user query (canonical key used by orchestrator)
 retry_count : int — how many critic-driven retries have occurred
 critic_feedback: dict | None — structured feedback from the critic agent

State keys WRITTEN:
 intent : PlannerIntent — route + agents + metadata
 plan : str — paragraph describing the overall plan
 subtasks : list — ordered list of {task_id, description, agent}
 tool : str — convenience copy of intent["route"]
 target_object : str — object to grasp/manipulate (ROBOT_PICKUP routes,
 "" for all other routes) — top-level key, this
 is what robot_planner_agent.py reads directly
 current_step : int — incremented step counter
 messages : list — updated message log
 needs_retry : bool — reset to False at planner stage
"""
import json
import logging
from typing import Dict, Any

from state.schema import (
    OrchestratorState, PlannerIntent, AgentMessage,
    add_message, Route, AgentName,
)
from llm.client import call_llm_json, system_user

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """You are the Planner Agent in a multi-agent AI orchestrator.

AVAILABLE ROUTES:
 "web" -> General knowledge / current events -> Research Agent (web search)
 "excel" -> Questions about an uploaded Excel file -> Research Agent (Excel)
 "image" -> Analyse a single image -> Vision Agent
 "video" -> Analyse a video -> Vision Agent
 "video_food" -> Analyse a food video + dietary advice -> Vision Agent + Nutrition Agent
 "robot_pickup" -> Pick up a physical object using robot arm -> Robot Perception Agent + Robot Planner Agent + MPC Controller

AVAILABLE AGENTS:
 "vision" -> Handles image/video analysis (local Qwen2.5-VL-7B)
 "nutrition" -> Gives dietary recommendations based on detected food
 "research" -> Handles web search and Excel file queries
 "robot_perception" -> Detects and segments objects in a scene (YOLO + SAM2)
 "robot_planner" -> Plans grasp pose and trajectory for robot arm
 "mpc_controller" -> Executes trajectory via Model Predictive Control

OUTPUT FORMAT (strict JSON, no markdown):
{
 "route" : "<web|excel|image|video|video_food|robot_pickup>",
 "agents_needed" : ["<agent1>"],
 "reasoning" : "<one sentence>",
 "requires_file" : <true|false>,
 "file_type" : "<image|video|excel|>",
 "target_object" : "<short noun phrase naming the object to grasp, e.g. 'red mug'; empty string if route != robot_pickup>",
 "confidence" : <1-10>,
 "plan" : "<paragraph>",
 "subtasks" : [
  {"task_id": "t1", "description": "<what>", "agent": "<who>"}
 ]
}

ROUTING RULES:
- If the user wants to pick up, grab, grasp, or move a physical object → use "robot_pickup"
- If an image_path is provided alongside a pickup request → still use "robot_pickup"
- "robot_pickup" always needs agents: ["robot_perception", "robot_planner", "mpc_controller"]
- If an image_path or video_path is provided without a pickup intent → use "image" or "video"
- "video_food" is for food/nutrition analysis of a video scene specifically
- For "robot_pickup", always extract target_object as a short noun phrase (e.g. "red mug", "the blue bottle") — this is the object the robot must grasp. For every other route, set target_object to "".
"""

# ── Main Node ──────────────────────────────────────────────────────────────────
def planner_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — determine route and build a structured plan.
    Reads state["user_input"] (canonical key) — NOT "user_query".
    Falls back gracefully if the LLM call fails.

    Args:
        state : OrchestratorState typed dict.

    Returns:
        Partial state update dict with intent, plan, subtasks, tool,
        target_object, current_step, messages, and needs_retry.
    """
    step = state.get("current_step", 0) + 1
    # ── FIX: use "user_input" — the canonical key set by run_pipeline() ───────
    user_input = state.get("user_input") or state.get("user_query", "")
    retry_count = state.get("retry_count", 0)
    critic_fb = state.get("critic_feedback")

    logger.info(
        "[planner] step=%d retry=%d query=%s",
        step, retry_count, user_input[:80],
    )

    # ── Build user prompt (with optional critic feedback on retry) ────────────
    user_prompt = f"USER QUERY: {user_input}"
    if retry_count > 0 and critic_fb:
        user_prompt += (
            f"\n\nCRITIC FEEDBACK (retry #{retry_count}):\n"
            f"Issues: {json.dumps(critic_fb.get('issues', []))}\n"
            f"Suggestions: {json.dumps(critic_fb.get('suggestions', []))}"
        )

    # ── Hint the planner when file paths are already in state ─────────────────
    # This prevents misrouting when the user says "analyse this" and the
    # orchestrator has already resolved the file path into state.
    if state.get("video_path") and "video" not in user_input.lower():
        user_prompt += "\n\n[SYSTEM HINT: A video_path is present in state — prefer a video route.]"
    if state.get("image_path") and "image" not in user_input.lower() and not state.get("video_path"):
        user_prompt += "\n\n[SYSTEM HINT: An image_path is present in state — prefer an image route.]"

    # ── Call LLM ──────────────────────────────────────────────────────────────
    try:
        result = call_llm_json(
            messages=system_user(PLANNER_SYSTEM, user_prompt),
            temperature=0.1,
            max_tokens=2000, # NEW — plan + subtasks JSON needs more than the 1024 default
            caller=AgentName.PLANNER,
        )
    except Exception as exc:
        logger.error("[planner] LLM failed: %s — falling back to keyword routing.", exc)
        result = _fallback_intent(user_input)

    # ── Build PlannerIntent ───────────────────────────────────────────────────
    route = result.get("route", Route.WEB)
    if route not in Route.ALL:
        logger.warning("[planner] Unknown route '%s' — defaulting to WEB.", route)
        route = Route.WEB

    intent: PlannerIntent = {
        "route": route,
        "agents_needed": result.get("agents_needed", [AgentName.RESEARCH]),
        "reasoning": result.get("reasoning", ""),
        "requires_file": result.get("requires_file", False),
        "file_type": result.get("file_type", ""),
        "target_object": result.get("target_object", ""), # NEW
        "confidence": result.get("confidence", 5),
    }

    plan = result.get("plan", f"Process user query: {user_input}")
    subtasks = result.get("subtasks", [
        {"task_id": "t1", "description": user_input, "agent": AgentName.RESEARCH}
    ])

    # ── Update message log ────────────────────────────────────────────────────
    updated_messages = add_message(
        state={**state, "current_step": step},
        role=AgentName.PLANNER,
        content=(
            f"Route={intent['route']} | Agents={intent['agents_needed']} | "
            f"Target={intent['target_object'] or 'n/a'} | "
            f"Confidence={intent['confidence']}/10 | Reasoning={intent['reasoning']}"
        ),
    )

    return {
        "intent": intent,
        "plan": plan,
        "subtasks": subtasks,
        "tool": intent["route"],
        # NEW — top-level key, this is what robot_planner_agent.py actually reads
        # via state.get("target_object"); nesting it only inside "intent" would
        # never reach the robot branch.
        "target_object": intent.get("target_object", ""),
        "current_step": step,
        "messages": updated_messages,
        "needs_retry": False,
    }

# ── Fallback Intent ────────────────────────────────────────────────────────────
def _extract_target_keyword(user_input: str) -> str:
    """
    Best-effort, LLM-free target-object guess: strip common command verbs
    and filler words, return whatever noun phrase remains.

    Only used when the LLM call fails — the LLM path is always preferred
    for real target extraction, since this heuristic has no grammar awareness
    and will do a poor job on anything but short imperative commands.

    Args:
        user_input : raw user query string.

    Returns:
        Best-effort noun phrase, or "" if nothing meaningful remains.
    """
    verbs_and_fillers = {
        "pick", "up", "grab", "grasp", "lift", "move", "place", "the",
        "a", "an", "robot", "please", "can", "you", "put", "down",
        "and", "it", "into", "onto", "on", "in", "to", "at", "target",
        "spot", "bin", "table", "tray", "box", "location", "position",
    }
    input_part = user_input.lower()
    for sep in [" and ", " then ", " onto ", " into ", " to "]:
        if sep in input_part:
            input_part = input_part.split(sep)[0]
            break
    words = [w for w in input_part.split() if w not in verbs_and_fillers]
    return " ".join(words).strip()

def _fallback_intent(user_input: str) -> Dict[str, Any]:
    """
    Keyword-based routing fallback when the LLM is unavailable.
    Checks for robot-related keywords first, then food/video keywords,
    then defaults to web search.

    Args:
        user_input : raw user query string.

    Returns:
        A result dict in the same shape as the LLM JSON response.
    """
    lower = user_input.lower()

    # ── Robot pickup ──────────────────────────────────────────────────────────
    robot_keywords = {"pick", "grab", "grasp", "lift", "move", "place", "robot"}
    if any(kw in lower for kw in robot_keywords):
        return {
            "route": Route.ROBOT_PICKUP,
            "agents_needed": [AgentName.ROBOT_PERC, AgentName.ROBOT_PLAN, AgentName.MPC],
            "reasoning": "LLM unavailable — detected robot pickup keywords",
            "requires_file": True,
            "file_type": "image",
            "target_object": _extract_target_keyword(user_input), # NEW
            "confidence": 1,
            "plan": f"Attempt robot pickup for: {user_input}",
            "subtasks": [
                {"task_id": "t1", "description": "Detect objects in scene", "agent": AgentName.ROBOT_PERC},
                {"task_id": "t2", "description": "Plan grasp", "agent": AgentName.ROBOT_PLAN},
                {"task_id": "t3", "description": "Execute via MPC", "agent": AgentName.MPC},
            ],
        }

    # ── Food video ────────────────────────────────────────────────────────────
    food_video_keywords = {"food", "buffet", "meal", "lunch", "dinner", "calorie", "nutrition"}
    if any(kw in lower for kw in food_video_keywords):
        return {
            "route": Route.VIDEO_FOOD,
            "agents_needed": [AgentName.VISION],
            "reasoning": "LLM unavailable — detected food/video keywords",
            "requires_file": True,
            "file_type": "video",
            "target_object": "", # NEW — not applicable outside robot_pickup
            "confidence": 1,
            "plan": f"Analyse food scene for: {user_input}",
            "subtasks": [
                {"task_id": "t1", "description": "Detect food items", "agent": AgentName.VISION},
            ],
        }

    # ── Default: web search ───────────────────────────────────────────────────
    return {
        "route": Route.WEB,
        "agents_needed": [AgentName.RESEARCH],
        "reasoning": "LLM unavailable — defaulting to web search",
        "requires_file": False,
        "file_type": "",
        "target_object": "", # NEW — not applicable outside robot_pickup
        "confidence": 1,
        "plan": f"Search the web for: {user_input}",
        "subtasks": [{"task_id": "t1", "description": user_input, "agent": AgentName.RESEARCH}],
    }
