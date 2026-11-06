# agents/synthesis_agent.py
"""
Synthesis Agent
===============
Final node in the ARIA LangGraph pipeline.

Responsibility:
 Gather all upstream agent outputs from state and produce a single,
 clean, actionable final_answer for the user.

Routing behaviour:
 Route.ROBOT_PICKUP -> robot-aware synthesis (grasp plan + scene description)
 All other routes -> food/nutrition synthesis (detected_foods, nutrition,
                     research, critic_feedback)

State keys READ:
 user_input : str — original user request (canonical key)
 tool : str — active route (Route constant)
 analysis : str | None — nutrition analysis from nutrition agent
 critic_feedback : dict | None — structured feedback from critic agent
 research_data : str | None — web / research agent output
 detected_foods : str | None — food labels from vision agent
 scene_description : str | None — robot perception scene summary
 grasp_plan : GraspPlan | None — grasp plan from robot planner agent
 robot_handoff : dict | None — full MPC handoff payload

State keys WRITTEN:
 final_answer : str — polished final answer returned to the user
 messages : list — updated message log
 current_step : int — incremented step counter
"""
from __future__ import annotations
import os
from typing import Any, Dict
from llm.client import client
from state.schema import (
    AgentName,
    OrchestratorState,
    Route,
    add_message,
)

# AgentState alias kept for any legacy imports elsewhere in the codebase
AgentState = OrchestratorState

MODEL = os.getenv("OPENAI_MODEL", "qwen-3.6-27b")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _get_text(msg: Any, fallback: str = "No answer generated.") -> str:
    """
    Safely extract text content from an OpenAI ChatCompletion message object.
    Handles models that expose a separate 'reasoning' field (e.g. o1, Qwen-3)
    by falling back to it when the main content is empty.
    Args:
        msg : ChatCompletion message object (response.choices[0].message)
        fallback : string to return if both content and reasoning are empty
    Returns:
        Non-empty string — never None.
    """
    return msg.content or getattr(msg, "reasoning", None) or fallback

def _format_critic_feedback(critic_feedback: Any) -> str:
    """
    Render critic_feedback (dict or str) into a readable string for the prompt.
    Args:
        critic_feedback : dict from critic agent or plain string
    Returns:
        Formatted string, or 'None' if empty.
    """
    if not critic_feedback:
        return "None"
    if isinstance(critic_feedback, dict):
        lines = []
        for k, v in critic_feedback.items():
            lines.append(f" {k}: {v}")
        return "\n".join(lines)
    return str(critic_feedback)

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def _build_food_prompt(state: OrchestratorState) -> str:
    """
    Build the synthesis prompt for food / nutrition routes.
    Reads: user_input, detected_foods, analysis, critic_feedback, research_data
    """
    # ── FIX: use "user_input" — the canonical key set by main.py / run_pipeline()
    query = state.get("user_input") or state.get("user_query") or ""
    detected_foods = state.get("detected_foods") or "Not detected."
    analysis = state.get("analysis") or "Not available."
    critic_raw = state.get("critic_feedback")
    research_data = state.get("research_data") or "Not available."
    critic_str = _format_critic_feedback(critic_raw)

    return f"""You are a final synthesis expert. Your job is to produce a clean,
clear and actionable final answer for the user.

User Query: {query}

Detected Foods at the event:
{detected_foods}

Nutritional Analysis:
{analysis}

Critic / Quality Feedback:
{critic_str}

Research Context:
{research_data}

Now write a final, polished, friendly and actionable answer for the user.
Include:
1. What to eat and why
2. What to avoid and why
3. Portion recommendations
4. Estimated calorie intake
5. Motivational closing tip
"""

def _build_robot_prompt(state: OrchestratorState) -> str:
    """
    Build the synthesis prompt for the robot_pickup route.
    Reads: user_input, scene_description, grasp_plan, robot_handoff
    The MPC Controller is external — synthesis only reports the planned
    grasp, not execution results (those are written post-graph by MPC).
    """
    # ── FIX: use "user_input" — the canonical key set by main.py / run_pipeline()
    query = state.get("user_input") or state.get("user_query") or ""
    scene_description = state.get("scene_description") or "No scene description available."
    grasp_plan = state.get("grasp_plan") or {}
    robot_handoff = state.get("robot_handoff") or {}

    # Summarise grasp plan fields for the prompt
    target_object = grasp_plan.get("target_object", robot_handoff.get("selected_items", "unknown"))
    confidence = grasp_plan.get("confidence", robot_handoff.get("confidence", "N/A"))
    reasoning = grasp_plan.get("reasoning", robot_handoff.get("reasoning", "Not provided."))
    gripper_width = grasp_plan.get("gripper_width", robot_handoff.get("gripper_width", "N/A"))
    grasp_pose = grasp_plan.get("grasp_pose", "Not computed.")
    pre_grasp_pose = grasp_plan.get("pre_grasp_pose", "Not computed.")

    return f"""You are a robotics task synthesis expert. Summarise the outcome of
the robot perception and planning pipeline for the user in clear, concise language.

User Request: {query}

Scene Description (from robot perception):
{scene_description}

Target Object : {target_object}
Grasp Pose : {grasp_pose}
Pre-Grasp Pose : {pre_grasp_pose}
Gripper Width : {gripper_width} m
Plan Confidence: {confidence}
Planner Reasoning:
{reasoning}

Write a brief, friendly status report that:
1. Confirms what object was identified and where it is in the scene
2. Describes the planned grasp approach (pre-grasp -> grasp -> lift)
3. States the confidence level and any caveats
4. Notes that the grasp plan has been handed off to the MPC Controller for execution
5. Ends with a short actionable note for the operator
"""

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LANGGRAPH NODE
# ══════════════════════════════════════════════════════════════════════════════
def synthesis_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — synthesise all upstream outputs into a final answer.
    Routing:
    Route.ROBOT_PICKUP -> robot-aware prompt (_build_robot_prompt)
    all other routes -> food/nutrition prompt (_build_food_prompt)
    Args:
        state : OrchestratorState typed dict populated by upstream agents.
    Returns:
        Partial state update dict:
        final_answer : str — polished answer for the user
        messages : list — updated message log
        current_step : int — incremented step counter
    """
    route = state.get("tool", Route.WEB)

    # ── Select prompt based on active route ───────────────────────────────────
    if route == Route.ROBOT_PICKUP:
        prompt = _build_robot_prompt(state)
        fallback = state.get("scene_description") or "Robot grasp plan prepared — awaiting MPC execution."
    else:
        prompt = _build_food_prompt(state)
        fallback = state.get("analysis") or "No answer generated."

    # ── Call LLM ──────────────────────────────────────────────────────────────
    try:
        if client is not None:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
            )
            result = _get_text(response.choices[0].message, fallback=fallback)
            print(f"[DEBUG synthesis_agent] final_answer generated ({len(result)} chars, route={route})")
        else:
            result = fallback
    except Exception as exc:
        print(f"[ERROR synthesis_agent] LLM call failed: {exc}")
        result = fallback

    # ── Update state ──────────────────────────────────────────────────────────
    updated_messages = add_message(
        state=state,
        role=AgentName.SYNTHESIZER,
        content=result,
    )
    return {
        "final_answer": result,
        "messages": updated_messages,
        "current_step": state.get("current_step", 0) + 1,
    }
