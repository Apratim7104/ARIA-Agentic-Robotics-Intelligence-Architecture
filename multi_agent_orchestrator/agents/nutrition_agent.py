# agents/nutrition_agent.py
"""
Nutrition Agent
===============
Provides personalised dietary recommendations based on food items
detected by the Vision Agent and any additional research context.

NOTE: This agent is NOT currently wired into orchestrator.py.
It is called as part of the VIDEO_FOOD route but the orchestrator
routes VIDEO_FOOD directly to vision_agent -> synthesizer.
To activate it, add it as a node between vision_agent and synthesizer
for the VIDEO_FOOD route in orchestrator.py.

State keys READ:
 user_input : str — canonical user query key
 detected_foods : str | None — JSON array of food items from vision agent
 research_data : str | None — web research context from research agent
 health_profile : dict | None — user dietary constraints

State keys WRITTEN:
 analysis : str — structured nutrition recommendations
 messages : list — updated message log
 current_step : int — incremented step counter
"""
from __future__ import annotations
import os
import logging
from typing import Dict, Any
from llm.client import client
from state.schema import OrchestratorState, AgentName, add_message

logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "qwen-3.6-27b")

def _get_text(msg: Any, fallback: str = "No analysis generated.") -> str:
    """
    Safely extract text from a ChatCompletion message object.
    Handles models with a separate 'reasoning' field (e.g. Qwen-3).
    """
    return msg.content or getattr(msg, "reasoning", None) or fallback

def nutrition_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — generate personalised nutrition recommendations.

    Args:
        state : OrchestratorState typed dict.

    Returns:
        Partial state update dict:
        analysis : str — structured nutrition advice
        messages : list — updated message log
        current_step : int — incremented step counter
    """
    # ── FIX: use canonical state keys ─────────────────────────────────────────
    user_input = state.get("user_input") or state.get("user_query") or ""
    detected_foods = state.get("detected_foods") or "No food items detected."
    research_data = state.get("research_data") or "" # ── FIX: was "research"
    health_profile = state.get("health_profile") or {}

    logger.info(
        "[nutrition_agent] Building nutrition analysis | query=%s",
        user_input[:60],
    )

    # ── Build health context string ───────────────────────────────────────────
    health_context = ""
    if health_profile:
        health_context = "\n".join(
            f" {k}: {v}" for k, v in health_profile.items()
        )
        health_context = f"\nUser Health Profile:\n{health_context}\n"

    prompt = f"""You are a certified nutritionist and dietitian.

User Query: {user_input}
{health_context}
Foods available at the event:
{detected_foods}

Additional research context:
{research_data}

Based on the user's personal goals and the available food items, provide:
1. Which foods they SHOULD eat and why
2. Which foods they should AVOID or limit and why
3. Recommended portion sizes
4. Estimated calories for the recommended meal
5. Any practical tips for eating at this event

Be specific, friendly, and actionable. Reference the actual food items detected.
"""

    try:
        if client is not None:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
            )
            result = _get_text(response.choices[0].message)
            logger.info(
                "[nutrition_agent] Analysis generated (%d chars).", len(result)
            )
        else:
            result = "Nutrition analysis unavailable: client not initialized."
    except Exception as exc:
        logger.error("[nutrition_agent] LLM call failed: %s", exc)
        result = f"Nutrition analysis unavailable: {exc}"

    # ── Update state ──────────────────────────────────────────────────────────
    updated_messages = add_message(
        state=state,
        role=AgentName.NUTRITION,
        content=result,
    )

    return {
        "analysis": result,
        "messages": updated_messages,
        "current_step": state.get("current_step", 0) + 1,
    }
