# agents/critic.py
import logging
from typing import Dict, Any
from state.schema import (
    OrchestratorState, CriticFeedback,
    add_message, AgentName,
)
from llm.client import call_llm_json, system_user

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_THRESHOLD = 7

CRITIC_SYSTEM = """You are the Critic Agent in a multi-agent AI quality review system.

Evaluate the final answer on:
 1. Completeness - Does it fully answer the user's question?
 2. Accuracy - Is the information correct and well-supported?
 3. Clarity - Is it well-structured and easy to understand?
 4. Relevance - Does it stay on topic?
 5. Tone - Is it professional and helpful?

Score 1-10 (10=perfect, 7=good, 5=acceptable, 3=poor, 1=fail).

OUTPUT FORMAT (strict JSON, no markdown):
{
 "approved" : <true|false>,
 "score" : <integer 1-10>,
 "issues" : ["<issue1>"],
 "suggestions" : ["<suggestion1>"],
 "retry_reason": "<one sentence or empty string>"
}

RULES:
- approved=true if score >= 7
- approved=false if score < 7 (give clear, actionable suggestions)
- If retry_count >= 2, ALWAYS set approved=true
"""

def critic_node(state: OrchestratorState) -> Dict[str, Any]:
    step = state.get("current_step", 0) + 1
    final_answer = state.get("final_answer", "")
    # FIX: main.py only ever sets "user_input" — "user_query" is never populated,
    # so the critic was always reviewing against a blank question.
    user_query = state.get("user_input") or state.get("user_query", "")
    retry_count = state.get("retry_count", 0)

    logger.info(f"[critic] step={step} retry={retry_count} answer_len={len(final_answer)}")

    # Force approve if max retries reached
    if retry_count >= MAX_RETRIES:
        logger.warning(f"[critic] Max retries reached -> force approve")
        return _build(state, _force_approve(step), step, False, retry_count)

    # Force reject if answer is empty
    if len(final_answer.strip()) < 50:
        fb: CriticFeedback = {
            "approved": False, "score": 2,
            "issues": ["Answer is too short or empty"],
            "suggestions": ["Generate a complete, detailed answer"],
            "retry_reason": "Answer is too short to be useful",
            "reviewed_at_step": step,
        }
        return _build(state, fb, step, True, retry_count + 1)

    prompt = (
        f"USER QUESTION: {user_query}\n\n"
        f"FINAL ANSWER TO REVIEW:\n{final_answer}\n\n"
        f"Current retry count: {retry_count} / {MAX_RETRIES}\n"
        "Evaluate and return your JSON review."
    )

    try:
        result = call_llm_json(
            messages=system_user(CRITIC_SYSTEM, prompt),
            temperature=0.1,
            max_tokens=2048, # RESTORED — client.py now accepts this properly
            caller=AgentName.CRITIC,
        )
    except Exception as exc:
        logger.error(f"[critic] LLM failed: {exc} -> force approve")
        return _build(state, _force_approve(step), step, False, retry_count)

    score = int(result.get("score", 7))
    approved = score >= RETRY_THRESHOLD
    fb: CriticFeedback = {
        "approved": approved,
        "score": score,
        "issues": result.get("issues", []),
        "suggestions": result.get("suggestions", []),
        "retry_reason": result.get("retry_reason", ""),
        "reviewed_at_step": step,
    }

    needs_retry = not approved
    new_retry = retry_count + 1 if needs_retry else retry_count
    logger.info(f"[critic] -> approved={approved} score={score} retry={needs_retry}")
    return _build(state, fb, step, needs_retry, new_retry)

def _build(state, feedback, step, needs_retry, retry_count):
    status = (
        f"APPROVED (score={feedback['score']}/10)"
        if feedback["approved"]
        else f"REJECTED (score={feedback['score']}/10) - {feedback['retry_reason']}"
    )
    updated = add_message(
        state={**state, "current_step": step},
        role=AgentName.CRITIC,
        content=f"Quality review: {status}",
    )
    return {
        "critic_feedback": feedback,
        "needs_retry": needs_retry,
        "retry_count": retry_count,
        "current_step": step,
        "messages": updated,
    }

def _force_approve(step: int) -> CriticFeedback:
    return CriticFeedback(
        approved=True, score=6, issues=[], suggestions=[],
        retry_reason="", reviewed_at_step=step,
    )
