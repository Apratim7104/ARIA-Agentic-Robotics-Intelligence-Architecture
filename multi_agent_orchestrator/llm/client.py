# llm/client.py
"""
Siemens LLM API Client
======================
Uses the OpenAI Python SDK pointed at the Siemens LLM endpoint.
The Siemens API is OpenAI-compatible — only the base_url and api_key differ.

Environment variables required (.env):
 SIEMENS_LLM_API_KEY : your Siemens Bearer token
 SIEMENS_LLM_BASE_URL : Siemens LLM endpoint base URL
 SIEMENS_LLM_MODEL : model name (e.g. "qwen-3.6-27b") [optional]
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("SIEMENS_LLM_MODEL", "qwen-3.6-27b")

# ── Siemens OpenAI-compatible client ──────────────────────────────────────────
try:
    from openai import OpenAI
    client = OpenAI(
        api_key = os.getenv("SIEMENS_LLM_API_KEY") or "placeholder",
        base_url = os.getenv("SIEMENS_LLM_BASE_URL"),
    )
except Exception:
    client = None

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def system_user(system: str, user: str) -> List[Dict[str, str]]:
    """
    Format a system + user string pair into the OpenAI messages list format.
    Args:
        system : system prompt (agent instructions / persona)
        user : user prompt (scene data, query, context)
    Returns:
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    Usage:
        messages = system_user(ROBOT_PLANNER_SYSTEM, user_prompt)
        result = call_llm_json(messages, temperature=0.05)
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

# ══════════════════════════════════════════════════════════════════════════════
# CORE API CALL
# ══════════════════════════════════════════════════════════════════════════════
def call_llm_json(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    caller: str = "unknown",
    model: str = LLM_MODEL,
    max_tokens: int = 1024, # FIX: now a real, overridable parameter
) -> Dict[str, Any]:
    """
    Call the Siemens LLM API and return the response parsed as a Python dict.
    The LLM is instructed (via the system prompt) to return strict JSON.
    This function handles:
    • The actual API call via the OpenAI SDK
    • Stripping markdown code fences if the LLM wraps JSON in ```json...```
    • JSON parsing with clear error messages on failure
    """
    if client is None:
        raise RuntimeError("OpenAI client is not available (openai library not installed)")

    logger.info(
        f"[llm.client] [{caller}] → Siemens LLM | "
        f"model={model} | temperature={temperature} | "
        f"messages={len(messages)} | max_tokens={max_tokens}"
    )

    # ── THE actual Siemens LLM API call ───────────────────────────────────────
    response = client.chat.completions.create(
        model = model,
        messages = messages, # type: ignore[arg-type]
        temperature = temperature,
        max_tokens = max_tokens, # FIX: uses the parameter instead of a hardcoded literal
    )
    raw_content = response.choices[0].message.content or ""
    logger.info(
        f"[llm.client] [{caller}] ← Response received "
        f"({len(raw_content)} chars) | "
        f"finish_reason={response.choices[0].finish_reason}"
    )

    # ── Strip markdown fences if LLM wraps JSON in ```json ... ``` ────────────
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[cleaned.index("\n") + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

    # ── Parse JSON ────────────────────────────────────────────────────────────
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            f"[llm.client] [{caller}] JSON parse failed: {exc}\n"
            f"Raw response:\n{raw_content}"
        )
        raise ValueError(
            f"LLM response is not valid JSON.\n"
            f"Parse error : {exc}\n"
            f"Raw content : {raw_content[:300]}"
        ) from exc

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL: plain text call (for non-JSON agents)
# ══════════════════════════════════════════════════════════════════════════════
def call_llm_text(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    caller: str = "unknown",
    model: str = LLM_MODEL,
    max_tokens: int = 1024, # FIX: now a real, overridable parameter
) -> str:
    """
    Call the Siemens LLM API and return the raw text response.
    Use this for agents that return free-form text instead of JSON
    (e.g. a scene narration agent, a user-facing explanation agent).
    """
    if client is None:
        raise RuntimeError("OpenAI client is not available (openai library not installed)")

    logger.info(
        f"[llm.client] [{caller}] → Siemens LLM (text) | "
        f"model={model} | temperature={temperature} | max_tokens={max_tokens}"
    )
    response = client.chat.completions.create(
        model = model,
        messages = messages, # type: ignore[arg-type]
        temperature = temperature,
        max_tokens = max_tokens, # FIX: uses the parameter instead of a hardcoded literal
    )
    raw_content = response.choices[0].message.content or ""
    logger.info(
        f"[llm.client] [{caller}] ← Text response "
        f"({len(raw_content)} chars)"
    )
    return raw_content
