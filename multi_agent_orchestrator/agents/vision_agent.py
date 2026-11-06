# agents/vision_agent.py
"""
Vision Agent
============
Uses the shared Qwen2.5-VL-7B-Instruct model (llm/vlm_client.py) for food
detection from video frames or single images.

Why local inference:
 - No dependency on the Siemens LLM API or any external endpoint.
 - Full control over model loading, quantization, and GPU placement.
 - Qwen2.5-VL-7B fits comfortably on a single A100/H100 GPU in bf16.

Model loading has moved to llm/vlm_client.py — this module no longer owns
its own private model/processor instance. robot_perception_agent.py also
imports from that same shared module (for its open-vocabulary YOLO fallback),
so keeping the model in one place avoids loading two separate 7B-parameter
instances into GPU memory.

Input state keys read:
 video_path : str | None — path to a video file (mp4, avi, etc.)
 image_path : str | None — path to a single image (jpg, png, etc.)
 (video_path takes priority if both are present)

Output state keys written:
 detected_foods : str — raw JSON array string of detected food items
 messages : list — updated message log (via add_message)
 current_step : int — incremented step counter

Video handling:
 We sample N evenly-spaced frames with OpenCV and pass them as PIL Images
 directly to the shared VLM client. N_FRAMES is configurable via the
 VISION_N_FRAMES env var (default 6).

Dependencies:
 pip install transformers accelerate qwen-vl-utils pillow opencv-python
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image

from llm.vlm_client import run_vlm_inference, MAX_NEW_TOKENS_GENERAL
from state.schema import AgentName, OrchestratorState, add_message

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
# Number of frames to sample from a video file.
N_FRAMES = int(os.getenv("VISION_N_FRAMES", "6"))

# Max new tokens for the VLM response (JSON array of food items).
# Defaults to the shared client's general-purpose value; override independently
# via VISION_MAX_TOKENS if food detection specifically needs a different cap.
MAX_NEW_TOKENS = int(os.getenv("VISION_MAX_TOKENS", str(MAX_NEW_TOKENS_GENERAL)))

# ── Prompt ─────────────────────────────────────────────────────────────────────
FOOD_DETECTION_PROMPT = """
You are a food recognition expert analyzing a lunch buffet scene.
Carefully examine the image(s) and identify ALL visible food items on the table.
For each food item, return a JSON array using this EXACT format:
[
 {
 "label": "<food name>",
 "cooking_method": "<fried/grilled/boiled/raw/baked/steamed>",
 "estimated_portion": "<small/medium/large>",
 "calorie_level": "<low/medium/high>",
 "box_2d": [y_min, x_min, y_max, x_max]
 }
]
CRITICAL RULES for box_2d:
- Coordinates MUST be in [y_min, x_min, y_max, x_max] order
- ALL values must be normalised between 0 and 1000
- y_min < y_max and x_min < x_max — always
- Each box must TIGHTLY wrap the actual food item
- Do NOT use "point" — always use "box_2d"
- Do NOT overlap boxes unnecessarily
- Cover every dish, plate, drink, and condiment visible
If multiple images are provided (video frames), combine detections across
all frames into a single unified list — do NOT duplicate the same item.
Return ONLY the raw JSON array. No markdown, no explanation, no code fences.
""".strip()

# ── Frame / Image Helpers ──────────────────────────────────────────────────────
def _load_image_as_pil(image_path: str) -> Image.Image:
    """
    Load an image from disk and return it as an RGB PIL Image.
    Args:
        image_path : absolute or relative path to the image file.
    Returns:
        PIL Image in RGB mode.
    Raises:
        FileNotFoundError : if image_path does not exist.
        ValueError : if OpenCV cannot decode the file.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[vision_agent] Image not found: {image_path}")
    if cv2 is not None:
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise ValueError(f"[vision_agent] cv2.imread failed for: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    return Image.open(image_path).convert("RGB")

def _sample_video_frames(video_path: str, n_frames: int = N_FRAMES) -> List[Image.Image]:
    """
    Sample N evenly-spaced frames from a video file as PIL Images.
    Args:
        video_path : path to the video file (mp4, avi, mov, etc.)
        n_frames : number of frames to sample (default: N_FRAMES env var)
    Returns:
        List of PIL Images (RGB), one per sampled frame.
        Returns an empty list if the video cannot be opened.
    """
    if not os.path.exists(video_path):
        logger.error("[vision_agent] Video file not found: %s", video_path)
        return []
    if cv2 is None:
        logger.warning("[vision_agent] cv2 is not installed; video frame sampling unavailable.")
        return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("[vision_agent] cv2.VideoCapture failed to open: %s", video_path)
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        logger.warning("[vision_agent] Video has 0 frames: %s", video_path)
        cap.release()
        return []
    # Evenly-spaced frame indices across the full video duration
    indices = [
        int(i * (total_frames - 1) / max(n_frames - 1, 1))
        for i in range(n_frames)
    ]
    pil_frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            logger.warning("[vision_agent] Could not read frame %d — skipping.", idx)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(rgb))
    cap.release()
    logger.info(
        "[vision_agent] Sampled %d/%d frames from: %s",
        len(pil_frames), n_frames, video_path,
    )
    return pil_frames

# ── JSON Parsing Helper ────────────────────────────────────────────────────────
def _safe_parse_json(raw: str) -> str:
    """
    Validate that the VLM response is a parseable JSON array.
    Strips accidental markdown fences if the model includes them
    despite the prompt instructions.
    Args:
        raw : raw string from the VLM response.
    Returns:
        Compact JSON string of the parsed list, or a JSON-encoded
        error list if parsing fails — downstream agents always receive
        a string that is safe to json.loads().
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            raise ValueError("Top-level value is not a JSON array.")
        logger.info("[vision_agent] JSON parse OK — %d food item(s) detected.", len(parsed))
        return json.dumps(parsed)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("[vision_agent] JSON parse failed: %s\nRaw: %s", exc, raw[:300])
        return json.dumps([{"label": "parse_error", "raw_response": raw[:500]}])

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LANGGRAPH NODE
# ══════════════════════════════════════════════════════════════════════════════
def vision_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — detect food items from a video or image using
    the shared Qwen2.5-VL-7B-Instruct model (llm/vlm_client.py).
    Priority:
    1. state["video_path"] — sample N frames, pass as PIL image sequence
    2. state["image_path"] — load single image, pass directly
    Inference runs entirely on the local GPU — no external API calls.
    Args:
        state : OrchestratorState typed dict.
                Must contain video_path or image_path.
    Returns:
        Partial state update dict:
        detected_foods : str — raw JSON array of food detections
        messages : list — updated message log
        current_step : int — incremented step counter
    """
    video_path = state.get("video_path")
    image_path = state.get("image_path")

    # ── Guard: nothing to process ─────────────────────────────────────────────
    if not video_path and not image_path:
        logger.warning("[vision_agent] No video_path or image_path in state — skipping.")
        fallback = "No image or video provided."
        updated_messages = add_message(
            state=state,
            role=AgentName.VISION,
            content=fallback,
        )
        return {
            "detected_foods": fallback,
            "messages": updated_messages,
            "current_step": state.get("current_step", 0) + 1,
        }

    # ── Load images / frames ──────────────────────────────────────────────────
    pil_images: List[Image.Image] = []
    if video_path:
        logger.info("[vision_agent] Sampling %d frames from video: %s", N_FRAMES, video_path)
        pil_images = _sample_video_frames(video_path, n_frames=N_FRAMES)
        if not pil_images:
            logger.error("[vision_agent] No frames extracted from video — aborting.")
            fallback = "Failed to extract frames from video."
            updated_messages = add_message(
                state=state, role=AgentName.VISION, content=fallback,
            )
            return {
                "detected_foods": fallback,
                "messages": updated_messages,
                "current_step": state.get("current_step", 0) + 1,
            }
        logger.info("[vision_agent] %d frame(s) ready for inference.", len(pil_images))
    else:
        logger.info("[vision_agent] Loading single image: %s", image_path)
        try:
            pil_images = [_load_image_as_pil(image_path)]
        except (FileNotFoundError, ValueError) as exc:
            logger.error("[vision_agent] Image loading failed: %s", exc)
            fallback = f"Image loading error: {exc}"
            updated_messages = add_message(
                state=state, role=AgentName.VISION, content=fallback,
            )
            return {
                "detected_foods": fallback,
                "messages": updated_messages,
                "current_step": state.get("current_step", 0) + 1,
            }

    # ── Run inference via the shared VLM client ───────────────────────────────
    logger.info(
        "[vision_agent] Running local inference via llm/vlm_client.py — %d image(s).",
        len(pil_images),
    )
    try:
        raw_output = run_vlm_inference(
            pil_images=pil_images,
            prompt_text=FOOD_DETECTION_PROMPT,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        logger.info("[vision_agent] Inference complete (%d chars).", len(raw_output))
    except Exception as exc:
        logger.error("[vision_agent] Local inference failed: %s", exc)
        raw_output = json.dumps([{
            "label": "inference_error",
            "cooking_method": "unknown",
            "estimated_portion": "unknown",
            "calorie_level": "unknown",
            "box_2d": [0, 0, 0, 0],
            "error": str(exc),
        }])

    # ── Validate and normalise JSON ───────────────────────────────────────────
    detected_foods = _safe_parse_json(raw_output)
    logger.info("[vision_agent] detected_foods written to state.")
    print(f"[vision_agent] Detected foods:\n{detected_foods}")

    # ── Update LangGraph state ────────────────────────────────────────────────
    updated_messages = add_message(
        state=state,
        role=AgentName.VISION,
        content=detected_foods,
    )
    return {
        "detected_foods": detected_foods,
        "messages": updated_messages,
        "current_step": state.get("current_step", 0) + 1,
    }
