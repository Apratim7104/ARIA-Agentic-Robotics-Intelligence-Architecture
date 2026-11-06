# llm/vlm_client.py
"""
Shared VLM Client — Qwen2.5-VL-7B-Instruct
===========================================
Single source of truth for the locally-loaded Qwen2.5-VL model and processor.

Why this module exists:
 Previously, vision_agent.py loaded its own private module-level Qwen2.5-VL
 instance. Now that robot_perception_agent.py also needs VLM access (as an
 open-vocabulary fallback when YOLO's fixed 80-class vocabulary misses the
 target object), that model must live in ONE shared place — otherwise two
 separate 7B-parameter model instances would get loaded into GPU memory,
 doubling VRAM usage for no benefit.
 vision_agent.py should import `model`, `processor`, and
 `run_vlm_inference()` from here instead of loading its own copy.

Two public entry points:
 run_vlm_inference(pil_images, prompt_text) -> str
 Generic — pass any list of PIL images + any text prompt, get the raw
 decoded text response back. Used by vision_agent.py for food detection.
 query_vlm_for_object_box(pil_image, target_object) -> dict | None
 Specific — asks the VLM to locate ONE named object in ONE image and
 return its bounding box. Used by robot_perception_agent.py as an
 open-vocabulary fallback when YOLO can't find the target_object.
 Returns None if the VLM reports the object isn't visible, or if
 parsing fails.

Model:
 Controlled via the VISION_MODEL_ID env var. Default:
 "Qwen/Qwen2.5-VL-7B-Instruct". Loaded ONCE at module import time
 (singleton pattern) — every caller across the whole process shares it.

cuDNN workaround (see agents/vision_agent.py history for full context):
 This host has a cuDNN 9 sublibrary version mismatch — the pip-installed
 nvidia-cudnn-cu13 wheel doesn't ship libcudnn_engines_tensor_ir.so /
 libcudnn_ext.so at all, so cuDNN's backend/graph API falls through to a
 mismatched system-wide cuDNN install and fails a version-consistency
 check at init (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH). Disabling
 cuDNN sidesteps this entirely, with negligible cost here since the ViT
 vision tower is attention/matmul-bound (cuBLAS, unaffected) — only a
 single small patch-embedding convolution is routed through cuDNN.
 This MUST run before the model is loaded below, and now lives here
 (not in vision_agent.py) since this is the new single load site.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
import torch

torch.backends.cudnn.enabled = False

from PIL import Image

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
MAX_NEW_TOKENS_GENERAL = int(os.getenv("VISION_MAX_TOKENS", "2000"))
MAX_NEW_TOKENS_BOX_QUERY = int(os.getenv("VISION_BOX_QUERY_MAX_TOKENS", "512"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model Singleton ────────────────────────────────────────────────────────────
model = None
processor = None

def _get_vlm_model_and_processor():
    global model, processor
    if model is None or processor is None:
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            logger.info("[vlm_client] Loading model: %s on device: %s", VISION_MODEL_ID, DEVICE)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                VISION_MODEL_ID,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model.eval()
            processor = AutoProcessor.from_pretrained(VISION_MODEL_ID)
            logger.info("[vlm_client] Model and processor loaded successfully.")
        except Exception as e:
            logger.warning("[vlm_client] VLM model loading deferred or failed: %s", e)
            raise
    return model, processor

# ── Prompt: single-object box query ────────────────────────────────────────────
BOX_QUERY_PROMPT_TEMPLATE = """You are an object localization expert.
TARGET OBJECT: "{target_object}"
Examine the image carefully and determine whether an object matching this
description is visible anywhere in the scene.
Return ONLY raw JSON (no markdown, no explanation) in this EXACT format:
{{
 "found": <true|false>,
 "box_2d": [y_min, x_min, y_max, x_max],
 "confidence": <0.0-1.0>
}}
RULES:
- Coordinates in box_2d MUST be normalised between 0 and 1000, in
 [y_min, x_min, y_max, x_max] order.
- y_min < y_max and x_min < x_max — always.
- The box must TIGHTLY wrap only the described object — not the whole scene,
 not a group of objects, not a nearby similar-looking object.
- If the target object is NOT visible anywhere in the image, set
 "found": false, "box_2d": [0, 0, 0, 0], "confidence": 0.0.
- Return ONLY the JSON object. No code fences, no extra text.
""".strip()

# ── Message / Inference Helpers ────────────────────────────────────────────────
def _build_messages(pil_images: List[Image.Image], prompt_text: str) -> List[Dict[str, Any]]:
    """
    Build the Qwen2.5-VL chat message list from PIL images + a text prompt.
    Args:
        pil_images : list of PIL Images to include in the message.
        prompt_text : the instruction text to append after the image(s).
    Returns:
        List of message dicts in Qwen2.5-VL chat format.
    """
    image_blocks = [{"type": "image", "image": img} for img in pil_images]
    image_blocks.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": image_blocks}]

def run_vlm_inference(
    pil_images: List[Image.Image],
    prompt_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS_GENERAL,
) -> str:
    """
    Generic VLM inference — pass any images + any prompt, get raw decoded text.
    This is the shared low-level call. vision_agent.py's food-detection flow
    and query_vlm_for_object_box() below both build on top of this.
    """
    vlm_model, vlm_processor = _get_vlm_model_and_processor()
    from qwen_vl_utils import process_vision_info

    messages = _build_messages(pil_images, prompt_text)
    text_prompt = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.inference_mode():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False, # greedy decoding for structured JSON output
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = vlm_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return raw_output

def _safe_parse_json_object(raw: str, caller: str = "vlm_client") -> Optional[Dict[str, Any]]:
    """
    Validate that a VLM response is a parseable JSON object (dict), stripping
    accidental markdown fences.
    Args:
        raw : raw string from the VLM response.
        caller : label for logging context.
    Returns:
        Parsed dict, or None if parsing fails or the top-level value isn't a dict.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Top-level value is not a JSON object.")
        return parsed
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(
            "[vlm_client] [%s] JSON parse failed: %s\nRaw: %s",
            caller, exc, raw[:300],
        )
        return None

# ── Public: single-object box query ────────────────────────────────────────────
def query_vlm_for_object_box(
    pil_image: Image.Image,
    target_object: str,
) -> Optional[Dict[str, Any]]:
    """
    Ask the VLM to locate ONE named object in ONE image.
    Used by robot_perception_agent.py as an open-vocabulary fallback when
    YOLO's fixed 80-class vocabulary fails to find state["target_object"].
    Args:
        pil_image : single RGB PIL Image (the full scene frame).
        target_object : natural-language description of the object to find,
                        e.g. "red mug", "salami", "the blue bottle".
    Returns:
        {
            "found": bool,
            "box_2d": [y_min, x_min, y_max, x_max] — normalised 0-1000,
            "confidence": float
        }
        or None if:
        - target_object is empty/blank,
        - the VLM response fails to parse as JSON,
        - the VLM reports the object is not visible ("found": false).
    """
    if not target_object or not target_object.strip():
        logger.warning("[vlm_client] query_vlm_for_object_box called with empty target_object.")
        return None

    prompt = BOX_QUERY_PROMPT_TEMPLATE.format(target_object=target_object.strip())
    logger.info(
        "[vlm_client] Querying VLM for target_object='%s' (open-vocabulary fallback)",
        target_object,
    )
    try:
        raw_output = run_vlm_inference(
            pil_images=[pil_image],
            prompt_text=prompt,
            max_new_tokens=MAX_NEW_TOKENS_BOX_QUERY,
        )
    except Exception as exc:
        logger.error("[vlm_client] VLM inference failed for target_object='%s': %s", target_object, exc)
        return None

    result = _safe_parse_json_object(raw_output, caller="query_vlm_for_object_box")
    if result is None:
        return None
    if not result.get("found", False):
        logger.info("[vlm_client] VLM reports target_object='%s' NOT visible.", target_object)
        return None

    box_2d = result.get("box_2d")
    if not box_2d or len(box_2d) != 4:
        logger.warning("[vlm_client] VLM returned found=true but malformed box_2d: %s", box_2d)
        return None

    logger.info(
        "[vlm_client] VLM found target_object='%s' | box_2d=%s | confidence=%.2f",
        target_object, box_2d, result.get("confidence", 0.0),
    )
    return {
        "found": True,
        "box_2d": box_2d,
        "confidence": float(result.get("confidence", 0.5)),
    }
