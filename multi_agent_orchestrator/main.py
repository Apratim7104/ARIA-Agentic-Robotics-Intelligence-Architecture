# main.py
"""
ARIA Pipeline Entry Point
=========================
CLI wrapper around the ARIA multi-agent LangGraph pipeline.

Usage:
 python main.py --query "What food is on the buffet?" --video path/to/video.mp4
 python main.py --query "Pick up the red cup" --image path/to/scene.jpg
 python main.py --query "What is the Siemens MindSphere platform?"

Env vars:
 LOG_LEVEL : Python logging level (default: INFO)
"""
import argparse
import logging
import uuid
import os

from graph.orchestrator import run_pipeline # ── FIX: import run_pipeline from orchestrator,
                                            # not 'app' from 'graph.builder'
from annotator import annotate_frame

# ── Logging ───────────────────────────────────────────────────────────────────
# ── FIX: level is configurable via LOG_LEVEL env var for production use
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("main")

def main():
    parser = argparse.ArgumentParser(description="Run the ARIA multi-agent pipeline.")
    parser.add_argument("--query", required=True, help="User query / instruction")
    parser.add_argument("--video", required=False, default=None, help="Path to video file")
    parser.add_argument("--image", required=False, default=None, help="Path to image file")
    args = parser.parse_args()

    # ── Build initial state ───────────────────────────────────────────────────
    # ── FIX: use "user_input" (canonical key) instead of "query"
    # ── FIX: add session_id — documented as required by run_pipeline()
    initial_state = {
        "user_input": args.query, # canonical key read by all agents
        "session_id": str(uuid.uuid4()), # unique session identifier
        "video_path": args.video,
        "image_path": args.image,
        "retry_count": 0,
    }

    logger.info(
        "Starting ARIA pipeline | session=%s | query=%s",
        initial_state["session_id"],
        args.query[:80],
    )

    # ── Run the multi-agent graph ─────────────────────────────────────────────
    result = run_pipeline(initial_state)

    # ── Print all state keys for debugging ───────────────────────────────────
    print("\n=== ALL STATE KEYS AND VALUES ===")
    for k, v in result.items():
        val_preview = str(v)[:200] if v is not None else repr(v)
        print(f" [{k}] => {val_preview}")

    # ── Annotate frame if video + detections are available ───────────────────
    if result.get("detected_foods") and result.get("video_path"):
        logger.info("Generating annotated frame from detected foods...")
        annotated_path = annotate_frame(
            video_path = result["video_path"],
            detected_foods_json = result["detected_foods"],
            output_path = "annotated_frame.jpg",
        )
        if annotated_path:
            print(f"\n Annotated frame saved -> {annotated_path}")
        else:
            print("\n⚠ Annotated frame could not be generated.")
    else:
        logger.info("Skipping annotation — no video path or detected foods in state.")

    # ── Print final answer ────────────────────────────────────────────────────
    print("\n=== FINAL ANSWER ===\n")
    print(
        result.get("final_answer")
        or result.get("analysis")
        or result.get("synthesis")
        or "No answer generated."
    )

if __name__ == "__main__":
    main()
