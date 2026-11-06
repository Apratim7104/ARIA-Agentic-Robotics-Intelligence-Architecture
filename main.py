"""
main.py - ARIA End-to-End Autonomous Robotic Manipulation System
================================================================
Master entrypoint unifying all 4 layers:
  Layer 1: Multi-Agent Orchestrator (LLM Planner, YOLOv8/VLM Perception, SAM2 Segmentation, Robot Planner, Critic, Synthesizer)
  Layer 2: Spatial Reasoning Layer (MuJoCo Scene Grounding, Camera Deprojection, Continuous Distance Tracking)
  Layer 3: Context-Conditioned Dynamics Model (CCDM v3.0, Dynamic Constraints & NMPC Cost Weights)
  Layer 4: ARIA Franka 7-DOF NMPC Controller (CasADi + IPOPT Closed-Loop Torque Control in MuJoCo Physics)

Usage:
  python main.py --prompt "Pick up the red cube and place it on the target spot"
  python main.py --prompt "Pick up the red cube" --gui
  python main.py --prompt "Pick up the red cube" --agent-only
  python main.py --verify
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

orch_dir = ROOT_DIR / "multi_agent_orchestrator"
if str(orch_dir) not in sys.path:
    sys.path.insert(0, str(orch_dir))


def run_agent_only(prompt: str, image_path: str | None = None) -> Dict[str, Any]:
    """Runs only Layer 1: Multi-Agent Orchestrator."""
    print("=" * 80)
    print("           ARIA LAYER 1: MULTI-AGENT ORCHESTRATOR STANDALONE")
    print("=" * 80)
    print(f" User Prompt: \"{prompt}\"")
    if image_path:
        print(f" Image Path : {image_path}")

    from multi_agent_orchestrator.graph.orchestrator import run_pipeline

    initial_state = {
        "user_input": prompt,
        "session_id": f"aria_agent_{int(time.time())}",
        "image_path": image_path,
        "depth_path": None,
    }

    t0 = time.perf_counter()
    final_state = run_pipeline(initial_state)
    elapsed = time.perf_counter() - t0

    intent = final_state.get("intent") or {}
    route = intent.get("route", "unknown")
    print("-" * 80)
    print(f" Pipeline completed in {elapsed:.3f}s")
    print(f" Route Selected    : {route}")
    print(f" Target Object     : {intent.get('target_object', 'None')}")

    det_objs = final_state.get("detected_objects", [])
    print(f" Detected Objects  : {len(det_objs)}")
    for obj in det_objs:
        label = obj.get("label", "unknown")
        src = obj.get("detector_src", "vision")
        centroid = obj.get("centroid_3d", [])
        print(f"   - [{src}] '{label}': 3D centroid = {centroid}")

    handoff = final_state.get("robot_handoff")
    if handoff:
        grasps = handoff.get("grasp_poses", [])
        print(f" Controller Handoff: {len(grasps)} 6-DOF grasp poses emitted")
        for gp in grasps:
            phase = gp.get("phase", "waypoint")
            pos = gp.get("position", {})
            print(f"   - Phase '{phase}': pos=({pos.get('x',0):.3f}, {pos.get('y',0):.3f}, {pos.get('z',0):.3f})")

    synth = final_state.get("synthesis", "")
    if synth:
        print("\n Synthesizer Summary:")
        print(f"   {synth[:300]}...")

    critique = final_state.get("critique")
    if critique:
        print(f"\n Critic Approved   : {critique.get('passed', False)}")

    print("=" * 80)
    return final_state


def main():
    parser = argparse.ArgumentParser(
        description="ARIA: Autonomous Robotic Manipulation System (Multi-Agent + Spatial + CCDM + NMPC)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Pick up the red cube and place it on the target spot",
        help="Natural language task prompt describing the manipulation intent."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        dest="prompt",
        help="Alias for --prompt."
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default=None,
        help="Path to scene image (optional, simulator renders camera frame automatically)."
    )
    parser.add_argument(
        "--steps-per-phase",
        type=int,
        default=25,
        help="Number of control steps per phase (default 25)."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch interactive passive MuJoCo 3D viewer window."
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start immediately without waiting for user spacebar in GUI."
    )
    parser.add_argument(
        "--agent-only",
        action="store_true",
        help="Run only Layer 1 Multi-Agent Orchestrator without physical simulation."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run full deployment verification audit suite."
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record simulation video to GIF."
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="simulation_video.gif",
        help="Output path for recorded simulation GIF."
    )

    args = parser.parse_args()

    if args.agent_only:
        run_agent_only(prompt=args.prompt, image_path=args.image)
        return

    from verify_deployment import run_deployment_verification

    run_deployment_verification(
        user_prompt=args.prompt,
        steps_per_phase=args.steps_per_phase,
        render_gui=args.gui,
        auto_start=args.auto_start,
        record_video=args.record_video,
        video_path=Path(args.video_path)
    )


if __name__ == "__main__":
    main()
