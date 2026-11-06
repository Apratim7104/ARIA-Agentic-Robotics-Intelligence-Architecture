import os
from dotenv import load_dotenv
load_dotenv()
import cv2
import numpy as np
from agents.robot_perception_agent import robot_perception_node

VIDEO_PATH = "/data/apratim/multi_agent_orchestrator/6760617-uhd_2560_1440_25fps.mp4"
N_FRAMES = 5
PALETTE = [
    (0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 0, 255),
    (128, 255, 0), (255, 128, 0),
]

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
indices = [int(i * (total - 1) / max(N_FRAMES - 1, 1)) for i in range(N_FRAMES)] if total > 0 else []
cap.release()

for idx in indices:
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Frame {idx}: could not read — skipping")
        continue
    frame_path = f"/data/apratim/multi_agent_orchestrator/tmp_frame_{idx}.jpg"
    cv2.imwrite(frame_path, frame)
    state = {
        "image_path": frame_path,
        "target_object": "", # no specific target requested — just report everything found
        "current_step": 0,
        "messages": [],
    }
    result = robot_perception_node(state)
    objs = result["detected_objects"]
    print(f"\nFrame {idx}/{total}: {len(objs)} object(s) detected")
    blended = frame.copy()
    for i, obj in enumerate(objs):
        color = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = [int(v) for v in obj["bbox_xyxy"]]
        mask = obj["mask"]
        print(
            f" [{i}] {obj['label']:20s} conf={obj['confidence']:.2f} "
            f"detector_src={obj['detector_src']:6s} centroid_src={obj['centroid_src']:6s} "
            f"depth={obj['depth_m']:.3f}m mask={'yes' if mask is not None else 'NO'}"
        )
        if mask is not None:
            colored_layer = np.zeros_like(frame)
            colored_layer[mask > 0] = color
            blended = cv2.addWeighted(blended, 1.0, colored_layer, 0.45, 0)
        cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)
        cv2.putText(blended, f"{obj['label']} {obj['confidence']:.2f}",
                    (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    out_path = f"robot_perception_test_frame_{idx}.jpg"
    cv2.imwrite(out_path, blended)
    print(f" Saved {out_path}")

print("\nDone — check robot_perception_test_frame_*.jpg for all detections + masks.")
