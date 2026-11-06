import os
from dotenv import load_dotenv
load_dotenv()
import cv2
import numpy as np
import torch

# Fresh process — cuDNN must be disabled again here, it does not persist across runs.
torch.backends.cudnn.enabled = False

from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

VIDEO_PATH = "/data/apratim/multi_agent_orchestrator/6760617-uhd_2560_1440_25fps.mp4"
N_FRAMES = 5

SAM2_CONFIG = os.getenv("SAM2_CONFIG", "sam2_hiera_small.yaml")
SAM2_CKPT = os.getenv("SAM2_CHECKPOINT", "checkpoints/sam2_hiera_small.pt")
print(f"Using SAM2_CONFIG={SAM2_CONFIG} SAM2_CHECKPOINT={SAM2_CKPT}")
if os.path.exists(SAM2_CKPT):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO("yolov8n.pt")
    predictor = SAM2ImagePredictor(build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device))
else:
    print(f"Checkpoint NOT FOUND at: {SAM2_CKPT}")

# Distinct BGR colors cycled per object so overlapping/adjacent masks stay
# visually separable on the same frame.
PALETTE = [
    (0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 0, 255),
    (128, 255, 0), (255, 128, 0),
]

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
indices = [int(i * (total - 1) / max(N_FRAMES - 1, 1)) for i in range(N_FRAMES)] if total > 0 else []

for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {idx}: could not read — skipping")
        continue
    results = yolo(frame, conf=0.35, verbose=False)
    boxes = results[0].boxes
    n_boxes = len(boxes)
    print(f"\nFrame {idx}/{total}: {n_boxes} YOLO detection(s)")
    if n_boxes == 0:
        continue
    # Encode the image once per frame, then run SAM2 on every box in it —
    # avoids re-encoding the same frame N times (SAM2's image encoder is the
    # expensive step; the box-conditioned mask decode is cheap per call).
    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    blended = frame.copy()
    for i in range(n_boxes):
        xyxy = boxes.xyxy[i].tolist()
        conf = float(boxes.conf[i])
        label = yolo.names[int(boxes.cls[i])]
        color = PALETTE[i % len(PALETTE)]
        masks, scores, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=np.array([xyxy], dtype=np.float32),
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8)
        print(f" [{i}] {label:20s} conf={conf:.2f} bbox={[round(v,1) for v in xyxy]} "
              f"mask_px={mask.sum()} sam2_score={scores[0]:.3f}")
        # Alpha-blend this object's mask into the running overlay in its own color
        colored_layer = np.zeros_like(frame)
        colored_layer[mask > 0] = color
        blended = cv2.addWeighted(blended, 1.0, colored_layer, 0.45, 0)
        x1, y1 = int(xyxy[0]), int(xyxy[1])
        cv2.putText(blended, f"{label} {conf:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    out_path = f"yolo_sam2_test_frame_{idx}.jpg"
    cv2.imwrite(out_path, blended)
    print(f" Saved {out_path} ({n_boxes} object(s) masked)")

cap.release()
print("\nDone — check yolo_sam2_test_frame_*.jpg for visual confirmation of ALL detected objects.")
