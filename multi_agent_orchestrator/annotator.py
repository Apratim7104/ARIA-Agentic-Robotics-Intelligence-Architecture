# annotator.py
try:
    import cv2
except ImportError:
    cv2 = None
import json
import os

def annotate_frame(
    video_path: str,
    detected_foods_json: str,
    output_path: str = "annotated_frame.jpg"
):
    """
    Extracts a representative frame from the video and draws
    properly scaled bounding boxes using Gemini's box_2d coordinates.
    Gemini returns coordinates in a 0-1000 normalized grid.
    This function scales them to the actual frame resolution.

    Args:
        video_path: Path to the input video file.
        detected_foods_json: JSON string from Gemini detection tool.
        output_path: Where to save the annotated image.

    Returns:
        output_path if successful, None otherwise.
    """
    if cv2 is None:
        print("[annotator] Warning: cv2 not installed, cannot annotate frame.")
        return None
    # ── 1. Parse detected foods ───────────────────────────────────────────────
    clean = detected_foods_json.strip()
    # Strip markdown code fences if present (safety fallback)
    if clean.startswith("```"):
        lines = clean.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        clean = "\n".join(lines).strip()

    # Validate it looks like JSON
    if not clean.startswith("["):
        print(f"[annotator] ⚠ Unexpected response format, trying to extract JSON...")
        start = clean.find("[")
        end = clean.rfind("]") + 1
        if start != -1 and end > start:
            clean = clean[start:end]
        else:
            print("[annotator] Could not find JSON array in response.")
            return None

    foods = json.loads(clean.strip())

    # ── 2. Extract a representative frame ────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[annotator] Cannot open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = max(0, int(total_frames * 0.20))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[annotator] Could not read frame from video.")
        return None

    h, w = frame.shape[:2]
    print(f"[annotator] Frame size: {w}x{h}")

    # ── 3. Colour map per calorie level ──────────────────────────────────────
    calorie_colours = {
        "low": (0, 200, 0),       # green
        "medium": (0, 165, 255),   # orange
        "high": (0, 0, 220),       # red
    }
    default_colour = (200, 200, 200)

    # ── 4. Draw bounding boxes ────────────────────────────────────────────────
    for item in foods:
        label = item.get("label", "unknown")
        cal_lvl = item.get("calorie_level", "medium").lower()
        colour = calorie_colours.get(cal_lvl, default_colour)
        method = item.get("cooking_method", "")
        portion = item.get("estimated_portion", "")

        # ── Prefer box_2d, fall back to point ────────────────────────────────
        if "box_2d" in item:
            # Gemini box_2d = [y_min, x_min, y_max, x_max] in 0-1000 range
            y_min_n, x_min_n, y_max_n, x_max_n = item["box_2d"]
            # Scale from 0-1000 normalized → actual pixel coordinates
            x1 = int((x_min_n / 1000.0) * w)
            y1 = int((y_min_n / 1000.0) * h)
            x2 = int((x_max_n / 1000.0) * w)
            y2 = int((y_max_n / 1000.0) * h)
        elif "point" in item:
            # Fallback: point also in 0-1000 normalized range
            # Build a small box around the point
            px = int((item["point"][0] / 1000.0) * w)
            py = int((item["point"][1] / 1000.0) * h)
            box_half_w = int(w * 0.08) # ~8% of frame width
            box_half_h = int(h * 0.08) # ~8% of frame height
            x1 = max(0, px - box_half_w)
            y1 = max(0, py - box_half_h)
            x2 = min(w, px + box_half_w)
            y2 = min(h, py + box_half_h)
        else:
            print(f"[annotator] ⚠ No box_2d or point for '{label}', skipping.")
            continue

        # ── Draw bounding box (thick outer + thin inner) ──────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3, cv2.LINE_AA)
        cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2),
                      (255, 255, 255), 1, cv2.LINE_AA)

        # ── Label background ──────────────────────────────────────────────────
        display_text = f"{label} [{cal_lvl} cal]"
        sub_text = f"{method} · {portion} portion"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness = 1
        (tw, th), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
        (sw, sh), _ = cv2.getTextSize(sub_text, font, 0.42, thickness)
        label_w = max(tw, sw) + 10
        label_h = th + sh + 14

        # Place label above box, clamp inside frame
        lx = max(0, min(x1, w - label_w))
        ly = max(label_h, y1)

        cv2.rectangle(frame,
                      (lx, ly - label_h),
                      (lx + label_w, ly),
                      colour, -1)
        cv2.rectangle(frame,
                      (lx, ly - label_h),
                      (lx + label_w, ly),
                      (255, 255, 255), 1)

        cv2.putText(frame, display_text,
                    (lx + 4, ly - sh - 8),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, sub_text,
                    (lx + 4, ly - 4),
                    font, 0.42, (220, 220, 220), thickness, cv2.LINE_AA)

    # ── 5. Legend ─────────────────────────────────────────────────────────────
    legend_items = [
        ("Low calorie", (0, 200, 0)),
        ("Medium calorie", (0, 165, 255)),
        ("High calorie", (0, 0, 220)),
    ]
    for i, (leg_label, leg_col) in enumerate(legend_items):
        lx0 = 18
        cy = 24 + i * 28
        cv2.rectangle(frame, (lx0, cy - 10), (lx0 + 20, cy + 10),
                      leg_col, -1)
        cv2.rectangle(frame, (lx0, cy - 10), (lx0 + 20, cy + 10),
                      (255, 255, 255), 1)
        cv2.putText(frame, leg_label, (lx0 + 26, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # ── 6. Title bar ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 36), (w, h), (30, 30, 30), -1)
    cv2.putText(frame,
                f"Food Detection · {len(foods)} items · Gemini VLM",
                (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 220, 255), 1, cv2.LINE_AA)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    cv2.imwrite(output_path, frame)
    print(f"[annotator] Saved → {output_path}")
    return output_path
