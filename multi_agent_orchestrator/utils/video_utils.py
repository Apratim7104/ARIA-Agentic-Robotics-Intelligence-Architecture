try:
    import cv2
except ImportError:
    cv2 = None
import base64
import os

def extract_key_frames(video_path: str, num_frames: int = 5) -> list[str]:
    """Extract evenly spaced frames from video and return as base64 strings."""
    if cv2 is None:
        return []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buffer).decode("utf-8")
            frames_b64.append(b64)
            
    cap.release()
    return frames_b64
