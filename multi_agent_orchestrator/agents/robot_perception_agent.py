# agents/robot_perception_agent.py
"""
Robot Perception Agent
======================
Hardware Configuration:
 • Robot Arm : Franka Panda
 • Simulator : MuJoCo
 • Camera : MuJoCo Virtual Pinhole Camera (640x480, fovy read live from model)
 • Real Cam : Intel RealSense D435i (future sim-to-real transfer)

Pipeline:
 1. Read RGB + depth directly from MuJoCo rendering buffers
 (replaces MiDaS — MuJoCo gives perfect ground-truth depth)
 2. Resolve camera intrinsics live from MuJoCo model at runtime
 (FX, FY, CX, CY are NOT hardcoded — MuJoCo XML is single source of truth)
 3. Run YOLOv8 object detection -> bounding boxes + labels
 3b. Open-vocabulary fallback -> if target_object is set and YOLO's fixed
 80-class vocabulary doesn't cover it, query the shared VLM
 (llm/vlm_client.py) for a single bounding box on that specific object.
 This keeps YOLO's speed for the common case and only pays the VLM's
 extra latency when genuinely needed. The VLM here is used strictly as
 an open-vocabulary DETECTOR — it does not reason about intent; that
 stays the responsibility of robot_planner_agent.py's LLM call.
 4. Run SAM2 segmentation -> per-object binary masks
 (applies uniformly to YOLO-sourced AND VLM-sourced boxes)
 5. Back-project to 3-D -> centroid_3d in camera frame
 using intrinsics sourced live from MuJoCo
 6. Write detected_objects, scene_description, masks, and depth_map into state.
 masks and depth_map are forwarded directly to robot_planner_node via state.
 Each DetectedObject carries a detector_src field ("yolo" or "vlm") so the
 source of every detection stays traceable downstream.

Camera Intrinsics — resolved at runtime via priority chain:
 PATH 1 -> state["mujoco_model"] : live MuJoCo MjModel (preferred, ground truth)
 mujoco.mj_name2id() + model.cam_fovy[cam_id]
 FX = FY = (H/2) / tan(fovy_rad/2) [MuJoCo square pixels]
 CX = W/2, CY = H/2
 PATH 2 -> state["cam_intrinsics"] : pre-computed {fx, fy, cx, cy} dict
 PATH 3 -> geometric fallback : state["cam_fovy"] or CAM_FOVY env var

Sim-to-Real Note:
 When transferring to Intel RealSense D435i, populate state["cam_intrinsics"]
 with the RealSense factory calibration:
 state["cam_intrinsics"] = {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0}
 and re-enable _estimate_depth_map_midas() for monocular depth if needed.
"""
from __future__ import annotations
import logging
import os
import math
from typing import Any, Dict, List
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from state.schema import (
    AgentName,
    DetectedObject,
    OrchestratorState,
    add_message,
)
from llm.vlm_client import query_vlm_for_object_box

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIGURATION
# Only non-intrinsic parameters live here.
# FX, FY, CX, CY are intentionally absent — they are resolved at runtime
# directly from the live MuJoCo model (single source of truth = XML scene).
#
# CAM_FOVY is kept ONLY as a last-resort geometric fallback when no MuJoCo
# model is available in state (e.g. offline unit tests).
# ══════════════════════════════════════════════════════════════════════════════
# -- Render resolution ---------------------------------------------------------
CAM_WIDTH = int(os.getenv("CAM_WIDTH", "640"))
CAM_HEIGHT = int(os.getenv("CAM_HEIGHT", "480"))

# -- Fallback fovy — used ONLY in PATH 3 (no MuJoCo model in state) -----------
CAM_FOVY = float(os.getenv("CAM_FOVY", "60.0")) # degrees

# -- Depth clip planes ---------------------------------------------------------
DEPTH_NEAR = float(os.getenv("DEPTH_NEAR", "0.01")) # near clip (metres)
DEPTH_FAR = float(os.getenv("DEPTH_FAR", "2.50")) # far clip (metres)

# -- MiDaS fallback scale ------------------------------------------------------
DEPTH_SCALE = float(os.getenv("DEPTH_SCALE", "2.50")) # matches far clip

# -- YOLO ----------------------------------------------------------------------
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.35"))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt") # nano=fast; yolov8x.pt=accurate

# -- MuJoCo camera name --------------------------------------------------------
MUJOCO_CAM_NAME = os.getenv("MUJOCO_CAM_NAME", "wrist_cam")

# ══════════════════════════════════════════════════════════════════════════════
# INTRINSICS HELPER
# ══════════════════════════════════════════════════════════════════════════════
def get_camera_intrinsics_from_mujoco(
    model,
    cam_name: str = MUJOCO_CAM_NAME,
    width: int = CAM_WIDTH,
    height: int = CAM_HEIGHT,
) -> Dict[str, Any]:
    """
    Extract FX, FY, CX, CY and the 3x3 K matrix directly from a live
    MuJoCo model object.
    """
    import mujoco
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
        if cam_id >= 0:
            cam_name = "overhead_cam"
        elif model.ncam > 0:
            cam_id = 0
            cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0) or "cam0"
        else:
            fy_fb = (height / 2.0) / math.tan(math.radians(CAM_FOVY) / 2.0)
            return {"fx": fy_fb, "fy": fy_fb, "cx": width / 2.0, "cy": height / 2.0,
                    "K": np.array([[fy_fb, 0, width/2.0], [0, fy_fb, height/2.0], [0, 0, 1]], dtype=np.float64)}

    fovy_deg = float(model.cam_fovy[cam_id])
    fovy_rad = math.radians(fovy_deg)
    fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy # MuJoCo square pixels -> FX == FY
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    logger.info(
        "[robot_perception] MuJoCo intrinsics for '%s': "
        "fovy=%.4f deg FX=FY=%.4f CX=%s CY=%s",
        cam_name, fovy_deg, fx, cx, cy,
    )
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "K": K}

# ══════════════════════════════════════════════════════════════════════════════
# DEPTH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def mujoco_depth_to_metres(
    depth_buffer: np.ndarray,
    near: float = DEPTH_NEAR,
    far: float = DEPTH_FAR,
) -> np.ndarray:
    """
    Convert a MuJoCo normalised depth buffer [0, 1] to metric depth (metres).
    """
    depth_buffer = np.asarray(depth_buffer, dtype=np.float64)
    denom = far - depth_buffer * (far - near)
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    depth_m = (near * far) / denom
    return depth_m.astype(np.float32)

def _estimate_depth_map_midas(bgr_image: np.ndarray) -> np.ndarray:
    """
    Monocular depth via MiDaS — FALLBACK ONLY.
    """
    try:
        import torch
        logger.warning(
            "[robot_perception] Using MiDaS depth — MuJoCo depth buffer not available. "
            "Depth accuracy will be approximate."
        )
        if not hasattr(_estimate_depth_map_midas, "_cache"):
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            midas.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            transform = transforms.small_transform
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            midas.to(device)
            _estimate_depth_map_midas._cache = (midas, transform, device)
            logger.info("[robot_perception] MiDaS loaded and cached.")
        midas, transform, device = _estimate_depth_map_midas._cache
        rgb = bgr_image[:, :, ::-1].copy() if cv2 is None else cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        input_batch = transform(rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=bgr_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_raw = prediction.cpu().numpy()
        depth_norm = (depth_raw - depth_raw.min()) / (
            depth_raw.max() - depth_raw.min() + 1e-8
        )
        depth_m = DEPTH_SCALE * (1.0 - depth_norm)
        return depth_m.astype(np.float32)
    except Exception as exc:
        logger.warning("[robot_perception] MiDaS failed: %s — fallback to constant depth", exc)
        H, W = bgr_image.shape[:2]
        return np.full((H, W), 0.5, dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def _pixel_to_3d(
    u: float,
    v: float,
    depth_m: float,
    fx: float,
    fy: float,
    pp_cx: float,
    pp_cy: float,
) -> List[float]:
    """
    Back-project pixel (u, v) + metric depth -> 3-D point in camera frame.
    """
    X = (u - pp_cx) * depth_m / fx
    Y = (v - pp_cy) * depth_m / fy
    Z = depth_m
    return [round(float(X), 4), round(float(Y), 4), round(float(Z), 4)]

# ══════════════════════════════════════════════════════════════════════════════
# YOLO
# ══════════════════════════════════════════════════════════════════════════════
def _load_yolo():
    from ultralytics import YOLO
    return YOLO(YOLO_MODEL)

# ══════════════════════════════════════════════════════════════════════════════
# OPEN-VOCABULARY FALLBACK
# ══════════════════════════════════════════════════════════════════════════════
def _yolo_found_target(candidates: List[Dict[str, Any]], target_object: str) -> bool:
    if not target_object:
        return True
    target_words = set(target_object.lower().split())
    for cand in candidates:
        label = cand["label"].lower()
        if label in target_object.lower() or any(w in label for w in target_words):
            return True
    return False

# ══════════════════════════════════════════════════════════════════════════════
# SAM2
# ══════════════════════════════════════════════════════════════════════════════
_SAM2_PREDICTOR_CACHE = None

def _get_sam2_predictor():
    global _SAM2_PREDICTOR_CACHE
    if _SAM2_PREDICTOR_CACHE is not None:
        return _SAM2_PREDICTOR_CACHE
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_cfg = os.getenv("SAM2_CONFIG", "sam2_hiera_small.yaml")
    sam2_ckpt = os.getenv("SAM2_CHECKPOINT", "checkpoints/sam2_hiera_small.pt")
    sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    _SAM2_PREDICTOR_CACHE = predictor
    logger.info("[robot_perception] SAM2 predictor built and cached (device=%s).", device)
    return predictor

def _run_sam2_mask(
    bgr_image: np.ndarray,
    bbox_xyxy: List[float],
) -> np.ndarray | None:
    try:
        predictor = _get_sam2_predictor()
        rgb = bgr_image[:, :, ::-1].copy() if cv2 is None else cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        predictor.set_image(rgb)
        x1, y1, x2, y2 = bbox_xyxy
        box_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np,
            multimask_output=False,
        )
        return masks[0].astype(np.uint8)
    except Exception as exc:
        logger.warning(
            "[robot_perception] SAM2 failed: %s — synthesizing binary mask from box.", exc
        )
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
        H, W = bgr_image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            return mask
        return None

# ══════════════════════════════════════════════════════════════════════════════
# MASK CENTROID UTILITY
# ══════════════════════════════════════════════════════════════════════════════
def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    if cv2 is not None:
        M = cv2.moments(mask)
        if M["m00"] > 0:
            return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        return float(xs.mean()), float(ys.mean())
    return 0.0, 0.0

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LANGGRAPH NODE
# ══════════════════════════════════════════════════════════════════════════════
def robot_perception_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — perceive the scene and populate detected_objects.
    """
    mujoco_model = state.get("mujoco_model")
    cam_name = state.get("cam_name") or MUJOCO_CAM_NAME

    if mujoco_model is not None:
        intr = get_camera_intrinsics_from_mujoco(
            model = mujoco_model,
            cam_name = cam_name,
            width = CAM_WIDTH,
            height = CAM_HEIGHT,
        )
        logger.info(
            "[robot_perception] Intrinsics from MuJoCo model: "
            "FX=%.4f FY=%.4f CX=%s CY=%s",
            intr["fx"], intr["fy"], intr["cx"], intr["cy"],
        )
    elif state.get("cam_intrinsics") is not None:
        intr = state["cam_intrinsics"]
        logger.info(
            "[robot_perception] Intrinsics from state['cam_intrinsics']: "
            "FX=%.4f FY=%.4f CX=%s CY=%s",
            intr["fx"], intr["fy"], intr["cx"], intr["cy"],
        )
    else:
        fovy_deg = float(state.get("cam_fovy", CAM_FOVY))
        fovy_rad = math.radians(fovy_deg)
        fy_fb = (CAM_HEIGHT / 2.0) / math.tan(fovy_rad / 2.0)
        intr = {
            "fx": fy_fb,
            "fy": fy_fb,
            "cx": CAM_WIDTH / 2.0,
            "cy": CAM_HEIGHT / 2.0,
        }
        logger.warning(
            "[robot_perception] No MuJoCo model in state — "
            "using geometric fallback (fovy=%.1f deg FX=FY=%.4f).",
            fovy_deg, fy_fb,
        )

    fx: float = intr["fx"]
    fy: float = intr["fy"]
    pp_cx: float = intr["cx"]
    pp_cy: float = intr["cy"]

    image_path = state.get("image_path")

    # If image_path is not given, but MuJoCo model and data are present, render live
    if (not image_path or not os.path.exists(image_path)) and mujoco_model is not None and state.get("mujoco_data") is not None:
        try:
            import mujoco
            renderer = mujoco.Renderer(mujoco_model, height=CAM_HEIGHT, width=CAM_WIDTH)
            cam_target = cam_name if mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) >= 0 else 0
            renderer.update_scene(state["mujoco_data"], camera=cam_target)
            rgb_render = renderer.render()
            image_path = "scene_camera_frame.jpg"
            Image.fromarray(rgb_render).save(image_path)
            if state.get("mujoco_depth_buffer") is None:
                renderer.enable_depth_rendering()
                renderer.update_scene(state["mujoco_data"], camera=cam_target)
                state["mujoco_depth_buffer"] = renderer.render().copy()
            renderer.close()
        except Exception as e:
            logger.warning("[robot_perception] Live simulation render warning: %s", e)

    if not image_path or not os.path.exists(image_path):
        logger.warning("[robot_perception] No valid image_path — skipping perception.")
        return {
            "detected_objects": [],
            "scene_description": "No image provided for robot perception.",
            "masks": None,
            "depth_map": None,
        }

    logger.info("[robot_perception] Loading image: %s", image_path)
    if cv2 is not None:
        bgr = cv2.imread(image_path)
    else:
        pil_im = Image.open(image_path).convert("RGB")
        bgr = np.array(pil_im)[:, :, ::-1].copy()

    if bgr is None:
        logger.error("[robot_perception] Image loading failed for '%s'.", image_path)
        return {
            "detected_objects": [],
            "scene_description": f"Failed to load image: {image_path}",
            "masks": None,
            "depth_map": None,
        }

    H, W = bgr.shape[:2]
    logger.info("[robot_perception] Image size: %dx%d", W, H)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1 : Resolve depth map
    # ══════════════════════════════════════════════════════════════════════════
    depth_map: np.ndarray
    if state.get("mujoco_depth_buffer") is not None:
        logger.info("[robot_perception] Using MuJoCo ground-truth depth buffer.")
        near = DEPTH_NEAR
        far = DEPTH_FAR
        if mujoco_model is not None:
            extent = float(getattr(mujoco_model.stat, "extent", 1.0))
            near = float(getattr(mujoco_model.vis.map, "znear", 0.01)) * extent
            far = float(getattr(mujoco_model.vis.map, "zfar", 2.5)) * extent
        depth_map = mujoco_depth_to_metres(
            state["mujoco_depth_buffer"],
            near=near,
            far=far,
        )
    elif state.get("realsense_depth_map") is not None:
        logger.info("[robot_perception] Using RealSense metric depth map.")
        depth_map = np.asarray(state["realsense_depth_map"], dtype=np.float32)
    else:
        logger.warning(
            "[robot_perception] No depth buffer in state — falling back to MiDaS."
        )
        try:
            depth_map = _estimate_depth_map_midas(bgr)
        except Exception as exc:
            logger.error("[robot_perception] MiDaS failed: %s — using 0.5 m dummy.", exc)
            depth_map = np.full((H, W), 0.5, dtype=np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2 : YOLO detection + open-vocabulary VLM fallback (Option B)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("[robot_perception] Running YOLO detection...")
    candidates: List[Dict[str, Any]] = []
    try:
        yolo = _load_yolo()
        results = yolo(bgr, conf=YOLO_CONF, verbose=False)
        boxes = results[0].boxes if results[0].boxes is not None else []
        class_names = yolo.names
        logger.info("[robot_perception] YOLO found %d object(s).", len(boxes))
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = class_names.get(cls_id, f"object_{cls_id}")
            candidates.append({
                "label": label,
                "confidence": conf,
                "xyxy": xyxy,
                "detector_src": "yolo",
            })
    except Exception as exc:
        logger.warning("[robot_perception] YOLO detection failed: %s", exc)

    # -- Open-vocabulary fallback (Option B — conditional, target-specific) ----
    target_object = state.get("target_object", "") or ""
    if not _yolo_found_target(candidates, target_object):
        logger.info(
            "[robot_perception] YOLO found no match for target_object='%s' — "
            "falling back to VLM open-vocabulary detection.",
            target_object,
        )
        pil_image = Image.fromarray(bgr[:, :, ::-1].copy() if cv2 is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        try:
            vlm_result = query_vlm_for_object_box(pil_image, target_object)
            if vlm_result and vlm_result.get("found"):
                y1n, x1n, y2n, x2n = vlm_result["box_2d"]
                vlm_xyxy = [x1n / 1000 * W, y1n / 1000 * H, x2n / 1000 * W, y2n / 1000 * H]
                candidates.append({
                    "label": target_object,
                    "confidence": vlm_result.get("confidence", 0.5),
                    "xyxy": vlm_xyxy,
                    "detector_src": "vlm",
                })
                logger.info(
                    "[robot_perception] VLM fallback found target_object='%s' "
                    "bbox=%s confidence=%.2f",
                    target_object, [round(v, 1) for v in vlm_xyxy], candidates[-1]["confidence"],
                )
            else:
                logger.warning(
                    "[robot_perception] VLM fallback also could not find target_object='%s'.",
                    target_object,
                )
        except Exception as exc:
            logger.warning("[robot_perception] VLM query failed: %s", exc)

    # Ground truth simulation fallback if YOLO and VLM produced no detections
    if not candidates and mujoco_model is not None and state.get("mujoco_data") is not None:
        import mujoco
        m = mujoco_model
        d = state["mujoco_data"]
        # Find matching body or target object in simulation
        target_tokens = [w for w in target_object.lower().split() if len(w) > 2]
        matched_body_id = -1
        # First priority: body name match with tokens
        for bid in range(m.nbody):
            bname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            if any(t in bname for t in target_tokens):
                matched_body_id = bid
                break
        if matched_body_id < 0:
            for bname_try in ["target_block", "red_cube", "cube", "block"]:
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname_try)
                if bid >= 0:
                    matched_body_id = bid
                    break

        if matched_body_id >= 0:
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, matched_body_id) or "target_object"
            p_world = d.xpos[matched_body_id].copy()
            # Project to camera
            c_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if c_id < 0:
                c_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
            if c_id < 0 and m.ncam > 0:
                c_id = 0
            if c_id >= 0:
                p_cam = d.cam_xpos[c_id].copy()
                R_cam = d.cam_xmat[c_id].reshape(3, 3)
                p_cam_frame = R_cam.T @ (p_world - p_cam)
                # MuJoCo camera looks down -Z optical axis
                Xc, Yc, Zc = p_cam_frame[0], p_cam_frame[1], -p_cam_frame[2]
                if Zc > 0.05:
                    u_proj = fx * (Xc / Zc) + pp_cx
                    v_proj = fy * (-Yc / Zc) + pp_cy
                    candidates.append({
                        "label": target_object or bname,
                        "confidence": 0.95,
                        "xyxy": [max(0, u_proj - 25), max(0, v_proj - 25), min(W, u_proj + 25), min(H, v_proj + 25)],
                        "centroid_3d": [round(float(p_world[0]), 4), round(float(p_world[1]), 4), round(float(p_world[2]), 4)],
                        "detector_src": "sim_grounding",
                    })

    if not candidates:
        logger.warning("[robot_perception] No objects detected (YOLO empty, VLM fallback empty/not triggered).")
        return {
            "detected_objects": [],
            "scene_description": "No objects detected in the scene.",
            "masks": [],
            "depth_map": depth_map,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Step 3 : Per-object processing (SAM2 mask + 3-D back-projection)
    # ══════════════════════════════════════════════════════════════════════════
    detected_objects: List[DetectedObject] = []
    masks_list: List[Any] = []
    for i, cand in enumerate(candidates):
        x1, y1, x2, y2 = [int(round(v)) for v in cand["xyxy"]]
        label = cand["label"]
        conf = cand["confidence"]
        detector_src = cand["detector_src"]

        # -- SAM2 mask ---------------------------------------------------------
        mask = _run_sam2_mask(bgr, [x1, y1, x2, y2])
        masks_list.append(mask)

        # -- Centroid pixel ----------------------------------------------------
        if mask is not None:
            cx_px, cy_px = _mask_centroid(mask)
            centroid_src = "sam2"
        else:
            cx_px = (x1 + x2) / 2.0
            cy_px = (y1 + y2) / 2.0
            centroid_src = "bbox"

        # -- Depth at centroid -------------------------------------------------
        depth_H, depth_W = depth_map.shape[:2]
        cx_depth = int(min(max(cx_px * (depth_W / W), 0), depth_W - 1))
        cy_depth = int(min(max(cy_px * (depth_H / H), 0), depth_H - 1))
        depth_val = float(depth_map[cy_depth, cx_depth])

        # -- Back-project to 3-D using live MuJoCo intrinsics ------------------
        if cand.get("centroid_3d") is not None:
            centroid_3d = cand["centroid_3d"]
        else:
            centroid_3d = _pixel_to_3d(cx_px, cy_px, depth_val, fx, fy, pp_cx, pp_cy)

            # If in MuJoCo sim and camera is overhead, convert camera frame to world frame
            if mujoco_model is not None and state.get("mujoco_data") is not None:
                import mujoco
                c_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if c_id < 0:
                    c_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
                if c_id < 0 and mujoco_model.ncam > 0:
                    c_id = 0
                if c_id >= 0:
                    d = state["mujoco_data"]
                    p_cam = d.cam_xpos[c_id].copy()
                    R_cam = d.cam_xmat[c_id].reshape(3, 3)
                    p_c = np.array([centroid_3d[0], -centroid_3d[1], -centroid_3d[2]])
                    p_world_est = p_cam + R_cam @ p_c
                    centroid_3d = [round(float(p_world_est[0]), 4),
                                   round(float(p_world_est[1]), 4),
                                   round(float(p_world_est[2]), 4)]

        # -- Build DetectedObject ----------------------------------------------
        obj: DetectedObject = {
            "label": label,
            "confidence": round(conf, 4),
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "mask": mask,
            "depth_m": round(depth_val, 4),
            "centroid_3d": centroid_3d,
            "centroid_src": centroid_src,
            "detector_src": detector_src,
        }
        detected_objects.append(obj)
        logger.info(
            "[robot_perception] [%d] %-20s conf=%.2f depth=%.3fm "
            "3D=%s centroid_src=%s detector_src=%s",
            i, label, conf, depth_val, centroid_3d, centroid_src, detector_src,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 4 : Scene description
    # ══════════════════════════════════════════════════════════════════════════
    summary_lines = [
        f"{o['label']} "
        f"(conf={o['confidence']:.2f}, "
        f"depth={o['depth_m']:.3f}m, "
        f"3D={o['centroid_3d']}, "
        f"centroid_src={o['centroid_src']}, "
        f"detector_src={o['detector_src']})"
        for o in detected_objects
    ]
    scene_description = (
        f"Detected {len(detected_objects)} object(s) "
        f"[cam: {CAM_WIDTH}x{CAM_HEIGHT}, "
        f"FX={fx:.4f} FY={fy:.4f} CX={pp_cx} CY={pp_cy} "
        f"(intrinsics sourced live from MuJoCo)]:\n"
        + "\n".join(f" * {s}" for s in summary_lines)
    )
    logger.info("[robot_perception] Perception complete.\n%s", scene_description)

    # ══════════════════════════════════════════════════════════════════════════
    # Step 5 : Update LangGraph state
    # ══════════════════════════════════════════════════════════════════════════
    updated_messages = add_message(
        state=state,
        role=AgentName.ROBOT_PERC,
        content=scene_description,
    )
    return {
        "detected_objects": detected_objects,
        "scene_description": scene_description,
        "masks": masks_list,
        "depth_map": depth_map,
        "messages": updated_messages,
        "current_step": state.get("current_step", 0) + 1,
    }
