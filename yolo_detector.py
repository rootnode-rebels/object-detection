"""
CPU-Optimized YOLOv8 Object Detection Module
Designed for Raspberry Pi 5 / Standard Hardware (No NPU Required)
Runs directly on CPU using OpenCV DNN with ARM NEON acceleration.
"""

import os
import sys
import time
import urllib.request
import hashlib
import numpy as np
import cv2

# COCO 80 Class Names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_files")
YOLO_ONNX_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx")
ASSISTIVE_ONNX_PATH = os.path.join(MODEL_DIR, "yolov8_assistive_92.onnx")
CLASSES_TXT_PATH = os.path.join(BASE_DIR, "classes.txt")
YOLO_ONNX_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"


YOLO_ONNX_SHA256 = "cc08cf9fa1f18cabd30089fa82564a4ec657f8868adc58a60db8d35306fc59bc"


def verify_file_sha256(file_path, expected_hash):
    """Computes SHA-256 checksum and compares against expected hash."""
    if not os.path.exists(file_path):
        return False
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower() == expected_hash.lower()


def download_yolo_onnx(target_path=YOLO_ONNX_PATH, expected_hash=YOLO_ONNX_SHA256):
    """Downloads the official YOLOv8n ONNX model (~12MB) with SHA-256 integrity verification (Fix for Finding F-05)."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 10_000_000:
        if expected_hash is None or verify_file_sha256(target_path, expected_hash):
            return target_path
        print(f"[WARNING] Local model hash mismatch or corruption detected. Re-downloading...")

    print(f"[INFO] Downloading official YOLOv8n ONNX model (~12MB)...")
    temp_path = target_path + ".tmp"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(YOLO_ONNX_URL, headers=headers)

    hasher = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=90) as response, open(temp_path, "wb") as f:
        chunk_size = 64 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            hasher.update(chunk)

    downloaded_hash = hasher.hexdigest().lower()
    if expected_hash and downloaded_hash != expected_hash.lower():
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise ValueError(
            f"Security Error: Downloaded model SHA-256 mismatch! Expected {expected_hash}, got {downloaded_hash}."
        )

    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)
    print(f"[INFO] YOLOv8n ONNX model verified & saved to {target_path} ({os.path.getsize(target_path):,} bytes, SHA-256 verified)")
    return target_path


class YOLODetector:
    """
    Lightweight, CPU-optimized YOLOv8 Object Detector.
    Runs entirely on standard Raspberry Pi 5 CPU without requiring an NPU or AI HAT.
    """
    def __init__(self, model_path=None, classes=None, conf_threshold=0.5, nms_threshold=0.45, input_size=(320, 320)):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size  # (320, 320) yields ~15-20 FPS on Pi 5 CPU; (640, 640) for max accuracy

        # Smart Model Selection:
        # 1. User-supplied model_path (if exists)
        # 2. Local 92-class Assistive ONNX model (if exists)
        # 3. Local YOLOv8n ONNX model
        # 4. Download YOLOv8n ONNX model
        chosen_model = None
        if model_path is not None and os.path.exists(model_path):
            chosen_model = model_path
        elif os.path.exists(ASSISTIVE_ONNX_PATH) and os.path.getsize(ASSISTIVE_ONNX_PATH) > 10_000_000:
            chosen_model = ASSISTIVE_ONNX_PATH
        elif os.path.exists(YOLO_ONNX_PATH) and os.path.getsize(YOLO_ONNX_PATH) > 10_000_000:
            chosen_model = YOLO_ONNX_PATH
        else:
            try:
                chosen_model = download_yolo_onnx(YOLO_ONNX_PATH)
            except Exception as e:
                print(f"[WARN] Could not download default YOLOv8 model: {e}")
                chosen_model = YOLO_ONNX_PATH

        # Smart Classes Selection:
        if classes is not None:
            self.classes = classes
        elif chosen_model == ASSISTIVE_ONNX_PATH and os.path.exists(CLASSES_TXT_PATH):
            with open(CLASSES_TXT_PATH, "r") as f:
                self.classes = [line.strip() for line in f if line.strip()]
        else:
            self.classes = COCO_CLASSES

        print(f"[INFO] Loading YOLOv8 ONNX model on CPU: {chosen_model}...")
        self.net = cv2.dnn.readNetFromONNX(chosen_model)

        # Optimize for CPU execution on Raspberry Pi (ARM NEON)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"[INFO] YOLOv8 CPU detector ready with {len(self.classes)} classes. Input resolution: {self.input_size[0]}x{self.input_size[1]}")

    def detect(self, frame):
        """
        Runs object detection on the input frame.
        Returns:
        [
            {
                "class": str,
                "confidence": float,
                "box": [x1, y1, x2, y2],
                "direction": "left" | "center" | "right"
            }, ...
        ]
        """
        if frame is None or self.net is None:
            return []

        orig_h, orig_w = frame.shape[:2]
        target_w, target_h = self.input_size

        # Create normalized 1/255 blob
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (target_w, target_h), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Output shape is [1, 84, 8400] -> Transpose to [8400, 84]
        predictions = np.transpose(outputs[0], (1, 0))

        boxes = []
        confidences = []
        class_ids = []

        x_scale = orig_w / target_w
        y_scale = orig_h / target_h

        # Vectorized candidate extraction (Fix for Finding F-02: <1.5ms vs 35-50ms in Python loop)
        scores = predictions[:, 4:]
        max_cids = np.argmax(scores, axis=1)
        max_confs = scores[np.arange(len(scores)), max_cids]

        mask = max_confs >= self.conf_threshold
        if np.any(mask):
            valid_preds = predictions[mask]
            confidences = max_confs[mask].astype(float).tolist()
            class_ids = max_cids[mask].astype(int).tolist()

            cx = valid_preds[:, 0]
            cy = valid_preds[:, 1]
            w = valid_preds[:, 2]
            h = valid_preds[:, 3]

            x1 = np.maximum(0, ((cx - 0.5 * w) * x_scale)).astype(int)
            y1 = np.maximum(0, ((cy - 0.5 * h) * y_scale)).astype(int)
            bw = (w * x_scale).astype(int)
            bh = (h * y_scale).astype(int)

            boxes = np.column_stack([x1, y1, bw, bh]).tolist()

        if len(boxes) == 0:
            return []

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detections = []
        if len(indices) > 0:
            flat_indices = indices.flatten() if hasattr(indices, 'flatten') else list(indices)
            for idx in flat_indices:
                box = boxes[idx]
                x1 = max(0, box[0])
                y1 = max(0, box[1])
                x2 = min(orig_w, x1 + box[2])
                y2 = min(orig_h, y1 + box[3])

                cid = class_ids[idx]
                class_name = self.classes[cid] if cid < len(self.classes) else f"object_{cid}"

                # 9-Zone Spatial Partitioning (Top 3, Middle 3, Bottom 3)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                # Column (Horizontal: Left, Middle, Right)
                if center_x < orig_w / 3.0:
                    col = "left"
                    col_label = "on your left"
                elif center_x > (orig_w * 2.0) / 3.0:
                    col = "right"
                    col_label = "on your right"
                else:
                    col = "middle"
                    col_label = "directly ahead"

                # Row (Vertical: Top/Overhead, Middle/Chest, Bottom/Ground)
                if center_y < orig_h / 3.0:
                    row = "top"
                    row_label = "overhead"
                elif center_y > (orig_h * 2.0) / 3.0:
                    row = "bottom"
                    row_label = "ground level"
                else:
                    row = "middle"
                    row_label = "chest level"

                if row == "middle" and col == "middle":
                    zone = "middle-center"
                    direction = "directly ahead"
                elif row == "middle":
                    zone = f"middle-{col}"
                    direction = col_label
                elif col == "middle":
                    zone = f"{row}-middle"
                    direction = f"{row_label} ahead"
                else:
                    zone = f"{row}-{col}"
                    direction = f"{row_label} {col_label}"

                detections.append({
                    "class": class_name,
                    "confidence": float(confidences[idx]),
                    "box": [x1, y1, x2, y2],
                    "direction": direction,
                    "zone": zone,
                    "row": row,
                    "col": col,
                    "center_x": center_x,
                    "center_y": center_y
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def close(self):
        """Release resources."""
        self.net = None


def format_spatial_announcement(detections, distance_meters=None, target_zone=None):
    """
    Constructs actionable spatial navigation voice guidance across the 9-part grid:
    Top Row:    top-left    | top-middle (overhead)     | top-right
    Middle Row: middle-left | middle-center (ahead)    | middle-right
    Bottom Row: bottom-left | bottom-middle (ground)    | bottom-right
    
    :param detections: List of detection dictionaries from YOLODetector.detect()
    :param distance_meters: Optional ultrasonic distance reading in meters
    :param target_zone: Optional filter (e.g. 'top-left', 'middle-center', 'bottom-middle', or 'left'/'middle'/'right')
    """
    # 9 spatial zones dictionary
    zones = {
        "top-left": [], "top-middle": [], "top-right": [],
        "middle-left": [], "middle-center": [], "middle-right": [],
        "bottom-left": [], "bottom-middle": [], "bottom-right": []
    }
    
    # Map detections into 9 zones
    for d in (detections or []):
        z = d.get("zone", "middle-center")
        # Backwards compatible mapping if old 3-zone names appear
        if z == "middle": z = "middle-center"
        elif z == "left": z = "middle-left"
        elif z == "right": z = "middle-right"

        if z in zones and d["class"] not in [item["class"] for item in zones[z]]:
            zones[z].append(d)

    # Consider middle blocked if ultrasonic distance is under 1.0m
    middle_proximity = (distance_meters is not None and distance_meters < 1.0)
    has_mid = bool(zones["middle-center"]) or middle_proximity
    has_low_ahead = bool(zones["bottom-middle"])
    has_overhead = bool(zones["top-middle"])

    dist_str = f" at {distance_meters:.1f} meters" if (distance_meters is not None and distance_meters < 3.0) else ""

    # 1. Single Targeted Query for any of the 9 zones
    if target_zone:
        # Alias normalization
        tz = target_zone.lower()
        if tz in ["middle", "ahead", "center"]: tz = "middle-center"
        elif tz == "left": tz = "middle-left"
        elif tz == "right": tz = "middle-right"
        elif tz in ["top", "overhead"]: tz = "top-middle"
        elif tz in ["bottom", "ground", "low"]: tz = "bottom-middle"

        items = zones.get(tz, [])
        names = " and ".join([it["class"] for it in items[:2]]) if items else None

        zone_names = {
            "top-left": "Top-left overhead",
            "top-middle": "Top-overhead ahead",
            "top-right": "Top-right overhead",
            "middle-left": "Middle left",
            "middle-center": "Directly ahead",
            "middle-right": "Middle right",
            "bottom-left": "Ground level on the left",
            "bottom-middle": "Ground level directly ahead",
            "bottom-right": "Ground level on the right"
        }
        z_label = zone_names.get(tz, tz)

        if not names:
            if tz == "middle-center" and middle_proximity:
                return f"Obstacle directly ahead{dist_str}. Step aside to the left or right."
            return f"{z_label} is clear."
        
        if tz == "bottom-middle":
            return f"Low obstacle directly ahead on the ground: {names}{dist_str}. Watch your step."
        elif tz == "top-middle":
            return f"Overhead obstacle directly ahead: {names}. Watch your head."
        elif tz == "middle-center":
            return f"Obstacle directly ahead: {names}{dist_str}. Go right or go left."
        elif "left" in tz:
            return f"Obstacle on the {tz.replace('-', ' ')}: {names}. Safe to go right."
        else:
            return f"Obstacle on the {tz.replace('-', ' ')}: {names}. Safe to go left."

    # 2. Priority Navigation Announcement (Overhead, Ground Hazards, Path Obstacles)
    # Check Ground Trip Hazards
    if has_low_ahead:
        low_names = " and ".join([it["class"] for it in zones["bottom-middle"][:2]])
        return f"Warning, ground obstacle directly ahead: {low_names}{dist_str}. Watch your step."

    # Check Overhead Hazard
    if has_overhead:
        top_names = " and ".join([it["class"] for it in zones["top-middle"][:2]])
        return f"Warning, overhead obstacle directly ahead: {top_names}. Duck or step aside."

    # Eye/Chest Level Obstacles
    m_name = (" and ".join([it["class"] for it in zones["middle-center"][:2]])) if zones["middle-center"] else ("obstacle" if middle_proximity else None)
    l_name = (" and ".join([it["class"] for it in zones["middle-left"][:2]])) if zones["middle-left"] else None
    r_name = (" and ".join([it["class"] for it in zones["middle-right"][:2]])) if zones["middle-right"] else None

    has_left = bool(l_name)
    has_right = bool(r_name)

    if not has_mid and not has_left and not has_right:
        return "Path clear ahead. Continue straight."
    elif has_left and not has_mid and not has_right:
        return f"Obstacle on your left: {l_name}. Path ahead is clear, go straight."
    elif has_right and not has_mid and not has_left:
        return f"Obstacle on your right: {r_name}. Path ahead is clear, go straight."
    elif has_mid and not has_left and not has_right:
        return f"Obstacle directly ahead: {m_name}{dist_str}. Go right or go left."
    elif has_mid and has_left and not has_right:
        return f"Obstacle ahead and on the left: {m_name}{dist_str}. Go right, path is clear on the right."
    elif has_mid and has_right and not has_left:
        return f"Obstacle ahead and on the right: {m_name}{dist_str}. Go left, path is clear on the left."
    elif not has_mid and has_left and has_right:
        return f"Obstacles on left ({l_name}) and right ({r_name}). Center path is clear, go straight."
    else:
        return f"Obstacles detected ahead ({m_name}), left, and right. Path blocked, please stop."


if __name__ == "__main__":
    print("[INFO] Running YOLODetector standalone test...")
    detector = YOLODetector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy)
    print(f"[STATUS] Test completed successfully. Detected {len(res)} objects on dummy frame.")
    detector.close()
