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


# Specific hazard category sets to empower natural phrasing for blind individuals
GROUND_HAZARD_CLASSES = {
    "staircase", "pothole", "speed breaker", "pedestrian crossing", "road barrier",
    "road divider", "traffic cone", "stone", "rock", "brick", "cement bag",
    "wooden plank", "manhole", "drain", "construction material"
}

OVERHEAD_HAZARD_CLASSES = {
    "tree branch", "road sign", "light", "street light", "electric pole"
}

VEHICLE_APPROACH_CLASSES = {
    "car", "bus", "truck", "auto-rickshaw", "motorcycle", "scooter", "bicycle"
}


class ApproachTracker:
    """
    Lightweight, multi-frame object tracker and approach motion detector.
    Specifically designed for assistive devices for visually impaired users.
    
    Capabilities:
    1. Tracks bounding boxes across consecutive video frames by centroid and IoU.
    2. Measures bounding box expansion rate (growth in area ratio over time window).
    3. Fuses ultrasonic proximity trend (decreasing distance over time).
    4. Categorizes approach motion:
       - 'fast' / 'rapid' (fast vehicle, running person/animal, or steep distance drop)
       - 'steady' (steady approach towards user or user closing in)
       - None (static or receding)
    """
    def __init__(self, max_missed_frames=8, min_growth_thresh=0.14, rapid_growth_thresh=0.35):
        self.next_track_id = 1
        self.tracks = {}  # track_id -> dict
        self.max_missed_frames = max_missed_frames
        self.min_growth_thresh = min_growth_thresh
        self.rapid_growth_thresh = rapid_growth_thresh
        self.prev_distance = None
        self.prev_distance_time = 0.0

    def update(self, detections, distance_meters=None):
        now = time.time()

        # Check ultrasonic distance delta if provided
        distance_dropping = False
        distance_rapid_drop = False
        if distance_meters is not None and 0.0 < distance_meters < 3.0:
            if self.prev_distance is not None:
                dt = max(0.01, now - self.prev_distance_time)
                dist_vel = (distance_meters - self.prev_distance) / dt  # m/s
                if dist_vel < -0.15:  # Closing distance faster than 0.15 m/s
                    distance_dropping = True
                if dist_vel < -0.50 or (self.prev_distance - distance_meters >= 0.20):
                    distance_rapid_drop = True
            self.prev_distance = distance_meters
            self.prev_distance_time = now

        matched_track_ids = set()
        for det in detections:
            box = det["box"]
            cls_name = det["class"]
            center = (det["center_x"], det["center_y"])
            area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))

            best_tid = None
            best_score = -1.0

            for tid, trk in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                if trk["class"] != cls_name:
                    continue

                # Compute IoU
                tb = trk["box"]
                ix1 = max(box[0], tb[0])
                iy1 = max(box[1], tb[1])
                ix2 = min(box[2], tb[2])
                iy2 = min(box[3], tb[3])
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter_area = iw * ih
                union_area = area + trk["area"] - inter_area
                iou = inter_area / max(1.0, union_area)

                # Center distance
                tc = trk["center"]
                dist = np.hypot(center[0] - tc[0], center[1] - tc[1])

                if iou > 0.20 or (dist < 130 and iou > 0.05) or dist < 65:
                    score = iou * 2.0 + (1.0 / (1.0 + dist * 0.02))
                    if score > best_score:
                        best_score = score
                        best_tid = tid

            if best_tid is not None:
                matched_track_ids.add(best_tid)
                trk = self.tracks[best_tid]
                trk["missed"] = 0
                trk["box"] = box
                trk["center"] = center
                trk["area"] = area
                trk["zone"] = det.get("zone", "middle-center")
                trk["history"].append((now, area, center, box))
                if len(trk["history"]) > 25:
                    trk["history"].pop(0)

                # Evaluate growth over time window (0.2s - 1.2s ago)
                is_approaching = False
                approach_speed = None
                growth_rate = 0.0

                if len(trk["history"]) >= 2:
                    baseline_entry = None
                    for hist_entry in trk["history"]:
                        age = now - hist_entry[0]
                        if 0.10 <= age <= 1.2:
                            baseline_entry = hist_entry
                            break
                    if baseline_entry is None:
                        baseline_entry = trk["history"][0]

                    baseline_area = baseline_entry[1]
                    growth_rate = (area - baseline_area) / max(1.0, baseline_area)

                    is_vehicle = cls_name in VEHICLE_APPROACH_CLASSES
                    is_center = ("middle" in det.get("zone", "") or "center" in det.get("zone", ""))

                    # Approaching condition: area expansion or distance dropping towards center object
                    if (growth_rate >= self.min_growth_thresh and area > 1000) or (is_center and distance_dropping):
                        is_approaching = True
                        if growth_rate >= self.rapid_growth_thresh or distance_rapid_drop or (is_vehicle and growth_rate >= 0.18):
                            approach_speed = "fast"
                        else:
                            approach_speed = "steady"

                trk["is_approaching"] = is_approaching
                trk["approach_speed"] = approach_speed
                trk["growth_rate"] = growth_rate

                det["track_id"] = best_tid
                det["is_approaching"] = is_approaching
                det["approach_speed"] = approach_speed
                det["growth_rate"] = growth_rate
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                matched_track_ids.add(tid)
                self.tracks[tid] = {
                    "id": tid,
                    "class": cls_name,
                    "box": box,
                    "center": center,
                    "area": area,
                    "zone": det.get("zone", "middle-center"),
                    "missed": 0,
                    "is_approaching": False,
                    "approach_speed": None,
                    "growth_rate": 0.0,
                    "history": [(now, area, center, box)]
                }
                det["track_id"] = tid
                det["is_approaching"] = False
                det["approach_speed"] = None
                det["growth_rate"] = 0.0

        stale_tids = []
        for tid, trk in self.tracks.items():
            if tid not in matched_track_ids:
                trk["missed"] += 1
                if trk["missed"] > self.max_missed_frames:
                    stale_tids.append(tid)

        for tid in stale_tids:
            del self.tracks[tid]

        return detections


class YOLODetector:
    """
    Lightweight, CPU-optimized YOLOv8 Object Detector with multi-frame approach tracking.
    Runs entirely on standard Raspberry Pi 5 CPU without requiring an NPU or AI HAT.
    """
    def __init__(self, model_path=None, classes=None, conf_threshold=0.5, nms_threshold=0.45, input_size=(320, 320)):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size  # (320, 320) yields ~15-20 FPS on Pi 5 CPU; (640, 640) for max accuracy
        self.tracker = ApproachTracker()

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
        elif chosen_model and (os.path.abspath(chosen_model) == os.path.abspath(ASSISTIVE_ONNX_PATH) or "assistive" in os.path.basename(chosen_model).lower()) and os.path.exists(CLASSES_TXT_PATH):
            with open(CLASSES_TXT_PATH, "r") as f:
                self.classes = [line.strip() for line in f if line.strip()]
        else:
            self.classes = COCO_CLASSES

        print(f"[INFO] Loading YOLOv8 ONNX model on CPU: {chosen_model}...")
        try:
            self.net = cv2.dnn.readNetFromONNX(chosen_model)
            # Optimize for CPU execution on Raspberry Pi (ARM NEON)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"[INFO] YOLOv8 CPU detector ready with {len(self.classes)} classes. Input resolution: {self.input_size[0]}x{self.input_size[1]}")
        except Exception as e:
            print(f"[ERROR] Failed to load ONNX model {chosen_model}: {e}")
            self.net = None

    def detect(self, frame, distance_meters=None):
        """
        Runs object detection on the input frame and tracks approaching objects.
        Returns:
        [
            {
                "class": str,
                "confidence": float,
                "box": [x1, y1, x2, y2],
                "direction": "directly ahead" | "on your left" | "on your right" ...,
                "zone": "middle-center" | "middle-left" | ...,
                "is_approaching": bool,
                "approach_speed": "fast" | "steady" | None,
                "track_id": int
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

        # Output shape is [1, num_attrs, num_anchors] -> Transpose to [num_anchors, num_attrs]
        predictions = np.transpose(outputs[0], (1, 0))

        boxes = []
        confidences = []
        class_ids = []

        x_scale = orig_w / target_w
        y_scale = orig_h / target_h

        # Vectorized candidate extraction (<1.5ms vs 35-50ms in Python loop)
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
            # Still update tracker to keep timestamps and prune stale tracks
            self.tracker.update([], distance_meters=distance_meters)
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

        # Multi-frame tracking to detect approaching objects and speeds
        detections = self.tracker.update(detections, distance_meters=distance_meters)
        return detections

    def close(self):
        """Release resources."""
        self.net = None


def format_spatial_announcement(detections, distance_meters=None, target_zone=None, is_full_scan=False):
    """
    Constructs actionable, life-critical spatial voice guidance specifically for blind individuals.
    
    Priority Hierarchy:
    1. Approaching Hazards (names specific object approaching towards the user, e.g. car, person, dog, bicycle).
    2. Ground Hazards (tripping/falling risks directly ahead: staircase, pothole, curb, drain, rocks).
    3. Overhead Hazards (head collision risks directly ahead: tree branch, low signs).
    4. Immediate Proximity Obstacles (< 1.0m ahead: "Stop! {object} directly in front of you at X meters").
    5. Path Navigation Obstacles (what is ahead, left, and right with clear evasion directions).
    6. All Clear ("Path clear ahead. Continue straight.").
    """
    zones = {
        "top-left": [], "top-middle": [], "top-right": [],
        "middle-left": [], "middle-center": [], "middle-right": [],
        "bottom-left": [], "bottom-middle": [], "bottom-right": []
    }

    approaching_objects = []

    for d in (detections or []):
        z = d.get("zone", "middle-center")
        if z == "middle": z = "middle-center"
        elif z == "left": z = "middle-left"
        elif z == "right": z = "middle-right"

        if z in zones and d["class"] not in [item["class"] for item in zones[z]]:
            zones[z].append(d)

        if d.get("is_approaching", False):
            approaching_objects.append(d)

    middle_proximity = (distance_meters is not None and 0.0 < distance_meters < 1.0)
    has_mid = bool(zones["middle-center"]) or middle_proximity
    has_low_ahead = bool(zones["bottom-middle"])
    has_overhead = bool(zones["top-middle"])

    dist_str = f" at {distance_meters:.1f} meters" if (distance_meters is not None and 0.0 < distance_meters < 3.0) else ""

    # =========================================================
    # 0. On-Demand Tactile Button Full Surroundings Scan
    # =========================================================
    if is_full_scan:
        scan_parts = []
        if approaching_objects:
            app_n = approaching_objects[0]["class"]
            scan_parts.append(f"Caution, {app_n} is approaching towards you")
        if zones["middle-center"]:
            mc_names = " and ".join([it["class"] for it in zones["middle-center"][:2]])
            scan_parts.append(f"{mc_names} directly in front of you{dist_str}")
        elif middle_proximity:
            scan_parts.append(f"obstacle directly in front of you{dist_str}")

        if zones["middle-left"]:
            ml_names = " and ".join([it["class"] for it in zones["middle-left"][:2]])
            scan_parts.append(f"{ml_names} on your left")
        if zones["middle-right"]:
            mr_names = " and ".join([it["class"] for it in zones["middle-right"][:2]])
            scan_parts.append(f"{mr_names} on your right")

        if zones["bottom-middle"]:
            bm_names = " and ".join([it["class"] for it in zones["bottom-middle"][:2]])
            scan_parts.append(f"low {bm_names} on the ground ahead")
        if zones["top-middle"]:
            tm_names = " and ".join([it["class"] for it in zones["top-middle"][:2]])
            scan_parts.append(f"overhead {tm_names} ahead")

        if scan_parts:
            # Action suggestion
            if not zones["middle-right"]:
                action = "Path is clear on the right."
            elif not zones["middle-left"]:
                action = "Path is clear on the left."
            elif not zones["middle-center"] and not middle_proximity:
                action = "Center path is clear to walk."
            else:
                action = "Path blocked, please step aside."
            return f"Surroundings scan: {', '.join(scan_parts)}. {action}"
        else:
            return "Surroundings scan: Path is completely clear ahead. Safe to walk straight."

    # =========================================================
    # 1. Targeted Directional Query (Manual button query by blind user)
    # =========================================================
    if target_zone:
        tz = target_zone.lower()
        if tz in ["middle", "ahead", "center", "infront"]: tz = "middle-center"
        elif tz in ["left"]: tz = "middle-left"
        elif tz in ["right"]: tz = "middle-right"
        elif tz in ["top", "overhead"]: tz = "top-middle"
        elif tz in ["bottom", "ground", "low"]: tz = "bottom-middle"

        items = zones.get(tz, [])
        names = " and ".join([it["class"] for it in items[:2]]) if items else None

        zone_labels = {
            "top-left": "Top-left overhead",
            "top-middle": "Overhead directly ahead",
            "top-right": "Top-right overhead",
            "middle-left": "On your left",
            "middle-center": "Directly in front of you",
            "middle-right": "On your right",
            "bottom-left": "On the ground to your left",
            "bottom-middle": "On the ground directly ahead",
            "bottom-right": "On the ground to your right"
        }
        z_label = zone_labels.get(tz, tz)

        if not names:
            if tz == "middle-center" and middle_proximity:
                return f"Obstacle directly in front of you{dist_str}. Step aside to the left or right."
            return f"{z_label} is clear."

        # Check if item in target zone is approaching
        target_approaching = any(it.get("is_approaching", False) for it in items)
        if target_approaching:
            if tz == "middle-center":
                return f"Caution, {names} is approaching towards you directly ahead{dist_str}."
            elif "left" in tz:
                return f"Caution, {names} is approaching on your left."
            elif "right" in tz:
                return f"Caution, {names} is approaching on your right."

        if tz == "bottom-middle":
            return f"Ground hazard directly ahead: {names}{dist_str}. Watch your step."
        elif tz == "top-middle":
            return f"Overhead obstacle directly ahead: {names}. Watch your head."
        elif tz == "middle-center":
            return f"{names.capitalize()} directly in front of you{dist_str}. Step aside to the left or right."
        elif "left" in tz:
            return f"{names.capitalize()} on your left. Safe to go straight or right."
        else:
            return f"{names.capitalize()} on your right. Safe to go straight or left."

    # =========================================================
    # 2. PRIORITY 1: APPROACHING OBJECTS (Life-Critical for Blind Person)
    # =========================================================
    if approaching_objects:
        # Sort approaching objects: center objects and vehicles first
        def _approach_priority(d):
            score = 0
            if d["class"] in VEHICLE_APPROACH_CLASSES:
                score += 50
            if "middle" in d.get("zone", "") or "center" in d.get("zone", ""):
                score += 40
            if d.get("approach_speed") == "fast":
                score += 30
            return score

        approaching_objects.sort(key=_approach_priority, reverse=True)
        top_app = approaching_objects[0]
        app_name = top_app["class"]
        app_zone = top_app.get("zone", "middle-center")
        is_fast = (top_app.get("approach_speed") == "fast") or (app_name in VEHICLE_APPROACH_CLASSES)

        if "middle" in app_zone and "center" in app_zone:
            if is_fast:
                return f"Warning! {app_name.capitalize()} approaching fast directly ahead{dist_str}! Step aside immediately."
            else:
                return f"Caution, {app_name} is approaching towards you directly ahead{dist_str}."
        elif "left" in app_zone:
            return f"Caution, {app_name} approaching on your left."
        elif "right" in app_zone:
            return f"Caution, {app_name} approaching on your right."
        elif "bottom" in app_zone:
            return f"Warning, {app_name} approaching directly ahead on the ground. Watch your step."
        elif "top" in app_zone:
            return f"Warning, overhead {app_name} approaching directly ahead. Watch your head."

    # =========================================================
    # 3. PRIORITY 2: GROUND HAZARDS (Fall / Trip Risks: Stairs, Potholes, Manholes)
    # =========================================================
    if has_low_ahead:
        low_names = " and ".join([it["class"] for it in zones["bottom-middle"][:2]])
        return f"Warning, ground hazard in front of you: {low_names}{dist_str}. Watch your step."

    # Also check if any center detection is explicitly a ground hazard class (e.g. staircase, pothole)
    center_ground_hazards = [it["class"] for it in zones["middle-center"] if it["class"] in GROUND_HAZARD_CLASSES]
    if center_ground_hazards:
        gh_name = " and ".join(center_ground_hazards[:2])
        return f"Warning, {gh_name} directly in front of you on the ground{dist_str}. Watch your step."

    # =========================================================
    # 4. PRIORITY 3: OVERHEAD HAZARDS (Head Bump Risks: Tree Branches, Signs)
    # =========================================================
    if has_overhead:
        top_names = " and ".join([it["class"] for it in zones["top-middle"][:2]])
        return f"Warning, overhead obstacle directly ahead: {top_names}. Watch your head."

    # =========================================================
    # 5. PRIORITY 4: IMMEDIATE PROXIMITY HAZARD (< 1.0 Meter in Front)
    # =========================================================
    m_name = (" and ".join([it["class"] for it in zones["middle-center"][:2]])) if zones["middle-center"] else None
    l_name = (" and ".join([it["class"] for it in zones["middle-left"][:2]])) if zones["middle-left"] else None
    r_name = (" and ".join([it["class"] for it in zones["middle-right"][:2]])) if zones["middle-right"] else None

    # Fallback to general obstacle if ultrasonic triggered without YOLO label
    front_label = m_name if m_name else ("obstacle" if middle_proximity else None)

    if middle_proximity and front_label:
        if not l_name and r_name:
            return f"Stop! {front_label.capitalize()} directly in front of you{dist_str}. Step aside to the left."
        elif not r_name and l_name:
            return f"Stop! {front_label.capitalize()} directly in front of you{dist_str}. Step aside to the right."
        elif not l_name and not r_name:
            return f"Stop! {front_label.capitalize()} directly in front of you{dist_str}. Step aside to the left or right."
        else:
            return f"Stop! {front_label.capitalize()} directly in front of you{dist_str}. Path blocked, please stop."

    # =========================================================
    # 6. PRIORITY 5: PATH NAVIGATION (Obstacles at Standard Distance)
    # =========================================================
    has_left = bool(l_name)
    has_right = bool(r_name)
    has_front = bool(front_label)

    if not has_front and not has_left and not has_right:
        return "Path clear ahead. Continue straight."
    elif has_left and not has_front and not has_right:
        return f"Obstacle on your left: {l_name}. Path ahead is clear, continue straight."
    elif has_right and not has_front and not has_left:
        return f"Obstacle on your right: {r_name}. Path ahead is clear, continue straight."
    elif has_front and not has_left and not has_right:
        return f"{front_label.capitalize()} directly in front of you{dist_str}. Safe to go right or left."
    elif has_front and has_left and not has_right:
        return f"{front_label.capitalize()} in front of you and {l_name} on the left{dist_str}. Safe to go right."
    elif has_front and has_right and not has_left:
        return f"{front_label.capitalize()} in front of you and {r_name} on the right{dist_str}. Safe to go left."
    elif not has_front and has_left and has_right:
        return f"Obstacles on left ({l_name}) and right ({r_name}). Center path is clear, continue straight."
    else:
        return f"Obstacles detected ahead ({front_label}), left, and right. Path blocked, please stop."


if __name__ == "__main__":
    print("[INFO] Running YOLODetector standalone test...")
    detector = YOLODetector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy)
    print(f"[STATUS] Test completed successfully. Detected {len(res)} objects on dummy frame.")
    detector.close()
