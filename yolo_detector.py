"""
CPU-Optimized YOLOv8 Object Detection Module
Designed for Raspberry Pi 5 / Standard Hardware (No NPU Required)
Runs directly on CPU using OpenCV DNN with ARM NEON acceleration.
"""

import os
import sys
import time
import urllib.request
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

MODEL_DIR = "model_files"
YOLO_ONNX_PATH = os.path.join(MODEL_DIR, "yolov8n.onnx")
ASSISTIVE_ONNX_PATH = os.path.join(MODEL_DIR, "yolov8_assistive_92.onnx")
CLASSES_TXT_PATH = "classes.txt"
YOLO_ONNX_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"


def download_yolo_onnx(target_path=YOLO_ONNX_PATH):
    """Downloads the official YOLOv8n ONNX model (~12MB) if not present."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 10_000_000:
        return target_path

    print(f"[INFO] Downloading official YOLOv8n ONNX model (~12MB)...")
    temp_path = target_path + ".tmp"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(YOLO_ONNX_URL, headers=headers)

    with urllib.request.urlopen(req, timeout=90) as response, open(temp_path, "wb") as f:
        chunk_size = 64 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)
    print(f"[INFO] YOLOv8n ONNX model saved to {target_path} ({os.path.getsize(target_path):,} bytes)")
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

        # Extract candidates
        for row in predictions:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence >= self.conf_threshold:
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                x1 = int((cx - 0.5 * w) * x_scale)
                y1 = int((cy - 0.5 * h) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)

                boxes.append([x1, y1, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                box = boxes[idx]
                x1 = max(0, box[0])
                y1 = max(0, box[1])
                x2 = min(orig_w, x1 + box[2])
                y2 = min(orig_h, y1 + box[3])

                cid = class_ids[idx]
                class_name = self.classes[cid] if cid < len(self.classes) else f"object_{cid}"

                # Calculate spatial direction (Left, Middle, Right)
                center_x = (x1 + x2) / 2.0
                if center_x < orig_w * 0.35:
                    direction = "on your left"
                    zone = "left"
                elif center_x > orig_w * 0.65:
                    direction = "on your right"
                    zone = "right"
                else:
                    direction = "in the middle"
                    zone = "middle"

                detections.append({
                    "class": class_name,
                    "confidence": float(confidences[idx]),
                    "box": [x1, y1, x2, y2],
                    "direction": direction,
                    "zone": zone,
                    "center_x": center_x,
                    "center_y": (y1 + y2) / 2.0
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def close(self):
        """Release resources."""
        self.net = None


def format_spatial_announcement(detections, distance_meters=None, target_zone=None):
    """
    Constructs natural, comprehensive spatial voice feedback (Left, Middle, Right)
    for visually impaired navigation.
    
    :param detections: List of detection dictionaries from YOLODetector.detect()
    :param distance_meters: Optional ultrasonic distance reading in meters
    :param target_zone: Optional filter ('left', 'middle', 'right') for targeted sector queries
    """
    if not detections:
        if distance_meters is not None and distance_meters < 1.0:
            return f"Obstacle in the middle at {distance_meters:.1f} meters. Please stop or step aside."
        return "Path clear ahead. No obstacles detected."

    # Group unique object labels by zone
    zones = {"middle": [], "left": [], "right": []}
    for d in detections:
        z = d.get("zone", "middle")
        if z in zones and d["class"] not in [item["class"] for item in zones[z]]:
            zones[z].append(d)

    # 1. Targeted single-sector query (e.g. user pressed [L], [M], or [R])
    if target_zone:
        items = zones.get(target_zone, [])
        label = "in the middle" if target_zone == "middle" else f"on your {target_zone}"
        if not items:
            return f"No objects detected {label}."
        names = " and ".join([it["class"] for it in items[:2]])
        if distance_meters is not None and distance_meters <= 3.0:
            return f"{names.capitalize()} detected {label} at {distance_meters:.1f} meters."
        return f"{names.capitalize()} detected {label}."

    # 2. Comprehensive multi-zone spatial announcement
    phrases = []

    # Priority 1: Middle / Center (Direct walking path of the user)
    if zones["middle"]:
        m_names = " and ".join([it["class"] for it in zones["middle"][:2]])
        if distance_meters is not None and distance_meters <= 3.0:
            phrases.append(f"In the middle: {m_names} at {distance_meters:.1f} meters")
        else:
            phrases.append(f"In the middle: {m_names}")
    elif distance_meters is not None and distance_meters < 1.0:
        phrases.append(f"Obstacle in the middle at {distance_meters:.1f} meters")

    # Priority 2: Left sector
    if zones["left"]:
        l_names = " and ".join([it["class"] for it in zones["left"][:2]])
        phrases.append(f"On your left: {l_names}")

    # Priority 3: Right sector
    if zones["right"]:
        r_names = " and ".join([it["class"] for it in zones["right"][:2]])
        phrases.append(f"On your right: {r_names}")

    if phrases:
        return ". ".join(phrases) + "."

    if distance_meters is not None and distance_meters < 1.0:
        return f"Warning: obstacle detected at {distance_meters:.1f} meters."

    return "Path clear ahead."


if __name__ == "__main__":
    print("[INFO] Running YOLODetector standalone test...")
    detector = YOLODetector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(dummy)
    print(f"[STATUS] Test completed successfully. Detected {len(res)} objects on dummy frame.")
    detector.close()
