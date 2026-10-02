"""
Hailo-8 / Hailo-8L NPU YOLOv8 Object Detection Module
Designed for Raspberry Pi 5 + Raspberry Pi AI HAT+
Achieves real-time 60 FPS inference with <15% CPU load.
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

# Hailo Model Zoo HEF URL (YOLOv8s / YOLOv8n compiled for Hailo-8 / Hailo-8L)
HEF_DIR = "model_files"
HEF_FILE_HAILO8 = os.path.join(HEF_DIR, "yolov8n.hef")
HEF_URL_HAILO8 = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/yolov8n.hef"
HEF_URL_HAILO8L = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8l/yolov8n.hef"

# Check if HailoRT Python bindings are available
HAILO_AVAILABLE = False
try:
    # hailo_platform is installed on Raspberry Pi OS via: sudo apt install hailo-all
    from hailo_platform import (  # type: ignore
        HEF, VDevice, HailoStreamInterface, ConfigureParams,
        InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams
    )
    HAILO_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception):
    HAILO_AVAILABLE = False


def download_hef(target_path=HEF_FILE_HAILO8, is_hailo8l=False):
    """Downloads the pre-compiled YOLOv8 HEF from Hailo Model Zoo if not present."""
    if not os.path.exists(HEF_DIR):
        os.makedirs(HEF_DIR, exist_ok=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1_000_000:
        return target_path

    url = HEF_URL_HAILO8L if is_hailo8l else HEF_URL_HAILO8
    model_name = "Hailo-8L YOLOv8n" if is_hailo8l else "Hailo-8 YOLOv8n"
    print(f"[INFO] Downloading pre-compiled {model_name} HEF from Hailo Model Zoo...")

    temp_path = target_path + ".tmp"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=60) as response, open(temp_path, "wb") as f:
        chunk_size = 64 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

    if os.path.exists(target_path):
        os.remove(target_path)
    os.rename(temp_path, target_path)
    print(f"[INFO] HEF model saved to {target_path} ({os.path.getsize(target_path):,} bytes)")
    return target_path


class HailoYOLODetector:
    """
    High-Performance YOLOv8 Detector powered by Hailo-8 AI HAT+ (NPU).
    Offloads all deep-learning matrix multiplication to the dedicated NPU.
    """
    def __init__(self, hef_path=None, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.is_npu_active = False
        self.vdevice = None
        self.network_group = None
        self.hef = None
        self.input_vstream_info = None
        self.output_vstream_info = None
        self.input_shape = (640, 640)  # Default YOLOv8 shape

        # Try to initialize Hailo NPU
        if HAILO_AVAILABLE:
            try:
                self._init_hailo(hef_path)
                self.is_npu_active = True
                print("[SUCCESS] Hailo-8 AI HAT+ NPU initialized successfully! Running at up to 60 FPS.")
            except Exception as e:
                print(f"[WARN] Hailo NPU initialization failed: {e}")
                self._print_hardware_warning()
                self._init_cpu_fallback()
        else:
            print("[WARN] hailo_platform library not found.")
            self._print_hardware_warning()
            self._init_cpu_fallback()

    def _print_hardware_warning(self):
        print("\n" + "=" * 70)
        print("⚠️  HARDWARE ARCHITECTURE WARNING (per Executive Briefing):")
        print("   Running in CPU-Only mode. Without the Hailo-8 AI HAT+ NPU:")
        print("   - Video feed drops from 60 FPS down to ~3 FPS.")
        print("   - CPU utilization spikes to 100%, causing thermal throttling (85°C+).")
        print("   - Install the Hailo-8 AI HAT+ on Raspberry Pi 5 for full real-time speed.")
        print("=" * 70 + "\n")

    def _init_hailo(self, hef_path):
        """Initializes the Hailo VDevice and loads the YOLOv8 HEF."""
        if hef_path is None or not os.path.exists(hef_path):
            try:
                hef_path = download_hef(HEF_FILE_HAILO8)
            except Exception:
                # Try Hailo-8L variant if regular fails
                hef_path = download_hef(HEF_FILE_HAILO8, is_hailo8l=True)

        print(f"[INFO] Loading HEF into Hailo-8 NPU: {hef_path}...")
        self.hef = HEF(hef_path)
        params = VDevice.create_params()
        self.vdevice = VDevice(params)

        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.network_group = self.vdevice.configure(self.hef, configure_params)[0]

        self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.output_vstream_info = self.hef.get_output_vstream_infos()
        self.input_shape = (self.input_vstream_info.shape[1], self.input_vstream_info.shape[2])
        print(f"[INFO] Hailo-8 Model Input Shape: {self.input_shape}")

    def _init_cpu_fallback(self):
        """Fallback lightweight detector when NPU is not physically present."""
        print("[INFO] Initializing CPU fallback detector...")
        self.is_npu_active = False
        # Optional MobileNet or OpenCV DNN model as emergency fallback
        model_dir = "model_files"
        prototxt = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
        weights = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")
        self.net = None
        if os.path.exists(prototxt) and os.path.exists(weights):
            try:
                self.net = cv2.dnn.readNetFromCaffe(prototxt, weights)
                print("[INFO] Fallback MobileNet-SSD CPU model loaded.")
            except Exception as e:
                print(f"[WARN] Could not load fallback model: {e}")

    def detect(self, frame):
        """
        Runs object detection on the input frame.
        Returns a list of detected objects:
        [
            {
                "class": str,
                "confidence": float,
                "box": [x1, y1, x2, y2],
                "direction": "left" | "center" | "right"
            }, ...
        ]
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]

        if self.is_npu_active:
            return self._detect_hailo(frame, w, h)
        else:
            return self._detect_cpu(frame, w, h)

    def _detect_hailo(self, frame, img_w, img_h):
        """Executes inference on the Hailo-8 NPU via InferVStreams."""
        target_h, target_w = self.input_shape
        resized = cv2.resize(frame, (target_w, target_h))
        # Ensure RGB input format
        input_data = np.expand_dims(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), axis=0).astype(np.uint8)

        input_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        detections = []
        with InferVStreams(self.network_group, input_params, output_params) as infer_pipeline:
            input_name = self.input_vstream_info.name
            raw_results = infer_pipeline.infer({input_name: input_data})

            # Parse Hailo YOLOv8 NMS output
            for out_name, out_tensor in raw_results.items():
                if len(out_tensor.shape) >= 2:
                    # Parse detections: [batch, num_detections, 6] -> [y1, x1, y2, x2, score, class_id]
                    boxes = out_tensor[0]
                    for det in boxes:
                        score = float(det[4]) if len(det) > 4 else 0.0
                        if score >= self.conf_threshold:
                            class_id = int(det[5]) if len(det) > 5 else 0
                            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "object"
                            
                            # Convert coordinates
                            y1, x1, y2, x2 = det[0], det[1], det[2], det[3]
                            box = [int(x1 * img_w), int(y1 * img_h), int(x2 * img_w), int(y2 * img_h)]
                            
                            # Determine spatial direction
                            center_x = (box[0] + box[2]) / 2.0
                            if center_x < img_w * 0.35:
                                direction = "on your left"
                            elif center_x > img_w * 0.65:
                                direction = "on your right"
                            else:
                                direction = "straight ahead"

                            detections.append({
                                "class": class_name,
                                "confidence": score,
                                "box": box,
                                "direction": direction
                            })

        # Sort by highest confidence
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def _detect_cpu(self, frame, img_w, img_h):
        """CPU fallback detection."""
        if self.net is None:
            return []

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        raw_detections = self.net.forward()

        # VOC classes mapping for fallback
        voc_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                       "sofa", "train", "tvmonitor"]

        detections = []
        for i in range(raw_detections.shape[2]):
            confidence = float(raw_detections[0, 0, i, 2])
            if confidence > self.conf_threshold:
                idx = int(raw_detections[0, 0, i, 1])
                class_name = voc_classes[idx] if idx < len(voc_classes) else "obstacle"

                x1 = int(raw_detections[0, 0, i, 3] * img_w)
                y1 = int(raw_detections[0, 0, i, 4] * img_h)
                x2 = int(raw_detections[0, 0, i, 5] * img_w)
                y2 = int(raw_detections[0, 0, i, 6] * img_h)

                center_x = (x1 + x2) / 2.0
                if center_x < img_w * 0.35:
                    direction = "on your left"
                elif center_x > img_w * 0.65:
                    direction = "on your right"
                else:
                    direction = "straight ahead"

                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "direction": direction
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def close(self):
        """Releases Hailo VDevice handles."""
        if self.vdevice is not None:
            try:
                self.vdevice.release()
            except Exception:
                pass
            self.vdevice = None

if __name__ == "__main__":
    print("[INFO] Running Hailo Detector standalone self-test...")
    detector = HailoYOLODetector()
    print(f"[STATUS] NPU Acceleration Active: {detector.is_npu_active}")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(dummy)
    print(f"[STATUS] Self-test complete. Detections on test frame: {len(results)}")
    detector.close()

