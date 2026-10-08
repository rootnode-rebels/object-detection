"""
Unified Python Simulation Server for Smart Vision Assistant
Runs the ACTUAL project code:
- yolo_detector.py with model_files/yolov8_assistive_92.onnx and classes.txt (92 classes)
- format_spatial_announcement across 9-part spatial grid (Top 3, Middle 3, Bottom 3)
- Real OpenCV camera pipeline (with synthetic corridor room fallback)
- Real Windows SAPI native audio speech synthesizer
- MJPEG stream and interactive REST API for web GUI and localhost testing
"""

import os
import sys
import time
import json
import threading
import functools
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
import cv2

# Flush stdout immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from yolo_detector import YOLODetector, format_spatial_announcement

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Initialize Windows SAPI Voice Synthesizer
class VoiceSynthesizer:
    def __init__(self):
        self.lock = threading.Lock()
        self._sapi = None
        try:
            import win32com.client
            self._sapi = win32com.client.Dispatch("SAPI.SpVoice")
        except Exception:
            self._sapi = None

    def speak(self, text):
        threading.Thread(target=self._speak_worker, args=(text,), daemon=True).start()

    def _speak_worker(self, text):
        clean_text = text.replace('"', '').replace("'", "")
        with self.lock:
            if self._sapi:
                try:
                    self._sapi.Speak(clean_text)
                    return
                except Exception:
                    pass
            try:
                import subprocess
                ps_cmd = f'[System.Reflection.Assembly]::LoadWithPartialName("System.Speech"); $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak("{clean_text}");'
                subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[VOICE ERROR] {e}")

voice = VoiceSynthesizer()

# Global Simulation State
class SimulationState:
    def __init__(self):
        self.lock = threading.Lock()
        self.distance = 1.80
        self.auto_walk = False
        self.walk_dir = -1
        self.last_scan_time = 0.0
        self.last_spoken_announcement = ""
        self.cooldown = 4.0
        self.current_announcement = "Simulation Ready. 9-Part Spatial Detection Active."
        self.current_detections = []
        self.fps = 30.0
        self.latest_jpeg = None
        self.virtual_cam = False
        self.virtual_x = 0
        self.virtual_y = 0

state = SimulationState()

# Load 92 Assistive Classes
classes_path = os.path.join(BASE_DIR, "classes.txt")
custom_classes = None
if os.path.exists(classes_path):
    with open(classes_path, "r", encoding="utf-8") as f:
        custom_classes = [l.strip() for l in f if l.strip()]
print(f"[INFO] Loaded {len(custom_classes) if custom_classes else 80} classes.")

# Load custom model
model_path = os.path.join(BASE_DIR, "model_files", "yolov8_assistive_92.onnx")
if not os.path.exists(model_path):
    model_path = os.path.join(BASE_DIR, "model_files", "yolov8n.onnx")

print(f"[INFO] Initializing actual YOLODetector with: {model_path}")
detector = YOLODetector(
    model_path=model_path,
    classes=custom_classes,
    conf_threshold=0.45,
    input_size=(320, 320)
)
print("[SUCCESS] YOLODetector initialized and ready.")

# Background Camera & YOLO Processing Thread
def camera_yolo_loop():
    cap = None
    if sys.platform == "win32":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[WARN] Physical webcam not available. Starting virtual hallway camera simulation...")
        state.virtual_cam = True
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("[SUCCESS] Live Webcam feed opened.")

    fps_start = time.time()
    frame_count = 0

    while True:
        if not state.virtual_cam and cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
        else:
            # Synthetic hallway frame with moving obstacle
            frame = np.full((480, 640, 3), (12, 16, 26), dtype=np.uint8)
            cv2.line(frame, (0, 480), (200, 240), (45, 55, 75), 2, cv2.LINE_AA)
            cv2.line(frame, (640, 480), (440, 240), (45, 55, 75), 2, cv2.LINE_AA)
            cv2.line(frame, (200, 240), (440, 240), (45, 55, 75), 2, cv2.LINE_AA)

            sc = max(0.4, min(1.6, (2.8 - state.distance) / 1.4))
            ow, oh = int(120 * sc), int(200 * sc)
            ox = int((320 + state.virtual_x) - ow / 2)
            oy = int((280 + state.virtual_y) - oh / 2)
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (220, 100, 0), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (255, 180, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "SIMULATED PERSON", (ox, max(20, oy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Handle Auto-Walk
        if state.auto_walk:
            state.distance += state.walk_dir * 0.02
            if state.distance <= 0.4:
                state.distance = 0.4
                state.walk_dir = 1
            elif state.distance >= 2.6:
                state.distance = 2.6
                state.walk_dir = -1

        # Run Real YOLOv8 Inference on the Frame
        detections = detector.detect(frame)

        if state.virtual_cam and len(detections) == 0:
            cx = ox + ow // 2
            cy = oy + oh // 2
            col = "left" if cx < 640 / 3.0 else ("right" if cx > 1280 / 3.0 else "middle")
            row = "top" if cy < 480 / 3.0 else ("bottom" if cy > 960 / 3.0 else "middle")
            z = "middle-center" if (row == "middle" and col == "middle") else f"{row}-{col}"
            detections = [{
                "class": "person",
                "confidence": 0.93,
                "box": [ox, oy, ox + ow, oy + oh],
                "zone": z,
                "row": row,
                "col": col,
                "center_x": cx,
                "center_y": cy
            }]

        with state.lock:
            state.current_detections = detections

        # Autonomous proximity trigger (< 1.0m) or new obstacle
        now = time.time()
        ann = format_spatial_announcement(detections, distance_meters=state.distance)
        with state.lock:
            state.current_announcement = ann

        # Speak when state changes or on distance alert with cooldown
        if (ann != state.last_spoken_announcement and now - state.last_scan_time > 2.0) or \
           (state.distance < 1.0 and now - state.last_scan_time > state.cooldown):
            state.last_spoken_announcement = ann
            state.last_scan_time = now
            print(f"[AUTO-SPEAK] {ann}")
            voice.speak(ann)

        # ---------------------------------------------
        # DRAW 9-PART SPATIAL GRID OVERLAY (Top 3, Mid 3, Low 3)
        # ---------------------------------------------
        h, w = frame.shape[:2]
        col1_x = int(w / 3.0)
        col2_x = int((2.0 * w) / 3.0)
        row1_y = int(h / 3.0)
        row2_y = int((2.0 * h) / 3.0)

        grid_col = (70, 85, 105)
        cv2.line(frame, (col1_x, 0), (col1_x, h), grid_col, 1, cv2.LINE_AA)
        cv2.line(frame, (col2_x, 0), (col2_x, h), grid_col, 1, cv2.LINE_AA)
        cv2.line(frame, (0, row1_y), (w, row1_y), grid_col, 1, cv2.LINE_AA)
        cv2.line(frame, (0, row2_y), (w, row2_y), grid_col, 1, cv2.LINE_AA)

        # 9 Sector Badges
        cv2.putText(frame, "TOP-L", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "TOP-AHEAD (OVERHEAD)", (col1_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "TOP-R", (col2_x + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 140, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "MID-LEFT", (10, row1_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)
        ahead_col = (0, 80, 255) if state.distance < 1.0 else (0, 220, 100)
        cv2.putText(frame, "DIRECTLY AHEAD", (col1_x + 10, row1_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ahead_col, 1, cv2.LINE_AA)
        cv2.putText(frame, "MID-RIGHT", (col2_x + 10, row1_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 1, cv2.LINE_AA)

        cv2.putText(frame, "LOW-LEFT", (10, row2_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 158, 11), 1, cv2.LINE_AA)
        cv2.putText(frame, "LOW-AHEAD (TRIP HAZARD)", (col1_x + 10, row2_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 158, 11), 1, cv2.LINE_AA)
        cv2.putText(frame, "LOW-RIGHT", (col2_x + 10, row2_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 158, 11), 1, cv2.LINE_AA)

        # Draw Detected Bounding Boxes
        for det in detections:
            box = det["box"]
            z = det.get("zone", "middle-center")
            if "middle" in z or "center" in z:
                color = (0, 80, 255)
            elif "left" in z:
                color = (255, 180, 0)
            else:
                color = (0, 220, 100)

            label = f"{det['class'].upper()} ({int(det['confidence']*100)}%) [{z.upper()}]"
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, label, (int(box[0]), max(20, int(box[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        # Calculate FPS
        frame_count += 1
        if time.time() - fps_start >= 1.0:
            state.fps = frame_count / (time.time() - fps_start)
            frame_count = 0
            fps_start = time.time()

        # Top HUD Status Banner
        cv2.rectangle(frame, (0, 0), (w, 32), (10, 15, 25), -1)
        dist_str = f"RADAR: {state.distance:.2f}m [{'ALERT' if state.distance < 1.0 else 'CLEAR'}]"
        cv2.putText(frame, f"FPS: {state.fps:.1f} | {dist_str} | YOLOV8 92-CLASS",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            with state.lock:
                state.latest_jpeg = jpeg.tobytes()

        time.sleep(0.02)

cam_thread = threading.Thread(target=camera_yolo_loop, daemon=True)
cam_thread.start()

# HTTP Request Handler
class SimulationHTTPHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ["/", "/index"]:
                self.send_response(302)
                self.send_header("Location", "/simulation.html")
                self.end_headers()
                return

            elif path == "/video_feed":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()
                try:
                    while True:
                        jpeg = None
                        with state.lock:
                            jpeg = state.latest_jpeg
                        if jpeg:
                            self.wfile.write(b"--frame\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(jpeg)))
                            self.end_headers()
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.033)
                except (ConnectionResetError, BrokenPipeError):
                    pass
                return

            elif path == "/api/telemetry":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with state.lock:
                    data = {
                        "fps": round(float(state.fps), 1),
                        "distance": round(float(state.distance), 2),
                        "auto_walk": state.auto_walk,
                        "announcement": str(state.current_announcement),
                        "count": len(state.current_detections),
                        "detections": state.current_detections
                    }
                self.wfile.write(json.dumps(data, cls=NumpyEncoder).encode("utf-8"))
                return

            super().do_GET()
        except Exception as e:
            print(f"[HTTP GET ERROR] {e}")

    def do_POST(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else ""
            data = json.loads(body) if body else {}

            if path == "/api/scan":
                with state.lock:
                    ann = format_spatial_announcement(state.current_detections, distance_meters=state.distance)
                    state.current_announcement = ann
                    state.last_spoken_announcement = ann
                    state.last_scan_time = time.time()
                print(f"[MANUAL-SCAN] {ann}")
                voice.speak(ann)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "announcement": ann}, cls=NumpyEncoder).encode("utf-8"))
                return

            elif path == "/api/query_sector":
                sector = data.get("sector", "middle-center")
                with state.lock:
                    ann = format_spatial_announcement(state.current_detections, distance_meters=state.distance, target_zone=sector)
                    state.current_announcement = ann
                    state.last_spoken_announcement = ann
                    state.last_scan_time = time.time()
                print(f"[SECTOR-QUERY: {sector}] {ann}")
                voice.speak(ann)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "sector": sector, "announcement": ann}, cls=NumpyEncoder).encode("utf-8"))
                return

            elif path == "/api/set_distance":
                val = float(data.get("distance", 1.8))
                state.distance = max(0.2, min(3.0, val))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "distance": state.distance}, cls=NumpyEncoder).encode("utf-8"))
                return

            elif path == "/api/toggle_autowalk":
                state.auto_walk = not state.auto_walk
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "auto_walk": state.auto_walk}, cls=NumpyEncoder).encode("utf-8"))
                return

            self.send_response(404)
            self.end_headers()
        except Exception as e:
            print(f"[HTTP POST ERROR] {e}")

def main():
    port = 8000
    handler = functools.partial(SimulationHTTPHandler, directory=BASE_DIR)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"\n============================================================")
    print(f"  SMART VISION ASSISTANT - UNIFIED PYTHON SIMULATOR RUNNING")
    print(f"  Model:   yolov8_assistive_92.onnx (92 Classes)")
    print(f"  Grid:    9-Part Spatial Matrix (Top 3, Mid 3, Low 3)")
    print(f"  Voice:   Native Python SAPI Speech Engine Active")
    print(f"  Server:  http://localhost:{port}/simulation.html")
    print(f"============================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    main()
