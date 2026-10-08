"""
Smart Vision & Obstacle Detection Assistant
Runs on Raspberry Pi 5 (CPU-Optimized, No NPU Required).
- Real-Time YOLOv8 Object Detection (80 COCO classes) via OpenCV DNN
- Ultrasonic Proximity Alert (HC-SR04) & Tactile Button Trigger
- Non-Blocking Spatial Voice Announcements (espeak)
- Real-Time HUD with Live FPS, CPU Load %, and Distance
"""

import cv2
import time
import os
import sys
import subprocess
import threading
import argparse

# Ensure current directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# CPU-Optimized YOLOv8 Detector & Spatial Formatter
from yolo_detector import YOLODetector, format_spatial_announcement

# Safe GPIO import for Raspberry Pi
try:
    from gpiozero import DistanceSensor, Button
    GPIO_AVAILABLE = True
except (ImportError, Exception) as e:
    GPIO_AVAILABLE = False
    DistanceSensor = None
    Button = None
    print(f"[WARN] gpiozero not available ({e}). Running in simulation/desktop mode.")

# Optional CPU / System Telemetry
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ==========================================
# 1. HARDWARE PIN CONFIGURATION
# ==========================================
PIN_TRIGGER = 23
PIN_ECHO = 24
PIN_BUTTON = 17

sensor = None
button = None

print("[INFO] Initializing Hardware Sensors...")
if GPIO_AVAILABLE:
    try:
        sensor = DistanceSensor(echo=PIN_ECHO, trigger=PIN_TRIGGER, max_distance=3.0)
        button = Button(PIN_BUTTON, pull_up=True)
        print("[INFO] GPIO ultrasonic sensor & button initialized successfully.")
    except Exception as e:
        print(f"[WARN] Could not initialize GPIO pins. Error: {e}")
        sensor = None
        button = None
else:
    print("[INFO] GPIO unavailable. Running in desktop/simulation mode.")

def get_distance():
    """Safely reads the distance from the ultrasonic sensor in meters."""
    if sensor is not None:
        try:
            dist = sensor.distance
            if dist is not None and 0.0 <= dist <= 3.0:
                return dist
        except Exception:
            pass
    return 3.0

def is_button_pressed():
    """Safely checks whether the hardware push button is pressed."""
    if button is not None:
        try:
            return bool(button.is_pressed)
        except Exception:
            return False
    return False

def get_cpu_utilization():
    """Returns current system CPU utilization percentage."""
    if PSUTIL_AVAILABLE:
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0
    try:
        with open("/proc/loadavg", "r") as f:
            load = float(f.read().split()[0])
            return min(100.0, load * 25.0)  # Approximate for 4-core Pi 5
    except Exception:
        return 0.0


# ==========================================
# 2. TEXT-TO-SPEECH (NON-BLOCKING)
# ==========================================
tts_process = None

def speak(text):
    """Speaks text asynchronously using espeak-ng/espeak (Linux/Pi) or native OS speech without blocking."""
    global tts_process
    if not text:
        return

    # Security: sanitize text to strictly allowed printable characters and punctuation
    clean_text = "".join(ch for ch in text if ch.isalnum() or ch in " ,.!?:;'-()\"")
    if not clean_text.strip():
        return

    try:
        print(f"[VOICE] SPEAKING: {clean_text}")
    except Exception:
        pass

    if sys.platform == "win32":
        def _win_speak():
            sanitized = clean_text.replace('"', '').replace("'", "")
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pass

            try:
                import win32com.client
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                voice.Speak(sanitized, 1 | 2)
                return
            except Exception:
                pass

            # Fallback
            try:
                import winsound
                import win32com.client
                import tempfile
                temp_wav = os.path.join(tempfile.gettempdir(), "_temp_assistant_tts.wav")
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                stream = win32com.client.Dispatch("SAPI.SpFileStream")
                stream.Open(temp_wav, 3, False)
                voice.AudioOutputStream = stream
                voice.Speak(sanitized, 0)
                stream.Close()
                winsound.PlaySound(temp_wav, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_PURGE)
                return
            except Exception:
                pass

        threading.Thread(target=_win_speak, daemon=True).start()
        return

    # Linux / Raspberry Pi: Clean up prior process and reap zombies
    if tts_process is not None:
        try:
            if tts_process.poll() is None:
                tts_process.terminate()
                try:
                    tts_process.wait(timeout=0.15)
                except Exception:
                    pass
            else:
                # Reap finished child process to prevent zombie accumulation
                tts_process.wait(timeout=0.05)
        except Exception:
            pass

    # Launch new speech without shell=True for strict command-injection prevention
    for engine in ['espeak-ng', 'espeak']:
        try:
            tts_process = subprocess.Popen([engine, '-ven+f3', '-s160', '-p50', clean_text])
            return
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"[WARN] TTS engine '{engine}' error: {e}")
            break

    if sys.platform == "darwin":
        try:
            tts_process = subprocess.Popen(["say", clean_text])
            return
        except Exception:
            pass

    print("[WARN] Neither 'espeak-ng' nor 'espeak' is installed. Run: sudo apt install espeak-ng -y")

def stop_tts():
    """Terminates active speech upon exit."""
    global tts_process
    if tts_process is not None:
        try:
            if tts_process.poll() is None:
                tts_process.terminate()
                tts_process.wait(timeout=0.5)
        except Exception:
            pass
        tts_process = None


# ==========================================
# 3. THREADED CAMERA STREAM (ZERO BUFFER LAG)
# ==========================================
class ThreadedCameraStream:
    """
    Continuously grabs frames in a background thread.
    Fixes OpenCV buffer delay so read() always returns the immediate live frame.
    Includes V4L2 acceleration on Raspberry Pi and automatic device fallback.
    """
    def __init__(self, src=0, width=640, height=480):
        self.cap = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

        # Try specified camera and fallback indices if needed
        candidate_indices = [src]
        for idx in [0, 1, 2]:
            if idx not in candidate_indices:
                candidate_indices.append(idx)

        for dev_idx in candidate_indices:
            try:
                if sys.platform.startswith("linux"):
                    self.cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
                else:
                    self.cap = cv2.VideoCapture(dev_idx)

                if self.cap is not None and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    grabbed, frame = self.cap.read()
                    if grabbed and frame is not None:
                        self.grabbed = grabbed
                        self.frame = frame
                        print(f"[INFO] Camera stream successfully opened on device index {dev_idx}")
                        break
                    else:
                        self.cap.release()
            except Exception:
                if self.cap is not None:
                    try: self.cap.release()
                    except Exception: pass

        if self.cap is not None and self.cap.isOpened():
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        else:
            print(f"[WARN] Could not open camera on indices {candidate_indices}. Check connection.")

    def _update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.02)
                continue
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed = bool(grabbed and frame is not None)
                if self.grabbed:
                    self.frame = frame
            if not grabbed:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened() and self.grabbed

    def stop(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


# ==========================================
# 4. MAIN APPLICATION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 CPU YOLOv8 Smart Vision Assistant")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLOv8 ONNX model")
    parser.add_argument("--resolution", type=int, default=320, choices=[320, 640],
                        help="YOLO input resolution: 320 for high FPS (~15-20 FPS on Pi 5), 640 for max accuracy")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    parser.add_argument("--classes", type=str, default=None,
                        help="Custom classes: comma-separated list or path to a text file containing class names")
    args = parser.parse_args()

    # Auto-detect assistive 92 model if not specified (script-relative path)
    default_model_path = os.path.join(BASE_DIR, "model_files", "yolov8_assistive_92.onnx")
    model_to_use = args.model
    if model_to_use is None and os.path.exists(default_model_path):
        model_to_use = default_model_path

    # Parse custom classes if provided, or auto-load classes.txt for assistive model
    default_classes_path = os.path.join(BASE_DIR, "classes.txt")
    classes_file = args.classes
    if classes_file is None and model_to_use == default_model_path and os.path.exists(default_classes_path):
        classes_file = default_classes_path

    custom_classes = None
    if classes_file:
        if os.path.exists(classes_file):
            with open(classes_file, "r") as f:
                custom_classes = [line.strip() for line in f if line.strip()]
        else:
            custom_classes = [c.strip() for c in classes_file.split(",") if c.strip()]
        print(f"[INFO] Loaded {len(custom_classes)} custom classes from {classes_file}")

    print("\n" + "=" * 65)
    print("  Raspberry Pi 5 Smart Vision Assistant (CPU-Optimized YOLOv8)")
    print(f"  Configuration: Resolution {args.resolution}x{args.resolution} | Confidence {args.conf}")
    print("=" * 65 + "\n")

    # Initialize CPU YOLOv8 Detector
    detector = YOLODetector(
        model_path=model_to_use,
        classes=custom_classes,
        conf_threshold=args.conf,
        input_size=(args.resolution, args.resolution)
    )

    if detector.net is None:
        err_msg = "Error: AI vision model failed to load. Please check model files."
        print(f"[ERROR] {err_msg}")
        speak(err_msg)
        time.sleep(3.5)
        sys.exit(1)

    # Initialize Camera
    print("[INFO] Starting Threaded Camera Stream...")
    cam = ThreadedCameraStream(src=args.camera, width=640, height=480)
    time.sleep(1.5)  # Warmup

    if not cam.is_opened():
        err_msg = "Warning: camera device not detected. Please verify camera cable connection."
        print(f"[ERROR] {err_msg}")
        speak(err_msg)
        time.sleep(3.5)
        sys.exit(1)

    startup_msg = "Smart vision assistant ready. Obstacle radar active."
    speak(startup_msg)

    last_scan_time = time.time()
    last_button_time = 0.0
    last_valid_frame_time = time.time()
    camera_lost_alerted = False
    current_announcement = startup_msg
    COOLDOWN = 3.0  # Cooldown between voice alerts
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                if time.time() - last_valid_frame_time > 3.0:
                    if not camera_lost_alerted:
                        speak("Warning: camera feed lost. Please stop and check camera connection.")
                        camera_lost_alerted = True
                time.sleep(0.01)
                continue

            last_valid_frame_time = time.time()
            if camera_lost_alerted:
                speak("Camera feed restored. Obstacle detection resumed.")
                camera_lost_alerted = False

            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Read proximity sensors first for multi-modal sensor fusion
            distance_meters = get_distance()
            button_raw = is_button_pressed()
            button_pressed = False
            if button_raw and (time.time() - last_button_time > 1.8):
                button_pressed = True
                last_button_time = time.time()

            # Run YOLOv8 detection on CPU with multi-frame approach tracking
            detections = detector.detect(frame, distance_meters=distance_meters)

            # Trigger condition: Obstacle detected in camera view OR distance < 1.0m OR Button pressed
            has_obstacle = (len(detections) > 0) or (distance_meters < 1.0)
            has_approaching = any(d.get("is_approaching", False) for d in detections)
            is_emergency = has_approaching or (distance_meters < 1.0)
            time_since_last_alert = time.time() - last_scan_time

            # Format announcement: tactile button triggers comprehensive on-demand scan
            candidate_announcement = format_spatial_announcement(
                detections,
                distance_meters=distance_meters,
                is_full_scan=button_pressed
            )
            state_changed = (candidate_announcement != current_announcement)

            # Fast response for blind safety:
            # - Instant alert on button press or emergency (approaching hazard / <1.0m)
            # - Fast 1.2s debouncing on normal navigation state changes
            # - Periodic reminder cooldown if obstacle persists
            should_alert = button_pressed or (state_changed and (is_emergency or time_since_last_alert > 1.2)) or (has_obstacle and time_since_last_alert > COOLDOWN)

            if should_alert:
                current_announcement = candidate_announcement
                if button_pressed:
                    trigger_reason = "Tactile Button Pressed (Full Surroundings Scan)"
                elif has_approaching:
                    app_names = [d["class"] for d in detections if d.get("is_approaching")]
                    trigger_reason = f"Approaching Hazard: {', '.join(app_names)}"
                elif len(detections) > 0:
                    trigger_reason = f"Obstacle Alert ({detections[0]['class']} in view)"
                elif distance_meters < 1.0:
                    trigger_reason = f"Proximity Alert ({distance_meters:.2f}m)"
                else:
                    trigger_reason = "Clear Path Navigation Update"

                print(f"\n[TRIGGER] {trigger_reason} -> \"{current_announcement}\"")
                speak(current_announcement)
                last_scan_time = time.time()

            # Graphical Telemetry Display
            if not args.headless:
                h, w = frame.shape[:2]
                x_left = int(w * 0.35)
                x_right = int(w * 0.65)
                cpu_load = get_cpu_utilization()

                # Draw 9-Part Spatial Grid (Top 3, Middle 3, Bottom 3)
                col1 = int(w / 3.0)
                col2 = int((2.0 * w) / 3.0)
                row1 = int(h / 3.0)
                row2 = int((2.0 * h) / 3.0)

                grid_color = (60, 70, 80)
                cv2.line(frame, (col1, 0), (col1, h), grid_color, 1, cv2.LINE_AA)
                cv2.line(frame, (col2, 0), (col2, h), grid_color, 1, cv2.LINE_AA)
                cv2.line(frame, (0, row1), (w, row1), grid_color, 1, cv2.LINE_AA)
                cv2.line(frame, (0, row2), (w, row2), grid_color, 1, cv2.LINE_AA)

                # Draw bounding boxes colored by approaching status and spatial zone
                for det in detections:
                    box = det["box"]
                    z = det.get("zone", "middle-center")
                    is_app = det.get("is_approaching", False)
                    app_spd = det.get("approach_speed")

                    if is_app:
                        color = (0, 0, 255) if app_spd == "fast" else (0, 140, 255)
                        label = f"[APPROACHING] {det['class']} ({app_spd or 'steady'})"
                    elif "middle" in z or "center" in z:
                        color = (0, 80, 255)  # Orange/Red for center/ahead
                        label = f"{det['class']} ({z})"
                    elif "left" in z:
                        color = (255, 180, 0)  # Cyan/Blue
                        label = f"{det['class']} ({z})"
                    else:
                        color = (0, 220, 100)  # Green
                        label = f"{det['class']} ({z})"

                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, label, (box[0], max(20, box[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                # Draw Telemetry HUD
                cv2.putText(frame, f"Mode: CPU YOLOv8 ({args.resolution}x{args.resolution})",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {current_fps:.1f} | CPU: {cpu_load:.1f}%",
                            (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
                dist_color = (0, 0, 255) if distance_meters < 1.0 else (0, 255, 255)
                cv2.putText(frame, f"Distance: {distance_meters:.2f}m",
                            (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.52, dist_color, 2)
                app_badge = "RADAR: APPROACHING HAZARD!" if has_approaching else "RADAR: MONITORING"
                app_badge_color = (0, 0, 255) if has_approaching else (200, 200, 200)
                cv2.putText(frame, app_badge,
                            (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.45, app_badge_color, 1)

                # Bottom Voice Transcript Banner
                cv2.rectangle(frame, (0, h - 35), (w, h), (20, 25, 35), -1)
                cv2.putText(frame, f"VOICE: \"{current_announcement}\"",
                            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)

                cv2.imshow("Raspberry Pi 5 Object Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted. Shutting down...")
    finally:
        print("[INFO] Cleaning up resources...")
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()
        if sensor is not None:
            try:
                sensor.close()
            except Exception:
                pass
        if button is not None:
            try:
                button.close()
            except Exception:
                pass
        stop_tts()
        print("[INFO] Cleanup complete. System safely halted.")

if __name__ == "__main__":
    main()
