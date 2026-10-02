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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CPU-Optimized YOLOv8 Detector
from yolo_detector import YOLODetector, YOLO_ONNX_PATH

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
    """Speaks text asynchronously using espeak without overlapping."""
    global tts_process
    print(f"🔊 SPEAKING: {text}")

    if tts_process is not None:
        try:
            if tts_process.poll() is None:
                tts_process.terminate()
        except Exception:
            pass

    try:
        tts_process = subprocess.Popen(['espeak', '-ven+f3', '-s155', text])
    except FileNotFoundError:
        print("[WARN] 'espeak' is not installed. Run: sudo apt install espeak")
    except Exception as e:
        print(f"[WARN] TTS error: {e}")

def stop_tts():
    """Terminates active speech upon exit."""
    global tts_process
    if tts_process is not None:
        try:
            if tts_process.poll() is None:
                tts_process.terminate()
                tts_process.wait(timeout=1.0)
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
    """
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.grabbed, self.frame = self.cap.read()
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.005)
                continue
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def is_opened(self):
        return self.cap.isOpened()

    def stop(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()


# ==========================================
# 4. MAIN APPLICATION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 CPU YOLOv8 Smart Vision Assistant")
    parser.add_argument("--model", type=str, default=YOLO_ONNX_PATH, help="Path to YOLOv8 ONNX model")
    parser.add_argument("--resolution", type=int, default=320, choices=[320, 640],
                        help="YOLO input resolution: 320 for high FPS (~15-20 FPS on Pi 5), 640 for max accuracy")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    parser.add_argument("--classes", type=str, default=None,
                        help="Custom classes: comma-separated list or path to a text file containing class names")
    args = parser.parse_args()

    # Parse custom classes if provided
    custom_classes = None
    if args.classes:
        if os.path.exists(args.classes):
            with open(args.classes, "r") as f:
                custom_classes = [line.strip() for line in f if line.strip()]
        else:
            custom_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        print(f"[INFO] Loaded {len(custom_classes)} custom classes: {custom_classes}")

    print("\n" + "=" * 65)
    print("  Raspberry Pi 5 Smart Vision Assistant (CPU-Optimized YOLOv8)")
    print(f"  Configuration: Resolution {args.resolution}x{args.resolution} | Confidence {args.conf}")
    print("=" * 65 + "\n")

    # Initialize CPU YOLOv8 Detector
    detector = YOLODetector(
        model_path=args.model,
        classes=custom_classes,
        conf_threshold=args.conf,
        input_size=(args.resolution, args.resolution)
    )

    # Initialize Camera
    print("[INFO] Starting Threaded Camera Stream...")
    cam = ThreadedCameraStream(src=0, width=640, height=480)
    time.sleep(1.5)  # Warmup

    if not cam.is_opened():
        print("[ERROR] Failed to open camera device. Please verify camera connection.")
        sys.exit(1)

    speak("AI vision system ready.")

    last_scan_time = 0.0
    COOLDOWN = 3.0  # Cooldown between voice alerts
    frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            # Run YOLOv8 detection on CPU
            detections = detector.detect(frame)

            # Read proximity sensors
            distance_meters = get_distance()
            button_pressed = is_button_pressed()

            # Trigger condition: Obstacle < 1m OR Button pressed
            time_since_last_alert = time.time() - last_scan_time
            if (distance_meters < 1.0 or button_pressed) and (time_since_last_alert > COOLDOWN):
                trigger_reason = "Button Pressed" if button_pressed else f"Proximity Alert ({distance_meters:.2f}m)"
                print(f"\n[TRIGGER] {trigger_reason}")

                if detections:
                    top_det = detections[0]
                    obj_name = top_det["class"]
                    direction = top_det["direction"]

                    if distance_meters <= 3.0:
                        announcement = f"{obj_name} detected {direction} at {distance_meters:.1f} meters."
                    else:
                        announcement = f"{obj_name} detected {direction}."
                    speak(announcement)
                else:
                    speak("Obstacle ahead, but cannot identify object.")

                last_scan_time = time.time()

            # Graphical Telemetry Display
            if not args.headless:
                cpu_load = get_cpu_utilization()

                # Draw bounding boxes
                for det in detections:
                    box = det["box"]
                    label = f"{det['class']} {int(det['confidence'] * 100)}%"
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, label, (box[0], max(20, box[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw Telemetry HUD
                cv2.putText(frame, f"Mode: CPU YOLOv8 ({args.resolution}x{args.resolution})",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {current_fps:.1f} | CPU: {cpu_load:.1f}%",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Distance: {distance_meters:.2f}m",
                            (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
