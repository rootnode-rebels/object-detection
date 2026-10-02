"""
Interactive PC Simulation & Demo for Assistive Vision Device
Runs directly on Windows / Mac / Linux PC (No Raspberry Pi or physical sensors required).

Features:
- Live Webcam feed with real-time YOLOv8 object detection HUD
- Simulated HC-SR04 Ultrasonic Distance Sensor (Keyboard controllable + Auto-walk mode)
- Simulated Hardware Tactile Push Button (Spacebar trigger)
- Native Windows/macOS/Linux Voice Audio Feedback (Speaks aloud)
- Spatial Awareness (Left / Straight Ahead / Right) with distance alerts

Controls:
  [SPACE]     : Press virtual hardware push button (triggers on-demand scan)
  [W] or [↑]  : Move closer to obstacle (decreases distance)
  [S] or [↓]  : Move farther from obstacle (increases distance)
  [A]         : Toggle Auto-walk simulation mode (oscillates distance)
  [Q] or [ESC]: Quit simulation
"""

import cv2
import time
import os
import sys
import threading
import subprocess
import argparse
import numpy as np

# Import YOLO detector
try:
    from yolo_detector import YOLODetector, YOLO_ONNX_PATH, COCO_CLASSES
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ==========================================
# 1. CROSS-PLATFORM VOICE FEEDBACK (NO ESPEAK REQUIRED)
# ==========================================
class VoiceSynthesizer:
    """Speaks text aloud using native OS speech engines without blocking the video stream."""
    def __init__(self):
        self.lock = threading.Lock()

    def speak(self, text):
        threading.Thread(target=self._speak_worker, args=(text,), daemon=True).start()

    def _speak_worker(self, text):
        with self.lock:
            print(f"🔊 SPEAKING: {text}")
            # Windows Native SAPI (zero dependencies)
            if sys.platform == "win32":
                clean_text = text.replace('"', '').replace("'", "")
                ps_cmd = f'(New-Object -ComObject SAPI.SpVoice).Speak("{clean_text}")'
                try:
                    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                                   capture_output=True, timeout=5)
                    return
                except Exception:
                    pass

            # macOS native say command
            elif sys.platform == "darwin":
                try:
                    subprocess.run(["say", text], capture_output=True, timeout=5)
                    return
                except Exception:
                    pass

            # Linux espeak command
            else:
                try:
                    subprocess.run(["espeak", "-ven+f3", "-s155", text], capture_output=True, timeout=5)
                    return
                except Exception:
                    pass


# ==========================================
# 2. PC SIMULATOR APPLICATION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="PC Simulator for Assistive Vision Device")
    parser.add_argument("--model", type=str, default=None, help="Path to custom ONNX model (optional)")
    parser.add_argument("--classes", type=str, default="classes.txt", help="Path to classes.txt file")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--resolution", type=int, default=320, help="Input resolution: 320 for high FPS")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    args = parser.parse_args()

    # Load custom classes if available
    custom_classes = None
    if args.classes and os.path.exists(args.classes):
        with open(args.classes, "r") as f:
            custom_classes = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(custom_classes)} custom classes from {args.classes}")

    # Initialize Voice Engine
    voice = VoiceSynthesizer()

    # Initialize Detector
    print("[INFO] Initializing YOLOv8 Object Detector...")
    detector = None
    if YOLO_AVAILABLE:
        try:
            detector = YOLODetector(
                model_path=args.model,
                classes=custom_classes,
                conf_threshold=args.conf,
                input_size=(args.resolution, args.resolution)
            )
            print("[SUCCESS] YOLOv8 Detector ready.")
        except Exception as e:
            print(f"[WARN] Could not initialize YOLOv8: {e}. Running in camera demo mode.")

    # Initialize Webcam
    print(f"[INFO] Opening webcam device {args.cam}...")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {args.cam}. Please verify your camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Simulation State
    distance_meters = 1.80
    auto_walk = False
    walk_direction = -1  # moving closer
    last_scan_time = 0.0
    COOLDOWN = 3.0
    current_announcement = "Simulation Ready. Press SPACE to query or W/S to adjust distance."

    voice.speak("Assistive vision simulation ready.")

    fps_start = time.time()
    frame_count = 0
    fps = 30.0

    print("\n" + "=" * 65)
    print("  ASSISTIVE VISION PC SIMULATOR RUNNING")
    print("  Controls:")
    print("    [SPACE]     : Push Button Trigger (Manual Query)")
    print("    [W] / [UP]  : Move closer to obstacle")
    print("    [S] / [DOWN]: Move farther from obstacle")
    print("    [A]         : Toggle Auto-walk radar simulation")
    print("    [Q] / [ESC] : Quit Simulation")
    print("=" * 65 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            if time.time() - fps_start >= 1.0:
                fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()

            # Handle Auto-walk mode
            if auto_walk:
                distance_meters += walk_direction * 0.03
                if distance_meters <= 0.4:
                    distance_meters = 0.4
                    walk_direction = 1
                elif distance_meters >= 2.5:
                    distance_meters = 2.5
                    walk_direction = -1

            # Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            manual_trigger = False

            if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                break
            elif key == ord(' '):  # SPACEBAR = Push Button
                manual_trigger = True
            elif key in [ord('w'), ord('W'), 82]:  # W or UP arrow
                distance_meters = max(0.2, distance_meters - 0.15)
            elif key in [ord('s'), ord('S'), 84]:  # S or DOWN arrow
                distance_meters = min(3.0, distance_meters + 0.15)
            elif key in [ord('a'), ord('A')]:  # Toggle auto-walk
                auto_walk = not auto_walk
                print(f"[SIM] Auto-walk mode: {'ENABLED' if auto_walk else 'DISABLED'}")

            # Run detection on current frame
            detections = []
            if detector is not None:
                detections = detector.detect(frame)

            # Trigger condition: Proximity < 1.0m OR Manual Button Trigger
            current_time = time.time()
            if (distance_meters < 1.0 or manual_trigger) and (current_time - last_scan_time > COOLDOWN):
                trigger_reason = "Button Pressed" if manual_trigger else f"Proximity Alert ({distance_meters:.2f}m)"
                print(f"\n[TRIGGER] {trigger_reason}")

                if detections:
                    top = detections[0]
                    obj_name = top["class"]
                    direction = top["direction"]
                    current_announcement = f"{obj_name} detected {direction} at {distance_meters:.1f} meters."
                else:
                    current_announcement = "Obstacle ahead, but cannot identify object."

                voice.speak(current_announcement)
                last_scan_time = current_time

            # ---------------------------------------------
            # DRAW TELEMETRY & HUD OVERLAY
            # ---------------------------------------------
            h, w = frame.shape[:2]

            # 1. Bounding Boxes
            for det in detections:
                box = det["box"]
                label = f"{det['class']} {int(det['confidence'] * 100)}%"
                color = (56, 189, 248)  # Cyan
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, label, (box[0], max(20, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 2. Top Banner
            banner_color = (15, 23, 42)
            cv2.rectangle(frame, (0, 0), (w, 85), banner_color, -1)
            cv2.line(frame, (0, 85), (w, 85), (56, 189, 248), 1)

            # Status indicators
            dist_color = (0, 0, 255) if distance_meters < 1.0 else (0, 255, 0)
            status_text = "PROXIMITY ALERT (<1.0m)" if distance_meters < 1.0 else "CLEAR PATH"
            cv2.putText(frame, f"DISTANCE: {distance_meters:.2f}m [{status_text}]",
                        (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dist_color, 2)

            walk_mode_str = "AUTO-WALK [ON]" if auto_walk else "MANUAL [W/S]"
            cv2.putText(frame, f"FPS: {fps:.1f} | Mode: {walk_mode_str} | [SPACE]: Push Button",
                        (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Radar Distance Bar
            bar_w = int((distance_meters / 3.0) * (w - 30))
            cv2.rectangle(frame, (15, 65), (15 + bar_w, 75), dist_color, -1)
            cv2.rectangle(frame, (15, 65), (w - 15, 75), (80, 80, 80), 1)

            # 3. Bottom Voice Feedback Transcript Banner
            cv2.rectangle(frame, (0, h - 45), (w, h), (15, 23, 42), -1)
            cv2.line(frame, (0, h - 45), (w, h - 45), (0, 255, 0), 1)
            cv2.putText(frame, f"SPEECH: \"{current_announcement}\"",
                        (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 180), 1)

            cv2.imshow("Smart Vision Assistant - PC Simulator", frame)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if detector is not None:
            detector.close()
        print("[INFO] PC Simulation ended cleanly.")


if __name__ == "__main__":
    main()
