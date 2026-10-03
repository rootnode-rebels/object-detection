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

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
        self.temp_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_temp_tts.wav")
        self._sapi_voice = None
        self._sapi_stream = None
        if sys.platform == "win32":
            try:
                import win32com.client
                self._sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                self._sapi_stream = win32com.client.Dispatch("SAPI.SpFileStream")
            except Exception:
                pass

    def speak(self, text):
        threading.Thread(target=self._speak_worker, args=(text,), daemon=True).start()

    def _speak_worker(self, text):
        with self.lock:
            try:
                print(f"[VOICE] SPEAKING: {text}")
            except Exception:
                pass
            # Windows Native SAPI via SpFileStream + winsound
            if sys.platform == "win32":
                clean_text = text.replace('"', '').replace("'", "")
                # Method 1: SAPI stream to temp wav + winsound (tested 100% working on Windows)
                try:
                    import winsound
                    if self._sapi_voice and self._sapi_stream:
                        self._sapi_stream.Open(self.temp_wav, 3, False)
                        self._sapi_voice.AudioOutputStream = self._sapi_stream
                        self._sapi_voice.Speak(clean_text)
                        self._sapi_stream.Close()
                        winsound.PlaySound(self.temp_wav, winsound.SND_FILENAME)
                        try:
                            if os.path.exists(self.temp_wav):
                                os.remove(self.temp_wav)
                        except Exception:
                            pass
                        return
                except Exception:
                    pass

                # Method 2: Fallback powershell System.Speech
                try:
                    ps_cmd = f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{clean_text}')"
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


def get_resource_path(relative_path):
    """Get absolute path to resource, works for source script and for PyInstaller frozen exe."""
    # 1. PyInstaller bundled temp folder
    if hasattr(sys, '_MEIPASS'):
        p = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(p):
            return p
    # 2. Next to executable
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    p = os.path.join(exe_dir, relative_path)
    if os.path.exists(p):
        return p
    # 3. Next to script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(script_dir, relative_path)
    if os.path.exists(p):
        return p
    return relative_path


# ==========================================
# 2. PC SIMULATOR APPLICATION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="PC Simulator for Assistive Vision Device")
    parser.add_argument("--model", type=str, default=None, help="Path to custom ONNX model (optional)")
    parser.add_argument("--classes", type=str, default=None, help="Path to classes.txt file")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--resolution", type=int, default=320, help="Input resolution: 320 for high FPS")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    args = parser.parse_args()

    # Auto-detect assistive 92 model and classes
    model_to_use = args.model
    if model_to_use is None:
        for candidate in [
            get_resource_path("model_files/yolov8_assistive_92.onnx"),
            get_resource_path("model_files/yolov8n.onnx"),
            "model_files/yolov8_assistive_92.onnx",
            "model_files/yolov8n.onnx"
        ]:
            if os.path.exists(candidate):
                model_to_use = candidate
                break

    classes_file = args.classes
    if classes_file is None or not os.path.exists(classes_file):
        for candidate in [get_resource_path("classes.txt"), "classes.txt"]:
            if os.path.exists(candidate):
                classes_file = candidate
                break

    # Load custom classes if available
    custom_classes = None
    if classes_file and os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            custom_classes = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(custom_classes)} custom classes from {classes_file}")

    # Initialize Voice Engine
    voice = VoiceSynthesizer()

    # Initialize Detector
    print("[INFO] Initializing YOLOv8 Object Detector...")
    detector = None
    if YOLO_AVAILABLE:
        try:
            detector = YOLODetector(
                model_path=model_to_use,
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
            target_sector = None

            if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                break
            elif key == ord(' '):  # SPACEBAR = Push Button (All Sectors)
                manual_trigger = True
            elif key in [ord('l'), ord('L')]:  # Query Left Sector
                manual_trigger = True
                target_sector = "left"
            elif key in [ord('m'), ord('M')]:  # Query Middle Sector
                manual_trigger = True
                target_sector = "middle"
            elif key in [ord('r'), ord('R')]:  # Query Right Sector
                manual_trigger = True
                target_sector = "right"
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
                if target_sector:
                    trigger_reason = f"Query {target_sector.upper()} Sector"
                elif manual_trigger:
                    trigger_reason = "Button Pressed (Full Spatial Scan)"
                else:
                    trigger_reason = f"Proximity Alert ({distance_meters:.2f}m)"

                print(f"\n[TRIGGER] {trigger_reason}")

                # Format spatial announcement across Left, Middle, and Right
                from yolo_detector import format_spatial_announcement
                current_announcement = format_spatial_announcement(
                    detections,
                    distance_meters=distance_meters,
                    target_zone=target_sector
                )

                voice.speak(current_announcement)
                last_scan_time = current_time

            # ---------------------------------------------
            # DRAW TELEMETRY & SPATIAL HUD OVERLAY
            # ---------------------------------------------
            h, w = frame.shape[:2]
            x_left = int(w * 0.35)
            x_right = int(w * 0.65)

            # 1. Spatial Zone Dividers (Left | Middle | Right)
            cv2.line(frame, (x_left, 85), (x_left, h - 45), (70, 80, 95), 1, cv2.LINE_AA)
            cv2.line(frame, (x_right, 85), (x_right, h - 45), (70, 80, 95), 1, cv2.LINE_AA)

            # Detect count per zone
            left_count = sum(1 for d in detections if d.get("zone") == "left")
            mid_count = sum(1 for d in detections if d.get("zone") == "middle")
            right_count = sum(1 for d in detections if d.get("zone") == "right")

            # Zone Header Badges
            # Left Header
            l_color = (56, 189, 248) if left_count > 0 else (120, 130, 140)
            cv2.putText(frame, f"[ LEFT: {left_count} ]", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, l_color, 1, cv2.LINE_AA)

            # Middle Header
            m_color = (0, 0, 255) if (mid_count > 0 and distance_meters < 1.0) else ((0, 220, 255) if mid_count > 0 else (120, 130, 140))
            cv2.putText(frame, f"[ MIDDLE: {mid_count} ]", (x_left + 15, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, m_color, 1, cv2.LINE_AA)

            # Right Header
            r_color = (56, 189, 248) if right_count > 0 else (120, 130, 140)
            cv2.putText(frame, f"[ RIGHT: {right_count} ]", (x_right + 15, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_color, 1, cv2.LINE_AA)

            # 2. Bounding Boxes
            for det in detections:
                box = det["box"]
                z = det.get("zone", "middle")
                if z == "middle":
                    color = (0, 80, 255)  # Orange/Red for center obstacle
                elif z == "left":
                    color = (255, 180, 0)  # Cyan/Blue
                else:
                    color = (0, 220, 100)  # Green

                label = f"{det['class']} ({z})"
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, label, (box[0], max(20, box[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 3. Top Banner
            banner_color = (15, 23, 42)
            cv2.rectangle(frame, (0, 0), (w, 85), banner_color, -1)
            cv2.line(frame, (0, 85), (w, 85), (56, 189, 248), 1)

            # Status indicators
            dist_color = (0, 0, 255) if distance_meters < 1.0 else (0, 255, 0)
            status_text = "PROXIMITY ALERT (<1.0m)" if distance_meters < 1.0 else "CLEAR PATH"
            cv2.putText(frame, f"DISTANCE: {distance_meters:.2f}m [{status_text}]",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, dist_color, 2)

            walk_mode_str = "AUTO-WALK [ON]" if auto_walk else "MANUAL [W/S]"
            cv2.putText(frame, f"FPS: {fps:.1f} | Mode: {walk_mode_str} | [SPACE]: Full Scan",
                        (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, "Hotkeys: [L] Left | [M] Middle | [R] Right | [W/S] Dist | [A] Auto | [Q] Quit",
                        (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 175, 190), 1)

            # Radar Distance Indicator Bar
            bar_w = int((distance_meters / 3.0) * (w - 30))
            cv2.rectangle(frame, (15, 74), (15 + bar_w, 80), dist_color, -1)
            cv2.rectangle(frame, (15, 74), (w - 15, 80), (60, 65, 75), 1)

            # 4. Bottom Voice Feedback Transcript Banner
            cv2.rectangle(frame, (0, h - 45), (w, h), (15, 23, 42), -1)
            cv2.line(frame, (0, h - 45), (w, h - 45), (0, 255, 180), 1)
            cv2.putText(frame, f"VOICE: \"{current_announcement}\"",
                        (15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 180), 1)

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
