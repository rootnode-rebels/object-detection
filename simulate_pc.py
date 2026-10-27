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
import tempfile
import numpy as np
import math
import re

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# Import YOLO detector & Spatial Formatter
try:
    from yolo_detector import YOLODetector, YOLO_ONNX_PATH, COCO_CLASSES, format_spatial_announcement
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ==========================================
# 1. CROSS-PLATFORM VOICE FEEDBACK (NO ESPEAK REQUIRED)
# ==========================================
class VoiceSynthesizer:
    """Speaks text aloud using native OS speech engines with calm pacing and anti-overload protection."""
    def __init__(self):
        self.lock = threading.Lock()
        self.is_speaking_flag = False
        self.last_speech_finished = 0.0
        self._sapi_voice = None
        self.temp_wav = os.path.join(tempfile.gettempdir(), "_smart_assistant_tts.wav")

    def is_speaking(self):
        with self.lock:
            return self.is_speaking_flag

    def time_since_finished(self):
        with self.lock:
            if self.is_speaking_flag:
                return 0.0
            if self.last_speech_finished == 0.0:
                return 999.0
            return time.time() - self.last_speech_finished

    def speak(self, text, force=False):
        if not text:
            return
        if not force and self.is_speaking():
            return
        threading.Thread(target=self._speak_worker, args=(text,), daemon=True).start()

    def _speak_worker(self, text):
        clean_text = text.replace('"', '').replace("'", "").strip()
        if not clean_text:
            return

        with self.lock:
            self.is_speaking_flag = True

        try:
            print(f"[VOICE] SPEAKING: {text}")
        except Exception:
            pass

        try:
            # Windows Native SAPI with comfortable conversational tempo
            if sys.platform == "win32":
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                    import win32com.client
                    voice = win32com.client.Dispatch("SAPI.SpVoice")
                    self._sapi_voice = voice
                    try:
                        voice.Rate = -2  # Calm, comfortable ~130 wpm
                        voice.Volume = 100
                    except Exception:
                        pass
                    # Synchronous speak (0) ensures is_speaking remains True until speech finishes
                    voice.Speak(clean_text, 0)
                    return
                except Exception:
                    pass

                # Fallback using async winsound
                try:
                    import winsound
                    import win32com.client
                    voice = win32com.client.Dispatch("SAPI.SpVoice")
                    try: voice.Rate = -2
                    except Exception: pass
                    stream = win32com.client.Dispatch("SAPI.SpFileStream")
                    stream.Open(self.temp_wav, 3, False)
                    voice.AudioOutputStream = stream
                    voice.Speak(clean_text, 0)
                    stream.Close()
                    winsound.PlaySound(self.temp_wav, winsound.SND_FILENAME | winsound.SND_PURGE)
                    return
                except Exception:
                    pass

                # Method 2: Fallback powershell System.Speech
                try:
                    ps_cmd = f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = -2; $s.Speak('{clean_text}')"
                    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                                   capture_output=True, timeout=10)
                    return
                except Exception:
                    pass

            # macOS native say command with rate 135
            elif sys.platform == "darwin":
                try:
                    subprocess.run(["say", "-r", "135", clean_text], capture_output=True, timeout=10)
                    return
                except Exception:
                    pass

            # Linux espeak command with steady 130 wpm
            else:
                for eng in ['espeak-ng', 'espeak']:
                    try:
                        subprocess.run([eng, "-ven-us", "-s130", "-p48", clean_text], capture_output=True, timeout=12)
                        return
                    except Exception:
                        pass
        finally:
            with self.lock:
                self.is_speaking_flag = False
                self.last_speech_finished = time.time()


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
# 2. HARDWARE SERIAL SONAR BRIDGE (OPTIONAL)
# ==========================================
class SerialSonarBridge:
    """Reads real-time distance packets from an external microcontroller (Arduino/ESP32) running sonar_firmware.ino."""
    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self.latest_distance = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        try:
            import serial
            self.ser = serial.Serial(self.port, self.baud, timeout=1.0)
            self.running = True
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
            print(f"[SUCCESS] Serial Sonar Bridge active on {self.port} @ {self.baud} baud.")
            return True
        except Exception as e:
            print(f"[WARN] Serial Sonar Bridge failed to connect to {self.port}: {e}")
            return False

    def _worker(self):
        while self.running:
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                m = re.search(r"[-+]?(?:\d*\.\d+|\d+)", line)
                if m:
                    val = float(m.group())
                    if val > 10.0:
                        val = val / 100.0  # Convert cm to meters
                    if 0.02 <= val <= 3.5:
                        with self.lock:
                            self.latest_distance = round(val, 2)
            except Exception:
                time.sleep(0.04)

    def get_distance(self):
        with self.lock:
            return self.latest_distance

    def stop(self):
        self.running = False
        try:
            if hasattr(self, 'ser') and self.ser is not None:
                self.ser.close()
        except Exception:
            pass


def draw_sonar_radar_widget(frame, distance_m, radar_angle, is_hardware=False):
    """
    Renders an acoustic sonar radar sweep display in the upper right quadrant of the HUD.
    Features:
    - 3 Concentric acoustic distance ranges (1.0m, 2.0m, 3.0m).
    - Rotating sweep beam simulating active ultrasonic pulses.
    - Blip dot indicating detected obstacle position.
    """
    h, w = frame.shape[:2]
    cx, cy = w - 85, 42
    r_max = 34

    # Background radar disc
    cv2.circle(frame, (cx, cy), r_max, (10, 20, 32), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r_max, (45, 65, 85), 1, cv2.LINE_AA)

    # Range rings for 1.0m, 2.0m, 3.0m
    cv2.circle(frame, (cx, cy), int(r_max * 0.33), (25, 45, 65), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), int(r_max * 0.66), (30, 55, 75), 1, cv2.LINE_AA)

    # Crosshairs
    cv2.line(frame, (cx - r_max, cy), (cx + r_max, cy), (35, 55, 75), 1)
    cv2.line(frame, (cx, cy - r_max), (cx, cy + r_max), (35, 55, 75), 1)

    # Rotating acoustic sweep ray
    rad = math.radians(radar_angle)
    sx = int(cx + r_max * math.cos(rad))
    sy = int(cy + r_max * math.sin(rad))
    cv2.line(frame, (cx, cy), (sx, sy), (0, 255, 180), 1, cv2.LINE_AA)

    # Obstacle blip on radar
    if distance_m is not None and 0.0 < distance_m <= 3.0:
        ratio = min(1.0, distance_m / 3.0)
        blip_r = int(ratio * (r_max - 4))
        bx = cx
        by = cy - blip_r
        blip_color = (0, 0, 255) if distance_m < 1.0 else ((0, 165, 255) if distance_m < 2.0 else (0, 255, 100))
        cv2.circle(frame, (bx, by), 4, blip_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 6, blip_color, 1, cv2.LINE_AA)

    # Widget Tag
    tag = "HW SONAR" if is_hardware else "SONAR RADAR"
    tag_color = (0, 255, 200) if is_hardware else (140, 180, 200)
    cv2.putText(frame, tag, (cx - 28, cy + r_max + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.30, tag_color, 1, cv2.LINE_AA)


# Proximity beep state
_last_beep_time = 0.0

def play_sonar_beep(dist, is_muted=False):
    """Triggers an acoustic proximity click/beep when an obstacle is within 1.0 meter."""
    global _last_beep_time
    if is_muted or dist is None or dist >= 1.0:
        return
    now = time.time()
    # Beep faster when closer: interval scales from 0.15s (very close) to 0.5s (1.0m)
    beep_interval = max(0.12, dist * 0.45)
    if now - _last_beep_time >= beep_interval:
        _last_beep_time = now
        if sys.platform == "win32":
            def _beep_worker():
                try:
                    import winsound
                    freq = int(1400 - (dist * 500))
                    winsound.Beep(max(400, min(2400, freq)), 35)
                except Exception:
                    pass
            threading.Thread(target=_beep_worker, daemon=True).start()


# ==========================================
# 3. PC SIMULATOR APPLICATION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="PC Simulator for Assistive Vision Device")
    parser.add_argument("--model", type=str, default=None, help="Path to custom ONNX model (optional)")
    parser.add_argument("--classes", type=str, default=None, help="Path to classes.txt file")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--resolution", type=int, default=320, help="Input resolution: 320 for high FPS")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--sonar-port", type=str, default=None, help="Serial COM port for USB HC-SR04 Arduino bridge (e.g. COM3 or /dev/ttyUSB0)")
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

    # Initialize Webcam with Virtual Camera Fallback
    print(f"[INFO] Opening webcam device {args.cam}...")
    cap = None
    if sys.platform == "win32":
        cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(args.cam)
    virtual_cam = False

    if not cap.isOpened():
        print(f"[WARN] Physical webcam {args.cam} not accessible (missing or in use).")
        print("[INFO] Starting Virtual Synthetic Camera Feed for offline PC simulation...")
        virtual_cam = True
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Prepare Display Window & Bring to Front
    WINDOW_TITLE = "Smart Vision Assistant - PC Simulator"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, 960, 720)
    try:
        cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    # Initialize Hardware Serial Sonar Bridge if requested
    serial_sonar = None
    if args.sonar_port:
        serial_sonar = SerialSonarBridge(args.sonar_port)
        if not serial_sonar.start():
            serial_sonar = None

    # Simulation State
    distance_meters = 1.80
    virtual_offset_x = 0  # Horizontal shift for simulated obstacle (-180 to +180)
    auto_walk = False
    walk_direction = -1  # moving closer
    radar_angle = 0
    sonar_beep_muted = False
    last_scan_time = time.time()
    COOLDOWN = 3.0
    current_announcement = "Simulation Ready. Automatic obstacle detection active."

    voice.speak("Assistive vision simulation ready.")

    fps_start = time.time()
    frame_count = 0
    fps = 30.0

    print("\n" + "=" * 65)
    print("  ASSISTIVE VISION PC SIMULATOR RUNNING")
    print(f"  Camera Mode   : {'VIRTUAL SYNTHETIC SCENE' if virtual_cam else 'LIVE WEBCAM'}")
    print(f"  Sonar Mode    : {'HARDWARE BRIDGE (' + args.sonar_port + ')' if serial_sonar else 'SIMULATED ACOUSTIC RADAR'}")
    print("  Auto Voice    : ENABLED (Speaks navigation cues in real time)")
    print("  Controls:")
    print("    [SPACE]     : Push Button Trigger (Manual Surroundings Query)")
    print("    [W] / [UP]  : Decrease Sonar Distance (Move closer to obstacle)")
    print("    [S] / [DOWN]: Increase Sonar Distance (Move farther from obstacle)")
    print("    [A]         : Toggle Sonar Auto-Walk Radar Mode")
    print("    [B]         : Toggle Sonar Proximity Acoustic Beep (Mute/Unmute)")
    print("    [0]         : Reset Sonar Distance to 2.0m")
    print("    [J] / [LEFT]: Shift obstacle to LEFT")
    print("    [K] / [RGHT]: Shift obstacle to RIGHT")
    print("    [C]         : Center obstacle")
    print("    [Q] / [ESC] : Quit Simulation")
    print("=" * 65 + "\n")

    try:
        while True:
            if not virtual_cam:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
            else:
                # Render simulated indoor hallway scene
                frame = np.full((480, 640, 3), (25, 30, 42), dtype=np.uint8)
                # Perspective room grid lines
                cv2.line(frame, (0, 480), (200, 240), (50, 65, 80), 2, cv2.LINE_AA)
                cv2.line(frame, (640, 480), (440, 240), (50, 65, 80), 2, cv2.LINE_AA)
                cv2.line(frame, (200, 240), (440, 240), (50, 65, 80), 2, cv2.LINE_AA)
                cv2.line(frame, (200, 0), (200, 240), (40, 50, 65), 1)
                cv2.line(frame, (440, 0), (440, 240), (40, 50, 65), 1)

                # Simulated obstacle size and horizontal position
                sc = max(0.4, min(1.8, (2.8 - distance_meters) / 1.4))
                ow, oh = int(120 * sc), int(200 * sc)
                ox, oy = int((320 + virtual_offset_x) - ow / 2), int(280 - oh / 2)
                cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (220, 100, 0), -1, cv2.LINE_AA)
                cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (255, 180, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, "SIMULATED PERSON", (ox - 10, max(20, oy - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                time.sleep(0.03)

            frame_count += 1
            if time.time() - fps_start >= 1.0:
                fps = frame_count / (time.time() - fps_start)
                frame_count = 0
                fps_start = time.time()

            # Update Sonar Radar sweep angle and distance
            radar_angle = (radar_angle + 8) % 360
            if serial_sonar is not None:
                hw_dist = serial_sonar.get_distance()
                if hw_dist is not None:
                    distance_meters = hw_dist
            elif auto_walk:
                distance_meters += walk_direction * 0.03
                if distance_meters <= 0.4:
                    distance_meters = 0.4
                    walk_direction = 1
                elif distance_meters >= 2.5:
                    distance_meters = 2.5
                    walk_direction = -1

            # Acoustic proximity warning beep for visually impaired guidance
            play_sonar_beep(distance_meters, is_muted=sonar_beep_muted)

            # Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            manual_trigger = False
            target_sector = None

            if key in [ord('q'), ord('Q'), 27]:  # Q or ESC
                break
            elif key == ord(' '):  # SPACEBAR = Push Button (All Sectors)
                manual_trigger = True
            elif key in [ord('l'), ord('L'), ord('4')]:  # Query Mid-Left
                manual_trigger = True
                target_sector = "middle-left"
            elif key in [ord('m'), ord('M'), ord('5')]:  # Query Directly Ahead
                manual_trigger = True
                target_sector = "middle-center"
            elif key in [ord('r'), ord('R'), ord('6')]:  # Query Mid-Right
                manual_trigger = True
                target_sector = "middle-right"
            elif key == ord('7'):  # Query Top-Left
                manual_trigger = True
                target_sector = "top-left"
            elif key == ord('8'):  # Query Top-Ahead (Overhead)
                manual_trigger = True
                target_sector = "top-middle"
            elif key == ord('9'):  # Query Top-Right
                manual_trigger = True
                target_sector = "top-right"
            elif key == ord('1'):  # Query Low-Left
                manual_trigger = True
                target_sector = "bottom-left"
            elif key == ord('2'):  # Query Low-Ahead (Trip Hazard)
                manual_trigger = True
                target_sector = "bottom-middle"
            elif key == ord('3'):  # Query Low-Right
                manual_trigger = True
                target_sector = "bottom-right"
            elif key in [ord('w'), ord('W'), 82]:  # W or UP arrow (decrease distance)
                distance_meters = max(0.2, distance_meters - 0.15)
            elif key in [ord('s'), ord('S'), 84]:  # S or DOWN arrow (increase distance)
                distance_meters = min(3.0, distance_meters + 0.15)
            elif key in [ord('b'), ord('B')]:  # B = Toggle Sonar Beep
                sonar_beep_muted = not sonar_beep_muted
                print(f"[SONAR] Proximity Beep: {'MUTED' if sonar_beep_muted else 'ACTIVE'}")
            elif key == ord('0'):  # 0 = Reset Sonar to 2.0m
                distance_meters = 2.0
            elif key in [ord('j'), ord('J'), 81]:  # J or LEFT arrow
                virtual_offset_x = max(-180, virtual_offset_x - 30)
            elif key in [ord('k'), ord('K'), 83]:  # K or RIGHT arrow
                virtual_offset_x = min(180, virtual_offset_x + 30)
            elif key in [ord('c'), ord('C')]:  # C = Center
                virtual_offset_x = 0
            elif key in [ord('a'), ord('A')]:  # Toggle auto-walk
                auto_walk = not auto_walk
                print(f"[SIM] Sonar Auto-walk mode: {'ENABLED' if auto_walk else 'DISABLED'}")

            # Run detection on current frame
            detections = []
            if detector is not None:
                detections = detector.detect(frame, distance_meters=distance_meters)
            if virtual_cam and len(detections) == 0:
                sim_center_x = ox + ow // 2
                sim_center_y = oy + oh // 2

                # 9-zone classification for simulated frame
                col = "left" if sim_center_x < 213 else ("right" if sim_center_x > 426 else "middle")
                row = "top" if sim_center_y < 160 else ("bottom" if sim_center_y > 320 else "middle")
                sim_zone = f"{row}-{col}" if (row != "middle" or col != "middle") else "middle-center"

                sim_dir = "directly ahead" if sim_zone == "middle-center" else f"{row} {col}"
                sim_det = [{
                    "class": "person",
                    "confidence": 0.92,
                    "box": [ox, oy, ox + ow, oy + oh],
                    "direction": sim_dir,
                    "zone": sim_zone,
                    "center": (sim_center_x, sim_center_y),
                    "center_x": sim_center_x,
                    "center_y": sim_center_y
                }]
                if detector is not None:
                    detections = detector.tracker.update(sim_det, distance_meters=distance_meters)
                else:
                    detections = sim_det

            # ---------------------------------------------
            # AUTOMATIC REAL-TIME OBSTACLE VOICE GUIDANCE
            # ---------------------------------------------
            # Automatically speaks navigation commands whenever an obstacle is detected!
            has_obstacle = (len(detections) > 0) or (distance_meters < 1.0)
            has_approaching = any(d.get("is_approaching", False) for d in detections)
            is_emergency = has_approaching or (distance_meters < 1.0)
            current_time = time.time()

            candidate_announcement = format_spatial_announcement(
                detections,
                distance_meters=distance_meters,
                target_zone=target_sector
            )

            # Speak automatically on obstacle detection, zone change, clear-path update, or manual trigger
            time_since = current_time - last_scan_time
            state_changed = (candidate_announcement != current_announcement)
            time_since_finished = voice.time_since_finished()
            is_speaking = voice.is_speaking()

            # Prevent voice acceleration & buffer cutoffs: enforce post-speech silence window
            # Emergency needs >= 1.8s silence after previous speech finished; standard needs >= 2.5s
            min_post_pause = 1.8 if is_emergency else 2.5
            pacing_ready = (not is_speaking) and (time_since_finished >= min_post_pause)

            should_alert = manual_trigger or (pacing_ready and ((state_changed and (is_emergency or time_since > 2.0)) or (has_obstacle and time_since > COOLDOWN)))

            if should_alert:
                current_announcement = candidate_announcement
                if target_sector:
                    trigger_reason = f"Query {target_sector.upper()} Sector"
                elif manual_trigger:
                    trigger_reason = "Button Pressed (Manual Scan)"
                elif has_approaching:
                    app_names = [d["class"] for d in detections if d.get("is_approaching")]
                    trigger_reason = f"Approaching Hazard: {', '.join(app_names).upper()}"
                elif len(detections) > 0:
                    obs_class = detections[0]['class']
                    obs_zone = detections[0].get('zone', 'path')
                    trigger_reason = f"Obstacle Alert: {obs_class} ({obs_zone.upper()})"
                elif distance_meters < 1.0:
                    trigger_reason = f"Proximity Alert ({distance_meters:.2f}m)"
                else:
                    trigger_reason = "Clear Path Navigation Update"

                print(f"\n[TRIGGER] {trigger_reason} -> \"{current_announcement}\"")
                voice.speak(current_announcement, force=manual_trigger)
                last_scan_time = current_time

            # ---------------------------------------------
            # DRAW TELEMETRY & SPATIAL HUD OVERLAY
            # ---------------------------------------------
            h, w = frame.shape[:2]
            x_left = int(w * 0.35)
            x_right = int(w * 0.65)

            # 1. 9-Part Spatial Grid Matrix (Top 3, Middle 3, Bottom 3)
            col1_x = int(w / 3.0)
            col2_x = int((2.0 * w) / 3.0)
            row1_y = int(85 + (h - 130) / 3.0)
            row2_y = int(85 + (2.0 * (h - 130)) / 3.0)

            # Grid lines (2 vertical, 2 horizontal)
            grid_color = (60, 75, 95)
            cv2.line(frame, (col1_x, 85), (col1_x, h - 45), grid_color, 1, cv2.LINE_AA)
            cv2.line(frame, (col2_x, 85), (col2_x, h - 45), grid_color, 1, cv2.LINE_AA)
            cv2.line(frame, (0, row1_y), (w, row1_y), grid_color, 1, cv2.LINE_AA)
            cv2.line(frame, (0, row2_y), (w, row2_y), grid_color, 1, cv2.LINE_AA)

            # 9 Grid Sector Labels
            cv2.putText(frame, "TOP-L [7]", (15, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 150, 165), 1, cv2.LINE_AA)
            cv2.putText(frame, "TOP-AHEAD (OVERHEAD) [8]", (col1_x + 10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "TOP-R [9]", (col2_x + 10, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 150, 165), 1, cv2.LINE_AA)

            cv2.putText(frame, "MID-LEFT [4/L]", (15, row1_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (56, 189, 248), 1, cv2.LINE_AA)
            cv2.putText(frame, "DIRECTLY AHEAD [5/M]", (col1_x + 10, row1_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 80, 255) if distance_meters < 1.0 else (0, 220, 100), 1, cv2.LINE_AA)
            cv2.putText(frame, "MID-RIGHT [6/R]", (col2_x + 10, row1_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (56, 189, 248), 1, cv2.LINE_AA)

            cv2.putText(frame, "LOW-LEFT [1]", (15, row2_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 150, 165), 1, cv2.LINE_AA)
            cv2.putText(frame, "LOW-AHEAD (TRIP HAZARD) [2]", (col1_x + 10, row2_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 140, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "LOW-RIGHT [3]", (col2_x + 10, row2_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 150, 165), 1, cv2.LINE_AA)

            # 2. Bounding Boxes
            for det in detections:
                box = det["box"]
                z = det.get("zone", "middle")
                is_app = det.get("is_approaching", False)
                app_spd = det.get("approach_speed")

                if is_app:
                    color = (0, 0, 255) if app_spd == "fast" else (0, 140, 255)
                    label = f"[APPROACHING] {det['class']} ({app_spd or 'steady'})"
                elif z == "middle" or "center" in z:
                    color = (0, 80, 255)  # Orange/Red for center obstacle
                    label = f"{det['class']} ({z})"
                elif "left" in z:
                    color = (255, 180, 0)  # Cyan/Blue
                    label = f"{det['class']} ({z})"
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

            # Draw Real-Time Sonar Acoustic Radar Widget
            draw_sonar_radar_widget(frame, distance_meters, radar_angle, is_hardware=(serial_sonar is not None))

            # Status indicators
            dist_color = (0, 0, 255) if distance_meters < 1.0 else (0, 255, 0)
            status_text = "PROXIMITY HAZARD (<1.0m)" if distance_meters < 1.0 else "CLEAR PATH"
            sonar_source = f"HARDWARE ({args.sonar_port})" if serial_sonar else ("AUTO-WALK [A]" if auto_walk else "SIMULATED [W/S]")
            cv2.putText(frame, f"HC-SR04 SONAR: {distance_meters:.2f}m [{status_text}]",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, dist_color, 2)

            radar_status = "[APPROACHING HAZARD!]" if has_approaching else "[RADAR OK]"
            radar_color = (0, 0, 255) if has_approaching else (0, 255, 180)
            beep_state = "MUTED" if sonar_beep_muted else "ON"
            cv2.putText(frame, f"FPS: {fps:.1f} | Source: {sonar_source} | Beep: {beep_state} | Status: {radar_status}",
                        (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, radar_color if has_approaching else (200, 200, 200), 1)
            cv2.putText(frame, "Hotkeys: [W/S] Sonar Dist | [A] Auto-Walk | [B] Sonar Beep | [0] Reset | [SPACE] Scan | [Q] Quit",
                        (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 175, 190), 1)

            # Radar Distance Indicator Bar
            bar_w = int((distance_meters / 3.0) * (w - 180))
            cv2.rectangle(frame, (15, 74), (15 + bar_w, 80), dist_color, -1)
            cv2.rectangle(frame, (15, 74), (w - 165, 80), (60, 65, 75), 1)

            # 4. Bottom Voice Feedback Transcript Banner
            cv2.rectangle(frame, (0, h - 45), (w, h), (15, 23, 42), -1)
            cv2.line(frame, (0, h - 45), (w, h - 45), (0, 255, 180), 1)
            cv2.putText(frame, f"VOICE: \"{current_announcement}\"",
                        (15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 180), 1)

            cv2.imshow(WINDOW_TITLE, frame)

    except KeyboardInterrupt:
        pass
    finally:
        if serial_sonar is not None:
            serial_sonar.stop()
        if cap is not None and not virtual_cam:
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        if detector is not None:
            detector.close()
        print("[INFO] PC Simulation ended cleanly.")


if __name__ == "__main__":
    main()
