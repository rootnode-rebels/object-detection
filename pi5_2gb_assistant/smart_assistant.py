"""
Smart Vision & Obstacle Detection Assistant
Runs on Raspberry Pi 5 (CPU-Optimized, No NPU Required).
- Real-Time YOLOv8 Object Detection (80 COCO classes) via OpenCV DNN
- Ultrasonic Proximity Alert (HC-SR04) & Tactile Button Trigger
- Non-Blocking Spatial Voice Announcements (espeak)
- Real-Time HUD with Live FPS, CPU Load %, and Distance
"""

import os
import sys

# Crucial memory optimization for 2GB RAM Raspberry Pi 5:
# Prevent glibc multi-threaded memory arena fragmentation (caps arenas to 2 instead of 32)
os.environ["MALLOC_ARENA_MAX"] = "2"

import cv2
import time
import subprocess
import threading
import argparse
import gc
import warnings
import glob
import re

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
sonar_disabled = True
sonar_reading = 3.0
sonar_thread = None
sonar_thread_running = False

def _sonar_latest_filter(queue):
    """
    Zero-hysteresis filter for HC-SR04 sonar queue.
    Prevents gpiozero from holding stale distances on open space or when an obstacle moves.
    """
    if not queue:
        return 1.0  # 1.0 * max_distance = 3.0m (clear)
    latest = queue[-1]
    if latest is None:
        return 1.0  # Echo timed out or open space -> immediately clear
    return latest

def _sonar_worker():
    """Background daemon thread to poll ultrasonic sensor without blocking the camera vision loop."""
    global sensor, sonar_reading, sonar_disabled, sonar_thread_running
    while sonar_thread_running and sensor is not None:
        try:
            d = sensor.distance
            if d is None or d >= 2.95:
                # Normal out-of-range (>3m) or clear open space
                sonar_reading = 3.0
            elif 0.02 <= d < 2.95:
                sonar_reading = round(float(d), 2)
                sonar_disabled = False
        except Exception:
            time.sleep(0.04)
        time.sleep(0.05)  # Sample at ~20 Hz asynchronously

def init_sensors(trigger_pin=23, echo_pin=24, button_pin=17, enable_sonar=True):
    """
    Initializes hardware GPIO sensors (HC-SR04 ultrasonic distance sensor & tactile button).
    Keeps sensor active continuously without falsely disabling on clear room startup.
    """
    global sensor, button, PIN_TRIGGER, PIN_ECHO, PIN_BUTTON, sonar_disabled, sonar_reading, sonar_thread, sonar_thread_running
    PIN_TRIGGER = trigger_pin
    PIN_ECHO = echo_pin
    PIN_BUTTON = button_pin

    if not GPIO_AVAILABLE:
        print("[INFO] GPIO unavailable. Running in desktop/simulation mode.")
        sonar_disabled = True
        return

    if sensor is not None:
        try: sensor.close()
        except Exception: pass
        sensor = None
    if button is not None:
        try: button.close()
        except Exception: pass
        button = None

    sonar_thread_running = False

    if enable_sonar:
        try:
            candidate_sensor = DistanceSensor(
                echo=PIN_ECHO,
                trigger=PIN_TRIGGER,
                max_distance=3.0,
                queue_len=2,
                partial=True
            )
            # Crucial gpiozero fix:
            # By default, gpiozero ignores None (ignore=frozenset({None})), which permanently
            # locks the sensor queue to old readings (e.g. 2.0m) whenever an echo is missed!
            # Clearing ignore and using _sonar_latest_filter ensures immediate, zero-lag updates.
            if hasattr(candidate_sensor, '_queue') and candidate_sensor._queue is not None:
                candidate_sensor._queue.ignore = frozenset()
                candidate_sensor._queue.average = _sonar_latest_filter

            sensor = candidate_sensor
            sonar_disabled = False
            sonar_reading = 3.0
            sonar_thread_running = True
            sonar_thread = threading.Thread(target=_sonar_worker, daemon=True)
            sonar_thread.start()
            print(f"[INFO] Ultrasonic hardware healthy! Multi-modal radar ACTIVE on Trigger={PIN_TRIGGER}, Echo={PIN_ECHO}.")
        except Exception as e:
            print(f"[INFO] Ultrasonic radar offline ({e}). Running vision-only mode.")
            sensor = None
            sonar_disabled = True
    else:
        print("[INFO] Ultrasonic radar disabled via --no-sonar flag.")
        sonar_disabled = True

    try:
        button = Button(PIN_BUTTON, pull_up=True)
        print(f"[INFO] Tactile push button initialized on GPIO {PIN_BUTTON}.")
    except Exception as e:
        print(f"[WARN] Could not initialize push button on GPIO {PIN_BUTTON}: {e}")
        button = None

def get_distance():
    """Returns the latest ultrasonic distance in O(1) time without blocking the video thread."""
    global sonar_reading, sonar_disabled
    if sonar_disabled:
        return 3.0
    return sonar_reading

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

def get_memory_utilization():
    """Returns (used_mb, total_mb, percent) for system RAM on Raspberry Pi / Linux."""
    if PSUTIL_AVAILABLE:
        try:
            vm = psutil.virtual_memory()
            return vm.used / (1024 * 1024), vm.total / (1024 * 1024), vm.percent
        except Exception:
            pass
    try:
        with open("/proc/meminfo", "r") as f:
            mem_info = {}
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    mem_info[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_kb = mem_info.get("MemTotal", 2097152)
            avail_kb = mem_info.get("MemAvailable", total_kb // 2)
            used_kb = max(0, total_kb - avail_kb)
            pct = (used_kb / total_kb) * 100.0 if total_kb > 0 else 0.0
            return used_kb / 1024.0, total_kb / 1024.0, pct
    except Exception:
        return 0.0, 2048.0, 0.0


# ==========================================
# 2. TEXT-TO-SPEECH (NON-BLOCKING WITH PACING PROTECTION)
# ==========================================
class SpeechCoordinator:
    """
    Cross-platform, thread-safe speech coordinator.
    Guarantees:
    1. Prevents voice acceleration & buffer piling: speech runs in a dedicated thread and accurately tracks is_speaking().
    2. True completion tracking: marks speech finished only after physical sound finishes playing from speakers.
    3. Natural, calm pacing:
       - Linux/Pi: espeak-ng/espeak at -ven-us -s130 -p48 (steady 130 wpm without chipmunk pitch/speeding up).
       - Windows: SAPI voice.Rate = -2 (relaxed conversational tempo).
       - macOS: say -r 135.
    4. Force/Interruption support: tactile push button halts current non-urgent speech immediately.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.current_process = None
        self.is_speaking_flag = False
        self.last_speech_started = 0.0
        self.last_speech_finished = 0.0
        self._sapi_voice = None

    def is_speaking(self):
        with self.lock:
            if self.is_speaking_flag:
                return True
            if self.current_process is not None:
                try:
                    return self.current_process.poll() is None
                except Exception:
                    pass
            return False

    def time_since_finished(self):
        with self.lock:
            if self.is_speaking_flag:
                return 0.0
            if self.last_speech_finished == 0.0:
                return 999.0
            return time.time() - self.last_speech_finished

    def stop_current(self):
        with self.lock:
            if self.current_process is not None:
                try:
                    if self.current_process.poll() is None:
                        self.current_process.terminate()
                        try:
                            self.current_process.wait(timeout=0.15)
                        except Exception:
                            self.current_process.kill()
                except Exception:
                    pass
                self.current_process = None

            if sys.platform == "win32" and self._sapi_voice is not None:
                try:
                    # SVSFlagsAsync (1) | SVSFPurgeBeforeSpeak (2) with empty string to immediately purge buffer
                    self._sapi_voice.Speak("", 1 | 2)
                except Exception:
                    pass

            self.is_speaking_flag = False
            self.last_speech_finished = time.time()

    def speak(self, text, force=False):
        if not text:
            return

        clean_text = "".join(ch for ch in text if ch.isalnum() or ch in " ,.!?:;'-()\"").strip()
        if not clean_text:
            return

        if not force and self.is_speaking():
            # Speech is already in progress - drop overlapping cues to prevent speech piling up
            return

        if force:
            self.stop_current()

        threading.Thread(target=self._play_worker, args=(clean_text,), daemon=True).start()

    def _play_worker(self, clean_text):
        with self.lock:
            self.is_speaking_flag = True
            self.last_speech_started = time.time()

        try:
            print(f"[VOICE] SPEAKING: {clean_text}")
        except Exception:
            pass

        try:
            if sys.platform == "win32":
                self._play_windows(clean_text)
            elif sys.platform.startswith("linux"):
                self._play_linux(clean_text)
            elif sys.platform == "darwin":
                self._play_darwin(clean_text)
            else:
                self._play_linux(clean_text)
        finally:
            with self.lock:
                self.is_speaking_flag = False
                self.current_process = None
                self.last_speech_finished = time.time()

    def _play_windows(self, clean_text):
        sanitized = clean_text.replace('"', '').replace("'", "")
        try:
            import pythoncom
            pythoncom.CoInitialize()
            import win32com.client
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            self._sapi_voice = voice
            # Set rate to -2 for clear, relaxed pacing (default 0 or unconfigured can be very fast)
            try:
                voice.Rate = -2
                voice.Volume = 100
            except Exception:
                pass
            # Synchronous speak (flag 0) in worker thread: ensures is_speaking_flag stays True until audio completes
            voice.Speak(sanitized, 0)
            return
        except Exception:
            pass

        # Fallback to PowerShell System.Speech with rate -2
        try:
            ps_cmd = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = -2; $s.Speak("{sanitized}");'
            proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self.lock:
                self.current_process = proc
            proc.wait(timeout=12.0)
        except Exception:
            pass

    def _play_linux(self, clean_text):
        # Calm, steady 110 words/minute (-s110) with 120ms word gaps (-g12) and relaxed pitch (-p45)
        for engine in ['espeak-ng', 'espeak']:
            try:
                proc = subprocess.Popen(
                    [engine, '-ven-us', '-s110', '-g12', '-p45', clean_text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                with self.lock:
                    self.current_process = proc
                proc.wait(timeout=15.0)
                return
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"[WARN] TTS engine '{engine}' error: {e}")
                break

        print("[WARN] Neither 'espeak-ng' nor 'espeak' is installed. Run: sudo apt install espeak-ng -y")

    def _play_darwin(self, clean_text):
        try:
            proc = subprocess.Popen(["say", "-r", "135", clean_text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self.lock:
                self.current_process = proc
            proc.wait(timeout=12.0)
        except Exception:
            pass


speech_coordinator = SpeechCoordinator()

def is_speaking():
    """Checks whether speech output is currently playing."""
    return speech_coordinator.is_speaking()

def speak(text, force=False):
    """Speaks text asynchronously without blocking, protected by speech_coordinator."""
    speech_coordinator.speak(text, force=force)

def stop_tts():
    """Terminates active speech upon exit."""
    speech_coordinator.stop_current()


# ==========================================
# 3. THREADED CAMERA STREAM (ZERO BUFFER LAG)
# ==========================================
def discover_capture_cameras(src=0):
    """
    Identifies genuine video capture devices on Linux / Raspberry Pi 5.
    Filters out internal Pi 5 hardware codecs, ISPs, and M2M queues (/dev/video19 - /dev/video35)
    which are NOT cameras and cause 10-second select() timeouts.
    """
    if not sys.platform.startswith("linux"):
        candidate_indices = [src]
        for idx in [0, 1, 2]:
            if idx not in candidate_indices:
                candidate_indices.append(idx)
        return candidate_indices

    candidate_indices = []

    # Priority 1: Check /dev/v4l/by-id/ (Definitive hardware identifiers for USB webcams)
    by_id_path = "/dev/v4l/by-id"
    if os.path.isdir(by_id_path):
        for link in sorted(os.listdir(by_id_path)):
            try:
                real_target = os.path.realpath(os.path.join(by_id_path, link))
                match = re.search(r"video(\d+)", real_target)
                if match:
                    idx = int(match.group(1))
                    if idx not in candidate_indices:
                        candidate_indices.append(idx)
                        print(f"[INFO] USB webcam identified: '{link}' -> /dev/video{idx}")
            except Exception:
                pass

    # Priority 2: Filter /sys/class/video4linux/ for true capture devices
    sys_v4l = "/sys/class/video4linux"
    blacklist = ["rp1-cfe", "bcm2835", "codec", "pisp", "unicam", "m2m", "isp", "metadata", "stats", "config"]
    if os.path.isdir(sys_v4l):
        for vdir in sorted(os.listdir(sys_v4l)):
            match = re.search(r"video(\d+)", vdir)
            if not match:
                continue
            idx = int(match.group(1))

            dev_name = ""
            name_file = os.path.join(sys_v4l, vdir, "name")
            if os.path.isfile(name_file):
                try:
                    with open(name_file, "r") as f:
                        dev_name = f.read().strip().lower()
                except Exception:
                    pass

            if any(term in dev_name for term in blacklist):
                continue

            driver_link = os.path.join(sys_v4l, vdir, "device", "driver")
            is_uvc = False
            if os.path.exists(driver_link):
                try:
                    drv_name = os.path.basename(os.path.realpath(driver_link)).lower()
                    if "uvc" in drv_name:
                        is_uvc = True
                except Exception:
                    pass

            if is_uvc and idx not in candidate_indices:
                candidate_indices.append(idx)
            elif not any(term in dev_name for term in blacklist) and idx < 10 and idx not in candidate_indices:
                candidate_indices.append(idx)

    # If user provided a specific src, ensure it's evaluated first
    if src not in candidate_indices and src < 10:
        candidate_indices.insert(0, src)
    elif not candidate_indices:
        candidate_indices = [0]

    return candidate_indices


class ThreadedCameraStream:
    """
    Continuously grabs frames in a background thread.
    Fixes OpenCV buffer delay so read() always returns the immediate live frame.
    Includes V4L2 acceleration on Raspberry Pi, hardware MJPG compression, and auto-reconnection.
    """
    def __init__(self, src=0, width=640, height=480):
        self.cap = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.dev_idx = src
        self.width = width
        self.height = height
        self.last_frame_time = time.time()

        candidate_indices = discover_capture_cameras(src=src)
        print(f"[INFO] Tested camera capture indices: {candidate_indices}")

        for dev_idx in candidate_indices:
            try:
                if sys.platform.startswith("linux"):
                    self.cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
                    if not self.cap.isOpened():
                        try: self.cap.release()
                        except Exception: pass
                        self.cap = cv2.VideoCapture(dev_idx)
                else:
                    self.cap = cv2.VideoCapture(dev_idx)

                if self.cap is not None and self.cap.isOpened():
                    # Request MJPG hardware compression to prevent USB bandwidth choke on Pi 5
                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    except Exception:
                        pass
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    try:
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                    except Exception:
                        pass

                    grabbed, frame = self.cap.read()
                    if grabbed and frame is not None:
                        self.grabbed = True
                        self.frame = frame
                        self.dev_idx = dev_idx
                        self.last_frame_time = time.time()
                        print(f"[INFO] Camera stream successfully opened on device index {dev_idx} (MJPG 30FPS mode)")
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
            print("\n" + "=" * 65)
            print("  [ERROR] Hardware Diagnostic: No USB Camera Detected on Raspberry Pi")
            print("  1. The USB camera is physically disconnected or its chip crashed from a power glitch.")
            print("     -> UNPLUG the USB camera cable from the Pi, wait 3 seconds, and PLUG IT BACK IN.")
            print("     -> Ensure it is plugged into a black USB 2.0 port.")
            print("  2. Verify in terminal: Run 'lsusb' and 'ls -l /dev/video*'.")
            print("     -> If the camera is receiving power, /dev/video0 will appear immediately.")
            print("=" * 65 + "\n")

    def _reopen(self):
        """Silently re-initializes the camera capture handle without crashing the application."""
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        time.sleep(0.25)
        try:
            if sys.platform.startswith("linux"):
                self.cap = cv2.VideoCapture(self.dev_idx, cv2.CAP_V4L2)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.dev_idx)
            else:
                self.cap = cv2.VideoCapture(self.dev_idx)

            if self.cap is not None and self.cap.isOpened():
                try: self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                except Exception: pass
                try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception: pass
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.last_frame_time = time.time()
                print(f"[INFO] Camera stream auto-recovered on device index {self.dev_idx}")
        except Exception as e:
            print(f"[WARN] Camera auto-recovery failed: {e}")

    def _update(self):
        consecutive_misses = 0
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.02)
                continue
            grabbed, frame = self.cap.read()
            with self.lock:
                if grabbed and frame is not None:
                    self.grabbed = True
                    self.frame = frame
                    self.last_frame_time = time.time()
                    consecutive_misses = 0
                else:
                    consecutive_misses += 1

            if not grabbed:
                time.sleep(0.01)
                # Auto-recovery: if stream stalled for > 1.2 seconds (~40 consecutive failed reads)
                if consecutive_misses > 40:
                    self._reopen()
                    consecutive_misses = 0

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            # Allow up to 3.0 seconds of wire jitter without failing read() if we have a frame
            if time.time() - self.last_frame_time > 3.0:
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
# 4. ASSISTIVE ALERT MANAGER (Strict Anti-Repetition & Track Memory)
# ==========================================
class AssistiveAlertManager:
    """
    Intelligent Alert State Manager for Visually Impaired Navigation.
    
    Guarantees:
    1. Strict One-Time Pronunciation: Once an obstacle/person is identified and announced,
       it is NOT pronounced again as long as it remains present in view.
    2. Dynamic Approach Exception: If a previously announced stationary object begins actively
       approaching towards the user (is_approaching=True), a caution alert is triggered.
    3. Emergency Distance Safety: Immediate stop warning if distance drops to <0.40m.
    4. Instant On-Demand Query: Tactile button press triggers an immediate full scan at any time.
    5. Confirmed Path Clear: Requires continuous clear frames before announcing 'Path clear ahead'.
    """
    def __init__(self, change_debounce=1.8, clear_delay=3.0, post_speech_cooldown=2.5, emergency_cooldown=1.8):
        self.change_debounce = change_debounce
        self.clear_delay = clear_delay
        self.post_speech_cooldown = post_speech_cooldown
        self.emergency_cooldown = emergency_cooldown
        
        self.last_spoken_time = 0.0
        self.last_spoken_sig = None
        self.last_spoken_text = ""
        
        # Track-level & semantic memory:
        self.announced_track_ids = set()       # Track IDs already announced
        self.announced_approaches = set()      # Track IDs where approach was already warned
        self.last_emergency_dist = None       # Distance of last critical proximity stop
        
        self.candidate_sig = None
        self.candidate_first_seen = 0.0
        self.candidate_text = ""
        
        self.clear_start_time = None

    def _make_sig(self, detections, emergency=False):
        # Stable semantic signature: object class and track_id (or col fallback)
        sig_items = []
        for d in detections:
            tid = d.get("track_id", d.get("col", "middle"))
            sig_items.append((d["class"], tid))
        return tuple(sorted(sig_items)) + (emergency,)

    def evaluate(self, detections, distance_meters, button_pressed):
        now = time.time()
        time_since_finished = speech_coordinator.time_since_finished()
        is_currently_speaking = speech_coordinator.is_speaking()
        
        current_track_ids = {d.get("track_id") for d in detections if d.get("track_id") is not None}
        
        # Prune announced tracks that have left the camera view
        if current_track_ids:
            self.announced_track_ids = self.announced_track_ids.intersection(current_track_ids)
            self.announced_approaches = self.announced_approaches.intersection(current_track_ids)
        else:
            self.announced_track_ids.clear()
            self.announced_approaches.clear()

        # 1. Tactile Button Pressed -> Immediate manual surroundings scan (always allowed)
        if button_pressed:
            announcement = format_spatial_announcement(detections, distance_meters=distance_meters, is_full_scan=True)
            self.last_spoken_time = now
            self.last_spoken_sig = self._make_sig(detections)
            self.last_spoken_text = announcement
            self.announced_track_ids.update(current_track_ids)
            self.candidate_sig = None
            self.clear_start_time = None
            return True, announcement, "Tactile Button Pressed (Full Surroundings Scan)"

        # 2. Critical Safety Proximity (< 0.40m = 40 cm Stop Hazard)
        is_critical_close = (distance_meters is not None and 0.02 <= distance_meters < 0.40)
        if is_critical_close:
            if is_currently_speaking:
                return False, None, None
            # Only repeat if distance dropped significantly (>= 15cm closer) or initial trigger
            if self.last_emergency_dist is None or (self.last_emergency_dist - distance_meters >= 0.15):
                announcement = format_spatial_announcement(detections, distance_meters=distance_meters)
                self.last_spoken_time = now
                self.last_spoken_sig = self._make_sig(detections, emergency=True)
                self.last_spoken_text = announcement
                self.last_emergency_dist = distance_meters
                return True, announcement, f"Critical Proximity Stop ({distance_meters*100:.0f}cm)"
            return False, None, None
        else:
            self.last_emergency_dist = None

        # 3. Dynamic Hazard: Newly approaching objects that haven't been warned yet
        newly_approaching = [
            d for d in detections 
            if d.get("is_approaching") and d.get("track_id") not in self.announced_approaches
        ]
        if newly_approaching:
            if is_currently_speaking or time_since_finished < self.emergency_cooldown:
                return False, None, None
            
            announcement = format_spatial_announcement(detections, distance_meters=distance_meters)
            self.last_spoken_time = now
            self.last_spoken_sig = self._make_sig(detections, emergency=True)
            self.last_spoken_text = announcement
            for d in newly_approaching:
                if d.get("track_id") is not None:
                    self.announced_approaches.add(d["track_id"])
                    self.announced_track_ids.add(d["track_id"])
            return True, announcement, "Approaching Hazard Alert"

        # 4. Speech pacing guard for standard navigation:
        if is_currently_speaking or time_since_finished < self.post_speech_cooldown:
            return False, None, None

        # 5. Obstacles in view (Standard navigation)
        if len(detections) > 0:
            self.clear_start_time = None
            current_sig = self._make_sig(detections)
            
            # Find any unannounced tracks in view
            unannounced = [d for d in detections if d.get("track_id") is not None and d.get("track_id") not in self.announced_track_ids]
            
            # If all objects in view have ALREADY been announced and signature hasn't changed:
            # STRICT ZERO REPETITION: Do NOT pronounce again!
            if not unannounced and current_sig == self.last_spoken_sig:
                return False, None, None

            # New obstacle detected or new state candidate:
            if current_sig != self.candidate_sig:
                self.candidate_sig = current_sig
                self.candidate_first_seen = now
                self.candidate_text = format_spatial_announcement(detections, distance_meters=distance_meters)
                return False, None, None
            else:
                # Candidate has been stable for change_debounce (1.8s)
                if now - self.candidate_first_seen >= self.change_debounce:
                    self.last_spoken_time = now
                    self.last_spoken_sig = current_sig
                    self.last_spoken_text = self.candidate_text
                    self.candidate_sig = None
                    self.announced_track_ids.update(current_track_ids)
                    main_cls = detections[0]['class']
                    main_dir = detections[0].get('direction', 'ahead')
                    return True, self.candidate_text, f"New Obstacle Identified ({main_cls} {main_dir})"
                return False, None, None

        # 6. Path is clear (len(detections) == 0 and distance >= 1.0m)
        self.candidate_sig = None
        if self.last_spoken_sig is not None and self.last_spoken_sig != () and self.last_spoken_sig != (False,):
            # Path was previously blocked! Wait for clear_delay to avoid flutter
            if self.clear_start_time is None:
                self.clear_start_time = now
                return False, None, None
            elif now - self.clear_start_time >= self.clear_delay:
                clear_msg = "Path clear ahead. Continue straight."
                self.last_spoken_time = now
                self.last_spoken_sig = ()
                self.last_spoken_text = clear_msg
                self.announced_track_ids.clear()
                self.announced_approaches.clear()
                return True, clear_msg, "Clear Path Navigation Update"
            return False, None, None

        return False, None, None
        self.candidate_sig = None
        if self.last_spoken_sig is not None and self.last_spoken_sig != () and self.last_spoken_sig != (False,):
            # Path was previously blocked! Wait for clear_delay to avoid flutter
            if self.clear_start_time is None:
                self.clear_start_time = now
                return False, None, None
            elif now - self.clear_start_time >= self.clear_delay:
                clear_msg = "Path clear ahead. Continue straight."
                self.last_spoken_time = now
                self.last_spoken_sig = ()
                self.last_spoken_text = clear_msg
                self.clear_start_time = None
                return True, clear_msg, "Clear Path Confirmed"
        
        return False, None, None


def print_cv_image_telemetry(frame, detections, current_fps, distance_meters, resolution=320, inference_ms=None):
    """
    Renders live Computer Vision and Image Processing telemetry directly in the terminal.
    Shows input frame resolution, letterbox tensor shape, inference latency, bounding boxes [X, Y, W, H],
    confidence scores, and a 3-zone spatial radar without saving any image files to disk.
    """
    h, w = frame.shape[:2]
    dist_str = f"{distance_meters:.2f}m" if (distance_meters is not None and 0.0 < distance_meters < 3.0) else "Clear (>3.0m)"
    inf_str = f"{inference_ms:.1f} ms" if inference_ms else f"~{1000.0 / max(1.0, current_fps):.1f} ms"

    print("\n" + "=" * 76)
    print(f" [CV IMAGE PROCESSING PIPELINE] | Frame: {w}x{h} -> Tensor: (1x3x{resolution}x{resolution})")
    print(f" Latency: {inf_str} (CPU DNN) | Rate: {current_fps:.1f} FPS | Sonar: {dist_str}")
    print("-" * 76)

    if detections:
        print(f" Detected Objects ({len(detections)} bounding boxes via OpenCV NMS):")
        for d in detections[:4]:
            box = d["box"]
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            c_name = d["class"].upper()
            conf = int(d["confidence"] * 100)
            z = d.get("zone", "middle-center").upper()
            v_dist = f"{d['visual_distance']:.1f}m" if d.get("visual_distance") else "N/A"
            app = " [APPROACHING!]" if d.get("is_approaching") else ""
            print(f"   * {c_name:<12} | Conf: {conf:>2}% | Box: [X:{box[0]:<3} Y:{box[1]:<3} W:{bw:<3} H:{bh:<3}] | Zone: {z:<13} | Dist: {v_dist}{app}")
    else:
        print(" Detection Status : Path clear ahead. No bounding boxes above threshold.")

    left_objs = [d["class"].upper() for d in detections if "left" in d.get("zone", "").lower()]
    mid_objs = [d["class"].upper() for d in detections if ("center" in d.get("zone", "").lower() or "ahead" in d.get("zone", "").lower() or d.get("zone", "").lower() in ["middle", "center", "middle-center", "top-middle", "bottom-middle"]) and "left" not in d.get("zone", "").lower() and "right" not in d.get("zone", "").lower()]
    right_objs = [d["class"].upper() for d in detections if "right" in d.get("zone", "").lower()]

    l_txt = left_objs[0] if left_objs else "CLEAR"
    m_txt = mid_objs[0] if mid_objs else "CLEAR"
    r_txt = right_objs[0] if right_objs else "CLEAR"

    print("-" * 76)
    print(f" [SPATIAL RADAR]:  [ LEFT: {l_txt} ]  ---  [ CENTER: {m_txt} ]  ---  [ RIGHT: {r_txt} ]")
    print("=" * 76)


# ==========================================
# 5. MAIN APPLICATION
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
    parser.add_argument("--trigger-pin", type=int, default=23, help="HC-SR04 trigger GPIO pin (default: 23)")
    parser.add_argument("--echo-pin", type=int, default=24, help="HC-SR04 echo GPIO pin (default: 24)")
    parser.add_argument("--button-pin", type=int, default=17, help="Tactile push button GPIO pin (default: 17)")
    parser.add_argument("--no-sonar", action="store_true", help="Disable ultrasonic sensor for pure high-speed camera mode")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable live terminal image processing telemetry")
    args = parser.parse_args()

    # Initialize GPIO Sensors with configured pins (or skip sonar if --no-sonar)
    init_sensors(
        trigger_pin=args.trigger_pin,
        echo_pin=args.echo_pin,
        button_pin=args.button_pin,
        enable_sonar=(not args.no_sonar)
    )

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

    # Auto-detect headless environment (SSH, service, or no GUI display on Pi 5)
    headless_mode = args.headless
    if not headless_mode and sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            print("[INFO] No active GUI display detected ($DISPLAY / $WAYLAND_DISPLAY unset).")
            print("[INFO] Automatically switching to --headless mode (optimized for Raspberry Pi 5 2GB).")
            headless_mode = True

    # Memory Pre-flight & Telemetry
    ram_used, ram_total, ram_pct = get_memory_utilization()
    print("\n" + "=" * 65)
    print("  Raspberry Pi 5 Smart Vision Assistant (CPU-Optimized YOLOv8)")
    print(f"  Configuration: Resolution {args.resolution}x{args.resolution} | Confidence {args.conf} | Mode: {'Headless' if headless_mode else 'GUI Window'}")
    print(f"  System RAM   : {ram_used:.0f} MB used / {ram_total:.0f} MB total ({ram_pct:.1f}%)")
    if ram_total <= 2300:
        print("  Pi 5 Variant : 2 GB RAM detected. Memory guard & tensor optimizations active.")
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

    alert_manager = AssistiveAlertManager(persistent_cooldown=25.0, change_debounce=1.8, clear_delay=3.5)
    last_button_time = 0.0
    last_valid_frame_time = time.time()
    camera_lost_alerted = False
    current_announcement = startup_msg
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
            if frame_count % 300 == 0:
                gc.collect()

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

            # Run YOLOv8 detection on CPU with multi-frame approach tracking and temporal confirmation
            detections = detector.detect(frame, distance_meters=distance_meters, filter_unconfirmed=True)
            has_approaching = any(d.get("is_approaching", False) for d in detections)

            # Live Terminal Image Processing Feedback (every 1.5 seconds)
            if not hasattr(main, '_last_print_time'):
                main._last_print_time = 0.0
            if time.time() - main._last_print_time > 1.5:
                if not args.no_telemetry:
                    inf_ms = getattr(detector, 'last_inference_ms', None)
                    print_cv_image_telemetry(frame, detections, current_fps, distance_meters, args.resolution, inf_ms)
                main._last_print_time = time.time()

            # Evaluate assistive alert state (anti-repetition, stateful debouncing & clear path confirmation)
            should_speak, announcement, trigger_reason = alert_manager.evaluate(
                detections=detections,
                distance_meters=distance_meters,
                button_pressed=button_pressed
            )

            if should_speak and announcement:
                current_announcement = announcement
                print(f"\n[TRIGGER] {trigger_reason} -> \"{current_announcement}\"")
                speak(current_announcement, force=button_pressed)

            # Graphical Telemetry Display
            if not headless_mode:
                h, w = frame.shape[:2]
                x_left = int(w * 0.35)
                x_right = int(w * 0.65)
                cpu_load = get_cpu_utilization()
                ram_u, ram_t, ram_p = get_memory_utilization()

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
                cv2.putText(frame, f"FPS: {current_fps:.1f} | CPU: {cpu_load:.0f}% | RAM: {ram_u:.0f}/{ram_t:.0f}MB ({ram_p:.0f}%)",
                            (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
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
