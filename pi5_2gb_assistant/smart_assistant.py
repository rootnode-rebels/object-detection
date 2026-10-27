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

def _sonar_worker():
    """Background daemon thread to poll ultrasonic sensor without blocking the camera vision loop."""
    global sensor, sonar_reading, sonar_disabled, sonar_thread_running
    consecutive_errors = 0
    while sonar_thread_running and not sonar_disabled and sensor is not None:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                d = sensor.distance
                has_no_echo = any("no echo" in str(item.message).lower() for item in w)
                if has_no_echo or d is None:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0
                    if 0.0 <= d <= 3.0:
                        sonar_reading = d

            if consecutive_errors >= 4:
                print("[WARN] Ultrasonic echo signal lost. Auto-switching to pure high-speed camera mode.")
                sonar_disabled = True
                break
        except Exception:
            consecutive_errors += 1
            if consecutive_errors >= 4:
                sonar_disabled = True
                break
        time.sleep(0.08)  # Sample at ~12 Hz asynchronously

def init_sensors(trigger_pin=23, echo_pin=24, button_pin=17, enable_sonar=True):
    """
    Initializes hardware GPIO sensors (HC-SR04 ultrasonic distance sensor & tactile button).
    Features non-blocking pre-flight probe: automatically detects if ECHO pin is desoldered,
    bypassing sonar with zero latency tonight, while seamlessly activating when soldered tomorrow!
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
            candidate_sensor = DistanceSensor(echo=PIN_ECHO, trigger=PIN_TRIGGER, max_distance=3.0)
            # Pre-flight probe: verify echo pulse responds before locking into radar mode
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                probe = candidate_sensor.distance
                has_no_echo = any("no echo" in str(item.message).lower() for item in w)
                if has_no_echo or probe is None:
                    print("[INFO] Ultrasonic ECHO pin not detected (hardware desoldered / unplugged).")
                    print("[INFO] Auto-engaging pure high-speed vision mode. Push button active on GPIO 17.")
                    candidate_sensor.close()
                    sensor = None
                    sonar_disabled = True
                else:
                    sensor = candidate_sensor
                    sonar_disabled = False
                    sonar_reading = probe
                    sonar_thread_running = True
                    sonar_thread = threading.Thread(target=_sonar_worker, daemon=True)
                    sonar_thread.start()
                    print(f"[INFO] Ultrasonic hardware healthy! Live echo verified: {probe:.2f}m. Multi-modal radar ACTIVE.")
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
        # Calm, steady 130 words/minute (-s130) with natural US English voice (-ven-us) prevents fast chipmunk voice
        for engine in ['espeak-ng', 'espeak']:
            try:
                proc = subprocess.Popen(
                    [engine, '-ven-us', '-s130', '-p48', clean_text],
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
# 4. ASSISTIVE ALERT MANAGER (Stateful Debouncing & Anti-Repetition)
# ==========================================
class AssistiveAlertManager:
    """
    Intelligent Alert State Manager for Visually Impaired Navigation.
    
    Guarantees:
    1. Zero repeating loop spam (25-second cooldown on static/unchanged scene).
    2. Instant alert on tactile push button press (full surroundings scan).
    3. Prevents voice acceleration & buffer piling: enforces strict post-speech silence windows
       (minimum 2.5s calm pause after speech finishes before new navigation cues, 1.8s for emergencies).
    4. Stable signatures (uses class and horizontal column instead of noisy micro-coordinates) so frame-to-frame jitter doesn't trigger speech bursts.
    5. Fully integrates with SpeechCoordinator.is_speaking() to ensure the assistant never interrupts itself or talks over ongoing speech.
    6. Confirms clear path (requires 2.0s of continuous clear frames before announcing 'Path clear ahead', preventing flutter).
    """
    def __init__(self, persistent_cooldown=25.0, change_debounce=1.2, clear_delay=2.0, post_speech_cooldown=2.5, emergency_cooldown=1.8):
        self.persistent_cooldown = persistent_cooldown
        self.change_debounce = change_debounce
        self.clear_delay = clear_delay
        self.post_speech_cooldown = post_speech_cooldown
        self.emergency_cooldown = emergency_cooldown
        
        self.last_spoken_time = 0.0
        self.last_spoken_sig = None
        self.last_spoken_text = ""
        
        self.candidate_sig = None
        self.candidate_first_seen = 0.0
        self.candidate_text = ""
        
        self.clear_start_time = None

    def _make_sig(self, detections, emergency=False):
        # Stable semantic signature: object class and horizontal column ('left', 'middle', 'right').
        # Using column instead of volatile 9-grid box coordinates prevents minor hand tremors or bounding box jitter
        # from changing the signature every 100 milliseconds.
        sig_items = []
        for d in detections:
            col = d.get("col", "middle")
            sig_items.append((d["class"], col))
        return tuple(sorted(sig_items)) + (emergency,)

    def evaluate(self, detections, distance_meters, button_pressed):
        now = time.time()
        time_since_finished = speech_coordinator.time_since_finished()
        is_currently_speaking = speech_coordinator.is_speaking()
        
        # 1. Tactile Button Pressed -> Immediate full surroundings scan (always allowed, forces ongoing speech to stop)
        if button_pressed:
            announcement = format_spatial_announcement(detections, distance_meters=distance_meters, is_full_scan=True)
            self.last_spoken_time = now
            self.last_spoken_sig = self._make_sig(detections)
            self.last_spoken_text = announcement
            self.candidate_sig = None
            self.clear_start_time = None
            return True, announcement, "Tactile Button Pressed (Full Surroundings Scan)"

        # 2. Emergency Hazard: approaching hazard OR immediate proximity < 0.85m
        has_approaching = any(d.get("is_approaching", False) for d in detections)
        sonar_prox = (distance_meters is not None and 0.0 < distance_meters < 0.85)
        visual_prox = any(
            ("middle" in d.get("zone", "") or "center" in d.get("zone", "")) and
            (d.get("visual_distance") is not None and d.get("visual_distance") < 0.85)
            for d in detections
        )
        is_emergency = has_approaching or sonar_prox or visual_prox

        if is_emergency:
            # If assistant is currently speaking, do not talk over itself
            if is_currently_speaking:
                return False, None, None

            # Enforce at least emergency_cooldown (1.8s) of calm pause after previous speech finished
            if time_since_finished < self.emergency_cooldown:
                return False, None, None

            current_sig = self._make_sig(detections, emergency=True)
            if current_sig != self.last_spoken_sig or (now - self.last_spoken_time >= 3.5):
                announcement = format_spatial_announcement(detections, distance_meters=distance_meters)
                self.last_spoken_time = now
                self.last_spoken_sig = current_sig
                self.last_spoken_text = announcement
                self.candidate_sig = None
                self.clear_start_time = None
                if has_approaching:
                    reason = "Approaching Hazard Alert"
                elif sonar_prox:
                    reason = f"Proximity Alert ({distance_meters:.2f}m)"
                else:
                    v_d = detections[0].get("visual_distance", 0.8)
                    reason = f"Visual Proximity Alert ({v_d:.1f}m)"
                return True, announcement, reason
            return False, None, None

        # 3. Speech pacing guard for standard navigation:
        # If speech is currently active OR post-speech calm window hasn't elapsed, wait!
        if is_currently_speaking or time_since_finished < self.post_speech_cooldown:
            return False, None, None

        # 4. Obstacles in view (Standard navigation)
        if len(detections) > 0:
            self.clear_start_time = None
            current_sig = self._make_sig(detections)
            
            # Scenario A: Same scene as previously spoken
            if current_sig == self.last_spoken_sig:
                if now - self.last_spoken_time >= self.persistent_cooldown:
                    announcement = format_spatial_announcement(detections, distance_meters=distance_meters)
                    self.last_spoken_time = now
                    self.last_spoken_text = announcement
                    return True, announcement, "Periodic Obstacle Reminder"
                return False, None, None
            
            # Scenario B: Scene has changed (new obstacle or direction change)
            if current_sig != self.candidate_sig:
                self.candidate_sig = current_sig
                self.candidate_first_seen = now
                self.candidate_text = format_spatial_announcement(detections, distance_meters=distance_meters)
                return False, None, None
            else:
                # Candidate has been stable for change_debounce seconds
                if now - self.candidate_first_seen >= self.change_debounce:
                    self.last_spoken_time = now
                    self.last_spoken_sig = current_sig
                    self.last_spoken_text = self.candidate_text
                    self.candidate_sig = None
                    main_cls = detections[0]['class']
                    main_dir = detections[0].get('direction', 'ahead')
                    return True, self.candidate_text, f"Scene State Change ({main_cls} {main_dir})"
                return False, None, None

        # 5. Path is clear (len(detections) == 0 and distance >= 1.0m)
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

    alert_manager = AssistiveAlertManager(persistent_cooldown=25.0, change_debounce=1.2, clear_delay=2.0)
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

            # Live Terminal Scanning Feedback every 1.5 seconds so user always sees camera is alive
            if not hasattr(main, '_last_print_time'):
                main._last_print_time = 0.0
            if time.time() - main._last_print_time > 1.5:
                if detections:
                    det_summary = ", ".join([f"{d['class']} ({d['confidence']*100:.0f}%)" for d in detections])
                    print(f"[LIVE SCAN] FPS: {current_fps:.1f} | Detections: [{det_summary}]")
                else:
                    print(f"[LIVE SCAN] FPS: {current_fps:.1f} | Scanning... Path clear ahead")
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
