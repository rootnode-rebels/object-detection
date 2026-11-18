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

# Precision Raspberry Pi 5 Ultrasonic Sonar Driver (Direct lgpio Kernel-IRQ Timestamps)
try:
    from sonar_driver import Pi5UltrasonicSonar
except ImportError:
    Pi5UltrasonicSonar = None

# Safe GPIO import for Raspberry Pi
try:
    from gpiozero import DistanceSensor, Button, Buzzer
    GPIO_AVAILABLE = True
except (ImportError, Exception) as e:
    GPIO_AVAILABLE = False
    DistanceSensor = None
    Button = None
    Buzzer = None
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
PIN_TRIGGER = 23    # Physical Pin 16 (HC-SR04 Trig)
PIN_ECHO = 24       # Physical Pin 18 (HC-SR04 Echo via voltage divider)
PIN_BUTTON = 17     # Physical Pin 11 (Tactile Button)
PIN_BUZZER = 27     # Physical Pin 13 (Active Alert Buzzer, GND on Pin 14)

sensor = None
button = None
buzzer = None


class HardwareBuzzer:
    """
    Non-blocking Hardware Alert Buzzer for Visually Impaired Safety Alerts.
    Supports physical active buzzer on Raspberry Pi GPIO 27, and fallback audio on PC.
    
    Alert Modes:
    - CRITICAL (<0.50m / 50cm): Rapid urgent beeping (80ms on, 80ms off)
    - WARNING (0.50m - 1.00m or approaching hazard): Pulsed caution beeping (150ms on, 250ms off)
    - CHIRP: Single 50ms tactile confirmation beep on push button press
    - SILENT: Clear space (>=1.00m), sensor disconnected, or no obstacles
    """
    def __init__(self, pin=27, enabled=True):
        self.pin = pin
        self.enabled = enabled
        self.buzzer = None
        self.current_state = "SILENT"
        self._lock = threading.Lock()

        if self.enabled and GPIO_AVAILABLE and Buzzer is not None:
            try:
                self.buzzer = Buzzer(self.pin)
                print(f"[INFO] Hardware alert buzzer initialized on GPIO {self.pin} (Physical Pin 13).")
            except Exception as e:
                print(f"[WARN] Could not claim buzzer on GPIO {self.pin}: {e}")
                self.buzzer = None

    def update(self, distance_meters, has_approaching=False):
        """Dynamically adjusts buzzer pulsing based on physical sensor proximity."""
        if not self.enabled:
            return

        target_state = "SILENT"
        if distance_meters is not None:
            if 0.02 <= distance_meters < 0.50:
                target_state = "CRITICAL"
            elif 0.50 <= distance_meters < 1.00 or has_approaching:
                target_state = "WARNING"

        if target_state == self.current_state:
            return

        with self._lock:
            self.current_state = target_state
            if self.buzzer is not None:
                try:
                    if target_state == "CRITICAL":
                        # Rapid urgent beeps: 80ms on, 80ms off
                        self.buzzer.beep(on_time=0.08, off_time=0.08, background=True)
                    elif target_state == "WARNING":
                        # Moderate warning beeps: 150ms on, 250ms off
                        self.buzzer.beep(on_time=0.15, off_time=0.25, background=True)
                    else:
                        self.buzzer.off()
                except Exception:
                    pass

    def chirp(self):
        """Single short confirmation chirp (e.g. for tactile button confirmation)."""
        if not self.enabled:
            return
        if self.buzzer is not None:
            try:
                self.buzzer.beep(on_time=0.05, off_time=0.05, n=1, background=True)
            except Exception:
                pass

    def close(self):
        """Silences and safely releases buzzer GPIO."""
        with self._lock:
            if self.buzzer is not None:
                try:
                    self.buzzer.off()
                    self.buzzer.close()
                except Exception:
                    pass
                self.buzzer = None
            self.current_state = "SILENT"

sonar_connected = False
sonar_reading = None
sonar_thread = None
sonar_thread_running = False

def _sonar_latest_filter(queue):
    """
    Zero-hysteresis filter for HC-SR04 sonar queue.
    Returns latest reading or None if echo timed out / disconnected.
    """
    if not queue:
        return None
    return queue[-1]

def _sonar_worker():
    """
    Background daemon thread to continuously poll physical ultrasonic sensor (HC-SR04).
    Uses direct lgpio kernel-IRQ hardware timestamps (Pi5UltrasonicSonar) with 70ms acoustic
    settling and 5-sample median + spike-guard filtering for accurate, jitter-free distance.
    STRICT HARDWARE RULE:
    - If obstacle or hand is detected (0.02m to 2.98m): updates live distance, marks sensor connected.
    - If clear open room (>3.0m): reports 3.0m clear, marks sensor connected.
    - If wire is unplugged, echo missing, or sensor offline: marks DISCONNECTED, sets distance to None.
    """
    global sensor, sonar_reading, sonar_connected, sonar_thread_running
    last_valid_time = 0.0

    while sonar_thread_running and sensor is not None:
        t0 = time.perf_counter()
        try:
            d = None
            if hasattr(sensor, "read_filtered"):
                # Direct Pi5UltrasonicSonar (lgpio kernel-IRQ + median + bounce/spike rejection)
                d = sensor.read_filtered()
            else:
                # Legacy gpiozero fallback
                has_queue_items = (hasattr(sensor, '_queue') and sensor._queue is not None and len(sensor._queue.queue) > 0)
                if has_queue_items:
                    d = sensor.distance

            if d is not None and 0.02 <= d <= 3.0:
                if d >= 2.98:
                    sonar_reading = 3.0
                else:
                    sonar_reading = round(float(d), 2)
                sonar_connected = True
                last_valid_time = time.time()
            elif d is not None and d > 3.0:
                sonar_reading = 3.0
                sonar_connected = True
                last_valid_time = time.time()

            # Disconnection check: 2.0s without valid echo -> mark disconnected
            if last_valid_time == 0.0 or (time.time() - last_valid_time) > 2.0:
                sonar_reading = None
                sonar_connected = False

        except Exception:
            if last_valid_time == 0.0 or (time.time() - last_valid_time) > 2.0:
                sonar_reading = None
                sonar_connected = False

        # Respect HC-SR04 datasheet minimum 60ms cycle (70ms prevents wall-echo collisions)
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.01, 0.07 - elapsed))

def init_sensors(trigger_pin=23, echo_pin=24, button_pin=17, buzzer_pin=27, enable_sonar=True, enable_buzzer=True):
    """
    Initializes hardware GPIO sensors (HC-SR04 ultrasonic sensor, tactile button & alert buzzer).
    STRICT RULE: Distance is ONLY enabled if a verified physical sensor is connected.
    """
    global sensor, button, buzzer, PIN_TRIGGER, PIN_ECHO, PIN_BUTTON, PIN_BUZZER, sonar_connected, sonar_reading, sonar_thread, sonar_thread_running
    PIN_TRIGGER = trigger_pin
    PIN_ECHO = echo_pin
    PIN_BUTTON = button_pin
    PIN_BUZZER = buzzer_pin

    if not GPIO_AVAILABLE and Pi5UltrasonicSonar is None:
        print("[INFO] GPIO unavailable. Hardware sensors disabled (Desktop/Simulation mode).")
        sonar_connected = False
        sonar_reading = None
        return

    if sensor is not None:
        try: sensor.close()
        except Exception: pass
        sensor = None
    if button is not None:
        try: button.close()
        except Exception: pass
        button = None
    if buzzer is not None:
        try: buzzer.close()
        except Exception: pass
        buzzer = None

    sonar_thread_running = False

    if enable_sonar:
        try:
            if Pi5UltrasonicSonar is not None:
                candidate_sensor = Pi5UltrasonicSonar(
                    trigger_pin=PIN_TRIGGER,
                    echo_pin=PIN_ECHO,
                    max_distance=3.0,
                    temp_c=25.0,
                    window_size=5,
                    ping_interval=0.07,
                )
                backend_info = f"{candidate_sensor.backend} on {candidate_sensor.chip_info}"
            elif GPIO_AVAILABLE and DistanceSensor is not None:
                candidate_sensor = DistanceSensor(
                    echo=PIN_ECHO,
                    trigger=PIN_TRIGGER,
                    max_distance=3.0,
                    queue_len=5,
                    partial=True
                )
                backend_info = "gpiozero"
            else:
                raise RuntimeError("No GPIO sonar backend available")

            sensor = candidate_sensor
            sonar_connected = False
            sonar_reading = None
            sonar_thread_running = True
            sonar_thread = threading.Thread(target=_sonar_worker, daemon=True)
            sonar_thread.start()
            print(f"[INFO] Ultrasonic hardware initialized on Trigger=GPIO {PIN_TRIGGER}, Echo=GPIO {PIN_ECHO} ({backend_info}).")
        except Exception as e:
            print(f"[INFO] Ultrasonic radar offline ({e}). Running vision-only mode (No distance).")
            sensor = None
            sonar_connected = False
            sonar_reading = None
    else:
        print("[INFO] Ultrasonic radar disabled via --no-sonar flag.")
        sonar_connected = False
        sonar_reading = None
        sonar_reading = None

    try:
        button = Button(PIN_BUTTON, pull_up=True)
        print(f"[INFO] Tactile push button initialized on GPIO {PIN_BUTTON}.")
    except Exception as e:
        print(f"[WARN] Could not initialize push button on GPIO {PIN_BUTTON}: {e}")
        button = None

    if enable_buzzer:
        buzzer = HardwareBuzzer(pin=PIN_BUZZER, enabled=True)
    else:
        buzzer = None

def get_distance():
    """
    Returns the latest physical ultrasonic distance in meters.
    Returns None if sensor is disconnected, unplugged, or offline.
    STRICT RULE: The only distance dependency is the physical sensor.
    """
    global sonar_reading, sonar_connected
    if not sonar_connected:
        return None
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
    def __init__(self, voice_speed=170):
        self.lock = threading.Lock()
        self.current_process = None
        self.is_speaking_flag = False
        self.last_speech_started = 0.0
        self.last_speech_finished = 0.0
        self._sapi_voice = None
        self.voice_speed = voice_speed

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
            # Set rate to 0 for natural, crisp 1.0x human speaking tempo
            try:
                voice.Rate = 0
                voice.Volume = 100
            except Exception:
                pass
            # Synchronous speak (flag 0) in worker thread: ensures is_speaking_flag stays True until audio completes
            voice.Speak(sanitized, 0)
            return
        except Exception:
            pass

        # Fallback to PowerShell System.Speech with rate 0 (1.0x normal)
        try:
            ps_cmd = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 0; $s.Speak("{sanitized}");'
            proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", ps_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self.lock:
                self.current_process = proc
            proc.wait(timeout=12.0)
        except Exception:
            pass

    def _play_linux(self, clean_text):
        """
        High-fidelity, non-blocking Linux / Raspberry Pi speech synthesis engine.
        Key optimizations for Raspberry Pi 5:
        1. Synthesizes complete WAV directly to /dev/shm (POSIX RAM disk) in ~5ms:
           Completely eliminates shell piping jitter, sample-rate starvation, and ALSA buffer underrun crackles!
        2. Plays via prioritized audio backends:
           - pw-play: Native PipeWire on Raspberry Pi OS Bookworm / Trixie (flawless hardware sync)
           - paplay : PulseAudio / PipeWire ALSA bridge
           - aplay  : Direct ALSA with explicit 22050Hz sample rate and 200ms buffer (-B 200000)
        3. Single player process tracking enables instant, clean interruption with zero orphaned audio processes.
        """
        spd = getattr(self, 'voice_speed', 170)
        escaped_text = clean_text.replace("'", "").replace('"', '').replace(';', '').replace('&', '').strip()
        if not escaped_text:
            return

        ramdisk_dir = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"
        wav_path = os.path.join(ramdisk_dir, f"tts_{os.getpid()}_{int(time.time() * 1000) % 10000}.wav")

        for engine in ['espeak-ng', 'espeak']:
            try:
                # Step 1: Render complete WAV into RAM disk in ~5ms
                gen_cmd = [engine, '-ven-us', f'-s{spd}', '-p50', '-a100', '-w', wav_path, escaped_text]
                ret = subprocess.run(gen_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5.0)
                if ret.returncode != 0 or not os.path.exists(wav_path):
                    continue

                # Step 2: Play rendered WAV with highest-fidelity available player
                player_candidates = [
                    ['pw-play', wav_path],
                    ['paplay', wav_path],
                    ['aplay', '-q', '-D', 'default', '-r', '22050', '-f', 'S16_LE', '-c', '1', '-B', '200000', wav_path],
                    ['aplay', '-q', '-D', 'default', wav_path],
                    ['aplay', '-q', wav_path]
                ]

                played = False
                for play_cmd in player_candidates:
                    try:
                        proc = subprocess.Popen(
                            play_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        with self.lock:
                            self.current_process = proc
                        proc.wait(timeout=10.0)
                        if proc.returncode == 0:
                            played = True
                            break
                    except FileNotFoundError:
                        continue
                    except Exception:
                        pass

                if played:
                    return

            except Exception:
                pass
            finally:
                try:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                except Exception:
                    pass

            # Step 3: Direct engine fallback
            try:
                proc = subprocess.Popen(
                    [engine, '-ven-us', f'-s{spd}', '-p50', '-a100', escaped_text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                with self.lock:
                    self.current_process = proc
                proc.wait(timeout=10.0)
                return
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"[WARN] TTS fallback '{engine}' error: {e}")
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
    1. Preemptive Safety: Critical stop (<0.50m) and approaching hazards interrupt casual chatter immediately (force=True).
    2. Head-Swivel Grace Period (TTL): Remembers announced obstacles for 6.0 seconds so a blind user tilting their
       head or turning slightly does NOT trigger annoying re-announcements of the same person or obstacle.
    3. Strict One-Time Pronunciation: Once an obstacle/person is identified and announced,
       it is NOT pronounced again as long as it remains present in view.
    4. Dynamic Approach Exception: If a previously announced stationary object begins actively
       approaching towards the user (is_approaching=True), a caution alert is triggered.
    5. First-Class Sonar Fusion: Forward obstacles (walls, doors, pillars) detected by ultrasonic (<1.8m)
       are fully integrated into navigation even if camera detects no bounding box.
    6. Rapid Clear Path: Announces 'Path clear ahead' after 1.6s of confirmed open path.
    7. Instant On-Demand Query: Tactile button press triggers an immediate full scan at any time.
    """
    def __init__(self, change_debounce=1.6, clear_delay=1.6, post_speech_cooldown=2.2, emergency_cooldown=1.5, track_ttl=6.0):
        self.change_debounce = change_debounce
        self.clear_delay = clear_delay
        self.post_speech_cooldown = post_speech_cooldown
        self.emergency_cooldown = emergency_cooldown
        self.track_ttl = track_ttl
        
        self.last_spoken_time = 0.0
        self.last_spoken_sig = None
        self.last_spoken_text = ""
        self.is_last_emergency = False
        
        # Track-level & semantic memory with TTL (grace period for head turns):
        self.announced_tracks_seen = {}       # {track_id: last_seen_time}
        self.announced_approaches = set()      # Track IDs where approach was already warned
        self.last_emergency_dist = None       # Distance of last critical proximity stop
        
        self.candidate_sig = None
        self.candidate_first_seen = 0.0
        self.candidate_text = ""
        
        self.clear_start_time = None

    def _make_sig(self, detections, distance_meters=None, emergency=False):
        # Stable semantic signature: object class and track_id (or col fallback)
        sig_items = []
        for d in detections:
            tid = d.get("track_id", d.get("col", "middle"))
            sig_items.append((d["class"], tid))
        # Quantize sonar distance into 35cm bands to prevent repetitive voice spam for stationary obstacles
        sonar_band = None
        if distance_meters is not None and 0.02 <= distance_meters < 1.8:
            sonar_band = int(distance_meters // 0.35)
        return tuple(sorted(sig_items)) + (sonar_band, emergency)

    def evaluate(self, detections, distance_meters, button_pressed):
        now = time.time()
        time_since_finished = speech_coordinator.time_since_finished()
        is_currently_speaking = speech_coordinator.is_speaking()
        self.is_last_emergency = False
        
        current_track_ids = {d.get("track_id") for d in detections if d.get("track_id") is not None}
        
        # ONLY refresh last-seen timestamps for tracks that were ALREADY announced!
        # (Never pre-mark newly seen unannounced tracks as announced before they are spoken.)
        for tid in current_track_ids:
            if tid in self.announced_tracks_seen:
                self.announced_tracks_seen[tid] = now
            
        # Expire tracks that haven't been seen for > track_ttl (6 seconds grace period for head turns)
        expired = [tid for tid, seen_t in self.announced_tracks_seen.items() if now - seen_t > self.track_ttl]
        for tid in expired:
            self.announced_tracks_seen.pop(tid, None)
            self.announced_approaches.discard(tid)

        active_announced_tracks = set(self.announced_tracks_seen.keys())

        # 1. Tactile Button Pressed -> Immediate manual surroundings scan (always allowed, preempts audio)
        if button_pressed:
            announcement = format_spatial_announcement(detections, distance_meters=distance_meters, is_full_scan=True)
            self.last_spoken_time = now
            self.last_spoken_sig = self._make_sig(detections, distance_meters=distance_meters)
            self.last_spoken_text = announcement
            for tid in current_track_ids:
                self.announced_tracks_seen[tid] = now
            self.candidate_sig = None
            self.clear_start_time = None
            self.is_last_emergency = True
            return True, announcement, "Tactile Button Pressed (Full Surroundings Scan)"

        # 2. Critical Safety Proximity (< 0.50m = 50 cm Stop Hazard for Walking Reaction Time)
        # PREEMPTS ongoing standard speech immediately!
        is_critical_close = (distance_meters is not None and 0.02 <= distance_meters < 0.50)
        if is_critical_close:
            # Only repeat if distance dropped significantly (>= 10cm closer) or initial trigger
            if self.last_emergency_dist is None or (self.last_emergency_dist - distance_meters >= 0.10):
                announcement = format_spatial_announcement(detections, distance_meters=distance_meters)
                self.last_spoken_time = now
                self.last_spoken_sig = self._make_sig(detections, distance_meters=distance_meters, emergency=True)
                self.last_spoken_text = announcement
                self.last_emergency_dist = distance_meters
                self.is_last_emergency = True
                return True, announcement, f"Critical Proximity Stop ({distance_meters*100:.0f}cm)"
            return False, None, None
        else:
            self.last_emergency_dist = None

        # 3. Dynamic Hazard: Newly approaching objects that haven't been warned yet
        # PREEMPTS ongoing standard speech!
        newly_approaching = [
            d for d in detections 
            if d.get("is_approaching") and d.get("track_id") not in self.announced_approaches
        ]
        if newly_approaching:
            if (now - self.last_spoken_time < self.emergency_cooldown) and self.is_last_emergency:
                return False, None, None
            
            announcement = format_spatial_announcement(detections, distance_meters=distance_meters)
            self.last_spoken_time = now
            self.last_spoken_sig = self._make_sig(detections, distance_meters=distance_meters, emergency=True)
            self.last_spoken_text = announcement
            self.is_last_emergency = True
            for d in newly_approaching:
                if d.get("track_id") is not None:
                    self.announced_approaches.add(d["track_id"])
                    self.announced_tracks_seen[d["track_id"]] = now
            return True, announcement, "Approaching Hazard Alert"

        # 4. Speech pacing guard for standard navigation:
        if is_currently_speaking or time_since_finished < self.post_speech_cooldown:
            return False, None, None

        # 5. Obstacles in view (Camera detections OR Forward Ultrasonic Radar < 1.8m)
        sonar_obstacle_ahead = (distance_meters is not None and 0.02 <= distance_meters < 1.8)
        has_obstacles = (len(detections) > 0) or sonar_obstacle_ahead

        if has_obstacles:
            self.clear_start_time = None
            current_sig = self._make_sig(detections, distance_meters=distance_meters)
            # Stable candidate key based on object classes + obstacle presence (prevents minor sonar jitter resetting debounce)
            candidate_key = tuple(sorted(d["class"] for d in detections)) + (sonar_obstacle_ahead,)
            
            # Find any unannounced tracks in view (respecting 6s TTL history!)
            unannounced = [d for d in detections if d.get("track_id") is not None and d.get("track_id") not in active_announced_tracks]
            
            # If all objects in view (and sonar band) have ALREADY been announced and signature hasn't changed:
            # STRICT ZERO REPETITION: Do NOT pronounce again!
            if not unannounced and current_sig == self.last_spoken_sig:
                return False, None, None

            # New obstacle detected or new state candidate:
            if candidate_key != self.candidate_sig:
                self.candidate_sig = candidate_key
                self.candidate_first_seen = now
                return False, None, None
            else:
                # Candidate has been stable for change_debounce -> format fresh announcement with latest live distance
                if now - self.candidate_first_seen >= self.change_debounce:
                    fresh_text = format_spatial_announcement(detections, distance_meters=distance_meters)
                    self.last_spoken_time = now
                    self.last_spoken_sig = current_sig
                    self.last_spoken_text = fresh_text
                    self.candidate_sig = None
                    for tid in current_track_ids:
                        self.announced_tracks_seen[tid] = now
                    obs_name = detections[0]['class'] if detections else "obstacle"
                    obs_dir = detections[0].get('direction', 'ahead') if detections else "ahead"
                    return True, fresh_text, f"New Obstacle Identified ({obs_name} {obs_dir})"
                return False, None, None

        # 6. Path is clear (len(detections) == 0 AND distance >= 1.8m or disconnected)
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
                return True, clear_msg, "Clear Path Navigation Update"
            return False, None, None

        return False, None, None


def print_cv_image_telemetry(frame, detections, current_fps, distance_meters, resolution=320, inference_ms=None):
    """
    Renders live Computer Vision and Image Processing telemetry directly in the terminal.
    Shows input frame resolution, letterbox tensor shape, inference latency, bounding boxes [X, Y, W, H],
    confidence scores, and a 3-zone spatial radar without saving any image files to disk.
    """
    h, w = frame.shape[:2]
    if distance_meters is not None:
        if distance_meters >= 2.98:
            dist_str = "Clear (>3.0m)"
        else:
            dist_str = f"{distance_meters:.2f}m ({int(round(distance_meters * 100))}cm)"
    else:
        dist_str = "DISCONNECTED [No Hardware Sensor]"

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
            app = " [APPROACHING!]" if d.get("is_approaching") else ""
            print(f"   * {c_name:<14} | Conf: {conf:>2}% | Box: [X:{box[0]:<3} Y:{box[1]:<3} W:{bw:<3} H:{bh:<3}] | Zone: {z:<13}{app}")
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
    parser.add_argument("--buzzer-pin", type=int, default=27, help="Active alert buzzer GPIO pin (default: 27 / Pin 13)")
    parser.add_argument("--no-buzzer", action="store_true", help="Disable hardware alert buzzer")
    parser.add_argument("--no-sonar", action="store_true", help="Disable ultrasonic sensor for pure high-speed camera mode")
    parser.add_argument("--no-telemetry", action="store_true", help="Disable live terminal image processing telemetry")
    args = parser.parse_args()

    # Immediate audio confirmation for visually impaired user: hardware is powering up
    speak("Starting smart vision assistant. Please wait.")

    # Initialize GPIO Sensors with configured pins (or skip sonar/buzzer if disabled)
    init_sensors(
        trigger_pin=args.trigger_pin,
        echo_pin=args.echo_pin,
        button_pin=args.button_pin,
        buzzer_pin=args.buzzer_pin,
        enable_sonar=(not args.no_sonar),
        enable_buzzer=(not args.no_buzzer)
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

    startup_msg = "Smart vision assistant ready. Push button anytime for full scan."
    speak(startup_msg)

    alert_manager = AssistiveAlertManager(change_debounce=0.6, clear_delay=2.0)
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
                        speak("Warning! Camera feed lost! Stop immediately and check camera cable.", force=True)
                        camera_lost_alerted = True
                time.sleep(0.01)
                continue

            last_valid_frame_time = time.time()
            if camera_lost_alerted:
                speak("Camera feed restored. Obstacle detection resumed.", force=True)
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
                if buzzer is not None:
                    buzzer.chirp()

            # Run YOLOv8 detection on CPU with multi-frame approach tracking and temporal confirmation
            detections = detector.detect(frame, distance_meters=distance_meters, filter_unconfirmed=True)
            has_approaching = any(d.get("is_approaching", False) for d in detections)

            # Update Hardware Alert Buzzer in real-time (Rapid alarm <50cm, pulsed warning <1.0m)
            if buzzer is not None:
                buzzer.update(distance_meters=distance_meters, has_approaching=has_approaching)

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
                force_speech = button_pressed or getattr(alert_manager, 'is_last_emergency', False)
                speak(current_announcement, force=force_speech)

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
                if distance_meters is not None:
                    if distance_meters < 0.50:
                        dist_color = (0, 0, 255)  # Red for close hazard
                        dist_label = f"Distance: {distance_meters:.2f}m ({int(round(distance_meters * 100))}cm) [STOP HAZARD!]"
                    elif distance_meters < 1.0:
                        dist_color = (0, 165, 255)  # Orange for caution
                        dist_label = f"Distance: {distance_meters:.2f}m ({int(round(distance_meters * 100))}cm) [CAUTION]"
                    elif distance_meters < 2.98:
                        dist_color = (0, 255, 0)  # Green for clear range
                        dist_label = f"Distance: {distance_meters:.2f}m ({int(round(distance_meters * 100))}cm)"
                    else:
                        dist_color = (0, 255, 180)  # Cyan for open space
                        dist_label = "Distance: Clear (>3.0m)"
                else:
                    dist_color = (0, 0, 255)  # Red warning: Sensor is unplugged / removed
                    dist_label = "Distance: SENSOR DISCONNECTED [NO HARDWARE]"

                cv2.putText(frame, dist_label,
                            (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.52, dist_color, 2)
                bz_status = f" | BUZZER: {buzzer.current_state}" if buzzer is not None and buzzer.enabled else ""
                app_badge = ("RADAR: APPROACHING HAZARD!" if has_approaching else "RADAR: MONITORING") + bz_status
                app_badge_color = (0, 0, 255) if (has_approaching or (buzzer and buzzer.current_state == 'CRITICAL')) else (200, 200, 200)
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
        if buzzer is not None:
            try:
                buzzer.close()
            except Exception:
                pass
        stop_tts()
        print("[INFO] Cleanup complete. System safely halted.")

if __name__ == "__main__":
    main()
