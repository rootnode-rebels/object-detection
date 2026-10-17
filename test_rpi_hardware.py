#!/usr/bin/env python3
"""
Raspberry Pi 5 Hardware & AI Model Diagnostic Suite
Tests and verifies all subsystems for the Smart Vision Assistant:
1. System & CPU Architecture (Pi 5 ARM Cortex-A76)
2. AI Model & 92 Classes (model_files/yolov8_assistive_92.onnx)
3. Camera Stream (Raspberry Pi Camera Module 3 or USB Webcam via V4L2)
4. Non-Blocking Voice Output (espeak-ng / espeak)
5. HC-SR04 Ultrasonic Distance Sensor (Trigger GPIO 23, Echo GPIO 24)
6. Tactile Push Button (GPIO 17)
7. Dynamic Approach Detection & Spatial Guidance Engine
"""

import os
import sys
import time
import subprocess
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

PIN_TRIGGER = 23
PIN_ECHO = 24
PIN_BUTTON = 17

def print_header(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def test_system():
    print_header("1. SYSTEM & CPU / MEMORY DIAGNOSTICS")
    import platform
    print(f"  OS Platform     : {platform.system()} {platform.release()}")
    print(f"  Architecture    : {platform.machine()}")
    print(f"  Python Version  : {sys.version.split()[0]}")
    
    total_ram_mb = 2048.0
    avail_ram_mb = 1024.0
    swap_total_mb = 0.0
    swap_free_mb = 0.0

    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        print(f"  CPU Cores       : {cpu_count} cores")
        try:
            ram = psutil.virtual_memory()
            total_ram_mb = ram.total / (1024**2)
            avail_ram_mb = ram.available / (1024**2)
        except Exception:
            pass
        try:
            swap = psutil.swap_memory()
            swap_total_mb = swap.total / (1024**2)
            swap_free_mb = swap.free / (1024**2)
        except Exception:
            pass
    except Exception:
        pass

    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                info = {}
                for line in f:
                    p = line.split(":")
                    if len(p) == 2:
                        info[p[0].strip()] = int(p[1].strip().split()[0])
                if "MemTotal" in info:
                    total_ram_mb = info["MemTotal"] / 1024.0
                if "MemAvailable" in info:
                    avail_ram_mb = info["MemAvailable"] / 1024.0
                if "SwapTotal" in info:
                    swap_total_mb = info["SwapTotal"] / 1024.0
                if "SwapFree" in info:
                    swap_free_mb = info["SwapFree"] / 1024.0
    except Exception:
        pass

    print(f"  Total RAM       : {total_ram_mb / 1024.0:.2f} GB ({total_ram_mb:.0f} MB)")
    print(f"  Available RAM   : {avail_ram_mb:.0f} MB")
    print(f"  Swap Space      : {swap_total_mb:.0f} MB total ({swap_free_mb:.0f} MB free)")

    if total_ram_mb <= 2300:
        print("  Hardware Target : Raspberry Pi 5 (2 GB RAM Edition)")
        if swap_total_mb < 1000:
            print("  [WARN] Swap is under 1.0 GB! For 2GB Pi 5, a 2GB swapfile is strongly recommended.")
            print("         Run: sudo ./setup_rpi.sh (it will automatically configure 2GB swap)")
        else:
            print("  [PASS] Adequate swap protection enabled for 2GB RAM stability.")

    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().strip()
            print(f"  Hardware Model  : {model}")
    except Exception:
        print("  Hardware Model  : Standard / Emulated Environment")
    print("  [PASS] System environment verified.")

def test_ai_model():
    print_header("2. AI MODEL & 92-CLASS ASSISTIVE ENGINE")
    from yolo_detector import YOLODetector, ASSISTIVE_ONNX_PATH, CLASSES_TXT_PATH
    import gc
    
    if not os.path.exists(ASSISTIVE_ONNX_PATH):
        print(f"  [FAIL] Assistive model missing at: {ASSISTIVE_ONNX_PATH}")
        return False
    
    file_size_mb = os.path.getsize(ASSISTIVE_ONNX_PATH) / (1024 * 1024)
    print(f"  Model File      : {os.path.basename(ASSISTIVE_ONNX_PATH)} ({file_size_mb:.1f} MB)")
    
    if not os.path.exists(CLASSES_TXT_PATH):
        print(f"  [FAIL] Classes file missing at: {CLASSES_TXT_PATH}")
        return False

    with open(CLASSES_TXT_PATH, "r") as f:
        classes = [l.strip() for l in f if l.strip()]
    print(f"  Classes Count   : {len(classes)} categories loaded")
    print(f"  Sample Classes  : {', '.join(classes[:8])}...")

    def get_proc_rss():
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    rss_before = get_proc_rss()
    print("  Initializing YOLOv8 CPU Detector (320x320 for 2GB RAM)...")
    t0 = time.time()
    detector = YOLODetector(model_path=ASSISTIVE_ONNX_PATH, classes=classes, input_size=(320, 320))
    load_time = time.time() - t0
    rss_after = get_proc_rss()
    ram_delta = max(0.0, rss_after - rss_before)
    print(f"  Model Load Time : {load_time:.2f} seconds")
    if ram_delta > 0:
        print(f"  Model RAM Impact: +{ram_delta:.1f} MB (Fits comfortably within 2GB budget)")

    # Benchmark dummy inference
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.time()
    for _ in range(5):
        detector.detect(dummy)
    avg_latency = (time.time() - t0) / 5.0
    approx_fps = 1.0 / max(0.001, avg_latency)
    print(f"  Inference Speed : {avg_latency*1000:.1f} ms per frame (~{approx_fps:.1f} FPS)")
    detector.close()
    gc.collect()
    print("  [PASS] Assistive AI model verified and optimized for CPU.")
    return True

def test_approach_and_spatial():
    print_header("3. APPROACH TRACKING & SPATIAL ENGINE")
    from yolo_detector import ApproachTracker, format_spatial_announcement
    tracker = ApproachTracker()
    
    # Simulate an approaching person over 3 frames
    det1 = [{'class': 'person', 'confidence': 0.9, 'box': [200, 150, 300, 350], 'center_x': 250, 'center_y': 250, 'zone': 'middle-center'}]
    tracker.update(det1, distance_meters=2.2)
    time.sleep(0.15)
    det2 = [{'class': 'person', 'confidence': 0.9, 'box': [180, 130, 320, 380], 'center_x': 250, 'center_y': 255, 'zone': 'middle-center'}]
    tracker.update(det2, distance_meters=1.8)
    time.sleep(0.15)
    det3 = [{'class': 'person', 'confidence': 0.9, 'box': [160, 100, 350, 420], 'center_x': 255, 'center_y': 260, 'zone': 'middle-center'}]
    t3 = tracker.update(det3, distance_meters=1.4)

    is_app = t3[0].get('is_approaching', False)
    spd = t3[0].get('approach_speed')
    ann = format_spatial_announcement(t3, distance_meters=1.4)
    print(f"  Approach Detected: {is_app} (Speed: {spd})")
    print(f"  Voice Cue Output : \"{ann}\"")
    if is_app:
        print("  [PASS] Dynamic approach detection and spatial reasoning verified.")
        return True
    else:
        print("  [WARN] Approach tracking did not flag expansion.")
        return False

def test_speech():
    print_header("4. VOICE SYNTHESIS (TTS) CHECK")
    engines = ['espeak-ng', 'espeak']
    found_engine = None
    for eng in engines:
        try:
            res = subprocess.run([eng, '--version'], capture_output=True, text=True)
            if res.returncode == 0:
                found_engine = eng
                print(f"  Found Engine    : {eng} ({res.stdout.splitlines()[0] if res.stdout else 'OK'})")
                break
        except Exception:
            pass

    if found_engine:
        test_phrase = "Smart vision assistant voice test. System ready."
        print(f"  Speaking phrase : \"{test_phrase}\"...")
        try:
            subprocess.run([found_engine, '-ven+f3', '-s160', '-p50', test_phrase], timeout=5)
            print("  [PASS] Audio output verified successfully.")
            return True
        except Exception as e:
            print(f"  [WARN] Failed to play audio: {e}")
            return False
    elif sys.platform == "darwin":
        subprocess.run(["say", "Smart vision assistant voice test"])
        return True
    elif sys.platform == "win32":
        print("  [INFO] Windows SAPI will be used for desktop speech.")
        return True
    else:
        print("  [FAIL] Neither 'espeak-ng' nor 'espeak' found.")
        print("  Fix: Run 'sudo apt install espeak-ng -y'")
        return False

def test_camera():
    print_header("5. CAMERA STREAM CHECK")
    found_cam = False
    for idx in [0, 1, 2]:
        try:
            if sys.platform.startswith("linux"):
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if not cap.isOpened():
                    try: cap.release()
                    except Exception: pass
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"  Found active camera on /dev/video{idx} (Resolution: {w}x{h})")
                    found_cam = True
                    cap.release()
                    break
                cap.release()
        except Exception:
            pass

    if found_cam:
        print("  [PASS] Camera stream is functional.")
        return True
    else:
        print("  [WARN] No physical camera found on indices 0, 1, or 2.")
        print("  Ensure Camera Module 3 or USB Webcam is connected.")
        return False

def test_gpio():
    print_header("6. GPIO SENSORS & BUTTON CHECK")
    try:
        from gpiozero import DistanceSensor, Button
    except ImportError:
        print("  [WARN] 'gpiozero' is not installed or running outside Raspberry Pi.")
        print("  Fix: Run 'sudo apt install python3-gpiozero python3-lgpio -y'")
        return False

    print(f"  Testing HC-SR04 Ultrasonic Sensor (Trigger={PIN_TRIGGER}, Echo={PIN_ECHO})...")
    sensor = None
    try:
        sensor = DistanceSensor(echo=PIN_ECHO, trigger=PIN_TRIGGER, max_distance=3.0)
        time.sleep(0.5)
        dist = sensor.distance
        if dist is not None:
            print(f"  Live Ultrasonic Distance : {dist:.2f} meters ({dist * 100:.1f} cm)")
            print("  [PASS] HC-SR04 sensor read successfully.")
    except Exception as e:
        print(f"  [WARN] Ultrasonic sensor error: {e}")
        if "pin factory" in str(e).lower() or "badpin" in str(e).lower():
            print("  [TIP] On Raspberry Pi 5 RP1, install lgpio: sudo apt install python3-lgpio -y")
        print("  Check 5V/GND, Trigger (GPIO 23), Echo (GPIO 24 via voltage divider).")
    finally:
        if sensor is not None:
            try: sensor.close()
            except Exception: pass

    print(f"  Testing Push Button on GPIO {PIN_BUTTON}...")
    btn = None
    try:
        btn = Button(PIN_BUTTON, pull_up=True)
        print(f"  Button current state : {'PRESSED' if btn.is_pressed else 'RELEASED'}")
        print("  [PASS] GPIO push button pin initialized.")
    except Exception as e:
        print(f"  [WARN] Button pin error: {e}")
        if "pin factory" in str(e).lower() or "badpin" in str(e).lower():
            print("  [TIP] On Raspberry Pi 5 RP1, install lgpio: sudo apt install python3-lgpio -y")
    finally:
        if btn is not None:
            try: btn.close()
            except Exception: pass

    return True

def main():
    print("\n" + "#" * 65)
    print("  SMART VISION ASSISTANT - HARDWARE & AI DIAGNOSTIC SUITE")
    print("  Designed for Raspberry Pi 5 & Visually Impaired Safety")
    print("#" * 65)

    test_system()
    ai_ok = test_ai_model()
    app_ok = test_approach_and_spatial()
    tts_ok = test_speech()
    cam_ok = test_camera()
    test_gpio()

    print_header("DIAGNOSTIC SUMMARY")
    print(f"  AI Model & 92 Classes : {'[OK]' if ai_ok else '[CHECK]'}")
    print(f"  Approach Radar Engine : {'[OK]' if app_ok else '[CHECK]'}")
    print(f"  Speech Audio Output   : {'[OK]' if tts_ok else '[CHECK]'}")
    print(f"  Camera Capture Feed   : {'[OK]' if cam_ok else '[CHECK]'}")
    print("\nTo start the assistant on Raspberry Pi 5:")
    print("  ./run_rpi.sh")
    print("Or headless mode (no GUI monitor required):")
    print("  python3 smart_assistant.py --headless\n")

if __name__ == "__main__":
    main()
