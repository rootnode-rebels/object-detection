import cv2
import time
import os
import urllib.request
import subprocess
import numpy as np
from gpiozero import DistanceSensor, Button

# ==========================================
# 1. HARDWARE CONFIGURATION (WIRING)
# ==========================================
# Ultrasonic Sensor
# VCC -> 5V, GND -> GND
# TRIG -> GPIO 23
# ECHO -> Voltage Converter (HV) -> Voltage Converter (LV) -> GPIO 24
PIN_TRIGGER = 23
PIN_ECHO = 24

# Push Button
# Pin 1 -> GPIO 17, Pin 2 -> GND
PIN_BUTTON = 17

# Initialize Hardware
print("[INFO] Initializing Hardware...")
try:
    sensor = DistanceSensor(echo=PIN_ECHO, trigger=PIN_TRIGGER, max_distance=3.0)
    button = Button(PIN_BUTTON, pull_up=True)
except Exception as e:
    print(f"[ERROR] Could not initialize GPIO. Are you running this on the Raspberry Pi? Error: {e}")

# ==========================================
# 2. AI MODEL SETUP (MobileNet-SSD for CPU)
# ==========================================
MODEL_DIR = "model_files"
PROTOTXT = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt")
WEIGHTS = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def download_models():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    if not os.path.exists(PROTOTXT):
        print("[INFO] Downloading AI Model config...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt", PROTOTXT)
    
    if not os.path.exists(WEIGHTS):
        print("[INFO] Downloading AI Model weights (this may take a minute)...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.caffemodel", WEIGHTS)

download_models()

print("[INFO] Loading AI Model into CPU...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, WEIGHTS)

# ==========================================
# 3. TEXT-TO-SPEECH HELPER
# ==========================================
def speak(text):
    """Uses the lightweight 'espeak' engine to talk through the speaker."""
    print(f"🔊 SPEAKING: {text}")
    try:
        # Runs espeak asynchronously so it doesn't freeze the camera
        subprocess.Popen(['espeak', '-ven+f3', '-s150', text])
    except FileNotFoundError:
        print("[ERROR] 'espeak' is not installed. Run: sudo apt-get install espeak")

# ==========================================
# 4. MAIN APPLICATION LOOP
# ==========================================
print("[INFO] Starting Camera...")
cap = cv2.VideoCapture(0) # 0 is usually the default Pi/USB camera
time.sleep(2.0) # Warm up camera

print("[INFO] System Ready. Press the button or walk within 1 meter of an object.")
speak("System is ready.")

last_scan_time = 0
COOLDOWN = 3.0 # Wait 3 seconds between scans so it doesn't talk over itself

try:
    while True:
        # Continuously read distance
        distance_meters = sensor.distance
        
        # Trigger Condition: Object is closer than 1 meter OR Button is pressed
        if (distance_meters < 1.0 or button.is_pressed) and (time.time() - last_scan_time > COOLDOWN):
            
            trigger_reason = "Button Pressed" if button.is_pressed else f"Proximity Alert ({distance_meters:.2f}m)"
            print(f"\n[TRIGGER] {trigger_reason}. Scanning environment...")
            
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                continue
            
            # Prepare frame for AI
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            
            # Run AI Inference on CPU
            net.setInput(blob)
            detections = net.forward()
            
            # Find the most confident object
            best_confidence = 0
            best_object = "unknown object"
            
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter out weak detections (less than 50% confidence)
                if confidence > 0.50:
                    idx = int(detections[0, 0, i, 1])
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_object = CLASSES[idx]
            
            # Announce the result
            if best_confidence > 0.50:
                announcement = f"{best_object} detected at {distance_meters:.1f} meters."
                speak(announcement)
            else:
                speak("Obstacle ahead, but cannot identify object.")
                
            last_scan_time = time.time()
            
        time.sleep(0.1) # Small sleep to prevent maxing out the CPU 100% on the loop

except KeyboardInterrupt:
    print("[INFO] Shutting down...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    sensor.close()
    button.close()
