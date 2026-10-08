# Raspberry Pi 5 Smart Vision Assistant - Deployment Guide

This guide details the setup and configuration required for running the real-time AI object detection assistant directly on the **Raspberry Pi 5 CPU** (no AI HAT or NPU required), specifically designed for visually impaired and blind users.

---

## 1. Hardware Requirements & Pinout

* **SBC**: Raspberry Pi 5 (4GB or 8GB RAM recommended)
* **Camera**: Raspberry Pi Camera Module 3 or standard USB Webcam
* **Sensors**:
  * **HC-SR04 Ultrasonic Distance Sensor**:
    * `VCC` -> 5V (Pin 2 or 4)
    * `GND` -> Ground (Pin 6)
    * `Trigger` -> GPIO 23 (Pin 16)
    * `Echo` -> **Voltage Divider** (1kΩ + 2kΩ resistors to convert 5V Echo to 3.3V safe logic) -> GPIO 24 (Pin 18)
  * **Tactile Push Button**:
    * Terminal 1 -> GPIO 17 (Pin 11)
    * Terminal 2 -> Ground (Pin 9 or 14)
* **Speaker**: 3.5mm audio jack, USB speaker, or Bluetooth headset for low-latency `espeak-ng` voice feedback
* **Power Supply**: Official 27W USB-C Power Supply

---

## 2. Raspberry Pi 5 Software Setup

### Step 1: Update System Packages
```bash
sudo apt update && sudo apt full-upgrade -y
```

### Step 2: Install System Dependencies
Install OpenCV, GPIOzero, psutil, NumPy, and the voice synthesis engine:
```bash
sudo apt install python3-opencv python3-gpiozero python3-psutil python3-numpy espeak-ng -y
```

---

## 3. Project Installation

### Step 1: Clone or Copy Project Files
```bash
cd ~
git clone https://github.com/rootnode-rebels/object-detection.git
cd object-detection
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

---

## 4. Hardware & AI Diagnostics Suite

Before launching the assistant, run the automated diagnostic tool to verify all physical connections, the 92-class AI model, approach tracking, and audio:

```bash
python3 test_rpi_hardware.py
```

This verifies:
1. CPU & System environment
2. YOLOv8 92-class model loading & inference speed
3. Real-time approach motion tracking & spatial logic
4. Speech synthesis engine (`espeak-ng`)
5. Camera stream capture via V4L2
6. HC-SR04 ultrasonic sensor readings & tactile button state

---

## 5. Running the Smart Assistant on Raspberry Pi 5

### Method A: One-Command Launcher (Recommended)
```bash
chmod +x run_rpi.sh
./run_rpi.sh
```

### Method B: Wearable / Headless Mode (For Portable Blind Assistance)
Runs without requiring an attached HDMI monitor or desktop window:
```bash
python3 smart_assistant.py --headless
```

### Method C: Desktop Window Mode (For Testing / Monitor)
Displays live video HUD with 9-part spatial grid and `[APPROACHING >>]` hazard radar:
```bash
python3 smart_assistant.py --resolution 320
```

---

## 6. How the Blind Safety & Approach Features Work

1. **Approaching Object Radar**:
   * Uses multi-frame bounding box optical expansion and sensor fusion to detect objects moving towards the user (e.g. oncoming person, bicycle, car, or user closing in on an obstacle).
   * Voice alerts:
     * *Rapid approach / Vehicle*: `"Warning! Car approaching fast directly ahead! Move aside immediately!"`
     * *Steady approach*: `"Caution, person is approaching towards you directly ahead at 1.8 meters."`
     * *Side approach*: `"Caution, bicycle approaching on your right."`

2. **In Front & Ground Hazards**:
   * *Critical Proximity (< 1.0m)*: `"Stop! Chair directly in front of you at 0.8 meters. Step aside to the right."`
   * *Trip / Fall Hazards*: `"Warning, ground hazard in front of you: staircase at 1.5 meters. Watch your step."`
   * *Overhead Hazards*: `"Warning, overhead obstacle directly ahead: tree branch. Watch your head."`
   * *Clear Path*: `"Path clear ahead. Continue straight."`

3. **Tactile Push Button**:
   * Pressing the button at any moment triggers an instant on-demand spatial scan of the surroundings.
