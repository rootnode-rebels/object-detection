# Raspberry Pi 5 Smart Vision Assistant - Deployment Guide

This guide details the setup and configuration required for running the real-time AI object detection assistant directly on the **Raspberry Pi 5 CPU** (no AI HAT or NPU required).

---

## 1. Hardware Requirements

* **SBC**: Raspberry Pi 5 (4GB or 8GB RAM recommended)
* **Camera**: Raspberry Pi Camera Module 3 or USB Webcam
* **Sensors**:
  * HC-SR04 Ultrasonic Sensor (Trigger -> GPIO 23, Echo -> 3.3V Voltage Divider -> GPIO 24)
  * Tactile Push Button (Pin 1 -> GPIO 17, Pin 2 -> GND)
* **Speaker**: 3.5mm audio jack or USB speaker for `espeak` feedback
* **Power Supply**: Official 27W USB-C Power Supply

---

## 2. Raspberry Pi 5 Software Setup

### Step 1: Update System Packages
```bash
sudo apt update && sudo apt full-upgrade -y
```

### Step 2: Install System Dependencies
Install OpenCV, GPIOzero, and the `espeak` voice engine:
```bash
sudo apt install python3-opencv python3-gpiozero python3-psutil python3-numpy espeak -y
```

---

## 3. Project Installation & Execution

### Step 1: Clone or Copy Project Files
```bash
cd ~/ai-object-detection
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### Step 3: Run the Smart Assistant
Launch with live graphical HUD:
```bash
python3 smart_assistant.py
```

**Options**:
* `--resolution 320`: Fast mode (~15-20 FPS on Pi 5 CPU, default)
* `--resolution 640`: High-resolution mode (maximum accuracy)
* `--headless`: Run without desktop GUI (for battery / portable use)
* `--conf 0.5`: Confidence threshold

```bash
# Example: Run headless in fast mode
python3 smart_assistant.py --resolution 320 --headless
```

### Step 4: Run Performance Benchmark
Validate the CPU frame rate and temperature on your device:
```bash
python3 benchmark_performance.py
```
