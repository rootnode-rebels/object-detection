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
cd ~
git clone https://github.com/rootnode-rebels/object-detection.git
cd object-detection
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

---

## 4. Generating the 92-Class Assistive Model (Method 1: YOLO-World)

To enable detection and voice announcements for all **92 custom objects** (staircases, doors, auto-rickshaws, potholes, switches, etc.) without manual dataset annotation:

### Option A: Generate in 30 Seconds via Free Google Colab (Recommended)
1. Open a new notebook at [Google Colab](https://colab.research.google.com).
2. Paste and run this single code block:
```python
!git clone https://github.com/rootnode-rebels/object-detection.git
%cd object-detection
!pip install ultralytics onnx
!python build_assistive_model.py --mode yoloworld --imgsz 320

from google.colab import files
files.download('model_files/yolov8_assistive_92.onnx')
```
3. Copy the downloaded `yolov8_assistive_92.onnx` into the `model_files/` folder on your Raspberry Pi.

### Option B: Generate Directly on Raspberry Pi 5
```bash
pip install ultralytics onnx --break-system-packages
python3 build_assistive_model.py --mode yoloworld --imgsz 320
```

---

## 5. Running the Smart Assistant

### Run with the 92-Class Assistive Model:
```bash
python3 smart_assistant.py --model model_files/yolov8_assistive_92.onnx --classes classes.txt --resolution 320
```

**Options**:
* `--resolution 320`: Fast mode (~15-20 FPS on Pi 5 CPU, default)
* `--resolution 640`: High-resolution mode (maximum accuracy)
* `--headless`: Run without desktop GUI (for wearable / portable use)
* `--conf 0.5`: Detection confidence threshold

### Run Performance Benchmark:
```bash
python3 benchmark_performance.py
```
