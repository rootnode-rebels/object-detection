# Raspberry Pi 5 + Hailo-8 AI HAT+ Deployment Guide

This guide details the setup and configuration required to fulfill the specifications outlined in **Executive Briefing: Hardware Architecture Justification**.

---

## 1. Hardware Architecture Requirements

* **SBC**: Raspberry Pi 5 (4GB or 8GB RAM).
* **NPU**: Raspberry Pi AI HAT+ (Hailo-8 26 TOPS or Hailo-8L 13 TOPS).
* **Connection**: PCIe ribbon cable connected directly to the Pi 5 PCIe FFC port.
* **Camera**: Raspberry Pi Camera Module 3 or high-speed USB camera.
* **Sensors**:
  * HC-SR04 Ultrasonic Sensor (Trigger -> GPIO 23, Echo -> 3.3V Divider -> GPIO 24).
  * Push Button (Pin 1 -> GPIO 17, Pin 2 -> GND).
* **Speaker**: 3.5mm audio jack or USB speaker for `espeak` feedback.

---

## 2. Raspberry Pi 5 PCIe & Driver Configuration

### Step 1: Enable PCIe Gen 3 Speed
Edit the Raspberry Pi boot configuration:
```bash
sudo nano /boot/firmware/config.txt
```
Append the following lines to the end of the file:
```ini
# Enable PCIe Gen 3 for Hailo-8 AI HAT+ Maximum Throughput
dtparam=pciex1
dtparam=pciex1_gen=3
```
Save and exit (`Ctrl + O`, `Enter`, `Ctrl + X`).

### Step 2: Install Hailo Software Stack
Update system packages and install Hailo runtime drivers, CLI utilities, and Python bindings:
```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install hailo-all python3-opencv python3-gpiozero python3-psutil espeak -y
```

### Step 3: Reboot the System
```bash
sudo reboot
```

### Step 4: Verify NPU Detection & Firmware
Run the official Hailo control utility to verify that the NPU is active:
```bash
hailortcli fw-control identify
```
You should see output similar to:
```text
Executing on device: 0000:01:00.0
Identifying board
Control Protocol Version: 2
Firmware Version: 4.17.0
Logger Version: 0
Device Architecture: HAILO8
Part Number: HM218B1C2FA
Product Name: HAILO-8 AI MODULE
```

---

## 3. Project Setup & Execution

### Step 1: Clone / Copy Project Files
```bash
cd ~/ai-object-detection-main
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt --break-system-packages
```

### Step 3: Run the Edge AI Smart Assistant
Launch with live graphical HUD:
```bash
python3 smart_assistant.py
```
Or launch in headless mode (for background service / headless battery operation):
```bash
python3 smart_assistant.py --headless
```

### Step 4: Validate Performance Against the Executive Briefing
Run the benchmark script to confirm performance targets:
```bash
python3 benchmark_performance.py
```

Expected validation targets:
* **Frame Rate**: ~60 FPS
* **CPU Utilization**: < 15%
* **SoC Temperature**: Stable, no thermal throttling
