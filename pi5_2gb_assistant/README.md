# Raspberry Pi 5 (2 GB RAM) - Smart Vision Assistant Deployment Package

This folder contains **all the necessary files** to run the Smart Vision Assistant on a **Raspberry Pi 5 with 2 GB RAM**. All Windows-only binaries, web simulator files, and heavy PyTorch model weights have been excluded, leaving a lean, production-ready package (~64 MB total).

---

## 1. Hardware Pinout & Wiring

| Component | Pin / Connection | Raspberry Pi 5 Physical Pin |
| :--- | :--- | :--- |
| **HC-SR04 VCC** | 5V Power | Pin 2 or Pin 4 (5V) |
| **HC-SR04 GND** | Ground | Pin 6 (GND) |
| **HC-SR04 Trigger** | GPIO 23 | Pin 16 (GPIO 23) |
| **HC-SR04 Echo** | **Voltage Divider** (1kΩ + 2kΩ) -> GPIO 24 | Pin 18 (GPIO 24) *(Crucial: Converts 5V logic to safe 3.3V)* |
| **Tactile Push Button** | Terminal 1 -> GPIO 17<br>Terminal 2 -> Ground | Pin 11 (GPIO 17)<br>Pin 9 or 14 (GND) |
| **Camera** | USB Webcam or Pi Camera Module 3 | USB 3.0 port or CAM/DISP port |
| **Audio Speaker** | 3.5mm Jack / USB Audio / Bluetooth | Low-latency audio for `espeak-ng` |

---

## 2. How to Transfer ("Dump") This Folder to Raspberry Pi 5

Choose any of the following methods to transfer the `pi5_2gb_assistant` folder to your Raspberry Pi:

### Method A: Via USB Flash Drive (Easiest)
1. Copy the `pi5_2gb_assistant` folder onto a USB flash drive.
2. Insert the USB drive into your Raspberry Pi 5.
3. Open a terminal on the Pi and copy it to your home directory:
   ```bash
   cp -r /media/$USER/*/"pi5_2gb_assistant" ~/"pi5_2gb_assistant"
   cd ~/"pi5_2gb_assistant"
   ```

### Method B: Via SCP / Network Transfer (From PC PowerShell or Terminal)
From your PC terminal, run (replace `pi` and `raspberrypi.local` with your Pi username and IP address):
```bash
scp -r "c:\Users\suhas\Desktop\test\ai-object-detection-main\pi5_2gb_assistant" pi@raspberrypi.local:/home/pi/
```
Then SSH into the Pi:
```bash
ssh pi@raspberrypi.local
cd ~/pi5_2gb_assistant
```

---

## 3. One-Click Setup on Raspberry Pi 5

Once inside the folder on your Pi, run the automated setup script:

```bash
chmod +x *.sh test_rpi_hardware.py smart_assistant.py
sudo ./setup_rpi.sh
```

### What `setup_rpi.sh` Automatically Configures:
1. **2GB Swap Protection**: Detects 2GB RAM and automatically configures a 2048 MB swapfile. This prevents the Linux Out-Of-Memory (OOM) killer from terminating the assistant during peak memory spikes.
2. **Swappiness Tuning**: Sets `vm.swappiness = 15` so the SD card / NVMe drive isn't thrashed and swap is only used for overflow protection.
3. **Pre-compiled APT Packages**: Installs `python3-opencv`, `python3-numpy`, `python3-gpiozero`, `python3-psutil`, and `espeak-ng` from Debian repositories. This avoids building massive C++ wheels in RAM.
4. **Hardware Permissions**: Adds your user to `video`, `audio`, and `gpio` groups.
5. **Systemd Auto-Start Service**: Configures `/etc/systemd/system/smart_assistant.service` with a `MemoryMax=1.6G` ceiling and `MALLOC_ARENA_MAX=2` to ensure wearable bootup reliability.

---

## 4. Run Hardware & Memory Diagnostics

Before launching, verify all sensors, memory limits, and the AI model:

```bash
python3 test_rpi_hardware.py
```

This diagnostic script tests:
* **System & RAM Diagnostics**: Verifies physical RAM, swap space, and memory headroom.
* **AI Model Verification**: Benchmarks model loading time and RAM delta (~120 MB RAM footprint).
* **Approach Tracking Radar**: Simulates expanding bounding boxes to verify hazard warnings.
* **Voice Output**: Tests `espeak-ng` text-to-speech audio.
* **Camera Capture**: Tests V4L2 video feed.
* **GPIO Sensors**: Reads distance from HC-SR04 and button state on GPIO 17.

---

## 5. Starting the Assistant

### Option 1: Standard Assistive Mode (Recommended)
Uses the 92-class Assistive Vision model tuned for blind navigation, indoor obstacles, and approaching hazards:
```bash
./run_rpi.sh
```

### Option 2: Ultra-Lightweight Mode (Lowest RAM Footprint)
Uses the standard 80 COCO-class YOLOv8n base model (total process RAM usage: ~65 MB):
```bash
./run_rpi.sh --ultra-light
```

### Option 3: Wearable / Headless Mode
Runs without requiring an attached HDMI monitor or desktop window (saves ~500 MB of RAM):
```bash
python3 smart_assistant.py --headless --resolution 320
```

---

## 6. Real-Time Memory & Health Monitoring

To check live RAM, swap, temperature, and assistant process memory on your 2GB Pi at any time:

```bash
./check_pi_memory.sh
```

---

## 7. Managing the Background Boot Service

If enabled during setup, the assistant will automatically speak and start whenever the Pi powers on.

```bash
sudo systemctl start smart_assistant.service    # Start background service
sudo systemctl stop smart_assistant.service     # Stop background service
sudo systemctl restart smart_assistant.service  # Restart service
sudo systemctl status smart_assistant.service   # View real-time service logs
```

---

## 8. Summary of 2GB RAM Optimizations

* **320x320 Input Resolution**: Keeps DNN activation buffers under 60 MB while maintaining ~18–24 FPS on the Pi 5 Cortex-A76 CPU.
* **V4L2 Single-Frame Buffer**: `CAP_PROP_BUFFERSIZE = 1` eliminates frame backlogs and prevents memory leaks.
* **Glibc Arena Limiting**: `MALLOC_ARENA_MAX = 2` prevents glibc heap fragmentation from multi-threaded OpenCV DNN operations.
* **Explicit Garbage Collection**: `gc.collect()` runs after model loading and periodically in the main loop.
* **Headless Auto-Detection**: Automatically runs headless when no graphical desktop is detected, saving ~500 MB of system RAM.
