# 🌟 Smart Vision Assistant – Complete Project Handover & Architecture Guide
**AI-Powered Assistive Navigation for the Visually Impaired on Raspberry Pi 5 (2GB RAM)**

---

## 📋 Executive Overview
This document is the master technical handover for the **Smart Vision Assistant**. It details every stage of the project: how the AI model was designed and trained in the cloud, why the architecture was engineered without PyTorch on edge hardware, how multi-modal sensor fusion protects blind users, and how to replicate, test, and deploy the entire system.

---

## 📂 Handover Directory Structure

```text
Smart_Vision_Assistant_Complete_Handover/
│
├── 01_HOW_WE_DID_EVERYTHING_HANDOVER_GUIDE.md  <-- You are here (Complete Master Documentation)
│
├── 02_AI_Model_Training_And_Export/             <-- Cloud & Colab training and export files
│   ├── Build_Assistive_92_Model.ipynb           (1-Click Google Colab Notebook)
│   ├── build_assistive_model.py                 (Automation script for model distillation)
│   ├── train_custom_yolo.py                     (Custom labeled dataset training pipeline)
│   ├── classes.txt                              (92 custom assistive classes list)
│   ├── yolov8s-worldv2.pt                       (Base YOLO-World v2 checkpoint - 25.9MB)
│   └── yolov8_assistive_92.onnx                 (Final compiled ONNX model - 48.4MB)
│
├── 03_Raspberry_Pi_Production_Code/             <-- Deployment package for Raspberry Pi 5
│   ├── smart_assistant.py                       (Master system: camera + sonar + TTS coordinator)
│   ├── yolo_detector.py                         (OpenCV DNN CPU inference engine)
│   ├── test_rpi_hardware.py                     (Pre-flight hardware diagnostic test)
│   ├── test_sonar.py                            (Ultrasonic radar live test)
│   ├── test_codebase_integrity.py               (Regression & safety test suite)
│   ├── setup_rpi.sh                             (Automated Pi environment setup script)
│   ├── run_rpi.sh                               (One-click startup launcher)
│   ├── check_pi_memory.sh                       (Pi RAM & temperature telemetry monitor)
│   ├── connect_bluetooth.sh                     (Audio earpiece auto-pair script)
│   ├── classes.txt                              (Class names for inference)
│   ├── requirements.txt                         (Pip dependencies)
│   ├── smart_assistant.service                  (Linux systemd auto-start service)
│   └── model_files/
│       └── yolov8_assistive_92.onnx             (Deployed AI model)
│
├── 04_PC_Simulation_And_Testing/                <-- Live demo & simulation suite for PC/Mac
│   ├── simulate_pc.py                           (PC simulator with virtual radar & obstacle generator)
│   ├── run_pc_demo.bat                          (1-Click Windows simulator launcher)
│   ├── run_simulation_server.py                 (HTTP browser video streamer)
│   ├── simulation.html                          (Interactive browser telemetry dashboard)
│   ├── benchmark_performance.py                 (FPS & latency benchmarking script)
│   ├── prototype_showcase.html                  (Complete visual prototype showcase)
│   └── model_files/
│       └── yolov8_assistive_92.onnx
│
├── 05_Technical_Documentation_And_Wiring/       <-- Hardware diagrams & whitepapers (PDFs)
│   ├── How_To_Upload_And_Run_On_Raspberry_Pi.pdf
│   ├── Hardware_Architecture_and_Wiring_Guide.pdf
│   ├── Architecture_Decisions_and_Rationale.pdf
│   ├── Raspberry_Pi_Blind_Assistant_Complete_Setup_Guide.pdf
│   ├── Raspberry_Pi_5_Wiring_Schematic_LLC.png  (Circuit schematic)
│   ├── training_and_architecture_explainer.html (Interactive architecture dashboard)
│   ├── PROTOTYPE_SPECIFICATION.md
│   └── SETUP_GUIDE.md
│
└── 06_Video_Demonstrations_And_Walkthroughs/    <-- 1080p Explainer Videos
    ├── Colab_Model_Training_And_Why_No_PyTorch_Walkthrough.mp4 (Step-by-step Colab screen walkthrough)
    ├── How_We_Trained_Our_AI_Model_And_Why_No_PyTorch.mp4      (Full presentation on architecture)
    └── NPU_vs_CPU_Video_Simulation.mp4                         (Side-by-side CPU vs NPU benchmark)
```

---

## 🧠 Phase 1: How We Built & Trained the AI Model

### 1. The Challenge with Generic AI (COCO 80)
Standard object detection models (COCO) detect items like *apples, giraffes, kites, and umbrellas*, but are completely blind to the daily hazards faced by visually impaired individuals:
- **Ground Hazards**: *staircases, potholes, open manholes, curbs, speed breakers, drains, loose rocks*.
- **Overhead Hazards**: *tree branches, low-hanging signs, electric poles*.
- **Navigation Cues**: *doors, windows, handrails, walls, pillars, electrical switchboards*.
- **Urban Transit**: *auto-rickshaws, motorcycles, scooters, barriers*.

### 2. The Solution: Open-Vocabulary Foundation Distillation
Rather than spending months capturing and labeling 50,000 images from scratch, we engineered our model using **Ultralytics YOLO-World v2** (`yolov8s-worldv2.pt`).
- YOLO-World incorporates a **Vision-Language CLIP Text Encoder**.
- It can bind arbitrary text vocabularies directly to its multi-scale visual detection heads.
- We defined **92 high-priority classes** in [`classes.txt`](file:///d:/projects/objrct%20detection/object-detection/classes.txt).

### 3. The Cloud Training Pipeline (Google Colab)
We automated the training in [`Build_Assistive_92_Model.ipynb`](file:///d:/projects/objrct%20detection/object-detection/Build_Assistive_92_Model.ipynb):
1. **Model Initialization**: Loaded `yolov8s-worldv2.pt` onto a cloud GPU.
2. **Vocabulary Binding**: Executed `model.set_classes(classes)` to inject the 92 text embeddings into the offline prediction head.
3. **Graph Freezing & Simplification**: Exported the model via `model.export(format="onnx", imgsz=320, dynamic=False, simplify=True)`.
4. **Result**: A standalone, 48.4 MB mathematical file—[`yolov8_assistive_92.onnx`](file:///d:/projects/objrct%20detection/object-detection/model_files/yolov8_assistive_92.onnx)—with all 92 classes baked permanently into its weights.

---

## ⚡ Phase 2: Why No PyTorch on Raspberry Pi? (The 2GB Architecture Decision)

The target deployment hardware is a **Raspberry Pi 5 with 2 GB RAM**.

### The PyTorch Memory Bottleneck
- PyTorch (`import torch`) requires **1.4 GB to 1.8 GB of RAM** just to load in memory.
- The Raspberry Pi OS and system daemons require ~500 MB.
- Launching PyTorch on a 2GB Pi immediately forces the system into heavy swap thrashing, crashing with an **Out Of Memory (OOM) kernel kill**.
- Furthermore, PyTorch wheels occupy over **2.5 GB of disk space**.

### The OpenCV DNN Solution
Instead of PyTorch, we run the model using **OpenCV DNN (`cv2.dnn`)**:
1. **~50 MB RAM Footprint**: Leaves 1.5 GB of free RAM for camera buffers, radar threads, and audio synthesis.
2. **0 MB Extra Storage**: OpenCV is already installed for camera capture; no 2.5 GB PyTorch library needed.
3. **ARM NEON SIMD Acceleration**: OpenCV DNN executes pure C++ matrix operations compiled with ARM NEON vectorization on the 4-core Cortex-A76, delivering **5.5 to 7.0 FPS** with **~175ms latency**.
4. **No External NPU Required**: Eliminates the cost, heat, and supply-chain dependency of external PCIe Hailo accelerators or Coral NPUs.

---

## 🛡️ Phase 3: Blind-First Multi-Modal Safety Engine

Blind navigation requires specialized software heuristics beyond simple bounding-box rendering:

### 1. Multi-Modal Radar Fusion (The "Silent Wall" Solution)
- **Problem**: Camera AI detects discrete objects (people, chairs), but produces **zero bounding boxes for flat drywall, blank pillars, or closed glass doors**.
- **Solution**: The HC-SR04 ultrasonic sensor pings continuously at 20 Hz. If an obstacle is closer than 1.8m, it is integrated as a first-class obstacle even with 0 camera detections:
  > *"There is an obstacle directly in front of you, at 1.2 meters. Move left or right to bypass."*
  > *The system will NEVER announce "Path clear ahead" if an ultrasonic obstacle is detected within 1.8m.*

### 2. Preemptive Audio Interruption
- **Problem**: If the user is about to collide with a pole (<50cm), standard systems fail to interrupt ongoing non-urgent speech.
- **Solution**: Proximity stops (<50cm) and fast-approaching vehicles immediately **sever** ongoing non-critical speech (`force=True`) to deliver an emergency alarm:
  > *"Stop! Close obstacle directly ahead, at 35 centimeters! Stop."*

### 3. Human Intuitive Centimeter Phrasing
- Distances under 1.0 meter are phrased in **centimeters** (e.g., *"at 45 centimeters"*, *"at 70 centimeters"*) rather than robotic decimals (*"at 0.45 meters"*).

### 4. Head-Swivel Grace Period (6.0s TTL Memory)
- Stationary obstacles and people are announced **once** on first identification.
- An internal **6-second Time-To-Live (TTL)** prevents the assistant from re-announcing the same person if the user tilts or turns their head momentarily.

### 5. On-Demand Tactile Button Scan
- Pressing the physical button on GPIO 17 immediately interrupts speech and delivers a full 360° surroundings scan:
  > *"Surroundings scan: a person directly ahead at 1.8 meters, chair on your left. Path clear on your right."*

---

## 🔌 Phase 4: Hardware Assembly & Circuit Diagram

### Pin Connections (Raspberry Pi 5 RP1 Header)

| Component | Pin Function | Pi Physical Pin | GPIO Name | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **HC-SR04 VCC** | Power | Pin 2 or 4 | 5V Power | Powers the sensor transmitter. |
| **HC-SR04 GND** | Ground | Pin 6, 9, or 14 | Ground | Shared system ground. |
| **HC-SR04 TRIG** | Trigger Output | Pin 16 | **GPIO 23** | 10µs trigger pulse from Pi (3.3V logic OK). |
| **HC-SR04 ECHO** | Echo Input | Pin 18 | **GPIO 24** | **CRITICAL: Must pass through 1kΩ / 2kΩ voltage divider** to reduce 5V echo to 3.3V safe level! |
| **Push Button** | Tactile Query | Pin 11 | **GPIO 17** | Other leg connected to Ground (Uses internal pull-up). |
| **USB Camera** | Video Stream | USB Port | USB 2.0 | Plug into a black USB 2.0 port for lower power/EMI. |
| **Audio** | Voice Feedback | Bluetooth / 3.5mm | System Audio | Connect any standard Bluetooth earpiece. |

*Refer to [`Raspberry_Pi_5_Wiring_Schematic_LLC.png`](file:///d:/projects/objrct%20detection/object-detection/pi5_2gb_assistant/Raspberry_Pi_5_Wiring_Schematic_LLC.png) for the full breadboard wiring schematic.*

---

## 🚀 Phase 5: How to Run Live Demonstrations

### 1. Running on a PC / Laptop (Simulation Mode)
*No Raspberry Pi required! Runs on Windows, Mac, or Linux using standard webcams:*
1. Navigate to `04_PC_Simulation_And_Testing/`.
2. Double-click [`run_pc_demo.bat`](file:///d:/projects/objrct%20detection/object-detection/run_pc_demo.bat) (or run `python simulate_pc.py`).
3. **Interactive Controls**:
   - `W` / `S` or Up/Down arrows: Increase/decrease simulated obstacle distance.
   - `J` / `K` or Left/Right arrows: Shift obstacle left/right.
   - `A`: Toggle automatic walking simulation.
   - `Space`: Push-button surroundings scan.

### 2. Running on the Raspberry Pi 5
1. Copy `03_Raspberry_Pi_Production_Code/` to the Pi:
   ```bash
   scp -r 03_Raspberry_Pi_Production_Code/* pi@<pi-ip>:~/assistant/
   ```
2. Run automated setup:
   ```bash
   cd ~/assistant
   chmod +x setup_rpi.sh run_rpi.sh
   ./setup_rpi.sh
   ```
3. Run pre-flight hardware diagnostic:
   ```bash
   python3 test_rpi_hardware.py
   ```
4. Start the assistant:
   ```bash
   # Windowed mode (if connected to a screen)
   ./run_rpi.sh

   # Headless wearable mode (pure audio, no monitor needed)
   python3 smart_assistant.py --headless
   ```

---

## 🏆 Project Achievements & Summary
- **100% Offline Edge Computing**: Zero reliance on cloud APIs or internet connection.
- **Optimized for 2GB RAM**: Runs at 6 FPS and 50MB RAM on Raspberry Pi 5 CPU without external NPUs.
- **92 Custom Assistive Classes**: Tailored specifically for blind navigation and hazard evasion.
- **Multi-Modal Reliability**: Ultrasonic echo fusion prevents walking into blank walls, windows, and doors.
- **Human-Centric Audio**: Preemptive emergency audio, non-repetitive memory, and calm 110 wpm speech pacing.
