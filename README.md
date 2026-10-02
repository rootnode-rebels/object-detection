# Real-Time Edge AI Object Detection & Assistive Vision

A real-time object detection and smart vision assistant engineered for **Raspberry Pi 5** using a CPU-optimized **YOLOv8** architecture (no dedicated NPU or AI HAT required).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rootnode-rebels/ai-object-detection/blob/main/Build_Assistive_92_Model.ipynb)

---

## 🚀 Key Features

* **CPU-Optimized YOLOv8 Inference**: Runs lightweight YOLOv8n ONNX via OpenCV DNN with ARM NEON acceleration, achieving ~15-20 FPS directly on the Raspberry Pi 5 CPU.
* **80 COCO Classes with Spatial Awareness**: Detects everyday objects and calculates their spatial position (*"on your left"*, *"straight ahead"*, *"on your right"*).
* **Buffer-Free High-Speed Video Pipeline**: Threaded camera stream eliminates OpenCV internal frame buffer delay, guaranteeing zero-latency live vision analysis.
* **Proximity & Manual Triggers**:
  * HC-SR04 ultrasonic sensor continuously monitors for obstacles closer than 1.0 meter.
  * Tactile push button enables instant on-demand voice queries.
* **Non-Blocking Text-to-Speech**: Integrated `espeak` engine speaks directional audio alerts asynchronously without freezing the video stream.
* **Real-Time Telemetry HUD**: Live onscreen overlay displaying current FPS, CPU load %, distance reading, and object bounding boxes.
* **Hardware Benchmark Tool**: Built-in benchmarking script to measure frame rates, latencies, CPU load, and SoC temperatures.

---

## 📊 System Specifications

| Component | Specification |
|---|---|
| **Host System** | Raspberry Pi 5 (Quad-core ARM Cortex-A76 @ 2.4 GHz) |
| **Deep Learning Model** | YOLOv8n ONNX (80 COCO object classes) |
| **Inference Engine** | OpenCV DNN (CPU Backend with ARM NEON SIMD) |
| **Resolution Modes** | `320x320` (~15-20 FPS fast mode) / `640x640` (high accuracy mode) |
| **Proximity Sensor** | HC-SR04 Ultrasonic Distance Sensor |
| **Manual Trigger** | Push button with internal pull-up |
| **Audio Feedback** | `espeak` Text-to-Speech Engine |

---

## 🔌 Hardware Connections & Wiring Graph

The setup connects the HC-SR04 ultrasonic sensor, tactile push button, camera, and speaker directly to the Raspberry Pi 5.

### Hardware Connection Graph

```mermaid
graph TD
    subgraph Raspberry_Pi_5 ["Raspberry Pi 5 Host"]
        CSI["MIPI CSI / USB Port"]
        AUDIO_OUT["3.5mm Jack / USB Audio"]
        
        subgraph GPIO_Header ["40-Pin GPIO Header"]
            PIN2["Pin 2: 5V Power"]
            PIN6["Pin 6 / 9: GND"]
            PIN16["Pin 16: GPIO 23 (TRIG)"]
            PIN18["Pin 18: GPIO 24 (ECHO Safe)"]
            PIN11["Pin 11: GPIO 17 (Button)"]
        end
    end

    subgraph Ultrasonic_Sensor ["HC-SR04 Ultrasonic Distance Sensor"]
        US_VCC["VCC (5V)"]
        US_GND["GND"]
        US_TRIG["TRIG (Trigger)"]
        US_ECHO["ECHO (5V Output)"]
    end

    subgraph Voltage_Divider ["Voltage Divider (3.3V Logic Protection)"]
        R1["Resistor R1: 1kΩ"]
        R2["Resistor R2: 2kΩ"]
        DIV_NODE["3.3V Safe Tap Node"]
    end

    subgraph Peripherals ["Peripherals & Feedback"]
        BTN["Tactile Push Button"]
        CAM["Pi Camera Module 3 / USB Cam"]
        SPK["Speaker / Headphones"]
    end

    %% Camera Link
    CAM -->|"Raw Video Stream"| CSI

    %% Audio Link
    AUDIO_OUT -->|"Audio Output"| SPK

    %% Ultrasonic Wiring
    PIN2 -->|"5V DC Power"| US_VCC
    PIN6 --- US_GND
    PIN16 -->|"Trigger Pulse"| US_TRIG

    %% Voltage Divider Wiring
    US_ECHO --> R1
    R1 --> DIV_NODE
    DIV_NODE --> PIN18
    DIV_NODE --> R2
    R2 --- PIN6

    %% Push Button Wiring
    PIN11 -->|"Pull-Up Input"| BTN
    BTN --- PIN6

    classDef rpi fill:#c51a4a,stroke:#800020,stroke-width:2px,color:#fff;
    classDef sensor fill:#2e7d32,stroke:#1b5e20,stroke-width:2px,color:#fff;
    classDef divider fill:#f57c00,stroke:#b26a00,stroke-width:2px,color:#fff;
    classDef io fill:#5e35b1,stroke:#311b92,stroke-width:2px,color:#fff;

    class Raspberry_Pi_5,GPIO_Header rpi;
    class Ultrasonic_Sensor sensor;
    class Voltage_Divider divider;
    class Peripherals,BTN,CAM,SPK io;
```

---

### Pinout & Wiring Table

| Raspberry Pi 5 Pin | Header Label | Connects To | Wire / Role | Notes |
|---|---|---|---|---|
| **Pin 2** | `5V Power` | HC-SR04 `VCC` | Power (Red) | Supplies 5V required by HC-SR04 |
| **Pin 6 / 9** | `GND` | HC-SR04 `GND`, Button Pin 2, Divider `GND` | Ground (Black) | Common ground reference |
| **Pin 16** | `GPIO 23` | HC-SR04 `TRIG` | Signal (Yellow/Orange) | Sends $10\mu\text{s}$ ultrasonic pulse |
| **Pin 18** | `GPIO 24` | Center node of Voltage Divider ($1\text{k}\Omega / 2\text{k}\Omega$) | Signal (Green) | Stepped-down 3.3V safe Echo input |
| **Pin 11** | `GPIO 17` | Push Button Terminal 1 | Signal (Blue) | Internal pull-up enabled in software |
| **CSI / USB** | `CAM / USB 3.0`| Camera Module 3 or USB Webcam | Video In | Live camera stream |
| **3.5mm / USB**| `Audio Jack` | Mini Speaker / Amplified Speaker | Audio Out | Spoken voice alerts via `espeak` |

> [!CAUTION]
> **Protecting Raspberry Pi 5 3.3V GPIO Pins**:
> The HC-SR04 sensor outputs a **5V pulse** on its `ECHO` pin. Connecting this directly to the Pi's GPIO 24 can damage the processor's GPIO bank.
> Always use a **voltage divider**:
>
> $$V_{\text{out}} = 5\text{V} \times \left(\frac{2000\,\Omega}{1000\,\Omega + 2000\,\Omega}\right) = 3.33\text{V}$$
>
> * Connect `ECHO` $\rightarrow$ $1\text{k}\Omega$ resistor ($R_1$).
> * Connect the other end of $R_1$ to `GPIO 24` and to a $2\text{k}\Omega$ resistor ($R_2$).
> * Connect the other end of $R_2$ to `GND`.

---

## 🔄 End-to-End System Workflow

```mermaid
flowchart TD
    subgraph Initialization ["1. System Boot & Initialization"]
        INIT_START(["System Start"]) --> YOLO_INIT["Download & Load YOLOv8n ONNX Model"]
        YOLO_INIT --> CAM_INIT["Start ThreadedCameraStream Background Thread"]
        CAM_INIT --> GPIO_INIT["Initialize GPIO: DistanceSensor & Button"]
        GPIO_INIT --> READY_VOICE["Speak: AI vision system ready"]
    end

    subgraph Streaming ["2. Continuous Video Pipeline"]
        CAM_INIT -.-> CAM_LOOP["Threaded Camera Capture Loop"]
        CAM_LOOP -->|"Read Live Frame"| BUFFER["Thread-Safe Frame Buffer"]
        BUFFER -.->|"Latest Live Frame"| CPU_INFER
    end

    subgraph Polling ["3. Sensor Polling Loop (Every 100ms)"]
        READY_VOICE --> LOOP_START["Poll Distance & Button State"]
        LOOP_START --> READ_SENSORS["Read get_distance and is_button_pressed"]
        READ_SENSORS --> TRIGGER_CHECK{"Trigger Condition Met?<br/>Distance under 1.0m OR Button Pressed<br/>AND Cooldown over 3.0s"}
    end

    TRIGGER_CHECK -- No --> LOOP_START

    subgraph AI_Inference ["4. CPU YOLOv8 Object Detection"]
        TRIGGER_CHECK -- Yes --> GRAB_FRAME["Fetch Freshest Frame from Buffer"]
        GRAB_FRAME --> RESIZE["Normalize Blob to 320x320 or 640x640"]
        RESIZE --> CPU_INFER["Run OpenCV DNN Forward Pass on CPU"]
        CPU_INFER --> NMS["Vectorized NMS Candidate Extraction"]
        NMS --> FILTER_CONF{"Any Detection over 50% Confidence?"}
    end

    subgraph Audio_Feedback ["5. Spatial Awareness & Voice Output"]
        FILTER_CONF -- Yes --> CALC_DIR["Calculate Spatial Direction:<br/>Left / Straight Ahead / Right"]
        CALC_DIR --> FORMAT_ALERT["Format Speech: Object detected (Direction) at (Distance)m"]
        FILTER_CONF -- No --> FORMAT_GENERIC["Format Speech: Obstacle ahead, cannot identify object"]
        FORMAT_ALERT --> SPEAK_ASYNC["Speak Asynchronously via espeak"]
        FORMAT_GENERIC --> SPEAK_ASYNC
    end

    subgraph Telemetry ["6. Real-Time HUD Overlay & Display"]
        SPEAK_ASYNC --> OVERLAY["Render Bounding Boxes + Direction Labels"]
        OVERLAY --> HUD["Display Live FPS, CPU Load %, and Distance"]
        HUD --> RESET_TIMER["Reset 3.0s Cooldown Timer"]
        RESET_TIMER --> LOOP_START
    end

    subgraph Teardown ["7. Safe Shutdown"]
        LOOP_START -.->|"SIGINT / Ctrl+C"| CLEANUP["Release Camera Stream"]
        CLEANUP --> CLOSE_NET["Release DNN Model"]
        CLOSE_NET --> CLOSE_GPIO["Close GPIO Handles & Terminate TTS"]
        CLOSE_GPIO --> EXIT(["System Safely Halted"])
    end
```

---

## ⚡ Installation & Quick Start

### Option A: Local PC Simulation (Windows / Mac / Linux)
You can test the entire pipeline on your PC with your webcam, simulated distance sensor, and native voice synthesis:

1. **One-Click Run (Windows):**
   Simply double-click `run_pc_demo.bat`.

2. **Manual Command:**
   ```bash
   python simulate_pc.py --model model_files/yolov8_assistive_92.onnx --classes classes.txt --resolution 320
   ```
   *Controls during PC simulation:*
   * `[SPACE]` : Press virtual hardware tactile button (triggers on-demand query)
   * `[W]` / `[UP]` : Move closer to obstacle (decreases ultrasonic distance)
   * `[S]` / `[DOWN]` : Move farther from obstacle (increases ultrasonic distance)
   * `[A]` : Toggle Auto-walk radar oscillation mode
   * `[Q]` / `[ESC]` : Exit simulation

3. **Interactive Hardware Web Simulator:**
   Open `simulation.html` in any web browser to interactively test the ultrasonic radar, push button, HUD, and voice synthesis in a GUI environment.

---

### Option B: Physical Raspberry Pi 5 Deployment

#### 1. Install System Dependencies
On Raspberry Pi OS (Bookworm):
```bash
sudo apt update && sudo apt install python3-opencv python3-gpiozero python3-psutil python3-numpy espeak -y
```

#### 2. Install Python Packages
```bash
pip install -r requirements.txt --break-system-packages
```

#### 3. Run the Smart Assistant

**Standard 92-Class Assistive Mode (~15-20 FPS on Pi 5 CPU):**
```bash
python3 smart_assistant.py --model model_files/yolov8_assistive_92.onnx --classes classes.txt --resolution 320
```

**COCO-80 Baseline Mode:**
```bash
python3 smart_assistant.py --resolution 320
```

**Headless Mode (for battery-powered wearable use):**
```bash
python3 smart_assistant.py --headless
```

#### 4. Run Benchmark Tool
```bash
python3 benchmark_performance.py
```

---

## 🎯 92 Supported Assistive Classes

The custom-trained assistive model detects 92 essential indoor, outdoor, and hazard objects:

* **Pedestrian & Navigation Hazards**: `staircase`, `door`, `pothole`, `manhole`, `speed breaker`, `pedestrian crossing`, `road barrier`, `road divider`, `traffic cone`, `drain`, `fence`, `gate`, `construction material`, `rock`, `stone`
* **Vehicles & Street Mobility**: `person`, `auto-rickshaw`, `car`, `bus`, `truck`, `motorcycle`, `scooter`, `bicycle`, `street light`, `traffic light`, `fire hydrant`
* **Indoor Furniture & Landmarks**: `chair`, `bench`, `table`, `desk`, `bed`, `sofa`, `cabinet`, `cupboard`, `shelf`, `switchboard`, `electrical socket`, `sink`, `toilet`
* **Everyday Objects**: `water bottle`, `backpack`, `mobile phone`, `laptop`, `book`, `cup`, `glass`, `plate`, `bowl`, `spoon`, `fork`

---

## 📁 Repository Structure

```text
├── run_pc_demo.bat             # 1-Click Windows PC demo launcher
├── simulate_pc.py              # Native PC simulation & CV demo with speech
├── simulation.html             # Interactive browser-based hardware simulator
├── smart_assistant.py          # Main real-time Raspberry Pi 5 assistant & HUD
├── yolo_detector.py            # CPU-optimized YOLOv8 inference engine
├── build_assistive_model.py    # Zero-shot CLIP encoder for custom classes
├── classes.txt                 # List of 92 assistive object classes
├── benchmark_performance.py    # Performance & thermal benchmarking tool
├── SETUP_GUIDE.md              # Raspberry Pi 5 hardware wiring & setup guide
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & wiring diagrams
└── .gitignore                  # Git ignore rules
```

---

## 📄 License
MIT License

