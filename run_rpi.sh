#!/bin/bash
# ==============================================================================
# Smart Vision Assistant - Raspberry Pi 5 (2GB RAM) Launcher
# Optimized for Visually Impaired Safety & Real-Time Performance
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Glibc memory allocation optimization for 2GB RAM:
# Constrains the number of memory arenas to avoid virtual and physical fragmentation
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

echo "================================================================="
echo "  Smart Vision Assistant - Raspberry Pi 5 Launcher"
echo "  Target Hardware : Raspberry Pi 5 (2 GB RAM Edition)"
echo "  Inference Engine: OpenCV DNN on ARM Cortex-A76 (No NPU required)"
echo "  Sensors         : Camera + HC-SR04 Ultrasonic + Tactile Push Button"
echo "  Features        : Dynamic Approach Alert + 9-Part Spatial Audio"
echo "================================================================="
echo ""

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed. Run: sudo apt install python3 -y"
    exit 1
fi

# Check espeak-ng / espeak
if ! command -v espeak-ng &> /dev/null && ! command -v espeak &> /dev/null; then
    echo "[WARNING] Neither espeak-ng nor espeak found. Installing espeak-ng..."
    sudo apt update && sudo apt install espeak-ng -y
fi

# Pre-flight RAM headroom check
AVAIL_RAM_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo "1024")
TOTAL_RAM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo "2048")

echo "[INFO] System RAM Status: ${AVAIL_RAM_MB} MB available of ${TOTAL_RAM_MB} MB total"
if [ "$AVAIL_RAM_MB" -lt 250 ]; then
    echo "[WARN] Available RAM is critically low (<250MB)!"
    echo "       Consider closing browser/desktop apps or running with --ultra-light."
fi

# Handle model selection (Assistive 92 vs Ultra-Light Base)
RUN_ARGS=()
USE_ULTRA_LIGHT=false

for arg in "$@"; do
    if [ "$arg" == "--ultra-light" ] || [ "$arg" == "--lightweight" ]; then
        USE_ULTRA_LIGHT=true
    else
        RUN_ARGS+=("$arg")
    fi
done

if [ "$USE_ULTRA_LIGHT" = true ]; then
    MODEL_PATH="model_files/yolov8n.onnx"
    echo "[INFO] Mode: ULTRA-LIGHT BASE (YOLOv8n 80-Class, ~65MB RAM Footprint)"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "[INFO] Downloading base model..."
        python3 -c "import sys; sys.path.insert(0, '.'); from yolo_detector import download_yolo_onnx; download_yolo_onnx()"
    fi
    MODEL_ARGS=(--model "$MODEL_PATH")
else
    MODEL_PATH="model_files/yolov8_assistive_92.onnx"
    echo "[INFO] Mode: ASSISTIVE 92-CLASS (Tuned for Obstacles & Blind Safety)"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "[WARN] Assistive model not found at $MODEL_PATH."
        echo "[INFO] Falling back to yolov8n.onnx..."
        MODEL_PATH="model_files/yolov8n.onnx"
        if [ ! -f "$MODEL_PATH" ]; then
            python3 -c "import sys; sys.path.insert(0, '.'); from yolo_detector import download_yolo_onnx; download_yolo_onnx()"
        fi
        MODEL_ARGS=(--model "$MODEL_PATH")
    else
        MODEL_ARGS=(--model "$MODEL_PATH" --classes classes.txt)
    fi
fi

# Auto-detect headless mode if no GUI display is attached
HAS_HEADLESS_ARG=false
for arg in "${RUN_ARGS[@]}"; do
    if [ "$arg" == "--headless" ]; then
        HAS_HEADLESS_ARG=true
        break
    fi
done

if [ "$HAS_HEADLESS_ARG" = false ] && [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
    echo "[INFO] No display server detected. Auto-enabling --headless mode."
    RUN_ARGS+=(--headless)
fi

# Ensure camera permissions if needed
if [ -e /dev/video0 ] && [ ! -r /dev/video0 ]; then
    sudo -n chmod 666 /dev/video* 2>/dev/null || true
fi

echo "[INFO] Starting Smart Vision Assistant on Raspberry Pi 5..."
echo "       Press Ctrl+C to stop."
echo ""

# Execute with 320x320 resolution (optimal FPS & lowest RAM for Pi 5 2GB)
python3 smart_assistant.py \
    "${MODEL_ARGS[@]}" \
    --resolution 320 \
    "${RUN_ARGS[@]}"
