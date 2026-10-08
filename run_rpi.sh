#!/bin/bash
# ==============================================================================
# Smart Vision Assistant - Raspberry Pi 5 Startup Script
# Optimized for Visually Impaired Safety & Real-Time Performance
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo "  Smart Vision Assistant - Raspberry Pi 5 Launcher"
echo "  AI Model : YOLOv8 92-Class Assistive (CPU Optimized)"
echo "  Sensors  : Camera + HC-SR04 Ultrasonic + Tactile Push Button"
echo "  Features : Dynamic Approach Alert + 9-Part Spatial Audio"
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

# Ensure model exists
MODEL_PATH="model_files/yolov8_assistive_92.onnx"
if [ ! -f "$MODEL_PATH" ]; then
    echo "[WARNING] Assistive model not found at $MODEL_PATH."
    echo "[INFO] Checking for default yolov8n.onnx..."
    if [ ! -f "model_files/yolov8n.onnx" ]; then
        echo "[INFO] Downloading base model..."
        python3 -c "import sys; sys.path.insert(0, '.'); from yolo_detector import download_yolo_onnx; download_yolo_onnx()"
    fi
fi

# Ensure camera permissions if needed
if [ -e /dev/video0 ] && [ ! -r /dev/video0 ]; then
    sudo -n chmod 666 /dev/video* 2>/dev/null || true
fi

echo "[INFO] Starting Smart Vision Assistant on Raspberry Pi 5..."
echo "       Press Ctrl+C to stop."
echo ""

# Execute with any forwarded command-line arguments (e.g. --headless)
python3 smart_assistant.py \
    --model model_files/yolov8_assistive_92.onnx \
    --classes classes.txt \
    --resolution 320 \
    "$@"
