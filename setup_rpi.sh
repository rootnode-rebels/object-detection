#!/bin/bash
# ==============================================================================
# One-Click Automated Setup for Raspberry Pi 5 Smart Vision Assistant
# Configures system, dependencies, audio, permissions, and auto-start on boot
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_USER="$(whoami)"

echo "================================================================="
echo "  Smart Vision Assistant - Raspberry Pi 5 Automated Setup"
echo "  Optimized for Visually Impaired Safety"
echo "================================================================="
echo ""
echo "[INFO] Running setup in directory: $SCRIPT_DIR"
echo "[INFO] Current user: $CURRENT_USER"
echo ""

# 1. Update APT Packages
echo "[STEP 1/6] Updating system package index..."
sudo apt update -y

# 2. Install Required System Dependencies
echo "[STEP 2/6] Installing OpenCV, GPIO, voice engine, and system tools..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-opencv \
    python3-gpiozero \
    python3-psutil \
    python3-numpy \
    espeak-ng \
    v4l-utils \
    alsa-utils

# 3. Install Python Dependencies
echo "[STEP 3/6] Installing Python requirements..."
cd "$SCRIPT_DIR"
pip install -r requirements.txt --break-system-packages

# 4. Set Permissions & User Groups
echo "[STEP 4/6] Setting executable permissions and hardware access groups..."
chmod 755 run_rpi.sh test_rpi_hardware.py setup_rpi.sh
sudo usermod -aG video,audio,gpio "$CURRENT_USER" 2>/dev/null || true

# 5. Verify Model File
echo "[STEP 5/6] Verifying 92-Class Assistive AI Model..."
MODEL_FILE="$SCRIPT_DIR/model_files/yolov8_assistive_92.onnx"
if [ -f "$MODEL_FILE" ]; then
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo "  [OK] Assistive model present ($MODEL_SIZE)"
else
    echo "  [WARN] Assistive model missing at $MODEL_FILE."
    echo "  Checking for yolov8n.onnx fallback..."
    if [ ! -f "$SCRIPT_DIR/model_files/yolov8n.onnx" ]; then
        python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from yolo_detector import download_yolo_onnx; download_yolo_onnx()"
    fi
fi

# 6. Configure Systemd Auto-Start Service (For Blind Wearable Bootup)
echo "[STEP 6/6] Configuring auto-start service (smart_assistant.service)..."
SERVICE_PATH="/etc/systemd/system/smart_assistant.service"

sudo bash -c "cat > $SERVICE_PATH" <<EOF
[Unit]
Description=Smart Vision Assistant for Visually Impaired
After=multi-user.target sound.target network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/smart_assistant.py --headless --resolution 320
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo "  [OK] Service installed at $SERVICE_PATH"

read -p "Would you like the assistant to auto-start automatically when the Pi powers on? (y/n) [y]: " AUTO_BOOT
AUTO_BOOT=${AUTO_BOOT:-y}

if [[ "$AUTO_BOOT" =~ ^[Yy]$ ]]; then
    sudo systemctl enable smart_assistant.service
    echo "  [ENABLED] Auto-start enabled! The assistant will speak and start whenever the Pi boots."
else
    echo "  [INFO] Auto-start not enabled. You can manually start the assistant using: ./run_rpi.sh"
fi

echo ""
echo "================================================================="
echo "  SETUP COMPLETE! Running hardware diagnostics..."
echo "================================================================="
python3 test_rpi_hardware.py

echo ""
echo "To start the assistant manually:"
echo "  ./run_rpi.sh"
echo ""
echo "To manage the auto-start background service:"
echo "  sudo systemctl start smart_assistant.service    # Start background service"
echo "  sudo systemctl stop smart_assistant.service     # Stop background service"
echo "  sudo systemctl status smart_assistant.service   # View live status"
echo ""
