#!/bin/bash
# ==============================================================================
# One-Click Automated Setup for Raspberry Pi 5 (2GB RAM Edition)
# Smart Vision Assistant for Visually Impaired Safety
#
# Configures:
# 1. System packages & pre-compiled ARM64 OpenCV
# 2. 2GB Swapfile + kernel swappiness protection (Prevents OOM on 2GB RAM)
# 3. Audio (espeak-ng), Camera (V4L2), and GPIO permissions
# 4. Glibc memory arena tuning (MALLOC_ARENA_MAX=2)
# 5. Background auto-start systemd service with memory ceilings
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Correctly detect the non-root login user when invoked with 'sudo ./setup_rpi.sh'
if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
    CURRENT_USER="$SUDO_USER"
else
    CURRENT_USER="$(whoami)"
fi

USER_UID=$(id -u "$CURRENT_USER" 2>/dev/null || echo "1000")

echo "================================================================="
echo "  Smart Vision Assistant - Raspberry Pi 5 Automated Setup"
echo "  Target Hardware: Raspberry Pi 5 (2 GB RAM Edition)"
echo "  Optimized for Visually Impaired Safety & Low Memory Footprint"
echo "================================================================="
echo ""
echo "[INFO] Running setup in directory: $SCRIPT_DIR"
echo "[INFO] Target user account      : $CURRENT_USER (UID: $USER_UID)"
echo ""

# 1. Memory Pre-Check & Swap Configuration for 2GB RAM Pi 5
echo "[STEP 1/7] Analyzing System RAM and Configuring Swap Safety Net..."
TOTAL_RAM_KB=$(grep -m1 MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
TOTAL_RAM_MB=$(( ${TOTAL_RAM_KB:-2097152} / 1024 ))
SWAP_TOTAL_KB=$(grep -m1 SwapTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
SWAP_TOTAL_MB=$(( ${SWAP_TOTAL_KB:-0} / 1024 ))

echo "  Physical RAM detected : ${TOTAL_RAM_MB} MB"
echo "  Current Swap active   : ${SWAP_TOTAL_MB} MB"

if [ "$SWAP_TOTAL_MB" -lt 1500 ]; then
    echo "  [OPTIMIZATION] Total swap is below 1.5GB. Configuring 2048MB swap for 2GB Pi 5 stability..."
    if [ -f /etc/dphys-swapfile ]; then
        sudo dphys-swapfile swapoff 2>/dev/null || true
        if grep -q "CONF_SWAPSIZE" /etc/dphys-swapfile; then
            sudo sed -i 's/^#*CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
        else
            echo "CONF_SWAPSIZE=2048" | sudo tee -a /etc/dphys-swapfile >/dev/null
        fi
        sudo dphys-swapfile setup
        sudo dphys-swapfile swapon
        echo "  [OK] /etc/dphys-swapfile expanded to 2048 MB."
    elif [ ! -f /swapfile ]; then
        echo "  Creating dedicated 2GB swapfile at /swapfile..."
        sudo fallocate -l 2G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
        sudo chmod 600 /swapfile
        sudo mkswap /swapfile
        sudo swapon /swapfile
        if ! grep -q '/swapfile' /etc/fstab; then
            echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
        fi
        echo "  [OK] 2GB swapfile active and registered in /etc/fstab."
    fi
else
    echo "  [OK] Swap configuration is sufficient (${SWAP_TOTAL_MB} MB)."
fi

# Set optimal swappiness (swappiness=15 reserves swap strictly for overflow without thrashing SD card)
sudo sysctl -w vm.swappiness=15 >/dev/null 2>&1 || true
if [ -d /etc/sysctl.d ]; then
    echo "vm.swappiness=15" | sudo tee /etc/sysctl.d/99-pi5-memory.conf >/dev/null
fi

# 2. Update APT Package Repositories
echo ""
echo "[STEP 2/7] Updating system package index..."
sudo apt update -y

# 3. Install Required Pre-Compiled System Dependencies via APT
# NOTE: Installing opencv, numpy, and psutil via APT avoids compiling heavy wheels in RAM!
echo ""
echo "[STEP 3/7] Installing pre-compiled OpenCV, GPIO, voice engine, and system tools..."
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

# 4. Install Python Dependencies safely with pip
echo ""
echo "[STEP 4/7] Verifying Python requirements..."
pip install -r requirements.txt --break-system-packages --no-cache-dir

# 5. Set Permissions & Hardware Access Groups
echo ""
echo "[STEP 5/7] Setting permissions and hardware access groups for $CURRENT_USER..."
chmod +x *.sh test_rpi_hardware.py smart_assistant.py 2>/dev/null || true
sudo usermod -aG video,audio,gpio "$CURRENT_USER" 2>/dev/null || true

# 6. Verify Model Files
echo ""
echo "[STEP 6/7] Verifying YOLOv8 Assistive AI Models..."
ASSISTIVE_MODEL="$SCRIPT_DIR/model_files/yolov8_assistive_92.onnx"
BASE_MODEL="$SCRIPT_DIR/model_files/yolov8n.onnx"

if [ -f "$ASSISTIVE_MODEL" ]; then
    M_SIZE=$(du -h "$ASSISTIVE_MODEL" | cut -f1)
    echo "  [OK] 92-Class Assistive model present ($M_SIZE)"
else
    echo "  [WARN] Assistive model missing at $ASSISTIVE_MODEL."
fi

if [ -f "$BASE_MODEL" ]; then
    B_SIZE=$(du -h "$BASE_MODEL" | cut -f1)
    echo "  [OK] Ultra-Light YOLOv8n base model present ($B_SIZE)"
else
    echo "  [INFO] Downloading base model fallback..."
    python3 -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from yolo_detector import download_yolo_onnx; download_yolo_onnx()"
fi

# 7. Configure Systemd Auto-Start Service (For Wearable Bootup with Memory Guards)
echo ""
echo "[STEP 7/7] Configuring systemd background service (smart_assistant.service)..."
SERVICE_PATH="/etc/systemd/system/smart_assistant.service"

sudo bash -c "cat > $SERVICE_PATH" <<EOF
[Unit]
Description=Smart Vision Assistant for Visually Impaired (Pi 5 2GB)
After=multi-user.target sound.target network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/smart_assistant.py --headless --resolution 320
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1
Environment=MALLOC_ARENA_MAX=2
Environment=XDG_RUNTIME_DIR=/run/user/$USER_UID
MemoryMax=1.6G
MemoryHigh=1.4G
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo "  [OK] Service installed at $SERVICE_PATH (Guarded with MemoryMax=1.6G)"

echo ""
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
echo "  SETUP COMPLETE! Running hardware & memory diagnostics..."
echo "================================================================="
if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
    sudo -u "$CURRENT_USER" python3 test_rpi_hardware.py
else
    python3 test_rpi_hardware.py
fi

echo ""
echo "Quick Management Commands:"
echo "  ./run_rpi.sh                # Start assistant manually (interactive/monitor)"
echo "  ./run_rpi.sh --ultra-light  # Start with ultra-light base model (~65MB RAM)"
echo "  ./check_pi_memory.sh        # Check live RAM, swap, and temperature"
echo ""
echo "Background Service Commands:"
echo "  sudo systemctl start smart_assistant.service    # Start background service"
echo "  sudo systemctl stop smart_assistant.service     # Stop background service"
echo "  sudo systemctl status smart_assistant.service   # View live status"
echo ""
