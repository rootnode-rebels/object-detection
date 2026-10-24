#!/bin/bash
# ==============================================================================
# HC-SR04 Ultrasonic Sonar Diagnostic Launcher for Raspberry Pi 5
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo "  Smart Vision Assistant - HC-SR04 Sonar Diagnostic Tool"
echo "  Target Hardware: Raspberry Pi 5 (Trigger GPIO 23, Echo GPIO 24)"
echo "================================================================="
echo ""

# Ensure lgpio and gpiozero are installed
if ! python3 -c "import gpiozero" &>/dev/null; then
    echo "[INFO] Installing python3-gpiozero and python3-lgpio for Pi 5..."
    sudo apt update && sudo apt install python3-gpiozero python3-lgpio -y
fi

python3 test_sonar.py "$@"
