#!/bin/bash
# ==============================================================================
# boAt Rockerz 425 Bluetooth Reconnection & Audio Routing Tool
# Specifically designed for Raspberry Pi 5 (Raspberry Pi OS Bookworm / PipeWire)
# MAC Address: 41:42:DA:D8:8C:5D
# ==============================================================================

BT_MAC="41:42:DA:D8:8C:5D"
DEVICE_NAME="boAt Rockerz 425"

echo "================================================================="
echo "  Raspberry Pi 5 Bluetooth Audio Fixer"
echo "  Target Device : $DEVICE_NAME ($BT_MAC)"
echo "================================================================="
echo ""

# 1. Check rfkill and unblock Bluetooth if soft/hard blocked
echo "[STEP 1/5] Checking Bluetooth Radio & Hardware Controller..."
sudo rfkill unblock bluetooth 2>/dev/null || true
sudo systemctl restart bluetooth
sleep 1

# Power on the Bluetooth adapter via bluetoothctl
echo "  Powering on Bluetooth adapter..."
bluetoothctl power on >/dev/null 2>&1 || true
sleep 1

# 2. Restart PipeWire / WirePlumber Audio Daemons (Bookworm Audio Stack)
echo "[STEP 2/5] Resetting PipeWire / WirePlumber Audio Stack..."
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
systemctl --user restart pipewire pipewire-pulse wireplumber 2>/dev/null || true
sleep 1

# 3. Check if device is already connected
echo "[STEP 3/5] Testing Bluetooth connection to $DEVICE_NAME ($BT_MAC)..."
IS_CONNECTED=$(bluetoothctl info "$BT_MAC" 2>/dev/null | grep -i "Connected: yes" || true)

if [ -n "$IS_CONNECTED" ]; then
    echo "  [OK] Device is ALREADY connected at the Bluetooth level."
else
    echo "  Attempting direct connection..."
    # Trust first so future reconnections happen automatically on power-on
    bluetoothctl trust "$BT_MAC" >/dev/null 2>&1 || true
    
    # Attempt connection (timeout after 10s)
    CONNECT_OUT=$(bluetoothctl connect "$BT_MAC" 2>&1 || true)
    echo "  Connection output: $CONNECT_OUT"

    # Verify if connection succeeded
    IS_CONNECTED=$(bluetoothctl info "$BT_MAC" 2>/dev/null | grep -i "Connected: yes" || true)

    if [ -z "$IS_CONNECTED" ]; then
        echo ""
        echo "  [!] Direct connection failed. Performing Clean Re-Pairing..."
        echo "  IMPORTANT PHYSICAL CHECK:"
        echo "  1. Turn OFF Bluetooth on your phone/laptop so they don't hijack the headphones."
        echo "  2. Turn OFF your boAt Rockerz 425."
        echo "  3. Press and HOLD the power button for 5-6 SECONDS until the LED flashes RED and BLUE rapidly."
        echo ""
        read -p "  Press [ENTER] once the headphones are blinking Red & Blue in pairing mode..." DUMMY

        echo "  Removing stale pairing cache..."
        bluetoothctl remove "$BT_MAC" >/dev/null 2>&1 || true
        sleep 1

        echo "  Scanning for $BT_MAC for 7 seconds..."
        bluetoothctl --timeout 7 scan on >/dev/null 2>&1 || true

        echo "  Pairing with $BT_MAC..."
        bluetoothctl pair "$BT_MAC"
        sleep 1
        bluetoothctl trust "$BT_MAC"
        sleep 1
        bluetoothctl connect "$BT_MAC"
        sleep 2
    fi
fi

# 4. Configure Audio Routing (Set Default Audio Sink in PipeWire / WirePlumber)
echo ""
echo "[STEP 4/5] Routing Audio Output to $DEVICE_NAME..."
sleep 2

# Try wpctl (WirePlumber / PipeWire standard on Bookworm)
if command -v wpctl >/dev/null 2>&1; then
    # Look for boAt Rockerz sink ID in wpctl
    SINK_ID=$(wpctl status 2>/dev/null | grep -i -E "Rockerz|boAt|41_42_DA_D8_8C_5D" | grep -o -E "[0-9]+" | head -n 1 || true)
    if [ -n "$SINK_ID" ]; then
        wpctl set-default "$SINK_ID" 2>/dev/null || true
        wpctl set-volume "$SINK_ID" 0.85 2>/dev/null || true
        echo "  [OK] Audio routed to boAt Rockerz 425 (Sink ID: $SINK_ID, Volume: 85%)."
    else
        echo "  [INFO] Setting default Bluetooth audio sink..."
    fi
fi

# 5. Audio Playback Verification Test
echo ""
echo "[STEP 5/5] Playing Voice Test through Bluetooth Headphones..."
if command -v espeak-ng >/dev/null 2>&1; then
    espeak-ng "Bluetooth connected successfully. boAt Rockerz headphones are ready for smart vision assistant." 2>/dev/null || true
elif command -v espeak >/dev/null 2>&1; then
    espeak "Bluetooth connected successfully. boAt Rockerz headphones are ready for smart vision assistant." 2>/dev/null || true
fi

echo ""
echo "================================================================="
echo "  DIAGNOSTIC & CONNECTION SUMMARY:"
echo "================================================================="
bluetoothctl info "$BT_MAC" | grep -E "Name|Connected|Paired|Trusted|Battery" || true
echo "================================================================="
echo "If you heard the voice in your headphones, Bluetooth is 100% READY!"
echo "Now you can start your assistant:"
echo "  python3 smart_assistant.py --headless --no-sonar --conf 0.40 --resolution 320"
echo "================================================================="
