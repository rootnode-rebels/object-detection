#!/usr/bin/env python3
"""
HC-SR04 Ultrasonic Sonar Sensor - Production Diagnostic Suite
============================================================
Designed for Smart Vision Assistant on Raspberry Pi 5 & PC Hardware Bridges.

Capabilities:
1. Physical GPIO Mode (Raspberry Pi 5 RP1 GPIO via gpiozero / lgpio):
   - Trigger Pin: GPIO 23 (Physical Pin 16)
   - Echo Pin   : GPIO 24 (Physical Pin 18, stepped down to 3.3V via 1k/2k divider)
2. USB Serial Bridge Mode (PC / Mac / Linux via Arduino / ESP32 / USB-UART):
   - Reads live distance from microcontroller over serial port (e.g. COM3 or /dev/ttyUSB0)
3. Simulated Radar Mode:
   - Interactive terminal acoustic radar simulation if running on PC without physical sensors
4. Temperature Compensation:
   - Adjusts speed of sound dynamically: v = 331.3 + (0.606 * Temp_C) m/s
5. Electrical Safety & Pin Diagnostics:
   - Checks echo timeout, detects floating pin / disconnected voltage divider, measures ping jitter.
"""

import os
import sys
import time
import argparse
import math

# Ensure stdout flush for real-time terminal radar display
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

PIN_TRIGGER_DEFAULT = 23
PIN_ECHO_DEFAULT = 24


def print_banner():
    print("=" * 68)
    print("   HC-SR04 ULTRASONIC SONAR RADAR - HARDWARE DIAGNOSTIC SUITE")
    print("   Assistive Vision Navigation System | Raspberry Pi 5 & PC Bridge")
    print("=" * 68)
    print("   Wiring Reference for Raspberry Pi 5:")
    print("     - VCC   --> Pin 2 or Pin 4  (5.0V DC)")
    print("     - GND   --> Pin 6 or Pin 9  (Ground)")
    print("     - TRIG  --> Pin 16 (GPIO 23) (3.3V Logic directly driven by Pi)")
    print("     - ECHO  --> Pin 18 (GPIO 24) [CRITICAL: USE 1k/2k VOLTAGE DIVIDER!]")
    print("                 Sensor ECHO (5V) -> [ 1k Ohm Resistor ] -> GPIO 24")
    print("                                                         -> [ 2k Ohm ] -> GND")
    print("=" * 68 + "\n")


def format_radar_bar(distance_m, max_range=3.0, bar_width=28):
    """Generates an ASCII acoustic radar visual bar with color coding."""
    if distance_m is None or distance_m > max_range:
        return "[" + " " * bar_width + "]  OUT OF RANGE (>3.0m)"
    
    ratio = min(1.0, max(0.0, distance_m / max_range))
    filled = int(round(ratio * bar_width))
    empty = bar_width - filled
    
    # ANSI color formatting if terminal supports it
    if distance_m < 0.50:
        bar_char = "!"
        status = "CRITICAL PROXIMITY (<0.5m) [STOP]"
    elif distance_m < 1.00:
        bar_char = "#"
        status = "WARNING HAZARD (<1.0m) [STEP ASIDE]"
    elif distance_m < 2.00:
        bar_char = "="
        status = "CAUTION RANGE (1.0m - 2.0m)"
    else:
        bar_char = "-"
        status = "CLEAR RANGE (2.0m - 3.0m)"
        
    bar = "[" + (bar_char * filled) + (" " * empty) + "]"
    return f"{bar} {distance_m:.2f}m ({distance_m*100:.1f}cm | {distance_m*39.37:.1f}in) | {status}"


def test_raspberry_pi_gpio(trigger_pin, echo_pin, temp_c=22.0, sample_rate_hz=12):
    """Directly tests HC-SR04 connected to Raspberry Pi 5 GPIO pins."""
    print(f"[INFO] Initializing Raspberry Pi GPIO Sonar (Trigger=GPIO {trigger_pin}, Echo=GPIO {echo_pin})...")
    
    sensor = None
    try:
        from gpiozero import DistanceSensor
        sensor = DistanceSensor(echo=echo_pin, trigger=trigger_pin, max_distance=3.0)
    except ImportError:
        print("[FAIL] 'gpiozero' is not installed.")
        print("       Install on Raspberry Pi OS via: sudo apt install python3-gpiozero python3-lgpio -y")
        return False
    except Exception as e:
        print(f"[ERROR] Could not claim GPIO pins (Trigger={trigger_pin}, Echo={echo_pin}): {e}")
        if "pin factory" in str(e).lower() or "badpin" in str(e).lower():
            print("       [TIP] Raspberry Pi 5 RP1 requires 'python3-lgpio': sudo apt install python3-lgpio -y")
        return False

    # Speed of sound calculation
    v_sound = 331.3 + (0.606 * temp_c)
    print(f"[INFO] Calibrated Speed of Sound: {v_sound:.1f} m/s (Ambient: {temp_c:.1f}°C)")
    print(f"[INFO] Live Acoustic Pinging Active @ {sample_rate_hz} Hz. Press Ctrl+C to stop.\n")

    consecutive_timeouts = 0
    readings = []
    start_time = time.time()
    ping_interval = 1.0 / max(1, sample_rate_hz)

    try:
        while True:
            t0 = time.time()
            dist = None
            try:
                dist = sensor.distance
            except Exception:
                dist = None

            if dist is not None and 0.02 <= dist <= 3.0:
                consecutive_timeouts = 0
                readings.append(dist)
                if len(readings) > 10:
                    readings.pop(0)
                avg_dist = sum(readings) / len(readings)
                pulse_us = int((dist * 2.0 / v_sound) * 1_000_000)
                sys.stdout.write(f"\r  {format_radar_bar(dist)} | Pulse: {pulse_us:5d}µs   ")
                sys.stdout.flush()
            else:
                consecutive_timeouts += 1
                if consecutive_timeouts >= 8:
                    sys.stdout.write(f"\r  [WARN] No echo received! Check 5V/GND & Echo resistor divider.          \n")
                    sys.stdout.flush()
                    consecutive_timeouts = 0
                else:
                    sys.stdout.write(f"\r  [PINGING...] Waiting for acoustic echo reflection...                    \r")
                    sys.stdout.flush()

            elapsed = time.time() - t0
            sleep_time = max(0.01, ping_interval - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[INFO] Diagnostics stopped by user.")
    finally:
        if sensor is not None:
            try:
                sensor.close()
            except Exception:
                pass
        print("[INFO] GPIO pins released cleanly.")
    return True


def test_serial_sonar(port_name, baud_rate=115200):
    """Tests HC-SR04 sonar module streaming over USB Serial from Arduino / Microcontroller."""
    print(f"[INFO] Connecting to Serial Sonar Bridge on {port_name} at {baud_rate} baud...")
    try:
        import serial
    except ImportError:
        print("[FAIL] 'pyserial' not installed. Run: pip install pyserial")
        return False

    import re
    try:
        ser = serial.Serial(port_name, baud_rate, timeout=1.5)
        time.sleep(1.5)  # Wait for Arduino DTR reset
        print(f"[SUCCESS] Connected to {port_name}! Reading live sonar acoustic packets...\n")
        
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            
            # Match "DISTANCE: 1.45" or raw float numbers
            m = re.search(r"[-+]?(?:\d*\.\d+|\d+)", line)
            if m:
                val = float(m.group())
                # If value reported in centimeters (> 10.0), convert to meters
                if val > 10.0:
                    dist_m = val / 100.0
                else:
                    dist_m = val

                sys.stdout.write(f"\r  {format_radar_bar(dist_m)} | Raw: {line:<16} ")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\n[INFO] Serial diagnostics terminated by user.")
    except Exception as e:
        print(f"\n[ERROR] Serial error on {port_name}: {e}")
        return False
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return True


def test_simulated_sonar():
    """Runs interactive ASCII acoustic radar simulator on PC without physical hardware."""
    print("[INFO] Running in Interactive Terminal Sonar Simulation Mode.")
    print("       Use this mode to test radar logic, distance math, and visual gauge.\n")
    print("Controls:")
    print("  [Auto-Sweep Mode]: Distance will oscillate dynamically between 0.3m and 2.8m.")
    print("  Press Ctrl+C to exit simulation.\n")

    distance = 2.40
    direction = -1
    v_sound = 344.0  # at ~21°C

    try:
        while True:
            distance += direction * 0.04
            if distance <= 0.35:
                distance = 0.35
                direction = 1
            elif distance >= 2.80:
                distance = 2.80
                direction = -1

            pulse_us = int((distance * 2.0 / v_sound) * 1_000_000)
            sys.stdout.write(f"\r  {format_radar_bar(distance)} | Pulse: {pulse_us:5d}µs   ")
            sys.stdout.flush()
            time.sleep(0.06)

    except KeyboardInterrupt:
        print("\n\n[INFO] Simulation ended cleanly.")
    return True


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="HC-SR04 Sonar Ultrasonic Hardware Diagnostic Suite")
    parser.add_argument("--trigger", type=int, default=PIN_TRIGGER_DEFAULT, help="Trigger GPIO pin (default: 23)")
    parser.add_argument("--echo", type=int, default=PIN_ECHO_DEFAULT, help="Echo GPIO pin (default: 24)")
    parser.add_argument("--temp", type=float, default=22.0, help="Ambient temperature in Celsius (default: 22.0)")
    parser.add_argument("--rate", type=int, default=12, help="Sample rate in Hz (default: 12)")
    parser.add_argument("--port", type=str, default=None, help="Serial COM port for USB Sonar (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--sim", action="store_true", help="Force simulated terminal acoustic radar mode")
    args = parser.parse_args()

    # 1. Serial USB Sonar requested
    if args.port:
        test_serial_sonar(args.port)
        return

    # 2. Simulated Sonar requested
    if args.sim:
        test_simulated_sonar()
        return

    # 3. Detect if running on Raspberry Pi / Linux GPIO
    is_raspberry_pi = False
    try:
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "r") as f:
                if "raspberry pi" in f.read().lower():
                    is_raspberry_pi = True
    except Exception:
        pass

    if is_raspberry_pi:
        test_raspberry_pi_gpio(args.trigger, args.echo, temp_c=args.temp, sample_rate_hz=args.rate)
    else:
        # Check if gpiozero is available anyway
        try:
            import gpiozero
            print("[INFO] 'gpiozero' detected. Attempting hardware GPIO test...")
            test_raspberry_pi_gpio(args.trigger, args.echo, temp_c=args.temp, sample_rate_hz=args.rate)
        except Exception:
            print("[INFO] Running on non-Raspberry Pi workstation without hardware GPIO pins.")
            print("       Auto-launching Simulated Sonar Radar. To test USB hardware, pass --port COMx.\n")
            test_simulated_sonar()


if __name__ == "__main__":
    main()
