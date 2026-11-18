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


def test_raspberry_pi_gpio(trigger_pin, echo_pin, temp_c=25.0, sample_rate_hz=12, use_polling=False):
    """Directly tests HC-SR04 connected to Raspberry Pi 5 GPIO pins using Pi5UltrasonicSonar."""
    print(f"[INFO] Initializing Raspberry Pi GPIO Sonar (Trigger=GPIO {trigger_pin}, Echo=GPIO {echo_pin})...")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sensor = None
    try:
        from sonar_driver import Pi5UltrasonicSonar
        sensor = Pi5UltrasonicSonar(
            trigger_pin=trigger_pin,
            echo_pin=echo_pin,
            max_distance=3.0,
            temp_c=temp_c,
            window_size=5,
            ping_interval=max(0.07, 1.0 / max(1, sample_rate_hz)),
            use_polling=use_polling,
        )
        print(f"[INFO] Sonar Backend : {sensor.backend} | Chip: {sensor.chip_info}")
        echo_idle = sensor.echo_pin_state
        if echo_idle is not None:
            idle_label = "LOW (Healthy)" if echo_idle == 0 else "HIGH (WARNING: Check resistor divider / 5V wiring!)"
            print(f"[INFO] ECHO Idle Pin : {idle_label}")
    except Exception as e:
        print(f"[ERROR] Could not initialize sonar on Trigger=GPIO {trigger_pin}, Echo=GPIO {echo_pin}: {e}")
        print("       [TIP] Raspberry Pi 5 RP1 requires 'python3-lgpio': sudo apt install python3-lgpio -y")
        return False

    # Speed of sound calculation
    v_sound = 331.3 + (0.606 * temp_c)
    print(f"[INFO] Calibrated Speed of Sound: {v_sound:.1f} m/s (Ambient: {temp_c:.1f}°C)")
    print(f"[INFO] Live Acoustic Pinging Active (70ms settling cycle). Press Ctrl+C to stop.\n")

    consecutive_timeouts = 0
    total_timeouts = 0
    checked_swap = False
    ping_interval = max(0.07, 1.0 / max(1, sample_rate_hz))

    try:
        while True:
            t0 = time.perf_counter()
            dist = sensor.read_filtered()
            raw_pulse_us = sensor.last_pulse_us
            status_code = sensor.last_status

            if dist is not None and 0.02 <= dist <= 3.0:
                consecutive_timeouts = 0
                total_timeouts = 0
                pulse_disp = int(round(raw_pulse_us)) if raw_pulse_us is not None else int((dist * 2.0 / v_sound) * 1_000_000)
                raw_disp = f"{sensor.last_raw_meters*100:.1f}cm" if sensor.last_raw_meters is not None else "---"
                sys.stdout.write(f"\r  {format_radar_bar(dist)} | Raw:{raw_disp:>6s} | Pulse:{pulse_disp:5d}µs   ")
                sys.stdout.flush()
            else:
                consecutive_timeouts += 1
                total_timeouts += 1

                # If no echo received after 12 attempts (~1 second), check if TRIG & ECHO are swapped
                if total_timeouts >= 12 and not checked_swap:
                    checked_swap = True
                    sys.stdout.write(f"\r  [TESTING] Checking if Trigger & Echo wires are reversed on breadboard...\n")
                    sys.stdout.flush()
                    try:
                        sensor.close()
                        swap_sensor = Pi5UltrasonicSonar(
                            trigger_pin=echo_pin,
                            echo_pin=trigger_pin,
                            max_distance=3.0,
                            temp_c=temp_c,
                            use_polling=use_polling,
                        )
                        sw_d, _ = swap_sensor.ping_once()
                        swap_sensor.close()
                        if sw_d is not None and 0.02 <= sw_d <= 3.0:
                            print(f"\n  [ALERT] REVERSED WIRING DETECTED! Echo was received on GPIO {trigger_pin} ({sw_d*100:.1f} cm)!")
                            print(f"          Fix: Swap physical Pin 16 (GPIO 23) and Pin 18 (GPIO 24) on your breadboard!\n")
                        # Re-open original pins
                        sensor = Pi5UltrasonicSonar(
                            trigger_pin=trigger_pin,
                            echo_pin=echo_pin,
                            max_distance=3.0,
                            temp_c=temp_c,
                            window_size=5,
                            ping_interval=ping_interval,
                            use_polling=use_polling,
                        )
                    except Exception:
                        pass

                if consecutive_timeouts >= 8:
                    sys.stdout.write(f"\r  [WARN] Status={status_code} | No valid echo! Check 5V VCC, GND & 1k/2k divider.      \n")
                    sys.stdout.flush()
                    consecutive_timeouts = 0
                else:
                    sys.stdout.write(f"\r  [PINGING...] Waiting for echo (Status: {status_code})...                          \r")
                    sys.stdout.flush()

            elapsed = time.perf_counter() - t0
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
    parser.add_argument("--temp", type=float, default=25.0, help="Ambient temperature in Celsius (default: 25.0)")
    parser.add_argument("--rate", type=int, default=12, help="Sample rate in Hz (default: 12)")
    parser.add_argument("--poll", action="store_true", help="Use direct lgpio polling mode instead of kernel IRQ timestamps")
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
        test_raspberry_pi_gpio(args.trigger, args.echo, temp_c=args.temp, sample_rate_hz=args.rate, use_polling=args.poll)
    else:
        # Check if lgpio or gpiozero is available anyway
        try:
            import lgpio
            print("[INFO] 'lgpio' detected. Attempting hardware GPIO test...")
            test_raspberry_pi_gpio(args.trigger, args.echo, temp_c=args.temp, sample_rate_hz=args.rate, use_polling=args.poll)
        except Exception:
            print("[INFO] Running on non-Raspberry Pi workstation without hardware GPIO pins.")
            print("       Auto-launching Simulated Sonar Radar. To test USB hardware, pass --port COMx.\n")
            test_simulated_sonar()


if __name__ == "__main__":
    main()
