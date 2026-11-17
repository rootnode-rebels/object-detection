#!/usr/bin/env python3
"""
Raw Hardware Pin Diagnostic for HC-SR04 on Raspberry Pi 5
Bypasses all smoothing queues and directly probes electrical signals on GPIO 23 and 24.
"""
import time
import sys

print("=" * 65)
print("   RAW ELECTRICAL PIN DIAGNOSTIC - HC-SR04 SONAR")
print("=" * 65)

try:
    from gpiozero import OutputDevice, InputDevice
except ImportError:
    print("[ERROR] gpiozero is not installed. Run: sudo apt install python3-gpiozero python3-lgpio -y")
    sys.exit(1)

TRIG_PIN = 23  # Physical Pin 16
ECHO_PIN = 24  # Physical Pin 18

try:
    trig = OutputDevice(TRIG_PIN)
    echo = InputDevice(ECHO_PIN)
except Exception as e:
    print(f"[ERROR] Could not claim GPIO pins {TRIG_PIN} / {ECHO_PIN}: {e}")
    print("If GPIO is busy, run: sudo killall -9 python3")
    sys.exit(1)

print(f"\n[STEP 1] Checking idle electrical state on GPIO {ECHO_PIN} (Physical Pin 18)...")
idle_val = echo.value
print(f"         Idle ECHO Pin Value = {idle_val}")
if idle_val == 1:
    print("  [ALERT] ECHO pin is stuck at HIGH (3.3V/5V) when idle!")
    print("          Likely Cause: Voltage divider shorted, bad breadboard rail, or wrong pin.")
else:
    print("  [PASS] Idle ECHO state is LOW (0V). Ready to listen.")

print(f"\n[STEP 2] Firing 10 acoustic pings and measuring microsecond pulses...")
success_count = 0

for i in range(1, 11):
    # Ensure trigger is low
    trig.off()
    time.sleep(0.04)

    # 10 microsecond trigger pulse
    trig.on()
    time.sleep(0.00001)
    trig.off()

    # Listen for ECHO rising edge (0 -> 1)
    t_start = time.perf_counter()
    timeout = t_start + 0.04  # 40ms timeout (~6.8 meters max)
    pulse_start = None
    pulse_end = None

    while time.perf_counter() < timeout:
        if echo.value == 1:
            pulse_start = time.perf_counter()
            break

    if pulse_start is None:
        print(f"  Ping #{i:2d}: FAILED -> ECHO pin never rose to HIGH (1).")
        continue

    # Listen for ECHO falling edge (1 -> 0)
    while time.perf_counter() < timeout:
        if echo.value == 0:
            pulse_end = time.perf_counter()
            break

    if pulse_end is None:
        print(f"  Ping #{i:2d}: FAILED -> ECHO stayed HIGH (timed out).")
        continue

    pulse_duration = pulse_end - pulse_start
    distance_cm = (pulse_duration * 34300) / 2.0
    distance_m = distance_cm / 100.0
    success_count += 1
    print(f"  Ping #{i:2d}: SUCCESS! Pulse: {pulse_duration*1_000_000:6.0f} µs | Distance: {distance_cm:5.1f} cm ({distance_m:.2f} m)")

print("\n" + "=" * 65)
print(f"  RESULTS: {success_count} / 10 Pings Successful.")
if success_count > 0:
    print("  >>> SENSOR IS PHYSICALLY FUNCTIONAL AND WORKING! <<<")
else:
    print("  >>> HARDWARE TROUBLESHOOTING GUIDE <<<")
    print("  1. If ECHO was stuck at 1:")
    print("     - Check for short circuit on breadboard.")
    print("  2. If ECHO never went to 1 (all pings failed):")
    print("     - TRIG and ECHO are reversed! Swap Pin 16 and Pin 18.")
    print("     - VCC is on 3.3V instead of 5V (Move RED wire to Pin 2).")
    print("     - 1k and 2k resistors are reversed (giving 1.6V instead of 3.3V).")
print("=" * 65)

trig.close()
echo.close()
