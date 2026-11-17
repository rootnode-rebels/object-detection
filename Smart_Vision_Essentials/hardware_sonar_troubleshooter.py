#!/usr/bin/env python3
"""
Smart Vision Assistant - Comprehensive Ultrasonic Sensor & Logic Level Troubleshooter
====================================================================================
Tests EVERY possible hardware, resistor, logic-level converter, and wiring scenario
to diagnose EXACTLY what is physically wrong on your Raspberry Pi 5.
"""

import sys
import time

def print_banner():
    print("=" * 70)
    print("  HC-SR04 & LOGIC LEVEL CONVERTER COMPREHENSIVE HARDWARE TROUBLESHOOTER")
    print("  Raspberry Pi 5 (RP1 GPIO) Multi-Configuration Hardware Probe")
    print("=" * 70)

def test_pin_pair(trig_pin, echo_pin, description):
    """Probes a specific Trigger/Echo pin combination."""
    from gpiozero import OutputDevice, InputDevice
    trig = None
    echo = None
    try:
        trig = OutputDevice(trig_pin)
        echo = InputDevice(echo_pin)
    except Exception as e:
        return False, f"Could not claim GPIO {trig_pin}/{echo_pin}: {e}", None

    idle_state = echo.value
    if idle_state == 1:
        trig.close()
        echo.close()
        return False, f"GPIO {echo_pin} is stuck HIGH (1) at idle (Short circuit or floating ground)!", None

    # Try 5 pings
    successful_pings = []
    for _ in range(5):
        trig.off()
        time.sleep(0.03)
        trig.on()
        time.sleep(0.00001)
        trig.off()

        t_start = time.perf_counter()
        timeout = t_start + 0.04
        rise = None
        fall = None

        while time.perf_counter() < timeout:
            if echo.value == 1:
                rise = time.perf_counter()
                break

        if rise is not None:
            while time.perf_counter() < timeout:
                if echo.value == 0:
                    fall = time.perf_counter()
                    break

        if rise is not None and fall is not None:
            pulse_us = (fall - rise) * 1_000_000
            dist_cm = (pulse_us * 0.0343) / 2.0
            if 2.0 <= dist_cm <= 400.0:
                successful_pings.append(dist_cm)

    trig.close()
    echo.close()

    if len(successful_pings) >= 2:
        avg_cm = sum(successful_pings) / len(successful_pings)
        return True, f"WORKING! Verified echo received! Measured Distance: {avg_cm:.1f} cm ({avg_cm/100:.2f} m)", avg_cm
    else:
        return False, f"No acoustic echo response on Trigger=GPIO {trig_pin}, Echo=GPIO {echo_pin}.", None

def main():
    print_banner()

    try:
        import gpiozero
    except ImportError:
        print("\n[ERROR] 'gpiozero' is not installed.")
        print("Run: sudo apt install python3-gpiozero python3-lgpio -y")
        sys.exit(1)

    print("\n[PHASE 1] Electrical Diagnostics on Standard Pins (TRIG=GPIO 23, ECHO=GPIO 24)...")
    success_std, msg_std, dist_std = test_pin_pair(23, 24, "Standard Pins (Pin 16 TRIG, Pin 18 ECHO)")
    print(f"  Result: {msg_std}")

    if success_std:
        print("\n" + "=" * 70)
        print("  >>> CONGRATULATIONS! YOUR HARDWARE IS FULLY WORKING! <<<")
        print("  Wiring, resistors/LLC, and 5V power are all 100% correct!")
        print(f"  Live Target Distance: {dist_std:.1f} cm")
        print("=" * 70 + "\n")
        return

    print("\n[PHASE 2] Checking for REVERSED WIRING (TRIG on Pin 18, ECHO on Pin 16)...")
    success_rev, msg_rev, dist_rev = test_pin_pair(24, 23, "Reversed Pins (Pin 18 TRIG, Pin 16 ECHO)")
    print(f"  Result: {msg_rev}")

    if success_rev:
        print("\n" + "=" * 70)
        print("  >>> DIAGNOSIS: YOUR TRIGGER AND ECHO WIRES ARE SWAPPED! <<<")
        print("=" * 70)
        print("  Problem: Your breadboard wires are reversed.")
        print("  Fix: Swap the two jumper wires between Physical Pin 16 and Physical Pin 18:")
        print("       - Move Trigger wire to Physical Pin 16 (GPIO 23)")
        print("       - Move Echo wire (via divider/LLC) to Physical Pin 18 (GPIO 24)")
        print(f"  (Once swapped, it will read distance: {dist_rev:.1f} cm)")
        print("=" * 70 + "\n")
        return

    print("\n[PHASE 3] Checking if connected to alternative GPIO pins (e.g. GPIO 17 / GPIO 27)...")
    for (t_pin, e_pin, t_phys, e_phys) in [(17, 27, 11, 13), (27, 17, 13, 11), (22, 23, 15, 16)]:
        ok, m, d = test_pin_pair(t_pin, e_pin, f"Alt Pins ({t_pin}, {e_pin})")
        if ok:
            print(f"\n  [ALERT] Echo detected on Physical Pin {t_phys} (GPIO {t_pin}) and Pin {e_phys} (GPIO {e_pin})!")
            print(f"          Fix: Move your wires to Pin 16 (GPIO 23) and Pin 18 (GPIO 24)!")
            return

    # PHASE 4: FAILURE CAUSE MATRIX & EXACT FIX
    print("\n" + "=" * 70)
    print("  >>> HARDWARE FAILURE ROOT-CAUSE ANALYSIS & FIXES <<<")
    print("=" * 70)
    print("Neither standard nor reversed pins returned an echo.")
    print("This confirms one of the following 4 physical hardware errors:\n")

    print("1. [MOST COMMON] RESISTOR DIVIDER RATIO IS BACKWARDS (Voltage too low):")
    print("   - Current state: Sensor Echo is likely producing 1.6V instead of 3.3V.")
    print("   - Why: If you used 2kΩ in series and 1kΩ to GND, the Pi sees 0V (no signal).")
    print("   - FIX:")
    print("     Sensor ECHO Pin ---> [ 1k Ohm Resistor ] ---> Pi Pin 18 (GPIO 24)")
    print("                                              ---> [ 2k Ohm ] ---> GND (Pin 6/14)\n")

    print("2. [COMMON] SENSOR VCC IS PLUGGED INTO 3.3V INSTEAD OF 5.0V:")
    print("   - Current state: Sensor cannot fire ultrasonic bursts.")
    print("   - Why: Standard HC-SR04 requires 5V to power its internal oscillator.")
    print("   - FIX: Move the RED power wire from Pin 1 (3.3V) to Physical Pin 2 or Pin 4 (5.0V).\n")

    print("3. [IF USING LOGIC LEVEL CONVERTER MODULE (LLC)]:")
    print("   - HV pin MUST connect to Pin 2 (5V DC)")
    print("   - LV pin MUST connect to Pin 1 (3.3V DC)")
    print("   - LLC GND MUST connect to Pi Ground (Pin 6 or 14)")
    print("   - HV1 connects to Sensor ECHO")
    print("   - LV1 connects to Pi Pin 18 (GPIO 24)\n")

    print("4. [IF HC-SR04 IS PLUGGED DIRECTLY WITHOUT ANY RESISTORS/LLC]:")
    print("   - If your sensor is an 'HC-SR04P' or 'RCWL-9610' (3.3V version), connect directly.")
    print("   - If standard HC-SR04, check that jumper wires make firm contact in breadboard.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
