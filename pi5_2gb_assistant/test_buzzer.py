#!/usr/bin/env python3
"""
Hardware Alert Buzzer Test Utility
Designed for Smart Vision Assistant on Raspberry Pi 5 & PC.

Hardware Wiring on Raspberry Pi 5:
  - Buzzer Positive (+) [Long leg / Red wire]   --> Physical Pin 13 (GPIO 27)
  - Buzzer Negative (-) [Short leg / Black wire] --> Physical Pin 14 (Ground)
  (Pin 13 & Pin 14 are side-by-side on the Pi header for zero-cable mess!)
"""

import sys
import time

PIN_BUZZER_DEFAULT = 27

def print_banner():
    print("=" * 68)
    print("   HARDWARE ALERT BUZZER - TEST & DIAGNOSTIC UTILITY")
    print("   Smart Vision Assistant for Visually Impaired Safety")
    print("=" * 68)
    print("   Wiring on Raspberry Pi 5 40-Pin Header:")
    print("     - Buzzer (+) --> Physical Pin 13 (GPIO 27)")
    print("     - Buzzer (-) --> Physical Pin 14 (Ground / GND)")
    print("   Type: Active 3.3V or 5V Piezo Buzzer")
    print("=" * 68 + "\n")

def test_raspberry_pi(pin=PIN_BUZZER_DEFAULT):
    print(f"[INFO] Testing Hardware Buzzer on GPIO {pin} (Physical Pin 13)...")
    try:
        from gpiozero import Buzzer
        bz = Buzzer(pin)
        
        print("\n1. Testing Short Tactile Confirmation Chirp (50ms)...")
        bz.beep(on_time=0.05, off_time=0.05, n=1, background=False)
        time.sleep(0.5)

        print("2. Testing Caution Warning Pulses (0.50m - 1.00m Obstacle)...")
        bz.beep(on_time=0.15, off_time=0.25, n=3, background=False)
        time.sleep(0.5)

        print("3. Testing Critical Proximity Emergency Alarm (<0.50m Stop Hazard)...")
        bz.beep(on_time=0.08, off_time=0.08, n=6, background=False)
        time.sleep(0.5)

        bz.off()
        bz.close()
        print("\n[SUCCESS] Buzzer test completed successfully!")
        return True
    except ImportError:
        print("[WARN] 'gpiozero' is not installed (running on non-Pi OS).")
        return False
    except Exception as e:
        print(f"[ERROR] Could not control GPIO {pin}: {e}")
        return False

def test_pc_simulation():
    print("[INFO] Running in PC / Laptop Simulation Mode...")
    if sys.platform == "win32":
        try:
            import winsound
            print("\n1. Simulating Short Confirmation Chirp...")
            winsound.Beep(1800, 60)
            time.sleep(0.4)

            print("2. Simulating Caution Warning Pulses...")
            for _ in range(3):
                winsound.Beep(1200, 150)
                time.sleep(0.15)
            time.sleep(0.4)

            print("3. Simulating Critical Proximity Emergency Alarm (<50cm)...")
            for _ in range(6):
                winsound.Beep(2200, 80)
                time.sleep(0.08)

            print("\n[SUCCESS] PC Audio Buzzer simulation complete!")
            return True
        except Exception as e:
            print(f"[WARN] PC sound error: {e}")
            return False
    else:
        print("PC simulation buzzer requires Windows or physical Raspberry Pi GPIO.")
        return False

def main():
    print_banner()
    is_pi = False
    try:
        from gpiozero import Buzzer
        # Test creating a dummy pin to verify real hardware factory
        bz_test = Buzzer(PIN_BUZZER_DEFAULT)
        bz_test.close()
        is_pi = True
    except Exception:
        is_pi = False

    if is_pi:
        test_raspberry_pi()
    else:
        test_pc_simulation()

if __name__ == "__main__":
    main()
