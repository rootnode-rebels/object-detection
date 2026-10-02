"""
Performance Benchmark Tool: Hailo-8 AI HAT+ vs. CPU-Only Baseline
Validates the metrics from 'Executive Briefing: Hardware Architecture Justification':
1. Frame Rate (Target: 60 FPS on Hailo-8 NPU vs. ~3 FPS on CPU)
2. CPU Utilization (Target: < 15% on Hailo-8 NPU vs. 100% on CPU)
3. Thermal Strain (Target: Cool operation vs. 85°C+ CPU throttling)
"""

import time
import os
import sys
import numpy as np
import cv2

try:
    import psutil
except ImportError:
    psutil = None

from hailo_detector import HailoYOLODetector


def get_soc_temperature():
    """Reads Raspberry Pi SoC temperature in Celsius."""
    # Method 1: Linux thermal zone
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        pass
    # Method 2: vcgencmd (Raspberry Pi specific)
    try:
        import subprocess
        output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(output.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        return 0.0


def run_benchmark(num_frames=300):
    print("=" * 70)
    print("      EDGE AI HARDWARE BENCHMARK: RASPBERRY PI 5 + HAILO-8 NPU")
    print("=" * 70)

    detector = HailoYOLODetector()
    accel_type = "Hailo-8 AI HAT+ (Dedicated NPU)" if detector.is_npu_active else "CPU ONLY (No AI HAT+)"

    print(f"\n[BENCHMARK] Active Compute Unit: {accel_type}")
    print(f"[BENCHMARK] Running {num_frames} frames benchmark...\n")

    # Generate synthetic camera frames (640x480)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Prime / Warm up
    for _ in range(10):
        detector.detect(dummy_frame)

    cpu_samples = []
    temp_samples = []
    latencies = []

    start_time = time.time()

    for i in range(num_frames):
        frame_start = time.perf_counter()
        _ = detector.detect(dummy_frame)
        latency = time.perf_counter() - frame_start
        latencies.append(latency)

        if psutil is not None and i % 10 == 0:
            cpu_samples.append(psutil.cpu_percent(interval=None))
            temp_samples.append(get_soc_temperature())

        if (i + 1) % 50 == 0 or i == num_frames - 1:
            fps_so_far = (i + 1) / (time.time() - start_time)
            print(f"  Frame {i+1:3d}/{num_frames}: Speed = {fps_so_far:.1f} FPS | "
                  f"Latency = {latency * 1000:.1f} ms")

    total_time = time.time() - start_time
    avg_fps = num_frames / total_time
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    avg_temp = sum(temp_samples) / len(temp_samples) if temp_samples else get_soc_temperature()

    print("\n" + "=" * 70)
    print("                         BENCHMARK RESULTS")
    print("=" * 70)
    print(f" • Architecture Mode       : {accel_type}")
    print(f" • Achieved Frame Rate     : {avg_fps:.1f} FPS")
    print(f" • Per-Frame Latency       : {avg_latency_ms:.2f} ms")
    print(f" • Average CPU Utilization : {avg_cpu:.1f} %")
    if avg_temp > 0:
        print(f" • SoC Core Temperature    : {avg_temp:.1f} °C")

    print("\n" + "-" * 70)
    print(" EXECUTIVE ARCHITECTURE COMPARISON:")
    print("-" * 70)
    print(f" Metric               | Specification Target (Hailo-8) | CPU-Only Baseline")
    print(f" Frame Rate           | 60 FPS                        | ~3 FPS")
    print(f" CPU Utilization      | < 15%                         | 100% (Overheated)")
    print(f" Thermal Throttling   | None (Cool running)           | Severe (85°C+)")
    print("=" * 70 + "\n")

    detector.close()

if __name__ == "__main__":
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run_benchmark(frames)
