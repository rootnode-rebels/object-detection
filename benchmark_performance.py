"""
Performance Benchmark Tool: Raspberry Pi 5 CPU YOLOv8
Measures:
1. Frame Rate (FPS) under CPU execution
2. Per-frame latency (ms)
3. CPU Utilization (%)
4. SoC Core Temperature (°C)
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

from yolo_detector import YOLODetector


def get_soc_temperature():
    """Reads Raspberry Pi SoC temperature in Celsius."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return float(f.read().strip()) / 1000.0
    except Exception:
        pass
    try:
        import subprocess
        output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(output.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        return 0.0


def run_benchmark(num_frames=150, resolution=320):
    print("=" * 70)
    print("      EDGE AI HARDWARE BENCHMARK: RASPBERRY PI 5 CPU (NO NPU)")
    print(f"      Configuration: YOLOv8n ONNX @ {resolution}x{resolution}")
    print("=" * 70)

    detector = YOLODetector(input_size=(resolution, resolution))

    print(f"\n[BENCHMARK] Warming up and running {num_frames} frames benchmark...\n")

    # Generate synthetic camera frames (640x480)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Prime / Warm up
    for _ in range(5):
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

        if (i + 1) % 25 == 0 or i == num_frames - 1:
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
    print(f" • Architecture Mode       : Raspberry Pi 5 CPU (YOLOv8n ONNX)")
    print(f" • Input Resolution        : {resolution} x {resolution}")
    print(f" • Achieved Frame Rate     : {avg_fps:.1f} FPS")
    print(f" • Per-Frame Latency       : {avg_latency_ms:.2f} ms")
    if avg_cpu > 0:
        print(f" • Average CPU Utilization : {avg_cpu:.1f} %")
    if avg_temp > 0:
        print(f" • SoC Core Temperature    : {avg_temp:.1f} °C")
    print("=" * 70 + "\n")

    detector.close()


if __name__ == "__main__":
    res = int(sys.argv[1]) if len(sys.argv) > 1 else 320
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run_benchmark(num_frames=frames, resolution=res)
