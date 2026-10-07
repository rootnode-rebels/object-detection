"""
Performance Benchmark Tool: Raspberry Pi 5 CPU YOLOv8
Measures:
1. Pure DNN Inference Latency (ms)
2. Spatial Post-Processing & Speech Formatting Latency (ms)
3. End-to-End Pipeline Latency & Sustained FPS
4. CPU Utilization (%)
5. SoC Core Temperature (°C)
"""

import time
import os
import sys
import argparse
import numpy as np
import cv2

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import psutil
except ImportError:
    psutil = None

from yolo_detector import YOLODetector, format_spatial_announcement


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


def run_benchmark(model_path=None, classes_path=None, num_frames=100, resolution=320):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(BASE_DIR, "model_files", "yolov8_assistive_92.onnx")
    default_classes = os.path.join(BASE_DIR, "classes.txt")

    # Auto-detect 92-class assistive model if present
    if model_path is None and os.path.exists(default_model):
        model_path = default_model

    if classes_path is None and model_path == default_model and os.path.exists(default_classes):
        classes_path = default_classes

    custom_classes = None
    if classes_path and os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            custom_classes = [line.strip() for line in f if line.strip()]

    print("\n" + "=" * 75)
    print("      EDGE AI HARDWARE BENCHMARK: RASPBERRY PI 5 CPU (NO NPU)")
    print(f"      Model       : {model_path or 'Default YOLOv8n ONNX'}")
    print(f"      Classes     : {len(custom_classes) if custom_classes else 80} categories")
    print(f"      Resolution  : {resolution} x {resolution}")
    print(f"      Test Frames : {num_frames} frames")
    print("=" * 75 + "\n")

    detector = YOLODetector(
        model_path=model_path,
        classes=custom_classes,
        input_size=(resolution, resolution)
    )

    print(f"[BENCHMARK] Warming up CPU and running {num_frames} frames benchmark...\n")

    # Generate synthetic camera frames (640x480) with realistic simulated scene
    dummy_frame = np.random.randint(40, 220, (480, 640, 3), dtype=np.uint8)

    # Prime / Warm up
    for _ in range(5):
        dets = detector.detect(dummy_frame)
        _ = format_spatial_announcement(dets, distance_meters=1.5)

    cpu_samples = []
    temp_samples = []
    infer_latencies = []
    spatial_latencies = []
    total_latencies = []

    start_time = time.time()

    for i in range(num_frames):
        t0 = time.perf_counter()
        detections = detector.detect(dummy_frame)
        t1 = time.perf_counter()

        _ = format_spatial_announcement(detections, distance_meters=1.2)
        t2 = time.perf_counter()

        infer_latency = t1 - t0
        spatial_latency = t2 - t1
        total_latency = t2 - t0

        infer_latencies.append(infer_latency)
        spatial_latencies.append(spatial_latency)
        total_latencies.append(total_latency)

        if psutil is not None and i % 10 == 0:
            cpu_samples.append(psutil.cpu_percent(interval=None))
            temp_samples.append(get_soc_temperature())

        if (i + 1) % 25 == 0 or i == num_frames - 1:
            fps_so_far = (i + 1) / (time.time() - start_time)
            print(f"  Frame {i+1:3d}/{num_frames}: Speed = {fps_so_far:5.1f} FPS | "
                  f"Infer = {infer_latency * 1000:5.1f} ms | "
                  f"Spatial HUD = {spatial_latency * 1000:4.2f} ms")

    total_time = time.time() - start_time
    avg_fps = num_frames / total_time
    avg_infer_ms = (sum(infer_latencies) / len(infer_latencies)) * 1000
    avg_spatial_ms = (sum(spatial_latencies) / len(spatial_latencies)) * 1000
    avg_total_ms = (sum(total_latencies) / len(total_latencies)) * 1000
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    avg_temp = sum(temp_samples) / len(temp_samples) if temp_samples else get_soc_temperature()

    print("\n" + "=" * 75)
    print("                         BENCHMARK RESULTS")
    print("=" * 75)
    print(f" * Architecture Mode             : Raspberry Pi 5 CPU (ARM NEON OpenCV DNN)")
    print(f" * Model Evaluated               : {os.path.basename(model_path) if model_path else 'yolov8n.onnx'}")
    print(f" * Active Vocabulary             : {len(detector.classes)} Classes")
    print(f" * Input Resolution              : {resolution} x {resolution}")
    print(f" * Sustained Pipeline Frame Rate : {avg_fps:.1f} FPS")
    print(f" * Pure DNN Inference Latency    : {avg_infer_ms:.2f} ms")
    print(f" * Spatial Sector Formatting     : {avg_spatial_ms:.2f} ms")
    print(f" * Total Per-Frame Latency       : {avg_total_ms:.2f} ms")
    if avg_cpu > 0:
        print(f" * Average CPU Utilization       : {avg_cpu:.1f} %")
    if avg_temp > 0:
        print(f" * SoC Core Temperature          : {avg_temp:.1f} °C")
    print("=" * 75 + "\n")

    detector.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 CPU YOLOv8 Performance Benchmark")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--classes", type=str, default=None, help="Path to classes.txt file")
    parser.add_argument("--resolution", type=int, default=320, choices=[320, 640], help="Input resolution (320 or 640)")
    parser.add_argument("--frames", type=int, default=100, help="Number of benchmark frames (default: 100)")
    
    # Support positional args for backwards compatibility (e.g. "python benchmark_performance.py 320 50")
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        res = int(sys.argv[1])
        frames = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
        run_benchmark(num_frames=frames, resolution=res)
    else:
        args = parser.parse_args()
        run_benchmark(
            model_path=args.model,
            classes_path=args.classes,
            num_frames=args.frames,
            resolution=args.resolution
        )
