"""
Custom YOLOv8 Training & ONNX Export Pipeline
Designed for Assistive Vision Device (Raspberry Pi 5)

Usage:
1. Organize your dataset into YOLO format (train/val images and labels).
2. Define your classes in data.yaml.
3. Run: python train_custom_yolo.py --data data.yaml --epochs 50 --imgsz 320
4. The script automatically exports the final model to 'model_files/yolov8_custom.onnx'
   which drops directly into smart_assistant.py!
"""

import os
import sys
import argparse

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


def train_and_export(data_yaml="data.yaml", epochs=50, imgsz=320, batch=16):
    if not ULTRALYTICS_AVAILABLE:
        print("\n[ERROR] 'ultralytics' package is required for training.")
        print("Install it with: pip install ultralytics\n")
        sys.exit(1)

    if not os.path.exists(data_yaml):
        print(f"\n[ERROR] Dataset configuration file '{data_yaml}' not found.")
        print("Please prepare your dataset and create a data.yaml file.")
        print("Example template:\n")
        print("""path: ./dataset
train: images/train
val: images/val
names:
  0: stairs
  1: door
  2: crosswalk
  3: pothole
""")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("      CUSTOM YOLOV8 MODEL TRAINING PIPELINE")
    print(f"      Data: {data_yaml} | Epochs: {epochs} | Resolution: {imgsz}x{imgsz}")
    print("=" * 65 + "\n")

    # 1. Load pre-trained nano model for fast transfer learning
    print("[INFO] Loading pre-trained YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    # 2. Train on custom dataset
    print(f"[INFO] Starting training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",  # or 0 if training on PC GPU
        plots=True
    )

    # 3. Export to CPU-optimized ONNX format for Raspberry Pi
    output_dir = "model_files"
    os.makedirs(output_dir, exist_ok=True)
    custom_onnx_target = os.path.join(output_dir, "yolov8_custom.onnx")

    print(f"\n[INFO] Exporting best model to CPU-optimized ONNX ({imgsz}x{imgsz})...")
    exported_path = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)

    # Move/copy to model_files directory
    if os.path.exists(exported_path):
        import shutil
        shutil.copy(exported_path, custom_onnx_target)
        print(f"\n[SUCCESS] Model successfully trained and exported to: {custom_onnx_target}")
        print("\nTo run your smart assistant with this custom model, simply run:")
        print(f"python smart_assistant.py --model {custom_onnx_target} --resolution {imgsz}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 model and export to ONNX")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml dataset file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size (default: 320 for high Pi 5 FPS)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    train_and_export(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch
    )
