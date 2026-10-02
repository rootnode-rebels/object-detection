"""
Assistive Vision Model Builder for 92 Custom Objects
Supports two methods:
1. Instant Zero-Shot Generation via YOLO-World:
   Creates a specialized ONNX model for all 92 classes without needing thousands of labeled images!
2. Standard Supervised Training via YOLOv8 Transfer Learning:
   For fine-tuning on custom labeled image datasets.
"""

import os
import sys
import argparse

CLASSES_FILE = "classes.txt"
OUTPUT_DIR = "model_files"


def load_classes(file_path=CLASSES_FILE):
    if not os.path.exists(file_path):
        print(f"[ERROR] Classes file '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, "r") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def build_with_yoloworld(model_name="yolov8s-worldv2.pt", imgsz=320):
    """
    Leverages YOLO-World open-vocabulary model to bind all 92 custom classes
    and exports a standalone, lightweight ONNX model ready for Raspberry Pi CPU.
    """
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        print("\n[ERROR] 'ultralytics' is required. Run: pip install ultralytics\n")
        sys.exit(1)

    classes = load_classes()
    print("\n" + "=" * 70)
    print(f"  BUILDING ASSISTIVE VISION MODEL FOR {len(classes)} CLASSES")
    print(f"  Base Architecture: {model_name} (YOLO-World)")
    print(f"  Resolution: {imgsz}x{imgsz}")
    print("=" * 70 + "\n")

    print(f"[INFO] Initializing base model: {model_name}...")
    model = YOLOWorld(model_name)

    print(f"[INFO] Embedding {len(classes)} custom classes into model vocabulary...")
    model.set_classes(classes)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_onnx = os.path.join(OUTPUT_DIR, "yolov8_assistive_92.onnx")

    print(f"[INFO] Exporting offline CPU-optimized ONNX model...")
    exported_file = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)

    if os.path.exists(exported_file):
        import shutil
        if os.path.exists(target_onnx):
            os.remove(target_onnx)
        shutil.move(exported_file, target_onnx)
        print(f"\n[SUCCESS] Model successfully generated: {target_onnx}")
        print("\nTo run your smart assistant with all 92 custom classes:")
        print(f"python smart_assistant.py --model {target_onnx} --classes {CLASSES_FILE} --resolution {imgsz}\n")


def generate_data_yaml(output_yaml="assistive_data.yaml"):
    """Generates a data.yaml template for all 92 classes for manual training."""
    classes = load_classes()
    with open(output_yaml, "w") as f:
        f.write("# Assistive Vision 92-Class Dataset Configuration\n")
        f.write("path: ./dataset\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: '{c}'\n")
    print(f"[INFO] Created dataset template: {output_yaml} with {len(classes)} classes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or prepare model for 92 assistive vision classes")
    parser.add_argument("--mode", choices=["yoloworld", "create-yaml"], default="yoloworld",
                        help="Mode: 'yoloworld' for instant model export, 'create-yaml' to generate dataset yaml")
    parser.add_argument("--imgsz", type=int, default=320, help="Model input size (default: 320 for Raspberry Pi)")
    args = parser.parse_args()

    if args.mode == "yoloworld":
        build_with_yoloworld(imgsz=args.imgsz)
    elif args.mode == "create-yaml":
        generate_data_yaml()
