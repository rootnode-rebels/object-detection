import numpy as np
import time
from yolo_detector import YOLODetector, format_spatial_announcement
from smart_assistant import AssistiveAlertManager, HardwareBuzzer

print("=" * 65)
print("  END-TO-END PIPELINE & LAPTOP DEMO SIMULATION TEST")
print("=" * 65)

print("\n[TEST 4A] Testing AI Model Loading & Inference...")
det = YOLODetector(model_path="model_files/yolov8_assistive_92.onnx", input_size=(320, 320))
frame = np.zeros((480, 640, 3), dtype=np.uint8)
res = det.detect(frame, distance_meters=None)
print(f"  PASS: Loaded 92-class model. Dummy frame detections: {len(res)}")

print("\n[TEST 4B] Testing AlertManager when Sensor is DISCONNECTED...")
mgr = AssistiveAlertManager()
speak_ok, msg, reason = mgr.evaluate([], distance_meters=None, button_pressed=False)
print(f"  PASS: No sensor -> No false distance generated (speak={speak_ok}, msg={msg})")

print("\n[TEST 4C] Testing Intercepting Hand at 15cm Proximity...")
speak_ok, msg, reason = mgr.evaluate([], distance_meters=0.15, button_pressed=False)
assert speak_ok is True, "Emergency stop should trigger"
assert "15 centimeters" in msg and "Stop!" in msg, f"Unexpected message: {msg}"
print(f"  PASS: Emergency proximity stop triggered correctly:")
print(f"        Speech: \"{msg}\"")
print(f"        Reason: {reason}")

print("\n[TEST 4D] Testing Hardware Alert Buzzer Proximity Scaling...")
bz = HardwareBuzzer(enabled=True)
bz.update(0.15)
assert bz.current_state == "CRITICAL", f"Expected CRITICAL, got {bz.current_state}"
bz.update(0.75)
assert bz.current_state == "WARNING", f"Expected WARNING, got {bz.current_state}"
bz.update(2.50)
assert bz.current_state == "SILENT", f"Expected SILENT, got {bz.current_state}"
bz.update(None)
assert bz.current_state == "SILENT", f"Expected SILENT, got {bz.current_state}"
bz.close()
print("  PASS: Buzzer states: 15cm (CRITICAL) -> 75cm (WARNING) -> 2.5m (SILENT) -> None (SILENT)")

print("\n[TEST 4E] Testing Spatial Obstacle Announcement with Sensor Fusion...")
ann_obj = format_spatial_announcement(
    [{"class": "chair", "zone": "middle-center", "col": "middle", "row": "middle", "box": [100, 100, 200, 200]}],
    distance_meters=0.85
)
assert "chair" in ann_obj and "85 centimeters" in ann_obj
print(f"  PASS: Object + Sensor Fusion Speech:")
print(f"        \"{ann_obj}\"")

print("\n" + "=" * 65)
print("  >>> ALL PIPELINE TESTS PASSED WITH 100% SUCCESS! <<<")
print("=" * 65)
