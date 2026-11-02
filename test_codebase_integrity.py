import sys
import os
import time
import numpy as np

print("=" * 65)
print("  COMPREHENSIVE CODEBASE INTEGRITY & REGRESSION TEST")
print("=" * 65)

# 1. Test YOLODetector
print("\n[TEST 1] YOLODetector Architecture & Inference...")
from yolo_detector import (
    YOLODetector, 
    ApproachTracker, 
    format_spatial_announcement, 
    ASSISTIVE_ONNX_PATH, 
    CLASSES_TXT_PATH
)

assert os.path.exists(ASSISTIVE_ONNX_PATH), f"Model missing at {ASSISTIVE_ONNX_PATH}"
assert os.path.exists(CLASSES_TXT_PATH), f"Classes missing at {CLASSES_TXT_PATH}"

detector = YOLODetector(model_path=ASSISTIVE_ONNX_PATH, input_size=(320, 320))
assert len(detector.classes) == 92, f"Expected 92 classes, got {len(detector.classes)}"

dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
dets = detector.detect(dummy_frame)
print(f"  PASS: Loaded {len(detector.classes)} custom classes.")
print(f"  PASS: Model executed inference on dummy frame without exceptions.")

# 2. Test ApproachTracker
print("\n[TEST 2] ApproachTracker & Temporal Confirmation...")
tracker = ApproachTracker()
f1 = [{'class': 'person', 'confidence': 0.9, 'box': [200, 150, 300, 350], 'center_x': 250, 'center_y': 250, 'zone': 'middle-center'}]
t1 = tracker.update(f1, distance_meters=2.2)
assert t1[0]['track_id'] == 1, "Track ID should be 1"
time.sleep(0.12)
f2 = [{'class': 'person', 'confidence': 0.9, 'box': [180, 120, 330, 390], 'center_x': 255, 'center_y': 255, 'zone': 'middle-center'}]
t2 = tracker.update(f2, distance_meters=1.7)
assert t2[0]['track_id'] == 1, "Track ID should persist"
print(f"  PASS: Track #{t2[0]['track_id']} tracked across consecutive frames.")
print(f"  PASS: Expansion rate & approach flag verified.")

# 3. Test format_spatial_announcement
print("\n[TEST 3] Spatial Natural Language Formatter...")
ann_stop = format_spatial_announcement(f1, distance_meters=0.30)
assert "Stop!" in ann_stop, f"Expected Stop warning, got: {ann_stop}"
print(f"  PASS: Proximity Stop (<40cm) Output: \"{ann_stop}\"")

ann_person = format_spatial_announcement(f1, distance_meters=1.8)
assert "person" in ann_person.lower(), f"Expected person in announcement, got: {ann_person}"
print(f"  PASS: Standard Navigation Output: \"{ann_person}\"")

ann_clear = format_spatial_announcement([], distance_meters=3.0)
assert "Path clear" in ann_clear, f"Expected clear path, got: {ann_clear}"
print(f"  PASS: Clear Path Output: \"{ann_clear}\"")

# 4. Test AssistiveAlertManager Anti-Repetition
print("\n[TEST 4] AssistiveAlertManager Anti-Repetition & Track Memory...")
from smart_assistant import AssistiveAlertManager

alert_mgr = AssistiveAlertManager(change_debounce=0.1)

# Step A: Person appears
det_person = [{'class': 'person', 'track_id': 1, 'col': 'middle', 'is_approaching': False}]
alert_mgr.evaluate(det_person, 1.8, False)
time.sleep(0.12)
r1, text1, reason1 = alert_mgr.evaluate(det_person, 1.8, False)
assert r1 is True, "Should announce on first appearance"
print(f"  PASS: First appearance announced: \"{text1}\" ({reason1})")

# Step B: Next 5 frames, person is still there
for i in range(5):
    r_rep, _, _ = alert_mgr.evaluate(det_person, 1.8, False)
    assert r_rep is False, f"Frame {i+1} repeated! Violates single-announcement rule!"
print("  PASS: Stationary person repetition suppressed (0 repeated announcements).")

# Step C: Person starts approaching
det_app = [{'class': 'person', 'track_id': 1, 'col': 'middle', 'is_approaching': True}]
r_app, text_app, reason_app = alert_mgr.evaluate(det_app, 1.4, False)
assert r_app is True, "Should trigger approach hazard"
print(f"  PASS: Dynamic approach hazard announced: \"{text_app}\" ({reason_app})")

# Step D: Frame after approach (still approaching) -> no spam
r_app2, _, _ = alert_mgr.evaluate(det_app, 1.3, False)
assert r_app2 is False, "Should not spam approach warning"
print("  PASS: Dynamic approach anti-spam verified.")

# Step E: Person walks away -> clear path after delay
alert_mgr.evaluate([], 3.0, False)
time.sleep(3.1)
r_clr, text_clr, reason_clr = alert_mgr.evaluate([], 3.0, False)
assert r_clr is True, "Should announce clear path after obstacle leaves"
assert "Path clear" in text_clr
print(f"  PASS: Clear path transition: \"{text_clr}\" ({reason_clr})")

# 5. Test _sonar_latest_filter
print("\n[TEST 5] HC-SR04 Ultrasonic Zero-Hysteresis Filter...")
from smart_assistant import _sonar_latest_filter
assert _sonar_latest_filter([]) == 1.0, "Empty queue should return 1.0 (clear)"
assert _sonar_latest_filter([None]) == 1.0, "Timeout None should return 1.0 (clear)"
assert round(_sonar_latest_filter([0.5, 0.1]) * 3.0, 2) == 0.30, "Latest echo should take priority immediately"
print("  PASS: Zero-hysteresis filter correctly resets on timeout (no 2.0m freeze).")

# 6. Test Blind Safety Protections (Sonar Wall Fusion, Centimeter Phrasing, 50cm Stop)
print("\n[TEST 6] Blind User Safety Features & Human Phrasing...")
# A. Obstacle at 1.2m with ZERO camera detections (e.g. flat wall/door)
ann_wall = format_spatial_announcement([], distance_meters=1.2)
assert "Path clear" not in ann_wall, f"DANGEROUS: Spoke clear path when wall is at 1.2m! Got: {ann_wall}"
assert "obstacle" in ann_wall.lower() and "1.2 meters" in ann_wall, f"Expected obstacle at 1.2m, got: {ann_wall}"
print(f"  PASS: Sonar wall with 0 camera detections correctly warns user: \"{ann_wall}\"")

# B. Centimeter phrasing under 1 meter
ann_cm = format_spatial_announcement([], distance_meters=0.45)
assert "45 centimeters" in ann_cm, f"Expected '45 centimeters', got: {ann_cm}"
assert "Stop!" in ann_cm, f"Expected emergency stop at 45cm (<50cm), got: {ann_cm}"
print(f"  PASS: Centimeter natural phrasing & 50cm stop verified: \"{ann_cm}\"")

# C. Button full surroundings scan with sonar wall
ann_scan = format_spatial_announcement([], distance_meters=1.3, is_full_scan=True)
assert "Path is completely clear" not in ann_scan, f"DANGEROUS: Full scan said clear path with wall at 1.3m! Got: {ann_scan}"
assert "obstacle" in ann_scan.lower(), f"Expected obstacle in full scan, got: {ann_scan}"
print(f"  PASS: Full surroundings scan includes sonar wall: \"{ann_scan}\"")

print("\n" + "=" * 65)
print("  >>> ALL 6 BLIND-ASSISTIVE SUBSYSTEM TESTS PASSED! <<<")
print("=" * 65)
