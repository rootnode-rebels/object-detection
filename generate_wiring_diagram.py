import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig = plt.figure(figsize=(20, 13), dpi=220, facecolor='#0b1120')
ax = fig.add_subplot(111)
ax.set_facecolor('#0b1120')
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis('off')

# Header Title Banner
ax.text(10, 12.4, 'RASPBERRY PI 5 - HC-SR04 ULTRASONIC SENSOR WIRING & LOGIC SHIFTING', 
        ha='center', va='center', color='#38bdf8', fontsize=18, fontweight='bold')
ax.text(10, 11.95, 'Safe 5V-to-3.3V Level Shifting Schematics | RP1 GPIO Compatibility Guide', 
        ha='center', va='center', color='#94a3b8', fontsize=12)

# ==============================================================================
# LEFT PANEL: METHOD 1 - TWO-RESISTOR VOLTAGE DIVIDER (RECOMMENDED & MOST COMMON)
# ==============================================================================
p1 = patches.FancyBboxPatch((0.5, 4.4), 9.2, 7.1, boxstyle='round,pad=0.25', facecolor='#1e293b', edgecolor='#38bdf8', lw=2)
ax.add_patch(p1)
ax.text(5.1, 11.2, 'METHOD 1: Two-Resistor Voltage Divider (Most Popular)', ha='center', va='center', color='#38bdf8', fontsize=13, fontweight='bold')
ax.text(5.1, 10.85, 'Connects directly using standard breadboard resistors (1k Ohm and 2k Ohm)', ha='center', va='center', color='#94a3b8', fontsize=9.5)

# HC-SR04 Sensor Box (Left Panel)
s1 = patches.FancyBboxPatch((1.0, 7.2), 2.2, 3.2, boxstyle='round,pad=0.15', facecolor='#0284c7', edgecolor='#e2e8f0', lw=2)
ax.add_patch(s1)
ax.text(2.1, 10.0, 'HC-SR04', ha='center', va='center', color='#ffffff', fontsize=12, fontweight='bold')
ax.text(2.1, 9.6, 'Ultrasonic', ha='center', va='center', color='#e0f2fe', fontsize=9)

# Sensor Pins (Left Panel)
pins_s1 = [
    ('VCC (5V)', 9.0, '#ef4444'),
    ('TRIG', 8.4, '#f97316'),
    ('ECHO (5V)', 7.8, '#eab308'),
    ('GND', 7.2, '#64748b')
]
for name, y, col in pins_s1:
    ax.add_patch(patches.Circle((3.2, y), 0.12, facecolor=col, edgecolor='#ffffff', lw=1))
    ax.text(3.0, y, name, ha='right', va='center', color='#ffffff', fontsize=9, fontweight='bold')

# Raspberry Pi 5 Header Box (Left Panel)
rpi1 = patches.FancyBboxPatch((7.0, 7.0), 2.3, 3.5, boxstyle='round,pad=0.15', facecolor='#15803d', edgecolor='#e2e8f0', lw=2)
ax.add_patch(rpi1)
ax.text(8.15, 10.1, 'Raspberry Pi 5', ha='center', va='center', color='#ffffff', fontsize=11, fontweight='bold')
ax.text(8.15, 9.75, '40-Pin Header', ha='center', va='center', color='#dcfce7', fontsize=8.5)

pins_rpi1 = [
    ('Pin 2 (5V DC)', 9.0, '#ef4444'),
    ('Pin 16 (GPIO 23)', 8.4, '#f97316'),
    ('Pin 18 (GPIO 24)', 7.8, '#38bdf8'),
    ('Pin 14 (GND)', 7.2, '#64748b')
]
for name, y, col in pins_rpi1:
    ax.add_patch(patches.Circle((7.0, y), 0.12, facecolor=col, edgecolor='#ffffff', lw=1))
    ax.text(7.2, y, name, ha='left', va='center', color='#ffffff', fontsize=9, fontweight='bold')

# Wires (Left Panel)
# 1. VCC (5V) -> Pin 2
ax.annotate('', xy=(7.0, 9.0), xytext=(3.2, 9.0),
            arrowprops=dict(arrowstyle='->', color='#ef4444', lw=2.5))
ax.text(5.1, 9.15, '5.0V Power Line (RED wire)', ha='center', va='bottom', color='#fca5a5', fontsize=8)

# 2. TRIG -> Pin 16
ax.annotate('', xy=(7.0, 8.4), xytext=(3.2, 8.4),
            arrowprops=dict(arrowstyle='->', color='#f97316', lw=2.5))
ax.text(5.1, 8.55, 'Direct 3.3V Trigger Pulse (ORANGE wire)', ha='center', va='bottom', color='#fdba74', fontsize=8)

# 3. ECHO -> Resistor Divider -> Pin 18
# Resistor R1 (1k)
ax.plot([3.2, 4.2], [7.8, 7.8], color='#eab308', lw=2.5)
r1_box = patches.Rectangle((4.2, 7.62), 0.8, 0.36, facecolor='#ca8a04', edgecolor='#fef08a', lw=1.5)
ax.add_patch(r1_box)
ax.text(4.6, 7.8, '1k Ohm', ha='center', va='center', color='#ffffff', fontsize=8.5, fontweight='bold')
ax.plot([5.0, 5.5], [7.8, 7.8], color='#38bdf8', lw=2.5)

# Center Junction Node (5.5, 7.8)
ax.add_patch(patches.Circle((5.5, 7.8), 0.14, facecolor='#38bdf8', edgecolor='#ffffff', lw=1.5))
ax.text(5.5, 8.1, '3.3V Out', ha='center', va='bottom', color='#38bdf8', fontsize=8.5, fontweight='bold')

# Connect Junction to Pin 18
ax.annotate('', xy=(7.0, 7.8), xytext=(5.5, 7.8),
            arrowprops=dict(arrowstyle='->', color='#38bdf8', lw=2.5))

# Connect Junction down through Resistor R2 (2k) to GND
ax.plot([5.5, 5.5], [7.8, 6.7], color='#38bdf8', lw=2.5)
r2_box = patches.Rectangle((5.32, 6.1), 0.36, 0.6, facecolor='#1e3a8a', edgecolor='#93c5fd', lw=1.5)
ax.add_patch(r2_box)
ax.text(5.5, 6.4, '2k Ohm', ha='center', va='center', color='#ffffff', fontsize=8.5, fontweight='bold')
ax.plot([5.5, 5.5], [6.1, 5.6], color='#64748b', lw=2.5)

# GND Bus bar
ax.plot([3.2, 5.5, 7.0], [5.6, 5.6, 5.6], color='#64748b', lw=2.5)
ax.plot([3.2, 3.2], [7.2, 5.6], color='#64748b', lw=2.5)
ax.plot([7.0, 7.0], [7.2, 5.6], color='#64748b', lw=2.5)
ax.text(5.1, 5.35, 'Common Ground (GND) Bus Line (BLACK / BROWN wire)', ha='center', va='top', color='#94a3b8', fontsize=8.5)

# Formula & Safety Card (Left Panel)
warn_box = patches.FancyBboxPatch((0.8, 4.6), 8.6, 0.65, boxstyle='round,pad=0.1', facecolor='#0f172a', edgecolor='#f59e0b', lw=1.5)
ax.add_patch(warn_box)
ax.text(5.1, 4.95, 'FORMULA: Vout = 5.0V * [ 2k / (1k + 2k) ] = 3.33V (Safe for Pi 5 RP1)', ha='center', va='center', color='#fde047', fontsize=8.8, fontweight='bold')
ax.text(5.1, 4.72, 'CRITICAL: Do NOT reverse! If 2k is first and 1k to GND, Vout=1.66V -> Pi reads NO ECHO!', ha='center', va='center', color='#f87171', fontsize=8)

# ==============================================================================
# RIGHT PANEL: METHOD 2 - LOGIC LEVEL CONVERTER (LLC MODULE)
# ==============================================================================
p2 = patches.FancyBboxPatch((10.3, 4.4), 9.2, 7.1, boxstyle='round,pad=0.25', facecolor='#1e293b', edgecolor='#a855f7', lw=2)
ax.add_patch(p2)
ax.text(14.9, 11.2, 'METHOD 2: 4-Channel Logic Level Shifter Module', ha='center', va='center', color='#c084fc', fontsize=13, fontweight='bold')
ax.text(14.9, 10.85, 'Active bidirectional MOSFET shifter for high-noise environments', ha='center', va='center', color='#94a3b8', fontsize=9.5)

# HC-SR04 Sensor Box (Right Panel)
s2 = patches.FancyBboxPatch((10.8, 7.2), 2.2, 3.2, boxstyle='round,pad=0.15', facecolor='#0284c7', edgecolor='#e2e8f0', lw=2)
ax.add_patch(s2)
ax.text(11.9, 10.0, 'HC-SR04', ha='center', va='center', color='#ffffff', fontsize=12, fontweight='bold')
ax.text(11.9, 9.6, 'Ultrasonic', ha='center', va='center', color='#e0f2fe', fontsize=9)

for name, y, col in pins_s1:
    ax.add_patch(patches.Circle((13.0, y), 0.12, facecolor=col, edgecolor='#ffffff', lw=1))
    ax.text(12.8, y, name, ha='right', va='center', color='#ffffff', fontsize=9, fontweight='bold')

# Logic Level Shifter Box (Right Panel Middle)
llc_box = patches.FancyBboxPatch((13.9, 6.2), 2.0, 3.6, boxstyle='round,pad=0.15', facecolor='#581c87', edgecolor='#e9d5ff', lw=2)
ax.add_patch(llc_box)
ax.text(14.9, 9.5, 'LLC Module', ha='center', va='center', color='#ffffff', fontsize=11, fontweight='bold')
ax.text(14.9, 9.15, 'Bi-Directional', ha='center', va='center', color='#e9d5ff', fontsize=8)

# LLC Left Pins (High Voltage 5V)
ax.text(14.1, 8.6, 'HV (5V)', ha='left', va='center', color='#fca5a5', fontsize=8, fontweight='bold')
ax.text(14.1, 7.8, 'HV1', ha='left', va='center', color='#fde047', fontsize=8, fontweight='bold')
ax.text(14.1, 7.0, 'GND', ha='left', va='center', color='#cbd5e1', fontsize=8, fontweight='bold')

# LLC Right Pins (Low Voltage 3.3V)
ax.text(15.7, 8.6, 'LV (3.3V)', ha='right', va='center', color='#67e8f9', fontsize=8, fontweight='bold')
ax.text(15.7, 7.8, 'LV1', ha='right', va='center', color='#67e8f9', fontsize=8, fontweight='bold')
ax.text(15.7, 7.0, 'GND', ha='right', va='center', color='#cbd5e1', fontsize=8, fontweight='bold')

# Raspberry Pi 5 Box (Right Panel)
rpi2 = patches.FancyBboxPatch((16.8, 6.2), 2.3, 4.3, boxstyle='round,pad=0.15', facecolor='#15803d', edgecolor='#e2e8f0', lw=2)
ax.add_patch(rpi2)
ax.text(17.95, 10.15, 'Raspberry Pi 5', ha='center', va='center', color='#ffffff', fontsize=11, fontweight='bold')
ax.text(17.95, 9.8, '40-Pin Header', ha='center', va='center', color='#dcfce7', fontsize=8.5)

pins_rpi2 = [
    ('Pin 2 (5V DC)', 9.2, '#ef4444'),
    ('Pin 1 (3.3V Ref)', 8.6, '#06b6d4'),
    ('Pin 16 (GPIO 23)', 8.0, '#f97316'),
    ('Pin 18 (GPIO 24)', 7.4, '#38bdf8'),
    ('Pin 6 (GND)', 6.6, '#64748b')
]
for name, y, col in pins_rpi2:
    ax.add_patch(patches.Circle((16.8, y), 0.12, facecolor=col, edgecolor='#ffffff', lw=1))
    ax.text(17.0, y, name, ha='left', va='center', color='#ffffff', fontsize=8.5, fontweight='bold')

# Wires (Right Panel)
# 5V Power: Pin 2 -> Sensor VCC & LLC HV
ax.plot([16.8, 14.9, 14.9], [9.2, 9.2, 8.6], color='#ef4444', lw=2)
ax.plot([14.9, 13.0], [9.2, 9.0], color='#ef4444', lw=2)

# 3.3V Power: Pin 1 -> LLC LV
ax.plot([16.8, 15.9], [8.6, 8.6], color='#06b6d4', lw=2)

# TRIG: Sensor TRIG -> Pin 16 Direct
ax.annotate('', xy=(16.8, 8.0), xytext=(13.0, 8.4),
            arrowprops=dict(arrowstyle='->', color='#f97316', lw=2.5))

# ECHO: Sensor ECHO -> LLC HV1 -> LLC LV1 -> Pin 18
ax.annotate('', xy=(13.9, 7.8), xytext=(13.0, 7.8),
            arrowprops=dict(arrowstyle='->', color='#eab308', lw=2.5))
ax.annotate('', xy=(16.8, 7.4), xytext=(15.9, 7.8),
            arrowprops=dict(arrowstyle='->', color='#38bdf8', lw=2.5))

# GND lines
ax.plot([13.0, 13.9], [7.2, 7.0], color='#64748b', lw=2)
ax.plot([15.9, 16.8], [7.0, 6.6], color='#64748b', lw=2)

# Explanatory card (Right Panel)
note2 = patches.FancyBboxPatch((10.6, 4.6), 8.6, 0.65, boxstyle='round,pad=0.1', facecolor='#0f172a', edgecolor='#a855f7', lw=1.5)
ax.add_patch(note2)
ax.text(14.9, 4.95, 'HOW IT WORKS: 5V Echo pulse enters HV1, steps down to 3.3V at LV1 safely.', ha='center', va='center', color='#e9d5ff', fontsize=8.8, fontweight='bold')
ax.text(14.9, 4.72, 'TRIG is driven directly by Pin 16 (3.3V is accepted as HIGH by 5V HC-SR04).', ha='center', va='center', color='#cbd5e1', fontsize=8)

# ==============================================================================
# BOTTOM PANEL: PIN HEADER REFERENCE & HARDWARE CHECKS
# ==============================================================================
bot = patches.FancyBboxPatch((0.5, 0.5), 19.0, 3.6, boxstyle='round,pad=0.25', facecolor='#1e293b', edgecolor='#64748b', lw=1.5)
ax.add_patch(bot)

ax.text(10, 3.8, 'COMPLETE RASPBERRY PI 5 GPIO PINOUT & WIRING QUICK-CHECK', ha='center', va='center', color='#f8fafc', fontsize=12, fontweight='bold')

# Draw 40-pin header visualization
pin_data = [
    (1, '3.3V', '#06b6d4', 'Power (LLC LV)'),
    (2, '5.0V', '#ef4444', 'Sensor VCC Power'),
    (6, 'GND', '#64748b', 'Power Ground'),
    (11, 'GPIO 17', '#3b82f6', 'Push Button (Scan)'),
    (13, 'GPIO 27', '#a855f7', 'Alert Buzzer (+)'),
    (14, 'GND', '#64748b', 'Buzzer Ground (-)'),
    (16, 'GPIO 23', '#f97316', 'HC-SR04 TRIGGER'),
    (18, 'GPIO 24', '#38bdf8', 'HC-SR04 ECHO (3.3V)')
]

x_start = 1.0
card_w = 2.15
card_gap = 0.1
for idx, (p_num, bcm, p_col, desc) in enumerate(pin_data):
    cx = x_start + idx * (card_w + card_gap)
    cbox = patches.FancyBboxPatch((cx, 1.4), card_w, 2.1, boxstyle='round,pad=0.1', facecolor='#0f172a', edgecolor=p_col, lw=1.5)
    ax.add_patch(cbox)
    ax.text(cx + card_w/2, 3.15, f'PIN {p_num}', ha='center', va='center', color='#ffffff', fontsize=11, fontweight='bold')
    ax.text(cx + card_w/2, 2.75, bcm, ha='center', va='center', color=p_col, fontsize=10, fontweight='bold')
    ax.text(cx + card_w/2, 2.05, desc, ha='center', va='center', color='#94a3b8', fontsize=8, wrap=True)

# 2 Quick Diagnosis Bullet Points
ax.text(1.5, 0.95, '[CHECK 1] VCC MUST be on Pin 2 or 4 (5V). Pin 1 (3.3V) will NOT power standard HC-SR04!', color='#f8fafc', fontsize=9, fontweight='bold')
ax.text(1.5, 0.70, '[CHECK 2] If test reports "No echo received", swap Pin 16 and Pin 18 (Trigger and Echo are frequently swapped).', color='#f8fafc', fontsize=9, fontweight='bold')

plt.tight_layout()
out1 = 'pi5_2gb_assistant/HC_SR04_Logic_Level_Wiring_Diagram.png'
out2 = r'C:\Users\suhas\.gemini\antigravity\brain\553fd0ff-805e-4e27-b3c1-4131bb6edf8c\HC_SR04_Logic_Level_Wiring_Diagram.png'
os.makedirs('pi5_2gb_assistant', exist_ok=True)
os.makedirs(os.path.dirname(out2), exist_ok=True)
plt.savefig(out1, facecolor='#0b1120', dpi=220)
plt.savefig(out2, facecolor='#0b1120', dpi=220)
print('Diagrams generated successfully:', out1, out2)
