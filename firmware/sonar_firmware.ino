/*
  ==============================================================================
  Smart Vision Assistant - Ultrasonic Sonar Serial Hardware Bridge
  Target Hardware: Arduino Uno / Nano / Mega / ESP32 / ESP8266 / Raspberry Pi Pico
  Sensor: HC-SR04 Ultrasonic Distance Sensor
  ==============================================================================
  
  Wiring Diagram:
    HC-SR04 VCC   --> 5V Pin
    HC-SR04 GND   --> GND Pin
    HC-SR04 TRIG  --> Digital Pin 9  (or configured TRIGGER_PIN)
    HC-SR04 ECHO  --> Digital Pin 10 (or configured ECHO_PIN)
    
  Data Format:
    Streams real-time calibrated distance to USB Serial at 115200 baud:
    DISTANCE: 1.42
    
  Compatibility:
    - Compatible with `simulate_pc.py --sonar-port COMx`
    - Compatible with `test_sonar.py --port COMx`
    - Compatible with Raspberry Pi USB serial input
*/

#define TRIGGER_PIN  9
#define ECHO_PIN     10
#define STATUS_LED   13

// Ultrasonic timing constants
// Speed of sound at 20°C is ~343 m/s = 0.0343 cm/µs
// Distance (cm) = (duration_µs * 0.0343) / 2 = duration_µs / 58.2
// Timeout: 25ms timeout corresponds to ~4.3 meters maximum range
const unsigned long ECHO_TIMEOUT_US = 25000;

// Median filter buffer for noise suppression
const int NUM_SAMPLES = 5;
float sample_buffer[NUM_SAMPLES];
int sample_idx = 0;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    // Wait for native USB CDC (Leonardo / Micro / Pico / ESP32-S3)
  }

  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(STATUS_LED, OUTPUT);

  digitalWrite(TRIGGER_PIN, LOW);
  digitalWrite(STATUS_LED, LOW);

  // Initialize median filter buffer
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sample_buffer[i] = 2.0; // default 2.0m
  }

  Serial.println(F("# SMART VISION ASSISTANT - ULTRASONIC SONAR BRIDGE READY"));
  Serial.println(F("# Baud: 115200 | Sensor: HC-SR04 | Max Range: 3.5m"));
}

float measureDistanceMeters() {
  // 1. Send clean 10-microsecond trigger pulse
  digitalWrite(TRIGGER_PIN, LOW);
  delayMicroseconds(4);
  digitalWrite(TRIGGER_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER_PIN, LOW);

  // 2. Measure high pulse duration on echo pin
  unsigned long duration = pulseIn(ECHO_PIN, HIGH, ECHO_TIMEOUT_US);

  if (duration == 0) {
    return -1.0; // Echo timeout or obstacle out of 3.5m range
  }

  // 3. Compute distance in meters
  // (duration in µs * 343 m/s) / (2 * 1,000,000 µs/s) = duration * 0.0001715
  float dist_m = (float)duration * 0.0001715;

  if (dist_m < 0.02 || dist_m > 3.50) {
    return -1.0; // Out of bounds for reliable navigation
  }

  return dist_m;
}

// Simple insertion sort to find median of last N samples
float getFilteredDistance(float new_val) {
  if (new_val > 0.0) {
    sample_buffer[sample_idx] = new_val;
    sample_idx = (sample_idx + 1) % NUM_SAMPLES;
  }

  float sorted[NUM_SAMPLES];
  for (int i = 0; i < NUM_SAMPLES; i++) {
    sorted[i] = sample_buffer[i];
  }

  for (int i = 1; i < NUM_SAMPLES; i++) {
    float key = sorted[i];
    int j = i - 1;
    while (j >= 0 && sorted[j] > key) {
      sorted[j + 1] = sorted[j];
      j--;
    }
    sorted[j + 1] = key;
  }

  return sorted[NUM_SAMPLES / 2]; // Return median
}

void loop() {
  float raw_dist = measureDistanceMeters();
  float filtered_dist = getFilteredDistance(raw_dist);

  // Blink indicator LED on proximity (< 1.0m)
  if (filtered_dist < 1.00 && filtered_dist > 0.0) {
    digitalWrite(STATUS_LED, HIGH);
  } else {
    digitalWrite(STATUS_LED, LOW);
  }

  // Output standard telemetry packet
  Serial.print(F("DISTANCE: "));
  Serial.println(filtered_dist, 2);

  // Sample at ~15-20 Hz for smooth real-time tracking
  delay(55);
}
