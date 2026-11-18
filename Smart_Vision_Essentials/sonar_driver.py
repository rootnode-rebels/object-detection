#!/usr/bin/env python3
"""
High-Precision HC-SR04 Ultrasonic Sonar Driver for Raspberry Pi 5 (RP1 / lgpio)
================================================================================
Replaces gpiozero's unvalidated DistanceSensor background polling with a direct
kernel-IRQ hardware-timestamped lgpio driver specifically engineered for Pi 5.

Why this fixes wrong/erratic distances on Raspberry Pi 5:
1. Hardware Kernel Timestamps (lgpio.callback):
   Uses Linux kernel GPIO IRQ timestamps (nanosecond resolution) recorded directly
   inside the RP1 interrupt handler — immune to Python GIL stalls and YOLOv8 CPU load.
2. Strict Trigger-to-Echo Correlation & Bounce Rejection:
   - Only accepts falling edges that follow a rising edge armed AFTER the trigger pulse.
   - Rejects <100us electrical/resistor-divider bounce glitches automatically.
   - Cross-checks pulse duration against wall-clock perf_counter_ns() to reject stale events.
   - Enables internal PULL-DOWN on ECHO so floating wires never generate ghost readings.
3. Datasheet Acoustic Settling (>=70ms cycle):
   Prevents late wall reflections from colliding with the next ultrasonic trigger.
4. 5-Sample Median + Outlier Spike Rejection Filter:
   Eliminates single-ping acoustic anomalies while preserving snappy ~140ms hand response.
5. Automatic Pi 5 RP1 Chip Detection:
   Finds 'pinctrl-rp1' across both gpiochip4 (Bookworm) and gpiochip0 (Trixie/6.6+).
   Gracefully falls back to gpiozero if lgpio is unavailable.
"""

import time
import threading
import statistics
from collections import deque


def _open_rp1_gpiochip(lgpio_mod):
    """
    Locates and opens the 40-pin header gpiochip on Raspberry Pi 5 (pinctrl-rp1)
    across all kernel versions (Bookworm gpiochip4 and Trixie/6.6+ gpiochip0).
    Falls back to standard Pi 4/3 chip 0 if RP1 is not present.
    """
    # Pass 1: Look explicitly for Raspberry Pi 5 RP1 controller
    for chip_num in (4, 0, 1, 2, 3):
        h = None
        try:
            h = lgpio_mod.gpiochip_open(chip_num)
            info = lgpio_mod.gpio_get_chip_info(h)
            # lgpio.gpio_get_chip_info returns [status, lines, name, label]
            label = str(info[3]).lower() if isinstance(info, (list, tuple)) and len(info) >= 4 else str(info).lower()
            if "rp1" in label:
                return h, chip_num, label
            lgpio_mod.gpiochip_close(h)
        except Exception:
            if h is not None:
                try:
                    lgpio_mod.gpiochip_close(h)
                except Exception:
                    pass

    # Pass 2: Standard Pi 4 / Pi 3 / generic pinctrl fallback
    for chip_num in (0, 4):
        h = None
        try:
            h = lgpio_mod.gpiochip_open(chip_num)
            info = lgpio_mod.gpio_get_chip_info(h)
            label = str(info[3]).lower() if isinstance(info, (list, tuple)) and len(info) >= 4 else f"gpiochip{chip_num}"
            return h, chip_num, label
        except Exception:
            if h is not None:
                try:
                    lgpio_mod.gpiochip_close(h)
                except Exception:
                    pass

    raise RuntimeError("Could not open any Linux gpiochip device (check gpio group permissions).")


class Pi5UltrasonicSonar:
    """
    Precision HC-SR04 Ultrasonic Distance Sensor driver.
    Uses direct lgpio hardware-IRQ nanosecond timestamps on Raspberry Pi 5,
    with automatic fallback to direct lgpio polling or gpiozero if needed.
    """

    def __init__(
        self,
        trigger_pin=23,
        echo_pin=24,
        max_distance=3.0,
        temp_c=25.0,
        window_size=5,
        ping_interval=0.07,
        use_polling=False,
    ):
        self.trigger_pin = int(trigger_pin)
        self.echo_pin = int(echo_pin)
        self.max_distance = float(max_distance)
        self.speed_of_sound = 331.3 + (0.606 * float(temp_c))  # m/s
        self.ping_interval = max(0.06, float(ping_interval))   # >=60ms per HC-SR04 datasheet
        self.use_polling = bool(use_polling)

        # Thread-safe IRQ capture state
        self._irq_lock = threading.Lock()
        self._armed = False
        self._rise_ns = None
        self._fall_ns = None
        self._done_event = threading.Event()

        # Rolling median filter & spike-guard state
        self._window = deque(maxlen=max(3, int(window_size)))
        self._pending_jump = None
        self._last_raw_m = None
        self._last_pulse_us = None
        self._last_status = "INIT"

        # Hardware handles
        self.backend = None
        self.chip_info = ""
        self._lgpio = None
        self._h = None
        self._cb = None
        self._gz_sensor = None
        self._closed = False

        # Auto-fallback counter if IRQ callbacks fail on an unusual kernel
        self._consecutive_irq_misses = 0

        self._init_hardware()

    def _init_hardware(self):
        # Priority 1: Direct lgpio on Raspberry Pi 5 / modern Raspberry Pi OS
        try:
            import lgpio as lgpio_mod

            h, chip_num, label = _open_rp1_gpiochip(lgpio_mod)
            self._lgpio = lgpio_mod
            self._h = h
            self.chip_info = f"gpiochip{chip_num} ({label})"

            # Claim TRIG as output LOW
            lgpio_mod.gpio_claim_output(h, self.trigger_pin, 0)

            # Pull-down flag compatible with both lgpio 0.1.x and 0.2.x
            pull_down = getattr(
                lgpio_mod,
                "SET_PULL_DOWN",
                getattr(lgpio_mod, "SET_BIAS_PULL_DOWN", 0),
            )

            if not self.use_polling:
                # Claim ECHO for kernel IRQ edge timestamps with internal pull-down
                lgpio_mod.gpio_claim_alert(h, self.echo_pin, lgpio_mod.BOTH_EDGES, pull_down)
                try:
                    lgpio_mod.gpio_set_debounce_micros(h, self.echo_pin, 0)
                except Exception:
                    pass
                self._cb = lgpio_mod.callback(h, self.echo_pin, lgpio_mod.BOTH_EDGES, self._on_edge)
                self.backend = "lgpio-irq"
            else:
                lgpio_mod.gpio_claim_input(h, self.echo_pin, pull_down)
                self.backend = "lgpio-poll"

            # Let TRIG settle LOW for 20ms before first ping
            time.sleep(0.02)
            return
        except Exception as e:
            # Clean up any partial lgpio handles before falling back
            self._cleanup_lgpio()
            lgpio_err = e

        # Priority 2: Fallback to gpiozero (for older Pi OS without lgpio)
        try:
            from gpiozero import DistanceSensor

            self._gz_sensor = DistanceSensor(
                echo=self.echo_pin,
                trigger=self.trigger_pin,
                max_distance=self.max_distance,
                queue_len=5,
                partial=True,
            )
            self._gz_sensor.speed_of_sound = self.speed_of_sound
            self.backend = "gpiozero"
            self.chip_info = "gpiozero-default"
            return
        except Exception as gz_err:
            raise RuntimeError(
                f"Failed to initialize ultrasonic sonar on TRIG=GPIO{self.trigger_pin}, "
                f"ECHO=GPIO{self.echo_pin}. lgpio: {lgpio_err} | gpiozero: {gz_err}"
            )

    def _on_edge(self, chip, gpio, level, tick):
        """
        Kernel IRQ edge handler called by lgpio C thread with nanosecond hardware timestamps.
        Rejects electrical bounce glitches (<100us) and enforces strict rise->fall ordering.
        """
        with self._irq_lock:
            if not self._armed:
                return
            if level == 1:
                if self._rise_ns is None:
                    self._rise_ns = tick
            elif level == 0:
                if self._rise_ns is not None:
                    dt_ns = tick - self._rise_ns
                    if dt_ns < 100_000:
                        # Glitch or rising-edge bounce (<100us / <1.7cm) -> discard and wait for true edge
                        self._rise_ns = None
                    else:
                        self._fall_ns = tick
                        self._armed = False
                        self._done_event.set()

    def _switch_to_polling(self):
        """Switches an open lgpio handle from IRQ alert mode to direct high-speed polling."""
        if self._lgpio is None or self._h is None or self.backend == "lgpio-poll":
            return
        try:
            if self._cb is not None:
                self._cb.cancel()
                self._cb = None
            try:
                self._lgpio.gpio_free(self._h, self.echo_pin)
            except Exception:
                pass
            pull_down = getattr(
                self._lgpio,
                "SET_PULL_DOWN",
                getattr(self._lgpio, "SET_BIAS_PULL_DOWN", 0),
            )
            self._lgpio.gpio_claim_input(self._h, self.echo_pin, pull_down)
            self.backend = "lgpio-poll"
            self.use_polling = True
        except Exception:
            pass

    def _restore_irq_mode(self):
        """Restores kernel IRQ alert mode on ECHO pin."""
        if self._lgpio is None or self._h is None or self.backend == "lgpio-irq":
            return
        try:
            try:
                self._lgpio.gpio_free(self._h, self.echo_pin)
            except Exception:
                pass
            pull_down = getattr(
                self._lgpio,
                "SET_PULL_DOWN",
                getattr(self._lgpio, "SET_BIAS_PULL_DOWN", 0),
            )
            self._lgpio.gpio_claim_alert(self._h, self.echo_pin, self._lgpio.BOTH_EDGES, pull_down)
            try:
                self._lgpio.gpio_set_debounce_micros(self._h, self.echo_pin, 0)
            except Exception:
                pass
            self._cb = self._lgpio.callback(self._h, self.echo_pin, self._lgpio.BOTH_EDGES, self._on_edge)
            self.backend = "lgpio-irq"
            self.use_polling = False
        except Exception:
            pass

    def _fire_trigger_pulse(self):
        """Emits a crisp 12-microsecond HIGH trigger pulse on TRIG."""
        lg = self._lgpio
        h = self._h
        trig = self.trigger_pin

        lg.gpio_write(h, trig, 0)
        t0 = time.perf_counter_ns()
        while (time.perf_counter_ns() - t0) < 2_000:
            pass

        lg.gpio_write(h, trig, 1)
        t1 = time.perf_counter_ns()
        while (time.perf_counter_ns() - t1) < 12_000:
            pass
        lg.gpio_write(h, trig, 0)

    def _ping_lgpio_irq(self):
        """Executes one hardware-timestamped ping using lgpio kernel IRQ alerts."""
        lg = self._lgpio
        h = self._h

        # Wait up to 35ms for ECHO to settle LOW before firing a new pulse
        if lg.gpio_read(h, self.echo_pin) == 1:
            deadline = time.perf_counter() + 0.035
            while lg.gpio_read(h, self.echo_pin) == 1:
                if time.perf_counter() > deadline:
                    self._last_status = "ECHO_STUCK_HIGH"
                    return None, None
                time.sleep(0.0005)

        with self._irq_lock:
            self._rise_ns = None
            self._fall_ns = None
            self._armed = True
            self._done_event.clear()

        t_trig_end = time.perf_counter_ns()
        self._fire_trigger_pulse()
        t_trig_end = time.perf_counter_ns()

        # Wait up to 42ms for echo (HC-SR04 max range pulse is ~23ms; open-space timeout pulse is ~38ms)
        got_echo = self._done_event.wait(timeout=0.042)
        t_done = time.perf_counter_ns()

        with self._irq_lock:
            self._armed = False
            rise_ns = self._rise_ns
            fall_ns = self._fall_ns

        if not got_echo or rise_ns is None or fall_ns is None:
            self._consecutive_irq_misses += 1
            self._last_status = "NO_ECHO"
            return None, None

        pulse_ns = fall_ns - rise_ns
        wall_ns = t_done - t_trig_end

        # Physical sanity check: pulse cannot be negative or longer than total wait window (+5ms margin)
        if pulse_ns <= 0 or pulse_ns > (wall_ns + 5_000_000):
            self._last_status = "INVALID_TIMESTAMP"
            return None, None

        self._consecutive_irq_misses = 0
        pulse_us = pulse_ns / 1000.0
        return self._pulse_us_to_distance(pulse_us)

    def _ping_lgpio_poll(self):
        """Executes one direct tight-loop polled ping using lgpio.gpio_read + perf_counter_ns."""
        lg = self._lgpio
        h = self._h
        echo = self.echo_pin

        # Wait for ECHO LOW
        if lg.gpio_read(h, echo) == 1:
            deadline = time.perf_counter() + 0.035
            while lg.gpio_read(h, echo) == 1:
                if time.perf_counter() > deadline:
                    self._last_status = "ECHO_STUCK_HIGH"
                    return None, None
                time.sleep(0.0005)

        self._fire_trigger_pulse()

        # Wait up to 15ms for ECHO to go HIGH (normally takes ~450us after TRIG)
        t_rise_deadline = time.perf_counter_ns() + 15_000_000
        while lg.gpio_read(h, echo) == 0:
            if time.perf_counter_ns() > t_rise_deadline:
                self._last_status = "NO_ECHO"
                return None, None
        t_rise = time.perf_counter_ns()

        # Wait up to 42ms for ECHO to go LOW
        t_fall_deadline = t_rise + 42_000_000
        while lg.gpio_read(h, echo) == 1:
            if time.perf_counter_ns() > t_fall_deadline:
                self._last_status = "ECHO_TIMEOUT"
                return None, None
        t_fall = time.perf_counter_ns()

        pulse_us = (t_fall - t_rise) / 1000.0
        return self._pulse_us_to_distance(pulse_us)

    def _pulse_us_to_distance(self, pulse_us):
        """
        Converts echo pulse duration in microseconds to distance in meters.
        HC-SR04 valid echo range:
          - 115 us .. 17,400 us -> 0.02 m .. 3.00 m (measured obstacle)
          - 17,400 us .. 45,000 us -> >3.00 m clear open space (includes 38ms open-room pulse)
          - <115 us or >45,000 us -> electrical noise / invalid
        """
        if pulse_us < 115.0:
            self._last_status = "GLITCH_SHORT"
            return None, pulse_us

        if pulse_us > 45_000.0:
            self._last_status = "GLITCH_LONG"
            return None, pulse_us

        dist_m = (pulse_us * 1e-6) * self.speed_of_sound / 2.0
        if dist_m > self.max_distance:
            self._last_status = "CLEAR_OPEN"
            return self.max_distance, pulse_us

        self._last_status = "OK"
        return dist_m, pulse_us

    def ping_once(self):
        """
        Performs a single validated ultrasonic measurement.
        Returns (raw_distance_meters_or_None, pulse_microseconds_or_None).
        """
        if self._closed:
            return None, None

        if self.backend == "lgpio-irq":
            dist, pulse_us = self._ping_lgpio_irq()
            # If IRQ never received a single edge after 8 tries, test polling once:
            # only stay in polling if polling actually succeeds (avoids leaving IRQ mode when sensor is simply unplugged)
            if dist is None and self._consecutive_irq_misses == 8:
                self._switch_to_polling()
                p_dist, p_us = self._ping_lgpio_poll()
                if p_dist is not None:
                    self._last_raw_m = p_dist
                    self._last_pulse_us = p_us
                    return p_dist, p_us
                else:
                    # Sensor is genuinely unplugged/silent -> restore IRQ mode cleanly
                    self._restore_irq_mode()
            self._last_raw_m = dist
            self._last_pulse_us = pulse_us
            return dist, pulse_us

        if self.backend == "lgpio-poll":
            dist, pulse_us = self._ping_lgpio_poll()
            self._last_raw_m = dist
            self._last_pulse_us = pulse_us
            return dist, pulse_us

        if self.backend == "gpiozero" and self._gz_sensor is not None:
            try:
                has_items = (
                    hasattr(self._gz_sensor, "_queue")
                    and self._gz_sensor._queue is not None
                    and len(self._gz_sensor._queue.queue) > 0
                )
                if not has_items:
                    self._last_status = "NO_ECHO"
                    return None, None
                d = float(self._gz_sensor.distance)
                if 0.02 <= d <= self.max_distance:
                    pulse_us = (d * 2.0 / self.speed_of_sound) * 1_000_000.0
                    self._last_status = "OK" if d < self.max_distance else "CLEAR_OPEN"
                    self._last_raw_m = d
                    self._last_pulse_us = pulse_us
                    return d, pulse_us
            except Exception:
                pass
            self._last_status = "NO_ECHO"
            return None, None

        return None, None

    def read_filtered(self):
        """
        Performs one measurement and applies median + spike-guard filtering.
        Returns filtered distance in meters (float), or None if no valid echo.
        """
        raw_m, _ = self.ping_once()
        if raw_m is None:
            return None

        # Single-ping spike guard: reject sudden >0.80m jumps unless confirmed by 2 consecutive pings
        if len(self._window) >= 3:
            current_med = statistics.median(self._window)
            if abs(raw_m - current_med) > 0.80:
                if self._pending_jump is None or abs(raw_m - self._pending_jump) > 0.35:
                    # First anomalous spike -> hold previous median until confirmed
                    self._pending_jump = raw_m
                    return round(float(current_med), 2)
                else:
                    # Confirmed real move (2 consecutive readings at new distance) -> flush old window
                    self._window.clear()
                    self._window.append(self._pending_jump)
                    self._window.append(raw_m)
                    self._pending_jump = None
                    return round(float(statistics.median(self._window)), 2)

        self._pending_jump = None
        self._window.append(raw_m)
        return round(float(statistics.median(self._window)), 2)

    @property
    def echo_pin_state(self):
        """Returns current logic level (0 or 1) of ECHO pin for diagnostics, or None."""
        if self._lgpio is not None and self._h is not None:
            try:
                return int(self._lgpio.gpio_read(self._h, self.echo_pin))
            except Exception:
                return None
        return None

    @property
    def last_raw_meters(self):
        return self._last_raw_m

    @property
    def last_pulse_us(self):
        return self._last_pulse_us

    @property
    def last_status(self):
        return self._last_status

    def _cleanup_lgpio(self):
        if self._cb is not None:
            try:
                self._cb.cancel()
            except Exception:
                pass
            self._cb = None
        if self._lgpio is not None and self._h is not None:
            for p in (self.trigger_pin, self.echo_pin):
                try:
                    self._lgpio.gpio_free(self._h, p)
                except Exception:
                    pass
            try:
                self._lgpio.gpiochip_close(self._h)
            except Exception:
                pass
        self._h = None

    def close(self):
        self._closed = True
        self._cleanup_lgpio()
        if self._gz_sensor is not None:
            try:
                self._gz_sensor.close()
            except Exception:
                pass
            self._gz_sensor = None
