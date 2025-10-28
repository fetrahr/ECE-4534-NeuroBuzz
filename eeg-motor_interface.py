#!/usr/bin/env python3
"""
EEG → Haptics interface using BrainFlow predefined metrics (RELAXATION, CONCENTRATION, STRESS).

- Streams EEG from an OpenBCI (or any BrainFlow-supported) board
- Computes BrainFlow metrics on a rolling window
- Maps metric scores to motor commands (intensity / pattern)
- I2C motor control is intentionally NOT implemented — clearly-marked stubs are provided
"""
from __future__ import annotations
import argparse
import signal
import sys
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from brainflow.ml_model import (
    MLModel,
    BrainFlowModelParams,
    BrainFlowMetrics,
    BrainFlowClassifiers,
)


# =========================
# Configuration
# =========================
@dataclass
class Config:
    board_id: int = BoardIds.CYTON_BOARD.value  # Change if not using Cyton
    sampling_rate: int = 250                    # Cyton default
    window_sec: float = 4.0                     # seconds of data per inference
    step_sec: float = 1.0                       # inference cadence
    bandpass_low: float = 1.0                   # Hz
    bandpass_high: float = 40.0                 # Hz
    notch: float = 60.0                         # set to 50.0 in EU

    # Channels of interest (override after connecting using BoardShim.get_eeg_channels)
    eeg_channels: List[int] | None = None

    # Metric thresholds (tune!)
    relax_low: float = 0.40
    relax_high: float = 0.70
    focus_low: float = 0.40
    focus_high: float = 0.70
    stress_low: float = 0.30
    stress_high: float = 0.60

    # I2C / DRV2605L placeholders — fill in when you wire hardware
    i2c_bus: int = 1
    drv2605l_addresses: List[int] = None  # e.g., [0x5A, 0x5B, 0x5C]
    use_i2c_mux: bool = False
    i2c_mux_address: int = 0x70  # Common TCA9548A address


# =========================
# I2C STUBS (fill in later)
# =========================
class HapticsDriver:
    """Placeholder for DRV2605L (or similar) control over I2C.

    Replace STUB sections with your actual I2C implementation.
    Suggested Python libs: `smbus2` or `periphery` on Raspberry Pi.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # STUB: Initialize I2C bus, mux, and drivers here
        # Example:
        #   from smbus2 import SMBus
        #   self.bus = SMBus(cfg.i2c_bus)
        #   if cfg.use_i2c_mux: self._select_mux_channel(0)
        pass

    def close(self):
        # STUB: Close I2C resources if needed
        pass

    def set_global_intensity(self, intensity: int):
        """Set overall vibration intensity 0..100.
        Map this to DRV2605L real registers when you implement.
        """
        intensity = int(np.clip(intensity, 0, 100))
        # STUB: write to I2C to set ERM/LRA amplitude registers or waveform library scaling
        # For multiple drivers, iterate self.cfg.drv2605l_addresses
        # Example:
        #   for addr in self.cfg.drv2605l_addresses:
        #       self._write_reg(addr, REG, value)
        pass

    def play_pattern(self, pattern_id: int, duration_ms: int):
        """Play a preconfigured haptic pattern for `duration_ms`.
        Map pattern_id to DRV2605L waveform sequence slots.
        """
        # STUB: select waveform sequence, trigger GO bit, sleep, then stop
        pass

    def stop_all(self):
        # STUB: send STOP/standby to all drivers
        pass

    # --- Optional helpers ---
    def _select_mux_channel(self, ch: int):
        # STUB: if using TCA9548A, write 1 << ch to mux address
        pass

    def _write_reg(self, addr: int, reg: int, val: int):
        # STUB: bus.write_byte_data(addr, reg, val)
        pass


# =========================
# EEG/BCI Runtime
# =========================
class EEGToHaptics:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.params = BrainFlowInputParams()
        self.board = BoardShim(cfg.board_id, self.params)
        self.stop_event = threading.Event()
        self.haptics = HapticsDriver(cfg)

        # Prepare ML models for BrainFlow predefined metrics
        self.models: Dict[str, MLModel] = {}
        for metric in (BrainFlowMetrics.RELAXATION, BrainFlowMetrics.CONCENTRATION, BrainFlowMetrics.STRESS):
            p = BrainFlowModelParams(metric.value, BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
            m = MLModel(p)
            m.prepare()
            self.models[metric.name] = m

        self.sr = None
        self.eeg_chs: List[int] = []
        self.window_samples = None
        self.step_samples = None

    # --------------- Lifecycle ---------------
    def start(self):
        BoardShim.enable_dev_board_logger()
        self.board.prepare_session()
        self.board.start_stream()

        # Resolve runtime params from board
        self.sr = BoardShim.get_sampling_rate(self.cfg.board_id)
        self.eeg_chs = BoardShim.get_eeg_channels(self.cfg.board_id) if self.cfg.eeg_channels is None else self.cfg.eeg_channels
        self.window_samples = int(self.cfg.window_sec * self.sr)
        self.step_samples = int(self.cfg.step_sec * self.sr)

        print(f"Connected. sr={self.sr} Hz, eeg_chs={self.eeg_chs}, window={self.window_samples} samples")

        # Warm-up buffer
        self._wait_for_samples(self.window_samples)

        try:
            self._run_loop()
        finally:
            self.shutdown()

    def shutdown(self):
        print("Shutting down...")
        self.haptics.stop_all()  # STUB will be no-op until implemented
        try:
            self.board.stop_stream()
        except Exception:
            pass
        try:
            self.board.release_session()
        except Exception:
            pass
        for m in self.models.values():
            try:
                m.release()
            except Exception:
                pass
        self.haptics.close()
        print("Goodbye.")

    # --------------- Core Loop ---------------
    def _run_loop(self):
        last_infer = time.time()
        residual = np.empty((0, 0))

        while not self.stop_event.is_set():
            # Sleep until next step (soft real-time)
            t_next = last_infer + self.cfg.step_sec
            dt = t_next - time.time()
            if dt > 0:
                time.sleep(dt)
            last_infer = time.time()

            # Get latest chunk
            data = self.board.get_board_data()
            if data.size == 0:
                continue

            eeg = data[self.eeg_chs, :]

            # Keep only last `window_samples` across calls
            # Concatenate with any residual (if you implement overlap)
            if eeg.shape[1] >= self.window_samples:
                eeg_win = eeg[:, -self.window_samples:]
            else:
                # If short, pull more data from internal ring buffer
                needed = self.window_samples - eeg.shape[1]
                old = self.board.get_current_board_data(needed)
                eeg_win = np.hstack([old[self.eeg_chs, :], eeg])
                if eeg_win.shape[1] > self.window_samples:
                    eeg_win = eeg_win[:, -self.window_samples:]

            # Preprocess per channel
            for ch_idx in range(eeg_win.shape[0]):
                DataFilter.detrend(eeg_win[ch_idx], DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(eeg_win[ch_idx], self.sr, self.cfg.bandpass_low, self.cfg.bandpass_high,
                                            4, FilterTypes.BUTTERWORTH.value, 0)
                if self.cfg.notch > 0:
                    DataFilter.perform_bandstop(eeg_win[ch_idx], self.sr, self.cfg.notch - 2.0, self.cfg.notch + 2.0,
                                                2, FilterTypes.BUTTERWORTH.value, 0)

            # Compute average band powers (across channels)
            bands = self._compute_bands(eeg_win)
            feature_vec = np.array([
                bands["delta"], bands["theta"], bands["alpha"], bands["beta"], bands["gamma"],
            ], dtype=np.float32)

            # Predict BrainFlow metrics
            scores = {
                "RELAXATION": self.models["RELAXATION"].predict(feature_vec),
                "CONCENTRATION": self.models["CONCENTRATION"].predict(feature_vec),
                "STRESS": self.models["STRESS"].predict(feature_vec),
            }

            # Map to haptics policy
            intensity, pattern, dur = self._policy(scores)

            # === I2C STUB: Send motor command ===
            # Replace next two lines by actual I2C writes to DRV2605L
            # Example implementation sketch:
            #   self.haptics.set_global_intensity(intensity)
            #   self.haptics.play_pattern(pattern_id=pattern, duration_ms=dur)
            print(f"scores={scores} -> intensity={intensity} pattern={pattern} duration_ms={dur}")

    # --------------- Helpers ---------------
    def _wait_for_samples(self, n: int, timeout: float = 5.0):
        t0 = time.time()
        while self.board.get_board_data_count() < n:
            time.sleep(0.05)
            if time.time() - t0 > timeout:
                break

    def _compute_bands(self, eeg_win: np.ndarray) -> Dict[str, float]:
        """Return mean band powers across channels for delta..gamma."""
        # BrainFlow helper uses Welch + integration internally
        avg_bands, _std_bands = DataFilter.get_avg_band_powers(
            eeg_win, self.eeg_chs, self.sr, True
        )
        # avg_bands order: delta, theta, alpha, beta, gamma
        return {
            "delta": float(avg_bands[0]),
            "theta": float(avg_bands[1]),
            "alpha": float(avg_bands[2]),
            "beta": float(avg_bands[3]),
            "gamma": float(avg_bands[4]),
        }

    def _policy(self, scores: Dict[str, float]) -> Tuple[int, int, int]:
        """Translate metric scores to (intensity %, pattern_id, duration_ms).

        Suggested behavior (tune as you like):
        - Encourage relaxed creative state: low RELAXATION → nudge user to relax
        - Encourage focus when concentration low
        - Avoid overstimulation when STRESS high (use short, gentle pulses)
        """
        relax = float(scores["RELAXATION"])  # 0..1
        focus = float(scores["CONCENTRATION"])  # 0..1
        stress = float(scores["STRESS"])  # 0..1

        # Default values
        intensity = 0
        pattern_id = 0  # Map to DRV2605L library pattern index later
        duration_ms = 150

        # Example heuristic policy:
        if stress > self.cfg.stress_high:
            # User is stressed → gentle brief cue (short pulse, low intensity)
            intensity = 30
            pattern_id = 1  # e.g., soft tap
            duration_ms = 120
        elif relax < self.cfg.relax_low:
            # Needs relaxation → longer pulse to remind to breathe/soften gaze
            intensity = 60
            pattern_id = 2  # e.g., slow ramp
            duration_ms = 400
        elif focus < self.cfg.focus_low:
            # Needs focus → quick double tap
            intensity = 50
            pattern_id = 3  # e.g., double click
            duration_ms = 180
        elif relax > self.cfg.relax_high and focus > self.cfg.focus_high and stress < self.cfg.stress_low:
            # Ideal creative zone → no vibration
            intensity = 0
            pattern_id = 0
            duration_ms = 0
        else:
            # Neutral maintenance cue: very light touch occasionally
            intensity = 20
            pattern_id = 4  # e.g., micro buzz
            duration_ms = 100

        return intensity, pattern_id, duration_ms


# =========================
# CLI / Entrypoint
# =========================

def _install_sigint(handler):
    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handler)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EEG→Haptics Interface (BrainFlow metrics)")
    parser.add_argument("--board", type=int, default=BoardIds.CYTON_BOARD.value, help="BrainFlow Board ID")
    parser.add_argument("--window", type=float, default=4.0, help="Seconds per inference window")
    parser.add_argument("--step", type=float, default=1.0, help="Seconds per inference step")
    parser.add_argument("--notch", type=float, default=60.0, help="Notch center frequency (0 to disable)")
    args = parser.parse_args(argv)

    cfg = Config(
        board_id=args.board,
        window_sec=args.window,
        step_sec=args.step,
        notch=args.notch,
        drv2605l_addresses=[0x5A],  # EDIT: add all your motor driver I2C addresses
    )

    runtime = EEGToHaptics(cfg)

    def _graceful(sig, frame):
        print("\nSignal received, exiting...")
        runtime.stop_event.set()

    _install_sigint(_graceful)

    try:
        runtime.start()
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
