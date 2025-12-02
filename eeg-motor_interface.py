import argparse
import time
import sys
import select

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

# ---------- Motor / I2C imports ----------
import board
import busio
from adafruit_bus_device.i2c_device import I2CDevice
import adafruit_drv2605

import numpy as np
import matplotlib.pyplot as plt

# ---------- Motor Config ----------
MUX_ADDR = 0x70
CHANNELS = [0, 1, 2]      # 3 motors on mux channels 0,1,2
USE_LRA = True            # True for LRA, False for ERM
LRA_LIBRARY = 6           # 6 for LRA, 1 for ERM
MAX_EFFECT = 123          # DRV2605 effect range 1..123
DEFAULT_EFFECT = 120        # simple click / buzz pattern


# ---------- I2C + MUX ----------
i2c = busio.I2C(board.SCL, board.SDA)

def tca_select(ch: int):
    """Select TCA9548A mux channel."""
    with I2CDevice(i2c, MUX_ADDR) as mux:
        mux.write(bytes([1 << ch]))
    time.sleep(0.002)

def make_drv_for_channel(ch: int, use_lra: bool, lib: int):
    """Create and configure a DRV2605 instance for a mux channel."""
    tca_select(ch)
    drv = adafruit_drv2605.DRV2605(i2c)
    if use_lra:
        drv.use_LRM()
        drv.library = lib
    else:
        drv.use_ERM()
        drv.library = 1
    return drv

drivers = [make_drv_for_channel(ch, USE_LRA, LRA_LIBRARY) for ch in CHANNELS]

def trigger_single_motor(motor_index: int, effect: int = DEFAULT_EFFECT):
    """
    Trigger a single motor by index (0,1,2).
    Uses sequence[0] with a single effect and calls play().
    """
    if not (0 <= motor_index < len(drivers)):
        return
    ch = CHANNELS[motor_index]
    drv = drivers[motor_index]
    tca_select(ch)
    drv.sequence[0] = adafruit_drv2605.Effect(effect)
    drv.play()

def run_motor_pattern(mindfulness_detected: bool,
                      restfulness_detected: bool,
                      motors_enabled: bool):
    """
    Map EEG metrics → motor patterns:

      - Mindfulness only  (True, False):
          buzz left->right

      - Restfulness only (False, True):
          buzz right->left

      - Neither or both (False, False) or (True, True):
          all motors buzz once (periodic "baseline" buzz)
    """
    if not motors_enabled:
        return

    # Mindfulness only: left -> right
    if mindfulness_detected and not restfulness_detected:
        for idx in CHANNELS:
            trigger_single_motor(idx)
            time.sleep(0.5)  # step timing between motors
        return

    # Restfulness only: right -> left
    if restfulness_detected and not mindfulness_detected:
        for idx in CHANNELS:
            trigger_single_motor(idx)
            time.sleep(0.5)
        return

    # Neither (or both) detected: buzz all motors once (periodic)
    for idx in CHANNELS:
        trigger_single_motor(idx)

def stop_all_motors():
    """Stop all DRV2605 outputs (best-effort)."""
    for ch, drv in zip(CHANNELS, drivers):
        try:
            tca_select(ch)
            drv.stop()
        except Exception:
            pass

def check_for_input(motors_enabled: bool):
    """
    Non-blocking check for user commands from stdin.

    - 's' + Enter: toggle motors on/off
    - 'r' + Enter: start record/playback sequence

    Returns: (motors_enabled, record_requested)
    """
    record_requested = False
    rlist, _, _ = select.select([sys.stdin], [], [], 0)
    if sys.stdin in rlist:
        line = sys.stdin.readline().strip().lower()
        if line == 's':
            motors_enabled = not motors_enabled
            state = "ENABLED" if motors_enabled else "DISABLED"
            print(f"[motors] {state}")
        elif line == 'r':
            record_requested = True
    return motors_enabled, record_requested

def collect_segment(board, duration_s, sampling_rate, with_stim=False):
    """
    Collect 'duration_s' seconds of data from the board.

    If with_stim is True, continuously run a simple stimulation pattern
    (buzz all three motors sequentially) during the segment.
    """
    collected = []
    t_start = time.time()

    while time.time() - t_start < duration_s:
        if with_stim:
            # simple pattern: left->right each loop
            for idx in [0, 1, 2]:
                trigger_single_motor(idx)
                time.sleep(0.05)

        # read whatever data accumulated since last call
        chunk = board.get_board_data()
        if chunk.size > 0:
            collected.append(chunk)

        # small sleep to avoid busy-waiting
        time.sleep(0.05)

    if collected:
        data = np.concatenate(collected, axis=1)
    else:
        # Fallback: grab last N samples if nothing collected for some reason
        n_samples = int(duration_s * sampling_rate)
        data = board.get_current_board_data(n_samples)
    return data

def run_record_playback(board, eeg_channels, sampling_rate,
                        mindfulness_model, restfulness_model):
    """
    Implements your 'r' mode:

      - Print explanation
      - Wait for Enter
      - 10s baseline (no stim)
      - 10s stimulation (motors buzzing)
      - FFT plots (baseline vs stim)
      - Metrics plots (mindfulness/restfulness baseline vs stim)
      - Save all data to file
    """
    print("\n=== Record / Playback Mode ===")
    print("This mode will:")
    print("  1) Record 10 seconds of EEG with NO motor stimulation.")
    print("  2) Then record 10 seconds of EEG WITH motor stimulation.")
    print("  3) Show FFT plots comparing both segments.")
    print("  4) Show mindfulness/restfulness scores for both segments.")
    print("  5) Save the recorded data and metrics to a file.\n")
    input("Press Enter to start the 10-second baseline (no motors)...")

    # Clear any old data from the buffer so we get clean segments
    _ = board.get_board_data()

    # 1) Baseline segment (no motors)
    print("Recording baseline (no motor stimulation) for 10 seconds...")
    stop_all_motors()
    baseline_data = collect_segment(board, duration_s=10,
                                    sampling_rate=sampling_rate,
                                    with_stim=False)
    print("Baseline recording done.\n")

    # 2) Stimulation segment
    input("Press Enter to start 10-second stimulation segment...")
    print("Recording with motor stimulation for 10 seconds...")
    # Motors are driven inside collect_segment(with_stim=True)
    stim_data = collect_segment(board, duration_s=10,
                                sampling_rate=sampling_rate,
                                with_stim=True)
    stop_all_motors()
    print("Stimulation recording done.\n")

    # ---------- Compute FFTs ----------
    # Use mean across EEG channels for a simple 1D signal per segment
    baseline_eeg = baseline_data[eeg_channels, :]
    stim_eeg = stim_data[eeg_channels, :]

    baseline_signal = baseline_eeg.mean(axis=0)
    stim_signal = stim_eeg.mean(axis=0)

    # FFT
    def compute_fft(sig, fs):
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spectrum = np.abs(np.fft.rfft(sig))
        return freqs, spectrum

    freqs_b, spec_b = compute_fft(baseline_signal, sampling_rate)
    freqs_s, spec_s = compute_fft(stim_signal, sampling_rate)

    # ---------- Compute metrics ----------
    def compute_metrics(segment):
        bands = DataFilter.get_avg_band_powers(segment, eeg_channels,
                                               sampling_rate, True)
        feature_vector = bands[0]
        m_score = float(mindfulness_model.predict(feature_vector)[0])
        r_score = float(restfulness_model.predict(feature_vector)[0])
        return m_score, r_score

    m_base, r_base = compute_metrics(baseline_data)
    m_stim, r_stim = compute_metrics(stim_data)

    print(f"Baseline Mindfulness:  {m_base:.3f}, Restfulness: {r_base:.3f}")
    print(f"Stim Mindfulness:      {m_stim:.3f}, Restfulness: {r_stim:.3f}\n")

    # ---------- Plot FFTs ----------
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("FFT - Baseline (no stimulation)")
    plt.plot(freqs_b, spec_b)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.subplot(2, 1, 2)
    plt.title("FFT - Stimulation")
    plt.plot(freqs_s, spec_s)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()

    # ---------- Plot metrics ----------
    labels = ["Baseline", "Stim"]
    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    width = 0.35

    plt.bar(x - width/2, [m_base, m_stim], width, label="Mindfulness")
    plt.bar(x + width/2, [r_base, r_stim], width, label="Restfulness")

    plt.xticks(x, labels)
    plt.ylabel("Score")
    plt.title("Mindfulness / Restfulness: Baseline vs Stim")
    plt.legend()
    plt.tight_layout()

    plt.show()

    # ---------- Save data ----------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"eeg_record_playback_{timestamp}.npz"
    np.savez(
        filename,
        baseline=baseline_data,
        stim=stim_data,
        sampling_rate=sampling_rate,
        eeg_channels=np.array(eeg_channels),
        baseline_mindfulness=m_base,
        baseline_restfulness=r_base,
        stim_mindfulness=m_stim,
        stim_restfulness=r_stim,
    )
    print(f"[saved] Record/playback data saved to {filename}\n")

def main():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    #parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
    #
    #                     required=True)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = '/dev/ttyUSB0'
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    board = BoardShim(BoardIds.CYTON_BOARD, params)
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)


    board.prepare_session()
    board.start_stream(45000, args.streamer_params)
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Start continuous EEG metric loop')
    print("Controls:")
    print("  s + Enter  -> toggle motors on/off")
    print("  r + Enter  -> record/playback mode (10s baseline + 10s stim)")
    print("  Ctrl+C     -> exit\n")
    
    mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()
    
    restfulness_params = BrainFlowModelParams(BrainFlowMetrics.RESTFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    restfulness = MLModel(restfulness_params)
    restfulness.prepare()
    
    MINDFULNESS_THRESHOLD = 0.7
    RESTFULNESS_THRESHOLD = 0.7

    eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
    
    window_sec = 4
    num_samples = int(window_sec * sampling_rate)

    motors_enabled = True

    try:
        while True:
            # Handle user input (s / r) without blocking
            motors_enabled, record_requested = check_for_input(motors_enabled)

            if record_requested:
                # Run the 20s record/playback experiment
                run_record_playback(board, eeg_channels, sampling_rate,
                                    mindfulness, restfulness)
                # After that, continue normal loop
                continue

            # let the buffer fill enough for one window
            time.sleep(window_sec)

            # get the most recent window of data
            data = board.get_current_board_data(num_samples)

            # skip if not enough data yet
            if data.shape[1] < num_samples:
                print("Not enough data yet, waiting...")
                continue

            # compute band powers and feature vector
            bands = DataFilter.get_avg_band_powers(
                data, eeg_channels, sampling_rate, True
            )
            feature_vector = bands[0]

            # compute metrics (each is a 1-element array -> take [0] and cast to float)
            mindfulness_score = float(mindfulness.predict(feature_vector)[0])
            restfulness_score = float(restfulness.predict(feature_vector)[0])

            mindfulness_detected = mindfulness_score >= MINDFULNESS_THRESHOLD
            restfulness_detected = restfulness_score >= RESTFULNESS_THRESHOLD

            print(
                f"Mindfulness: {mindfulness_score:.3f} "
                f"(detected={mindfulness_detected}), "
                f"Restfulness: {restfulness_score:.3f} "
                f"(detected={restfulness_detected})"
            )
            print()  # blank line between updates

            # ---- drive motor pattern based on metrics ----
            run_motor_pattern(
                mindfulness_detected,
                restfulness_detected,
                motors_enabled
            )

    except KeyboardInterrupt:
        print("Stopping on user interrupt...")

    finally:
        # clean up
        board.stop_stream()
        board.release_session()
        mindfulness.release()
        restfulness.release()

        # stop motors on exit
        stop_all_motors()



if __name__ == "__main__":
    main()