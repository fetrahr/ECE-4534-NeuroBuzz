import argparse
import time
import sys

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

import numpy as np
import pyqtgraph as pg

def collect_segment(board, duration_s, sampling_rate):
    """
    Collect 'duration_s' seconds of data from the board.
    """
    collected = []
    t_start = time.time()

    while time.time() - t_start < duration_s:
        chunk = board.get_board_data()
        if chunk.size > 0:
            collected.append(chunk)
        time.sleep(0.05)

    if collected:
        data = np.concatenate(collected, axis=1)
    else:
        n_samples = int(duration_s * sampling_rate)
        data = board.get_current_board_data(n_samples)
    return data


def main():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=int, required=False, default=0)
    parser.add_argument('--ip-port', type=int, required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, required=False, default=0)
    parser.add_argument('--ip-address', type=str, required=False, default='')
    parser.add_argument('--serial-port', type=str, required=False, default='')
    parser.add_argument('--mac-address', type=str, required=False, default='')
    parser.add_argument('--other-info', type=str, required=False, default='')
    parser.add_argument('--streamer-params', type=str, required=False, default='')
    parser.add_argument('--serial-number', type=str, required=False, default='')
    parser.add_argument('--file', type=str, required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = '/dev/ttyUSB0'  # hard-coded for your setup
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
    print("Running real-time EEG + bandpower viewer. Ctrl+C to exit.\n")

    mindfulness_params = BrainFlowModelParams(
        BrainFlowMetrics.MINDFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()

    restfulness_params = BrainFlowModelParams(
        BrainFlowMetrics.RESTFULNESS.value,
        BrainFlowClassifiers.DEFAULT_CLASSIFIER.value
    )
    restfulness = MLModel(restfulness_params)
    restfulness.prepare()

    MINDFULNESS_THRESHOLD = 0.7
    RESTFULNESS_THRESHOLD = 0.7

    eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))

    window_sec = 4
    num_samples = int(window_sec * sampling_rate)

    # ---------- pyqtgraph setup ----------
    app = pg.mkQApp("EEG + Bandpower Viewer")
    win = pg.GraphicsLayoutWidget(title="EEG + Bandpower")
    win.resize(900, 700)

    # Top: EEG time series
    eeg_plot = win.addPlot(row=0, col=0, title="EEG Time Series (last window)")
    eeg_plot.setLabel('bottom', 'Time', units='s')
    eeg_plot.setLabel('left', 'Amplitude', units='µV')

    eeg_curves = []
    for i, ch in enumerate(eeg_channels):
        curve = eeg_plot.plot(pen=i, name=f"Ch {ch}")
        eeg_curves.append(curve)

    # Bottom: Bandpower time series
    band_plot = win.addPlot(row=1, col=0, title="Bandpower Time Series")
    band_plot.setLabel('bottom', 'Time', units='s')
    band_plot.setLabel('left', 'Power (a.u.)')

    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    band_curves = []
    for i, name in enumerate(band_names):
        curve = band_plot.plot(pen=i, name=name)
        band_curves.append(curve)

    band_times = []
    band_history = [[] for _ in band_names]
    t0 = time.time()

    win.show()

    try:
        while True:
            # keep Qt GUI responsive
            app.processEvents()

            # let the buffer fill enough for one window
            time.sleep(window_sec)

            # get the most recent window of data
            data = board.get_current_board_data(num_samples)

            # skip if not enough data yet
            if data.shape[1] < num_samples:
                print("Not enough data yet, waiting...")
                continue

            # ---- EEG time-series plot ----
            eeg_window = data[eeg_channels, :]  # shape: (n_channels, num_samples)
            t_eeg = np.linspace(-window_sec, 0, num_samples)
            for i, curve in enumerate(eeg_curves):
                if i < eeg_window.shape[0]:
                    curve.setData(t_eeg, eeg_window[i, :])

            y_min = float(np.min(eeg_window))
            y_max = float(np.max(eeg_window))
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0
            eeg_plot.setYRange(y_min, y_max)

            # ---- band powers ----
            bands = DataFilter.get_avg_band_powers(
                data, eeg_channels, sampling_rate, True
            )
            feature_vector = bands[0]  # delta, theta, alpha, beta, gamma

            t_now = time.time() - t0
            band_times.append(t_now)
            fv = np.asarray(feature_vector).flatten()
            for i in range(len(band_names)):
                if i < len(fv):
                    band_history[i].append(float(fv[i]))
                    band_curves[i].setData(band_times, band_history[i])

            # ---- metrics ----
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
            print()

    except KeyboardInterrupt:
        print("Stopping on user interrupt...")

    finally:
        board.stop_stream()
        board.release_session()
        mindfulness.release()
        restfulness.release()


if __name__ == "__main__":
    main()
