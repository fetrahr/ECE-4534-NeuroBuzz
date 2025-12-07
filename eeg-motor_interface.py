import argparse
import time
import sys

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

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

    # ---------- pyqtgraph + Qt setup ----------
    app = pg.mkQApp("EEG + Bandpower Viewer")

    # Main window widget and layout
    main_win = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout(main_win)

    # --- Scroll area for EEG channel plots ---
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)

    eeg_container = QtWidgets.QWidget()
    eeg_layout = QtWidgets.QVBoxLayout(eeg_container)

    eeg_plots = []
    eeg_curves = []

    # One PlotWidget per EEG channel, stacked vertically
    for i, ch in enumerate(eeg_channels):
        pw = pg.PlotWidget()
        pw.setMinimumHeight(120)  # so ~4 fit on screen
        pw.setLabel('left', f"Ch {ch}", units='µV')
        pw.setLabel('bottom', 'Time', units='s')  # label on EVERY plot

        # Link x-axes so zoom/pan stay aligned
        if i > 0:
            pw.setXLink(eeg_plots[0])

        curve = pw.plot()
        eeg_plots.append(pw)
        eeg_curves.append(curve)
        eeg_layout.addWidget(pw)

    eeg_container.setLayout(eeg_layout)
    scroll.setWidget(eeg_container)

    # Add scroll area to main layout
    main_layout.addWidget(scroll, stretch=3)

    # --- Bandpower plot at the bottom (last 4 seconds only) ---
    band_plot = pg.PlotWidget(title="Bandpower Time Series (last 4 s)")
    band_plot.setLabel('bottom', 'Time', units='s')
    band_plot.setLabel('left', 'Power', units='a.u.')

    # Band names + frequency ranges (typical BrainFlow defaults)
    band_info = [
        ("Delta", "1–4 Hz"),
        ("Theta", "4–8 Hz"),
        ("Alpha", "8–13 Hz"),
        ("Beta",  "13–30 Hz"),
        ("Gamma", "30–45 Hz"),
    ]

    legend = band_plot.addLegend()

    band_curves = []
    for i, (name, freq_range) in enumerate(band_info):
        label = f"{name} ({freq_range})"
        curve = band_plot.plot(pen=i, name=label)
        band_curves.append(curve)

    band_times = []
    band_history = [[] for _ in band_info]
    band_window_sec = 4.0  # show last 4 seconds
    t0 = time.time()

    main_layout.addWidget(band_plot, stretch=1)

    main_win.setLayout(main_layout)
    main_win.resize(900, 700)
    main_win.show()


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

            # ---- EEG: per-channel plots ----
            eeg_window = data[eeg_channels, :]  # shape: (n_channels, num_samples)
            t_eeg = np.linspace(-window_sec, 0, num_samples)

            for ch_idx, curve in enumerate(eeg_curves):
                if ch_idx < eeg_window.shape[0]:
                    chan_data = eeg_window[ch_idx, :]
                    curve.setData(t_eeg, chan_data)

                    # auto-scale Y for each channel separately
                    y_min = float(np.min(chan_data))
                    y_max = float(np.max(chan_data))
                    if y_min == y_max:
                        y_min -= 1.0
                        y_max += 1.0
                    eeg_plots[ch_idx].setYRange(y_min, y_max)

            # ---- band powers (delta..gamma) ----
            bands = DataFilter.get_avg_band_powers(
                data, eeg_channels, sampling_rate, True
            )
            feature_vector = bands[0]  # delta, theta, alpha, beta, gamma

            # ---- update bandpower time-series (last 4 seconds only) ----
            t_now = time.time() - t0
            band_times.append(t_now)
            fv = np.asarray(feature_vector).flatten()
            for i in range(len(band_info)):
                if i < len(fv):
                    band_history[i].append(float(fv[i]))

            # Trim history to last 4 seconds
            t_min = t_now - band_window_sec
            # Drop from the front while too old
            while band_times and band_times[0] < t_min:
                band_times.pop(0)
                for hist in band_history:
                    if hist:
                        hist.pop(0)

            # Update curves with trimmed data
            for i, curve in enumerate(band_curves):
                curve.setData(band_times, band_history[i])

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
