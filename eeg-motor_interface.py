import argparse
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import (
    MLModel,
    BrainFlowMetrics,
    BrainFlowClassifiers,
    BrainFlowModelParams,
)

import numpy as np
import pyqtgraph as pg


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

    # For the dummy board, we basically don't need any connection params
    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port  # ignored by synthetic board
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    # --- use BrainFlow's synthetic (dummy) board here ---
    board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
    master_board_id = board.get_board_id()
    sampling_rate = BoardShim.get_sampling_rate(master_board_id)

    board.prepare_session()
    board.start_stream(45000, args.streamer_params)
    BoardShim.log_message(LogLevels.LEVEL_INFO.value,
                          'Start continuous EEG metric loop (synthetic board)')
    print("Running real-time EEG + bandpower viewer with SYNTHETIC_BOARD. Ctrl+C to exit.\n")

    # --- ML models ---
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
    n_ch = len(eeg_channels)

    window_sec = 4.0
    num_samples = int(window_sec * sampling_rate)

    # ---------- pyqtgraph setup ----------
    app = pg.mkQApp("EEG + Bandpower Viewer")
    win = pg.GraphicsLayoutWidget(title="EEG + Bandpower (Synthetic Board)")
    win.resize(1000, 800)

    # === EEG plot (all channels stacked like OpenBCI) ===
    eeg_plot = win.addPlot(row=0, col=0, title="EEG Time Series (last window)")
    eeg_plot.setLabel('bottom', 'Time', units='s')
    eeg_plot.getAxis('left').setTicks([])  # no y ticks
    eeg_plot.showGrid(x=True, y=True, alpha=0.2)

    spacing = 100.0  # vertical spacing between channels

    eeg_curves = []
    channel_labels = []

    for i, ch in enumerate(eeg_channels):
        # distinct colors per channel
        pen = pg.intColor(i, hues=n_ch, maxValue=255)

        curve = eeg_plot.plot(pen=pen)
        eeg_curves.append(curve)

        label = pg.TextItem(f"Ch {ch}", color=pen, anchor=(1, 0.5))
        eeg_plot.addItem(label)
        channel_labels.append(label)

    eeg_plot.setYRange(-spacing, spacing * (n_ch + 1))

    # === Bandpower plot (bottom) ===
    band_plot = win.addPlot(row=1, col=0, title="Bandpower Time Series (last 4 s)")
    band_plot.setLabel('bottom', 'Time', units='s')
    band_plot.setLabel('left', 'Power', units='a.u.')
    band_plot.showGrid(x=True, y=True, alpha=0.2)

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
        pen = pg.intColor(i, hues=len(band_info), maxValue=255)
        curve = band_plot.plot(pen=pen, name=label)
        band_curves.append(curve)

    band_times = []
    band_history = [[] for _ in band_info]
    band_window_sec = 4.0
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

            # ========== EEG update ==========
            eeg_window = data[eeg_channels, :]  # shape: (n_ch, num_samples)
            t_eeg = np.linspace(-window_sec, 0, num_samples)

            max_abs = float(np.max(np.abs(eeg_window))) if eeg_window.size > 0 else 1.0
            if max_abs < 1e-6:
                max_abs = 1.0
            scale = 0.4 * spacing / max_abs  # scale to fit within strips

            for idx, curve in enumerate(eeg_curves):
                if idx < eeg_window.shape[0]:
                    sig = eeg_window[idx, :].astype(float)
                    sig = sig - np.mean(sig)
                    sig = sig * scale

                    offset = (n_ch - 1 - idx) * spacing
                    curve.setData(t_eeg, sig + offset)

                    channel_labels[idx].setPos(t_eeg[0], offset)

            eeg_plot.setXRange(-window_sec, 0.0, padding=0)

            # ========== Bandpower update ==========
            bands = DataFilter.get_avg_band_powers(
                data, eeg_channels, sampling_rate, True
            )
            feature_vector = bands[0]  # delta, theta, alpha, beta, gamma

            t_now = time.time() - t0
            band_times.append(t_now)

            fv = np.asarray(feature_vector).flatten()
            for i in range(len(band_info)):
                if i < len(fv):
                    band_history[i].append(float(fv[i]))
                    band_curves[i].setData(band_times, band_history[i])

            band_plot.setXRange(t_now - band_window_sec, t_now, padding=0)

            # ========== Metrics update ==========
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
