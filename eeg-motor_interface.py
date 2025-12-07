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
from pyqtgraph.Qt import QtCore, QtWidgets


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
    BoardShim.log_message(
        LogLevels.LEVEL_INFO.value,
        'Start continuous EEG metric loop (synthetic board)'
    )
    print("Running real-time EEG + bandpower viewer with SYNTHETIC_BOARD. "
          "Close the window or Ctrl+C in terminal to exit.\n")

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

    # Window length for analysis (seconds) and number of samples per window
    window_sec = 4.0
    num_samples = int(window_sec * sampling_rate)

    # ---------- pyqtgraph / Qt setup ----------
    app = pg.mkQApp("EEG + Bandpower Viewer")
    win = pg.GraphicsLayoutWidget(title="EEG + Bandpower (Synthetic Board)")
    win.resize(1000, 800)

    # === EEG plot (all channels stacked like OpenBCI) ===
    eeg_plot = win.addPlot(row=0, col=0, title="EEG Time Series (last window)")
    eeg_plot.setLabel('bottom', 'Time', units='s')
    eeg_plot.getAxis('left').setTicks([])       # no y ticks
    eeg_plot.getAxis('left').setStyle(showValues=False)
    eeg_plot.showGrid(x=True, y=True, alpha=0.2)

    spacing = 100.0  # vertical spacing between channels in display units

    eeg_curves = []
    channel_labels = []

    for i, ch in enumerate(eeg_channels):
        # distinct colors per channel
        pen = pg.intColor(i, hues=n_ch, maxValue=255)

        curve = eeg_plot.plot(pen=pen)
        eeg_curves.append(curve)

        label = pg.TextItem(f"Ch {ch}", color=pen, anchor=(0, 0.5))  # left-middle anchor
        eeg_plot.addItem(label)
        channel_labels.append(label)

    rms_labels = []
    for i in range(n_ch):
        rms_label = pg.TextItem(
            "",                       # text set in update()
            color='w',
            anchor=(1, 0.5),          # right-middle anchor
            fill=pg.mkBrush(0, 0, 0, 180)  # black-ish background for box effect
        )
        eeg_plot.addItem(rms_label)
        rms_labels.append(rms_label)

    eeg_plot.setYRange(-spacing, spacing * (n_ch + 1))

    # === Bandpower plot (bottom) ===
    band_plot = win.addPlot(row=1, col=0, title="Bandpower Time Series (last 4 s)")
    band_plot.setLabel('bottom', 'Time', units='s')
    band_plot.setLabel('left', 'Power', units='a.u.')
    band_plot.showGrid(x=True, y=True, alpha=0.2)

    # --- make left axes the same width so x-axes line up ---
    left_axis_width = 60  # tweak this until it looks nice
    eeg_plot.getAxis('left').setWidth(left_axis_width)
    band_plot.getAxis('left').setWidth(left_axis_width)

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

    # --- Metric window (separate) ---
    metrics_win = QtWidgets.QWidget()
    metrics_win.setWindowTitle("EEG Metrics")
    metrics_layout = QtWidgets.QHBoxLayout(metrics_win)

    # Left side: single indicator box + label
    indicator_layout = QtWidgets.QVBoxLayout()

    # colors for metrics
    mind_color = (0, 200, 255)   # cyan-ish
    rest_color = (0, 255, 0)     # green

    # Status box
    status_box = QtWidgets.QFrame()
    status_box.setFixedSize(40, 40)
    status_box.setStyleSheet("background-color: gray; border: 1px solid white;")

    # Status label
    status_label = QtWidgets.QLabel("Neither detected")
    status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    indicator_layout.addWidget(status_box, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
    indicator_layout.addWidget(status_label)

    metrics_layout.addLayout(indicator_layout)

    # Right side: metric plot
    metrics_plot = pg.PlotWidget(title="EEG Metrics (last 4 s)")
    metrics_plot.setLabel('bottom', 'Time', units='s')
    metrics_plot.setLabel('left', 'Metric', units='')
    metrics_plot.setYRange(0.0, 1.0)
    metrics_plot.setXRange(-4.0, 0.0, padding=0)
    metrics_plot.showGrid(x=True, y=True, alpha=0.2)

    # Curves for metrics
    mind_pen = pg.mkPen(color=mind_color, width=2)
    rest_pen = pg.mkPen(color=rest_color, width=2)

    mind_curve = metrics_plot.plot(pen=mind_pen, name="Mindfulness")
    rest_curve = metrics_plot.plot(pen=rest_pen, name="Restfulness")

    # Threshold lines (dotted)
    mind_thresh_line = pg.InfiniteLine(
        pos=MINDFULNESS_THRESHOLD,
        angle=0,
        pen=pg.mkPen(color=mind_color, style=QtCore.Qt.PenStyle.DotLine)
    )
    rest_thresh_line = pg.InfiniteLine(
        pos=RESTFULNESS_THRESHOLD,
        angle=0,
        pen=pg.mkPen(color=rest_color, style=QtCore.Qt.PenStyle.DotLine)
    )
    metrics_plot.addItem(mind_thresh_line)
    metrics_plot.addItem(rest_thresh_line)

    # History for metrics
    metric_times = []
    mind_history = []
    rest_history = []

    metrics_layout.addWidget(metrics_plot)
    metrics_win.resize(600, 300)
    metrics_win.show()

    # --- timing / update configuration ---
    update_speed_ms = 50          # how often to refresh plots (20 Hz)
    metrics_period = 1.0          # how often to recompute metrics/print
    last_metrics_time = time.time()

    win.show()

    # ---------- update function called by QTimer ----------
    def update():
        nonlocal last_metrics_time

        # get the most recent window of data
        data = board.get_current_board_data(num_samples)

        # skip if not enough data yet
        if data.shape[1] < num_samples:
            return

        # ========== EEG update (every timer tick) ==========
        eeg_window = data[eeg_channels, :]  # shape: (n_ch, num_samples)
        t_eeg = np.linspace(-window_sec, 0, num_samples)

        max_abs = float(np.max(np.abs(eeg_window))) if eeg_window.size > 0 else 1.0
        if max_abs < 1e-6:
            max_abs = 1.0
        # scale to fit within strip spacing
        scale = 0.4 * spacing / max_abs

        # Get current view range, so labels/boxes track the visible area
        vb = eeg_plot.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        x_left = x_min + 0.02 * (x_max - x_min)    # 2% in from left for channel labels
        x_right = x_max - 0.02 * (x_max - x_min)   # 2% in from right for RMS boxes

        for idx, curve in enumerate(eeg_curves):
            if idx < eeg_window.shape[0]:
                raw_chan = eeg_window[idx, :].astype(float)

                # draw signal (detrended + scaled + offset)
                sig = raw_chan - np.mean(raw_chan)
                sig = sig * scale

                offset = (n_ch - 1 - idx) * spacing
                curve.setData(t_eeg, sig + offset)

                # left-side channel name label
                channel_labels[idx].setPos(x_left, offset)

                # right-side RMS box (RMS in original units)
                rms_val = np.sqrt(np.mean(raw_chan ** 2))
                rms_labels[idx].setPos(x_right, offset)
                rms_labels[idx].setText(f"{rms_val:5.1f} µV")

        eeg_plot.setXRange(-window_sec, 0.0, padding=0)

        # ========== Bandpower update (EVERY frame) ==========
        now = time.time()
        bands = DataFilter.get_avg_band_powers(
            data, eeg_channels, sampling_rate, True
        )
        feature_vector = bands[0]  # delta, theta, alpha, beta, gamma

        t_now = now - t0
        band_times.append(t_now)
        fv = np.asarray(feature_vector).flatten()

        # relative times for bandpower plot (last 4 s shown)
        band_times_rel = [t - t_now for t in band_times]

        for i, curve in enumerate(band_curves):
            if i < len(fv):
                band_history[i].append(float(fv[i]))
                curve.setData(band_times_rel, band_history[i])

        band_plot.setXRange(-band_window_sec, 0.0, padding=0)

        # ========== Metrics update (slower rate) ==========
        if now - last_metrics_time < metrics_period:
            return

        last_metrics_time = now

        mindfulness_score = float(mindfulness.predict(feature_vector)[0])
        restfulness_score = float(restfulness.predict(feature_vector)[0])

        mindfulness_detected = mindfulness_score >= MINDFULNESS_THRESHOLD
        restfulness_detected = restfulness_score >= RESTFULNESS_THRESHOLD

        # --- update metric history for plotting ---
        metric_t_now = now - t0
        metric_times.append(metric_t_now)
        mind_history.append(mindfulness_score)
        rest_history.append(restfulness_score)

        metric_times_rel = [t - metric_t_now for t in metric_times]
        mind_curve.setData(metric_times_rel, mind_history)
        rest_curve.setData(metric_times_rel, rest_history)
        metrics_plot.setXRange(-4.0, 0.0, padding=0)

        # --- update single indicator box + label ---
        gray_style = "background-color: gray; border: 1px solid white;"
        mind_style = f"background-color: rgb({mind_color[0]},{mind_color[1]},{mind_color[2]}); border: 1px solid white;"
        rest_style = f"background-color: rgb({rest_color[0]},{rest_color[1]},{rest_color[2]}); border: 1px solid white;"

        if not mindfulness_detected and not restfulness_detected:
            status_box.setStyleSheet(gray_style)
            status_label.setText("Neither detected")
        else:
            # if both detected, choose whichever metric is stronger
            if mindfulness_detected and (not restfulness_detected or mindfulness_score >= restfulness_score):
                status_box.setStyleSheet(mind_style)
                status_label.setText("Mindfulness detected")
            else:
                status_box.setStyleSheet(rest_style)
                status_label.setText("Restfulness detected")

        # optional: still print to console
        print(
            f"Mindfulness: {mindfulness_score:.3f} "
            f"(detected={mindfulness_detected}), "
            f"Restfulness: {restfulness_score:.3f} "
            f"(detected={restfulness_detected})"
        )
        print()

    # ---------- QTimer driving the update loop ----------
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(update_speed_ms)

    try:
        QtWidgets.QApplication.instance().exec()
    finally:
        board.stop_stream()
        board.release_session()
        mindfulness.release()
        restfulness.release()


if __name__ == "__main__":
    main()
