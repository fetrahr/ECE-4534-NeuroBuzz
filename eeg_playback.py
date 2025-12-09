import sys
import csv
import os
import argparse
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


WINDOW_SEC = 4.0      # length of visible window (seconds)
UPDATE_DT = 0.05      # playback time step (seconds)


class MotorPlaybackWidget(QtWidgets.QWidget):
    """
    Playback version of MotorVisualizationWidget.

    - `events`: list of (t_rel, motor_idx, effect_id), where t_rel is seconds
      relative to global t0.
    - Shows last `plot_duration_s` seconds in a scrolling fashion (right = now).
    """

    def __init__(self, motors, events, plot_duration_s=4.0, parent=None):
        super().__init__(parent)
        self.motors = motors
        self.events = events  # list[(t_rel, motor_idx, effect_id)]
        self.plot_duration_s = plot_duration_s

        self.line_height = 25
        self.motor_margin = 10
        self.current_time = 0.0

        self.setMinimumHeight(
            len(motors) * (self.line_height + self.motor_margin) + 20
        )

    def set_current_time(self, t):
        """Set current playback time (seconds, relative to t0)."""
        self.current_time = t
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        W = self.width()
        H = self.height()
        current_time = self.current_time

        label_width = 30
        plot_area_width = max(1, W - label_width)
        scale_factor = plot_area_width / float(self.plot_duration_s)

        # --- Draw motor lines and labels ---
        for i, name in enumerate(self.motors):
            y_center = (i * (self.line_height + self.motor_margin)) + self.line_height

            # Motor name label (use first char or full name)
            painter.setFont(QtGui.QFont('Arial', 9))
            painter.setPen(QtCore.Qt.black)
            painter.drawText(
                QtCore.QRect(0, y_center - self.line_height // 2, label_width, self.line_height),
                QtCore.Qt.AlignCenter,
                str(name)[0]
            )

            # Horizontal baseline
            painter.setPen(QtGui.QPen(QtCore.Qt.gray, 2))
            painter.drawLine(label_width, y_center, W, y_center)

            # Vertical grid markers every 1 second (optional)
            painter.setPen(QtGui.QPen(QtCore.Qt.lightGray, 1, QtCore.Qt.DashLine))
            for s in range(0, int(self.plot_duration_s) + 1):
                x_pos = W - (s * scale_factor)
                if x_pos > label_width:
                    painter.drawLine(int(x_pos), 0, int(x_pos), H)

        # --- Draw events in the last plot_duration_s seconds ---
        effect_duration_s = 0.2

        # Filter events that are within the visible window
        visible_events = [
            e for e in self.events
            if 0.0 <= current_time - e[0] <= self.plot_duration_s
        ]

        for t_event, motor_idx, effect_id in visible_events:
            if motor_idx < 0 or motor_idx >= len(self.motors):
                continue

            motor_line_y = (motor_idx * (self.line_height + self.motor_margin)) + self.line_height

            time_diff = current_time - t_event
            x_end = W - (time_diff * scale_factor)
            x_start = x_end - (effect_duration_s * scale_factor)

            if x_end < label_width:
                continue

            indent_box = QtCore.QRect(
                int(x_start),
                motor_line_y - self.line_height // 2,
                int(x_end - x_start),
                self.line_height
            )

            painter.fillRect(indent_box, QtGui.QColor('#3f88c5'))

            painter.setPen(QtCore.Qt.white)
            painter.setFont(QtGui.QFont('Arial', 8, QtGui.QFont.Bold))
            painter.drawText(indent_box, QtCore.Qt.AlignmentFlag.AlignCenter, str(effect_id))

        painter.end()


def load_eeg_raw_csv(path):
    """
    Expected format:
      brainflow_ts_us,ch_0,ch_1,...,ch_N

    Returns:
      ts_rel (N,), eeg_data (N, C), channel_ids (list[int])
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        if len(header) < 2 or header[0] != "brainflow_ts":
            # but user said "UNIX timestamps in microseconds", so we treat as microseconds column anyway
            # first column is timestamp in microseconds
            pass

        # channel labels are everything after timestamp
        channel_labels = header[1:]
        channel_ids = []
        for lbl in channel_labels:
            if lbl.startswith("ch_"):
                try:
                    channel_ids.append(int(lbl[3:]))
                except ValueError:
                    channel_ids.append(lbl)
            else:
                channel_ids.append(lbl)

        ts_list = []
        rows = []
        for row in reader:
            if not row:
                continue
            ts_us = float(row[0])
            ts_list.append(ts_us / 1e6)  # convert µs -> seconds
            rows.append([float(x) for x in row[1:]])

    ts = np.array(ts_list, dtype=float)
    data = np.array(rows, dtype=float)  # shape (N, C)
    return ts, data, channel_ids


def load_features_csv(path):
    """
    Expected header:
      brainflow_ts,delta,theta,alpha,beta,gamma,
      mindfulness_score,restfulness_score,
      mindfulness_detected,restfulness_detected

    All timestamps in microseconds.

    Returns:
      ts_rel (F,),
      bands (F,5),
      mind_score (F,),
      rest_score (F,),
      mind_det (F,), rest_det (F,)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        # simple index-based parse based on the format we defined earlier
        # if you changed the header, adjust here
        ts_idx = 0
        delta_idx = 1
        theta_idx = 2
        alpha_idx = 3
        beta_idx = 4
        gamma_idx = 5
        mind_score_idx = 6
        rest_score_idx = 7
        mind_det_idx = 8
        rest_det_idx = 9

        ts_list = []
        bands_list = []
        mind_scores = []
        rest_scores = []
        mind_dets = []
        rest_dets = []

        for row in reader:
            if not row:
                continue
            ts_us = float(row[ts_idx])
            ts_list.append(ts_us / 1e6)

            bands_list.append([
                float(row[delta_idx]),
                float(row[theta_idx]),
                float(row[alpha_idx]),
                float(row[beta_idx]),
                float(row[gamma_idx]),
            ])
            mind_scores.append(float(row[mind_score_idx]))
            rest_scores.append(float(row[rest_score_idx]))
            mind_dets.append(int(row[mind_det_idx]))
            rest_dets.append(int(row[rest_det_idx]))

    ts = np.array(ts_list, dtype=float)
    bands = np.array(bands_list, dtype=float)
    mind_scores = np.array(mind_scores, dtype=float)
    rest_scores = np.array(rest_scores, dtype=float)
    mind_dets = np.array(mind_dets, dtype=int)
    rest_dets = np.array(rest_dets, dtype=int)

    return ts, bands, mind_scores, rest_scores, mind_dets, rest_dets


def load_motor_csv(path):
    """
    Expected format (from logging formatter `'%(asctime)s,%(message)s'`,
    with asctime changed to unix timestamp in µs):

      ts_us,action,motors_str,effects_str

    motors_str: "0;1;2"
    effects_str: "10;10;10"

    Returns:
      ts_rel (M,), events (list[(t_rel, motor_idx, effect_id)]), num_motors
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    events_raw = []  # (ts_sec, motor_idx, effect_id)

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) < 4:
                # unexpected; skip
                continue

            ts_us = float(row[0])
            ts_sec = ts_us / 1e6

            # action = row[1]  # e.g. "finger_once", "all_once", etc. (we don't need it here)
            motors_str = row[2]
            effects_str = row[3]

            motor_indices = [int(x) for x in motors_str.split(";") if x.strip() != ""]
            effect_ids = [int(x) for x in effects_str.split(";") if x.strip() != ""]

            for m_idx, eff in zip(motor_indices, effect_ids):
                events_raw.append((ts_sec, m_idx, eff))

    if not events_raw:
        return np.array([]), [], 0

    # Use all timestamps to get relative times later
    ts_all = np.array([e[0] for e in events_raw], dtype=float)
    # We'll shift by global t0 in the main loader; for now we return absolute seconds
    return ts_all, events_raw


class PlaybackWindow(QtWidgets.QMainWindow):
    def __init__(self, raw_path, feat_path, motor_path):
        super().__init__()
        self.setWindowTitle("EEG + Bandpower + Metrics + Motor Playback")

        # --- Load data from CSVs ---
        raw_ts_abs, raw_data, channel_ids = load_eeg_raw_csv(raw_path)
        feat_ts_abs, bands, mind_scores, rest_scores, mind_dets, rest_dets = load_features_csv(feat_path)
        motor_ts_abs, motor_events_raw = load_motor_csv(motor_path)

        # Determine global t0 from all available streams
        times_for_t0 = []
        if raw_ts_abs.size > 0:
            times_for_t0.append(raw_ts_abs[0])
        if feat_ts_abs.size > 0:
            times_for_t0.append(feat_ts_abs[0])
        if motor_ts_abs.size > 0:
            times_for_t0.append(motor_ts_abs[0])

        if not times_for_t0:
            raise RuntimeError("No timestamps found in any CSV.")

        t0 = min(times_for_t0)

        # Convert to relative times (seconds from t0)
        self.raw_ts = raw_ts_abs - t0
        self.raw_data = raw_data   # shape (N_samples, N_channels)
        self.channel_ids = channel_ids

        self.feat_ts = feat_ts_abs - t0
        self.bands = bands         # (F, 5)
        self.mind_scores = mind_scores
        self.rest_scores = rest_scores
        self.mind_dets = mind_dets
        self.rest_dets = rest_dets

        # Motor events list: convert to (t_rel, motor_idx, effect_id)
        self.motor_events = []
        self.num_motors = 0
        if motor_events_raw:
            self.motor_events = [(e[0] - t0, e[1], e[2]) for e in motor_events_raw]
            self.num_motors = max(e[1] for e in motor_events_raw) + 1
        else:
            self.num_motors = 0

        # Playback time range
        self.playback_time = 0.0
        # end time: max of all streams
        t_end_candidates = []
        if self.raw_ts.size > 0:
            t_end_candidates.append(self.raw_ts[-1])
        if self.feat_ts.size > 0:
            t_end_candidates.append(self.feat_ts[-1])
        if self.motor_events:
            t_end_candidates.append(max(e[0] for e in self.motor_events))
        self.t_end = max(t_end_candidates) if t_end_candidates else 0.0

        # --- Build UI ---
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vlayout = QtWidgets.QVBoxLayout(central)
        vlayout.setContentsMargins(10, 10, 10, 10)
        vlayout.setSpacing(8)

        # EEG stacked plot
        self.eeg_plot = pg.PlotWidget(title="EEG Time Series (Playback)")
        self.eeg_plot.setLabel('bottom', 'Time', units='s')
        self.eeg_plot.getAxis('left').setTicks([])
        self.eeg_plot.getAxis('left').setStyle(showValues=False)
        self.eeg_plot.showGrid(x=True, y=True, alpha=0.2)

        self.eeg_spacing = 100.0
        n_ch = self.raw_data.shape[1] if self.raw_data.size > 0 else 0
        self.eeg_curves = []
        self.eeg_labels = []
        self.eeg_rms_labels = []

        for i in range(n_ch):
            pen = pg.intColor(i, hues=max(1, n_ch), maxValue=255)
            curve = self.eeg_plot.plot(pen=pen)
            self.eeg_curves.append(curve)

            label = pg.TextItem(f"Ch {self.channel_ids[i]}", color=pen, anchor=(0, 0.5))
            self.eeg_plot.addItem(label)
            self.eeg_labels.append(label)

            rms_label = pg.TextItem(
                "",
                color="w",
                anchor=(1, 0.5),
                fill=pg.mkBrush(0, 0, 0, 180),
            )
            self.eeg_plot.addItem(rms_label)
            self.eeg_rms_labels.append(rms_label)

        if n_ch > 0:
            self.eeg_plot.setYRange(-self.eeg_spacing, self.eeg_spacing * (n_ch + 1))

        vlayout.addWidget(self.eeg_plot, stretch=3)

        # Bandpower plot
        self.band_plot = pg.PlotWidget(title="Bandpower (Playback, last 4 s)")
        self.band_plot.setLabel('bottom', 'Time', units='s')
        self.band_plot.setLabel('left', 'Power', units='a.u.')
        self.band_plot.showGrid(x=True, y=True, alpha=0.2)

        band_info = [
            ("Delta", "1–4 Hz"),
            ("Theta", "4–8 Hz"),
            ("Alpha", "8–13 Hz"),
            ("Beta", "13–30 Hz"),
            ("Gamma", "30–45 Hz"),
        ]
        self.band_curves = []
        legend = self.band_plot.addLegend()
        for i, (name, freq_range) in enumerate(band_info):
            label = f"{name} ({freq_range})"
            pen = pg.intColor(i, hues=len(band_info), maxValue=255)
            curve = self.band_plot.plot(pen=pen, name=label)
            self.band_curves.append(curve)
        vlayout.addWidget(self.band_plot, stretch=2)

        # Metrics plot (Mindfulness & Restfulness)
        self.metrics_plot = pg.PlotWidget(title="EEG Metrics (Playback, last 4 s)")
        self.metrics_plot.setLabel('bottom', 'Time', units='s')
        self.metrics_plot.setLabel('left', 'Metric', units='')
        self.metrics_plot.setYRange(0.0, 1.0)
        self.metrics_plot.showGrid(x=True, y=True, alpha=0.2)

        mind_color = (0, 200, 255)
        rest_color = (0, 255, 0)
        self.mind_curve = self.metrics_plot.plot(
            pen=pg.mkPen(color=mind_color, width=2), name="Mindfulness"
        )
        self.rest_curve = self.metrics_plot.plot(
            pen=pg.mkPen(color=rest_color, width=2), name="Restfulness"
        )

        vlayout.addWidget(self.metrics_plot, stretch=2)

        # Motor playback widget
        if self.num_motors > 0:
            motor_names = [str(i) for i in range(self.num_motors)]
            self.motor_widget = MotorPlaybackWidget(
                motors=motor_names,
                events=self.motor_events,
                plot_duration_s=WINDOW_SEC,
            )
        else:
            self.motor_widget = MotorPlaybackWidget(
                motors=["0"], events=[], plot_duration_s=WINDOW_SEC
            )

        vlayout.addWidget(self.motor_widget, stretch=2)

        # Align left axis widths of EEG and band plots
        left_axis_width = 60
        self.eeg_plot.getAxis('left').setWidth(left_axis_width)
        self.band_plot.getAxis('left').setWidth(left_axis_width)

        # Timer for playback
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(int(UPDATE_DT * 1000))

    def on_timer(self):
        # Stop when we pass beyond the last timestamp
        if self.playback_time > self.t_end + WINDOW_SEC:
            self.timer.stop()
            return

        t = self.playback_time
        self.playback_time += UPDATE_DT

        t_start = t - WINDOW_SEC

        # --- EEG playback ---
        if self.raw_ts.size > 0:
            mask = (self.raw_ts >= t_start) & (self.raw_ts <= t)
            if np.any(mask):
                ts_seg = self.raw_ts[mask] - t  # shift to [-WINDOW_SEC, 0]
                eeg_seg = self.raw_data[mask, :]  # (M, n_ch)
                n_ch = eeg_seg.shape[1]

                max_abs = np.max(np.abs(eeg_seg)) if eeg_seg.size > 0 else 1.0
                if max_abs < 1e-6:
                    max_abs = 1.0
                scale = 0.4 * self.eeg_spacing / max_abs

                # current view box X range for label placement
                self.eeg_plot.setXRange(-WINDOW_SEC, 0.0, padding=0)
                vb = self.eeg_plot.getViewBox()
                x_min, x_max = vb.viewRange()[0]
                x_left = x_min + 0.02 * (x_max - x_min)
                x_right = x_max - 0.02 * (x_max - x_min)

                for idx, curve in enumerate(self.eeg_curves):
                    if idx >= n_ch:
                        break
                    raw_chan = eeg_seg[:, idx]
                    sig = raw_chan - np.mean(raw_chan)
                    sig = sig * scale

                    offset = (n_ch - 1 - idx) * self.eeg_spacing
                    curve.setData(ts_seg, sig + offset)

                    # Channel name label
                    self.eeg_labels[idx].setPos(x_left, offset)

                    # RMS label
                    rms_val = np.sqrt(np.mean(raw_chan ** 2))
                    self.eeg_rms_labels[idx].setPos(x_right, offset)
                    self.eeg_rms_labels[idx].setText(f"{rms_val:5.1f} µV")
            else:
                # clear if no data in this window
                for curve in self.eeg_curves:
                    curve.clear()

        # --- Bandpower playback ---
        if self.feat_ts.size > 0:
            mask_f = (self.feat_ts >= t_start) & (self.feat_ts <= t)
            if np.any(mask_f):
                ts_feat = self.feat_ts[mask_f] - t  # [-WINDOW_SEC, 0]
                bands_seg = self.bands[mask_f, :]   # (K, 5)
                for i, curve in enumerate(self.band_curves):
                    if i < bands_seg.shape[1]:
                        curve.setData(ts_feat, bands_seg[:, i])
            else:
                for curve in self.band_curves:
                    curve.clear()
            self.band_plot.setXRange(-WINDOW_SEC, 0.0, padding=0)

        # --- Metrics playback ---
        if self.feat_ts.size > 0:
            mask_f = (self.feat_ts >= t_start) & (self.feat_ts <= t)
            if np.any(mask_f):
                ts_feat = self.feat_ts[mask_f] - t
                mind_seg = self.mind_scores[mask_f]
                rest_seg = self.rest_scores[mask_f]
                self.mind_curve.setData(ts_feat, mind_seg)
                self.rest_curve.setData(ts_feat, rest_seg)
            else:
                self.mind_curve.clear()
                self.rest_curve.clear()
            self.metrics_plot.setXRange(-WINDOW_SEC, 0.0, padding=0)

        # --- Motor playback ---
        self.motor_widget.set_current_time(t)


def main():
    parser = argparse.ArgumentParser(
        description="Playback EEG + Bandpower + Metrics + Motor CSV logs."
    )
    parser.add_argument("raw_csv", help="Path to raw EEG CSV (eeg_session_raw_*.csv)")
    parser.add_argument("features_csv", help="Path to features CSV (eeg_session_features_*.csv)")
    parser.add_argument("motor_csv", help="Path to motor log CSV from glove app")
    args = parser.parse_args()

    app = pg.mkQApp("EEG/Motor Playback")

    win = PlaybackWindow(args.raw_csv, args.features_csv, args.motor_csv)
    win.resize(1000, 900)
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
