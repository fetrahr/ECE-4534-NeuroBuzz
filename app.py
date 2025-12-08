import sys
import time
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QSpinBox,
    QPushButton, QCheckBox, QLineEdit, QDoubleSpinBox,
    QGroupBox, QFrame, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

# Local imports
from data_logger import DataLogger
from eeg_manager import EEGManager
from gui_widgets import MotorVisualizationWidget, GloveCanvas

# Mock Haptics if missing
try:
    from haptic_glove import HapticGlove, MAX_EFFECT
except ImportError:
    MAX_EFFECT = 123
    class HapticGlove:
        def __init__(self, channels): pass
        def play_effect(self, c, e): pass
        def play_all(self, e): pass
        def stop_all(self): pass

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
CHANGE_DICT = {"none": 0, "ascending": 1, "descending": -1}

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated EEG & Haptic Controller")
        self.resize(1300, 900)
        
        # --- Managers ---
        self.logger = DataLogger()
        self.eeg = EEGManager()
        self.glove = HapticGlove(channels=[0, 1, 2, 3, 4])
        
        # --- State ---
        self.logging_enabled = False
        self.fingers = [{"name": n, "effect_val": 1, "toggle_val": False} for n in FINGER_NAMES]
        self.shared_effect_val = 1
        self.delay_val = 2.0
        self.dwell_val = 0.2
        self.effect_mode = "none"
        self.eeg_scale_val = 0.1  # Sensitivity control
        
        # Automation State
        self.repeat_all = False
        self.last_all_time = None
        self.last_haptic_time = [None] * 5
        self.ripple_running = False
        self.repeat_ripple = False
        self.ripple_order = [0, 1, 2, 3, 4]
        self.ripple_index = 0
        
        # --- Init ---
        self.eeg.start_stream()
        self.init_ui()
        
        # --- Timers ---
        self.haptic_timer = QTimer(self)
        self.haptic_timer.timeout.connect(self.update_haptics)
        self.haptic_timer.start(25)

        self.eeg_timer = QTimer(self)
        self.eeg_timer.timeout.connect(self.update_eeg_visuals)
        self.eeg_timer.start(50)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # === LEFT PANE (Controls) ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        # 1. Config
        left_layout.addWidget(self.build_config_panel())
        
        # 2. Logging
        left_layout.addWidget(self.build_log_controls())
        
        # 3. Actions
        left_layout.addWidget(self.build_action_controls())
        
        # 4. Finger Matrix
        h_matrix = QHBoxLayout()
        h_matrix.addWidget(self.build_finger_matrix())
        
        self.glove_canvas = GloveCanvas("glove.png", self.play_finger_once)
        self.glove_canvas.setFixedSize(300, 250) 
        h_matrix.addWidget(self.glove_canvas)
        
        left_layout.addLayout(h_matrix)
        left_layout.addStretch()
        
        main_layout.addWidget(left_widget, 1)

        # === RIGHT PANE (Visuals) ===
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)
        
        # 1. EEG Plot Section
        right_layout.addWidget(QLabel("<b>EEG Time Series</b>"))
        # Note: We remove the 'title' arg from GraphicsLayoutWidget to use the QLabel instead
        self.eeg_widget = pg.GraphicsLayoutWidget()
        self.setup_eeg_plot()
        right_layout.addWidget(self.eeg_widget, 2)
        
        # 2. Bandpower Section
        right_layout.addWidget(QLabel("<b>Bandpower</b>"))
        self.band_widget = pg.GraphicsLayoutWidget()
        self.setup_band_plot()
        right_layout.addWidget(self.band_widget, 1)
        
        # 3. Motor Section
        # We wrap this in a frame to match styling if needed, but keep margins tight
        motor_frame = QFrame()
        motor_frame.setFrameShape(QFrame.NoFrame) 
        v_motor = QVBoxLayout(motor_frame)
        v_motor.setContentsMargins(0,0,0,0) # Tight margins
        v_motor.setSpacing(5)
        
        v_motor.addWidget(QLabel("<b>Haptic Timeline</b>"))
        self.motor_viz = MotorVisualizationWidget()
        
        # Try to align motor viz if it has a way to set margins (Depends on internal implementation)
        # If it is a pyqtgraph widget internally, you would set its left axis width here too.
        
        v_motor.addWidget(self.motor_viz)
        
        right_layout.addWidget(motor_frame, 1)
        
        main_layout.addLayout(right_layout, 2)
        
        # Link X-Axes for synchronization
        self.plot_bp.setXLink(self.plot_ts)

    # --- UI Builders ---
    def build_config_panel(self):
        grp = QGroupBox("Configuration")
        layout = QGridLayout(grp)
        
        layout.addWidget(QLabel("Delay (s):"), 0, 0)
        self.spin_delay = QDoubleSpinBox()
        self.spin_delay.setRange(0.01, 60); self.spin_delay.setValue(2.0)
        self.spin_delay.valueChanged.connect(lambda v: setattr(self, 'delay_val', v))
        layout.addWidget(self.spin_delay, 0, 1)

        layout.addWidget(QLabel("Dwell (s):"), 0, 2)
        self.spin_dwell = QDoubleSpinBox()
        self.spin_dwell.setRange(0.01, 60); self.spin_dwell.setValue(0.2)
        self.spin_dwell.valueChanged.connect(lambda v: setattr(self, 'dwell_val', v))
        layout.addWidget(self.spin_dwell, 0, 3)
        
        layout.addWidget(QLabel("uV Scale:"), 1, 0)
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.01, 1000.0); self.spin_scale.setValue(0.1); self.spin_scale.setSingleStep(0.01)
        self.spin_scale.valueChanged.connect(lambda v: setattr(self, 'eeg_scale_val', v))
        layout.addWidget(self.spin_scale, 1, 1)

        self.lbl_mode = QLabel("Mode: none")
        layout.addWidget(self.lbl_mode, 1, 2)
        btn_mode = QPushButton("Toggle Mode")
        btn_mode.clicked.connect(self.toggle_mode)
        layout.addWidget(btn_mode, 1, 3)

        btn_stop = QPushButton("STOP ALL MOTORS")
        btn_stop.setStyleSheet("background-color: #ffcccc; color: red; font-weight: bold;")
        btn_stop.clicked.connect(self.stop_all)
        layout.addWidget(btn_stop, 2, 0, 1, 4)
        return grp

    def build_action_controls(self):
        grp = QGroupBox("Hand Actions")
        layout = QVBoxLayout(grp)
        
        h1 = QHBoxLayout()
        btn_all = QPushButton("Play All")
        btn_all.clicked.connect(self.play_all_once)
        h1.addWidget(btn_all)
        self.chk_rep_all = QCheckBox("Repeat All")
        self.chk_rep_all.stateChanged.connect(lambda s: setattr(self, 'repeat_all', s == Qt.Checked))
        h1.addWidget(self.chk_rep_all)
        layout.addLayout(h1)
        
        h2 = QHBoxLayout()
        btn_rip = QPushButton("Play Ripple")
        btn_rip.clicked.connect(lambda: self.start_ripple(False))
        h2.addWidget(btn_rip)
        self.chk_rep_rip = QCheckBox("Repeat Ripple")
        self.chk_rep_rip.stateChanged.connect(self.toggle_repeat_ripple)
        h2.addWidget(self.chk_rep_rip)
        layout.addLayout(h2)
        
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Order:"))
        self.txt_order = QLineEdit("0,1,2,3,4")
        h3.addWidget(self.txt_order)
        btn_ord = QPushButton("Set")
        btn_ord.clicked.connect(self.update_order)
        h3.addWidget(btn_ord)
        layout.addLayout(h3)
        return grp

    def build_log_controls(self):
        grp = QGroupBox("Logging")
        layout = QHBoxLayout(grp)
        self.chk_log = QCheckBox("Record Data")
        self.chk_log.stateChanged.connect(self.toggle_logging)
        layout.addWidget(self.chk_log)
        self.lbl_log_status = QLabel("Status: OFF")
        self.lbl_log_status.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_log_status)
        return grp

    def build_finger_matrix(self):
        grp = QGroupBox("Fingers")
        layout = QGridLayout(grp)
        layout.setSpacing(5)
        
        layout.addWidget(QLabel("<b>Eff</b>"), 0, 1)
        layout.addWidget(QLabel("<b>Play</b>"), 0, 2)
        layout.addWidget(QLabel("<b>Tog</b>"), 0, 3)
        
        layout.addWidget(QLabel("<i>All</i>"), 1, 0)
        spin_shared = QSpinBox()
        spin_shared.setRange(1, MAX_EFFECT); spin_shared.setValue(1); spin_shared.setFixedWidth(50)
        spin_shared.valueChanged.connect(lambda v: setattr(self, 'shared_effect_val', v))
        layout.addWidget(spin_shared, 1, 1)
        btn_set = QPushButton("Set")
        btn_set.setFixedWidth(50)
        btn_set.clicked.connect(self.set_all_shared)
        layout.addWidget(btn_set, 1, 2)
        
        self.spinboxes = []
        self.checkboxes = []
        for i, f in enumerate(self.fingers):
            r = i + 2
            layout.addWidget(QLabel(f['name']), r, 0)
            s = QSpinBox()
            s.setRange(1, MAX_EFFECT); s.setValue(1); s.setFixedWidth(50)
            s.valueChanged.connect(lambda v, x=i: self.update_finger_eff(x, v))
            layout.addWidget(s, r, 1)
            self.spinboxes.append(s)
            
            b = QPushButton("Play")
            b.setFixedWidth(50)
            b.clicked.connect(lambda _, x=i: self.play_finger_once(x))
            layout.addWidget(b, r, 2)
            
            c = QCheckBox()
            c.stateChanged.connect(lambda s, x=i: self.update_finger_tog(x, s))
            layout.addWidget(c, r, 3)
            self.checkboxes.append(c)
        return grp

    # --- Plot Setup ---
    def setup_eeg_plot(self):
        self.plot_ts = self.eeg_widget.addPlot()
        self.plot_ts.showGrid(x=True, y=True, alpha=0.3)
        # self.plot_ts.setLabel('bottom', 'Time (s)') # Optional: hide bottom label to save space
        
        # === KEY FIX: FIXED WIDTH FOR ALIGNMENT ===
        # By setting width to 60px, we force the plot to start at the exact same pixel
        # regardless of whether the label is "100" or "10000"
        self.plot_ts.getAxis('left').setWidth(60) 
        
        # Remove ticks/values if you want it super clean, but Fixed Width is better for alignment
        # self.plot_ts.showAxis('left', False) 
        
        self.curves_ts = []
        self.labels_ts = []
        n_ch = len(self.eeg.eeg_channels)
        self.ts_spacing = 100.0
        
        for i, ch in enumerate(self.eeg.eeg_channels):
            pen = pg.intColor(i, n_ch, maxValue=255)
            c = self.plot_ts.plot(pen=pen)
            self.curves_ts.append(c)
            txt = pg.TextItem(f"Ch {ch}", color=pen, anchor=(0, 0.5))
            self.plot_ts.addItem(txt)
            self.labels_ts.append(txt)
            
        self.plot_ts.setYRange(-self.ts_spacing, self.ts_spacing * (n_ch + 1))
        # LOCK X AXIS:
        self.plot_ts.setXRange(-4, 0, padding=0)
        self.plot_ts.setMouseEnabled(y=False, x=False)

    def setup_band_plot(self):
        self.plot_bp = self.band_widget.addPlot()
        self.plot_bp.addLegend(offset=(30, 30))
        self.plot_bp.showGrid(x=True, y=True, alpha=0.3)
        # self.plot_bp.setLabel('left', 'Power')
        
        # === KEY FIX: MATCH WIDTH WITH EEG PLOT ===
        self.plot_bp.getAxis('left').setWidth(60)
        
        names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        self.curves_bp = []
        self.bp_history = [[] for _ in names]
        self.bp_times = []
        self.t0 = time.time()
        
        for i, name in enumerate(names):
            pen = pg.intColor(i, 5)
            c = self.plot_bp.plot(pen=pen, name=name)
            self.curves_bp.append(c)

    # --- Update Loops ---
    def update_haptics(self):
        now = time.time()
        if self.repeat_all:
            if self.last_all_time is None or (now - self.last_all_time) > self.delay_val:
                self.play_all_once()
                self.last_all_time = now
        else:
            self.last_all_time = None
        for i, f in enumerate(self.fingers):
            if f['toggle_val']:
                last = self.last_haptic_time[i]
                if last is None or (now - last) > self.delay_val:
                    self.play_finger_once(i)
                    self.last_haptic_time[i] = now
            else:
                self.last_haptic_time[i] = None

    def update_eeg_visuals(self):
        win_size = int(4.0 * self.eeg.sampling_rate)
        data = self.eeg.get_data(win_size)
        if data.shape[1] == 0: return
        
        # FIX: Clamp data to window size to prevent axis growth
        if data.shape[1] > win_size:
            data = data[:, -win_size:]

        # --- LOGGING: Raw Data ---
        # Note: log_raw handles deduplication internally
        if self.logging_enabled:
            self.logger.log_raw(data, self.eeg.ts_channel, self.eeg.eeg_channels)

        # Plot EEG
        t_axis = np.linspace(-4, 0, data.shape[1])
        n_ch = len(self.eeg.eeg_channels)
        vb = self.plot_ts.getViewBox()
        x_min, x_max = vb.viewRange()[0]
        x_lbl = x_min + 0.02 * (x_max - x_min)

        for i, curve in enumerate(self.curves_ts):
            chan_idx = self.eeg.eeg_channels[i]
            raw = data[chan_idx, :].astype(float)
            raw -= np.mean(raw)
            # Apply Manual Scale
            raw *= self.eeg_scale_val
            
            offset = (n_ch - 1 - i) * self.ts_spacing
            curve.setData(t_axis, raw + offset)
            self.labels_ts[i].setPos(x_lbl, offset)

        # Plot Bandpower
        bands = self.eeg.get_bands(data)
        feature_vector = bands[0]
        t_now = time.time() - self.t0
        self.bp_times.append(t_now)
        if len(self.bp_times) > 100:
            self.bp_times.pop(0)
            for h in self.bp_history: h.pop(0)
            
        for i, val in enumerate(feature_vector):
            self.bp_history[i].append(val)
            t_rel = [t - t_now for t in self.bp_times]
            self.curves_bp[i].setData(t_rel, self.bp_history[i])
        self.plot_bp.setXRange(-4, 0)

        # Metrics
        mind, rest = self.eeg.predict_metrics(feature_vector)
        
        # --- LOGGING: Features ---
        if self.logging_enabled:
            # Use the last timestamp from the window as the feature timestamp
            bf_ts = data[self.eeg.ts_channel, -1]
            self.logger.log_features(bf_ts, feature_vector, mind, rest, mind>0.7, rest>0.7)

    # --- Haptic Logic ---
    def play_finger_once(self, idx):
        eff = self.fingers[idx]['effect_val']
        self.glove.play_effect(idx, eff)
        self.motor_viz.visualize_effect(idx, eff)
        
        # --- LOGGING: Haptic Event ---
        if self.logging_enabled:
            self.logger.log_haptic(time.time(), "SingleFinger", f"Finger {idx}", eff, int(self.dwell_val*1000))
            
        self.apply_mode_change(idx)

    def play_all_once(self):
        effs = [f['effect_val'] for f in self.fingers]
        self.glove.play_all(effs)
        for i in range(5):
            self.motor_viz.visualize_effect(i, effs[i])
            self.apply_mode_change(i)
            
        # --- LOGGING: Haptic Event ---
        if self.logging_enabled:
            # We log one event for the group trigger
            self.logger.log_haptic(time.time(), "AllFingers", "Trig 0-4", effs[0], int(self.dwell_val*1000))

    def start_ripple(self, repeat):
        if self.ripple_running: return
        self.update_order()
        self.ripple_running = True
        self.repeat_ripple = repeat
        self.ripple_index = 0
        self.ripple_step()

    def ripple_step(self):
        if not self.ripple_running: return
        if self.ripple_index >= len(self.ripple_order):
            if self.repeat_ripple and self.chk_rep_rip.isChecked():
                self.ripple_index = 0
                QTimer.singleShot(max(1, int(self.delay_val*1000)), self.ripple_step)
                return
            else:
                self.ripple_running = False
                self.chk_rep_rip.setChecked(False)
                return
        midx = self.ripple_order[self.ripple_index]
        if 0 <= midx < 5:
            eff = self.fingers[midx]['effect_val']
            self.glove.play_effect(midx, eff)
            self.motor_viz.visualize_effect(midx, eff)
            self.apply_mode_change(midx)
            
            # --- LOGGING: Haptic Event ---
            if self.logging_enabled:
                self.logger.log_haptic(time.time(), "RippleStep", f"Finger {midx}", eff, int(self.dwell_val*1000))
                
        self.ripple_index += 1
        QTimer.singleShot(max(1, int(self.dwell_val*1000)), self.ripple_step)

    def apply_mode_change(self, idx):
        chg = CHANGE_DICT[self.effect_mode]
        if chg == 0: return
        val = self.fingers[idx]['effect_val'] + chg
        if val > MAX_EFFECT: val = 1
        if val < 1: val = MAX_EFFECT
        self.fingers[idx]['effect_val'] = val
        self.spinboxes[idx].setValue(val)

    def stop_all(self):
        self.glove.stop_all()
        self.repeat_all = False; self.chk_rep_all.setChecked(False)
        self.ripple_running = False; self.chk_rep_rip.setChecked(False)
        for c in self.checkboxes: c.setChecked(False)

    def update_order(self):
        try:
            txt = self.txt_order.text()
            nums = [int(x) for x in txt.split(",") if x.strip()]
            if sorted(nums) != [0,1,2,3,4]: raise ValueError
            self.ripple_order = nums
        except:
            QMessageBox.warning(self, "Invalid Order", "Must be 0-4 exactly once.")

    def toggle_repeat_ripple(self, state):
        self.repeat_ripple = (state == Qt.Checked)
        if self.repeat_ripple and not self.ripple_running:
            self.start_ripple(True)

    def toggle_logging(self, state):
        if state == Qt.Checked:
            # Start Logging with the channels from the EEG manager
            self.logger.start(self.eeg.eeg_channels)
            self.logging_enabled = True
            self.lbl_log_status.setText(f"REC: {os.path.basename(self.logger.session_dir)}")
            self.lbl_log_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.logging_enabled = False
            self.logger.stop()
            self.lbl_log_status.setText("Status: OFF")
            self.lbl_log_status.setStyleSheet("color: gray;")

    def update_finger_eff(self, i, v): self.fingers[i]['effect_val'] = v
    def update_finger_tog(self, i, s): self.fingers[i]['toggle_val'] = (s == Qt.Checked)
    def set_all_shared(self):
        for i, s in enumerate(self.spinboxes): s.setValue(self.shared_effect_val)
    def toggle_mode(self):
        modes = ["none", "ascending", "descending"]
        self.effect_mode = modes[(modes.index(self.effect_mode)+1)%3]
        self.lbl_mode.setText(f"Mode: {self.effect_mode}")

    def closeEvent(self, e):
        self.eeg.stop_stream()
        if self.logging_enabled: self.logger.stop()
        self.stop_all()
        super().closeEvent(e)

if __name__ == "__main__":
    if not os.path.exists("glove.png"):
        from PyQt5.QtGui import QImage
        img = QImage(300, 200, QImage.Format_RGB32); img.fill(Qt.white); img.save("glove.png")
    
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=False) 
    win = MainApp()
    win.show()
    sys.exit(app.exec_())