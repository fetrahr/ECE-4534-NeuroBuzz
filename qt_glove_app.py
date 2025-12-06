import sys
import time
import os
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QSpinBox,
    QPushButton, QCheckBox, QLineEdit, QDoubleSpinBox,
    QGroupBox, QFrame, QMessageBox
)
from PyQt5.QtGui import QPixmap, QFont, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer, QLocale, QRect

# Import the existing HapticGlove class and constants
from haptic_glove import HapticGlove, MAX_EFFECT
from glove_app import FINGER_NAMES, FINGER_TO_MOTOR, FINGERTIP_COORDS, CHANGE_DICT

# --- Configuration for Qt Conversion ---
GLOVE_IMAGE_PATH = "glove.png" 

class MotorVisualizationWidget(QWidget):
    def __init__(self, motors=FINGER_NAMES, plot_duration_s=10):
        super().__init__()
        self.motors = motors
        self.plot_duration_s = plot_duration_s
        self.timeline = []  # Stores: (timestamp, motor_idx, effect_id)
        
        # Visualization constants
        self.line_height = 25
        self.motor_margin = 10
        self.start_time = time.time()
        
        # Set minimum height based on 5 motors + labels
        self.setMinimumHeight(len(motors) * (self.line_height + self.motor_margin) + 20)
        
        # Timer for redrawing the plot (e.g., 20 times per second)
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update) # Calls paintEvent
        self.plot_timer.start(50) 
        
    def visualize_effect(self, motor_idx, effect_id):
        """Records the effect event in the timeline."""
        # Record the time and details of the event
        event = (time.time(), motor_idx, effect_id)
        self.timeline.append(event)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get dimensions
        W = self.width()
        H = self.height()
        current_time = time.time()
        
        # --- Drawing Setup ---
        
        # Text/Label Space (Left margin for motor names)
        label_width = 30
        plot_area_width = W - label_width
        
        # Scale: pixels per second
        scale_factor = plot_area_width / self.plot_duration_s
        
        # --- Draw Motor Lines and Labels ---
        for i, name in enumerate(self.motors):
            y_center = (i * (self.line_height + self.motor_margin)) + self.line_height
            
            # 1. Draw Motor Name Label
            painter.setFont(QFont('Arial', 9))
            painter.setPen(Qt.black)
            painter.drawText(QRect(0, y_center - self.line_height // 2, label_width, self.line_height), 
                             Qt.AlignCenter, name[0])
            
            # 2. Draw Horizontal Timeline (Flat line)
            painter.setPen(QPen(Qt.gray, 2))
            painter.drawLine(label_width, y_center, W, y_center)
            
            # 3. Draw Axis Markers (e.g., every 2 seconds)
            painter.setPen(QPen(Qt.lightGray, 1, Qt.DashLine))
            for s in range(0, self.plot_duration_s, 2):
                x_pos = W - (s * scale_factor)
                if x_pos > label_width:
                    painter.drawLine(int(x_pos), 0, int(x_pos), H)
            
        # --- Process and Draw Events ---
        
        # Remove old events from the timeline
        self.timeline = [e for e in self.timeline if e[0] > (current_time - self.plot_duration_s)]
        
        # Draw all remaining events
        effect_duration_s = 0.2  # Duration the visual indent is active
        
        for t, motor_idx, effect_id in self.timeline:
            motor_line_y = (motor_idx * (self.line_height + self.motor_margin)) + self.line_height
            
            # Calculate event position: (Current time - Event time) * Scale = Pixels from Right edge
            time_difference = current_time - t
            x_end = W - (time_difference * scale_factor)
            x_start = x_end - (effect_duration_s * scale_factor)
            
            # Constrain to plot area
            if x_end < label_width:
                continue
            
            # Drawing the indent (vertical line/box)
            indent_box = QRect(int(x_start), motor_line_y - self.line_height // 2, 
                               int(x_end - x_start), self.line_height)
            
            # Fill the box
            painter.fillRect(indent_box, QColor('#3f88c5'))
            
            # Draw Effect ID Text
            painter.setPen(Qt.white)
            painter.setFont(QFont('Arial', 8, QFont.Bold))
            painter.drawText(indent_box, Qt.AlignCenter, str(effect_id))
            
        painter.end()

# --- Qt GUI Application Class ---

class QtHapticGloveGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Haptic Glove Controller (Qt)")
        
        self.channels = [0, 1, 2, 3, 4]
        self.glove = HapticGlove(channels=self.channels)

        # --- State Variables ---
        self.fingers = []
        self.finger_last_time = [None] * 5

        for name in FINGER_NAMES:
            self.fingers.append({
                "name": name,
                "effect_val": 1,
                "toggle_val": False,
            })
        
        self.shared_effect_val = 1
        self.write_timestamps_val = False
        self.repeat_all_val = False
        self.repeat_ripple_val = False

        self.delay_time_val = 2.0
        self.dwell_time_val = 0.2

        self.order = [0, 1, 2, 3, 4]
        self.order_str_val = "0,1,2,3,4"

        self.effect_mode_val = "none" # none / ascending / descending
        self.all_last_time = None

        self.ripple_running = False
        self.ripple_repeat = False
        self.ripple_index = 0

        self.log_name_val = "session"
        self.logger = None

        self.init_ui()
        
        # Setup periodic update timer (mimics periodic_update() in glove_app.py)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.periodic_update)
        self.timer.start(25) 
        
    # --- Logging Setup ---

    def init_logger(self):
        dirname = "haptic_data"
        os.makedirs(dirname, exist_ok=True)

        name = self.log_name_val.strip() or "session"
        path = os.path.join(dirname, f"{name}.csv")

        logger = logging.getLogger("haptics")

        if logger.handlers:
            logger.handlers.clear()

        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(path, mode="a")
        formatter = logging.Formatter("%(asctime)s,%(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        self.logger = logger
        
    def log_event(self, action, motors, effects):
        if not self.write_timestamps_val:
            return

        motors_str = ";".join(str(m) for m in motors)
        effects_str = ";".join(str(e) for e in effects)
        self.logger.info(f"{action},{motors_str},{effects_str}")
    
    def write_timestamps_toggled(self, state):
        self.write_timestamps_val = state == Qt.Checked
        if self.write_timestamps_val:
            self.init_logger()

    # --- UI Initialization ---

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout: Left Pane | Right Pane
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # --- Left Pane (VBox for ALL Controls and Hand) ---
        left_vbox = QVBoxLayout()
        left_vbox.setSpacing(10)
        
        # 1. Top Controls (ALL Logic)
        top_controls_frame = QFrame()
        top_controls_frame.setFrameShape(QFrame.StyledPanel)
        top_controls_vbox = QVBoxLayout(top_controls_frame)
        self.build_full_top_controls(top_controls_vbox) # Now uses the full controls builder
        left_vbox.addWidget(top_controls_frame)

        # 2. Controls & Glove (HBox)
        h_controls_glove = QHBoxLayout()
        
        # 2a. Individual Finger Controls (Left of Glove)
        finger_group = QGroupBox("Finger Effect Control")
        finger_layout = QGridLayout(finger_group)
        self.build_finger_controls(finger_layout)
        h_controls_glove.addWidget(finger_group)

        # 2b. Glove Canvas + Buttons (Right of Individual Controls)
        self.canvas_widget = QWidget()
        self.build_glove_canvas(self.canvas_widget)
        h_controls_glove.addWidget(self.canvas_widget, 0)

        left_vbox.addLayout(h_controls_glove)
        left_vbox.addStretch(1)

        main_layout.addLayout(left_vbox)

        # --- Right Pane (Dedicated Visualization Plot) ---
        right_vbox = QVBoxLayout()
        right_vbox.setSpacing(10)
        
        reserved_frame = QFrame()
        reserved_frame.setFrameShape(QFrame.Box)
        reserved_frame.setFrameShadow(QFrame.Sunken)
        reserved_vbox = QVBoxLayout(reserved_frame)
        
        # Motor Visualization Widget (New: Full width and scrolling)
        self.motor_viz = MotorVisualizationWidget()
        reserved_vbox.addWidget(self.motor_viz)
        
        # EEG Reserved Space (Placeholder Label, placed below the plot)
        reserved_label = QLabel("Reserved Space for EEG/Motor Data")
        reserved_label.setAlignment(Qt.AlignCenter)
        reserved_label.setFont(QFont('Arial', 10, QFont.Medium, True)) 
        reserved_vbox.addWidget(reserved_label, 1)
        
        right_vbox.addWidget(reserved_frame, 1)
        main_layout.addLayout(right_vbox, 1) # Stretch factor 1 to let the right side expand

        self.adjustSize()

    def build_finger_controls(self, layout: QGridLayout):
    

        # --- Row 0: Shared Effect (Compacted) ---
        
        # Change label from "Shared Effect ID:" to "Shared Effect:"
        shared_label = QLabel("Shared Effect:")
        layout.addWidget(shared_label, 0, 0, alignment=Qt.AlignLeft)
        
        self.shared_spin = QSpinBox()
        self.shared_spin.setRange(1, MAX_EFFECT)
        self.shared_spin.setValue(self.shared_effect_val)
        self.shared_spin.setFixedWidth(50)
        self.shared_spin.valueChanged.connect(self.update_shared_effect_val)
        layout.addWidget(self.shared_spin, 0, 1, alignment=Qt.AlignLeft)
        
        # Change button text from "Set all fingers" to "Set all"
        set_all_btn = QPushButton("Set all")
        set_all_btn.setFixedWidth(65) # Reduced width for compaction
        set_all_btn.clicked.connect(self.set_all_effects_from_shared)
        layout.addWidget(set_all_btn, 0, 2, 1, 2, alignment=Qt.AlignLeft)
        
        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator, 1, 0, 1, 4) 

        # --- Row 2: Individual Finger Headers (Smaller font) ---
        header_row = 2
        
        finger_label = QLabel("<b>Finger</b>")
        layout.addWidget(finger_label, header_row, 0, alignment=Qt.AlignLeft)
        
        effect_label = QLabel("<b>Effect</b>")
        layout.addWidget(effect_label, header_row, 1, alignment=Qt.AlignCenter)
        
        play_label = QLabel("<b>Play Once</b>")
        layout.addWidget(play_label, header_row, 2, alignment=Qt.AlignCenter)
        
        toggle_label = QLabel("<b>Toggle</b>")
        layout.addWidget(toggle_label, header_row, 3, alignment=Qt.AlignCenter)

        # --- Rows 3-7: Individual Finger Rows (Smaller font) ---
        for idx, finger in enumerate(self.fingers):
            row = idx + 3
            name = finger["name"]

            name_label = QLabel(name)
            layout.addWidget(name_label, row, 0, alignment=Qt.AlignLeft)

            spin = QSpinBox()
            spin.setRange(1, MAX_EFFECT)
            spin.setValue(finger["effect_val"])
            spin.setFixedWidth(50)
            spin.valueChanged.connect(lambda val, i=idx: self.update_finger_effect_val(i, val))
            layout.addWidget(spin, row, 1, alignment=Qt.AlignCenter)

            play_btn = QPushButton("Play")
            play_btn.setFixedWidth(60)
            play_btn.clicked.connect(lambda _, i=idx: self.play_finger_once(i))
            layout.addWidget(play_btn, row, 2, alignment=Qt.AlignCenter)

            toggle_chk = QCheckBox("On")
            toggle_chk.setChecked(finger["toggle_val"])
            toggle_chk.stateChanged.connect(lambda state, i=idx: self.on_finger_toggle_changed(i, state))
            layout.addWidget(toggle_chk, row, 3, alignment=Qt.AlignLeft)

    def build_glove_canvas(self, parent: QWidget):
        # Load image and set up a QLabel to display it
        try:
            self.glove_pixmap = QPixmap(GLOVE_IMAGE_PATH)
            self.glove_label = QLabel()
            self.glove_label.setPixmap(self.glove_pixmap)
            self.glove_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            
            canvas_layout = QGridLayout(parent)
            canvas_layout.addWidget(self.glove_label, 0, 0)
            canvas_layout.setContentsMargins(0, 0, 0, 0)
            canvas_layout.setSpacing(0)
            
            # Place buttons on fingertips
            for name, (x, y) in FINGERTIP_COORDS.items():
                motor_idx = FINGER_TO_MOTOR[name]
                btn = QPushButton(name[0])
                btn.setFixedSize(30, 30)
                btn.clicked.connect(lambda _, i=motor_idx: self.play_finger_once(i))
                
                btn_size = btn.sizeHint().width()
                btn.move(x - btn_size // 2, y - btn_size // 2)
                
                btn.setParent(parent) 

        except Exception as e:
            fallback_label = QLabel(f"Error loading {GLOVE_IMAGE_PATH}: {e}")
            fallback_label.setAlignment(Qt.AlignCenter)
            VBoxLayout(parent).addWidget(fallback_label)

    def build_full_top_controls(self, layout: QVBoxLayout):
        
        # --- Timing / Delay / Dwell / Effect Mode / STOP ---
        timing_frame = QFrame()
        timing_vbox = QVBoxLayout(timing_frame)
        timing_vbox.setContentsMargins(0, 0, 0, 0)
        
        # Delay / Dwell / STOP button
        h_timing = QHBoxLayout()
        h_timing.addWidget(QLabel("Delay (s):"))
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.01, 60.0)
        self.delay_spin.setDecimals(2)
        self.delay_spin.setValue(self.delay_time_val)
        self.delay_spin.setFixedWidth(60)
        self.delay_spin.valueChanged.connect(self.update_delay_time)
        h_timing.addWidget(self.delay_spin)

        h_timing.addWidget(QLabel("Dwell (s):"))
        self.dwell_spin = QDoubleSpinBox()
        self.dwell_spin.setRange(0.01, 60.0)
        self.dwell_spin.setDecimals(2)
        self.dwell_spin.setValue(self.dwell_time_val)
        self.dwell_spin.setFixedWidth(60)
        self.dwell_spin.valueChanged.connect(self.update_dwell_time)
        h_timing.addWidget(self.dwell_spin)
        
        h_timing.addStretch(1)

        stop_btn = QPushButton("Stop all motors")
        stop_btn.clicked.connect(self.stop_all)
        stop_btn.setFixedWidth(120)
        h_timing.addWidget(stop_btn)
        
        timing_vbox.addLayout(h_timing)

        # Effect Mode
        h_mode = QHBoxLayout()
        self.effect_mode_label = QLabel(f"Mode: {self.effect_mode_val}")
        h_mode.addWidget(self.effect_mode_label)
        toggle_mode_btn = QPushButton("Toggle none/asc/desc")
        toggle_mode_btn.clicked.connect(self.toggle_effect_mode)
        h_mode.addWidget(toggle_mode_btn)
        h_mode.addStretch(1)
        timing_vbox.addLayout(h_mode)
        
        layout.addWidget(timing_frame)


        # --- Whole-hand Actions (Play All / Ripple) ---
        actions_frame = QFrame()
        actions_vbox = QVBoxLayout(actions_frame)
        actions_vbox.setContentsMargins(0, 0, 0, 0)

        h_actions = QHBoxLayout()
        play_all_btn = QPushButton("Play all once")
        play_all_btn.clicked.connect(self.play_all_once)
        h_actions.addWidget(play_all_btn)
        self.repeat_all_chk = QCheckBox("Repeat all (uses Delay)")
        self.repeat_all_chk.stateChanged.connect(lambda state: self.update_repeat_all(state))
        h_actions.addWidget(self.repeat_all_chk)
        h_actions.addStretch(1)
        actions_vbox.addLayout(h_actions)
        
        h_ripple = QHBoxLayout()
        play_ripple_btn = QPushButton("Play ripple once")
        play_ripple_btn.clicked.connect(lambda: self.start_ripple(repeat=False))
        h_ripple.addWidget(play_ripple_btn)
        self.repeat_ripple_chk = QCheckBox("Repeat ripple (uses Delay and Dwell)")
        self.repeat_ripple_chk.stateChanged.connect(self.on_repeat_ripple_toggled)
        h_ripple.addWidget(self.repeat_ripple_chk)
        h_ripple.addStretch(1)
        actions_vbox.addLayout(h_ripple)
        
        layout.addWidget(actions_frame)


        # --- Ripple Order Controls ---
        order_frame = QFrame()
        order_vbox = QVBoxLayout(order_frame)
        order_vbox.setContentsMargins(0, 0, 0, 0)
        
        h_order = QHBoxLayout()
        h_order.addWidget(QLabel("Ripple Order (0-4, comma-separated):"))
        self.order_edit = QLineEdit(self.order_str_val)
        self.order_edit.setFixedWidth(100)
        h_order.addWidget(self.order_edit)
        update_order_btn = QPushButton("Update order")
        update_order_btn.clicked.connect(self.update_order_from_text)
        h_order.addWidget(update_order_btn)
        h_order.addStretch(1)
        order_vbox.addLayout(h_order)
        
        layout.addWidget(order_frame)


        # --- Logging Controls ---
        log_frame = QFrame()
        log_vbox = QVBoxLayout(log_frame)
        log_vbox.setContentsMargins(0, 0, 0, 0)
        
        h_log = QHBoxLayout()
        log_chk = QCheckBox("Write timestamps to csv")
        log_chk.stateChanged.connect(self.write_timestamps_toggled)
        h_log.addWidget(log_chk)
        
        h_log.addWidget(QLabel("Filename (without extension):"))
        self.log_edit = QLineEdit(self.log_name_val)
        self.log_edit.setFixedWidth(100)
        self.log_edit.textChanged.connect(self.update_log_name)
        h_log.addWidget(self.log_edit)
        
        h_log.addStretch(1)
        log_vbox.addLayout(h_log)
        
        layout.addWidget(log_frame)
        
        layout.addStretch(1)

    # --- Value Update Methods (for Qt widgets to update state) ---

    def update_finger_effect_val(self, idx, val):
        self.fingers[idx]["effect_val"] = val

    def update_shared_effect_val(self, val):
        self.shared_effect_val = val
        
    def update_delay_time(self, val):
        self.delay_time_val = val
        
    def update_dwell_time(self, val):
        self.dwell_time_val = val
        
    def update_repeat_all(self, state):
        self.repeat_all_val = state == Qt.Checked
        if not self.repeat_all_val:
            self.all_last_time = None

    def update_log_name(self, text):
        self.log_name_val = text

    # --- Action Methods (Mirrors glove_app.py logic) ---

    def set_next_effect_for_motor(self, motor_idx):
        base_effect = self.fingers[motor_idx]["effect_val"]
        mode = self.effect_mode_val
        change = CHANGE_DICT[mode]
        next_effect = (base_effect + change) % (MAX_EFFECT + 1)
        if next_effect == 0:
            next_effect = MAX_EFFECT if change == -1 else 1
            
        # Update internal state and GUI widget
        self.fingers[motor_idx]["effect_val"] = next_effect
        # Find the SpinBox and update it
        layout = self.centralWidget().findChild(QGroupBox).findChild(QGridLayout)
        # Assuming finger controls are in the first QGroupBox and the QGridLayout is the first one found
        spinbox = layout.itemAtPosition(motor_idx + 3, 1).widget() # +3 due to 2 header rows and 1 separator row
        if isinstance(spinbox, QSpinBox):
            spinbox.setValue(next_effect)

    def play_finger_once(self, finger_idx):
        effect_id = self.fingers[finger_idx]["effect_val"]
        self.glove.play_effect(finger_idx, effect_id)
        self.log_event("finger_once", [finger_idx], [effect_id])
        self.set_next_effect_for_motor(finger_idx)
        self.motor_viz.visualize_effect(finger_idx, effect_id)

    def play_all_once(self):
        effects = [self.fingers[i]["effect_val"] for i in range(len(self.fingers))]
        self.glove.play_all(effects)
        self.log_event("all_once", list(range(len(self.fingers))), effects)
        for i in range(len(self.fingers)):
            self.set_next_effect_for_motor(i)

    def on_finger_toggle_changed(self, finger_idx, state):
        self.fingers[finger_idx]["toggle_val"] = state == Qt.Checked
        self.finger_last_time[finger_idx] = None

    def set_all_effects_from_shared(self):
        shared_id = self.shared_effect_val
        for finger in self.fingers:
            finger["effect_val"] = shared_id
        
        layout = self.centralWidget().findChild(QGroupBox).findChild(QGridLayout)
        for idx in range(len(self.fingers)):
            spinbox = layout.itemAtPosition(idx + 3, 1).widget() # +3 offset
            if isinstance(spinbox, QSpinBox):
                spinbox.setValue(shared_id)

    def toggle_effect_mode(self):
        mode = self.effect_mode_val
        if mode == "none":
            mode = "ascending"
        elif mode == "ascending":
            mode = "descending"
        else:
            mode = "none"

        self.effect_mode_val = mode
        self.effect_mode_label.setText(f"Mode: {mode}")

    def on_repeat_ripple_toggled(self, state):
        self.repeat_ripple_val = state == Qt.Checked
        if self.repeat_ripple_val:
            self.ripple_repeat = True
            if not self.ripple_running:
                self.start_ripple(repeat=True)
        else:
            self.ripple_repeat = False

    def start_ripple(self, repeat: bool):
        if self.ripple_running:
            return
        self.update_order_from_text()
        self.ripple_running = True
        self.ripple_repeat = repeat
        self.ripple_index = 0
        self.ripple_step()

    def ripple_step(self):
        if not self.ripple_running:
            return

        if self.ripple_index == len(self.order):
            if self.ripple_repeat and self.repeat_ripple_val:
                self.ripple_index = 0
                delay_ms = max(1, int(self.delay_time_val * 1000))
                QTimer.singleShot(delay_ms, self.ripple_step)
                return
            else:
                self.ripple_running = False
                self.repeat_ripple_chk.setChecked(False)
                return

        motor_idx = self.order[self.ripple_index]
        if 0 <= motor_idx < len(self.fingers):
            effect_id = self.fingers[motor_idx]["effect_val"]
            self.glove.play_effect(motor_idx, effect_id)
            self.set_next_effect_for_motor(motor_idx)
            self.log_event("ripple_step", [motor_idx], [effect_id])
            self.motor_viz.visualize_effect(motor_idx, effect_id)

        self.ripple_index += 1
        try:
            dwell_ms = max(1, int(self.dwell_time_val * 1000))
        except:
            dwell_ms = 200
        QTimer.singleShot(dwell_ms, self.ripple_step)

    def update_order_from_text(self):
        text = self.order_edit.text()
        try:
            parts = [p.strip() for p in text.split(",") if p.strip() != ""]
            nums = [int(p) for p in parts]
        except ValueError:
            QMessageBox.critical(self, "Order Error", "Order must be comma-separated integers.")
            return

        if sorted(nums) != list(range(5)):
            QMessageBox.critical(self, "Order Error", "Order must contain each of 0,1,2,3,4 exactly once.")
            return

        self.order = nums
        self.order_str_val = text

    def stop_all(self):
        self.glove.stop_all()

        layout = self.centralWidget().findChild(QGroupBox).findChild(QGridLayout)
        for idx, finger in enumerate(self.fingers):
            self.fingers[idx]["toggle_val"] = False
            self.finger_last_time[idx] = None
            
            checkbox = layout.itemAtPosition(idx + 3, 3).widget() # +3 offset
            if isinstance(checkbox, QCheckBox):
                checkbox.setChecked(False)

        self.repeat_all_val = False
        self.all_last_time = None
        self.repeat_all_chk.setChecked(False)

        self.ripple_running = False
        self.ripple_repeat = False
        self.ripple_index = 0
        self.repeat_ripple_val = False
        self.repeat_ripple_chk.setChecked(False)

    # --- Periodic Update Loop (Qt Timer) ---
    def periodic_update(self):
        now = time.time()
        try:
            delay = max(self.delay_time_val, 0.01)
        except:
            delay = 2.0

        # Per-finger toggle mode
        for idx, finger in enumerate(self.fingers):
            if finger["toggle_val"]:
                last = self.finger_last_time[idx]
                if last is None or (now - last) >= delay:
                    try:
                        eff = finger["effect_val"]
                    except:
                        eff = 1
                    self.glove.play_effect(idx, eff)
                    self.log_event("finger_toggle", [idx], [eff])
                    self.finger_last_time[idx] = now

                    self.motor_viz.visualize_effect(idx, eff)
            else:
                self.finger_last_time[idx] = None

        # Repeat-all mode
        if self.repeat_all_val:
            last = self.all_last_time
            if last is None or (now - last) >= delay:
                self.play_all_once()
                self.all_last_time = now
        else:
            self.all_last_time = None

    def closeEvent(self, event):
        self.stop_all()
        super().closeEvent(event)


if __name__ == "__main__":
    if not os.path.exists(GLOVE_IMAGE_PATH):
        print(f"Error: Required image file '{GLOVE_IMAGE_PATH}' not found.")
        print("Please ensure 'glove.png' is in the same directory as this script.")
        sys.exit(1)
        
    app = QApplication(sys.argv)
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates)) 
    gui = QtHapticGloveGUI()
    gui.show()
    sys.exit(app.exec_())