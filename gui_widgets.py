import time
from collections import deque
from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QGridLayout
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, QRect

FINGER_TO_MOTOR = {"Thumb": 0, "Index": 1, "Middle": 2, "Ring": 3, "Pinky": 4}
FINGERTIP_COORDS = {
    "Thumb":  (53, 137), "Index":  (152, 46), "Middle": (209, 42),
    "Ring":   (259, 67), "Pinky":  (293, 122),
}

class MotorVisualizationWidget(QWidget):
    def __init__(self, motors=["Thumb", "Index", "Middle", "Ring", "Pinky"], plot_duration_s=10):
        super().__init__()
        self.motors = motors
        self.plot_duration_s = plot_duration_s
        self.line_height = 25
        self.motor_margin = 10
        
        # Optimization: Use deque for efficient popping from the left
        self.timeline = deque() 
        self.setMinimumHeight(len(motors) * (self.line_height + self.motor_margin) + 20)
        
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update)
        # Optimization: Reduce update rate from 50ms (20Hz) to 66ms (15Hz) if dragging, 
        # but 50ms is standard. We optimize the paintEvent instead.
        self.plot_timer.start(50) 
        
    def visualize_effect(self, motor_idx, effect_id):
        self.timeline.append((time.time(), motor_idx, effect_id))
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        current_time = time.time()
        
        label_width = 30
        plot_area_width = W - label_width
        scale_factor = plot_area_width / self.plot_duration_s
        cutoff_time = current_time - self.plot_duration_s

        # 1. Draw Static Grid (Fast)
        for i, name in enumerate(self.motors):
            y_center = (i * (self.line_height + self.motor_margin)) + self.line_height
            # Label
            painter.setFont(QFont('Arial', 9))
            painter.setPen(Qt.black)
            painter.drawText(QRect(0, y_center - self.line_height // 2, label_width, self.line_height), 
                             Qt.AlignCenter, name[0])
            # Line
            painter.setPen(QPen(Qt.gray, 2))
            painter.drawLine(label_width, y_center, W, y_center)
        
        # 2. Prune old events (Optimization)
        while self.timeline and self.timeline[0][0] < cutoff_time:
            self.timeline.popleft()
            
        # 3. Draw Events
        effect_duration_s = 0.2
        painter.setPen(Qt.white)
        painter.setFont(QFont('Arial', 8, QFont.Bold))
        
        # Optimization: Only draw what is visible (already pruned mostly, but safe check)
        for t, motor_idx, effect_id in self.timeline:
            y_center = (motor_idx * (self.line_height + self.motor_margin)) + self.line_height
            
            x_end = W - ((current_time - t) * scale_factor)
            x_start = x_end - (effect_duration_s * scale_factor)
            
            # Simple visibility check
            if x_end < label_width: continue
            
            rect = QRect(int(x_start), y_center - self.line_height // 2, 
                         int(x_end - x_start), self.line_height)
            
            painter.fillRect(rect, QColor('#3f88c5'))
            painter.drawText(rect, Qt.AlignCenter, str(effect_id))
        
        painter.end()

class GloveCanvas(QWidget):
    def __init__(self, image_path, callback):
        super().__init__()
        self.callback = callback
        try:
            self.pixmap = QPixmap(image_path)
        except:
            self.pixmap = None
            
        layout = QGridLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        self.bg_label = QLabel()
        if self.pixmap:
            self.bg_label.setPixmap(self.pixmap)
            self.bg_label.setScaledContents(False)
        else:
            self.bg_label.setText("Glove Image Missing")
        self.bg_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.bg_label, 0, 0)
        
        # Overlay Buttons
        for name, (x, y) in FINGERTIP_COORDS.items():
            motor_idx = FINGER_TO_MOTOR[name]
            btn = QPushButton(name[0], self)
            btn.setFixedSize(30, 30)
            btn.clicked.connect(lambda _, i=motor_idx: self.callback(i))
            # Adjust position slightly for button center
            btn.move(x - 5, y - 5)