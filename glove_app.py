import time
import tkinter as tk
from tkinter import ttk, messagebox
import logging, csv, io, os

from haptic_glove import HapticGlove, MAX_EFFECT

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_TO_MOTOR = {
    "Thumb": 0,
    "Index": 1,
    "Middle": 2,
    "Ring": 3,
    "Pinky": 4,
}

# Coordinates for drawing fingertip circles on the glove image
FINGERTIP_COORDS = {
    "Thumb":  (53, 137),
    "Index":  (152, 46),
    "Middle": (209, 42),
    "Ring":   (259, 67),
    "Pinky":  (293, 122),
}

CHANGE_DICT = {
    "none": 0,
    "ascending": 1,
    "descending": -1,
}

class HapticGloveGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Haptic Glove Controller")
        self.resizable(False, False)

        # Initialize hardware
        self.channels = [0,1,2,3,4]
        self.glove = HapticGlove(channels=self.channels)

        # Per-finger: effect number + toggled flag
        self.fingers = []  # list of dicts: {"name", "effect_var", "toggle_var"}
        self.finger_last_time = [None] * 5  # last time each finger fired in toggle mode

        for name in FINGER_NAMES:
            effect_var = tk.IntVar(value=1)
            toggle_var = tk.BooleanVar(value=False)
            self.fingers.append({
                "name": name,
                "effect_var": effect_var,
                "toggle_var": toggle_var,
            })

        # Shared / whole-hand variables
        self.shared_effect_var = tk.IntVar(value=1)
        self.write_timestamps_var = tk.BooleanVar(value=False)
        self.repeat_all_var = tk.BooleanVar(value=False)
        self.repeat_ripple_var = tk.BooleanVar(value=False)

        self.delay_time_var = tk.DoubleVar(value=2.0)   # between groups (seconds)
        self.dwell_time_var = tk.DoubleVar(value=0.2)   # between motors in ripple (seconds)

        # Order for ripple
        self.order = [0, 1, 2, 3, 4]
        self.order_str_var = tk.StringVar(value="0,1,2,3,4")

        # Effect increment mode: none / ascending / descending
        self.effect_mode_var = tk.StringVar(value="none")

        # For "repeat all" timing
        self.all_last_time = None

        # Ripple state machine
        self.ripple_running = False
        self.ripple_repeat = False
        self.ripple_index = 0

        # Logging filename
        self.log_name_var = tk.StringVar(value="session")

        # ------------- Layout -------------

        self.build_layout()

        # ------------- Periodic update loop -------------
        self.periodic_update()

    def init_logger(self):
        dirname = "haptic_data"
        os.makedirs(dirname, exist_ok=True)

        name = self.log_name_var.get().strip() or "session"
        path = os.path.join(dirname, f"{name}.csv")

        logger = logging.getLogger("haptics")

        # Prevent duplicate handlers if this gets re-run
        if logger.handlers:
            logger.handlers.clear()

        logger.setLevel(logging.INFO)

        # Create a normal CSV file handler
        handler = logging.FileHandler(path, mode="a")
        formatter = logging.Formatter("%(asctime)s,%(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        self.logger = logger


    def build_layout(self):
        # Two columns: left (finger + glove), right (main controls)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)

        left_frame = ttk.Frame(self)
        right_frame = ttk.Frame(self)

        left_frame.grid(row=0, column=0, sticky="nws", padx=10, pady=10)
        right_frame.grid(row=0, column=1, sticky="nes", padx=10, pady=10)

        # Left: top = finger controls, bottom = glove canvas
        finger_frame = ttk.Frame(left_frame)
        finger_frame.pack(fill="x", side="top", pady=(0, 10))

        self.build_finger_controls(finger_frame)

        # canvas_frame = ttk.LabelFrame(left_frame, text="Glove View")
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(side="bottom")

        self.build_glove_canvas(canvas_frame)

        # Right: main controls
        self.build_main_controls(right_frame)

    def build_finger_controls(self, parent):
        # Header row
        ttk.Label(parent, text="Finger").grid(row=0, column=0, padx=4, pady=2)
        ttk.Label(parent, text="Effect").grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(parent, text="Play Once").grid(row=0, column=2, padx=4, pady=2)
        ttk.Label(parent, text="Toggle").grid(row=0, column=3, padx=4, pady=2)

        for idx, finger in enumerate(self.fingers):
            row = idx + 1
            name = finger["name"]
            effect_var = finger["effect_var"]
            toggle_var = finger["toggle_var"]

            ttk.Label(parent, text=name).grid(row=row, column=0, sticky="w")

            spin = tk.Spinbox(parent, from_=1, to=MAX_EFFECT, textvariable=effect_var, width=5)
            spin.grid(row=row, column=1, padx=2)

            func1=lambda i=idx : self.play_finger_once(i)
            play_btn = ttk.Button(parent, text="Play", command=func1)
            play_btn.grid(row=row, column=2, padx=2)

            func2=lambda i=idx : self.on_finger_toggle_changed(i)
            toggle_chk = ttk.Checkbutton(parent, text="On", variable=toggle_var, command=func2)
            toggle_chk.grid(row=row, column=3, padx=2)
    
    def build_glove_canvas(self, parent):
        # load image
        self.glove_img = tk.PhotoImage(file="glove.png")

        height = self.glove_img.height()
        width  = self.glove_img.width()

        self.canvas = tk.Canvas(parent, width=width, height=height)
        self.canvas.pack()

        # Draw glove
        self.canvas.create_image(0, 0, image=self.glove_img, anchor="nw")

        # Place buttons on fingertips (play once for that finger)
        for name, (x, y) in FINGERTIP_COORDS.items():
            motor_idx = FINGER_TO_MOTOR[name]
            func=lambda i=motor_idx : self.play_finger_once(i)
            btn = ttk.Button(self.canvas, text=name[0], width=2, command=func)
            self.canvas.create_window(x, y, window=btn)

    def build_main_controls(self, parent):
        # Shared effect + set-all
        shared_frame = ttk.Frame(parent)
        shared_frame.pack(fill="x", pady=4)
        ttk.Label(shared_frame, text="Shared Effect ID:").grid(row=0, column=0, sticky="w")

        shared_spin = tk.Spinbox(shared_frame, from_=1, to=MAX_EFFECT, textvariable=self.shared_effect_var, width=6)
        shared_spin.grid(row=0, column=1, padx=4, sticky="w")

        set_all_btn = ttk.Button(shared_frame, text="Set all fingers", command=self.set_all_effects_from_shared)
        set_all_btn.grid(row=0, column=2, padx=4)

        # Delay / dwell
        timing_frame = ttk.Frame(parent)
        timing_frame.pack(fill="x", pady=4)

        ttk.Label(timing_frame, text="Delay (s):").grid(row=0, column=0, sticky="w")
        delay_entry = ttk.Entry(timing_frame, textvariable=self.delay_time_var, width=6)
        delay_entry.grid(row=0, column=1, padx=4, sticky="w")

        ttk.Label(timing_frame, text="Dwell (s):").grid(row=1, column=0, sticky="w")
        dwell_entry = ttk.Entry(timing_frame, textvariable=self.dwell_time_var, width=6)
        dwell_entry.grid(row=1, column=1, padx=4, sticky="w")

        # Effect mode (none / ascending / descending)
        effect_mode_frame = ttk.Frame(parent)
        effect_mode_frame.pack(fill="x", pady=4)

        self.effect_mode_label = ttk.Label(effect_mode_frame, text="Mode: none")
        self.effect_mode_label.grid(row=0, column=0, sticky="w")

        toggle_mode_btn = ttk.Button(effect_mode_frame, text="Toggle none/asc/desc", command=self.toggle_effect_mode)
        toggle_mode_btn.grid(row=0, column=1, padx=4, sticky="w")

        # Whole-hand actions
        all_frame = ttk.Frame(parent)
        all_frame.pack(fill="x", pady=4)

        play_all_once_btn = ttk.Button(all_frame, text="Play all once", command=self.play_all_once)
        play_all_once_btn.grid(row=0, column=0, padx=4, pady=2, sticky="w")
        repeat_all_chk = ttk.Checkbutton(all_frame, text="Repeat all (uses Delay)", variable=self.repeat_all_var)
        repeat_all_chk.grid(row=0, column=1, padx=4, pady=2, sticky="w")

        play_ripple_once_btn = ttk.Button(all_frame, text="Play ripple once", command=lambda: self.start_ripple(repeat=False))
        play_ripple_once_btn.grid(row=1, column=0, padx=4, pady=2, sticky="w")
        self.repeat_ripple_chk = ttk.Checkbutton(all_frame, text="Repeat ripple (uses Delay and Dwell)", variable=self.repeat_ripple_var, command=self.on_repeat_ripple_toggled)
        self.repeat_ripple_chk.grid(row=1, column=1, padx=4, pady=2, sticky="w") 

        # Order controls
        order_frame = ttk.Frame(parent) 
        order_frame.pack(fill="x", pady=4)

        ttk.Label(order_frame, text="Ripple Order (0-4, comma-separated):").grid(row=0, column=0, sticky="w")
        order_entry = ttk.Entry(order_frame, textvariable=self.order_str_var, width=18)
        order_entry.grid(row=0, column=1, padx=4, sticky="w")

        update_order_btn = ttk.Button(order_frame, text="Update order", command=self.update_order_from_text)
        update_order_btn.grid(row=0, column=2, padx=4, sticky="w")

        # Logging controls
        log_frame = ttk.Frame(parent)
        log_frame.pack(fill="x", pady=4)

        log_chk = ttk.Checkbutton(log_frame, text="Write timestamps to csv", variable=self.write_timestamps_var, command=self.write_timestamps_toggled)
        log_chk.grid(row=0, column=0, sticky="w")

        ttk.Label(log_frame, text="Filename (without extension):").grid(row=1, column=0, sticky="w", pady=(4, 0))
        log_entry = ttk.Entry(log_frame, textvariable=self.log_name_var, width=18)
        log_entry.grid(row=1, column=1, padx=4, sticky="w", pady=(4, 0))

        ttk.Label(log_frame, text="Saved under ./haptic_data/").grid(row=2, column=0, columnspan=2, sticky="w")

        # Stop all
        stop_frame = ttk.Frame(parent)
        stop_frame.pack(fill="x", pady=8)

        stop_btn = ttk.Button(stop_frame, text="Stop all motors", command=self.stop_all)
        stop_btn.pack(side="left")

    # Logging function
    def log_event(self, action, motors, effects):
        if not self.write_timestamps_var.get():
            return

        motors_str = ";".join(str(m) for m in motors)
        effects_str = ";".join(str(e) for e in effects)

        # Format: action,motorlist,effectlist
        self.logger.info(f"{action},{motors_str},{effects_str}")

    # Reinitialize logger if writing timestamps is toggled on
    def write_timestamps_toggled(self):
        if self.write_timestamps_var.get():
            self.init_logger()

    # Set effect ID for motor based on effect mode
    def set_next_effect_for_motor(self, motor_idx):
        base_effect = self.fingers[motor_idx]["effect_var"].get()
        mode = self.effect_mode_var.get()
        change = CHANGE_DICT[mode]
        next_effect = (base_effect + change) % (MAX_EFFECT + 1)
        if next_effect == 0:
            next_effect = MAX_EFFECT if change == -1 else 1
        self.fingers[motor_idx]["effect_var"].set(next_effect)

    # play single finger once
    def play_finger_once(self, finger_idx):
        effect_id = self.fingers[finger_idx]["effect_var"].get()
        self.glove.play_effect(finger_idx, effect_id)
        self.log_event("finger_once", [finger_idx], [effect_id])
        self.set_next_effect_for_motor(finger_idx)

    # Play all motors once
    def play_all_once(self):
        effects = [self.fingers[i]["effect_var"].get() for i in range(len(self.fingers))]
        self.glove.play_all(effects)
        self.log_event("all_once", list(range(len(self.fingers))), effects)
        for i in range(len(self.fingers)):
            self.set_next_effect_for_motor(i)

    # finger toggle changed, reset timing
    def on_finger_toggle_changed(self, finger_idx):
        self.finger_last_time[finger_idx] = None

    # set all finger effects from shared effect
    def set_all_effects_from_shared(self):
        shared_id = self.shared_effect_var.get()
        for finger in self.fingers:
            finger["effect_var"].set(shared_id)

    # cycle effect mode: none -> ascending -> descending -> none
    def toggle_effect_mode(self):
        mode = self.effect_mode_var.get()
        if mode == "none":
            mode = "ascending"
        elif mode == "ascending":
            mode = "descending"
        else:
            mode = "none"

        self.effect_mode_var.set(mode)
        self.effect_mode_label.config(text=f"Mode: {mode}")

    def start_ripple(self, repeat: bool):
        if self.ripple_running:
            return
        self.update_order_from_text()
        self.ripple_running = True
        self.ripple_repeat = repeat
        self.ripple_index = 0
        self.ripple_step()

    # Repeat ripple toggled
    def on_repeat_ripple_toggled(self):
        if self.repeat_ripple_var.get():
            self.ripple_repeat = True
            if not self.ripple_running:
                self.start_ripple(repeat=True)
        else:
            self.ripple_repeat = False  # current cycle finishes, then stops

    # Lopping step for ripple effect
    def ripple_step(self):
        if not self.ripple_running:
            return

        if self.ripple_index == len(self.order):
            # One full cycle complete
            if self.ripple_repeat and self.repeat_ripple_var.get():
                # Wait Delay before next cycle
                self.ripple_index = 0
                delay_ms = max(1, int(self.delay_time_var.get() * 1000))
                self.after(delay_ms, self.ripple_step)
                return
            else:
                # Done
                self.ripple_running = False
                return

        # Play next motor in order
        motor_idx = self.order[self.ripple_index]
        if 0 <= motor_idx < len(self.fingers):
            self.glove.play_effect(motor_idx, self.fingers[motor_idx]["effect_var"].get())
            self.set_next_effect_for_motor(motor_idx)
            self.log_event("ripple_step", [motor_idx], [self.fingers[motor_idx]["effect_var"].get()])

        self.ripple_index += 1
        try:
            dwell_ms = max(1, int(self.dwell_time_var.get() * 1000))
        except:
            dwell_ms = 200
        self.after(dwell_ms, self.ripple_step)

    # update ripple order from text entry
    def update_order_from_text(self):
        text = self.order_str_var.get()
        try:
            parts = [p.strip() for p in text.split(",") if p.strip() != ""]
            nums = [int(p) for p in parts]
        except ValueError:
            messagebox.showerror("Order Error", "Order must be comma-separated integers.")
            return

        if sorted(nums) != sorted(range(5)):
            messagebox.showerror("Order Error", "Order must contain each of 0,1,2,3,4 exactly once.")
            return

        self.order = nums


    # Stop all motors and reset states
    def stop_all(self):
        self.glove.stop_all()

        # Reset timing/state for per-finger toggles
        for idx, finger in enumerate(self.fingers):
            finger["toggle_var"].set(False)
            self.finger_last_time[idx] = None

        # Reset repeat-all state
        self.repeat_all_var.set(False)
        self.all_last_time = None

        # Reset ripple state
        self.ripple_running = False
        self.ripple_repeat = False
        self.ripple_index = 0
        self.repeat_ripple_var.set(False)

    
    # Periodic update function, nonblocking, state based (loop)
    def periodic_update(self):
        now = time.time()
        try:
            delay = max(self.delay_time_var.get(), 0.01)  # minimum delay of 10ms
        except:
            delay = 2.0

        # Per-finger toggle mode (always uses per-finger effect, not effect_mode)
        for idx, finger in enumerate(self.fingers):
            if finger["toggle_var"].get():
                last = self.finger_last_time[idx]
                if last is None or (now - last) >= delay:
                    try:
                        eff = finger["effect_var"].get()
                    except:
                        eff = 1
                    self.glove.play_effect(idx, eff)
                    self.log_event("finger_toggle", [idx], [eff])
                    self.finger_last_time[idx] = now
            else:
                self.finger_last_time[idx] = None

        # Repeat-all mode (uses effect_mode)
        if self.repeat_all_var.get():
            last = self.all_last_time
            if last is None or (now - last) >= delay:
                self.play_all_once()
                self.all_last_time = now
        else:
            self.all_last_time = None

        # schedule next tick
        self.after(25, self.periodic_update)  # ~40 Hz update


if __name__ == "__main__":
    app = HapticGloveGUI()
    app.mainloop()
