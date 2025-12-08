import csv
import os
import time
from datetime import datetime

class DataLogger:
    """
    Logs data to 3 distinct CSV files to separate high-frequency raw data 
    from lower-frequency features and sparse haptic events.
    
    Files:
      1) Raw samples: [timestamp, ch_1, ch_2...]
      2) Features:    [timestamp, delta, theta, alpha, beta, gamma, 
                       mindfulness, restfulness, mind_detected, rest_detected]
      3) Haptics:     [timestamp, event_type, details, intensity, duration_ms]
    """
    def __init__(self):
        self.raw_file = None
        self.feat_file = None
        self.hapt_file = None
        
        self.raw_writer = None
        self.feat_writer = None
        self.hapt_writer = None
        
        self.last_logged_ts = None  # To prevent duplicate logging of raw samples
        self.session_dir = None

    def start(self, eeg_channels, base_dir="data_logs", prefix="session"):
        """
        Opens file handles and writes CSV headers.
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = base_dir # Stored for UI reference
        
        # Define file paths
        raw_path = os.path.join(base_dir, f"{prefix}_{timestamp_str}_eeg_raw.csv")
        feat_path = os.path.join(base_dir, f"{prefix}_{timestamp_str}_eeg_features.csv")
        hapt_path = os.path.join(base_dir, f"{prefix}_{timestamp_str}_haptic_events.csv")

        # Open files
        self.raw_file = open(raw_path, "w", newline="")
        self.feat_file = open(feat_path, "w", newline="")
        self.hapt_file = open(hapt_path, "w", newline="")

        self.raw_writer = csv.writer(self.raw_file)
        self.feat_writer = csv.writer(self.feat_file)
        self.hapt_writer = csv.writer(self.hapt_file)
        
        self.last_logged_ts = None

        # 1. Write Raw Header
        # brainflow_ts is always the first column
        raw_header = ["brainflow_ts"] + [f"ch_{ch}" for ch in eeg_channels]
        self.raw_writer.writerow(raw_header)

        # 2. Write Feature Header
        feat_header = [
            "brainflow_ts",
            "delta", "theta", "alpha", "beta", "gamma",
            "mindfulness_score", "restfulness_score",
            "mindfulness_detected", "restfulness_detected",
        ]
        self.feat_writer.writerow(feat_header)

        # 3. Write Haptic Header
        hapt_header = ["system_ts", "event_type", "details", "intensity", "duration_ms"]
        self.hapt_writer.writerow(hapt_header)

        print(f"[DataLogger] Started.\n  Raw: {raw_path}\n  Feat: {feat_path}\n  Hapt: {hapt_path}")

    def log_raw(self, data, timestamp_channel, eeg_channels):
        """
        Logs new raw samples from the BrainFlow buffer.
        Checks timestamps to ensure no duplicates are written if buffers overlap.
        
        data: 2D numpy array from board.get_board_data() or get_current_board_data()
        timestamp_channel: index of the timestamp row
        eeg_channels: list of indices for EEG rows
        """
        if self.raw_writer is None:
            return

        timestamps = data[timestamp_channel, :]  # shape: (num_samples,)
        if timestamps.size == 0:
            return

        # Only log samples with timestamp > last_logged_ts
        for col in range(timestamps.shape[0]):
            ts = float(timestamps[col])
            if (self.last_logged_ts is None) or (ts > self.last_logged_ts):
                row = [ts]
                for ch_idx in eeg_channels:
                    row.append(float(data[ch_idx, col]))
                self.raw_writer.writerow(row)

        self.last_logged_ts = float(timestamps[-1])

    def log_features(self, feature_ts, bands, mind_score, rest_score,
                     mind_detected, rest_detected):
        """
        Logs computed features (approx 1Hz or window-based).
        """
        if self.feat_writer is None:
            return
        
        # bands is expected to be [delta, theta, alpha, beta, gamma]
        row = [
            float(feature_ts),
            float(bands[0]), float(bands[1]), float(bands[2]),
            float(bands[3]), float(bands[4]),
            float(mind_score), float(rest_score),
            int(bool(mind_detected)), int(bool(rest_detected)),
        ]
        self.feat_writer.writerow(row)

    def log_haptic(self, timestamp, event_type, details, intensity, duration_ms):
        """
        Logs haptic trigger events.
        """
        if self.hapt_writer is None:
            return
            
        row = [timestamp, event_type, details, intensity, duration_ms]
        self.hapt_writer.writerow(row)
        self.hapt_file.flush() # Flush immediately for sparse events

    def stop(self):
        """
        Closes all file handles safely.
        """
        if self.raw_file: self.raw_file.close()
        if self.feat_file: self.feat_file.close()
        if self.hapt_file: self.hapt_file.close()
        
        self.raw_file = None
        self.feat_file = None
        self.hapt_file = None
        print("[DataLogger] Logs closed.")