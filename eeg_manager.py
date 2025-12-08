import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

class EEGManager:
    def __init__(self, board_id=BoardIds.SYNTHETIC_BOARD.value):
        self.board_id = board_id
        params = BrainFlowInputParams()
        self.board = BoardShim(self.board_id, params)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.ts_channel = BoardShim.get_timestamp_channel(self.board_id)
        
        self.mindfulness = self._load_model(BrainFlowMetrics.MINDFULNESS)
        self.restfulness = self._load_model(BrainFlowMetrics.RESTFULNESS)
        self.is_streaming = False

    def _load_model(self, metric):
        try:
            params = BrainFlowModelParams(metric.value, BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
            model = MLModel(params)
            model.prepare()
            return model
        except: return None

    def start_stream(self):
        if not self.is_streaming:
            self.board.prepare_session()
            self.board.start_stream(45000)
            self.is_streaming = True

    def stop_stream(self):
        if self.is_streaming:
            if self.board.is_prepared():
                self.board.stop_stream()
                self.board.release_session()
            self.is_streaming = False
            if self.mindfulness: self.mindfulness.release()
            if self.restfulness: self.restfulness.release()

    def get_data(self, num_samples):
        return self.board.get_current_board_data(num_samples)
    
    def get_bands(self, data):
        return DataFilter.get_avg_band_powers(data, self.eeg_channels, self.sampling_rate, True)

    def predict_metrics(self, feature_vector):
        mind = self.mindfulness.predict(feature_vector)[0] if self.mindfulness else 0.0
        rest = self.restfulness.predict(feature_vector)[0] if self.restfulness else 0.0
        return mind, rest