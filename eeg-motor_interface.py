import argparse
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

# ---------- Motor / I2C imports ----------
import board
import busio
from adafruit_bus_device.i2c_device import I2CDevice
import adafruit_drv2605

# ---------- Motor Config ----------
MUX_ADDR = 0x70
CHANNELS = [0, 1, 2]      # 3 motors on mux channels 0,1,2
USE_LRA = True            # True for LRA, False for ERM
LRA_LIBRARY = 6           # 6 for LRA, 1 for ERM
MAX_EFFECT = 123          # DRV2605 effect range 1..123
DEFAULT_EFFECT = 1        # simple click / buzz pattern


# ---------- I2C + MUX ----------
i2c = busio.I2C(board.SCL, board.SDA)

def tca_select(ch: int):
    """Select TCA9548A mux channel."""
    with I2CDevice(i2c, MUX_ADDR) as mux:
        mux.write(bytes([1 << ch]))
    time.sleep(0.002)

def make_drv_for_channel(ch: int, use_lra: bool, lib: int):
    """Create and configure a DRV2605 instance for a mux channel."""
    tca_select(ch)
    drv = adafruit_drv2605.DRV2605(i2c)
    if use_lra:
        drv.use_LRM()
        drv.library = lib
    else:
        drv.use_ERM()
        drv.library = 1
    return drv

drivers = [make_drv_for_channel(ch, USE_LRA, LRA_LIBRARY) for ch in CHANNELS]

def trigger_single_motor(motor_index: int, effect: int = DEFAULT_EFFECT):
    """
    Trigger a single motor by index (0,1,2).
    Uses sequence[0] with a single effect and calls play().
    """
    if not (0 <= motor_index < len(drivers)):
        return
    ch = CHANNELS[motor_index]
    drv = drivers[motor_index]
    tca_select(ch)
    drv.sequence[0] = adafruit_drv2605.Effect(effect)
    drv.play()


def buzz_from_metrics(mindfulness_detected: bool, restfulness_detected: bool):
    """
    Map EEG metrics → motors:
      - Motor 0: buzz when NEITHER mindfulness nor restfulness detected
      - Motor 1: buzz when mindfulness detected
      - Motor 2: buzz when restfulness detected
    """
    # 1) "Neutral / no metric" buzz: only when both are False
    if not mindfulness_detected and not restfulness_detected:
        trigger_single_motor(0)

    # 2) Mindfulness buzz
    if mindfulness_detected:
        trigger_single_motor(1)

    # 3) Restfulness buzz
    if restfulness_detected:
        trigger_single_motor(2)

def main():
    BoardShim.enable_board_logger()
    DataFilter.enable_data_logger()
    MLModel.enable_ml_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    #parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
    #
    #                     required=True)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = '/dev/ttyUSB0'
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
    
    mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    mindfulness = MLModel(mindfulness_params)
    mindfulness.prepare()
    
    restfulness_params = BrainFlowModelParams(BrainFlowMetrics.RESTFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
    restfulness = MLModel(restfulness_params)
    restfulness.prepare()
    
    MINDFULNESS_THRESHOLD = 0.7
    RESTFULNESS_THRESHOLD = 0.7

    eeg_channels = BoardShim.get_eeg_channels(int(master_board_id))
    
    window_sec = 4
    num_samples = int(window_sec * sampling_rate)

    try:
        while True:
            # give the buffer time to fill
            time.sleep(window_sec)

            # get the most recent window of data
            data = board.get_current_board_data(num_samples)

            # safety check: skip iteration if not enough data yet
            if data.shape[1] < num_samples:
                print("Not enough data yet, waiting...")
                continue

            # compute band powers and feature vector
            bands = DataFilter.get_avg_band_powers(data, eeg_channels, sampling_rate, True)
            feature_vector = bands[0]

            # compute metrics
            mindfulness_score = float(mindfulness.predict(feature_vector)[0])
            restfulness_score = float(restfulness.predict(feature_vector)[0])

            # booleans indicating whether each state is "detected"
            mindfulness_detected = mindfulness_score >= MINDFULNESS_THRESHOLD
            restfulness_detected = restfulness_score >= RESTFULNESS_THRESHOLD

            # print so you can see what's happening
            
            print(
                f"Mindfulness: {mindfulness_score:.3f} "
                f"(detected={mindfulness_score >= MINDFULNESS_THRESHOLD}), "
                f"Restfulness: {restfulness_score:.3f} "
                f"(detected={restfulness_score >= RESTFULNESS_THRESHOLD})"
            )
            print()

            buzz_from_metrics(mindfulness_detected, restfulness_detected)

    except KeyboardInterrupt:
        print("Stopping on user interrupt...")

    finally:
        # clean up
        board.stop_stream()
        board.release_session()
        mindfulness.release()
        restfulness.release()

        # stop motors on exit
        for ch, drv in zip(CHANNELS, drivers):
            try:
                tca_select(ch)
                drv.stop()
            except Exception:
                pass



if __name__ == "__main__":
    main()