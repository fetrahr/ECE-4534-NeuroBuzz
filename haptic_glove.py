import time, board, busio
from adafruit_bus_device.i2c_device import I2CDevice
import adafruit_drv2605

MUX_ADDR = 0x70
MAX_EFFECT = 123

class HapticGlove:
    def __init__(self, channels=None, library=6):
        if channels is None:
            channels = [0,1,2,3,4]

        self.channels = channels
        # initialize I2C bus for future use
        self.i2c = busio.I2C(board.SCL, board.SDA)

        # initialize drivers for each motor
        self.drivers = []
        for ch in channels:
            self._tca_select(ch)
            drv = adafruit_drv2605.DRV2605(self.i2c)
            drv.use_LRM()
            self.drivers.append(drv)

    # select channel on the TCA9548A mux
    def _tca_select(self, ch: int):
        with I2CDevice(self.i2c, MUX_ADDR) as mux:
            mux.write(bytes([1 << ch]))
        time.sleep(0.001) # 1ms delay to allow mux to switch

    # clamp effect ID to valid range 1 to MAX_EFFECT
    def clamp_effect(self, e: int) -> int:
        return max(1, min(MAX_EFFECT, e))

    # play a single effect on one motor
    def play_effect(self, motor_idx: int, effect_id: int):
        effect_id = self.clamp_effect(effect_id)
        ch = self.channels[motor_idx]
        drv = self.drivers[motor_idx]

        self._tca_select(ch)
        drv.sequence[0] = adafruit_drv2605.Effect(effect_id)
        self._tca_select(ch)
        drv.play()

    # play effects on all motors, effect_ids can be a single ID or a list
    def play_all(self, effect_ids):
        # if single effect ID given, replicate for all motors
        if isinstance(effect_ids, int):
            effect_ids = [self.clamp_effect(effect_ids)] * len(self.channels)

        # program sequences
        for (ch, drv, eff) in zip(self.channels, self.drivers, effect_ids):
            eff = self.clamp_effect(eff)
            self._tca_select(ch)
            drv.sequence[0] = adafruit_drv2605.Effect(eff)

        # then trigger all
        for (ch, drv) in zip(self.channels, self.drivers):
            self._tca_select(ch)
            drv.play()

    # stop all motors
    def stop_all(self):
        for ch, drv in zip(self.channels, self.drivers):
            try:
                self._tca_select(ch)
                drv.stop()
            except Exception:
                pass

    # simple ripple pattern, each motor in sequence
    def ripple(self, effect_id: int, dwell_s: float = 0.1, repeat: int = 1):
        effect_id = self.clamp_effect(effect_id)

        for _ in range(repeat):
            for motor_idx in range(len(self.drivers)):
                self.play_effect(motor_idx, effect_id)
                time.sleep(dwell_s)