#!/usr/bin/env python3
"""
Goal: Execute a time-stamped script that triggers specific DRV2605
motor effects on specific motors (via TCA9548A I2C mux) at precise times.

This uses the same "technology" as drv_demo.py: board/busio I2C, a TCA9548A
selector, and adafruit_drv2605 drivers. It is designed to be a building block
that later can be embedded into a larger real-time controller.

Supported input formats (per line, comments start with '#'):
  1) CSV:    time_s, channel, effect[, dwell_s]
     e.g.,   0.00, 0,  1, 0.50
             0.10, 1, 45
             1.20, 2, 12, 0.30

  2) KV:     at=<seconds> ch=<channel> eff=<effect> [dwell=<seconds>]
     e.g.,   at=0.00 ch=0 eff=1 dwell=0.50

Times are absolute from the moment the program starts running (t=0 at start).

Notes:
- "channel" refers to the TCA9548A mux channel for the motor (0..7).
- "effect" must be in the DRV2605 ROM effect range (typically 1..123).
- "dwell" is an optional pause after triggering the effect before the next
  command is considered; if omitted the global default dwell applies only as a
  pacing delay, not as a modifier of the effect itself.
- Simultaneous events (identical time stamps) are handled in input order.

Usage example:
  python motor_script_runner.py --file commands.txt \
      --channels 0,1,2 --mux-addr 0x70 --lra --lib 6 --default-dwell 0.10

Flags:
  --file           Path to the script file (required)
  --channels       Comma-sep list of mux channels that exist (default: 0,1,2)
  --mux-addr       I2C address of the TCA9548A (default: 0x70)
  --lra/--erm      Select LRA (library 6 by default) vs ERM (library 1)
  --lib            Force a specific DRV2605 library (int)
  --default-dwell  Default dwell seconds between triggers when a line omits it
  --dry-run        Do not touch hardware; print what would happen
  --start-delay    Delay (sec) before t=0 starts; useful to arm & step aside
  --skew-warn      Warn if we are late by more than this many ms (default 10)

"""
from __future__ import annotations
import argparse, time, sys, re
from dataclasses import dataclass
from typing import List, Optional

# --- Optional hardware deps (allow --dry-run without hardware) ---
try:
    import board, busio
    from adafruit_bus_device.i2c_device import I2CDevice
    import adafruit_drv2605
except Exception as e:
    board = busio = I2CDevice = adafruit_drv2605 = None  # type: ignore

MAX_EFFECT = 123

@dataclass
class Cmd:
    t: float      # absolute time (s) from program start
    ch: int       # mux channel (0..7)
    eff: int      # DRV2605 effect id (1..123)
    dwell: float  # pacing delay after this trigger (s)
    line_no: int  # for diagnostics
    raw: str      # original line

# --------- Parsing ---------
_csv_re = re.compile(r"^\s*([^#].*?)\s*$")
_kv_re  = re.compile(r"(\w+)\s*=\s*([^\s]+)")

def _parse_float(v: str, name: str, line_no: int) -> float:
    try:
        return float(v)
    except Exception:
        raise ValueError(f"line {line_no}: invalid {name}='{v}' (float expected)")

def _parse_int(v: str, name: str, line_no: int) -> int:
    try:
        return int(v)
    except Exception:
        raise ValueError(f"line {line_no}: invalid {name}='{v}' (int expected)")

def parse_script(path: str, default_dwell: float) -> List[Cmd]:
    cmds: List[Cmd] = []
    with open(path, 'r') as f:
        for i, raw in enumerate(f, start=1):
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            # CSV format? Split by commas if it looks like CSV
            if ',' in s and not ('=' in s):
                parts = [p.strip() for p in s.split(',')]
                if len(parts) < 3 or len(parts) > 4:
                    raise ValueError(f"line {i}: CSV needs 3 or 4 fields: time_s, ch, eff[, dwell_s]")
                t    = _parse_float(parts[0], 'time_s', i)
                ch   = _parse_int  (parts[1], 'ch', i)
                eff  = _parse_int  (parts[2], 'eff', i)
                dwell = _parse_float(parts[3], 'dwell', i) if len(parts) == 4 else default_dwell
                cmds.append(Cmd(t, ch, eff, dwell, i, raw.rstrip('\n')))
                continue
            # KV format
            kvs = dict((m.group(1).lower(), m.group(2)) for m in _kv_re.finditer(s))
            if not kvs:
                raise ValueError(f"line {i}: could not parse: {s}")
            missing = [k for k in ('at','ch','eff') if k not in kvs]
            if missing:
                raise ValueError(f"line {i}: missing keys {missing}; need at=<s> ch=<int> eff=<int>")
            t   = _parse_float(kvs['at'], 'at',  i)
            ch  = _parse_int  (kvs['ch'], 'ch',  i)
            eff = _parse_int  (kvs['eff'],'eff', i)
            dwell = _parse_float(kvs.get('dwell', str(default_dwell)), 'dwell', i)
            cmds.append(Cmd(t, ch, eff, dwell, i, raw.rstrip('\n')))
    # sort by time, keep stable order for simultaneous events
    cmds.sort(key=lambda c: c.t)
    return cmds

# --------- Hardware helpers ---------
class HW:
    def __init__(self, mux_addr: int, channels: List[int], use_lra: bool, lib: Optional[int], dry_run: bool):
        self.mux_addr = mux_addr
        self.channels = channels
        self.use_lra  = use_lra
        self.lib      = lib if lib is not None else (6 if use_lra else 1)
        self.dry_run  = dry_run
        self.i2c = None
        self._mux_dev = None
        self.drivers = {}
        if not dry_run:
            if board is None or busio is None or I2CDevice is None or adafruit_drv2605 is None:
                raise RuntimeError("Hardware libraries not available; run with --dry-run or install deps.")
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self._mux_dev = I2CDevice(self.i2c, mux_addr)
            for ch in channels:
                self.drivers[ch] = self._make_drv(ch)

    def _tca_select(self, channel: int):
        if self.dry_run:
            return
        if channel not in range(8):
            raise ValueError(f"Invalid mux channel {channel}")
        buf = bytearray([1 << channel])
        with self._mux_dev as dev:
            dev.write(buf)

    def _make_drv(self, ch: int):
        self._tca_select(ch)
        drv = adafruit_drv2605.DRV2605(self.i2c)
        # Configure library and mode
        drv.use_LRM() if self.use_lra else drv.use_ERM()
        try:
            drv.library = self.lib
        except Exception:
            pass  # some versions expose only default libs; ignore if unavailable
        # Simple: use slot 0 for single effect playback
        drv.sequence[0] = adafruit_drv2605.Pause(0)  # placeholder
        return drv

    def play_effect(self, ch: int, eff: int):
        if eff < 1 or eff > MAX_EFFECT:
            raise ValueError(f"Effect {eff} out of range 1..{MAX_EFFECT}")
        if self.dry_run:
            print(f"PLAY ch={ch} eff={eff}")
            return
        if ch not in self.drivers:
            raise ValueError(f"Channel {ch} not initialized (valid: {self.channels})")
        self._tca_select(ch)
        drv = self.drivers[ch]
        drv.sequence[0] = adafruit_drv2605.Effect(eff)
        drv.play()

    def stop_all(self):
        if self.dry_run:
            print("STOP ALL")
            return
        for ch, drv in self.drivers.items():
            try:
                self._tca_select(ch)
                drv.stop()
            except Exception:
                pass

# --------- Scheduler ---------

def run_schedule(cmds: List[Cmd], hw: HW, start_delay: float, skew_warn_ms: float):
    if start_delay > 0:
        print(f"Arming... starting in {start_delay:.2f}s")
        time.sleep(start_delay)
    t0 = time.monotonic()
    print(f"t=0 at {time.strftime('%H:%M:%S')}")
    for idx, c in enumerate(cmds):
        due = t0 + c.t
        now = time.monotonic()
        # Sleep until due (if in the past, run immediately and warn)
        if now < due:
            time.sleep(due - now)
        else:
            lateness_ms = (now - due) * 1000.0
            if lateness_ms > skew_warn_ms:
                print(f"[WARN] skew {lateness_ms:.1f}ms on line {c.line_no}: {c.raw}")
        # Execute
        ts = time.monotonic() - t0
        print(f"{ts:8.3f}s | ch={c.ch} eff={c.eff} (line {c.line_no})")
        hw.play_effect(c.ch, c.eff)
        # Optional dwell pacing
        if c.dwell > 0:
            time.sleep(c.dwell)
    hw.stop_all()

# --------- CLI ---------

def main(argv=None):
    p = argparse.ArgumentParser(description="Run scheduled DRV2605 effects from a script file")
    p.add_argument('--file', required=True, help='Script file path')
    p.add_argument('--channels', default='0,1,2', help='Comma‑sep mux channels that exist (default 0,1,2)')
    p.add_argument('--mux-addr', default='0x70', help='TCA9548A I2C address (default 0x70)')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--lra', action='store_true', help='Use LRA mode (default)')
    g.add_argument('--erm', action='store_true', help='Use ERM mode')
    p.add_argument('--lib', type=int, default=None, help='Force DRV library number (e.g., 6 for LRA, 1 for ERM)')
    p.add_argument('--default-dwell', type=float, default=0.10, help='Default dwell seconds for lines that omit dwell')
    p.add_argument('--dry-run', action='store_true', help='Do not touch hardware; print actions only')
    p.add_argument('--start-delay', type=float, default=0.0, help='Delay before treating t=0 (sec)')
    p.add_argument('--skew-warn', type=float, default=10.0, help='Warn if late by more than this many ms')

    args = p.parse_args(argv)

    try:
        channels = [int(x) for x in args.channels.split(',') if x.strip() != '']
    except Exception:
        p.error('--channels must be comma‑separated integers like 0,1,2')
    try:
        mux_addr = int(args.mux_addr, 0)
    except Exception:
        p.error('--mux-addr must be an int (e.g., 0x70)')

    use_lra = True
    if args.erm:
        use_lra = False
    elif args.lra:
        use_lra = True
    # else: default True

    # Parse file
    cmds = parse_script(args.file, args.default_dwell)
    if not cmds:
        print('No commands found; exiting.')
        return 0

    # Echo plan
    print('--- Plan ---')
    for c in cmds:
        print(f" t={c.t:7.3f}s  ch={c.ch}  eff={c.eff:3d}  dwell={c.dwell:.3f}s  # line {c.line_no}")
    print('------------')

    # Init hardware and run
    hw = HW(mux_addr=mux_addr, channels=channels, use_lra=use_lra, lib=args.lib, dry_run=args.dry_run)
    try:
        run_schedule(cmds, hw, args.start_delay, args.skew_warn)
        return 0
    except KeyboardInterrupt:
        print('\n[CTRL+C] stopping…')
        hw.stop_all()
        return 130

if __name__ == '__main__':
    sys.exit(main())
