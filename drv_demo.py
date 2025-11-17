#!/usr/bin/env python3
import sys, time, re, curses
import board, busio
from adafruit_bus_device.i2c_device import I2CDevice
import adafruit_drv2605

# ---------- Config ----------
MUX_ADDR = 0x70
CHANNELS = [0, 1, 2]      # 3 motors on mux channels 0,1,2
USE_LRA = True            # True for LRA, False for ERM
LRA_LIBRARY = 6           # 6 for LRA, 1 for ERM
DWELL_S_DEFAULT = 0.5
MAX_EFFECT = 123

# ---------- I2C + MUX ----------
i2c = busio.I2C(board.SCL, board.SDA)

def tca_select(ch: int):
    with I2CDevice(i2c, MUX_ADDR) as mux:
        mux.write(bytes([1 << ch]))
    time.sleep(0.002)

def make_drv_for_channel(ch: int, use_lra: bool, lib: int):
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

# ---------- App state ----------
running = True
mode = "repeat"          # "repeat", "scan", or "seq"
effect_id = 1
seq_triplet = None
dwell_s = DWELL_S_DEFAULT
last_tick = 0.0

HELP_LINES = [
  "Controls:",
  "  <number>      Set single effect and REPEAT mode (e.g., 70)",
  "  a             SCAN mode (auto-advance 1..123)",
  "  seq 1,2,3     Per-motor effects: CH0=1, CH1=2, CH2=3 (SEQ mode)",
  "  clearseq      Exit SEQ mode (back to REPEAT)",
  "  n / p         Next / Previous effect (REPEAT)",
  "  + / -         Increment / decrement effect",
  "  r             Reset effect to 1",
  "  d <sec>       Set dwell seconds (e.g., d 0.3)",
  "  space / s     Start/Stop toggle",
  "  h             Help",
  "  q             Quit",
]

def clamp_effect(e):
    if e < 1:  return MAX_EFFECT
    if e > MAX_EFFECT: return 1
    return e

def parse_seq_triplet(text: str):
    m = re.findall(r"\d+", text)
    if len(m) != 3:
        return None
    vals = list(map(int, m))
    return vals if all(1 <= v <= MAX_EFFECT for v in vals) else None

def trigger_all(effect_or_triplet):
    if isinstance(effect_or_triplet, list):
        effects = effect_or_triplet
    else:
        effects = [effect_or_triplet] * len(CHANNELS)

    for ch, drv, eff in zip(CHANNELS, drivers, effects):
        tca_select(ch)
        drv.sequence[0] = adafruit_drv2605.Effect(eff)
    for ch, drv in zip(CHANNELS, drivers):
        tca_select(ch)
        drv.play()

def handle_command(cmd: str, log_append):
    global running, mode, effect_id, dwell_s, seq_triplet
    if not cmd:
        return
    low = cmd.strip().lower()

    if low in (" ", "s"):
        running = not running
        log_append(f"[state] {'RUNNING' if running else 'STOPPED'}"); return
    if low == "a":
        mode = "scan"; running = True; seq_triplet = None
        log_append("[mode] SCAN"); return
    if low == "n":
        effect_id = clamp_effect(effect_id + 1); mode = "repeat"; seq_triplet = None
        log_append(f"[effect] {effect_id} (REPEAT)"); return
    if low == "p":
        effect_id = clamp_effect(effect_id - 1); mode = "repeat"; seq_triplet = None
        log_append(f"[effect] {effect_id} (REPEAT)"); return
    if low == "r":
        effect_id = 1; mode = "repeat"; seq_triplet = None
        log_append("[reset] effect -> 1"); return
    if low == "+":
        effect_id = clamp_effect(effect_id + 1)
        log_append(f"[effect] {effect_id}"); return
    if low == "-":
        effect_id = clamp_effect(effect_id - 1)
        log_append(f"[effect] {effect_id}"); return
    if low.startswith("d "):
        try:
            dwell_s = max(0.05, float(low.split()[1]))
            log_append(f"[info] dwell={dwell_s:.3f}s")
        except Exception:
            log_append("[warn] Usage: d <seconds>  (e.g., d 0.3)")
        return
    if low.startswith("seq"):
        trip = parse_seq_triplet(low)
        if trip:
            seq_triplet = trip; mode = "seq"; running = True
            log_append(f"[mode] SEQ {seq_triplet}")
        else:
            log_append("[warn] Usage: seq 1,2,3   (each 1..123)")
        return
    if low == "clearseq":
        seq_triplet = None; mode = "repeat"
        log_append("[mode] REPEAT"); return
    if low == "h":
        for line in HELP_LINES: log_append(line)
        return
    if low == "q":
        raise KeyboardInterrupt

    if low.isdigit():
        val = int(low)
        if 1 <= val <= MAX_EFFECT:
            effect_id = val; mode = "repeat"; running = True; seq_triplet = None
            log_append(f"[effect] {effect_id} (REPEAT)")
        else:
            log_append(f"[warn] effect must be 1..{MAX_EFFECT}")
        return

    log_append("[warn] Unknown command. Press 'h' for help.")

# ---------------- Curses UI ----------------
def run_curses(stdscr):
    global last_tick, effect_id

    curses.curs_set(1)              # show cursor in input line
    stdscr.nodelay(True)            # nonblocking getch()
    stdscr.keypad(True)
    curses.use_default_colors()

    # Layout:
    # - Top area: scrollback log (N-2 lines)
    # - Line N-1: status line (non-scroll)
    # - Line N: input line ">> ..."

    height, width = stdscr.getmaxyx()
    log_h = max(5, height - 2)
    log_win = curses.newwin(log_h, width, 0, 0)
    status_win = curses.newwin(1, width, log_h, 0)
    input_win  = curses.newwin(1, width, log_h + 1, 0)

    log_lines = []
    input_buf = ""
    last_status_draw = 0.0

    def log_append(s: str):
        nonlocal log_lines
        ts = time.strftime("%H:%M:%S")
        wrapped = []
        for line in s.splitlines() or [""]:
            # simple wrap
            while len(line) > width - 2:
                wrapped.append(line[:width-2])
                line = line[width-2:]
            wrapped.append(line)
        for w in wrapped:
            log_lines.append(f"{ts} {w}")
        if len(log_lines) > 1000:
            log_lines[:] = log_lines[-1000:]

    # print help once
    for line in HELP_LINES: log_append(line)

    def draw_log():
        log_win.erase()
        start = max(0, len(log_lines) - log_h)
        view = log_lines[start:]
        y = 0
        for line in view:
            log_win.addnstr(y, 0, line, width - 1)
            y += 1
        log_win.noutrefresh()

    def draw_status():
        status_win.erase()
        run = "RUN" if running else "STOP"
        m = mode.upper()
        eff = f"{seq_triplet}" if (mode == "seq" and seq_triplet) else f"{effect_id}"
        text = f"[{run}] mode={m} dwell={dwell_s:.2f}s effects={eff}"
        status_win.addnstr(0, 0, text.ljust(width - 1), width - 1)
        status_win.noutrefresh()

    def draw_input():
        input_win.erase()
        prompt = ">> "
        s = prompt + input_buf
        input_win.addnstr(0, 0, s, width - 1)
        # place cursor at end
        input_win.move(0, min(len(s), width - 1))
        input_win.noutrefresh()

    try:
        while True:
            # TICK: drive motors on schedule
            now = time.monotonic()
            if running and (now - last_tick) >= dwell_s:
                to_play = seq_triplet if (mode == "seq" and seq_triplet) else effect_id
                try:
                    trigger_all(to_play)
                except Exception as e:
                    log_append(f"[error] trigger: {e}")
                last_tick = now
                if mode == "scan":
                    effect_id = clamp_effect(effect_id + 1)

            # UI: read keys
            try:
                ch = stdscr.getch()
            except curses.error:
                ch = -1

            if ch != -1:
                if ch in (curses.KEY_ENTER, 10, 13):
                    cmd = input_buf.strip()
                    input_buf = ""
                    try:
                        handle_command(cmd, log_append)
                    except KeyboardInterrupt:
                        break
                elif ch in (curses.KEY_BACKSPACE, 127, 8):
                    input_buf = input_buf[:-1]
                elif ch == 21:  # Ctrl+U clear line
                    input_buf = ""
                elif ch == 3:   # Ctrl+C
                    break
                elif 0 <= ch < 256:
                    input_buf += chr(ch)

            # Draw (limit status redraw rate a bit)
            if now - last_status_draw > 0.01:
                draw_log()
                draw_status()
                draw_input()
                curses.doupdate()
                last_status_draw = now

            time.sleep(0.005)

    finally:
        # stop motors on exit
        for ch, drv in zip(CHANNELS, drivers):
            try:
                tca_select(ch); drv.stop()
            except Exception:
                pass

def main():
    curses.wrapper(run_curses)

if __name__ == "__main__":
    main()
