"""
capture_serial.py - record the ESP32 1 kHz accelerometer stream to .npy
 
Usage (Windows example):
    python capture_serial.py --port COM5 --seconds 30 --out desk_quiet.npy
    python capture_serial.py --port COM5 --seconds 600 --out fan_speed1_healthy.npy --note "speed 1, no clip"
    python capture_serial.py --port COM5 --seconds 30 --out desk_phone.npy --plot
 
Output:
    <out>.npy        float32 array, shape (N, 3), columns = ax, ay, az in g, fs = 1000 Hz
    <out>.json       metadata: fs, gaps, bad lines, per-axis stats, your note
 
Requires: pip install pyserial numpy   (matplotlib only for --plot)
"""
import argparse
import json
import sys
import time
from datetime import datetime
 
import numpy as np
import serial
 
FS = 1000
LSB_PER_G = 16384.0
 
 
def reset_board(ser):
    # DevKitC auto-reset circuit: asserting RTS pulls EN low. Keep DTR off (DTR pulls IO0 = bootloader).
    ser.dtr = False
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False
 
 
def parse(line):
    parts = line.split(",")
    if len(parts) != 4:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=500000)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--out", required=True, help="output .npy filename")
    ap.add_argument("--note", default="", help="free text saved in the .json (speed, clip, mounting...)")
    ap.add_argument("--plot", action="store_true", help="show time + spectrum plots after capture")
    a = ap.parse_args()
 
    try:
        ser = serial.Serial(a.port, a.baud, timeout=2)
    except serial.SerialException as e:
        sys.exit(f"Could not open {a.port}: {e}\nIs Arduino Serial Monitor still open? Close it.")
 
    reset_board(ser)
 
    # Wait for the READY banner (or live data, if the board didn't reset)
    t0 = time.time()
    first = None
    while True:
        s = ser.readline().decode(errors="ignore").strip()
        if s.startswith("# READY"):
            print(s)
            break
        if s.startswith("# ERR"):
            sys.exit(s)
        v = parse(s)
        if v is not None:
            print("Board already streaming (no reset seen) - starting capture.")
            first = v
            break
        if time.time() - t0 > 10:
            sys.exit("No READY line in 10 s. Check port, baud (500000), and that C_mpu_stream is flashed. "
                     "Try pressing EN on the board.")
 
    n_target = int(a.seconds * FS)
    seqs, rows = [], []
    if first:
        seqs.append(first[0]); rows.append(first[1:])
    bad = i2c_err = 0
    t_start = time.time()
    print(f"Recording {a.seconds:.0f} s ({n_target} samples)...")
 
    while len(rows) < n_target:
        raw = ser.readline()
        if not raw:
            sys.exit("Serial timeout - the stream stopped. Check USB cable / wiring.")
        s = raw.decode(errors="ignore").strip()
        if not s:
            continue
        if s.startswith("#"):
            if "I2C_ERR" in s:
                i2c_err += 1
            else:
                print(s)
            continue
        v = parse(s)
        if v is None:
            bad += 1
            continue
        seqs.append(v[0]); rows.append(v[1:])
        if len(rows) % 10000 == 0:
            print(f"  {len(rows) / FS:5.0f} s captured")
 
    elapsed = time.time() - t_start
    ser.close()
 
    seqs = np.array(seqs, dtype=np.int64)
    d = np.diff(seqs)
    missed = int(np.sum(d[d > 1] - 1))
    g = np.array(rows, dtype=np.float32) / LSB_PER_G
 
    np.save(a.out, g)
 
    # Per-axis summary: gravity mean, vibration std, dominant frequency
    stats = {}
    print("\naxis   mean(g)    std(g)    dominant freq (Hz)")
    for i, name in enumerate(["ax", "ay", "az"]):
        x = g[:, i] - g[:, i].mean()
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        f = np.fft.rfftfreq(len(x), 1 / FS)
        spec[f < 3] = 0                      # ignore DC / very slow drift
        fpk = float(f[np.argmax(spec)])
        stats[name] = {"mean_g": float(g[:, i].mean()), "std_g": float(x.std()), "peak_hz": fpk}
        print(f"{name}   {g[:, i].mean():+8.4f}  {x.std():8.5f}   {fpk:8.1f}")
 
    meta = {
        "file": a.out, "fs": FS, "columns": ["ax", "ay", "az"], "units": "g",
        "samples": int(len(g)), "wall_seconds": round(elapsed, 2),
        "missed_samples": missed, "bad_lines": bad, "i2c_errors": i2c_err,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "note": a.note, "axes": stats,
    }
    meta_path = a.out.rsplit(".", 1)[0] + ".json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
 
    print(f"\nSaved {a.out} shape {g.shape} and {meta_path}")
    print(f"Effective rate: {len(g) / elapsed:.1f} samples/s (wall clock, expect ~1000)")
    loss = 100 * missed / max(1, len(g) + missed)
    flag = "OK" if loss < 0.1 else "WARNING: too many drops"
    print(f"Missed samples: {missed} ({loss:.3f}%)  bad lines: {bad}  I2C errors: {i2c_err}  -> {flag}")
 
    if a.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 7))
        t = np.arange(min(len(g), 2000)) / FS
        for i, name in enumerate(["ax", "ay", "az"]):
            ax[0].plot(t, g[:len(t), i] - g[:, i].mean(), label=name, lw=0.8)
            x = g[:, i] - g[:, i].mean()
            f = np.fft.rfftfreq(len(x), 1 / FS)
            ax[1].semilogy(f, np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-9, label=name, lw=0.8)
        ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("accel - mean (g)"); ax[0].legend()
        ax[1].set_xlabel("frequency (Hz)"); ax[1].set_ylabel("|FFT|"); ax[1].set_xlim(0, 500); ax[1].legend()
        ax[0].set_title(a.out)
        plt.tight_layout()
        plt.show()
 
 
if __name__ == "__main__":
    main()
 