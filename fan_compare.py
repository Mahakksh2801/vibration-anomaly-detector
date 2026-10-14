"""
fan_compare.py - healthy vs fault overview for the report.

Builds one summary table and one figure per speed showing the time trace and
the averaged spectrum with the healthy and fault runs overlaid. This is the
"the sensor can see the fault at all" evidence, independent of the detectors.

Usage (from Project/data):
    python fan_compare.py
"""

import argparse
import csv
import os

import numpy as np

import fan_features as ff

PAIRS = [
    ("off", "fan_off.npy", None),
    ("speed 1", "fan_speed1_test.npy", "fan_speed1_fault.npy"),
    ("speed 2", "fan_speed2_test.npy", "fan_speed2_fault.npy"),
    ("speed 3", "fan_speed3_test.npy", "fan_speed3_fault.npy"),
]
AXES = ("ax", "ay", "az")


def welch(x, nperseg=1024):
    """Averaged periodogram. Returns freq, power. Keeps scipy out of the deps."""
    win = np.hanning(nperseg)
    hop = nperseg // 2
    segs = [x[s:s + nperseg] for s in range(0, len(x) - nperseg + 1, hop)]
    acc = np.zeros(nperseg // 2 + 1)
    for seg in segs:
        acc += np.abs(np.fft.rfft((seg - seg.mean()) * win)) ** 2
    acc /= max(len(segs), 1)
    return np.fft.rfftfreq(nperseg, 1.0 / ff.FS), acc


def summarize(path):
    sig = ff.load_capture(path)
    meta = ff.load_meta(path)
    row = {"file": os.path.basename(path), "samples": len(sig),
           "note": meta.get("note", "")}
    for i, name in enumerate(AXES):
        x = sig[:, i]
        f, p = welch(x)
        band = f >= 5.0
        row[f"{name}_mean_g"] = round(float(x.mean()), 4)
        row[f"{name}_std_g"] = round(float(x.std()), 5)
        row[f"{name}_dom_hz"] = round(float(f[band][p[band].argmax()]), 1)
    return row, sig


def main():
    ap = argparse.ArgumentParser(description="Healthy vs fault overview")
    ap.add_argument("--data-dir", default=".")
    args = ap.parse_args()

    rows = []
    present = []
    for label, healthy, fault in PAIRS:
        for kind, fname in (("healthy", healthy), ("fault", fault)):
            if fname is None:
                continue
            path = os.path.join(args.data_dir, fname)
            if not os.path.exists(path):
                print(f"skipping {fname} (not found)")
                continue
            row, sig = summarize(path)
            row["case"] = label
            row["condition"] = kind
            rows.append(row)
            present.append((label, kind, sig))

    if not rows:
        raise SystemExit("no recordings found")

    head = f"{'case':9s} {'cond':8s} {'ax std':>8s} {'ay std':>8s} {'az std':>8s} {'dom Hz':>7s}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['case']:9s} {r['condition']:8s} {r['ax_std_g']:8.5f} "
              f"{r['ay_std_g']:8.5f} {r['az_std_g']:8.5f} {r['ax_dom_hz']:7.1f}")

    fields = ["case", "condition", "file", "samples", "note"] + [
        f"{a}_{m}" for a in AXES for m in ("mean_g", "std_g", "dom_hz")]
    out_csv = os.path.join(args.data_dir, "fan_data_summary.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nSaved fan_data_summary.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    speeds = sorted({lab for lab, kind, _ in present if lab != "off"})
    if not speeds:
        return
    fig, axes = plt.subplots(len(speeds), 2, figsize=(11, 3.2 * len(speeds)), squeeze=False)
    for r, speed in enumerate(speeds):
        for kind, color in (("healthy", "tab:blue"), ("fault", "tab:red")):
            match = [s for lab, k, s in present if lab == speed and k == kind]
            if not match:
                continue
            sig = match[0]
            t = np.arange(2000) / ff.FS
            axes[r][0].plot(t, sig[:2000, 0] - sig[:2000, 0].mean(),
                            color=color, lw=0.7, label=kind)
            f, p = welch(sig[:, 0])
            axes[r][1].semilogy(f, p, color=color, lw=0.8, label=kind)
        axes[r][0].set_title(f"{speed} - ax time trace")
        axes[r][0].set_xlabel("time (s)")
        axes[r][0].set_ylabel("g")
        axes[r][0].legend(fontsize=8)
        axes[r][0].grid(alpha=0.3)
        axes[r][1].set_title(f"{speed} - ax spectrum")
        axes[r][1].set_xlabel("frequency (Hz)")
        axes[r][1].set_xlim(0, 300)
        axes[r][1].legend(fontsize=8)
        axes[r][1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.data_dir, "fan_healthy_vs_fault.png"), dpi=130)
    print("Saved fan_healthy_vs_fault.png")


if __name__ == "__main__":
    main()