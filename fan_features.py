"""
fan_features.py - shared feature extraction for the fan rig.

Every other script in Stage 3 imports from here so that the Python
training path and the ESP32 firmware compute EXACTLY the same features.
If you change anything in this file you must re-run:
    fan_train.py  ->  export_header.py  ->  re-flash fan_monitor.ino

Feature vector layout (24 values, 8 per axis, axis order ax, ay, az):
    [0] rms            root mean square of the mean-removed window
    [1] p2p            peak-to-peak
    [2] kurtosis       4th moment / rms^4  (impulsiveness)
    [3] crest          peak / rms
    [4] log band 5-50 Hz      (fraction of spectral energy, natural log)
    [5] log band 50-100 Hz
    [6] log band 100-200 Hz
    [7] log band 200-500 Hz
"""

import json
import os

import numpy as np

FS = 1000.0          # sample rate, Hz  (must match C_mpu_stream.ino)
WINDOW = 256         # samples per window (0.256 s)
HOP = 128            # 50% overlap
BANDS = [(5.0, 50.0), (50.0, 100.0), (100.0, 200.0), (200.0, 500.0)]
N_FEAT_PER_AXIS = 4 + len(BANDS)
N_FEAT = 3 * N_FEAT_PER_AXIS
EPS = 1e-12

FEATURE_NAMES = [
    f"{ax}_{name}"
    for ax in ("ax", "ay", "az")
    for name in ("rms", "p2p", "kurt", "crest", "bE5_50", "bE50_100", "bE100_200", "bE200_500")
]

# Hann window, precomputed once. The ESP32 uses the same formula.
_HANN = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WINDOW) / WINDOW)
_FREQ = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
_BAND_MASKS = [(_FREQ >= lo) & (_FREQ < hi) for lo, hi in BANDS]


def load_capture(path):
    """Load a .npy produced by capture_serial.py and return float g, shape (N, 3).

    Handles both storage conventions:
      * raw int16 counts at +/-2 g full scale (16384 LSB/g)
      * already-converted float g
    """
    raw = np.load(path)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f"{path}: expected shape (N, 3), got {raw.shape}")
    x = raw.astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer) or np.abs(x).max() > 20.0:
        x = x / 16384.0          # +/-2 g full scale
    return x


def load_meta(path):
    """Load the sidecar .json written by capture_serial.py, or {} if absent."""
    meta_path = os.path.splitext(path)[0] + ".json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}
    return {}


def window_features(win):
    """Feature vector for one window. win: (WINDOW, 3) float g -> (24,) float."""
    out = np.empty(N_FEAT, dtype=np.float64)
    k = 0
    for axis in range(3):
        x = win[:, axis].astype(np.float64)
        x = x - x.mean()

        rms = np.sqrt(np.mean(x * x))
        peak = np.max(np.abs(x))
        out[k + 0] = rms
        out[k + 1] = x.max() - x.min()
        out[k + 2] = np.mean(x ** 4) / (rms ** 4 + EPS)
        out[k + 3] = peak / (rms + EPS)

        spec = np.abs(np.fft.rfft(x * _HANN)) ** 2
        total = spec.sum() + EPS
        for b, mask in enumerate(_BAND_MASKS):
            out[k + 4 + b] = np.log(spec[mask].sum() / total + EPS)
        k += N_FEAT_PER_AXIS
    return out


def extract(signal, window=WINDOW, hop=HOP):
    """Slide over a full recording. signal: (N, 3) float g -> (n_windows, 24)."""
    n = signal.shape[0]
    if n < window:
        raise ValueError(f"recording too short: {n} samples < window {window}")
    starts = range(0, n - window + 1, hop)
    return np.stack([window_features(signal[s:s + window]) for s in starts])


def extract_file(path):
    """Convenience: path to .npy -> (n_windows, 24) feature matrix."""
    return extract(load_capture(path))