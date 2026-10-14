import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

FS = 12000          # verified
WINDOW = 4096       # samples per window (~0.34 s)

# Frequency bands (Hz) for energy features — all within MPU-6050's future range
BANDS = [(5, 50), (50, 120), (120, 250), (250, 500)]

def extract_features(signal, fs, window):
    """Chop signal into windows, return one feature row per window."""
    n_windows = len(signal) // window
    rows = []
    for i in range(n_windows):
        seg = signal[i*window : (i+1)*window]

        rms = np.sqrt(np.mean(seg**2))

        spec = np.abs(np.fft.rfft(seg * np.hanning(window))) / window
        freqs = np.fft.rfftfreq(window, d=1/fs)

        peak_freq = freqs[np.argmax(spec[1:]) + 1]   # skip the 0 Hz bin

        band_energies = []
        for lo, hi in BANDS:
            mask = (freqs >= lo) & (freqs < hi)
            band_energies.append(np.sum(spec[mask]**2))

        rows.append([rms, peak_freq] + band_energies)
    return np.array(rows)

healthy = sio.loadmat("97.mat")["X097_DE_time"].flatten()
faulty  = sio.loadmat("105.mat")["X105_DE_time"].flatten()

feat_h = extract_features(healthy, FS, WINDOW)
feat_f = extract_features(faulty, FS, WINDOW)

names = ["RMS", "PeakFreq", "E_5-50", "E_50-120", "E_120-250", "E_250-500"]

print(f"Healthy: {feat_h.shape[0]} windows, Faulty: {feat_f.shape[0]} windows")
print(f"{'Feature':<12}{'Healthy mean':>14}{'Faulty mean':>14}{'Ratio':>8}")
for j, name in enumerate(names):
    h, f = feat_h[:, j].mean(), feat_f[:, j].mean()
    print(f"{name:<12}{h:>14.5f}{f:>14.5f}{f/h:>8.1f}")

np.save("features_healthy.npy", feat_h)
np.save("features_faulty.npy", feat_f)

plt.figure(figsize=(8, 6))
plt.scatter(feat_h[:, 0], feat_h[:, 4], label="Healthy", alpha=0.7)
plt.scatter(feat_f[:, 0], feat_f[:, 4], label="Faulty", alpha=0.7, color="tab:red")
plt.xlabel("RMS")
plt.ylabel("Energy 120-250 Hz")
plt.title("Feature space: healthy vs faulty windows")
plt.legend()
plt.tight_layout()
plt.savefig("feature_scatter.png", dpi=150)
plt.show()