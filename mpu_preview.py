import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate

FS_ORIG = 12000
DECIM = 12                      # 12 kHz -> 1 kHz, like the MPU-6050
FS = FS_ORIG // DECIM           # 1000 Hz -> sees 0-500 Hz only
WINDOW = 512                    # ~0.51 s at 1 kHz, power of two

BANDS = [(5, 50), (50, 120), (120, 250), (250, 500)]

def extract_features(signal, fs, window):
    n_windows = len(signal) // window
    rows = []
    for i in range(n_windows):
        seg = signal[i*window : (i+1)*window]
        rms = np.sqrt(np.mean(seg**2))
        spec = np.abs(np.fft.rfft(seg * np.hanning(window))) / window
        freqs = np.fft.rfftfreq(window, d=1/fs)
        peak_freq = freqs[np.argmax(spec[1:]) + 1]
        band_energies = [np.sum(spec[(freqs >= lo) & (freqs < hi)]**2)
                         for lo, hi in BANDS]
        rows.append([rms, peak_freq] + band_energies)
    return np.array(rows)

healthy = sio.loadmat("97.mat")["X097_DE_time"].flatten()
faulty  = sio.loadmat("105.mat")["X105_DE_time"].flatten()

# Simulate the cheap sensor: anti-alias filter + downsample to 1 kHz
healthy_lo = decimate(healthy, DECIM)
faulty_lo  = decimate(faulty, DECIM)

feat_h = extract_features(healthy_lo, FS, WINDOW)
feat_f = extract_features(faulty_lo, FS, WINDOW)
print(f"Windows: healthy {len(feat_h)}, faulty {len(feat_f)}")

N_TRAIN = int(len(feat_h) * 0.7)
train, test_h, test_f = feat_h[:N_TRAIN], feat_h[N_TRAIN:], feat_f

mu = train.mean(axis=0)
cov = np.cov(train, rowvar=False)
cov += np.eye(cov.shape[0]) * 1e-9 * np.trace(cov)
cov_inv = np.linalg.inv(cov)

def mahalanobis(x):
    d = x - mu
    return float(np.sqrt(d @ cov_inv @ d))

score_train = np.array([mahalanobis(x) for x in train])
score_h = np.array([mahalanobis(x) for x in test_h])
score_f = np.array([mahalanobis(x) for x in test_f])
threshold = score_train.mean() + 3 * score_train.std()

print(f"Train scores : mean {score_train.mean():.2f}, max {score_train.max():.2f}")
print(f"Test healthy : mean {score_h.mean():.2f}, max {score_h.max():.2f}")
print(f"Test faulty  : mean {score_f.mean():.2f}, min {score_f.min():.2f}")
print(f"Threshold    : {threshold:.2f}")
print(f"False positives: {np.sum(score_h > threshold)}/{len(score_h)}")
print(f"False negatives: {np.sum(score_f <= threshold)}/{len(score_f)}")

plt.figure(figsize=(10, 5))
n1, n2 = len(score_train), len(score_h)
plt.scatter(range(n1), score_train, label="Healthy (train)", alpha=0.7)
plt.scatter(range(n1, n1+n2), score_h, label="Healthy (test)", alpha=0.7, color="tab:green")
plt.scatter(range(n1+n2, n1+n2+len(score_f)), score_f, label="Faulty (test)", alpha=0.7, color="tab:red")
plt.axhline(threshold, color="k", linestyle="--", label=f"Threshold = {threshold:.1f}")
plt.yscale("log")
plt.xlabel("Window index")
plt.ylabel("Mahalanobis distance (log)")
plt.title("Detector at 1 kHz sampling (MPU-6050 simulation)")
plt.legend()
plt.tight_layout()
plt.savefig("mpu_preview_scores.png", dpi=150)
plt.show()