import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

torch.manual_seed(0); np.random.seed(0)

# --- Rebuild 1 kHz features (same as mpu_preview.py) ---
import scipy.io as sio
from scipy.signal import decimate

FS, WINDOW = 1000, 512
BANDS = [(5, 50), (50, 120), (120, 250), (250, 500)]

def extract_features(signal, fs, window):
    n = len(signal) // window
    rows = []
    for i in range(n):
        seg = signal[i*window:(i+1)*window]
        rms = np.sqrt(np.mean(seg**2))
        spec = np.abs(np.fft.rfft(seg * np.hanning(window))) / window
        freqs = np.fft.rfftfreq(window, d=1/fs)
        pf = freqs[np.argmax(spec[1:]) + 1]
        be = [np.sum(spec[(freqs >= lo) & (freqs < hi)]**2) for lo, hi in BANDS]
        rows.append([rms, pf] + be)
    return np.array(rows)

healthy = decimate(sio.loadmat("97.mat")["X097_DE_time"].flatten(), 12)
faulty  = decimate(sio.loadmat("105.mat")["X105_DE_time"].flatten(), 12)
feat_h = extract_features(healthy, FS, WINDOW)
feat_f = extract_features(faulty, FS, WINDOW)

N_TRAIN = int(len(feat_h) * 0.7)
train_raw, test_h_raw, test_f_raw = feat_h[:N_TRAIN], feat_h[N_TRAIN:], feat_f

# ================= Detector 1: Mahalanobis =================
mu = train_raw.mean(axis=0)
cov = np.cov(train_raw, rowvar=False)
cov += np.eye(cov.shape[0]) * 1e-9 * np.trace(cov)
cov_inv = np.linalg.inv(cov)
maha = lambda A: np.sqrt(np.einsum('ij,jk,ik->i', A - mu, cov_inv, A - mu))

m_train, m_h, m_f = maha(train_raw), maha(test_h_raw), maha(test_f_raw)

# ================= Detector 2: Autoencoder =================
f_mean, f_std = train_raw.mean(axis=0), train_raw.std(axis=0) + 1e-9
norm = lambda A: (A - f_mean) / f_std
Xtr = torch.tensor(norm(train_raw), dtype=torch.float32)

model = nn.Sequential(nn.Linear(6,4), nn.ReLU(), nn.Linear(4,3), nn.ReLU(),
                      nn.Linear(3,4), nn.ReLU(), nn.Linear(4,6))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(2000):
    opt.zero_grad(); loss = ((model(Xtr) - Xtr)**2).mean(); loss.backward(); opt.step()

def ae_err(A):
    with torch.no_grad():
        t = torch.tensor(norm(A), dtype=torch.float32)
        return ((model(t) - t)**2).mean(dim=1).numpy()

a_train, a_h, a_f = ae_err(train_raw), ae_err(test_h_raw), ae_err(test_f_raw)

# ============ Fair thresholds: 99th percentile of training ============
thr_m = np.percentile(m_train, 99)
thr_a = np.percentile(a_train, 99)

def report(name, s_h, s_f, thr):
    fp, fn = np.sum(s_h > thr), np.sum(s_f <= thr)
    margin = s_f.min() / s_h.max()
    print(f"{name:<14} FP {fp}/{len(s_h)}  FN {fn}/{len(s_f)}  "
          f"healthy-max {s_h.max():.3g}  faulty-min {s_f.min():.3g}  margin {margin:.2f}x")
    return fp, fn

print(f"1 kHz features | train {N_TRAIN}, test_h {len(test_h_raw)}, test_f {len(test_f_raw)}")
report("Mahalanobis", m_h, m_f, thr_m)
report("Autoencoder", a_h, a_f, thr_a)

# ============ Threshold-free comparison: ROC / AUC ============
y_true  = np.concatenate([np.zeros(len(m_h)), np.ones(len(m_f))])
auc_m = roc_auc_score(y_true, np.concatenate([m_h, m_f]))
auc_a = roc_auc_score(y_true, np.concatenate([a_h, a_f]))
print(f"AUC  Mahalanobis: {auc_m:.4f}   Autoencoder: {auc_a:.4f}")

fpr_m, tpr_m, _ = roc_curve(y_true, np.concatenate([m_h, m_f]))
fpr_a, tpr_a, _ = roc_curve(y_true, np.concatenate([a_h, a_f]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(fpr_m, tpr_m, label=f"Mahalanobis (AUC {auc_m:.3f})", marker="o")
axes[0].plot(fpr_a, tpr_a, label=f"Autoencoder (AUC {auc_a:.3f})", marker="s")
axes[0].plot([0,1], [0,1], "k:", linewidth=0.8)
axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC — 1 kHz features"); axes[0].legend()

for scores, label, c in [(np.concatenate([m_h/thr_m, a_h/thr_a]), "", None)]:
    pass  # (placeholder—right panel below)

axes[1].scatter(m_h/thr_m, a_h/thr_a, label="Healthy (test)", color="tab:green", alpha=0.7)
axes[1].scatter(m_f/thr_m, a_f/thr_a, label="Faulty (test)", color="tab:red", alpha=0.7)
axes[1].axvline(1, color="k", linestyle="--", linewidth=0.8)
axes[1].axhline(1, color="k", linestyle="--", linewidth=0.8)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel("Mahalanobis score / its threshold")
axes[1].set_ylabel("AE error / its threshold")
axes[1].set_title("Per-window agreement map")
axes[1].legend()
plt.tight_layout()
plt.savefig("head_to_head.png", dpi=150)
plt.show()