import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

feat_h = np.load("features_healthy.npy")
feat_f = np.load("features_faulty.npy")

N_TRAIN = 40
train_raw, test_h_raw, test_f_raw = feat_h[:N_TRAIN], feat_h[N_TRAIN:], feat_f

# --- Normalize using TRAINING stats only ---
f_mean = train_raw.mean(axis=0)
f_std  = train_raw.std(axis=0) + 1e-9
train  = (train_raw  - f_mean) / f_std
test_h = (test_h_raw - f_mean) / f_std
test_f = (test_f_raw - f_mean) / f_std

X = torch.tensor(train, dtype=torch.float32)

# --- Tiny autoencoder: 6 -> 4 -> 3 -> 4 -> 6 ---
model = nn.Sequential(
    nn.Linear(6, 4), nn.ReLU(),
    nn.Linear(4, 3), nn.ReLU(),
    nn.Linear(3, 4), nn.ReLU(),
    nn.Linear(4, 6),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(2000):
    opt.zero_grad()
    loss = loss_fn(model(X), X)
    loss.backward()
    opt.step()
    if epoch % 400 == 0:
        print(f"epoch {epoch:4d}  train loss {loss.item():.5f}")

def recon_error(arr):
    with torch.no_grad():
        t = torch.tensor(arr, dtype=torch.float32)
        return ((model(t) - t)**2).mean(dim=1).numpy()

err_train = recon_error(train)
err_h     = recon_error(test_h)
err_f     = recon_error(test_f)
threshold = err_train.mean() + 3 * err_train.std()

print(f"\nTrain error  : mean {err_train.mean():.4f}, max {err_train.max():.4f}")
print(f"Test healthy : mean {err_h.mean():.4f}, max {err_h.max():.4f}")
print(f"Test faulty  : mean {err_f.mean():.4f}, min {err_f.min():.4f}")
print(f"Threshold    : {threshold:.4f}")
print(f"False positives: {np.sum(err_h > threshold)}/{len(err_h)}")
print(f"False negatives: {np.sum(err_f <= threshold)}/{len(err_f)}")

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params}  (~{n_params*4} bytes as float32)")

plt.figure(figsize=(10, 5))
n1, n2 = len(err_train), len(err_h)
plt.scatter(range(n1), err_train, label="Healthy (train)", alpha=0.7)
plt.scatter(range(n1, n1+n2), err_h, label="Healthy (test)", alpha=0.7, color="tab:green")
plt.scatter(range(n1+n2, n1+n2+len(err_f)), err_f, label="Faulty (test)", alpha=0.7, color="tab:red")
plt.axhline(threshold, color="k", linestyle="--", label=f"Threshold = {threshold:.3f}")
plt.yscale("log")
plt.xlabel("Window index")
plt.ylabel("Reconstruction error (log)")
plt.title("Anomaly scores: autoencoder detector")
plt.legend()
plt.tight_layout()
plt.savefig("autoencoder_scores.png", dpi=150)
plt.show()