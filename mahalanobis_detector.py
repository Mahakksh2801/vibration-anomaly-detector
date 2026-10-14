import numpy as np
import matplotlib.pyplot as plt

feat_h = np.load("features_healthy.npy")
feat_f = np.load("features_faulty.npy")

# --- Train/test split: detector may ONLY learn from healthy data ---
N_TRAIN = 40
train      = feat_h[:N_TRAIN]     # learn baseline from these
test_h     = feat_h[N_TRAIN:]     # unseen healthy — should score LOW
test_f     = feat_f               # faulty — should score HIGH

# --- "Training" = describing the healthy cloud ---
mu  = train.mean(axis=0)
cov = np.cov(train, rowvar=False)
cov += np.eye(cov.shape[0]) * 1e-9 * np.trace(cov)   # regularization
cov_inv = np.linalg.inv(cov)

def mahalanobis(x):
    d = x - mu
    return float(np.sqrt(d @ cov_inv @ d))

score_train = np.array([mahalanobis(x) for x in train])
score_h     = np.array([mahalanobis(x) for x in test_h])
score_f     = np.array([mahalanobis(x) for x in test_f])

# --- Threshold: set from healthy training scores ONLY ---
threshold = score_train.mean() + 3 * score_train.std()

print(f"Train scores    : mean {score_train.mean():.2f}, max {score_train.max():.2f}")
print(f"Test healthy    : mean {score_h.mean():.2f}, max {score_h.max():.2f}")
print(f"Test faulty     : mean {score_f.mean():.2f}, min {score_f.min():.2f}")
print(f"Threshold       : {threshold:.2f}")
fp = np.sum(score_h > threshold)
fn = np.sum(score_f <= threshold)
print(f"False positives : {fp}/{len(score_h)} healthy windows flagged")
print(f"False negatives : {fn}/{len(score_f)} faulty windows missed")

plt.figure(figsize=(10, 5))
x1 = np.arange(len(score_train))
x2 = np.arange(len(score_train), len(score_train) + len(score_h))
x3 = np.arange(len(score_train) + len(score_h),
               len(score_train) + len(score_h) + len(score_f))
plt.scatter(x1, score_train, label="Healthy (train)", alpha=0.7)
plt.scatter(x2, score_h, label="Healthy (test)", alpha=0.7, color="tab:green")
plt.scatter(x3, score_f, label="Faulty (test)", alpha=0.7, color="tab:red")
plt.axhline(threshold, color="k", linestyle="--", label=f"Threshold = {threshold:.1f}")
plt.yscale("log")
plt.xlabel("Window index")
plt.ylabel("Mahalanobis distance (log scale)")
plt.title("Anomaly scores: statistical drift detector")
plt.legend()
plt.tight_layout()
plt.savefig("mahalanobis_scores.png", dpi=150)
plt.show()