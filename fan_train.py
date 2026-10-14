"""
fan_train.py - train both detectors on HEALTHY fan data only.

This is the "self-baselining" step: the device never sees a fault during
training. It learns what normal looks like and flags everything else.

Usage (from Project/data):
    python fan_train.py
    python fan_train.py --healthy fan_speed1_test.npy fan_speed2_test.npy fan_speed3_test.npy
    python fan_train.py --latent 8 --epochs 400

Outputs:
    fan_baseline.npz   - everything export_header.py and fan_evaluate.py need
    train_curve.png    - autoencoder loss curve (sanity check)
"""

import argparse
import os

import numpy as np

import fan_features as ff

DEFAULT_HEALTHY = [
    "fan_speed1_test.npy",
    "fan_speed2_test.npy",
    "fan_speed3_test.npy",
]


# ---------------------------------------------------------------- Mahalanobis

def fit_mahalanobis(z, shrinkage=0.10):
    """Covariance with Ledoit-Wolf style shrinkage toward a diagonal target.

    Shrinkage matters here: 24 features from ~230 windows per speed is not a
    lot of data, and the raw covariance is close to singular. Blending in a
    scaled identity keeps the inverse stable.
    """
    cov = np.cov(z, rowvar=False)
    target = np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    cov = (1.0 - shrinkage) * cov + shrinkage * target
    # A = L^-1 from cov = L L^T, so that d^2 = ||A (x - mu)||^2
    L = np.linalg.cholesky(cov)
    A = np.linalg.solve(L, np.eye(cov.shape[0]))
    return cov, A


def mahalanobis_score(z, A):
    """Squared Mahalanobis distance for standardized rows z."""
    return np.sum((z @ A.T) ** 2, axis=1)


# ---------------------------------------------------------------- autoencoder

def init_ae(n_in, n_hidden, rng):
    """Xavier init for a 24 -> h -> 24 autoencoder (tanh encoder, linear decoder)."""
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / (n_in + n_hidden)), (n_in, n_hidden)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, np.sqrt(2.0 / (n_in + n_hidden)), (n_hidden, n_in)),
        "b2": np.zeros(n_in),
    }


def ae_forward(p, x):
    h = np.tanh(x @ p["W1"] + p["b1"])
    return h, h @ p["W2"] + p["b2"]


def ae_score(p, x):
    """Per-window reconstruction MSE."""
    _, xr = ae_forward(p, x)
    return np.mean((xr - x) ** 2, axis=1)


def train_ae(x, n_hidden=8, epochs=400, batch=32, lr=3e-3, seed=0, noise=0.02):
    """Plain-numpy denoising autoencoder with Adam. No TensorFlow needed.

    Keeping it numpy means the exported weights are trivially portable to C,
    and the whole model is 416 floats - small enough to live in ESP32 flash.
    """
    rng = np.random.default_rng(seed)
    p = init_ae(x.shape[1], n_hidden, rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    n = x.shape[0]
    history = []
    step = 0

    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb = x[idx]
            xin = xb + rng.normal(0, noise, xb.shape)   # denoising -> less overfit

            h = np.tanh(xin @ p["W1"] + p["b1"])
            xr = h @ p["W2"] + p["b2"]
            err = xr - xb
            loss = np.mean(err ** 2)
            epoch_loss += loss * len(idx)

            g = {}
            d_out = 2.0 * err / (err.shape[0] * err.shape[1])
            g["W2"] = h.T @ d_out
            g["b2"] = d_out.sum(axis=0)
            d_h = (d_out @ p["W2"].T) * (1.0 - h ** 2)
            g["W1"] = xin.T @ d_h
            g["b1"] = d_h.sum(axis=0)

            step += 1
            for k in p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mhat = m[k] / (1 - b1 ** step)
                vhat = v[k] / (1 - b2 ** step)
                p[k] -= lr * mhat / (np.sqrt(vhat) + eps)
        history.append(epoch_loss / n)
    return p, np.array(history)


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Train Mahalanobis + autoencoder on healthy fan data")
    ap.add_argument("--healthy", nargs="+", default=DEFAULT_HEALTHY,
                    help="healthy .npy recordings (one per speed)")
    ap.add_argument("--data-dir", default=".", help="folder holding the .npy files")
    ap.add_argument("--out", default="fan_baseline.npz")
    ap.add_argument("--latent", type=int, default=8, help="autoencoder hidden units")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="fraction of healthy windows held out for threshold setting")
    ap.add_argument("--fpr", type=float, default=0.01,
                    help="target false-positive rate on held-out healthy windows")
    ap.add_argument("--shrinkage", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = [os.path.join(args.data_dir, p) for p in args.healthy]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing healthy recordings: " + ", ".join(missing))

    print("Extracting features from healthy recordings")
    blocks, labels = [], []
    for path in paths:
        feats = ff.extract_file(path)
        blocks.append(feats)
        labels.extend([os.path.basename(path)] * len(feats))
        print(f"  {os.path.basename(path):28s} {feats.shape[0]:5d} windows")
    X = np.vstack(blocks)
    print(f"  total {X.shape[0]} windows x {X.shape[1]} features")

    # Train / validation split, stratified by stepping through the stack so
    # every speed contributes to both halves.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0])
    n_val = int(round(args.val_frac * X.shape[0]))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Xval = X[tr_idx], X[val_idx]

    # Standardize using TRAIN statistics only.
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma < 1e-9] = 1e-9
    Ztr = (Xtr - mu) / sigma
    Zval = (Xval - mu) / sigma

    print("\nFitting Mahalanobis model")
    cov, A = fit_mahalanobis(Ztr, shrinkage=args.shrinkage)
    maha_val = mahalanobis_score(Zval, A)
    maha_thr = float(np.quantile(maha_val, 1.0 - args.fpr))
    print(f"  healthy d^2  median {np.median(maha_val):8.2f}   "
          f"threshold @ {args.fpr:.0%} FPR = {maha_thr:.2f}")

    print(f"\nTraining autoencoder 24 -> {args.latent} -> 24 for {args.epochs} epochs")
    params, history = train_ae(Ztr, n_hidden=args.latent, epochs=args.epochs, seed=args.seed)
    ae_val = ae_score(params, Zval)
    ae_thr = float(np.quantile(ae_val, 1.0 - args.fpr))
    print(f"  final train MSE {history[-1]:.5f}")
    print(f"  healthy MSE  median {np.median(ae_val):8.5f}   "
          f"threshold @ {args.fpr:.0%} FPR = {ae_thr:.5f}")

    np.savez(
        os.path.join(args.data_dir, args.out),
        mu=mu, sigma=sigma, A=A, cov=cov,
        W1=params["W1"], b1=params["b1"], W2=params["W2"], b2=params["b2"],
        maha_thr=maha_thr, ae_thr=ae_thr,
        fs=ff.FS, window=ff.WINDOW, hop=ff.HOP,
        n_feat=ff.N_FEAT, latent=args.latent, fpr=args.fpr,
        healthy_files=np.array(args.healthy),
        history=history,
    )
    print(f"\nSaved {args.out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3.5))
        plt.semilogy(history)
        plt.xlabel("epoch")
        plt.ylabel("train MSE")
        plt.title("autoencoder training")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.data_dir, "train_curve.png"), dpi=130)
        print("Saved train_curve.png")
    except ImportError:
        pass

    print("\nNext: python fan_evaluate.py")


if __name__ == "__main__":
    main()