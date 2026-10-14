"""
fan_evaluate.py - score healthy vs fault fan data with both detectors.

Produces the numbers and figures that go straight into the report:
  * per-speed detection rate at the trained threshold
  * ROC AUC for each detector
  * separation margin (median fault score / threshold) - the headline metric
    you already used for the CWRU comparison in Stage 2

Usage (from Project/data):
    python fan_evaluate.py
    python fan_evaluate.py --baseline fan_baseline.npz

Outputs:
    fan_results.csv
    fan_detector_scores.png
    fan_roc.png
"""

import argparse
import csv
import os

import numpy as np

import fan_features as ff
from fan_train import ae_score, mahalanobis_score

PAIRS = [
    ("speed 1", "fan_speed1_test.npy", "fan_speed1_fault.npy"),
    ("speed 2", "fan_speed2_test.npy", "fan_speed2_fault.npy"),
    ("speed 3", "fan_speed3_test.npy", "fan_speed3_fault.npy"),
]


def roc_auc(neg, pos):
    """AUC via the rank-sum identity. No sklearn dependency."""
    scores = np.concatenate([neg, pos])
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = len(pos), len(neg)
    r_pos = ranks[n_neg:].sum()
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser(description="Evaluate both detectors on fan healthy vs fault data")
    ap.add_argument("--baseline", default="fan_baseline.npz")
    ap.add_argument("--data-dir", default=".")
    args = ap.parse_args()

    bl_path = os.path.join(args.data_dir, args.baseline)
    if not os.path.exists(bl_path):
        raise SystemExit(f"{args.baseline} not found - run fan_train.py first")
    bl = np.load(bl_path, allow_pickle=True)

    mu, sigma, A = bl["mu"], bl["sigma"], bl["A"]
    params = {k: bl[k] for k in ("W1", "b1", "W2", "b2")}
    maha_thr, ae_thr = float(bl["maha_thr"]), float(bl["ae_thr"])

    def score(path):
        z = (ff.extract_file(path) - mu) / sigma
        return mahalanobis_score(z, A), ae_score(params, z)

    rows = []
    collected = {}
    print(f"thresholds   mahalanobis d^2 > {maha_thr:.2f}    autoencoder MSE > {ae_thr:.5f}\n")
    header = f"{'case':10s} {'n':>5s} {'maha med':>10s} {'maha det%':>10s} {'AE med':>10s} {'AE det%':>9s}"
    print(header)
    print("-" * len(header))

    for label, healthy_file, fault_file in PAIRS:
        hp = os.path.join(args.data_dir, healthy_file)
        fp = os.path.join(args.data_dir, fault_file)
        if not (os.path.exists(hp) and os.path.exists(fp)):
            print(f"{label:10s} skipped (missing recording)")
            continue

        mh, ah = score(hp)
        mf, af = score(fp)
        collected[label] = (mh, mf, ah, af)

        for kind, m, a in (("healthy", mh, ah), ("fault", mf, af)):
            m_det = 100.0 * np.mean(m > maha_thr)
            a_det = 100.0 * np.mean(a > ae_thr)
            print(f"{label + ' ' + kind:18s} {len(m):5d} "
                  f"{np.median(m):10.2f} {m_det:9.1f}% {np.median(a):10.5f} {a_det:8.1f}%")
            rows.append({
                "case": label, "condition": kind, "windows": len(m),
                "maha_median": round(float(np.median(m)), 4),
                "maha_detect_pct": round(float(m_det), 2),
                "ae_median": round(float(np.median(a)), 6),
                "ae_detect_pct": round(float(a_det), 2),
            })

        m_auc = roc_auc(mh, mf)
        a_auc = roc_auc(ah, af)
        m_margin = np.median(mf) / maha_thr
        a_margin = np.median(af) / ae_thr
        print(f"{'':10s} AUC  mahalanobis {m_auc:.4f}   autoencoder {a_auc:.4f}")
        print(f"{'':10s} margin (median fault / threshold)  "
              f"mahalanobis {m_margin:.2f}x   autoencoder {a_margin:.2f}x\n")
        rows[-1]["maha_auc"] = round(float(m_auc), 4)
        rows[-1]["ae_auc"] = round(float(a_auc), 4)
        rows[-1]["maha_margin"] = round(float(m_margin), 3)
        rows[-1]["ae_margin"] = round(float(a_margin), 3)

    if not rows:
        raise SystemExit("no healthy/fault pairs found")

    out_csv = os.path.join(args.data_dir, "fan_results.csv")
    fields = ["case", "condition", "windows", "maha_median", "maha_detect_pct",
              "ae_median", "ae_detect_pct", "maha_auc", "ae_auc", "maha_margin", "ae_margin"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Saved fan_results.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(collected)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7), squeeze=False)
    for col, (label, (mh, mf, ah, af)) in enumerate(collected.items()):
        ax = axes[0][col]
        ax.semilogy(mh, ".", ms=3, label="healthy")
        ax.semilogy(np.arange(len(mf)) + len(mh), mf, ".", ms=3, label="fault")
        ax.axhline(maha_thr, color="k", ls="--", lw=1, label="threshold")
        ax.set_title(f"{label} - Mahalanobis")
        ax.set_ylabel("d$^2$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1][col]
        ax.semilogy(ah, ".", ms=3, label="healthy")
        ax.semilogy(np.arange(len(af)) + len(ah), af, ".", ms=3, label="fault")
        ax.axhline(ae_thr, color="k", ls="--", lw=1, label="threshold")
        ax.set_title(f"{label} - autoencoder")
        ax.set_ylabel("reconstruction MSE")
        ax.set_xlabel("window index")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.data_dir, "fan_detector_scores.png"), dpi=130)
    print("Saved fan_detector_scores.png")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, (mh, mf, ah, af) in collected.items():
        for name, neg, pos, style in (("maha", mh, mf, "-"), ("AE", ah, af, "--")):
            thresholds = np.unique(np.concatenate([neg, pos]))
            tpr = [np.mean(pos > t) for t in thresholds]
            fpr = [np.mean(neg > t) for t in thresholds]
            ax.plot(fpr, tpr, style, lw=1.4, label=f"{label} {name}")
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC - fan rig")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.data_dir, "fan_roc.png"), dpi=130)
    print("Saved fan_roc.png")


if __name__ == "__main__":
    main()