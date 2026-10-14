"""
verify_header.py - prove baseline.h matches fan_baseline.npz before you flash.

Parses the generated C header, rebuilds the detectors from the parsed floats,
and re-scores a recording. If the max difference is ~1e-6 the export is good;
anything larger means export_header.py and fan_features.py have drifted apart.

Usage (from Project/data):
    python verify_header.py
    python verify_header.py --check fan_speed3_fault.npy
"""

import argparse
import os
import re

import numpy as np

import fan_features as ff
from fan_train import ae_score, mahalanobis_score


def parse_header(path):
    with open(path) as fh:
        text = fh.read()
    out = {}
    for name, size, body in re.findall(
            r"const float (\w+)\[(\d+)\] PROGMEM = \{(.*?)\};", text, re.S):
        vals = [float(v.rstrip("f")) for v in re.findall(r"-?[\d.eE+-]+f", body)]
        if len(vals) != int(size):
            raise SystemExit(f"{name}: declared {size} values, parsed {len(vals)}")
        out[name] = np.array(vals, dtype=np.float64)
    for name, val in re.findall(r"#define (BASE_\w+)\s+([-\d.eE+]+)f?", text):
        out[name] = float(val)
    return out


def main():
    ap = argparse.ArgumentParser(description="Verify baseline.h against fan_baseline.npz")
    ap.add_argument("--header", default="baseline.h")
    ap.add_argument("--baseline", default="fan_baseline.npz")
    ap.add_argument("--check", default="fan_speed3_fault.npy",
                    help="recording to score with both versions")
    ap.add_argument("--data-dir", default=".")
    args = ap.parse_args()

    hdr_path = os.path.join(args.data_dir, args.header)
    npz_path = os.path.join(args.data_dir, args.baseline)
    rec_path = os.path.join(args.data_dir, args.check)
    for p in (hdr_path, npz_path, rec_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    h = parse_header(hdr_path)
    bl = np.load(npz_path, allow_pickle=True)
    n = int(bl["n_feat"])
    lat = int(bl["latent"])

    print("shape and threshold checks")
    checks = [
        ("n_feat", int(h["BASE_N_FEAT"]), n),
        ("latent", int(h["BASE_LATENT"]), lat),
        ("window", int(h["BASE_WINDOW"]), int(bl["window"])),
        ("hop", int(h["BASE_HOP"]), int(bl["hop"])),
    ]
    ok = True
    for name, got, want in checks:
        flag = "ok" if got == want else "MISMATCH"
        ok &= got == want
        print(f"  {name:8s} header {got:6d}  npz {want:6d}  {flag}")

    feats = ff.extract_file(rec_path)

    # reference path, straight from the npz
    z_ref = (feats - bl["mu"]) / bl["sigma"]
    maha_ref = mahalanobis_score(z_ref, bl["A"])
    ae_ref = ae_score({k: bl[k] for k in ("W1", "b1", "W2", "b2")}, z_ref)

    # rebuilt path, straight from the parsed header
    z_hdr = (feats - h["BASE_MU"]) / h["BASE_SIGMA"]
    A = h["BASE_A"].reshape(n, n)
    maha_hdr = np.sum((z_hdr @ A.T) ** 2, axis=1)
    W1 = h["BASE_W1"].reshape(n, lat)
    W2 = h["BASE_W2"].reshape(lat, n)
    hid = np.tanh(z_hdr @ W1 + h["BASE_B1"])
    ae_hdr = np.mean((hid @ W2 + h["BASE_B2"] - z_hdr) ** 2, axis=1)

    dm = np.max(np.abs(maha_hdr - maha_ref) / (np.abs(maha_ref) + 1e-9))
    da = np.max(np.abs(ae_hdr - ae_ref) / (np.abs(ae_ref) + 1e-9))
    print(f"\nscored {os.path.basename(rec_path)}  ({len(feats)} windows)")
    print(f"  mahalanobis max relative error {dm:.2e}")
    print(f"  autoencoder max relative error {da:.2e}")

    agree_m = np.mean((maha_hdr > h["BASE_MAHA_THR"]) == (maha_ref > float(bl["maha_thr"])))
    agree_a = np.mean((ae_hdr > h["BASE_AE_THR"]) == (ae_ref > float(bl["ae_thr"])))
    print(f"  decision agreement   mahalanobis {agree_m:.1%}   autoencoder {agree_a:.1%}")

    good = ok and dm < 1e-5 and da < 1e-5 and agree_m == 1.0 and agree_a == 1.0
    print("\n" + ("PASS - safe to flash" if good else "FAIL - do not flash, re-run export_header.py"))
    raise SystemExit(0 if good else 1)


if __name__ == "__main__":
    main()