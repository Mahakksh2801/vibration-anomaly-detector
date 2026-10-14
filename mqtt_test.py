"""
mqtt_test.py - replay a fan recording to the Stage 4 broker as if it were the ESP32.

Lets you prove the dashboard path works before the firmware is flashed, and
lets you demo the fault alarm without running the fan.

Usage:
    pip install paho-mqtt
    python mqtt_test.py --file fan_speed3_fault.npy --host localhost
    python mqtt_test.py --file fan_speed3_test.npy --speed 4
"""

import argparse
import json
import os
import time

import numpy as np

import fan_features as ff
from fan_train import ae_score, mahalanobis_score


def main():
    ap = argparse.ArgumentParser(description="Replay a recording to the MQTT broker")
    ap.add_argument("--file", default="fan_speed3_fault.npy")
    ap.add_argument("--baseline", default="fan_baseline.npz")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--device-id", default="fan_rig_01")
    ap.add_argument("--topic", default=None, help="default sensor/<device-id>/data")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    ap.add_argument("--dry-run", action="store_true", help="print payloads, do not connect")
    args = ap.parse_args()

    topic = args.topic or f"sensor/{args.device_id}/data"
    bl = np.load(os.path.join(args.data_dir, args.baseline), allow_pickle=True)
    feats = ff.extract_file(os.path.join(args.data_dir, args.file))
    z = (feats - bl["mu"]) / bl["sigma"]
    maha = mahalanobis_score(z, bl["A"])
    ae = ae_score({k: bl[k] for k in ("W1", "b1", "W2", "b2")}, z)
    maha_thr, ae_thr = float(bl["maha_thr"]), float(bl["ae_thr"])

    client = None
    if not args.dry_run:
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            raise SystemExit("pip install paho-mqtt (or use --dry-run)")
        client = mqtt.Client(client_id=f"{args.device_id}_replay")
        client.connect(args.host, args.port, 60)
        client.loop_start()
        print(f"connected to {args.host}:{args.port}, publishing to {topic}")

    period = (ff.HOP / ff.FS) / max(args.speed, 1e-3)
    streak = 0
    try:
        for i, (d2, mse) in enumerate(zip(maha, ae)):
            bad = d2 > maha_thr or mse > ae_thr
            streak = streak + 1 if bad else 0
            payload = {
                "device_id": args.device_id,
                "window": i,
                "maha": round(float(d2), 3),
                "maha_thr": round(maha_thr, 3),
                "ae": round(float(mse), 6),
                "ae_thr": round(ae_thr, 6),
                "rms_x": round(float(feats[i, 0]), 4),
                "rms_y": round(float(feats[i, 8]), 4),
                "rms_z": round(float(feats[i, 16]), 4),
                "anomaly": bool(streak >= 3),
                "i2c_errors": 0,
                "source": "replay:" + os.path.basename(args.file),
            }
            line = json.dumps(payload)
            if client:
                client.publish(topic, line)
            if i % 10 == 0 or payload["anomaly"]:
                print(f"w={i:4d} maha={d2:10.2f} ae={mse:9.5f} "
                      f"{'ANOM' if bad else 'ok'}{' [ALARM]' if payload['anomaly'] else ''}")
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if client:
            client.loop_stop()
            client.disconnect()
    print(f"published {len(maha)} windows")


if __name__ == "__main__":
    main()