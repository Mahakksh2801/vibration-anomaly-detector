"""
Replay all healthy + fault recordings to the dashboard in sequence.
Shows the complete project results on one screen.

Usage:
    python replay_all.py --host localhost --speed 4
    python replay_all.py --host 192.168.1.42 --speed 2
"""

import subprocess
import time
import argparse

RECORDINGS = [
    ("fan_speed1_test.npy", "Speed 1 - Healthy"),
    ("fan_speed1_fault.npy", "Speed 1 - Fault"),
    ("fan_speed2_test.npy", "Speed 2 - Healthy"),
    ("fan_speed2_fault.npy", "Speed 2 - Fault"),
    ("fan_speed3_test.npy", "Speed 3 - Healthy"),
    ("fan_speed3_fault.npy", "Speed 3 - Fault"),
]

def main():
    ap = argparse.ArgumentParser(description="Replay all project recordings")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between recordings")
    args = ap.parse_args()

    print(f"\nReplaying all 6 recordings to {args.host}:{args.port} at {args.speed}× speed\n")

    for filename, label in RECORDINGS:
        print(f"▶ {label}...")
        try:
            subprocess.run([
                "python", "mqtt_test.py",
                "--file", filename,
                "--host", args.host,
                "--port", str(args.port),
                "--speed", str(args.speed),
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"✗ Failed to replay {filename}")
            return

        print(f"✓ {label} done\n")
        time.sleep(args.pause)

    print("All recordings replayed. Refresh the dashboard to see the full history.")

if __name__ == "__main__":
    main()