import os, tempfile, csv
import numpy as np

from src.metrics import driving_metrics


def test_driving_metrics():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame","timestamp","speed_kmh","steering_pred","throttle","brake",
                    "x","y","z","yaw","collision","lane_invasion"])
        for i in range(5):
            w.writerow([i, float(i), 20.0, 0.1, 0.5, 0.0, float(i), 0.0, 0.0, 0.0, 0, 0])
    m = driving_metrics(p)
    assert m["frames"] == 5
    assert m["collisions"] == 0
