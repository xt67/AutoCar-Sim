"""Hyperparameter tuning for behavioral cloning.

Runs a small grid over learning rate and augmentation strength, training each
config and recording metrics. Heavy: run on a GPU or overnight.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import ensure_dir, save_json


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning")
    parser.add_argument("--data", default=str(config.DATA_DIR / "collected"))
    parser.add_argument("--lrs", default="1e-2,1e-3,1e-4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--augment-levels", default="none,light,heavy",
                        help="none: no aug; light: flip+brightness; heavy: +blur+shadow")
    parser.add_argument("--out", default=str(config.LOGS_DIR / "tune"))
    args = parser.parse_args()

    ensure_dir(args.out)
    lrs = [float(x) for x in args.lrs.split(",")]
    levels = [x.strip() for x in args.augment_levels.split(",")]

    results = []
    for lr, level in itertools.product(lrs, levels):
        exp = f"tune_lr{lr}_aug{level}"
        cmd = [sys.executable, "-m", "src.train", "--data", args.data,
               "--experiment", exp, "--epochs", str(args.epochs),
               "--lr", str(lr), "--batch", str(config.BATCH_SIZE)]
        if level != "none":
            cmd.append("--augment")
        print(f"\n=== Running {exp} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=False)
        # read metrics
        mpath = os.path.join(str(config.LOGS_DIR), exp, "metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                m = json.load(f)
            results.append({"lr": lr, "augment": level, "metrics": m})
            print(f"  test MAE: {m['test']['mae']:.4f}")

    save_json(results, os.path.join(args.out, "tune_results.json"))
    # summary
    print("\n=== Tuning summary (sorted by test MAE) ===")
    results.sort(key=lambda r: r["metrics"]["test"]["mae"])
    for r in results:
        print(f"lr={r['lr']:.0e}  aug={r['augment']:6s}  "
              f"test_MAE={r['metrics']['test']['mae']:.4f}  "
              f"val_MAE={r['metrics']['val']['mae']:.4f}")


if __name__ == "__main__":
    main()
