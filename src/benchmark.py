"""Repeated CARLA evaluation of existing checkpoints without overwriting them."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from src.metrics import driving_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for model_index, model in enumerate(args.models):
        for run in range(args.runs):
            log = root / f"drive_model{model_index}_run{run}.csv"
            command = [sys.executable, "-m", "src.drive", "--model", model, "--frames", str(args.frames),
                       "--out", str(log), "--seed", str(42 + run), "--spawn-index", str(run), "--traffic", "0"]
            completed = subprocess.run(command, timeout=600)
            result = {"model":model, "sha256":hashlib.sha256(Path(model).read_bytes()).hexdigest(),
                      "seed":42+run, "spawn_index":run, "traffic":0, "returncode":completed.returncode,
                      "limitations":"Existing checkpoint; offline split provenance unverified. No traffic; short run."}
            if completed.returncode == 0:
                result["metrics"] = driving_metrics(log)
            results.append(result)
            (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            if completed.returncode:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
