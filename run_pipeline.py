"""One-command end-to-end demo on synthetic data (no CARLA needed)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(args):
    print(">>", " ".join(args))
    subprocess.run(args, check=True)


if __name__ == "__main__":
    data = ROOT / "data" / "synthetic"
    run([PY, "-m", "src.synthetic_data", "--frames", "2000", "--out", str(data)])
    run([PY, "-m", "src.train", "--data", str(data), "--epochs", "5",
         "--augment", "--model", "light"])
    print("\nPipeline complete. Launch dashboard with:")
    print("  streamlit run app/dashboard.py")
