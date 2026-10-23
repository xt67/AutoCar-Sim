"""One-command launcher: starts CARLA server + Streamlit dashboard."""
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
CARLA_EXE = os.path.join(BASE, "CarlaUE4.exe")
PYTHON = sys.executable


def wait_for_carla(timeout=120):
    import carla
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            c = carla.Client("127.0.0.1", 2000)
            c.set_timeout(5)
            c.get_world()
            return True
        except Exception:
            time.sleep(3)
    return False


def main():
    # 1. Start CARLA server if not already running
    try:
        import carla
        c = carla.Client("127.0.0.1", 2000)
        c.set_timeout(3)
        c.get_world()
        print("[run_all] CARLA already running.")
    except Exception:
        if not os.path.exists(CARLA_EXE):
            print(f"[run_all] CarlaUE4.exe not found at {CARLA_EXE}")
            sys.exit(1)
        print("[run_all] Starting CARLA server...")
        subprocess.Popen([CARLA_EXE], cwd=BASE)
        print("[run_all] Waiting for CARLA to boot (up to 120s)...")
        if not wait_for_carla():
            print("[run_all] CARLA did not start in time.")
            sys.exit(1)
        print("[run_all] CARLA ready.")

    # 2. Start Streamlit dashboard
    print("[run_all] Starting Streamlit dashboard...")
    print("[run_all] Open http://localhost:8501 in your browser.")
    subprocess.run([PYTHON, "-m", "streamlit", "run", "app/dashboard.py"], cwd=BASE)


if __name__ == "__main__":
    main()
