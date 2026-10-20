# ML-Based Autonomous Car Driving Simulation and Performance Visualization

A complete pipeline: **CARLA simulator** -> data collection -> behavioral-cloning ML model -> autonomous driving -> performance visualization dashboard.

## Tech stack

| Layer | Choice |
|---|---|
| Simulator | CARLA 0.9.16 (Unreal Engine, 3D, sensors, weather, traffic) |
| Language | Python 3.12 (required by CARLA wheel) |
| ML | TensorFlow/Keras (NVIDIA end-to-end CNN) |
| CV/IO | OpenCV, Pillow, NumPy, Pandas |
| Viz | Streamlit + Plotly (+ Matplotlib) |
| Control | CARLA Traffic Manager / BehaviorAgent (expert) |

Why CARLA over Udacity Term-3 sim: CARLA gives a true 3D world with cameras, lidar, radar, GNSS, IMU, weather and traffic, plus a Python API and ready-made expert agents. Udacity's sim is lighter but camera-only and less flexible for "performance visualization" work. CARLA is already bundled in this folder.

> Note: TensorFlow >= 2.11 has no native GPU support on Windows. For GPU training use WSL2 + CUDA, or swap the Keras backend for PyTorch. CPU training works fine for the included small models.

## Setup

```bash
# 1. Python 3.12 environment (conda)
conda create -n carla python=3.12 -y
conda activate carla

# 2. Install CARLA python wheel
pip install PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl

# 3. Install ML deps
pip install -r requirements.txt
```

## Run order

### 0. Launch the simulator (server)
```bash
CarlaUE4.exe
```
Wait for the CARLA window to appear.

### 1. Collect expert driving data
```bash
python -m src.collect_data --frames 5000 --out data/collected --behavior normal --traffic 20
```

### Manual driving data (for behavioral cloning)

Start `manual_control.py` in one terminal. In a second terminal, from the
project root, run:

```bash
python -m src.collect_manual_data --output data/manual_collected/run_001
```

This attaches only a front camera to the already controlled `hero` vehicle; it
does not spawn or control a second vehicle. Images and training-compatible
`telemetry.csv` are saved in the named run folder. Stop the recorder with `Ctrl+C` in its own
terminal. Use `run_002`, `run_003`, and so on for later sessions.

### 2. Train the steering model
```bash
python -m src.train --data data/collected --model nvidia --epochs 30 --augment
```
Use `--model light` for a faster CPU run. Synthetic data (no CARLA) for a quick smoke test:
```bash
python -m src.synthetic_data --frames 2000 --out data/synthetic
python -m src.train --data data/synthetic --model light --epochs 5 --augment
```
Or run the whole demo pipeline:
```bash
python run_pipeline.py
```

### 3. Run autonomous driving
```bash
python -m src.drive --model models/steering_model.keras --frames 1000 --target-speed 25
```

### 4. Visualize performance
```bash
streamlit run app/dashboard.py
```

## Project layout

```
config.py                 central paths & hyperparameters
src/
  carla_client.py          CARLA connection, vehicle & sensors
  collect_data.py          expert (BehaviorAgent) data collection
  synthetic_data.py        offline synthetic dataset for testing
  preprocess.py            crop/resize/normalize + augmentation
  model.py                 NVIDIA & light CNNs
  train.py                 training pipeline + metrics
  drive.py                 autonomous driving client
  metrics.py               training/driving KPIs
  utils.py
app/dashboard.py           Streamlit performance dashboard
run_pipeline.py            end-to-end demo (synthetic)
```

## Performance metrics

Training: val MSE, val MAE, best epoch, training time.
Driving: avg/max speed, speed std, steering mean/std, steering smoothness (mean abs delta), distance traveled, collisions, lane invasions, collision/lane-invasion rates.

## Tips

- Synchronous mode + fixed delta (0.05s) keeps camera frames aligned with control labels.
- For more robust models, collect across multiple maps/weather: `--map Town03`, and vary weather via CARLA.
- Add traffic with `--traffic N` for harder scenarios.
- The dashboard reads any `telemetry.csv` under the data dir and any `*.csv` in the logs dir.
