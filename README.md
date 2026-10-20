# AI-Based Autonomous Driving Using Behavioral Cloning

University mini-project: collect human driving data in CARLA, train a CNN to predict steering (behavioral cloning), run autonomous inference, and compare **baseline vs. augmented** training.

Research question: *How do training data quality and image augmentation affect steering prediction accuracy and driving performance in a CNN-based behavioral-cloning model?*

## Tech stack

| Layer | Choice |
|---|---|
| Simulator | CARLA 0.9.16 (Unreal Engine, 3D) |
| Python | 3.12 (required by CARLA wheel) |
| ML | TensorFlow/Keras |
| CV/IO | OpenCV, NumPy, Pandas |
| Viz | Matplotlib, Streamlit + Plotly (optional) |

## CARLA control (use the provided examples)

This project does **not** reimplement driving controls. It relies on CARLA's own Python API examples:

- `PythonAPI/examples/manual_control.py` — human-driven vehicle (spawns the `hero` car you drive with WASD/arrows).
- `PythonAPI/examples/generate_traffic.py` — spawns background traffic in autopilot.

Start the simulator server first, then use these examples for car control.

## Setup

```bash
conda create -n carla python=3.12 -y
conda activate carla
pip install PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl
pip install -r requirements.txt
```

## Run order

### 0. Launch CARLA server
```bash
CarlaUE4.exe
```

### 1a. Collect human driving data (manual mode)
Terminal 1 — background traffic:
```bash
python PythonAPI/examples/generate_traffic.py -n 20
```
Terminal 2 — drive the car yourself:
```bash
python PythonAPI/examples/manual_control.py
```
Terminal 3 — record data from the hero vehicle (no duplicate car, no control takeover):
```bash
python -m src.collect_data --mode manual --frames 5000 --out data/collected
```

### 1b. Or collect automated expert data (autopilot mode)
```bash
python -m src.collect_data --mode autopilot --frames 5000 --out data/collected --traffic 20
```

### 2. Train baseline
```bash
python -m src.train --data data/collected --experiment baseline --epochs 30
```

### 3. Train augmented (same split, same settings)
```bash
python -m src.train --data data/collected --experiment augmented --augment --epochs 30
```

### 4. Evaluate
```bash
python -m src.evaluate --model models/baseline/model.keras --data data/collected --out reports/baseline/eval
python -m src.evaluate --model models/augmented/model.keras --data data/collected --out reports/augmented/eval
```

### 5. Autonomous driving
```bash
python -m src.drive --model models/augmented/model.keras --frames 1000
```
Manual override: create a file `logs/STOP` or press Ctrl+C (safe shutdown applies full brake).

### 6. Dashboard (optional)
```bash
streamlit run app/dashboard.py
```

### Smoke test (no CARLA)
```bash
python run_pipeline.py
python -m pytest -q
```

### Hyperparameter tuning
```bash
python -m src.tune --data data/collected --lrs 1e-2,1e-3,1e-4 --epochs 10
```

### Multi-map / weather collection
```bash
python -m src.collect_data --mode autopilot --frames 5000 --out data/collected \\
  --maps Town01,Town03,Town05 --weathers ClearNoon,CloudyNoon,SoftRainNoon --traffic 20
```

### Multi-sensor (RGB + depth)
```bash
python -m src.collect_data --mode autopilot --frames 2000 --out data/collected --depth --traffic 20
python -m src.train --data data/collected --experiment multisensor --depth --model multisensor --epochs 30
```

### Reinforcement-learning baseline
```bash
python -m src.rl_baseline --episodes 5 --steps 200
```

## Limitation note (manual labels)
When recording `manual_control.py` driving, this project reads `vehicle.get_control()` from the hero actor. That returns the last control applied by the manual-control process, aligned to the next world tick, so labels have an approximate one-tick (~0.05 s) delay. The collector never fabricates labels; if no hero vehicle exists it errors out. For perfectly synchronized labels, run the collector as the control owner (autopilot mode) or extend `manual_control.py` to stream controls (not done here to avoid modifying CARLA examples).

## Project layout

```
config.py
src/
  carla_client.py      CARLA connection & sensors
  collect_data.py      manual + autopilot collection
  synthetic_data.py    offline synthetic dataset
  preprocess.py        crop/resize/normalize + augmentation
  model.py             CNN architectures
  train.py             baseline/augmented training + experiment tracking
  evaluate.py          metrics + plots
  tune.py              hyperparameter grid search
  rl_baseline.py       DQN reinforcement-learning baseline
  drive.py             autonomous inference
  metrics.py           KPIs
  utils.py
app/dashboard.py       Streamlit dashboard
tests/                 pytest smoke tests
```

## Experiments
Both runs use an identical, persisted 70/15/15 split (`data/splits.json`). Only the training set is augmented; validation/test stay untouched. Each run saves config, history, metrics and plots under `logs/<experiment>/` and `reports/<experiment>/`.

## Git
Datasets, weights, logs, venvs and CARLA binaries are gitignored. Commit source only.
