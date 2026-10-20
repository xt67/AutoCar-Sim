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

## Setup

```bash
conda create -n carla python=3.12 -y
conda activate carla
pip install PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl
pip install -r requirements.txt
```

## Run order

### 0. Launch CARLA
```bash
CarlaUE4.exe
```

### 1a. Collect human driving data (manual mode)
In one terminal:
```bash
python PythonAPI/examples/manual_control.py
```
In another:
```bash
python -m src.collect_data --mode manual --frames 5000 --out data/collected
```
The collector attaches a front camera to the `hero` vehicle that `manual_control.py` drives. It does **not** spawn a vehicle or apply control. Labels come from `vehicle.get_control()` with a documented ~one-tick delay (see Limitation below).

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
