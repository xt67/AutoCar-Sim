# AI-Based Autonomous Driving Using Behavioral Cloning

Collect human driving data in CARLA, train a CNN to predict steering, and compare **baseline vs. augmented** training.

![Demo](docs/images/demo.gif)

## Historical results (original run)

| Metric | Baseline | Augmented |
|---|---|---|
| Test MAE (steering) | 0.0124 | 0.0131 |
| Test MSE (steering) | 0.00073 | 0.00072 |
| Autonomous drive distance (300 frames) | 93.2 m | 88.9 m |
| Autonomous drive collisions (300 frames) | 0 | 0 |
| Autonomous drive lane invasions (300 frames) | 9 | 4 |
| Avg speed (km/h) | 22.4 | 21.4 |
| Steering smoothness (MAD) | 0.0123 | 0.0191 |

Augmentation slightly reduced test MSE but increased test MAE. In this short driving run it reduced lane invasions, while the baseline travelled farther. Longer repeated runs are needed to establish a reliable advantage.

## Dashboard

![Dashboard](docs/images/screenshot_dashboard.png)

```bash
streamlit run app/dashboard.py
```

## 3D Visualization

![3D Visualization](docs/images/screenshot_3d.png)

```bash
python -m src.visualize_3d
```

## Training Curves

![Training](docs/images/screenshot_training.png)

## Setup

### Dashboard only (Streamlit Cloud / lightweight)
```bash
pip install -r requirements.txt
```

### Full pipeline (local, Python 3.12 required for CARLA)
```bash
conda create -n carla python=3.12 -y
conda activate carla
pip install PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl
pip install -r requirements-full.txt -c requirements-test.txt
```

## Run

### 1. Launch CARLA
```bash
CarlaUE4.exe
```

### 2. Collect data (pick one)

**Manual (you drive):**
```bash
python PythonAPI/examples/generate_traffic.py -n 20
python PythonAPI/examples/manual_control.py
python -m src.collect_data --mode manual --frames 5000 --out data/collected
```

**Autopilot (automated expert):**
```bash
python -m src.collect_data --mode autopilot --frames 5000 --out data/collected --traffic 20
```

### 3. Train
```bash
python -m src.train --data data/collected --experiment baseline --epochs 30
python -m src.train --data data/collected --experiment augmented --augment --epochs 30
```

### 4. Evaluate
```bash
python -m src.evaluate --model models/baseline/model.keras --data data/collected
python -m src.evaluate --model models/augmented/model.keras --data data/collected
```

### 5. Autonomous driving
```bash
python -m src.drive --model models/augmented/model.keras --frames 300
```

### Smoke test (no CARLA)
```bash
python run_pipeline.py
python -m pytest -q
```

## Project structure

```
config.py              paths & hyperparameters
src/
  carla_client.py      CARLA connection & sensors
  collect_data.py      manual + autopilot collection
  preprocess.py        crop/resize/normalize + augmentation
  model.py             NVIDIA CNN + variants
  train.py             baseline/augmented training
  evaluate.py          metrics + plots
  drive.py             autonomous inference
  metrics.py           KPIs
  tune.py              hyperparameter tuning
  rl_baseline.py       DQN baseline
  visualize_3d.py      3D visualization
app/dashboard.py       Streamlit dashboard
tests/                 pytest smoke tests
```

## Tech stack

CARLA 0.9.16 · Python 3.12 · TensorFlow/Keras · OpenCV · NumPy · Pandas · Matplotlib · Plotly · Streamlit

## Notes

- Training data, models, logs, and reports are gitignored (local only).
- GPU not used on native Windows (TF ≥2.11); CPU training is fast enough.
- Manual control labels are sampled alongside cached frame snapshots and explicitly marked approximate; use synchronous expert collection for aligned observation-to-next-control labels.
- Augmented model: slightly worse offline MAE but fewer lane invasions online.
- Both models: 0 collisions over 300 frames on Town10HD at ~22 km/h.

## Git

Source code, tests, and docs are on GitHub (`feature/behavioral-cloning` branch). Generated data/models/logs stay local.


## Release validation (11 November 2026)

See [review and measured results](PROJECT_REVIEW.md) and [raw CARLA runs](docs/validation/carla-20261111.json). In three new short, traffic-free runs per checkpoint, baseline completed all three without collision flags; augmented collided and stalled in all three. These runs supersede any implication that the older zero-collision table demonstrates general driving reliability. Original checkpoint split provenance is unknown.

![Verified dashboard](docs/images/release-dashboard.png)

New training writes a fingerprinted `splits.json` beside the checkpoint. Evaluation requires it by default and scores only held-out samples. For genuinely unseen external data, explicitly pass `--external-dataset`. Adjacent frames are no longer randomly shuffled between splits. Datasets with at least three recording IDs hold out whole recordings; legacy recordings use chronological splits with a default 30-frame embargo. Collect more independent drives for stronger generalization evidence.

Missing RGB/depth pairs now stop training with an actionable error. Old depth-trained checkpoints must be retrained after the channel-order fix. RGB/depth collection stores sensor frame IDs and simulation time. Collection refuses an existing telemetry file, so select a fresh output directory for each run.

```bash
python -m pytest tests -q
python -m src.benchmark --models models/baseline/model.keras models/augmented/model.keras --runs 3 --frames 300 --out reports/new-benchmark
```

Launch CARLA with `-RenderOffScreen` for background camera evaluation. CARLA no-rendering mode cannot supply camera input and is rejected by the driving command.

## License

AutoCar-Sim original contributions are licensed under the [MIT License](LICENSE). The existing CARLA copyright notice is preserved. Third-party dependencies, datasets and assets retain their respective terms.
