"""Streamlit performance visualization dashboard."""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.metrics import training_metrics, driving_metrics

st.set_page_config(page_title="CARLA ML Driving - Performance", layout="wide")
st.title("ML-Based Autonomous Car Driving - Performance Visualization")


def safe_read_csv(path):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return None
        df = pd.read_csv(path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


# ---------- Sidebar ----------
st.sidebar.header("Data sources")
data_dir = st.sidebar.text_input("Data dir", str(config.DATA_DIR))
logs_dir = st.sidebar.text_input("Logs dir", str(config.LOGS_DIR))

# discover experiments
experiments = []
if os.path.isdir(logs_dir):
    for name in sorted(os.listdir(logs_dir)):
        d = os.path.join(logs_dir, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "training.csv")):
            experiments.append(name)

st.sidebar.header("Experiment")
if experiments:
    sel_exp = st.sidebar.selectbox("Experiment", experiments)
else:
    sel_exp = None
    st.sidebar.info("No experiment folders found under logs/.")

# discover drive logs
drive_logs = []
if os.path.isdir(logs_dir):
    for root, _, files in os.walk(logs_dir):
        for fn in files:
            if fn.startswith("drive_") and fn.endswith(".csv"):
                drive_logs.append(os.path.join(root, fn))

# ---------- KPI row ----------
st.header("Overview")
cols = st.columns(4)
if sel_exp:
    hist_path = os.path.join(logs_dir, sel_exp, "history.json")
    metrics_path = os.path.join(logs_dir, sel_exp, "metrics.json")
    if os.path.exists(hist_path):
        tm = training_metrics(hist_path)
        cols[0].metric("Best val loss", f"{tm['best_val_loss']:.4f}" if tm['best_val_loss'] is not None else "n/a")
        cols[1].metric("Best val MAE", f"{tm['best_val_mae']:.4f}" if tm['best_val_mae'] is not None else "n/a")
        cols[2].metric("Epochs", tm["epochs_run"])
        cols[3].metric("Train samples", tm["train_samples"])
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        st.json(m)
else:
    cols[0].metric("Best val loss", "n/a")
    st.info("Run training first to populate experiments.")

# ---------- Training curves ----------
st.header("Training curves")
if sel_exp:
    train_csv = os.path.join(logs_dir, sel_exp, "training.csv")
    tdf = safe_read_csv(train_csv)
    if tdf is not None:
        fig = go.Figure()
        if "loss" in tdf.columns:
            fig.add_trace(go.Scatter(y=tdf["loss"], name="train loss"))
        if "val_loss" in tdf.columns:
            fig.add_trace(go.Scatter(y=tdf["val_loss"], name="val loss"))
        fig.update_layout(title="Loss", xaxis_title="epoch", yaxis_title="MSE")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        if "mae" in tdf.columns:
            fig2.add_trace(go.Scatter(y=tdf["mae"], name="train mae"))
        if "val_mae" in tdf.columns:
            fig2.add_trace(go.Scatter(y=tdf["val_mae"], name="val mae"))
        fig2.update_layout(title="MAE", xaxis_title="epoch", yaxis_title="MAE")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write(f"No training.csv data for {sel_exp}.")

# ---------- Driving telemetry ----------
st.header("Driving telemetry")
if drive_logs:
    selected = st.selectbox("Telemetry file", drive_logs)
    ddf = safe_read_csv(selected)
    if ddf is not None:
        dm = driving_metrics(selected)
        if dm:
            c = st.columns(4)
            c[0].metric("Frames", dm["frames"])
            c[1].metric("Avg speed (km/h)", f"{dm['avg_speed_kmh']:.1f}")
            c[2].metric("Collisions", dm["collisions"])
            c[3].metric("Lane invasions", dm["lane_invasions"])
            c2 = st.columns(4)
            c2[0].metric("Distance (m)", f"{dm['distance_m']:.1f}")
            c2[1].metric("Steer smoothness", f"{dm['steering_smoothness_mad']:.4f}")
            c2[2].metric("Max speed", f"{dm['max_speed_kmh']:.1f}")
            c2[3].metric("Lane inv. rate", f"{dm['lane_invasion_rate']:.3f}")
        fig = px.line(ddf, y=[col for col in ["speed_kmh", "steering_pred", "throttle", "brake"] if col in ddf.columns],
                      title="Telemetry over time")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Trajectory")
        if "x" in ddf.columns and "y" in ddf.columns:
            figt = px.scatter(ddf, x="x", y="y",
                              color="speed_kmh" if "speed_kmh" in ddf.columns else None,
                              title="Vehicle trajectory")
            figt.update_traces(mode="lines+markers", marker_size=3)
            st.plotly_chart(figt, use_container_width=True)
else:
    st.write("No driving telemetry (drive_*.csv) found in logs dir.")

# ---------- Dataset overview ----------
st.header("Dataset overview")
candidates = []
if os.path.isdir(data_dir):
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn == "telemetry.csv":
                candidates.append(os.path.join(root, fn))
if candidates:
    dsel = st.selectbox("Dataset telemetry", candidates)
    ddf2 = safe_read_csv(dsel)
    if ddf2 is not None:
        st.write(f"Samples: {len(ddf2)}")
        if "steering" in ddf2.columns:
            fig = px.histogram(ddf2, x="steering", nbins=40, title="Steering distribution")
            st.plotly_chart(fig, use_container_width=True)
        if "speed_kmh" in ddf2.columns:
            fig = px.histogram(ddf2, x="speed_kmh", nbins=40, title="Speed distribution")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No dataset telemetry.csv found.")

st.sidebar.markdown("---")
st.sidebar.markdown("Train: `python -m src.train --data data/collected --experiment baseline --epochs 30`")
st.sidebar.markdown("Drive: `python -m src.drive --model models/baseline/model.keras --frames 1000`")
