"""Streamlit performance visualization dashboard — modern dark theme."""
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

# ---------- Page config ----------
st.set_page_config(
    page_title="CARLA ML Driving",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    /* Dark modern theme */
    .stApp { background-color: #0d1117; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #e6edf3 !important; font-weight: 600; }
    .stMetric { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
                padding: 12px 16px; }
    .stMetric label { color: #8b949e !important; font-size: 0.85rem; }
    .stMetric [data-testid="stMetricValue"] { color: #e6edf3; font-size: 1.4rem; font-weight: 600; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .stSelectbox label, .stTextInput label { color: #8b949e; }
    div[data-testid="stExpander"] { background: #161b22; border: 1px solid #30363d; border-radius: 8px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
            padding: 20px; margin-bottom: 16px; }
    .card-title { color: #e6edf3; font-size: 1.1rem; font-weight: 600; margin-bottom: 12px; }
    hr { border-color: #30363d; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#e6edf3", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
)


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
with st.sidebar:
    st.markdown("## 🚗 CARLA ML Driving")
    st.markdown("---")
    data_dir = st.text_input("Data dir", str(config.DATA_DIR))
    logs_dir = st.text_input("Logs dir", str(config.LOGS_DIR))

    experiments = []
    if os.path.isdir(logs_dir):
        for name in sorted(os.listdir(logs_dir)):
            d = os.path.join(logs_dir, name)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "training.csv")):
                experiments.append(name)

    sel_exp = st.selectbox("Experiment", experiments) if experiments else None

    drive_logs = []
    if os.path.isdir(logs_dir):
        for root, _, files in os.walk(logs_dir):
            for fn in files:
                if fn.startswith("drive_") and fn.endswith(".csv"):
                    drive_logs.append(os.path.join(root, fn))

    st.markdown("---")
    st.caption("Train: `python -m src.train --data data/collected --experiment baseline --epochs 30`")
    st.caption("Drive: `python -m src.drive --model models/baseline/model.keras --frames 300`")


# ---------- Header ----------
st.markdown("# 🚗 ML-Based Autonomous Driving")
st.markdown("Performance visualization dashboard for behavioral cloning experiments.")
st.markdown("---")

# ---------- KPI Row ----------
if sel_exp:
    hist_path = os.path.join(logs_dir, sel_exp, "history.json")
    metrics_path = os.path.join(logs_dir, sel_exp, "metrics.json")

    col1, col2, col3, col4 = st.columns(4)
    if os.path.exists(hist_path):
        tm = training_metrics(hist_path)
        col1.metric("Best Val Loss", f"{tm['best_val_loss']:.4f}" if tm['best_val_loss'] is not None else "—")
        col2.metric("Best Val MAE", f"{tm['best_val_mae']:.4f}" if tm['best_val_mae'] is not None else "—")
        col3.metric("Epochs Run", tm["epochs_run"])
        col4.metric("Train Samples", f"{tm['train_samples']:,}")

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        st.markdown("#### Test-set results")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Test MAE", f"{m['test']['mae']:.4f}")
        mc2.metric("Test MSE", f"{m['test']['mse']:.4f}")
        mc3.metric("Val MAE", f"{m['val']['mae']:.4f}")
        mc4.metric("Val MSE", f"{m['val']['mse']:.4f}")

st.markdown("---")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Training", "🛣️ Driving", "📊 Dataset", "🧊 3D View", "📋 Experiment Info"])

# ---- Tab 1: Training ----
with tab1:
    if sel_exp:
        train_csv = os.path.join(logs_dir, sel_exp, "training.csv")
        tdf = safe_read_csv(train_csv)
        if tdf is not None:
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                if "loss" in tdf.columns:
                    fig.add_trace(go.Scatter(y=tdf["loss"], name="Train", line=dict(color="#58a6ff", width=2)))
                if "val_loss" in tdf.columns:
                    fig.add_trace(go.Scatter(y=tdf["val_loss"], name="Val", line=dict(color="#f0883e", width=2)))
                fig.update_layout(title="Loss (MSE)", xaxis_title="Epoch", yaxis_title="MSE", **PLOTLY_THEME)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure()
                if "mae" in tdf.columns:
                    fig2.add_trace(go.Scatter(y=tdf["mae"], name="Train", line=dict(color="#58a6ff", width=2)))
                if "val_mae" in tdf.columns:
                    fig2.add_trace(go.Scatter(y=tdf["val_mae"], name="Val", line=dict(color="#f0883e", width=2)))
                fig2.update_layout(title="MAE", xaxis_title="Epoch", yaxis_title="MAE", **PLOTLY_THEME)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info(f"No training.csv for {sel_exp}.")
    else:
        st.info("Select an experiment in the sidebar.")

# ---- Tab 2: Driving ----
with tab2:
    if drive_logs:
        selected = st.selectbox("Telemetry file", drive_logs, key="drive_sel")
        ddf = safe_read_csv(selected)
        if ddf is not None:
            dm = driving_metrics(selected)
            if dm:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Frames", f"{dm['frames']:,}")
                k2.metric("Avg Speed", f"{dm['avg_speed_kmh']:.1f} km/h")
                k3.metric("Collisions", dm["collisions"])
                k4.metric("Lane Inv.", dm["lane_invasions"])
                k5, k6, k7, k8 = st.columns(4)
                k5.metric("Distance", f"{dm['distance_m']:.1f} m")
                k6.metric("Max Speed", f"{dm['max_speed_kmh']:.1f} km/h")
                k7.metric("Steer Smoothness", f"{dm['steering_smoothness_mad']:.4f}")
                k8.metric("Lane Inv. Rate", f"{dm['lane_invasion_rate']:.3f}")

            # Telemetry chart
            available = [c for c in ["speed_kmh", "steering_pred", "throttle", "brake"] if c in ddf.columns]
            if available:
                fig = px.line(ddf, y=available, title="Telemetry Over Time")
                fig.update_layout(**PLOTLY_THEME)
                st.plotly_chart(fig, use_container_width=True)

            # Trajectory
            if "x" in ddf.columns and "y" in ddf.columns:
                figt = px.scatter(ddf, x="x", y="y",
                                  color="speed_kmh" if "speed_kmh" in ddf.columns else None,
                                  title="Vehicle Trajectory")
                figt.update_traces(mode="lines+markers", marker_size=3)
                figt.update_layout(**PLOTLY_THEME)
                st.plotly_chart(figt, use_container_width=True)
    else:
        st.info("No drive_*.csv files found in logs/.")

# ---- Tab 3: Dataset ----
with tab3:
    candidates = []
    if os.path.isdir(data_dir):
        for root, _, files in os.walk(data_dir):
            for fn in files:
                if fn == "telemetry.csv":
                    candidates.append(os.path.join(root, fn))
    if candidates:
        dsel = st.selectbox("Dataset", candidates, key="ds_sel")
        ddf2 = safe_read_csv(dsel)
        if ddf2 is not None:
            st.markdown(f"**Samples:** {len(ddf2):,}")
            c1, c2 = st.columns(2)
            with c1:
                if "steering" in ddf2.columns:
                    fig = px.histogram(ddf2, x="steering", nbins=40, title="Steering Distribution")
                    fig.update_layout(**PLOTLY_THEME)
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if "speed_kmh" in ddf2.columns:
                    fig = px.histogram(ddf2, x="speed_kmh", nbins=40, title="Speed Distribution")
                    fig.update_layout(**PLOTLY_THEME)
                    st.plotly_chart(fig, use_container_width=True)
            if "collision" in ddf2.columns:
                cc1, cc2 = st.columns(2)
                cc1.metric("Collisions", int(ddf2["collision"].sum()))
                if "lane_invasion" in ddf2.columns:
                    cc2.metric("Lane Invasions", int(ddf2["lane_invasion"].sum()))
    else:
        st.info("No telemetry.csv found in data dir.")

# ---- Tab 4: 3D View ----
with tab4:
    st.markdown("### 3D Visualizations")
    st.markdown("Interactive 3D views — rotate, zoom, and explore.")

    # 3D CNN Architecture
    st.markdown("#### CNN Architecture (NVIDIA End-to-End)")
    layers = [
        ("Input (RGB)", 66, 200, 3, "#58a6ff"),
        ("Conv1 24ch", 31, 98, 24, "#f0883e"),
        ("Conv2 36ch", 14, 47, 36, "#f0883e"),
        ("Conv3 48ch", 5, 22, 48, "#f0883e"),
        ("Conv4 64ch", 3, 20, 64, "#f85149"),
        ("Conv5 64ch", 1, 18, 64, "#f85149"),
        ("Flatten", 1, 1, 1152, "#8b949e"),
        ("Dense 100", 1, 1, 100, "#3fb950"),
        ("Dense 50", 1, 1, 50, "#3fb950"),
        ("Dense 10", 1, 1, 10, "#3fb950"),
        ("Steering", 1, 1, 1, "#f85149"),
    ]
    fig_cnn = go.Figure()
    xscale = 2.5
    for idx, (name, h, w, c, color) in enumerate(layers):
        hh = max(h / 8.0, 0.8)
        ww = max(w / 15.0, 0.8)
        x0 = idx * xscale
        x1 = x0 + 1.5
        # Box as mesh
        verts = [(x0,-hh/2,-ww/2),(x1,-hh/2,-ww/2),(x1,hh/2,-ww/2),(x0,hh/2,-ww/2),
                 (x0,-hh/2,ww/2),(x1,-hh/2,ww/2),(x1,hh/2,ww/2),(x0,hh/2,ww/2)]
        tris = [(0,1,2),(0,2,3),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(2,3,7),(2,7,6),(0,3,7),(0,7,4),(1,2,6),(1,6,5)]
        fig_cnn.add_trace(go.Mesh3d(
            x=[v[0] for v in verts], y=[v[1] for v in verts], z=[v[2] for v in verts],
            i=[t[0] for t in tris], j=[t[1] for t in tris], k=[t[2] for t in tris],
            color=color, opacity=0.7, name=f"{name} (H={h} W={w} C={c})",
            hoverinfo="name", flatshading=True,
        ))
        fig_cnn.add_trace(go.Scatter3d(
            x=[(x0+x1)/2], y=[hh/2+0.8], z=[0],
            mode="text", text=[f"<b>{name}</b><br>H={h} W={w} C={c}"],
            textfont=dict(size=10, color="#ffffff"), showlegend=False, hoverinfo="none",
        ))
        if idx < len(layers) - 1:
            fig_cnn.add_trace(go.Scatter3d(
                x=[x1, x1+(xscale-1.5)], y=[0, 0], z=[0, 0],
                mode="lines", line=dict(color="#484f58", width=2, dash="dot"),
                showlegend=False, hoverinfo="none",
            ))
    fig_cnn.update_layout(
        title="NVIDIA End-to-End CNN Architecture",
        scene=dict(
            xaxis=dict(title="Layer", showgrid=False, color="#e6edf3"),
            yaxis=dict(title="Height", showgrid=True, gridcolor="#30363d", color="#e6edf3"),
            zaxis=dict(title="Width", showgrid=True, gridcolor="#30363d", color="#e6edf3"),
            aspectmode="manual", aspectratio=dict(x=2.5, y=1, z=1),
            camera=dict(eye=dict(x=1.6, y=1.0, z=1.2)),
            bgcolor="#0d1117",
        ),
        **{k: v for k, v in PLOTLY_THEME.items() if k not in ("margin",)},
        height=600, showlegend=True,
        legend=dict(font=dict(size=9, color="#e6edf3"), bgcolor="rgba(13,17,23,0.8)"),
    )
    st.plotly_chart(fig_cnn, use_container_width=True)

    # 3D Trajectory from drive data
    st.markdown("#### 3D Driving Trajectory")
    if drive_logs:
        sel_traj = st.selectbox("Telemetry file", drive_logs, key="traj3d_sel")
        tdf = safe_read_csv(sel_traj)
        if tdf is not None and "x" in tdf.columns and "y" in tdf.columns:
            fig_t = go.Figure()
            color_col = "speed_kmh" if "speed_kmh" in tdf.columns else None
            fig_t.add_trace(go.Scatter3d(
                x=tdf["x"], y=tdf["y"],
                z=tdf["speed_kmh"] if "speed_kmh" in tdf.columns else [0]*len(tdf),
                mode="lines",
                line=dict(color=tdf["speed_kmh"] if "speed_kmh" in tdf.columns else "#58a6ff",
                          colorscale="Turbo", width=5,
                          colorbar=dict(title="Speed (km/h)") if "speed_kmh" in tdf.columns else None),
                name="Path",
                hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Speed: %{z:.1f} km/h",
            ))
            if "collision" in tdf.columns:
                coll = tdf[tdf["collision"] == 1]
                if not coll.empty:
                    fig_t.add_trace(go.Scatter3d(
                        x=coll["x"], y=coll["y"], z=coll["speed_kmh"],
                        mode="markers", marker=dict(size=6, color="#f85149", symbol="x"),
                        name="Collisions",
                    ))
            fig_t.update_layout(
                title="Driving Trajectory (Height = Speed)",
                scene=dict(
                    xaxis=dict(title="X (m)", color="#e6edf3", gridcolor="#30363d"),
                    yaxis=dict(title="Y (m)", color="#e6edf3", gridcolor="#30363d"),
                    zaxis=dict(title="Speed (km/h)", color="#e6edf3", gridcolor="#30363d"),
                    bgcolor="#0d1117",
                ),
                **{k: v for k, v in PLOTLY_THEME.items() if k not in ("margin",)},
                height=600,
            )
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("No x/y columns in telemetry.")
    else:
        st.info("No drive telemetry found. Run `python -m src.drive` first.")

    # 3D DQN Q-values
    st.markdown("#### DQN Q-Network (Steering Actions)")
    actions = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    q_vals = [0.12, 0.25, 0.41, 0.68, 0.95, 0.72, 0.48, 0.30, 0.15]
    fig_q = go.Figure()
    for a, q in zip(actions, q_vals):
        fig_q.add_trace(go.Scatter3d(
            x=[a, a], y=[0, 0], z=[0, q],
            mode="lines", line=dict(color="#58a6ff", width=8),
            showlegend=False, hoverinfo="none",
        ))
        fig_q.add_trace(go.Scatter3d(
            x=[a], y=[0], z=[q],
            mode="markers+text",
            marker=dict(size=10 + q * 12, color=q, colorscale="Plasma",
                        colorbar=dict(title="Q-value"), line=dict(color="#ffffff", width=1)),
            text=[f"steer={a:.2f}<br>Q={q:.2f}"],
            textposition="top center",
            textfont=dict(size=10, color="#ffffff"),
            showlegend=False, hoverinfo="text",
        ))
    fig_q.update_layout(
        title="DQN Q-Values per Steering Action",
        scene=dict(
            xaxis=dict(title="Steering", tickvals=actions, color="#e6edf3", gridcolor="#30363d"),
            yaxis=dict(title="", showticklabels=False, color="#e6edf3"),
            zaxis=dict(title="Q-value", color="#e6edf3", gridcolor="#30363d"),
            bgcolor="#0d1117",
        ),
        **{k: v for k, v in PLOTLY_THEME.items() if k not in ("margin",)},
        height=500,
    )
    st.plotly_chart(fig_q, use_container_width=True)

# ---- Tab 5: Experiment Info ----
with tab5:
    if sel_exp:
        cfg_path = os.path.join(logs_dir, sel_exp, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            st.json(cfg)
        else:
            st.info("No config.json found.")
    else:
        st.info("Select an experiment in the sidebar.")

st.markdown("---")
st.caption("CARLA ML Driving Dashboard · Built with Streamlit + Plotly")