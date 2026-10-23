"""3D visualization of the CNN architecture and project algorithms."""
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import ensure_dir


def make_box(x0, x1, y0, y1, z0, z1, color, name, opacity=0.6):
    """Create a 3D box mesh."""
    verts = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    i = [0, 0, 0, 0, 4, 4, 4, 4, 0, 1, 2, 3, 0, 1, 2, 3, 0, 4, 1, 5, 2, 6, 3, 7]
    # triangles
    tris = [
        (0,1,2),(0,2,3), (4,5,6),(4,6,7),
        (0,1,5),(0,5,4), (2,3,7),(2,7,6),
        (0,3,7),(0,7,4), (1,2,6),(1,6,5),
    ]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    ii = [t[0] for t in tris]
    jj = [t[1] for t in tris]
    kk = [t[2] for t in tris]
    return go.Mesh3d(x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                     color=color, opacity=opacity, name=name,
                     hoverinfo="name", showlegend=True)


def cnn_3d():
    """3D visualization of NVIDIA end-to-end CNN layers."""
    # (name, H, W, C, color)
    layers = [
        ("Input (RGB)", 66, 200, 3, "lightblue"),
        ("Conv1 24ch 5x5 s2", 31, 98, 24, "orange"),
        ("Conv2 36ch 5x5 s2", 14, 47, 36, "orange"),
        ("Conv3 48ch 5x5 s2", 5, 22, 48, "orange"),
        ("Conv4 64ch 3x3", 3, 20, 64, "salmon"),
        ("Conv5 64ch 3x3", 1, 18, 64, "salmon"),
        ("Flatten", 1, 1, 1152, "gray"),
        ("Dense 100", 1, 1, 100, "green"),
        ("Dense 50", 1, 1, 50, "green"),
        ("Dense 10", 1, 1, 10, "green"),
        ("Steering (1)", 1, 1, 1, "red"),
    ]
    fig = go.Figure()
    x_offset = 0
    xscale = 2.0
    for idx, (name, h, w, c, color) in enumerate(layers):
        # scale spatial dims for visibility
        hh = max(h / 10.0, 0.5)
        ww = max(w / 20.0, 0.5)
        cc = min(c / 10.0, 6.0)
        x0 = idx * xscale
        x1 = x0 + 0.8
        box = make_box(x0, x1, -hh/2, hh/2, -ww/2, ww/2, color, f"{name} (C={c})")
        fig.add_trace(box)
        # label
        fig.add_trace(go.Scatter3d(
            x=[(x0+x1)/2], y=[hh/2 + 0.5], z=[0],
            mode="text", text=[f"{name}<br>H={h} W={w} C={c}"],
            textfont=dict(size=9, color="black"), showlegend=False,
            hoverinfo="none"))

    fig.update_layout(
        title="NVIDIA End-to-End CNN — 3D Layer Architecture",
        scene=dict(
            xaxis_title="Layer depth",
            yaxis_title="Height",
            zaxis_title="Width",
            aspectmode="manual",
            aspectratio=dict(x=2, y=1, z=1),
        ),
        width=1200, height=700,
        legend=dict(font=dict(size=9)),
    )
    return fig


def qnetwork_3d():
    """3D visualization of DQN Q-network (steering action values)."""
    actions = np.linspace(-1.0, 1.0, 9)
    q_values = np.random.rand(9)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=actions, y=[0]*9, z=q_values,
        mode="markers+text",
        marker=dict(size=8 + q_values*20, color=q_values, colorscale="Viridis",
                    colorbar=dict(title="Q-value")),
        text=[f"a={a:.2f}<br>Q={q:.3f}" for a, q in zip(actions, q_values)],
        textposition="top center",
        name="Q-values",
    ))
    # Add vertical lines from z=0 to each point
    for a, q in zip(actions, q_values):
        fig.add_trace(go.Scatter3d(
            x=[a, a], y=[0, 0], z=[0, q],
            mode="lines", line=dict(color="gray", width=2), showlegend=False,
        ))
    fig.update_layout(
        title="DQN Q-Network — Q-values per steering action",
        scene=dict(
            xaxis=dict(title="Steering action"),
            yaxis=dict(title="", showticklabels=False),
            zaxis_title="Q-value",
        ),
        width=800, height=600,
    )
    return fig


def trajectory_3d():
    """3D visualization of a driving trajectory with speed as height."""
    # Try to load real drive telemetry
    import pandas as pd
    path = os.path.join(config.LOGS_DIR, "drive_baseline.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # synthetic demo trajectory
        t = np.linspace(0, 4*np.pi, 200)
        df = pd.DataFrame({
            "x": 50*np.cos(t), "y": 50*np.sin(t),
            "speed_kmh": 10 + 15*np.abs(np.sin(t)),
            "steering_pred": 0.3*np.sin(t),
        })
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df["x"], y=df["y"], z=df["speed_kmh"],
        mode="lines+markers",
        marker=dict(size=3, color=df["speed_kmh"], colorscale="Turbo", colorbar=dict(title="Speed km/h")),
        line=dict(color=df["speed_kmh"], colorscale="Turbo", width=4),
        name="Trajectory",
    ))
    fig.update_layout(
        title="Driving Trajectory (height = speed)",
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Speed (km/h)"),
        width=900, height=700,
    )
    return fig


def main():
    ensure_dir(config.REPORTS_DIR)
    out_dir = config.REPORTS_DIR

    # CNN architecture 3D
    fig1 = cnn_3d()
    fig1.write_html(str(out_dir / "cnn_3d.html"))
    print(f"Saved {out_dir / 'cnn_3d.html'}")

    # DQN 3D
    fig2 = qnetwork_3d()
    fig2.write_html(str(out_dir / "dqn_3d.html"))
    print(f"Saved {out_dir / 'dqn_3d.html'}")

    # Trajectory 3D
    fig3 = trajectory_3d()
    fig3.write_html(str(out_dir / "trajectory_3d.html"))
    print(f"Saved {out_dir / 'trajectory_3d.html'}")

    # Combined dashboard
    combined = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["CNN Architecture", "DQN Q-values", "Driving Trajectory"],
    )
    # Add traces from each figure
    for trace in fig1.data:
        combined.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        combined.add_trace(trace, row=1, col=2)
    for trace in fig3.data:
        combined.add_trace(trace, row=1, col=3)
    combined.update_layout(title="3D Visualization Dashboard", width=1800, height=600, showlegend=False)
    combined.write_html(str(out_dir / "3d_dashboard.html"))
    print(f"Saved {out_dir / '3d_dashboard.html'}")

    # Open in browser
    import webbrowser
    webbrowser.open(str(out_dir / "3d_dashboard.html"))


if __name__ == "__main__":
    main()
