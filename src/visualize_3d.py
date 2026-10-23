"""3D visualization of the CNN architecture and project algorithms.

Generates large, clean, interactive 3D plots saved as HTML files.
Run:  python -m src.visualize_3d
"""
import os
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import ensure_dir


LAYER_COLORS = {
    "input": "#4FC3F7",   # light blue
    "conv": "#FF8A65",    # orange
    "dense": "#81C784",   # green
    "output": "#E57373",  # red
    "flat": "#BDBDBD",    # gray
}


def make_box(x0, x1, y0, y1, z0, z1, color, name, opacity=0.75):
    verts = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    tris = [
        (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (0, 3, 7), (0, 7, 4), (1, 2, 6), (1, 6, 5),
    ]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    ii = [t[0] for t in tris]
    jj = [t[1] for t in tris]
    kk = [t[2] for t in tris]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
        color=color, opacity=opacity, name=name,
        hoverinfo="name", showlegend=True,
        flatshading=True,
    )


def cnn_3d():
    """Large, clean 3D visualization of NVIDIA end-to-end CNN layers."""
    layers = [
        ("Input (RGB)", 66, 200, 3, "input"),
        ("Conv1  24 ch\n5×5 stride 2", 31, 98, 24, "conv"),
        ("Conv2  36 ch\n5×5 stride 2", 14, 47, 36, "conv"),
        ("Conv3  48 ch\n5×5 stride 2", 5, 22, 48, "conv"),
        ("Conv4  64 ch\n3×3", 3, 20, 64, "conv"),
        ("Conv5  64 ch\n3×3", 1, 18, 64, "conv"),
        ("Flatten\n1152", 1, 1, 1152, "flat"),
        ("Dense 100", 1, 1, 100, "dense"),
        ("Dense 50", 1, 1, 50, "dense"),
        ("Dense 10", 1, 1, 10, "dense"),
        ("Steering", 1, 1, 1, "output"),
    ]

    fig = go.Figure()
    xscale = 3.0
    for idx, (name, h, w, c, kind) in enumerate(layers):
        # Scale dimensions for visibility but keep proportions
        hh = max(h / 8.0, 1.0)
        ww = max(w / 15.0, 1.0)
        depth = max(min(c / 12.0, 8.0), 1.0)
        color = LAYER_COLORS[kind]

        x0 = idx * xscale
        x1 = x0 + 1.6
        box = make_box(x0, x1, -hh / 2, hh / 2, -ww / 2, ww / 2, color,
                       f"{name}  |  H={h} W={w} C={c}")
        fig.add_trace(box)

        # Label above the box
        fig.add_trace(go.Scatter3d(
            x=[(x0 + x1) / 2],
            y=[hh / 2 + 1.5],
            z=[0],
            mode="text",
            text=[f"<b>{name}</b><br>H={h}  W={w}  C={c}"],
            textfont=dict(size=11, color="#222222"),
            showlegend=False,
            hoverinfo="none",
        ))

        # Arrow-like connector to next layer
        if idx < len(layers) - 1:
            fig.add_trace(go.Scatter3d(
                x=[x1, x1 + (xscale - 1.6)],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line=dict(color="#999999", width=2, dash="dot"),
                showlegend=False,
                hoverinfo="none",
            ))

    fig.update_layout(
        title=dict(text="NVIDIA End-to-End CNN — 3D Layer Architecture",
                   font=dict(size=20)),
        scene=dict(
            xaxis=dict(title="Layer Depth →", showgrid=False, zeroline=False),
            yaxis=dict(title="Height", showgrid=True, gridcolor="#eeeeee"),
            zaxis=dict(title="Width", showgrid=True, gridcolor="#eeeeee"),
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=1, z=1),
            camera=dict(eye=dict(x=1.8, y=1.2, z=1.0)),
            bgcolor="#fafafa",
        ),
        width=1400,
        height=800,
        legend=dict(font=dict(size=11), itemsizing="constant"),
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="#ffffff",
    )
    return fig


def qnetwork_3d():
    """Large 3D visualization of DQN Q-values per steering action."""
    actions = np.linspace(-1.0, 1.0, 9)
    # Simulated Q-values (replace with real model output)
    q_values = np.array([0.12, 0.25, 0.41, 0.68, 0.95, 0.72, 0.48, 0.30, 0.15])
    action_labels = ["-1.0", "-0.75", "-0.5", "-0.25", "0.0",
                     "0.25", "0.5", "0.75", "1.0"]

    fig = go.Figure()

    # Bars
    for a, q, label in zip(actions, q_values, action_labels):
        fig.add_trace(go.Scatter3d(
            x=[a, a], y=[0, 0], z=[0, q],
            mode="lines",
            line=dict(color="#1976D2", width=8),
            showlegend=False,
            hoverinfo="none",
        ))
        fig.add_trace(go.Scatter3d(
            x=[a], y=[0], z=[q],
            mode="markers+text",
            marker=dict(size=12 + q * 15, color=q, colorscale="Viridis",
                        colorbar=dict(title="Q-value", x=1.02),
                        line=dict(color="white", width=1)),
            text=[f"steer={label}<br>Q={q:.3f}"],
            textposition="top center",
            textfont=dict(size=10, color="#333333"),
            showlegend=False,
            hoverinfo="text",
        ))

    # Base plane
    fig.add_trace(go.Mesh3d(
        x=[-1.2, 1.2, 1.2, -1.2],
        y=[-0.5, -0.5, 0.5, 0.5],
        z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#E0E0E0", opacity=0.3, showlegend=False, hoverinfo="none",
    ))

    fig.update_layout(
        title=dict(text="DQN Q-Network — Q-values per Steering Action",
                   font=dict(size=20)),
        scene=dict(
            xaxis=dict(title="Steering Action", tickvals=list(actions),
                       ticktext=action_labels, showgrid=True, gridcolor="#eeeeee"),
            yaxis=dict(title="", showticklabels=False, showgrid=False),
            zaxis=dict(title="Q-value", showgrid=True, gridcolor="#eeeeee",
                       range=[0, 1.2]),
            aspectmode="manual",
            aspectratio=dict(x=2, y=0.5, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            bgcolor="#fafafa",
        ),
        width=1000,
        height=700,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="#ffffff",
    )
    return fig


def trajectory_3d():
    """Large 3D driving trajectory with speed as height."""
    import pandas as pd

    path = os.path.join(config.LOGS_DIR, "drive_baseline.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        title_suffix = "(Baseline model)"
    else:
        # Synthetic demo trajectory
        t = np.linspace(0, 4 * np.pi, 300)
        df = pd.DataFrame({
            "x": 50 * np.cos(t),
            "y": 50 * np.sin(t),
            "speed_kmh": 10 + 15 * np.abs(np.sin(t)),
            "steering_pred": 0.3 * np.sin(t),
        })
        title_suffix = "(Demo trajectory)"

    fig = go.Figure()

    # Trajectory line colored by speed
    fig.add_trace(go.Scatter3d(
        x=df["x"], y=df["y"], z=df["speed_kmh"],
        mode="lines",
        line=dict(
            color=df["speed_kmh"],
            colorscale="Turbo",
            width=6,
            colorbar=dict(title="Speed (km/h)", x=1.02, len=0.6),
        ),
        name="Path",
        hoverinfo="text",
        text=[f"Speed: {s:.1f} km/h" for s in df["speed_kmh"]],
    ))

    # Markers at collision points
    if "collision" in df.columns:
        collisions = df[df["collision"] == 1]
        if not collisions.empty:
            fig.add_trace(go.Scatter3d(
                x=collisions["x"], y=collisions["y"], z=collisions["speed_kmh"],
                mode="markers",
                marker=dict(size=8, color="red", symbol="x"),
                name="Collisions",
            ))

    fig.update_layout(
        title=dict(text=f"Driving Trajectory — 3D View {title_suffix}",
                   font=dict(size=20)),
        scene=dict(
            xaxis=dict(title="X (m)", showgrid=True, gridcolor="#eeeeee"),
            yaxis=dict(title="Y (m)", showgrid=True, gridcolor="#eeeeee"),
            zaxis=dict(title="Speed (km/h)", showgrid=True, gridcolor="#eeeeee"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.6),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor="#fafafa",
        ),
        width=1200,
        height=800,
        margin=dict(l=20, r=80, t=60, b=20),
        paper_bgcolor="#ffffff",
        legend=dict(font=dict(size=12)),
    )
    return fig


def combined_dashboard():
    """Create a single large combined HTML with all three visualizations."""
    fig1 = cnn_3d()
    fig2 = qnetwork_3d()
    fig3 = trajectory_3d()

    # Write each as separate HTML
    ensure_dir(config.REPORTS_DIR)
    fig1.write_html(str(config.REPORTS_DIR / "cnn_3d.html"))
    fig2.write_html(str(config.REPORTS_DIR / "dqn_3d.html"))
    fig3.write_html(str(config.REPORTS_DIR / "trajectory_3d.html"))

    # Combined page as HTML with iframes
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>3D Visualization Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #1976D2; text-align: center; }}
        .viz {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 8px;
                background: white; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        iframe {{ width: 100%; height: 700px; border: none; }}
    </style>
</head>
<body>
    <h1>3D Visualization Dashboard</h1>
    <div class="viz">
        <h2>CNN Architecture (NVIDIA End-to-End)</h2>
        <iframe src="cnn_3d.html"></iframe>
    </div>
    <div class="viz">
        <h2>DQN Q-Network (Steering Actions)</h2>
        <iframe src="dqn_3d.html"></iframe>
    </div>
    <div class="viz">
        <h2>Driving Trajectory (Height = Speed)</h2>
        <iframe src="trajectory_3d.html"></iframe>
    </div>
</body>
</html>"""
    combined_path = config.REPORTS_DIR / "3d_dashboard.html"
    with open(combined_path, "w") as f:
        f.write(html)
    print(f"Saved {combined_path}")
    return combined_path


def main():
    ensure_dir(config.REPORTS_DIR)
    combined_path = combined_dashboard()
    print("Generated:")
    print(f"  {config.REPORTS_DIR / 'cnn_3d.html'}")
    print(f"  {config.REPORTS_DIR / 'dqn_3d.html'}")
    print(f"  {config.REPORTS_DIR / 'trajectory_3d.html'}")
    print(f"  {config.REPORTS_DIR / '3d_dashboard.html'}")

    import webbrowser
    webbrowser.open(str(combined_path))


if __name__ == "__main__":
    main()
