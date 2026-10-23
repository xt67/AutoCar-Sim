"""3D visualization of the CNN architecture and project algorithms.

Dark theme, large readable labels, interactive HTML output.
Run:  python -m src.visualize_3d
"""
import os
import sys

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import ensure_dir

# Dark theme colors
BG = "#0d1117"
GRID = "#30363d"
TEXT = "#e6edf3"
LAYER_COLORS = {
    "input": "#58a6ff",
    "conv": "#f0883e",
    "dense": "#3fb950",
    "output": "#f85149",
    "flat": "#8b949e",
}


def make_box(x0, x1, y0, y1, z0, z1, color, name, opacity=0.85):
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
        flatshading=True, lighting=dict(ambient=0.6, diffuse=0.8),
    )


def cnn_3d():
    layers = [
        ("Input (RGB)", 66, 200, 3, "input"),
        ("Conv1  24ch\n5×5 s2", 31, 98, 24, "conv"),
        ("Conv2  36ch\n5×5 s2", 14, 47, 36, "conv"),
        ("Conv3  48ch\n5×5 s2", 5, 22, 48, "conv"),
        ("Conv4  64ch\n3×3", 3, 20, 64, "conv"),
        ("Conv5  64ch\n3×3", 1, 18, 64, "conv"),
        ("Flatten\n1152", 1, 1, 1152, "flat"),
        ("Dense 100", 1, 1, 100, "dense"),
        ("Dense 50", 1, 1, 50, "dense"),
        ("Dense 10", 1, 1, 10, "dense"),
        ("Steering", 1, 1, 1, "output"),
    ]

    fig = go.Figure()
    xscale = 3.5
    for idx, (name, h, w, c, kind) in enumerate(layers):
        hh = max(h / 8.0, 1.2)
        ww = max(w / 15.0, 1.2)
        depth = max(min(c / 12.0, 8.0), 1.2)
        color = LAYER_COLORS[kind]

        x0 = idx * xscale
        x1 = x0 + 1.8
        box = make_box(x0, x1, -hh / 2, hh / 2, -ww / 2, ww / 2, color,
                       f"{name}  |  H={h} W={w} C={c}")
        fig.add_trace(box)

        # Label in front of the box (at z offset so it's readable from default angle)
        fig.add_trace(go.Scatter3d(
            x=[(x0 + x1) / 2],
            y=[hh / 2 + 1.0],
            z=[ww / 2 + 1.5],
            mode="text",
            text=[f"<b>{name}</b><br>H={h} W={w} C={c}"],
            textfont=dict(size=12, color=TEXT),
            showlegend=False,
            hoverinfo="none",
        ))

        if idx < len(layers) - 1:
            fig.add_trace(go.Scatter3d(
                x=[x1, x1 + (xscale - 1.8)],
                y=[0, 0],
                z=[0, 0],
                mode="lines",
                line=dict(color="#484f58", width=2, dash="dot"),
                showlegend=False,
                hoverinfo="none",
            ))

    fig.update_layout(
        title=dict(text="NVIDIA End-to-End CNN — 3D Layer Architecture",
                   font=dict(size=22, color=TEXT)),
        scene=dict(
            xaxis=dict(title="Layer Depth →", showgrid=True, gridcolor=GRID,
                       zeroline=False, color=TEXT),
            yaxis=dict(title="Height", showgrid=True, gridcolor=GRID, color=TEXT),
            zaxis=dict(title="Width", showgrid=True, gridcolor=GRID, color=TEXT),
            aspectmode="manual",
            aspectratio=dict(x=2.5, y=1, z=1),
            camera=dict(eye=dict(x=1.6, y=1.0, z=1.2)),
            bgcolor=BG,
        ),
        width=1500,
        height=850,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, size=12),
        legend=dict(font=dict(size=11, color=TEXT), itemsizing="constant",
                    bgcolor="rgba(13,17,23,0.8)", bordercolor=GRID),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def qnetwork_3d():
    actions = np.linspace(-1.0, 1.0, 9)
    q_values = np.array([0.12, 0.25, 0.41, 0.68, 0.95, 0.72, 0.48, 0.30, 0.15])
    action_labels = ["-1.0", "-0.75", "-0.5", "-0.25", "0.0",
                     "0.25", "0.5", "0.75", "1.0"]

    fig = go.Figure()

    for a, q, label in zip(actions, q_values, action_labels):
        fig.add_trace(go.Scatter3d(
            x=[a, a], y=[0, 0], z=[0, q],
            mode="lines",
            line=dict(color="#58a6ff", width=10),
            showlegend=False,
            hoverinfo="none",
        ))
        fig.add_trace(go.Scatter3d(
            x=[a], y=[0], z=[q],
            mode="markers+text",
            marker=dict(size=14 + q * 12, color=q, colorscale="Plasma",
                        colorbar=dict(title="Q-value", x=1.02, len=0.5,
                                      tickfont=dict(color=TEXT)),
                        line=dict(color="#ffffff", width=1)),
            text=[f"steer={label}<br>Q={q:.3f}"],
            textposition="top center",
            textfont=dict(size=11, color=TEXT),
            showlegend=False,
            hoverinfo="text",
        ))

    # Base plane
    fig.add_trace(go.Mesh3d(
        x=[-1.2, 1.2, 1.2, -1.2],
        y=[-0.5, -0.5, 0.5, 0.5],
        z=[0, 0, 0, 0],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="#21262d", opacity=0.5, showlegend=False, hoverinfo="none",
    ))

    fig.update_layout(
        title=dict(text="DQN Q-Network — Q-values per Steering Action",
                   font=dict(size=20, color=TEXT)),
        scene=dict(
            xaxis=dict(title="Steering Action", tickvals=list(actions),
                       ticktext=action_labels, showgrid=True, gridcolor=GRID,
                       color=TEXT),
            yaxis=dict(title="", showticklabels=False, showgrid=False, color=TEXT),
            zaxis=dict(title="Q-value", showgrid=True, gridcolor=GRID,
                       color=TEXT, range=[0, 1.2]),
            aspectmode="manual",
            aspectratio=dict(x=2, y=0.5, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            bgcolor=BG,
        ),
        width=1000,
        height=700,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, size=12),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def trajectory_3d():
    import pandas as pd

    path = os.path.join(config.LOGS_DIR, "drive_baseline.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        title_suffix = "(Baseline model)"
    else:
        t = np.linspace(0, 4 * np.pi, 300)
        df = pd.DataFrame({
            "x": 50 * np.cos(t), "y": 50 * np.sin(t),
            "speed_kmh": 10 + 15 * np.abs(np.sin(t)),
            "steering_pred": 0.3 * np.sin(t),
        })
        title_suffix = "(Demo trajectory)"

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=df["x"], y=df["y"], z=df["speed_kmh"],
        mode="lines",
        line=dict(
            color=df["speed_kmh"],
            colorscale="Turbo",
            width=6,
            colorbar=dict(title=dict(text="Speed (km/h)", font=dict(color=TEXT)),
                          x=1.02, len=0.6, tickfont=dict(color=TEXT)),
        ),
        name="Path",
        hoverinfo="text",
        text=[f"Speed: {s:.1f} km/h" for s in df["speed_kmh"]],
    ))

    if "collision" in df.columns:
        collisions = df[df["collision"] == 1]
        if not collisions.empty:
            fig.add_trace(go.Scatter3d(
                x=collisions["x"], y=collisions["y"], z=collisions["speed_kmh"],
                mode="markers",
                marker=dict(size=8, color="#f85149", symbol="x"),
                name="Collisions",
            ))

    fig.update_layout(
        title=dict(text=f"Driving Trajectory — 3D View {title_suffix}",
                   font=dict(size=20, color=TEXT)),
        scene=dict(
            xaxis=dict(title="X (m)", showgrid=True, gridcolor=GRID, color=TEXT),
            yaxis=dict(title="Y (m)", showgrid=True, gridcolor=GRID, color=TEXT),
            zaxis=dict(title="Speed (km/h)", showgrid=True, gridcolor=GRID, color=TEXT),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.6),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor=BG,
        ),
        width=1200,
        height=800,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, size=12),
        margin=dict(l=20, r=80, t=60, b=20),
        legend=dict(font=dict(size=12, color=TEXT)),
    )
    return fig


def combined_dashboard():
    fig1 = cnn_3d()
    fig2 = qnetwork_3d()
    fig3 = trajectory_3d()

    ensure_dir(config.REPORTS_DIR)
    fig1.write_html(str(config.REPORTS_DIR / "cnn_3d.html"))
    fig2.write_html(str(config.REPORTS_DIR / "dqn_3d.html"))
    fig3.write_html(str(config.REPORTS_DIR / "trajectory_3d.html"))

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>3D Visualization Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: {BG}; color: {TEXT}; }}
        h1 {{ color: #58a6ff; text-align: center; }}
        .viz {{ margin: 20px 0; border: 1px solid {GRID}; border-radius: 8px;
                background: #161b22; padding: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
        iframe {{ width: 100%; height: 750px; border: none; }}
        h2 {{ color: #e6edf3; }}
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
