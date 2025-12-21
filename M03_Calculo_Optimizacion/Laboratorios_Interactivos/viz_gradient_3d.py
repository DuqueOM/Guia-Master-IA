from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x**2 + 4 * y**2


def grad_f(x: float, y: float) -> np.ndarray:
    return np.array([2 * x, 8 * y], dtype=float)


def run_gd(
    lr: float = 0.1, steps: int = 20, x0: float = 2.5, y0: float = 2.5
) -> np.ndarray:
    x, y = float(x0), float(y0)
    path = [(x, y, float(f(x, y)))]
    for _ in range(int(steps)):
        g = grad_f(x, y)
        x = x - lr * float(g[0])
        y = y - lr * float(g[1])
        path.append((x, y, float(f(x, y))))
    return np.array(path, dtype=float)


def make_figure(
    lr: float, steps: int, grid_lim: float = 3.0, grid_n: int = 80
) -> go.Figure:
    grid = np.linspace(-float(grid_lim), float(grid_lim), int(grid_n))
    X, Y = np.meshgrid(grid, grid)
    Z = f(X, Y)

    path = run_gd(lr=lr, steps=steps)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z, opacity=0.85, colorscale="Viridis", showscale=False)
    )
    fig.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines+markers",
            line={"color": "red", "width": 6},
            marker={"size": 4},
            name="GD path",
        )
    )

    fig.update_layout(
        title=f"Gradient Descent (3D): lr={lr}, steps={steps}",
        scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "f(x,y)"},
        height=650,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )

    return fig


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Visualización 3D de Gradient Descent (Plotly). "
            "Recomendado para M03: ejecuta con distintos learning rates y compara convergencia vs divergencia."
        )
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--grid-lim", type=float, default=3.0)
    parser.add_argument("--grid-n", type=int, default=80)
    parser.add_argument("--html-out", default="gradient_descent_3d.html")
    parser.add_argument(
        "--show", action="store_true", help="Abrir en el visor del entorno (si aplica)"
    )
    args = parser.parse_args()

    fig = make_figure(
        lr=float(args.lr),
        steps=int(args.steps),
        grid_lim=float(args.grid_lim),
        grid_n=int(args.grid_n),
    )

    out_path = Path(args.html_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"saved: {out_path}")

    if args.show:
        fig.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
