import matplotlib.pyplot as plt
import numpy as np


def _grid_lines(lim: float = 5.0, n_lines: int = 11, n_points: int = 200):
    xs = np.linspace(-lim, lim, n_lines)
    ys = np.linspace(-lim, lim, n_lines)
    t = np.linspace(-lim, lim, n_points)

    vertical = [(np.full_like(t, x), t) for x in xs]
    horizontal = [(t, np.full_like(t, y)) for y in ys]
    return vertical, horizontal


def _apply(A: np.ndarray, x: np.ndarray, y: np.ndarray):
    pts = np.vstack([x, y])
    out = A @ pts
    return out[0], out[1]


def plot_linear_transformation(
    A: np.ndarray,
    lim: float = 5.0,
    n_lines: int = 11,
    show_eigen: bool = True,
    title: str | None = None,
):
    A = np.asarray(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("A must be a 2x2 matrix")

    v_lines, h_lines = _grid_lines(lim=lim, n_lines=n_lines)

    fig, ax = plt.subplots(figsize=(7, 7))

    for x, y in v_lines:
        ax.plot(x, y, color="#cbd5e1", linewidth=0.8)
    for x, y in h_lines:
        ax.plot(x, y, color="#cbd5e1", linewidth=0.8)

    for x, y in v_lines:
        xt, yt = _apply(A, x, y)
        ax.plot(xt, yt, color="#2563eb", linewidth=0.9, alpha=0.85)
    for x, y in h_lines:
        xt, yt = _apply(A, x, y)
        ax.plot(xt, yt, color="#2563eb", linewidth=0.9, alpha=0.85)

    ax.axhline(0, color="#0f172a", linewidth=1)
    ax.axvline(0, color="#0f172a", linewidth=1)

    if show_eigen:
        vals, vecs = np.linalg.eig(A)
        vals = np.real_if_close(vals)
        vecs = np.real_if_close(vecs)

        for i in range(2):
            v = vecs[:, i]
            if np.iscomplexobj(v):
                continue
            v = np.real(v)
            v = v / (np.linalg.norm(v) + 1e-12)
            scale = lim
            ax.plot(
                [-scale * v[0], scale * v[0]],
                [-scale * v[1], scale * v[1]],
                color="#f97316",
                linewidth=2.0,
                alpha=0.9,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim * 2, lim * 2)
    ax.set_ylim(-lim * 2, lim * 2)
    ax.grid(True, alpha=0.15)

    if title is None:
        title = f"A = {A.tolist()}"
    ax.set_title(title)

    return fig, ax


if __name__ == "__main__":
    matrices = {
        "Stretch X": np.array([[2.0, 0.0], [0.0, 1.0]]),
        "Rotate 90°": np.array([[0.0, -1.0], [1.0, 0.0]]),
        "Shear": np.array([[1.0, 1.0], [0.0, 1.0]]),
    }

    for name, A in matrices.items():
        plot_linear_transformation(A, title=name)
    plt.show()
