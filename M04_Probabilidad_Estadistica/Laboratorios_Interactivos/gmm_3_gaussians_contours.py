from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def _make_positive_definite(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    return cov + eps * np.eye(cov.shape[0])


def sample_gmm_2d(
    n: int,
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)

    means = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)

    k = int(weights.shape[0])
    comp_ids = rng.choice(k, size=int(n), p=weights)

    xs = np.zeros((n, 2), dtype=np.float64)
    for j in range(k):
        idx = np.where(comp_ids == j)[0]
        if idx.size == 0:
            continue
        cov_j = _make_positive_definite(covs[j])
        xs[idx] = rng.multivariate_normal(mean=means[j], cov=cov_j, size=idx.size)

    return xs, comp_ids


def grid_density(
    xx: np.ndarray,
    yy: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    xy = np.column_stack([xx.ravel(), yy.ravel()])

    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)

    k = int(weights.shape[0])

    comp_dens = []
    mix = np.zeros((xy.shape[0],), dtype=np.float64)
    for j in range(k):
        rv = multivariate_normal(mean=means[j], cov=_make_positive_definite(covs[j]))
        d = rv.pdf(xy)
        comp_dens.append(d.reshape(xx.shape))
        mix += weights[j] * d

    mix = mix.reshape(xx.shape)
    return mix, comp_dens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid", type=int, default=250)
    parser.add_argument(
        "--out",
        default="",
        help="si se especifica, guarda la figura (png/pdf) en esa ruta",
    )
    args = parser.parse_args()

    weights = np.array([0.35, 0.40, 0.25], dtype=np.float64)
    means = np.array(
        [
            [-2.0, -1.0],
            [1.5, 1.0],
            [0.0, -2.5],
        ]
    )
    covs = np.array(
        [
            [[1.0, 0.6], [0.6, 1.2]],
            [[0.7, -0.2], [-0.2, 0.5]],
            [[0.4, 0.0], [0.0, 1.0]],
        ]
    )

    x, comp = sample_gmm_2d(args.n, weights, means, covs, seed=args.seed)
    print(
        f"samples: X.shape={x.shape} components={np.bincount(comp, minlength=3).tolist()}"
    )

    pad = 1.5
    x_min, x_max = float(x[:, 0].min() - pad), float(x[:, 0].max() + pad)
    y_min, y_max = float(x[:, 1].min() - pad), float(x[:, 1].max() + pad)

    xs = np.linspace(x_min, x_max, int(args.grid))
    ys = np.linspace(y_min, y_max, int(args.grid))
    xx, yy = np.meshgrid(xs, ys)

    mix, comps = grid_density(xx, yy, weights, means, covs)

    fig, ax = plt.subplots(figsize=(9, 7))

    colors = ["tab:blue", "tab:orange", "tab:green"]
    for j in range(3):
        mask = comp == j
        ax.scatter(
            x[mask, 0], x[mask, 1], s=8, alpha=0.35, color=colors[j], label=f"comp {j}"
        )

    ax.contour(xx, yy, mix, levels=12, linewidths=1.2, colors="black", alpha=0.6)

    for j, dens in enumerate(comps):
        ax.contour(xx, yy, dens, levels=6, linewidths=1.0, colors=colors[j], alpha=0.55)

    ax.set_title("GMM 2D: datos de 3 gaussianas + contornos (componentes y mezcla)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"saved: {out_path}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
