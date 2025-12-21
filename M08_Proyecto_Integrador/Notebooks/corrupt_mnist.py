from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CorruptionConfig:
    gaussian_std: float
    saltpepper_prob: float
    invert_prob: float
    nan_prob: float
    clip: bool


def _as_float01(X: np.ndarray) -> tuple[np.ndarray, bool]:
    X = np.asarray(X)
    if X.dtype.kind in {"f"}:
        return X.astype(np.float32, copy=False), True
    return X.astype(np.float32) / 255.0, False


def _back_to_original_scale(
    X01: np.ndarray, was_float: bool, original_dtype: np.dtype
) -> np.ndarray:
    if was_float:
        out = X01
    else:
        out = X01 * 255.0

    if np.issubdtype(original_dtype, np.integer):
        out = np.rint(out)
        out = np.clip(out, 0, 255)
        return out.astype(original_dtype)

    return out.astype(original_dtype)


def corrupt_features(X: np.ndarray, cfg: CorruptionConfig, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    original_dtype = np.asarray(X).dtype
    X01, was_float = _as_float01(X)

    Xc = X01.copy()

    if cfg.gaussian_std > 0:
        noise = rng.normal(0.0, float(cfg.gaussian_std), size=Xc.shape).astype(
            np.float32
        )
        Xc = Xc + noise

    if cfg.saltpepper_prob > 0:
        mask = rng.random(size=Xc.shape) < float(cfg.saltpepper_prob)
        sp = rng.integers(0, 2, size=Xc.shape, dtype=np.int8)
        Xc[mask] = sp[mask].astype(np.float32)

    if cfg.invert_prob > 0:
        if Xc.ndim == 2:
            per_sample = rng.random(size=(Xc.shape[0],)) < float(cfg.invert_prob)
            Xc[per_sample] = 1.0 - Xc[per_sample]
        else:
            per_sample = rng.random(size=(Xc.shape[0],)) < float(cfg.invert_prob)
            Xc[per_sample, ...] = 1.0 - Xc[per_sample, ...]

    if cfg.nan_prob > 0:
        nan_mask = rng.random(size=Xc.shape) < float(cfg.nan_prob)
        Xc[nan_mask] = np.nan

    if cfg.clip:
        Xc = np.clip(Xc, 0.0, 1.0)

    return _back_to_original_scale(
        Xc, was_float=was_float, original_dtype=original_dtype
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(str(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **arrays)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Corrompe datasets tipo MNIST (arrays) para forzar un Dirty Data Check. "
            "Entrada/salida recomendada: .npz con claves X_train/X_test o X/y."
        )
    )

    parser.add_argument(
        "--in", dest="in_path", required=True, help="Ruta a .npz o .npy"
    )
    parser.add_argument(
        "--out", dest="out_path", required=True, help="Ruta de salida .npz"
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gaussian-std", type=float, default=0.15)
    parser.add_argument("--saltpepper-prob", type=float, default=0.02)
    parser.add_argument("--invert-prob", type=float, default=0.10)
    parser.add_argument("--nan-prob", type=float, default=0.001)
    parser.add_argument("--no-clip", action="store_true")

    parser.add_argument(
        "--keys",
        default="X_train,X_test,X",
        help="Claves a corromper si la entrada es .npz (CSV).",
    )

    args = parser.parse_args()

    cfg = CorruptionConfig(
        gaussian_std=float(args.gaussian_std),
        saltpepper_prob=float(args.saltpepper_prob),
        invert_prob=float(args.invert_prob),
        nan_prob=float(args.nan_prob),
        clip=not bool(args.no_clip),
    )

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if in_path.suffix == ".npy":
        X = np.load(str(in_path), allow_pickle=False)
        X_corrupt = corrupt_features(X, cfg, seed=int(args.seed))
        _save_npz(out_path, {"X": X_corrupt})
        print(f"saved: {out_path} (keys: X)")
        return 0

    if in_path.suffix != ".npz":
        raise ValueError("Input must be .npz or .npy")

    arrays = _load_npz(in_path)
    keys = {k.strip() for k in str(args.keys).split(",") if k.strip()}

    changed = []
    for k in sorted(keys):
        if k in arrays:
            arrays[k] = corrupt_features(arrays[k], cfg, seed=int(args.seed))
            changed.append(k)

    arrays["_corruption_json"] = np.array(
        json.dumps(
            {
                "gaussian_std": cfg.gaussian_std,
                "saltpepper_prob": cfg.saltpepper_prob,
                "invert_prob": cfg.invert_prob,
                "nan_prob": cfg.nan_prob,
                "clip": cfg.clip,
                "seed": int(args.seed),
            },
            sort_keys=True,
        )
    )

    _save_npz(out_path, arrays)

    print(f"saved: {out_path}")
    print(f"corrupted_keys: {', '.join(changed) if changed else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
