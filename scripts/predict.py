from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch no está instalado.\n"
            "Instala con: pip install torch torchvision\n"
            "o (si usas pyproject): pip install '.[pytorch]'"
        ) from e


def _load_image_28x28(path: Path) -> np.ndarray:
    import matplotlib.image as mpimg

    img = mpimg.imread(str(path))

    if img.ndim == 3:
        img = img[..., 0]

    if img.shape != (28, 28):
        raise ValueError(f"La imagen debe ser 28x28. Recibido: {img.shape}")

    img = img.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    return img


def _load_input(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(str(path), allow_pickle=False)
        arr = np.asarray(arr)
        if arr.shape == (28, 28):
            return arr.astype(np.float32)
        if arr.shape == (784,):
            return arr.reshape(28, 28).astype(np.float32)
        raise ValueError(f".npy debe ser (28,28) o (784,). Recibido: {arr.shape}")

    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return _load_image_28x28(path)

    raise ValueError("input must be .png/.jpg/.jpeg or .npy")


def _build_model(arch: str):
    import torch.nn as nn

    if arch != "SimpleCNN_v1":
        raise ValueError(f"Arquitectura desconocida: {arch}")

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def main() -> int:
    _require_torch()

    parser = argparse.ArgumentParser(
        description="Load a saved CNN and run prediction on one image."
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint .pt creado por scripts/train_cnn_pytorch.py",
    )
    parser.add_argument(
        "--input", required=True, help="Imagen 28x28 (.png/.jpg) o array (.npy)"
    )
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = parser.parse_args()

    import torch

    ckpt_path = Path(args.ckpt)
    in_path = Path(args.input)

    if args.device == "auto":
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        device = torch.device(args.device)

    payload = torch.load(str(ckpt_path), map_location="cpu")
    arch = payload.get("arch", "SimpleCNN_v1")
    state_dict = payload["state_dict"]

    model = _build_model(arch)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    img = _load_input(in_path)
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,28,28)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        conf = float(torch.max(probs, dim=1).values.item())

    print(f"pred: {pred}")
    print(f"confidence: {conf:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
