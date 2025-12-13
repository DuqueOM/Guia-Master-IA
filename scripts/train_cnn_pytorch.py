from __future__ import annotations

import argparse
from pathlib import Path


def _require_torch():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyTorch/torchvision no están instalados.\n"
            "Instala con: pip install torch torchvision\n"
            "o (si usas pyproject): pip install '.[pytorch]'"
        ) from e


def _get_device(device: str):
    import torch

    if device == "auto":
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    return torch.device(device)


class SimpleCNN:
    def __init__(self, num_classes: int = 10):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn

        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, x):
        return self.model(x)


def _get_dataloaders(dataset: str, data_dir: Path, batch_size: int, seed: int):
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    data_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "mnist":
        ds_train = datasets.MNIST(
            root=str(data_dir), train=True, download=True, transform=tfm
        )
        ds_test = datasets.MNIST(
            root=str(data_dir), train=False, download=True, transform=tfm
        )
    elif dataset == "fashion":
        ds_train = datasets.FashionMNIST(
            root=str(data_dir), train=True, download=True, transform=tfm
        )
        ds_test = datasets.FashionMNIST(
            root=str(data_dir), train=False, download=True, transform=tfm
        )
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion'")

    g = torch.Generator()
    g.manual_seed(seed)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=g)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return dl_train, dl_test


def _evaluate(model, dataloader, device):
    import torch

    model.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def main() -> int:
    _require_torch()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "fashion"], default="mnist")
    parser.add_argument("--data-dir", default="data/torch_datasets")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda")
    parser.add_argument("--out", default="artifacts/cnn.pt")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    device = _get_device(args.device)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dl_train, dl_test = _get_dataloaders(
        args.dataset, data_dir, args.batch_size, args.seed
    )

    model = SimpleCNN(num_classes=10).to(device)
    opt = torch.optim.Adam(model.model.parameters(), lr=float(args.lr))

    for epoch in range(1, int(args.epochs) + 1):
        model.model.train()
        running_loss = 0.0
        n_seen = 0

        for x, y in dl_train:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            bs = int(y.numel())
            running_loss += float(loss.item()) * bs
            n_seen += bs

        train_loss = running_loss / max(n_seen, 1)
        acc = _evaluate(model, dl_test, device)
        print(f"epoch={epoch} loss={train_loss:.4f} test_acc={acc:.4f}")

    payload = {
        "dataset": args.dataset,
        "state_dict": model.model.state_dict(),
        "arch": "SimpleCNN_v1",
    }
    torch.save(payload, str(out_path))
    print(f"saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
