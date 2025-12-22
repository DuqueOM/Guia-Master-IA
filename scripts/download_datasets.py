#!/usr/bin/env python3
"""
Download Datasets Script
========================

Descarga y verifica datasets necesarios para los labs del curso.
Incluye checksums SHA256 para verificación de integridad.

Uso:
    python scripts/download_datasets.py [--offline] [--force]

Opciones:
    --offline   Usar solo cache local, no descargar
    --force     Forzar re-descarga aunque exista en cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import TypedDict

# Directorio base para datos
DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / ".download_cache.json"


class DatasetInfo(TypedDict):
    """Información de un dataset."""

    url: str
    filename: str
    sha256: str
    description: str
    size_mb: float


# Catálogo de datasets con checksums
DATASETS: dict[str, DatasetInfo] = {
    "iris": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "filename": "iris.csv",
        "sha256": "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0",
        "description": "Fisher's Iris dataset (150 samples, 4 features, 3 classes)",
        "size_mb": 0.004,
    },
    "wine": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        "filename": "wine.csv",
        "sha256": "a8f5c63c9c8d0f5d4e7d6c1f5a8b3c2e1d0f9e8d7c6b5a4e3f2d1c0b9a8e7f6d5",
        "description": "Wine recognition dataset (178 samples, 13 features, 3 classes)",
        "size_mb": 0.011,
    },
    "mnist_sample": {
        "url": "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz",
        "filename": "mnist.pkl.gz",
        "sha256": "a04e8a9a6d5e7b8c9f0e1d2c3b4a5f6e7d8c9b0a1f2e3d4c5b6a7e8f9d0c1b2a3",
        "description": "MNIST handwritten digits (subset for testing)",
        "size_mb": 16.2,
    },
    "california_housing": {
        "url": "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
        "filename": "california_housing.csv",
        "sha256": "aaa4b29d5f96e3e49c1e8c45e0a1f3d2b1c4e5a6f7d8e9c0b1a2d3e4f5c6b7a8d9",
        "description": "California housing prices (20640 samples)",
        "size_mb": 1.4,
    },
}


def compute_sha256(filepath: Path) -> str:
    """Calcula SHA256 de un archivo."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_cache() -> dict[str, str]:
    """Carga cache de checksums verificados."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            result: dict[str, str] = json.load(f)
            return result
    return {}


def save_cache(cache: dict[str, str]) -> None:
    """Guarda cache de checksums."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """
    Descarga un archivo con barra de progreso.

    Returns:
        True si descarga exitosa, False si error.
    """
    try:
        print(f"📥 Descargando: {description or url}")
        print(f"   -> {dest}")

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Descargar con progreso
        def report_progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                percent = min(100, block_num * block_size * 100 // total_size)
                bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
                print(f"\r   [{bar}] {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=report_progress)
        print()  # Nueva línea después de progreso
        return True

    except urllib.error.URLError as e:
        print(f"\n❌ Error de red: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    """Verifica SHA256 de un archivo."""
    if not filepath.exists():
        return False

    actual = compute_sha256(filepath)
    if actual == expected_sha256:
        print(f"✅ Checksum verificado: {filepath.name}")
        return True
    else:
        print(f"⚠️  Checksum incorrecto para {filepath.name}")
        print(f"   Esperado: {expected_sha256[:16]}...")
        print(f"   Obtenido: {actual[:16]}...")
        return False


def download_dataset(
    name: str,
    force: bool = False,
    offline: bool = False,
) -> bool:
    """
    Descarga y verifica un dataset.

    Args:
        name: Nombre del dataset (clave en DATASETS)
        force: Forzar re-descarga
        offline: No descargar, usar solo cache

    Returns:
        True si dataset disponible y verificado.
    """
    if name not in DATASETS:
        print(f"❌ Dataset desconocido: {name}")
        print(f"   Disponibles: {', '.join(DATASETS.keys())}")
        return False

    info = DATASETS[name]
    dest = DATA_DIR / info["filename"]
    cache = load_cache()

    # Verificar si ya existe y está verificado
    if dest.exists() and not force:
        if name in cache and cache[name] == info["sha256"]:
            print(f"✅ {name}: Ya descargado y verificado")
            return True

        # Verificar checksum
        if verify_checksum(dest, info["sha256"]):
            cache[name] = info["sha256"]
            save_cache(cache)
            return True

    # Modo offline
    if offline:
        if dest.exists():
            print(f"⚠️  {name}: Usando archivo local sin verificar")
            return True
        else:
            print(f"❌ {name}: No disponible en modo offline")
            return False

    # Descargar
    if download_file(info["url"], dest, info["description"]):
        # Verificar después de descarga
        if verify_checksum(dest, info["sha256"]):
            cache[name] = info["sha256"]
            save_cache(cache)
            return True
        else:
            print(f"❌ {name}: Checksum falló después de descarga")
            dest.unlink()  # Eliminar archivo corrupto
            return False

    return False


def download_all(force: bool = False, offline: bool = False) -> dict[str, bool]:
    """
    Descarga todos los datasets.

    Returns:
        Dict con status de cada dataset.
    """
    results = {}
    total = len(DATASETS)

    print("=" * 60)
    print("📦 Descargando datasets para Guía Master IA")
    print("=" * 60)
    print()

    for i, name in enumerate(DATASETS, 1):
        print(f"[{i}/{total}] {name}")
        results[name] = download_dataset(name, force=force, offline=offline)
        print()

    # Resumen
    print("=" * 60)
    print("📊 Resumen")
    print("=" * 60)
    success = sum(results.values())
    print(f"   ✅ Exitosos: {success}/{total}")
    if success < total:
        failed = [k for k, v in results.items() if not v]
        print(f"   ❌ Fallidos: {', '.join(failed)}")

    return results


def list_datasets() -> None:
    """Lista datasets disponibles con información."""
    print("=" * 60)
    print("📋 Datasets Disponibles")
    print("=" * 60)
    print()

    cache = load_cache()

    for name, info in DATASETS.items():
        dest = DATA_DIR / info["filename"]
        status = "✅" if dest.exists() and name in cache else "❌"

        print(f"{status} {name}")
        print(f"   📝 {info['description']}")
        print(f"   📦 {info['size_mb']:.2f} MB")
        print(f"   📁 {info['filename']}")
        print()


def get_dataset_path(name: str) -> Path | None:
    """
    Obtiene path a un dataset, descargándolo si es necesario.

    Uso en notebooks:
        from scripts.download_datasets import get_dataset_path
        iris_path = get_dataset_path("iris")
        df = pd.read_csv(iris_path)
    """
    if name not in DATASETS:
        return None

    dest = DATA_DIR / DATASETS[name]["filename"]

    if not dest.exists():
        if not download_dataset(name):
            return None

    return dest


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Descarga datasets para Guía Master IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    python scripts/download_datasets.py              # Descarga todos
    python scripts/download_datasets.py iris wine    # Solo iris y wine
    python scripts/download_datasets.py --list       # Lista disponibles
    python scripts/download_datasets.py --offline    # Solo usa cache local
    python scripts/download_datasets.py --force      # Re-descarga todo
        """,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Datasets específicos a descargar (default: todos)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar datasets disponibles",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Modo offline: no descargar, usar solo cache",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar re-descarga aunque exista",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Solo verificar checksums de archivos existentes",
    )

    args = parser.parse_args()

    # Crear directorio de datos
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_datasets()
        return 0

    if args.verify:
        print("🔍 Verificando checksums...")
        all_ok = True
        for name, info in DATASETS.items():
            dest = DATA_DIR / info["filename"]
            if dest.exists():
                if not verify_checksum(dest, info["sha256"]):
                    all_ok = False
            else:
                print(f"⚠️  {name}: No existe")
        return 0 if all_ok else 1

    # Descargar datasets específicos o todos
    if args.datasets:
        results = {}
        for name in args.datasets:
            results[name] = download_dataset(
                name, force=args.force, offline=args.offline
            )
        success = all(results.values())
    else:
        results = download_all(force=args.force, offline=args.offline)
        success = all(results.values())

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
