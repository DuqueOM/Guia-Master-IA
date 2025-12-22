#!/usr/bin/env python3
"""
Carga Robusta de Datos para M08 Proyecto Integrador
====================================================

Este módulo proporciona funciones para cargar el dataset de Kaggle
"NLP with Disaster Tweets" de forma robusta, con soporte para:
- Detección automática de rutas comunes
- Montaje de Google Drive en Colab
- Mensajes de error amigables con instrucciones

Uso:
    from src.data_loader import load_train_data
    df = load_train_data()
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def running_in_colab() -> bool:
    """Detecta si el código se ejecuta en Google Colab."""
    return "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ


def try_mount_gdrive_if_colab() -> None:
    """Monta Google Drive automáticamente si estamos en Colab."""
    if not running_in_colab():
        return
    try:
        from google.colab import drive  # type: ignore

        print("🔎 Detectado Google Colab. Montando Google Drive...")
        drive.mount("/content/drive")
        print("✅ Drive montado en /content/drive")
    except Exception as e:  # noqa: BLE001
        print("⚠️ No se pudo montar Google Drive automáticamente.")
        print(f"   Detalle: {e}")


def find_dataset_file(filename: str = "train.csv") -> Path | None:
    """
    Busca el archivo del dataset en ubicaciones típicas.

    Args:
        filename: Nombre del archivo a buscar (default: train.csv)

    Returns:
        Path al archivo si existe, None si no se encuentra.
    """
    candidates = [
        # Rutas relativas desde diferentes ubicaciones
        Path(filename),
        Path(f"data/{filename}"),
        Path(f"data/raw/{filename}"),
        Path(f"datasets/{filename}"),
        Path(f"datasets/raw/{filename}"),
        # Rutas desde raíz del proyecto
        Path(f"M08_Proyecto_Integrador/data/{filename}"),
        Path(f"M08_Proyecto_Integrador/data/raw/{filename}"),
        Path(f"M08_Proyecto_Integrador/datasets/{filename}"),
        # Rutas Colab
        Path(f"/content/{filename}"),
        Path(f"/content/data/{filename}"),
        Path(f"/content/data/raw/{filename}"),
        # Rutas Google Drive
        Path(f"/content/drive/MyDrive/{filename}"),
        Path(f"/content/drive/MyDrive/data/{filename}"),
        Path(
            f"/content/drive/MyDrive/Guia-Master-IA/M08_Proyecto_Integrador/data/{filename}"
        ),
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def print_download_instructions() -> None:
    """Imprime instrucciones para descargar el dataset."""
    print("\n" + "=" * 60)
    print("⚠️  ERROR: No se encuentra el dataset train.csv")
    print("=" * 60)
    print("\nPasos para obtener los datos:\n")
    print("1) Ve a: https://www.kaggle.com/c/nlp-getting-started/data")
    print("2) Descarga: train.csv (y opcionalmente test.csv)")
    print("3) Colócalo en una de estas ubicaciones:")
    print("   - M08_Proyecto_Integrador/data/raw/train.csv")
    print("   - M08_Proyecto_Integrador/datasets/train.csv")
    print("   - ./data/train.csv (relativo al notebook)")
    print("\n" + "=" * 60 + "\n")


def load_train_data(filename: str = "train.csv") -> pd.DataFrame:
    """
    Carga el dataset de entrenamiento de forma robusta.

    Args:
        filename: Nombre del archivo (default: train.csv)

    Returns:
        DataFrame con los datos de entrenamiento.

    Raises:
        FileNotFoundError: Si no se encuentra el archivo.
    """
    # Intentar montar Drive si estamos en Colab
    try_mount_gdrive_if_colab()

    # Buscar el archivo
    path = find_dataset_file(filename)

    if path is None:
        print_download_instructions()
        raise FileNotFoundError(f"{filename} no encontrado. Ver instrucciones arriba.")

    print(f"✅ Dataset encontrado: {path}")
    df = pd.read_csv(path)
    print(f"   Filas: {len(df):,} | Columnas: {df.shape[1]}")

    return df


if __name__ == "__main__":
    # Test de carga
    try:
        df = load_train_data()
        print(df.head())
    except FileNotFoundError:
        print("Ejecuta las instrucciones anteriores para continuar.")
