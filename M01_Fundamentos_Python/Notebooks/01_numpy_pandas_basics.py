#!/usr/bin/env python3
"""
Notebook M01: Fundamentos de NumPy y Pandas
============================================
Ejercicios prácticos para dominar las herramientas base de ML.

Ejecutar: python 01_numpy_pandas_basics.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)

# =============================================================================
# PARTE 1: NumPy - Arrays y Operaciones Vectorizadas
# =============================================================================

print("=" * 60)
print("PARTE 1: NumPy Fundamentals")
print("=" * 60)

# --- 1.1 Creación de Arrays ---
print("\n--- 1.1 Creación de Arrays ---")

# Desde lista
arr_lista = np.array([1, 2, 3, 4, 5])  # Array 1D desde lista Python
print(f"Desde lista: {arr_lista}, shape: {arr_lista.shape}")

# Arrays especiales
zeros = np.zeros((3, 4))  # Matriz de ceros 3x4
ones = np.ones((2, 3))  # Matriz de unos 2x3
eye = np.eye(3)  # Matriz identidad 3x3
rango = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8] - start, stop, step
linspace = np.linspace(0, 1, 5)  # 5 puntos equiespaciados entre 0 y 1

print(f"zeros shape: {zeros.shape}")
print(f"eye:\n{eye}")
print(f"linspace: {linspace}")

# --- 1.2 Indexing y Slicing ---
print("\n--- 1.2 Indexing y Slicing ---")

X = np.arange(12).reshape(3, 4)  # Matriz 3x4 con valores 0-11
print(f"X:\n{X}")
print(f"X[0, 0] = {X[0, 0]}")  # Elemento (0,0)
print(f"X[1, :] = {X[1, :]}")  # Fila 1 completa
print(f"X[:, 2] = {X[:, 2]}")  # Columna 2 completa
print(f"X[0:2, 1:3] =\n{X[0:2, 1:3]}")  # Submatriz

# --- 1.3 Broadcasting ---
print("\n--- 1.3 Broadcasting ---")

A = np.array([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
b = np.array([10, 20, 30])  # Shape (3,)
resultado = A + b  # Broadcasting: b se "expande" a (2, 3)
print(f"A + b =\n{resultado}")

# Regla de broadcasting: dimensiones compatibles si son iguales o una es 1
c = np.array([[100], [200]])  # Shape (2, 1)
resultado2 = A + c  # c se expande a (2, 3)
print(f"A + c =\n{resultado2}")

# --- 1.4 Operaciones Vectorizadas (¡NO usar loops!) ---
print("\n--- 1.4 Operaciones Vectorizadas ---")

data = rng.standard_normal(1000, dtype=float)  # 1000 números aleatorios N(0,1)

# MAL (lento):
# suma = 0
# for x in data:
#     suma += x
# media_loop = suma / len(data)

# BIEN (vectorizado):
media = np.mean(data)  # Promedio vectorizado
std = np.std(data)  # Desviación estándar
print(f"Media: {media:.4f}, Std: {std:.4f}")

# Operaciones elemento a elemento
arr = np.array([1, 4, 9, 16])
print(f"sqrt: {np.sqrt(arr)}")  # Raíz cuadrada
print(f"exp: {np.exp(arr[:2])}")  # Exponencial
print(f"log: {np.log(arr)}")  # Logaritmo natural

# --- 1.5 Shapes y Reshape ---
print("\n--- 1.5 Shapes y Reshape ---")

original = np.arange(24)  # Shape (24,)
reshaped = original.reshape(4, 6)  # Shape (4, 6)
reshaped_3d = original.reshape(2, 3, 4)  # Shape (2, 3, 4)

print(f"Original shape: {original.shape}")
print(f"Reshaped 2D: {reshaped.shape}")
print(f"Reshaped 3D: {reshaped_3d.shape}")

# Flatten vs Ravel
flat = reshaped.flatten()  # Copia
ravel = reshaped.ravel()  # Vista (más eficiente)
print(f"Flatten shape: {flat.shape}")

# =============================================================================
# PARTE 2: Pandas - DataFrames y Manipulación de Datos
# =============================================================================

print("\n" + "=" * 60)
print("PARTE 2: Pandas Fundamentals")
print("=" * 60)

# --- 2.1 Creación de DataFrames ---
print("\n--- 2.1 Creación de DataFrames ---")

# Desde diccionario
data_dict = {
    "nombre": ["Alice", "Bob", "Charlie", "Diana"],
    "edad": [25, 30, 35, 28],
    "salario": [50000, 60000, 70000, 55000],
    "departamento": ["IT", "HR", "IT", "Finance"],
}
df = pd.DataFrame(data_dict)
print(df)

# --- 2.2 Selección de Datos ---
print("\n--- 2.2 Selección de Datos ---")

print(f"Columna 'nombre':\n{df['nombre']}")
print(f"\nFilas donde edad > 28:\n{df[df['edad'] > 28]}")
column_slice = ["nombre", "edad", "salario"]
print(f"\nloc (columnas específicas):\n{df.loc[0:2, column_slice]}")
print(f"\niloc (índices): df.iloc[0:2, 0:2]:\n{df.iloc[0:2, 0:2]}")

# --- 2.3 Operaciones Comunes ---
print("\n--- 2.3 Operaciones Comunes ---")

print(f"Estadísticas:\n{df.describe()}")
print(f"\nValores únicos en departamento: {df['departamento'].unique()}")
print(f"Conteo por departamento:\n{df['departamento'].value_counts()}")

# GroupBy
print(
    f"\nSalario promedio por departamento:\n{df.groupby('departamento')['salario'].mean()}"
)

# --- 2.4 Manejo de Datos Faltantes ---
print("\n--- 2.4 Manejo de Datos Faltantes ---")

df_con_nan = df.copy()
df_con_nan.loc[1, "salario"] = np.nan  # Introducir NaN
print(f"DataFrame con NaN:\n{df_con_nan}")
print(f"\nNaN por columna:\n{df_con_nan.isna().sum()}")

# Opciones para manejar NaN
df_dropna = df_con_nan.dropna()  # Eliminar filas con NaN
df_fillna = df_con_nan.fillna(df_con_nan["salario"].mean())  # Rellenar con media
print(f"\nRellenado con media:\n{df_fillna}")

# =============================================================================
# PARTE 3: Ejercicios de Práctica
# =============================================================================

print("\n" + "=" * 60)
print("PARTE 3: Ejercicios de Práctica")
print("=" * 60)

# Ejercicio 1: Normalización (muy común en ML)
print("\n--- Ejercicio 1: Normalización ---")
X_raw = (
    rng.standard_normal((100, 5), dtype=float) * 10 + 50
)  # Datos con media ~50, std ~10

# Z-score normalization: (x - mean) / std
X_mean = X_raw.mean(axis=0)  # Media por columna
X_std = X_raw.std(axis=0)  # Std por columna
X_normalized = (X_raw - X_mean) / X_std  # Broadcasting automático

print(f"Antes - Media: {X_raw.mean(axis=0)[:3]}")
print(f"Después - Media: {X_normalized.mean(axis=0)[:3]}")
print(f"Después - Std: {X_normalized.std(axis=0)[:3]}")

# Ejercicio 2: One-Hot Encoding manual
print("\n--- Ejercicio 2: One-Hot Encoding ---")
categorias = np.array([0, 1, 2, 0, 1])  # 3 categorías
n_categorias = 3
one_hot = np.eye(n_categorias)[categorias]  # Truco elegante con indexing
print(f"Categorías: {categorias}")
print(f"One-hot:\n{one_hot}")

# Ejercicio 3: Distancia Euclidiana vectorizada
print("\n--- Ejercicio 3: Distancia Euclidiana ---")
punto_a = np.array([1, 2, 3])
punto_b = np.array([4, 5, 6])
distancia = np.sqrt(np.sum((punto_a - punto_b) ** 2))  # ||a - b||_2
distancia_alt = np.linalg.norm(punto_a - punto_b)  # Forma más limpia
print(f"Distancia: {distancia:.4f} (alternativa: {distancia_alt:.4f})")

print("\n" + "=" * 60)
print("✅ Notebook M01 completado")
print("=" * 60)
