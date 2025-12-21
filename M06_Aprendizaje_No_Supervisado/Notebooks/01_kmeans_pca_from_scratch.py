#!/usr/bin/env python3
"""
Notebook M06: K-Means y PCA desde Cero
======================================
Implementación de algoritmos de aprendizaje no supervisado.

Ejecutar: python 01_kmeans_pca_from_scratch.py
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(seed=42)  # Reproducibilidad

# =============================================================================
# PARTE 1: K-Means Clustering
# =============================================================================

print("=" * 60)
print("PARTE 1: K-Means Clustering")
print("=" * 60)


def kmeans(
    X: np.ndarray,  # Datos (N, D)
    k: int,  # Número de clusters
    max_iters: int = 100,  # Máximo de iteraciones
    tol: float = 1e-4,  # Tolerancia para convergencia
) -> tuple[np.ndarray, np.ndarray]:
    """
    Algoritmo K-Means (Lloyd's algorithm).

    1. Inicializar centroides aleatoriamente
    2. Repetir hasta convergencia:
       a. Asignar cada punto al centroide más cercano
       b. Recalcular centroides como media de puntos asignados

    Returns:
        centroids: (K, D) - Centroides finales
        labels: (N,) - Asignación de cluster para cada punto
    """
    N, D = X.shape  # N muestras, D dimensiones

    # 1. Inicialización: elegir K puntos aleatorios como centroides
    random_indices = rng.choice(N, size=k, replace=False)  # K índices únicos
    centroids = X[random_indices].copy()  # Centroides iniciales (K, D)

    for iteration in range(max_iters):
        # 2a. Asignación: cada punto al centroide más cercano
        # Calcular distancias de cada punto a cada centroide
        # distances[i, j] = ||X[i] - centroids[j]||²
        distances = np.zeros((N, k))  # Matriz de distancias (N, K)
        for j in range(k):
            diff = X - centroids[j]  # (N, D) - Broadcasting
            distances[:, j] = np.sum(diff**2, axis=1)  # Distancia² a centroide j

        labels = np.argmin(distances, axis=1)  # (N,) - Cluster más cercano

        # 2b. Actualización: recalcular centroides
        new_centroids = np.zeros_like(centroids)  # (K, D)
        for j in range(k):
            mask = labels == j  # Puntos en cluster j
            if np.sum(mask) > 0:  # Evitar división por 0
                new_centroids[j] = X[mask].mean(axis=0)  # Media de puntos
            else:
                new_centroids[j] = centroids[j]  # Mantener si cluster vacío

        # Verificar convergencia
        shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))  # Movimiento total
        centroids = new_centroids  # Actualizar

        if shift < tol:
            print(f"  Convergió en iteración {iteration + 1}")
            break

    return centroids, labels


def kmeans_plusplus_init(X: np.ndarray, k: int) -> np.ndarray:
    """
    Inicialización K-Means++ (mejor que aleatoria).

    1. Elegir primer centroide aleatoriamente
    2. Para cada centroide adicional:
       - Calcular D(x)² = distancia al centroide más cercano
       - Elegir siguiente centroide con probabilidad ∝ D(x)²
    """
    N, D = X.shape
    centroids = np.zeros((k, D))  # (K, D)

    # Primer centroide aleatorio
    idx = int(rng.integers(0, N))
    centroids[0] = X[idx]

    for i in range(1, k):
        # Calcular distancia mínima al centroide más cercano para cada punto
        min_distances = np.full(N, np.inf)  # Inicializar con infinito
        for j in range(i):
            dist = np.sum((X - centroids[j]) ** 2, axis=1)  # Distancia² a centroide j
            min_distances = np.minimum(min_distances, dist)  # Mínimo

        # Probabilidad proporcional a D(x)²
        probs = min_distances / min_distances.sum()  # Normalizar
        idx = int(rng.choice(N, p=probs))  # Elegir con probabilidad ponderada
        centroids[i] = X[idx]

    return centroids


# --- Demo K-Means ---
print("\n--- Demo K-Means ---")

# Generar datos sintéticos: 3 clusters
cluster1 = rng.standard_normal((50, 2)) + np.array([0, 0])  # Cluster en (0, 0)
cluster2 = rng.standard_normal((50, 2)) + np.array([5, 5])  # Cluster en (5, 5)
cluster3 = rng.standard_normal((50, 2)) + np.array([0, 5])  # Cluster en (0, 5)
X_demo = np.vstack([cluster1, cluster2, cluster3])  # (150, 2)

print(f"Datos: {X_demo.shape}")
centroids, labels = kmeans(X_demo, k=3)
print(f"Centroides encontrados:\n{centroids}")
print(f"Distribución de labels: {np.bincount(labels)}")

# =============================================================================
# PARTE 2: PCA (Principal Component Analysis)
# =============================================================================

print("\n" + "=" * 60)
print("PARTE 2: PCA desde Cero")
print("=" * 60)


def pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA usando SVD (método numéricamente estable).

    Pasos:
    1. Centrar datos (restar media)
    2. Calcular SVD: X_centered = U @ Σ @ V^T
    3. Componentes principales = primeras columnas de V
    4. Proyección: X_transformed = X_centered @ V[:, :n_components]

    Returns:
        X_transformed: Datos proyectados (N, n_components)
        components: Componentes principales (D, n_components)
        explained_variance_ratio: Varianza explicada por componente
    """
    N, D = X.shape

    # 1. Centrar datos
    mean = X.mean(axis=0)  # (D,)
    X_centered = X - mean  # (N, D)

    # 2. SVD
    # X_centered = U @ diag(S) @ Vh
    # donde Vh tiene los componentes principales como filas
    U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)  # SVD económica

    # 3. Componentes principales (columnas de V = filas de Vh transpuestas)
    components = Vh[:n_components].T  # (D, n_components)

    # 4. Proyección
    X_transformed = X_centered @ components  # (N, n_components)

    # 5. Varianza explicada
    total_variance = np.sum(S**2) / (N - 1)  # Varianza total
    explained_variance = (S**2) / (N - 1)  # Varianza por componente
    explained_variance_ratio = explained_variance[:n_components] / total_variance

    return X_transformed, components, explained_variance_ratio


# --- Demo PCA ---
print("\n--- Demo PCA ---")

# Datos con correlación
n_samples = 200
X_corr = rng.standard_normal((n_samples, 2))  # Datos base
X_corr[:, 1] = 0.8 * X_corr[:, 0] + 0.2 * X_corr[:, 1]  # Correlación fuerte

print(f"Datos originales: {X_corr.shape}")
X_pca, components, var_ratio = pca(X_corr, n_components=2)
print(f"Componentes principales:\n{components}")
print(f"Varianza explicada: {var_ratio}")
print(f"Varianza total explicada: {var_ratio.sum():.4f}")

# --- PCA para reducción dimensional ---
print("\n--- PCA para Reducción Dimensional ---")

# Datos de alta dimensión
X_high_dim = rng.standard_normal((100, 50))  # 100 muestras, 50 features
X_reduced, _, var_ratio_high = pca(X_high_dim, n_components=10)
print(f"Original: {X_high_dim.shape} → Reducido: {X_reduced.shape}")
print(f"Varianza explicada por 10 componentes: {var_ratio_high.sum():.4f}")

# =============================================================================
# PARTE 3: Métricas de Evaluación
# =============================================================================

print("\n" + "=" * 60)
print("PARTE 3: Métricas de Evaluación")
print("=" * 60)


def inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    """
    Inercia (Within-Cluster Sum of Squares).

    WCSS = Σ Σ ||x - μ_k||²
           k  x∈C_k

    Mide qué tan compactos son los clusters.
    Menor es mejor.
    """
    total = 0.0
    for k in range(len(centroids)):
        mask = labels == k  # Puntos en cluster k
        if np.sum(mask) > 0:
            diff = X[mask] - centroids[k]  # Diferencia a centroide
            total += np.sum(diff**2)  # Suma de distancias²
    return float(total)


def silhouette_score_manual(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette Score (simplificado).

    Para cada punto i:
    - a(i) = distancia promedio a otros puntos en su cluster
    - b(i) = distancia promedio al cluster más cercano (que no sea el suyo)
    - s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Score global = promedio de s(i)
    Rango: [-1, 1], mayor es mejor
    """
    N = len(X)
    unique_labels = np.unique(labels)
    K = len(unique_labels)

    if K == 1:
        return 0.0  # No tiene sentido con 1 cluster

    silhouette_values = np.zeros(N)

    for i in range(N):
        # a(i): distancia promedio intra-cluster
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Excluir el punto mismo
        if np.sum(same_cluster) > 0:
            a_i = np.mean(np.sqrt(np.sum((X[same_cluster] - X[i]) ** 2, axis=1)))
        else:
            a_i = 0.0

        # b(i): distancia promedio al cluster más cercano
        b_i = np.inf
        for k in unique_labels:
            if k != labels[i]:
                other_cluster = labels == k
                if np.sum(other_cluster) > 0:
                    mean_dist = np.mean(
                        np.sqrt(np.sum((X[other_cluster] - X[i]) ** 2, axis=1))
                    )
                    b_i = min(b_i, mean_dist)

        # Silhouette para punto i
        if max(a_i, b_i) > 0:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)

    return float(np.mean(silhouette_values))


# --- Demo Métricas ---
print("\n--- Demo Métricas ---")

inertia_val = inertia(X_demo, centroids, labels)
silhouette_val = silhouette_score_manual(X_demo, labels)
print(f"Inercia (WCSS): {inertia_val:.2f}")
print(f"Silhouette Score: {silhouette_val:.4f}")

# =============================================================================
# PARTE 4: Método del Codo (Elbow Method)
# =============================================================================

print("\n" + "=" * 60)
print("PARTE 4: Método del Codo")
print("=" * 60)

print("\n--- Buscando K óptimo ---")
inertias = []
for k in range(1, 8):
    c, labels_k = kmeans(X_demo, k=k, max_iters=50)
    inertia_k = inertia(X_demo, c, labels_k)
    inertias.append(inertia_k)
    print(f"  K={k}: Inercia={inertia_k:.2f}")

print("\nEl 'codo' típicamente está en K=3 (donde la mejora marginal decrece)")

print("\n" + "=" * 60)
print("✅ Notebook M06 completado")
print("=" * 60)
