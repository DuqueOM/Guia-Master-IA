#!/usr/bin/env python3
"""
Simulacro de Examen: CSCA 5632 - Unsupervised Learning
======================================================

Módulo: M06 - Aprendizaje No Supervisado
Tiempo Estimado: 90 minutos
Puntuación Total: 100 puntos

Estructura:
- Parte A: Preguntas Teóricas (30 puntos)
- Parte B: Ejercicios de Código (70 puntos)

Criterio para aprobar con B: >= 80 puntos

Ejecutar tests: pytest tests/test_simulacro_csca5632.py -v
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

rng = np.random.default_rng(42)

# =============================================================================
# PARTE A: PREGUNTAS TEÓRICAS (30 puntos)
# =============================================================================

print("=" * 70)
print("PARTE A: PREGUNTAS TEÓRICAS (30 puntos)")
print("=" * 70)

# -----------------------------------------------------------------------------
# Pregunta A1 (8 puntos): K-Means
# -----------------------------------------------------------------------------
# a) ¿Qué función objetivo minimiza K-Means? Escribe la fórmula. (3 pts)
# b) ¿Por qué K-Means puede converger a un mínimo local? (2 pts)
# c) ¿Cómo ayuda K-Means++ a obtener mejores resultados? (3 pts)

respuesta_A1_a: str = ""  # TODO: Fórmula de la función objetivo
respuesta_A1_b: str = ""  # TODO: Explicación mínimo local
respuesta_A1_c: str = ""  # TODO: Explicación K-Means++


# -----------------------------------------------------------------------------
# Pregunta A2 (8 puntos): PCA
# -----------------------------------------------------------------------------
# Una matriz de datos X ∈ R^(100×50) se reduce a 10 componentes con PCA.
#
# a) ¿Cuáles son las dimensiones de la matriz de componentes principales V? (2 pts)
# b) ¿Cuáles son las dimensiones de los datos transformados? (2 pts)
# c) Si el primer componente explica 60% de varianza y el segundo 20%,
#    ¿cuánta varianza total explican los primeros 2 componentes? (2 pts)
# d) ¿Qué significa que los componentes principales sean ortogonales? (2 pts)

respuesta_A2_a: str = ""  # TODO: Dimensiones de V
respuesta_A2_b: str = ""  # TODO: Dimensiones de X_transformed
respuesta_A2_c: float = 0.0  # TODO: Varianza explicada
respuesta_A2_d: str = ""  # TODO: Explicación ortogonalidad


# -----------------------------------------------------------------------------
# Pregunta A3 (7 puntos): GMM y EM
# -----------------------------------------------------------------------------
# a) ¿Cuál es la diferencia principal entre K-Means y GMM? (3 pts)
# b) En el algoritmo EM, ¿qué se calcula en el E-step y qué en el M-step? (4 pts)

respuesta_A3_a: str = ""  # TODO: Diferencia K-Means vs GMM
respuesta_A3_b: str = ""  # TODO: E-step y M-step


# -----------------------------------------------------------------------------
# Pregunta A4 (7 puntos): Sistemas de Recomendación
# -----------------------------------------------------------------------------
# Una matriz de ratings R tiene dimensiones (1000 usuarios × 5000 items).
# Se factoriza como R ≈ P × Q^T con k=50 factores latentes.
#
# a) ¿Cuáles son las dimensiones de P y Q? (3 pts)
# b) ¿Por qué la factorización de matrices ayuda con el problema de sparsity? (4 pts)

respuesta_A4_a: str = ""  # TODO: Dimensiones de P y Q
respuesta_A4_b: str = ""  # TODO: Explicación sparsity


# =============================================================================
# PARTE B: EJERCICIOS DE CÓDIGO (70 puntos)
# =============================================================================

print("\n" + "=" * 70)
print("PARTE B: EJERCICIOS DE CÓDIGO (70 puntos)")
print("=" * 70)


# -----------------------------------------------------------------------------
# Ejercicio B1 (25 puntos): K-Means desde Cero
# -----------------------------------------------------------------------------
def euclidean_distance(x1: NDArray[np.float64], x2: NDArray[np.float64]) -> float:
    """Calcula distancia Euclidiana entre dos puntos."""
    # TODO: Implementar
    return 0.0  # Placeholder


def kmeans(
    X: NDArray[np.float64],
    k: int,
    max_iters: int = 100,
    tol: float = 1e-4,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Implementa K-Means clustering.

    Parámetros
    ----------
    X : NDArray[np.float64]
        Datos de entrada (n_samples, n_features).
    k : int
        Número de clusters.
    max_iters : int
        Máximo de iteraciones.
    tol : float
        Tolerancia para convergencia.

    Retorna
    -------
    tuple[NDArray[np.float64], NDArray[np.int64]]
        - centroids: Centroides finales (k, n_features)
        - labels: Asignación de cluster para cada punto (n_samples,)

    Algoritmo:
    ----------
    1. Inicializar centroides aleatoriamente (k puntos de X)
    2. Repetir hasta convergencia:
       a. Asignar cada punto al centroide más cercano
       b. Recalcular centroides como media de puntos asignados
    """
    n_samples, n_features = X.shape

    # TODO: Inicializar centroides (elegir k puntos aleatorios de X)
    indices = rng.choice(n_samples, size=k, replace=False)
    centroids = X[indices].copy()

    labels = np.zeros(n_samples, dtype=np.int64)

    for _ in range(max_iters):
        # TODO: Paso 1 - Asignar cada punto al centroide más cercano
        pass

        # TODO: Paso 2 - Recalcular centroides
        pass

        # TODO: Verificar convergencia

    return centroids, labels


def compute_inertia(
    X: NDArray[np.float64],
    centroids: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    """
    Calcula la inercia (suma de distancias² al centroide asignado).

    Esta es la función objetivo que K-Means minimiza.
    """
    # TODO: Implementar
    return 0.0  # Placeholder


# Test B1
print("\n--- Test B1: K-Means ---")
# Generar 3 clusters sintéticos
cluster1 = rng.standard_normal((50, 2)) + np.array([0, 0])
cluster2 = rng.standard_normal((50, 2)) + np.array([5, 5])
cluster3 = rng.standard_normal((50, 2)) + np.array([0, 5])
X_kmeans = np.vstack([cluster1, cluster2, cluster3])

centroids, labels = kmeans(X_kmeans, k=3)
print(f"Centroides:\n{centroids}")
print(f"Distribución de labels: {np.bincount(labels)}")


# -----------------------------------------------------------------------------
# Ejercicio B2 (25 puntos): PCA desde Cero
# -----------------------------------------------------------------------------
def pca(
    X: NDArray[np.float64],
    n_components: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Implementa PCA usando SVD.

    Parámetros
    ----------
    X : NDArray[np.float64]
        Datos de entrada (n_samples, n_features).
    n_components : int
        Número de componentes principales a retener.

    Retorna
    -------
    tuple[NDArray, NDArray, NDArray]
        - X_transformed: Datos proyectados (n_samples, n_components)
        - components: Componentes principales (n_features, n_components)
        - explained_variance_ratio: Varianza explicada por componente

    Algoritmo:
    ----------
    1. Centrar datos (restar media)
    2. Calcular SVD: X_centered = U @ Σ @ V^T
    3. Componentes = columnas de V (eigenvectors de X^T X)
    4. Proyección: X_transformed = X_centered @ V[:, :n_components]
    """
    n_samples, n_features = X.shape

    # TODO: Paso 1 - Centrar datos
    X_centered = X - X.mean(axis=0)  # noqa: F841

    # TODO: Paso 2 - Calcular SVD
    # U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # TODO: Paso 3 - Calcular varianza explicada
    # explained_variance = S² / (n_samples - 1)
    # explained_variance_ratio = explained_variance / sum(explained_variance)

    # TODO: Paso 4 - Proyectar datos
    # components = Vt.T[:, :n_components]
    # X_transformed = X_centered @ components

    # Placeholder
    X_transformed = np.zeros((n_samples, n_components))
    components = np.zeros((n_features, n_components))
    explained_variance_ratio = np.zeros(n_components)

    return X_transformed, components, explained_variance_ratio


# Test B2
print("\n--- Test B2: PCA ---")
X_pca = rng.standard_normal((100, 10))
X_transformed, components, var_ratio = pca(X_pca, n_components=3)
print(f"Shape original: {X_pca.shape}")
print(f"Shape transformado: {X_transformed.shape}")
print(f"Varianza explicada: {var_ratio}")


# -----------------------------------------------------------------------------
# Ejercicio B3 (20 puntos): Matrix Factorization para Recomendación
# -----------------------------------------------------------------------------
def matrix_factorization_sgd(
    R: NDArray[np.float64],
    k: int,
    n_epochs: int = 100,
    lr: float = 0.01,
    reg: float = 0.1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Factorización de matrices con SGD para sistemas de recomendación.

    Aproxima R ≈ P @ Q.T donde R es la matriz de ratings.

    Parámetros
    ----------
    R : NDArray[np.float64]
        Matriz de ratings (n_users, n_items). NaN = no rating.
    k : int
        Número de factores latentes.
    n_epochs : int
        Número de épocas.
    lr : float
        Learning rate.
    reg : float
        Parámetro de regularización L2.

    Retorna
    -------
    tuple[NDArray, NDArray]
        - P: Matriz de usuarios (n_users, k)
        - Q: Matriz de items (n_items, k)

    Algoritmo:
    ----------
    Para cada rating conocido r_ui:
    1. Predecir: r̂_ui = p_u · q_i
    2. Calcular error: e_ui = r_ui - r̂_ui
    3. Actualizar:
       p_u = p_u + lr * (e_ui * q_i - reg * p_u)
       q_i = q_i + lr * (e_ui * p_u - reg * q_i)
    """
    n_users, n_items = R.shape

    # TODO: Inicializar P y Q con valores pequeños aleatorios
    P = rng.standard_normal((n_users, k)) * 0.1
    Q = rng.standard_normal((n_items, k)) * 0.1

    # Encontrar índices de ratings conocidos (no NaN)
    _known_ratings = ~np.isnan(R)  # noqa: F841

    for _ in range(n_epochs):
        # TODO: Iterar sobre ratings conocidos
        # TODO: Calcular predicción y error
        # TODO: Actualizar P y Q
        pass

    return P, Q


def predict_rating(
    P: NDArray[np.float64],
    Q: NDArray[np.float64],
    user_idx: int,
    item_idx: int,
) -> float:
    """Predice el rating de un usuario para un item."""
    # TODO: Implementar r̂_ui = p_u · q_i
    return 0.0  # Placeholder


# Test B3
print("\n--- Test B3: Matrix Factorization ---")
# Matriz pequeña de ratings (NaN = no rating)
R_test = np.array(
    [
        [5, 3, np.nan, 1],
        [4, np.nan, np.nan, 1],
        [1, 1, np.nan, 5],
        [1, np.nan, np.nan, 4],
        [np.nan, 1, 5, 4],
    ]
)
P, Q = matrix_factorization_sgd(R_test, k=2, n_epochs=100)
print(f"Shape P: {P.shape}")
print(f"Shape Q: {Q.shape}")

# Predecir rating faltante
pred = predict_rating(P, Q, user_idx=0, item_idx=2)
print(f"Rating predicho [0, 2]: {pred:.2f}")


# =============================================================================
# VALIDACIÓN FINAL
# =============================================================================

print("\n" + "=" * 70)
print("VALIDACIÓN FINAL")
print("=" * 70)


def validar_simulacro() -> dict[str, bool]:
    """Valida todas las respuestas del simulacro."""
    resultados: dict[str, bool] = {}

    # Validar A2 (PCA dimensiones)
    resultados["A2_varianza"] = abs(respuesta_A2_c - 0.80) < 0.01

    # Validar B1 (K-Means)
    X_val = np.vstack(
        [
            rng.standard_normal((30, 2)) + np.array([0, 0]),
            rng.standard_normal((30, 2)) + np.array([10, 10]),
        ]
    )
    centroids_val, labels_val = kmeans(X_val, k=2)
    # Verificar que encontró 2 clusters distintos
    resultados["B1_kmeans"] = len(np.unique(labels_val)) == 2

    # Validar B2 (PCA)
    X_pca_val = rng.standard_normal((50, 5))
    X_t, comp, var = pca(X_pca_val, n_components=2)
    resultados["B2_pca_shape"] = X_t.shape == (50, 2)
    resultados["B2_pca_variance"] = len(var) == 2 and bool(np.all(var >= 0))

    # Validar B3 (Matrix Factorization)
    P_val, Q_val = matrix_factorization_sgd(R_test, k=2, n_epochs=50)
    resultados["B3_mf_shapes"] = P_val.shape == (5, 2) and Q_val.shape == (4, 2)

    return resultados


# Ejecutar validación
print("\n🔍 Validando respuestas...")
resultados = validar_simulacro()

puntos = 0
for test, passed in resultados.items():
    status = "✅" if passed else "❌"
    pts = 12 if passed else 0
    puntos += pts
    print(f"  {status} {test}: {pts} pts")

print(f"\n📊 PUNTUACIÓN ESTIMADA: {puntos}/70 (solo código)")
print("   + Parte Teórica: /30 (requiere revisión manual)")

if puntos >= 56:
    print("\n🎉 ¡Vas bien! El código cumple el criterio para B.")
else:
    print("\n⚠️ Necesitas revisar las implementaciones antes del examen real.")

print("\n" + "=" * 70)
print("FIN DEL SIMULACRO")
print("=" * 70)
