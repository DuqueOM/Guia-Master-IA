#!/usr/bin/env python3
"""
Simulacro de Examen: CSCA 5622 - Supervised Learning
=====================================================

Módulo: M05 - Aprendizaje Supervisado
Tiempo Estimado: 90 minutos
Puntuación Total: 100 puntos

Estructura:
- Parte A: Preguntas Teóricas (30 puntos)
- Parte B: Ejercicios de Código (70 puntos)

Criterio para aprobar con B: >= 80 puntos

Instrucciones:
1. Ejecutar todas las celdas en orden
2. Completar las funciones marcadas con # TODO
3. Ejecutar los tests al final para validar
4. No modificar las funciones de test

Ejecutar tests: pytest tests/test_simulacro_csca5622.py -v
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

rng = np.random.default_rng(42)

# =============================================================================
# PARTE A: PREGUNTAS TEÓRICAS (30 puntos)
# =============================================================================
# Responde en las variables indicadas (string)

print("=" * 70)
print("PARTE A: PREGUNTAS TEÓRICAS (30 puntos)")
print("=" * 70)

# -----------------------------------------------------------------------------
# Pregunta A1 (8 puntos): Bias-Variance Tradeoff
# -----------------------------------------------------------------------------
# Un modelo de regresión tiene MSE = 25 en el conjunto de test.
# Sabes que Bias² = 9 y la varianza irreducible (ruido) = 4.
#
# a) ¿Cuál es la Variance del modelo? (2 pts)
# b) ¿El modelo sufre más de underfitting o overfitting? Justifica. (3 pts)
# c) ¿Qué acción tomarías para mejorar el modelo? (3 pts)

respuesta_A1_a: float = 0.0  # TODO: Reemplazar con el valor correcto
respuesta_A1_b: str = ""  # TODO: "underfitting" o "overfitting" + justificación
respuesta_A1_c: str = ""  # TODO: Acción específica


# -----------------------------------------------------------------------------
# Pregunta A2 (7 puntos): Regularización
# -----------------------------------------------------------------------------
# En regresión Ridge, la función de costo es:
# J(w) = MSE + λ||w||₂²
#
# a) ¿Qué sucede con los pesos w cuando λ → ∞? (2 pts)
# b) ¿Qué sucede cuando λ = 0? (2 pts)
# c) ¿Por qué Ridge NO produce pesos exactamente 0 pero Lasso sí? (3 pts)

respuesta_A2_a: str = ""  # TODO
respuesta_A2_b: str = ""  # TODO
respuesta_A2_c: str = ""  # TODO


# -----------------------------------------------------------------------------
# Pregunta A3 (8 puntos): Métricas de Clasificación
# -----------------------------------------------------------------------------
# Un clasificador de spam tiene la siguiente matriz de confusión:
#
#                    Predicho
#                  Spam    No-Spam
# Real  Spam       80        20
#       No-Spam    10       890
#
# a) Calcula Precision para la clase "Spam" (2 pts)
# b) Calcula Recall para la clase "Spam" (2 pts)
# c) Si el costo de un False Negative (spam no detectado) es 10x mayor
#    que un False Positive, ¿qué métrica priorizarías? (4 pts)

respuesta_A3_precision: float = 0.0  # TODO
respuesta_A3_recall: float = 0.0  # TODO
respuesta_A3_c: str = ""  # TODO


# -----------------------------------------------------------------------------
# Pregunta A4 (7 puntos): Árboles de Decisión
# -----------------------------------------------------------------------------
# a) ¿Qué mide el Gini Impurity? Escribe la fórmula. (3 pts)
# b) ¿Por qué Random Forest reduce la varianza comparado con un solo árbol? (4 pts)

respuesta_A4_a: str = ""  # TODO
respuesta_A4_b: str = ""  # TODO


# =============================================================================
# PARTE B: EJERCICIOS DE CÓDIGO (70 puntos)
# =============================================================================

print("\n" + "=" * 70)
print("PARTE B: EJERCICIOS DE CÓDIGO (70 puntos)")
print("=" * 70)


# -----------------------------------------------------------------------------
# Ejercicio B1 (20 puntos): Regresión Lineal desde Cero
# -----------------------------------------------------------------------------
def linear_regression_gradient_descent(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
) -> NDArray[np.float64]:
    """
    Implementa Regresión Lineal usando Gradiente Descendente.

    Parámetros
    ----------
    X : NDArray[np.float64]
        Matriz de características (n_samples, n_features).
        Ya incluye columna de 1s para el bias.
    y : NDArray[np.float64]
        Vector objetivo (n_samples,).
    learning_rate : float
        Tasa de aprendizaje α.
    n_iterations : int
        Número de iteraciones.

    Retorna
    -------
    NDArray[np.float64]
        Vector de pesos optimizados (n_features,).

    Fórmulas:
    ---------
    - Predicción: ŷ = Xw
    - Gradiente MSE: ∇w = (2/n) * X^T * (Xw - y)
    - Actualización: w = w - α * ∇w
    """
    n_samples, n_features = X.shape

    # TODO: Inicializar pesos con ceros
    weights = np.zeros(n_features)  # Placeholder

    # TODO: Implementar gradiente descendente
    for _ in range(n_iterations):
        # TODO: Calcular predicciones
        # TODO: Calcular gradiente
        # TODO: Actualizar pesos
        pass

    return weights


# Test B1
print("\n--- Test B1: Regresión Lineal ---")
X_test = np.column_stack([np.ones(100), rng.standard_normal((100, 2))])
true_weights = np.array([2.0, 1.5, -0.5])
y_test = X_test @ true_weights + rng.standard_normal(100) * 0.1

weights_pred = linear_regression_gradient_descent(X_test, y_test, learning_rate=0.1)
print(f"Pesos verdaderos: {true_weights}")
print(f"Pesos estimados:  {weights_pred}")


# -----------------------------------------------------------------------------
# Ejercicio B2 (25 puntos): Regresión Logística desde Cero
# -----------------------------------------------------------------------------
def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Función sigmoide: σ(z) = 1 / (1 + exp(-z))."""
    # TODO: Implementar sigmoid (manejar overflow)
    return np.zeros_like(z)  # Placeholder


def logistic_regression_gradient_descent(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    learning_rate: float = 0.1,
    n_iterations: int = 1000,
) -> NDArray[np.float64]:
    """
    Implementa Regresión Logística usando Gradiente Descendente.

    Parámetros
    ----------
    X : NDArray[np.float64]
        Matriz de características (n_samples, n_features).
    y : NDArray[np.float64]
        Vector de labels binarios (n_samples,) con valores 0 o 1.
    learning_rate : float
        Tasa de aprendizaje α.
    n_iterations : int
        Número de iteraciones.

    Retorna
    -------
    NDArray[np.float64]
        Vector de pesos optimizados (n_features,).

    Fórmulas:
    ---------
    - Predicción: p = σ(Xw)
    - Gradiente BCE: ∇w = (1/n) * X^T * (p - y)
    - Actualización: w = w - α * ∇w
    """
    n_samples, n_features = X.shape

    # TODO: Inicializar pesos
    weights = np.zeros(n_features)

    # TODO: Implementar gradiente descendente
    for _ in range(n_iterations):
        # TODO: Calcular probabilidades con sigmoid
        # TODO: Calcular gradiente
        # TODO: Actualizar pesos
        pass

    return weights


def predict_proba(
    X: NDArray[np.float64], weights: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Predice probabilidades P(y=1|X)."""
    # TODO: Implementar
    return np.zeros(X.shape[0])  # Placeholder


def predict(
    X: NDArray[np.float64], weights: NDArray[np.float64], threshold: float = 0.5
) -> NDArray[np.int64]:
    """Predice clases binarias."""
    # TODO: Implementar
    return np.zeros(X.shape[0], dtype=np.int64)  # Placeholder


# Test B2
print("\n--- Test B2: Regresión Logística ---")
X_log = np.column_stack([np.ones(200), rng.standard_normal((200, 2))])
true_w_log = np.array([0.0, 2.0, -1.5])
y_log = (sigmoid(X_log @ true_w_log) > 0.5).astype(int)

weights_log = logistic_regression_gradient_descent(X_log, y_log, learning_rate=0.5)
preds = predict(X_log, weights_log)
accuracy = np.mean(preds == y_log)
print(f"Accuracy: {accuracy:.2%}")


# -----------------------------------------------------------------------------
# Ejercicio B3 (25 puntos): Árbol de Decisión - Gini Impurity
# -----------------------------------------------------------------------------
def gini_impurity(y: NDArray[np.int64]) -> float:
    """
    Calcula el Gini Impurity de un conjunto de labels.

    Gini = 1 - Σ p_i²

    donde p_i es la proporción de la clase i.

    Parámetros
    ----------
    y : NDArray[np.int64]
        Vector de labels (n_samples,).

    Retorna
    -------
    float
        Gini impurity entre 0 (puro) y 0.5 (máxima impureza para binario).
    """
    if len(y) == 0:
        return 0.0

    # TODO: Implementar Gini impurity
    return 0.0  # Placeholder


def information_gain(
    y_parent: NDArray[np.int64],
    y_left: NDArray[np.int64],
    y_right: NDArray[np.int64],
) -> float:
    """
    Calcula el Information Gain de un split.

    IG = Gini(parent) - [n_left/n * Gini(left) + n_right/n * Gini(right)]

    Parámetros
    ----------
    y_parent : NDArray[np.int64]
        Labels del nodo padre.
    y_left : NDArray[np.int64]
        Labels del hijo izquierdo.
    y_right : NDArray[np.int64]
        Labels del hijo derecho.

    Retorna
    -------
    float
        Information gain (siempre >= 0).
    """
    n = len(y_parent)
    if n == 0:
        return 0.0

    # TODO: Implementar information gain
    return 0.0  # Placeholder


def find_best_split(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
) -> tuple[int, float, float]:
    """
    Encuentra el mejor split para un nodo.

    Parámetros
    ----------
    X : NDArray[np.float64]
        Matriz de características (n_samples, n_features).
    y : NDArray[np.int64]
        Vector de labels (n_samples,).

    Retorna
    -------
    tuple[int, float, float]
        - best_feature: índice de la mejor característica
        - best_threshold: valor del umbral
        - best_gain: information gain del split
    """
    best_feature = 0
    best_threshold = 0.0
    best_gain = 0.0

    n_samples, n_features = X.shape

    # TODO: Iterar sobre features y thresholds
    # TODO: Encontrar el split con mayor information gain

    return best_feature, best_threshold, best_gain


# Test B3
print("\n--- Test B3: Gini Impurity ---")
y_pure = np.array([1, 1, 1, 1])
y_impure = np.array([0, 0, 1, 1])
y_mixed = np.array([0, 0, 0, 1, 1, 1, 1, 1])

print(f"Gini puro (esperado ~0.0): {gini_impurity(y_pure):.4f}")
print(f"Gini 50/50 (esperado ~0.5): {gini_impurity(y_impure):.4f}")
print(f"Gini 3/5 (esperado ~0.469): {gini_impurity(y_mixed):.4f}")


# =============================================================================
# VALIDACIÓN FINAL
# =============================================================================

print("\n" + "=" * 70)
print("VALIDACIÓN FINAL")
print("=" * 70)


def validar_simulacro() -> dict[str, bool]:
    """Valida todas las respuestas del simulacro."""
    resultados: dict[str, bool] = {}

    # Validar A1
    resultados["A1_variance"] = (
        abs(respuesta_A1_a - 12.0) < 0.1
    )  # MSE = Bias² + Var + Noise
    resultados["A1_diagnostico"] = "overfitting" in respuesta_A1_b.lower()

    # Validar A3 (métricas)
    resultados["A3_precision"] = abs(respuesta_A3_precision - 80 / 90) < 0.01
    resultados["A3_recall"] = abs(respuesta_A3_recall - 80 / 100) < 0.01

    # Validar B1 (regresión lineal)
    X_val = np.column_stack([np.ones(50), rng.standard_normal((50, 2))])
    true_w = np.array([1.0, 2.0, -1.0])
    y_val = X_val @ true_w
    w_pred = linear_regression_gradient_descent(
        X_val, y_val, learning_rate=0.1, n_iterations=1000
    )
    resultados["B1_linear_reg"] = np.allclose(w_pred, true_w, atol=0.1)

    # Validar B2 (sigmoid)
    resultados["B2_sigmoid"] = np.allclose(
        sigmoid(np.array([0.0])), np.array([0.5]), atol=0.01
    )

    # Validar B3 (gini)
    resultados["B3_gini_pure"] = abs(gini_impurity(np.array([1, 1, 1])) - 0.0) < 0.01
    resultados["B3_gini_impure"] = abs(gini_impurity(np.array([0, 1])) - 0.5) < 0.01

    return resultados


# Ejecutar validación
print("\n🔍 Validando respuestas...")
resultados = validar_simulacro()

puntos = 0
for test, passed in resultados.items():
    status = "✅" if passed else "❌"
    pts = 10 if passed else 0
    puntos += pts
    print(f"  {status} {test}: {pts} pts")

print(f"\n📊 PUNTUACIÓN ESTIMADA: {puntos}/70 (solo código)")
print("   + Parte Teórica: /30 (requiere revisión manual)")

if puntos >= 56:  # 80% de 70
    print("\n🎉 ¡Vas bien! El código cumple el criterio para B.")
else:
    print("\n⚠️ Necesitas revisar las implementaciones antes del examen real.")

print("\n" + "=" * 70)
print("FIN DEL SIMULACRO")
print("=" * 70)
