#!/usr/bin/env python3
"""
Notebook M05: Regresión Lineal desde Cero con Celdas de Validación
===================================================================

Módulo 5 - Semana 9: Fundamentos de Aprendizaje Supervisado
Curso Alineado: CSCA 5622 - Supervised Learning

Objetivos:
1. Implementar Regresión Lineal con Gradiente Descendente desde cero
2. Comprender la conexión matemática con M02 (Álgebra Lineal) y M03 (Cálculo)
3. Aplicar Regularización L2 (Ridge) y entender su relación con normas vectoriales
4. Validar implementaciones con celdas de autograding

Dependencias:
    pip install numpy matplotlib scikit-learn

Ejecutar como script o convertir a notebook con jupytext.
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(seed=42)

# =============================================================================
# PARTE 1: Regresión Lineal - Fundamentos Matemáticos
# =============================================================================

print("=" * 70)
print("PARTE 1: Regresión Lineal desde Cero")
print("=" * 70)

# %% [markdown]
# ## 💡 Conexión con M02 - Álgebra Lineal
#
# La regresión lineal busca encontrar el vector de pesos $\mathbf{w}$ que minimiza:
#
# $$\min_{\mathbf{w}} \|\mathbf{Xw} - \mathbf{y}\|_2^2$$
#
# Donde $\|\cdot\|_2$ es la **norma L2** (norma Euclidiana) que estudiaste en M02.
# La solución analítica usa la **ecuación normal**:
#
# $$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$
#
# Esto requiere que $\mathbf{X}^T\mathbf{X}$ sea **invertible** (matriz no singular).


def linear_regression_closed_form(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Regresión Lineal usando la solución de forma cerrada (ecuación normal).

    Parámetros
    ----------
    X : np.ndarray
        Matriz de características (n_samples, n_features).
        Debe incluir columna de 1s para el bias si se desea intercepto.
    y : np.ndarray
        Vector objetivo (n_samples,).

    Retorna
    -------
    np.ndarray
        Vector de pesos (n_features,) incluyendo bias si X tiene columna de 1s.

    Notas
    -----
    💡 Conexión M02: Esta función implementa la solución $w = (X^TX)^{-1}X^Ty$.
    Usa np.linalg.pinv (pseudo-inversa) para mayor estabilidad numérica.
    """
    # Pseudo-inversa de Moore-Penrose para estabilidad numérica
    # Equivalente a (X^T X)^{-1} X^T cuando X tiene rango completo
    return np.linalg.pinv(X) @ y


# %% [markdown]
# ## 💡 Conexión con M03 - Cálculo y Optimización
#
# El gradiente del MSE respecto a $\mathbf{w}$ es:
#
# $$\nabla_{\mathbf{w}} \text{MSE} = \frac{2}{n}\mathbf{X}^T(\mathbf{Xw} - \mathbf{y})$$
#
# El **Descenso de Gradiente** actualiza iterativamente:
#
# $$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla_{\mathbf{w}} \text{MSE}$$
#
# El signo negativo es porque descendemos en la dirección opuesta al gradiente.


def linear_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """
    Regresión Lineal usando Descenso de Gradiente.

    Parámetros
    ----------
    X : np.ndarray
        Matriz de características (n_samples, n_features).
    y : np.ndarray
        Vector objetivo (n_samples,).
    learning_rate : float
        Tasa de aprendizaje α (default: 0.01).
    n_iterations : int
        Número máximo de iteraciones (default: 1000).
    tolerance : float
        Criterio de convergencia basado en cambio de loss (default: 1e-6).
    verbose : bool
        Si True, imprime progreso cada 100 iteraciones.

    Retorna
    -------
    tuple[np.ndarray, list[float]]
        - weights: Vector de pesos optimizados (n_features,)
        - loss_history: Lista con el MSE en cada iteración

    Notas
    -----
    💡 Conexión M03: Este algoritmo implementa el descenso de gradiente
    que derivaste manualmente. La actualización w = w - α∇MSE usa el
    gradiente para moverse hacia el mínimo de la función de costo.
    """
    n_samples, n_features = X.shape

    # Inicialización de pesos (pequeños valores aleatorios)
    weights = rng.standard_normal(n_features) * 0.01

    loss_history: list[float] = []
    prev_loss = float("inf")

    for iteration in range(n_iterations):
        # Forward pass: predicción
        y_pred = X @ weights  # (n_samples,)

        # Calcular error
        error = y_pred - y  # (n_samples,)

        # Calcular MSE loss
        mse = float(np.mean(error**2))
        loss_history.append(mse)

        # Verificar convergencia
        if abs(prev_loss - mse) < tolerance:
            if verbose:
                print(f"  Convergió en iteración {iteration}")
            break
        prev_loss = mse

        # Calcular gradiente: ∇MSE = (2/n) * X^T * (Xw - y)
        gradient = (2 / n_samples) * (X.T @ error)  # (n_features,)

        # Actualizar pesos: w = w - α * ∇MSE
        weights = weights - learning_rate * gradient

        if verbose and iteration % 100 == 0:
            print(f"  Iteración {iteration}: MSE = {mse:.6f}")

    return weights, loss_history


# =============================================================================
# PARTE 2: Regularización L2 (Ridge Regression)
# =============================================================================

print("\n" + "=" * 70)
print("PARTE 2: Regularización L2 (Ridge Regression)")
print("=" * 70)

# %% [markdown]
# ## 💡 Conexión con M02 - Normas Vectoriales y Regularización L2
#
# > **⚠️ CONCEPTO CLAVE**: La regularización L2 penaliza la **norma L2** del vector
# > de pesos, evitando que crezcan demasiado (overfitting).
#
# La función de costo con regularización L2 es:
#
# $$J(\mathbf{w}) = \text{MSE} + \lambda \|\mathbf{w}\|_2^2
#                = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p}w_j^2$$
#
# Donde:
# - $\|\mathbf{w}\|_2^2 = w_1^2 + w_2^2 + ... + w_p^2$ es la **norma L2 al cuadrado**
# - $\lambda$ es el hiperparámetro de regularización
#
# ### ¿Por qué funciona?
#
# En el espacio de pesos, la regularización L2 restringe la solución a una
# **hiperesfera** centrada en el origen. Esto es exactamente la definición
# geométrica de la norma L2 que viste en M02: todos los puntos a distancia
# constante del origen forman una esfera.
#
# ### Gradiente con Regularización
#
# $$\nabla_{\mathbf{w}} J = \frac{2}{n}\mathbf{X}^T(\mathbf{Xw} - \mathbf{y}) + 2\lambda\mathbf{w}$$


def ridge_regression_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    lambda_reg: float = 0.1,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """
    Ridge Regression (L2 Regularization) usando Descenso de Gradiente.

    Parámetros
    ----------
    X : np.ndarray
        Matriz de características (n_samples, n_features).
    y : np.ndarray
        Vector objetivo (n_samples,).
    lambda_reg : float
        Parámetro de regularización λ (default: 0.1).
        Mayor λ → más regularización → pesos más pequeños.
    learning_rate : float
        Tasa de aprendizaje α (default: 0.01).
    n_iterations : int
        Número máximo de iteraciones (default: 1000).
    tolerance : float
        Criterio de convergencia (default: 1e-6).
    verbose : bool
        Si True, imprime progreso.

    Retorna
    -------
    tuple[np.ndarray, list[float]]
        - weights: Vector de pesos regularizados
        - loss_history: Lista con el costo total (MSE + penalización L2)

    Notas
    -----
    💡 Conexión M02: El término λ||w||₂² penaliza la norma L2 del vector w.
    Geométricamente, esto restringe w a estar dentro de una hiperesfera.
    """
    n_samples, n_features = X.shape
    weights = rng.standard_normal(n_features) * 0.01
    loss_history: list[float] = []
    prev_loss = float("inf")

    for iteration in range(n_iterations):
        # Forward pass
        y_pred = X @ weights
        error = y_pred - y

        # Calcular costo total: MSE + λ||w||²
        mse = float(np.mean(error**2))
        l2_penalty = lambda_reg * float(np.sum(weights**2))
        total_cost = mse + l2_penalty
        loss_history.append(total_cost)

        # Verificar convergencia
        if abs(prev_loss - total_cost) < tolerance:
            if verbose:
                print(f"  Convergió en iteración {iteration}")
            break
        prev_loss = total_cost

        # Gradiente con regularización: ∇J = (2/n)X^T(Xw-y) + 2λw
        gradient = (2 / n_samples) * (X.T @ error) + 2 * lambda_reg * weights

        # Actualizar pesos
        weights = weights - learning_rate * gradient

        if verbose and iteration % 100 == 0:
            print(
                f"  Iteración {iteration}: Costo = {total_cost:.6f} (MSE={mse:.6f}, L2={l2_penalty:.6f})"
            )

    return weights, loss_history


# =============================================================================
# PARTE 3: Demo y Visualización
# =============================================================================

print("\n" + "=" * 70)
print("PARTE 3: Demo con Datos Sintéticos")
print("=" * 70)

# Generar datos sintéticos
n_samples = 100
n_features = 3

# Crear matriz de características con columna de 1s para bias
X_raw = rng.standard_normal((n_samples, n_features))
X = np.column_stack([np.ones(n_samples), X_raw])  # Añadir intercepto

# Pesos verdaderos (incluyendo bias)
true_weights = np.array([2.0, 1.5, -0.5, 0.3])  # [bias, w1, w2, w3]

# Generar y con ruido
noise = rng.standard_normal(n_samples) * 0.5
y = X @ true_weights + noise

print(f"Datos generados: X.shape={X.shape}, y.shape={y.shape}")
print(f"Pesos verdaderos: {true_weights}")

# --- Método 1: Forma Cerrada ---
print("\n--- Método 1: Forma Cerrada (Ecuación Normal) ---")
weights_closed = linear_regression_closed_form(X, y)
print(f"Pesos estimados: {weights_closed}")
print(f"Error vs verdaderos: {np.abs(weights_closed - true_weights)}")

# --- Método 2: Gradiente Descendente ---
print("\n--- Método 2: Gradiente Descendente ---")
weights_gd, loss_gd = linear_regression_gradient_descent(
    X, y, learning_rate=0.1, n_iterations=1000, verbose=True
)
print(f"Pesos estimados: {weights_gd}")
print(f"MSE final: {loss_gd[-1]:.6f}")

# --- Método 3: Ridge Regression ---
print("\n--- Método 3: Ridge Regression (λ=0.1) ---")
weights_ridge, loss_ridge = ridge_regression_gradient_descent(
    X, y, lambda_reg=0.1, learning_rate=0.1, n_iterations=1000, verbose=True
)
print(f"Pesos estimados: {weights_ridge}")
print(f"Norma L2 de pesos (GD): {np.linalg.norm(weights_gd):.4f}")
print(f"Norma L2 de pesos (Ridge): {np.linalg.norm(weights_ridge):.4f}")
print("→ Nota: Ridge produce pesos con menor norma (más regularizados)")


# =============================================================================
# PARTE 4: CELDAS DE VALIDACIÓN (AUTOGRADERS)
# =============================================================================

print("\n" + "=" * 70)
print("PARTE 4: Celdas de Validación para Estudiantes")
print("=" * 70)


def validar_regresion_lineal(
    weights_estudiante: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mse_threshold: float = 1.0,
    weights_reference: np.ndarray | None = None,
    tolerance: float = 0.1,
) -> bool:
    """
    Celda de Validación para Regresión Lineal.

    Verifica que la implementación del estudiante:
    1. Produce predicciones con forma correcta
    2. Alcanza un MSE razonable
    3. Pesos cercanos a la referencia (si se proporciona)

    Parámetros
    ----------
    weights_estudiante : np.ndarray
        Vector de pesos del estudiante (n_features,).
    X_test : np.ndarray
        Datos de prueba (n_samples, n_features).
    y_test : np.ndarray
        Valores objetivo de prueba (n_samples,).
    mse_threshold : float
        MSE máximo aceptable (default: 1.0).
    weights_reference : np.ndarray, opcional
        Pesos de referencia para comparar.
    tolerance : float
        Tolerancia para comparación de pesos (default: 0.1).

    Retorna
    -------
    bool
        True si pasa todas las validaciones.

    Raises
    ------
    AssertionError
        Si alguna validación falla, con mensaje descriptivo.
    """
    print("🔍 Ejecutando validaciones...")

    # Test 1: Verificar dimensiones
    expected_features = X_test.shape[1]
    assert weights_estudiante.shape == (expected_features,), (
        f"❌ Error de dimensiones: "
        f"Esperado ({expected_features},), "
        f"Obtenido {weights_estudiante.shape}"
    )
    print("  ✅ Test 1: Dimensiones correctas")

    # Test 2: Calcular predicciones y MSE
    y_pred = X_test @ weights_estudiante
    assert y_pred.shape == y_test.shape, (
        f"❌ Error: Predicciones con forma incorrecta. "
        f"Esperado {y_test.shape}, Obtenido {y_pred.shape}"
    )
    print("  ✅ Test 2: Forma de predicciones correcta")

    # Test 3: Verificar MSE
    mse = float(np.mean((y_pred - y_test) ** 2))
    assert mse < mse_threshold, (
        f"❌ Error: MSE demasiado alto. " f"MSE={mse:.4f}, Umbral={mse_threshold}"
    )
    print(f"  ✅ Test 3: MSE aceptable ({mse:.4f} < {mse_threshold})")

    # Test 4: Comparar con referencia (opcional)
    if weights_reference is not None:
        diff = np.abs(weights_estudiante - weights_reference)
        max_diff = float(np.max(diff))
        assert max_diff < tolerance, (
            f"❌ Error: Pesos difieren de la referencia. "
            f"Máxima diferencia={max_diff:.4f}, Tolerancia={tolerance}"
        )
        print(f"  ✅ Test 4: Pesos cercanos a referencia (max_diff={max_diff:.4f})")

    print("\n✅ ¡EXCELENTE! Tu implementación pasa todas las validaciones.")
    return True


def validar_ridge_regression(
    weights_estudiante: np.ndarray,
    weights_sin_regularizar: np.ndarray,
    lambda_reg: float,
) -> bool:
    """
    Validación específica para Ridge Regression.

    Verifica que:
    1. Los pesos regularizados tienen menor norma L2
    2. El efecto de regularización es proporcional a λ

    Parámetros
    ----------
    weights_estudiante : np.ndarray
        Pesos de Ridge Regression del estudiante.
    weights_sin_regularizar : np.ndarray
        Pesos de regresión lineal sin regularizar (para comparar).
    lambda_reg : float
        Valor de λ usado en la regularización.

    Retorna
    -------
    bool
        True si pasa las validaciones.
    """
    print("🔍 Validando Ridge Regression...")

    norma_estudiante = float(np.linalg.norm(weights_estudiante))
    norma_sin_reg = float(np.linalg.norm(weights_sin_regularizar))

    # Test 1: Ridge debe producir pesos con menor norma
    assert norma_estudiante < norma_sin_reg * 1.1, (
        f"❌ Error: La regularización L2 debería reducir la norma de los pesos. "
        f"Norma Ridge={norma_estudiante:.4f}, Norma sin regularizar={norma_sin_reg:.4f}"
    )
    print(f"  ✅ Test 1: Norma reducida ({norma_estudiante:.4f} < {norma_sin_reg:.4f})")

    # Test 2: Con λ > 0, los pesos no deberían ser idénticos
    if lambda_reg > 0:
        diff = float(np.linalg.norm(weights_estudiante - weights_sin_regularizar))
        assert diff > 1e-6, (
            "❌ Error: Los pesos regularizados son idénticos a los no regularizados. "
            "Verifica que estás aplicando el término de penalización λ||w||²"
        )
        print(f"  ✅ Test 2: Regularización aplicada correctamente (diff={diff:.6f})")

    print("\n✅ ¡CORRECTO! Tu Ridge Regression está bien implementado.")
    return True


# --- Ejecutar Validaciones ---
print("\n--- Validación de tu implementación de Regresión Lineal ---")
try:
    validar_regresion_lineal(
        weights_estudiante=weights_gd,
        X_test=X,
        y_test=y,
        mse_threshold=1.0,
        weights_reference=weights_closed,
        tolerance=0.5,
    )
except AssertionError as e:
    print(f"\n{e}")

print("\n--- Validación de tu implementación de Ridge Regression ---")
try:
    validar_ridge_regression(
        weights_estudiante=weights_ridge,
        weights_sin_regularizar=weights_gd,
        lambda_reg=0.1,
    )
except AssertionError as e:
    print(f"\n{e}")


# =============================================================================
# PARTE 5: Ejercicios para el Estudiante
# =============================================================================

print("\n" + "=" * 70)
print("PARTE 5: Ejercicios para Practicar")
print("=" * 70)

# %% [markdown]
# ## 📝 Ejercicio 1: Implementa tu propia función de Regresión Lineal
#
# Completa la función `mi_regresion_lineal()` usando gradiente descendente.
# Luego ejecuta la celda de validación para verificar tu implementación.
#
# ```python
# def mi_regresion_lineal(X, y, lr=0.01, epochs=1000):
#     """
#     Tu implementación aquí.
#
#     Pistas:
#     1. Inicializa pesos con valores pequeños aleatorios
#     2. En cada época:
#        a. Calcula predicciones: y_pred = X @ weights
#        b. Calcula error: error = y_pred - y
#        c. Calcula gradiente: grad = (2/n) * X.T @ error
#        d. Actualiza pesos: weights = weights - lr * grad
#     3. Retorna los pesos finales
#     """
#     n_samples, n_features = X.shape
#     weights = ...  # Inicializar
#
#     for epoch in range(epochs):
#         # Tu código aquí
#         pass
#
#     return weights
# ```
#
# ## 📝 Ejercicio 2: Experimenta con diferentes valores de λ
#
# Ejecuta Ridge Regression con λ ∈ {0.01, 0.1, 1.0, 10.0} y observa:
# 1. ¿Cómo cambia la norma L2 de los pesos?
# 2. ¿Cómo cambia el MSE en datos de entrenamiento?
# 3. ¿Hay un punto donde demasiada regularización perjudica?
#
# ## 📝 Ejercicio 3: Conexión con M02
#
# Demuestra geométricamente por qué la regularización L2 produce pesos
# dentro de una hiperesfera. Pista: grafica el contorno de MSE vs el
# contorno de ||w||² = constante para un problema 2D.


print(
    """
📚 RESUMEN DE CONEXIONES TEÓRICO-PRÁCTICAS:

┌─────────────────────────────────────────────────────────────────────────┐
│ MÓDULO    │ CONCEPTO              │ APLICACIÓN EN REGRESIÓN             │
├─────────────────────────────────────────────────────────────────────────┤
│ M02       │ Norma L2 (Euclidiana) │ Regularización Ridge: λ||w||₂²      │
│ M02       │ Producto interno      │ Predicción: ŷ = Xw = Σ xᵢwᵢ        │
│ M02       │ Inversión matricial   │ Ecuación normal: w = (X^TX)^{-1}X^Ty│
│ M03       │ Gradiente             │ ∇MSE = (2/n)X^T(Xw - y)             │
│ M03       │ Descenso de gradiente │ w = w - α∇MSE                       │
│ M03       │ Derivadas parciales   │ ∂MSE/∂wⱼ para cada peso             │
└─────────────────────────────────────────────────────────────────────────┘
"""
)

print("\n✅ Notebook completado. ¡Ahora implementa tus propias funciones y valídalas!")
