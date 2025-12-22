#!/usr/bin/env python3
"""
Simulacro de Examen: CSCA 5642 - Deep Learning
===============================================

Módulo: M07 - Deep Learning
Tiempo Estimado: 90 minutos
Puntuación Total: 100 puntos

Estructura:
- Parte A: Preguntas Teóricas (30 puntos)
- Parte B: Ejercicios de Código (70 puntos)

Criterio para aprobar con B: >= 80 puntos

Ejecutar tests: pytest tests/test_simulacro_csca5642.py -v
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
# Pregunta A1 (8 puntos): Backpropagation
# -----------------------------------------------------------------------------
# Considera una red con una capa oculta: x → h → y
# h = σ(W₁x + b₁), y = W₂h + b₂
# Loss: L = ½(y - t)²
#
# a) Escribe la expresión para ∂L/∂W₂ (3 pts)
# b) Escribe la expresión para ∂L/∂W₁ usando la regla de la cadena (3 pts)
# c) ¿Por qué el gradiente puede "desvanecerse" en redes profundas con sigmoid? (2 pts)

respuesta_A1_a: str = ""  # TODO: ∂L/∂W₂
respuesta_A1_b: str = ""  # TODO: ∂L/∂W₁
respuesta_A1_c: str = ""  # TODO: Vanishing gradient


# -----------------------------------------------------------------------------
# Pregunta A2 (8 puntos): CNNs
# -----------------------------------------------------------------------------
# Una imagen de entrada tiene dimensiones 32×32×3.
# Se aplica una convolución con 16 filtros de 5×5, stride=1, padding=0.
#
# a) ¿Cuáles son las dimensiones de la salida? (3 pts)
# b) ¿Cuántos parámetros tiene esta capa (incluyendo bias)? (3 pts)
# c) ¿Por qué las CNNs son más eficientes que MLPs para imágenes? (2 pts)

respuesta_A2_a: str = ""  # TODO: Dimensiones de salida
respuesta_A2_b: int = 0  # TODO: Número de parámetros
respuesta_A2_c: str = ""  # TODO: Eficiencia de CNNs


# -----------------------------------------------------------------------------
# Pregunta A3 (7 puntos): Regularización
# -----------------------------------------------------------------------------
# a) Explica cómo funciona Dropout durante entrenamiento vs inferencia. (3 pts)
# b) ¿Por qué Batch Normalization actúa como regularizador? (2 pts)
# c) ¿Qué es Early Stopping y cómo previene overfitting? (2 pts)

respuesta_A3_a: str = ""  # TODO: Dropout train vs inference
respuesta_A3_b: str = ""  # TODO: BatchNorm como regularizador
respuesta_A3_c: str = ""  # TODO: Early Stopping


# -----------------------------------------------------------------------------
# Pregunta A4 (7 puntos): RNNs y LSTMs
# -----------------------------------------------------------------------------
# a) ¿Cuál es el problema principal de RNNs vanilla para secuencias largas? (2 pts)
# b) ¿Cómo resuelve LSTM este problema? Menciona las gates. (3 pts)
# c) ¿Cuándo usarías Bidirectional LSTM vs unidireccional? (2 pts)

respuesta_A4_a: str = ""  # TODO: Problema de RNN vanilla
respuesta_A4_b: str = ""  # TODO: Solución LSTM
respuesta_A4_c: str = ""  # TODO: Bidirectional vs unidirectional


# =============================================================================
# PARTE B: EJERCICIOS DE CÓDIGO (70 puntos)
# =============================================================================

print("\n" + "=" * 70)
print("PARTE B: EJERCICIOS DE CÓDIGO (70 puntos)")
print("=" * 70)


# -----------------------------------------------------------------------------
# Ejercicio B1 (20 puntos): Forward Pass de MLP
# -----------------------------------------------------------------------------
def relu(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Función de activación ReLU: max(0, z)."""
    # TODO: Implementar
    return np.maximum(0, z)


def relu_derivative(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Derivada de ReLU: 1 si z > 0, else 0."""
    # TODO: Implementar
    return (z > 0).astype(np.float64)


def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Función softmax para clasificación multiclase.

    softmax(z)_i = exp(z_i) / Σ exp(z_j)

    Tip: Restar max(z) para estabilidad numérica.
    """
    # TODO: Implementar con estabilidad numérica
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return np.asarray(exp_z / np.sum(exp_z, axis=-1, keepdims=True), dtype=np.float64)


def mlp_forward(
    X: NDArray[np.float64],
    weights: list[NDArray[np.float64]],
    biases: list[NDArray[np.float64]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """
    Forward pass de un MLP con ReLU en capas ocultas y softmax en salida.

    Parámetros
    ----------
    X : NDArray[np.float64]
        Datos de entrada (batch_size, input_dim).
    weights : list[NDArray]
        Lista de matrices de pesos [W1, W2, ...].
    biases : list[NDArray]
        Lista de vectores de bias [b1, b2, ...].

    Retorna
    -------
    tuple[list[NDArray], list[NDArray]]
        - activations: Lista de activaciones de cada capa [a0, a1, ..., output]
        - pre_activations: Lista de valores pre-activación [z1, z2, ...]
    """
    activations = [X]
    pre_activations = []

    current_input = X
    n_layers = len(weights)

    for i in range(n_layers):
        # TODO: Calcular z = W @ a + b
        z = current_input @ weights[i] + biases[i]
        pre_activations.append(z)

        # TODO: Aplicar activación (ReLU para ocultas, softmax para última)
        if i < n_layers - 1:
            a = relu(z)
        else:
            a = softmax(z)

        activations.append(a)
        current_input = a

    return activations, pre_activations


# Test B1
print("\n--- Test B1: MLP Forward Pass ---")
X_mlp = rng.standard_normal((5, 4))  # 5 samples, 4 features
W1 = rng.standard_normal((4, 8)) * 0.1
b1 = np.zeros(8)
W2 = rng.standard_normal((8, 3)) * 0.1
b2 = np.zeros(3)

activations, pre_activations = mlp_forward(X_mlp, [W1, W2], [b1, b2])
print(f"Input shape: {X_mlp.shape}")
print(f"Hidden shape: {activations[1].shape}")
print(f"Output shape: {activations[2].shape}")
print(f"Output sums to 1? {np.allclose(activations[2].sum(axis=1), 1)}")


# -----------------------------------------------------------------------------
# Ejercicio B2 (25 puntos): Backward Pass y Gradientes
# -----------------------------------------------------------------------------
def cross_entropy_loss(
    y_pred: NDArray[np.float64],
    y_true: NDArray[np.int64],
) -> float:
    """
    Calcula Cross-Entropy Loss.

    L = -1/n * Σ log(y_pred[i, y_true[i]])

    Parámetros
    ----------
    y_pred : NDArray[np.float64]
        Probabilidades predichas (batch_size, n_classes).
    y_true : NDArray[np.int64]
        Labels verdaderos como índices (batch_size,).

    Retorna
    -------
    float
        Cross-entropy loss promedio.
    """
    n = len(y_true)
    # TODO: Implementar (añadir epsilon para estabilidad)
    eps = 1e-15
    log_probs = np.log(np.clip(y_pred[np.arange(n), y_true], eps, 1 - eps))
    return float(-np.mean(log_probs))


def cross_entropy_gradient(
    y_pred: NDArray[np.float64],
    y_true: NDArray[np.int64],
) -> NDArray[np.float64]:
    """
    Gradiente de Cross-Entropy + Softmax combinado.

    Para softmax + CE, el gradiente simplificado es:
    ∂L/∂z = y_pred - y_onehot

    Parámetros
    ----------
    y_pred : NDArray[np.float64]
        Probabilidades predichas (batch_size, n_classes).
    y_true : NDArray[np.int64]
        Labels verdaderos (batch_size,).

    Retorna
    -------
    NDArray[np.float64]
        Gradiente respecto a z (batch_size, n_classes).
    """
    n = len(y_true)
    n_classes = y_pred.shape[1]

    # TODO: Crear one-hot encoding de y_true
    y_onehot = np.zeros((n, n_classes), dtype=np.float64)
    y_onehot[np.arange(n), y_true] = 1

    # TODO: Calcular gradiente
    grad = (y_pred - y_onehot) / n
    return grad


def mlp_backward(
    activations: list[NDArray[np.float64]],
    pre_activations: list[NDArray[np.float64]],
    weights: list[NDArray[np.float64]],
    y_true: NDArray[np.int64],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """
    Backward pass de un MLP.

    Parámetros
    ----------
    activations : list[NDArray]
        Activaciones de cada capa [a0, a1, ..., output].
    pre_activations : list[NDArray]
        Valores pre-activación [z1, z2, ...].
    weights : list[NDArray]
        Matrices de pesos.
    y_true : NDArray[np.int64]
        Labels verdaderos.

    Retorna
    -------
    tuple[list[NDArray], list[NDArray]]
        - grad_weights: Gradientes de pesos [∂L/∂W1, ∂L/∂W2, ...]
        - grad_biases: Gradientes de biases [∂L/∂b1, ∂L/∂b2, ...]
    """
    n_layers = len(weights)
    grad_weights: list[NDArray[np.float64]] = []
    grad_biases: list[NDArray[np.float64]] = []

    # TODO: Gradiente de la capa de salida (softmax + CE)
    delta = cross_entropy_gradient(activations[-1], y_true)

    # Backpropagate
    for i in range(n_layers - 1, -1, -1):
        # TODO: Gradiente de pesos: ∂L/∂W = a^T @ delta
        grad_w = activations[i].T @ delta
        grad_weights.insert(0, grad_w)

        # TODO: Gradiente de bias: ∂L/∂b = sum(delta, axis=0)
        grad_b = np.sum(delta, axis=0)
        grad_biases.insert(0, grad_b)

        if i > 0:
            # TODO: Propagar delta a capa anterior
            delta = (delta @ weights[i].T) * relu_derivative(pre_activations[i - 1])

    return grad_weights, grad_biases


# Test B2
print("\n--- Test B2: MLP Backward Pass ---")
y_true_mlp = np.array([0, 1, 2, 0, 1])  # 5 samples, 3 classes
loss = cross_entropy_loss(activations[2], y_true_mlp)
print(f"Cross-Entropy Loss: {loss:.4f}")

grad_w, grad_b = mlp_backward(activations, pre_activations, [W1, W2], y_true_mlp)
print(f"Gradiente W1 shape: {grad_w[0].shape}")
print(f"Gradiente W2 shape: {grad_w[1].shape}")


# -----------------------------------------------------------------------------
# Ejercicio B3 (25 puntos): Implementar una Capa Convolucional
# -----------------------------------------------------------------------------
def conv2d_forward(
    X: NDArray[np.float64],
    W: NDArray[np.float64],
    b: NDArray[np.float64],
    stride: int = 1,
    padding: int = 0,
) -> NDArray[np.float64]:
    """
    Forward pass de convolución 2D.

    Parámetros
    ----------
    X : NDArray[np.float64]
        Input (batch_size, height, width, in_channels).
    W : NDArray[np.float64]
        Filtros (filter_h, filter_w, in_channels, out_channels).
    b : NDArray[np.float64]
        Bias (out_channels,).
    stride : int
        Stride de la convolución.
    padding : int
        Zero-padding.

    Retorna
    -------
    NDArray[np.float64]
        Output (batch_size, out_height, out_width, out_channels).

    Fórmula de dimensiones:
    -----------------------
    out_height = (height + 2*padding - filter_h) // stride + 1
    out_width = (width + 2*padding - filter_w) // stride + 1
    """
    batch_size, height, width, in_channels = X.shape
    filter_h, filter_w, _, out_channels = W.shape

    # Calcular dimensiones de salida
    out_height = (height + 2 * padding - filter_h) // stride + 1
    out_width = (width + 2 * padding - filter_w) // stride + 1

    # TODO: Aplicar padding si es necesario
    if padding > 0:
        X_padded = np.pad(
            X,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode="constant",
        )
    else:
        X_padded = X

    # TODO: Inicializar output
    output = np.zeros((batch_size, out_height, out_width, out_channels))

    # TODO: Realizar convolución
    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride
            h_end = h_start + filter_h
            w_start = j * stride
            w_end = w_start + filter_w

            # Extraer región
            region = X_padded[:, h_start:h_end, w_start:w_end, :]

            # Convolución: sum over (filter_h, filter_w, in_channels)
            for k in range(out_channels):
                output[:, i, j, k] = (
                    np.sum(region * W[:, :, :, k], axis=(1, 2, 3)) + b[k]
                )

    return output


def max_pool2d(
    X: NDArray[np.float64],
    pool_size: int = 2,
    stride: int = 2,
) -> NDArray[np.float64]:
    """
    Max Pooling 2D.

    Parámetros
    ----------
    X : NDArray[np.float64]
        Input (batch_size, height, width, channels).
    pool_size : int
        Tamaño de la ventana de pooling.
    stride : int
        Stride del pooling.

    Retorna
    -------
    NDArray[np.float64]
        Output con dimensiones reducidas.
    """
    batch_size, height, width, channels = X.shape

    out_height = (height - pool_size) // stride + 1
    out_width = (width - pool_size) // stride + 1

    output = np.zeros((batch_size, out_height, out_width, channels))

    # TODO: Aplicar max pooling
    for i in range(out_height):
        for j in range(out_width):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size

            region = X[:, h_start:h_end, w_start:w_end, :]
            output[:, i, j, :] = np.max(region, axis=(1, 2))

    return output


# Test B3
print("\n--- Test B3: Conv2D Forward ---")
X_conv = rng.standard_normal((2, 8, 8, 3))  # 2 images, 8x8, 3 channels
W_conv = rng.standard_normal((3, 3, 3, 16)) * 0.1  # 16 filters, 3x3
b_conv = np.zeros(16)

out_conv = conv2d_forward(X_conv, W_conv, b_conv, stride=1, padding=0)
print(f"Input shape: {X_conv.shape}")
print(f"Output shape: {out_conv.shape}")
print("Expected output shape: (2, 6, 6, 16)")

out_pool = max_pool2d(out_conv, pool_size=2, stride=2)
print(f"After MaxPool: {out_pool.shape}")


# =============================================================================
# VALIDACIÓN FINAL
# =============================================================================

print("\n" + "=" * 70)
print("VALIDACIÓN FINAL")
print("=" * 70)


def validar_simulacro() -> dict[str, bool]:
    """Valida todas las respuestas del simulacro."""
    resultados: dict[str, bool] = {}

    # Validar A2 (CNN parámetros)
    # (5*5*3 + 1) * 16 = 1216
    resultados["A2_params"] = respuesta_A2_b == 1216

    # Validar B1 (Forward pass)
    X_test = rng.standard_normal((3, 4))
    W1_test = rng.standard_normal((4, 5)) * 0.1
    b1_test = np.zeros(5)
    W2_test = rng.standard_normal((5, 2)) * 0.1
    b2_test = np.zeros(2)

    acts, _ = mlp_forward(X_test, [W1_test, W2_test], [b1_test, b2_test])
    resultados["B1_softmax_sum"] = np.allclose(acts[-1].sum(axis=1), 1)
    resultados["B1_shapes"] = acts[1].shape == (3, 5) and acts[2].shape == (3, 2)

    # Validar B2 (Cross-entropy)
    y_pred_test = np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]])
    y_true_test = np.array([0, 1, 0])
    loss_test = cross_entropy_loss(y_pred_test, y_true_test)
    resultados["B2_ce_loss"] = 0 < loss_test < 1  # Reasonable range

    # Validar B3 (Conv2D)
    X_c = rng.standard_normal((1, 6, 6, 1))
    W_c = rng.standard_normal((3, 3, 1, 4)) * 0.1
    b_c = np.zeros(4)
    out_c = conv2d_forward(X_c, W_c, b_c, stride=1, padding=0)
    resultados["B3_conv_shape"] = out_c.shape == (1, 4, 4, 4)

    out_p = max_pool2d(out_c, pool_size=2, stride=2)
    resultados["B3_pool_shape"] = out_p.shape == (1, 2, 2, 4)

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
