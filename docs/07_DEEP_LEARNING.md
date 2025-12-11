# Módulo 07 - Deep Learning

> **🎯 Objetivo:** Implementar MLP con backprop + entender fundamentos de CNNs
> **Fase:** 2 - Núcleo de ML | **Semanas 17-20**
> **Curso del Pathway:** Introduction to Deep Learning

---

## 🧠 ¿Por Qué Deep Learning?

```
DEEP LEARNING = Redes Neuronales Multicapa + Arquitecturas Especializadas

Ventajas sobre ML clásico:
├── Aprende features automáticamente (no feature engineering manual)
├── Puede modelar relaciones NO LINEALES complejas
├── Escala con más datos y más compute
└── Estado del arte en visión (CNNs), NLP (Transformers), etc.

Desventajas:
├── Requiere más datos
├── "Caja negra" - menos interpretable
└── Costoso computacionalmente
```

---

## 📚 Contenido del Módulo

| Semana | Tema | Entregable |
|--------|------|------------|
| 17 | Perceptrón y MLP | `activations.py` + forward pass |
| 18 | Backpropagation | `backward()` con Chain Rule |
| 19 | **CNNs: Teoría** | Entender convolución, pooling, stride |
| 20 | Optimizadores y Entrenamiento | `neural_network.py` completo |

---

## 💻 Parte 1: Perceptrón y Activaciones

### 1.1 La Neurona Artificial

```python
import numpy as np

"""
NEURONA ARTIFICIAL (Perceptrón)

Inspiración biológica:
- Recibe señales de entrada (dendrites)
- Procesa y decide si "dispara" (soma)
- Envía señal de salida (axon)

Modelo matemático:
    z = Σ wᵢxᵢ + b = w·x + b  (combinación lineal)
    a = σ(z)                    (activación)

Donde:
- x: vector de entradas
- w: vector de pesos (learnable)
- b: bias (learnable)
- σ: función de activación (introduce no-linealidad)
"""

def perceptron(x: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Un perceptrón simple.

    Args:
        x: entrada (n_features,)
        w: pesos (n_features,)
        b: bias

    Returns:
        salida activada
    """
    z = np.dot(w, x) + b
    return 1 if z > 0 else 0  # Función escalón
```

### 1.2 Funciones de Activación

```python
import numpy as np

class Activations:
    """Funciones de activación y sus derivadas."""

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid: σ(z) = 1 / (1 + e^(-z))

        Rango: (0, 1)
        Uso: Capa de salida para clasificación binaria
        Problema: Vanishing gradient para |z| grande
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """σ'(z) = σ(z) · (1 - σ(z)) = a · (1 - a)"""
        return a * (1 - a)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """
        ReLU: f(z) = max(0, z)

        Rango: [0, ∞)
        Uso: Capas ocultas (default moderno)
        Ventaja: No vanishing gradient para z > 0
        Problema: "Dying ReLU" si z < 0 siempre
        """
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """ReLU'(z) = 1 si z > 0, 0 si z ≤ 0"""
        return (z > 0).astype(float)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """
        Tanh: f(z) = (e^z - e^(-z)) / (e^z + e^(-z))

        Rango: (-1, 1)
        Uso: Alternativa a sigmoid (centrado en 0)
        """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(a: np.ndarray) -> np.ndarray:
        """tanh'(z) = 1 - tanh²(z) = 1 - a²"""
        return 1 - a ** 2

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax: softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)

        Rango: (0, 1), suma = 1
        Uso: Capa de salida para clasificación multiclase
        Output: probabilidades de cada clase
        """
        # Restar máximo para estabilidad numérica
        z_shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


# Demo
z = np.array([-2, -1, 0, 1, 2])
act = Activations()

print("z:", z)
print("sigmoid:", act.sigmoid(z))
print("relu:", act.relu(z))
print("tanh:", act.tanh(z))
print("softmax:", act.softmax(z))
```

### 1.3 El Problema XOR

```python
"""
XOR: La limitación del Perceptrón Simple

XOR truth table:
    x1  x2  |  y
    0   0   |  0
    0   1   |  1
    1   0   |  1
    1   1   |  0

Un perceptrón simple NO puede resolver XOR porque:
- XOR no es linealmente separable
- No existe una línea que separe las clases

Solución: Red multicapa (MLP)
- Una capa oculta puede aprender features intermedias
- Combinación de features no lineales resuelve XOR
"""

# Datos XOR
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Un perceptrón simple no puede aprender esto
# Necesitamos una red con al menos una capa oculta
```

---

## 💻 Parte 2: Forward Propagation

### 2.1 Arquitectura MLP

```python
"""
MLP - Multilayer Perceptron

Arquitectura típica:
    Input Layer → Hidden Layer(s) → Output Layer

Ejemplo para clasificación binaria:
    x (n_features) → h (n_hidden) → y (1)

Forward Pass:
    z₁ = W₁x + b₁        (capa 1: lineal)
    a₁ = σ(z₁)           (capa 1: activación)
    z₂ = W₂a₁ + b₂       (capa 2: lineal)
    a₂ = σ(z₂)           (capa 2: activación = output)

Dimensiones:
    x: (n_features,)
    W₁: (n_hidden, n_features)
    b₁: (n_hidden,)
    z₁, a₁: (n_hidden,)
    W₂: (n_output, n_hidden)
    b₂: (n_output,)
    z₂, a₂: (n_output,)
"""
```

### 2.2 Implementación Forward Pass

```python
import numpy as np
from typing import List, Dict

class Layer:
    """Una capa de la red neuronal."""

    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        """
        Args:
            input_size: número de entradas
            output_size: número de neuronas
            activation: 'relu', 'sigmoid', 'tanh', 'softmax', 'linear'
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Inicialización Xavier/He
        if activation == 'relu':
            # He initialization para ReLU
            std = np.sqrt(2.0 / input_size)
        else:
            # Xavier initialization
            std = np.sqrt(1.0 / input_size)

        self.W = np.random.randn(output_size, input_size) * std
        self.b = np.zeros(output_size)

        # Cache para backprop
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass de una capa.

        z = Wx + b
        a = activation(z)
        """
        self.cache['x'] = x

        # Transformación lineal
        z = self.W @ x + self.b
        self.cache['z'] = z

        # Activación
        if self.activation == 'relu':
            a = np.maximum(0, z)
        elif self.activation == 'sigmoid':
            a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            a = np.tanh(z)
        elif self.activation == 'softmax':
            z_shifted = z - np.max(z)
            exp_z = np.exp(z_shifted)
            a = exp_z / np.sum(exp_z)
        else:  # linear
            a = z

        self.cache['a'] = a
        return a


class NeuralNetwork:
    """Red Neuronal Multicapa."""

    def __init__(self, layer_sizes: List[int], activations: List[str]):
        """
        Args:
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            activations: ['relu', 'relu', ..., 'sigmoid'] para cada capa
        """
        assert len(activations) == len(layer_sizes) - 1

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass a través de todas las capas."""
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicción para múltiples muestras."""
        predictions = []
        for x in X:
            output = self.forward(x)
            if len(output) == 1:
                predictions.append(1 if output[0] > 0.5 else 0)
            else:
                predictions.append(np.argmax(output))
        return np.array(predictions)


# Demo
net = NeuralNetwork(
    layer_sizes=[2, 4, 1],  # 2 inputs → 4 hidden → 1 output
    activations=['relu', 'sigmoid']
)

# Forward pass
x = np.array([0.5, 0.3])
output = net.forward(x)
print(f"Input: {x}")
print(f"Output: {output}")
```

---

## 💻 Parte 3: Backpropagation

### 3.1 Funciones de Pérdida

```python
import numpy as np

def binary_cross_entropy(y_true: float, y_pred: float, eps: float = 1e-15) -> float:
    """
    Binary Cross-Entropy Loss.

    L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

    Args:
        y_true: etiqueta real (0 o 1)
        y_pred: predicción (probabilidad)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_derivative(y_true: float, y_pred: float, eps: float = 1e-15) -> float:
    """
    Derivada de BCE respecto a y_pred.

    ∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -y_true / y_pred + (1 - y_true) / (1 - y_pred)

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Categorical Cross-Entropy para multiclase.

    L = -Σᵢ yᵢ·log(ŷᵢ)

    Args:
        y_true: one-hot encoded (k,)
        y_pred: probabilidades softmax (k,)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred))

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)
```

### 3.2 Backpropagation: La Chain Rule en Acción

```python
"""
BACKPROPAGATION

Objetivo: Calcular ∂L/∂W y ∂L/∂b para cada capa.

Usando Chain Rule:
    ∂L/∂W₂ = ∂L/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂W₂
    ∂L/∂W₁ = ∂L/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁

Patrón:
1. Calcular ∂L/∂a_output (derivada de la loss)
2. Para cada capa, de atrás hacia adelante:
   a. δ = ∂L/∂z = ∂L/∂a · ∂a/∂z (error de la capa)
   b. ∂L/∂W = δ · x^T
   c. ∂L/∂b = δ
   d. Propagar: ∂L/∂a_prev = W^T · δ
"""

def backward_layer(layer, dL_da: np.ndarray) -> tuple:
    """
    Backward pass de una capa.

    Args:
        layer: capa con cache del forward pass
        dL_da: gradiente de la loss respecto a la activación

    Returns:
        dL_dx: gradiente respecto a la entrada
        dL_dW: gradiente respecto a los pesos
        dL_db: gradiente respecto al bias
    """
    z = layer.cache['z']
    x = layer.cache['x']
    a = layer.cache['a']

    # Derivada de la activación: ∂a/∂z
    if layer.activation == 'sigmoid':
        da_dz = a * (1 - a)
    elif layer.activation == 'relu':
        da_dz = (z > 0).astype(float)
    elif layer.activation == 'tanh':
        da_dz = 1 - a ** 2
    elif layer.activation == 'softmax':
        # Para softmax + cross-entropy, usamos el gradiente simplificado
        da_dz = np.ones_like(z)  # se maneja especialmente
    else:  # linear
        da_dz = np.ones_like(z)

    # δ = ∂L/∂z = ∂L/∂a · ∂a/∂z
    delta = dL_da * da_dz

    # Gradientes
    dL_dW = np.outer(delta, x)
    dL_db = delta
    dL_dx = layer.W.T @ delta

    return dL_dx, dL_dW, dL_db
```

### 3.3 Red Neuronal Completa con Backprop

```python
import numpy as np
from typing import List, Tuple

class NeuralNetworkFull:
    """Red Neuronal con Backpropagation completo."""

    def __init__(self, layer_sizes: List[int], activations: List[str]):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)

        self.loss_history = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward(self, y_true: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Backward pass: calcula gradientes para todas las capas.

        Returns:
            Lista de (dW, db) para cada capa
        """
        gradients = []

        # Obtener predicción (última activación)
        y_pred = self.layers[-1].cache['a']

        # Gradiente inicial: ∂L/∂a_output
        # Para sigmoid + BCE: simplificado a (y_pred - y_true)
        # Para softmax + CCE: también (y_pred - y_true)
        if self.layers[-1].activation in ['sigmoid', 'softmax']:
            dL_da = y_pred - y_true
        else:
            # MSE: 2(y_pred - y_true)
            dL_da = 2 * (y_pred - y_true)

        # Propagar hacia atrás
        for layer in reversed(self.layers):
            dL_dx, dL_dW, dL_db = backward_layer(layer, dL_da)
            gradients.insert(0, (dL_dW, dL_db))
            dL_da = dL_dx

        return gradients

    def update_weights(self, gradients: List[Tuple], learning_rate: float):
        """Actualiza pesos usando gradient descent."""
        for layer, (dW, db) in zip(self.layers, gradients):
            layer.W -= learning_rate * dW
            layer.b -= learning_rate * db

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.1,
        verbose: bool = True
    ):
        """Entrena la red."""
        for epoch in range(epochs):
            total_loss = 0

            for xi, yi in zip(X, y):
                # Forward
                output = self.forward(xi)

                # Loss
                if isinstance(yi, (int, float)):
                    yi_arr = np.array([yi])
                else:
                    yi_arr = yi
                loss = binary_cross_entropy(yi_arr[0], output[0])
                total_loss += loss

                # Backward
                gradients = self.backward(yi_arr)

                # Update
                self.update_weights(gradients, learning_rate)

            avg_loss = total_loss / len(X)
            self.loss_history.append(avg_loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            output = self.forward(x)
            predictions.append(1 if output[0] > 0.5 else 0)
        return np.array(predictions)


# Demo: Resolver XOR
print("=== Entrenando para XOR ===")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

net = NeuralNetworkFull(
    layer_sizes=[2, 4, 1],
    activations=['tanh', 'sigmoid']
)

net.fit(X_xor, y_xor, epochs=5000, learning_rate=0.5, verbose=True)

print("\n=== Predicciones XOR ===")
for x, y in zip(X_xor, y_xor):
    pred = net.forward(x)[0]
    print(f"{x} -> {pred:.4f} (target: {y})")
```

---

## 💻 Parte 4: Optimizadores

### 4.1 SGD (Stochastic Gradient Descent)

```python
class SGD:
    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate

    def update(self, layer, dW: np.ndarray, db: np.ndarray):
        layer.W -= self.lr * dW
        layer.b -= self.lr * db
```

### 4.2 SGD con Momentum

```python
class SGDMomentum:
    """
    SGD con Momentum.

    v_t = β·v_{t-1} + (1-β)·∇L
    θ = θ - lr·v_t

    Momentum ayuda a:
    - Acelerar convergencia
    - Escapar de mínimos locales
    - Reducir oscilaciones
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def update(self, layer, dW: np.ndarray, db: np.ndarray, layer_id: int):
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                'W': np.zeros_like(dW),
                'b': np.zeros_like(db)
            }

        v = self.velocities[layer_id]

        # Actualizar velocidad
        v['W'] = self.momentum * v['W'] + (1 - self.momentum) * dW
        v['b'] = self.momentum * v['b'] + (1 - self.momentum) * db

        # Actualizar parámetros
        layer.W -= self.lr * v['W']
        layer.b -= self.lr * v['b']
```

### 4.3 Adam Optimizer

```python
class Adam:
    """
    Adam: Adaptive Moment Estimation.

    Combina:
    - Momentum (primer momento)
    - RMSprop (segundo momento)

    m_t = β₁·m_{t-1} + (1-β₁)·g_t       (momentum)
    v_t = β₂·v_{t-1} + (1-β₂)·g_t²      (velocidad adaptativa)
    m̂_t = m_t / (1 - β₁^t)              (corrección de bias)
    v̂_t = v_t / (1 - β₂^t)
    θ = θ - lr · m̂_t / (√v̂_t + ε)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer, dW: np.ndarray, db: np.ndarray, layer_id: int):
        if layer_id not in self.m:
            self.m[layer_id] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}
            self.v[layer_id] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}

        self.t += 1
        m, v = self.m[layer_id], self.v[layer_id]

        # Actualizar momentos
        m['W'] = self.beta1 * m['W'] + (1 - self.beta1) * dW
        m['b'] = self.beta1 * m['b'] + (1 - self.beta1) * db
        v['W'] = self.beta2 * v['W'] + (1 - self.beta2) * dW**2
        v['b'] = self.beta2 * v['b'] + (1 - self.beta2) * db**2

        # Corrección de bias
        m_hat_W = m['W'] / (1 - self.beta1**self.t)
        m_hat_b = m['b'] / (1 - self.beta1**self.t)
        v_hat_W = v['W'] / (1 - self.beta2**self.t)
        v_hat_b = v['b'] / (1 - self.beta2**self.t)

        # Actualizar parámetros
        layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
        layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
```

---

## 📦 Entregable del Módulo

### `neural_network.py`

```python
"""
Neural Network Module

Implementación desde cero de:
- MLP (Multilayer Perceptron)
- Backpropagation
- Optimizadores (SGD, Momentum, Adam)
- Funciones de activación

Autor: [Tu nombre]
Módulo: 06 - Deep Learning
"""

import numpy as np
from typing import List, Tuple, Optional


# ============================================================
# ACTIVACIONES
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def tanh_deriv(a):
    return 1 - a**2

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


# ============================================================
# CAPA
# ============================================================

class Layer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu'):
        self.activation = activation
        scale = np.sqrt(2.0 / input_size) if activation == 'relu' else np.sqrt(1.0 / input_size)
        self.W = np.random.randn(output_size, input_size) * scale
        self.b = np.zeros(output_size)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        z = self.W @ x + self.b
        self.cache['z'] = z

        if self.activation == 'relu':
            a = relu(z)
        elif self.activation == 'sigmoid':
            a = sigmoid(z)
        elif self.activation == 'tanh':
            a = np.tanh(z)
        elif self.activation == 'softmax':
            a = softmax(z)
        else:
            a = z

        self.cache['a'] = a
        return a

    def backward(self, dL_da: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z, x, a = self.cache['z'], self.cache['x'], self.cache['a']

        if self.activation == 'sigmoid':
            da_dz = sigmoid_deriv(a)
        elif self.activation == 'relu':
            da_dz = relu_deriv(z)
        elif self.activation == 'tanh':
            da_dz = tanh_deriv(a)
        else:
            da_dz = np.ones_like(z)

        delta = dL_da * da_dz
        dL_dW = np.outer(delta, x)
        dL_db = delta
        dL_dx = self.W.T @ delta

        return dL_dx, dL_dW, dL_db


# ============================================================
# OPTIMIZADORES
# ============================================================

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers, gradients):
        for layer, (dW, db) in zip(layers, gradients):
            layer.W -= self.lr * dW
            layer.b -= self.lr * db


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, layers, gradients):
        self.t += 1
        for i, (layer, (dW, db)) in enumerate(zip(layers, gradients)):
            if i not in self.m:
                self.m[i] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}
                self.v[i] = {'W': np.zeros_like(dW), 'b': np.zeros_like(db)}

            self.m[i]['W'] = self.beta1 * self.m[i]['W'] + (1 - self.beta1) * dW
            self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * db
            self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1 - self.beta2) * dW**2
            self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * db**2

            m_hat_W = self.m[i]['W'] / (1 - self.beta1**self.t)
            m_hat_b = self.m[i]['b'] / (1 - self.beta1**self.t)
            v_hat_W = self.v[i]['W'] / (1 - self.beta2**self.t)
            v_hat_b = self.v[i]['b'] / (1 - self.beta2**self.t)

            layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


# ============================================================
# RED NEURONAL
# ============================================================

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], activations: List[str]):
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
                       for i in range(len(layer_sizes)-1)]
        self.loss_history = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true: np.ndarray) -> List[Tuple]:
        y_pred = self.layers[-1].cache['a']
        dL_da = y_pred - y_true

        gradients = []
        for layer in reversed(self.layers):
            dL_da, dW, db = layer.backward(dL_da)
            gradients.insert(0, (dW, db))
        return gradients

    def fit(self, X, y, epochs=1000, lr=0.1, optimizer='sgd', verbose=True):
        opt = Adam(lr) if optimizer == 'adam' else SGD(lr)

        for epoch in range(epochs):
            total_loss = 0
            for xi, yi in zip(X, y):
                yi_arr = np.atleast_1d(yi)
                output = self.forward(xi)

                # BCE loss
                output_clip = np.clip(output, 1e-15, 1-1e-15)
                loss = -np.sum(yi_arr * np.log(output_clip) + (1-yi_arr) * np.log(1-output_clip))
                total_loss += loss

                gradients = self.backward(yi_arr)
                opt.step(self.layers, gradients)

            self.loss_history.append(total_loss / len(X))
            if verbose and epoch % (epochs//10) == 0:
                print(f"Epoch {epoch}: Loss = {self.loss_history[-1]:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([1 if self.forward(x)[0] > 0.5 else 0 for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":
    print("=== Test: XOR Problem ===")
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0, 1, 1, 0])

    net = NeuralNetwork([2, 4, 1], ['tanh', 'sigmoid'])
    net.fit(X, y, epochs=5000, lr=0.5, verbose=True)

    print("\nPredicciones:")
    for xi, yi in zip(X, y):
        pred = net.forward(xi)[0]
        print(f"{xi} -> {pred:.4f} (target: {yi})")

    print(f"\nAccuracy: {net.score(X, y):.2%}")
    print("\n✓ Test XOR completado!")
```

---

## 💻 Parte 5: CNNs - Redes Convolucionales (Semana 19)

> ⚠️ **Nota:** En este módulo NO implementamos CNNs desde cero (es complejo). El objetivo es **entender la teoría** para el curso de Deep Learning de CU Boulder.

### 5.1 ¿Por Qué CNNs para Imágenes?

```
PROBLEMA CON MLP PARA IMÁGENES:

Imagen MNIST: 28x28 = 784 píxeles
MLP fully connected a capa de 256 neuronas:
  → 784 × 256 = 200,704 parámetros (¡solo primera capa!)

Imagen HD: 1920x1080x3 = 6,220,800 píxeles
  → Imposible conectar todo con todo

SOLUCIÓN: CONVOLUCIÓN
- Procesar regiones locales (no toda la imagen)
- Compartir pesos (el mismo filtro en toda la imagen)
- Detectar patrones sin importar su posición
```

### 5.2 La Operación de Convolución

```python
import numpy as np

def convolve2d_simple(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolución 2D simplificada (para entender el concepto).

    La convolución desliza un kernel (filtro) sobre la imagen
    y calcula el producto punto en cada posición.

    Args:
        image: Imagen de entrada (H, W)
        kernel: Filtro (kH, kW), típicamente 3x3 o 5x5

    Returns:
        Feature map (H-kH+1, W-kW+1)
    """
    H, W = image.shape
    kH, kW = kernel.shape

    # Tamaño del output (sin padding)
    out_H = H - kH + 1
    out_W = W - kW + 1

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            # Extraer región de la imagen
            region = image[i:i+kH, j:j+kW]
            # Producto punto con el kernel
            output[i, j] = np.sum(region * kernel)

    return output


# Ejemplo: Detección de bordes verticales
image = np.array([
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1],
])

# Kernel Sobel para bordes verticales
sobel_vertical = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

edges = convolve2d_simple(image, sobel_vertical)
print("Feature map (bordes verticales):")
print(edges)
```

### 5.3 Conceptos Clave de CNNs

```
┌─────────────────────────────────────────────────────────────────┐
│  VOCABULARIO CNN                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  KERNEL (FILTRO)                                                │
│  ├── Matriz pequeña (3x3, 5x5) que detecta patrones             │
│  ├── Los valores del kernel son APRENDIDOS (backprop)           │
│  └── Diferentes kernels detectan diferentes features            │
│                                                                 │
│  STRIDE                                                         │
│  ├── Cuántos píxeles se mueve el kernel en cada paso            │
│  ├── stride=1: mueve 1 píxel (output grande)                    │
│  └── stride=2: mueve 2 píxeles (output más pequeño)             │
│                                                                 │
│  PADDING                                                        │
│  ├── Añadir ceros alrededor de la imagen                        │
│  ├── 'valid': sin padding (output más pequeño)                  │
│  └── 'same': padding para mantener tamaño                       │
│                                                                 │
│  POOLING                                                        │
│  ├── Reduce dimensiones (downsampling)                          │
│  ├── Max Pooling: toma el máximo de cada región                 │
│  └── Average Pooling: toma el promedio                          │
│                                                                 │
│  FEATURE MAP                                                    │
│  └── Output de aplicar un filtro (lo que "ve" el filtro)        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Cálculo de Dimensiones (Importante para Exámenes)

```python
def output_size(input_size: int, kernel_size: int,
                stride: int = 1, padding: int = 0) -> int:
    """
    Fórmula para calcular tamaño del output de convolución.

    output_size = floor((input + 2*padding - kernel) / stride) + 1
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


# Ejemplos típicos de examen:
print("=== Ejercicios de dimensiones ===")

# Ejemplo 1: MNIST sin padding
# Input: 28x28, Kernel: 5x5, Stride: 1, Padding: 0
out = output_size(28, 5, stride=1, padding=0)
print(f"MNIST 28x28, kernel 5x5, stride 1: output = {out}x{out}")  # 24x24

# Ejemplo 2: Con padding 'same'
# Para mantener tamaño con kernel 3x3, necesitas padding=1
out = output_size(28, 3, stride=1, padding=1)
print(f"MNIST 28x28, kernel 3x3, padding 1: output = {out}x{out}")  # 28x28

# Ejemplo 3: Max Pooling 2x2 stride 2
out = output_size(24, 2, stride=2, padding=0)
print(f"24x24, pooling 2x2 stride 2: output = {out}x{out}")  # 12x12
```

### 5.5 Arquitectura Típica de CNN

```
┌─────────────────────────────────────────────────────────────────┐
│  ARQUITECTURA LENET-5 (Clásica para MNIST)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: 28x28x1 (imagen grayscale)                              │
│         │                                                       │
│         ▼                                                       │
│  [CONV 5x5, 6 filtros] → 24x24x6                                │
│         │                                                       │
│         ▼                                                       │
│  [ReLU]                                                         │
│         │                                                       │
│         ▼                                                       │
│  [MaxPool 2x2] → 12x12x6                                        │
│         │                                                       │
│         ▼                                                       │
│  [CONV 5x5, 16 filtros] → 8x8x16                                │
│         │                                                       │
│         ▼                                                       │
│  [ReLU]                                                         │
│         │                                                       │
│         ▼                                                       │
│  [MaxPool 2x2] → 4x4x16 = 256 neuronas                          │
│         │                                                       │
│         ▼                                                       │
│  [Flatten] → 256                                                │
│         │                                                       │
│         ▼                                                       │
│  [FC 120] → 120                                                 │
│         │                                                       │
│         ▼                                                       │
│  [FC 84] → 84                                                   │
│         │                                                       │
│         ▼                                                       │
│  [FC 10 + Softmax] → 10 clases (dígitos 0-9)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 Max Pooling

```python
def max_pool2d(x: np.ndarray, pool_size: int = 2) -> np.ndarray:
    """
    Max Pooling 2D.

    Reduce dimensiones tomando el máximo de cada región.
    Hace la red más robusta a pequeñas traslaciones.

    Args:
        x: Feature map (H, W)
        pool_size: Tamaño de la ventana (típicamente 2)

    Returns:
        Pooled output (H//pool_size, W//pool_size)
    """
    H, W = x.shape
    out_H, out_W = H // pool_size, W // pool_size

    output = np.zeros((out_H, out_W))

    for i in range(out_H):
        for j in range(out_W):
            region = x[i*pool_size:(i+1)*pool_size,
                      j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)

    return output


# Ejemplo
feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [3, 2, 1, 0],
    [1, 2, 3, 4]
])

pooled = max_pool2d(feature_map, pool_size=2)
print("Original 4x4:")
print(feature_map)
print("\nMax Pooled 2x2:")
print(pooled)  # [[6, 4], [3, 4]]
```

### 5.7 Por Qué Funcionan las CNNs

```
INTUICIÓN:

1. CAPAS INICIALES: Detectan features simples
   - Bordes horizontales, verticales, diagonales
   - Cambios de color, texturas

2. CAPAS MEDIAS: Combinan features simples
   - Esquinas, curvas, patrones

3. CAPAS PROFUNDAS: Features de alto nivel
   - Partes de objetos (ojos, ruedas, letras)

4. CAPAS FINALES: Objetos completos
   - "Esto es un 7", "Esto es un gato"

VENTAJAS CLAVE:
├── Parameter sharing: mismo filtro en toda la imagen
├── Sparse connectivity: cada output depende de región local
├── Translation invariance: detecta patrones sin importar posición
└── Hierarchical features: de simple a complejo
```

### 5.8 Recursos para Profundizar en CNNs

| Recurso | Descripción |
|---------|-------------|
| [3B1B - But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) | Intuición visual |
| [CS231n Stanford](http://cs231n.stanford.edu/) | Curso completo de CNNs |
| Deep Learning Book, Cap. 9 | Teoría formal |

---

## 📝 Derivación Analítica: Backpropagation a Mano (v3.2)

> 🎓 **Simulación de Examen:** *"Derive las ecuaciones de backpropagation para una red de 2 capas"*. Este es un clásico de exámenes de posgrado.

### Red de 2 Capas: Derivación Completa

**Arquitectura:**
- Input: $x$ (vector de features)
- Capa 1: $z_1 = W_1 x + b_1$, $a_1 = \sigma(z_1)$
- Capa 2: $z_2 = W_2 a_1 + b_2$, $\hat{y} = \sigma(z_2)$
- Loss: $L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$

#### Paso 1: Gradiente de la Capa de Salida

$$\frac{\partial L}{\partial z_2} = \hat{y} - y = \delta_2$$

(Resultado elegante gracias a la combinación sigmoid + cross-entropy)

$$\frac{\partial L}{\partial W_2} = \delta_2 \cdot a_1^T$$

$$\frac{\partial L}{\partial b_2} = \delta_2$$

#### Paso 2: Propagar el Error Hacia Atrás (Capa Oculta)

$$\frac{\partial L}{\partial a_1} = W_2^T \delta_2$$

$$\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \odot \sigma'(z_1) = W_2^T \delta_2 \odot a_1 \odot (1 - a_1) = \delta_1$$

$$\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^T$$

$$\frac{\partial L}{\partial b_1} = \delta_1$$

#### Resumen: Las 4 Ecuaciones de Backprop

```
┌─────────────────────────────────────────────────────────────┐
│ ECUACIONES DE BACKPROPAGATION                               │
│                                                             │
│ 1. δ_L = ∇_a L ⊙ σ'(z_L)     Error en capa final           │
│                                                             │
│ 2. δ_l = (W_{l+1}^T δ_{l+1}) ⊙ σ'(z_l)   Propagar atrás    │
│                                                             │
│ 3. ∂L/∂W_l = δ_l · a_{l-1}^T   Gradiente de pesos           │
│                                                             │
│ 4. ∂L/∂b_l = δ_l              Gradiente de bias             │
└─────────────────────────────────────────────────────────────┘
```

### Tu Entregable

Escribe en un documento (Markdown o LaTeX):
1. Derivación completa de backprop para red de 2 capas
2. Por qué $\delta_L = \hat{y} - y$ cuando usamos sigmoid + cross-entropy
3. Diagrama de grafo computacional mostrando el flujo de gradientes

---

## 🧪 Overfit on Small Batch: Debugging de Redes Neuronales (v3.3)

> ⚠️ **CRÍTICO:** Esta es la técnica #1 de debugging en Deep Learning. Si tu red no puede hacer overfitting en 10 ejemplos, tiene un bug.

### El Principio

```
REGLA DE ORO DEL DEBUGGING EN DL:

Una red neuronal DEBE poder memorizar un dataset pequeño.

Si entrenas con:
- 10 ejemplos
- Muchas épocas (1000+)
- Sin regularización

El loss DEBE llegar a ~0.00 (o muy cercano).

Si NO llega a 0 → TU IMPLEMENTACIÓN TIENE UN BUG
```

### Por Qué Funciona

```
┌─────────────────────────────────────────────────────────────┐
│ OVERFIT TEST                                                │
│                                                             │
│ Dataset pequeño (10 ejemplos):                              │
│ - Capacidad de la red >> complejidad del dataset            │
│ - La red puede "memorizar" cada ejemplo perfectamente       │
│ - Loss debe → 0 si backprop funciona                        │
│                                                             │
│ Si loss NO baja:                                            │
│ - Gradiente mal calculado                                   │
│ - Learning rate incorrecto                                  │
│ - Arquitectura rota (dimensiones)                           │
│ - Bug en forward o backward pass                            │
└─────────────────────────────────────────────────────────────┘
```

### Script: `overfit_test.py` (Entregable Obligatorio v3.3)

```python
"""
Overfit Test - Validación de Redes Neuronales
Si tu red no puede hacer overfit en 10 ejemplos, está rota.

Autor: [Tu nombre]
Módulo: 07 - Deep Learning
"""
import numpy as np
from typing import List, Tuple


def overfit_test(
    model,
    X_small: np.ndarray,
    y_small: np.ndarray,
    epochs: int = 2000,
    target_loss: float = 0.01,
    verbose: bool = True
) -> Tuple[bool, List[float]]:
    """
    Test de overfitting: la red debe memorizar un dataset pequeño.

    Args:
        model: Tu red neuronal (debe tener .fit() y .forward())
        X_small: Dataset pequeño (10-20 ejemplos)
        y_small: Labels del dataset
        epochs: Épocas de entrenamiento
        target_loss: Loss objetivo (default: 0.01)
        verbose: Mostrar progreso

    Returns:
        (passed, loss_history)
    """
    if verbose:
        print("=" * 60)
        print("OVERFIT TEST: ¿Puede tu red memorizar 10 ejemplos?")
        print("=" * 60)
        print(f"Dataset size: {len(y_small)}")
        print(f"Epochs: {epochs}")
        print(f"Target loss: {target_loss}")
        print("-" * 60)

    # Entrenar
    loss_history = []
    for epoch in range(epochs):
        # Forward pass para todos los ejemplos
        total_loss = 0.0
        for i in range(len(y_small)):
            output = model.forward(X_small[i])
            loss = np.mean((output - y_small[i]) ** 2)  # MSE
            total_loss += loss

            # Backward y update (asumiendo que model tiene estos métodos)
            model.backward(y_small[i])
            model.update(learning_rate=0.1)

        avg_loss = total_loss / len(y_small)
        loss_history.append(avg_loss)

        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch:4d}: Loss = {avg_loss:.6f}")

    final_loss = loss_history[-1]
    passed = final_loss < target_loss

    if verbose:
        print("-" * 60)
        print(f"Final Loss: {final_loss:.6f}")
        if passed:
            print("✓ PASSED: Tu red puede hacer overfitting")
            print("  → El forward y backward pass funcionan correctamente")
        else:
            print("✗ FAILED: Tu red NO puede hacer overfitting")
            print("  → Revisa tu implementación de backprop")
            print("  Posibles causas:")
            print("  - Gradiente mal calculado")
            print("  - Learning rate muy bajo")
            print("  - Bug en forward pass")
            print("  - Dimensiones incorrectas")

    return passed, loss_history


# ============================================================
# EJEMPLO: Test con XOR (debe pasar)
# ============================================================

def test_xor_overfit():
    """Test: Una red pequeña debe resolver XOR perfectamente."""
    print("\n" + "=" * 60)
    print("TEST: Overfit on XOR Problem")
    print("=" * 60)

    # XOR dataset (4 ejemplos)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float64)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=np.float64)

    # Crear red simple (2 -> 8 -> 1)
    # NOTA: Reemplaza esto con tu clase NeuralNetwork
    class SimpleNet:
        def __init__(self):
            np.random.seed(42)
            self.W1 = np.random.randn(8, 2) * 0.5
            self.b1 = np.zeros((8, 1))
            self.W2 = np.random.randn(1, 8) * 0.5
            self.b2 = np.zeros((1, 1))

            # Cache para backprop
            self.cache = {}

        def sigmoid(self, z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        def forward(self, x):
            x = x.reshape(-1, 1)
            z1 = self.W1 @ x + self.b1
            a1 = self.sigmoid(z1)
            z2 = self.W2 @ a1 + self.b2
            a2 = self.sigmoid(z2)

            self.cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
            return a2.flatten()

        def backward(self, y_true):
            y_true = np.array(y_true).reshape(-1, 1)
            a2 = self.cache['a2']
            a1 = self.cache['a1']
            x = self.cache['x']

            # Gradientes
            dz2 = a2 - y_true
            self.dW2 = dz2 @ a1.T
            self.db2 = dz2

            da1 = self.W2.T @ dz2
            dz1 = da1 * a1 * (1 - a1)
            self.dW1 = dz1 @ x.T
            self.db1 = dz1

        def update(self, learning_rate):
            self.W1 -= learning_rate * self.dW1
            self.b1 -= learning_rate * self.db1
            self.W2 -= learning_rate * self.dW2
            self.b2 -= learning_rate * self.db2

    # Ejecutar test
    model = SimpleNet()
    passed, history = overfit_test(model, X, y, epochs=2000, target_loss=0.01)

    # Verificar predicciones finales
    print("\nPredicciones finales:")
    for i in range(len(X)):
        pred = model.forward(X[i])
        print(f"  Input: {X[i]} → Pred: {pred[0]:.3f} (Target: {y[i][0]})")

    return passed


if __name__ == "__main__":
    test_xor_overfit()
```

### Checklist de Debugging con Overfit Test

| Síntoma | Diagnóstico | Solución |
|---------|-------------|----------|
| Loss no baja | Gradiente = 0 o NaN | Verificar derivadas con grad_check |
| Loss baja muy lento | Learning rate muy bajo | Aumentar LR (probar 0.1, 0.5, 1.0) |
| Loss oscila mucho | Learning rate muy alto | Reducir LR |
| Loss sube | Signos invertidos en gradiente | Revisar forward/backward |
| Loss = NaN | Overflow en exp/softmax | Usar versiones numéricamente estables |

---

## 🎯 El Reto del Tablero Blanco (Metodología Feynman)

Explica en **máximo 5 líneas** sin jerga técnica:

1. **¿Qué es backpropagation?**
   > Pista: Piensa en "culpar" a cada peso por el error.

2. **¿Por qué ReLU es mejor que sigmoid en capas ocultas?**
   > Pista: Piensa en qué pasa con el gradiente de sigmoid cuando z es muy grande o muy pequeño.

3. **¿Qué hace una convolución en una imagen?**
   > Pista: Piensa en "deslizar una lupa" buscando un patrón específico.

4. **¿Por qué usamos pooling?**
   > Pista: Piensa en "resumir" una región y hacerla más pequeña.

---

## ✅ Checklist de Finalización (v3.3)

### Conocimiento
- [ ] Entiendo la analogía neurona biológica → neurona artificial
- [ ] Implementé sigmoid, ReLU, tanh, softmax y sus derivadas
- [ ] Entiendo por qué XOR no es linealmente separable
- [ ] Implementé forward pass para MLP
- [ ] Entiendo la Chain Rule aplicada a backpropagation
- [ ] Implementé backward pass calculando gradientes
- [ ] Implementé SGD, SGD+Momentum y Adam
- [ ] Mi red resuelve el problema XOR

### CNNs (Teoría)
- [ ] Entiendo qué es convolución, stride, padding y pooling
- [ ] Puedo calcular dimensiones de output de una CNN
- [ ] Conozco la arquitectura LeNet-5

### Entregables de Código
- [ ] `neural_network.py` con tests pasando
- [ ] `mypy src/` pasa sin errores
- [ ] `pytest tests/` pasa sin errores

### Overfit Test (v3.3 - Obligatorio)
- [ ] **`overfit_test.py` implementado**
- [ ] **Mi red hace overfit en XOR (loss < 0.01)**
- [ ] Si el test falla, debuggeé con grad_check

### Derivación Analítica (Obligatorio)
- [ ] Derivé las ecuaciones de backprop a mano
- [ ] Documento con derivación completa (Markdown o LaTeX)
- [ ] Diagrama de grafo computacional

### Metodología Feynman
- [ ] Puedo explicar backpropagation en 5 líneas sin jerga
- [ ] Puedo explicar ReLU vs sigmoid en 5 líneas
- [ ] Puedo explicar convolución en 5 líneas
- [ ] Puedo explicar pooling en 5 líneas

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [06_UNSUPERVISED_LEARNING](06_UNSUPERVISED_LEARNING.md) | [00_INDICE](00_INDICE.md) | [08_PROYECTO_MNIST](08_PROYECTO_MNIST.md) |
