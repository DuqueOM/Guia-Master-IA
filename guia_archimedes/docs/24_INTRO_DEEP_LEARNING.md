# Módulo 09 - Introducción al Deep Learning

> **🎯 Objetivo:** Dominar fundamentos de redes neuronales y backpropagation  
> **⭐ PATHWAY LÍNEA 1:** Introduction to Deep Learning

---

## 🧠 Analogía: El Cerebro Artificial

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   RED NEURONAL = Funciones Compuestas que Aprenden                          │
│   ────────────────────────────────────────────────                          │
│                                                                             │
│   NEURONA BIOLÓGICA:          NEURONA ARTIFICIAL:                           │
│   ────────────────────        ────────────────────                          │
│   Dendritas → Soma → Axón     Inputs → Σ(wx+b) → Activación → Output        │
│                                                                             │
│        x₁ ──w₁──┐                                                           │
│                 │                                                           │
│        x₂ ──w₂──┼──▶ Σ ──▶ f(z) ──▶ y                                     │
│                 │    (suma)  (activ)                                        │
│        x₃ ──w₃──┘                                                           │
│               +b                                                            │
│                                                                             │
│   z = w₁x₁ + w₂x₂ + w₃x₃ + b                                                │
│   y = f(z)                                                                  │
│                                                                             │
│   ¿POR QUÉ "PROFUNDO"?                                                      │
│   ──────────────────────                                                    │
│   Múltiples capas permiten aprender representaciones jerárquicas:           │
│                                                                             │
│   Capa 1: Bordes, texturas                                                  │
│   Capa 2: Formas simples                                                    │
│   Capa 3: Partes de objetos                                                 │
│   Capa N: Conceptos complejos                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Perceptrón y Neurona](#1-perceptron)
2. [Funciones de Activación](#2-activaciones)
3. [Redes Multicapa (MLP)](#3-mlp)
4. [Backpropagation](#4-backpropagation)
5. [Optimización y Regularización](#5-optimizacion)
6. [Arquitecturas Especiales (CNN, RNN)](#6-arquitecturas)

---

## 1. Perceptrón y Neurona {#1-perceptron}

### 1.1 Perceptrón Simple

```python
from typing import List, Tuple
import math
import random

class Perceptron:
    """Single layer perceptron (binary classifier).
    
    The simplest neural network: one neuron.
    Can only learn linearly separable patterns.
    
    Model: y = sign(w·x + b)
    """
    
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.weights = [random.uniform(-1, 1) for _ in range(n_features)]
        self.bias = random.uniform(-1, 1)
    
    def predict_one(self, x: List[float]) -> int:
        """Predict for single sample."""
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return 1 if z >= 0 else 0
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict for multiple samples."""
        return [self.predict_one(x) for x in X]
    
    def fit(self, X: List[List[float]], y: List[int], 
            epochs: int = 100) -> 'Perceptron':
        """Train perceptron using perceptron learning rule.
        
        Update rule: w = w + lr × (y - ŷ) × x
        Only updates when prediction is wrong.
        """
        for _ in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_pred = self.predict_one(xi)
                error = yi - y_pred
                
                if error != 0:
                    errors += 1
                    for j in range(len(self.weights)):
                        self.weights[j] += self.lr * error * xi[j]
                    self.bias += self.lr * error
            
            if errors == 0:
                break  # Converged
        
        return self
```

### 1.2 Neurona con Activación Continua

```python
class Neuron:
    """Single neuron with continuous activation.
    
    More expressive than perceptron.
    Can use different activation functions.
    """
    
    def __init__(
        self, 
        n_inputs: int, 
        activation: str = 'sigmoid'
    ):
        self.weights = [random.gauss(0, 0.1) for _ in range(n_inputs)]
        self.bias = 0.0
        self.activation = activation
        
        # For backprop
        self.last_input: List[float] = []
        self.last_z: float = 0.0
        self.last_output: float = 0.0
    
    def _activate(self, z: float) -> float:
        """Apply activation function."""
        if self.activation == 'sigmoid':
            if z < -500:
                return 0.0
            elif z > 500:
                return 1.0
            return 1.0 / (1.0 + math.exp(-z))
        elif self.activation == 'relu':
            return max(0, z)
        elif self.activation == 'tanh':
            return math.tanh(z)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activation_derivative(self, z: float) -> float:
        """Derivative of activation function."""
        if self.activation == 'sigmoid':
            s = self._activate(z)
            return s * (1 - s)
        elif self.activation == 'relu':
            return 1.0 if z > 0 else 0.0
        elif self.activation == 'tanh':
            return 1 - math.tanh(z) ** 2
        elif self.activation == 'linear':
            return 1.0
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, x: List[float]) -> float:
        """Forward pass."""
        self.last_input = x
        self.last_z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        self.last_output = self._activate(self.last_z)
        return self.last_output
```

---

## 2. Funciones de Activación {#2-activaciones}

### 2.1 Comparación

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FUNCIONES DE ACTIVACIÓN                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SIGMOID: σ(z) = 1 / (1 + e⁻ᶻ)                                             │
│   ─────────────────────────────                                             │
│   Rango: (0, 1)                                                             │
│   Uso: Output para probabilidad binaria                                     │
│   Problema: Vanishing gradient para |z| grande                              │
│        ___________                                                          │
│       /                                                                     │
│   ───/───────────                                                           │
│                                                                             │
│   TANH: tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)                                   │
│   ────────────────────────────────────────                                  │
│   Rango: (-1, 1)                                                            │
│   Uso: Capas ocultas (centrado en 0)                                        │
│          ___                                                                │
│         /                                                                   │
│   _____/                                                                    │
│                                                                             │
│   ReLU: f(z) = max(0, z)                                                    │
│   ──────────────────────                                                    │
│   Rango: [0, ∞)                                                             │
│   Uso: ESTÁNDAR para capas ocultas                                          │
│   Ventaja: No vanishing gradient, rápido                                    │
│   Problema: "Dying ReLU" (neuronas muertas)                                 │
│            /                                                                │
│   ________/                                                                 │
│                                                                             │
│   Leaky ReLU: f(z) = max(αz, z), α ≈ 0.01                                   │
│   ──────────────────────────────────────                                    │
│   Soluciona dying ReLU                                                      │
│            /                                                                │
│   _      /                                                                  │
│    \____/                                                                   │
│                                                                             │
│   Softmax: σ(z)ᵢ = eᶻⁱ / Σeᶻʲ                                               │
│   ───────────────────────────                                               │
│   Rango: (0, 1), suma = 1                                                   │
│   Uso: Output para clasificación multiclase                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementación

```python
def sigmoid(z: float) -> float:
    """Sigmoid activation."""
    if z < -500:
        return 0.0
    elif z > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(z: float) -> float:
    """Derivative of sigmoid."""
    s = sigmoid(z)
    return s * (1 - s)


def relu(z: float) -> float:
    """ReLU activation."""
    return max(0, z)


def relu_derivative(z: float) -> float:
    """Derivative of ReLU."""
    return 1.0 if z > 0 else 0.0


def leaky_relu(z: float, alpha: float = 0.01) -> float:
    """Leaky ReLU activation."""
    return z if z > 0 else alpha * z


def softmax(z: List[float]) -> List[float]:
    """Softmax for vector (numerically stable)."""
    max_z = max(z)
    exp_z = [math.exp(zi - max_z) for zi in z]
    sum_exp = sum(exp_z)
    return [e / sum_exp for e in exp_z]
```

---

## 3. Redes Multicapa (MLP) {#3-mlp}

### 3.1 Arquitectura

```
MULTILAYER PERCEPTRON (MLP):
────────────────────────────

INPUT       HIDDEN 1     HIDDEN 2     OUTPUT
LAYER       LAYER        LAYER        LAYER

  x₁ ─────┬───────────┬────────────┬─────▶ ŷ₁
          │           │            │
  x₂ ─────┼───────────┼────────────┼─────▶ ŷ₂
          │           │            │
  x₃ ─────┴───────────┴────────────┴─────▶ ŷ₃

FORWARD PROPAGATION:
h₁ = f(W₁x + b₁)      # Primera capa oculta
h₂ = f(W₂h₁ + b₂)     # Segunda capa oculta
ŷ = g(W₃h₂ + b₃)      # Output (g puede ser softmax)

PARÁMETROS TOTALES:
Para arquitectura [input, h1, h2, output] = [784, 128, 64, 10]:
W₁: 784×128 + 128 = 100,480
W₂: 128×64 + 64 = 8,256
W₃: 64×10 + 10 = 650
Total: ~109,000 parámetros
```

### 3.2 Implementación

```python
class Layer:
    """A single layer in a neural network."""
    
    def __init__(
        self, 
        n_inputs: int, 
        n_neurons: int, 
        activation: str = 'relu'
    ):
        # Xavier initialization
        limit = math.sqrt(6 / (n_inputs + n_neurons))
        self.weights = [
            [random.uniform(-limit, limit) for _ in range(n_inputs)]
            for _ in range(n_neurons)
        ]
        self.biases = [0.0] * n_neurons
        self.activation = activation
        
        # Cache for backprop
        self.inputs: List[float] = []
        self.z: List[float] = []  # Pre-activation
        self.outputs: List[float] = []
        
        # Gradients
        self.weight_gradients: List[List[float]] = []
        self.bias_gradients: List[float] = []
    
    def _activate(self, z: float) -> float:
        """Apply activation function."""
        if self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'relu':
            return relu(z)
        elif self.activation == 'tanh':
            return math.tanh(z)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activation_derivative(self, z: float) -> float:
        """Derivative of activation function."""
        if self.activation == 'sigmoid':
            return sigmoid_derivative(z)
        elif self.activation == 'relu':
            return relu_derivative(z)
        elif self.activation == 'tanh':
            return 1 - math.tanh(z) ** 2
        elif self.activation == 'linear':
            return 1.0
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass through layer."""
        self.inputs = inputs
        self.z = []
        self.outputs = []
        
        for neuron_idx in range(len(self.weights)):
            # Linear combination
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            ) + self.biases[neuron_idx]
            self.z.append(z)
            
            # Activation
            self.outputs.append(self._activate(z))
        
        return self.outputs
    
    def backward(self, output_gradients: List[float]) -> List[float]:
        """Backward pass: compute gradients."""
        n_neurons = len(self.weights)
        n_inputs = len(self.weights[0])
        
        # Gradient of activation
        activation_gradients = [
            output_gradients[i] * self._activation_derivative(self.z[i])
            for i in range(n_neurons)
        ]
        
        # Weight gradients
        self.weight_gradients = [
            [activation_gradients[i] * self.inputs[j] for j in range(n_inputs)]
            for i in range(n_neurons)
        ]
        
        # Bias gradients
        self.bias_gradients = activation_gradients[:]
        
        # Input gradients (for previous layer)
        input_gradients = [0.0] * n_inputs
        for j in range(n_inputs):
            for i in range(n_neurons):
                input_gradients[j] += activation_gradients[i] * self.weights[i][j]
        
        return input_gradients
    
    def update(self, learning_rate: float) -> None:
        """Update weights using computed gradients."""
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= learning_rate * self.weight_gradients[i][j]
            self.biases[i] -= learning_rate * self.bias_gradients[i]


class NeuralNetwork:
    """Multilayer Perceptron neural network."""
    
    def __init__(self, layer_sizes: List[int], activations: List[str] = None):
        """
        Args:
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            activations: activation for each layer (default: relu + linear)
        """
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 2) + ['linear']
        
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                layer_sizes[i], 
                layer_sizes[i + 1], 
                activations[i]
            )
            self.layers.append(layer)
    
    def forward(self, x: List[float]) -> List[float]:
        """Forward pass through all layers."""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, loss_gradient: List[float]) -> None:
        """Backward pass through all layers."""
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
    
    def update(self, learning_rate: float) -> None:
        """Update all layers."""
        for layer in self.layers:
            layer.update(learning_rate)
    
    def predict(self, X: List[List[float]]) -> List[List[float]]:
        """Predict for batch."""
        return [self.forward(x) for x in X]
```

---

## 4. Backpropagation {#4-backpropagation}

### 4.1 La Regla de la Cadena

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   BACKPROPAGATION = Regla de la Cadena Aplicada                             │
│   ─────────────────────────────────────────────                             │
│                                                                             │
│   Objetivo: ∂L/∂wᵢⱼ (cómo cambiar cada peso para reducir el loss)           │
│                                                                             │
│   Forward:  x → [W₁] → h₁ → [W₂] → h₂ → [W₃] → ŷ → L                        │
│                                                                             │
│   Backward: x ← [∂] ← h₁ ← [∂] ← h₂ ← [∂] ← ŷ ← ∂L/∂ŷ                       │
│                                                                             │
│   REGLA DE LA CADENA:                                                       │
│   ───────────────────                                                       │
│   ∂L/∂W₂ = ∂L/∂ŷ × ∂ŷ/∂h₂ × ∂h₂/∂W₂                                         │
│                                                                             │
│   Para cada capa:                                                           │
│   1. Recibir gradiente de la capa siguiente (∂L/∂output)                    │
│   2. Multiplicar por derivada de la activación (∂output/∂z)                 │
│   3. Calcular gradientes de pesos: ∂L/∂W = (grad) × input                   │
│   4. Pasar gradiente a capa anterior: ∂L/∂input = Wᵀ × (grad)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Funciones de Pérdida

```python
def mse_loss(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Squared Error loss."""
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


def mse_gradient(y_true: List[float], y_pred: List[float]) -> List[float]:
    """Gradient of MSE loss with respect to predictions."""
    n = len(y_true)
    return [2 * (yp - yt) / n for yt, yp in zip(y_true, y_pred)]


def binary_cross_entropy(y_true: List[float], y_pred: List[float]) -> float:
    """Binary cross-entropy loss."""
    eps = 1e-15
    loss = 0.0
    for yt, yp in zip(y_true, y_pred):
        yp = max(min(yp, 1 - eps), eps)  # Clip to avoid log(0)
        loss -= yt * math.log(yp) + (1 - yt) * math.log(1 - yp)
    return loss / len(y_true)


def bce_gradient(y_true: List[float], y_pred: List[float]) -> List[float]:
    """Gradient of binary cross-entropy."""
    eps = 1e-15
    return [
        ((yp - yt) / (yp * (1 - yp) + eps)) / len(y_true)
        for yt, yp in zip(y_true, y_pred)
    ]


def categorical_cross_entropy(y_true: List[int], y_pred: List[List[float]]) -> float:
    """Cross-entropy for multi-class classification.
    
    y_true: class indices
    y_pred: softmax probabilities
    """
    eps = 1e-15
    loss = 0.0
    for i, (true_class, pred_probs) in enumerate(zip(y_true, y_pred)):
        pred = max(pred_probs[true_class], eps)
        loss -= math.log(pred)
    return loss / len(y_true)
```

### 4.3 Training Loop Completo

```python
def train_network(
    network: NeuralNetwork,
    X_train: List[List[float]],
    y_train: List[List[float]],
    epochs: int = 100,
    learning_rate: float = 0.01,
    batch_size: int = 32,
    verbose: bool = True
) -> List[float]:
    """Train neural network with mini-batch gradient descent.
    
    Returns list of losses per epoch.
    """
    n_samples = len(X_train)
    losses = []
    
    for epoch in range(epochs):
        # Shuffle data
        indices = list(range(n_samples))
        random.shuffle(indices)
        
        epoch_loss = 0.0
        n_batches = 0
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]
            
            batch_loss = 0.0
            
            for idx in batch_indices:
                x = X_train[idx]
                y = y_train[idx]
                
                # Forward
                y_pred = network.forward(x)
                
                # Loss
                loss = mse_loss(y, y_pred)
                batch_loss += loss
                
                # Backward
                gradient = mse_gradient(y, y_pred)
                network.backward(gradient)
                
                # Update
                network.update(learning_rate)
            
            epoch_loss += batch_loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_samples
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses
```

---

## 5. Optimización y Regularización {#5-optimizacion}

### 5.1 Optimizadores

```python
class SGD:
    """Stochastic Gradient Descent with momentum."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities: Dict = {}
    
    def update(self, param_id: str, param: List[float], 
               gradient: List[float]) -> List[float]:
        """Update parameters."""
        if param_id not in self.velocities:
            self.velocities[param_id] = [0.0] * len(param)
        
        v = self.velocities[param_id]
        
        for i in range(len(param)):
            v[i] = self.momentum * v[i] - self.lr * gradient[i]
            param[i] += v[i]
        
        return param


class Adam:
    """Adam optimizer (simplified)."""
    
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
        self.m: Dict = {}  # First moment
        self.v: Dict = {}  # Second moment
        self.t: int = 0    # Time step
    
    def update(self, param_id: str, param: List[float], 
               gradient: List[float]) -> List[float]:
        """Update parameters using Adam."""
        self.t += 1
        
        if param_id not in self.m:
            self.m[param_id] = [0.0] * len(param)
            self.v[param_id] = [0.0] * len(param)
        
        m = self.m[param_id]
        v = self.v[param_id]
        
        for i in range(len(param)):
            # Update biased first moment
            m[i] = self.beta1 * m[i] + (1 - self.beta1) * gradient[i]
            
            # Update biased second moment
            v[i] = self.beta2 * v[i] + (1 - self.beta2) * gradient[i] ** 2
            
            # Bias correction
            m_hat = m[i] / (1 - self.beta1 ** self.t)
            v_hat = v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameter
            param[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon)
        
        return param
```

### 5.2 Regularización

```python
def l2_regularization(weights: List[List[float]], lambda_: float) -> float:
    """L2 regularization term: λ × Σ w²."""
    total = 0.0
    for layer_weights in weights:
        for row in layer_weights:
            total += sum(w ** 2 for w in row)
    return lambda_ * total


def l2_gradient(weight: float, lambda_: float) -> float:
    """Gradient of L2 regularization: 2λw."""
    return 2 * lambda_ * weight


def dropout(layer_output: List[float], keep_prob: float = 0.8, 
            training: bool = True) -> List[float]:
    """Dropout regularization.
    
    Randomly zeros out neurons during training.
    Scales outputs during inference.
    """
    if not training:
        return layer_output
    
    result = []
    for val in layer_output:
        if random.random() < keep_prob:
            result.append(val / keep_prob)  # Inverted dropout
        else:
            result.append(0.0)
    
    return result
```

### 5.3 Batch Normalization (Concepto)

```
BATCH NORMALIZATION:
────────────────────

Normaliza las activaciones de cada capa:

1. Calcular μ y σ del batch
2. Normalizar: x̂ = (x - μ) / σ
3. Escalar y desplazar: y = γx̂ + β (parámetros aprendidos)

BENEFICIOS:
• Permite learning rates más altas
• Reduce dependencia de inicialización
• Actúa como regularizador
• Acelera el entrenamiento

NOTA: Comportamiento diferente en train vs inference
• Train: estadísticas del batch
• Inference: estadísticas acumuladas (running mean/var)
```

---

## 6. Arquitecturas Especiales (CNN, RNN) {#6-arquitecturas}

### 6.1 Convolutional Neural Networks (CNN)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   CONVOLUTIONAL NEURAL NETWORKS                                             │
│   ─────────────────────────────                                             │
│                                                                             │
│   Para datos con estructura espacial (imágenes).                            │
│                                                                             │
│   CONVOLUCIÓN:                                                              │
│   ─────────────                                                             │
│   Filtro 3×3 deslizándose sobre la imagen:                                  │
│                                                                             │
│   Input Image        Filter        Feature Map                              │
│   ┌─────────┐       ┌───┐         ┌───────┐                                 │
│   │ 1 2 3 4 │   *   │a b│    =    │ . . . │                                 │
│   │ 5 6 7 8 │       │c d│         │ . . . │                                 │
│   │ 9 . . . │       └───┘         └───────┘                                 │
│   └─────────┘                                                               │
│                                                                             │
│   output[i,j] = Σ input[i+k, j+l] × filter[k,l]                             │
│                                                                             │
│   POOLING:                                                                  │
│   ─────────                                                                 │
│   Reduce dimensión espacial:                                                │
│   • Max Pooling: toma el máximo de cada región                              │
│   • Average Pooling: promedio de cada región                                │
│                                                                             │
│   ARQUITECTURA TÍPICA:                                                      │
│   ────────────────────                                                      │
│   [Conv → ReLU → Pool] × N → Flatten → [Dense] × M → Output                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Recurrent Neural Networks (RNN)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   RECURRENT NEURAL NETWORKS                                                 │
│   ─────────────────────────                                                 │
│                                                                             │
│   Para secuencias (texto, series temporales).                               │
│                                                                             │
│   ESTADO OCULTO:                                                            │
│   ───────────────                                                           │
│   hₜ = f(Wₓₕ × xₜ + Wₕₕ × hₜ₋₁ + b)                                            │
│                                                                             │
│   El estado anterior influye en el actual.                                  │
│                                                                             │
│       x₁        x₂        x₃        x₄                                      │
│        ↓         ↓         ↓         ↓                                      │
│       [h] ───▶ [h] ───▶ [h] ───▶ [h]                                      │
│        ↓         ↓         ↓         ↓                                      │
│       y₁        y₂        y₃        y₄                                      │
│                                                                             │
│   PROBLEMA: Vanishing/Exploding Gradients                                   │
│   SOLUCIÓN: LSTM, GRU (gated architectures)                                 │
│                                                                             │
│   LSTM (Long Short-Term Memory):                                            │
│   ──────────────────────────────                                            │
│   • Forget gate: qué olvidar del estado anterior                            │
│   • Input gate: qué nueva información agregar                               │
│   • Output gate: qué output producir                                        │
│   • Cell state: memoria a largo plazo                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Convolution Simplificada

```python
def convolve_2d(
    image: List[List[float]], 
    kernel: List[List[float]]
) -> List[List[float]]:
    """2D convolution (no padding, stride=1).
    
    Simplified implementation for understanding.
    """
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    
    output_h = h - kh + 1
    output_w = w - kw + 1
    
    output = [[0.0] * output_w for _ in range(output_h)]
    
    for i in range(output_h):
        for j in range(output_w):
            total = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    total += image[i + ki][j + kj] * kernel[ki][kj]
            output[i][j] = total
    
    return output


def max_pool_2d(
    feature_map: List[List[float]], 
    pool_size: int = 2
) -> List[List[float]]:
    """Max pooling with given pool size."""
    h, w = len(feature_map), len(feature_map[0])
    output_h = h // pool_size
    output_w = w // pool_size
    
    output = [[0.0] * output_w for _ in range(output_h)]
    
    for i in range(output_h):
        for j in range(output_w):
            max_val = float('-inf')
            for pi in range(pool_size):
                for pj in range(pool_size):
                    val = feature_map[i * pool_size + pi][j * pool_size + pj]
                    max_val = max(max_val, val)
            output[i][j] = max_val
    
    return output
```

---

## ⚠️ Mejores Prácticas

```
DEEP LEARNING BEST PRACTICES:
─────────────────────────────

DATOS:
• Más datos > modelo más complejo
• Augmentación para aumentar datos
• Normalización de inputs

ARQUITECTURA:
• Empezar simple, agregar complejidad si es necesario
• ReLU para capas ocultas
• Batch normalization después de capas densas

ENTRENAMIENTO:
• Adam optimizer por defecto
• Learning rate scheduling
• Early stopping
• Validación para detectar overfitting

DEBUGGING:
• Verificar que loss disminuye en train pequeño
• Graficar loss curves
• Monitorear gradientes (no vanishing/exploding)
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 24.1: Perceptrón para AND/OR
Entrenar perceptrón en funciones lógicas simples.

### Ejercicio 24.2: MLP para XOR
Red de 2 capas para resolver XOR (no linealmente separable).

### Ejercicio 24.3: MNIST desde Cero
Clasificar dígitos con MLP implementado manualmente.

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) | Curso | 🔴 Obligatorio |
| [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Videos | 🔴 Obligatorio |
| [Neural Networks from Scratch](https://nnfs.io/) | Libro | 🟡 Recomendado |

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [23_ML_NO_SUPERVISADO](23_ML_NO_SUPERVISADO.md) | [00_INDICE](00_INDICE.md) | [12_PROYECTO_INTEGRADOR](12_PROYECTO_INTEGRADOR.md) |
