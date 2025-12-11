# Módulo 05 - Supervised Learning

> **🎯 Objetivo:** Dominar regresión lineal, logística y métricas de evaluación
> **Fase:** 2 - Núcleo de ML | **Semanas 9-12**
> **Curso del Pathway:** Introduction to Machine Learning: Supervised Learning

---

## 🧠 ¿Qué es Supervised Learning?

```
APRENDIZAJE SUPERVISADO

Tenemos:
- Datos de entrada X (features)
- Etiquetas Y (targets/labels)

Objetivo: Aprender una función f tal que f(X) ≈ Y

Tipos principales:
├── REGRESIÓN: Y es continuo (precio, temperatura)
│   └── Output: número real
└── CLASIFICACIÓN: Y es discreto (spam/no spam, dígito 0-9)
    └── Output: clase o probabilidad
```

---

## 📚 Contenido del Módulo

| Semana | Tema | Entregable |
|--------|------|------------|
| 9 | Regresión Lineal | `linear_regression.py` |
| 10 | Regresión Logística | `logistic_regression.py` |
| 11 | Métricas de Evaluación | `metrics.py` |
| 12 | Validación y Regularización | Cross-validation, L1/L2 |

---

## 💻 Parte 1: Regresión Lineal

### 1.1 Modelo

```python
import numpy as np

"""
REGRESIÓN LINEAL

Hipótesis: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
         = θᵀx (forma matricial)

Donde:
- θ (theta): parámetros/pesos del modelo
- x: vector de features (con x₀ = 1 para el bias)

En forma matricial para múltiples muestras:
    ŷ = Xθ

Donde:
- X: matriz (m × n+1) con m muestras y n features + columna de 1s
- θ: vector (n+1 × 1) de parámetros
- ŷ: vector (m × 1) de predicciones
"""

def add_bias_term(X: np.ndarray) -> np.ndarray:
    """Añade columna de 1s para el término de bias."""
    m = X.shape[0]
    return np.column_stack([np.ones(m), X])

def predict_linear(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predicción lineal: ŷ = Xθ"""
    return X @ theta
```

### 1.2 Función de Costo (MSE)

```python
import numpy as np

def mse_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Mean Squared Error Cost Function.

    J(θ) = (1/2m) Σᵢ (h(xᵢ) - yᵢ)²
         = (1/2m) ||Xθ - y||²

    El factor 1/2 es por conveniencia (cancela con la derivada).
    """
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def mse_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Gradiente del MSE respecto a θ.

    ∂J/∂θ = (1/m) Xᵀ(Xθ - y)
    """
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    return (1 / m) * X.T @ errors
```

### 1.3 Solución Cerrada (Normal Equation)

```python
import numpy as np

def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solución cerrada para regresión lineal.

    θ = (XᵀX)⁻¹ Xᵀy

    Ventajas:
    - No requiere iteraciones
    - No hay hiperparámetros (learning rate)

    Desventajas:
    - O(n³) por la inversión de matriz
    - No funciona si XᵀX es singular
    - No escala bien para n grande (>10,000 features)
    """
    XtX = X.T @ X
    Xty = X.T @ y

    # Usar solve en lugar de inv para estabilidad numérica
    theta = np.linalg.solve(XtX, Xty)
    return theta
```

### 1.4 Gradient Descent para Regresión

```python
import numpy as np
from typing import List, Tuple

class LinearRegression:
    """Regresión Lineal implementada desde cero."""

    def __init__(self):
        self.theta = None
        self.cost_history = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'gradient_descent',
        learning_rate: float = 0.01,
        n_iterations: int = 1000
    ) -> 'LinearRegression':
        """
        Entrena el modelo.

        Args:
            X: features (m, n)
            y: targets (m,)
            method: 'gradient_descent' o 'normal_equation'
            learning_rate: tasa de aprendizaje (solo para GD)
            n_iterations: número de iteraciones (solo para GD)
        """
        # Añadir bias
        X_b = add_bias_term(X)
        m, n = X_b.shape

        if method == 'normal_equation':
            self.theta = normal_equation(X_b, y)
        else:
            # Inicializar theta con ceros o valores pequeños
            self.theta = np.zeros(n)

            for i in range(n_iterations):
                # Calcular gradiente
                gradient = mse_gradient(X_b, y, self.theta)

                # Actualizar theta
                self.theta = self.theta - learning_rate * gradient

                # Guardar costo para monitoreo
                cost = mse_cost(X_b, y, self.theta)
                self.cost_history.append(cost)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice valores."""
        X_b = add_bias_term(X)
        return X_b @ self.theta

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# Demo
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5  # y = 4 + 3x + ruido

model = LinearRegression()
model.fit(X, y, method='gradient_descent', learning_rate=0.1, n_iterations=1000)

print(f"Parámetros aprendidos: {model.theta}")
print(f"Esperados: [4, 3]")
print(f"R² score: {model.score(X, y):.4f}")
```

---

## 💻 Parte 2: Regresión Logística

### 2.1 Función Sigmoid

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Función sigmoid/logística.

    σ(z) = 1 / (1 + e^(-z))

    Propiedades:
    - Rango: (0, 1) - perfecto para probabilidades
    - σ(0) = 0.5
    - σ'(z) = σ(z)(1 - σ(z))
    """
    # Clip para evitar overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Visualizar
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)
plt.figure(figsize=(8, 4))
plt.plot(z, sigmoid(z))
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Función Sigmoid')
plt.grid(True)
# plt.show()
```

### 2.2 Hipótesis Logística

```python
"""
REGRESIÓN LOGÍSTICA

No predice un valor continuo, sino la PROBABILIDAD de pertenecer a la clase 1.

h(x) = P(y=1|x; θ) = σ(θᵀx)

Decisión:
- Si h(x) ≥ 0.5 → predicir clase 1
- Si h(x) < 0.5 → predicir clase 0

Equivalente a:
- Si θᵀx ≥ 0 → clase 1
- Si θᵀx < 0 → clase 0

El "decision boundary" está en θᵀx = 0
"""

def predict_proba(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predice probabilidad de clase 1."""
    return sigmoid(X @ theta)

def predict_class(X: np.ndarray, theta: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Predice clase (0 o 1)."""
    return (predict_proba(X, theta) >= threshold).astype(int)
```

### 2.3 Binary Cross-Entropy Loss

```python
import numpy as np

def binary_cross_entropy(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Binary Cross-Entropy (Log Loss).

    J(θ) = -(1/m) Σᵢ [yᵢ log(hᵢ) + (1-yᵢ) log(1-hᵢ)]

    Donde hᵢ = σ(θᵀxᵢ)

    Por qué esta función de costo:
    - Es convexa (tiene un único mínimo global)
    - Penaliza mucho las predicciones muy incorrectas
    - Es la derivación de Maximum Likelihood Estimation
    """
    m = len(y)
    h = sigmoid(X @ theta)

    # Clip para evitar log(0)
    h = np.clip(h, eps, 1 - eps)

    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def bce_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Gradiente de Binary Cross-Entropy.

    ∂J/∂θ = (1/m) Xᵀ(h - y)

    ¡Tiene la misma forma que el gradiente del MSE!
    Esto es porque derivamos σ(z) y la derivada σ'(z) = σ(z)(1-σ(z))
    cancela parte de la expresión.
    """
    m = len(y)
    h = sigmoid(X @ theta)
    return (1/m) * X.T @ (h - y)
```

### 2.4 Implementación Completa

```python
import numpy as np
from typing import List

class LogisticRegression:
    """Regresión Logística implementada desde cero."""

    def __init__(self):
        self.theta = None
        self.cost_history = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.1,
        n_iterations: int = 1000
    ) -> 'LogisticRegression':
        """Entrena con gradient descent."""
        # Añadir bias
        X_b = np.column_stack([np.ones(len(X)), X])
        m, n = X_b.shape

        # Inicializar
        self.theta = np.zeros(n)

        for i in range(n_iterations):
            # Gradiente
            gradient = bce_gradient(X_b, y, self.theta)

            # Actualizar
            self.theta = self.theta - learning_rate * gradient

            # Guardar costo
            cost = binary_cross_entropy(X_b, y, self.theta)
            self.cost_history.append(cost)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice probabilidades."""
        X_b = np.column_stack([np.ones(len(X)), X])
        return sigmoid(X_b @ self.theta)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predice clases."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy."""
        return np.mean(self.predict(X) == y)


# Demo con datos sintéticos
np.random.seed(42)

# Generar datos de dos clases
n_samples = 200
X_class0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
X_class1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Entrenar
model = LogisticRegression()
model.fit(X, y, learning_rate=0.1, n_iterations=1000)

print(f"Accuracy: {model.score(X, y):.2%}")
print(f"Parámetros: {model.theta}")
```

---

## 💻 Parte 3: Métricas de Evaluación

### 3.1 Matriz de Confusión

```python
import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de confusión.

    Para clasificación binaria:

                    Predicho
                    0       1
    Real    0      TN      FP
            1      FN      TP

    - TP (True Positive): Predijo 1, era 1
    - TN (True Negative): Predijo 0, era 0
    - FP (False Positive): Predijo 1, era 0 (Error Tipo I)
    - FN (False Negative): Predijo 0, era 1 (Error Tipo II)
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    return cm

def extract_tp_tn_fp_fn(y_true: np.ndarray, y_pred: np.ndarray):
    """Extrae TP, TN, FP, FN para clasificación binaria."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn
```

### 3.2 Accuracy, Precision, Recall, F1

```python
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Proporción de predicciones correctas.

    Problema: Puede ser engañoso con clases desbalanceadas.
    Si 99% son clase 0, predecir siempre 0 da 99% accuracy.
    """
    return np.mean(y_true == y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)

    De todos los que predije como positivos, ¿cuántos realmente lo son?

    Alta precisión = pocos falsos positivos.
    Importante cuando el costo de FP es alto (ej: spam → inbox).
    """
    tp, tn, fp, fn = extract_tp_tn_fp_fn(y_true, y_pred)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall (Sensitivity, True Positive Rate) = TP / (TP + FN)

    De todos los positivos reales, ¿cuántos capturé?

    Alto recall = pocos falsos negativos.
    Importante cuando el costo de FN es alto (ej: detección de cáncer).
    """
    tp, tn, fp, fn = extract_tp_tn_fp_fn(y_true, y_pred)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 = 2 * (precision * recall) / (precision + recall)

    Media armónica de precision y recall.

    Útil cuando quieres un balance entre ambas métricas.
    F1 alto solo si AMBAS precision y recall son altas.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Specificity (True Negative Rate) = TN / (TN + FP)

    De todos los negativos reales, ¿cuántos identifiqué?
    """
    tp, tn, fp, fn = extract_tp_tn_fp_fn(y_true, y_pred)
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)
```

### 3.3 Clase Metrics Completa

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class ClassificationReport:
    """Reporte de métricas de clasificación."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    confusion_matrix: np.ndarray

    def __str__(self) -> str:
        cm = self.confusion_matrix
        return f"""
Classification Report
=====================
Accuracy:    {self.accuracy:.4f}
Precision:   {self.precision:.4f}
Recall:      {self.recall:.4f}
F1 Score:    {self.f1:.4f}
Specificity: {self.specificity:.4f}

Confusion Matrix:
           Pred 0  Pred 1
Actual 0   {cm[0,0]:5d}   {cm[0,1]:5d}
Actual 1   {cm[1,0]:5d}   {cm[1,1]:5d}
"""

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationReport:
    """Genera reporte completo de métricas."""
    return ClassificationReport(
        accuracy=accuracy(y_true, y_pred),
        precision=precision(y_true, y_pred),
        recall=recall(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        specificity=specificity(y_true, y_pred),
        confusion_matrix=confusion_matrix(y_true, y_pred)
    )

# Demo
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])

report = classification_report(y_true, y_pred)
print(report)
```

---

## 💻 Parte 4: Validación y Regularización

### 4.1 Train/Test Split

```python
import numpy as np

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = None
) -> tuple:
    """
    Divide datos en conjuntos de entrenamiento y prueba.

    Args:
        X: features
        y: targets
        test_size: proporción para test (0-1)
        random_state: semilla para reproducibilidad
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(y)
    indices = np.random.permutation(n)

    test_size_n = int(n * test_size)
    test_indices = indices[:test_size_n]
    train_indices = indices[test_size_n:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
```

### 4.2 K-Fold Cross Validation

```python
import numpy as np
from typing import List, Tuple

def k_fold_split(n: int, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Genera índices para K-Fold Cross Validation.

    Returns:
        Lista de (train_indices, val_indices) para cada fold
    """
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n

        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        folds.append((train_indices, val_indices))

    return folds

def cross_validate(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    **model_params
) -> dict:
    """
    Realiza K-Fold Cross Validation.

    Returns:
        Dict con scores de cada fold y promedio
    """
    folds = k_fold_split(len(y), k)
    scores = []

    for i, (train_idx, val_idx) in enumerate(folds):
        # Split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train
        model = model_class()
        model.fit(X_train, y_train, **model_params)

        # Evaluate
        score = model.score(X_val, y_val)
        scores.append(score)

    return {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }

# Demo
# cv_results = cross_validate(LogisticRegression, X, y, k=5, learning_rate=0.1, n_iterations=500)
# print(f"CV Accuracy: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
```

### 4.3 Regularización

```python
import numpy as np

class LogisticRegressionRegularized:
    """Logistic Regression con regularización L1/L2."""

    def __init__(self, regularization: str = 'l2', lambda_: float = 0.01):
        """
        Args:
            regularization: 'l1', 'l2', o None
            lambda_: fuerza de regularización
        """
        self.regularization = regularization
        self.lambda_ = lambda_
        self.theta = None
        self.cost_history = []

    def _cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Costo con regularización."""
        m = len(y)
        h = sigmoid(X @ self.theta)
        h = np.clip(h, 1e-15, 1 - 1e-15)

        # Cross-entropy base
        bce = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

        # Regularización (excluir bias theta[0])
        if self.regularization == 'l2':
            # Ridge: λ/2m * Σθⱼ²
            reg = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:] ** 2)
        elif self.regularization == 'l1':
            # Lasso: λ/m * Σ|θⱼ|
            reg = (self.lambda_ / m) * np.sum(np.abs(self.theta[1:]))
        else:
            reg = 0

        return bce + reg

    def _gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradiente con regularización."""
        m = len(y)
        h = sigmoid(X @ self.theta)

        # Gradiente base
        grad = (1/m) * X.T @ (h - y)

        # Regularización (excluir bias)
        if self.regularization == 'l2':
            reg_grad = np.concatenate([[0], (self.lambda_ / m) * self.theta[1:]])
        elif self.regularization == 'l1':
            reg_grad = np.concatenate([[0], (self.lambda_ / m) * np.sign(self.theta[1:])])
        else:
            reg_grad = 0

        return grad + reg_grad

    def fit(self, X: np.ndarray, y: np.ndarray,
            learning_rate: float = 0.1, n_iterations: int = 1000):
        X_b = np.column_stack([np.ones(len(X)), X])
        self.theta = np.zeros(X_b.shape[1])

        for _ in range(n_iterations):
            gradient = self._gradient(X_b, y)
            self.theta -= learning_rate * gradient
            self.cost_history.append(self._cost(X_b, y))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.column_stack([np.ones(len(X)), X])
        return (sigmoid(X_b @ self.theta) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)
```

---

## 📦 Entregable del Módulo

### `supervised_learning.py`

```python
"""
Supervised Learning Module

Implementación desde cero de:
- Linear Regression (con Normal Equation y Gradient Descent)
- Logistic Regression (con regularización L1/L2)
- Métricas de evaluación
- Cross Validation

Autor: [Tu nombre]
Módulo: 04 - Supervised Learning
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


# ============================================================
# REGRESIÓN LINEAL
# ============================================================

class LinearRegression:
    def __init__(self):
        self.theta = None
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            method: str = 'normal', lr: float = 0.01, n_iter: int = 1000):
        X_b = add_bias(X)

        if method == 'normal':
            self.theta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
        else:
            m, n = X_b.shape
            self.theta = np.zeros(n)
            for _ in range(n_iter):
                grad = (1/m) * X_b.T @ (X_b @ self.theta - y)
                self.theta -= lr * grad
                self.cost_history.append(np.mean((X_b @ self.theta - y)**2))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return add_bias(X) @ self.theta

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / ss_tot


# ============================================================
# REGRESIÓN LOGÍSTICA
# ============================================================

class LogisticRegression:
    def __init__(self, reg: str = None, lambda_: float = 0.01):
        self.reg = reg
        self.lambda_ = lambda_
        self.theta = None
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            lr: float = 0.1, n_iter: int = 1000):
        X_b = add_bias(X)
        m, n = X_b.shape
        self.theta = np.zeros(n)

        for _ in range(n_iter):
            h = sigmoid(X_b @ self.theta)
            grad = (1/m) * X_b.T @ (h - y)

            if self.reg == 'l2':
                grad[1:] += (self.lambda_/m) * self.theta[1:]
            elif self.reg == 'l1':
                grad[1:] += (self.lambda_/m) * np.sign(self.theta[1:])

            self.theta -= lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(add_bias(X) @ self.theta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# ============================================================
# MÉTRICAS
# ============================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2*p*r/(p+r) if (p+r) > 0 else 0

def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    cm[0, 0] = np.sum((y_true == 0) & (y_pred == 0))
    cm[0, 1] = np.sum((y_true == 0) & (y_pred == 1))
    cm[1, 0] = np.sum((y_true == 1) & (y_pred == 0))
    cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))
    return cm


# ============================================================
# VALIDACIÓN
# ============================================================

def train_test_split(X, y, test_size=0.2, seed=None):
    if seed: np.random.seed(seed)
    n = len(y)
    idx = np.random.permutation(n)
    split = int(n * test_size)
    return X[idx[split:]], X[idx[:split]], y[idx[split:]], y[idx[:split]]

def cross_validate(model_class, X, y, k=5, **params):
    n = len(y)
    idx = np.random.permutation(n)
    fold_size = n // k
    scores = []

    for i in range(k):
        val_idx = idx[i*fold_size:(i+1)*fold_size]
        train_idx = np.concatenate([idx[:i*fold_size], idx[(i+1)*fold_size:]])

        model = model_class()
        model.fit(X[train_idx], y[train_idx], **params)
        scores.append(model.score(X[val_idx], y[val_idx]))

    return {'scores': scores, 'mean': np.mean(scores), 'std': np.std(scores)}


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Test Linear Regression
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5

    lr = LinearRegression()
    lr.fit(X, y)
    print(f"Linear Regression R²: {lr.score(X, y):.4f}")

    # Test Logistic Regression
    X_c0 = np.random.randn(50, 2) + [-2, -2]
    X_c1 = np.random.randn(50, 2) + [2, 2]
    X_clf = np.vstack([X_c0, X_c1])
    y_clf = np.array([0]*50 + [1]*50)

    log_reg = LogisticRegression()
    log_reg.fit(X_clf, y_clf)
    print(f"Logistic Regression Accuracy: {log_reg.score(X_clf, y_clf):.4f}")

    # Test metrics
    y_true = np.array([0,0,0,1,1,1,1,1])
    y_pred = np.array([0,0,1,1,1,0,1,1])
    print(f"Precision: {precision(y_true, y_pred):.4f}")
    print(f"Recall: {recall(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")

    # Test CV
    cv = cross_validate(LogisticRegression, X_clf, y_clf, k=5, lr=0.1, n_iter=500)
    print(f"CV Score: {cv['mean']:.4f} ± {cv['std']:.4f}")

    print("\n✓ Todos los tests pasaron!")
```

---

## 📝 Derivación Analítica: El Entregable de Lápiz y Papel (v3.2)

> 🎓 **Simulación de Examen:** En la maestría te pedirán: *"Derive la regla de actualización de pesos para Logistic Regression"*. Debes poder hacerlo a mano.

### Derivación del Gradiente de Logistic Regression

**Objetivo:** Derivar $\frac{\partial L}{\partial w}$ para la función de costo Cross-Entropy.

#### Paso 1: Definir la Función de Costo

$$L(w) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

Donde:
- $\hat{y}_i = \sigma(w^T x_i) = \frac{1}{1 + e^{-w^T x_i}}$
- $\sigma(z)$ es la función sigmoid

#### Paso 2: Derivar la Sigmoid

$$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$

**Demostración:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
$$\frac{d\sigma}{dz} = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)(1 - \sigma(z))$$

#### Paso 3: Aplicar la Regla de la Cadena

Para un solo ejemplo $(x_i, y_i)$:

$$\frac{\partial L_i}{\partial w} = \frac{\partial L_i}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial w}$$

Donde $z_i = w^T x_i$

**Calcular cada término:**

1. $\frac{\partial L_i}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i} + \frac{1 - y_i}{1 - \hat{y}_i}$

2. $\frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i(1 - \hat{y}_i)$

3. $\frac{\partial z_i}{\partial w} = x_i$

#### Paso 4: Simplificar

$$\frac{\partial L_i}{\partial w} = \left( -\frac{y_i}{\hat{y}_i} + \frac{1 - y_i}{1 - \hat{y}_i} \right) \cdot \hat{y}_i(1 - \hat{y}_i) \cdot x_i$$

Simplificando el término entre paréntesis:
$$= \frac{-y_i(1 - \hat{y}_i) + (1-y_i)\hat{y}_i}{\hat{y}_i(1 - \hat{y}_i)} \cdot \hat{y}_i(1 - \hat{y}_i) \cdot x_i$$
$$= (-y_i + y_i\hat{y}_i + \hat{y}_i - y_i\hat{y}_i) \cdot x_i$$
$$= (\hat{y}_i - y_i) \cdot x_i$$

#### Resultado Final

$$\boxed{\frac{\partial L}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) x_i = \frac{1}{n} X^T (\hat{y} - y)}$$

**Forma vectorizada (para código):**
```python
gradient = (1/n) * X.T @ (y_pred - y_true)
```

### Tu Entregable

Escribe en un documento (Markdown o LaTeX):
1. La derivación completa del gradiente de Cross-Entropy
2. La derivación de la regla de actualización: $w \leftarrow w - \alpha \nabla L$
3. Por qué el gradiente tiene la forma $(\hat{y} - y)$ (interpretación geométrica)

---

## 🎯 El Reto del Tablero Blanco (Metodología Feynman)

Explica en **máximo 5 líneas** sin jerga técnica:

1. **¿Por qué usamos sigmoid en clasificación?**
   > Pista: Piensa en probabilidades entre 0 y 1.

2. **¿Por qué Cross-Entropy y no MSE para clasificación?**
   > Pista: Piensa en qué pasa cuando $\hat{y} \approx 0$ pero $y = 1$.

3. **¿Qué significa "One-vs-All"?**
   > Pista: Piensa en cómo clasificar 10 dígitos con clasificadores binarios.

---

## 🔍 Shadow Mode: Validación con sklearn (v3.3)

> ⚠️ **Regla:** sklearn está **prohibido para aprender**, pero es **necesario para validar**. Si tu implementación difiere significativamente de sklearn, tienes un bug.

### Protocolo de Validación (Viernes de Fase 2)

```python
"""
Shadow Mode - Validación de Implementaciones
Compara tu código desde cero vs sklearn para detectar bugs.

Regla: Si la diferencia de accuracy es >5%, revisar matemáticas.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.linear_model import LinearRegression as SklearnLinReg
from sklearn.metrics import accuracy_score, mean_squared_error

# Importar tu implementación
# from src.logistic_regression import LogisticRegression as MyLR
# from src.linear_regression import LinearRegression as MyLinReg


def shadow_mode_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Compara tu Logistic Regression vs sklearn.

    Los coeficientes y accuracy deben ser casi idénticos.
    """
    print("=" * 60)
    print("SHADOW MODE: Logistic Regression")
    print("=" * 60)

    # ========== TU IMPLEMENTACIÓN ==========
    # my_model = MyLR()
    # my_model.fit(X_train, y_train, lr=0.1, n_iter=1000)
    # my_pred = my_model.predict(X_test)
    # my_acc = accuracy_score(y_test, my_pred)
    # my_weights = my_model.weights

    # Placeholder (reemplazar con tu código)
    my_acc = 0.85
    my_weights = np.zeros(X_train.shape[1])

    # ========== SKLEARN (GROUND TRUTH) ==========
    sklearn_model = SklearnLR(max_iter=1000, solver='lbfgs')
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    sklearn_weights = sklearn_model.coef_.flatten()

    # ========== COMPARACIÓN ==========
    acc_diff = abs(my_acc - sklearn_acc)
    weight_diff = np.linalg.norm(my_weights - sklearn_weights[:len(my_weights)])

    print(f"\n📊 RESULTADOS:")
    print(f"  Tu Accuracy:     {my_acc:.4f}")
    print(f"  sklearn Accuracy: {sklearn_acc:.4f}")
    print(f"  Diferencia:       {acc_diff:.4f}")

    print(f"\n📐 PESOS:")
    print(f"  Diferencia L2 de pesos: {weight_diff:.4f}")

    # Veredicto
    print("\n" + "-" * 60)
    if acc_diff < 0.05:
        print("✓ PASSED: Tu implementación es correcta")
        return True
    else:
        print("✗ FAILED: Diferencia significativa - revisa tu matemática")
        print("  Posibles causas:")
        print("  - Gradiente mal calculado")
        print("  - Learning rate muy alto/bajo")
        print("  - Falta de normalización de datos")
        return False


def shadow_mode_linear_regression(X_train, y_train, X_test, y_test):
    """
    Compara tu Linear Regression vs sklearn.
    """
    print("=" * 60)
    print("SHADOW MODE: Linear Regression")
    print("=" * 60)

    # ========== TU IMPLEMENTACIÓN ==========
    # my_model = MyLinReg()
    # my_model.fit(X_train, y_train)
    # my_pred = my_model.predict(X_test)
    # my_mse = mean_squared_error(y_test, my_pred)

    # Placeholder
    my_mse = 0.5

    # ========== SKLEARN ==========
    sklearn_model = SklearnLinReg()
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)

    # ========== COMPARACIÓN ==========
    mse_ratio = my_mse / sklearn_mse if sklearn_mse > 0 else float('inf')

    print(f"\n📊 RESULTADOS:")
    print(f"  Tu MSE:     {my_mse:.4f}")
    print(f"  sklearn MSE: {sklearn_mse:.4f}")
    print(f"  Ratio:       {mse_ratio:.2f}x")

    print("\n" + "-" * 60)
    if mse_ratio < 1.1:  # Dentro del 10%
        print("✓ PASSED: Tu implementación es correcta")
        return True
    else:
        print("✗ FAILED: Tu MSE es significativamente mayor")
        return False


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    # Dataset de clasificación
    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=42
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    # Dataset de regresión
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=10, noise=10, random_state=42
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Ejecutar Shadow Mode
    shadow_mode_logistic_regression(X_train_c, y_train_c, X_test_c, y_test_c)
    print("\n")
    shadow_mode_linear_regression(X_train_r, y_train_r, X_test_r, y_test_r)
```

### Checklist Shadow Mode

| Día | Algoritmo | Validar |
|-----|-----------|---------|
| Viernes Sem 10 | Linear Regression | MSE ≈ sklearn |
| Viernes Sem 11 | Logistic Regression | Accuracy ≈ sklearn |
| Viernes Sem 12 | Métricas | Precision/Recall = sklearn |

---

## ✅ Checklist de Finalización (v3.3)

### Conocimiento
- [ ] Implementé regresión lineal con Normal Equation y GD
- [ ] Entiendo MSE y su gradiente
- [ ] Implementé regresión logística desde cero
- [ ] Entiendo sigmoid y binary cross-entropy
- [ ] Puedo calcular TP, TN, FP, FN de una matriz de confusión
- [ ] Implementé accuracy, precision, recall, F1
- [ ] Implementé train/test split
- [ ] Implementé K-fold cross validation
- [ ] Entiendo regularización L1 vs L2

### Shadow Mode (v3.3 - Obligatorio)
- [ ] **Linear Regression**: Mi MSE ≈ sklearn (ratio < 1.1)
- [ ] **Logistic Regression**: Mi Accuracy ≈ sklearn (diff < 5%)

### Entregables de Código
- [ ] `logistic_regression.py` con tests pasando
- [ ] `mypy src/` pasa sin errores
- [ ] `pytest tests/` pasa sin errores

### Derivación Analítica (Obligatorio)
- [ ] Derivé el gradiente de Cross-Entropy a mano
- [ ] Documento con derivación completa (Markdown o LaTeX)
- [ ] Puedo explicar por qué $\nabla L = X^T(\hat{y} - y)$

### Metodología Feynman
- [ ] Puedo explicar sigmoid en 5 líneas sin jerga
- [ ] Puedo explicar Cross-Entropy vs MSE en 5 líneas
- [ ] Puedo explicar One-vs-All en 5 líneas

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [04_PROBABILIDAD_ML](04_PROBABILIDAD_ML.md) | [00_INDICE](00_INDICE.md) | [06_UNSUPERVISED_LEARNING](06_UNSUPERVISED_LEARNING.md) |
