# Módulo 07 - Machine Learning Supervisado

> **🎯 Objetivo:** Dominar algoritmos de aprendizaje supervisado  
> **⭐ PATHWAY LÍNEA 1:** Introduction to Machine Learning: Supervised Learning

---

## 🧠 Analogía: Aprender de Ejemplos con Respuestas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   APRENDIZAJE SUPERVISADO = Un Maestro con Respuestas                       │
│   ────────────────────────────────────────────────────                      │
│                                                                             │
│   ENTRADA (X)         ETIQUETA (y)         MODELO APRENDE                   │
│   ──────────────      ────────────         ──────────────                   │
│   Foto de gato   →    "gato"          →    Patrones de gatos                │
│   Email spam     →    "spam"          →    Palabras sospechosas             │
│   Precio casa    →    $500,000        →    Relación features/precio         │
│                                                                             │
│   OBJETIVO:                                                                 │
│   Encontrar función f(X) ≈ y que GENERALICE a datos nuevos                  │
│                                                                             │
│   DOS TIPOS:                                                                │
│   ┌─────────────────┐        ┌─────────────────┐                            │
│   │  REGRESIÓN      │        │  CLASIFICACIÓN  │                            │
│   │  y ∈ ℝ (número) │        │  y ∈ {0,1,...}  │                            │
│   │  Ej: precio     │        │  Ej: spam/no    │                            │
│   └─────────────────┘        └─────────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Fundamentos del Aprendizaje Supervisado](#1-fundamentos)
2. [Regresión Lineal y Logística](#2-regresion)
3. [Árboles de Decisión](#3-arboles)
4. [K-Nearest Neighbors](#4-knn)
5. [Support Vector Machines](#5-svm)
6. [Evaluación de Modelos](#6-evaluacion)

---

## 1. Fundamentos del Aprendizaje Supervisado {#1-fundamentos}

### 1.1 El Pipeline de ML

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE ML SUPERVISADO                           │
│                                                                             │
│   DATOS ──▶ PREPROCESO ──▶ SPLIT ──▶ TRAIN ──▶ EVALUATE ──▶ DEPLOY       │
│                                                                             │
│   1. Recolectar datos (X, y)                                                │
│   2. Limpiar, normalizar, codificar                                         │
│   3. Dividir en train/validation/test (70/15/15)                            │
│   4. Entrenar modelo en train set                                           │
│   5. Evaluar en validation, ajustar hiperparámetros                         │
│   6. Evaluación final en test set                                           │
│   7. Desplegar si el rendimiento es satisfactorio                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Bias-Variance Tradeoff ⭐ FUNDAMENTAL

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ERROR = BIAS² + VARIANCE + RUIDO IRREDUCIBLE                              │
│                                                                             │
│   BIAS (Sesgo):                                                             │
│   ─────────────                                                             │
│   Error por suposiciones incorrectas del modelo.                            │
│   Modelo muy simple → Alto bias → UNDERFITTING                              │
│                                                                             │
│   VARIANCE (Varianza):                                                      │
│   ─────────────────────                                                     │
│   Sensibilidad a fluctuaciones en datos de entrenamiento.                   │
│   Modelo muy complejo → Alta varianza → OVERFITTING                         │
│                                                                             │
│        Error                                                                │
│          ▲                                                                  │
│          │    \                    /                                        │
│          │     \     Total       /                                          │
│          │      \   Error      /                                            │
│          │       \    ▼      /                                              │
│          │        ╲────────╱                                                │
│          │   Bias  ╲      ╱  Variance                                       │
│          │    ▼     ╲    ╱     ▼                                            │
│          │───────────╲──╱────────────▶ Complejidad                          │
│          │            \/                                                    │
│          │         Óptimo                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Implementación de Train/Test Split

```python
from typing import List, Tuple, TypeVar
import random

T = TypeVar('T')

def train_test_split(
    X: List[T], 
    y: List[T], 
    test_size: float = 0.2,
    random_state: int = None
) -> Tuple[List[T], List[T], List[T], List[T]]:
    """Split data into training and test sets.
    
    Args:
        X: Features
        y: Labels
        test_size: Fraction for test set
        random_state: Seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        random.seed(random_state)
    
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)
    
    n_test = int(n * test_size)
    
    test_indices = set(indices[:n_test])
    
    X_train = [X[i] for i in range(n) if i not in test_indices]
    X_test = [X[i] for i in range(n) if i in test_indices]
    y_train = [y[i] for i in range(n) if i not in test_indices]
    y_test = [y[i] for i in range(n) if i in test_indices]
    
    return X_train, X_test, y_train, y_test


def k_fold_split(
    n_samples: int, 
    k: int = 5
) -> List[Tuple[List[int], List[int]]]:
    """Generate K-fold cross-validation indices.
    
    Returns list of (train_indices, val_indices) for each fold.
    """
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    fold_size = n_samples // k
    folds = []
    
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n_samples
        
        val_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        
        folds.append((train_indices, val_indices))
    
    return folds
```

---

## 2. Regresión Lineal y Logística {#2-regresion}

### 2.1 Regresión Lineal desde Cero

```python
import math

class LinearRegression:
    """Linear regression using gradient descent.
    
    Model: y = Xw + b
    Loss: MSE = (1/n) Σ(yᵢ - ŷᵢ)²
    
    Gradient:
        ∂L/∂w = -(2/n) Xᵀ(y - ŷ)
        ∂L/∂b = -(2/n) Σ(y - ŷ)
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.loss_history: List[float] = []
    
    def fit(self, X: List[List[float]], y: List[float]) -> 'LinearRegression':
        """Train the model using gradient descent."""
        n_samples = len(X)
        n_features = len(X[0])
        
        # Initialize weights
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        for _ in range(self.n_iter):
            # Predictions
            y_pred = self._predict(X)
            
            # Calculate gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                for j in range(n_features):
                    dw[j] += (2 / n_samples) * error * X[i][j]
                db += (2 / n_samples) * error
            
            # Update weights
            for j in range(n_features):
                self.weights[j] -= self.lr * dw[j]
            self.bias -= self.lr * db
            
            # Track loss
            loss = self._mse(y, y_pred)
            self.loss_history.append(loss)
        
        return self
    
    def _predict(self, X: List[List[float]]) -> List[float]:
        """Make predictions."""
        predictions = []
        for xi in X:
            pred = self.bias + sum(w * x for w, x in zip(self.weights, xi))
            predictions.append(pred)
        return predictions
    
    def predict(self, X: List[List[float]]) -> List[float]:
        """Public prediction method."""
        return self._predict(X)
    
    def _mse(self, y_true: List[float], y_pred: List[float]) -> float:
        """Mean Squared Error."""
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
```

### 2.2 Regresión Logística desde Cero ⭐

```python
class LogisticRegression:
    """Logistic regression for binary classification.
    
    Model: P(y=1|x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx + b)))
    
    Loss: Binary Cross-Entropy
        L = -(1/n) Σ [yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]
    
    Gradient:
        ∂L/∂w = (1/n) Xᵀ(p̂ - y)
        ∂L/∂b = (1/n) Σ(p̂ - y)
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights: List[float] = []
        self.bias: float = 0.0
    
    def _sigmoid(self, z: float) -> float:
        """Sigmoid activation function."""
        if z < -500:
            return 0.0
        elif z > 500:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))
    
    def fit(self, X: List[List[float]], y: List[int]) -> 'LogisticRegression':
        """Train using gradient descent."""
        n_samples = len(X)
        n_features = len(X[0])
        
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        for _ in range(self.n_iter):
            # Forward pass
            linear = [
                self.bias + sum(w * x for w, x in zip(self.weights, xi))
                for xi in X
            ]
            predictions = [self._sigmoid(z) for z in linear]
            
            # Gradients
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = predictions[i] - y[i]
                for j in range(n_features):
                    dw[j] += (1 / n_samples) * error * X[i][j]
                db += (1 / n_samples) * error
            
            # Update
            for j in range(n_features):
                self.weights[j] -= self.lr * dw[j]
            self.bias -= self.lr * db
        
        return self
    
    def predict_proba(self, X: List[List[float]]) -> List[float]:
        """Predict probabilities."""
        return [
            self._sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, xi)))
            for xi in X
        ]
    
    def predict(self, X: List[List[float]], threshold: float = 0.5) -> List[int]:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]
```

### 2.3 Regularización (L1/L2)

```python
class RidgeRegression(LinearRegression):
    """Linear regression with L2 regularization.
    
    Loss = MSE + λ Σ wⱼ²
    
    L2 shrinks weights but doesn't set them to zero.
    Equivalent to MAP estimation with Gaussian prior.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 alpha: float = 1.0):
        super().__init__(learning_rate, n_iterations)
        self.alpha = alpha  # Regularization strength
    
    def fit(self, X: List[List[float]], y: List[float]) -> 'RidgeRegression':
        """Train with L2 regularization."""
        n_samples = len(X)
        n_features = len(X[0])
        
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        for _ in range(self.n_iter):
            y_pred = self._predict(X)
            
            # Gradients with regularization
            dw = [0.0] * n_features
            db = 0.0
            
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                for j in range(n_features):
                    dw[j] += (2 / n_samples) * error * X[i][j]
                db += (2 / n_samples) * error
            
            # Add L2 penalty gradient
            for j in range(n_features):
                dw[j] += 2 * self.alpha * self.weights[j]
            
            # Update
            for j in range(n_features):
                self.weights[j] -= self.lr * dw[j]
            self.bias -= self.lr * db
        
        return self
```

---

## 3. Árboles de Decisión {#3-arboles}

### 3.1 Concepto y Criterios de Split

```
ÁRBOL DE DECISIÓN:
──────────────────

                    [Feature₁ < 5?]
                    /              \
                  Yes              No
                 /                   \
        [Feature₂ < 3?]           [Clase: B]
         /          \
       Yes          No
       /              \
   [Clase: A]     [Clase: C]

CRITERIOS DE SPLIT:
• GINI IMPURITY (Clasificación):
  Gini(S) = 1 - Σ pᵢ²
  Mide "impureza" de un nodo

• INFORMATION GAIN (Clasificación):
  IG = Entropy(parent) - Σ (nⱼ/n) × Entropy(childⱼ)
  Entropy(S) = -Σ pᵢ log₂(pᵢ)

• MSE (Regresión):
  Split que minimiza varianza en nodos hijos
```

### 3.2 Implementación Simplificada

```python
from typing import Dict, Any, Optional
from collections import Counter

class DecisionTreeClassifier:
    """Simple decision tree for classification.
    
    Uses Gini impurity for splitting.
    """
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples = min_samples_split
        self.tree: Optional[Dict[str, Any]] = None
    
    def _gini(self, y: List[int]) -> float:
        """Calculate Gini impurity."""
        if not y:
            return 0.0
        counts = Counter(y)
        n = len(y)
        return 1.0 - sum((count / n) ** 2 for count in counts.values())
    
    def _best_split(
        self, 
        X: List[List[float]], 
        y: List[int]
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find best feature and threshold to split on."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = len(X[0])
        parent_gini = self._gini(y)
        n = len(y)
        
        for feature_idx in range(n_features):
            # Get unique values for thresholds
            values = sorted(set(x[feature_idx] for x in X))
            thresholds = [(values[i] + values[i+1]) / 2 
                          for i in range(len(values) - 1)]
            
            for threshold in thresholds:
                # Split data
                left_y = [y[i] for i in range(n) if X[i][feature_idx] <= threshold]
                right_y = [y[i] for i in range(n) if X[i][feature_idx] > threshold]
                
                if not left_y or not right_y:
                    continue
                
                # Calculate information gain
                left_gini = self._gini(left_y)
                right_gini = self._gini(right_y)
                weighted_gini = (len(left_y) * left_gini + 
                                 len(right_y) * right_gini) / n
                gain = parent_gini - weighted_gini
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(
        self, 
        X: List[List[float]], 
        y: List[int], 
        depth: int
    ) -> Dict[str, Any]:
        """Recursively build the tree."""
        n_samples = len(y)
        n_classes = len(set(y))
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples or 
            n_classes == 1):
            # Create leaf node
            return {"leaf": True, "class": Counter(y).most_common(1)[0][0]}
        
        # Find best split
        feature, threshold, gain = self._best_split(X, y)
        
        if feature is None or gain <= 0:
            return {"leaf": True, "class": Counter(y).most_common(1)[0][0]}
        
        # Split data
        left_indices = [i for i in range(n_samples) if X[i][feature] <= threshold]
        right_indices = [i for i in range(n_samples) if X[i][feature] > threshold]
        
        left_X = [X[i] for i in left_indices]
        left_y = [y[i] for i in left_indices]
        right_X = [X[i] for i in right_indices]
        right_y = [y[i] for i in right_indices]
        
        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1)
        }
    
    def fit(self, X: List[List[float]], y: List[int]) -> 'DecisionTreeClassifier':
        """Build the decision tree."""
        self.tree = self._build_tree(X, y, 0)
        return self
    
    def _predict_one(self, x: List[float], node: Dict[str, Any]) -> int:
        """Predict class for single sample."""
        if node["leaf"]:
            return node["class"]
        
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict classes for samples."""
        return [self._predict_one(x, self.tree) for x in X]
```

---

## 4. K-Nearest Neighbors {#4-knn}

### 4.1 Implementación

```python
class KNearestNeighbors:
    """K-Nearest Neighbors classifier.
    
    Non-parametric: no training, just stores data.
    Prediction: vote of k nearest neighbors.
    
    Time: O(n × d) per prediction (n samples, d dimensions)
    """
    
    def __init__(self, k: int = 3):
        self.k = k
        self.X_train: List[List[float]] = []
        self.y_train: List[int] = []
    
    def fit(self, X: List[List[float]], y: List[int]) -> 'KNearestNeighbors':
        """Store training data."""
        self.X_train = X
        self.y_train = y
        return self
    
    def _euclidean_distance(self, x1: List[float], x2: List[float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
    
    def _predict_one(self, x: List[float]) -> int:
        """Predict class for single sample."""
        # Calculate distances to all training samples
        distances = [
            (self._euclidean_distance(x, x_train), y_train)
            for x_train, y_train in zip(self.X_train, self.y_train)
        ]
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda d: d[0])
        k_nearest = distances[:self.k]
        
        # Vote
        votes = Counter(label for _, label in k_nearest)
        return votes.most_common(1)[0][0]
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict classes."""
        return [self._predict_one(x) for x in X]
```

---

## 5. Support Vector Machines {#5-svm}

### 5.1 Concepto

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   SUPPORT VECTOR MACHINES                                                   │
│   ───────────────────────                                                   │
│                                                                             │
│   Objetivo: Encontrar el HIPERPLANO que separa las clases                   │
│             con el MÁXIMO MARGEN.                                           │
│                                                                             │
│        ○  ○                                                                 │
│     ○        ○                                                              │
│        ○                     ← Support Vectors (los más cercanos)           │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─        ← Margen                                       │
│   ═══════════════════════    ← Hiperplano separador (wᵀx + b = 0)           │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─        ← Margen                                       │
│        ●                     ← Support Vectors                              │
│     ●        ●                                                              │
│        ●  ●                                                                 │
│                                                                             │
│   MARGEN = 2 / ||w||                                                        │
│   Maximizar margen = Minimizar ||w||²                                       │
│                                                                             │
│   KERNEL TRICK:                                                             │
│   Para datos no linealmente separables, proyectar a dimensión superior      │
│   donde SÍ sean separables (sin calcular la proyección explícitamente).     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Implementación Simplificada (Lineal)

```python
class LinearSVM:
    """Linear SVM using gradient descent (simplified).
    
    Minimizes: (1/2)||w||² + C × Σ max(0, 1 - yᵢ(wᵀxᵢ + b))
    
    Hinge loss for classification.
    """
    
    def __init__(self, learning_rate: float = 0.001, 
                 n_iterations: int = 1000, C: float = 1.0):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.C = C  # Regularization parameter
        self.weights: List[float] = []
        self.bias: float = 0.0
    
    def fit(self, X: List[List[float]], y: List[int]) -> 'LinearSVM':
        """Train SVM. Labels should be -1 or 1."""
        # Convert 0/1 to -1/1 if needed
        y = [1 if label == 1 else -1 for label in y]
        
        n_samples = len(X)
        n_features = len(X[0])
        
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        for _ in range(self.n_iter):
            for i in range(n_samples):
                # Check if sample satisfies margin constraint
                margin = y[i] * (sum(w * x for w, x in zip(self.weights, X[i])) + self.bias)
                
                if margin >= 1:
                    # Correctly classified with margin
                    # Only regularization gradient
                    for j in range(n_features):
                        self.weights[j] -= self.lr * self.weights[j]
                else:
                    # Misclassified or within margin
                    # Regularization + hinge loss gradient
                    for j in range(n_features):
                        self.weights[j] -= self.lr * (self.weights[j] - 
                                                       self.C * y[i] * X[i][j])
                    self.bias -= self.lr * (-self.C * y[i])
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict class labels (0 or 1)."""
        predictions = []
        for x in X:
            score = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
            predictions.append(1 if score >= 0 else 0)
        return predictions
```

---

## 6. Evaluación de Modelos {#6-evaluacion}

### 6.1 Métricas de Clasificación

```python
def confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
    """Calculate confusion matrix components.
    
    Returns:
        TP, TN, FP, FN counts
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Accuracy = (TP + TN) / Total."""
    cm = confusion_matrix(y_true, y_pred)
    total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
    return (cm["TP"] + cm["TN"]) / total if total > 0 else 0.0


def precision(y_true: List[int], y_pred: List[int]) -> float:
    """Precision = TP / (TP + FP).
    
    Of all predicted positive, how many are actually positive?
    """
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["TP"] + cm["FP"]
    return cm["TP"] / denom if denom > 0 else 0.0


def recall(y_true: List[int], y_pred: List[int]) -> float:
    """Recall = TP / (TP + FN).
    
    Of all actual positive, how many did we find?
    Also called Sensitivity or True Positive Rate.
    """
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["TP"] + cm["FN"]
    return cm["TP"] / denom if denom > 0 else 0.0


def f1_score(y_true: List[int], y_pred: List[int]) -> float:
    """F1 = 2 × (Precision × Recall) / (Precision + Recall).
    
    Harmonic mean of precision and recall.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
```

### 6.2 Métricas de Regresión

```python
def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """MSE = (1/n) Σ(yᵢ - ŷᵢ)²."""
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """RMSE = √MSE."""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """MAE = (1/n) Σ|yᵢ - ŷᵢ|."""
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)


def r_squared(y_true: List[float], y_pred: List[float]) -> float:
    """R² = 1 - SS_res / SS_tot.
    
    Proportion of variance explained.
    """
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((y - mean_y) ** 2 for y in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
```

### 6.3 Cross-Validation

```python
def cross_validate(
    model_class,
    X: List[List[float]], 
    y: List, 
    k: int = 5,
    **model_params
) -> List[float]:
    """K-fold cross-validation.
    
    Returns accuracy for each fold.
    """
    folds = k_fold_split(len(X), k)
    scores = []
    
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        score = accuracy(y_val, y_pred)
        scores.append(score)
    
    return scores
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 22.1: Regresión Lineal
Implementar y entrenar regresión lineal en datos sintéticos.

### Ejercicio 22.2: Clasificador de Spam
Usar Logistic Regression para clasificar emails.

### Ejercicio 22.3: Comparar Modelos
Evaluar Decision Tree vs KNN vs Logistic Regression con cross-validation.

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning) | Curso | 🔴 Obligatorio |
| [Hands-On ML Book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) | Libro | 🔴 Obligatorio |
| [StatQuest: ML](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) | Videos | 🟡 Recomendado |

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [21_CADENAS_MARKOV_MONTECARLO](21_CADENAS_MARKOV_MONTECARLO.md) | [00_INDICE](00_INDICE.md) | [23_ML_NO_SUPERVISADO](23_ML_NO_SUPERVISADO.md) |
