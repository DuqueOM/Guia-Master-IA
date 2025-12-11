# Módulo 08 - Proyecto Final: MNIST Analyst

> **🎯 Objetivo:** Pipeline end-to-end que demuestra competencia en las 3 áreas del Pathway
> **Fase:** 3 - Proyecto Integrador | **Semanas 21-24** (4 semanas)
> **Dataset:** MNIST (dígitos escritos a mano, 28×28 píxeles)

---

## 🧠 ¿Qué Estamos Construyendo?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PROYECTO: END-TO-END HANDWRITTEN DIGIT ANALYSIS PIPELINE                  │
│   ─────────────────────────────────────────────────────────                 │
│                                                                             │
│   LÍNEA 1: MACHINE LEARNING (3 créditos) - DEMOSTRADO EN 4 SEMANAS          │
│   ├── Semana 21: EDA + PCA + K-Means                                        │
│   ├── Semana 22: Logistic Regression One-vs-All                             │
│   ├── Semana 23: MLP con Backprop                                           │
│   └── Semana 24: Comparación de Modelos + Informe                           │
│                                                                             │
│   RESULTADO:                                                                │
│   Un pipeline que analiza, agrupa y clasifica dígitos MNIST                 │
│   usando algoritmos implementados 100% desde cero.                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

> 💡 **Nota v3.1:** MNIST es un dataset simple. 4 semanas son suficientes para un proyecto bien estructurado.

---

## 📚 Estructura del Proyecto

### Cronograma (4 Semanas)

| Semana | Fase | Materia Demostrada | Entregable |
|--------|------|-------------------|------------|
| 21 | EDA + No Supervisado | Unsupervised Algorithms | PCA + K-Means funcionando |
| 22 | Clasificación Clásica | Supervised Learning | Logistic Regression OvA |
| 23 | Deep Learning | Introduction to Deep Learning | MLP con backprop |
| 24 | Benchmark + Informe | Integración | MODEL_COMPARISON.md |

### Estructura de Archivos

```
mnist-analyst/
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Cargar y preprocesar MNIST (Módulo 01)
│   ├── linear_algebra.py      # Operaciones vectoriales (Módulo 02)
│   ├── pca.py                 # PCA desde cero (Módulo 06)
│   ├── kmeans.py              # K-Means desde cero (Módulo 06)
│   ├── logistic_regression.py # Logistic multiclase (Módulo 05)
│   ├── neural_network.py      # MLP con backprop (Módulo 07)
│   ├── metrics.py             # Métricas de evaluación (Módulo 05)
│   └── pipeline.py            # Pipeline integrado
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pca_visualization.ipynb
│   ├── 03_kmeans_clustering.ipynb
│   ├── 04_logistic_classification.ipynb
│   └── 05_neural_network_benchmark.ipynb
│
├── tests/
│   └── test_*.py
│
├── docs/
│   └── MODEL_COMPARISON.md
│
├── README.md
└── requirements.txt
```

---

## 💻 Parte 1: Cargar MNIST

### 1.1 Data Loader

```python
"""
MNIST Dataset Loader

MNIST contiene:
- 60,000 imágenes de entrenamiento
- 10,000 imágenes de test
- Cada imagen: 28x28 píxeles grayscale (0-255)
- 10 clases: dígitos 0-9

Formato aplanado: cada imagen es un vector de 784 dimensiones
"""

import numpy as np
import struct
import gzip
from pathlib import Path
from typing import Tuple


def load_mnist_images(filepath: str) -> np.ndarray:
    """
    Carga imágenes MNIST desde archivo IDX.

    Formato IDX:
    - 4 bytes: magic number
    - 4 bytes: número de imágenes
    - 4 bytes: número de filas
    - 4 bytes: número de columnas
    - resto: píxeles (unsigned bytes)
    """
    with gzip.open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)
    return images


def load_mnist_labels(filepath: str) -> np.ndarray:
    """Carga etiquetas MNIST."""
    with gzip.open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(data_dir: str = 'data/mnist') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga dataset MNIST completo.

    Returns:
        X_train: (60000, 784)
        y_train: (60000,)
        X_test: (10000, 784)
        y_test: (10000,)
    """
    data_dir = Path(data_dir)

    X_train = load_mnist_images(data_dir / 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(data_dir / 'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(data_dir / 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(data_dir / 't10k-labels-idx1-ubyte.gz')

    return X_train, y_train, X_test, y_test


def normalize_data(X: np.ndarray) -> np.ndarray:
    """Normaliza píxeles a rango [0, 1]."""
    return X.astype(np.float64) / 255.0


def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convierte etiquetas a one-hot encoding."""
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


# Alternativa: generar datos sintéticos si no tienes MNIST
def generate_synthetic_mnist(n_samples: int = 1000, seed: int = 42) -> Tuple:
    """
    Genera datos sintéticos similares a MNIST para pruebas.
    """
    np.random.seed(seed)

    X = np.random.rand(n_samples, 784)  # Imágenes aleatorias
    y = np.random.randint(0, 10, n_samples)  # Etiquetas aleatorias

    # Split 80/20
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]
```

### 1.2 Visualización

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_digits(X: np.ndarray, y: np.ndarray, n_samples: int = 25):
    """Visualiza una cuadrícula de dígitos."""
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2*n_rows))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n_samples:
            img = X[i].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Label: {y[i]}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_digit_single(x: np.ndarray, title: str = ''):
    """Visualiza un solo dígito."""
    plt.figure(figsize=(4, 4))
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
```

---

## 💻 Parte 2: Exploración No Supervisada (Semanas 21-22)

### 2.1 PCA para Visualización

```python
"""
SEMANA 21: PCA en MNIST

Objetivo: Reducir de 784 dimensiones a 2-3 para visualización.

Preguntas a responder:
1. ¿Cuánta varianza se retiene con pocos componentes?
2. ¿Se separan visualmente las clases en 2D?
3. ¿Qué "aprenden" las componentes principales?
"""

import numpy as np
from typing import Tuple

class PCA:
    """PCA implementado desde cero (del Módulo 05)."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> 'PCA':
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # SVD (más estable que eigendecomposition)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        self.components_ = Vt[:self.n_components].T
        variance = (S ** 2) / (len(X) - 1)
        self.explained_variance_ratio_ = variance[:self.n_components] / np.sum(variance)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        return X_pca @ self.components_.T + self.mean_


def analyze_pca_mnist(X: np.ndarray, y: np.ndarray):
    """Análisis PCA completo de MNIST."""

    # 1. PCA con diferentes números de componentes
    print("=== Análisis de Varianza Explicada ===")
    pca_full = PCA(n_components=min(50, X.shape[1]))
    pca_full.fit(X)

    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)

    for n in [2, 10, 50]:
        if n <= len(cumulative_var):
            print(f"  {n} componentes: {cumulative_var[n-1]:.2%} varianza")

    # 2. Visualización 2D
    print("\n=== Proyección 2D ===")
    pca_2d = PCA(n_components=2)
    X_2d = pca_2d.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for digit in range(10):
        mask = y == digit
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   alpha=0.5, label=str(digit), s=10)
    plt.legend()
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    plt.title('MNIST en 2D (PCA)')
    plt.show()

    # 3. Visualizar componentes principales
    print("\n=== Componentes Principales como Imágenes ===")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    pca_10 = PCA(n_components=10)
    pca_10.fit(X)

    for i, ax in enumerate(axes.flatten()):
        component = pca_10.components_[:, i].reshape(28, 28)
        ax.imshow(component, cmap='RdBu')
        ax.set_title(f'PC{i+1}')
        ax.axis('off')
    plt.suptitle('Top 10 Componentes Principales')
    plt.tight_layout()
    plt.show()

    return pca_2d, X_2d
```

### 2.2 K-Means Clustering

```python
"""
SEMANA 22: K-Means en MNIST

Objetivo: Agrupar dígitos SIN usar etiquetas.

Preguntas a responder:
1. ¿K-Means encuentra los 10 dígitos?
2. ¿Qué tan puros son los clusters?
3. ¿Cómo se ven los centroides?
"""

import numpy as np

class KMeans:
    """K-Means implementado desde cero (del Módulo 05)."""

    def __init__(self, n_clusters: int = 10, max_iter: int = 100, seed: int = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids_plusplus(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ initialization."""
        if self.seed:
            np.random.seed(self.seed)

        n_samples = len(X)
        centroids = [X[np.random.randint(n_samples)]]

        for _ in range(1, self.n_clusters):
            distances = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
            probs = distances / distances.sum()
            centroids.append(X[np.random.choice(n_samples, p=probs)])

        return np.array(centroids)

    def fit(self, X: np.ndarray) -> 'KMeans':
        self.centroids = self._init_centroids_plusplus(X)

        for _ in range(self.max_iter):
            # Asignar
            distances = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])
            self.labels_ = np.argmin(distances, axis=1)

            # Actualizar
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0)
                                      if np.sum(self.labels_ == k) > 0
                                      else self.centroids[k]
                                      for k in range(self.n_clusters)])

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.inertia_ = sum(np.sum((X[self.labels_ == k] - self.centroids[k])**2)
                           for k in range(self.n_clusters))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)


def analyze_kmeans_mnist(X: np.ndarray, y: np.ndarray):
    """Análisis K-Means de MNIST."""

    print("=== K-Means Clustering ===")
    kmeans = KMeans(n_clusters=10, seed=42)
    kmeans.fit(X)

    # 1. Visualizar centroides
    print("\n=== Centroides (promedio de cada cluster) ===")
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flatten()):
        centroid = kmeans.centroids[i].reshape(28, 28)
        ax.imshow(centroid, cmap='gray')
        ax.set_title(f'Cluster {i}')
        ax.axis('off')
    plt.suptitle('Centroides K-Means')
    plt.tight_layout()
    plt.show()

    # 2. Analizar pureza de clusters
    print("\n=== Pureza de Clusters ===")
    print("Cluster | Dígito Dominante | Pureza")
    print("-" * 40)

    total_correct = 0
    for cluster in range(10):
        cluster_mask = kmeans.labels_ == cluster
        cluster_labels = y[cluster_mask]

        if len(cluster_labels) > 0:
            dominant_digit = np.bincount(cluster_labels).argmax()
            purity = np.sum(cluster_labels == dominant_digit) / len(cluster_labels)
            total_correct += np.sum(cluster_labels == dominant_digit)
            print(f"   {cluster}    |        {dominant_digit}         | {purity:.2%}")

    overall_purity = total_correct / len(y)
    print(f"\nPureza Global: {overall_purity:.2%}")

    return kmeans
```

---

## 💻 Parte 3: Clasificación Supervisada (Semanas 23-24)

### 3.1 Logistic Regression One-vs-All

```python
"""
SEMANAS 23-24: Logistic Regression Multiclase

Estrategia One-vs-All (OvA):
- Entrenar 10 clasificadores binarios
- Cada uno: "¿Es este dígito X o no?"
- Predicción: elegir la clase con mayor probabilidad
"""

import numpy as np
from typing import List

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionBinary:
    """Logistic Regression binario."""

    def __init__(self, lr: float = 0.1, n_iter: int = 100, reg: float = 0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.reg = reg  # L2 regularization
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionBinary':
        n_samples, n_features = X.shape
        self.theta = np.zeros(n_features)

        for _ in range(self.n_iter):
            h = sigmoid(X @ self.theta)
            grad = (1/n_samples) * X.T @ (h - y) + (self.reg/n_samples) * self.theta
            self.theta -= self.lr * grad

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X @ self.theta)


class LogisticRegressionOvA:
    """Logistic Regression One-vs-All para clasificación multiclase."""

    def __init__(self, n_classes: int = 10, lr: float = 0.1, n_iter: int = 100):
        self.n_classes = n_classes
        self.lr = lr
        self.n_iter = n_iter
        self.classifiers: List[LogisticRegressionBinary] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionOvA':
        """Entrena un clasificador por clase."""
        # Añadir bias
        X_b = np.column_stack([np.ones(len(X)), X])

        self.classifiers = []
        for c in range(self.n_classes):
            print(f"  Entrenando clasificador para clase {c}...", end='\r')
            y_binary = (y == c).astype(int)
            clf = LogisticRegressionBinary(self.lr, self.n_iter)
            clf.fit(X_b, y_binary)
            self.classifiers.append(clf)

        print("  Entrenamiento completado.                ")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades para cada clase."""
        X_b = np.column_stack([np.ones(len(X)), X])
        probs = np.column_stack([clf.predict_proba(X_b) for clf in self.classifiers])
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice la clase con mayor probabilidad."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy."""
        return np.mean(self.predict(X) == y)


def train_logistic_mnist(X_train, y_train, X_test, y_test):
    """Entrena y evalúa Logistic Regression en MNIST."""

    print("=== Logistic Regression One-vs-All ===")

    # Entrenar
    lr_model = LogisticRegressionOvA(n_classes=10, lr=0.1, n_iter=200)
    lr_model.fit(X_train, y_train)

    # Evaluar
    train_acc = lr_model.score(X_train, y_train)
    test_acc = lr_model.score(X_test, y_test)

    print(f"\nTrain Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:  {test_acc:.2%}")

    # Métricas detalladas
    y_pred = lr_model.predict(X_test)

    print("\n=== Métricas por Clase ===")
    print("Dígito | Precision | Recall | F1-Score")
    print("-" * 45)

    for digit in range(10):
        tp = np.sum((y_test == digit) & (y_pred == digit))
        fp = np.sum((y_test != digit) & (y_pred == digit))
        fn = np.sum((y_test == digit) & (y_pred != digit))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"   {digit}   |   {precision:.3f}   |  {recall:.3f}  |   {f1:.3f}")

    # Matriz de confusión
    print("\n=== Matriz de Confusión ===")
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_test, y_pred):
        cm[true, pred] += 1

    print("    " + "  ".join(str(i) for i in range(10)))
    for i in range(10):
        print(f"{i}: " + " ".join(f"{cm[i,j]:3d}" for j in range(10)))

    return lr_model
```

---

## 💻 Parte 4: Deep Learning (Semanas 25-26)

### 4.1 MLP para MNIST

```python
"""
SEMANAS 25-26: Neural Network para MNIST

Arquitectura:
- Input: 784 (28x28 píxeles aplanados)
- Hidden 1: 128 neuronas, ReLU
- Hidden 2: 64 neuronas, ReLU
- Output: 10 neuronas, Softmax

Objetivo: Superar a Logistic Regression
"""

import numpy as np
from typing import List, Tuple

# Funciones de activación
def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


class NeuralNetworkMNIST:
    """Red Neuronal optimizada para MNIST."""

    def __init__(self, layer_sizes: List[int] = [784, 128, 64, 10], seed: int = 42):
        """
        Args:
            layer_sizes: [input, hidden1, hidden2, ..., output]
        """
        np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        # Inicializar pesos (He initialization para ReLU)
        self.weights = []
        self.biases = []

        for i in range(self.n_layers - 1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)

        self.cache = {}
        self.loss_history = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.cache['a0'] = x
        a = x

        for i in range(self.n_layers - 2):
            z = self.weights[i] @ a + self.biases[i]
            a = relu(z)
            self.cache[f'z{i+1}'] = z
            self.cache[f'a{i+1}'] = a

        # Última capa: softmax
        z = self.weights[-1] @ a + self.biases[-1]
        a = softmax(z)
        self.cache[f'z{self.n_layers-1}'] = z
        self.cache[f'a{self.n_layers-1}'] = a

        return a

    def backward(self, y_true: np.ndarray) -> Tuple[List, List]:
        """Backward pass."""
        y_pred = self.cache[f'a{self.n_layers-1}']

        # Gradiente de softmax + cross-entropy
        dz = y_pred - y_true

        dW_list = []
        db_list = []

        for i in range(self.n_layers - 2, -1, -1):
            a_prev = self.cache[f'a{i}']

            dW = np.outer(dz, a_prev)
            db = dz

            dW_list.insert(0, dW)
            db_list.insert(0, db)

            if i > 0:
                da_prev = self.weights[i].T @ dz
                z_prev = self.cache[f'z{i}']
                dz = da_prev * relu_deriv(z_prev)

        return dW_list, db_list

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        verbose: bool = True
    ):
        """Entrena la red con mini-batch SGD."""
        n_samples = len(X)

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Acumular gradientes del batch
                dW_accum = [np.zeros_like(w) for w in self.weights]
                db_accum = [np.zeros_like(b) for b in self.biases]

                for x, y_true_label in zip(X_batch, y_batch):
                    # One-hot encode
                    y_one_hot = np.zeros(10)
                    y_one_hot[y_true_label] = 1

                    # Forward
                    y_pred = self.forward(x)

                    # Loss
                    loss = -np.sum(y_one_hot * np.log(np.clip(y_pred, 1e-15, 1)))
                    total_loss += loss

                    # Backward
                    dW_list, db_list = self.backward(y_one_hot)

                    for j in range(len(self.weights)):
                        dW_accum[j] += dW_list[j]
                        db_accum[j] += db_list[j]

                # Update
                batch_len = len(X_batch)
                for j in range(len(self.weights)):
                    self.weights[j] -= learning_rate * dW_accum[j] / batch_len
                    self.biases[j] -= learning_rate * db_accum[j] / batch_len

            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)

            if verbose:
                train_acc = self.score(X[:1000], y[:1000])
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.2%}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clases."""
        return np.array([np.argmax(self.forward(x)) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy."""
        return np.mean(self.predict(X) == y)


def train_neural_network_mnist(X_train, y_train, X_test, y_test):
    """Entrena y evalúa Neural Network en MNIST."""

    print("=== Neural Network (MLP) ===")
    print("Arquitectura: 784 → 128 → 64 → 10")

    nn = NeuralNetworkMNIST([784, 128, 64, 10])
    nn.fit(X_train, y_train, epochs=10, batch_size=32, learning_rate=0.01)

    train_acc = nn.score(X_train, y_train)
    test_acc = nn.score(X_test, y_test)

    print(f"\nTrain Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy:  {test_acc:.2%}")

    return nn
```

---

## 💻 Parte 5: Benchmark y Comparación

### 5.1 Pipeline Completo

```python
"""
Pipeline completo que ejecuta todos los análisis
y compara los modelos.
"""

import numpy as np
import time

def run_mnist_pipeline(X_train, y_train, X_test, y_test, use_subset: bool = True):
    """
    Ejecuta el pipeline completo de MNIST.

    Args:
        use_subset: Si True, usa solo 10k samples para rapidez
    """
    if use_subset:
        X_train = X_train[:10000]
        y_train = y_train[:10000]
        X_test = X_test[:2000]
        y_test = y_test[:2000]

    # Normalizar
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    print("=" * 60)
    print("MNIST ANALYST PIPELINE")
    print("=" * 60)
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print("=" * 60)

    results = {}

    # === FASE 1: Unsupervised ===
    print("\n" + "=" * 60)
    print("FASE 1: EXPLORACIÓN NO SUPERVISADA")
    print("=" * 60)

    # PCA
    print("\n[PCA]")
    pca = PCA(n_components=50)
    pca.fit(X_train)
    print(f"Varianza explicada (50 PCs): {sum(pca.explained_variance_ratio_):.2%}")

    # K-Means
    print("\n[K-Means]")
    start = time.time()
    kmeans = KMeans(n_clusters=10, seed=42)
    kmeans.fit(X_train)
    kmeans_time = time.time() - start
    print(f"Inercia: {kmeans.inertia_:.2f}")
    print(f"Tiempo: {kmeans_time:.2f}s")

    # === FASE 2: Supervised ===
    print("\n" + "=" * 60)
    print("FASE 2: CLASIFICACIÓN SUPERVISADA")
    print("=" * 60)

    # Logistic Regression
    print("\n[Logistic Regression One-vs-All]")
    start = time.time()
    lr_model = LogisticRegressionOvA(n_classes=10, lr=0.1, n_iter=100)
    lr_model.fit(X_train, y_train)
    lr_time = time.time() - start
    lr_acc = lr_model.score(X_test, y_test)
    print(f"Test Accuracy: {lr_acc:.2%}")
    print(f"Tiempo: {lr_time:.2f}s")
    results['Logistic Regression'] = lr_acc

    # === FASE 3: Deep Learning ===
    print("\n" + "=" * 60)
    print("FASE 3: DEEP LEARNING")
    print("=" * 60)

    # Neural Network
    print("\n[Neural Network MLP]")
    start = time.time()
    nn = NeuralNetworkMNIST([784, 128, 64, 10])
    nn.fit(X_train, y_train, epochs=5, batch_size=32, learning_rate=0.01, verbose=False)
    nn_time = time.time() - start
    nn_acc = nn.score(X_test, y_test)
    print(f"Test Accuracy: {nn_acc:.2%}")
    print(f"Tiempo: {nn_time:.2f}s")
    results['Neural Network'] = nn_acc

    # === COMPARACIÓN ===
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 60)

    print("\nModelo               | Accuracy | Mejora vs LR")
    print("-" * 50)
    baseline = results['Logistic Regression']
    for name, acc in results.items():
        improvement = ((acc - baseline) / baseline) * 100 if name != 'Logistic Regression' else 0
        print(f"{name:<20} | {acc:.2%}    | {improvement:+.1f}%")

    # === ANÁLISIS ===
    print("\n" + "=" * 60)
    print("ANÁLISIS: ¿Por qué NN es mejor?")
    print("=" * 60)
    print("""
1. NO-LINEALIDAD: ReLU permite aprender fronteras no lineales.
   Logistic Regression solo puede aprender fronteras lineales.

2. REPRESENTACIÓN JERÁRQUICA: Las capas ocultas aprenden features
   de complejidad creciente (bordes → formas → dígitos).

3. CAPACIDAD: Más parámetros = puede memorizar patrones más complejos.
   Pero cuidado con overfitting si hay pocos datos.

4. COMPOSICIÓN: La red compone funciones simples (lineales + activaciones)
   para aproximar funciones complejas.
""")

    return results
```

---

## 📦 Entregable Final

### `MODEL_COMPARISON.md`

```markdown
# Model Comparison Report - MNIST Analyst

## Executive Summary

This project demonstrates competency in all three courses of the Machine Learning
Pathway (Line 1) through a complete analysis of the MNIST dataset.

## Results

| Model | Test Accuracy | Training Time |
|-------|---------------|---------------|
| K-Means (Unsupervised) | N/A (clustering) | ~5s |
| Logistic Regression | ~85-90% | ~30s |
| Neural Network (MLP) | ~95%+ | ~60s |

## Analysis

### Why does the Neural Network outperform Logistic Regression?

1. **Non-linearity**: ReLU activations allow learning non-linear decision boundaries
2. **Hierarchical features**: Hidden layers learn increasingly abstract representations
3. **Capacity**: More parameters enable capturing complex patterns

### Mathematical Explanation

Logistic Regression:
```
ŷ = σ(Wx + b)
```
- Single linear transformation + sigmoid
- Can only learn linear decision boundaries

Neural Network:
```
ŷ = softmax(W₃ · ReLU(W₂ · ReLU(W₁x + b₁) + b₂) + b₃)
```
- Multiple non-linear transformations
- Universal function approximator

### PCA Insights

- 50 components retain ~85% of variance
- First 2 components show partial class separation
- Principal components capture stroke patterns

### K-Means Insights

- Cluster centroids resemble average digit shapes
- Some digits (1, 7) cluster well; others (4, 9) overlap
- Unsupervised clustering achieves ~60% purity

## Conclusion

The Neural Network achieves the highest accuracy by learning hierarchical,
non-linear representations of the input images. This project demonstrates
practical implementation of Supervised Learning, Unsupervised Learning,
and Deep Learning algorithms from scratch.
```

---

## 📊 Análisis Bias-Variance (v3.2)

> 🎓 **Concepto Central de la Maestría:** Entender el tradeoff Bias-Variance es fundamental para diseñar modelos de ML.

### El Tradeoff Bias-Variance

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ERROR TOTAL = BIAS² + VARIANCE + RUIDO IRREDUCIBLE            │
│                                                                 │
│   BIAS (Sesgo): Error por suposiciones simplificadoras          │
│   - Modelo muy simple → NO captura patrones → UNDERFITTING      │
│                                                                 │
│   VARIANCE (Varianza): Error por sensibilidad a datos           │
│   - Modelo muy complejo → Memoriza ruido → OVERFITTING          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Análisis de los Modelos del Proyecto

| Modelo | Bias | Variance | Comportamiento Esperado |
|--------|------|----------|-------------------------|
| **Logistic Regression** | Alto | Baja | Underfitting (solo límites lineales) |
| **MLP pequeño (128-64)** | Medio | Media | Balance óptimo |
| **MLP grande (512-256-128)** | Bajo | Alta | Riesgo de overfitting |

### Tu Entregable: Sección en MODEL_COMPARISON.md

Añade una sección que responda:

1. **¿Por qué Logistic Regression tiene alto bias?**
   - Solo puede aprender fronteras de decisión lineales
   - MNIST tiene patrones no lineales (curvas, esquinas)

2. **¿Por qué MLP puede tener alta variance?**
   - Muchos parámetros pueden memorizar ejemplos de entrenamiento
   - Solución: Regularización L2, Dropout, Early Stopping

3. **Experimento práctico:**
   - Entrenar MLP con diferentes tamaños
   - Graficar train_accuracy vs test_accuracy
   - Identificar punto de overfitting

```python
# Código para análisis Bias-Variance
def train_and_evaluate(hidden_sizes: list, X_train, y_train, X_test, y_test):
    """Entrenar modelos de diferentes tamaños y comparar."""
    results = []

    for sizes in hidden_sizes:
        model = NeuralNetwork(layers=[784] + list(sizes) + [10])
        model.fit(X_train, y_train, epochs=100)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        gap = train_acc - test_acc  # Gap grande = overfitting

        results.append({
            'hidden_sizes': sizes,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': gap
        })

    return results

# Experimento
sizes_to_test = [
    (32,),           # Muy pequeño (alto bias)
    (128, 64),       # Medio (balanceado)
    (512, 256, 128)  # Grande (alta variance)
]
```

---

## 📝 Formato de Informe: Paper Científico (v3.2)

> 💎 **Profesionalismo:** El notebook final debe tener el formato de un paper académico.

### Estructura del Jupyter Notebook Final

```markdown
# MNIST Digit Classification: A From-Scratch Implementation

## Abstract
[3-4 oraciones resumiendo objetivo, métodos y resultados principales]

## 1. Introduction
- Problema: clasificación de dígitos escritos a mano
- Motivación: demostrar competencia en ML
- Contribución: implementación 100% desde cero

## 2. Dataset
- Descripción de MNIST (60K train, 10K test, 28x28 pixels)
- Preprocesamiento aplicado

## 3. Methodology
### 3.1 Unsupervised Learning
- PCA: reducción dimensional
- K-Means: clustering

### 3.2 Supervised Learning
- Logistic Regression One-vs-All

### 3.3 Deep Learning
- MLP architecture: 784→128→64→10
- Training: SGD with momentum

## 4. Results
### 4.1 PCA Analysis
[Gráficos de varianza explicada, visualización 2D]

### 4.2 K-Means Clustering
[Centroides, pureza de clusters]

### 4.3 Model Comparison
[Tabla comparativa, matriz de confusión]

### 4.4 Bias-Variance Analysis
[Gap train-test para diferentes modelos]

## 5. Discussion
- ¿Por qué MLP supera a Logistic Regression?
- Limitaciones del estudio
- Trabajo futuro

## 6. Conclusion
[2-3 oraciones de cierre]

## References
- LeCun, Y., et al. "Gradient-based learning applied to document recognition."
- Deep Learning Book (Goodfellow et al.)
```

---

## 🔎 Análisis de Errores: Nivel Senior (v3.3)

> 💎 **Profesionalismo:** No solo muestres el accuracy. Muestra las imágenes que la red falló y explica por qué.

### Por Qué Es Importante

```
ANÁLISIS DE ERRORES = LO QUE SEPARA JUNIOR DE SENIOR

Junior: "Mi modelo tiene 92% accuracy"

Senior: "Mi modelo tiene 92% accuracy. Los errores se concentran en:
        - Confusión 4↔9 (formas similares)
        - Confusión 3↔8 (curvas similares)
        - Dígitos mal escritos o cortados
        Esto sugiere que el modelo necesita más ejemplos
        de estos casos difíciles o data augmentation."
```

### Script: Análisis de Errores

```python
"""
Error Analysis - Visualización de Fallos del Modelo
Nivel Senior: No solo accuracy, también entender los errores.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def analyze_errors(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_errors: int = 20
) -> dict:
    """
    Analiza y visualiza los errores del modelo.

    Args:
        model: Modelo entrenado (con .predict())
        X_test: Datos de test
        y_test: Labels de test
        n_errors: Número de errores a visualizar

    Returns:
        Diccionario con análisis completo
    """
    # Predicciones
    y_pred = model.predict(X_test)

    # Identificar errores
    errors_mask = y_pred != y_test
    error_indices = np.where(errors_mask)[0]

    print("=" * 60)
    print("ANÁLISIS DE ERRORES")
    print("=" * 60)
    print(f"Total errores: {len(error_indices)} / {len(y_test)}")
    print(f"Error rate: {100 * len(error_indices) / len(y_test):.2f}%")

    # Matriz de confusión de errores
    confusion_pairs = {}
    for idx in error_indices:
        pair = (y_test[idx], y_pred[idx])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    # Top confusiones
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: -x[1])

    print("\n📊 TOP CONFUSIONES:")
    for (true, pred), count in sorted_pairs[:10]:
        print(f"  {true} → {pred}: {count} errores")

    # Visualizar errores
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle("Ejemplos de Errores del Modelo", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < min(n_errors, len(error_indices)):
            idx = error_indices[i]
            img = X_test[idx].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"True: {y_test[idx]}, Pred: {y_pred[idx]}",
                        color='red', fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=150)
    plt.show()

    return {
        'n_errors': len(error_indices),
        'error_rate': len(error_indices) / len(y_test),
        'confusion_pairs': sorted_pairs,
        'error_indices': error_indices
    }


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float]
) -> None:
    """
    Visualiza curvas de aprendizaje para diagnóstico Bias-Variance.

    - Train alto, Val alto → Underfitting (High Bias)
    - Train bajo, Val alto → Overfitting (High Variance)
    - Train bajo, Val bajo → Buen modelo
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curves: Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Learning Curves: Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Diagnóstico
    final_gap = train_accs[-1] - val_accs[-1]

    if val_accs[-1] < 0.7:
        diagnosis = "⚠️ UNDERFITTING: Modelo muy simple o poco entrenamiento"
    elif final_gap > 0.1:
        diagnosis = "⚠️ OVERFITTING: Gap train-val > 10%"
    else:
        diagnosis = "✓ BUEN AJUSTE: Modelo generaliza bien"

    fig.suptitle(f"Diagnóstico: {diagnosis}", fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    plt.show()

    print("\n📈 DIAGNÓSTICO BIAS-VARIANCE:")
    print(f"  Train Accuracy Final: {train_accs[-1]:.4f}")
    print(f"  Val Accuracy Final:   {val_accs[-1]:.4f}")
    print(f"  Gap:                  {final_gap:.4f}")
    print(f"  → {diagnosis}")
```

### Sección Obligatoria en MODEL_COMPARISON.md

```markdown
## Error Analysis

### Top Confusiones del Modelo

| True → Pred | Count | Explicación |
|-------------|-------|-------------|
| 4 → 9 | 23 | Formas similares (bucle arriba) |
| 9 → 4 | 18 | Formas similares |
| 3 → 8 | 15 | Curvas similares |
| 7 → 1 | 12 | Trazo vertical dominante |

### Visualización de Errores

![Errores del modelo](error_analysis.png)

### Interpretación

Los errores se concentran principalmente en dígitos con formas similares.
Esto sugiere que:
1. El modelo captura bien las features principales
2. Features más finas (bucles, cruces) necesitan más ejemplos
3. Data augmentation podría ayudar
```

---

## ✅ Checklist de Finalización (v3.3)

### Semana 21: EDA + No Supervisado
- [ ] PCA reduce MNIST a 2D/50D con visualización
- [ ] Analicé varianza explicada por componente
- [ ] K-Means agrupa dígitos sin etiquetas
- [ ] Visualicé centroides como imágenes 28x28

### Semana 22: Clasificación Supervisada
- [ ] Logistic Regression One-vs-All funcional
- [ ] Accuracy >85% en test set
- [ ] Calculé Precision, Recall, F1 por clase
- [ ] Analicé matriz de confusión

### Semana 23: Deep Learning
- [ ] MLP con arquitectura 784→128→64→10
- [ ] Forward y backward pass implementados
- [ ] Mini-batch SGD funcionando
- [ ] Accuracy >90% en test set

### Semana 24: Benchmark + Informe
- [ ] MODEL_COMPARISON.md completo
- [ ] README.md profesional en inglés

### Requisitos v3.3
- [ ] **Análisis Bias-Variance** con experimento práctico
- [ ] **Notebook en formato Paper** (Abstract, Methods, Results, Discussion)
- [ ] **Análisis de Errores** con visualización de fallos
- [ ] **Curvas de Aprendizaje** con diagnóstico Bias-Variance
- [ ] Sección "Error Analysis" en MODEL_COMPARISON.md
- [ ] `mypy src/` pasa sin errores
- [ ] `pytest tests/` pasa sin errores

### Metodología Feynman
- [ ] Puedo explicar por qué MLP supera a Logistic en 5 líneas
- [ ] Puedo explicar Bias vs Variance en 5 líneas
- [ ] Puedo explicar por qué 4↔9 se confunden frecuentemente

---

## 🔗 Navegación

| Anterior | Índice |
|----------|--------|
| [07_DEEP_LEARNING](07_DEEP_LEARNING.md) | [00_INDICE](00_INDICE.md) |
