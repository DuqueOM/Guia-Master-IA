# Módulo 08 - Proyecto Final: MNIST Analyst

> **🎯 Objetivo:** Pipeline end-to-end que demuestra competencia en las 3 áreas del Pathway
> **Fase:** 3 - Proyecto Integrador | **Semanas 21-24** (4 semanas)
> **Dataset:** MNIST (dígitos escritos a mano, 28×28 píxeles) / **Fashion-MNIST** (alternativo, mismo formato)

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

> 💡 **Nota v3.3:** MNIST es un dataset simple. 4 semanas son suficientes para un proyecto bien estructurado.

> 💡 **Nota (upgrade):** si quieres un benchmark más realista, usa **Fashion-MNIST**. Mantiene 28×28 y 10 clases, pero es más difícil y diferencia mejor LR vs MLP/CNN.

---

## 📚 Estructura del Proyecto

### Cronograma (4 Semanas)

| Semana | Fase | Materia Demostrada | Entregable |
|--------|------|-------------------|------------|
| 21 | EDA + No Supervisado | Unsupervised Algorithms | PCA + K-Means funcionando |
| 22 | Clasificación Clásica | Supervised Learning | Logistic Regression OvA |
| 23 | Deep Learning | Introduction to Deep Learning | MLP con backprop |
| 24 | Benchmark + Informe | Integración | MODEL_COMPARISON.md + deployment mínimo |

Evaluación (rúbrica):

- [study_tools/RUBRICA_v1.md](../study_tools/RUBRICA_v1.md) (scope `M08` en `rubrica.csv`)
- Condición dura de admisión: **PB-23 ≥ 80/100** (si PB-23 < 80 ⇒ estado “Aún no listo” aunque el total global sea alto)

Notas prácticas (Week 24):

- **Fashion-MNIST (alternativo):** en vez de MNIST dígitos, corre el benchmark en Fashion-MNIST para ver degradación realista.
- **Dirty Data Check:** genera un dataset corrupto (ruido/NaNs/inversión) con `scripts/corrupt_mnist.py` y documenta cómo lo limpiaste.
- **Deployment mínimo:** entrena y guarda una CNN con `scripts/train_cnn_pytorch.py` y luego predice una imagen 28×28 con `scripts/predict.py`.

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

## 🎯 Ejercicios por fase (progresivos) + Soluciones

Reglas:

- **Intenta primero** sin mirar la solución.
- **Timebox sugerido:** 30–90 min por ejercicio.
- **Éxito mínimo:** tu solución debe pasar los `assert`.

Nota: los ejercicios usan **datos sintéticos** para que sean reproducibles sin descargar MNIST. La idea es validar *invariantes* del pipeline (shapes, estabilidad numérica, métricas, convergencia, reproducibilidad).

---

### Ejercicio 8.1: Reproducibilidad (seed) y split determinista

#### Enunciado

1) **Básico**

- Implementa un split train/test reproducible basado en una semilla.

2) **Intermedio**

- Verifica que la misma semilla produce el mismo split.

3) **Avanzado**

- Verifica que no se pierden muestras y que train/test no se solapan.

#### Solución

```python
import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 0):
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


np.random.seed(0)
X = np.random.randn(100, 784)
y = np.random.randint(0, 10, size=(100,))

Xtr1, Xte1, ytr1, yte1 = train_test_split(X, y, test_size=0.25, seed=123)
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, test_size=0.25, seed=123)

assert np.allclose(Xtr1, Xtr2)
assert np.allclose(Xte1, Xte2)
assert np.all(ytr1 == ytr2)
assert np.all(yte1 == yte2)

assert Xtr1.shape[0] + Xte1.shape[0] == X.shape[0]
assert len(np.intersect1d(Xtr1[:, 0], Xte1[:, 0])) <= X.shape[0]
```

---

### Ejercicio 8.2: Invariantes de datos tipo MNIST (shapes + rangos)

#### Enunciado

1) **Básico**

- Simula imágenes `uint8` en `[0,255]` con shape `(n, 784)`.

2) **Intermedio**

- Normaliza a `float` en `[0,1]`.

3) **Avanzado**

- Verifica que no aparecen `NaN/inf` y que `dtype` es float.

#### Solución

```python
import numpy as np

rng = np.random.default_rng(0)
n = 256
X_uint8 = rng.integers(0, 256, size=(n, 784), dtype=np.uint8)

X = X_uint8.astype(np.float32) / 255.0

assert X.shape == (n, 784)
assert X.dtype in (np.float32, np.float64)
assert np.isfinite(X).all()
assert X.min() >= 0.0
assert X.max() <= 1.0
```

---

### Ejercicio 8.3: One-hot encoding (multiclase)

#### Enunciado

1) **Básico**

- Implementa `one_hot(y, num_classes=10)`.

2) **Intermedio**

- Verifica que cada fila suma 1.

3) **Avanzado**

- Verifica que `argmax(one_hot(y)) == y`.

#### Solución

```python
import numpy as np

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y).astype(int)
    Y = np.zeros((y.size, num_classes), dtype=float)
    Y[np.arange(y.size), y] = 1.0
    return Y


y = np.array([0, 2, 9, 2, 1])
Y = one_hot(y, num_classes=10)

assert Y.shape == (y.size, 10)
assert np.allclose(np.sum(Y, axis=1), 1.0)
assert np.all(np.argmax(Y, axis=1) == y)
```

---

### Ejercicio 8.4: PCA vía SVD (varianza explicada + reconstrucción)

#### Enunciado

1) **Básico**

- Implementa PCA con SVD para reducir a `k` componentes.

2) **Intermedio**

- Calcula `explained_variance_ratio` y verifica que está ordenada (de mayor a menor).

3) **Avanzado**

- Reconstruye con `k=10` y `k=50` y verifica que el error baja.

#### Solución

```python
import numpy as np

def pca_svd_fit_transform(X: np.ndarray, k: int):
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k].T
    Z = Xc @ Vk
    var = (S ** 2) / (Xc.shape[0] - 1)
    ratio = var / np.sum(var)
    return Z, Vk, mu, ratio


def pca_reconstruct(Z: np.ndarray, Vk: np.ndarray, mu: np.ndarray) -> np.ndarray:
    return Z @ Vk.T + mu


rng = np.random.default_rng(1)
X = rng.normal(size=(300, 784)).astype(np.float64)

Z10, V10, mu, ratio = pca_svd_fit_transform(X, k=10)
Z50, V50, mu2, ratio2 = pca_svd_fit_transform(X, k=50)

assert np.allclose(mu, mu2)
assert ratio[0] >= ratio[1]
assert ratio2[0] >= ratio2[1]

X10 = pca_reconstruct(Z10, V10, mu)
X50 = pca_reconstruct(Z50, V50, mu)

err10 = np.linalg.norm(X - X10)
err50 = np.linalg.norm(X - X50)

assert err50 <= err10 + 1e-12
```

---

### Ejercicio 8.5: K-Means (inercia) - una iteración debe no aumentar J

#### Enunciado

1) **Básico**

- Implementa asignación por distancia euclidiana al centroide más cercano.

2) **Intermedio**

- Implementa update de centroides como promedio (manejando clusters vacíos).

3) **Avanzado**

- Verifica que la inercia `J` no aumenta tras una iteración.

#### Solución

```python
import numpy as np

def assign_labels(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    D2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
    return np.argmin(D2, axis=1)


def update_centroids(X: np.ndarray, labels: np.ndarray, C: np.ndarray) -> np.ndarray:
    C_new = C.copy()
    for j in range(C.shape[0]):
        mask = labels == j
        if np.any(mask):
            C_new[j] = np.mean(X[mask], axis=0)
    return C_new


def inertia(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    diffs = X - C[labels]
    return float(np.sum(diffs ** 2))


rng = np.random.default_rng(2)
X = np.vstack([
    rng.normal(loc=-1.0, scale=0.5, size=(100, 2)),
    rng.normal(loc=+1.0, scale=0.5, size=(100, 2)),
])
C0 = np.array([[-1.0, 1.0], [1.0, -1.0]])

labels0 = assign_labels(X, C0)
J0 = inertia(X, C0, labels0)

C1 = update_centroids(X, labels0, C0)
labels1 = assign_labels(X, C1)
J1 = inertia(X, C1, labels1)

assert J1 <= J0 + 1e-12
```

---

### Ejercicio 8.6: Logistic Regression OvA - gradient check (una clase)

#### Enunciado

1) **Básico**

- Implementa sigmoid y BCE para una clase binaria `y_c`.

2) **Intermedio**

- Implementa gradiente `∇w = (1/n) X^T (p - y_c)`.

3) **Avanzado**

- Verifica una coordenada del gradiente con diferencias centrales.

#### Solución

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def bce_from_logits(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, eps: float = 1e-15) -> float:
    p = sigmoid(X @ w + b)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def grad_w(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    p = sigmoid(X @ w + b)
    return (X.T @ (p - y)) / X.shape[0]


rng = np.random.default_rng(3)
n, d = 120, 50
X = rng.normal(size=(n, d))
y = (rng.random(size=(n, 1)) < 0.4).astype(float)

w = rng.normal(size=(d, 1)) * 0.1
b = 0.0

g = grad_w(X, y, w, b)

idx = 7
h = 1e-6
E = np.zeros_like(w)
E[idx, 0] = 1.0
L_plus = bce_from_logits(X, y, w + h * E, b)
L_minus = bce_from_logits(X, y, w - h * E, b)
g_num = (L_plus - L_minus) / (2.0 * h)

assert np.isclose(g[idx, 0], g_num, rtol=1e-4, atol=1e-6)
```

---

### Ejercicio 8.7: MLP (sanity) - overfit mini-batch

#### Enunciado

1) **Básico**

- Implementa un MLP mínimo `784→32→10` (ReLU + softmax) y cross-entropy.

2) **Intermedio**

- Entrena sobre un set tiny (p.ej. 64 ejemplos) y verifica que el loss baja.

3) **Avanzado**

- Verifica que logra accuracy alta en entrenamiento (overfit).

#### Solución

```python
import numpy as np

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_deriv(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(float)


def logsumexp(z: np.ndarray, axis: int = -1, keepdims: bool = True) -> np.ndarray:
    m = np.max(z, axis=axis, keepdims=True)
    return m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(z - logsumexp(z))


def cross_entropy(y_onehot: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1.0)
    return float(-np.mean(np.sum(y_onehot * np.log(p), axis=1)))


rng = np.random.default_rng(4)
n, d_in, d_h, d_out = 64, 784, 32, 10
X = rng.normal(size=(n, d_in))
y = rng.integers(0, d_out, size=(n,))
Y = np.zeros((n, d_out), dtype=float)
Y[np.arange(n), y] = 1.0

W1 = rng.normal(size=(d_in, d_h)) * 0.01
b1 = np.zeros(d_h)
W2 = rng.normal(size=(d_h, d_out)) * 0.01
b2 = np.zeros(d_out)

lr = 1.0
loss0 = None
for _ in range(200):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    P = softmax(Z2)
    loss = cross_entropy(Y, P)
    if loss0 is None:
        loss0 = loss

    dZ2 = (P - Y) / n
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

loss_end = cross_entropy(Y, softmax(relu(X @ W1 + b1) @ W2 + b2))
pred = np.argmax(softmax(relu(X @ W1 + b1) @ W2 + b2), axis=1)
acc = float(np.mean(pred == y))

assert loss_end <= loss0
assert acc > 0.6
```

---

### Ejercicio 8.8: Métricas (confusion matrix + F1 macro)

#### Enunciado

1) **Básico**

- Implementa `confusion_matrix(y_true, y_pred, k)`.

2) **Intermedio**

- Implementa precision/recall/F1 por clase.

3) **Avanzado**

- Implementa F1 macro y verifica rango `[0,1]`.

#### Solución

```python
import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def prf_from_cm(cm: np.ndarray):
    k = cm.shape[0]
    eps = 1e-12
    precision = np.zeros(k)
    recall = np.zeros(k)
    f1 = np.zeros(k)
    for c in range(k):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        precision[c] = tp / (tp + fp + eps)
        recall[c] = tp / (tp + fn + eps)
        f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + eps)
    return precision, recall, f1


y_true = np.array([0, 1, 2, 2, 2, 1])
y_pred = np.array([0, 2, 2, 2, 1, 1])
cm = confusion_matrix(y_true, y_pred, k=3)

prec, rec, f1 = prf_from_cm(cm)
f1_macro = float(np.mean(f1))

assert cm.shape == (3, 3)
assert 0.0 <= f1_macro <= 1.0
```

---

### Ejercicio 8.9: Comparación de modelos (tabla consistente)

#### Enunciado

1) **Básico**

- Dado un dict `{modelo: accuracy}`, construye una lista ordenada de mejor a peor.

2) **Intermedio**

- Verifica que el mejor accuracy es el primero.

3) **Avanzado**

- Verifica que todos los accuracies están en `[0,1]`.

#### Solución

```python
import numpy as np

results = {
    "K-Means": 0.00,
    "Logistic Regression": 0.88,
    "MLP": 0.94,
}

items = sorted(results.items(), key=lambda kv: kv[1], reverse=True)

assert items[0][1] == max(results.values())
for _, acc in items:
    assert 0.0 <= acc <= 1.0
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

Incluye una visualización (por ejemplo una cuadrícula con las imágenes mal clasificadas) en tu reporte final.

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
 - [ ] Benchmark alternativo: probé **Fashion-MNIST** (o justifiqué por qué no)
 - [ ] Dirty Data Check: generé un dataset corrupto con `scripts/corrupt_mnist.py` y documenté limpieza
 - [ ] Deployment mínimo: entrené una CNN con `scripts/train_cnn_pytorch.py` y guardé el checkpoint
 - [ ] Deployment mínimo: ejecuté `scripts/predict.py` sobre una imagen 28×28 y reporté predicción

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
