# Módulo 06 - Unsupervised Learning

> **🎯 Objetivo:** Dominar K-Means clustering y PCA para reducción dimensional  
> **Fase:** 2 - Núcleo de ML | **Semanas 13-16**  
> **Curso del Pathway:** Unsupervised Algorithms in Machine Learning

---

## 🧠 ¿Qué es Unsupervised Learning?

```
APRENDIZAJE NO SUPERVISADO

Tenemos:
- Datos de entrada X (features)
- NO tenemos etiquetas Y

Objetivo: Encontrar estructura oculta en los datos

Tipos principales:
├── CLUSTERING: Agrupar puntos similares
│   └── K-Means, DBSCAN, Hierarchical
├── REDUCCIÓN DIMENSIONAL: Comprimir features
│   └── PCA, t-SNE, UMAP
└── DETECCIÓN DE ANOMALÍAS: Encontrar outliers
    └── Isolation Forest, GMM
```

---

## 📚 Contenido del Módulo

| Semana | Tema | Entregable |
|--------|------|------------|
| 13 | K-Means Clustering | `kmeans.py` |
| 14 | Evaluación de Clusters | Métricas de clustering |
| 15 | PCA | `pca.py` |
| 16 | PCA Aplicado + GMM | Compresión de imágenes |

---

## 💻 Parte 1: K-Means Clustering

### 1.1 Algoritmo de Lloyd

```python
import numpy as np

"""
K-MEANS CLUSTERING (Algoritmo de Lloyd)

Objetivo: Particionar n puntos en k clusters, minimizando la
varianza intra-cluster (inercia).

Algoritmo:
1. Inicializar k centroides (aleatorio o k-means++)
2. Repetir hasta convergencia:
   a. ASIGNAR: cada punto al centroide más cercano
   b. ACTUALIZAR: mover cada centroide al promedio de sus puntos
3. Retornar centroides y asignaciones

Función objetivo (minimizar):
    J = Σᵢ Σⱼ ||xⱼ - μᵢ||²
    
Donde xⱼ pertenece al cluster i con centroide μᵢ
"""

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia euclidiana entre dos puntos."""
    return np.sqrt(np.sum((a - b) ** 2))

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Asigna cada punto al centroide más cercano.
    
    Args:
        X: datos (n_samples, n_features)
        centroids: centroides actuales (k, n_features)
    
    Returns:
        labels: índice del cluster para cada punto (n_samples,)
    """
    n_samples = X.shape[0]
    k = centroids.shape[0]
    
    # Calcular distancia de cada punto a cada centroide
    distances = np.zeros((n_samples, k))
    for i in range(k):
        distances[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))
    
    # Asignar al más cercano
    return np.argmin(distances, axis=1)

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Actualiza centroides como el promedio de los puntos asignados.
    
    Args:
        X: datos
        labels: asignaciones actuales
        k: número de clusters
    
    Returns:
        nuevos centroides
    """
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features))
    
    for i in range(k):
        points_in_cluster = X[labels == i]
        if len(points_in_cluster) > 0:
            centroids[i] = np.mean(points_in_cluster, axis=0)
    
    return centroids
```

### 1.2 K-Means++ Initialization

```python
import numpy as np

def kmeans_plus_plus_init(X: np.ndarray, k: int, random_state: int = None) -> np.ndarray:
    """
    Inicialización K-Means++.
    
    Mejor que inicialización aleatoria porque:
    - Elige centroides que están lejos entre sí
    - Reduce la probabilidad de mala convergencia
    - Garantiza O(log k) de la solución óptima
    
    Algoritmo:
    1. Elegir primer centroide aleatoriamente
    2. Para cada centroide restante:
       a. Calcular distancia de cada punto al centroide más cercano
       b. Elegir nuevo centroide con probabilidad proporcional a d²
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features))
    
    # Primer centroide aleatorio
    first_idx = np.random.randint(n_samples)
    centroids[0] = X[first_idx]
    
    # Centroides restantes
    for c in range(1, k):
        # Calcular distancia al centroide más cercano para cada punto
        distances = np.zeros(n_samples)
        for i in range(n_samples):
            min_dist = float('inf')
            for j in range(c):
                dist = np.sum((X[i] - centroids[j]) ** 2)
                min_dist = min(min_dist, dist)
            distances[i] = min_dist
        
        # Probabilidad proporcional a d²
        probabilities = distances / np.sum(distances)
        
        # Elegir nuevo centroide
        new_idx = np.random.choice(n_samples, p=probabilities)
        centroids[c] = X[new_idx]
    
    return centroids
```

### 1.3 Implementación Completa

```python
import numpy as np
from typing import Tuple

class KMeans:
    """K-Means Clustering implementado desde cero."""
    
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        tol: float = 1e-4,
        init: str = 'kmeans++',
        random_state: int = None
    ):
        """
        Args:
            n_clusters: número de clusters (k)
            max_iter: máximo de iteraciones
            tol: tolerancia para convergencia
            init: 'kmeans++' o 'random'
            random_state: semilla para reproducibilidad
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """Inicializa centroides."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.init == 'kmeans++':
            return kmeans_plus_plus_init(X, self.n_clusters, self.random_state)
        else:
            # Inicialización aleatoria
            indices = np.random.choice(len(X), self.n_clusters, replace=False)
            return X[indices].copy()
    
    def _compute_inertia(self, X: np.ndarray) -> float:
        """
        Calcula inercia (within-cluster sum of squares).
        
        Inercia = Σᵢ Σⱼ ||xⱼ - μᵢ||²
        """
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels_ == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """Entrena el modelo."""
        # Inicializar centroides
        self.centroids = self._init_centroids(X)
        
        for iteration in range(self.max_iter):
            # Guardar centroides anteriores
            old_centroids = self.centroids.copy()
            
            # Paso 1: Asignar puntos a clusters
            self.labels_ = assign_clusters(X, self.centroids)
            
            # Paso 2: Actualizar centroides
            self.centroids = update_centroids(X, self.labels_, self.n_clusters)
            
            # Verificar convergencia
            centroid_shift = np.sum((self.centroids - old_centroids) ** 2)
            if centroid_shift < self.tol:
                break
        
        self.n_iter_ = iteration + 1
        self.inertia_ = self._compute_inertia(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clusters para nuevos datos."""
        return assign_clusters(X, self.centroids)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Entrena y predice."""
        self.fit(X)
        return self.labels_


# Demo
np.random.seed(42)

# Generar datos sintéticos (3 clusters)
cluster1 = np.random.randn(100, 2) + [0, 0]
cluster2 = np.random.randn(100, 2) + [5, 5]
cluster3 = np.random.randn(100, 2) + [10, 0]
X = np.vstack([cluster1, cluster2, cluster3])

# Entrenar
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

print(f"Iteraciones: {kmeans.n_iter_}")
print(f"Inercia: {kmeans.inertia_:.2f}")
print(f"Centroides:\n{kmeans.centroids}")
```

---

## 💻 Parte 2: Evaluación de Clusters

### 2.1 Inercia (Within-Cluster Sum of Squares)

```python
def compute_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Inercia: suma de distancias cuadradas al centroide.
    
    Menor inercia = clusters más compactos.
    
    Problema: siempre disminuye al aumentar k.
    Solución: usar método del codo.
    """
    inertia = 0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia
```

### 2.2 Método del Codo (Elbow Method)

```python
import numpy as np
import matplotlib.pyplot as plt

def elbow_method(X: np.ndarray, k_range: range) -> list:
    """
    Método del codo para elegir k óptimo.
    
    Busca el punto donde añadir más clusters
    no reduce significativamente la inercia.
    """
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return inertias

def plot_elbow(k_range: range, inertias: list):
    """Visualiza el método del codo."""
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertias, 'bo-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.grid(True)
    plt.show()

# Demo
# inertias = elbow_method(X, range(1, 11))
# plot_elbow(range(1, 11), inertias)
```

### 2.3 Silhouette Score

```python
import numpy as np

def silhouette_sample(X: np.ndarray, labels: np.ndarray, idx: int) -> float:
    """
    Calcula silhouette para un solo punto.
    
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    Donde:
    - a(i): distancia promedio a puntos del mismo cluster
    - b(i): distancia promedio mínima a puntos de otro cluster
    
    Rango: [-1, 1]
    - 1: punto bien asignado
    - 0: punto en frontera entre clusters
    - -1: punto mal asignado
    """
    point = X[idx]
    label = labels[idx]
    
    # a(i): distancia promedio intra-cluster
    same_cluster = X[labels == label]
    if len(same_cluster) > 1:
        a = np.mean([np.sqrt(np.sum((point - p) ** 2)) 
                     for p in same_cluster if not np.array_equal(p, point)])
    else:
        a = 0
    
    # b(i): distancia promedio al cluster más cercano
    unique_labels = np.unique(labels)
    b = float('inf')
    for other_label in unique_labels:
        if other_label != label:
            other_cluster = X[labels == other_label]
            if len(other_cluster) > 0:
                avg_dist = np.mean([np.sqrt(np.sum((point - p) ** 2)) 
                                   for p in other_cluster])
                b = min(b, avg_dist)
    
    if b == float('inf'):
        return 0
    
    return (b - a) / max(a, b)

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette Score promedio para todos los puntos.
    
    Mayor es mejor (max = 1).
    """
    scores = [silhouette_sample(X, labels, i) for i in range(len(X))]
    return np.mean(scores)


# Demo
# score = silhouette_score(X, labels)
# print(f"Silhouette Score: {score:.4f}")
```

---

## 💻 Parte 3: PCA (Principal Component Analysis)

### 3.1 Concepto

```python
"""
PCA - ANÁLISIS DE COMPONENTES PRINCIPALES

Objetivo: Reducir dimensionalidad preservando la máxima varianza.

Idea:
1. Centrar los datos (restar media)
2. Encontrar direcciones de máxima varianza (eigenvectors)
3. Proyectar datos en las top-k direcciones

Matemáticamente:
- Las componentes principales son los eigenvectors de la matriz de covarianza
- Los eigenvalues indican cuánta varianza captura cada componente

Aplicaciones:
- Visualización (reducir a 2D/3D)
- Preprocesamiento (eliminar ruido, reducir features)
- Compresión de datos/imágenes
"""
```

### 3.2 PCA via Eigendecomposition

```python
import numpy as np

def pca_eigen(X: np.ndarray, n_components: int) -> tuple:
    """
    PCA usando eigendecomposition de la matriz de covarianza.
    
    Pasos:
    1. Centrar datos: X_centered = X - mean(X)
    2. Calcular matriz de covarianza: Σ = (1/(n-1)) X^T X
    3. Eigendecomposition: Σv = λv
    4. Ordenar eigenvectors por eigenvalue descendente
    5. Proyectar: X_pca = X_centered @ V[:, :k]
    
    Returns:
        X_pca: datos transformados
        components: eigenvectors (componentes principales)
        explained_variance_ratio: proporción de varianza por componente
    """
    # 1. Centrar
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # 2. Matriz de covarianza
    n_samples = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    
    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Convertir a reales (puede haber componentes imaginarias pequeñas)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
    # 4. Ordenar por eigenvalue descendente
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. Seleccionar top k componentes
    components = eigenvectors[:, :n_components]
    
    # 6. Proyectar
    X_pca = X_centered @ components
    
    # 7. Varianza explicada
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues[:n_components] / total_variance
    
    return X_pca, components, explained_variance_ratio, mean
```

### 3.3 PCA via SVD (Más Estable)

```python
import numpy as np

def pca_svd(X: np.ndarray, n_components: int) -> tuple:
    """
    PCA usando SVD (Singular Value Decomposition).
    
    Más estable numéricamente que eigendecomposition.
    
    Si X = UΣV^T, entonces:
    - V contiene las componentes principales
    - Σ²/(n-1) son los eigenvalues (varianzas)
    """
    # 1. Centrar
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    
    # 2. SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # 3. Componentes principales (filas de Vt, o columnas de V)
    components = Vt[:n_components].T
    
    # 4. Proyectar
    X_pca = X_centered @ components
    
    # 5. Varianza explicada
    n_samples = X.shape[0]
    variance = (S ** 2) / (n_samples - 1)
    explained_variance_ratio = variance[:n_components] / np.sum(variance)
    
    return X_pca, components, explained_variance_ratio, mean
```

### 3.4 Implementación Completa

```python
import numpy as np

class PCA:
    """Principal Component Analysis implementado desde cero."""
    
    def __init__(self, n_components: int = 2):
        """
        Args:
            n_components: número de componentes a retener
        """
        self.n_components = n_components
        self.components_ = None  # (n_features, n_components)
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """Calcula componentes principales."""
        # Centrar
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Componentes principales
        self.components_ = Vt[:self.n_components].T
        
        # Varianza explicada
        n_samples = X.shape[0]
        variance = (S ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = variance[:self.n_components] / np.sum(variance)
        self.singular_values_ = S[:self.n_components]
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Proyecta datos a espacio de componentes principales."""
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit y transform en un paso."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Reconstruye datos desde el espacio PCA.
        
        X_reconstructed = X_pca @ components.T + mean
        
        Nota: hay pérdida de información si n_components < n_features
        """
        return X_pca @ self.components_.T + self.mean_
    
    def get_covariance(self) -> np.ndarray:
        """Retorna matriz de covarianza aproximada."""
        return self.components_ @ np.diag(self.singular_values_ ** 2) @ self.components_.T


# Demo
np.random.seed(42)

# Datos correlacionados en 3D
n_samples = 200
X = np.random.randn(n_samples, 3)
X[:, 1] = X[:, 0] * 2 + np.random.randn(n_samples) * 0.1  # y correlacionado con x
X[:, 2] = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"Shape original: {X.shape}")
print(f"Shape reducido: {X_pca.shape}")
print(f"Varianza explicada: {pca.explained_variance_ratio_}")
print(f"Varianza total: {np.sum(pca.explained_variance_ratio_):.2%}")
```

### 3.5 Reconstrucción y Error

```python
import numpy as np

def reconstruction_error(X: np.ndarray, pca: PCA) -> float:
    """
    Calcula el error de reconstrucción.
    
    Error = ||X - X_reconstructed||² / ||X||²
    """
    X_pca = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    
    error = np.sum((X - X_reconstructed) ** 2)
    total = np.sum((X - np.mean(X, axis=0)) ** 2)
    
    return error / total

def choose_n_components(X: np.ndarray, variance_threshold: float = 0.95) -> int:
    """
    Elige número de componentes para retener cierta varianza.
    
    Args:
        variance_threshold: proporción de varianza a retener (ej: 0.95 = 95%)
    """
    # PCA con todos los componentes
    pca = PCA(n_components=min(X.shape))
    pca.fit(X)
    
    # Varianza acumulada
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Encontrar n_components
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    return n_components, cumulative_variance
```

---

## 💻 Parte 4: Aplicaciones de PCA

### 4.1 Compresión de Imágenes

```python
import numpy as np

def compress_image_pca(image: np.ndarray, n_components: int) -> tuple:
    """
    Comprime una imagen usando PCA.
    
    Args:
        image: imagen grayscale (height, width)
        n_components: número de componentes a retener
    
    Returns:
        imagen comprimida, pca model
    """
    # Tratar filas como muestras
    pca = PCA(n_components=n_components)
    image_pca = pca.fit_transform(image)
    
    # Reconstruir
    image_reconstructed = pca.inverse_transform(image_pca)
    
    return image_reconstructed, pca

def compression_ratio_pca(original_shape: tuple, n_components: int) -> float:
    """Calcula ratio de compresión."""
    height, width = original_shape
    original_size = height * width
    # Almacenamos: componentes + proyecciones + media
    compressed_size = n_components * width + height * n_components + width
    return compressed_size / original_size
```

### 4.2 Visualización en 2D

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_pca_2d(X: np.ndarray, labels: np.ndarray = None, title: str = "PCA"):
    """Reduce a 2D y visualiza."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    
    if labels is not None:
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       label=f'Clase {label}', alpha=0.7)
        plt.legend()
    else:
        plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
```

---

## 📦 Entregable del Módulo

### `unsupervised_learning.py`

```python
"""
Unsupervised Learning Module

Implementación desde cero de:
- K-Means Clustering (con K-Means++ initialization)
- PCA (Principal Component Analysis)
- Métricas de evaluación de clusters

Autor: [Tu nombre]
Módulo: 05 - Unsupervised Learning
"""

import numpy as np
from typing import Tuple, List


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

def kmeans_plus_plus(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:
    """Inicialización K-Means++."""
    if seed: np.random.seed(seed)
    n = len(X)
    centroids = [X[np.random.randint(n)]]
    
    for _ in range(1, k):
        distances = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])
        probs = distances / distances.sum()
        centroids.append(X[np.random.choice(n, p=probs)])
    
    return np.array(centroids)


class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, seed=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        self.centroids = kmeans_plus_plus(X, self.n_clusters, self.seed)
        
        for i in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            # Asignar
            distances = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])
            self.labels_ = np.argmin(distances, axis=1)
            
            # Actualizar
            for j in range(self.n_clusters):
                points = X[self.labels_ == j]
                if len(points) > 0:
                    self.centroids[j] = points.mean(axis=0)
            
            if np.sum((self.centroids - old_centroids)**2) < self.tol:
                break
        
        self.n_iter_ = i + 1
        self.inertia_ = sum(np.sum((X[self.labels_ == j] - self.centroids[j])**2) 
                           for j in range(self.n_clusters))
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


# ============================================================
# PCA
# ============================================================

class PCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
    
    def fit(self, X: np.ndarray) -> 'PCA':
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        self.components_ = Vt[:self.n_components].T
        variance = (S**2) / (len(X) - 1)
        self.explained_variance_ratio_ = variance[:self.n_components] / variance.sum()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        return X_pca @ self.components_.T + self.mean_


# ============================================================
# MÉTRICAS
# ============================================================

def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Within-cluster sum of squares."""
    return sum(np.sum((X[labels == i] - centroids[i])**2) 
               for i in range(len(centroids)))

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score promedio."""
    n = len(X)
    scores = []
    
    for i in range(n):
        # a: distancia promedio intra-cluster
        same = X[labels == labels[i]]
        a = np.mean([np.sqrt(np.sum((X[i] - x)**2)) for x in same if not np.array_equal(x, X[i])])
        
        # b: distancia promedio al cluster más cercano
        b = float('inf')
        for label in np.unique(labels):
            if label != labels[i]:
                other = X[labels == label]
                if len(other) > 0:
                    b = min(b, np.mean([np.sqrt(np.sum((X[i] - x)**2)) for x in other]))
        
        if b == float('inf'):
            scores.append(0)
        else:
            scores.append((b - a) / max(a, b))
    
    return np.mean(scores)


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    # Test K-Means
    c1 = np.random.randn(50, 2) + [0, 0]
    c2 = np.random.randn(50, 2) + [5, 5]
    c3 = np.random.randn(50, 2) + [10, 0]
    X = np.vstack([c1, c2, c3])
    
    kmeans = KMeans(n_clusters=3, seed=42)
    labels = kmeans.fit_predict(X)
    
    print(f"K-Means Inertia: {kmeans.inertia_:.2f}")
    print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")
    
    # Test PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    
    print(f"\nPCA Varianza explicada: {pca.explained_variance_ratio_}")
    print(f"Error reconstrucción: {np.mean((X - X_reconstructed)**2):.6f}")
    
    print("\n✓ Todos los tests pasaron!")
```

---

## ✅ Checklist de Finalización

- [ ] Implementé K-Means con inicialización K-Means++
- [ ] Entiendo el algoritmo de Lloyd (asignar-actualizar)
- [ ] Puedo calcular inercia y usarla para el método del codo
- [ ] Implementé silhouette score
- [ ] Implementé PCA usando SVD
- [ ] Entiendo varianza explicada y puedo elegir n_components
- [ ] Puedo reconstruir datos desde PCA
- [ ] Apliqué PCA para visualización 2D
- [ ] Todos los tests del módulo pasan

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [04_SUPERVISED_LEARNING](04_SUPERVISED_LEARNING.md) | [00_INDICE](00_INDICE.md) | [06_DEEP_LEARNING](06_DEEP_LEARNING.md) |
