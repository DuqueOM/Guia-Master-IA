# Módulo 08 - Machine Learning No Supervisado

> **🎯 Objetivo:** Dominar clustering, reducción de dimensionalidad y detección de anomalías  
> **⭐ PATHWAY LÍNEA 1:** Unsupervised Algorithms in Machine Learning

---

## 🧠 Analogía: Encontrar Patrones sin Respuestas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   APRENDIZAJE NO SUPERVISADO = Explorar sin Mapa                            │
│   ──────────────────────────────────────────────                            │
│                                                                             │
│   NO HAY ETIQUETAS (y)                                                      │
│   Solo tenemos datos (X)                                                    │
│                                                                             │
│   OBJETIVO: Descubrir estructura oculta en los datos                        │
│                                                                             │
│   ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐     │
│   │     CLUSTERING      │  │     REDUCCIÓN DE    │  │   DETECCIÓN DE  │     │
│   │                     │  │   DIMENSIONALIDAD   │  │    ANOMALÍAS    │     │
│   │  Agrupar similares  │  │  Comprimir datos    │  │  Encontrar raros│     │
│   │                     │  │                     │  │                 │     │
│   │    ●●●    ○○○       │  │  100D → 2D          │  │     ●●●●        │     │
│   │   ●●●●   ○○○○       │  │  (mantener info)    │  │    ●●●●★        │     │
│   │    ●●     ○○        │  │                     │  │     ●●●●        │     │
│   └─────────────────────┘  └─────────────────────┘  └─────────────────┘     │
│                                                                             │
│   APLICACIONES:                                                             │
│   • Segmentación de clientes                                                │
│   • Visualización de datos de alta dimensión                                │
│   • Detección de fraude                                                     │
│   • Compresión de datos                                                     │
│   • Preprocesamiento para ML supervisado                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Fundamentos de Clustering](#1-fundamentos)
2. [K-Means Clustering](#2-kmeans)
3. [Clustering Jerárquico](#3-jerarquico)
4. [DBSCAN](#4-dbscan)
5. [PCA - Reducción de Dimensionalidad](#5-pca)
6. [Detección de Anomalías](#6-anomalias)

---

## 1. Fundamentos de Clustering {#1-fundamentos}

### 1.1 ¿Qué es Clustering?

```
CLUSTERING:
───────────

Particionar n objetos en k grupos (clusters) donde:
• Objetos dentro del mismo cluster son SIMILARES
• Objetos en diferentes clusters son DIFERENTES

MEDIDAS DE SIMILITUD/DISTANCIA:
────────────────────────────────

EUCLIDIANA: d(x,y) = √Σ(xᵢ - yᵢ)²
• La más común
• Sensible a escala

MANHATTAN: d(x,y) = Σ|xᵢ - yᵢ|
• Para grids

COSENO: sim(x,y) = (x·y) / (||x|| × ||y||)
• Para texto, documentos
• Ignora magnitud, solo dirección

¿CUÁNTOS CLUSTERS?
• Elbow method
• Silhouette score
• Domain knowledge
```

### 1.2 Métricas de Evaluación

```python
from typing import List, Dict
import math
from collections import defaultdict

def euclidean_distance(x: List[float], y: List[float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x, y)))


def manhattan_distance(x: List[float], y: List[float]) -> float:
    """Manhattan (L1) distance."""
    return sum(abs(xi - yi) for xi, yi in zip(x, y))


def cosine_similarity(x: List[float], y: List[float]) -> float:
    """Cosine similarity between vectors."""
    dot = sum(xi * yi for xi, yi in zip(x, y))
    norm_x = math.sqrt(sum(xi ** 2 for xi in x))
    norm_y = math.sqrt(sum(yi ** 2 for yi in y))
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return dot / (norm_x * norm_y)


def inertia(X: List[List[float]], labels: List[int], 
            centroids: List[List[float]]) -> float:
    """Within-cluster sum of squares (WCSS).
    
    Sum of squared distances from each point to its centroid.
    Lower is better (more compact clusters).
    """
    total = 0.0
    for x, label in zip(X, labels):
        dist = euclidean_distance(x, centroids[label])
        total += dist ** 2
    return total


def silhouette_sample(
    X: List[List[float]], 
    labels: List[int], 
    idx: int
) -> float:
    """Silhouette coefficient for a single sample.
    
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    
    a(i) = average distance to points in same cluster
    b(i) = average distance to points in nearest other cluster
    
    Range: [-1, 1], higher is better
    """
    x = X[idx]
    label = labels[idx]
    
    # Calculate a(i): mean distance to same cluster
    same_cluster = [
        euclidean_distance(x, X[j]) 
        for j in range(len(X)) 
        if labels[j] == label and j != idx
    ]
    a = sum(same_cluster) / len(same_cluster) if same_cluster else 0
    
    # Calculate b(i): min mean distance to other clusters
    other_clusters = set(labels) - {label}
    b = float('inf')
    
    for other_label in other_clusters:
        other_points = [
            euclidean_distance(x, X[j])
            for j in range(len(X))
            if labels[j] == other_label
        ]
        if other_points:
            mean_dist = sum(other_points) / len(other_points)
            b = min(b, mean_dist)
    
    if b == float('inf'):
        b = 0
    
    if max(a, b) == 0:
        return 0.0
    return (b - a) / max(a, b)


def silhouette_score(X: List[List[float]], labels: List[int]) -> float:
    """Average silhouette coefficient for all samples."""
    scores = [silhouette_sample(X, labels, i) for i in range(len(X))]
    return sum(scores) / len(scores)
```

---

## 2. K-Means Clustering {#2-kmeans}

### 2.1 Algoritmo

```
K-MEANS ALGORITHM:
──────────────────

1. INICIALIZAR: Elegir k centroides aleatorios
2. ASIGNAR: Cada punto al centroide más cercano
3. ACTUALIZAR: Mover cada centroide al promedio de sus puntos
4. REPETIR 2-3 hasta convergencia

CONVERGENCIA:
• Las asignaciones no cambian, O
• Centroides se mueven menos que ε

COMPLEJIDAD: O(n × k × d × i)
• n = puntos
• k = clusters
• d = dimensiones
• i = iteraciones
```

### 2.2 Implementación

```python
import random

class KMeans:
    """K-Means clustering algorithm.
    
    Partitions n samples into k clusters by minimizing
    within-cluster sum of squares (inertia).
    """
    
    def __init__(self, n_clusters: int = 3, max_iterations: int = 100,
                 tolerance: float = 1e-4, random_state: int = None):
        self.k = n_clusters
        self.max_iter = max_iterations
        self.tol = tolerance
        self.random_state = random_state
        self.centroids: List[List[float]] = []
        self.labels: List[int] = []
        self.inertia_: float = 0.0
    
    def _init_centroids(self, X: List[List[float]]) -> List[List[float]]:
        """Initialize centroids randomly from data points."""
        if self.random_state is not None:
            random.seed(self.random_state)
        
        indices = random.sample(range(len(X)), self.k)
        return [X[i][:] for i in indices]  # Copy points
    
    def _assign_clusters(self, X: List[List[float]]) -> List[int]:
        """Assign each point to nearest centroid."""
        labels = []
        for x in X:
            distances = [euclidean_distance(x, c) for c in self.centroids]
            labels.append(distances.index(min(distances)))
        return labels
    
    def _update_centroids(
        self, 
        X: List[List[float]], 
        labels: List[int]
    ) -> List[List[float]]:
        """Update centroids to mean of assigned points."""
        n_features = len(X[0])
        new_centroids = []
        
        for k in range(self.k):
            cluster_points = [X[i] for i in range(len(X)) if labels[i] == k]
            
            if cluster_points:
                centroid = [
                    sum(p[d] for p in cluster_points) / len(cluster_points)
                    for d in range(n_features)
                ]
            else:
                # Empty cluster: keep old centroid
                centroid = self.centroids[k]
            
            new_centroids.append(centroid)
        
        return new_centroids
    
    def _centroid_shift(
        self, 
        old: List[List[float]], 
        new: List[List[float]]
    ) -> float:
        """Calculate total movement of centroids."""
        return sum(euclidean_distance(o, n) for o, n in zip(old, new))
    
    def fit(self, X: List[List[float]]) -> 'KMeans':
        """Fit K-Means to data."""
        self.centroids = self._init_centroids(X)
        
        for _ in range(self.max_iter):
            # Assign points to clusters
            self.labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, self.labels)
            
            # Check convergence
            shift = self._centroid_shift(self.centroids, new_centroids)
            self.centroids = new_centroids
            
            if shift < self.tol:
                break
        
        # Calculate final inertia
        self.inertia_ = inertia(X, self.labels, self.centroids)
        
        return self
    
    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict cluster for new points."""
        return self._assign_clusters(X)
    
    def fit_predict(self, X: List[List[float]]) -> List[int]:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels


def elbow_method(X: List[List[float]], max_k: int = 10) -> List[float]:
    """Calculate inertia for different k values.
    
    Plot inertia vs k to find "elbow" point.
    """
    inertias = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return inertias
```

### 2.3 K-Means++: Mejor Inicialización

```python
def kmeans_plus_plus_init(X: List[List[float]], k: int) -> List[List[float]]:
    """K-Means++ initialization.
    
    Spreads initial centroids by selecting them with probability
    proportional to their distance from existing centroids.
    
    Improves convergence and final clustering quality.
    """
    n = len(X)
    centroids = []
    
    # First centroid: random
    centroids.append(X[random.randint(0, n - 1)][:])
    
    for _ in range(1, k):
        # Calculate distance to nearest centroid for each point
        distances = []
        for x in X:
            min_dist = min(euclidean_distance(x, c) for c in centroids)
            distances.append(min_dist ** 2)  # Square for probability
        
        # Sample with probability proportional to distance²
        total = sum(distances)
        probs = [d / total for d in distances]
        
        r = random.random()
        cumulative = 0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                centroids.append(X[i][:])
                break
    
    return centroids
```

---

## 3. Clustering Jerárquico {#3-jerarquico}

### 3.1 Concepto

```
CLUSTERING JERÁRQUICO:
──────────────────────

AGLOMERATIVO (Bottom-up):
1. Empezar: cada punto es su propio cluster
2. Fusionar: los dos clusters más cercanos
3. Repetir hasta tener k clusters (o 1)

DIVISIVO (Top-down):
1. Empezar: todos los puntos en un cluster
2. Dividir: el cluster más grande
3. Repetir hasta tener k clusters

LINKAGE (Distancia entre clusters):
• SINGLE: min distancia entre puntos
• COMPLETE: max distancia entre puntos
• AVERAGE: promedio de distancias
• WARD: minimiza varianza al fusionar

DENDROGRAMA:
Visualización del proceso de fusión/división
```

### 3.2 Implementación Aglomerativa

```python
class AgglomerativeClustering:
    """Agglomerative (bottom-up) hierarchical clustering.
    
    Uses single linkage by default.
    """
    
    def __init__(self, n_clusters: int = 2, linkage: str = 'single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels: List[int] = []
        self.merge_history: List[tuple] = []
    
    def _cluster_distance(
        self, 
        cluster1: List[int], 
        cluster2: List[int], 
        distances: List[List[float]]
    ) -> float:
        """Calculate distance between two clusters."""
        if self.linkage == 'single':
            # Minimum distance between any two points
            return min(
                distances[i][j] 
                for i in cluster1 
                for j in cluster2
            )
        elif self.linkage == 'complete':
            # Maximum distance
            return max(
                distances[i][j] 
                for i in cluster1 
                for j in cluster2
            )
        elif self.linkage == 'average':
            # Average distance
            total = sum(
                distances[i][j] 
                for i in cluster1 
                for j in cluster2
            )
            return total / (len(cluster1) * len(cluster2))
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")
    
    def fit(self, X: List[List[float]]) -> 'AgglomerativeClustering':
        """Build hierarchical clustering."""
        n = len(X)
        
        # Precompute pairwise distances
        distances = [
            [euclidean_distance(X[i], X[j]) for j in range(n)]
            for i in range(n)
        ]
        
        # Initialize: each point is its own cluster
        clusters = [[i] for i in range(n)]
        
        while len(clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(
                        clusters[i], clusters[j], distances
                    )
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Record merge
            self.merge_history.append((merge_i, merge_j, min_dist))
            
            # Merge clusters
            clusters[merge_i] = clusters[merge_i] + clusters[merge_j]
            clusters.pop(merge_j)
        
        # Assign labels
        self.labels = [0] * n
        for cluster_idx, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels[point_idx] = cluster_idx
        
        return self
    
    def fit_predict(self, X: List[List[float]]) -> List[int]:
        """Fit and return labels."""
        self.fit(X)
        return self.labels
```

---

## 4. DBSCAN {#4-dbscan}

### 4.1 Concepto

```
DBSCAN (Density-Based Spatial Clustering):
──────────────────────────────────────────

NO requiere especificar número de clusters.
Encuentra clusters de forma arbitraria.
Detecta outliers automáticamente.

PARÁMETROS:
• ε (eps): Radio del vecindario
• min_samples: Mínimo puntos para ser core point

TIPOS DE PUNTOS:
• CORE: Tiene ≥ min_samples vecinos en radio ε
• BORDER: No es core, pero es vecino de core
• NOISE: No es core ni border (outlier)

ALGORITMO:
1. Para cada punto sin visitar:
   a. Si es core: crear nuevo cluster, expandir
   b. Si no: marcar como ruido (puede cambiar a border)
```

### 4.2 Implementación

```python
class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise.
    
    Finds clusters based on density, automatically detecting outliers.
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels: List[int] = []
    
    def _get_neighbors(
        self, 
        X: List[List[float]], 
        idx: int
    ) -> List[int]:
        """Find all points within eps distance."""
        neighbors = []
        for i in range(len(X)):
            if euclidean_distance(X[idx], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def fit(self, X: List[List[float]]) -> 'DBSCAN':
        """Fit DBSCAN clustering."""
        n = len(X)
        self.labels = [-1] * n  # -1 = unassigned/noise
        
        cluster_id = 0
        
        for i in range(n):
            if self.labels[i] != -1:
                continue  # Already processed
            
            neighbors = self._get_neighbors(X, i)
            
            if len(neighbors) < self.min_samples:
                # Noise point (might become border later)
                continue
            
            # Start new cluster
            self.labels[i] = cluster_id
            
            # Expand cluster
            seed_set = neighbors[:]
            j = 0
            
            while j < len(seed_set):
                q = seed_set[j]
                
                if self.labels[q] == -1:
                    # Was noise, now border
                    self.labels[q] = cluster_id
                
                if self.labels[q] != -1 and self.labels[q] != cluster_id:
                    # Already in another cluster
                    j += 1
                    continue
                
                if self.labels[q] == -1 or self.labels[q] == cluster_id:
                    self.labels[q] = cluster_id
                    
                    q_neighbors = self._get_neighbors(X, q)
                    
                    if len(q_neighbors) >= self.min_samples:
                        # q is also core point
                        for neighbor in q_neighbors:
                            if neighbor not in seed_set:
                                seed_set.append(neighbor)
                
                j += 1
            
            cluster_id += 1
        
        return self
    
    def fit_predict(self, X: List[List[float]]) -> List[int]:
        """Fit and return labels (-1 for noise)."""
        self.fit(X)
        return self.labels
```

---

## 5. PCA - Reducción de Dimensionalidad {#5-pca}

### 5.1 Concepto

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PRINCIPAL COMPONENT ANALYSIS (PCA)                                        │
│   ───────────────────────────────────                                       │
│                                                                             │
│   OBJETIVO: Reducir dimensiones preservando máxima varianza                 │
│                                                                             │
│   DATOS ORIGINALES (2D):         PROYECCIÓN (1D):                           │
│         y                                                                   │
│         ▲    * *                       PC1                                  │
│         │   * * *                 ─────●──●──●──●──●─────▶                 │
│         │  * * *                       (máxima varianza)                    │
│         │ * * *                                                             │
│         └──────────▶ x                                                     │
│                                                                             │
│   PASOS:                                                                    │
│   1. Centrar datos (restar media)                                           │
│   2. Calcular matriz de covarianza                                          │
│   3. Encontrar eigenvectors/eigenvalues                                     │
│   4. Ordenar por eigenvalue (mayor = más varianza)                          │
│   5. Proyectar datos en top-k eigenvectors                                  │
│                                                                             │
│   APLICACIONES:                                                             │
│   • Visualización (reducir a 2D/3D)                                         │
│   • Compresión de datos                                                     │
│   • Preprocesamiento para ML (reducir ruido)                                │
│   • Decorrelacionar features                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Implementación Simplificada

```python
class PCA:
    """Principal Component Analysis for dimensionality reduction.
    
    Simplified implementation using power iteration for eigenvectors.
    """
    
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components: List[List[float]] = []  # Principal components
        self.mean: List[float] = []
        self.explained_variance: List[float] = []
    
    def _center_data(self, X: List[List[float]]) -> List[List[float]]:
        """Center data by subtracting mean."""
        n, d = len(X), len(X[0])
        
        # Calculate mean
        self.mean = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
        
        # Subtract mean
        centered = [
            [X[i][j] - self.mean[j] for j in range(d)]
            for i in range(n)
        ]
        
        return centered
    
    def _covariance_matrix(self, X: List[List[float]]) -> List[List[float]]:
        """Compute covariance matrix."""
        n, d = len(X), len(X[0])
        
        cov = [[0.0] * d for _ in range(d)]
        
        for i in range(d):
            for j in range(d):
                cov[i][j] = sum(X[k][i] * X[k][j] for k in range(n)) / (n - 1)
        
        return cov
    
    def _power_iteration(
        self, 
        matrix: List[List[float]], 
        n_iterations: int = 100
    ) -> tuple[List[float], float]:
        """Find dominant eigenvector using power iteration."""
        d = len(matrix)
        
        # Random initial vector
        v = [random.random() for _ in range(d)]
        norm = math.sqrt(sum(x ** 2 for x in v))
        v = [x / norm for x in v]
        
        for _ in range(n_iterations):
            # Matrix-vector multiplication
            v_new = [
                sum(matrix[i][j] * v[j] for j in range(d))
                for i in range(d)
            ]
            
            # Normalize
            norm = math.sqrt(sum(x ** 2 for x in v_new))
            if norm == 0:
                break
            v_new = [x / norm for x in v_new]
            
            v = v_new
        
        # Rayleigh quotient for eigenvalue
        Av = [sum(matrix[i][j] * v[j] for j in range(d)) for i in range(d)]
        eigenvalue = sum(v[i] * Av[i] for i in range(d))
        
        return v, eigenvalue
    
    def _deflate(
        self, 
        matrix: List[List[float]], 
        eigenvector: List[float], 
        eigenvalue: float
    ) -> List[List[float]]:
        """Remove contribution of found eigenvector from matrix."""
        d = len(matrix)
        deflated = [row[:] for row in matrix]
        
        for i in range(d):
            for j in range(d):
                deflated[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]
        
        return deflated
    
    def fit(self, X: List[List[float]]) -> 'PCA':
        """Fit PCA to data."""
        # Center data
        X_centered = self._center_data(X)
        
        # Compute covariance matrix
        cov = self._covariance_matrix(X_centered)
        
        # Find principal components using power iteration + deflation
        self.components = []
        self.explained_variance = []
        
        for _ in range(self.n_components):
            eigenvector, eigenvalue = self._power_iteration(cov)
            self.components.append(eigenvector)
            self.explained_variance.append(eigenvalue)
            cov = self._deflate(cov, eigenvector, eigenvalue)
        
        return self
    
    def transform(self, X: List[List[float]]) -> List[List[float]]:
        """Project data onto principal components."""
        # Center data using fitted mean
        d = len(X[0])
        X_centered = [
            [X[i][j] - self.mean[j] for j in range(d)]
            for i in range(len(X))
        ]
        
        # Project onto components
        transformed = []
        for x in X_centered:
            projection = [
                sum(x[j] * self.components[k][j] for j in range(d))
                for k in range(self.n_components)
            ]
            transformed.append(projection)
        
        return transformed
    
    def fit_transform(self, X: List[List[float]]) -> List[List[float]]:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def explained_variance_ratio(self) -> List[float]:
        """Proportion of variance explained by each component."""
        total = sum(self.explained_variance)
        return [v / total for v in self.explained_variance]
```

---

## 6. Detección de Anomalías {#6-anomalias}

### 6.1 Métodos

```
DETECCIÓN DE ANOMALÍAS:
───────────────────────

BASADO EN ESTADÍSTICAS:
• Z-score: puntos con |z| > 3 son outliers
• IQR: puntos fuera de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

BASADO EN DENSIDAD:
• Local Outlier Factor (LOF)
• DBSCAN (noise points = outliers)

BASADO EN DISTANCIA:
• Puntos lejos de sus vecinos
• Isolation Forest

BASADO EN CLUSTERING:
• Puntos lejos de todos los centroides
• Clusters muy pequeños
```

### 6.2 Implementación de LOF

```python
class LocalOutlierFactor:
    """Local Outlier Factor for anomaly detection.
    
    Compares local density of a point to its neighbors.
    High LOF = lower density than neighbors = potential outlier.
    """
    
    def __init__(self, n_neighbors: int = 20, threshold: float = 1.5):
        self.k = n_neighbors
        self.threshold = threshold
        self.lof_scores: List[float] = []
    
    def _k_neighbors(
        self, 
        X: List[List[float]], 
        idx: int
    ) -> List[tuple]:
        """Find k nearest neighbors and their distances."""
        distances = [
            (i, euclidean_distance(X[idx], X[i]))
            for i in range(len(X)) if i != idx
        ]
        distances.sort(key=lambda x: x[1])
        return distances[:self.k]
    
    def _reachability_distance(
        self, 
        X: List[List[float]], 
        idx: int, 
        neighbor_idx: int,
        k_distances: Dict[int, float]
    ) -> float:
        """Reachability distance from idx to neighbor.
        
        max(k-distance(neighbor), actual_distance)
        """
        actual_dist = euclidean_distance(X[idx], X[neighbor_idx])
        return max(k_distances[neighbor_idx], actual_dist)
    
    def _local_reachability_density(
        self, 
        X: List[List[float]], 
        idx: int,
        neighbors: Dict[int, List[tuple]],
        k_distances: Dict[int, float]
    ) -> float:
        """LRD = inverse of average reachability distance to neighbors."""
        neighbor_list = neighbors[idx]
        
        if not neighbor_list:
            return 0.0
        
        total_reach_dist = sum(
            self._reachability_distance(X, idx, n_idx, k_distances)
            for n_idx, _ in neighbor_list
        )
        
        if total_reach_dist == 0:
            return float('inf')
        
        return len(neighbor_list) / total_reach_dist
    
    def fit_predict(self, X: List[List[float]]) -> List[int]:
        """Calculate LOF and return outlier labels.
        
        Returns 1 for inliers, -1 for outliers.
        """
        n = len(X)
        
        # Find k neighbors for all points
        neighbors = {}
        k_distances = {}
        
        for i in range(n):
            knn = self._k_neighbors(X, i)
            neighbors[i] = knn
            k_distances[i] = knn[-1][1] if knn else 0  # Distance to k-th neighbor
        
        # Calculate LRD for all points
        lrd = {}
        for i in range(n):
            lrd[i] = self._local_reachability_density(
                X, i, neighbors, k_distances
            )
        
        # Calculate LOF
        self.lof_scores = []
        
        for i in range(n):
            neighbor_list = neighbors[i]
            
            if not neighbor_list or lrd[i] == 0:
                self.lof_scores.append(1.0)
                continue
            
            # LOF = average ratio of neighbor's LRD to own LRD
            lof = sum(
                lrd[n_idx] / lrd[i] if lrd[i] != float('inf') else 0
                for n_idx, _ in neighbor_list
            ) / len(neighbor_list)
            
            self.lof_scores.append(lof)
        
        # Classify as outlier if LOF > threshold
        labels = [
            -1 if score > self.threshold else 1 
            for score in self.lof_scores
        ]
        
        return labels
```

---

## ⚠️ Cuándo Usar Cada Algoritmo

```
GUÍA DE SELECCIÓN:
──────────────────

K-MEANS:
• Clusters esféricos de tamaño similar
• Conoces número de clusters
• Datos grandes (escalable)

HIERARCHICAL:
• Quieres dendrograma
• No conoces k
• Datos pequeños/medianos

DBSCAN:
• Clusters de forma arbitraria
• Hay outliers
• No conoces k

PCA:
• Reducir dimensionalidad
• Visualización
• Decorrelacionar features
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 23.1: K-Means en datos sintéticos
Crear blobs y encontrar clusters óptimos con elbow method.

### Ejercicio 23.2: Comparar algoritmos
K-Means vs DBSCAN en datos con forma de luna.

### Ejercicio 23.3: PCA para visualización
Reducir MNIST a 2D y visualizar dígitos.

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html) | Docs | 🔴 Obligatorio |
| [StatQuest: PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ) | Video | 🔴 Obligatorio |
| [Visualizing DBSCAN](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) | Interactivo | 🟡 Recomendado |

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [22_ML_SUPERVISADO](22_ML_SUPERVISADO.md) | [00_INDICE](00_INDICE.md) | [24_INTRO_DEEP_LEARNING](24_INTRO_DEEP_LEARNING.md) |
