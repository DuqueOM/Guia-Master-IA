# Módulo 06 - Unsupervised Learning

> **🎯 Objetivo:** Dominar K-Means clustering y PCA para reducción dimensional
> **Fase:** 2 - Núcleo de ML | **Semanas 13-16**
> **Curso del Pathway:** Unsupervised Algorithms in Machine Learning

---

<a id="m06-0"></a>

## 🧭 Cómo usar este módulo (modo 0→100)

**Propósito:** que puedas:

- encontrar estructura sin etiquetas (clustering)
- reducir dimensionalidad con rigor (PCA)
- decidir cuándo NO usar estos métodos

### Objetivos de aprendizaje (medibles)

Al terminar este módulo podrás:

- **Implementar** K-Means (Lloyd + K-Means++).
- **Evaluar** clustering con inercia/codo y silhouette (entendiendo limitaciones).
- **Implementar** PCA con SVD y usar varianza explicada para elegir `n_components`.
- **Diagnosticar** cuándo K-Means/PCA fallan y proponer alternativas.

Enlaces rápidos:

- [GLOSARIO.md](GLOSARIO.md)
- [RECURSOS.md](RECURSOS.md)
- [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
- [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Recursos (cuándo usarlos)

| Prioridad | Recurso | Cuándo usarlo en este módulo | Para qué |
|----------|---------|------------------------------|----------|
| **Obligatorio** | `study_tools/DRILL_DIMENSIONES_NUMPY.md` | Semana 13–16, cada vez que implementes distancias/proyecciones y se rompan shapes | Evitar errores silenciosos en broadcasting/`axis` |
| **Obligatorio** | `study_tools/DIARIO_ERRORES.md` | Cuando K-Means produzca clusters vacíos, `NaN` o PCA devuelva resultados inestables | Registrar el caso y dejarlo “debuggeable” |
| **Complementario** | [3Blue1Brown: Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Semana 15 (PCA), al ver varianza/proyecciones/autovectores | Intuición visual para PCA |
| **Complementario** | [VisuAlgo](https://visualgo.net/en) | Semana 13–14, al estudiar el comportamiento iterativo de K-Means y su sensibilidad a inicialización | Visualizar algoritmos paso a paso para construir intuición |
| **Complementario** | [Mathematics for ML (book)](https://mml-book.github.io/) | Semana 15–16, al formalizar covarianza, eigen/SVD | Notación y derivaciones más rigurosas |
| **Opcional** | [RECURSOS.md](RECURSOS.md) | Al terminar el módulo (para profundizar en clustering/reducción dimensional) | Elegir material extra sin dispersarte |

---

## 🧠 ¿Qué es Unsupervised Learning?

```text
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

## 🧩 Micro-Capítulo Maestro: PCA (Principal Component Analysis) — Nivel: Avanzado

### 1) Intuición: la mejor foto

Imagina que tienes un objeto 3D (tus datos en alta dimensión) y solo puedes tomar una “foto” en 2D.

- Si tomas la foto desde un ángulo malo, la sombra se ve “aplastada” y pierdes estructura.
- Si tomas la foto desde el ángulo correcto, la sombra conserva la mayor cantidad de información posible.

PCA busca matemáticamente ese ángulo: la proyección donde la **varianza proyectada** es máxima.

### 2) Derivación lógica (covarianza → eigen)

1) **Centrar**

Mueves el origen para que el promedio sea 0:

`X_c = X - μ`

2) **Covarianza**

La matriz de covarianza captura cómo “se estiran” los datos:

`Σ = (1/(n-1)) X_c^T X_c`

Si (en 2D) `Σ = [[10, 0],[0, 1]]`, significa: hay mucha más varianza en X que en Y.

3) **Eigenvectors y eigenvalues**

- Los **eigenvectors** de `Σ` apuntan en direcciones principales de estiramiento.
- Los **eigenvalues** dicen cuánta varianza hay en esas direcciones.

PCA elige los eigenvectors con eigenvalues más grandes y proyecta ahí.

### 3) Por qué SVD suele ser mejor que eigen en código

Si calculas `X_c^T X_c` puedes amplificar problemas numéricos (estás “cuadrando” escalas).

En cambio, con SVD:

`X_c = U S V^T`

se obtienen las componentes principales directamente desde `V` sin formar explícitamente `Σ`.

Regla práctica:

- **En teoría:** PCA = eigen de la covarianza.
- **En práctica:** PCA = SVD de `X_c` (más estable; es lo que usan implementaciones modernas).

---

## 💻 Parte 1: K-Means Clustering

### 1.0 K-Means — Nivel: intermedio (core del Pathway)

**Propósito:** pasar de “sé que K-Means agrupa puntos” a **poder implementarlo desde cero, elegir `k` con criterio y detectar cuándo NO usarlo**.

#### Objetivos de aprendizaje (medibles)

Al terminar esta parte podrás:

- **Explicar** la función objetivo de K-Means (inercia) y por qué usa distancia euclidiana.
- **Aplicar** el algoritmo de Lloyd (asignar → actualizar → repetir) y reconocer convergencia.
- **Implementar** K-Means++ y justificar por qué mejora la inicialización.
- **Analizar** fallas típicas: clusters vacíos, sensibilidad a escala, mínimos locales.
- **Evaluar** resultados usando inercia y silhouette (y entender limitaciones de ambas).

#### Prerrequisitos

- De `Módulo 01`: NumPy (broadcasting, `axis`, shapes).
- De `Módulo 02`: norma L2 / distancia euclidiana.

Enlaces rápidos:

- [GLOSARIO: K-Means](GLOSARIO.md#k-means)
- [GLOSARIO: K-Means++](GLOSARIO.md#k-means-1)
- [GLOSARIO: Inertia](GLOSARIO.md#inertia)
- [GLOSARIO: Clustering](GLOSARIO.md#clustering)

#### Resumen ejecutivo (big idea)

K-Means alterna dos pasos que **siempre reducen (o no aumentan)** la inercia:

- **Asignación:** cada punto va al centroide más cercano.
- **Actualización:** cada centroide se mueve al promedio de sus puntos.

Esto garantiza que el algoritmo converge (en iteraciones finitas), pero **no garantiza el mínimo global**: por eso la inicialización (K-Means++) importa.

#### Intuición → formalización

##### a) Intuición

K-Means intenta poner `k` “imanes” (centroides) y moverlos hasta que cada imán represente bien a los puntos que atrajo.

##### a.1 Intuición geométrica: Voronoi tessellation (territorios)

Una forma visual de entender K-Means:

- pones `k` centroides como “semillas” en el plano
- cada semilla **reclama el territorio** de los puntos más cercanos

Eso induce un particionado del espacio en **celdas de Voronoi**: regiones poligonales donde todos los puntos están más cerca de un centroide que de cualquier otro.

En cada iteración de Lloyd:

- **Asignación:** recalculas las celdas (quién pertenece a quién)
- **Actualización:** cada semilla se mueve al centro de masa de su celda

##### b) Formalización

Función objetivo:

`J = Σᵢ Σ_{x∈Cᵢ} ||x - μᵢ||²`

Donde:

- `μᵢ` es el centroide del cluster `i`.
- `Cᵢ` es el conjunto de puntos asignados al cluster `i`.

##### c) Condiciones donde K-Means funciona bien

- clusters “redondos” / aproximadamente esféricos
- tamaños similares
- distancia euclidiana tiene sentido (features en la misma escala)

##### d) Casos donde falla (y cómo detectarlo)

- clusters alargados/no convexos (ej.: “dos lunas”)
- escalas distintas sin normalizar (una feature domina)
- outliers fuertes arrastran centroides

#### Actividades activas (aprendizaje activo)

- **Retrieval practice (3–5 min):** escribe sin mirar:
  - los dos pasos del algoritmo de Lloyd
  - la función objetivo `J`
- **Ejercicio de diagnóstico:** crea 2 features con escalas distintas y observa cómo cambia el clustering si normalizas.

#### Debugging / validación (v5)

- Si obtienes resultados raros, revisa primero:
  - shapes (`X: (n_samples, n_features)`, `centroids: (k, n_features)`, `labels: (n_samples,)`)
  - `NaN` por clusters vacíos (centroide sin puntos)
- Registra hallazgos en `study_tools/DIARIO_ERRORES.md`.
- Antes de usar un dataset real “sucio”, aplica `study_tools/DIRTY_DATA_CHECK.md`.
- Para integrar el protocolo completo:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

#### Cheat sheet (repaso rápido)

- **Paso 1:** `labels = argmin(||x - μᵢ||²)`
- **Paso 2:** `μᵢ = mean(points_in_cluster_i)`
- **Convergencia:** `||μ_new - μ_old||² < tol`
- **Riesgo:** mínimos locales → usar K-Means++ y/o múltiples inicializaciones

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

### 2.0 Evaluación — cómo decidir si el clustering “tiene sentido”

**Propósito:** evitar el error común de “K-Means siempre devuelve clusters, entonces siempre sirve”. Aquí aprendes a **medir calidad** y a entender por qué esas métricas pueden engañar.

#### Objetivos de aprendizaje (medibles)

- **Explicar** qué mide la inercia y por qué siempre baja al subir `k`.
- **Aplicar** el método del codo como heurística (no como regla matemática).
- **Interpretar** silhouette score (qué significa cerca de 1, 0 y valores negativos).
- **Analizar** cuándo no puedes validar bien (porque no hay ground truth).

Enlaces rápidos:

- [GLOSARIO: Inertia](GLOSARIO.md#inertia)
- [GLOSARIO: Silhouette Score](GLOSARIO.md#silhouette-score)

#### Resumen ejecutivo

- **Inercia:** mide compactación interna; útil para comparar `k`, pero sesgada (siempre favorece `k` grande).
- **Silhouette:** mezcla cohesión y separación; útil para comparar modelos, pero costosa de calcular de forma exacta.

#### Actividades activas

- Ejecuta elbow + silhouette sobre el mismo dataset y escribe una conclusión:
  - ¿coinciden en el `k`?
  - si no coinciden, ¿por qué podría pasar?

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

### 3.0 PCA — Nivel: intermedio (reducción dimensional con rigor)

**Propósito:** pasar de “PCA reduce dimensiones” a **poder derivar su lógica, implementarlo con SVD y usar varianza explicada para tomar decisiones**.

#### Objetivos de aprendizaje (medibles)

Al terminar esta parte podrás:

- **Explicar** por qué PCA encuentra direcciones de máxima varianza (y qué NO significa eso).
- **Aplicar** el pipeline correcto: centrar → descomponer (SVD) → proyectar → reconstruir.
- **Implementar** PCA con SVD y calcular `explained_variance_ratio_`.
- **Elegir** `n_components` por varianza acumulada y justificar el trade-off.
- **Diagnosticar** errores típicos: no centrar datos, confundir componentes con scores, reconstrucción incorrecta.

#### Motivación / por qué importa

En la mayoría de los problemas reales, la intuición visual se pierde en espacios de alta dimensión (ej.: cientos o miles de features). PCA te permite:

- **Visualizar** en 2D/3D sin tirar información “a ojo”.
- **Eliminar ruido** (quedándote con las direcciones dominantes de variación).
- **Comprimir** (reconstruir aproximaciones controlando el error).

Regla práctica: PCA no “encuentra lo que separa clases”; encuentra lo que **más varía**.

#### Prerrequisitos

- De `Módulo 02`: SVD (intuición) y producto matricial.
- De `Módulo 02`: matriz de covarianza, eigenvalues y eigenvectors.
- De `Módulo 02`: proyección (producto punto) y norma.
- De `Módulo 01`: manipulación de shapes y `axis`.

Enlaces rápidos:

- [GLOSARIO: PCA](GLOSARIO.md#pca-principal-component-analysis)
- [GLOSARIO: SVD](GLOSARIO.md#svd-singular-value-decomposition)
- [RECURSOS.md](RECURSOS.md)

#### Resumen ejecutivo (big idea)

PCA crea un nuevo sistema de coordenadas donde:

- el eje 1 (PC1) captura la mayor varianza,
- el eje 2 (PC2) captura la mayor varianza restante, y así sucesivamente,

y luego te permite quedarte con los primeros `k` ejes para comprimir.

#### Intuición → formalización

##### a) Intuición

Si tus datos viven cerca de un plano dentro de un espacio 100D, PCA intenta encontrar ese plano (o subespacio) para representar los datos con menos números.

Analogía: “buscar el mejor ángulo para tomar una foto”

- Tienes un objeto 3D (tus datos en alta dimensión).
- Una foto 2D pierde información.
- PCA elige el **ángulo de cámara** que preserva la mayor “información” medible como **varianza**.

Metáfora complementaria (baguette): imagina una nube de puntos alargada como una baguette flotando en 3D. Si tomas la foto desde la punta, parece un círculo (pierdes estructura). Si la tomas de lado, ves su longitud real. PCA busca ese “lado” matemáticamente.

Ojo: “más varianza” no significa “más útil para clasificar”; solo significa “más dispersión”.

##### b) Conceptos clave (glosario mínimo)

- **Varianza:** dispersión de los datos; PCA busca maximizarla *después* de proyectar.
- **Matriz de covarianza (`Σ`):** matriz simétrica que describe cómo varían las variables y cómo co-varían entre sí.
- **Eigenvector (vector propio):** dirección que no cambia (salvo escala) al aplicar `Σ`; en PCA, son los ejes principales.
- **Eigenvalue (valor propio):** varianza capturada en la dirección de su eigenvector.
- **Componente principal:** eje (eigenvector) ordenado por eigenvalue descendente.

##### c) Formalización mínima

- Centrar: `X_c = X - mean(X)`
- SVD: `X_c = U S Vᵀ`
- Componentes principales: columnas de `V` (o filas de `Vᵀ`)
- Proyección a `k` componentes: `Z = X_c @ V_k`
- Reconstrucción: `X_hat = Z @ V_kᵀ + mean`

##### c.1 Maximizando la varianza (derivación lógica → ecuación de eigenvalores)

Idea: buscas un vector unitario `u` (dirección) tal que la varianza de la proyección `uᵀx` sea máxima.

Si `x` está centrado, la varianza proyectada es:

`Var(uᵀx) = uᵀ Σ u`

Planteas el problema:

`max_u  uᵀ Σ u   s.a.  ||u||₂ = 1`

Con multiplicadores de Lagrange, la condición de óptimo lleva a:

`Σu = λu`

Interpretación directa:

- `u` es un componente principal.
- `λ` es la varianza capturada por ese componente.

##### c.2 Relación SVD ↔ eigenvalues (por qué SVD es el método preferido)

Si `X_c` son los datos centrados y haces:

```
X_c = U S Vᵀ
```

Entonces la covarianza muestral es:

```
Σ = (1/(n-1)) X_cᵀ X_c
  = (1/(n-1)) (V S Uᵀ)(U S Vᵀ)
  = V (S²/(n-1)) Vᵀ
```

Conclusión:

- **Los eigenvectors de `Σ`** son las columnas de `V`.
- **Los eigenvalues de `Σ`** son `S²/(n-1)`.

Esto conecta directamente con `Módulo 02` (eigenvalues/eigenvectors) y explica por qué PCA “vía SVD” suele ser más estable.

##### c.3 Worked example: PCA manual en 2D (rotación de ejes)

Supón datos 2D que “viven” casi sobre la diagonal `y = x`.

1) Centrar los datos:

```
X_c = X - mean(X)
```

2) Imagina que la covarianza queda (caso idealizado):

```
Σ = [[1, 1],
     [1, 1]]
```

3) Sus eigenvectors (direcciones principales) son:

- `v1 = (1, 1)/√2`  (dirección diagonal)
- `v2 = (1, -1)/√2` (dirección anti-diagonal)

Y sus eigenvalues:

- `λ1 = 2` (mucha varianza en la diagonal)
- `λ2 = 0` (casi nada en la anti-diagonal)

4) Proyección a 1D:

```
z = X_c @ v1
```

Interpretación: rotaste ejes y te quedaste solo con el eje donde “vive” casi toda la variación.

##### c.4 Worked example (numérico): covarianza y primer componente a mano

Datos centrados (6 puntos):

```text
X = [(-1,-1), (-2,-1), (-3,-2), (1,1), (2,1), (3,2)]
```

1) Construye `X` como matriz `(n_samples, 2)` y calcula:

`Σ = (1/(n-1)) Xᵀ X`

Aquí:

```text
XᵀX = [[28, 18],
       [18, 12]]
n-1 = 5
```

Por tanto:

```text
Σ = [[5.6, 3.6],
     [3.6, 2.4]]
```

2) Eigenvalues/eigenvectors (aprox.)

- `λ1 ≈ 7.94`, `λ2 ≈ 0.06`
- primer eigenvector (normalizado) `u1 ≈ (0.84, 0.55)`

3) Varianza explicada del primer componente:

`λ1/(λ1+λ2) ≈ 7.94/8.00 ≈ 99.3%`

Lectura: la nube está casi en una línea; proyectar a 1D conserva casi toda la estructura.

#### Algoritmo (paso a paso)

1) Centrar (y típicamente escalar si tus features tienen unidades distintas).
2) SVD de `X_c` (recomendado) o eigen de `Σ`.
3) Elegir `k` por varianza acumulada (y/o error de reconstrucción).
4) Proyectar `Z = X_c @ V_k`.
5) (Opcional) Reconstruir `X_hat = Z @ V_kᵀ + mean` para medir pérdida.

#### Implementación práctica (código)

En esta guía ya tienes:

- `pca_eigen(...)` en **3.2** (útil para entender la teoría).
- `pca_svd(...)` y la clase `PCA` en **3.3–3.4** (recomendado para práctica).

#### Evaluación formativa (rápida)

Pregunta: si `λ1 = 9` y `λ2 = 1`, ¿qué proporción de varianza captura PC1?

Respuesta: `9/(9+1) = 90%`.

#### Actividades activas

- **Retrieval practice:** escribe las 4 ecuaciones (centrar, SVD, proyectar, reconstruir).
- **Experimento mínimo:** genera datos 3D correlacionados, reduce a 1D y reporta:
  - varianza explicada
  - error de reconstrucción

#### Errores comunes

- **No centrar**: PCA se sesga hacia la media (resultado incorrecto).
- **Confundir `components` vs `X_pca`**: componentes son ejes; `X_pca` son coordenadas en esos ejes.
- **Elegir `n_components` “a ojo”**: usar varianza acumulada.

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

## 🧩 Consolidación (PCA)

### Errores comunes

- **No centrar:** si no restas la media, el primer componente puede capturar “offset” en vez de estructura.
- **Confundir `components` con `X_pca`:**
  - `components` = ejes
  - `X_pca` = coordenadas en esos ejes
- **Elegir `n_components` sin criterio:** usa varianza acumulada + error de reconstrucción.

### Debugging / validación (v5)

- Verifica:
  - `X_centered.mean(axis=0)` cerca de 0
  - shapes: `components: (n_features, k)`, `X_pca: (n_samples, k)`
- Si tu reconstrucción explota, revisa `X_hat = Z @ V_kᵀ + mean`.
- Registra hallazgos en `study_tools/DIARIO_ERRORES.md`.
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Reto Feynman (tablero blanco)

Explica en 5 líneas o menos:

1) ¿Por qué PCA “elige un ángulo” y qué significa “máxima varianza”?
2) ¿Por qué `S²/(n-1)` son eigenvalues de la covarianza?
3) ¿Qué pierde la reconstrucción cuando `k < n_features`?

---

## 💻 Parte 5: Gaussian Mixture Models (GMM)

### 5.0 GMM — Nivel: intermedio/avanzado (clustering probabilístico)

**Propósito:** pasar de “K-Means agrupa” a **entender cuándo K-Means es geométricamente incorrecto** y usar un modelo que capture **clusters elípticos** y asignación “suave” (*soft clustering*).

#### Objetivos de aprendizaje (medibles)

Al terminar este bloque podrás:

- **Explicar** por qué K-Means asume clusters aproximadamente esféricos (misma varianza en todas direcciones).
- **Describir** un GMM como “mezcla de Gaussianas” con una variable latente de componente.
- **Derivar** la idea operacional del algoritmo EM (E-step y M-step) a nivel implementable.
- **Interpretar** *responsibilities* `γ(z_k)` como probabilidad de pertenencia.
- **Diagnosticar** fallas típicas: colapso de covarianzas, sensibilidad a inicialización, singularidad.

#### Intuición geométrica: clusters elípticos y pertenencia suave

Imagina que tus datos forman “nubes” alargadas:

- K-Means solo puede poner centroides y partir el espacio por regiones de Voronoi con distancia euclidiana.
- GMM asume que cada cluster es una **Gaussiana** con su propia forma:
  - media `μ_k` (centro)
  - covarianza `Σ_k` (orientación y elongación)

La diferencia clave es que GMM no dice “este punto es del cluster 2”. Dice:

> “Este punto es 70% del componente 2 y 30% del componente 1”.

Eso es extremadamente útil cuando los clusters se solapan.

#### Conceptos clave (glosario mínimo)

- **Mezcla:** combinación ponderada de distribuciones.
- **Pesos `π_k`:** probabilidades a priori de cada componente (suman 1).
- **Variable latente `z`:** indica qué componente “generó” el punto.
- **Responsibilities `γ_{ik}`:** `P(z=k | x_i)`.
- **EM (Expectation-Maximization):** alterna “asignar probabilidades” y “re-estimar parámetros”.

#### Formalización mínima

Modelo:

`p(x) = Σ_{k=1..K} π_k  N(x | μ_k, Σ_k)`

Log-likelihood de datos `X = {x_i}`:

`ℓ = Σ_i log( Σ_k π_k N(x_i | μ_k, Σ_k) )`

No puedes maximizar esto de forma cerrada por el `log(Σ ...)`. EM lo hace iterativamente.

#### EM (idea implementable)

**E-step:** calcula responsibilities

`γ_{ik} = P(z=k | x_i) = (π_k N(x_i|μ_k,Σ_k)) / (Σ_j π_j N(x_i|μ_j,Σ_j))`

**M-step:** actualiza parámetros usando promedios ponderados

- `N_k = Σ_i γ_{ik}`
- `π_k = N_k / n`
- `μ_k = (1/N_k) Σ_i γ_{ik} x_i`
- `Σ_k = (1/N_k) Σ_i γ_{ik} (x_i-μ_k)(x_i-μ_k)ᵀ`

#### Worked example (mínimo, 1D para ver EM sin álgebra pesada)

Supón puntos 1D `x = [-2, -1, 0, 2, 3]` y `K=2`.

Idea:

1) Inicializas dos Gaussianas (medias distintas).
2) En E-step, los puntos negativos tienen `γ` alto para el componente “izquierdo” y bajo para el derecho.
3) En M-step, la media izquierda se va hacia el promedio ponderado de los negativos, la derecha hacia los positivos.
4) Repites hasta que el log-likelihood deja de mejorar.

La intuición: es como K-Means, pero en vez de asignar “duro”, asignas *responsabilities* y actualizas con pesos.

#### Cuándo usar GMM vs K-Means (regla práctica)

- **Usa K-Means** si esperas clusters aproximadamente esféricos, bien separados y quieres simplicidad/velocidad.
- **Usa GMM** si:
  - esperas **clusters elípticos** o con varianzas distintas por dirección
  - hay **solapamiento** y necesitas pertenencia probabilística
  - quieres un modelo generativo simple para densidad

#### Errores comunes / debugging

- **No estandarizar features:** si una dimensión domina, la covarianza se distorsiona.
- **Singularidad/collapse:** una `Σ_k` puede volverse casi singular si un componente “se queda” con muy pocos puntos.
- **Inicialización pobre:** EM converge a óptimos locales; iniciar con K-Means suele ayudar.

---

## 🚫 Cuándo NO usar K-Means / PCA (y qué hacer en su lugar)

### Diagnóstico rápido (regla práctica)

Si no puedes justificar “por qué este método tiene sentido para este dataset”, asume que estás en zona de riesgo.

#### K-Means: señales de que NO es buena idea

- **Geometría incorrecta:** clusters no convexos (formas tipo “dos lunas”) o estructuras alargadas.
- **Densidades muy distintas:** un cluster muy denso y otro muy disperso.
- **Outliers fuertes:** centroides se mueven para “perseguir” outliers.
- **Escalas distintas:** una feature domina la distancia euclidiana.

**Síntomas medibles típicos:**

- El **método del codo** no muestra un “codo” claro.
- **Silhouette score** bajo (cerca de 0) o negativo.
- Resultados muy distintos entre distintas inicializaciones.

**Qué hacer en su lugar (según el problema):**

- **Clusters elípticos (varianza diferente por dirección):** GMM (Gaussian Mixture Models).
- **Clusters con formas arbitrarias y ruido:** DBSCAN / HDBSCAN (no implementados aquí, pero recomendados).
- **Estructura jerárquica:** Hierarchical clustering.

#### PCA: señales de que NO es buena idea

- **Señal no alineada con varianza:** la dirección con mayor varianza no es la que separa clases (común en tareas supervisadas).
- **Relación no lineal:** datos sobre un manifold curvo (PCA lineal pierde estructura).
- **Interpretación equivocada:** usar PCA como “selector de features importantes” sin analizar varianza explicada y reconstrucción.

**Síntomas medibles típicos:**

- Necesitas muchos componentes para llegar a 95% de varianza (PCA no está comprimiendo bien).
- La visualización en 2D parece “mezclar” todo sin estructura (ojo: esto no prueba que no haya estructura, pero es una señal).

**Qué hacer en su lugar (según el objetivo):**

- **Visualización no lineal:** t-SNE / UMAP (útiles para explorar, no para entrenar modelos lineales directamente).
- **Compresión aprendida:** autoencoders (Módulo 07, enfoque DL).
- **Si solo quieres acelerar:** reducir features por ingeniería o seleccionar por dominio.

### Checklist de decisión (antes de usar el método)

- **Datos escalados:** ¿features comparables? (si no, normaliza).
- **Outliers:** ¿hay outliers? (si sí, documenta su impacto).
- **Objetivo real:** ¿quieres compresión, visualización, o clustering interpretable?

Integración con ejecución y validación:

- [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
- [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)
- Diario: `study_tools/DIARIO_ERRORES.md`

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
Módulo: 06 - Unsupervised Learning
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

## 🔍 Shadow Mode: Validación con sklearn (v3.3)

> ⚠️ **Regla:** sklearn está **prohibido para aprender**, pero es **útil para validar**. Si tus resultados difieren de forma grande y consistente, primero asume bug.

### Protocolo mínimo

- **K-Means:** comparar inercia y silhouette para el mismo `k`.
- **PCA:** comparar `explained_variance_ratio_` y reconstrucción aproximada.

```python
"""
Shadow Mode - Unsupervised Learning
Comparación: implementaciones desde cero vs sklearn.
"""

import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.decomposition import PCA as SklearnPCA


def shadow_mode_kmeans(X: np.ndarray, k: int = 3, seed: int = 42) -> None:
    """Compara inercia de tu K-Means vs sklearn."""
    # Tu implementación
    # my = KMeans(n_clusters=k, random_state=seed)
    # my_labels = my.fit_predict(X)
    # my_inertia = my.inertia_

    # Placeholder (reemplazar con tu código)
    my_inertia = 0.0

    # sklearn
    sk = SklearnKMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)
    sk.fit(X)

    print("=" * 60)
    print("SHADOW MODE: K-Means")
    print("=" * 60)
    print(f"Tu inercia:      {my_inertia:.4f}")
    print(f"sklearn inertia: {sk.inertia_:.4f}")


def shadow_mode_pca(X: np.ndarray, n_components: int = 2) -> None:
    """Compara varianza explicada de tu PCA vs sklearn."""
    # Tu implementación
    # my = PCA(n_components=n_components)
    # X_my = my.fit_transform(X)

    # sklearn
    sk = SklearnPCA(n_components=n_components)
    sk.fit(X)

    print("=" * 60)
    print("SHADOW MODE: PCA")
    print("=" * 60)
    print(f"sklearn explained_variance_ratio_: {sk.explained_variance_ratio_}")
```

---

## 🧭 Puente al Módulo 08 (MNIST Analyst)

En la Semana 21 del proyecto:

- **PCA:** lo usas para reducir MNIST y visualizar estructura en 2D (y para acelerar métodos posteriores).
- **K-Means:** lo usas para agrupar dígitos sin etiquetas y visualizar centroides como “prototipos”.

Checklist de integración:

- **Entrada:** MNIST normalizado a `[0, 1]`.
- **PCA 2D:** gráfico con clusters/colores.
- **K-Means:** elegir `k=10` y analizar si los clusters se alinean con dígitos.
- **Salida:** guarda figuras y conclusiones para el informe.

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
| [05_SUPERVISED_LEARNING](05_SUPERVISED_LEARNING.md) | [00_INDICE](00_INDICE.md) | [07_DEEP_LEARNING](07_DEEP_LEARNING.md) |
