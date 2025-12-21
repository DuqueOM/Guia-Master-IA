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

### 🧪 Ver para Entender (Laboratorios Interactivos)

- Guía central: [INTERACTIVE_LABS.md](../../Recursos_Adicionales/INTERACTIVE_LABS.md)
- PCA: rotación manual 3D → proyección 2D (intuición de varianza máxima) + referencia SVD:
  - `streamlit run M06_unsupervised/pca_rotation_plotly_app.py`

Enlaces rápidos:

- [GLOSARIO.md](GLOSARIO.md)
- [RECURSOS.md](RECURSOS.md)
- [PLAN_V4_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V4_ESTRATEGICO.md)
- [PLAN_V5_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V5_ESTRATEGICO.md)
- Evaluación (rúbrica): [Herramientas_Estudio/RUBRICA_v1.md](../Herramientas_Estudio/RUBRICA_v1.md) (scope `M06` en `rubrica.csv`; incluye PB-16)

### Recursos (cuándo usarlos)

| Prioridad | Recurso | Cuándo usarlo en este módulo | Para qué |
|----------|---------|------------------------------|----------|
| **Obligatorio** | `Herramientas_Estudio/DRILL_DIMENSIONES_NUMPY.md` | Semana 13–16, cada vez que implementes distancias/proyecciones y se rompan shapes | Evitar errores silenciosos en broadcasting/`axis` |
| **Obligatorio** | `Herramientas_Estudio/DIARIO_ERRORES.md` | Cuando K-Means produzca clusters vacíos, `NaN` o PCA devuelva resultados inestables | Registrar el caso y dejarlo “debuggeable” |
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
- Registra hallazgos en `Herramientas_Estudio/DIARIO_ERRORES.md`.
- Antes de usar un dataset real “sucio”, aplica `Herramientas_Estudio/DIRTY_DATA_CHECK.md`.
- Para integrar el protocolo completo:
  - [PLAN_V4_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V5_ESTRATEGICO.md)

#### Cheat sheet (repaso rápido)

- **Paso 1:** `labels = argmin(||x - μᵢ||²)`
- **Paso 2:** `μᵢ = mean(points_in_cluster_i)`
- **Convergencia:** `||μ_new - μ_old||² < tol`
- **Riesgo:** mínimos locales → usar K-Means++ y/o múltiples inicializaciones

### 1.1 Algoritmo de Lloyd

```python
import numpy as np  # Importa NumPy: se usa para RNG, arrays, distancias cuadráticas y muestreo probabilístico en K-Means++

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

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:  # Distancia euclidiana entre dos puntos.
    """Distancia euclidiana entre dos puntos."""
    return np.sqrt(np.sum((a - b) ** 2))  # Calcula ||a-b||₂: resta vectorial, eleva al cuadrado, suma por dimensión y aplica raíz

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:  # Asigna cada punto al centroide más cercano.
    """
    Asigna cada punto al centroide más cercano.

    Args:
        X: datos (n_samples, n_features)
        centroids: centroides actuales (k, n_features)

    Returns:
        labels: índice del cluster para cada punto (n_samples,)
    """
    n_samples = X.shape[0]  # Número de puntos: define cuántas filas tendrá la matriz de distancias y la longitud de labels
    k = centroids.shape[0]  # Número de centroides/clusters: define cuántas columnas tendrá la matriz de distancias

    # Calcular distancia de cada punto a cada centroide
    distances = np.zeros((n_samples, k))  # Reserva distancias (n,k): cada entrada [j,i] será la distancia de X[j] al centroide i
    for i in range(k):  # Recorre centroides: calcula distancias de TODOS los puntos a un centroide a la vez (vectorizado por filas)
        distances[:, i] = np.sqrt(np.sum((X - centroids[i]) ** 2, axis=1))  # Distancia Euclídea por punto: sum over features y sqrt para cada fila

    # Asignar al más cercano
    return np.argmin(distances, axis=1)  # Label por punto: índice i del centroide con distancia mínima (argmin sobre columnas)

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:  # Actualiza centroides como el promedio de los puntos asignados.
    """
    Actualiza centroides como el promedio de los puntos asignados.

    Args:
        X: datos
        labels: asignaciones actuales
        k: número de clusters

    Returns:
        nuevos centroides
    """
    n_features = X.shape[1]  # Dimensionalidad d: número de features por punto, define el ancho del array de centroides
    centroids = np.zeros((k, n_features))  # Inicializa centroides nuevos (k,d): se llenan con medias por cluster

    for i in range(k):  # Recorre cada cluster i: recalcula su centroide como promedio de sus puntos asignados
        points_in_cluster = X[labels == i]  # Selecciona puntos asignados al cluster i: indexación booleana
        if len(points_in_cluster) > 0:  # Evita cluster vacío: sin puntos no se puede calcular media (mean sobre vacío -> warning/NaN)
            centroids[i] = np.mean(points_in_cluster, axis=0)  # Media por feature: define el nuevo centroide como el "centro" de su nube

    return centroids  # Devuelve centroides actualizados (k,d): promedios por cluster para el siguiente paso de asignación
```

### 1.2 K-Means++ Initialization

```python
import numpy as np  # Importa NumPy: se usa para RNG, distancias cuadráticas y muestreo ponderado en la inicialización K-Means++

def kmeans_plus_plus_init(X: np.ndarray, k: int, random_state: int = None) -> np.ndarray:  # Inicializa centroides con K-Means++ (mejora convergencia vs random)
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
    if random_state is not None:  # Si se provee semilla, fijamos el RNG para reproducibilidad del muestreo de centroides
        np.random.seed(random_state)  # Setea semilla global: controla randint/choice usados abajo

    n_samples, n_features = X.shape  # Extrae shapes: n puntos y d features para dimensionar estructuras y muestrear índices
    centroids = np.zeros((k, n_features))  # Reserva matriz de centroides: (k,d) para ir llenándola iterativamente

    # Primer centroide aleatorio
    first_idx = np.random.randint(n_samples)  # Elige índice inicial uniforme: primer centroide se toma al azar (paso 1 de K-Means++)
    centroids[0] = X[first_idx]  # Copia el primer centroide desde X: garantiza que el centroide es un punto real del dataset

    # Centroides restantes
    for c in range(1, k):  # Para cada centroide restante: selecciona un nuevo centroide sesgado hacia puntos lejanos
        # Calcular distancia al centroide más cercano para cada punto
        distances = np.zeros(n_samples)  # Vector d² mínimo por punto: almacenará la distancia^2 al centroide más cercano
        for i in range(n_samples):  # Recorre cada muestra i: calcula su distancia al centroide más cercano (entre los ya elegidos)
            min_dist = float('inf')  # Inicializa mínimo: se actualizará comparando con cada centroide existente
            for j in range(c):  # Recorre centroides ya elegidos (0..c-1): busca el más cercano al punto i
                dist = np.sum((X[i] - centroids[j]) ** 2)  # Distancia^2 a centroide j: evita sqrt y preserva orden para argmin
                min_dist = min(min_dist, dist)  # Actualiza mínimo: mantiene la menor distancia^2 encontrada hasta ahora
            distances[i] = min_dist  # Guarda d² mínimo para el punto i: define su probabilidad de ser elegido

        # Probabilidad proporcional a d²
        probabilities = distances / np.sum(distances)  # Normaliza d² a distribución: suma 1 y prioriza puntos lejanos a centroides actuales

        # Elegir nuevo centroide
        new_idx = np.random.choice(n_samples, p=probabilities)  # Samplea índice según probs: implementa el sesgo K-Means++ (paso 2)
        centroids[c] = X[new_idx]  # Asigna el nuevo centroide: toma un punto real de X para evitar centroides fuera del soporte

    return centroids  # Devuelve centroides iniciales (k,d): se pasan a K-Means/Lloyd para empezar iteraciones desde una buena semilla
```

### 1.3 Implementación Completa

```python
import numpy as np  # Importa NumPy: se usa para RNG, álgebra vectorizada y operaciones de distancia/centroides
from typing import Tuple  # Importa typing: documenta tipos de retorno/entradas (no afecta runtime)

class KMeans:  # Implementa K-Means desde cero: alterna asignación de clusters y actualización de centroides hasta convergencia
    """K-Means Clustering implementado desde cero."""  # Docstring: describe la clase; es un literal string y no cambia el comportamiento

    def __init__(  # Inicializa hiperparámetros y atributos del modelo
        self,  # Referencia a la instancia: permite setear atributos persistentes del estimador
        n_clusters: int = 3,  # k: cantidad de clusters/centroides a aprender
        max_iter: int = 300,  # Máximo de iteraciones: tope de seguridad si no converge
        tol: float = 1e-4,  # Tolerancia de convergencia: umbral para detener cuando los centroides cambian muy poco
        init: str = 'kmeans++',  # Estrategia de inicialización: 'kmeans++' o 'random'
        random_state: int = None  # Semilla opcional: hace reproducibles tanto init como sampling aleatorio
    ):  # Cierra firma: se ejecuta al definir el método
        """
        Args:
            n_clusters: número de clusters (k)
            max_iter: máximo de iteraciones
            tol: tolerancia para convergencia
            init: 'kmeans++' o 'random'
            random_state: semilla para reproducibilidad
        """
        self.n_clusters = n_clusters  # Guarda k: se reutiliza en fit/predict y en loops internos
        self.max_iter = max_iter  # Guarda límite de iteraciones: controla el ciclo de entrenamiento
        self.tol = tol  # Guarda tolerancia: define criterio de parada por desplazamiento de centroides
        self.init = init  # Guarda modo de init: decide cómo se eligen centroides iniciales
        self.random_state = random_state  # Guarda semilla: permite reproducibilidad de resultados

        self.centroids = None  # Placeholder: centroides aprendidos (k, n_features); se setea en fit
        self.labels_ = None  # Placeholder: asignación por muestra (n_samples,); se setea en fit
        self.inertia_ = None  # Placeholder: SSE final dentro de clusters; se computa al final del fit
        self.n_iter_ = 0  # Contador de iteraciones ejecutadas: útil para diagnóstico de convergencia

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:  # Inicializa centroides de acuerdo al modo elegido
        """Inicializa centroides."""  # Docstring: describe el helper; no altera la lógica
        if self.random_state is not None:  # Si hay semilla, fijamos el RNG global para reproducibilidad del muestreo de centroides
            np.random.seed(self.random_state)  # Setea semilla: controla np.random.choice/np.random.randint usados abajo

        if self.init == 'kmeans++':  # Rama 1: init informado por distancia (mejor que random para evitar mínimos malos)
            return kmeans_plus_plus_init(X, self.n_clusters, self.random_state)  # Devuelve centroides iniciales con K-Means++
        else:  # Rama 2: init aleatorio (baseline) para comparación/rapidez
            # Inicialización aleatoria
            indices = np.random.choice(len(X), self.n_clusters, replace=False)  # Samplea k índices distintos: evita centroides duplicados
            return X[indices].copy()  # Copia centroides iniciales: evita aliasing con X al modificarlos durante el training

    def _compute_inertia(self, X: np.ndarray) -> float:  # Calcula inercia/SSE dentro de clusters para el estado actual
        """
        Calcula inercia (within-cluster sum of squares).

        Inercia = Σᵢ Σⱼ ||xⱼ - μᵢ||²
        """
        inertia = 0  # Acumulador SSE: suma distancias cuadradas de cada punto a su centroide asignado
        for i in range(self.n_clusters):  # Itera clusters: computa contribución por centroide i
            cluster_points = X[self.labels_ == i]  # Selecciona puntos asignados al cluster i: indexación booleana
            if len(cluster_points) > 0:  # Evita cluster vacío: si no hay puntos, su SSE contribuye 0
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)  # Suma distancias^2 al centroide i
        return inertia  # Devuelve SSE total: se usa en elbow method y diagnóstico

    def fit(self, X: np.ndarray) -> 'KMeans':  # Entrena el modelo: aprende centroides y labels para X
        """Entrena el modelo."""  # Docstring: describe método; no modifica el entrenamiento
        # Inicializar centroides
        self.centroids = self._init_centroids(X)  # Setea centroides iniciales: punto de partida del loop iterativo

        for iteration in range(self.max_iter):  # Loop principal: alterna asignación (E-step) y actualización (M-step)
            # Guardar centroides anteriores
            old_centroids = self.centroids.copy()  # Copia para medir desplazamiento: criterio de convergencia

            # Paso 1: Asignar puntos a clusters
            self.labels_ = assign_clusters(X, self.centroids)  # Asigna cada punto al centroide más cercano (por distancia^2)

            # Paso 2: Actualizar centroides
            self.centroids = update_centroids(X, self.labels_, self.n_clusters)  # Recalcula centroides como media de puntos asignados

            # Verificar convergencia
            centroid_shift = np.sum((self.centroids - old_centroids) ** 2)  # Shift global: suma de desplazamientos^2 entre iteraciones
            if centroid_shift < self.tol:  # Si el cambio total es pequeño, asumimos convergencia (ya no mejora materialmente)
                break  # Sale temprano: ahorra cómputo manteniendo la solución estable

        self.n_iter_ = iteration + 1  # Guarda iteraciones efectivas (iteration es 0-index)
        self.inertia_ = self._compute_inertia(X)  # Calcula inercia final: métrica interna del ajuste

        return self  # Permite chaining (kmeans.fit(X).predict(X)) y acceso a atributos entrenados

    def predict(self, X: np.ndarray) -> np.ndarray:  # Predice cluster para datos nuevos usando centroides ya aprendidos
        """Predice clusters para nuevos datos."""  # Docstring: describe uso; no cambia la predicción
        return assign_clusters(X, self.centroids)  # Reutiliza misma asignación: calcula distancias a centroides y retorna argmin

    def fit_predict(self, X: np.ndarray) -> np.ndarray:  # Atajo: fit + retorna labels en una sola llamada
        """Entrena y predice."""  # Docstring: describe atajo
        self.fit(X)  # Entrena primero: actualiza centroides/labels/inercia
        return self.labels_  # Devuelve labels aprendidas: resultado principal del clustering


# Demo
np.random.seed(42)  # Fija semilla: hace reproducible el dataset sintético del demo

# Generar datos sintéticos (3 clusters)
cluster1 = np.random.randn(100, 2) + [0, 0]  # Cluster 1: nube gaussiana centrada en (0,0)
cluster2 = np.random.randn(100, 2) + [5, 5]  # Cluster 2: nube gaussiana centrada en (5,5)
cluster3 = np.random.randn(100, 2) + [10, 0]  # Cluster 3: nube gaussiana centrada en (10,0)
X = np.vstack([cluster1, cluster2, cluster3])  # Dataset final: concatena clusters => shape (300,2)

# Entrenar
kmeans = KMeans(n_clusters=3, random_state=42)  # Crea estimador con k=3: coincide con generación sintética
labels = kmeans.fit_predict(X)  # Ajusta el modelo y obtiene labels: debería separar bien los 3 grupos

print(f"Iteraciones: {kmeans.n_iter_}")  # Muestra iteraciones: indica rapidez de convergencia
print(f"Inercia: {kmeans.inertia_:.2f}")  # Muestra SSE final: menor suele implicar clusters más compactos (pero depende de k)
print(f"Centroides:\n{kmeans.centroids}")  # Muestra centroides aprendidos: aproximan los centros de las nubes
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
def compute_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:  # Calcula inercia/SSE: suma de distancias^2 de puntos a su centroide (métrica interna de compactación)
    """
    Inercia: suma de distancias cuadradas al centroide.

    Menor inercia = clusters más compactos.

    Problema: siempre disminuye al aumentar k.
    Solución: usar método del codo.
    """
    inertia = 0  # Acumulador SSE: suma de distancias cuadráticas intra-cluster (cuanto menor, más compactos los clusters)
    for i, centroid in enumerate(centroids):  # Recorre centroides: agrega contribución de cada cluster i a la inercia total
        cluster_points = X[labels == i]  # Selecciona puntos asignados al cluster i: subconjunto sobre el que se mide compactación
        inertia += np.sum((cluster_points - centroid) ** 2)  # Suma ||x-μ_i||^2 sobre puntos del cluster i: define la SSE intra-cluster
    return inertia  # Devuelve inercia total: se usa para comparar k en el elbow method (aunque siempre decrece al aumentar k)
```

### 2.2 Método del Codo (Elbow Method)

```python
import numpy as np  # Importa NumPy: se usa para almacenar/operar con listas de inercia y manejar rangos de k
import matplotlib.pyplot as plt  # Importa Matplotlib: se usa para graficar la curva de inercia vs k ("codo")

def elbow_method(X: np.ndarray, k_range: range) -> list:  # Ejecuta KMeans para múltiples k y devuelve la lista de inercias para detectar el “codo”
    """
    Método del codo para elegir k óptimo.

    Busca el punto donde añadir más clusters
    no reduce significativamente la inercia.
    """
    inertias = []  # Lista de inercia por k: se llena en el loop y luego se grafica para buscar el “codo”

    for k in k_range:  # Itera candidatos k: prueba distintos números de clusters para ver cómo cae la inercia
        kmeans = KMeans(n_clusters=k, random_state=42)  # Instancia KMeans: fija random_state para que comparaciones entre k sean reproducibles
        kmeans.fit(X)  # Ajusta el modelo: ejecuta Lloyd y aprende centroides/labels; calcula inercia final
        inertias.append(kmeans.inertia_)  # Guarda la inercia del modelo: suma de distancias cuadráticas intra-cluster (menor es mejor pero sesga a k alto)

    return inertias  # Devuelve lista alineada con k_range: se usa para graficar y detectar visualmente el punto de codo

def plot_elbow(k_range: range, inertias: list):  # Grafica la curva k vs inercia para elegir k por criterio visual/heurístico
    """Visualiza el método del codo."""
    plt.figure(figsize=(8, 5))  # Crea figura: define tamaño para una lectura clara de la curva
    plt.plot(list(k_range), inertias, 'bo-')  # Curva k vs inercia: puntos azules con línea (visualiza tendencia y posible “codo”)
    plt.xlabel('Número de clusters (k)')  # Etiqueta eje x: variable controlada (cantidad de clusters)
    plt.ylabel('Inercia')  # Etiqueta eje y: métrica interna que siempre baja con k (no debe optimizarse “a ciegas”)
    plt.title('Método del Codo')  # Título: contextualiza la gráfica
    plt.grid(True)  # Activa grilla: facilita comparar caídas relativas entre k consecutivos
    plt.show()  # Renderiza la figura: muestra el plot al usuario

# Demo
# inertias = elbow_method(X, range(1, 11))
# plot_elbow(range(1, 11), inertias)
```

### 2.3 Silhouette Score

```python
import numpy as np  # Importa NumPy: se usa para sqrt/sum/mean, comparaciones, uniques y manejo de arrays

def silhouette_sample(X: np.ndarray, labels: np.ndarray, idx: int) -> float:  # Silhouette puntual: calcula s(i) para una muestra i
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
    point = X[idx]  # Punto i: vector de features para el que se calcula silhouette
    label = labels[idx]  # Label del punto i: define su cluster para calcular cohesión (a) y separación (b)

    # a(i): distancia promedio intra-cluster
    same_cluster = X[labels == label]  # Puntos del mismo cluster: se usan para promedio intra-cluster a(i)
    if len(same_cluster) > 1:  # Si el cluster tiene más de 1 punto, a(i) se define como promedio de distancias a los demás
        a = np.mean([np.sqrt(np.sum((point - p) ** 2))  # Distancia Euclídea a cada punto del mismo cluster
                     for p in same_cluster if not np.array_equal(p, point)])  # Excluye el propio punto: evita incluir distancia 0
    else:  # Si el cluster es unitario (solo i), no hay vecinos intra-cluster para promediar: usamos convención a=0
        a = 0  # Edge case: cluster unitario (solo el punto) => cohesión se define como 0 por convención

    # b(i): distancia promedio al cluster más cercano
    unique_labels = np.unique(labels)  # Clusters presentes: se iteran para buscar el cluster alternativo más cercano
    b = float('inf')  # Inicializa b(i): se busca el mínimo promedio a cualquier cluster distinto
    for other_label in unique_labels:  # Itera clusters alternativos: busca el cluster “vecino” con menor distancia media (b(i))
        if other_label != label:  # Omite el propio cluster: b(i) se define respecto a otros clusters
            other_cluster = X[labels == other_label]  # Puntos del cluster candidato: se usan para distancia media inter-cluster
            if len(other_cluster) > 0:  # Evita mean sobre vacío: aunque np.unique suele garantizar que el cluster existe, es defensa adicional
                avg_dist = np.mean([np.sqrt(np.sum((point - p) ** 2))  # Distancia Euclídea a cada punto del cluster candidato
                                   for p in other_cluster])  # Promedio: distancia media de i al cluster other_label
                b = min(b, avg_dist)  # Actualiza mejor b(i): elige el cluster con menor distancia media

    if b == float('inf'):  # Edge case: no se encontró cluster alternativo (labels degenerados) => b(i) no está definido
        return 0  # Convención: score neutral cuando no hay comparación posible

    return (b - a) / max(a, b)  # Fórmula silhouette puntual: normaliza para acotar en [-1,1] y comparar cohesión vs separación

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:  # Silhouette global: promedio de s(i) sobre todas las muestras
    """
    Silhouette Score promedio para todos los puntos.

    Mayor es mejor (max = 1).
    """
    scores = [silhouette_sample(X, labels, i) for i in range(len(X))]  # Calcula s(i) para cada punto: comprensión lista para promediar
    return np.mean(scores)  # Promedia scores por punto: devuelve silhouette global del clustering


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
5) (Opcional) Reconstruir `X_hat = Z @ V_kᵀ + mean`.

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
import numpy as np  # Importa NumPy: se usa para medias, covarianza, eigen, ordenamientos y proyecciones en PCA (vía eig)

def pca_eigen(X: np.ndarray, n_components: int) -> tuple:  # PCA por eigendecomposition: forma covarianza y extrae eigenvectors/eigenvalues
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
    mean = np.mean(X, axis=0)  # Media por feature: se resta para que la covarianza represente variación alrededor de 0
    X_centered = X - mean  # Centra datos: elimina offset por feature para que PCA encuentre direcciones de varianza

    # 2. Matriz de covarianza
    n_samples = X.shape[0]  # Número de muestras n: se usa para normalizar covarianza con factor (n-1)
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)  # Covarianza (d,d): Xc^T Xc /(n-1) (asumiendo centrado)

    # 3. Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # Eig: encuentra λ y v tales que cov v = λ v (direcciones principales)

    # Convertir a reales (puede haber componentes imaginarias pequeñas)
    eigenvalues = eigenvalues.real  # Descarta parte imaginaria pequeña: puede aparecer por errores numéricos en eig
    eigenvectors = eigenvectors.real  # Mantiene eigenvectors reales: PCA real espera ejes en ℝ^d

    # 4. Ordenar por eigenvalue descendente
    idx = np.argsort(eigenvalues)[::-1]  # Índices ordenados desc: prioriza componentes que explican más varianza
    eigenvalues = eigenvalues[idx]  # Reordena eigenvalues: queda λ1 ≥ λ2 ≥ ... para seleccionar top-k
    eigenvectors = eigenvectors[:, idx]  # Reordena columnas de eigenvectors: alinea v_j con eigenvalue λ_j ordenado

    # 5. Seleccionar top k componentes
    components = eigenvectors[:, :n_components]  # Toma primeras k columnas: matriz (d,k) de componentes principales

    # 6. Proyectar
    X_pca = X_centered @ components  # Proyección lineal: (n,d)@(d,k)->(n,k) da coordenadas en el subespacio de máxima varianza

    # 7. Varianza explicada
    total_variance = np.sum(eigenvalues)  # Varianza total: suma de eigenvalues equivale al trace(cov) (varianza total en d dims)
    explained_variance_ratio = eigenvalues[:n_components] / total_variance  # Ratio por componente: fracción de varianza explicada por cada λ_j

    return X_pca, components, explained_variance_ratio, mean  # Devuelve proyección, ejes, ratios y media para poder reconstruir/inferir
```

### 3.3 PCA via SVD (Más Estable)

```python
import numpy as np  # Importa NumPy: se usa para SVD, medias, proyecciones y varianza explicada en PCA (vía SVD)

def pca_svd(X: np.ndarray, n_components: int) -> tuple:  # PCA por SVD: alternativa más estable que eig para obtener componentes principales
    """
    PCA usando SVD (Singular Value Decomposition).

    Más estable numéricamente que eigendecomposition.

    Si X = UΣV^T, entonces:
    - V contiene las componentes principales
    - Σ²/(n-1) son los eigenvalues (varianzas)
    """
    # 1. Centrar
    mean = np.mean(X, axis=0)  # Media por feature: se resta para centrar y alinear PCA con covarianza (no con offsets)
    X_centered = X - mean  # Centra X: elimina offset para que SVD capture direcciones de varianza

    # 2. SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # SVD compacta: Xc=U diag(S) Vt; estable numéricamente

    # 3. Componentes principales (filas de Vt, o columnas de V)
    components = Vt[:n_components].T  # Componentes (d,k): toma primeras k filas de Vt y transpone para usar como matriz de proyección

    # 4. Proyectar
    X_pca = X_centered @ components  # Proyecta a k dims: coordenadas en el subespacio principal (scores)

    # 5. Varianza explicada
    n_samples = X.shape[0]  # n: se usa en el factor (n-1) para convertir S^2 en varianzas (eigenvalues)
    variance = (S ** 2) / (n_samples - 1)  # Varianza por componente: S^2/(n-1) corresponde a eigenvalues de cov(X)
    explained_variance_ratio = variance[:n_components] / np.sum(variance)  # Ratio truncado: var explicada por cada una de las k componentes

    return X_pca, components, explained_variance_ratio, mean  # Devuelve proyección, ejes (d,k), ratios y media para reconstrucción
```

### 3.4 Implementación Completa

```python
import numpy as np  # Importa NumPy: se usa para SVD, medias, proyecciones y generación de datos sintéticos en el demo

class PCA:  # Implementa PCA desde cero: aprende ejes principales (componentes) y permite proyectar/reconstruir
    """Principal Component Analysis implementado desde cero."""  # Docstring: describe la clase; no afecta cálculos

    def __init__(self, n_components: int = 2):  # Inicializa PCA con k componentes a retener (dimensión reducida)
        """
        Args:
            n_components: número de componentes a retener
        """
        self.n_components = n_components  # Guarda k: se usa para truncar Vt y para shapes de proyección
        self.components_ = None  # (n_features, n_components)  # Placeholder: ejes principales aprendidos (columnas)
        self.explained_variance_ratio_ = None  # Placeholder: fracción de varianza explicada por cada componente
        self.mean_ = None  # Placeholder: media por feature para centrar y descentrar (inverse_transform)

    def fit(self, X: np.ndarray) -> 'PCA':  # Ajusta PCA: estima media, componentes y varianza explicada a partir de X
        """Calcula componentes principales."""  # Docstring: describe fit; no altera el resultado
        # Centrar
        self.mean_ = np.mean(X, axis=0)  # Media por columna: PCA estándar requiere centrar para capturar covarianza
        X_centered = X - self.mean_  # Centra X: elimina offset para que SVD capture direcciones de varianza

        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # SVD: X=U S V^T; V contiene direcciones principales en espacio de features

        # Componentes principales
        self.components_ = Vt[:self.n_components].T  # Componentes (d,k): toma primeras k filas de Vt y transpone para usar como matriz de proyección

        # Varianza explicada
        n_samples = X.shape[0]  # n: se usa en el factor (n-1) para convertir S^2 en varianzas (eigenvalues)
        variance = (S ** 2) / (n_samples - 1)  # Varianza por componente: S^2/(n-1) corresponde a eigenvalues de cov(X)
        self.explained_variance_ratio_ = variance[:self.n_components] / np.sum(variance)  # Normaliza por varianza total: proporción explicada
        self.singular_values_ = S[:self.n_components]  # Guarda valores singulares truncados: útiles para covarianza aproximada

        return self  # Permite chaining (pca.fit(X).transform(X)) y acceso a atributos entrenados

    def transform(self, X: np.ndarray) -> np.ndarray:  # Proyecta X al subespacio PCA (coordenadas en base de componentes)
        """Proyecta datos a espacio de componentes principales."""  # Docstring: describe proyección
        X_centered = X - self.mean_  # Centra con la media aprendida: garantiza consistencia entre train y test
        return X_centered @ self.components_  # Proyección lineal: (n_samples,n_features)@(n_features,k) -> (n_samples,k)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:  # Atajo: ajusta PCA y devuelve la proyección en una llamada
        """Fit y transform en un paso."""  # Docstring: describe atajo
        self.fit(X)  # Aprende media/componentes: actualiza estado interno
        return self.transform(X)  # Proyecta X usando el estado recién aprendido

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:  # Reconstruye aproximación en espacio original desde coordenadas PCA
        """
        Reconstruye datos desde el espacio PCA.

        X_reconstructed = X_pca @ components.T + mean

        Nota: hay pérdida de información si n_components < n_features
        """
        return X_pca @ self.components_.T + self.mean_  # Re-proyecta a features y suma media: reconstrucción es aproximada si k<n_features

    def get_covariance(self) -> np.ndarray:  # Aproxima cov(X) usando componentes y valores singulares (relación con Σ^2)
        """Retorna matriz de covarianza aproximada."""  # Docstring: describe la salida (matriz n_features x n_features)
        return self.components_ @ np.diag(self.singular_values_ ** 2) @ self.components_.T  # Reconstruye cov aprox en base PCA


# Demo
np.random.seed(42)  # Fija semilla: hace reproducible el demo (mismos datos => mismas métricas/resultados)

# Datos correlacionados en 3D
n_samples = 200  # Cantidad de muestras sintéticas: controla tamaño del dataset para el ejemplo
X = np.random.randn(n_samples, 3)  # Genera datos base iid: luego se induce correlación entre columnas
X[:, 1] = X[:, 0] * 2 + np.random.randn(n_samples) * 0.1  # y correlacionado con x
X[:, 2] = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1  # z correlacionado con x e y: crea estructura subespacial

# PCA
pca = PCA(n_components=2)  # Instancia PCA para reducir de 3D a 2D: debería capturar casi toda la varianza
X_pca = pca.fit_transform(X)  # Ajusta PCA y proyecta: obtiene coordenadas (n_samples,2)

print(f"Shape original: {X.shape}")  # Reporta shape original: (n,3) para verificar dimensiones del dataset
print(f"Shape reducido: {X_pca.shape}")  # Reporta shape reducido: debe ser (n,2) por n_components=2
print(f"Varianza explicada: {pca.explained_variance_ratio_}")  # Muestra varianza por componente: debería ser alta en datos correlacionados
print(f"Varianza total: {np.sum(pca.explained_variance_ratio_):.2%}")  # Suma varianza explicada: indica cuánto conserva la reducción
```

### 3.5 Reconstrucción y Error

```python
import numpy as np  # Importa NumPy: se usa para sum/mean/cumsum/argmax y cálculo de errores en utilidades de PCA

def reconstruction_error(X: np.ndarray, pca: PCA) -> float:  # Error de reconstrucción relativo: cuantifica pérdida al proyectar y reconstruir
    """
    Calcula el error de reconstrucción.

    Error = ||X - X_reconstructed||² / ||X||²
    """
    X_pca = pca.transform(X)  # Proyecta X al subespacio PCA: obtiene coordenadas de dimensión reducida
    X_reconstructed = pca.inverse_transform(X_pca)  # Reconstruye a espacio original: aproxima X usando solo n_components

    error = np.sum((X - X_reconstructed) ** 2)  # SSE de reconstrucción: energía del residuo (cuánto “se perdió” al comprimir)
    total = np.sum((X - np.mean(X, axis=0)) ** 2)  # SSE total alrededor de la media: normaliza para obtener un error relativo comparable

    return error / total  # Retorna fracción de varianza no explicada (aprox): más bajo implica mejor reconstrucción

def choose_n_components(X: np.ndarray, variance_threshold: float = 0.95) -> int:  # Elige k mínimo tal que la varianza acumulada supere un umbral
    """
    Elige número de componentes para retener cierta varianza.

    Args:
        variance_threshold: proporción de varianza a retener (ej: 0.95 = 95%)
    """
    # PCA con todos los componentes
    pca = PCA(n_components=min(X.shape))  # Ajusta PCA con el máximo posible: k=min(n_samples,n_features) para capturar toda la varianza
    pca.fit(X)  # Entrena PCA completo: llena explained_variance_ratio_ para luego acumular y decidir k

    # Varianza acumulada
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)  # Suma acumulada: varianza explicada por las primeras j componentes

    # Encontrar n_components
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1  # Primer índice donde se supera el umbral (+1 por 0-index)

    return n_components, cumulative_variance  # Devuelve k elegido y la curva acumulada: permite auditar visualmente la decisión
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
- Registra hallazgos en `Herramientas_Estudio/DIARIO_ERRORES.md`.
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V5_ESTRATEGICO.md)

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

- [PLAN_V4_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V4_ESTRATEGICO.md)
- [PLAN_V5_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V5_ESTRATEGICO.md)
- Diario: `Herramientas_Estudio/DIARIO_ERRORES.md`

## 💻 Parte 4: Aplicaciones de PCA

### 4.1 Compresión de Imágenes

```python
import numpy as np  # Importa NumPy: representa la imagen como array y soporta operaciones vectorizadas usadas por PCA

def compress_image_pca(image: np.ndarray, n_components: int) -> tuple:  # Compresión PCA: reduce dimensionalidad de filas y reconstruye una aproximación
    """
    Comprime una imagen usando PCA.

    Args:
        image: imagen grayscale (height, width)
        n_components: número de componentes a retener

    Returns:
        imagen comprimida, pca model
    """
    # Tratar filas como muestras
    pca = PCA(n_components=n_components)  # Instancia PCA: retiene k componentes para comprimir cada fila (ancho) de la imagen
    image_pca = pca.fit_transform(image)  # Proyecta filas: (height,width)->(height,k) reduce dimensión horizontal conservando varianza

    # Reconstruir
    image_reconstructed = pca.inverse_transform(image_pca)  # Reconstruye a width original: aproxima la imagen usando solo k componentes

    return image_reconstructed, pca  # Devuelve imagen reconstruida y el modelo PCA: permite inspeccionar varianza/errores

def compression_ratio_pca(original_shape: tuple, n_components: int) -> float:  # Estima ratio de compresión: compara números a guardar vs tamaño original (heurístico)
    """Calcula ratio de compresión."""
    height, width = original_shape  # Desempaqueta shape: alto y ancho para estimar tamaños de almacenamiento
    original_size = height * width  # Tamaño original: número de píxeles (asumiendo 1 valor por pixel)
    # Almacenamos: componentes + proyecciones + media
    compressed_size = n_components * width + height * n_components + width  # Estima parámetros a guardar: componentes + scores + media (aprox)
    return compressed_size / original_size  # Ratio: <1 implica compresión (menos números que almacenar que la imagen original)
```

### 4.2 Visualización en 2D

```python
import numpy as np  # Importa NumPy: se usa para obtener labels únicos y crear máscaras booleanas de selección
import matplotlib.pyplot as plt  # Importa Matplotlib: se usa para dibujar el scatter 2D y la leyenda/ejes

def visualize_pca_2d(X: np.ndarray, labels: np.ndarray = None, title: str = "PCA"):  # Reduce a 2D con PCA y grafica (coloreando por label si existe)
    """Reduce a 2D y visualiza."""
    pca = PCA(n_components=2)  # Instancia PCA 2D: elige 2 componentes para poder graficar en un plano
    X_2d = pca.fit_transform(X)  # Ajusta y proyecta: transforma X (n,d) a coordenadas (n,2)

    plt.figure(figsize=(10, 6))  # Crea figura: define tamaño para legibilidad (ancho/alto en pulgadas)

    if labels is not None:  # Si hay etiquetas, colorea por clase/cluster para interpretar separación en el plano PCA
        for label in np.unique(labels):  # Itera clases únicas: crea una nube por label para la leyenda
            mask = labels == label  # Máscara booleana: selecciona puntos que pertenecen a la clase actual
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1],  # Scatter por clase: x=PC1, y=PC2 para los puntos filtrados
                       label=f'Clase {label}', alpha=0.7)  # Etiqueta/alpha: identifica clase y hace puntos semi-translúcidos
        plt.legend()  # Muestra leyenda: permite mapear colores a clases
    else:  # Caso sin labels: se grafica todo en un solo color para ver estructura global sin segmentación
        plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)  # Scatter sin labels: muestra la estructura global sin separar por clase

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')  # Etiqueta eje x: incluye % varianza explicada por PC1
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')  # Etiqueta eje y: incluye % varianza explicada por PC2
    plt.title(title)  # Título del gráfico: permite contextualizar dataset/experimento
    plt.grid(True, alpha=0.3)  # Grilla suave: mejora lectura de densidades/posiciones
    plt.show()  # Renderiza la figura: despliega el plot en pantalla/notebook
```

---

## 🎯 Ejercicios por tema (progresivos) + Soluciones

Reglas:

- **Intenta primero** sin mirar la solución.
- **Timebox sugerido:** 25–60 min por ejercicio.
- **Éxito mínimo:** tu solución debe pasar los `assert`.

---

### Ejercicio 6.1: Distancias vectorizadas (K-Means) - shapes y argmin

#### Enunciado

1) **Básico**

- Dado `X` con shape `(n,d)` y centroides `C` con shape `(k,d)`, construye una matriz `D2` con shape `(n,k)` donde `D2[i,j] = ||X_i - C_j||^2`.

2) **Intermedio**

- Obtén asignaciones `labels = argmin_j D2[i,j]`.

3) **Avanzado**

- Verifica por `assert` que el resultado coincide con un cálculo manual en un punto.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para construir arrays demo, broadcasting y verificaciones numéricas con asserts

X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])  # Dataset pequeño (n=4,d=2): puntos 2D para validar distancias y argmin
C = np.array([[0.0, 0.0], [2.0, 2.0]])  # Centroides (k=2,d=2): dos centros candidatos para asignación por distancia^2

# (n,1,d) - (1,k,d) -> (n,k,d)
diff = X[:, None, :] - C[None, :, :]  # Broadcasting: resta cada centroide a cada punto para obtener tensor (n,k,d) de diferencias
D2 = np.sum(diff ** 2, axis=2)  # Distancias^2 (n,k): suma sobre d para obtener ||X_i - C_j||^2 sin sqrt (más eficiente)

assert D2.shape == (X.shape[0], C.shape[0])  # Verifica shape: debe ser (n,k) para poder hacer argmin por punto

labels = np.argmin(D2, axis=1)  # Asignación: elige el centroide más cercano por punto (argmin sobre k)
assert labels.shape == (X.shape[0],)  # Verifica shape de labels: un label por muestra
assert labels.min() >= 0 and labels.max() < C.shape[0]  # Verifica rango: labels debe ser un índice válido en [0, k-1]

i = 2  # X[i] = [0,2]
manual0 = np.sum((X[i] - C[0]) ** 2)  # Distancia^2 manual a C0: sirve para comprobar que el cálculo vectorizado es correcto
manual1 = np.sum((X[i] - C[1]) ** 2)  # Distancia^2 manual a C1: segunda comparación para el mismo punto i
assert np.isclose(D2[i, 0], manual0)  # Verifica D2 vectorizado vs manual: entrada (i,0) coincide numéricamente
assert np.isclose(D2[i, 1], manual1)  # Verifica D2 vectorizado vs manual: entrada (i,1) coincide numéricamente
assert labels[i] == int(np.argmin([manual0, manual1]))  # Verifica argmin: el label debe coincidir con el mínimo de las distancias manuales
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 6.1: Distancias vectorizadas (shapes + broadcasting + argmin)</strong></summary>

#### 1) Metadatos
- **Título:** De `||x-c||²` a una matriz `(n,k)` sin loops
- **ID (opcional):** `M06-E06_1`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio
- **Dependencias:** Broadcasting + `axis` (M01), norma L2 (M02)

#### 2) Objetivos
- Construir `D2:(n,k)` sin bucles sobre `n` ni `k`.
- Elegir el `axis` correcto en `sum` y `argmin`.
- Debuggear shapes con un ejemplo pequeño y verificable.

#### 3) Errores comunes
- Reducir el eje equivocado en `sum` (debe ser el eje de features `d`).
- Calcular `sqrt` sin necesidad (para `argmin`, dist y dist² ordenan igual).
- Usar `argmin(axis=0)` (contesta otra pregunta).

#### 4) Nota docente
- Pide que el alumno explique qué representa cada eje de `D2`.
</details>

---

### Ejercicio 6.2: Paso de actualización (centroides como promedio)

#### Enunciado

1) **Básico**

- Dado `X` y `labels`, recalcula `C_new[j] = mean(X[labels==j])`.

2) **Intermedio**

- Verifica shapes y que no aparecen `NaN`.

3) **Avanzado**

- Maneja el caso de cluster vacío: si no hay puntos para un `j`, conserva el centroide anterior.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para arrays, broadcasting, argmin y validación con np.isfinite en el ejemplo

X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])  # Dataset 2D: dos grupos alrededor de (0,0) y (10,10) para probar actualización de centroides
C = np.array([[0.0, 0.0], [10.0, 10.0]])  # Centroides iniciales: uno por cada grupo para asignación por distancia^2

diff = X[:, None, :] - C[None, :, :]  # Tensor (n,k,d): diferencias punto-centroide por broadcasting para calcular distancias en lote
labels = np.argmin(np.sum(diff ** 2, axis=2), axis=1)  # Asignación por distancia^2: elige el centroide más cercano para cada punto

C_new = C.copy()  # Inicializa centroides nuevos: copia para poder conservar centroides si un cluster queda vacío
for j in range(C.shape[0]):  # Recorre clusters: recalcula centroide j como promedio de los puntos asignados
    mask = labels == j  # Máscara booleana: selecciona puntos del cluster j
    if np.any(mask):  # Evita cluster vacío: sin puntos, mean produciría NaN y rompería el algoritmo
        C_new[j] = np.mean(X[mask], axis=0)  # Actualiza centroide: media por feature minimiza SSE con labels fijos

assert C_new.shape == C.shape  # Verifica shape: la actualización no debe cambiar dimensionalidad ni número de centroides
assert np.isfinite(C_new).all()  # Verifica finitud: asegura que no se generaron NaN/inf por clusters vacíos u operaciones inválidas
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 6.2: Actualización de centroides (promedios + clusters vacíos)</strong></summary>

#### 1) Metadatos
- **Título:** Por qué el centroide es la media (y qué hacer si un cluster queda vacío)
- **ID (opcional):** `M06-E06_2`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio

#### 2) Ideas clave
- Con `labels` fijos, la media minimiza `Σ ||x-μ||²`.
- Si `labels==j` no selecciona puntos, `mean` sobre slice vacío produce `NaN`.

#### 3) Estrategias para cluster vacío
- Conservar el centroide anterior (simple y estable).
- Reinicializar en un punto aleatorio de `X`.
- Reinicializar en el punto con mayor error (más avanzado).

#### 4) Errores comunes
- Promediar con `axis=1` (debe ser `axis=0` para obtener un vector `(d,)`).
- No validar con `np.isfinite` y propagar `NaN`.

#### 5) Nota docente
- Pide que el alumno cree a propósito un cluster vacío y explique el fallo.
</details>

---

### Ejercicio 6.3: Inercia (función objetivo) + monotonía de Lloyd

#### Enunciado

1) **Básico**

- Implementa `inertia(X, C, labels) = sum_i ||X_i - C_{labels_i}||^2`.

2) **Intermedio**

- Ejecuta 1 iteración de Lloyd (asignación → actualización) y compara inercia.

3) **Avanzado**

- Verifica que la inercia **no aumenta** tras la iteración (debe bajar o quedar igual).

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para broadcasting, sumas, argmin y generación de datos sintéticos

def assign_labels(X: np.ndarray, C: np.ndarray) -> np.ndarray:  # Asigna cada punto al centroide más cercano (Lloyd step: asignación)
    D2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)  # Distancias^2 a cada centroide: (n,1,d)-(1,k,d)->(n,k,d) y suma en d
    return np.argmin(D2, axis=1)  # Label por punto: índice del centroide con menor distancia^2


def update_centroids(X: np.ndarray, labels: np.ndarray, C: np.ndarray) -> np.ndarray:  # Recalcula centroides como media de puntos asignados (Lloyd step: actualización)
    C_new = C.copy()  # Copia centroides: evita modificar C in-place (mantiene comparaciones/diagnóstico coherentes)
    for j in range(C.shape[0]):  # Itera cada cluster j: actualiza su centro si tiene puntos asignados
        mask = labels == j  # Máscara booleana: selecciona los puntos cuya etiqueta es j
        if np.any(mask):  # Solo actualiza si hay puntos: evita mean sobre vacío y conserva centroide si cluster quedó vacío
            C_new[j] = np.mean(X[mask], axis=0)  # Nuevo centroide: media minimiza SSE intra-cluster para labels fijos
    return C_new  # Devuelve centroides actualizados: se usan en la siguiente asignación


def inertia(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:  # Inercia/SSE: suma distancias^2 de puntos a su centroide asignado
    diffs = X - C[labels]  # Residuales por punto: resta el centroide correspondiente a cada label (indexación avanzada)
    return float(np.sum(diffs ** 2))  # SSE total: escalar float útil para asserts/prints (no depende de dtype)


np.random.seed(0)  # Fija semilla: hace reproducible el experimento/validación
X = np.vstack([  # Construye dataset: concatena dos nubes gaussianas (dos clusters) para probar Lloyd
    np.random.randn(50, 2) + np.array([0.0, 0.0]),  # Cluster 0: 50 puntos alrededor de (0,0)
    np.random.randn(50, 2) + np.array([5.0, 5.0]),  # Cluster 1: 50 puntos alrededor de (5,5)
])  # Cierra vstack: X queda con shape (100,2)

C0 = np.array([[0.0, 5.0], [5.0, 0.0]])  # Centroides iniciales “cruzados”: no coinciden con centros reales a propósito
labels0 = assign_labels(X, C0)  # Asignación inicial (E-step): etiqueta por punto según C0
J0 = inertia(X, C0, labels0)  # Inercia inicial: SSE antes de la actualización de centroides

C1 = update_centroids(X, labels0, C0)  # Actualiza centroides (M-step): recomputa C con labels0 fijos
labels1 = assign_labels(X, C1)  # Re-asigna con centroides nuevos: completa 1 iteración de Lloyd
J1 = inertia(X, C1, labels1)  # Inercia tras 1 iteración: debe no aumentar (monotonía)

assert J1 <= J0 + 1e-12  # Monotonía de Lloyd: la inercia baja o se mantiene (tolerancia por flotantes)
assert J0 >= 0.0 and J1 >= 0.0  # Inercia no-negativa: suma de cuadrados nunca debe ser negativa
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 6.3: Inercia y monotonía de Lloyd (convergencia ≠ óptimo global)</strong></summary>

#### 1) Metadatos
- **Título:** Qué mide la inercia y por qué Lloyd la baja
- **ID (opcional):** `M06-E06_3`
- **Duración estimada:** 30–75 min
- **Nivel:** Intermedio

#### 2) Idea central
- Asignación: con `C` fijo, elegir el centro más cercano minimiza `J` respecto a `labels`.
- Actualización: con `labels` fijos, poner cada centro en la media minimiza `J` respecto a `C`.
- Alternar ambos pasos ⇒ `J` baja o queda igual.

#### 3) Convergencia ≠ óptimo global
- Lloyd converge, pero depende de la inicialización y puede caer en mínimos locales.
- Por eso K-Means++ y reinicios múltiples son estándar.

#### 4) Debugging
- Si `J` aumenta, casi siempre es un error de `axis`, indexado (`C[labels]`) o `NaN`.

#### 5) Nota docente
- Pide que el alumno explique en 2 líneas: “converge” vs “mejor clustering posible”.
</details>

---

### Ejercicio 6.4: K-Means++ (probabilidades correctas)

#### Enunciado

1) **Básico**

- Implementa K-Means++ para elegir `k` centroides desde `X`.

2) **Intermedio**

- Verifica que los centroides pertenecen a `X`.

3) **Avanzado**

- Verifica que las probabilidades de muestreo suman 1 (en cada paso).

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para RNG moderno, operaciones vectorizadas y aserciones numéricas en K-Means++

def kmeans_plus_plus(X: np.ndarray, k: int, seed: int = 0) -> np.ndarray:  # Inicialización K-Means++: elige centroides separados para mejorar el arranque de Lloyd
    rng = np.random.default_rng(seed)  # Crea generador RNG local: evita depender del estado global de np.random y hace reproducible la selección
    n = X.shape[0]  # Número de muestras: define el rango válido de índices para muestrear puntos de X
    centroids = [X[rng.integers(n)]]  # Elige primer centroide uniforme: toma un punto real de X para iniciar la lista de centroides

    for _ in range(1, k):  # Itera para seleccionar los k-1 centroides restantes: en cada paso recalcula distancias al centroide más cercano
        C = np.array(centroids)  # Apila centroides actuales a array (c,d): facilita broadcasting contra X para calcular distancias en lote
        d2 = np.min(np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2), axis=1)  # d² mínimo por punto: distancia^2 al centroide más cercano (para probabilidad K-Means++)
        assert np.all(d2 >= 0)  # Chequeo: distancias cuadradas no deben ser negativas (sirve para detectar NaNs/errores numéricos)
        probs = d2 / np.sum(d2)  # Normaliza a distribución: cada punto se elige con probabilidad proporcional a su d² (más lejos => más probable)
        assert np.isclose(np.sum(probs), 1.0)  # Valida normalización: la suma de probabilidades debe ser 1 (tolerancia de float)
        centroids.append(X[rng.choice(n, p=probs)])  # Samplea nuevo centroide según probs: implementa la regla de muestreo de K-Means++

    return np.array(centroids)  # Devuelve centroides iniciales: shape (k, n_features) para arrancar el loop de K-Means


np.random.seed(1)  # Fija semilla global: hace reproducible la generación de datos sintéticos del ejemplo
X = np.random.randn(30, 2)  # Genera dataset demo (30,2): puntos 2D para verificar que los centroides elegidos pertenecen a X
C = kmeans_plus_plus(X, k=3, seed=123)  # Inicializa 3 centroides con K-Means++: se valida output y pertenencia a X

assert C.shape == (3, 2)  # Verifica shape: debe haber k centroides y cada uno con d=2 features
for j in range(C.shape[0]):  # Recorre centroides devueltos: valida que cada centroide sea exactamente uno de los puntos del dataset
    assert np.any(np.all(np.isclose(X, C[j]), axis=1))  # Chequea pertenencia: existe una fila en X (casi igual) a cada centroide C[j]
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 6.4: K-Means++ (probabilidades correctas)</strong></summary>

#### 1) Metadatos
- **Título:** Inicialización que reduce mínimos locales malos
- **ID (opcional):** `M06-E06_4`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio

#### 2) Idea clave
- K-Means++ elige nuevos centroides con probabilidad proporcional a la distancia² al centroide más cercano.
- Intuición: fuerza a que los centroides iniciales queden separados, cubriendo mejor el espacio.

#### 3) Chequeos importantes
- `d2 >= 0` siempre (son distancias cuadradas).
- `probs` debe sumar 1.
- Los centroides seleccionados deben ser puntos existentes de `X`.

#### 4) Caso borde
- Si todos los puntos ya están a distancia 0 de algún centroide (`sum(d2)=0`), no hay señal para muestrear: en práctica puedes romper el loop o elegir aleatorio.

#### 5) Nota docente
- Pide que el alumno compare K-Means con init aleatoria vs K-Means++ en un dataset con dos clusters separados.
</details>

---

### Ejercicio 6.5: Sensibilidad a escala (por qué normalizar importa)

#### Enunciado

1) **Básico**

- Construye un ejemplo donde escalar una feature cambie la asignación al centroide más cercano.

2) **Intermedio**

- Calcula labels con una escala `s=0.1` y con `s=10`.

3) **Avanzado**

- Verifica que hay al menos un punto cuyo label cambia.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para broadcasting, argmin y scaling por feature para ilustrar sensibilidad a escala

def assign_labels(X: np.ndarray, C: np.ndarray) -> np.ndarray:  # Asigna labels por distancia^2: mismo criterio que K-Means (sin actualizar centroides)
    D2 = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)  # Matriz (n,k) de distancias^2: usa broadcasting para evitar loops
    return np.argmin(D2, axis=1)  # Devuelve label por punto: índice del centroide más cercano (mínimo sobre k)


# Punto cerca en x pero lejos en y (y domina si la escalas)
X = np.array([  # Define punto de prueba: se elige para que el segundo eje pueda dominar al escalarlo
    [2.0, 0.0],  # Punto (x=2,y=0): su distancia depende de cuánto pese el eje y en la métrica euclídea
], dtype=float)  # Fuerza dtype float: evita enteros y hace explícitas operaciones de scaling y distancias
C = np.array([  # Define dos centroides: uno en origen y otro en (2,2) para que el punto cambie de asignación con scaling
    [0.0, 0.0],  # Centroide 0: origen, cercano en y cuando y está poco escalado
    [2.0, 2.0],  # Centroide 1: comparte x con el punto pero difiere en y, clave para provocar cambio al escalar y
], dtype=float)  # dtype float: mantiene coherencia numérica con X para comparaciones de distancia

labels_s_small = assign_labels(X * np.array([1.0, 0.1]), C * np.array([1.0, 0.1]))  # Escala y por 0.1: reduce su contribución a la distancia
labels_s_big = assign_labels(X * np.array([1.0, 10.0]), C * np.array([1.0, 10.0]))  # Escala y por 10: amplifica su contribución y puede cambiar el argmin

assert labels_s_small.shape == (1,)  # Verifica shape: un label para el único punto en X
assert labels_s_big.shape == (1,)  # Verifica shape: se preserva el formato tras el re-escalado
assert labels_s_small[0] != labels_s_big[0]  # Verifica sensibilidad: el label debe cambiar al modificar escalas (distancia euclídea cambia)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 6.5: Sensibilidad a escala (normalización)</strong></summary>

#### 1) Metadatos
- **Título:** Por qué K-Means necesita features comparables
- **ID (opcional):** `M06-E06_5`
- **Duración estimada:** 20–45 min
- **Nivel:** Intermedio

#### 2) Idea clave
- K-Means optimiza distancias euclidianas: si una feature está en escala 100× mayor, domina la distancia.

#### 3) Regla práctica
- Antes de K-Means/PCA, suele ser obligatorio:
  - estandarizar (media 0, var 1) o
  - normalizar por rango, según el dominio.

#### 4) Nota docente
- Pide que el alumno explique por qué “normalizar cambia el significado de ‘cerca’”.
</details>

---

### Ejercicio 6.6: PCA vía SVD (shapes + varianza explicada ordenada)

#### Enunciado

1) **Básico**

- Centra `X` y calcula `U,S,Vt = svd(Xc)`.

2) **Intermedio**

- Proyecta a `k=2` componentes y verifica shapes.

3) **Avanzado**

- Calcula varianza explicada y verifica que está ordenada de mayor a menor.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para SVD, generación de datos sintéticos y validación de shapes/propiedades

def pca_svd(X: np.ndarray, k: int):  # PCA por SVD: aprende componentes principales sin formar la covarianza explícita
    Xc = X - X.mean(axis=0)  # Centra datos: PCA requiere media cero por feature para que SVD capture direcciones de máxima varianza
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # SVD compacta: Xc=U diag(S) Vt; Vt contiene direcciones principales en espacio de features
    comps = Vt[:k].T  # Selecciona k componentes: toma primeras k filas de Vt y transpone => matriz (d,k) para proyectar
    Xk = Xc @ comps  # Proyección a k dims: (n,d)@(d,k)->(n,k) da coordenadas en el subespacio PCA
    var = (S ** 2) / (Xc.shape[0] - 1)  # Varianza por componente: S^2/(n-1) equivale a eigenvalues de covarianza (cuando X está centrado)
    ratio = var / np.sum(var)  # Ratio de varianza explicada: normaliza para obtener proporciones que suman 1 sobre todas las componentes
    return Xk, comps, ratio[:k]  # Devuelve proyección, componentes (d,k) y ratios truncados (k,) para inspección/validación


np.random.seed(0)  # Fija semilla global: hace reproducible el dataset sintético usado para validar PCA
n = 300  # Número de muestras: controla tamaño del dataset (más n => estimaciones de varianza más estables)
z = np.random.randn(n)  # Latente 1D: variable base que induce correlaciones lineales entre columnas
X = np.stack([z, 2.0 * z + 0.1 * np.random.randn(n), -z + 0.1 * np.random.randn(n)], axis=1)  # Construye X (n,3): features correlacionadas para que PCA tenga estructura

X2, comps, r = pca_svd(X, k=2)  # Aplica PCA a 2 componentes: obtiene proyección 2D, matriz de componentes y ratios de varianza

assert X2.shape == (n, 2)  # Verifica proyección: n filas (muestras) y k=2 columnas (componentes)
assert comps.shape == (3, 2)  # Verifica componentes: d=3 features originales y k=2 ejes principales retenidos
assert r.shape == (2,)  # Verifica ratios: debe haber exactamente k proporciones de varianza explicada
assert r[0] >= r[1]  # Verifica orden: la primera componente debe explicar >= varianza que la segunda
assert 0.0 <= r.sum() <= 1.0  # Verifica rango: suma parcial de ratios debe estar entre 0 y 1 (al truncar, suele ser < 1)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 6.6: PCA vía SVD (shapes + varianza explicada)</strong></summary>

#### 1) Metadatos
- **Título:** PCA estable en código (SVD) sin construir covarianza
- **ID (opcional):** `M06-E06_6`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio/Avanzado

#### 2) Shapes que debes poder justificar
- `X:(n,d)` → `Xc:(n,d)` (centrado)
- `Vt:(d,d)` (o `(d,rank)` si `full_matrices=False` y `n<d`)
- `comps = Vt[:k].T → (d,k)`
- `Xk = Xc @ comps → (n,k)`

#### 3) Varianza explicada
- Con SVD, los valores singulares `S` te dan varianzas: `var = S^2/(n-1)`.
- El ratio `var/sum(var)` indica qué porcentaje explica cada componente.

#### 4) Nota docente
- Pide que el alumno explique por qué centrar `X` es obligatorio para PCA.
</details>

---

### Ejercicio 6.7: Reconstrucción PCA (error decrece al aumentar componentes)

#### Enunciado

1) **Básico**

- Reconstruye `X` desde `k` componentes: `X_rec = Xc @ V_k @ V_k^T + mean`.

2) **Intermedio**

- Compara el error de reconstrucción con `k=1` vs `k=2`.

3) **Avanzado**

- Verifica que el error con `k=2` es menor o igual.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para SVD, medias, normas y generación de datos sintéticos del ejercicio

def pca_reconstruct(X: np.ndarray, k: int) -> np.ndarray:  # Reconstrucción PCA: proyecta a k dims y vuelve al espacio original (aprox)
    mu = X.mean(axis=0)  # Calcula media por feature: se usa para centrar y luego descentrar (reconstrucción en el sistema original)
    Xc = X - mu  # Centra X: PCA trabaja sobre datos de media cero para que los ejes representen covarianza
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # SVD: obtiene base ortonormal de componentes; Vt contiene vectores principales
    Vk = Vt[:k].T  # Toma subespacio de dimensión k: matriz (d,k) con los k ejes principales (columnas)
    Xk = Xc @ Vk  # Proyección a subespacio: coordenadas (n,k) en la base PCA truncada
    X_rec = Xk @ Vk.T + mu  # Reconstruye: vuelve a (n,d) aplicando el proyector Vk Vk^T y re-suma la media
    return X_rec  # Devuelve reconstrucción aproximada: error debe no aumentar al incrementar k


np.random.seed(1)  # Fija semilla global: hace reproducible el dataset sintético para comparar errores de reconstrucción
n = 200  # Número de muestras: tamaño del dataset para el test de monotonía del error
z = np.random.randn(n)  # Latente 1D: induce correlación lineal entre columnas para que PCA capture estructura en pocas componentes
X = np.stack([z, 2.0 * z + 0.2 * np.random.randn(n), -z + 0.2 * np.random.randn(n)], axis=1)  # Construye X (n,3): features correlacionadas + ruido

X1 = pca_reconstruct(X, k=1)  # Reconstrucción con 1 componente: mayor compresión => típicamente mayor error
X2 = pca_reconstruct(X, k=2)  # Reconstrucción con 2 componentes: subespacio más grande => error no debe aumentar

err1 = np.linalg.norm(X - X1)  # Error de reconstrucción k=1: norma Frobenius (por defecto) del residuo total
err2 = np.linalg.norm(X - X2)  # Error de reconstrucción k=2: debería ser <= err1 por propiedad de proyecciones ortogonales

assert err2 <= err1 + 1e-12  # Verifica monotonía: permitir epsilon numérico por redondeo en SVD/multiplicaciones
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 6.7: Reconstrucción PCA (sesgo vs compresión)</strong></summary>

#### 1) Metadatos
- **Título:** Más componentes ⇒ menos error (pero menos compresión)
- **ID (opcional):** `M06-E06_7`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio

#### 2) Idea clave
- `Vk Vk^T` es el proyector al subespacio de dimensión `k`.
- Si aumentas `k`, el subespacio crece y la proyección puede “explicar” más energía ⇒ el error no aumenta.

#### 3) Nota docente
- Pide que el alumno conecte “error de reconstrucción” con “varianza explicada acumulada”.
</details>

---

### (Bonus) Ejercicio 6.8: Silhouette (implementación mínima para dataset pequeño)

#### Enunciado

- Implementa silhouette para un dataset pequeño.
- Verifica que el score promedio está en `[-1, 1]`.

#### Solución

```python
import numpy as np  # Importa NumPy: se usa para álgebra vectorizada, broadcasting y funciones de agregación

def pairwise_dist(X: np.ndarray) -> np.ndarray:  # Distancias pairwise: construye matriz (n,n) de distancias Euclídeas
    D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)  # Distancias^2 por broadcasting: (n,1,d)-(1,n,d)->(n,n,d) y suma en d
    return np.sqrt(np.maximum(D2, 0.0))  # Raíz para Euclídea y clamp numérico: evita sqrt de valores negativos por redondeo


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:  # Silhouette promedio: s(i)=(b(i)-a(i))/max(a(i),b(i)) en [-1,1]
    X = np.asarray(X, dtype=float)  # Normaliza entrada a float: garantiza operaciones de distancia y medias en tipo numérico estable
    labels = np.asarray(labels, dtype=int)  # Normaliza labels a int: facilita comparaciones e indexación booleana por cluster
    D = pairwise_dist(X)  # Precalcula distancias pairwise (n,n): se reutiliza para a(i) y b(i) sin recomputar distancias
    n = X.shape[0]  # Número de puntos: controla el loop que calcula s(i) por cada muestra
    uniq = np.unique(labels)  # Clusters únicos presentes: define el conjunto de clusters a evaluar para b(i)
    s = np.zeros(n, dtype=float)  # Vector de silhouettes por punto: se promedia al final
    for i in range(n):  # Recorre cada punto i: silhouette se define punto a punto
        same = labels == labels[i]  # Máscara del cluster de i: selecciona puntos del mismo cluster
        same[i] = False  # Excluye el propio punto: evita distancia 0 consigo mismo en el promedio intra-cluster
        a = np.mean(D[i, same]) if np.any(same) else 0.0  # a(i): distancia media intra-cluster; 0 si i está solo en su cluster

        b = np.inf  # Inicializa b(i): buscamos el mínimo promedio a cualquier cluster distinto (si no hay, queda inf)
        for c in uniq:  # Itera clusters candidatos: calcula distancia media de i a cada cluster distinto
            if c == labels[i]:  # Omite el cluster propio: b(i) se define sobre otros clusters
                continue  # Salta a siguiente cluster candidato
            mask = labels == c  # Máscara del cluster candidato c: selecciona sus puntos
            if np.any(mask):  # Asegura que hay puntos en el cluster: evita mean sobre array vacío
                b = min(b, float(np.mean(D[i, mask])))  # Actualiza mejor b(i): toma el cluster con menor distancia media

        if b == np.inf:  # Edge case: no existe otro cluster válido (o labels degenerados), entonces b(i) no se define
            s[i] = 0.0  # Convención: score neutral si no hay comparación posible
        else:  # Caso normal: existe otro cluster para comparar, así que podemos calcular s(i) con a(i) y b(i)
            denom = max(a, b)  # Denominador estándar: normaliza para acotar en [-1,1] y evita dividir por valores pequeños
            s[i] = 0.0 if denom == 0.0 else (b - a) / denom  # Calcula s(i): si denom=0 (distancias 0), fuerza 0 para evitar NaN
    return float(np.mean(s))  # Promedio final: silhouette global del clustering


X = np.array([[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.1, 4.9]])  # Dataset mini 2D: dos clusters bien separados (cerca de (0,0) y (5,5))
labels = np.array([0, 0, 1, 1])  # Etiquetas de cluster: agrupa los dos primeros y los dos últimos
score = silhouette_score(X, labels)  # Calcula silhouette: debería ser positivo si clusters están bien definidos
assert -1.0 <= score <= 1.0  # Invariante del score: silhouette siempre cae en el rango [-1,1]
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 6.8: Silhouette (intuición y límites)</strong></summary>

#### 1) Metadatos
- **Título:** Métrica de clustering “interna” (sin etiquetas)
- **ID (opcional):** `M06-E06_8`
- **Duración estimada:** 30–75 min
- **Nivel:** Avanzado

#### 2) Intuición
- Para cada punto:
  - `a` = distancia media a su propio cluster
  - `b` = mejor (mínima) distancia media a otro cluster
- `s = (b-a)/max(a,b)` ∈ [-1, 1]

#### 3) Limitaciones
- Requiere distancias pairwise: costo O(n²) (por eso lo hacemos “mini”).
- Depende de la métrica de distancia (igual que K-Means).

#### 4) Nota docente
- Pide que el alumno interprete 3 casos: `s≈1`, `s≈0`, `s<0`.
</details>

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

import numpy as np  # Importa NumPy: base para álgebra lineal, RNG, broadcasting y operaciones vectorizadas en clustering/PCA
from typing import Tuple, List  # Importa typing: documenta retornos/colecciones (no afecta runtime)


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

def kmeans_plus_plus(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:  # Inicialización K-Means++: elige centroides separados para mejorar convergencia
    """Inicialización K-Means++."""  # Docstring 1-línea: describe propósito; se ejecuta como literal string y no cambia el algoritmo
    if seed: np.random.seed(seed)  # Fija semilla si es truthy: hace reproducible la inicialización (nota: seed=0 no entra por este if)
    n = len(X)  # Número de muestras: se usa para muestrear índices válidos al escoger centroides
    centroids = [X[np.random.randint(n)]]  # Elige primer centroide al azar: punto inicial para el esquema de selección probabilística

    for _ in range(1, k):  # Selecciona los k-1 centroides restantes: cada paso agrega un centroide nuevo
        distances = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])  # Distancia^2 al centroide más cercano: define qué tan “mal cubierto” está cada punto
        probs = distances / distances.sum()  # Normaliza a distribución: puntos más lejanos tienen mayor probabilidad de ser elegidos
        centroids.append(X[np.random.choice(n, p=probs)])  # Samplea nuevo centroide según probs: mejora separación inicial de clusters

    return np.array(centroids)  # Devuelve centroides iniciales (k,d): salida que se usa como init en K-Means/Lloyd


class KMeans:  # Implementación simple de K-Means: alterna asignación de clusters y actualización de centroides hasta converger
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, seed=None):  # Configura hiperparámetros (k, iteraciones, tolerancia, semilla)
        self.n_clusters = n_clusters  # k: número de clusters/centroides a aprender
        self.max_iter = max_iter  # Límite de iteraciones: evita loops infinitos si no converge
        self.tol = tol  # Tolerancia: umbral para decidir convergencia (en este código, compara shift cuadrático)
        self.seed = seed  # Semilla opcional: se pasa a K-Means++ para reproducibilidad de inicialización
        self.centroids = None  # Centroides aprendidos: se setea en fit y luego se usa en predict
        self.labels_ = None  # Etiquetas por muestra (cluster asignado): output principal del clustering
        self.inertia_ = None  # Inercia final: suma de distancias cuadradas intra-cluster (métrica interna)
        self.n_iter_ = 0  # Iteraciones ejecutadas: útil para diagnóstico (convergió rápido vs lento)

    def fit(self, X: np.ndarray) -> 'KMeans':  # Entrena K-Means sobre X: aprende centroides y asignaciones
        self.centroids = kmeans_plus_plus(X, self.n_clusters, self.seed)  # Inicializa centroides: un buen init reduce iteraciones y malos mínimos

        for i in range(self.max_iter):  # Loop EM-like: alterna asignación (E-step) y actualización (M-step)
            old_centroids = self.centroids.copy()  # Guarda centroides previos: permite medir desplazamiento para criterio de parada

            # Asignar
            distances = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])  # Matriz (n_samples,k) de distancias^2 a cada centroide
            self.labels_ = np.argmin(distances, axis=1)  # Asigna cada punto al centroide más cercano: minimiza SSE localmente

            # Actualizar
            for j in range(self.n_clusters):  # Recalcula cada centroide j usando los puntos asignados
                points = X[self.labels_ == j]  # Subconjunto del cluster j: todas las muestras cuyo label es j
                if len(points) > 0:  # Evita cluster vacío: si no hay puntos, se conserva el centroide anterior
                    self.centroids[j] = points.mean(axis=0)  # Nuevo centroide: promedio (minimiza SSE para ese cluster)

            if np.sum((self.centroids - old_centroids)**2) < self.tol:  # Criterio de convergencia: shift total cuadrático bajo tolerancia
                break  # Detiene iteraciones: ya no cambia significativamente la solución

        self.n_iter_ = i + 1  # Guarda iteraciones realmente ejecutadas (i es 0-index)
        self.inertia_ = sum(np.sum((X[self.labels_ == j] - self.centroids[j])**2)  # SSE por cluster: suma distancias^2 de puntos al centroide asignado
                          for j in range(self.n_clusters))  # Suma sobre todos los clusters: métrica interna usada en elbow method
        return self  # Permite chaining (kmeans.fit(X).labels_)

    def predict(self, X: np.ndarray) -> np.ndarray:  # Predice labels para nuevos datos usando centroides ya aprendidos
        distances = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])  # Distancias^2 a centroides aprendidos: (n,k)
        return np.argmin(distances, axis=1)  # Retorna índice del centroide más cercano para cada muestra

    def fit_predict(self, X: np.ndarray) -> np.ndarray:  # Convenience: entrena y devuelve labels en una sola llamada
        self.fit(X)  # Ejecuta entrenamiento: produce centroides y labels_
        return self.labels_  # Retorna labels aprendidas: evita llamar fit() y luego acceder a labels_


# ============================================================
# PCA
# ============================================================

class PCA:  # PCA vía SVD: aprende ejes principales y proyecta datos a un subespacio de menor dimensión
    def __init__(self, n_components: int = 2):  # Configura cuántas componentes (dimensión reducida) se desea retener
        self.n_components = n_components  # k: número de componentes principales a conservar
        self.components_ = None  # Matriz de componentes: se setea en fit (n_features, k)
        self.explained_variance_ratio_ = None  # Fracción de varianza explicada por cada componente: útil para decidir k
        self.mean_ = None  # Media por feature: necesaria para centrar en fit y para transformar/invertir consistentemente

    def fit(self, X: np.ndarray) -> 'PCA':  # Ajusta PCA: calcula media, componentes y varianza explicada a partir de X
        self.mean_ = X.mean(axis=0)  # Media por columna: centrar es obligatorio para PCA estándar (captura covarianza, no offset)
        X_centered = X - self.mean_  # Centra datos: elimina el sesgo de traslación para que SVD capture direcciones de máxima varianza

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # SVD compacta: descompone X centrado para extraer componentes principales (Vt)

        self.components_ = Vt[:self.n_components].T  # Toma las k filas principales de Vt y transpone: (n_features,k) para proyección X@components_
        variance = (S**2) / (len(X) - 1)  # Eigenvalues de covarianza: Σ^2/(n-1) corresponde a varianza por componente
        self.explained_variance_ratio_ = variance[:self.n_components] / variance.sum()  # Proporción explicada: normaliza por varianza total

        return self  # Devuelve instancia entrenada: permite chaining y acceso a componentes/ratios aprendidos

    def transform(self, X: np.ndarray) -> np.ndarray:  # Proyecta datos al subespacio PCA (dimensión k)
        return (X - self.mean_) @ self.components_  # Centra con la misma media aprendida y proyecta: (n,k)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:  # Atajo: fit + transform en una sola llamada
        self.fit(X)  # Aprende componentes y media
        return self.transform(X)  # Devuelve proyección PCA sin requerir llamada extra

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:  # Reconstruye aproximación en espacio original desde coordenadas PCA
        return X_pca @ self.components_.T + self.mean_  # Re-proyecta a features y re-agrega la media: reconstrucción pierde info si k<n_features


# ============================================================
# MÉTRICAS
# ============================================================

def inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:  # Inercia/SSE: suma de distancias^2 intra-cluster (métrica interna)
    """Within-cluster sum of squares."""  # Docstring 1-línea: define la métrica; cuenta como literal ejecutado
    return sum(np.sum((X[labels == i] - centroids[i])**2)  # SSE por cluster i: distancias^2 de sus puntos al centroide i
               for i in range(len(centroids)))  # Suma sobre todos los centroides: se usa en elbow method y diagnóstico

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:  # Silhouette promedio: combina cohesión (a) y separación (b) sin ground truth
    """Silhouette score promedio."""  # Docstring 1-línea: explica salida; se ejecuta como string literal
    n = len(X)  # Número de puntos: controla el loop externo del cálculo por muestra
    scores = []  # Acumula s_i por punto: luego se promedia para el score global

    for i in range(n):  # Recorre cada punto i: silhouette requiere evaluar su cohesión/separación relativa
        # a: distancia promedio intra-cluster
        same = X[labels == labels[i]]  # Puntos del mismo cluster que i (incluye a i): base para cohesión intra-cluster
        a = np.mean([np.sqrt(np.sum((X[i] - x)**2)) for x in same if not np.array_equal(x, X[i])])  # Distancia media a otros del mismo cluster (excluye i)

        # b: distancia promedio al cluster más cercano
        b = float('inf')  # Inicializa b con infinito: buscamos el mínimo promedio a cualquier cluster alternativo
        for label in np.unique(labels):  # Recorre clusters existentes: evalúa el “cluster vecino” más cercano en distancia promedio
            if label != labels[i]:  # Excluye el cluster propio: b se define como mejor cluster distinto
                other = X[labels == label]  # Puntos del cluster candidato: se usa para distancia media inter-cluster
                if len(other) > 0:  # Evita clusters vacíos: no aportan un promedio definido
                    b = min(b, np.mean([np.sqrt(np.sum((X[i] - x)**2)) for x in other]))  # Actualiza mínimo: elige cluster alternativo más cercano

        if b == float('inf'):  # Si no existió cluster alternativo válido (edge case), no se puede definir b correctamente
            scores.append(0)  # Convención simple: retorna 0 para ese punto (neutral)
        else:  # Caso normal: existe un cluster alternativo; se calcula s(i) comparando cohesión (a) vs separación (b)
            scores.append((b - a) / max(a, b))  # Fórmula silhouette: s=(b-a)/max(a,b) en [-1,1]; >0 indica buena asignación

    return np.mean(scores)  # Promedia s_i: define el silhouette global del clustering (más alto => mejor separación/cohesión)


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":  # Entry point: permite ejecutar este módulo como script para correr pruebas rápidas
    np.random.seed(42)  # Fija semilla global: hace reproducible el dataset sintético y por tanto los resultados del test

    # Test K-Means
    c1 = np.random.randn(50, 2) + [0, 0]  # Cluster 1: 50 puntos alrededor de (0,0)
    c2 = np.random.randn(50, 2) + [5, 5]  # Cluster 2: 50 puntos alrededor de (5,5)
    c3 = np.random.randn(50, 2) + [10, 0]  # Cluster 3: 50 puntos alrededor de (10,0)
    X = np.vstack([c1, c2, c3])  # Dataset final: concatena clusters (150,2) para probar K-Means y PCA

    kmeans = KMeans(n_clusters=3, seed=42)  # Instancia K-Means con k=3: coincide con la generación sintética
    labels = kmeans.fit_predict(X)  # Entrena y obtiene labels: debe separar aproximadamente los 3 grupos

    print(f"K-Means Inertia: {kmeans.inertia_:.2f}")  # Reporta inercia final: útil para comparar con otros k/datasets
    print(f"Silhouette Score: {silhouette_score(X, labels):.4f}")  # Reporta silhouette: idealmente cercano a 1 si clusters bien separados

    # Test PCA
    pca = PCA(n_components=2)  # Instancia PCA para reducir 2D (aquí X ya es 2D, sirve para validar pipeline)
    X_pca = pca.fit_transform(X)  # Ajusta y transforma: obtiene proyección en el subespacio (n,2)
    X_reconstructed = pca.inverse_transform(X_pca)  # Reconstruye desde PCA: útil para medir error de reconstrucción

    print(f"\nPCA Varianza explicada: {pca.explained_variance_ratio_}")  # Muestra proporción de varianza por componente: sanity check
    print(f"Error reconstrucción: {np.mean((X - X_reconstructed)**2):.6f}")  # Mide MSE de reconstrucción: debe ser pequeño si k es suficiente

    print("\n✓ Todos los tests pasaron!")  # Mensaje final: indica ejecución completa del bloque de pruebas sin excepciones
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

import numpy as np  # Importa NumPy: se usa para type hints, posibles conversiones y operaciones auxiliares
from sklearn.cluster import KMeans as SklearnKMeans  # Importa KMeans de sklearn: baseline para validar resultados (no para aprender)
from sklearn.decomposition import PCA as SklearnPCA  # Importa PCA de sklearn: baseline para comparar varianza explicada


def shadow_mode_kmeans(X: np.ndarray, k: int = 3, seed: int = 42) -> None:  # Compara tu K-Means (from scratch) contra sklearn (referencia)
    """Compara inercia de tu K-Means vs sklearn."""
    # Tu implementación
    # my = KMeans(n_clusters=k, random_state=seed)
    # my_labels = my.fit_predict(X)
    # my_inertia = my.inertia_

    # Placeholder (reemplazar con tu código)
    my_inertia = 0.0  # Placeholder: aquí debe ir la inercia de TU implementación (se deja en 0 para que el ejemplo sea ejecutable)

    # sklearn
    sk = SklearnKMeans(n_clusters=k, init="k-means++", n_init=10, random_state=seed)  # Instancia sklearn KMeans: usa K-Means++ y reinicios para estabilidad
    sk.fit(X)  # Ajusta sklearn KMeans: aprende centroides y calcula inercia interna en sk.inertia_

    print("=" * 60)  # Separador visual: hace más legible la salida en consola
    print("SHADOW MODE: K-Means")  # Encabezado: indica que esta sección corresponde a la comparación de K-Means
    print("=" * 60)  # Repite separador: encuadra el bloque de resultados
    print(f"Tu inercia:      {my_inertia:.4f}")  # Reporta la inercia de tu implementación (placeholder hasta reemplazar)
    print(f"sklearn inertia: {sk.inertia_:.4f}")  # Reporta inercia de sklearn: referencia para detectar discrepancias grandes


def shadow_mode_pca(X: np.ndarray, n_components: int = 2) -> None:  # Compara PCA from scratch vs sklearn en varianza explicada
    """Compara varianza explicada de tu PCA vs sklearn."""
    # Tu implementación
    # my = PCA(n_components=n_components)
    # X_my = my.fit_transform(X)

    # sklearn
    sk = SklearnPCA(n_components=n_components)  # Instancia sklearn PCA: calcula componentes principales por SVD internamente
    sk.fit(X)  # Ajusta PCA: estima explained_variance_ratio_ para comparar con tu implementación

    print("=" * 60)  # Separador visual: delimita salida del bloque PCA
    print("SHADOW MODE: PCA")  # Encabezado: indica comparación de PCA
    print("=" * 60)  # Repite separador: mantiene consistencia con el bloque anterior
    print(f"sklearn explained_variance_ratio_: {sk.explained_variance_ratio_}")  # Varianza explicada de sklearn: baseline para tu PCA
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

# 📘 Extensión Académica: Nivel MS-AI (University of Colorado Boulder Pathway)

> Rigor matemático formal, contexto histórico y conexiones teóricas profundas.

---

## A.1 Contexto Histórico

- **1901:** Karl Pearson — PCA
- **1957:** Lloyd — K-means (publicado 1982)
- **1977:** Dempster, Laird, Rubin — Algoritmo EM
- **2008:** van der Maaten & Hinton — t-SNE

### El Paradigma No Supervisado

A diferencia del supervisado, **no hay etiquetas**. El objetivo es descubrir **estructura latente**.

---

## A.2 Analogía: El Arqueólogo

Un arqueólogo encuentra fragmentos de cerámica:

- **Sin etiquetas:** No sabe la civilización
- **Objetivo:** Agrupar por similitud
- **Descubrimiento:** Identifica culturas desconocidas

> **Aprendizaje no supervisado = encontrar patrones ocultos sin guía externa.**

---

## A.3 PCA: Derivación Formal

### Objetivo de Optimización

Maximizar varianza de la proyección:

$$\max_{\mathbf{w}} \mathbf{w}^T \Sigma \mathbf{w} \quad \text{s.t.} \quad \|\mathbf{w}\|_2 = 1$$

### Solución

$\mathbf{w}$ es el **eigenvector** de $\Sigma$ con mayor eigenvalue.

### Conexión con SVD

Para $X$ centrado: $X = U\Sigma V^T$

- Componentes principales = columnas de $V$
- Scores = $XV = U\Sigma$

---

## A.4 K-Means: Análisis Teórico

### Objetivo

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

### Propiedades

- **Convergencia garantizada** (a mínimo local)
- **Complejidad:** $O(nKdI)$
- **NP-hard** encontrar óptimo global

### K-Means++

Inicialización que garantiza:

$$\mathbb{E}[J] \leq 8(\ln k + 2) \cdot J_{\text{OPT}}$$

---

## A.5 GMM y Algoritmo EM

### Modelo

$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

### EM: Garantías

- **Monotonía:** $\mathcal{L}(\theta^{(t+1)}) \geq \mathcal{L}(\theta^{(t)})$
- **Convergencia:** A punto estacionario
- **Limitación:** Puede converger a máximo local

---

## A.6 t-SNE y UMAP

### t-SNE: Objetivo

Minimizar KL divergence entre distribuciones de similitud:

$$D_{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### UMAP: Ventajas

- Más rápido: $O(n^{1.14})$ vs $O(n^2)$
- Mejor preservación global
- Puede proyectar nuevos datos

---

## A.7 Conexiones con MS-AI Pathway

| Concepto | Curso | Aplicación |
|----------|-------|------------|
| PCA | DTSA 5510 | Reducción dimensional |
| K-Means | DTSA 5510 | Clustering |
| GMM | DTSA 5510 | Modelos generativos |
| t-SNE/UMAP | DTSA 5510 | Visualización |

---

## A.8 Referencias Académicas

1. **Bishop, C.M. (2006).** *PRML*, Chapters 9, 12.
2. **van der Maaten, L., & Hinton, G. (2008).** "Visualizing Data using t-SNE." JMLR.
3. **McInnes, L., et al. (2018).** "UMAP: Uniform Manifold Approximation." arXiv.

---

*Extensión académica del MS-AI Pathway de la University of Colorado Boulder.*

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [M05_Aprendizaje_Supervisado](../../M05_Aprendizaje_Supervisado/) | [README](../../README.md) | [M07_Deep_Learning](../../M07_Deep_Learning/) |
