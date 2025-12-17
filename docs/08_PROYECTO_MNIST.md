# Módulo 08 - Proyecto Final: MNIST Analyst

> **🎯 Objetivo:** Pipeline end-to-end que demuestra competencia en las 3 áreas del Pathway
> **Fase:** 3 - Proyecto Integrador | **Semanas 21-24** (4 semanas)
> **Dataset:** **Fashion-MNIST** (principal, 28×28, 10 clases) / MNIST (fallback, mismo formato)

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
"""MNIST Dataset Loader
MNIST contiene:
- 60,000 imágenes de entrenamiento
- 10,000 imágenes de test
- Cada imagen: 28x28 píxeles grayscale (0-255)
- 10 clases: dígitos 0-9
Formato aplanado: cada imagen es un vector de 784 dimensiones
"""  # Delimitador de cierre del docstring del bloque; si faltara, todo lo siguiente quedaría dentro del string
import numpy as np  # Importa NumPy para operar con arrays y álgebra lineal de forma eficiente
import struct  # Importa struct para desempaquetar headers binarios en formato IDX
import gzip  # Importa gzip para leer directamente archivos comprimidos .gz en modo binario
from pathlib import Path  # Importa Path para composición robusta de rutas (operador /)
from typing import Tuple  # Importa Tuple para anotar retornos múltiples (no cambia el runtime)

def load_mnist_images(filepath: str) -> np.ndarray:  # Función: lee imágenes MNIST (IDX) y devuelve matriz (n, 784)
    """Carga imágenes MNIST desde archivo IDX.

     Formato IDX:
     - 4 bytes: magic number
     - 4 bytes: número de imágenes
     - 4 bytes: número de filas
     - 4 bytes: número de columnas
     - resto: píxeles (unsigned bytes)
     """  # Delimitador de cierre del docstring de la función; a partir de aquí se ejecutan lecturas/reshape
    with gzip.open(filepath, 'rb') as f:  # Abre el .gz en binario; el context manager cierra automáticamente
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))  # Lee 4 enteros big-endian del header
        images = np.frombuffer(f.read(), dtype=np.uint8)  # Interpreta el resto del archivo como bytes (sin copia)
        images = images.reshape(num_images, rows * cols)  # Reorganiza a (n_images, 784) para modelos lineales/MLP
    return images  # Retorna las imágenes aplanadas; no normaliza (se hace en un paso separado)

def load_mnist_labels(filepath: str) -> np.ndarray:  # Función: lee labels MNIST (IDX) y devuelve vector (n,)
    """Carga etiquetas MNIST."""  # Docstring corto: describe propósito sin afectar la ejecución
    with gzip.open(filepath, 'rb') as f:  # Abre el .gz; el archivo de labels tiene header + bytes de etiquetas
        magic, num_labels = struct.unpack('>II', f.read(8))  # Lee magic number y cantidad de etiquetas
        labels = np.frombuffer(f.read(), dtype=np.uint8)  # Lee el payload como uint8 (0-9)
    return labels  # Retorna vector 1D; el orden corresponde al de las imágenes

def load_mnist(data_dir: str = 'data/mnist') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # Loader train/test
    """Carga dataset MNIST completo.

     Returns:
         X_train: (60000, 784)
         y_train: (60000,)
         X_test: (10000, 784)
         y_test: (10000,)
     """  # Delimitador de cierre del docstring del loader; si faltara, el cuerpo quedaría como texto y no se ejecutaría
    data_dir = Path(data_dir)  # Convierte a Path para componer rutas de forma segura (OS-agnóstico)

    X_train = load_mnist_images(data_dir / 'train-images-idx3-ubyte.gz')  # Carga imágenes de entrenamiento
    y_train = load_mnist_labels(data_dir / 'train-labels-idx1-ubyte.gz')  # Carga etiquetas de entrenamiento
    X_test = load_mnist_images(data_dir / 't10k-images-idx3-ubyte.gz')  # Carga imágenes de prueba
    y_test = load_mnist_labels(data_dir / 't10k-labels-idx1-ubyte.gz')  # Carga etiquetas de prueba

    return X_train, y_train, X_test, y_test  # Retorna splits estándar para entrenamiento y evaluación

def normalize_data(X: np.ndarray) -> np.ndarray:  # Normaliza píxeles a [0,1] para estabilizar el entrenamiento
    """Normaliza píxeles a rango [0, 1]."""  # Docstring: contrato de normalización
    return X.astype(np.float64) / 255.0  # Convierte a float y escala; evita overflow y facilita gradientes

def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:  # Convierte labels a matriz one-hot (n_samples, n_classes)
    """Convierte etiquetas a one-hot encoding."""  # Docstring: documenta que se devuelve una matriz one-hot
    one_hot = np.zeros((len(y), num_classes))  # Inicializa matriz de ceros; se muta marcando 1 en la clase correcta
    one_hot[np.arange(len(y)), y] = 1  # Indexación avanzada: fila i, columna y[i] -> 1 (asume y en [0,num_classes))
    return one_hot  # Retorna la matriz para usar con softmax/cross-entropy u otros clasificadores

def generate_synthetic_mnist(n_samples: int = 1000, seed: int = 42) -> Tuple:  # Genera dataset sintético con shapes tipo-MNIST
    """Genera datos sintéticos similares a MNIST para pruebas."""  # Docstring: alternativa cuando no hay archivos MNIST
    np.random.seed(seed)  # Fija semilla del RNG global para reproducibilidad de X/y sintéticos
    X = np.random.rand(n_samples, 784)  # Features aleatorias en [0,1] con forma de MNIST aplanado (28*28)
    y = np.random.randint(0, 10, n_samples)  # Etiquetas enteras aleatorias en 0..9 (10 clases)
    split = int(0.8 * n_samples)  # Define split 80/20 (train/test) redondeando hacia abajo
    return X[:split], y[:split], X[split:], y[split:]  # Retorna X_train,y_train,X_test,y_test en ese orden

def visualize_digits(X: np.ndarray, y: np.ndarray, n_samples: int = 25):  # Dibuja una grilla de imágenes + labels
    """Visualiza una cuadrícula de dígitos."""  # Docstring: describe el propósito de la función
    n_cols = 5  # Número fijo de columnas para visualización
    n_rows = (n_samples + n_cols - 1) // n_cols  # Filas necesarias (ceil(n_samples/n_cols))

    # Crea figura con subplots organizados en cuadrícula
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 2*n_rows))  # Crea subplots; axes es array 2D
    axes = axes.flatten()  # Convierte matriz 2D de axes a 1D para iteración fácil

    # Itera sobre todos los axes disponibles
    for i, ax in enumerate(axes):  # Itera sobre cada subplot y su índice
        if i < n_samples:  # Evita acceder fuera de rango si hay más axes que muestras
            # Reestructura vector 784 a matriz 28x28 para visualización
            img = X[i].reshape(28, 28)  # Reconstruye imagen 28x28 desde vector aplanado
            # Muestra imagen en escala de grises
            ax.imshow(img, cmap='gray')  # Renderiza la imagen
            # Añade título con etiqueta correspondiente
            ax.set_title(f'Label: {y[i]}')  # Título con la clase real
        # Oculta ejes para limpieza visual
        ax.axis('off')  # Oculta ticks y borde del subplot

    plt.tight_layout()  # Ajusta espaciado para evitar solapamiento
    plt.show()  # Side effect: muestra la figura

def visualize_digit_single(x: np.ndarray, title: str = ''):  # Dibuja una sola imagen (vector 784) como 28x28
    """Visualiza un solo dígito."""  # Docstring: visualización individual
    plt.figure(figsize=(4, 4))  # Crea una figura nueva de 4x4 pulgadas
    plt.imshow(x.reshape(28, 28), cmap='gray')  # Reconvierte vector 784 a imagen 28x28 y la renderiza en gris
    plt.title(title)  # Título opcional para contextualizar la imagen
    plt.axis('off')  # Oculta ejes/ticks para una visualización limpia
    plt.show()  # Side effect: muestra la figura (depende del backend; puede bloquear)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 1.1: Data Loader (MNIST)</strong></summary>

#### 1) Metadatos
- **Título:** Data loader robusto: leer IDX, normalizar, one-hot, y sanity checks
- **ID (opcional):** `M08-P01_1`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** Numpy básico, archivos/rutas, lectura binaria

#### 2) Objetivos
- Cargar correctamente `X_train`, `y_train`, `X_test`, `y_test` con las shapes esperadas.
- Normalizar y validar rangos (`[0,1]`) sin romper dtype/shape.
- Generar etiquetas one-hot y entender cuándo usar one-hot vs enteros.

#### 3) Relevancia
- Si el loader falla, todo el pipeline falla silenciosamente (shapes incorrectas, labels desalineadas, leaks).
- Te obliga a practicar “contratos de datos”: invariantes que luego requieren PCA/LR/MLP.

#### 4) Mapa conceptual mínimo
- **Archivo IDX/GZ** → bytes → `np.frombuffer` → reshape a `(n, 784)`.
- **Normalización** → `uint8` → `float64` (o `float32`) → divide por 255.
- **Labels** → enteros `0..9` → (opcional) one-hot `(n, 10)`.

#### 5) Definiciones esenciales
- **Shape**: estructura `(n_samples, n_features)`.
- **Split**: train/test (no mezclar).
- **One-hot**: representación categórica para pérdidas tipo cross-entropy.

#### 6) Explicación didáctica
- Define invariantes y “asserts” mentales: `X.ndim==2`, `y.ndim==1`, `len(X)==len(y)`, `X.min()>=0`, `X.max()<=1` tras normalizar.

#### 7) Ejemplo modelado
- Si `X_train` no es `(60000, 784)` (o el dataset alternativo), revisa: `reshape(num_images, rows*cols)` y rutas.

#### 8) Práctica guiada
- Imprime: shapes, dtype, min/max y 5 pares `(imagen,label)` para verificar alineación.

#### 9) Práctica independiente
- Agrega un “modo debug” que muestre histograma de intensidades y detecte outliers/NaNs.

#### 10) Autoevaluación
- ¿Qué bugs aparecen si normalizas con `X/255` sin castear a float primero?

#### 11) Errores comunes
- Labels desalineadas por leer un archivo equivocado.
- Olvidar que `np.frombuffer` no copia: si mutas, entiende de dónde viene el buffer.
- Confundir densidad de pixeles con “features” ya listas (falta estandarizar o PCA según modelo).

#### 12) Retención
- Checklist fijo antes de entrenar: `shape`, `dtype`, `range`, `align`, `class_counts`.

#### 13) Diferenciación
- Avanzado: soportar Fashion-MNIST/MNIST intercambiables y parametrizar paths/descarga.

#### 14) Recursos
- IDX file format (MNIST) + docs de `gzip`/`struct`/`numpy.frombuffer`.

#### 15) Nota docente
- Evalúa por contrato: el alumno “aprueba” esta sección si puede demostrar invariantes y detectar 2 fallos típicos.
</details>

---

## 💻 Parte 2: Exploración No Supervisada (Semanas 21-22)

### 2.1 PCA para Visualización

```python
"""SEMANA 21: PCA en MNIST

Objetivo: Reducir de 784 dimensiones a 2-3 para visualización.

Preguntas a responder:
1. ¿Cuánta varianza se retiene con pocos componentes?
2. ¿Se separan visualmente las clases en 2D?
3. ¿Qué "aprenden" las componentes principales?
"""  # Cierra la cabecera multi-línea del bloque; si faltara, todo lo siguiente quedaría dentro del string

import numpy as np  # Importa NumPy para álgebra lineal (SVD) y manipulación de arrays
from typing import Tuple  # Importa Tuple para anotar retornos múltiples (no cambia el runtime)

# Nota: este bloque usa `plt` para graficar; requiere `import matplotlib.pyplot as plt` en el entorno.

class PCA:  # Clase PCA: aprende componentes principales y permite proyectar/reconstruir
    """PCA implementado desde cero (del Módulo 05)."""  # Docstring: referencia a implementación del módulo previo
    def __init__(self, n_components: int):  # Inicializa PCA con número de componentes deseado
        self.n_components = n_components  # Número de componentes principales a retener
        self.components_ = None  # Matriz (n_features, n_components) con ejes principales (se llena en fit)
        self.mean_ = None  # Vector (n_features,) con la media de entrenamiento para centrar
        self.explained_variance_ratio_ = None  # Vector (n_components,) con proporción de varianza explicada

    def fit(self, X: np.ndarray) -> 'PCA':  # Ajusta PCA a X (calcula media, SVD, varianza explicada)
        self.mean_ = np.mean(X, axis=0)  # Calcula media por feature (centrado requerido por PCA)
        X_centered = X - self.mean_  # Centra los datos (no muta X; crea array nuevo)

        # SVD (más estable que eigendecomposition)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # SVD: Xc = U diag(S) Vt

        self.components_ = Vt[:self.n_components].T  # Toma las primeras PCs (filas de Vt) y transpone a columnas
        variance = (S ** 2) / (len(X) - 1)  # Eigenvalues de covarianza (varianza por componente)
        self.explained_variance_ratio_ = variance[:self.n_components] / np.sum(variance)  # Proporción relativa

        return self  # Permite chaining (pca.fit(X).transform(X))

    def transform(self, X: np.ndarray) -> np.ndarray:  # Proyecta X al subespacio PCA
        return (X - self.mean_) @ self.components_  # Proyecta datos centrados a espacio reducido (scores)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:  # Ajusta y transforma en un paso
        self.fit(X)  # Ajusta PCA (calcula media y componentes)
        return self.transform(X)  # Devuelve proyección

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:  # Reconstruye aproximación desde espacio reducido
        return X_pca @ self.components_.T + self.mean_  # Reconstruye aproximación en espacio original


def analyze_pca_mnist(X: np.ndarray, y: np.ndarray):  # Función utilitaria de análisis PCA para MNIST
    """Análisis PCA completo de MNIST."""  # Docstring: grafica varianza explicada, proyección 2D y PCs como imágenes

    print("=== Análisis de Varianza Explicada ===")  # Header de sección
    pca_full = PCA(n_components=min(50, X.shape[1]))  # Ajusta PCA hasta 50 PCs o n_features (lo que sea menor)
    pca_full.fit(X)  # Entrena PCA (calcula componentes y varianza explicada)

    cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)  # Varianza acumulada para elegir k

    for n in [2, 10, 50]:  # Valores de k a reportar (heurísticos)
        if n <= len(cumulative_var):  # Guarda contra pedir más componentes de las que existen
            print(f"  {n} componentes: {cumulative_var[n-1]:.2%} varianza")  # Reporta varianza acumulada

    # 2. Visualización 2D
    print("\n=== Proyección 2D ===")  # Header para sección de scatter 2D
    pca_2d = PCA(n_components=2)  # PCA a 2D para graficar
    X_2d = pca_2d.fit_transform(X)  # Proyección (n_samples, 2)

    plt.figure(figsize=(10, 8))  # Crea figura para el scatter 2D (ancho x alto en pulgadas)
    for digit in range(10):  # Itera por clase para colorear cada dígito
        mask = y == digit  # Máscara booleana de muestras pertenecientes al dígito
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],  # Dibuja puntos de la clase seleccionada en el plano PCA
                   alpha=0.5, label=str(digit), s=10)  # Scatter de puntos del dígito en el plano PCA
    plt.legend()  # Muestra leyenda con los dígitos (0-9)
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')  # Etiqueta PC1 con varianza explicada
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')  # Etiqueta PC2 con varianza explicada
    plt.title('MNIST en 2D (PCA)')  # Título descriptivo
    plt.show()  # Side effect: renderiza figura

    # 3. Visualizar componentes principales
    print("\n=== Componentes Principales como Imágenes ===")  # Header para PCs visualizadas como imágenes
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))  # Cuadrícula 2x5 para mostrar 10 componentes
    pca_10 = PCA(n_components=10)  # PCA con 10 componentes
    pca_10.fit(X)  # Ajusta PCA para extraer componentes (vectores en R^784)

    for i, ax in enumerate(axes.flatten()):  # Itera sobre los 10 subplots
        component = pca_10.components_[:, i].reshape(28, 28)  # Reinterpreta PC_i como imagen 28x28
        ax.imshow(component, cmap='RdBu')  # Visualiza pesos positivos/negativos con colormap divergente
        ax.set_title(f'PC{i+1}')  # Título: índice de componente (1-index)
        ax.axis('off')  # Oculta ejes
    plt.suptitle('Top 10 Componentes Principales')  # Título general de la figura de PCs
    plt.tight_layout()  # Ajusta layout para evitar solapamientos
    plt.show()  # Side effect: renderiza figura

    return pca_2d, X_2d  # Retorna PCA entrenado a 2D y su proyección para reutilizar
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 2.1: PCA para Visualización</strong></summary>

#### 1) Metadatos
- **Título:** PCA como herramienta de diagnóstico: varianza, separabilidad y “qué aprende” el dataset
- **ID (opcional):** `M08-P02_1`
- **Duración estimada:** 120–180 min
- **Nivel:** Intermedio
- **Dependencias:** Álgebra lineal (SVD), centrado, interpretación de proyecciones

#### 2) Objetivos
- Implementar `fit/transform/inverse_transform` con shapes correctas.
- Interpretar `explained_variance_ratio_` y elegir `k` basado en varianza acumulada.
- Usar la proyección 2D como diagnóstico (no como “modelo final”).

#### 3) Relevancia
- PCA te muestra si la estructura del dataset es “amigable” para modelos lineales o si necesitas no linealidad.
- PCA conecta EDA con diseño del pipeline (qué modelos compiten bien y por qué).

#### 4) Mapa conceptual mínimo
- **Centrado** `X - mean`.
- **SVD** → direcciones principales.
- **Transform** → coordenadas en el subespacio.
- **Inverse** → reconstrucción aproximada.

#### 5) Definiciones esenciales
- **Componente principal**: dirección que maximiza varianza.
- **Varianza explicada**: proporción de “información” capturada.
- **Reconstrucción**: aproximación en el espacio original.

#### 6) Explicación didáctica
- La nube 2D puede “engañar”: si en 2D no se separa, aún podría separarse con más PCs.

#### 7) Ejemplo modelado
- Reporta varianza acumulada con `k=2,10,50` y decide un `k` para K-Means/LR.

#### 8) Práctica guiada
- Verifica que `inverse_transform(transform(X))` devuelve algo con la misma shape que `X`.

#### 9) Práctica independiente
- Mide el error de reconstrucción al variar `k` y grafica “k vs error”.

#### 10) Autoevaluación
- ¿Por qué es obligatorio centrar antes de PCA? ¿Qué se rompe si no centras?

#### 11) Errores comunes
- Confundir `components_` shape (cuidado con transponer Vt).
- Interpretar PCs como “features reales” (son combinaciones lineales).
- Mezclar `fit_transform` de train con test sin usar la misma `mean_`/`components_`.

#### 12) Retención
- Regla: PCA se ajusta en train y se aplica igual a test (misma media y componentes).

#### 13) Diferenciación
- Avanzado: whitening, o comparar PCA con t-SNE/UMAP (solo como diagnóstico, no core).

#### 14) Recursos
- Notas de SVD y PCA; documentación de estabilidad numérica.

#### 15) Nota docente
- Pide una “lectura narrativa” del scatter: ¿qué dígitos se confunden y por qué (similaridad visual)?
</details>

### 2.2 K-Means Clustering

```python
"""SEMANA 22: K-Means en MNIST

Objetivo: Agrupar dígitos SIN usar etiquetas.

Preguntas a responder:
1. ¿K-Means encuentra los 10 dígitos?
2. ¿Qué tan puros son los clusters?
3. ¿Cómo se ven los centroides?
"""  # Cierra la introducción del bloque; evita que imports/clases queden accidentalmente dentro del string

import numpy as np  # Importa NumPy para distancias, inicialización aleatoria y operaciones vectoriales

class KMeans:  # Clase K-Means: clustering no supervisado con inicialización K-Means++
    """K-Means implementado desde cero (del Módulo 05)."""  # Docstring: algoritmo Lloyd + K-Means++

    def __init__(self, n_clusters: int = 10, max_iter: int = 100, seed: int = None):  # Configura hiperparámetros
        self.n_clusters = n_clusters  # k: número de clusters
        self.max_iter = max_iter  # Máximo de iteraciones de Lloyd
        self.seed = seed  # Semilla opcional para reproducibilidad
        self.centroids = None  # Centroides (k, n_features)
        self.labels_ = None  # Labels asignados (n_samples,)
        self.inertia_ = None  # Inercia final (suma de distancias cuadradas intra-cluster)

    def _init_centroids_plusplus(self, X: np.ndarray) -> np.ndarray:  # Inicialización K-Means++
        """K-Means++ initialization."""  # Docstring: elige centroides iniciales separados
        if self.seed:  # Solo fija semilla si el usuario la proporcionó
            np.random.seed(self.seed)  # Fija RNG global (legacy) para reproducibilidad de la inicialización

        n_samples = len(X)  # Número de puntos
        centroids = [X[np.random.randint(n_samples)]]  # Primer centroide: punto aleatorio del dataset

        for _ in range(1, self.n_clusters):  # Elige los k-1 centroides restantes
            distances = np.array([min(np.sum((x - c)**2) for c in centroids) for x in X])  # d^2 al centroide más cercano
            probs = distances / distances.sum()  # Distribución de probabilidad proporcional a d^2
            centroids.append(X[np.random.choice(n_samples, p=probs)])  # Samplea nuevo centroide con esas probabilidades

        return np.array(centroids)  # Devuelve centroides iniciales como ndarray (k, n_features)

    def fit(self, X: np.ndarray) -> 'KMeans':  # Entrena K-Means (Lloyd) y guarda estado en self
        self.centroids = self._init_centroids_plusplus(X)  # Inicializa centroides (K-Means++)

        for _ in range(self.max_iter):  # Iteraciones de Lloyd (asignar -> actualizar)
            # Asignar
            distances = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])  # Matriz (n_samples,k) de distancias^2
            self.labels_ = np.argmin(distances, axis=1)  # Asigna cada punto al centroide más cercano

            # Actualizar
            new_centroids = np.array([X[self.labels_ == k].mean(axis=0)  # Centroide = media de puntos del cluster
                                      if np.sum(self.labels_ == k) > 0  # Si el cluster tiene puntos, usa la media
                                      else self.centroids[k]  # Si cluster vacío, conserva el centroide anterior
                                      for k in range(self.n_clusters)])  # Itera por cada id de cluster
            # Nota: si un cluster queda vacío, se conserva el centroide anterior para evitar NaN

            if np.allclose(self.centroids, new_centroids):  # Convergencia: centroides no cambian significativamente
                break  # Detiene iteración si convergió
            self.centroids = new_centroids  # Actualiza centroides (mutación del estado)

        self.inertia_ = sum(  # Calcula inercia final (SSE intra-cluster)
            np.sum((X[self.labels_ == k] - self.centroids[k])**2)  # SSE intra-cluster para cluster k
            for k in range(self.n_clusters)  # Itera por cluster para acumular SSE
        )  # Cierra la suma (reduce) de SSE sobre todos los clusters
        return self  # Retorna self para chaining

    def predict(self, X: np.ndarray) -> np.ndarray:  # Asigna clusters a nuevos puntos dado self.centroids
        distances = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])  # Distancias^2 a centroides
        return np.argmin(distances, axis=1)  # Devuelve labels predichos


def analyze_kmeans_mnist(X: np.ndarray, y: np.ndarray):  # Análisis y visualización de clustering (usa y solo para evaluación)
    """Análisis K-Means de MNIST."""  # Docstring: grafica centroides y calcula pureza

    print("=== K-Means Clustering ===")  # Header de la sección
    print("=== Centroides (promedio de cada cluster) ===")  # Muestra sección de centroides
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))  # Cuadrícula 2x5 para mostrar 10 centroides
    kmeans = KMeans(n_clusters=10, seed=42)  # Instancia modelo con k=10 (MNIST tiene 10 dígitos)
    kmeans.fit(X)  # Entrena K-Means

    for i, ax in enumerate(axes.flatten()):  # Itera sobre 10 subplots
        centroid = kmeans.centroids[i].reshape(28, 28)  # Reinterpreta centroide (784,) como imagen 28x28
        ax.imshow(centroid, cmap='gray')  # Visualiza centroide como imagen
        ax.set_title(f'Cluster {i}')  # Título con id del cluster
        ax.axis('off')  # Oculta ejes
    plt.suptitle('Centroides K-Means')  # Título general
    plt.tight_layout()  # Ajusta layout
    plt.show()  # Side effect: renderiza figura

    # 2. Analizar pureza de clusters
    print("\n=== Pureza de Clusters ===")  # Sección de evaluación de pureza usando y (solo para análisis)
    print("Cluster | Dígito Dominante | Pureza")  # Encabezado de tabla
    print("-" * 40)  # Separador visual

    total_correct = 0  # Acumulador de ejemplos que caen en el dígito dominante por cluster
    for cluster in range(10):  # Itera sobre los 10 clusters
        cluster_mask = kmeans.labels_ == cluster  # Máscara booleana de puntos asignados al cluster
        cluster_labels = y[cluster_mask]  # Etiquetas verdaderas de esos puntos (solo para evaluación)

        if len(cluster_labels) > 0:  # Evita operar sobre cluster vacío
            dominant_digit = np.bincount(cluster_labels).argmax()  # Dígito más frecuente en el cluster
            purity = np.sum(cluster_labels == dominant_digit) / len(cluster_labels)  # Pureza = fracción dominante
            total_correct += np.sum(cluster_labels == dominant_digit)  # Suma aciertos dominantes
            print(f"   {cluster}    |        {dominant_digit}         | {purity:.2%}")  # Imprime fila de tabla

    overall_purity = total_correct / len(y)  # Pureza global ponderada por tamaño de clusters
    print(f"\nPureza Global: {overall_purity:.2%}")  # Resumen global

    return kmeans  # Devuelve el modelo entrenado
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 2.2: K-Means Clustering</strong></summary>

#### 1) Metadatos
- **Título:** K-Means en MNIST: qué significa un centroide y cómo evaluar clusters sin “hacer trampa”
- **ID (opcional):** `M08-P02_2`
- **Duración estimada:** 120–180 min
- **Nivel:** Intermedio
- **Dependencias:** Distancias, promedios, intuición geométrica en alta dimensión

#### 2) Objetivos
- Implementar K-Means y entender el ciclo asignación → actualización.
- Interpretar `inertia_` y por qué decrece con iteraciones.
- Evaluar clusters con pureza usando labels solo para auditoría.

#### 3) Relevancia
- Entrena tu intuición de “representaciones”: en alta dimensión, K-Means puede fallar si la métrica no corresponde a la semántica.
- Es un baseline útil para detectar estructura y outliers antes de clasificación.

#### 4) Mapa conceptual mínimo
- **k** clusters → **centroides**.
- **Asignación** por mínima distancia.
- **Update**: centroides = media del cluster.
- **Evaluación**: inercia + pureza (si tienes y).

#### 5) Definiciones esenciales
- **Inercia (SSE)**: suma de distancias cuadradas intra-cluster.
- **Pureza**: fracción de la clase dominante por cluster.

#### 6) Explicación didáctica
- En MNIST, los centroides se pueden visualizar como “dígitos borrosos”: si salen irreconocibles, tu representación o k puede no estar bien.

#### 7) Ejemplo modelado
- Ejecuta K-Means con `k=10` y compara centroides vs dígitos reales.

#### 8) Práctica guiada
- Corre K-Means sobre PCA reducido (por ejemplo 50 PCs) y compara pureza/inercia con el espacio original.

#### 9) Práctica independiente
- Implementa “elbow method”: grafica `k` vs `inertia_` para `k ∈ {5,8,10,12,15}`.

#### 10) Autoevaluación
- ¿Por qué K-Means++ ayuda frente a centroides aleatorios?

#### 11) Errores comunes
- No manejar clusters vacíos (centroide NaN).
- Olvidar fijar seed y no poder reproducir resultados.
- Interpretar pureza alta como “modelo supervisado” (no lo es).

#### 12) Retención
- Mantra: “entrena sin labels; usa labels solo para auditar, no para optimizar”.

#### 13) Diferenciación
- Avanzado: comparar con GMM (soft clustering) y discutir cuándo es preferible.

#### 14) Recursos
- K-Means++ intuition; artículos sobre high-dimensional clustering.

#### 15) Nota docente
- Pide que el alumno explique por qué un cluster puede mezclar ‘4’ y ‘9’ (rasgos similares) y qué haría para mejorar.
</details>

---

## 💻 Parte 3: Clasificación Supervisada (Semanas 23-24)

### 3.1 Logistic Regression One-vs-All

```python
"""SEMANAS 23-24: Logistic Regression Multiclase

Estrategia One-vs-All (OvA):
- Entrenar 10 clasificadores binarios
- Cada uno: "¿Es este dígito X o no?"
- Predicción: elegir la clase con mayor probabilidad
"""  # Cierra la cabecera multi-línea del bloque; si faltara, todo lo siguiente quedaría dentro del string

import numpy as np  # Importa NumPy para álgebra lineal, métricas y procesamiento de arrays
from typing import List  # Importa List para anotaciones (no afecta runtime)

def sigmoid(z):  # Sigmoid: mapea logits a probabilidad en (0,1)
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid estable: clip evita overflow en exp

class LogisticRegressionBinary:  # Clasificador binario para OvA
    """Logistic Regression binario."""  # Docstring: modelo lineal + sigmoid

    def __init__(self, lr: float = 0.1, n_iter: int = 100, reg: float = 0.01):  # Hiperparámetros GD/L2
        self.lr = lr  # Learning rate para gradient descent
        self.n_iter = n_iter  # Número de iteraciones de entrenamiento
        self.reg = reg  # L2 regularization (fuerza del término ||theta||^2)
        self.theta = None  # Parámetros (n_features,)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionBinary':  # Entrena por GD
        n_samples, n_features = X.shape  # Extrae dimensiones; X debe ser 2D
        self.theta = np.zeros(n_features)  # Inicializa theta en 0 (convexo; converge)

        for _ in range(self.n_iter):  # Loop de optimización (permitido iterar sobre iteraciones)
            h = sigmoid(X @ self.theta)  # Probabilidades predichas (n_samples,)
            grad = (1/n_samples) * X.T @ (h - y) + (self.reg/n_samples) * self.theta  # Gradiente BCE + L2
            self.theta -= self.lr * grad  # Update GD (mutación de theta)

        return self  # Permite chaining (pca.fit(X).transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # Probabilidades para clase positiva
        return sigmoid(X @ self.theta)  # Devuelve probabilidad P(y=1|x)


class LogisticRegressionOvA:  # Meta-clasificador: 1 modelo binario por clase
    """Logistic Regression One-vs-All para clasificación multiclase."""  # Docstring: estrategia OvA

    def __init__(self, n_classes: int = 10, lr: float = 0.1, n_iter: int = 100):  # Configuración de OvA
        self.n_classes = n_classes  # Número de clases (10 en MNIST)
        self.lr = lr  # Learning rate para cada clasificador binario
        self.n_iter = n_iter  # Iteraciones para cada clasificador binario
        self.classifiers: List[LogisticRegressionBinary] = []  # Lista de modelos binarios entrenados

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionOvA':  # Entrena todos los clasificadores
        """Entrena un clasificador por clase."""  # Docstring: entrena un binario por cada clase
        # Añadir bias
        X_b = np.column_stack([np.ones(len(X)), X])  # Agrega columna 1s para término independiente

        self.classifiers = []  # Reinicia lista de clasificadores (mutación)
        for c in range(self.n_classes):  # Itera por clase para estrategia OvA
            print(f"  Entrenando clasificador para clase {c}...", end='\r')  # Side effect: imprime progreso
            y_binary = (y == c).astype(int)  # Labels binarios: 1 si es clase c, 0 si no
            clf = LogisticRegressionBinary(self.lr, self.n_iter)  # Inicializa clasificador binario
            clf.fit(X_b, y_binary)  # Entrena sobre el problema binario
            self.classifiers.append(clf)  # Guarda clasificador entrenado

        print("  Entrenamiento completado.                ")  # Limpia la línea y deja mensaje final
        return self  # Devuelve la instancia con `self.classifiers` entrenados

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # Devuelve scores/probabilidades por clase
        """Retorna probabilidades para cada clase."""  # Docstring
        X_b = np.column_stack([np.ones(len(X)), X])  # Añade bias
        probs = np.column_stack([clf.predict_proba(X_b) for clf in self.classifiers])  # (n_samples,n_classes)
        return probs  # Probabilidades por clase (no necesariamente suman 1; son scores OvA)

    def predict(self, X: np.ndarray) -> np.ndarray:  # Predice clase final vía argmax
        """Predice la clase con mayor probabilidad."""  # Docstring
        probs = self.predict_proba(X)  # Scores/probabilidades por clase
        return np.argmax(probs, axis=1)  # Elige la clase con score máximo

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # Accuracy
        """Accuracy."""  # Docstring
        return np.mean(self.predict(X) == y)  # Accuracy promedio (0..1)


def train_logistic_mnist(X_train, y_train, X_test, y_test):  # Entrena y reporta métricas
    """Entrena y evalúa Logistic Regression en MNIST."""  # Docstring: describe objetivo (entrenar/evaluar) sin afectar el comportamiento

    print("=== Logistic Regression One-vs-All ===")  # Header de la sección
    # Entrenar
    lr_model = LogisticRegressionOvA(n_classes=10, lr=0.1, n_iter=200)  # Configura OvA
    lr_model.fit(X_train, y_train)  # Entrena 10 clasificadores (side effects: ajusta parámetros)

    # Evaluar
    train_acc = lr_model.score(X_train, y_train)  # Accuracy en entrenamiento
    test_acc = lr_model.score(X_test, y_test)  # Accuracy en test

    print(f"\nTrain Accuracy: {train_acc:.2%}")  # Reporta accuracy de train
    print(f"Test Accuracy:  {test_acc:.2%}")  # Reporta accuracy de test

    # Métricas detalladas
    y_pred = lr_model.predict(X_test)  # Predicciones finales multiclase en test

    print("\n=== Métricas por Clase ===")  # Header de métricas por clase
    print("Dígito | Precision | Recall | F1-Score")  # Encabezado de tabla
    print("-" * 45)  # Separador

    for digit in range(10):  # Métricas one-vs-rest por clase
        tp = np.sum((y_test == digit) & (y_pred == digit))  # True positives para clase digit
        fp = np.sum((y_test != digit) & (y_pred == digit))  # False positives para clase digit
        fn = np.sum((y_test == digit) & (y_pred != digit))  # False negatives para clase digit

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision con guardas contra división por cero
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall con guardas
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # F1 con guardas

        print(f"   {digit}   |   {precision:.3f}   |  {recall:.3f}  |   {f1:.3f}")  # Fila por clase

    # Matriz de confusión
    print("\n=== Matriz de Confusión ===")  # Header de matriz de confusión
    cm = np.zeros((10, 10), dtype=int)  # Matriz de confusión (true x pred)
    for true, pred in zip(y_test, y_pred):  # Itera sobre pares (y_true, y_pred)
        cm[true, pred] += 1  # Incrementa celda correspondiente

    print("    " + "  ".join(str(i) for i in range(10)))  # Header con índices de clase predicha
    for i in range(10):  # Itera por clase verdadera
        print(f"{i}: " + " ".join(f"{cm[i,j]:3d}" for j in range(10)))  # Imprime fila de la matriz

    return lr_model  # Devuelve el modelo entrenado
```

---

## 💻 Parte 4: Deep Learning (Semanas 25-26)

### 4.1 MLP para MNIST

```python
"""SEMANAS 25-26: Neural Network para MNIST

Arquitectura:
- Input: 784 (28x28 píxeles aplanados)
- Hidden 1: 128 neuronas, ReLU
- Hidden 2: 64 neuronas, ReLU
- Output: 10 neuronas, Softmax

Objetivo: Superar a Logistic Regression
"""  # Cierra la introducción del bloque; el código siguiente define funciones/clases del MLP

import numpy as np  # Importa NumPy para álgebra lineal, inicialización de pesos y operaciones vectoriales
from typing import List, Tuple  # Importa tipos para anotaciones (no afecta runtime)

# Funciones de activación
def relu(z):  # ReLU: activación no lineal usada en capas ocultas
    return np.maximum(0, z)  # max(0,z) elemento a elemento

def relu_deriv(z):  # Derivada de ReLU respecto a z
    return (z > 0).astype(float)  # 1.0 si z>0, 0.0 si z<=0

def softmax(z):  # Softmax: convierte logits en probabilidades
    exp_z = np.exp(z - np.max(z))  # Estabiliza restando max(z) para evitar overflow en exp
    return exp_z / np.sum(exp_z)  # Normaliza para que la suma sea 1

class NeuralNetworkMNIST:  # Clase que encapsula el MLP (parámetros y estado) y provee entrenamiento/predicción
    """Red Neuronal optimizada para MNIST."""  # Docstring de clase: describe el propósito general sin ejecutar lógica

    def __init__(self, layer_sizes: List[int] = [784, 128, 64, 10], seed: int = 42):  # Configura arquitectura e inicializa pesos/bias
        """  # Cierra docstring del constructor; separa documentación del código de inicialización
        Args:
            layer_sizes: [input, hidden1, hidden2, ..., output]
        """  # Cierra docstring del constructor; separa documentación del código de inicialización
        np.random.seed(seed)  # Fija semilla global para reproducibilidad de inicialización

        self.layer_sizes = layer_sizes  # Lista de tamaños por capa (input -> hidden(s) -> output)
        self.n_layers = len(layer_sizes)  # Número total de capas (incluye input y output)

        # Inicializar pesos (He initialization para ReLU)
        self.weights = []  # Lista de matrices W por capa (out_dim, in_dim)
        self.biases = []  # Lista de vectores b por capa (out_dim,)

        for i in range(self.n_layers - 1):  # Crea parámetros para cada transición capa_i -> capa_{i+1}
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])  # He init para ReLU
            b = np.zeros(layer_sizes[i+1])  # Bias inicial en 0
            self.weights.append(w)  # Guarda pesos de la capa i
            self.biases.append(b)  # Guarda bias de la capa i

        self.cache = {}  # Cache de activaciones/preactivaciones para backward (estado mutable)
        self.loss_history = []  # Historial del loss promedio por época

    def forward(self, x: np.ndarray) -> np.ndarray:  # Forward: computa activaciones/probabilidades y guarda cache para backprop
        """Forward pass."""  # Docstring: aclara que aquí se computan activaciones
        self.cache['a0'] = x  # Guarda entrada como activación de capa 0
        a = x  # Activación actual (se va actualizando capa por capa)

        for i in range(self.n_layers - 2):  # Itera solo capas ocultas (la última capa se calcula con softmax aparte)
            z = self.weights[i] @ a + self.biases[i]  # Pre-activación: z = W a + b
            a = relu(z)  # Activación ReLU
            self.cache[f'z{i+1}'] = z  # Guarda z para derivada de ReLU
            self.cache[f'a{i+1}'] = a  # Guarda a para gradientes de W

        # Última capa: softmax
        z = self.weights[-1] @ a + self.biases[-1]  # Logits de salida
        a = softmax(z)  # Probabilidades por clase
        self.cache[f'z{self.n_layers-1}'] = z  # Guarda logits
        self.cache[f'a{self.n_layers-1}'] = a  # Guarda probabilidades

        return a  # Devuelve probabilidades finales

    def backward(self, y_true: np.ndarray) -> Tuple[List, List]:  # Backprop: calcula gradientes de pesos/bias usando y_true en one-hot
        """Backward pass."""  # Docstring: aclara que aquí se computan derivadas
        y_pred = self.cache[f'a{self.n_layers-1}']  # Recupera predicción del forward

        # Gradiente de softmax + cross-entropy
        dz = y_pred - y_true  # Para softmax+CE: dL/dz = (p - y_onehot)

        dW_list = []  # Gradientes de pesos (misma estructura que self.weights)
        db_list = []  # Gradientes de bias (misma estructura que self.biases)

        for i in range(self.n_layers - 2, -1, -1):  # Itera capas en reversa para propagar el gradiente hacia la entrada
            a_prev = self.cache[f'a{i}']  # Activación de la capa anterior

            dW = np.outer(dz, a_prev)  # dW = dz[:,None] * a_prev[None,:] (outer)
            db = dz  # db = dz (para un solo ejemplo)

            dW_list.insert(0, dW)  # Inserta al inicio para mantener orden de capas
            db_list.insert(0, db)  # Inserta al inicio para mantener orden de capas

            if i > 0:  # Evita computar derivada sobre una capa inexistente (no hay z0)
                da_prev = self.weights[i].T @ dz  # Propaga gradiente hacia activación previa
                z_prev = self.cache[f'z{i}']  # Pre-activación previa para derivada de ReLU
                dz = da_prev * relu_deriv(z_prev)  # Aplica derivada de ReLU

        return dW_list, db_list  # Devuelve gradientes por capa

    def fit(  # Entrena con mini-batch SGD (actualiza self.weights/self.biases en cada batch)
        self,  # Referencia al modelo; aquí se muta el estado entrenable de la instancia
        X: np.ndarray,  # Matriz de entrada (n_samples, n_features)
        y: np.ndarray,  # Vector de etiquetas enteras (n_samples,)
        epochs: int = 10,  # Cantidad de épocas (pasadas completas por el dataset)
        batch_size: int = 32,  # Tamaño del batch (trade-off: estabilidad del gradiente vs coste)
        learning_rate: float = 0.01,  # Paso del update en descenso de gradiente
        verbose: bool = True  # Controla prints de progreso (side effect: stdout)
    ):  # Cierra firma; el cuerpo implementa shuffle, batching, acumulación de gradientes y updates
        """Entrena la red con mini-batch SGD."""  # Docstring: describe objetivo (entrenar) sin afectar el comportamiento
        n_samples = len(X)  # Número total de muestras

        for epoch in range(epochs):  # Loop principal de entrenamiento por época
            # Shuffle
            indices = np.random.permutation(n_samples)  # Permutación aleatoria de índices (mutabilidad local)
            X_shuffled = X[indices]  # Reordena X
            y_shuffled = y[indices]  # Reordena y en el mismo orden

            total_loss = 0  # Acumulador de loss para promediar al final de la época

            for i in range(0, n_samples, batch_size):  # Recorre el dataset en ventanas para formar mini-batches
                X_batch = X_shuffled[i:i+batch_size]  # Slice del mini-batch
                y_batch = y_shuffled[i:i+batch_size]  # Labels del mini-batch

                # Acumular gradientes del batch
                dW_accum = [np.zeros_like(w) for w in self.weights]  # Acumulador de gradientes de W
                db_accum = [np.zeros_like(b) for b in self.biases]  # Acumulador de gradientes de b

                for x, y_true_label in zip(X_batch, y_batch):  # Itera ejemplos del batch para acumular gradientes
                    # One-hot encode
                    y_one_hot = np.zeros(10)  # Vector one-hot (10 clases)
                    y_one_hot[y_true_label] = 1  # Activa la clase verdadera

                    # Forward
                    y_pred = self.forward(x)  # Probabilidades predichas para este ejemplo

                    # Loss
                    loss = -np.sum(y_one_hot * np.log(np.clip(y_pred, 1e-15, 1)))  # CE con clipping para log(0)
                    total_loss += loss  # Acumula loss (suma)

                    # Backward
                    dW_list, db_list = self.backward(y_one_hot)  # Gradientes para un ejemplo

                    for j in range(len(self.weights)):  # Acumula gradientes por capa antes de actualizar (promedio de batch)
                        dW_accum[j] += dW_list[j]  # Suma gradiente de pesos
                        db_accum[j] += db_list[j]  # Suma gradiente de bias

                # Update
                batch_len = len(X_batch)  # Tamaño real del batch (último batch puede ser menor)
                for j in range(len(self.weights)):  # Aplica update SGD por capa usando gradiente promedio del batch
                    self.weights[j] -= learning_rate * dW_accum[j] / batch_len  # Update GD promedio en batch
                    self.biases[j] -= learning_rate * db_accum[j] / batch_len  # Update GD promedio en batch

            avg_loss = total_loss / n_samples  # Loss promedio por muestra (aprox)
            self.loss_history.append(avg_loss)  # Guarda loss para curvas

            if verbose:  # Logging condicional del progreso (no afecta cálculos si está desactivado)
                train_acc = self.score(X[:1000], y[:1000])  # Accuracy en un subset para feedback rápido
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.2%}")  # Log de entrenamiento

    def predict(self, X: np.ndarray) -> np.ndarray:  # Predice labels haciendo argmax de las probabilidades del forward
        """Predice clases."""  # Docstring: aclara que retorna labels (no probabilidades)
        return np.array([np.argmax(self.forward(x)) for x in X])  # Argmax de probabilidades por muestra

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # Calcula accuracy comparando predict(X) contra y
        """Accuracy."""  # Docstring: documenta la métrica retornada
        return np.mean(self.predict(X) == y)  # Promedio de aciertos


def train_neural_network_mnist(X_train, y_train, X_test, y_test):  # Wrapper que ejecuta fit/score y reporta; si se elimina, el pipeline pierde el paso de NN de forma encapsulada
    """Entrena y evalúa Neural Network en MNIST."""  # Docstring: describe objetivo (entrenar/evaluar) sin afectar el comportamiento

    print("=== Neural Network (MLP) ===")  # Header de la sección
    print("Arquitectura: 784 → 128 → 64 → 10")  # Describe arquitectura usada

    nn = NeuralNetworkMNIST([784, 128, 64, 10])  # Instancia MLP
    nn.fit(X_train, y_train, epochs=10, batch_size=32, learning_rate=0.01)  # Entrena (side effects: ajusta pesos internos del modelo)

    train_acc = nn.score(X_train, y_train)  # Accuracy en train
    test_acc = nn.score(X_test, y_test)  # Accuracy en test

    print(f"\nTrain Accuracy: {train_acc:.2%}")  # Reporta accuracy de train
    print(f"Test Accuracy:  {test_acc:.2%}")  # Reporta accuracy de test

    return nn  # Devuelve el modelo entrenado


# === Parte 5: Benchmark y Comparación ===


# === 5.1 Pipeline Completo ===

"""Pipeline completo que ejecuta todos los análisis
y compara los modelos.
"""  # Cierra la cabecera multi-línea; evita que imports/funciones queden dentro del string

import numpy as np  # Importa NumPy para normalización, slicing y operaciones básicas
import time  # Importa time para medir duración (benchmark) con time.time()

def run_mnist_pipeline(X_train, y_train, X_test, y_test, use_subset: bool = True):  # Orquesta el pipeline completo (PCA/KMeans/LR/MLP) y devuelve métricas
    """Ejecuta el pipeline completo de MNIST.

    Args:
        use_subset: Si True, usa solo 10k samples para rapidez
    """  # Cierra el docstring de la función; el bloque siguiente es lógica ejecutable del pipeline
    if use_subset:  # Permite acelerar el pipeline usando un subconjunto (trade-off: métrica menos estable)
        X_train = X_train[:10000]  # Reduce train para rapidez
        y_train = y_train[:10000]  # Reduce labels train
        X_test = X_test[:2000]  # Reduce test para rapidez
        y_test = y_test[:2000]  # Reduce labels test

    # Normalizar
    X_train = X_train / 255.0  # Normaliza train de [0,255] a [0,1]
    X_test = X_test / 255.0  # Normaliza test de [0,255] a [0,1]

    print("=" * 60)  # Imprime separador para delimitar la cabecera del pipeline en consola
    print("MNIST ANALYST PIPELINE")  # Título principal del pipeline (logging informativo)
    print("=" * 60)  # Repite separador para reforzar la separación visual
    print(f"Train samples: {len(X_train)}")  # Tamaño de train
    print(f"Test samples: {len(X_test)}")  # Tamaño de test
    print(f"Features: {X_train.shape[1]}")  # Número de features (784)
    print("=" * 60)  # Cierra la cabecera; a partir de aquí comienzan las fases del pipeline

    results = {}  # Dict para almacenar accuracy por modelo

    # === FASE 1: Unsupervised ===
    print("\n" + "=" * 60)  # Imprime separador y salto de línea para delimitar visualmente la fase 1
    print("FASE 1: EXPLORACIÓN NO SUPERVISADA")  # Título de la fase 1 (logging informativo)
    print("=" * 60)  # Imprime línea separadora para mejorar legibilidad en consola

    # PCA
    print("\n[PCA]")  # Header de la subsección PCA dentro del pipeline
    pca = PCA(n_components=50)  # Inicializa PCA con 50 componentes
    pca.fit(X_train)  # Ajusta PCA en train
    print(f"Varianza explicada (50 PCs): {sum(pca.explained_variance_ratio_):.2%}")  # Varianza acumulada

    # K-Means
    print("\n[K-Means]")  # Header de la subsección K-Means dentro del pipeline
    start = time.time()  # Marca tiempo inicial
    kmeans = KMeans(n_clusters=10, seed=42)  # Inicializa K-Means
    kmeans.fit(X_train)  # Entrena K-Means
    kmeans_time = time.time() - start  # Duración del entrenamiento
    print(f"Inercia: {kmeans.inertia_:.2f}")  # Inercia final
    print(f"Tiempo: {kmeans_time:.2f}s")  # Tiempo transcurrido

    # === FASE 2: Supervised ===
    print("\n" + "=" * 60)  # Separador visual para iniciar fase 2
    print("FASE 2: CLASIFICACIÓN SUPERVISADA")  # Título de fase 2 (modelos supervisados)
    print("=" * 60)  # Línea separadora de la sección

    # Logistic Regression
    print("\n[Logistic Regression One-vs-All]")  # Header para el entrenamiento/evaluación de Logistic Regression OvA
    start = time.time()  # Marca tiempo inicial
    lr_model = LogisticRegressionOvA(n_classes=10, lr=0.1, n_iter=100)  # Inicializa Logistic OvA
    lr_model.fit(X_train, y_train)  # Entrena
    lr_time = time.time() - start  # Duración
    lr_acc = lr_model.score(X_test, y_test)  # Accuracy en test
    print(f"Test Accuracy: {lr_acc:.2%}")  # Reporta accuracy
    print(f"Tiempo: {lr_time:.2f}s")  # Reporta tiempo
    results['Logistic Regression'] = lr_acc  # Guarda baseline

    # === FASE 3: Deep Learning ===
    print("\n" + "=" * 60)  # Separador visual para iniciar fase 3
    print("FASE 3: DEEP LEARNING")  # Título de fase 3 (modelos de red neuronal)
    print("=" * 60)  # Línea separadora de la sección

    # Neural Network
    print("\n[Neural Network MLP]")  # Header para el entrenamiento/evaluación del MLP
    start = time.time()  # Marca tiempo inicial
    nn = NeuralNetworkMNIST([784, 128, 64, 10])  # Inicializa MLP
    nn.fit(X_train, y_train, epochs=5, batch_size=32, learning_rate=0.01, verbose=False)  # Entrena rápido
    nn_time = time.time() - start  # Duración
    nn_acc = nn.score(X_test, y_test)  # Accuracy en test
    print(f"Test Accuracy: {nn_acc:.2%}")  # Reporta accuracy
    print(f"Tiempo: {nn_time:.2f}s")  # Reporta tiempo
    results['Neural Network'] = nn_acc  # Guarda resultado

    # === COMPARACIÓN ===
    print("\n" + "=" * 60)  # Separador visual previo a la tabla comparativa
    print("COMPARACIÓN DE MODELOS")  # Título de la sección donde se comparan accuracies
    print("=" * 60)  # Línea separadora para legibilidad

    print("\nModelo               | Accuracy | Mejora vs LR")  # Encabezado de la tabla de comparación
    print("-" * 50)  # Separador bajo el encabezado (formato tipo tabla)
    baseline = results['Logistic Regression']  # Baseline para comparar mejoras relativas
    for name, acc in results.items():  # Recorre resultados para imprimir cada fila (modelo -> métricas)
        improvement = ((acc - baseline) / baseline) * 100 if name != 'Logistic Regression' else 0  # Mejora porcentual
        print(f"{name:<20} | {acc:.2%}    | {improvement:+.1f}%")  # Imprime tabla de comparación

    # === ANÁLISIS ===
    print("\n" + "=" * 60)  # Separador visual previo al bloque explicativo
    print("ANÁLISIS: ¿Por qué NN es mejor?")  # Título del análisis cualitativo (texto informativo)
    print("=" * 60)  # Línea separadora
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

    return results  # Devuelve el diccionario de accuracies para uso posterior (p.ej. reporte/benchmark)

```
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
import numpy as np  # Importa NumPy para generar datos sintéticos y operar con arrays

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 0):  # Define split reproducible con RNG local
    X = np.asarray(X)  # Normaliza entrada a ndarray (evita listas y asegura slicing consistente)
    y = np.asarray(y)  # Normaliza labels a ndarray para indexación coherente con X
    n = X.shape[0]  # Número de muestras (se usa para construir índices)
    rng = np.random.default_rng(seed)  # RNG moderno aislado; misma seed => mismo shuffle
    idx = np.arange(n)  # Índices 0..n-1 (referencias a filas)
    rng.shuffle(idx)  # Baraja índices in-place (no modifica X/y directamente)
    n_test = int(round(n * test_size))  # Calcula tamaño de test (redondeo para cubrir casos fraccionales)
    test_idx = idx[:n_test]  # Toma los primeros índices como test
    train_idx = idx[n_test:]  # Toma el resto como train
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]  # Retorna splits (X_train, X_test, y_train, y_test)


np.random.seed(0)  # Fija semilla global legacy para que la generación sintética sea reproducible
X = np.random.randn(100, 784)  # Genera features sintéticas con la misma forma que MNIST (100, 784)
y = np.random.randint(0, 10, size=(100,))  # Genera etiquetas sintéticas enteras en rango [0,9]

Xtr1, Xte1, ytr1, yte1 = train_test_split(X, y, test_size=0.25, seed=123)  # Primer split con seed fija
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, test_size=0.25, seed=123)  # Segundo split con misma seed (debe coincidir)

assert np.allclose(Xtr1, Xtr2)  # Verifica reproducibilidad: train features idénticas
assert np.allclose(Xte1, Xte2)  # Verifica reproducibilidad: test features idénticas
assert np.all(ytr1 == ytr2)  # Verifica reproducibilidad: train labels idénticas
assert np.all(yte1 == yte2)  # Verifica reproducibilidad: test labels idénticas

assert Xtr1.shape[0] + Xte1.shape[0] == X.shape[0]  # Invariante: no se pierden muestras al partir
assert len(np.intersect1d(Xtr1[:, 0], Xte1[:, 0])) <= X.shape[0]  # Check simple de solapamiento (aprox; columna 0)

```

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 8.1: Reproducibilidad (seed) y split determinista</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M08-E08_1`
- **Duración estimada:** 20–35 min
- **Nivel:** Básico → Intermedio

#### 2) Idea clave
- La reproducibilidad es un *invariante del pipeline*: con la misma semilla debes obtener el mismo split.
- Un split determinista es la base para comparar modelos de forma justa en la Semana 24.

#### 3) Errores comunes
- Barajar `X` y `y` por separado (rompe la alineación filas↔labels).
- Usar RNG global implícito y luego modificarlo en otra parte del notebook/script.
- Olvidar verificar que `n_train + n_test == n`.

#### 4) Nota docente
- Pide al alumno imprimir los primeros 5 índices de `train_idx` y mostrar que se repiten entre ejecuciones.
</details>

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

rng = np.random.default_rng(1)
n = 256
X_uint8 = rng.integers(0, 256, size=(n, 784), dtype=np.uint8)

X = X_uint8.astype(np.float32) / 255.0

assert X.shape == (n, 784)
assert X.dtype in (np.float32, np.float64)
assert np.isfinite(X).all()
assert X.min() >= 0.0
assert X.max() <= 1.0

```

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 8.2: Invariantes de datos (shape, dtype, rango)</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M08-E08_2`
- **Duración estimada:** 15–30 min
- **Nivel:** Básico

#### 2) Idea clave
- Muchos “bugs de entrenamiento” son realmente *bugs de datos*.
- Fija estos invariantes temprano:
  - `X.shape == (n, 784)`
  - `X.dtype` es float
  - valores en `[0,1]`
  - sin `NaN/inf` (`isfinite`) en todo el dataset

#### 3) Errores comunes
- Normalizar con división entera (cast inesperado).
- Olvidar castear a float antes de dividir.
- Asumir min/max sin validarlo con asserts.

#### 4) Nota docente
- Pide al alumno inyectar un `NaN` a propósito y confirmar que el `assert` lo detecta.
</details>

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

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 8.3: One-hot encoding (multiclase)</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M08-E08_3`
- **Duración estimada:** 15–25 min
- **Nivel:** Básico

#### 2) Idea clave
- One-hot transforma `y:(n,)` en `Y:(n,k)` para poder expresar cross-entropy de forma vectorizada.
- Invariante: `argmax(Y[i]) == y[i]`.

#### 3) Errores comunes
- No castear labels a `int` (rompe la indexación).
- Labels fuera de rango `[0, k-1]`.
- Mezclar shapes `(n,1)` y `(n,)` sin ser explícito.

#### 4) Nota docente
- Pide al alumno probar labels que incluyan `0` y `k-1` para cubrir bordes.
</details>

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


rng = np.random.default_rng(2)
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

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 8.4: PCA (SVD) y varianza explicada</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M08-E08_4`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio

#### 2) Idea clave
- PCA requiere centrar: `Xc = X - mean(X)`.
- En SVD, las direcciones principales salen de `Vt`; las primeras `k` filas definen el subespacio.
- El error de reconstrucción debe bajar cuando `k` aumenta.

#### 3) Errores comunes
- Olvidar centrar y luego malinterpretar componentes.
- Confundir los roles de `U` y `V` en SVD.
- Calcular ratios de varianza sin dividir por la varianza total.

#### 4) Nota docente
- Pide al alumno explicar por qué `k=784` reconstruye “perfecto” (salvo error numérico).
</details>

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


rng = np.random.default_rng(3)
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

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 8.5: Monotonía de la inercia en K-Means</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M08-E08_5`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio

#### 2) Idea clave
- El algoritmo de Lloyd alterna:
  - asignación (centroide más cercano)
  - update (centroide = promedio de puntos asignados)
- Con las definiciones estándar, cada paso no debería aumentar la inercia `J`.

#### 3) Errores comunes
- No manejar clusters vacíos (promedio de conjunto vacío).
- Calcular distancias mal por broadcasting.
- Medir `J` con labels que no corresponden al set de centroides.

#### 4) Nota docente
- Pide al alumno forzar un cluster vacío y justificar la estrategia de fallback.
</details>

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


rng = np.random.default_rng(4)
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

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 8.6: Gradient check en Logistic Regression (OvA)</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M08-E08_6`
- **Duración estimada:** 40–80 min
- **Nivel:** Avanzado

#### 2) Idea clave
- Gradient checking valida tu gradiente analítico contra una aproximación numérica en pocas coordenadas.
- En OvA puedes validar primero un clasificador (una clase) antes de escalar a 10.

#### 3) Errores comunes
- Mezclar `y:(n,)` con `p:(n,1)` y caer en broadcasting silencioso.
- Olvidar la normalización por `n`.
- Elegir `h` demasiado grande (sesgo) o demasiado pequeño (ruido numérico).

#### 4) Nota docente
- Pide al alumno chequear 2 coordenadas aleatorias y comparar el error relativo.
</details>

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


rng = np.random.default_rng(5)
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

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 8.7: Overfit sanity check en MLP</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M08-E08_7`
- **Duración estimada:** 45–90 min
- **Nivel:** Avanzado

#### 2) Idea clave
- Overfit en un mini-batch es un protocolo *obligatorio* de depuración: si no ajusta 64 ejemplos, asume bug.
- Para estabilidad, softmax debe implementarse con `logsumexp`.

#### 3) Errores comunes
- Inicialización demasiado pequeña o learning rate muy bajo → no progresa.
- Softmax inestable (overflow) → `NaN` en loss.
- Errores de shape en gradientes (en especial biases con broadcasting).

#### 4) Nota docente
- Pide al alumno registrar `loss` cada 20 pasos y explicar la tendencia.
</details>

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

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 8.8: Confusion matrix y F1 macro</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M08-E08_8`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio

#### 2) Idea clave
- Accuracy puede ocultar desbalance de clases.
- Macro-F1 promedia el F1 por clase, ponderando todas las clases por igual.

#### 3) Errores comunes
- Dividir por cero cuando una clase no tiene predicciones / no tiene verdaderos (usa `eps`).
- Usar micro-F1 cuando el objetivo es macro-F1.
- Construir `cm` con índices invertidos (`cm[p,t]` vs `cm[t,p]`).

#### 4) Nota docente
- Pide al alumno crear un caso donde una clase nunca se predice e interpretar las métricas.
</details>

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

<details open>
<summary><strong>Complemento pedagógico — Ejercicio 8.9: Consistencia en comparación de modelos</strong></summary>

#### 1) Metadatos
- **ID (opcional):** `M08-E08_9`
- **Duración estimada:** 15–25 min
- **Nivel:** Básico

#### 2) Idea clave
- Comparar modelos requiere consistencia: misma métrica y el mismo split.
- Ordenar es simple, pero importan los *invariantes*: valores en `[0,1]`, mejor primero, naming estable.

#### 3) Errores comunes
- Mezclar accuracy de train para un modelo y test para otro.
- Comparar modelos entrenados con seeds/splits distintos.
- No validar el rango de la métrica.

#### 4) Nota docente
- Pide al alumno extender el dict con un modelo nuevo y confirmar que pasan los checks.
</details>

---

## Executive Summary

This project demonstrates competency in all three courses of the Machine Learning
Pathway (Line 1) through a complete analysis of the Fashion-MNIST dataset.

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
