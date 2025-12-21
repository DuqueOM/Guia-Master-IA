# Módulo 01 — Computación Científica con Python: NumPy y Pandas

> **Nivel:** Preparación MS-AI (University of Colorado Boulder Pathway)
> **Duración:** Semanas 1-2 | **Créditos Equivalentes:** 2 unidades
> **Prerrequisitos:** Python básico (variables, funciones, estructuras de control)

---

## Índice del Módulo

1. [Fundamentos de Computación Numérica](#1-fundamentos-de-computación-numérica)
2. [NumPy: El Corazón del Stack Científico](#2-numpy-el-corazón-del-stack-científico)
3. [Pandas: Manipulación de Datos Tabulares](#3-pandas-manipulación-de-datos-tabulares)
4. [Broadcasting: Aritmética de Tensores](#4-broadcasting-aritmética-de-tensores)
5. [Vectorización y Rendimiento](#5-vectorización-y-rendimiento)
6. [Laboratorios Visuales Interactivos](#6-laboratorios-visuales-interactivos)

---

# 1. Fundamentos de Computación Numérica

## 1.1 Contexto Histórico y Motivación

### El Problema Computacional del Siglo XX

En 1945, John von Neumann publicó el *First Draft of a Report on the EDVAC*, estableciendo la arquitectura de memoria compartida que domina la computación moderna. Sin embargo, esta arquitectura presenta un **cuello de botella crítico**: la velocidad del procesador supera exponencialmente la velocidad de acceso a memoria (el llamado *memory wall*).

**Implicación para Machine Learning:** Los algoritmos de ML son *memory-bound*, no *compute-bound*. El costo dominante no es la operación aritmética, sino mover datos entre niveles de caché y memoria principal.

### Por Qué Python Puro es Inaceptable

Python es un lenguaje interpretado con tipado dinámico. Cada operación elemental incurre en:

1. **Dispatch dinámico:** Resolución de tipos en tiempo de ejecución
2. **Boxing/Unboxing:** Cada número es un objeto completo con metadatos
3. **Reference counting:** Gestión automática de memoria con overhead por operación

**Consecuencia cuantificable:**

$$
T_{\text{Python}} \approx 100 \cdot T_{\text{C/Fortran}}
$$

Para $n = 10^6$ operaciones, esto significa la diferencia entre 2ms y 200ms—inaceptable en pipelines de ML donde estas operaciones se ejecutan millones de veces durante el entrenamiento.

### La Solución: Bibliotecas Compiladas con Interfaz Python

NumPy (Numerical Python) fue desarrollado por Travis Oliphant en 2005, consolidando los proyectos *Numeric* (1995) y *Numarray* (2001). Su arquitectura sigue un principio elegante:

> **Principio de NumPy:** Exponer una API de alto nivel en Python que delegue el cómputo intensivo a rutinas compiladas en C/Fortran optimizadas para la arquitectura del hardware.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PYTHON (Alto nivel)                          │
│   - Sintaxis legible                                                │
│   - Gestión automática de memoria                                   │
│   - Prototipado rápido                                              │
├─────────────────────────────────────────────────────────────────────┤
│                     NUMPY C-API (Puente)                            │
│   - Traducción de objetos Python a buffers de memoria               │
│   - Dispatch de operaciones a rutinas optimizadas                   │
├─────────────────────────────────────────────────────────────────────┤
│                    BLAS/LAPACK (Bajo nivel)                         │
│   - Operaciones vectorizadas en C/Fortran                           │
│   - Optimizaciones específicas de arquitectura (SIMD, AVX)          │
│   - Paralelización automática en múltiples cores                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1.2 Analogía de Alto Impacto: La Fábrica de Chocolates

Imagina que tienes que producir 1 millón de chocolates:

**Método Python Puro (Artesanal):**
- Un maestro chocolatero hace cada chocolate individualmente
- Examina cada ingrediente, decide cómo mezclarlo, lo moldea a mano
- Tiempo: 1 chocolate por segundo = 11.5 días

**Método NumPy (Industrial):**
- Una línea de producción automatizada procesa lotes de 1000 chocolates
- Los ingredientes fluyen en bloques homogéneos por cintas transportadoras optimizadas
- Tiempo: 1000 chocolates por segundo = 16 minutos

La clave no es que la fábrica sea "más rápida"—es que **procesa en bloque** y **elimina la toma de decisiones por unidad**.

---

## 1.3 Rigor Matemático: El Modelo de Memoria

### Definición Formal: ndarray

Un `ndarray` de NumPy es una tupla $(B, S, D, T)$ donde:

- $B$ : **Buffer** — Bloque contiguo de memoria de $N$ bytes
- $S$ : **Shape** — Tupla $(d_1, d_2, \ldots, d_k)$ con $\prod_{i=1}^{k} d_i = N / \text{sizeof}(T)$
- $D$ : **Strides** — Tupla $(s_1, s_2, \ldots, s_k)$ donde $s_i$ indica bytes entre elementos consecutivos en dimensión $i$
- $T$ : **Dtype** — Tipo de dato homogéneo (e.g., `float64`, `int32`)

### Fórmula de Acceso a Elemento

Para un array $A$ con shape $(d_1, \ldots, d_k)$ y strides $(s_1, \ldots, s_k)$, el elemento $A[i_1, \ldots, i_k]$ se ubica en:

$$
\text{offset} = \sum_{j=1}^{k} i_j \cdot s_j
$$

**Implicación crítica:** La operación de *reshape* no copia datos—solo reinterpreta los strides.

### Ejemplo Concreto

```python
import numpy as np

# Crear array con datos en memoria contigua
a = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float64)

print(f"Shape: {a.shape}")       # (2, 3)
print(f"Strides: {a.strides}")   # (24, 8) — 24 bytes por fila, 8 bytes por elemento
print(f"Dtype: {a.dtype}")       # float64 (8 bytes por número)

# Verificación: 24 = 3 elementos × 8 bytes/elemento
assert a.strides[0] == a.shape[1] * a.itemsize
```

---

## 1.4 Asunciones del Modelo

Para que NumPy funcione eficientemente, el dataset debe cumplir:

| Asunción | Descripción | Violación Común |
|----------|-------------|-----------------|
| **Homogeneidad** | Todos los elementos del mismo tipo | Mezclar strings con números |
| **Tamaño conocido** | Dimensiones fijas tras creación | Append dinámico en loops |
| **Memoria suficiente** | El array completo cabe en RAM | Datasets > memoria disponible |
| **Alineación** | Datos contiguos en memoria | Slices discontinuos (views) |

**Cuando las asunciones fallan:**

- **Datos heterogéneos:** Usar Pandas (maneja columnas de tipos mixtos)
- **Datos que no caben en memoria:** Usar Dask, Vaex, o streaming
- **Append dinámico:** Pre-alocar con `np.empty()` o usar listas y convertir al final

---

## 1.5 Pensamiento Crítico: Escenarios de Fallo

### Escenario 1: El Dtype Silencioso

```python
import numpy as np

# PELIGRO: El dtype se infiere automáticamente
a = np.array([1, 2, 3])           # dtype=int64
b = np.array([1.0, 2.0, 3.0])     # dtype=float64

# Operación entre tipos: promoción silenciosa
c = a + b  # dtype=float64 (correcto)

# PERO: asignación in-place NO promociona
a += 0.5   # ¡Se trunca a int! a = [1, 2, 3], no [1.5, 2.5, 3.5]
```

**Regla de oro:** Siempre declarar `dtype=np.float64` explícitamente para datos de ML.

### Escenario 2: La Vista Fantasma

```python
import numpy as np

X = np.array([[1, 2, 3],
              [4, 5, 6]])

# Esto es una VISTA, no una copia
fila = X[0]
fila[0] = 999  # ¡Modifica X original!

print(X[0, 0])  # 999

# Solución: copia explícita
fila_segura = X[0].copy()
```

**Diagnóstico:** `np.shares_memory(X, fila)` retorna `True` si comparten buffer.

---

## 1.6 Comparativa: NumPy vs Alternativas

| Criterio | NumPy | PyTorch Tensors | JAX Arrays | CuPy |
|----------|-------|-----------------|------------|------|
| **Backend** | CPU (BLAS) | CPU/GPU | XLA (CPU/GPU/TPU) | GPU (CUDA) |
| **Autograd** | ❌ No | ✅ Sí | ✅ Sí (funcional) | ❌ No |
| **API** | Estándar de facto | Similar a NumPy | Similar a NumPy | Idéntica a NumPy |
| **Uso principal** | Prototipado, preproceso | Deep Learning | Investigación | GPU computing |

**Recomendación para el Pathway:** Dominar NumPy primero—todos los demás frameworks imitan su API.

---

# 2. NumPy: El Corazón del Stack Científico

## 2.1 Creación de Arrays

### Contexto Histórico

El diseño de constructores de NumPy refleja patrones de MATLAB (1984) y APL (1966), lenguajes pioneros en computación matricial. La filosofía es: **hacer lo común fácil y lo complejo posible**.

### Constructores Fundamentales

```python
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# DESDE DATOS EXISTENTES
# ══════════════════════════════════════════════════════════════════════

# Conversión explícita con control de tipo
data = [1, 2, 3, 4, 5]
arr = np.array(data, dtype=np.float64)
assert arr.dtype == np.float64

# Desde nested lists (matriz)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]], dtype=np.float64)
assert matrix.shape == (2, 3)

# ══════════════════════════════════════════════════════════════════════
# ARRAYS INICIALIZADOS
# ══════════════════════════════════════════════════════════════════════

# Ceros: útil para acumuladores
zeros = np.zeros((100, 50), dtype=np.float64)  # Shape (100, 50)

# Unos: útil para inicialización de pesos
ones = np.ones((64, 128), dtype=np.float64)

# Identidad: matriz cuadrada con 1s en diagonal
I = np.eye(4, dtype=np.float64)  # I @ x = x para todo x

# Sin inicializar: MÁS RÁPIDO pero contiene basura
buffer = np.empty((1000, 1000), dtype=np.float64)  # ¡No asumir valores!

# ══════════════════════════════════════════════════════════════════════
# SECUENCIAS
# ══════════════════════════════════════════════════════════════════════

# arange: como range() pero retorna array
# CUIDADO: con floats puede dar longitud inesperada
indices = np.arange(0, 10, 1)  # [0, 1, 2, ..., 9]

# linspace: PREFERIDO para floats — garantiza N puntos exactos
x = np.linspace(0, 2*np.pi, 100)  # 100 puntos entre 0 y 2π (inclusive)
assert len(x) == 100
assert x[0] == 0
assert np.isclose(x[-1], 2*np.pi)

# ══════════════════════════════════════════════════════════════════════
# ALEATORIOS (Generador moderno)
# ══════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(seed=42)  # Reproducibilidad

# Normal estándar: μ=0, σ=1
normal = rng.standard_normal((100, 10))

# Uniforme en [0, 1)
uniform = rng.random((100, 10))

# Enteros uniformes en [low, high)
integers = rng.integers(low=0, high=10, size=(5, 5))
```

### Rigor Matemático: ¿Por qué `linspace` sobre `arange` para floats?

El problema con `arange` y floats:

$$
\texttt{arange}(0, 1, 0.1) \rightarrow [0.0, 0.1, 0.2, \ldots, 0.9] \quad \text{(10 elementos)}
$$

Pero debido a errores de punto flotante:

$$
0.1 + 0.1 + \ldots + 0.1 \neq 1.0
$$

En algunos sistemas, `arange(0, 1, 0.1)` puede retornar 10 u 11 elementos dependiendo del redondeo.

**`linspace` garantiza:**

$$
x_i = a + i \cdot \frac{b - a}{n - 1}, \quad i \in \{0, 1, \ldots, n-1\}
$$

---

## 2.2 Indexación y Slicing

### El Contrato de Indexación NumPy

NumPy extiende la indexación de Python con semántica multidimensional:

```python
import numpy as np

# Crear matriz de ejemplo
A = np.arange(20).reshape(4, 5)
# [[ 0,  1,  2,  3,  4],
#  [ 5,  6,  7,  8,  9],
#  [10, 11, 12, 13, 14],
#  [15, 16, 17, 18, 19]]

# ══════════════════════════════════════════════════════════════════════
# INDEXACIÓN BÁSICA (retorna VISTAS)
# ══════════════════════════════════════════════════════════════════════

# Elemento escalar
elem = A[1, 2]  # 7 (fila 1, columna 2)

# Fila completa
row = A[1]      # [5, 6, 7, 8, 9] — shape (5,), ES UNA VISTA
row = A[1, :]   # Equivalente explícito

# Columna completa
col = A[:, 2]   # [2, 7, 12, 17] — shape (4,), ES UNA VISTA

# Submatriz (slice 2D)
sub = A[1:3, 2:4]  # [[7, 8], [12, 13]] — shape (2, 2), VISTA

# ══════════════════════════════════════════════════════════════════════
# FANCY INDEXING (retorna COPIAS)
# ══════════════════════════════════════════════════════════════════════

# Selección con lista de índices
rows = A[[0, 2, 3]]           # Filas 0, 2, 3 — shape (3, 5), ES COPIA
elems = A[[0, 1, 2], [4, 3, 2]]  # A[0,4], A[1,3], A[2,2] = [4, 8, 12]

# ══════════════════════════════════════════════════════════════════════
# INDEXACIÓN BOOLEANA (retorna COPIAS, siempre 1D)
# ══════════════════════════════════════════════════════════════════════

mask = A > 10
filtered = A[mask]  # [11, 12, 13, 14, 15, 16, 17, 18, 19] — shape (9,)

# Combinación de condiciones: usar & (and), | (or), ~ (not)
# IMPORTANTE: paréntesis obligatorios por precedencia de operadores
mask2 = (A > 5) & (A < 15)
filtered2 = A[mask2]  # [6, 7, 8, 9, 10, 11, 12, 13, 14]
```

### Teorema Fundamental: Vista vs Copia

| Operación | Resultado | Modifica Original |
|-----------|-----------|-------------------|
| `A[i]`, `A[i:j]`, `A[:, k]` | Vista | ✅ Sí |
| `A[[i, j, k]]` | Copia | ❌ No |
| `A[mask]` | Copia | ❌ No |
| `A.copy()` | Copia | ❌ No |

**Verificación programática:**

```python
import numpy as np

A = np.arange(10)
view = A[2:5]
copy = A[[2, 3, 4]]

print(np.shares_memory(A, view))   # True
print(np.shares_memory(A, copy))   # False
```

---

## 2.3 Operaciones Matriciales

### Contexto: BLAS y el Legado de Fortran

Las operaciones matriciales de NumPy delegan a **BLAS** (Basic Linear Algebra Subprograms), una especificación de 1979 implementada en Fortran. Las implementaciones modernas (OpenBLAS, Intel MKL, Apple Accelerate) incluyen:

- **Paralelización automática** en múltiples cores
- **Vectorización SIMD** (Single Instruction, Multiple Data)
- **Optimización de caché** mediante *blocking*

Por esto, `A @ B` no es "una multiplicación"—es **miles de líneas de código optimizado** ejecutándose bajo el capó.

### El Operador `@` vs `*`

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]], dtype=np.float64)

B = np.array([[5, 6],
              [7, 8]], dtype=np.float64)

# ══════════════════════════════════════════════════════════════════════
# MULTIPLICACIÓN ELEMENTO A ELEMENTO (Hadamard)
# ══════════════════════════════════════════════════════════════════════

C_hadamard = A * B
# [[ 5, 12],
#  [21, 32]]
# C[i,j] = A[i,j] * B[i,j]

# ══════════════════════════════════════════════════════════════════════
# PRODUCTO MATRICIAL (matmul)
# ══════════════════════════════════════════════════════════════════════

C_matmul = A @ B
# [[19, 22],
#  [43, 50]]
# C[i,j] = Σₖ A[i,k] * B[k,j]
```

### Formalización del Producto Matricial

Para $A \in \mathbb{R}^{m \times n}$ y $B \in \mathbb{R}^{n \times p}$:

$$
C = AB \in \mathbb{R}^{m \times p}, \quad C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$

**Regla de compatibilidad dimensional:**

$$
(m \times \underbrace{n}_{\text{deben coincidir}}) \times (\underbrace{n}_{\text{deben coincidir}} \times p) \rightarrow (m \times p)
$$

### Casos Especiales en ML

```python
import numpy as np

# Datos típicos de ML
N, D, K = 100, 784, 10  # 100 muestras, 784 features, 10 clases
X = np.random.randn(N, D)  # Batch de inputs
W = np.random.randn(D, K)  # Pesos de capa densa
b = np.random.randn(K)     # Bias

# Forward pass de capa lineal
Z = X @ W + b  # (N, D) @ (D, K) + (K,) → (N, K)
assert Z.shape == (N, K)

# Vector de pesos para regresión
w = np.random.randn(D)     # (D,)
y_pred = X @ w             # (N, D) @ (D,) → (N,)
assert y_pred.shape == (N,)
```

---

## 2.4 Reshape: Reinterpretación de Memoria

### Principio Fundamental

`reshape` **no mueve datos**—solo cambia la interpretación de los strides:

```python
import numpy as np

a = np.arange(12)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Todos estos comparten el MISMO buffer de memoria
b = a.reshape(3, 4)   # (3, 4)
c = a.reshape(4, 3)   # (4, 3)
d = a.reshape(2, 2, 3)  # (2, 2, 3)

assert np.shares_memory(a, b)
assert np.shares_memory(a, c)
assert np.shares_memory(a, d)

# El invariante es el tamaño total
assert a.size == b.size == c.size == d.size == 12
```

### El Parámetro `-1`: Inferencia Automática

NumPy puede calcular **una** dimensión automáticamente:

```python
import numpy as np

X = np.random.randn(100, 784)  # 100 imágenes de 28×28 aplanadas

# Reshape a formato de imagen (inferir primera dimensión)
X_img = X.reshape(-1, 28, 28)
assert X_img.shape == (100, 28, 28)

# Aplanar de vuelta
X_flat = X_img.reshape(X_img.shape[0], -1)
assert X_flat.shape == (100, 784)
```

### Error Común: `(n,)` vs `(n, 1)`

```python
import numpy as np

v = np.array([1, 2, 3])  # Shape (3,) — vector 1D

# NO es lo mismo que:
v_col = v.reshape(-1, 1)  # Shape (3, 1) — matriz columna
v_row = v.reshape(1, -1)  # Shape (1, 3) — matriz fila

# Diferencias en operaciones
print(v @ v)          # Producto punto: escalar = 14
print(v_col @ v_row)  # Outer product: matriz (3, 3)
# [[1, 2, 3],
#  [2, 4, 6],
#  [3, 6, 9]]
```

---

# 3. Pandas: Manipulación de Datos Tabulares

## 3.1 Contexto y Motivación

### El Problema de los Datos Reales

NumPy asume datos homogéneos. Pero los datasets reales contienen:

- **Columnas de tipos mixtos:** numéricos, categóricos, fechas, texto
- **Valores faltantes:** representados como `NaN`, `None`, o códigos especiales
- **Metadatos:** nombres de columnas, índices con significado

**Pandas** (Panel Data, desarrollado por Wes McKinney en 2008 para análisis financiero) resuelve esto con estructuras que combinan la eficiencia de NumPy con la flexibilidad de etiquetas.

### Arquitectura DataFrame

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DataFrame                                   │
├─────────────┬─────────────┬─────────────┬─────────────┬────────────┤
│   Index     │   Col_A     │   Col_B     │   Col_C     │   Col_D    │
│  (object)   │  (float64)  │  (int64)    │  (object)   │ (datetime) │
├─────────────┼─────────────┼─────────────┼─────────────┼────────────┤
│  "row_0"    │    1.5      │     10      │   "cat"     │ 2024-01-01 │
│  "row_1"    │    2.3      │     20      │   "dog"     │ 2024-01-02 │
│  "row_2"    │    NaN      │     30      │   "cat"     │ 2024-01-03 │
└─────────────┴─────────────┴─────────────┴─────────────┴────────────┘
                    ↓               ↓            ↓            ↓
               np.ndarray      np.ndarray   np.ndarray   np.ndarray
               (float64)       (int64)      (object)     (datetime64)
```

**Cada columna es un `np.ndarray` independiente** con su propio dtype.

## 3.2 Pipeline Canónico: CSV → Modelo

```python
import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# CAPA 1: CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════

df = pd.read_csv(
    'data/dataset.csv',
    dtype={'id': str, 'category': 'category'},  # Tipos explícitos
    parse_dates=['timestamp'],                   # Parsear fechas
    na_values=['', 'NA', 'null', '-999']         # Valores a tratar como NaN
)

# ══════════════════════════════════════════════════════════════════════
# CAPA 2: INSPECCIÓN (EDA Mínimo)
# ══════════════════════════════════════════════════════════════════════

print(df.shape)           # (n_rows, n_cols)
print(df.dtypes)          # Tipos de cada columna
print(df.info())          # Resumen con memoria y nulos
print(df.describe())      # Estadísticas de columnas numéricas

# Conteo de nulos
null_counts = df.isnull().sum()
null_pct = df.isnull().mean() * 100
print(null_pct.sort_values(ascending=False))

# ══════════════════════════════════════════════════════════════════════
# CAPA 3: LIMPIEZA
# ══════════════════════════════════════════════════════════════════════

# Estrategia de imputación por tipo de variable
df_clean = df.copy()

# Numéricas: mediana (robusta a outliers)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(median_val)

# Categóricas: moda o categoría "Unknown"
categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    df_clean[col] = df_clean[col].fillna('Unknown')

# Verificar que no quedan nulos
assert df_clean.isnull().sum().sum() == 0, "Aún hay valores nulos"

# ══════════════════════════════════════════════════════════════════════
# CAPA 4: PREPARACIÓN PARA ML
# ══════════════════════════════════════════════════════════════════════

# Seleccionar features y target
feature_cols = ['feat_1', 'feat_2', 'feat_3', 'feat_4']
target_col = 'target'

X = df_clean[feature_cols].to_numpy(dtype=np.float64)
y = df_clean[target_col].to_numpy(dtype=np.float64)

# Validaciones críticas
assert X.ndim == 2, f"X debe ser 2D, got {X.ndim}D"
assert y.ndim == 1, f"y debe ser 1D, got {y.ndim}D"
assert X.shape[0] == y.shape[0], "X e y deben tener mismo número de muestras"
assert not np.any(np.isnan(X)), "X contiene NaN"
assert not np.any(np.isnan(y)), "y contiene NaN"

print(f"X: {X.shape}, dtype={X.dtype}")
print(f"y: {y.shape}, dtype={y.dtype}")
```

## 3.3 Operaciones Avanzadas

### GroupBy: Split-Apply-Combine

El patrón **Split-Apply-Combine** (Wickham, 2011) es fundamental para agregaciones:

```python
import pandas as pd
import numpy as np

# Datos de ejemplo
df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'B'],
    'value': [10, 20, 30, 40, 50]
})

# Split-Apply-Combine
result = df.groupby('category')['value'].agg(['mean', 'std', 'count'])
#          mean       std  count
# A        15.0  7.071068      2
# B        40.0 10.000000      3
```

### Merge: Combinación de DataFrames

```python
import pandas as pd

# Dos tablas relacionadas
users = pd.DataFrame({
    'user_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

transactions = pd.DataFrame({
    'user_id': [1, 1, 2, 4],
    'amount': [100, 200, 150, 300]
})

# Inner join (solo matches)
merged_inner = pd.merge(users, transactions, on='user_id', how='inner')

# Left join (todos los users, NaN si no hay transacción)
merged_left = pd.merge(users, transactions, on='user_id', how='left')
```

---

# 4. Broadcasting: Aritmética de Tensores

## 4.1 Contexto Histórico

El broadcasting fue introducido en APL (1966) y refinado en NumPy siguiendo la "regla de compatibilidad dimensional". Es la característica más poderosa y más peligrosa de NumPy.

## 4.2 La Regla de Broadcasting

**Definición formal:** Dos arrays son compatibles para broadcasting si, para cada dimensión (alineando desde la derecha), los tamaños son iguales o uno de ellos es 1.

### Algoritmo de Broadcasting

1. **Padding:** Si los arrays tienen diferente número de dimensiones, agregar 1s al principio del shape más corto
2. **Verificación:** Para cada par de dimensiones, verificar compatibilidad
3. **Expansión:** Las dimensiones de tamaño 1 se "estiran" para coincidir

### Ejemplo Detallado

```python
import numpy as np

A = np.ones((3, 4))      # Shape: (3, 4)
b = np.array([1, 2, 3, 4])  # Shape: (4,)

# Paso 1: Padding
# A: (3, 4)
# b: (1, 4)  ← se agrega 1 al principio

# Paso 2: Verificación
# Dimensión 0: 3 vs 1 → compatible (1 se estira)
# Dimensión 1: 4 vs 4 → compatible (iguales)

# Paso 3: Resultado
C = A + b  # Shape: (3, 4)
# Cada fila de A se suma con b
```

### Visualización de Broadcasting

```
A (3, 4):                  b (4,) → (1, 4):
┌───┬───┬───┬───┐         ┌───┬───┬───┬───┐
│ 1 │ 1 │ 1 │ 1 │         │ 1 │ 2 │ 3 │ 4 │
├───┼───┼───┼───┤         └───┴───┴───┴───┘
│ 1 │ 1 │ 1 │ 1 │              ↓ broadcast
├───┼───┼───┼───┤         ┌───┬───┬───┬───┐
│ 1 │ 1 │ 1 │ 1 │         │ 1 │ 2 │ 3 │ 4 │
└───┴───┴───┴───┘         ├───┼───┼───┼───┤
                          │ 1 │ 2 │ 3 │ 4 │
A + b:                    ├───┼───┼───┼───┤
┌───┬───┬───┬───┐         │ 1 │ 2 │ 3 │ 4 │
│ 2 │ 3 │ 4 │ 5 │         └───┴───┴───┴───┘
├───┼───┼───┼───┤
│ 2 │ 3 │ 4 │ 5 │
├───┼───┼───┼───┤
│ 2 │ 3 │ 4 │ 5 │
└───┴───┴───┴───┘
```

## 4.3 Aplicación Crítica: Normalización de Features

```python
import numpy as np

# Dataset: N muestras, D features
N, D = 1000, 50
X = np.random.randn(N, D) * 10 + 5  # Media ≈ 5, std ≈ 10

# Normalización Z-score por columna (feature-wise)
# μ y σ deben tener shape (D,) o (1, D) para broadcast correcto

mu = X.mean(axis=0)           # Shape: (D,)
sigma = X.std(axis=0) + 1e-8  # Shape: (D,), +eps para evitar división por cero

X_normalized = (X - mu) / sigma  # Broadcasting: (N, D) - (D,) / (D,) → (N, D)

# Verificación
assert X_normalized.shape == X.shape
assert np.allclose(X_normalized.mean(axis=0), 0, atol=1e-10)
assert np.allclose(X_normalized.std(axis=0), 1, atol=1e-10)
```

## 4.4 El Error Silencioso: Broadcasting Incorrecto

```python
import numpy as np

X = np.random.randn(100, 50)  # (N, D)

# ❌ INCORRECTO: media global (escalar)
mu_wrong = X.mean()  # Shape: () — escalar
X_wrong = X - mu_wrong  # Resta el mismo valor a TODO

# ✅ CORRECTO: media por columna
mu_correct = X.mean(axis=0)  # Shape: (D,)
X_correct = X - mu_correct   # Resta media de cada feature

# El código "funciona" en ambos casos, pero semánticamente son MUY diferentes
```

**Regla de seguridad:** Siempre usar `keepdims=True` cuando no estés seguro:

```python
import numpy as np

X = np.random.randn(100, 50)

mu = X.mean(axis=0, keepdims=True)  # Shape: (1, 50)
sigma = X.std(axis=0, keepdims=True)  # Shape: (1, 50)

X_norm = (X - mu) / sigma  # Broadcasting explícito y seguro
```

---

# 5. Vectorización y Rendimiento

## 5.1 Por Qué Vectorizar

### Benchmark Comparativo

```python
import numpy as np
import time

def benchmark(func, *args, n_runs=100):
    """Mide tiempo promedio de ejecución."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)
    return np.mean(times) * 1000  # ms

# Datos de prueba
N = 100_000
a = np.random.randn(N)
b = np.random.randn(N)

# Versión con loop
def dot_loop(a, b):
    result = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Versión vectorizada
def dot_vectorized(a, b):
    return np.dot(a, b)

# Benchmark
t_loop = benchmark(dot_loop, a, b, n_runs=10)
t_vec = benchmark(dot_vectorized, a, b, n_runs=100)

print(f"Loop:       {t_loop:.2f} ms")
print(f"Vectorized: {t_vec:.4f} ms")
print(f"Speedup:    {t_loop/t_vec:.0f}x")
# Típicamente: 50-200x más rápido
```

### Análisis del Speedup

El speedup no es lineal y depende de:

1. **Overhead de interpretación:** Cada iteración en Python tiene costo fijo
2. **Cache locality:** Accesos contiguos en memoria son más rápidos
3. **SIMD:** Operaciones vectoriales procesan múltiples datos por instrucción
4. **Optimizaciones del compilador:** NumPy está compilado con `-O3`

## 5.2 Patrones de Vectorización

### Patrón 1: Eliminación de Loops Explícitos

```python
import numpy as np

X = np.random.randn(1000, 100)  # Datos
y = np.random.randn(1000)       # Target

# ❌ CON LOOP
def mse_loop(X, y, w):
    n = len(y)
    total = 0.0
    for i in range(n):
        pred = 0.0
        for j in range(len(w)):
            pred += X[i, j] * w[j]
        total += (pred - y[i]) ** 2
    return total / n

# ✅ VECTORIZADO
def mse_vectorized(X, y, w):
    pred = X @ w  # Predicciones vectorizadas
    return np.mean((pred - y) ** 2)
```

### Patrón 2: Uso de `np.where` en lugar de condicionales

```python
import numpy as np

x = np.random.randn(10000)

# ❌ CON LOOP
def relu_loop(x):
    result = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            result[i] = x[i]
    return result

# ✅ VECTORIZADO con np.where
def relu_where(x):
    return np.where(x > 0, x, 0)

# ✅ VECTORIZADO con np.maximum (aún más rápido)
def relu_maximum(x):
    return np.maximum(x, 0)
```

### Patrón 3: Distancias Euclidianas en Batch

Problema: Calcular distancia entre cada par de puntos en dos conjuntos.

```python
import numpy as np

# X: (N, D), Y: (M, D)
# Resultado: (N, M) donde D[i,j] = ||X[i] - Y[j]||²

def pairwise_distances_loop(X, Y):
    N, D = X.shape
    M = Y.shape[0]
    distances = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            diff = X[i] - Y[j]
            distances[i, j] = np.sum(diff ** 2)
    return distances

def pairwise_distances_vectorized(X, Y):
    # ||x - y||² = ||x||² + ||y||² - 2·x·y
    X_sqnorm = np.sum(X ** 2, axis=1, keepdims=True)  # (N, 1)
    Y_sqnorm = np.sum(Y ** 2, axis=1, keepdims=True)  # (M, 1)
    cross = X @ Y.T  # (N, M)
    return X_sqnorm + Y_sqnorm.T - 2 * cross  # Broadcasting: (N, 1) + (1, M) - (N, M)

# Verificación
np.random.seed(42)
X = np.random.randn(100, 50)
Y = np.random.randn(80, 50)

D_loop = pairwise_distances_loop(X, Y)
D_vec = pairwise_distances_vectorized(X, Y)

assert np.allclose(D_loop, D_vec)
```

---

# 6. Laboratorios Visuales Interactivos

## 6.1 Lab 1: Explorador de Broadcasting (Streamlit)

```python
"""
Archivo: visual_labs/m01_broadcasting_explorer.py
Ejecutar: streamlit run visual_labs/m01_broadcasting_explorer.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Broadcasting Explorer", layout="wide")
st.title("🔬 Explorador de Broadcasting NumPy")

st.markdown("""
### Regla de Broadcasting
Dos arrays son compatibles si, para cada dimensión (de derecha a izquierda):
1. Las dimensiones son iguales, O
2. Una de ellas es 1
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Array A")
    a_rows = st.slider("Filas de A", 1, 5, 3, key="a_rows")
    a_cols = st.slider("Columnas de A", 1, 5, 4, key="a_cols")
    A = np.arange(a_rows * a_cols).reshape(a_rows, a_cols)
    st.write(f"Shape: {A.shape}")
    st.dataframe(A)

with col2:
    st.subheader("Array B")
    b_shape_type = st.selectbox(
        "Tipo de B",
        ["Vector fila (1, n)", "Vector columna (m, 1)", "Escalar ()", "Matriz (m, n)"]
    )

    if b_shape_type == "Vector fila (1, n)":
        B = np.arange(a_cols).reshape(1, a_cols) * 10
    elif b_shape_type == "Vector columna (m, 1)":
        B = np.arange(a_rows).reshape(a_rows, 1) * 10
    elif b_shape_type == "Escalar ()":
        B = np.array(100)
    else:
        B = np.arange(a_rows * a_cols).reshape(a_rows, a_cols) * 10

    st.write(f"Shape: {B.shape}")
    if B.ndim == 0:
        st.write(f"Valor: {B}")
    else:
        st.dataframe(B)

st.subheader("Resultado: A + B")

try:
    C = A + B
    st.success(f"✅ Broadcasting exitoso! Shape resultado: {C.shape}")
    st.dataframe(C)

    # Visualización
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Array A", "Array B (broadcast)", "A + B"]
    )

    fig.add_trace(go.Heatmap(z=A, colorscale='Blues', showscale=False), row=1, col=1)

    B_broadcast = np.broadcast_to(B, C.shape) if B.ndim > 0 else np.full_like(C, B)
    fig.add_trace(go.Heatmap(z=B_broadcast, colorscale='Reds', showscale=False), row=1, col=2)

    fig.add_trace(go.Heatmap(z=C, colorscale='Viridis', showscale=False), row=1, col=3)

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

except ValueError as e:
    st.error(f"❌ Broadcasting falló: {e}")
    st.markdown("""
    **Diagnóstico:** Las shapes no son compatibles.
    Revisa que cada dimensión cumpla la regla.
    """)
```

## 6.2 Lab 2: Comparador de Rendimiento (Streamlit)

```python
"""
Archivo: visual_labs/m01_performance_benchmark.py
Ejecutar: streamlit run visual_labs/m01_performance_benchmark.py
"""
import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go

st.set_page_config(page_title="NumPy Performance Lab", layout="wide")
st.title("⚡ Laboratorio de Rendimiento: Loop vs Vectorizado")

operation = st.selectbox(
    "Selecciona operación",
    ["Producto punto", "Suma de elementos", "Distancia Euclidiana", "Normalización Z-score"]
)

sizes = st.multiselect(
    "Tamaños a probar (N)",
    [100, 1000, 10000, 100000, 1000000],
    default=[1000, 10000, 100000]
)

if st.button("🚀 Ejecutar Benchmark"):
    results = {"N": [], "Loop (ms)": [], "Vectorizado (ms)": [], "Speedup": []}

    progress = st.progress(0)

    for idx, N in enumerate(sizes):
        a = np.random.randn(N)
        b = np.random.randn(N)

        # Loop version
        if operation == "Producto punto":
            def loop_func():
                result = 0.0
                for i in range(N):
                    result += a[i] * b[i]
                return result
            vec_func = lambda: np.dot(a, b)
        elif operation == "Suma de elementos":
            def loop_func():
                result = 0.0
                for i in range(N):
                    result += a[i]
                return result
            vec_func = lambda: np.sum(a)
        elif operation == "Distancia Euclidiana":
            def loop_func():
                result = 0.0
                for i in range(N):
                    result += (a[i] - b[i]) ** 2
                return np.sqrt(result)
            vec_func = lambda: np.linalg.norm(a - b)
        else:  # Normalización
            def loop_func():
                mean = sum(a) / N
                std = np.sqrt(sum((x - mean)**2 for x in a) / N)
                return [(x - mean) / std for x in a]
            vec_func = lambda: (a - np.mean(a)) / np.std(a)

        # Benchmark
        n_runs = max(1, 100000 // N)

        start = time.perf_counter()
        for _ in range(n_runs):
            loop_func()
        t_loop = (time.perf_counter() - start) / n_runs * 1000

        start = time.perf_counter()
        for _ in range(n_runs):
            vec_func()
        t_vec = (time.perf_counter() - start) / n_runs * 1000

        results["N"].append(N)
        results["Loop (ms)"].append(t_loop)
        results["Vectorizado (ms)"].append(t_vec)
        results["Speedup"].append(t_loop / t_vec)

        progress.progress((idx + 1) / len(sizes))

    # Mostrar resultados
    st.subheader("Resultados")
    st.dataframe(results)

    # Gráfico
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Loop", x=[str(n) for n in results["N"]], y=results["Loop (ms)"]))
    fig.add_trace(go.Bar(name="Vectorizado", x=[str(n) for n in results["N"]], y=results["Vectorizado (ms)"]))
    fig.update_layout(
        barmode='group',
        xaxis_title="N (elementos)",
        yaxis_title="Tiempo (ms)",
        yaxis_type="log"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Speedup promedio", f"{np.mean(results['Speedup']):.0f}x")
```

## 6.3 Animación Manim: Memoria Contigua vs Dispersa

```python
"""
Archivo: visual_labs/m01_memory_animation.py
Ejecutar: manim -pql visual_labs/m01_memory_animation.py MemoryLayoutAnimation
"""
from manim import *

class MemoryLayoutAnimation(Scene):
    def construct(self):
        # Título
        title = Text("Memoria: Lista Python vs NumPy Array", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Lista Python (referencias dispersas)
        list_title = Text("Lista Python", font_size=24, color=RED)
        list_title.move_to(LEFT * 4 + UP * 2)

        # Crear cajas de referencia
        ref_boxes = VGroup()
        obj_boxes = VGroup()
        arrows = VGroup()

        for i in range(5):
            ref = Square(side_length=0.5, color=RED)
            ref.move_to(LEFT * 4 + DOWN * i * 0.7)
            ref_boxes.add(ref)

            obj = Square(side_length=0.5, color=YELLOW)
            obj.move_to(LEFT * (2 - i * 0.3) + DOWN * (i * 0.5 - 1) + RIGHT * np.random.uniform(-0.5, 0.5))
            obj_boxes.add(obj)

            arrow = Arrow(ref.get_right(), obj.get_left(), buff=0.1, color=GRAY)
            arrows.add(arrow)

        # NumPy Array (contiguo)
        np_title = Text("NumPy Array", font_size=24, color=GREEN)
        np_title.move_to(RIGHT * 3 + UP * 2)

        np_boxes = VGroup()
        for i in range(5):
            box = Square(side_length=0.5, color=GREEN)
            box.move_to(RIGHT * 3 + DOWN * i * 0.6)
            num = Text(str(i+1), font_size=20)
            num.move_to(box)
            np_boxes.add(VGroup(box, num))

        # Animaciones
        self.play(Write(list_title), Write(np_title))
        self.play(
            LaggedStart(*[Create(box) for box in ref_boxes], lag_ratio=0.1),
            LaggedStart(*[Create(box) for box in np_boxes], lag_ratio=0.1),
        )
        self.play(
            LaggedStart(*[Create(obj) for obj in obj_boxes], lag_ratio=0.1),
            LaggedStart(*[Create(arrow) for arrow in arrows], lag_ratio=0.1),
        )
        self.wait()

        # Explicación
        exp1 = Text("Disperso en memoria", font_size=18, color=RED)
        exp1.next_to(ref_boxes, DOWN)

        exp2 = Text("Contiguo en memoria", font_size=18, color=GREEN)
        exp2.next_to(np_boxes, DOWN)

        self.play(Write(exp1), Write(exp2))
        self.wait()

        # Conclusión
        conclusion = Text(
            "Acceso contiguo = mejor uso de caché = más velocidad",
            font_size=24,
            color=BLUE
        )
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(2)
```

---

# 7. Pensamiento Crítico y Edge Cases

## 7.1 Escenario de Fallo 1: Overflow Silencioso

```python
import numpy as np

# Arrays de enteros tienen límites
a = np.array([2**62, 2**62], dtype=np.int64)
b = a + a  # ¡OVERFLOW! Sin error ni warning

print(b)  # Números negativos o basura

# Solución: usar float64 o verificar límites
a_safe = np.array([2**62, 2**62], dtype=np.float64)
b_safe = a_safe + a_safe  # Funciona correctamente
```

## 7.2 Escenario de Fallo 2: Broadcasting Semántico Incorrecto

```python
import numpy as np

# Supongamos que queremos sumar bias a cada muestra
X = np.random.randn(100, 10)  # 100 muestras, 10 features
b = np.random.randn(100)       # ¿Bias? NO - esto es por muestra, no por feature

# El broadcasting "funciona" pero está MAL semánticamente
# X: (100, 10)
# b: (100,) → se convierte en (100, 1) o (1, 100)?

# NumPy hace: (100, 10) + (100,) → error porque 10 ≠ 100

# Si b fuera (10,):
b_correct = np.random.randn(10)  # Bias por feature
Z = X + b_correct  # (100, 10) + (10,) → (100, 10) ✅
```

## 7.3 Comparativa: NumPy vs Alternativas

| Escenario | Usar NumPy | Usar Alternativa |
|-----------|------------|------------------|
| Prototipado rápido | ✅ | |
| Datos < 1GB en RAM | ✅ | |
| Necesitas autograd | | PyTorch/JAX |
| GPU computing | | CuPy/JAX |
| Datos > RAM | | Dask/Vaex |
| Streaming data | | Generadores Python |

---

# 8. Evaluación y Autoevaluación

## 8.1 Checklist de Competencias

Antes de avanzar al Módulo 02, verifica que puedes:

- [ ] Crear arrays con `np.zeros`, `np.ones`, `np.eye`, `np.random.default_rng`
- [ ] Predecir shapes resultantes de operaciones matriciales
- [ ] Identificar cuándo una operación produce vista vs copia
- [ ] Implementar normalización Z-score vectorizada
- [ ] Explicar por qué NumPy es más rápido que Python puro
- [ ] Usar `axis` y `keepdims` correctamente en agregaciones
- [ ] Convertir un DataFrame de Pandas a arrays NumPy listos para ML

## 8.2 Ejercicio Integrador

```python
"""
Ejercicio: Implementar un pipeline completo de preprocesamiento

Dado un dataset CSV con:
- Columnas numéricas con algunos NaN
- Una columna categórica 'target'

Producir:
- X: np.ndarray de shape (n_samples, n_features), dtype=float64, sin NaN
- y: np.ndarray de shape (n_samples,), dtype=int64, con target codificado

Restricciones:
- No usar sklearn
- Imputar con mediana
- Verificar todas las shapes con assert
"""

import pandas as pd
import numpy as np

def prepare_ml_data(csv_path: str, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline de preparación de datos para ML."""
    # Tu implementación aquí
    pass

# Tests (deben pasar)
# X, y = prepare_ml_data('data/test.csv', 'target')
# assert X.ndim == 2
# assert y.ndim == 1
# assert X.shape[0] == y.shape[0]
# assert X.dtype == np.float64
# assert not np.any(np.isnan(X))
```

---

## Referencias y Lecturas Adicionales

1. **Harris, C.R., et al. (2020).** "Array programming with NumPy." *Nature*, 585, 357-362.
2. **McKinney, W. (2017).** *Python for Data Analysis*, 2nd ed. O'Reilly Media.
3. **Van Der Walt, S., Colbert, S.C., & Varoquaux, G. (2011).** "The NumPy Array: A Structure for Efficient Numerical Computation." *Computing in Science & Engineering*, 13(2), 22-30.
4. **3Blue1Brown - Linear Algebra Series:** [youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

---

*Módulo desarrollado siguiendo el curriculum del MS-AI Pathway de la University of Colorado Boulder.*
