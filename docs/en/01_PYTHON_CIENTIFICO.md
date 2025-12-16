# Module 01 - Scientific Python + Pandas

> **🎯 Goal:** master Pandas for real data + NumPy for math
> **Phase:** 1 - Foundations | **Weeks 1–2**
> **Prerequisites:** basic Python (variables, functions, lists, loops)

---

<a id="m01-0"></a>

## 🧭 How to use this module (0→100 mode)

**Purpose:** move from “I know basic Python” to **working with real datasets and producing model-ready arrays** (what you’ll use throughout the Pathway).

### Learning objectives (measurable)

By the end of this module you can:

- **Use** Pandas to load, explore, and clean real datasets.
- **Convert** datasets to `np.ndarray` with ML-correct shapes (`X` and `y`).
- **Explain** what vectorization is and why NumPy avoids Python loops.
- **Diagnose** common shape pitfalls (`(n,)` vs `(n,1)`, silent broadcasting, views vs copies).

### Prerequisites

- Basic Python (loops, functions, lists, dictionaries).

Quick links:

- [GLOSSARY: NumPy](GLOSARIO.md#numpy)
- [GLOSSARY: Broadcasting](GLOSARIO.md#broadcasting)
- [GLOSSARY: Vectorization](GLOSARIO.md#vectorization)
- [RECURSOS.md](RECURSOS.md)

### Integration with Plan v4/v5

- Daily shapes drill: `../../study_tools/DRILL_DIMENSIONES_NUMPY.md`
- Error log: `../../study_tools/DIARIO_ERRORES.md`
- Evaluation (rubric): [study_tools/RUBRICA_v1.md](../../study_tools/RUBRICA_v1.md) (scope `M01` in `rubrica.csv`)
- Full protocol:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Resources (when to use them)

| Priority | Resource | When to use it in this module | Why |
|----------|---------|------------------------------|----------|
| **Required** | [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/) | Week 1, before starting `DataFrame/Series` and cleaning | Official reference for the canonical load/EDA/clean workflow |
| **Required** | [NumPy Documentation (absolute beginners)](https://numpy.org/doc/stable/user/absolute_beginners.html) | Week 2, when `ndarray`, `dtype`, `reshape`, `axis`, broadcasting show up | Official source for shape/axis questions |
| **Required** | `../../study_tools/DRILL_DIMENSIONES_NUMPY.md` | Any time you get a shape wrong / before the exit checklist | Automate shape intuition |
| **Recommended** | [Real Python - NumPy](https://realpython.com/numpy-tutorial/) | After completing broadcasting + vectorization (Week 2) | Consolidate idiomatic patterns with practical examples |
| **Optional** | [RECURSOS.md](RECURSOS.md) | At the end of the module (to plan reinforcement) | Choose deeper paths without losing focus |

### Exit criteria (when you can move on)

- You can prepare `X` and `y` from a CSV without dtype/shape errors.
- You can explain `axis=0` vs `axis=1` and predict shapes without running code.
- You can demonstrate vectorized speedup (benchmark) and justify it.

## 🧠 Why this module?

### The problem with pure Python for ML

```python
# ❌ This is NOT how you do Machine Learning
def dot_product_slow(a: list, b: list) -> float:  # Producto punto usando listas (enfoque lento)
    """Dot product with a Python loop - SLOW."""  # Docstring: explica que el cálculo usa un loop de Python
    result = 0  # Acumulador del resultado (se va sumando término por término)
    for i in range(len(a)):  # Recorre cada índice; cada iteración tiene overhead de Python
        result += a[i] * b[i]  # Multiplica elemento a elemento y acumula en el escalar result
    return result  # Devuelve el producto punto final como float (o int si las entradas lo son)

# For vectors of 1 million elements:
# Time: ~200ms
```

```python
# ✅ This IS how you do Machine Learning
import numpy as np  # NumPy ejecuta operaciones en C/BLAS, evitando el overhead de bucles de Python

def dot_product_fast(a: np.ndarray, b: np.ndarray) -> float:  # Producto punto con arrays NumPy (enfoque rápido)
    """Vectorized dot product - FAST."""  # Docstring: versión vectorizada, pensada para rendimiento
    return np.dot(a, b)  # Llama a una rutina optimizada (posible BLAS) para calcular el producto punto

# For vectors of 1 million elements:
# Time: ~2ms (100x faster)
```

### Connection to the Pathway

In the CU Boulder courses:

- **Supervised Learning:** matrix multiplications for regression
- **Unsupervised Learning:** PCA requires matrix decompositions
- **Deep Learning:** forward/backward passes are matrix operations

**Without NumPy, you can’t do efficient ML.**

---

## 📚 Module content

### Week 1: Pandas + basic NumPy

```
┌─────────────────────────────────────────────────────────────────┐
│  DAY 1: Pandas - DataFrame and Series                           │
│  DAY 2: Pandas - Loading CSVs (read_csv, head, info)            │
│  DAY 3: Pandas - Cleaning (dropna, fillna, dtypes)              │
│  DAY 4: NumPy - Arrays and dtypes                               │
│  DAY 5: NumPy - Indexing and slicing                            │
│  DAY 6: Pandas → NumPy (df.values, df.to_numpy())               │
└─────────────────────────────────────────────────────────────────┘
```

### Week 2: Vectorized NumPy

```
┌─────────────────────────────────────────────────────────────────┐
│  DAY 1: Broadcasting                                            │
│  DAY 2: Matrix product (@, np.dot, np.matmul) + reshape/flatten │
│  DAY 3: OOP for ML (v5.1): class Tensor (__init__, __add__, @)  │
│  DAY 4: Axis-based reductions and ops                           │
│  DAY 5: RNG and synthetic data generation                       │
│  DAY 6: Deliverable: Pandas → NumPy pipeline                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Key Concepts

### 0. Essential Pandas (Days 1–3)

#### Why Pandas?

In real-world ML, data comes in messy CSVs, not in perfect NumPy arrays. Before applying any algorithm you need to:

1. **Load data** from files
2. **Explore** structure and types
3. **Clean** missing values and errors
4. **Convert** to NumPy for the model

```python
import pandas as pd  # Pandas: lectura y manipulación de datos tabulares (DataFrame)
import numpy as np  # NumPy: base numérica (útil para conversiones y operaciones matemáticas)

# ========== DATA LOADING ==========
# Load CSV
df = pd.read_csv('data/iris.csv')  # Carga el CSV en un DataFrame (cada columna es una variable)

# First rows
print(df.head())  # Muestra las primeras filas para inspeccionar rápidamente los datos

# DataFrame info
print(df.info())  # Resume columnas, tipos (dtypes) y conteo de valores no nulos
#  Column         Non-Null Count  Dtype
# ---  ------         --------------  -----
#  0   sepal_length   150 non-null    float64
#  1   sepal_width    150 non-null    float64
#  2   petal_length   150 non-null    float64
#  3   petal_width    150 non-null    float64
#  4   species        150 non-null    object

# Basic statistics
print(df.describe())  # Estadísticas descriptivas (solo numéricas por defecto): mean, std, min, etc.
```

#### Data Cleaning

```python
import pandas as pd  # Pandas para construir y limpiar un DataFrame con datos faltantes

# Create DataFrame with messy data
df = pd.DataFrame({  # Crea un DataFrame desde un diccionario de listas (columnas)
    'edad': [25, 30, None, 45, 50],
    'salario': [50000, 60000, 70000, None, 90000],
    'ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Sevilla', None]
})  # None será interpretado como NaN en columnas numéricas

# ========== FIND NULLS ==========
print(df.isnull().sum())  # Cuenta cuántos NaN/None hay por columna
# edad       1
# salario    1
# ciudad     1

# ========== DROP ROWS WITH NULLS ==========
df_clean = df.dropna()  # Elimina filas que tengan al menos un valor nulo
print(f"Rows after dropna: {len(df_clean)}")  # Mide cuántas filas sobrevivieron al filtrado

# ========== FILL NULLS ==========
df_filled = df.copy()  # Crea una copia para no modificar el DataFrame original (evita efectos colaterales)
df_filled['edad'] = df_filled['edad'].fillna(df_filled['edad'].mean())  # Imputa edad faltante con la media
df_filled['salario'] = df_filled['salario'].fillna(df_filled['salario'].median())  # Imputa salario con la mediana (más robusta)
df_filled['ciudad'] = df_filled['ciudad'].fillna('Unknown')  # Imputa categorías faltantes con un marcador

print(df_filled)  # Imprime el DataFrame ya imputado para validar el resultado
```

#### Selection and Filtering

```python
import pandas as pd  # Pandas: DataFrames/Series, selección de columnas y filtrado de filas

df = pd.read_csv('data/iris.csv')  # Carga el dataset Iris desde un CSV en un DataFrame

# ========== SELECT COLUMNS ==========
# One column (Series)
sepal_length = df['sepal_length']  # Selecciona 1 columna => devuelve una Series (vector 1D con índice)

# Multiple columns (DataFrame)
features = df[['sepal_length', 'sepal_width']]  # Selecciona varias columnas => devuelve un DataFrame

# ========== FILTER ROWS ==========
# Simple condition
setosa = df[df['species'] == 'setosa']  # Filtra filas usando una máscara booleana (solo especie setosa)

# Multiple conditions
large_setosa = df[(df['species'] == 'setosa') & (df['sepal_length'] > 5)]  # Combina condiciones con & (usar paréntesis)

# ========== LOC and ILOC ==========
# loc: label-based
df.loc[0:5, ['sepal_length', 'species']]  # Selección por etiquetas (filas 0..5 y columnas específicas)

# iloc: position-based (like NumPy)
df.iloc[0:5, 0:2]  # Selección por posición (similar a slicing de NumPy)
```

#### From Pandas to NumPy (Day 6)

```python
import pandas as pd  # Pandas para preparar features/target en DataFrame/Series
import numpy as np  # NumPy para trabajar luego con ndarrays (matrices/vectores para ML)

df = pd.read_csv('data/iris.csv')  # Carga el CSV en memoria antes de separar X e y

# ========== SPLIT FEATURES AND TARGET ==========
# Features (X) - all numeric columns
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()  # Convierte columnas numéricas a ndarray 2D
print(f"X shape: {X.shape}")  # Verifica shape: (n_muestras, n_features)
print(f"X dtype: {X.dtype}")  # Verifica dtype: normalmente float64 tras lectura del CSV

# Target (y) - map categories to numbers
y = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).to_numpy()  # Codifica etiquetas (str) a enteros y convierte a ndarray 1D
print(f"y shape: {y.shape}")  # Verifica shape: (n,) (vector 1D), típico para targets

# ========== VERIFY ==========
print(f"Type of X: {type(X)}")  # Confirma que X es numpy.ndarray (requerido por la mayoría de modelos)
print(f"Type of y: {type(y)}")  # Confirma que y también es numpy.ndarray

# Now X and y are ready for ML algorithms
```

---
### 1. Arrays vs Lists

#### Intuition: “contiguous memory” (NumPy) vs “scattered boxes” (lists)

Think of a **Python list** as a row of little boxes that store **references** to objects; those objects can be **scattered** across memory. NumPy, instead, aims to represent an `ndarray` as a **contiguous block** of numbers of the same type (homogeneous). That decision enables:

- **Real vectorization:** inner loops in C (highly optimized).
- **Better CPU cache usage:** reading contiguous data is faster.
- **Less overhead:** there isn’t “one object per number”.

Mini mental diagram:

```
List (references):  [ * ] -> obj1   [ * ] -> obj2   [ * ] -> obj3   ...
                       |              |              |
                      mem@A          mem@Z          mem@K

NumPy (contiguous):   [ 1.0 ][ 2.0 ][ 3.0 ][ 4.0 ] ...  (same dtype)
```

```python
import numpy as np  # NumPy: arreglos homogéneos (ndarray) y operaciones vectorizadas eficientes

# Python list
lista = [1, 2, 3, 4, 5]  # Lista de Python: colección de referencias; no está optimizada para matemáticas masivas

# NumPy array
array = np.array([1, 2, 3, 4, 5])  # ndarray: bloque contiguo de números (mismo dtype), ideal para ML

# Key differences:
# 1. Homogeneous type (all elements share the same type)
# 2. Fixed-size buffer after creation
# 3. Vectorized operations
# 4. Contiguous storage in memory
```

### 2. Array creation

```python
import numpy as np  # Importa NumPy para crear arreglos y generar datos numéricos

# From a list
a = np.array([1, 2, 3])  # Convierte una lista de Python en un ndarray

# Special arrays
zeros = np.zeros((3, 4))        # Crea una matriz 3x4 llena de ceros (dtype float por defecto)
ones = np.ones((2, 3))          # Crea una matriz 2x3 llena de unos
identity = np.eye(4)            # Crea la matriz identidad 4x4 (unos en la diagonal)
random = np.random.randn(3, 3)  # Crea una matriz 3x3 con valores N(0,1) (distribución normal estándar)

# Sequences
rango = np.arange(0, 10, 2)     # Genera una secuencia: inicio=0, fin=10 (excl.), paso=2
linspace = np.linspace(0, 1, 5) # Genera 5 puntos igualmente espaciados entre 0 y 1 (incluye ambos extremos)

print(f"Shape of zeros: {zeros.shape}")  # Imprime la forma (filas, columnas) del array
print(f"Dtype of zeros: {zeros.dtype}")  # Imprime el tipo de dato: importante para precisión y memoria
```

### 3. Indexing and slicing

```python
import numpy as np  # NumPy para indexado multi-dimensional y slicing eficiente

# Create a 2D matrix
matrix = np.array([  # Construye una matriz 3x3 (ndarray 2D) a partir de una lista de listas
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])  # Cada sublista representa una fila

# Element access
print(matrix[0, 0])      # Acceso por (fila, columna): elemento [0,0]
print(matrix[1, 2])      # Acceso por (fila, columna): elemento [1,2]

# Slicing
print(matrix[0, :])      # Toma la fila 0 completa (':' significa "todo" en esa dimensión)
print(matrix[:, 1])      # Toma la columna 1 completa
print(matrix[0:2, 1:3])  # Submatriz: filas 0..1 y columnas 1..2 (el límite superior es exclusivo)

# Boolean indexing
print(matrix[matrix > 5])  # Filtra por condición y devuelve un vector 1D con los elementos que cumplen
```

### 4. Broadcasting

#### Worked Example: `(3, 1) + (1, 3)` step by step

Goal: understand **why** it works without loops.

1) Define two arrays with a “size-1” dimension:

- `A.shape = (3, 1)` (column)
- `B.shape = (1, 3)` (row)

2) Key rule: if in a dimension one of the sizes is `1`, NumPy can **“stretch”** that dimension to match the other.

3) Final result: both behave like `(3, 3)` and are added element by element.

```python
import numpy as np  # NumPy para demostrar broadcasting (ajuste de shapes sin bucles)

A = np.array([[1], [2], [3]])        # shape: (3, 1)  # Vector columna: 3 filas, 1 columna
B = np.array([[10, 20, 30]])         # shape: (1, 3)  # Vector fila: 1 fila, 3 columnas

# Broadcasting:
# A is repeated horizontally 3 times
# B is repeated vertically 3 times
C = A + B                             # shape: (3, 3)  # Resultado: matriz 3x3 por suma elemento a elemento

print("A:\n", A)  # Imprime A para ver su forma original (columna)
print("B:\n", B)  # Imprime B para ver su forma original (fila)
print("C = A + B:\n", C)  # Imprime el resultado tras broadcasting
```

```python
import numpy as np  # NumPy para ejemplos de broadcasting con escalares, vectores y matrices

# Broadcasting: operate on arrays of different shapes

# Scalar + Array
a = np.array([1, 2, 3])  # Vector 1D
print(a + 10)  # [11, 12, 13]  # El escalar 10 se "expande" a cada elemento del vector

# Vector + Matrix (automatic broadcasting)
matrix = np.array([  # Matriz 2D (2 filas, 3 columnas)
    [1, 2, 3],
    [4, 5, 6]
])
vector = np.array([10, 20, 30])  # Vector 1D con 3 elementos (coincide con el número de columnas)

# The vector is "expanded" to match the matrix
print(matrix + vector)  # Suma por filas: el vector se aplica a cada fila automáticamente
# [[11, 22, 33],
#  [14, 25, 36]]

# Broadcasting rule:
# Dimensions must be equal OR one of them must be 1
```

### 5. Aggregations and axes

#### Visualization: what does each axis “collapse”?

Rule of thumb:

- `axis=0` **collapses rows** → you get “one output per column”
- `axis=1` **collapses columns** → you get “one output per row”

Example with a `2x3` matrix:

```
X = [[1, 2, 3],
     [4, 5, 6]]

sum(axis=0) = [1+4, 2+5, 3+6] = [5, 7, 9]
sum(axis=1) = [1+2+3, 4+5+6] = [6, 15]
```

```python
import numpy as np  # NumPy para agregaciones globales y por eje (axis)

matrix = np.array([  # Crea una matriz 2x3 para ilustrar axis=0 y axis=1
    [1, 2, 3],
    [4, 5, 6]
])

# Global aggregations
print(np.sum(matrix))   # 21 (sum of all elements)  # Suma total de todos los elementos
print(np.mean(matrix))  # 3.5 (mean of all elements)  # Media global (sobre toda la matriz)
print(np.std(matrix))   # 1.707... (standard deviation)  # Desviación estándar global

# Axis-based aggregations
# axis=0: collapse rows (operate on columns)
print(np.sum(matrix, axis=0))  # [5, 7, 9]  # Una suma por columna (resultado tiene longitud = n_columnas)

# axis=1: collapse columns (operate on rows)
print(np.sum(matrix, axis=1))  # [6, 15]  # Una suma por fila (resultado tiene longitud = n_filas)

# Axis visualization:
# ┌─────────────┐
# │ axis=0 ↓    │
# │ [1, 2, 3]   │ → axis=1
# │ [4, 5, 6]   │ → axis=1
# └─────────────┘
```

### 6. Matrix operations

```python
import numpy as np  # NumPy para operaciones matriciales (element-wise y producto matricial)

A = np.array([[1, 2], [3, 4]])  # Matriz A (2x2)
B = np.array([[5, 6], [7, 8]])  # Matriz B (2x2)

# Element-wise operations
print(A + B)   # Addition  # Suma elemento a elemento (misma shape requerida)
print(A * B)   # Element-wise multiplication (Hadamard)  # Multiplicación elemento a elemento
print(A / B)   # Element-wise division  # División elemento a elemento

# Matrix product (what you'll use in ML)
print(A @ B)           # @ operator (Python 3.5+)  # Producto matricial: combina filas de A con columnas de B
print(np.matmul(A, B)) # matmul function  # Equivalente a @ para 2D (y generaliza a más dims)
print(np.dot(A, B))    # dot function  # Para 2D también hace producto matricial (ojo: en 1D cambia semántica)

# Result:
# [[19, 22],
#  [43, 50]]

# Transpose
print(A.T)  # Transpuesta: intercambia filas por columnas
# [[1, 3],
#  [2, 4]]
```

### 7. Vectorization: removing loops

```python
import numpy as np  # NumPy permite vectorizar: operar sobre arrays completos sin loops explícitos de Python

# ❌ WITH A LOOP (slow)
def normalize_loop(data: list) -> list:
    """Normalize data using a loop."""
    mean = sum(data) / len(data)  # Calcula la media con sum/len (operaciones Python)
    std = (sum((x - mean)**2 for x in data) / len(data)) ** 0.5  # Varianza -> raíz (std), todo en Python
    return [(x - mean) / std for x in data]  # Normaliza elemento a elemento con list comprehension

# ✅ VECTORIZED (fast)
def normalize_vectorized(data: np.ndarray) -> np.ndarray:
    """Normalize data using vectorization."""
    return (data - np.mean(data)) / np.std(data)  # Opera sobre el array completo; los bucles internos están en C

# Example
data = np.random.randn(1000000)  # Genera 1 millón de floats ~ N(0,1)

# The vectorized version is ~100x faster
normalized = normalize_vectorized(data)  # Ejecuta la versión vectorizada (en general mucho más rápida)
```

### 8. Universal functions (ufuncs)

```python
import numpy as np  # ufuncs: funciones universales (element-wise) optimizadas en C

x = np.array([1, 2, 3, 4, 5])  # Vector de entrada (ndarray 1D)

# Math functions (applied element by element)
print(np.exp(x))      # e^x  # Aplica exp a cada elemento
print(np.log(x))      # ln(x)  # Aplica log natural a cada elemento
print(np.sqrt(x))     # √x  # Raíz cuadrada por elemento
print(np.sin(x))      # sin(x)  # Seno por elemento

# Important for ML:
# Sigmoid: σ(x) = 1 / (1 + e^(-x))
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))  # Función logística: convierte valores reales a (0,1)

# ReLU: max(0, x)
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)  # ReLU: recorta negativos a 0, deja positivos igual

print(sigmoid(np.array([-2, -1, 0, 1, 2])))
# [0.119, 0.269, 0.5, 0.731, 0.881]
```

### 9. Reshape and shape manipulation

```python
import numpy as np  # Reshape: reorganiza la vista de los datos sin cambiar su contenido

# Create a 1D array
a = np.arange(12)  # [0, 1, 2, ..., 11]  # Crea un vector 1D con 12 enteros consecutivos

# Reshape to 2D
matrix = a.reshape(3, 4)  # Reinterpreta el vector como matriz 3x4 (mismo buffer de datos)
print(matrix.shape)  # (3, 4)  # Confirma la forma (filas, columnas)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

# Reshape to 3D
tensor = a.reshape(2, 2, 3)  # Reinterpreta como tensor 3D (2x2x3)
print(tensor.shape)  # (2, 2, 3)  # Verifica la forma del tensor

# Flatten: back to 1D
flat = matrix.flatten()  # Aplana a 1D (nota: flatten() devuelve una copia)
print(flat.shape)  # (12,)  # Forma del vector resultante

# -1 to infer a dimension automatically
auto = a.reshape(4, -1)  # (4, 3)  # -1 hace que NumPy infiera la dimensión faltante
auto = a.reshape(-1, 6)  # (2, 6)  # Aquí se reasigna 'auto': ahora infiere el número de filas
```

### 9.1 OOP for ML (v5.1): mini `Tensor` framework

**Practical goal:** before you get to neural networks (where you must handle `self`, state, and operator overloading), build a tiny abstraction that behaves like a simple “tensor”.

#### What you must internalize (no fluff)

- **Class vs instance:** the class is the blueprint; an instance is the actual object.
- **`self`:** the current instance; this is where state lives.
- **State:** values stored on the object (`self.data`, `self.shape`).
- **Operators:** `+` calls `__add__`, `@` calls `__matmul__`.

#### Deliverable (workshop)

- Implement a `Tensor` class that:
  - accepts a Python list or a `np.ndarray` in `__init__`
  - stores an internal `self.shape`
  - implements `__add__` and `__matmul__` using NumPy internally

#### Reference implementation

```python
import numpy as np  # NumPy: convert inputs to arrays and reuse fast vector/matrix ops
from typing import Union  # Union: allow multiple input types for the constructor

ArrayLike = Union[list, np.ndarray]  # Supported inputs: Python list or NumPy ndarray

class Tensor:  # Minimal container to practice OOP for ML (state + operators)
    def __init__(self, data: ArrayLike):  # Constructor: build internal state from raw input
        self.data = np.array(data, dtype=float)  # Normalize to float ndarray for consistent math
        self.shape = self.data.shape  # Keep shape as part of the object state (useful for debugging)

    def __add__(self, other: "Tensor") -> "Tensor":  # Define + as element-wise addition
        if not isinstance(other, Tensor):  # If the other operand is not a Tensor, we cannot add
            return NotImplemented  # Standard Python signal: this operation is not supported
        return Tensor(self.data + other.data)  # Return a NEW Tensor (do not mutate self)

    def __matmul__(self, other: "Tensor") -> "Tensor":  # Define @ as matrix multiplication
        if not isinstance(other, Tensor):  # Validate type to avoid confusing runtime errors
            return NotImplemented  # Let Python try a reflected operation if available
        return Tensor(self.data @ other.data)  # Use NumPy matmul and wrap the output

    def __repr__(self) -> str:  # Friendly representation when you print a Tensor
        return f"Tensor(shape={self.shape}, data={self.data})"  # Show shape + raw data for quick inspection
```

#### Exercises (with `assert`) — your minimum acceptable bar

```python
import numpy as np  # NumPy for array comparisons (allclose) and building reference arrays

# 1) State: shape must reflect the internal ndarray
t = Tensor([1, 2, 3])  # Build from a Python list
assert t.shape == (3,)  # Shape of a 1D vector

# 2) Addition: + must call __add__
a = Tensor([1, 2, 3])  # Tensor A
b = Tensor([10, 20, 30])  # Tensor B
c = a + b  # Should return a new Tensor
assert isinstance(c, Tensor)  # Result must be a Tensor
assert np.allclose(c.data, np.array([11.0, 22.0, 33.0]))  # Element-wise addition
assert c.shape == (3,)  # Shape should be preserved

# 3) Matmul: @ must call __matmul__
A = Tensor([[1, 2], [3, 4]])  # 2x2 matrix
x = Tensor([1, 1])  # Vector with shape (2,)
y = A @ x  # Matrix-vector product -> shape (2,)
assert np.allclose(y.data, np.array([3.0, 7.0]))  # [1,2]·[1,1]=3 and [3,4]·[1,1]=7
assert y.shape == (2,)  # Output shape check

# 4) Shape mismatch: must fail if dimensions are not compatible
try:  # We expect NumPy to raise ValueError on incompatible shapes
    _ = Tensor([[1, 2, 3], [4, 5, 6]]) @ Tensor([1, 2])  # (2,3) @ (2,) is invalid
    assert False  # If it did not fail, this test must fail
except ValueError:  # NumPy raises ValueError for matmul dimension mismatches
    pass  # Success: we expected this error
```

### 10. Random data generation

```python
import numpy as np  # Generación de datos aleatorios para simulación, pruebas y ML

# Set seed for reproducibility
np.random.seed(42)  # Fija la semilla: mismos "aleatorios" en cada ejecución (reproducibilidad)

# Uniform distribution [0, 1)
uniform = np.random.rand(3, 3)  # Matriz 3x3 con distribución uniforme en [0,1)

# Standard normal distribution (mean=0, std=1)
normal = np.random.randn(3, 3)  # Matriz 3x3 con N(0,1)

# Custom normal distribution
custom_normal = np.random.normal(loc=5, scale=2, size=(100,))  # Vector de 100 valores ~ N(5,2)

# Random integers
integers = np.random.randint(0, 10, size=(3, 3))  # Enteros aleatorios en [0,10) con shape 3x3

# Shuffle
data = np.arange(10)  # Vector [0..9]
np.random.shuffle(data)  # Mezcla IN-PLACE (modifica el array original)

# Sampling without replacement
sample = np.random.choice(data, size=5, replace=False)  # Muestra 5 elementos sin repetir (sin reemplazo)
```

---

## 📊 Type hints with NumPy

```python
import numpy as np  # NumPy para operaciones vectorizadas
from numpy.typing import NDArray  # NDArray permite anotar arrays con dtype específico para mypy

# Type hints for arrays
def normalize(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize an array of floats."""
    return (data - np.mean(data)) / np.std(data)  # Normaliza: (x - mean)/std; retorna ndarray float64

# Generic type hints
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot product of two vectors."""
    return float(np.dot(a, b))  # np.dot devuelve escalar NumPy; float() lo convierte a float de Python

# With mypy
# pip install numpy-stubs
```

---

## ⚡ Benchmark: List vs NumPy

```python
import numpy as np  # NumPy para benchmark de operaciones vectorizadas (np.dot)
import time  # time.time() para medir tiempos (segundos) de forma simple
from typing import List  # List[float] para anotar listas con floats

def benchmark_dot_product():
    """Compare list vs NumPy performance."""
    size = 1_000_000  # Tamaño del vector: 1 millón de elementos (lo bastante grande para ver diferencia)

    # Create data
    list_a: List[float] = [float(i) for i in range(size)]  # Lista A: floats creados en Python (lento pero simple)
    list_b: List[float] = [float(i) for i in range(size)]  # Lista B
    array_a = np.array(list_a)  # Convierte lista -> ndarray (contiguo) para cálculo rápido
    array_b = np.array(list_b)  # Convierte lista -> ndarray

    # List benchmark
    start = time.time()  # Marca tiempo inicial
    result_list = sum(a * b for a, b in zip(list_a, list_b))  # Producto punto con generador + sum (todo en Python)
    time_list = time.time() - start  # Tiempo total del cálculo con listas

    # NumPy benchmark
    start = time.time()  # Marca tiempo inicial
    result_numpy = np.dot(array_a, array_b)  # Producto punto vectorizado (rutina optimizada)
    time_numpy = time.time() - start  # Tiempo total del cálculo con NumPy

    print(f"List:   {time_list:.4f}s")  # Reporta segundos con listas
    print(f"NumPy:  {time_numpy:.4f}s")  # Reporta segundos con NumPy
    print(f"Speedup: {time_list/time_numpy:.1f}x")  # Factor de aceleración (list_time / numpy_time)

    # Verify same results
    assert abs(result_list - result_numpy) < 1e-6  # Verifica que ambos métodos dan casi el mismo resultado

if __name__ == "__main__":
    benchmark_dot_product()  # Ejecuta el benchmark al correr el script directamente

# Typical output:
# List:   0.1523s
# NumPy:  0.0015s
# Speedup: 101.5x
```

---

## 🎯 Topic-based progressive exercises + solutions

Rules:

- **Try first** without looking at solutions.
- **Suggested timebox:** 10–15 min per exercise before checking.
- **Minimum success:** your solution must pass the `assert` checks.

---

### Exercise 1.1: Pandas - DataFrame and Series

#### Prompt

1) **Basic**

- Create a `DataFrame` called `df` with columns `edad`, `salario`, `ciudad` (5 rows).
- Extract `salario` as a `Series` and compute its mean.

2) **Intermediate**

- Create a new column `salario_k` with `salario / 1000`.
- Sort the `DataFrame` by `salario` descending.

3) **Advanced**

- Compute, per `ciudad`, the mean of `salario` and the row count (in a single table).

#### Solution

```python
import pandas as pd  # Pandas para crear DataFrames, operar columnas y hacer groupby/agg

df = pd.DataFrame(  # Construye un DataFrame a partir de un diccionario (columnas -> listas)
    {
        "edad": [25, 30, 30, 45, 50],  # Columna numérica: edades (5 filas)
        "salario": [50000, 60000, 61000, 80000, 90000],  # Columna numérica: salarios
        "ciudad": ["Madrid", "Barcelona", "Madrid", "Sevilla", "Madrid"],  # Columna categórica: ciudad
    }
)

salario = df["salario"]  # Selecciona una columna => devuelve una Series
mean_salary = salario.mean()  # Calcula la media de la Series (devuelve float)

df["salario_k"] = df["salario"] / 1000  # Crea nueva columna con salario en miles (vectorizado)
df_sorted = df.sort_values("salario", ascending=False)  # Ordena el DataFrame por salario (descendente)

summary = (  # Pipeline: agrupa por ciudad y calcula métricas agregadas en una sola tabla
    df.groupby("ciudad", as_index=False)  # Agrupa filas por ciudad; as_index=False mantiene 'ciudad' como columna
    .agg(salario_mean=("salario", "mean"), n=("salario", "size"))  # media de salario y conteo de filas por grupo
    .sort_values("salario_mean", ascending=False)  # Ordena por la media (mayor a menor)
)

assert isinstance(mean_salary, float)  # Verifica que el resultado de mean() sea un float
assert "salario_k" in df.columns  # Verifica que la nueva columna fue creada
assert df_sorted.iloc[0]["salario"] == df["salario"].max()  # El primer salario tras ordenar debe ser el máximo
assert set(summary.columns) == {"ciudad", "salario_mean", "n"}  # Verifica columnas esperadas del resumen
```

---

### Exercise 1.2: Pandas - Cleaning (missing values, dtypes, duplicates)

#### Prompt

1) **Basic**

- Create a `DataFrame` with missing values in `edad` and `salario`.
- Count nulls per column.

2) **Intermediate**

- Fill `edad` with the mean.
- Fill `salario` with the median.

3) **Advanced**

- Add a duplicate row on purpose.
- Drop duplicates.
- Convert `edad` to `int` **after** imputing.

#### Solution

```python
import pandas as pd  # Pandas para crear DataFrames y aplicar operaciones de limpieza
import numpy as np  # NumPy para validar dtypes (por ejemplo np.int64)

df = pd.DataFrame(  # Crea un DataFrame con valores faltantes (None -> NaN)
    {
        "edad": [25, None, 30, 45, None],  # 'edad' tiene 2 faltantes (None)
        "salario": [50000, 60000, None, 80000, 90000],  # 'salario' tiene 1 faltante
        "ciudad": ["Madrid", "Barcelona", "Madrid", "Sevilla", "Madrid"],  # Columna categórica sin nulos
    }
)

nulls = df.isnull().sum()  # Cuenta nulos por columna (para verificar antes de imputar)

df2 = df.copy()  # Copia para no modificar el DataFrame original
df2["edad"] = df2["edad"].fillna(df2["edad"].mean())  # Imputa 'edad' con la media (tras ignorar NaN)
df2["salario"] = df2["salario"].fillna(df2["salario"].median())  # Imputa 'salario' con la mediana (robusta)

df3 = pd.concat([df2, df2.iloc[[0]]], ignore_index=True)  # Agrega una fila duplicada (la fila 0) a propósito
df3 = df3.drop_duplicates()  # Elimina duplicados (por todas las columnas) y vuelve a dejar el dataset limpio
df3["edad"] = df3["edad"].round().astype(int)  # Convierte a int DESPUÉS de imputar (si no, fallaría por NaN)

assert nulls["edad"] == 2  # Verifica que realmente había 2 nulos en 'edad'
assert nulls["salario"] == 1  # Verifica que realmente había 1 nulo en 'salario'
assert df2.isnull().sum().sum() == 0  # Tras imputar, no debe quedar ningún nulo
assert len(df3) == len(df2)  # Tras agregar duplicado y quitarlo, el tamaño debe volver al original
assert df3["edad"].dtype == np.int64 or str(df3["edad"].dtype).startswith("int")  # Verifica dtype entero (varía por plataforma)
```

---

### Exercise 1.3: Pandas - Selection and filtering (`loc`, `iloc`, boolean masks)

#### Prompt

Use this `DataFrame`:

```python
import pandas as pd  # Pandas para construir un DataFrame de ejemplo (enunciado)

df = pd.DataFrame(  # Crea un DataFrame pequeño para practicar selección y filtrado
    {
        "sepal_length": [5.1, 4.9, 5.8, 6.0, 5.4],  # Columna numérica: largo del sépalo
        "sepal_width": [3.5, 3.0, 2.7, 2.2, 3.9],  # Columna numérica: ancho del sépalo
        "species": ["setosa", "setosa", "versicolor", "virginica", "setosa"],  # Columna categórica: especie
    }
)
```

1) **Basic**

- Extract columns `sepal_length` and `species`.

2) **Intermediate**

- Filter rows where `species == "setosa"` and `sepal_length > 5.0`.

3) **Advanced**

- Compute the mean of `sepal_length` by `species`.
- Return it sorted from highest to lowest.

#### Solution

```python
import pandas as pd  # Pandas para selección de columnas, filtros booleanos y agregaciones

df = pd.DataFrame(  # Reconstruye el DataFrame del enunciado para aplicar las operaciones
    {
        "sepal_length": [5.1, 4.9, 5.8, 6.0, 5.4],
        "sepal_width": [3.5, 3.0, 2.7, 2.2, 3.9],
        "species": ["setosa", "setosa", "versicolor", "virginica", "setosa"],
    }
)

subset = df[["sepal_length", "species"]]  # Selecciona dos columnas => DataFrame con esas columnas

filtered = df[(df["species"] == "setosa") & (df["sepal_length"] > 5.0)]  # Máscara booleana con 2 condiciones

means = (  # Pipeline: agrupa por especie y calcula media de sepal_length por grupo
    df.groupby("species", as_index=False)  # Agrupa filas por especie; as_index=False deja 'species' como columna
    .agg(sepal_length_mean=("sepal_length", "mean"))  # Calcula la media de sepal_length por especie
    .sort_values("sepal_length_mean", ascending=False)  # Ordena de mayor a menor media
)

assert list(subset.columns) == ["sepal_length", "species"]  # Verifica que el subset tiene las columnas correctas
assert (filtered["species"] == "setosa").all()  # En el filtrado, todas las filas deben ser setosa
assert (filtered["sepal_length"] > 5.0).all()  # En el filtrado, todas las filas deben cumplir sepal_length > 5
assert means.iloc[0]["sepal_length_mean"] >= means.iloc[-1]["sepal_length_mean"]  # Verifica que está ordenado desc
```

---

### Exercise 1.4: NumPy - Arrays and `dtype`

#### Prompt

1) **Basic**

- Create:
  - a vector of 10 zeros
  - a `3x3` matrix of ones
  - a `4x4` identity matrix

2) **Intermediate**

- Create `v = np.array([1, 2, 3])`.
- Convert `v` to `float64`.
- Verify that `v / 2` produces floats.

3) **Advanced**

- Reproduce the common dtype bug with in-place division:
  - create `a = np.array([1, 2, 3])`
  - apply `a /= 2`
  - explain the result with an expected `assert`

#### Solution

```python
import numpy as np  # NumPy para crear arrays con dtype controlado y demostrar bugs comunes

z = np.zeros(10)  # Vector de 10 ceros (shape: (10,))
ones = np.ones((3, 3))  # Matriz 3x3 llena de unos
I = np.eye(4)  # Matriz identidad 4x4

v = np.array([1, 2, 3])  # Array de enteros (dtype int por defecto)
v_f = v.astype(np.float64)  # Convierte a float64 para asegurar divisiones en punto flotante
half = v_f / 2  # División "normal" => produce floats (sin truncar)

a = np.array([1, 2, 3])  # Array int
a /= 2  # División IN-PLACE sobre int => trunca (bug común si esperas floats)

assert z.shape == (10,)  # Verifica forma del vector
assert ones.shape == (3, 3)  # Verifica forma de la matriz de unos
assert I.shape == (4, 4)  # Verifica forma de la identidad
assert v_f.dtype == np.float64  # Verifica conversión de dtype
assert half.dtype == np.float64  # Verifica que la división produce floats
assert np.array_equal(a, np.array([0, 1, 1]))  # Verifica el truncamiento (1/2=0, 2/2=1, 3/2=1)
```

---

### Exercise 1.5: NumPy - Indexing and slicing

#### Prompt

Given:

```python
import numpy as np  # NumPy para crear arrays y practicar indexing/slicing
X = np.arange(20).reshape(4, 5)  # Crea una matriz 4x5 con valores 0..19 (útil para verificar índices)
```

1) **Basic**

- Extract the element at row 2, column 3.

2) **Intermediate**

- Extract:
  - the full row 1
  - the full column 4
  - the submatrix rows 1–2, columns 2–4

3) **Advanced**

- Use boolean indexing to extract elements greater than 10.
- Verify that every element in the result satisfies `> 10`.

#### Solution

```python
import numpy as np  # NumPy para construir la matriz y extraer slices con indexado

X = np.arange(20).reshape(4, 5)  # Matriz 4x5 con números consecutivos

e = X[2, 3]  # Elemento en fila 2, columna 3 (indexado base 0)

row1 = X[1, :]  # Fila 1 completa (vector de longitud 5)
col4 = X[:, 4]  # Columna 4 completa (vector de longitud 4)
sub = X[1:3, 2:5]  # Submatriz: filas 1..2 y columnas 2..4 (límites superiores excluyentes)

gt10 = X[X > 10]  # Indexado booleano: extrae todos los elementos mayores que 10 (resultado 1D)

assert e == 13  # Verifica el valor esperado en esa posición
assert row1.shape == (5,)  # Verifica que row1 sea un vector de 5 elementos
assert col4.shape == (4,)  # Verifica que col4 sea un vector de 4 elementos
assert sub.shape == (2, 3)  # Verifica shape de la submatriz (2 filas, 3 columnas)
assert (gt10 > 10).all()  # Verifica que todos los elementos del resultado cumplan la condición
```

---

### Exercise 1.6: NumPy - Broadcasting

#### Prompt

1) **Basic**

- Without loops, add 100 to each element of a `3x3` matrix.

2) **Intermediate**

- Given `A` shape `(4, 3)` and `v` shape `(3,)`, add `v` to every row.

3) **Advanced**

- Given `X` shape `(n, d)`, normalize per column: `X_norm = (X - mean) / (std + eps)`.
- **Important:** output must keep shape `(n, d)`.

#### Solution

```python
import numpy as np  # NumPy para crear arrays y aplicar broadcasting/normalización sin bucles

M = np.arange(9).reshape(3, 3)  # Matriz 3x3 con valores 0..8
M2 = M + 100  # Suma escalar: 100 se "expande" a cada elemento (broadcasting)

A = np.arange(12).reshape(4, 3)  # Matriz 4x3
v = np.array([10, 20, 30])  # Vector (3,) compatible con las columnas de A
B = A + v  # Broadcasting por filas: v se suma a cada fila de A

X = np.random.randn(100, 5)  # Dataset sintético: 100 muestras, 5 features
eps = 1e-8  # Epsilon para evitar división por cero si alguna std fuera 0
mean = X.mean(axis=0)  # Media por columna (shape: (5,))
std = X.std(axis=0)  # Desviación estándar por columna (shape: (5,))
X_norm = (X - mean) / (std + eps)  # Normalización por feature; mantiene shape (100,5)

assert M2.shape == (3, 3)  # Verifica que la suma escalar no cambie la forma
assert B.shape == (4, 3)  # Verifica que el broadcasting por filas preserve la forma de A
assert X_norm.shape == (100, 5)  # Verifica que la normalización mantenga (n,d)
```

---

### Exercise 1.7: NumPy - Matrix product (`@`, `np.dot`, `np.matmul`)

#### Prompt

1) **Basic**

- Compute `A @ B` with:
  - `A` shape `(2, 3)`
  - `B` shape `(3, 2)`

2) **Intermediate**

- Demonstrate the difference between:
  - element-wise multiplication `A * B`
  - matrix product `A @ B`
  using `2x2` matrices.

3) **Advanced**

- Implement a linear prediction `y_hat = X @ w + b` with:
  - `X` shape `(n, d)`
  - `w` shape `(d,)`
  - `b` scalar
- Verify the shape of `y_hat`.

#### Solution

```python
import numpy as np  # NumPy para producto matricial (@) y para crear datos de ejemplo

A = np.array([[1, 2, 3], [4, 5, 6]])  # A tiene shape (2,3)
B = np.array([[1, 0], [0, 1], [1, 1]])  # B tiene shape (3,2)
C = A @ B  # Producto matricial => shape (2,2)

U = np.array([[1, 2], [3, 4]])  # Matriz 2x2
V = np.array([[10, 20], [30, 40]])  # Matriz 2x2
hadamard = U * V  # Multiplicación elemento a elemento (Hadamard), NO es producto matricial
matmul = U @ V  # Producto matricial real (combinación fila-columna)

X = np.random.randn(50, 3)  # Matriz de features (n=50, d=3)
w = np.array([0.1, -0.2, 0.3])  # Vector de pesos (d,)
b = 0.5  # Bias escalar
y_hat = X @ w + b  # Predicción lineal: resultado shape (n,) cuando w es (d,)

assert C.shape == (2, 2)  # (2,3) @ (3,2) => (2,2)
assert hadamard.shape == (2, 2)  # Element-wise conserva shape
assert matmul.shape == (2, 2)  # Producto matricial 2x2 -> 2x2
assert y_hat.shape == (50,)  # (50,3) @ (3,) => (50,)
```

---

### Exercise 1.8: NumPy - `reshape`, `flatten`, `transpose`

#### Prompt

1) **Basic**

- Create `a = np.arange(12)` and reshape it to a `(3, 4)` matrix.

2) **Intermediate**

- Transpose that matrix and verify the shape.

3) **Advanced**

- Reshape the same data to a tensor `(2, 2, 3)`.
- Return to 1D and verify you recover 12 elements.

#### Solution

```python
import numpy as np  # NumPy para reshape/transpose y verificación de shapes

a = np.arange(12)  # Vector 1D con 12 elementos
M = a.reshape(3, 4)  # Reinterpreta a como matriz 3x4
MT = M.T  # Transpuesta: cambia (3,4) -> (4,3)

T = a.reshape(2, 2, 3)  # Reinterpreta el mismo buffer como tensor 3D (2x2x3)
flat = T.reshape(-1)  # Regresa a 1D usando -1 para inferir el tamaño (debe ser 12)

assert M.shape == (3, 4)  # Verifica forma de la matriz
assert MT.shape == (4, 3)  # Verifica forma tras transponer
assert T.shape == (2, 2, 3)  # Verifica forma del tensor
assert flat.shape == (12,)  # Verifica que volvimos a 12 elementos
assert np.array_equal(flat, a)  # Verifica que el contenido se mantiene al re-reshape
```

---

### Exercise 1.9: NumPy - Aggregations and `axis`

#### Prompt

Let:

```python
import numpy as np  # NumPy para agregaciones y manejo de ejes (axis)
X = np.array([[1, 2, 3], [4, 5, 6]])  # Matriz 2x3 de ejemplo
```

1) **Basic**

- Compute `X.sum()` and verify the result.

2) **Intermediate**

- Compute `X.sum(axis=0)` and `X.sum(axis=1)`.
- Predict the shapes before running.

3) **Advanced**

- Compute the per-column mean with `keepdims=True`.
- Subtract it from `X` and verify the output shape.

#### Solution

```python
import numpy as np  # NumPy para sumas/medias globales y por eje (axis)

X = np.array([[1, 2, 3], [4, 5, 6]])  # Matriz 2x3

s_all = X.sum()  # Suma global (todos los elementos) => escalar
s0 = X.sum(axis=0)  # Suma por columnas => shape (3,)
s1 = X.sum(axis=1)  # Suma por filas => shape (2,)

mu = X.mean(axis=0, keepdims=True)  # Media por columna manteniendo dimensión => shape (1,3)
X_centered = X - mu  # Centrado por columnas; broadcasting de (1,3) sobre (2,3)

assert s_all == 21  # 1+2+3+4+5+6 = 21
assert s0.shape == (3,)  # Tres columnas
assert s1.shape == (2,)  # Dos filas
assert mu.shape == (1, 3)  # keepdims=True mantiene la dimensión de filas
assert X_centered.shape == (2, 3)  # Centrado mantiene shape
assert np.allclose(X_centered.mean(axis=0), 0.0)  # Media por columna ≈ 0 tras centrar
```

---

### Exercise 1.10: NumPy - `random` and synthetic data

#### Prompt

1) **Basic**

- Set a seed and generate 5 values with `np.random.randn`.

2) **Intermediate**

- Generate a synthetic regression dataset:
  - `X` shape `(200, 2)`
  - `w_true` shape `(2,)`
  - `y = X @ w_true + noise`

3) **Advanced**

- Standardize `X` per column (approximately `mean=0`, `std=1`).
- Verify with `np.allclose` (reasonable tolerance).

#### Solution

```python
import numpy as np  # NumPy para generar datos aleatorios y estandarizarlos por columna

np.random.seed(42)  # Semilla para reproducibilidad
z = np.random.randn(5)  # 5 valores ~ N(0,1)

n = 200  # Número de muestras
X = np.random.randn(n, 2)  # Matriz de features (200x2)
w_true = np.array([1.5, -0.7])  # Pesos reales (2,)
noise = 0.1 * np.random.randn(n)  # Ruido gaussiano (escala 0.1)
y = X @ w_true + noise  # Target: combinación lineal + ruido

eps = 1e-8  # Epsilon para evitar división por cero
X_mean = X.mean(axis=0)  # Media por columna
X_std = X.std(axis=0)  # Std por columna
Xz = (X - X_mean) / (X_std + eps)  # Estandariza por columna (aprox mean=0, std=1)

assert z.shape == (5,)  # Verifica tamaño del vector aleatorio
assert X.shape == (200, 2)  # Verifica shape de features
assert w_true.shape == (2,)  # Verifica shape de pesos
assert y.shape == (200,)  # Verifica shape del target
assert np.allclose(Xz.mean(axis=0), np.zeros(2), atol=1e-7)  # Media ~ 0 por columna
assert np.allclose(Xz.std(axis=0), np.ones(2), atol=1e-6)  # Std ~ 1 por columna
```

---

### (Bonus) Exercise 1.11: Vectorization + activation functions (mastery)

#### Prompt

1) **Vectorization**

- Implement Euclidean distance without loops.

2) **Activations**

- Implement:
  - `sigmoid`
  - `relu`
  - `softmax` (numerically stable)

#### Solution

```python
import numpy as np  # NumPy para vectorización y activaciones (sigmoid/relu/softmax)

def euclidean_distance_vectorized(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b  # Diferencia vectorizada elemento a elemento
    return float(np.sqrt(np.sum(diff * diff)))  # sqrt(sum((a-b)^2)); float() convierte escalar NumPy a float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))  # Sigmoid: mapea reales a (0,1)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)  # ReLU: 0 si x<0, x si x>=0


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)  # Asegura que x sea ndarray
    x_shift = x - np.max(x)  # Estabilización numérica: evita overflow en exp
    exps = np.exp(x_shift)  # Exponenciales elemento a elemento
    return exps / np.sum(exps)  # Normaliza para que sumen 1 (distribución de prob.)


a = np.array([1.0, 2.0, 3.0])  # Vector a
b = np.array([1.0, 1.0, 1.0])  # Vector b
d = euclidean_distance_vectorized(a, b)  # Distancia euclidiana entre a y b

assert np.isclose(d, np.sqrt(0**2 + 1**2 + 2**2))  # Verifica distancia: sqrt((0)^2+(1)^2+(2)^2)
assert np.isclose(sigmoid(np.array([0.0]))[0], 0.5)  # Sigmoid(0)=0.5
assert relu(np.array([-5.0, 5.0])).tolist() == [0.0, 5.0]  # ReLU recorta negativos
assert np.isclose(softmax(np.array([1.0, 2.0, 3.0])).sum(), 1.0)  # Softmax debe sumar 1
```

---

## 📦 Module deliverable

### Script: `benchmark_vectorization.py`

```python
"""
Benchmark: vector operations List vs NumPy

This script compares the performance of common operations
using pure Python lists vs NumPy arrays.

Operations compared:
1. Dot product
2. Normalization
3. Euclidean distance
4. Matrix sum

Author: [Your name]
Date: [Date]
"""

import numpy as np  # NumPy para operaciones vectorizadas y generación de arrays de prueba
import time  # time.time() para medir tiempos (benchmark) de forma simple
from typing import List, Tuple, Callable  # Tipos para anotar listas/tuplas y callables (mejor legibilidad)
from dataclasses import dataclass  # dataclass para agrupar resultados del benchmark de forma clara


@dataclass  # Genera automáticamente __init__, __repr__, etc. para este contenedor de resultados
class BenchmarkResult:
    """Benchmark result."""
    operation: str  # Nombre de la operación medida (p. ej., "Dot Product")
    time_list: float  # Tiempo promedio por iteración usando listas (segundos)
    time_numpy: float  # Tiempo promedio por iteración usando NumPy (segundos)
    speedup: float  # Aceleración: time_list / time_numpy


def benchmark(
    func_list: Callable,
    func_numpy: Callable,
    args_list: Tuple,
    args_numpy: Tuple,
    operation_name: str,
    iterations: int = 100
) -> BenchmarkResult:
    """Run a comparative benchmark."""

    # List benchmark
    start = time.time()  # Inicio del cronómetro para la versión con listas
    for _ in range(iterations):  # Repite varias veces para promediar y reducir ruido
        func_list(*args_list)  # Llama a la función "lenta" con argumentos de tipo lista
    time_list = (time.time() - start) / iterations  # Tiempo promedio por iteración

    # NumPy benchmark
    start = time.time()  # Inicio del cronómetro para la versión con NumPy
    for _ in range(iterations):  # Misma cantidad de iteraciones para comparar justo
        func_numpy(*args_numpy)  # Llama a la función vectorizada con argumentos ndarray
    time_numpy = (time.time() - start) / iterations  # Tiempo promedio por iteración

    return BenchmarkResult(  # Empaqueta y devuelve el resultado del benchmark
        operation=operation_name,  # Nombre de la operación (texto)
        time_list=time_list,  # Tiempo (listas)
        time_numpy=time_numpy,  # Tiempo (NumPy)
        speedup=time_list / time_numpy  # Cuántas veces es más rápido NumPy
    )


# === IMPLEMENT YOUR FUNCTIONS HERE ===

def dot_product_list(a: List[float], b: List[float]) -> float:
    """Dot product with lists."""
    # TODO: Implement
    pass  # Placeholder: aquí debes implementar el producto punto con bucle/zip/sum


def dot_product_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product with NumPy."""
    # TODO: Implement
    pass  # Placeholder: aquí debes implementar np.dot(a, b)


def normalize_list(data: List[float]) -> List[float]:
    """Normalize with lists."""
    # TODO: Implement
    pass  # Placeholder: normalización con listas (calcular mean/std y aplicar (x-mean)/std)


def normalize_numpy(data: np.ndarray) -> np.ndarray:
    """Normalize with NumPy."""
    # TODO: Implement
    pass  # Placeholder: normalización vectorizada con np.mean/np.std


def euclidean_distance_list(a: List[float], b: List[float]) -> float:
    """Euclidean distance with lists."""
    # TODO: Implement
    pass  # Placeholder: distancia euclidiana con listas (sqrt(sum((ai-bi)^2)))


def euclidean_distance_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance with NumPy."""
    # TODO: Implement
    pass  # Placeholder: distancia vectorizada con np.sqrt(np.sum((a-b)**2))


def matrix_sum_list(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix sum with lists."""
    # TODO: Implement
    pass  # Placeholder: suma elemento a elemento con listas anidadas


def matrix_sum_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix sum with NumPy."""
    # TODO: Implement
    pass  # Placeholder: suma directa A + B


def main():
    """Run all benchmarks."""
    size = 10000  # Tamaño de los vectores para las operaciones 1D (ajusta para tu máquina)

    # Create data de prueba
    list_a = [float(i) for i in range(size)]  # Lista A de floats (creación en Python)
    list_b = [float(i) for i in range(size)]  # Lista B
    array_a = np.array(list_a)  # Convierte a ndarray para cálculo vectorizado
    array_b = np.array(list_b)  # Convierte a ndarray

    matrix_size = 100  # Tamaño de matrices cuadradas para la prueba de suma de matrices
    list_matrix_a = [[float(i*j) for j in range(matrix_size)]  # Matriz A como listas anidadas
                     for i in range(matrix_size)]
    list_matrix_b = [[float(i+j) for j in range(matrix_size)]  # Matriz B como listas anidadas
                     for i in range(matrix_size)]
    array_matrix_a = np.array(list_matrix_a)  # Conversión a ndarray 2D
    array_matrix_b = np.array(list_matrix_b)  # Conversión a ndarray 2D

    # Run benchmarks
    results = []  # Lista donde acumulamos BenchmarkResult de cada operación

    results.append(benchmark(  # Benchmark para producto punto
        dot_product_list, dot_product_numpy,
        (list_a, list_b), (array_a, array_b),
        "Dot Product"
    ))

    results.append(benchmark(  # Benchmark para normalización
        normalize_list, normalize_numpy,
        (list_a,), (array_a,),
        "Normalization"
    ))

    results.append(benchmark(  # Benchmark para distancia euclidiana
        euclidean_distance_list, euclidean_distance_numpy,
        (list_a, list_b), (array_a, array_b),
        "Euclidean Distance"
    ))

    results.append(benchmark(  # Benchmark para suma de matrices
        matrix_sum_list, matrix_sum_numpy,
        (list_matrix_a, list_matrix_b), (array_matrix_a, array_matrix_b),
        "Matrix Sum"
    ))

    # Display results
    print("\n" + "="*60)  # Línea separadora
    print("BENCHMARK: List vs NumPy")  # Título del reporte
    print("="*60)  # Línea separadora
    print(f"{'Operation':<25} {'List (ms)':<12} {'NumPy (ms)':<12} {'Speedup':<10}")  # Encabezado de tabla
    print("-"*60)  # Separador de encabezado

    for r in results:  # Recorre cada resultado para imprimir una fila
        print(f"{r.operation:<25} {r.time_list*1000:<12.4f} {r.time_numpy*1000:<12.4f} {r.speedup:<10.1f}x")  # Convierte a ms

    print("="*60)  # Cierre de tabla
    print(f"\nAverage speedup: {sum(r.speedup for r in results)/len(results):.1f}x")  # Promedio de speedup


if __name__ == "__main__":
    main()

```

---

## 🐛 Debugging NumPy: Errors That Will Waste Your Time (v3.2)

> ⚠️ **CRITICAL:** These 5 errors are the most frequent in Phases 1 and 2. Fixing them now prevents hours of frustration.

### Error 1: Shape Mismatch - `(5,)` vs `(5,1)`

```python
import numpy as np  # NumPy para crear arrays y diagnosticar problemas de shape (dimensiones)

# PROBLEM: Vector 1D vs column vector
v1 = np.array([1, 2, 3, 4, 5])      # Shape: (5,) - Vector 1D  # Vector 1D: una sola dimensión
v2 = np.array([[1], [2], [3], [4], [5]])  # Shape: (5, 1) - column vector  # Matriz columna: 5 filas, 1 columna

print(f"v1.shape: {v1.shape}")  # (5,)  # Confirma que es 1D (no tiene segunda dimensión)
print(f"v2.shape: {v2.shape}")  # (5, 1)  # Confirma que es 2D (columna)

# THIS FAILS in Linear Regression:
# If X has shape (100, 5) and theta has shape (5,), the result is (100,)
# If theta has shape (5, 1), the result is (100, 1)

# SOLUTION: Use reshape or keepdims
v1_column = v1.reshape(-1, 1)  # (5,) → (5, 1)  # reshape(-1,1) fuerza un vector columna; -1 infiere filas
v1_column_alt = v1[:, np.newaxis]  # Alternative  # Alternativa equivalente: inserta un eje nuevo

# RULE: For ML, feature vectors should be (n, 1), not (n,)
```

### Error 2: Incorrect silent broadcasting

```python
import numpy as np  # NumPy para trabajar con arrays aleatorios y agregaciones por eje (axis)

# PROBLEM: Broadcasting does not fail, but gives incorrect results  # PROBLEMA: no falla, pero produce resultados incorrectos
X = np.random.randn(100, 5)  # 100 samples, 5 features  # Dataset: 100 filas (muestras), 5 columnas (features)
mean_wrong = np.mean(X)      # INCORRECT! Mean of the ENTIRE array  # Devuelve un escalar (pierdes info por feature)
mean_correct = np.mean(X, axis=0)  # Correct: mean per feature (shape: (5,))  # Media por columna (una por feature)

print(f"mean_wrong shape: {np.array(mean_wrong).shape}")  # () - scalar  # Shape vacío: es un escalar
print(f"mean_correct shape: {mean_correct.shape}")  # (5,)  # Vector de 5 medias (una por columna)

# RULE: Always specify axis= in aggregations  # REGLA: especifica siempre axis= para evitar errores silenciosos
# axis=0: operates over rows (result per column)  # axis=0: reduce filas -> obtienes un valor por columna (por feature)
# axis=1: operates over columns (result per row)  # axis=1: reduce columnas -> obtienes un valor por fila (por muestra)
```

### Error 3: Unexpected in-place modification

```python
import numpy as np  # NumPy para demostrar que los slices suelen ser vistas (views) y no copias

# PROBLEM: NumPy slices are VIEWS, not copies  # PROBLEMA: los slices comparten memoria con el original
original = np.array([1, 2, 3, 4, 5])  # Array original
slice_view = original[1:4]  # Slice: devuelve una VIEW (apunta al mismo buffer)
slice_view[0] = 999  # Modifica la vista => también modifica el original

print(original)  # [1, 999, 3, 4, 5] - ORIGINAL MODIFIED!  # Evidencia del bug: cambió el original

# SOLUTION: Use .copy() explicitly  # SOLUCIÓN: crea una copia real cuando necesites aislar cambios
original = np.array([1, 2, 3, 4, 5])  # Reinicia el original para mostrar el caso correcto
slice_copy = original[1:4].copy()  # copy(): ahora sí crea un buffer independiente
slice_copy[0] = 999  # Modifica la copia => NO afecta al original

print(original)  # [1, 2, 3, 4, 5] - Original intact  # Confirma que el original quedó intacto
```

### Error 4: Division by zero during normalization

```python
import numpy as np  # NumPy para calcular media/desviación y mostrar el riesgo de dividir por 0

# PROBLEM: Division by zero when std = 0  # PROBLEMA: si std=0, normalizar produce inf/NaN
data = np.array([5, 5, 5, 5, 5])  # Todos los valores iguales => varianza 0
std = np.std(data)  # 0.0  # Desviación estándar cero
normalized = (data - np.mean(data)) / std  # RuntimeWarning: divide by zero  # Divide entre 0 => inf/NaN

# SOLUTION: Add epsilon  # SOLUCIÓN: suma un epsilon para estabilizar la división
epsilon = 1e-8  # Pequeña constante para estabilizar divisiones
normalized_safe = (data - np.mean(data)) / (std + epsilon)  # Evita división por cero

# RULE: Siempre usar epsilon en divisiones (especialmente en softmax, normalizaciones)
```

### Error 5: Incorrect data types

```python
import numpy as np  # NumPy para ejemplificar cómo el dtype afecta resultados (truncamiento)

# PROBLEM: Operations with ints when you need floats  # PROBLEMA: usar int cuando necesitas float puede truncar resultados
a = np.array([1, 2, 3])  # dtype: int64  # Por defecto, enteros
b = a / 2  # dtype: float64 (OK en Python 3)  # División NO in-place => promueve a float

# BUT in in-place operations:  # PERO en operaciones in-place (sobre el mismo array):
a = np.array([1, 2, 3])  # Vuelve a int
a /= 2  # a remains int64, it truncates!  # In-place: intenta guardar floats en int => trunca
print(a)  # [0, 1, 1] - TRUNCATED!  # Se pierde precisión

# SOLUTION: Specify dtype when creating  # SOLUCIÓN: crea el array con dtype float desde el inicio
a = np.array([1, 2, 3], dtype=np.float64)  # Crea directamente como float
a /= 2  # In-place sobre float => conserva decimales
print(a)  # [0.5, 1.0, 1.5] - Correct  # Resultado correcto

# RULE: For ML, always use dtype=np.float64 or np.float32  # REGLA: en ML usa floats (mejor estabilidad numérica)
```

---

## 🛠️ Professional code standards (v3.2)

> 💎 **v3.2 philosophy:** code is not considered done until it passes `mypy`, `ruff`, and `pytest`.

### Professional environment setup

```bash
# Create a virtual environment
python -m venv .venv  # Crea un entorno virtual local (aislado) en la carpeta .venv
source .venv/bin/activate  # Linux/Mac  # Activa el entorno (usa el Python/pip de .venv)
# .venv\Scripts\activate   # Windows  # Alternativa de activación en Windows

# Install quality tools
pip install numpy pandas matplotlib  # Instala dependencias principales (runtime)
pip install mypy ruff pytest  # Instala herramientas de calidad: typing, lint/format y tests

# pyproject.toml file (create at the project root)
```

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"  # Versión objetivo de Python para el chequeo de tipos
warn_return_any = true  # Advierte cuando una función retorna Any (pérdida de precisión)
warn_unused_ignores = true  # Advierte si hay "# type: ignore" innecesarios
disallow_untyped_defs = true  # Exige anotaciones en defs (evita funciones sin tipos)

[tool.ruff]
line-length = 100  # Longitud máxima de línea
select = ["E", "F", "W", "I", "UP"]  # Reglas: errores/estilo/imports/modernización

[tool.pytest.ini_options]
testpaths = ["tests"]  # Directorio donde pytest buscará tests
python_files = "test_*.py"  # Patrón de nombres de archivos de test
```

### Example: Code with type hints

```python
# src/linear_algebra.py
"""Linear algebra operations from scratch."""
import numpy as np  # NumPy para operaciones vectorizadas (sum, sqrt) y arrays
from numpy.typing import NDArray  # Tipado: NDArray permite anotar ndarrays con dtype (útil para mypy)


def dot_product(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """
    Compute the dot product of two vectors.

    Args:
        a: First vector (n,)
        b: Second vector (n,)

    Returns:
        The dot product (scalar)

    Raises:
        ValueError: If the vectors have different shapes
    """
    if a.shape != b.shape:  # Validación: el producto punto requiere shapes iguales
        raise ValueError(f"Incompatible shapes: {a.shape} vs {b.shape}")  # Falla con mensaje claro
    return float(np.sum(a * b))  # Multiplica elemento a elemento y suma; float() -> float nativo


def norm_l2(v: NDArray[np.float64]) -> float:
    """Compute the L2 (Euclidean) norm of a vector."""
    return float(np.sqrt(np.sum(v ** 2)))  # sqrt(sum(v^2)) => definición de norma L2
```

### Example: Tests with pytest

```python
# tests/test_linear_algebra.py
"""Unit tests for linear_algebra.py."""
import numpy as np  # NumPy para construir vectores/matrices de prueba
import pytest  # pytest para asserts y verificación de excepciones
from src.linear_algebra import dot_product, norm_l2  # Importa funciones a testear


class TestDotProduct:
    """Tests for the dot_product function."""

    def test_dot_product_basic(self) -> None:
        """Basic test: [1,2,3] · [4,5,6] = 32"""
        a = np.array([1.0, 2.0, 3.0])  # Vector a
        b = np.array([4.0, 5.0, 6.0])  # Vector b
        assert dot_product(a, b) == 32.0  # Verifica 1*4 + 2*5 + 3*6

    def test_dot_product_orthogonal(self) -> None:
        """Orthogonal vectors have dot product = 0"""
        a = np.array([1.0, 0.0])  # Unitario en x
        b = np.array([0.0, 1.0])  # Unitario en y
        assert dot_product(a, b) == 0.0  # Ortogonales => producto punto 0

    def test_dot_product_shape_mismatch(self) -> None:
        """Should raise ValueError if shapes do not match"""
        a = np.array([1.0, 2.0])  # Shape (2,)
        b = np.array([1.0, 2.0, 3.0])  # Shape (3,)
        with pytest.raises(ValueError):  # Debe fallar por incompatibilidad de shapes
            dot_product(a, b)


class TestNormL2:
    """Tests for the norm_l2 function."""

    def test_norm_unit_vector(self) -> None:
        """A unit vector has norm 1"""
        v = np.array([1.0, 0.0, 0.0])  # Vector unitario
        assert norm_l2(v) == 1.0  # Norma de unitario = 1

    def test_norm_345(self) -> None:
        """3-4-5 triangle: norm of [3,4] = 5"""
        v = np.array([3.0, 4.0])  # Vector (3,4)
        assert norm_l2(v) == 5.0  # sqrt(3^2 + 4^2) = 5
```

### Verification commands

```bash
# Run at the project root:

# 1. Type checking (mypy)
mypy src/  # Chequea tipos estáticos y detecta inconsistencias

# 2. Style/linting (ruff)
ruff check src/  # Lint: errores comunes (imports, variables sin usar, estilo)
ruff format src/  # Auto-format: aplica formato automáticamente

# 3. Run tests (pytest)
pytest tests/ -v  # Ejecuta tests en modo verboso

# 4. All together (before each commit)
mypy src/ && ruff check src/ && pytest tests/ -v  # Pipeline mínimo antes de commitear
```

---

## 🎯 The Whiteboard Challenge (Feynman method)

> 📝 **Instruction:** After implementing code, you should be able to explain the algorithm in **at most 5 lines** without technical jargon. If you can’t, go back to the theory.

### Example: Broadcasting

**❌ Technical explanation (bad):**
"Broadcasting is NumPy's ability to perform element-wise operations between arrays of different shapes by implicitly expanding dimensions according to compatibility rules."

**✅ Feynman explanation (good):**
"When you add a number to a list, NumPy automatically adds that number to EACH element. It's as if the number were "copied" so it has the same size as the list. The same happens between lists of different sizes, as long as one of them has size 1 in some dimension."

### Your challenge for Module 01:

Explain in 5 lines or less:
1. Why is NumPy faster than Python lists?
2. What does `axis=0` vs `axis=1` mean?
3. Why is `.copy()` important?

---

## ✅ Completion checklist (v3.2)

### Knowledge
- [ ] I can create 1D, 2D, and 3D arrays with NumPy
- [ ] I understand array indexing and slicing
- [ ] I can explain broadcasting and use it
- [ ] I can compute axis-based aggregations (axis)
- [ ] I can rewrite loops as vectorized operations
- [ ] I know the differences between `@`, `np.dot`, `np.matmul`
- [ ] I know the 5 common NumPy errors and their solutions

### Code deliverables
- [ ] `benchmark_vectorization.py` implemented
- [ ] NumPy vs list speedup is >50x in my tests
- [ ] `mypy src/` passes with no errors
- [ ] `ruff check src/` passes with no errors
- [ ] At least 3 tests passing with `pytest`

### Feynman method
- [ ] I can explain broadcasting in 5 lines without jargon
- [ ] I can explain axis=0 vs axis=1 in 5 lines without jargon
- [ ] I can explain why .copy() is important

---

## 🔗 Navigation

| Previous | Index | Next |
|----------|--------|-----------|
| - | [00_INDICE](00_INDICE.md) | [02_ALGEBRA_LINEAL_ML](02_ALGEBRA_LINEAL_ML.md) |
