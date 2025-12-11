# Módulo 01 - Python Científico + Pandas

> **🎯 Objetivo:** Dominar Pandas para datos + NumPy para matemáticas  
> **Fase:** 1 - Fundamentos | **Semanas 1-2**  
> **Prerrequisitos:** Python básico (variables, funciones, listas, loops)

---

## 🧠 ¿Por Qué Este Módulo?

### El Problema con Python Puro para ML

```python
# ❌ Así NO se hace en Machine Learning
def dot_product_slow(a: list, b: list) -> float:
    """Producto punto con loop - LENTO."""
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Para vectores de 1 millón de elementos:
# Tiempo: ~200ms
```

```python
# ✅ Así SÍ se hace en Machine Learning
import numpy as np

def dot_product_fast(a: np.ndarray, b: np.ndarray) -> float:
    """Producto punto vectorizado - RÁPIDO."""
    return np.dot(a, b)

# Para vectores de 1 millón de elementos:
# Tiempo: ~2ms (100x más rápido)
```

### Conexión con el Pathway

En los cursos de CU Boulder:
- **Supervised Learning:** Multiplicaciones de matrices para regresión
- **Unsupervised Learning:** PCA requiere descomposición de matrices
- **Deep Learning:** Forward/backward pass son operaciones matriciales

**Sin NumPy, no puedes hacer ML eficiente.**

---

## 📚 Contenido del Módulo

### Semana 1: Pandas + NumPy Básico

```
┌─────────────────────────────────────────────────────────────────┐
│  DÍA 1: Pandas - DataFrame y Series                             │
│  DÍA 2: Pandas - Carga de CSVs (read_csv, head, info)           │
│  DÍA 3: Pandas - Limpieza (dropna, fillna, dtypes)              │
│  DÍA 4: NumPy - Arrays y dtypes                                 │
│  DÍA 5: NumPy - Indexing y Slicing                              │
│  DÍA 6: Pandas → NumPy (df.values, df.to_numpy())               │
└─────────────────────────────────────────────────────────────────┘
```

### Semana 2: NumPy Vectorizado

```
┌─────────────────────────────────────────────────────────────────┐
│  DÍA 1: Broadcasting                                            │
│  DÍA 2: Producto matricial (@, np.dot, np.matmul)               │
│  DÍA 3: Reshape, flatten, transpose                             │
│  DÍA 4: Agregaciones y operaciones con ejes                     │
│  DÍA 5: Random y generación de datos sintéticos                 │
│  DÍA 6: Entregable: Pipeline Pandas → NumPy                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Conceptos Clave

### 0. Pandas Esencial (Días 1-3)

#### ¿Por Qué Pandas?

En el mundo real de ML, los datos vienen en CSVs sucios, no en arrays NumPy perfectos. Antes de aplicar cualquier algoritmo necesitas:

1. **Cargar datos** desde archivos
2. **Explorar** estructura y tipos
3. **Limpiar** valores faltantes y errores
4. **Convertir** a NumPy para el modelo

```python
import pandas as pd
import numpy as np

# ========== CARGA DE DATOS ==========
# Cargar CSV
df = pd.read_csv('data/iris.csv')

# Primeras filas
print(df.head())

# Información del DataFrame
print(df.info())
#  Column         Non-Null Count  Dtype  
# ---  ------         --------------  -----  
#  0   sepal_length   150 non-null    float64
#  1   sepal_width    150 non-null    float64
#  2   petal_length   150 non-null    float64
#  3   petal_width    150 non-null    float64
#  4   species        150 non-null    object 

# Estadísticas básicas
print(df.describe())
```

#### Limpieza de Datos

```python
import pandas as pd

# Crear DataFrame con datos sucios
df = pd.DataFrame({
    'edad': [25, 30, None, 45, 50],
    'salario': [50000, 60000, 70000, None, 90000],
    'ciudad': ['Madrid', 'Barcelona', 'Madrid', 'Sevilla', None]
})

# ========== DETECTAR NULOS ==========
print(df.isnull().sum())
# edad       1
# salario    1
# ciudad     1

# ========== ELIMINAR FILAS CON NULOS ==========
df_clean = df.dropna()  # Elimina filas con cualquier nulo
print(f"Filas después de dropna: {len(df_clean)}")  # 2

# ========== RELLENAR NULOS ==========
df_filled = df.copy()
df_filled['edad'] = df_filled['edad'].fillna(df_filled['edad'].mean())
df_filled['salario'] = df_filled['salario'].fillna(df_filled['salario'].median())
df_filled['ciudad'] = df_filled['ciudad'].fillna('Desconocido')

print(df_filled)
```

#### Selección y Filtrado

```python
import pandas as pd

df = pd.read_csv('data/iris.csv')

# ========== SELECCIONAR COLUMNAS ==========
# Una columna (Serie)
sepal_length = df['sepal_length']

# Múltiples columnas (DataFrame)
features = df[['sepal_length', 'sepal_width']]

# ========== FILTRAR FILAS ==========
# Condición simple
setosa = df[df['species'] == 'setosa']

# Múltiples condiciones
large_setosa = df[(df['species'] == 'setosa') & (df['sepal_length'] > 5)]

# ========== LOC e ILOC ==========
# loc: por etiquetas
df.loc[0:5, ['sepal_length', 'species']]

# iloc: por posición (como NumPy)
df.iloc[0:5, 0:2]
```

#### De Pandas a NumPy (Día 6)

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data/iris.csv')

# ========== SEPARAR FEATURES Y TARGET ==========
# Features (X) - todas las columnas numéricas
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()
print(f"X shape: {X.shape}")  # (150, 4)
print(f"X dtype: {X.dtype}")  # float64

# Target (y) - convertir categorías a números
y = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2}).to_numpy()
print(f"y shape: {y.shape}")  # (150,)

# ========== VERIFICAR ==========
print(f"Tipo X: {type(X)}")  # <class 'numpy.ndarray'>
print(f"Tipo y: {type(y)}")  # <class 'numpy.ndarray'>

# Ahora X e y están listos para algoritmos de ML
```

---

### 1. Arrays vs Listas

```python
import numpy as np

# Lista de Python
lista = [1, 2, 3, 4, 5]

# Array de NumPy
array = np.array([1, 2, 3, 4, 5])

# Diferencias clave:
# 1. Tipo homogéneo (todos los elementos del mismo tipo)
# 2. Tamaño fijo después de creación
# 3. Operaciones vectorizadas
# 4. Almacenamiento contiguo en memoria
```

### 2. Creación de Arrays

```python
import numpy as np

# Desde lista
a = np.array([1, 2, 3])

# Arrays especiales
zeros = np.zeros((3, 4))        # Matriz 3x4 de ceros
ones = np.ones((2, 3))          # Matriz 2x3 de unos
identity = np.eye(4)            # Matriz identidad 4x4
random = np.random.randn(3, 3)  # Matriz 3x3 valores normales

# Secuencias
rango = np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5) # [0, 0.25, 0.5, 0.75, 1]

print(f"Shape de zeros: {zeros.shape}")  # (3, 4)
print(f"Dtype de zeros: {zeros.dtype}")  # float64
```

### 3. Indexing y Slicing

```python
import numpy as np

# Crear matriz 2D
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Acceso a elementos
print(matrix[0, 0])      # 1 (fila 0, columna 0)
print(matrix[1, 2])      # 6 (fila 1, columna 2)

# Slicing
print(matrix[0, :])      # [1, 2, 3] (toda la fila 0)
print(matrix[:, 1])      # [2, 5, 8] (toda la columna 1)
print(matrix[0:2, 1:3])  # [[2, 3], [5, 6]] (submatriz)

# Indexing booleano
print(matrix[matrix > 5])  # [6, 7, 8, 9]
```

### 4. Broadcasting

```python
import numpy as np

# Broadcasting: operar arrays de diferentes shapes

# Escalar + Array
a = np.array([1, 2, 3])
print(a + 10)  # [11, 12, 13]

# Vector + Matriz (broadcasting automático)
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
vector = np.array([10, 20, 30])

# El vector se "expande" para coincidir con la matriz
print(matrix + vector)
# [[11, 22, 33],
#  [14, 25, 36]]

# Regla de broadcasting:
# Las dimensiones deben ser iguales O una de ellas debe ser 1
```

### 5. Agregaciones y Ejes

```python
import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# Agregaciones globales
print(np.sum(matrix))   # 21 (suma de todos)
print(np.mean(matrix))  # 3.5 (promedio de todos)
print(np.std(matrix))   # 1.707... (desviación estándar)

# Agregaciones por eje
# axis=0: colapsar filas (operar columnas)
print(np.sum(matrix, axis=0))  # [5, 7, 9]

# axis=1: colapsar columnas (operar filas)
print(np.sum(matrix, axis=1))  # [6, 15]

# Visualización de ejes:
# ┌─────────────┐
# │ axis=0 ↓    │
# │ [1, 2, 3]   │ → axis=1
# │ [4, 5, 6]   │ → axis=1
# └─────────────┘
```

### 6. Operaciones Matriciales

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Operaciones elemento a elemento
print(A + B)   # Suma
print(A * B)   # Multiplicación elemento a elemento (Hadamard)
print(A / B)   # División elemento a elemento

# Producto matricial (lo que usarás en ML)
print(A @ B)           # Operador @ (Python 3.5+)
print(np.matmul(A, B)) # Función matmul
print(np.dot(A, B))    # Función dot

# Resultado:
# [[19, 22],
#  [43, 50]]

# Transpuesta
print(A.T)
# [[1, 3],
#  [2, 4]]
```

### 7. Vectorización: Eliminar Loops

```python
import numpy as np

# ❌ CON LOOP (lento)
def normalize_loop(data: list) -> list:
    """Normalizar datos con loop."""
    mean = sum(data) / len(data)
    std = (sum((x - mean)**2 for x in data) / len(data)) ** 0.5
    return [(x - mean) / std for x in data]

# ✅ VECTORIZADO (rápido)
def normalize_vectorized(data: np.ndarray) -> np.ndarray:
    """Normalizar datos vectorizado."""
    return (data - np.mean(data)) / np.std(data)

# Ejemplo
data = np.random.randn(1000000)

# La versión vectorizada es ~100x más rápida
normalized = normalize_vectorized(data)
```

### 8. Funciones Universales (ufuncs)

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])

# Funciones matemáticas (aplicadas elemento a elemento)
print(np.exp(x))      # e^x
print(np.log(x))      # ln(x)
print(np.sqrt(x))     # √x
print(np.sin(x))      # sin(x)

# Importante para ML:
# Sigmoid: σ(x) = 1 / (1 + e^(-x))
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

# ReLU: max(0, x)
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

print(sigmoid(np.array([-2, -1, 0, 1, 2])))
# [0.119, 0.269, 0.5, 0.731, 0.881]
```

### 9. Reshape y Manipulación de Forma

```python
import numpy as np

# Crear array 1D
a = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape a 2D
matrix = a.reshape(3, 4)
print(matrix.shape)  # (3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

# Reshape a 3D
tensor = a.reshape(2, 2, 3)
print(tensor.shape)  # (2, 2, 3)

# Flatten: volver a 1D
flat = matrix.flatten()
print(flat.shape)  # (12,)

# -1 para inferir dimensión automáticamente
auto = a.reshape(4, -1)  # (4, 3)
auto = a.reshape(-1, 6)  # (2, 6)
```

### 10. Generación de Datos Aleatorios

```python
import numpy as np

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Distribución uniforme [0, 1)
uniform = np.random.rand(3, 3)

# Distribución normal (media=0, std=1)
normal = np.random.randn(3, 3)

# Distribución normal personalizada
custom_normal = np.random.normal(loc=5, scale=2, size=(100,))

# Enteros aleatorios
integers = np.random.randint(0, 10, size=(3, 3))

# Shuffle (mezclar)
data = np.arange(10)
np.random.shuffle(data)

# Muestreo sin reemplazo
sample = np.random.choice(data, size=5, replace=False)
```

---

## 📊 Type Hints con NumPy

```python
import numpy as np
from numpy.typing import NDArray

# Type hints para arrays
def normalize(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normaliza un array de floats."""
    return (data - np.mean(data)) / np.std(data)

# Type hints genéricos
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula el producto punto de dos vectores."""
    return float(np.dot(a, b))

# Con mypy
# pip install numpy-stubs
```

---

## ⚡ Benchmark: Lista vs NumPy

```python
import numpy as np
import time
from typing import List

def benchmark_dot_product():
    """Compara rendimiento de lista vs NumPy."""
    size = 1_000_000
    
    # Crear datos
    list_a: List[float] = [float(i) for i in range(size)]
    list_b: List[float] = [float(i) for i in range(size)]
    array_a = np.array(list_a)
    array_b = np.array(list_b)
    
    # Benchmark lista
    start = time.time()
    result_list = sum(a * b for a, b in zip(list_a, list_b))
    time_list = time.time() - start
    
    # Benchmark NumPy
    start = time.time()
    result_numpy = np.dot(array_a, array_b)
    time_numpy = time.time() - start
    
    print(f"Lista:  {time_list:.4f}s")
    print(f"NumPy:  {time_numpy:.4f}s")
    print(f"Speedup: {time_list/time_numpy:.1f}x")
    
    # Verificar resultados iguales
    assert abs(result_list - result_numpy) < 1e-6

if __name__ == "__main__":
    benchmark_dot_product()
    
# Output típico:
# Lista:  0.1523s
# NumPy:  0.0015s
# Speedup: 101.5x
```

---

## 🎯 Ejercicios

### Ejercicio 1.1: Crear Arrays
```python
# Crear:
# 1. Vector de 10 ceros
# 2. Matriz 3x3 de unos
# 3. Matriz identidad 4x4
# 4. Vector de 0 a 99
# 5. 20 valores equiespaciados entre 0 y 2π
```

### Ejercicio 1.2: Indexing
```python
# Dada la matriz:
matrix = np.arange(20).reshape(4, 5)

# Extraer:
# 1. Elemento en fila 2, columna 3
# 2. Toda la fila 1
# 3. Toda la columna 4
# 4. Submatriz filas 1-2, columnas 2-4
# 5. Elementos mayores que 10
```

### Ejercicio 1.3: Broadcasting
```python
# Sin usar loops:
# 1. Sumar 100 a cada elemento de una matriz 3x3
# 2. Multiplicar cada fila por un vector diferente
# 3. Normalizar cada columna (restar media, dividir por std)
```

### Ejercicio 1.4: Vectorización
```python
# Reescribir sin loops:
def euclidean_distance_loop(a: list, b: list) -> float:
    total = 0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return total ** 0.5

# Tu versión vectorizada:
def euclidean_distance_vectorized(a: np.ndarray, b: np.ndarray) -> float:
    pass  # Implementar
```

### Ejercicio 1.5: Funciones de Activación
```python
# Implementar las siguientes funciones de activación:

def sigmoid(x: np.ndarray) -> np.ndarray:
    """σ(x) = 1 / (1 + e^(-x))"""
    pass

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU(x) = max(0, x)"""
    pass

def softmax(x: np.ndarray) -> np.ndarray:
    """softmax(x)_i = e^(x_i) / Σ e^(x_j)"""
    pass

# Verificar:
# sigmoid(0) ≈ 0.5
# relu(-5) = 0, relu(5) = 5
# softmax([1,2,3]).sum() ≈ 1.0
```

---

## 📦 Entregable del Módulo

### Script: `benchmark_vectorization.py`

```python
"""
Benchmark: Operaciones vectoriales Lista vs NumPy

Este script compara el rendimiento de operaciones comunes
usando listas de Python puras vs arrays de NumPy.

Operaciones comparadas:
1. Producto punto
2. Normalización
3. Distancia euclidiana
4. Suma de matrices

Autor: [Tu nombre]
Fecha: [Fecha]
"""

import numpy as np
import time
from typing import List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Resultado de un benchmark."""
    operation: str
    time_list: float
    time_numpy: float
    speedup: float


def benchmark(
    func_list: Callable,
    func_numpy: Callable,
    args_list: Tuple,
    args_numpy: Tuple,
    operation_name: str,
    iterations: int = 100
) -> BenchmarkResult:
    """Ejecuta benchmark comparativo."""
    
    # Benchmark lista
    start = time.time()
    for _ in range(iterations):
        func_list(*args_list)
    time_list = (time.time() - start) / iterations
    
    # Benchmark NumPy
    start = time.time()
    for _ in range(iterations):
        func_numpy(*args_numpy)
    time_numpy = (time.time() - start) / iterations
    
    return BenchmarkResult(
        operation=operation_name,
        time_list=time_list,
        time_numpy=time_numpy,
        speedup=time_list / time_numpy
    )


# === IMPLEMENTAR TUS FUNCIONES AQUÍ ===

def dot_product_list(a: List[float], b: List[float]) -> float:
    """Producto punto con listas."""
    # TODO: Implementar
    pass


def dot_product_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Producto punto con NumPy."""
    # TODO: Implementar
    pass


def normalize_list(data: List[float]) -> List[float]:
    """Normalizar con listas."""
    # TODO: Implementar
    pass


def normalize_numpy(data: np.ndarray) -> np.ndarray:
    """Normalizar con NumPy."""
    # TODO: Implementar
    pass


def euclidean_distance_list(a: List[float], b: List[float]) -> float:
    """Distancia euclidiana con listas."""
    # TODO: Implementar
    pass


def euclidean_distance_numpy(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia euclidiana con NumPy."""
    # TODO: Implementar
    pass


def matrix_sum_list(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Suma de matrices con listas."""
    # TODO: Implementar
    pass


def matrix_sum_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Suma de matrices con NumPy."""
    # TODO: Implementar
    pass


def main():
    """Ejecutar todos los benchmarks."""
    size = 10000
    
    # Crear datos de prueba
    list_a = [float(i) for i in range(size)]
    list_b = [float(i) for i in range(size)]
    array_a = np.array(list_a)
    array_b = np.array(list_b)
    
    matrix_size = 100
    list_matrix_a = [[float(i*j) for j in range(matrix_size)] 
                     for i in range(matrix_size)]
    list_matrix_b = [[float(i+j) for j in range(matrix_size)] 
                     for i in range(matrix_size)]
    array_matrix_a = np.array(list_matrix_a)
    array_matrix_b = np.array(list_matrix_b)
    
    # Ejecutar benchmarks
    results = []
    
    results.append(benchmark(
        dot_product_list, dot_product_numpy,
        (list_a, list_b), (array_a, array_b),
        "Producto Punto"
    ))
    
    results.append(benchmark(
        normalize_list, normalize_numpy,
        (list_a,), (array_a,),
        "Normalización"
    ))
    
    results.append(benchmark(
        euclidean_distance_list, euclidean_distance_numpy,
        (list_a, list_b), (array_a, array_b),
        "Distancia Euclidiana"
    ))
    
    results.append(benchmark(
        matrix_sum_list, matrix_sum_numpy,
        (list_matrix_a, list_matrix_b), (array_matrix_a, array_matrix_b),
        "Suma de Matrices"
    ))
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("BENCHMARK: Lista vs NumPy")
    print("="*60)
    print(f"{'Operación':<25} {'Lista (ms)':<12} {'NumPy (ms)':<12} {'Speedup':<10}")
    print("-"*60)
    
    for r in results:
        print(f"{r.operation:<25} {r.time_list*1000:<12.4f} {r.time_numpy*1000:<12.4f} {r.speedup:<10.1f}x")
    
    print("="*60)
    print(f"\nSpeedup promedio: {sum(r.speedup for r in results)/len(results):.1f}x")


if __name__ == "__main__":
    main()
```

---

## 🐛 Debugging NumPy: Errores que te Harán Perder el Tiempo (v3.2)

> ⚠️ **CRÍTICO:** Estos 5 errores son los más frecuentes en las Fases 1 y 2. Resolverlos ahora previene horas de frustración.

### Error 1: Shape Mismatch - `(5,)` vs `(5,1)`

```python
import numpy as np

# PROBLEMA: Vector 1D vs Vector Columna
v1 = np.array([1, 2, 3, 4, 5])      # Shape: (5,) - Vector 1D
v2 = np.array([[1], [2], [3], [4], [5]])  # Shape: (5, 1) - Vector columna

print(f"v1.shape: {v1.shape}")  # (5,)
print(f"v2.shape: {v2.shape}")  # (5, 1)

# ESTO FALLA en Regresión Lineal:
# Si X tiene shape (100, 5) y theta tiene shape (5,), el resultado es (100,)
# Si theta tiene shape (5, 1), el resultado es (100, 1)

# SOLUCIÓN: Usar reshape o keepdims
v1_columna = v1.reshape(-1, 1)  # (5,) → (5, 1)
v1_columna_alt = v1[:, np.newaxis]  # Alternativa

# REGLA: Para ML, los vectores de features deben ser (n, 1), no (n,)
```

### Error 2: Broadcasting Silencioso Incorrecto

```python
import numpy as np

# PROBLEMA: Broadcasting no falla, pero da resultados incorrectos
X = np.random.randn(100, 5)  # 100 samples, 5 features
mean_wrong = np.mean(X)      # ¡INCORRECTO! Media de TODO el array
mean_correct = np.mean(X, axis=0)  # Correcto: media por feature (shape: (5,))

print(f"mean_wrong shape: {np.array(mean_wrong).shape}")  # () - escalar
print(f"mean_correct shape: {mean_correct.shape}")  # (5,)

# REGLA: Siempre especifica axis= en agregaciones
# axis=0: opera sobre filas (resultado por columna)
# axis=1: opera sobre columnas (resultado por fila)
```

### Error 3: Modificación In-Place Inesperada

```python
import numpy as np

# PROBLEMA: Los slices de NumPy son VISTAS, no copias
original = np.array([1, 2, 3, 4, 5])
slice_view = original[1:4]
slice_view[0] = 999

print(original)  # [1, 999, 3, 4, 5] - ¡ORIGINAL MODIFICADO!

# SOLUCIÓN: Usar .copy() explícitamente
original = np.array([1, 2, 3, 4, 5])
slice_copy = original[1:4].copy()
slice_copy[0] = 999

print(original)  # [1, 2, 3, 4, 5] - Original intacto
```

### Error 4: División por Cero en Normalización

```python
import numpy as np

# PROBLEMA: División por cero cuando std = 0
data = np.array([5, 5, 5, 5, 5])
std = np.std(data)  # 0.0
normalized = (data - np.mean(data)) / std  # RuntimeWarning: divide by zero

# SOLUCIÓN: Añadir epsilon
epsilon = 1e-8
normalized_safe = (data - np.mean(data)) / (std + epsilon)

# REGLA: Siempre usar epsilon en divisiones (especialmente en softmax, normalizaciones)
```

### Error 5: Tipos de Datos Incorrectos

```python
import numpy as np

# PROBLEMA: Operaciones con int cuando necesitas float
a = np.array([1, 2, 3])  # dtype: int64
b = a / 2  # dtype: float64 (OK en Python 3)

# PERO en operaciones in-place:
a = np.array([1, 2, 3])
a /= 2  # a sigue siendo int64, se trunca!
print(a)  # [0, 1, 1] - ¡TRUNCADO!

# SOLUCIÓN: Especificar dtype al crear
a = np.array([1, 2, 3], dtype=np.float64)
a /= 2
print(a)  # [0.5, 1.0, 1.5] - Correcto

# REGLA: Para ML, siempre usar dtype=np.float64 o np.float32
```

---

## 🛠️ Estándares de Código Profesional (v3.2)

> 💎 **Filosofía v3.2:** El código no se considera terminado hasta que pase `mypy`, `ruff` y `pytest`.

### Configuración del Entorno Profesional

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar herramientas de calidad
pip install numpy pandas matplotlib
pip install mypy ruff pytest

# Archivo pyproject.toml (crear en la raíz del proyecto)
```

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

### Ejemplo: Código con Type Hints

```python
# src/linear_algebra.py
"""Operaciones de álgebra lineal desde cero."""
import numpy as np
from numpy.typing import NDArray


def dot_product(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """
    Calcula el producto punto de dos vectores.
    
    Args:
        a: Primer vector (n,)
        b: Segundo vector (n,)
    
    Returns:
        El producto punto (escalar)
    
    Raises:
        ValueError: Si los vectores tienen shapes diferentes
    """
    if a.shape != b.shape:
        raise ValueError(f"Shapes incompatibles: {a.shape} vs {b.shape}")
    return float(np.sum(a * b))


def norm_l2(v: NDArray[np.float64]) -> float:
    """Calcula la norma L2 (euclidiana) de un vector."""
    return float(np.sqrt(np.sum(v ** 2)))
```

### Ejemplo: Tests con pytest

```python
# tests/test_linear_algebra.py
"""Tests unitarios para linear_algebra.py"""
import numpy as np
import pytest
from src.linear_algebra import dot_product, norm_l2


class TestDotProduct:
    """Tests para la función dot_product."""
    
    def test_dot_product_basic(self) -> None:
        """Test básico: [1,2,3] · [4,5,6] = 32"""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert dot_product(a, b) == 32.0
    
    def test_dot_product_orthogonal(self) -> None:
        """Vectores ortogonales tienen producto punto = 0"""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert dot_product(a, b) == 0.0
    
    def test_dot_product_shape_mismatch(self) -> None:
        """Debe lanzar ValueError si shapes no coinciden"""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            dot_product(a, b)


class TestNormL2:
    """Tests para la función norm_l2."""
    
    def test_norm_unit_vector(self) -> None:
        """Vector unitario tiene norma 1"""
        v = np.array([1.0, 0.0, 0.0])
        assert norm_l2(v) == 1.0
    
    def test_norm_345(self) -> None:
        """Triángulo 3-4-5: norma de [3,4] = 5"""
        v = np.array([3.0, 4.0])
        assert norm_l2(v) == 5.0
```

### Comandos de Verificación

```bash
# Ejecutar en la raíz del proyecto:

# 1. Verificar tipos (mypy)
mypy src/

# 2. Verificar estilo (ruff)
ruff check src/
ruff format src/  # Auto-formatear

# 3. Ejecutar tests (pytest)
pytest tests/ -v

# 4. Todo junto (antes de cada commit)
mypy src/ && ruff check src/ && pytest tests/ -v
```

---

## 🎯 El Reto del Tablero Blanco (Metodología Feynman)

> 📝 **Instrucción:** Después de implementar código, debes poder explicar el algoritmo en **máximo 5 líneas** sin usar jerga técnica. Si no puedes, vuelve a la teoría.

### Ejemplo: Broadcasting

**❌ Explicación técnica (mala):**
"Broadcasting es la capacidad de NumPy de realizar operaciones elemento a elemento entre arrays de diferentes shapes mediante la expansión implícita de dimensiones según reglas de compatibilidad."

**✅ Explicación Feynman (buena):**
"Cuando sumas un número a una lista, NumPy automáticamente suma ese número a CADA elemento. Es como si el número se 'copiara' para que tenga el mismo tamaño que la lista. Lo mismo pasa entre listas de diferentes tamaños, siempre que una de ellas tenga tamaño 1 en alguna dimensión."

### Tu Reto para el Módulo 01:

Explica en 5 líneas o menos:
1. ¿Por qué NumPy es más rápido que listas de Python?
2. ¿Qué significa `axis=0` vs `axis=1`?
3. ¿Por qué `.copy()` es importante?

---

## ✅ Checklist de Finalización (v3.2)

### Conocimiento
- [ ] Puedo crear arrays 1D, 2D y 3D con NumPy
- [ ] Entiendo indexing y slicing de arrays
- [ ] Puedo explicar broadcasting y usarlo
- [ ] Sé calcular agregaciones por eje (axis)
- [ ] Puedo reescribir loops como operaciones vectorizadas
- [ ] Conozco las diferencias entre `@`, `np.dot`, `np.matmul`
- [ ] Conozco los 5 errores comunes de NumPy y sus soluciones

### Entregables de Código
- [ ] `benchmark_vectorization.py` implementado
- [ ] El speedup de NumPy vs lista es >50x en mis pruebas
- [ ] `mypy src/` pasa sin errores
- [ ] `ruff check src/` pasa sin errores
- [ ] Al menos 3 tests con `pytest` pasando

### Metodología Feynman
- [ ] Puedo explicar broadcasting en 5 líneas sin jerga
- [ ] Puedo explicar axis=0 vs axis=1 en 5 líneas sin jerga
- [ ] Puedo explicar por qué .copy() es importante

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| - | [00_INDICE](00_INDICE.md) | [02_ALGEBRA_LINEAL_ML](02_ALGEBRA_LINEAL_ML.md) |
