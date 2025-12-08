# Módulo 03 - Álgebra Lineal para ML

> **🎯 Objetivo:** Implementar operaciones vectoriales y matriciales desde cero  
> **Fase:** Fundamentos | **Prerrequisito para:** Módulos 04-09

---

## 🧠 Analogía: Vectores como Flechas, Matrices como Transformaciones

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   VECTOR = Una flecha en el espacio                                         │
│   ─────────────────────────────────                                         │
│                                                                             │
│   v = [3, 4]                                                                │
│                    ↗ (3, 4)                                                 │
│                  ╱                                                          │
│                ╱                                                            │
│              ╱  │                                                           │
│            ╱    │ 4                                                         │
│          ╱      │                                                           │
│        ●────────┘                                                           │
│     origen  3                                                               │
│                                                                             │
│   En Archimedes:                                                            │
│   • Cada documento es un vector en el "espacio de palabras"                 │
│   • Dimensión = número de palabras únicas                                   │
│   • Valor = frecuencia/importancia de cada palabra                          │
│   • Similitud = qué tan "paralelos" son dos vectores                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Vectores como Listas](#1-vectores)
2. [Operaciones Vectoriales](#2-operaciones-vectoriales)
3. [Producto Punto y Norma](#3-producto-punto)
4. [Matrices como Listas de Listas](#4-matrices)
5. [Operaciones Matriciales](#5-operaciones-matriciales)

---

## 1. Vectores como Listas {#1-vectores}

### 1.1 Representación

```python
# Un vector es simplemente una lista de números
Vector = list[float]

# Ejemplos
v1: Vector = [1.0, 2.0, 3.0]      # Vector 3D
v2: Vector = [0.5, 0.3, 0.8, 0.2] # Vector 4D

# En el contexto de TF-IDF:
# Cada posición corresponde a una palabra del vocabulario
# El valor es la importancia de esa palabra en el documento

vocabulary = ["python", "java", "code", "tutorial"]
doc_vector: Vector = [0.8, 0.0, 0.5, 0.3]
# Significa: mucho "python", nada de "java", algo de "code" y "tutorial"
```

### 1.2 Acceso y Longitud

```python
def get_dimension(v: Vector) -> int:
    """Return the dimension (number of components) of a vector."""
    return len(v)


def get_component(v: Vector, index: int) -> float:
    """Get specific component of vector."""
    if index < 0 or index >= len(v):
        raise IndexError(f"Index {index} out of range for vector of dimension {len(v)}")
    return v[index]
```

---

## 2. Operaciones Vectoriales {#2-operaciones-vectoriales}

### 2.1 Suma de Vectores

```
┌─────────────────────────────────────────────────────────────────┐
│  SUMA: Componente a componente                                  │
│                                                                 │
│  [1, 2, 3] + [4, 5, 6] = [1+4, 2+5, 3+6] = [5, 7, 9]            │
│                                                                 │
│  Geométricamente: "poner una flecha al final de otra"           │
└─────────────────────────────────────────────────────────────────┘
```

```python
def add_vectors(v1: Vector, v2: Vector) -> Vector:
    """Add two vectors component-wise.
    
    Args:
        v1: First vector.
        v2: Second vector (must have same dimension as v1).
    
    Returns:
        New vector with sum of corresponding components.
    
    Raises:
        ValueError: If vectors have different dimensions.
    
    Example:
        >>> add_vectors([1, 2, 3], [4, 5, 6])
        [5, 7, 9]
    """
    if len(v1) != len(v2):
        raise ValueError(f"Dimension mismatch: {len(v1)} vs {len(v2)}")
    
    return [a + b for a, b in zip(v1, v2)]
```

### 2.2 Resta de Vectores

```python
def subtract_vectors(v1: Vector, v2: Vector) -> Vector:
    """Subtract v2 from v1 component-wise.
    
    Example:
        >>> subtract_vectors([5, 7, 9], [4, 5, 6])
        [1, 2, 3]
    """
    if len(v1) != len(v2):
        raise ValueError(f"Dimension mismatch: {len(v1)} vs {len(v2)}")
    
    return [a - b for a, b in zip(v1, v2)]
```

### 2.3 Multiplicación por Escalar

```
┌─────────────────────────────────────────────────────────────────┐
│  ESCALAR × VECTOR: Multiplica cada componente                   │
│                                                                 │
│  3 × [1, 2, 3] = [3×1, 3×2, 3×3] = [3, 6, 9]                    │
│                                                                 │
│  Geométricamente: "estirar" o "encoger" la flecha               │
│  - Escalar > 1: estira                                          │
│  - 0 < Escalar < 1: encoge                                      │
│  - Escalar < 0: invierte dirección                              │
└─────────────────────────────────────────────────────────────────┘
```

```python
def scalar_multiply(scalar: float, v: Vector) -> Vector:
    """Multiply a vector by a scalar.
    
    Example:
        >>> scalar_multiply(3, [1, 2, 3])
        [3, 6, 9]
        >>> scalar_multiply(0.5, [4, 6])
        [2.0, 3.0]
    """
    return [scalar * component for component in v]
```

---

## 3. Producto Punto y Norma {#3-producto-punto}

### 3.1 Producto Punto (Dot Product)

```
┌─────────────────────────────────────────────────────────────────┐
│  PRODUCTO PUNTO: Suma de productos componente a componente      │
│                                                                 │
│  [1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32     │
│                                                                 │
│  SIGNIFICADO GEOMÉTRICO:                                        │
│  v1 · v2 = |v1| × |v2| × cos(θ)                                 │
│                                                                 │
│  Donde θ es el ángulo entre los vectores.                       │
│                                                                 │
│  Si cos(θ) = 1 (θ = 0°): vectores paralelos, misma dirección    │
│  Si cos(θ) = 0 (θ = 90°): vectores perpendiculares              │
│  Si cos(θ) = -1 (θ = 180°): direcciones opuestas                │
│                                                                 │
│  EN ARCHIMEDES: Mide qué tan "similares" son dos documentos     │
└─────────────────────────────────────────────────────────────────┘
```

```python
def dot_product(v1: Vector, v2: Vector) -> float:
    """Compute dot product of two vectors.
    
    Also known as inner product or scalar product.
    
    Args:
        v1: First vector.
        v2: Second vector (same dimension as v1).
    
    Returns:
        Scalar result of dot product.
    
    Example:
        >>> dot_product([1, 2, 3], [4, 5, 6])
        32
        >>> dot_product([1, 0], [0, 1])  # Perpendicular
        0
    """
    if len(v1) != len(v2):
        raise ValueError(f"Dimension mismatch: {len(v1)} vs {len(v2)}")
    
    return sum(a * b for a, b in zip(v1, v2))
```

### 3.2 Norma (Magnitud)

```
┌─────────────────────────────────────────────────────────────────┐
│  NORMA = Longitud del vector                                    │
│                                                                 │
│  ||v|| = √(v₁² + v₂² + ... + vₙ²)                               │
│                                                                 │
│  Ejemplo: ||[3, 4]|| = √(9 + 16) = √25 = 5                      │
│                                                                 │
│  Nota: ||v||² = v · v  (producto punto consigo mismo)           │
└─────────────────────────────────────────────────────────────────┘
```

```python
import math


def magnitude(v: Vector) -> float:
    """Compute the magnitude (length/norm) of a vector.
    
    Also known as Euclidean norm or L2 norm.
    
    Formula: ||v|| = sqrt(v1² + v2² + ... + vn²)
    
    Example:
        >>> magnitude([3, 4])
        5.0
        >>> magnitude([1, 0, 0])
        1.0
    """
    return math.sqrt(sum(component ** 2 for component in v))


def magnitude_squared(v: Vector) -> float:
    """Compute squared magnitude (avoids sqrt for comparisons).
    
    Useful when you only need to compare magnitudes.
    """
    return sum(component ** 2 for component in v)
```

### 3.3 Normalización (Vector Unitario)

```python
def normalize(v: Vector) -> Vector:
    """Return unit vector (magnitude = 1) in same direction.
    
    Formula: v̂ = v / ||v||
    
    Example:
        >>> normalize([3, 4])
        [0.6, 0.8]  # magnitude = 1.0
    
    Raises:
        ValueError: If vector has zero magnitude.
    """
    mag = magnitude(v)
    
    if mag == 0:
        raise ValueError("Cannot normalize zero vector")
    
    return [component / mag for component in v]
```

### 3.4 Similitud de Coseno (¡Crucial para Archimedes!)

```
┌─────────────────────────────────────────────────────────────────┐
│  SIMILITUD DE COSENO                                            │
│                                                                 │
│                    v1 · v2                                      │
│  cos(θ) = ─────────────────────                                 │
│            ||v1|| × ||v2||                                      │
│                                                                 │
│  Resultado:                                                     │
│  • 1: Vectores idénticos en dirección (muy similares)           │
│  • 0: Vectores perpendiculares (nada en común)                  │
│  • -1: Direcciones opuestas (para TF-IDF, raro)                 │
│                                                                 │
│  EN ARCHIMEDES: Documentos con palabras similares tendrán       │
│  coseno cercano a 1                                             │
└─────────────────────────────────────────────────────────────────┘
```

```python
def cosine_similarity(v1: Vector, v2: Vector) -> float:
    """Compute cosine similarity between two vectors.
    
    Measures the cosine of the angle between vectors.
    
    Returns:
        Value between -1 and 1 (usually 0 to 1 for TF-IDF).
        1 = identical direction, 0 = perpendicular.
    
    Example:
        >>> cosine_similarity([1, 0], [1, 0])
        1.0
        >>> cosine_similarity([1, 0], [0, 1])
        0.0
        >>> cosine_similarity([1, 1], [1, 1])
        1.0
    """
    if len(v1) != len(v2):
        raise ValueError(f"Dimension mismatch: {len(v1)} vs {len(v2)}")
    
    dot = dot_product(v1, v2)
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    
    # Handle zero vectors
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot / (mag1 * mag2)
```

---

## 4. Matrices como Listas de Listas {#4-matrices}

### 4.1 Representación

```python
# Matriz = lista de filas, cada fila es un vector
Matrix = list[list[float]]

# Ejemplo: matriz 2x3 (2 filas, 3 columnas)
m: Matrix = [
    [1, 2, 3],  # Fila 0
    [4, 5, 6]   # Fila 1
]

# Acceso: m[fila][columna]
# m[0][0] = 1, m[0][2] = 3, m[1][1] = 5
```

### 4.2 Funciones de Información

```python
def get_shape(m: Matrix) -> tuple[int, int]:
    """Return (rows, columns) of matrix.
    
    Example:
        >>> get_shape([[1, 2, 3], [4, 5, 6]])
        (2, 3)
    """
    if not m:
        return (0, 0)
    return (len(m), len(m[0]))


def get_element(m: Matrix, row: int, col: int) -> float:
    """Get element at (row, col)."""
    return m[row][col]


def get_row(m: Matrix, row: int) -> Vector:
    """Get a row as vector."""
    return m[row].copy()


def get_column(m: Matrix, col: int) -> Vector:
    """Get a column as vector."""
    return [row[col] for row in m]
```

---

## 5. Operaciones Matriciales {#5-operaciones-matriciales}

### 5.1 Suma de Matrices

```python
def add_matrices(m1: Matrix, m2: Matrix) -> Matrix:
    """Add two matrices element-wise.
    
    Matrices must have same dimensions.
    
    Example:
        >>> add_matrices([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[6, 8], [10, 12]]
    """
    rows1, cols1 = get_shape(m1)
    rows2, cols2 = get_shape(m2)
    
    if (rows1, cols1) != (rows2, cols2):
        raise ValueError(f"Shape mismatch: {(rows1, cols1)} vs {(rows2, cols2)}")
    
    return [
        [m1[i][j] + m2[i][j] for j in range(cols1)]
        for i in range(rows1)
    ]
```

### 5.2 Multiplicación por Escalar

```python
def scalar_multiply_matrix(scalar: float, m: Matrix) -> Matrix:
    """Multiply matrix by scalar.
    
    Example:
        >>> scalar_multiply_matrix(2, [[1, 2], [3, 4]])
        [[2, 4], [6, 8]]
    """
    return [
        [scalar * element for element in row]
        for row in m
    ]
```

### 5.3 Transpuesta

```
┌─────────────────────────────────────────────────────────────────┐
│  TRANSPUESTA: Intercambiar filas y columnas                     │
│                                                                 │
│  Original (2x3):        Transpuesta (3x2):                      │
│  [1, 2, 3]              [1, 4]                                  │
│  [4, 5, 6]              [2, 5]                                  │
│                         [3, 6]                                  │
│                                                                 │
│  Fórmula: T[i][j] = M[j][i]                                     │
└─────────────────────────────────────────────────────────────────┘
```

```python
def transpose(m: Matrix) -> Matrix:
    """Transpose a matrix (swap rows and columns).
    
    Example:
        >>> transpose([[1, 2, 3], [4, 5, 6]])
        [[1, 4], [2, 5], [3, 6]]
    """
    if not m:
        return []
    
    rows, cols = get_shape(m)
    
    return [
        [m[i][j] for i in range(rows)]
        for j in range(cols)
    ]
```

### 5.4 Producto Matriz-Vector

```
┌─────────────────────────────────────────────────────────────────┐
│  MATRIZ × VECTOR                                                │
│                                                                 │
│  [1, 2, 3]     [1]     [1×1 + 2×2 + 3×3]     [14]               │
│  [4, 5, 6]  ×  [2]  =  [4×1 + 5×2 + 6×3]  =  [32]               │
│               [3]                                               │
│                                                                 │
│  Cada elemento del resultado = producto punto de una fila       │
│  con el vector                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
def matrix_vector_multiply(m: Matrix, v: Vector) -> Vector:
    """Multiply matrix by vector.
    
    Matrix must have columns = len(v).
    Result has length = rows of matrix.
    
    Example:
        >>> matrix_vector_multiply([[1, 2, 3], [4, 5, 6]], [1, 2, 3])
        [14, 32]
    """
    rows, cols = get_shape(m)
    
    if cols != len(v):
        raise ValueError(f"Dimension mismatch: matrix has {cols} cols, vector has {len(v)} elements")
    
    return [dot_product(row, v) for row in m]
```

### 5.5 Producto Matriz-Matriz (Opcional pero útil)

```python
def matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix:
    """Multiply two matrices.
    
    m1 must have cols = m2 rows.
    Result shape: (m1 rows, m2 cols).
    
    Example:
        >>> matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        [[19, 22], [43, 50]]
    """
    rows1, cols1 = get_shape(m1)
    rows2, cols2 = get_shape(m2)
    
    if cols1 != rows2:
        raise ValueError(f"Cannot multiply: m1 has {cols1} cols, m2 has {rows2} rows")
    
    # Result[i][j] = dot product of m1 row i with m2 column j
    result = []
    for i in range(rows1):
        row = []
        for j in range(cols2):
            val = sum(m1[i][k] * m2[k][j] for k in range(cols1))
            row.append(val)
        result.append(row)
    
    return result
```

---

## ⚠️ Errores Comunes

### Error 1: Modificar vectores originales

```python
# ❌ Modifica el vector original
def normalize_bad(v: Vector) -> Vector:
    mag = magnitude(v)
    for i in range(len(v)):
        v[i] /= mag  # Modifica v!
    return v

# ✅ Crear nuevo vector
def normalize_good(v: Vector) -> Vector:
    mag = magnitude(v)
    return [x / mag for x in v]
```

### Error 2: División por cero en normalización

```python
# ❌ Falla con vector cero
def normalize_bad(v):
    mag = magnitude(v)
    return [x / mag for x in v]  # ZeroDivisionError!

# ✅ Manejar caso especial
def normalize_good(v):
    mag = magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize zero vector")
    return [x / mag for x in v]
```

### Error 3: Comparar floats con ==

```python
# ❌ Puede fallar por precisión de punto flotante
if magnitude(v) == 1.0:
    print("Unit vector")

# ✅ Usar tolerancia
if abs(magnitude(v) - 1.0) < 1e-9:
    print("Unit vector")
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 10.1: Operaciones Vectoriales
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-101)

### Ejercicio 10.2: Producto Punto y Norma
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-102)

### Ejercicio 10.3: Similitud de Coseno
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-103)

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [3Blue1Brown: Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Video | 🔴 Obligatorio |
| [Mathematics for ML: Linear Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning) | Curso | 🔴 Obligatorio |
| [Khan Academy Linear Algebra](https://www.khanacademy.org/math/linear-algebra) | Curso | 🟡 Recomendado |

---

## 🔗 Referencias del Glosario

- [Vector](GLOSARIO.md#vector)
- [Matriz](GLOSARIO.md#matriz)
- [Producto Punto](GLOSARIO.md#producto-punto)
- [Norma](GLOSARIO.md#norma)
- [Similitud de Coseno](GLOSARIO.md#similitud-coseno)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [09_BINARY_SEARCH](09_BINARY_SEARCH.md) | [00_INDICE](00_INDICE.md) | [11_TFIDF_COSENO](11_TFIDF_COSENO.md) |
