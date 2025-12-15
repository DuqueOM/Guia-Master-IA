# Module 02 - Linear Algebra for Machine Learning

> **🎯 Goal:** Master vectors, matrices, norms, and eigenvalues for ML
> **Phase:** 1 - Math Foundations | **Weeks 3–5**
> **Prerequisites:** Module 01 (Scientific Python with NumPy)

---

<a id="m02-0"></a>

## 🧭 How to use this module (0→100 mode)

**Purpose:** be able to read and write the mathematical “grammar” of ML:

- `ŷ = Xθ` (supervised)
- projections and bases (PCA)
- decompositions (SVD)

### Learning objectives (measurable)

By the end of this module you can:

- **Use** dot product and cosine similarity to measure similarity between vectors.
- **Implement** norms and distances (L1/L2/L∞) and explain their role in regularization.
- **Reason** about shapes in matrix operations (avoid silent bugs).
- **Explain** eigenvalues/eigenvectors as “principal directions” and connect them to PCA.
- **Explain** SVD and why it is the preferred, numerically stable method for PCA.

### Prerequisites

- `Module 01` (NumPy, vectorization, shapes).

Quick links:

- [GLOSSARY: Dot Product](GLOSARIO.md#dot-product)
- [GLOSSARY: Matrix Multiplication](GLOSARIO.md#matrix-multiplication)
- [GLOSSARY: L1 Norm](GLOSARIO.md#l1-norm-manhattan)
- [GLOSSARY: L2 Norm](GLOSARIO.md#l2-norm-euclidean)
- [GLOSSARY: SVD](GLOSARIO.md#svd-singular-value-decomposition)
- [RECURSOS.md](RECURSOS.md)

### Integration with Plan v4/v5

- Daily shape reinforcement: `../../study_tools/DRILL_DIMENSIONES_NUMPY.md`
- Simulations: `../../study_tools/SIMULACRO_EXAMEN_TEORICO.md`
- Evaluation (rubric): [study_tools/RUBRICA_v1.md](../../study_tools/RUBRICA_v1.md) (scope `M02` in `rubrica.csv`)
- Full protocol:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Resources (when to use them)

| Priority | Resource | When to use it in this module | Why |
|----------|---------|------------------------------|----------|
| **Required** | `../../study_tools/DRILL_DIMENSIONES_NUMPY.md` | Any time a multiplication/projection changes the shape unexpectedly | Avoid silent shape bugs |
| **Required** | [3Blue1Brown: Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Weeks 3–4, before matrices/eigen/SVD (and if you feel “mechanical” using `@`/`eig`) | Build strong geometric intuition |
| **Recommended** | Interactive plots in Jupyter (`matplotlib` + `plotly` + `ipywidgets`) | Weeks 3–5, when studying linear transforms / eigenvectors | See grids deform and build intuition by experimentation |
| **Recommended** | [Mathematics for ML: Linear Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning) | Week 5, when entering eigenvalues/SVD | Formalize with guided exercises |
| **Optional** | [Mathematics for ML (book)](https://mml-book.github.io/) | After eigen/SVD (for depth) | Go deeper into notation and derivations |
| **Optional** | [RECURSOS.md](RECURSOS.md) | When planning reinforcement for PCA (Module 06) | Choose extra practice materials |

## 🧠 Why Linear Algebra for ML?

### Vector space intuition (the missing link)

If you only think of matrices as “tables of numbers”, you might be able to write `np.linalg.eig(A)`, but you won’t understand what you’re computing. The central idea is:

> A matrix is a **function** that transforms space: it stretches, rotates, shears, or collapses it.

#### 1) Vectors as movement (not as points)

A vector `v = [x, y]` can be seen as a **displacement**:

- start at the origin
- walk `x` in X
- walk `y` in Y

Suggested visualization (draw it): vector addition as “walking two movements in a row”.

#### 2) Matrices as grid deformation

Imagine a square grid on the plane. Multiplying by a matrix `A` deforms the whole grid:

- parallel lines stay parallel
- the origin does not move
- squares become parallelograms

Mental examples:

- `[[2, 0], [0, 1]]` stretches space in X by a factor of 2.
- If `det(A) = 0`, you collapse the 2D plane into a line (or a point): you lose dimension.

This explains why a matrix with determinant 0 is not invertible: you cannot “un-collapse” a line to get back a plane.

#### 3) Dot product as a “shadow” (projection)

Geometric reading: `a·b = ||a|| ||b|| cos(θ)` measures how much of `a` points in the direction of `b`.

Direct ML application:

- `w·x` measures how aligned your input `x` is with pattern `w`.

#### 4) Eigenvectors: axes that do not rotate

When a matrix rotates/stretches space, almost all vectors change direction. But some vectors are “stubborn”: they only scale.

- **Eigenvector:** a direction that does not rotate under `A`.
- **Eigenvalue:** how much that direction stretches/shrinks.

Suggested visualization (for PCA): imagine aligning a “camera” with those natural axes.

In PCA (Module 06), those axes (eigenvectors of the covariance) are the directions with the most variance.

### Direct connections with the Pathway

| Concept | Use in ML | Pathway course |
|----------|-----------|-------------------|
| **Dot product** | Similarity, predictions | Supervised Learning |
| **L1/L2 norms** | Regularization, distances | Supervised Learning |
| **Eigenvalues** | PCA, dimensionality reduction | Unsupervised Learning |
| **Matrix multiplication** | Forward pass in networks | Deep Learning |
| **SVD** | Compression, PCA | Unsupervised Learning |

### The math behind ML

```
# Common places where linear algebra appears in ML (compact notation)
# Note: ŷ is the prediction; σ is a non-linear activation (e.g., sigmoid/ReLU)
Linear Regression:     ŷ = Xθ                 (matrix-vector multiplication: features X, weights θ)
Logistic Regression:   ŷ = σ(Xθ)              (same Xθ, then apply σ)
Neural Network:        ŷ = σ(W₃σ(W₂σ(W₁x)))   (stacked linear layers + activations)
PCA:                   X_reduced = XV         (project X onto eigenvectors V)
```

---

## 📚 Module content

### Week 3: Vectors and basic operations
### Week 4: Norms and distances
### Week 5: Matrices, eigenvalues, and SVD

---

## 💻 Part 1: Vectors

### 1.1 Definición Geométrica y Algebraica

```python
import numpy as np  # NumPy: represent vectors as arrays and generate random data
import matplotlib.pyplot as plt  # Matplotlib: visualize 2D vectors

# A vector is an ordered list of numbers
# Geometrically: an arrow with direction and magnitude

# Vector in R² (2 dimensions)
v = np.array([3, 4])  # 2D vector: (x=3, y=4)

# Vector in R³ (3 dimensions)
w = np.array([1, 2, 3])  # 3D vector: (x=1, y=2, z=3)

# Vector in R^n (n dimensions) - common in ML
# Example: a 28x28 image has 784 features once flattened
image_vector = np.random.randn(784)  # Simulated feature vector

# 2D visualization
def plot_vector(v, origin=[0, 0], color='blue', label=None):
    """Draw a vector starting at the given origin."""
    plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color=color, label=label)  # Arrow

plt.figure(figsize=(8, 8))  # Square figure for clean geometry
plot_vector(np.array([3, 4]), color='blue', label='v = [3, 4]')  # First vector
plot_vector(np.array([2, 1]), color='red', label='w = [2, 1]')  # Second vector for comparison
plt.xlim(-1, 5)  # X-axis range
plt.ylim(-1, 5)  # Y-axis range
plt.grid(True)  # Grid helps interpret coordinates
plt.axhline(y=0, color='k', linewidth=0.5)  # Draw x-axis
plt.axvline(x=0, color='k', linewidth=0.5)  # Draw y-axis
plt.legend()  # Show labels
plt.title('Vectors in R²')  # Plot title
plt.show()  # Render the figure
```

### 1.2 Operaciones con Vectores

#### Formalización: Producto punto como “sombra/proyección”

**Intuición:** el producto punto te dice cuánto del vector `a` está “apuntando” en la dirección de `b`. Si imaginas una linterna proyectando `a` sobre la línea de `b`, el producto punto está relacionado con el tamaño de esa **sombra**.

Dos fórmulas que debes dominar:

```
a·b = ||a|| · ||b|| · cos(θ)

proj_b(a) = (a·b / b·b) · b
```

Interpretación rápida:

- si `a·b` es grande y positivo → apuntan parecido
- si `a·b ≈ 0` → son casi ortogonales (poca “sombra”)
- si `a·b` es negativo → apuntan en sentidos opuestos

**Por qué importa en ML:** muchas predicciones son de la forma `ŷ = Xθ` (sumas de productos punto). Entenderlo geométricamente evita que el modelo sea “caja negra”.

```python
import numpy as np  # NumPy for vector operations and dot products

# Vectores de ejemplo
a = np.array([1, 2, 3])  # Example vector a (e.g., features)
b = np.array([4, 5, 6])  # Example vector b (e.g., weights)

# === SUMA DE VECTORES ===
# (a + b)ᵢ = aᵢ + bᵢ
suma = a + b
print(f"a + b = {suma}")  # [5, 7, 9]

# === RESTA DE VECTORES ===
resta = a - b
print(f"a - b = {resta}")  # [-3, -3, -3]

# === MULTIPLICACIÓN POR ESCALAR ===
# (c·a)ᵢ = c·aᵢ
escalar = 2 * a
print(f"2·a = {escalar}")  # [2, 4, 6]

# === PRODUCTO PUNTO (DOT PRODUCT) ===
# a·b = Σᵢ aᵢ·bᵢ
# Resultado: escalar
dot = np.dot(a, b)
print(f"a·b = {dot}")  # 1*4 + 2*5 + 3*6 = 32

# Alternativamente:
dot_alt = a @ b  # @ is the dot product for 1D vectors (equivalent to np.dot)
dot_sum = np.sum(a * b)  # Element-wise multiply then sum (manual dot product)
print(f"Verificación: {dot_alt}, {dot_sum}")  # Sanity check: all methods should match
```

### 1.3 Interpretación Geométrica del Producto Punto

```python
import numpy as np  # NumPy for dot, norms, arccos, and clipping

def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula el ángulo entre dos vectores.

    cos(θ) = (a·b) / (||a|| ||b||)
    """
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # cos(θ) = (a·b)/(||a|| ||b||)
    # Clip to avoid floating point drift outside [-1, 1] (arccos domain)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_rad = np.arccos(cos_theta)  # Convert cosine to angle in radians
    theta_deg = np.degrees(theta_rad)  # Convert radians to degrees (human-friendly)
    return theta_deg  # Return the angle in degrees

# Ejemplos
v1 = np.array([1, 0])  # x-axis
v2 = np.array([0, 1])  # y-axis (orthogonal to x)
v3 = np.array([1, 1])  # 45° direction
v4 = np.array([-1, 0])  # opposite to x-axis

print(f"Ángulo entre [1,0] y [0,1]: {angle_between_vectors(v1, v2):.0f}°")  # 90°  # Orthogonal vectors
print(f"Ángulo entre [1,0] y [1,1]: {angle_between_vectors(v1, v3):.0f}°")  # 45°  # Diagonal vs x-axis
print(f"Ángulo entre [1,0] y [-1,0]: {angle_between_vectors(v1, v4):.0f}°") # 180° # Opposite directions

# Interpretación para ML:
# - Producto punto alto → vectores similares (mismo "sentido")
# - Producto punto ≈ 0 → vectores ortogonales (independientes)
# - Producto punto negativo → vectores opuestos
```

### 1.4 Proyección de Vectores

```python
import numpy as np  # NumPy for dot products and working with vectors as arrays

def project(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Proyecta el vector a sobre el vector b.

    proj_b(a) = (a·b / b·b) · b

    Útil para: PCA, regresión, descomposición de señales
    """
    scalar = np.dot(a, b) / np.dot(b, b)  # Projection scalar: (a·b)/(b·b)
    return scalar * b  # Scale b to get the projected vector (same direction as b)

# Ejemplo
a = np.array([3, 4])  # Vector to project
b = np.array([1, 0])  # Unit vector along x (projection direction)

proyeccion = project(a, b)  # Compute the projection of a onto b
print(f"Proyección de {a} sobre {b}: {proyeccion}")  # [3, 0]  # Keeps only the x-component

# La proyección nos da "cuánto" de a está en la dirección de b
```

---

## 💻 Part 2: Norms and distances

### 2.1 Norma L2 (Euclidiana)

```python
import numpy as np  # NumPy for vectorized math and reference implementation (linalg.norm)

def l2_norm(x: np.ndarray) -> float:
    """
    Norma L2 (Euclidiana): longitud del vector.

    ||x||₂ = √(Σᵢ xᵢ²)

    Uso en ML:
    - Regularización Ridge
    - Normalización de vectores
    - Distancia euclidiana
    """
    return np.sqrt(np.sum(x ** 2))  # sqrt(sum(x_i^2)): square -> sum -> square root

# Equivalente en NumPy
x = np.array([3, 4])  # Classic 3-4-5 vector
print(f"||x||₂ = {l2_norm(x)}")           # 5.0  # Our implementation
print(f"NumPy:  {np.linalg.norm(x)}")     # 5.0  # NumPy default is L2
print(f"NumPy:  {np.linalg.norm(x, 2)}")  # 5.0 (especificando ord=2)

# Vector unitario (normalizado)
def normalize(x: np.ndarray) -> np.ndarray:
    """Convierte vector a longitud 1."""
    return x / np.linalg.norm(x)  # Divide by the norm to make ||x|| = 1 (unit vector)

x_unit = normalize(x)  # Normalize x
print(f"Unitario: {x_unit}")  # [0.6, 0.8]  # Same direction, scaled length
print(f"Norma del unitario: {np.linalg.norm(x_unit)}")  # 1.0  # Sanity check
```

### 2.2 Norma L1 (Manhattan)

```python
import numpy as np  # NumPy for abs/sum and norm computations

def l1_norm(x: np.ndarray) -> float:
    """
    L1 norm (Manhattan): sum of absolute values.

    ||x||₁ = Σᵢ |xᵢ|

    ML use:
    - Lasso regularization (promotes sparsity)
    - Robustness to outliers
    """
    return np.sum(np.abs(x))  # Sum of absolute values: Σ|x_i|

x = np.array([3, -4, 5])  # Mixed-sign vector to show abs() effect
print(f"||x||₁ = {l1_norm(x)}")                  # 12  # |3|+|−4|+|5| = 12
print(f"NumPy:  {np.linalg.norm(x, 1)}")         # 12.0  # Validation with NumPy

# L1 vs L2 comparison (intuition)
# L1 penalizes all magnitudes linearly
# L2 penalizes large magnitudes more (because of the square)
```

### 2.3 Norma L∞ (Máximo)

```python
import numpy as np  # NumPy for abs/max and infinity norm

def linf_norm(x: np.ndarray) -> float:
    """
    L∞ norm (max norm): maximum absolute value.

    ||x||∞ = max(|xᵢ|)
    """
    return np.max(np.abs(x))  # max(|x_i|)

x = np.array([3, -7, 5])  # Dominant magnitude is 7 (from -7)
print(f"||x||∞ = {linf_norm(x)}")            # 7
print(f"NumPy:  {np.linalg.norm(x, np.inf)}") # 7.0  # Validation with NumPy
```

### 2.4 Distancia Euclidiana

```python
import numpy as np  # NumPy for vector norms and efficient distance computations

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance between two points.

    d(a, b) = ||a - b||₂ = √(Σᵢ (aᵢ - bᵢ)²)

    ML use:
    - KNN (k-nearest neighbors)
    - K-Means (cluster assignment)
    - Similarity / nearest-neighbor search
    """
    return np.linalg.norm(a - b)  # ||a-b||₂

# Ejemplo
p1 = np.array([0, 0])  # Point 1
p2 = np.array([3, 4])  # Point 2 (3-4-5 triangle)
print(f"Distancia: {euclidean_distance(p1, p2)}")  # 5.0

# For many points (efficient)
def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise distance matrix between all rows in X.
    X: (n_samples, n_features)
    Returns: (n_samples, n_samples)
    """
    # Using broadcasting + the identity:
    # ||a - b||² = ||a||² + ||b||² - 2(a·b)
    sq_norms = np.sum(X ** 2, axis=1)  # Row-wise squared norms: ||x_i||^2
    distances_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T  # All pairwise squared distances
    distances_sq = np.maximum(distances_sq, 0)  # Guard against tiny negatives from floating point error
    return np.sqrt(distances_sq)  # Convert squared distances to distances

# Test
X = np.array([[0, 0], [3, 4], [1, 1]])  # 3 points in 2D
D = pairwise_distances(X)  # Pairwise distance matrix
print("Matriz de distancias:")
print(D)
```

### 2.5 Similitud Coseno

```python
import numpy as np  # NumPy for dot product and vector norms

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity: measures the angle between vectors.

    sim(a, b) = (a·b) / (||a|| ||b||)

    Rango: [-1, 1]
    - 1: vectores idénticos (misma dirección)
    - 0: vectores ortogonales
    - -1: vectores opuestos

    ML use:
    - NLP (document similarity)
    - Recommender systems
    - Embeddings / representation learning
    """
    dot_product = np.dot(a, b)  # Alignment between vectors
    norm_a = np.linalg.norm(a)  # ||a||
    norm_b = np.linalg.norm(b)  # ||b||

    if norm_a == 0 or norm_b == 0:  # Edge case: zero vector => avoid division by 0
        return 0.0

    return dot_product / (norm_a * norm_b)  # cos(θ) = (a·b)/(||a|| ||b||)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia coseno = 1 - similitud coseno."""
    return 1 - cosine_similarity(a, b)  # Turn similarity into a distance-like quantity

# Ejemplos
v1 = np.array([1, 0, 0])  # Base vector
v2 = np.array([1, 0, 0])  # Identical to v1
v3 = np.array([0, 1, 0])  # Orthogonal to v1
v4 = np.array([-1, 0, 0])  # Opposite direction to v1

print(f"Similitud (idénticos):  {cosine_similarity(v1, v2)}")   # 1.0
print(f"Similitud (ortogonales): {cosine_similarity(v1, v3)}")  # 0.0
print(f"Similitud (opuestos):    {cosine_similarity(v1, v4)}")  # -1.0
```

---

## 💻 Part 3: Matrices

### 3.1 Operaciones Básicas

```python
import numpy as np  # NumPy for creating matrices (2D arrays) and doing matrix operations

# Create matrices
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # Shape: (2, 3)  # 2 rows, 3 columns

B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])  # Shape: (3, 2)  # 3 rows, 2 columns

# === SUMA Y RESTA ===
# Solo para matrices del mismo shape
C = np.array([[1, 2, 3], [4, 5, 6]])
print(f"A + C =\n{A + C}")  # Element-wise addition (requires same shape)

# === MULTIPLICACIÓN POR ESCALAR ===
print(f"2·A =\n{2 * A}")  # Scalar multiplication (scales every entry)

# === PRODUCTO MATRICIAL ===
# (m×n) @ (n×p) = (m×p)
# A(2×3) @ B(3×2) = (2×2)
AB = A @ B
print(f"A @ B =\n{AB}")  # Matrix product: (2x3) @ (3x2) -> (2x2)
# [[58, 64],
#  [139, 154]]

# Verificación manual del elemento [0,0]:
# 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58 ✓

# === TRANSPUESTA ===
print(f"A^T =\n{A.T}")  # Transpose swaps rows/columns
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

### 3.2 Matriz por Vector (Transformación Lineal)

```python
import numpy as np  # NumPy for trig functions and matrix-vector multiplication

# La multiplicación matriz-vector es una TRANSFORMACIÓN LINEAL
# y = Ax transforma el vector x al espacio de y

# Ejemplo: Rotación 90° en R²
theta = np.pi / 2  # 90 degrees (in radians)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])  # 2x2 rotation matrix

x = np.array([1, 0])  # Input vector
y = R @ x  # Apply the linear transform
print(f"Rotar [1,0] 90°: {y}")  # [0, 1]  # Expected: x-axis -> y-axis

# En ML: y = Wx + b (capa de red neuronal)
W = np.random.randn(10, 784)  # Pesos: 784 entradas → 10 salidas
b = np.random.randn(10)         # Bias (one per output unit)
x = np.random.randn(784)        # Input (flattened image)

y = W @ x + b  # Layer output: (10,784)@(784,) + (10,) -> (10,)
print(f"Shape de y: {y.shape}")  # (10,)
```

### 3.3 Matriz Inversa

```python
import numpy as np  # NumPy for matrix inverse and linear algebra exceptions

def safe_inverse(A: np.ndarray) -> np.ndarray:
    """
    Calcula la inversa de A si existe.
    A @ A⁻¹ = A⁻¹ @ A = I

    Uso en ML:
    - Solución cerrada de regresión lineal: θ = (X^T X)⁻¹ X^T y
    - Whitening en PCA
    """
    try:
        return np.linalg.inv(A)  # Compute A^{-1} (A must be square and non-singular)
    except np.linalg.LinAlgError:
        print("Matriz no invertible (singular)")  # Singular => det(A)=0 (no inverse)
        return None  # Signal failure

# Ejemplo
A = np.array([
    [4, 7],
    [2, 6]
])

A_inv = safe_inverse(A)
print(f"A⁻¹ =\n{A_inv}")  # Print the inverse (if it exists)

# Verificar: A @ A⁻¹ = I
identity = A @ A_inv
print(f"A @ A⁻¹ ≈ I:\n{np.round(identity, 10)}")  # Round to highlight identity despite float error

# NOTA: En ML, evita calcular inversas cuando sea posible
# Usa np.linalg.solve() en su lugar (más estable numéricamente)
```

### 3.4 Solución de Sistemas Lineales

```python
import numpy as np  # NumPy for solving linear systems with solve()

# Sistema: Ax = b
# Encontrar x

A = np.array([
    [3, 1],
    [1, 2]
])  # Coefficient matrix
b = np.array([9, 8])  # Right-hand side vector

# Método 1: Inversa (NO RECOMENDADO)
x_inv = np.linalg.inv(A) @ b  # Works, but less stable and often slower

# Método 2: solve (RECOMENDADO - más estable)
x_solve = np.linalg.solve(A, b)  # Preferred: solves Ax=b directly

print(f"Solución: x = {x_solve}")  # [2, 3]

# Verificar
print(f"A @ x = {A @ x_solve}")    # [9, 8] ✓  # Check that Ax reproduces b
```

---

## 💻 Part 4: Eigenvalues and eigenvectors

### 4.1 Concepto

#### Intuición física: el globo terráqueo (eigenvector como eje)

Imagina que tomas un globo terráqueo y lo haces girar.

- Casi todos los puntos de la superficie se mueven.
- Pero hay una línea “especial” que no cambia de dirección: el eje que conecta los polos.

Ese eje es la metáfora del **eigenvector**: una dirección que la transformación “respeta” (no la gira, solo la escala).

El **eigenvalue** te dice cuánto se estira/encoge esa dirección.

#### Código generador de intuición (obligatorio): rejilla deformada por una matriz

Para dejar de ver matrices como tablas y empezar a verlas como “máquinas que deforman el espacio”, usa el script:

- [`visualizations/viz_transformations.py`](../../visualizations/viz_transformations.py)

Ejecución:

```bash
# Run the script that draws a grid and shows how a matrix transforms space
python3 visualizations/viz_transformations.py  # Requires plotting dependencies (e.g., matplotlib)
```

Ejercicio:

- prueba matrices como `[[2, 0], [0, 1]]`, `[[0, -1], [1, 0]]`, `[[1, 1], [0, 1]]`
- observa cómo se deforma la rejilla y cómo se comporta un eigenvector (si existe en R²)

#### Worked example: Eigenvalues de una matriz 2×2 (a mano)

Antes de usar `np.linalg.eig`, hazlo una vez “a mano” para fijar la idea.

Para:

```
A = [[2, 1],
     [1, 2]]
```

1) Buscamos `λ` tal que exista un `v ≠ 0` cumpliendo `Av = λv`. Eso equivale a:

```
(A - λI)v = 0
```

2) Para que haya solución no trivial, el determinante debe ser 0:

```
det(A - λI) = 0

det([[2-λ, 1],
     [1, 2-λ]]) = (2-λ)^2 - 1
```

3) Resolver:

```
(2-λ)^2 - 1 = 0
2-λ = ±1
λ ∈ {3, 1}
```

Esto coincide con lo que imprime el código (eigenvalues `[3, 1]`).

```python
import numpy as np  # NumPy for eigendecomposition and matrix operations

"""
EIGENVALUES (Autovalores) y EIGENVECTORS (Autovectores)

Definición: Av = λv
- v: eigenvector (vector que solo se escala, no cambia dirección)
- λ: eigenvalue (factor de escala)

Interpretación:
- Los eigenvectors son las "direcciones principales" de una transformación
- Los eigenvalues indican cuánto se estira/comprime en cada dirección

Uso en ML:
- PCA: eigenvectors de la matriz de covarianza son las componentes principales
- PageRank: eigenvector dominante de la matriz de transición
- Estabilidad de sistemas dinámicos
"""

# Simple example
A = np.array([  # Symmetric 2x2 matrix (nice for intuition)
    [2, 1],
    [1, 2]
])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)  # Returns (λ, V) with A @ V = V @ diag(λ)

print(f"Eigenvalues: {eigenvalues}")     # [3, 1] (scaling factors)
print(f"Eigenvectors:\n{eigenvectors}")  # Columns are eigenvectors (directions)

# Verify: Av = λv
v1 = eigenvectors[:, 0]  # First eigenvector (column 0)
### 4.2 Eigenvalues para PCA

#### Conexión Línea 2: Covarianza como esperanza (estadística)

En estadística, la matriz de covarianza se define conceptualmente como:

```
Cov(X) = E[(X - μ)(X - μ)^T]
```

Este puente es clave para el curso de **Statistical Estimation** (Línea 2): la misma idea de “esperanza” aparece en MLE, varianza, estimadores y pruebas.

```python
import numpy as np  # NumPy for centering, covariance, and eigendecomposition

def pca_via_eigen(X: np.ndarray, n_components: int) -> tuple:
    """
    PCA usando eigendecomposition de la matriz de covarianza.

    Args:
        X: datos (n_samples, n_features)
        n_components: número de componentes a retener

    Returns:
        X_transformed: datos proyectados
        components: eigenvectors (componentes principales)
        explained_variance: varianza explicada por cada componente
    """
    # 1. Center data (subtract mean per feature)
    X_centered = X - np.mean(X, axis=0)

    # 2. Calcular matriz de covarianza
    # Cov = (1/n) X^T X
    n_samples = X.shape[0]  # Number of samples (rows)
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)  # Sample covariance: (1/(n-1)) X^T X

    # 3. Calcular eigenvalues y eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # Eigenvalues ~ variances; eigenvectors ~ directions

    # 4. Ordenar por eigenvalue (mayor a menor)
    idx = np.argsort(eigenvalues)[::-1]  # Sort indices by descending eigenvalue
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. Seleccionar top n_components
    components = eigenvectors[:, :n_components].real  # Keep top components; take real part for stability

    # 6. Proyectar datos
    X_transformed = X_centered @ components  # Project onto principal directions

    # 7. Calcular varianza explicada
    total_variance = np.sum(eigenvalues)  # Total variance
    explained_variance = eigenvalues[:n_components].real / total_variance  # Explained variance ratio

    return X_transformed, components, explained_variance

# Demo
np.random.seed(42)  # Fixed seed for reproducibility
X = np.random.randn(100, 5)  # Synthetic dataset: 100 samples, 5 features

X_pca, components, var_explained = pca_via_eigen(X, n_components=2)

print(f"Shape original: {X.shape}")  # (n_samples, n_features)
print(f"Shape reducido: {X_pca.shape}")  # (n_samples, n_components)
print(f"Varianza explicada: {var_explained}")  # Ratio per component
print(f"Varianza total explicada: {np.sum(var_explained):.2%}")  # Total explained variance
```

### 5.1 Concepto

```python
import numpy as np  # NumPy for SVD (linalg.svd) and reconstruction

"""
SVD: Singular Value Decomposition

A = U Σ V^T

- U: matriz ortogonal (m×m) - vectores singulares izquierdos
- Σ: matriz diagonal (m×n) - valores singulares (σ₁ ≥ σ₂ ≥ ... ≥ 0)
- V^T: matriz ortogonal (n×n) - vectores singulares derechos

Ventajas sobre Eigendecomposition:
- Funciona para CUALQUIER matriz (no solo cuadradas)
- Más estable numéricamente
- Los valores singulares siempre son no-negativos

Uso en ML:
- PCA (método preferido)
- Compresión de imágenes
- Sistemas de recomendación (matrix factorization)
- Regularización (truncated SVD)
"""

# Ejemplo
A = np.array([  # Non-square matrix (SVD works for any shape)
    [1, 2],
    [3, 4],
    [5, 6]
])  # 3×2

U, S, Vt = np.linalg.svd(A, full_matrices=False)  # Economy SVD keeps minimal shapes

print(f"U shape: {U.shape}")   # (3, 2)  # Left singular vectors
print(f"S shape: {S.shape}")   # (2,)    # Singular values (σ)
print(f"Vt shape: {Vt.shape}") # (2, 2)  # Right singular vectors (transposed)

# Reconstruir A
A_reconstructed = U @ np.diag(S) @ Vt  # Rebuild A from U, Σ, V^T
print(f"\n¿A ≈ U Σ V^T? {np.allclose(A, A_reconstructed)}")  # Should be True
```

### 5.2 PCA via SVD (Método Preferido)

```python
import numpy as np  # NumPy for centering data and running SVD

def pca_via_svd(X: np.ndarray, n_components: int) -> tuple:
    """
    PCA usando SVD (más estable que eigendecomposition).

    La relación: si X = UΣV^T, entonces:
    - V contiene las componentes principales
    - Σ²/(n-1) son las varianzas (eigenvalues de X^TX)
    """
    # 1. Center data
    X_centered = X - np.mean(X, axis=0)  # Center features so PCA captures variance, not mean offsets

    # 2. SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # X = U·diag(S)·Vt

    # 3. Principal components (rows of Vt)
    components = Vt[:n_components]

    # 4. Project data
    X_transformed = X_centered @ components.T

    # 5. Explained variance
    variance = (S ** 2) / (X.shape[0] - 1)  # S^2/(n-1) relates to covariance eigenvalues
    explained_variance_ratio = variance[:n_components] / np.sum(variance)

    return X_transformed, components, explained_variance_ratio

# Demo
np.random.seed(42)  # Fixed seed for reproducibility
X = np.random.randn(100, 10)  # Synthetic dataset

X_pca, components, var_ratio = pca_via_svd(X, n_components=3)

print(f"Varianza explicada por componente: {var_ratio}")  # Per-component explained variance ratio
print(f"Varianza total explicada: {np.sum(var_ratio):.2%}")  # Total explained variance
```

### 5.3 Compresión de Imágenes con SVD

```python
import numpy as np  # NumPy for SVD and representing images as arrays

def compress_image_svd(image: np.ndarray, k: int) -> np.ndarray:
    """
    Comprime una imagen usando truncated SVD.

    Args:
        image: matriz 2D (grayscale) o 3D (RGB)
        k: número de valores singulares a retener

    Returns:
        imagen comprimida
    """
    if len(image.shape) == 2:  # 2D case: grayscale image (m×n)
        # Grayscale
        U, S, Vt = np.linalg.svd(image, full_matrices=False)  # SVD of the image matrix
        compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]  # Truncated SVD: keep only k components
        return np.clip(compressed, 0, 255).astype(np.uint8)  # Clip to valid range and cast
    else:
        # RGB: comprimir cada canal
        compressed = np.zeros_like(image)  # Allocate output image
        for i in range(3):  # Loop channels: 0=R, 1=G, 2=B
            compressed[:, :, i] = compress_image_svd(image[:, :, i], k)  # Compress each channel
        return compressed

def compression_ratio(original_shape: tuple, k: int) -> float:
    """Calcula ratio de compresión."""
    m, n = original_shape[:2]  # Height (m) and width (n)
    original_size = m * n  # Pixel count (per channel)
    compressed_size = k * (m + n + 1)  # Rough parameter count: U(m×k)+S(k)+Vt(k×n)
    return compressed_size / original_size

# Demo (sin cargar imagen real)
# Simular imagen 100x100
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # Fake grayscale image in [0,255]

for k in [5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)  # Approx reconstruction with k singular values
    ratio = compression_ratio(image.shape, k)  # Estimated compression ratio
    print(f"k={k}: ratio={ratio:.2%}")

#### Prompt

1) **Basic**

- Create two vectors `a` and `b` in `R^3`.
- Compute `a + b`, `a - b`, and `3*a`.

2) **Intermediate**

- Verify with `assert` that addition is commutative: `a + b == b + a`.

3) **Advanced**

- Convert a 1D vector `x` with shape `(3,)` into a column vector `(3, 1)` and verify shapes.

#### Solution

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

s = a + b
d = a - b
scaled = 3 * a

assert np.allclose(a + b, b + a)

x = np.array([7.0, 8.0, 9.0])
x_col = x.reshape(-1, 1)
assert x.shape == (3,)
assert x_col.shape == (3, 1)
```

---

### Exercise 2.2: Dot product, angle, and projection

#### Prompt

1) **Basic**

- Compute `a·b` in 3 ways: `np.dot(a,b)`, `a @ b`, and `np.sum(a*b)`.

2) **Intermediate**

- Implement `cos_theta = (a·b)/(||a|| ||b||)` and verify it is in `[-1, 1]`.

3) **Advanced**

- Implement the projection `proj_b(a) = (a·b)/(b·b) * b`.
- Verify the residual `r = a - proj_b(a)` is orthogonal to `b` (`r·b ≈ 0`).

#### Solution

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

d1 = np.dot(a, b)
d2 = a @ b
d3 = np.sum(a * b)
assert np.isclose(d1, d2) and np.isclose(d2, d3)

cos_theta = d1 / (np.linalg.norm(a) * np.linalg.norm(b))
cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
assert -1.0 <= cos_theta <= 1.0

proj = (np.dot(a, b) / np.dot(b, b)) * b
r = a - proj
assert np.isclose(np.dot(r, b), 0.0, atol=1e-10)
```

---

### Exercise 2.3: L1/L2/L∞ norms (intuition + verification)

#### Prompt

1) **Basic**

- Compute `||x||_1`, `||x||_2`, `||x||_∞` for `x = [3, -4, 12]`.

2) **Intermediate**

- Verify they match `np.linalg.norm(x, ord=...)`.

3) **Advanced**

- Verify the inequality `||x||_∞ <= ||x||_2 <= ||x||_1`.

#### Solution

```python
import numpy as np

x = np.array([3.0, -4.0, 12.0])

n1 = np.sum(np.abs(x))
n2 = np.sqrt(np.sum(x * x))
ninf = np.max(np.abs(x))

assert np.isclose(n1, np.linalg.norm(x, 1))
assert np.isclose(n2, np.linalg.norm(x, 2))
assert np.isclose(ninf, np.linalg.norm(x, np.inf))

assert ninf <= n2 + 1e-12
assert n2 <= n1 + 1e-12
```

---

### Exercise 2.4: Distances (euclidean and manhattan) + distance matrix

#### Prompt

1) **Basic**

- Compute the Euclidean distance between `p1=[0,0]` and `p2=[3,4]`.

2) **Intermediate**

- Compute the Manhattan distance for the same points.

3) **Advanced**

- Given a matrix `X` with 3 points, build a Euclidean distance matrix `D` of shape `3x3`.
- Verify `D` is symmetric and has zeros on the diagonal.

#### Solution

```python
import numpy as np

p1 = np.array([0.0, 0.0])
p2 = np.array([3.0, 4.0])

d2 = np.linalg.norm(p2 - p1)
d1 = np.sum(np.abs(p2 - p1))

assert np.isclose(d2, 5.0)
assert np.isclose(d1, 7.0)

X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
sq_norms = np.sum(X ** 2, axis=1)
D_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (X @ X.T)
D_sq = np.maximum(D_sq, 0.0)
D = np.sqrt(D_sq)

assert D.shape == (3, 3)
assert np.allclose(D, D.T)
assert np.allclose(np.diag(D), 0.0)
```

---

### Exercise 2.5: Cosine similarity (and the zero-vector case)

#### Prompt

1) **Basic**

- Verify that identical vectors have cosine similarity ≈ 1.

2) **Intermediate**

- Verify that orthogonal vectors have cosine similarity ≈ 0.

3) **Advanced**

- Define cosine similarity when one of the vectors is zero, and avoid division by zero.

#### Solution

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


v1 = np.array([1.0, 2.0, 3.0])
v2 = np.array([1.0, 2.0, 3.0])
v3 = np.array([1.0, 0.0, 0.0])
v4 = np.array([0.0, 1.0, 0.0])
z = np.array([0.0, 0.0, 0.0])

assert np.isclose(cosine_similarity(v1, v2), 1.0)
assert np.isclose(cosine_similarity(v3, v4), 0.0)
assert cosine_similarity(v1, z) == 0.0
```

---

### Exercise 2.6: Matrix multiplication and shape reasoning

#### Prompt

1) **Basic**

- Given `A` shape `(2,3)` and `B` shape `(3,2)`, compute `A @ B` and verify the result is `(2,2)`.

2) **Intermediate**

- For a dataset `X` with shape `(n,d)`, verify:
  - `X.T @ X` has shape `(d,d)`
  - `X @ X.T` has shape `(n,n)`

3) **Advanced**

- Implement `y_hat = X @ w + b` with `w` shape `(d,)` and scalar `b`.

#### Solution

```python
import numpy as np

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
C = A @ B
assert C.shape == (2, 2)

n, d = 7, 4
X = np.random.randn(n, d)
assert (X.T @ X).shape == (d, d)
assert (X @ X.T).shape == (n, n)

w = np.random.randn(d)
b = 0.25
y_hat = X @ w + b
assert y_hat.shape == (n,)
```

---

### Exercise 2.7: Linear systems: `solve` vs inverse (stability)

#### Prompt

1) **Basic**

- Solve `Ax=b` using `np.linalg.solve`.

2) **Intermediate**

- Solve with `np.linalg.inv(A) @ b` and compare results.

3) **Advanced**

- Construct a singular matrix and verify `np.linalg.solve` raises an error.

#### Solution

```python
import numpy as np

A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])

x_solve = np.linalg.solve(A, b)
x_inv = np.linalg.inv(A) @ b

assert np.allclose(A @ x_solve, b)
assert np.allclose(x_solve, x_inv)

S = np.array([[1.0, 2.0], [2.0, 4.0]])
try:
    np.linalg.solve(S, np.array([1.0, 1.0]))
    raise AssertionError("Expected LinAlgError for a singular matrix")
except np.linalg.LinAlgError:
    pass
```

---

### Exercise 2.8: Eigenvalues/eigenvectors (verify Av=λv)

#### Prompt

1) **Basic**

- Compute eigenvalues/eigenvectors of a symmetric 2x2 matrix.

2) **Intermediate**

- Verify numerically `A @ v ≈ λ v` for each pair.

3) **Advanced**

- For a symmetric matrix, verify eigenvectors are orthogonal.

#### Solution

```python
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 2.0]])
vals, vecs = np.linalg.eig(A)

for i in range(2):
    v = vecs[:, i]
    lam = vals[i]
    assert np.allclose(A @ v, lam * v)

v0 = vecs[:, 0]
v1 = vecs[:, 1]
assert np.isclose(np.dot(v0, v1), 0.0, atol=1e-10)
```

---

### Exercise 2.9: PCA (eigen vs SVD) - shape consistency

#### Prompt

1) **Basic**

- Generate a dataset `X` with shape `(200, 3)` with correlated features.

2) **Intermediate**

- Implement PCA via eigen decomposition of the covariance and reduce to 2D.

3) **Advanced**

- Implement PCA via SVD and verify:
  - same output shapes
  - explained variance is sorted descending

#### Solution

```python
import numpy as np

np.random.seed(0)
n = 200
z = np.random.randn(n)
X = np.stack([z, 2.0 * z + 0.1 * np.random.randn(n), -z + 0.1 * np.random.randn(n)], axis=1)


def pca_eigen(X: np.ndarray, k: int):
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    vals, vecs = np.linalg.eig(cov)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx].real
    vecs = vecs[:, idx].real
    comps = vecs[:, :k]
    Xk = Xc @ comps
    ratio = vals[:k] / np.sum(vals)
    return Xk, comps, ratio


def pca_svd(X: np.ndarray, k: int):
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:k].T
    Xk = Xc @ comps
    var = (S ** 2) / (Xc.shape[0] - 1)
    ratio = var[:k] / np.sum(var)
    return Xk, comps, ratio


X_e, C_e, r_e = pca_eigen(X, 2)
X_s, C_s, r_s = pca_svd(X, 2)

assert X_e.shape == (n, 2)
assert X_s.shape == (n, 2)
assert C_e.shape == (3, 2)
assert C_s.shape == (3, 2)
assert r_e.shape == (2,)
assert r_s.shape == (2,)
assert r_e[0] >= r_e[1]
assert r_s[0] >= r_s[1]
```

---

### Exercise 2.10: SVD - reconstruction and error (truncated SVD)

#### Prompt

1) **Basic**

- Compute the SVD of a matrix `A`.

2) **Intermediate**

- Reconstruct `A` exactly and verify `A ≈ U Σ V^T`.

3) **Advanced**

- Reconstruct with rank `k=1` and `k=2` and verify the error decreases.

#### Solution

```python
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
U, S, Vt = np.linalg.svd(A, full_matrices=False)

A_full = U @ np.diag(S) @ Vt
assert np.allclose(A, A_full)

def trunc(U, S, Vt, k: int):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

A1 = trunc(U, S, Vt, 1)
A2 = trunc(U, S, Vt, 2)

err1 = np.linalg.norm(A - A1)
err2 = np.linalg.norm(A - A2)
assert err2 <= err1 + 1e-12
```

---

### (Bonus) Exercise 2.11: From linear algebra to ML - closed-form regression

#### Prompt

- Generate `X` and `y` for a linear model `y = Xw + noise`.
- Estimate `w_hat` with the normal equation **using `solve`**: `(X^T X) w = X^T y`.
- Verify `w_hat` is close to `w_true`.

#### Solution

```python
import numpy as np

np.random.seed(1)
n, d = 300, 3
X = np.random.randn(n, d)
w_true = np.array([0.5, -1.2, 2.0])
noise = 0.1 * np.random.randn(n)
y = X @ w_true + noise

XtX = X.T @ X
Xty = X.T @ y
w_hat = np.linalg.solve(XtX, Xty)

assert w_hat.shape == (d,)
assert np.linalg.norm(w_hat - w_true) < 0.2
```

## 📦 Entregable del Módulo

### Librería: `linear_algebra.py`

```python
"""
Linear Algebra Library for Machine Learning

Implementación desde cero de operaciones fundamentales.
Usando NumPy para eficiencia pero entendiendo las matemáticas.

Autor: [Tu nombre]
Módulo: 02 - Álgebra Lineal para ML
"""

import numpy as np
from typing import Tuple, Optional


# ============================================================
# PARTE 1: OPERACIONES CON VECTORES
# ============================================================

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Producto punto de dos vectores.

    a·b = Σᵢ aᵢ·bᵢ
    """
    assert a.shape == b.shape, "Vectores deben tener mismo shape"
    return float(np.sum(a * b))


def vector_angle(a: np.ndarray, b: np.ndarray) -> float:
    """
    Ángulo entre dos vectores en grados.

    cos(θ) = (a·b) / (||a|| ||b||)
    """
    cos_theta = dot_product(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos_theta = np.clip(cos_theta, -1, 1)
    return float(np.degrees(np.arccos(cos_theta)))


def project_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Proyección del vector a sobre el vector b.

    proj_b(a) = (a·b / b·b) · b
    """
    scalar = dot_product(a, b) / dot_product(b, b)
    return scalar * b


# ============================================================
# PARTE 2: NORMAS
# ============================================================

def l1_norm(x: np.ndarray) -> float:
    """Norma L1 (Manhattan): ||x||₁ = Σ|xᵢ|"""
    return float(np.sum(np.abs(x)))


def l2_norm(x: np.ndarray) -> float:
    """Norma L2 (Euclidiana): ||x||₂ = √(Σxᵢ²)"""
    return float(np.sqrt(np.sum(x ** 2)))


def linf_norm(x: np.ndarray) -> float:
    """Norma L∞ (Máximo): ||x||∞ = max|xᵢ|"""
    return float(np.max(np.abs(x)))


def normalize(x: np.ndarray, ord: int = 2) -> np.ndarray:
    """Normaliza vector a norma 1."""
    norm = np.linalg.norm(x, ord=ord)
    if norm == 0:
        return x
    return x / norm


# ============================================================
# PARTE 3: DISTANCIAS
# ============================================================

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia Euclidiana: d(a,b) = ||a-b||₂"""
    return l2_norm(a - b)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia Manhattan: d(a,b) = ||a-b||₁"""
    return l1_norm(a - b)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Similitud coseno: sim(a,b) = (a·b) / (||a|| ||b||)
    Rango: [-1, 1]
    """
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product(a, b) / (norm_a * norm_b)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia coseno: 1 - similitud_coseno"""
    return 1 - cosine_similarity(a, b)


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """
    Matriz de distancias euclidianas entre todos los pares.

    Args:
        X: matriz (n_samples, n_features)
    Returns:
        D: matriz (n_samples, n_samples) de distancias
    """
    sq_norms = np.sum(X ** 2, axis=1)
    D_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T
    D_sq = np.maximum(D_sq, 0)  # Evitar negativos por errores numéricos
    return np.sqrt(D_sq)


# ============================================================
# PARTE 4: EIGENVALUES Y PCA
# ============================================================

def eigendecomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula eigenvalues y eigenvectors, ordenados por eigenvalue descendente.

    Returns:
        eigenvalues: array de eigenvalues (ordenados)
        eigenvectors: matriz donde columna i es el eigenvector i
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Ordenar por eigenvalue descendente
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real

    return eigenvalues, eigenvectors


def pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Principal Component Analysis via SVD.

    Args:
        X: datos (n_samples, n_features)
        n_components: número de componentes

    Returns:
        X_transformed: datos proyectados (n_samples, n_components)
        components: componentes principales (n_components, n_features)
        explained_variance_ratio: proporción de varianza explicada
    """
    # Centrar datos
    X_centered = X - np.mean(X, axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Componentes principales
    components = Vt[:n_components]

    # Proyectar
    X_transformed = X_centered @ components.T

    # Varianza explicada
    variance = (S ** 2) / (X.shape[0] - 1)
    explained_variance_ratio = variance[:n_components] / np.sum(variance)

    return X_transformed, components, explained_variance_ratio


# ============================================================
# TESTS
# ============================================================

def run_tests():
    """Ejecuta tests básicos."""
    print("Ejecutando tests...")

    # Test producto punto
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert abs(dot_product(a, b) - 32) < 1e-10
    print("✓ dot_product")

    # Test normas
    x = np.array([3, 4])
    assert abs(l2_norm(x) - 5) < 1e-10
    assert abs(l1_norm(x) - 7) < 1e-10
    print("✓ normas")

    # Test distancias
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    assert abs(euclidean_distance(p1, p2) - 5) < 1e-10
    print("✓ distancias")

    # Test similitud coseno
    v1 = np.array([1, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0, 1])
    assert abs(cosine_similarity(v1, v2) - 1) < 1e-10
    assert abs(cosine_similarity(v1, v3)) < 1e-10
    print("✓ cosine_similarity")

    # Test PCA
    np.random.seed(42)
    X = np.random.randn(50, 10)
    X_pca, _, var_ratio = pca(X, 3)
    assert X_pca.shape == (50, 3)
    assert np.sum(var_ratio) <= 1.0
    print("✓ PCA")

    print("\n¡Todos los tests pasaron!")


if __name__ == "__main__":
    run_tests()
```

---

## 🧩 Consolidation (common errors + v5 debugging + Feynman challenge)

### Common errors

- **Confundir dot product con multiplicación elemento-a-elemento:** `a*b` no es `a·b`.
- **Shapes silenciosos:** `a` con shape `(n,)` vs `(n,1)` cambia resultados al multiplicar.
- **Invertir matrices innecesariamente:** evita `inv(A) @ b` y prefiere `solve(A, b)`.
- **PCA sin centrar:** si no haces `X_centered = X - mean`, PCA sale mal.
- **Signo de eigenvectors:** el signo de un eigenvector puede cambiar (`v` o `-v`); no es un bug.

### Debugging / validation (v5)

- Verifica `shapes` en cada operación matricial.
- Si aparece `nan/inf`, revisa escalas y operaciones sensibles.
- Registra hallazgos en `../../study_tools/DIARIO_ERRORES.md`.
- Full protocol:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Feynman challenge (whiteboard)

Explica en 5 líneas o menos:

1) ¿Por qué `a·b` es una “sombra” y qué significa que sea negativo?
2) ¿Por qué PCA usa eigenvectors de la covarianza?
3) ¿Qué te da SVD que sea más estable que eigendecomposition?

---

## ✅ Completion checklist

- [ ] Puedo calcular producto punto y explicar su significado geométrico
- [ ] Entiendo las diferencias entre normas L1, L2, L∞
- [ ] Puedo calcular distancia euclidiana y similitud coseno
- [ ] Sé multiplicar matrices y entiendo las dimensiones resultantes
- [ ] Puedo explicar qué son eigenvalues/eigenvectors y su uso en PCA
- [ ] Entiendo SVD y puedo usarlo para compresión/PCA
- [ ] Implementé `linear_algebra.py` con todos los tests pasando
- [ ] Puedo proyectar datos usando PCA y explicar varianza explicada

---

## 🔗 Navigation

| Previous | Index | Next |
|----------|--------|-----------|
| [01_PYTHON_CIENTIFICO](01_PYTHON_CIENTIFICO.md) | [00_INDICE](00_INDICE.md) | [03_CALCULO_MULTIVARIANTE](03_CALCULO_MULTIVARIANTE.md) |
