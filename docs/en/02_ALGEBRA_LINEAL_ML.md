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

### 1.1 Geometric and algebraic definition

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

### 1.2 Vector operations

#### Formalization: dot product as “shadow/projection”

**Intuition:** the dot product tells you how much of vector `a` is “pointing” in the direction of `b`. If you imagine a flashlight projecting `a` onto the line of `b`, the dot product is related to the size of that **shadow**.

Two formulas you should master:

```
a·b = ||a|| · ||b|| · cos(θ)

proj_b(a) = (a·b / b·b) · b
```

Quick interpretation:

- if `a·b` is large and positive → they point in a similar direction
- if `a·b ≈ 0` → they are almost orthogonal (little “shadow”)
- if `a·b` is negative → they point in opposite directions

**Why it matters in ML:** many predictions have the form `ŷ = Xθ` (sums of dot products). Understanding it geometrically keeps the model from feeling like a “black box”.

```python
import numpy as np  # NumPy for vector operations and dot products

# Example vectors
a = np.array([1, 2, 3])  # Example vector a (e.g., features)
b = np.array([4, 5, 6])  # Example vector b (e.g., weights)

# === VECTOR ADDITION ===
# (a + b)ᵢ = aᵢ + bᵢ
suma = a + b
print(f"a + b = {suma}")  # [5, 7, 9]

# === VECTOR SUBTRACTION ===
resta = a - b
print(f"a - b = {resta}")  # [-3, -3, -3]

# === SCALAR MULTIPLICATION ===
# (c·a)ᵢ = c·aᵢ
escalar = 2 * a
print(f"2·a = {escalar}")  # [2, 4, 6]

# === DOT PRODUCT ===
# a·b = Σᵢ aᵢ·bᵢ
# Result: scalar
dot = np.dot(a, b)
print(f"a·b = {dot}")  # 1*4 + 2*5 + 3*6 = 32

# Alternatively:
dot_alt = a @ b  # @ is the dot product for 1D vectors (equivalent to np.dot)
dot_sum = np.sum(a * b)  # Element-wise multiply then sum (manual dot product)
print(f"Check: {dot_alt}, {dot_sum}")  # Sanity check: all methods should match
```

### 1.3 Geometric interpretation of the dot product

```python
import numpy as np  # NumPy for dot, norms, arccos, and clipping

def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the angle between two vectors.

    cos(θ) = (a·b) / (||a|| ||b||)
    """
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # cos(θ) = (a·b)/(||a|| ||b||)
    # Clip to avoid floating point drift outside [-1, 1] (arccos domain)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_rad = np.arccos(cos_theta)  # Convert cosine to angle in radians
    theta_deg = np.degrees(theta_rad)  # Convert radians to degrees (human-friendly)
    return theta_deg  # Return the angle in degrees

# Examples
v1 = np.array([1, 0])  # x-axis
v2 = np.array([0, 1])  # y-axis (orthogonal to x)
v3 = np.array([1, 1])  # 45° direction
v4 = np.array([-1, 0])  # opposite to x-axis

print(f"Angle between [1,0] and [0,1]: {angle_between_vectors(v1, v2):.0f}°")  # 90°  # Orthogonal vectors
print(f"Angle between [1,0] and [1,1]: {angle_between_vectors(v1, v3):.0f}°")  # 45°  # Diagonal vs x-axis
print(f"Angle between [1,0] and [-1,0]: {angle_between_vectors(v1, v4):.0f}°") # 180° # Opposite directions

# Interpretation for ML:
# - High dot product → similar vectors (same direction)
# - Dot product ≈ 0 → orthogonal vectors (independent)
# - Negative dot product → opposite vectors
```

### 1.4 Vector projection

```python
import numpy as np  # NumPy for dot products and working with vectors as arrays

def project(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Project vector a onto vector b.

    proj_b(a) = (a·b / b·b) · b

    Useful for: PCA, regression, signal decomposition
    """
    scalar = np.dot(a, b) / np.dot(b, b)  # Projection scalar: (a·b)/(b·b)
    return scalar * b  # Scale b to get the projected vector (same direction as b)

# Example
a = np.array([3, 4])  # Vector to project
b = np.array([1, 0])  # Unit vector along x (projection direction)

proyeccion = project(a, b)  # Compute the projection of a onto b
print(f"Projection of {a} onto {b}: {proyeccion}")  # [3, 0]  # Keeps only the x-component

# The projection tells us how much of a lies in the direction of b
```

---

## 💻 Part 2: Norms and distances

### 2.1 L2 norm (Euclidean)

```python
import numpy as np  # NumPy for vectorized math and reference implementation (linalg.norm)

def l2_norm(x: np.ndarray) -> float:
    """
    L2 norm (Euclidean): vector length.

    ||x||₂ = √(Σᵢ xᵢ²)

    ML use:
    - Ridge regularization
    - Vector normalization
    - Euclidean distance
    """
    return np.sqrt(np.sum(x ** 2))  # sqrt(sum(x_i^2)): square -> sum -> square root

# NumPy equivalent
x = np.array([3, 4])  # Classic 3-4-5 vector
print(f"||x||₂ = {l2_norm(x)}")           # 5.0  # Our implementation
print(f"NumPy:  {np.linalg.norm(x)}")     # 5.0  # NumPy default is L2
print(f"NumPy:  {np.linalg.norm(x, 2)}")  # 5.0 (specifying ord=2)

# Unit vector (normalized)
def normalize(x: np.ndarray) -> np.ndarray:
    """Convert a vector to length 1."""
    return x / np.linalg.norm(x)  # Divide by the norm to make ||x|| = 1 (unit vector)

x_unit = normalize(x)  # Normalize x
print(f"Unit vector: {x_unit}")  # [0.6, 0.8]  # Same direction, scaled length
print(f"Norm of unit vector: {np.linalg.norm(x_unit)}")  # 1.0  # Sanity check
```

### 2.2 L1 norm (Manhattan)

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

### 2.3 L∞ norm (maximum)

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

### 2.4 Euclidean distance

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

# Example
p1 = np.array([0, 0])  # Point 1
p2 = np.array([3, 4])  # Point 2 (3-4-5 triangle)
print(f"Distance: {euclidean_distance(p1, p2)}")  # 5.0

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
print("Distance matrix:")
print(D)
```

### 2.5 Cosine similarity

```python
import numpy as np  # NumPy for dot product and vector norms

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity: measures the angle between vectors.

    sim(a, b) = (a·b) / (||a|| ||b||)

    Range: [-1, 1]
    - 1: identical vectors (same direction)
    - 0: orthogonal vectors
    - -1: opposite vectors

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
    """Cosine distance = 1 - cosine similarity."""
    return 1 - cosine_similarity(a, b)  # Turn similarity into a distance-like quantity

# Examples
v1 = np.array([1, 0, 0])  # Base vector
v2 = np.array([1, 0, 0])  # Identical to v1
v3 = np.array([0, 1, 0])  # Orthogonal to v1
v4 = np.array([-1, 0, 0])  # Opposite direction to v1

print(f"Similarity (identical):    {cosine_similarity(v1, v2)}")  # 1.0
print(f"Similarity (orthogonal):   {cosine_similarity(v1, v3)}")  # 0.0
print(f"Similarity (opposite):     {cosine_similarity(v1, v4)}")  # -1.0
```

---

## 💻 Part 3: Matrices

### 3.1 Basic operations

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

# === ADDITION AND SUBTRACTION ===
# Only for matrices with the same shape
C = np.array([[1, 2, 3], [4, 5, 6]])
print(f"A + C =\n{A + C}")  # Element-wise addition (requires same shape)

# === SCALAR MULTIPLICATION ===
print(f"2·A =\n{2 * A}")  # Scalar multiplication (scales every entry)

# === MATRIX PRODUCT ===
# (m×n) @ (n×p) = (m×p)
# A(2×3) @ B(3×2) = (2×2)
AB = A @ B
print(f"A @ B =\n{AB}")  # Matrix product: (2x3) @ (3x2) -> (2x2)
# [[58, 64],
#  [139, 154]]

# Manual check for element [0,0]:
# 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58

# === TRANSPOSE ===
print(f"A^T =\n{A.T}")  # Transpose swaps rows/columns
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

### 3.2 Matrix-vector multiplication (linear transformation)

```python
import numpy as np  # NumPy for trig functions and matrix-vector multiplication

# Matrix-vector multiplication is a LINEAR TRANSFORMATION
# y = Ax transforms the vector x into the space of y

# Example: 90° rotation in R²
theta = np.pi / 2  # 90 degrees (in radians)
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])  # 2x2 rotation matrix

x = np.array([1, 0])  # Input vector
y = R @ x  # Apply the linear transform
print(f"Rotate [1,0] by 90°: {y}")  # [0, 1]  # Expected: x-axis -> y-axis

# In ML: y = Wx + b (neural network layer)
W = np.random.randn(10, 784)  # Weights: 784 inputs → 10 outputs
b = np.random.randn(10)         # Bias (one per output unit)
x = np.random.randn(784)        # Input (flattened image)

y = W @ x + b  # Layer output: (10,784)@(784,) + (10,) -> (10,)
print(f"Shape of y: {y.shape}")  # (10,)
```

### 3.3 Matrix inverse

```python
import numpy as np  # NumPy for matrix inverse and linear algebra exceptions

def safe_inverse(A: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of A if it exists.
    A @ A⁻¹ = A⁻¹ @ A = I

    ML use:
    - Closed-form linear regression: θ = (X^T X)⁻¹ X^T y
    - Whitening in PCA
    """
    try:
        return np.linalg.inv(A)  # Compute A^{-1} (A must be square and non-singular)
    except np.linalg.LinAlgError:
        print("Matrix is not invertible (singular)")  # Singular => det(A)=0 (no inverse)
        return None  # Signal failure

# Example
A = np.array([
    [4, 7],
    [2, 6]
])

A_inv = safe_inverse(A)
print(f"A⁻¹ =\n{A_inv}")  # Print the inverse (if it exists)

# Verify: A @ A⁻¹ = I
identity = A @ A_inv
print(f"A @ A⁻¹ ≈ I:\n{np.round(identity, 10)}")  # Round to highlight identity despite float error

# NOTE: In ML, avoid computing inverses when possible
# Use np.linalg.solve() instead (more numerically stable)
```

### 3.4 Solving linear systems

```python
import numpy as np  # NumPy for solving linear systems with solve()

# System: Ax = b
# Find x

A = np.array([
    [3, 1],
    [1, 2]
])  # Coefficient matrix
b = np.array([9, 8])  # Right-hand side vector

# Method 1: inverse (NOT RECOMMENDED)
x_inv = np.linalg.inv(A) @ b  # Works, but less stable and often slower

# Method 2: solve (RECOMMENDED - more stable)
x_solve = np.linalg.solve(A, b)  # Preferred: solves Ax=b directly

print(f"Solution: x = {x_solve}")  # [2, 3]

# Verify
print(f"A @ x = {A @ x_solve}")    # [9, 8] ✓  # Check that Ax reproduces b
```

---

## 💻 Part 4: Eigenvalues and eigenvectors

### 4.1 Concept

#### Physical intuition: the globe (eigenvector as an axis)

Imagine you take a globe and spin it.

- Almost all points on the surface move.
- But there is a “special” line that does not change direction: the axis connecting the poles.

That axis is a metaphor for the **eigenvector**: a direction that the transformation “respects” (it does not rotate it, it only scales it).

The **eigenvalue** tells you how much that direction stretches/shrinks.

#### Intuition-building code (required): grid deformed by a matrix

To stop seeing matrices as tables and start seeing them as “machines that deform space”, use the script:

- [`visualizations/viz_transformations.py`](../../visualizations/viz_transformations.py)

Run:

```bash
# Run the script that draws a grid and shows how a matrix transforms space
python3 visualizations/viz_transformations.py  # Requires plotting dependencies (e.g., matplotlib)
```

Exercise:

- try matrices like `[[2, 0], [0, 1]]`, `[[0, -1], [1, 0]]`, `[[1, 1], [0, 1]]`
- observe how the grid deforms and how an eigenvector behaves (if one exists in R²)

#### Worked example: Eigenvalues of a 2×2 matrix (by hand)

Before using `np.linalg.eig`, do it once “by hand” to lock in the idea.

For:

```
A = [[2, 1],
     [1, 2]]
```

1) We look for `λ` such that there exists a `v ≠ 0` satisfying `Av = λv`. This is equivalent to:

```
(A - λI)v = 0
```

2) For a non-trivial solution to exist, the determinant must be 0:

```
det(A - λI) = 0

det([[2-λ, 1],
     [1, 2-λ]]) = (2-λ)^2 - 1
```

3) Solve:

```
(2-λ)^2 - 1 = 0
2-λ = ±1
λ ∈ {3, 1}
```

This matches what the code prints (eigenvalues `[3, 1]`).

```python
import numpy as np  # NumPy for eigendecomposition and matrix operations

"""
EIGENVALUES and EIGENVECTORS

Definition: Av = λv
- v: eigenvector (a vector that only scales, it does not change direction)
- λ: eigenvalue (scaling factor)

Interpretation:
- Eigenvectors are the “principal directions” of a transformation
- Eigenvalues tell you how much the transformation stretches/compresses along each direction

ML use:
- PCA: eigenvectors of the covariance matrix are the principal components
- PageRank: dominant eigenvector of the transition matrix
- Stability of dynamical systems
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
lambda1 = eigenvalues[0]  # Eigenvalue associated with v1

Av = A @ v1  # Apply A to v1
lambda_v = lambda1 * v1  # Scale v1 by λ1

print(f"\nVerification Av = λv:")
print(f"Av     = {Av}")  # Should match λv (up to floating point tolerance)
print(f"λv     = {lambda_v}")
print(f"Equal? {np.allclose(Av, lambda_v)}")  # allclose handles small float errors
```

### 4.2 Eigenvalues for PCA

#### Connection Line 2: Covariance as expectation (statistics)

In statistics, the covariance matrix is conceptually defined as:

```
Cov(X) = E[(X - μ)(X - μ)^T]
```

This bridge is key for the **Statistical Estimation** course (Line 2): the same idea of “expectation” appears in MLE, variance, estimators, and hypothesis tests.

```python
import numpy as np  # NumPy for centering, covariance, and eigendecomposition

def pca_via_eigen(X: np.ndarray, n_components: int) -> tuple:
    """
    PCA using eigendecomposition of the covariance matrix.

    Args:
        X: data (n_samples, n_features)
        n_components: number of components to keep

    Returns:
        X_transformed: projected data
        components: eigenvectors (principal components)
        explained_variance: explained variance ratio per component
    """
    # 1. Center data (subtract mean per feature)
    X_centered = X - np.mean(X, axis=0)

    # 2. Compute covariance matrix
    # Cov = (1/n) X^T X
    n_samples = X.shape[0]  # Number of samples (rows)
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)  # Sample covariance: (1/(n-1)) X^T X

    # 3. Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # Eigenvalues ~ variances; eigenvectors ~ directions

    # 4. Sort by eigenvalue (largest to smallest)
    idx = np.argsort(eigenvalues)[::-1]  # Sort indices by descending eigenvalue
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. Select top n_components
    components = eigenvectors[:, :n_components].real  # Keep top components; take real part for stability

    # 6. Project data
    X_transformed = X_centered @ components  # Project onto principal directions

    # 7. Compute explained variance
    total_variance = np.sum(eigenvalues)  # Total variance
    explained_variance = eigenvalues[:n_components].real / total_variance  # Explained variance ratio

    return X_transformed, components, explained_variance

# Demo
np.random.seed(42)  # Fixed seed for reproducibility
X = np.random.randn(100, 5)  # Synthetic dataset: 100 samples, 5 features

X_pca, components, var_explained = pca_via_eigen(X, n_components=2)

print(f"Original shape: {X.shape}")  # (n_samples, n_features)
print(f"Reduced shape: {X_pca.shape}")  # (n_samples, n_components)
print(f"Explained variance ratio: {var_explained}")  # Ratio per component
print(f"Total explained variance: {np.sum(var_explained):.2%}")  # Total explained variance
```

---

## 💻 Part 5: SVD (Singular Value Decomposition)

### 5.1 Concept

```python
import numpy as np  # NumPy for SVD (linalg.svd) and reconstruction

"""
SVD: Singular Value Decomposition

A = U Σ V^T

- U: orthogonal matrix (m×m) - left singular vectors
- Σ: diagonal matrix (m×n) - singular values (σ₁ ≥ σ₂ ≥ ... ≥ 0)
- V^T: orthogonal matrix (n×n) - right singular vectors

Advantages over eigendecomposition:
- Works for ANY matrix (not only square)
- More numerically stable
- Singular values are always non-negative

ML use:
- PCA (preferred method)
- Image compression
- Recommender systems (matrix factorization)
- Regularization (truncated SVD)
"""

# Example
A = np.array([  # Non-square matrix (SVD works for any shape)
    [1, 2],
    [3, 4],
    [5, 6]
])  # 3×2

U, S, Vt = np.linalg.svd(A, full_matrices=False)  # Economy SVD keeps minimal shapes

print(f"U shape: {U.shape}")   # (3, 2)  # Left singular vectors
print(f"S shape: {S.shape}")   # (2,)    # Singular values (σ)
print(f"Vt shape: {Vt.shape}") # (2, 2)  # Right singular vectors (transposed)

# Reconstruct A
A_reconstructed = U @ np.diag(S) @ Vt  # Rebuild A from U, Σ, V^T
print(f"\nA ≈ U Σ V^T? {np.allclose(A, A_reconstructed)}")  # Should be True
```

### 5.2 PCA via SVD (preferred method)

```python
import numpy as np  # NumPy for centering data and running SVD

def pca_via_svd(X: np.ndarray, n_components: int) -> tuple:
    """
    PCA using SVD (more stable than eigendecomposition).

    Relationship: if X = UΣV^T, then:
    - V contains the principal components
    - Σ²/(n-1) are the variances (eigenvalues of X^T X)
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

print(f"Explained variance per component: {var_ratio}")  # Per-component explained variance ratio
print(f"Total explained variance: {np.sum(var_ratio):.2%}")  # Total explained variance
```

### 5.3 Image compression with SVD

```python
import numpy as np  # NumPy for SVD and representing images as arrays

def compress_image_svd(image: np.ndarray, k: int) -> np.ndarray:
    """
    Compress an image using truncated SVD.

    Args:
        image: 2D (grayscale) or 3D (RGB) image matrix
        k: number of singular values to keep

    Returns:
        compressed image
    """
    if len(image.shape) == 2:  # 2D case: grayscale image (m×n)
        # Grayscale
        U, S, Vt = np.linalg.svd(image, full_matrices=False)  # SVD of the image matrix
        compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]  # Truncated SVD: keep only k components
        return np.clip(compressed, 0, 255).astype(np.uint8)  # Clip to valid range and cast
    else:
        # RGB: compress each channel
        compressed = np.zeros_like(image)  # Allocate output image
        for i in range(3):  # Loop channels: 0=R, 1=G, 2=B
            compressed[:, :, i] = compress_image_svd(image[:, :, i], k)  # Compress each channel
        return compressed

def compression_ratio(original_shape: tuple, k: int) -> float:
    """Compute compression ratio."""
    m, n = original_shape[:2]  # Height (m) and width (n)
    original_size = m * n  # Pixel count (per channel)
    compressed_size = k * (m + n + 1)  # Rough parameter count: U(m×k)+S(k)+Vt(k×n)
    return compressed_size / original_size

# Demo (without loading a real image)
# Simulate a 100x100 image
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # Fake grayscale image in [0,255]

for k in [5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)  # Approx reconstruction with k singular values
    ratio = compression_ratio(image.shape, k)  # Estimated compression ratio
    print(f"k={k}: ratio={ratio:.2%}")
```

---
## 🎯 Topic-based progressive exercises + solutions

 Rules:

 - **Try first** without looking at solutions.
 - **Suggested timebox:** 15–25 min per exercise.
 - **Minimum success:** your solution must pass the `assert` checks.

 ---

### Exercise 2.1: Vectors - basic operations and shapes

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
import numpy as np  # NumPy for arrays, vectorized operations, and numeric comparisons

a = np.array([1.0, 2.0, 3.0])  # Define vector a in R^3
b = np.array([4.0, 5.0, 6.0])  # Define vector b in R^3

s = a + b  # Element-wise addition (vector sum)
d = a - b  # Element-wise subtraction (vector difference)
scaled = 3 * a  # Scalar multiplication (scales every component)

assert np.allclose(a + b, b + a)  # Verify commutativity of addition

x = np.array([7.0, 8.0, 9.0])  # 1D vector with shape (3,)
x_col = x.reshape(-1, 1)  # Convert to column vector shape (3, 1)
assert x.shape == (3,)  # Confirm original shape
assert x_col.shape == (3, 1)  # Confirm column-vector shape
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
import numpy as np  # NumPy for dot products, norms, clipping, and numeric checks

a = np.array([1.0, 2.0, 3.0])  # Define vector a
b = np.array([4.0, 5.0, 6.0])  # Define vector b

d1 = np.dot(a, b)  # Dot product via np.dot
d2 = a @ b  # Dot product via @ (1D @ 1D)
d3 = np.sum(a * b)  # Dot product as sum of element-wise products
assert np.isclose(d1, d2) and np.isclose(d2, d3)  # All three methods should match

cos_theta = d1 / (np.linalg.norm(a) * np.linalg.norm(b))  # cos(theta) = (a·b)/(||a|| ||b||)
cos_theta = float(np.clip(cos_theta, -1.0, 1.0))  # Clip for numeric stability and cast to float
assert -1.0 <= cos_theta <= 1.0  # cos(theta) must be in [-1, 1]

proj = (np.dot(a, b) / np.dot(b, b)) * b  # Projection of a onto b
r = a - proj  # Residual component (should be orthogonal to b)
assert np.isclose(np.dot(r, b), 0.0, atol=1e-10)  # Verify orthogonality: r·b ≈ 0
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
import numpy as np  # NumPy for abs/sum/max, sqrt, and reference norms

x = np.array([3.0, -4.0, 12.0])  # Example vector with mixed signs

n1 = np.sum(np.abs(x))  # L1 norm: sum of absolute values
n2 = np.sqrt(np.sum(x * x))  # L2 norm: sqrt of sum of squares
ninf = np.max(np.abs(x))  # L-infinity norm: maximum absolute value

assert np.isclose(n1, np.linalg.norm(x, 1))  # Verify against NumPy's L1 norm
assert np.isclose(n2, np.linalg.norm(x, 2))  # Verify against NumPy's L2 norm
assert np.isclose(ninf, np.linalg.norm(x, np.inf))  # Verify against NumPy's L∞ norm

assert ninf <= n2 + 1e-12  # Check inequality: ||x||∞ <= ||x||2
assert n2 <= n1 + 1e-12  # Check inequality: ||x||2 <= ||x||1
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
import numpy as np  # NumPy for arrays, norms, broadcasting, and numeric assertions

p1 = np.array([0.0, 0.0])  # Point 1 in R^2
p2 = np.array([3.0, 4.0])  # Point 2 in R^2

d2 = np.linalg.norm(p2 - p1)  # Euclidean distance (L2 norm)
d1 = np.sum(np.abs(p2 - p1))  # Manhattan distance (L1 norm)

assert np.isclose(d2, 5.0)  # 3-4-5 triangle
assert np.isclose(d1, 7.0)  # |3| + |4| = 7

X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])  # 3 points (n=3) in 2D
sq_norms = np.sum(X ** 2, axis=1)  # Squared norms ||x_i||^2 for each row (shape: (n,))
D_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (X @ X.T)  # Pairwise squared distances via ||a-b||^2
D_sq = np.maximum(D_sq, 0.0)  # Clamp negatives caused by floating point error
D = np.sqrt(D_sq)  # Pairwise Euclidean distance matrix

assert D.shape == (3, 3)  # Distance matrix must be n×n
assert np.allclose(D, D.T)  # Distances are symmetric
assert np.allclose(np.diag(D), 0.0)  # Distance to self is zero
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
import numpy as np  # NumPy for norms, dot product, and testing

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
     na = np.linalg.norm(a)  # ||a||_2
     nb = np.linalg.norm(b)  # ||b||_2
     if na == 0.0 or nb == 0.0:
         return 0.0  # Define cosine similarity with a zero vector as 0 to avoid division by zero
     return float(np.dot(a, b) / (na * nb))  # cos(theta) = (a·b)/(||a|| ||b||)


v1 = np.array([1.0, 2.0, 3.0])  # Reference vector
v2 = np.array([1.0, 2.0, 3.0])  # Identical to v1
v3 = np.array([1.0, 0.0, 0.0])  # Unit vector along x-axis
v4 = np.array([0.0, 1.0, 0.0])  # Unit vector along y-axis (orthogonal to v3)
z = np.array([0.0, 0.0, 0.0])  # Zero vector

assert np.isclose(cosine_similarity(v1, v2), 1.0)  # Identical direction => cosine ≈ 1
assert np.isclose(cosine_similarity(v3, v4), 0.0)  # Orthogonal => cosine ≈ 0
assert cosine_similarity(v1, z) == 0.0  # Zero-vector convention
```

---

### Exercise 2.6: Matrix multiplication and shape reasoning

#### Prompt

1) **Basic**

- Compute `A @ B` where `A` is `(2,3)` and `B` is `(3,2)`.

2) **Intermediate**

- For a dataset `X` with shape `(n,d)`, verify:
  - `X.T @ X` has shape `(d,d)`
  - `X @ X.T` has shape `(n,n)`

3) **Advanced**

- Implement `y_hat = X @ w + b` with `w` shape `(d,)` and scalar `b`.

#### Solution

```python
import numpy as np  # NumPy for matrix multiplication and random data

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Shape (2, 3)
B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # Shape (3, 2)
C = A @ B  # Matrix product => shape (2, 2)
assert C.shape == (2, 2)  # (2,3)@(3,2)=(2,2)

n, d = 7, 4  # n samples, d features
X = np.random.randn(n, d)  # Dataset matrix with shape (n, d)
assert (X.T @ X).shape == (d, d)  # Feature-feature (Gram) matrix
assert (X @ X.T).shape == (n, n)  # Sample-sample (Gram) matrix

w = np.random.randn(d)  # Weight vector with shape (d,)
b = 0.25  # Scalar bias
y_hat = X @ w + b  # Linear model prediction (broadcasts bias)
assert y_hat.shape == (n,)  # One prediction per sample
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
 import numpy as np  # NumPy for linear algebra routines and numerical assertions

 A = np.array([[3.0, 1.0], [1.0, 2.0]])  # Coefficient matrix (2x2)
 b = np.array([9.0, 8.0])  # Right-hand side vector (2,)

 x_solve = np.linalg.solve(A, b)  # Preferred: solve Ax=b without explicitly inverting A
 x_inv = np.linalg.inv(A) @ b  # Alternative (less stable): x = A^{-1} b

 assert np.allclose(A @ x_solve, b)  # Solution must satisfy the original equation
 assert np.allclose(x_solve, x_inv)  # For well-conditioned A, both methods should match closely

 S = np.array([[1.0, 2.0], [2.0, 4.0]])  # Singular matrix (second row is 2x first row)
 try:
     np.linalg.solve(S, np.array([1.0, 1.0]))  # Should fail: no unique solution for singular matrices
     raise AssertionError("Expected LinAlgError for a singular matrix")  # If we get here, the test should fail
 except np.linalg.LinAlgError:
     pass  # Expected path
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
 import numpy as np  # NumPy for eigendecomposition, dot products, and numeric checks

 A = np.array([[2.0, 1.0], [1.0, 2.0]])  # Symmetric matrix (eigenvectors should be orthogonal)
 vals, vecs = np.linalg.eig(A)  # Returns eigenvalues (vals) and eigenvectors as columns (vecs)

 for i in range(2):
     v = vecs[:, i]  # i-th eigenvector
     lam = vals[i]  # corresponding eigenvalue
     assert np.allclose(A @ v, lam * v)  # Verify Av = λv

 v0 = vecs[:, 0]  # First eigenvector
 v1 = vecs[:, 1]  # Second eigenvector
 assert np.isclose(np.dot(v0, v1), 0.0, atol=1e-10)  # For symmetric A, eigenvectors are orthogonal
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
 import numpy as np  # NumPy for random data, eig/SVD, and shape checks

 np.random.seed(0)  # Reproducibility
 n = 200  # Number of samples
 z = np.random.randn(n)  # Latent 1D factor
 X = np.stack(
     [
         z,  # Feature 1
         2.0 * z + 0.1 * np.random.randn(n),  # Feature 2 correlated with z
         -z + 0.1 * np.random.randn(n),  # Feature 3 anti-correlated with z
     ],
     axis=1,
 )  # Shape (n, 3)


 def pca_eigen(X: np.ndarray, k: int):
     Xc = X - X.mean(axis=0)  # Center features (PCA assumes zero-mean)
     cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)  # Sample covariance matrix (3x3)
     vals, vecs = np.linalg.eig(cov)  # Eigenvalues ~ variances; eigenvectors ~ principal directions
     idx = np.argsort(vals)[::-1]  # Sort by descending variance
     vals = vals[idx].real  # Ensure real (numerical safety)
     vecs = vecs[:, idx].real  # Reorder eigenvectors accordingly
     comps = vecs[:, :k]  # Keep top-k principal components (3xk)
     Xk = Xc @ comps  # Project centered data onto k components (n x k)
     ratio = vals[:k] / np.sum(vals)  # Explained variance ratio per component
     return Xk, comps, ratio


 def pca_svd(X: np.ndarray, k: int):
     Xc = X - X.mean(axis=0)  # Center data
     U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Xc = U diag(S) Vt
     comps = Vt[:k].T  # Top-k right-singular vectors => principal directions (3xk)
     Xk = Xc @ comps  # Project onto components (n x k)
     var = (S ** 2) / (Xc.shape[0] - 1)  # Singular values relate to covariance eigenvalues
     ratio = var[:k] / np.sum(var)  # Explained variance ratio
     return Xk, comps, ratio


 X_e, C_e, r_e = pca_eigen(X, 2)  # PCA via covariance eigendecomposition
 X_s, C_s, r_s = pca_svd(X, 2)  # PCA via SVD (often preferred numerically)

 assert X_e.shape == (n, 2)  # Reduced data shape
 assert X_s.shape == (n, 2)
 assert C_e.shape == (3, 2)  # Components shape: (n_features, k)
 assert C_s.shape == (3, 2)
 assert r_e.shape == (2,)  # Explained variance ratios per component
 assert r_s.shape == (2,)
 assert r_e[0] >= r_e[1]  # Ratios should be sorted descending
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
 import numpy as np  # NumPy for SVD, reconstruction, and norms

 A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Example non-square matrix (3x2)
 U, S, Vt = np.linalg.svd(A, full_matrices=False)  # Economy SVD: A = U diag(S) Vt

 A_full = U @ np.diag(S) @ Vt  # Exact reconstruction using all singular values
 assert np.allclose(A, A_full)  # Should match original (up to floating point tolerance)

 def trunc(U, S, Vt, k: int):
     return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]  # Rank-k approximation (truncated SVD)

 A1 = trunc(U, S, Vt, 1)  # Best rank-1 approximation
 A2 = trunc(U, S, Vt, 2)  # Best rank-2 approximation (here: full rank for 3x2)

 err1 = np.linalg.norm(A - A1)  # Reconstruction error (Frobenius norm by default)
 err2 = np.linalg.norm(A - A2)
 assert err2 <= err1 + 1e-12  # Error should decrease as rank increases
 ```

 ---

 ### (Bonus) Exercise 2.11: From linear algebra to ML - closed-form regression

 #### Prompt

 - Generate `X` and `y` for a linear model `y = Xw + noise`.
 - Estimate `w_hat` with the normal equation **using `solve`**: `(X^T X) w = X^T y`.
 - Verify `w_hat` is close to `w_true`.

 #### Solution

 ```python
 import numpy as np  # NumPy for random data generation and linear solves

 np.random.seed(1)  # Reproducibility
 n, d = 300, 3  # n samples, d features
 X = np.random.randn(n, d)  # Design matrix
 w_true = np.array([0.5, -1.2, 2.0])  # Ground-truth weights
 noise = 0.1 * np.random.randn(n)  # Additive noise
 y = X @ w_true + noise  # Targets: linear model with noise

 XtX = X.T @ X  # Normal equation left side (d x d)
 Xty = X.T @ y  # Normal equation right side (d,)
 w_hat = np.linalg.solve(XtX, Xty)  # Solve (X^T X) w = X^T y

 assert w_hat.shape == (d,)  # Estimated weight vector shape
 assert np.linalg.norm(w_hat - w_true) < 0.2  # Should recover weights reasonably well
 ```

 ## 📦 Module Deliverable

 ### Library: `linear_algebra.py`

 ```python
 """
 Linear Algebra Library for Machine Learning

 From-scratch implementation of fundamental operations.
 Uses NumPy for efficiency while keeping the math explicit.

 Author: [Your name]
 Module: 02 - Linear Algebra for ML
 """

 import numpy as np  # NumPy provides fast, reliable vector/matrix routines
 from typing import Tuple  # Type hints for multi-return linear algebra helpers

 # ============================================================
 # PART 1: VECTOR OPERATIONS
 # ============================================================

 def dot_product(a: np.ndarray, b: np.ndarray) -> float:  # Compute the dot product Σ(a_i * b_i)
     """
     Dot product of two vectors.

     a·b = Σᵢ aᵢ·bᵢ
     """
     assert a.shape == b.shape, "Vectors must have the same shape"  # Element-wise multiplication requires aligned shapes
     return float(np.sum(a * b))  # Multiply element-wise and sum: Σ(a_i * b_i)


 def vector_angle(a: np.ndarray, b: np.ndarray) -> float:  # Compute the angle between vectors using arccos of cosine similarity
     """
     Angle between two vectors in degrees.

     cos(θ) = (a·b) / (||a|| ||b||)
     """
     cos_theta = dot_product(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # Compute cos(θ) from dot product and magnitudes
     cos_theta = np.clip(cos_theta, -1, 1)  # Numerical safety: keep within arccos domain
     return float(np.degrees(np.arccos(cos_theta)))  # arccos -> radians, then convert to degrees


 def project_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # Project vector a onto direction b
     """
     Projection of vector a onto vector b.

     proj_b(a) = (a·b / b·b) · b
     """
     scalar = dot_product(a, b) / dot_product(b, b)  # Scalar coefficient for the projection along b
     return scalar * b  # Scale direction vector b to get the projected vector


 # ============================================================
 # PART 2: NORMS
 # ============================================================

 def l1_norm(x: np.ndarray) -> float:  # Compute L1 norm: Σ|x_i|
     """L1 norm (Manhattan): ||x||₁ = Σ|xᵢ|"""  # Measures total absolute magnitude
     return float(np.sum(np.abs(x)))  # Sum absolute values: Σ|x_i|


 def l2_norm(x: np.ndarray) -> float:  # Compute L2 norm: sqrt(Σ x_i^2)
     """L2 norm (Euclidean): ||x||₂ = √(Σxᵢ²)"""  # Standard vector length
     return float(np.sqrt(np.sum(x ** 2)))  # sqrt(Σ x_i^2)


 def linf_norm(x: np.ndarray) -> float:  # Compute L∞ norm: max(|x_i|)
     """L∞ norm (max): ||x||∞ = max|xᵢ|"""  # Maximum absolute component
     return float(np.max(np.abs(x)))  # max(|x_i|)


 def normalize(x: np.ndarray, ord: int = 2) -> np.ndarray:  # Normalize x so that ||x||_ord == 1
     """Normalize a vector to have norm 1 (for the chosen `ord`)."""  # Common preprocessing step in ML
     norm = np.linalg.norm(x, ord=ord)  # Compute the norm (default: L2)
     if norm == 0:  # Edge case: the zero vector has no direction (avoid division by zero)
         return x  # Convention: return unchanged
     return x / norm  # Scale so that ||x|| = 1 while preserving direction


 # ============================================================
 # PART 3: DISTANCES
 # ============================================================

 def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:  # Compute Euclidean distance between two points/vectors
     """Euclidean distance: d(a,b) = ||a-b||₂"""  # Straight-line distance
     return l2_norm(a - b)  # Distance is the L2 norm of the difference vector


 def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:  # Compute Manhattan (L1) distance
     """Manhattan distance: d(a,b) = ||a-b||₁"""  # City-block distance
     return l1_norm(a - b)  # Distance is the L1 norm of the difference vector


 def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:  # Compute cosine similarity (normalized dot product)
     """
     Cosine similarity: sim(a,b) = (a·b) / (||a|| ||b||)
     Range: [-1, 1]
     """
     norm_a = l2_norm(a)  # ||a|| (magnitude) for normalization
     norm_b = l2_norm(b)  # ||b|| (magnitude) for normalization
     if norm_a == 0 or norm_b == 0:  # If either vector is zero, cosine similarity is undefined
         return 0.0  # Convention: return 0.0 to avoid division by zero
     return dot_product(a, b) / (norm_a * norm_b)  # cos(θ) computed from normalized dot product


 def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:  # Convert cosine similarity to a distance-like quantity
     """Cosine distance: 1 - cosine_similarity."""  # Turns similarity into a distance-like measure
     return 1 - cosine_similarity(a, b)  # Higher similarity => smaller distance


 def pairwise_euclidean(X: np.ndarray) -> np.ndarray:  # Compute all-pairs distances efficiently via algebraic identity
     """
     Pairwise Euclidean distance matrix for all samples.

     Args:
         X: array of shape (n_samples, n_features)
     Returns:
         D: array of shape (n_samples, n_samples) with distances
     """
     sq_norms = np.sum(X ** 2, axis=1)  # Precompute ||x_i||^2 for each row i (shape: (n_samples,))
     D_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T  # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
     D_sq = np.maximum(D_sq, 0)  # Avoid small negative values due to floating-point rounding
     return np.sqrt(D_sq)  # Convert squared distances to Euclidean distances


 # ============================================================
 # PART 4: EIGENVALUES AND PCA
 # ============================================================

 def eigendecomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # Compute (eigenvalues, eigenvectors) and sort by eigenvalue
     """
     Compute eigenvalues and eigenvectors, sorted by descending eigenvalue.

     Returns:
         eigenvalues: eigenvalues array (sorted)
         eigenvectors: matrix whose i-th column is the i-th eigenvector
     """
     eigenvalues, eigenvectors = np.linalg.eig(A)  # Eigendecomposition A v = λ v (may return complex dtype)

     idx = np.argsort(eigenvalues)[::-1]  # Sort indices by eigenvalue magnitude (descending)
     eigenvalues = eigenvalues[idx].real  # Reorder and keep the real part (common for real symmetric A)
     eigenvectors = eigenvectors[:, idx].real  # Reorder eigenvectors to match sorted eigenvalues

     return eigenvalues, eigenvectors  # (λ, V) where columns of V correspond to λ


 def pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Reduce dimensionality using PCA via SVD
     """
     Principal Component Analysis via SVD.

     Args:
         X: data matrix of shape (n_samples, n_features)
         n_components: number of principal components to keep

     Returns:
         X_transformed: projected data of shape (n_samples, n_components)
         components: principal directions of shape (n_components, n_features)
         explained_variance_ratio: variance explained by each kept component
     """
     X_centered = X - np.mean(X, axis=0)  # Center each feature: PCA must run on zero-mean variables

     U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # SVD: X = U · diag(S) · Vt

     components = Vt[:n_components]  # Rows of Vt are principal directions; keep the first k

     X_transformed = X_centered @ components.T  # Project data onto the principal subspace

     variance = (S ** 2) / (X.shape[0] - 1)  # Component variances relate to singular values squared
     explained_variance_ratio = variance[:n_components] / np.sum(variance)  # Fraction of total variance per component

     return X_transformed, components, explained_variance_ratio


 # ============================================================
 # TESTS
 # ============================================================

 def run_tests() -> None:  # Execute quick sanity checks to validate the core linear algebra helpers
     """Run a small sanity-check suite for the functions in this file."""  # Minimal, fast unit-like checks
     print("Running tests...")  # Visible feedback when executing this module directly

     a = np.array([1, 2, 3])  # Sample vector for dot product tests
     b = np.array([4, 5, 6])  # Another sample vector (same shape as `a`)
     assert abs(dot_product(a, b) - 32) < 1e-10  # 1*4 + 2*5 + 3*6 = 32
     print("✓ dot_product")  # Confirm the check passed

     x = np.array([3, 4])  # 3-4-5 right triangle vector (useful for norm checks)
     assert abs(l2_norm(x) - 5) < 1e-10  # L2 norm should be sqrt(3^2 + 4^2) = 5
     assert abs(l1_norm(x) - 7) < 1e-10  # L1 norm should be |3| + |4| = 7
     print("✓ norms")  # Confirm the norm checks passed

     p1 = np.array([0, 0])  # Point at the origin
     p2 = np.array([3, 4])  # Point 5 units away from origin in Euclidean distance
     assert abs(euclidean_distance(p1, p2) - 5) < 1e-10  # Distance must be 5
     print("✓ distances")  # Confirm distance computation is correct

     v1 = np.array([1, 0])  # Unit vector along x-axis
     v2 = np.array([1, 0])  # Same direction as v1
     v3 = np.array([0, 1])  # Orthogonal to v1
     assert abs(cosine_similarity(v1, v2) - 1) < 1e-10  # Same direction => cosine similarity 1
     assert abs(cosine_similarity(v1, v3)) < 1e-10  # Orthogonal => cosine similarity 0
     print("✓ cosine_similarity")  # Confirm cosine similarity behaves as expected

     np.random.seed(42)  # Make this test deterministic across runs
     X = np.random.randn(50, 10)  # Synthetic dataset: 50 samples, 10 features
     X_pca, _, var_ratio = pca(X, 3)  # Reduce to 3 principal components
     assert X_pca.shape == (50, 3)  # Output must have the requested number of components
     assert np.sum(var_ratio) <= 1.0  # Explained variance ratios must sum to <= 1
     print("✓ PCA")  # Confirm PCA runs and returns consistent shapes

     print("\nAll tests passed!")  # Final success message


 if __name__ == "__main__":  # Entry point when running this file as a script
     run_tests()  # Run tests when executed as a script (not when imported)

```
---

## 🧩 Consolidation (common errors + v5 debugging + Feynman challenge)

### Common errors

- **Confusing dot product with element-wise multiplication:** `a*b` is not `a·b`.
- **Silent shape issues:** `a` with shape `(n,)` vs `(n,1)` can change results when multiplying.
- **Unnecessary matrix inverses:** avoid `inv(A) @ b` and prefer `solve(A, b)`.
- **PCA without centering:** if you skip `X_centered = X - mean`, PCA results are wrong.
- **Eigenvector sign ambiguity:** an eigenvector can flip sign (`v` or `-v`); this is not a bug.

### Debugging / validation (v5)

- Verify `shapes` on every matrix operation.
- If `nan/inf` appears, check scales and numerically sensitive operations.
- Log findings in `../../study_tools/DIARIO_ERRORES.md`.
- Full protocol:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Feynman challenge (whiteboard)

Explain in 5 lines or less:

1) Why is `a·b` a “shadow”, and what does it mean if it is negative?
2) Why does PCA use eigenvectors of the covariance matrix?
3) What does SVD give you that is more stable than eigendecomposition?

---

## ✅ Completion checklist

- [ ] I can compute the dot product and explain its geometric meaning
- [ ] I understand the differences between L1, L2, and L∞ norms
- [ ] I can compute Euclidean distance and cosine similarity
- [ ] I can multiply matrices and understand the resulting dimensions
- [ ] I can explain eigenvalues/eigenvectors and their use in PCA
- [ ] I understand SVD and can use it for compression/PCA
- [ ] I implemented `linear_algebra.py` with all tests passing
- [ ] I can project data using PCA and explain explained variance

---

## 🔗 Navigation

| Previous | Index | Next |
|----------|--------|-----------|
| [01_PYTHON_CIENTIFICO](01_PYTHON_CIENTIFICO.md) | [00_INDICE](00_INDICE.md) | [03_CALCULO_MULTIVARIANTE](03_CALCULO_MULTIVARIANTE.md) |
