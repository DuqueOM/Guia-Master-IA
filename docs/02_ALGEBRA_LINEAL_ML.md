# Módulo 02 - Álgebra Lineal para Machine Learning

> **🎯 Objetivo:** Dominar vectores, matrices, normas y eigenvalues para ML
> **Fase:** 1 - Fundamentos Matemáticos | **Semanas 3-5**
> **Prerrequisitos:** Módulo 01 (Python Científico con NumPy)

---

<a id="m02-0"></a>

## 🧭 Cómo usar este módulo (modo 0→100)

**Propósito:** que puedas leer y escribir la “gramática” matemática de ML:

- `ŷ = Xθ` (supervised)
- proyecciones y bases (PCA)
- descomposiciones (SVD)

### Objetivos de aprendizaje (medibles)

Al terminar este módulo podrás:

- **Aplicar** producto punto y similitud coseno para medir “parecido” entre vectores.
- **Implementar** normas y distancias (L1/L2/L∞) y explicar su rol en regularización.
- **Razonar** shapes en operaciones matriciales (evitar bugs silenciosos).
- **Explicar** eigenvalues/eigenvectors como “direcciones principales” y conectarlo con PCA.
- **Explicar** SVD y por qué es el método preferido para PCA numéricamente estable.

### Prerrequisitos

- `Módulo 01` (NumPy, vectorización, shapes).

Enlaces rápidos:

- [GLOSARIO: Dot Product](GLOSARIO.md#dot-product)
- [GLOSARIO: Matrix Multiplication](GLOSARIO.md#matrix-multiplication)
- [GLOSARIO: L1 Norm](GLOSARIO.md#l1-norm-manhattan)
- [GLOSARIO: L2 Norm](GLOSARIO.md#l2-norm-euclidean)
- [GLOSARIO: SVD](GLOSARIO.md#svd-singular-value-decomposition)
- [RECURSOS.md](RECURSOS.md)

### Integración con Plan v4/v5

- Refuerzo diario de shapes: `study_tools/DRILL_DIMENSIONES_NUMPY.md`
- Simulacros: `study_tools/SIMULACRO_EXAMEN_TEORICO.md`
- Evaluación (rúbrica): [study_tools/RUBRICA_v1.md](../study_tools/RUBRICA_v1.md) (scope `M02` en `rubrica.csv`)
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Recursos (cuándo usarlos)

| Prioridad | Recurso | Cuándo usarlo en este módulo | Para qué |
|----------|---------|------------------------------|----------|
| **Obligatorio** | `study_tools/DRILL_DIMENSIONES_NUMPY.md` | Cada vez que una multiplicación/proyección te cambie el shape de forma inesperada | Evitar bugs silenciosos por shapes |
| **Obligatorio** | [3Blue1Brown: Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Semana 3–4, antes de entrar a matrices/eigen/SVD (y si te sientes “mecánico” con `@`/`eig`) | Construir intuición geométrica sólida |
| **Complementario** | Plot interactivo en Jupyter (`matplotlib` + `plotly` + `ipywidgets`) | Semana 3–5, cuando estudies transformaciones lineales / eigenvectors | Ver “rejillas deformándose” y construir intuición geométrica por experimentación |
| **Complementario** | [Mathematics for ML: Linear Algebra](https://www.coursera.org/learn/linear-algebra-machine-learning) | Semana 5, al entrar a eigenvalues/SVD | Formalizar con ejercicios guiados |
| **Opcional** | [Mathematics for ML (book)](https://mml-book.github.io/) | Después de terminar eigen/SVD (para profundizar) | Profundizar en notación y demostraciones |
| **Opcional** | [RECURSOS.md](RECURSOS.md) | Al planificar refuerzo para PCA (M06) | Elegir materiales de práctica adicionales |

## 🧠 ¿Por Qué Álgebra Lineal para ML?

### Intuición del espacio vectorial (el eslabón perdido)

Si solo piensas en matrices como “tablas de números”, vas a poder escribir `np.linalg.eig(A)` pero no vas a entender qué estás calculando. La idea central es:

> Una matriz es una **función** que transforma el espacio: lo estira, lo rota, lo inclina o lo aplasta.

#### 1) Vectores como movimiento (no como puntos)

Un vector `v = [x, y]` puede verse como un **desplazamiento**:

- empezar en el origen
- caminar `x` en X
- caminar `y` en Y

Visualización sugerida (dibújalo): suma de vectores como “caminar dos movimientos seguidos”.

#### 2) Matrices como deformación de una rejilla (grid)

Imagina una rejilla cuadrada en el plano. Multiplicar por una matriz `A` deforma toda la rejilla:

- líneas paralelas siguen paralelas
- el origen no se mueve
- los cuadrados se vuelven paralelogramos

Ejemplos mentales:

- `[[2, 0], [0, 1]]` estira el espacio en X al doble.
- Si `det(A) = 0`, aplastas el plano 2D en una línea (o un punto): pierdes dimensión.

Esto explica por qué una matriz con determinante 0 no es invertible: no puedes “des-aplastar” una línea para volver a hacer un plano.

#### 3) Producto punto como “sombra” (proyección)

Lectura geométrica: `a·b = ||a|| ||b|| cos(θ)` mide cuánto de `a` apunta en la dirección de `b`.

Aplicación directa en ML:

- `w·x` mide qué tan alineado está tu input `x` con el patrón `w`.

#### 4) Eigenvectors: los ejes que no se mueven

Cuando una matriz rota/estira el espacio, casi todos los vectores cambian de dirección. Pero algunos vectores son “tercos”: solo se escalan.

- **Eigenvector:** dirección que no gira bajo `A`.
- **Eigenvalue:** cuánto se estiró/encogió esa dirección.

Visualización sugerida (para PCA): imagina que quieres alinear una “cámara” con esos ejes naturales.

En PCA (M06), esos ejes (eigenvectors de la covarianza) son los ejes donde hay más varianza.

### Conexiones Directas con el Pathway

| Concepto | Uso en ML | Curso del Pathway |
|----------|-----------|-------------------|
| **Producto punto** | Similitud, predicciones | Supervised Learning |
| **Normas L1/L2** | Regularización, distancias | Supervised Learning |
| **Eigenvalues** | PCA, reducción dimensional | Unsupervised Learning |
| **Multiplicación matricial** | Forward pass en redes | Deep Learning |
| **SVD** | Compresión, PCA | Unsupervised Learning |

### La Matemática Detrás de ML

```
Regresión Lineal:     ŷ = Xθ           (multiplicación matriz-vector)
Logistic Regression:  ŷ = σ(Xθ)        (+ función de activación)
Neural Network:       ŷ = σ(W₃σ(W₂σ(W₁x)))  (capas de multiplicaciones)
PCA:                  X_reduced = XV    (proyección a eigenvectors)
```

---

## 📚 Contenido del Módulo

### Semana 3: Vectores y Operaciones Básicas
### Semana 4: Normas y Distancias
### Semana 5: Matrices, Eigenvalues y SVD

---

## 💻 Parte 1: Vectores

### 1.1 Definición Geométrica y Algebraica

```python
import numpy as np
import matplotlib.pyplot as plt

# Un vector es una lista ordenada de números
# Geométricamente: flecha con dirección y magnitud

# Vector en R² (2 dimensiones)
v = np.array([3, 4])

# Vector en R³ (3 dimensiones)
w = np.array([1, 2, 3])

# Vector en R^n (n dimensiones) - común en ML
# Ejemplo: imagen 28x28 = 784 dimensiones
image_vector = np.random.randn(784)

# Visualización 2D
def plot_vector(v, origin=[0, 0], color='blue', label=None):
    """Dibuja un vector desde el origen."""
    plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color=color, label=label)

plt.figure(figsize=(8, 8))
plot_vector(np.array([3, 4]), color='blue', label='v = [3, 4]')
plot_vector(np.array([2, 1]), color='red', label='w = [2, 1]')
plt.xlim(-1, 5)
plt.ylim(-1, 5)
plt.grid(True)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.legend()
plt.title('Vectores en R²')
plt.show()
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
import numpy as np

# Vectores de ejemplo
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

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
dot_alt = a @ b
dot_sum = np.sum(a * b)
print(f"Verificación: {dot_alt}, {dot_sum}")
```

### 1.3 Interpretación Geométrica del Producto Punto

```python
import numpy as np

def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula el ángulo entre dos vectores.

    cos(θ) = (a·b) / (||a|| ||b||)
    """
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # Clip para evitar errores numéricos fuera de [-1, 1]
    cos_theta = np.clip(cos_theta, -1, 1)
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    return theta_deg

# Ejemplos
v1 = np.array([1, 0])
v2 = np.array([0, 1])
v3 = np.array([1, 1])
v4 = np.array([-1, 0])

print(f"Ángulo entre [1,0] y [0,1]: {angle_between_vectors(v1, v2):.0f}°")  # 90°
print(f"Ángulo entre [1,0] y [1,1]: {angle_between_vectors(v1, v3):.0f}°")  # 45°
print(f"Ángulo entre [1,0] y [-1,0]: {angle_between_vectors(v1, v4):.0f}°") # 180°

# Interpretación para ML:
# - Producto punto alto → vectores similares (mismo "sentido")
# - Producto punto ≈ 0 → vectores ortogonales (independientes)
# - Producto punto negativo → vectores opuestos
```

### 1.4 Proyección de Vectores

```python
import numpy as np

def project(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Proyecta el vector a sobre el vector b.

    proj_b(a) = (a·b / b·b) · b

    Útil para: PCA, regresión, descomposición de señales
    """
    scalar = np.dot(a, b) / np.dot(b, b)
    return scalar * b

# Ejemplo
a = np.array([3, 4])
b = np.array([1, 0])  # Vector unitario en x

proyeccion = project(a, b)
print(f"Proyección de {a} sobre {b}: {proyeccion}")  # [3, 0]

# La proyección nos da "cuánto" de a está en la dirección de b
```

---

## 💻 Parte 2: Normas y Distancias

### 2.1 Norma L2 (Euclidiana)

```python
import numpy as np

def l2_norm(x: np.ndarray) -> float:
    """
    Norma L2 (Euclidiana): longitud del vector.

    ||x||₂ = √(Σᵢ xᵢ²)

    Uso en ML:
    - Regularización Ridge
    - Normalización de vectores
    - Distancia euclidiana
    """
    return np.sqrt(np.sum(x ** 2))

# Equivalente en NumPy
x = np.array([3, 4])
print(f"||x||₂ = {l2_norm(x)}")           # 5.0
print(f"NumPy:  {np.linalg.norm(x)}")     # 5.0
print(f"NumPy:  {np.linalg.norm(x, 2)}")  # 5.0 (especificando ord=2)

# Vector unitario (normalizado)
def normalize(x: np.ndarray) -> np.ndarray:
    """Convierte vector a longitud 1."""
    return x / np.linalg.norm(x)

x_unit = normalize(x)
print(f"Unitario: {x_unit}")  # [0.6, 0.8]
print(f"Norma del unitario: {np.linalg.norm(x_unit)}")  # 1.0
```

### 2.2 Norma L1 (Manhattan)

```python
import numpy as np

def l1_norm(x: np.ndarray) -> float:
    """
    Norma L1 (Manhattan): suma de valores absolutos.

    ||x||₁ = Σᵢ |xᵢ|

    Uso en ML:
    - Regularización Lasso (promueve sparsity)
    - Robustez a outliers
    """
    return np.sum(np.abs(x))

x = np.array([3, -4, 5])
print(f"||x||₁ = {l1_norm(x)}")                  # 12
print(f"NumPy:  {np.linalg.norm(x, 1)}")         # 12.0

# Comparación L1 vs L2
# L1 penaliza todos los valores igualmente
# L2 penaliza más los valores grandes (cuadrado)
```

### 2.3 Norma L∞ (Máximo)

```python
import numpy as np

def linf_norm(x: np.ndarray) -> float:
    """
    Norma L∞: máximo valor absoluto.

    ||x||∞ = max(|xᵢ|)
    """
    return np.max(np.abs(x))

x = np.array([3, -7, 5])
print(f"||x||∞ = {linf_norm(x)}")            # 7
print(f"NumPy:  {np.linalg.norm(x, np.inf)}") # 7.0
```

### 2.4 Distancia Euclidiana

```python
import numpy as np

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distancia Euclidiana entre dos puntos.

    d(a, b) = ||a - b||₂ = √(Σᵢ (aᵢ - bᵢ)²)

    Uso en ML:
    - KNN (k-nearest neighbors)
    - K-Means (asignación a clusters)
    - Evaluación de similaridad
    """
    return np.linalg.norm(a - b)

# Ejemplo
p1 = np.array([0, 0])
p2 = np.array([3, 4])
print(f"Distancia: {euclidean_distance(p1, p2)}")  # 5.0

# Para múltiples puntos (eficiente)
def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Calcula matriz de distancias entre todos los puntos.
    X: matriz (n_samples, n_features)
    Retorna: matriz (n_samples, n_samples)
    """
    # Usando broadcasting
    # ||a - b||² = ||a||² + ||b||² - 2(a·b)
    sq_norms = np.sum(X ** 2, axis=1)
    distances_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T
    distances_sq = np.maximum(distances_sq, 0)  # Evitar negativos por errores numéricos
    return np.sqrt(distances_sq)

# Test
X = np.array([[0, 0], [3, 4], [1, 1]])
D = pairwise_distances(X)
print("Matriz de distancias:")
print(D)
```

### 2.5 Similitud Coseno

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Similitud coseno: mide el ángulo entre vectores.

    sim(a, b) = (a·b) / (||a|| ||b||)

    Rango: [-1, 1]
    - 1: vectores idénticos (misma dirección)
    - 0: vectores ortogonales
    - -1: vectores opuestos

    Uso en ML:
    - NLP (similitud de documentos)
    - Sistemas de recomendación
    - Embeddings
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distancia coseno = 1 - similitud coseno."""
    return 1 - cosine_similarity(a, b)

# Ejemplos
v1 = np.array([1, 0, 0])
v2 = np.array([1, 0, 0])
v3 = np.array([0, 1, 0])
v4 = np.array([-1, 0, 0])

print(f"Similitud (idénticos):  {cosine_similarity(v1, v2)}")   # 1.0
print(f"Similitud (ortogonales): {cosine_similarity(v1, v3)}")  # 0.0
print(f"Similitud (opuestos):    {cosine_similarity(v1, v4)}")  # -1.0
```

---

## 💻 Parte 3: Matrices

### 3.1 Operaciones Básicas

```python
import numpy as np

# Crear matrices
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # Shape: (2, 3)

B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])  # Shape: (3, 2)

# === SUMA Y RESTA ===
# Solo para matrices del mismo shape
C = np.array([[1, 2, 3], [4, 5, 6]])
print(f"A + C =\n{A + C}")

# === MULTIPLICACIÓN POR ESCALAR ===
print(f"2·A =\n{2 * A}")

# === PRODUCTO MATRICIAL ===
# (m×n) @ (n×p) = (m×p)
# A(2×3) @ B(3×2) = (2×2)
AB = A @ B
print(f"A @ B =\n{AB}")
# [[58, 64],
#  [139, 154]]

# Verificación manual del elemento [0,0]:
# 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58 ✓

# === TRANSPUESTA ===
print(f"A^T =\n{A.T}")
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

### 3.2 Matriz por Vector (Transformación Lineal)

```python
import numpy as np

# La multiplicación matriz-vector es una TRANSFORMACIÓN LINEAL
# y = Ax transforma el vector x al espacio de y

# Ejemplo: Rotación 90° en R²
theta = np.pi / 2  # 90 grados
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

x = np.array([1, 0])
y = R @ x
print(f"Rotar [1,0] 90°: {y}")  # [0, 1]

# En ML: y = Wx + b (capa de red neuronal)
W = np.random.randn(10, 784)  # Pesos: 784 entradas → 10 salidas
b = np.random.randn(10)        # Bias
x = np.random.randn(784)       # Input (imagen aplanada)

y = W @ x + b  # Output de la capa
print(f"Shape de y: {y.shape}")  # (10,)
```

### 3.3 Matriz Inversa

```python
import numpy as np

def safe_inverse(A: np.ndarray) -> np.ndarray:
    """
    Calcula la inversa de A si existe.
    A @ A⁻¹ = A⁻¹ @ A = I

    Uso en ML:
    - Solución cerrada de regresión lineal: θ = (X^T X)⁻¹ X^T y
    - Whitening en PCA
    """
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        print("Matriz no invertible (singular)")
        return None

# Ejemplo
A = np.array([
    [4, 7],
    [2, 6]
])

A_inv = safe_inverse(A)
print(f"A⁻¹ =\n{A_inv}")

# Verificar: A @ A⁻¹ = I
identity = A @ A_inv
print(f"A @ A⁻¹ ≈ I:\n{np.round(identity, 10)}")

# NOTA: En ML, evita calcular inversas cuando sea posible
# Usa np.linalg.solve() en su lugar (más estable numéricamente)
```

### 3.4 Solución de Sistemas Lineales

```python
import numpy as np

# Sistema: Ax = b
# Encontrar x

A = np.array([
    [3, 1],
    [1, 2]
])
b = np.array([9, 8])

# Método 1: Inversa (NO RECOMENDADO)
x_inv = np.linalg.inv(A) @ b

# Método 2: solve (RECOMENDADO - más estable)
x_solve = np.linalg.solve(A, b)

print(f"Solución: x = {x_solve}")  # [2, 3]

# Verificar
print(f"A @ x = {A @ x_solve}")    # [9, 8] ✓
```

---

## 💻 Parte 4: Eigenvalues y Eigenvectors

### 4.1 Concepto

#### Intuición física: el globo terráqueo (eigenvector como eje)

Imagina que tomas un globo terráqueo y lo haces girar.

- Casi todos los puntos de la superficie se mueven.
- Pero hay una línea “especial” que no cambia de dirección: el eje que conecta los polos.

Ese eje es la metáfora del **eigenvector**: una dirección que la transformación “respeta” (no la gira, solo la escala).

El **eigenvalue** te dice cuánto se estira/encoge esa dirección.

#### Código generador de intuición (obligatorio): rejilla deformada por una matriz

Para dejar de ver matrices como tablas y empezar a verlas como “máquinas que deforman el espacio”, usa el script:

- [`visualizations/viz_transformations.py`](../visualizations/viz_transformations.py)

Ejecución:

```bash
python3 visualizations/viz_transformations.py
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
import numpy as np

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

# Ejemplo simple
A = np.array([
    [2, 1],
    [1, 2]
])

# Calcular eigenvalues y eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")    # [3, 1]
print(f"Eigenvectors:\n{eigenvectors}") # columnas son los eigenvectors

# Verificar: Av = λv
v1 = eigenvectors[:, 0]  # primer eigenvector
lambda1 = eigenvalues[0]  # primer eigenvalue

Av = A @ v1
lambda_v = lambda1 * v1

print(f"\nVerificación Av = λv:")
print(f"Av     = {Av}")
print(f"λv     = {lambda_v}")
print(f"¿Iguales? {np.allclose(Av, lambda_v)}")
```

### 4.2 Eigenvalues para PCA

#### Conexión Línea 2: Covarianza como esperanza (estadística)

En estadística, la matriz de covarianza se define conceptualmente como:

```
Cov(X) = E[(X - μ)(X - μ)^T]
```

En la práctica, como no conocemos la distribución real, usamos la versión muestral:

```
Σ ≈ (1/(n-1)) X_centered^T X_centered
```

Este puente es clave para el curso de **Statistical Estimation** (Línea 2): la misma idea de “esperanza” aparece en MLE, varianza, estimadores y pruebas.

```python
import numpy as np

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
    # 1. Centrar datos (restar media)
    X_centered = X - np.mean(X, axis=0)

    # 2. Calcular matriz de covarianza
    # Cov = (1/n) X^T X
    n_samples = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

    # 3. Calcular eigenvalues y eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. Ordenar por eigenvalue (mayor a menor)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 5. Seleccionar top n_components
    components = eigenvectors[:, :n_components].real

    # 6. Proyectar datos
    X_transformed = X_centered @ components

    # 7. Calcular varianza explicada
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues[:n_components].real / total_variance

    return X_transformed, components, explained_variance

# Demo
np.random.seed(42)
X = np.random.randn(100, 5)  # 100 muestras, 5 features

X_pca, components, var_explained = pca_via_eigen(X, n_components=2)

print(f"Shape original: {X.shape}")
print(f"Shape reducido: {X_pca.shape}")
print(f"Varianza explicada: {var_explained}")
print(f"Varianza total explicada: {np.sum(var_explained):.2%}")
```

---

## 💻 Parte 5: SVD (Singular Value Decomposition)

### 5.1 Concepto

```python
import numpy as np

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
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])  # 3×2

U, S, Vt = np.linalg.svd(A, full_matrices=False)

print(f"U shape: {U.shape}")   # (3, 2)
print(f"S shape: {S.shape}")   # (2,) - valores singulares
print(f"Vt shape: {Vt.shape}") # (2, 2)

# Reconstruir A
A_reconstructed = U @ np.diag(S) @ Vt
print(f"\n¿A ≈ U Σ V^T? {np.allclose(A, A_reconstructed)}")
```

### 5.2 PCA via SVD (Método Preferido)

```python
import numpy as np

def pca_via_svd(X: np.ndarray, n_components: int) -> tuple:
    """
    PCA usando SVD (más estable que eigendecomposition).

    La relación: si X = UΣV^T, entonces:
    - V contiene las componentes principales
    - Σ²/(n-1) son las varianzas (eigenvalues de X^TX)
    """
    # 1. Centrar datos
    X_centered = X - np.mean(X, axis=0)

    # 2. SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # 3. Componentes principales (filas de Vt)
    components = Vt[:n_components]

    # 4. Proyectar datos
    X_transformed = X_centered @ components.T

    # 5. Varianza explicada
    variance = (S ** 2) / (X.shape[0] - 1)
    explained_variance_ratio = variance[:n_components] / np.sum(variance)

    return X_transformed, components, explained_variance_ratio

# Demo
np.random.seed(42)
X = np.random.randn(100, 10)

X_pca, components, var_ratio = pca_via_svd(X, n_components=3)

print(f"Varianza explicada por componente: {var_ratio}")
print(f"Varianza total explicada: {np.sum(var_ratio):.2%}")
```

### 5.3 Compresión de Imágenes con SVD

```python
import numpy as np

def compress_image_svd(image: np.ndarray, k: int) -> np.ndarray:
    """
    Comprime una imagen usando truncated SVD.

    Args:
        image: matriz 2D (grayscale) o 3D (RGB)
        k: número de valores singulares a retener

    Returns:
        imagen comprimida
    """
    if len(image.shape) == 2:
        # Grayscale
        U, S, Vt = np.linalg.svd(image, full_matrices=False)
        compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        return np.clip(compressed, 0, 255).astype(np.uint8)
    else:
        # RGB: comprimir cada canal
        compressed = np.zeros_like(image)
        for i in range(3):
            compressed[:, :, i] = compress_image_svd(image[:, :, i], k)
        return compressed

def compression_ratio(original_shape: tuple, k: int) -> float:
    """Calcula ratio de compresión."""
    m, n = original_shape[:2]
    original_size = m * n
    compressed_size = k * (m + n + 1)  # U[:,:k], S[:k], Vt[:k,:]
    return compressed_size / original_size

# Demo (sin cargar imagen real)
# Simular imagen 100x100
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

for k in [5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    ratio = compression_ratio(image.shape, k)
    print(f"k={k}: ratio={ratio:.2%}")
```

---

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

## 🧩 Consolidación (errores comunes + debugging v5 + reto Feynman)

### Errores comunes

- **Confundir dot product con multiplicación elemento-a-elemento:** `a*b` no es `a·b`.
- **Shapes silenciosos:** `a` con shape `(n,)` vs `(n,1)` cambia resultados al multiplicar.
- **Invertir matrices innecesariamente:** evita `inv(A) @ b` y prefiere `solve(A, b)`.
- **PCA sin centrar:** si no haces `X_centered = X - mean`, PCA sale mal.
- **Signo de eigenvectors:** el signo de un eigenvector puede cambiar (`v` o `-v`); no es un bug.

### Debugging / validación (v5)

- Verifica `shapes` en cada operación matricial.
- Si aparece `nan/inf`, revisa escalas y operaciones sensibles.
- Registra hallazgos en `study_tools/DIARIO_ERRORES.md`.
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Reto Feynman (tablero blanco)

Explica en 5 líneas o menos:

1) ¿Por qué `a·b` es una “sombra” y qué significa que sea negativo?
2) ¿Por qué PCA usa eigenvectors de la covarianza?
3) ¿Qué te da SVD que sea más estable que eigendecomposition?

---

## ✅ Checklist de Finalización

- [ ] Puedo calcular producto punto y explicar su significado geométrico
- [ ] Entiendo las diferencias entre normas L1, L2, L∞
- [ ] Puedo calcular distancia euclidiana y similitud coseno
- [ ] Sé multiplicar matrices y entiendo las dimensiones resultantes
- [ ] Puedo explicar qué son eigenvalues/eigenvectors y su uso en PCA
- [ ] Entiendo SVD y puedo usarlo para compresión/PCA
- [ ] Implementé `linear_algebra.py` con todos los tests pasando
- [ ] Puedo proyectar datos usando PCA y explicar varianza explicada

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [01_PYTHON_CIENTIFICO](01_PYTHON_CIENTIFICO.md) | [00_INDICE](00_INDICE.md) | [03_CALCULO_MULTIVARIANTE](03_CALCULO_MULTIVARIANTE.md) |
