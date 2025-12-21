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

### 🧪 Ver para Entender (Laboratorios Interactivos)

- Guía central: [INTERACTIVE_LABS.md](../../Recursos_Adicionales/INTERACTIVE_LABS.md)
- App (Streamlit): `2×2` deformando el espacio + eigenvectors
  - `streamlit run M02_Algebra_Lineal/Laboratorios_Interactivos/transformacion_lineal_app.py`
- Animación (Manim): transformación lineal (shear)
  - `manim -pqh M02_Algebra_Lineal/Laboratorios_Interactivos/animacion_matriz.py AnimacionMatriz`

### Ajuste recomendado (Semanas 3–5, sin cambiar fechas): +2 días de Transformaciones Lineales

Antes de “entrar” a PCA/SVD, dedica 2 días extra a que una matriz deje de ser una tabla y se vuelva una **deformación geométrica**.

- **Día 1 (concepto):** transformaciones lineales como función `R²→R²`, base canónica, determinante como cambio de área.
- **Día 2 (concepto + visual):** eigenvectors como direcciones que la transformación “respeta”.

Ejecución obligatoria (visualización):

- [`viz_transformations.py`](../Laboratorios_Interactivos/viz_transformations.py)

Prompt sugerido para IA (si lo usas):

- "Genera un script de Python usando matplotlib que visualice cómo una matriz de 2x2 deforma una rejilla de puntos unitarios. Quiero ver visualmente qué significa un Eigenvector."

Resultado esperado (criterio de salida de estos 2 días):

- Puedes mirar una matriz `2×2` y predecir cualitativamente si **rota**, **estira**, **inclina (shear)** o **aplasta**.
- Puedes explicar con un dibujo qué significa `Av = λv`.

### Ritmo semanal recomendado (aplicado a Semanas 3–5)

- **Lunes y Martes (Concepto):** lectura + tablero blanco (matriz como transformación; shapes como contrato).
- **Miércoles y Jueves (Implementación):** ejercicios de la guía + asserts de shapes + mini-validaciones.
- **Viernes (Romper cosas):** cambia una matriz “bonita” por una mala (det≈0, rotación + shear) y explica el síntoma (pérdida de dimensión, eigenvalues complejos, inestabilidad).

### Cápsula (obligatoria): Grafo Computacional Manual (puente a M03/M07)

Antes de programar backprop, tienes que poder hacer esto en papel:

1. **Dibuja el grafo** (nodos = operaciones; flechas = dependencias).
2. **Anota shapes** en cada arista (esto evita el 80% de bugs en ML).
3. **Haz forward** y guarda valores intermedios.
4. **Escribe gradientes locales** (derivadas de cada operación).
5. **Propaga hacia atrás** multiplicando gradientes (Chain Rule).

Diagrama mínimo (con shapes):

```
X:(N,D) ──► z = X@w + b : (N,) ──► a = sigmoid(z) : (N,) ──► L(a,y):(scalar)
              w:(D,)  b:()                 y:(N,)
```

Ejemplo “a mano” (ejecutable) con **disciplina de shapes** y comentarios línea‑por‑línea:

```python
import numpy as np  # NumPy: arrays + operaciones vectorizadas


def sigmoid(z: np.ndarray) -> np.ndarray:  # Sigmoide: mapea valores reales a (0,1) elemento a elemento
    return 1.0 / (1.0 + np.exp(-z))  # Implementación estable para valores moderados (ver Log-Sum-Exp en M04)


# ======== (1) DATOS + SHAPES ========
N = 4  # N: número de muestras del batch (mini-batch)
D = 3  # D: número de features por muestra

X = np.random.randn(N, D).astype(float)  # X:(N,D) matriz de features
assert X.shape == (N, D)  # Assert shape: contrato de X

w = np.random.randn(D).astype(float)  # w:(D,) vector de pesos
assert w.shape == (D,)  # Assert shape: contrato de w

b = 0.1  # b:() bias escalar (se suma por broadcasting)

y_true = (np.random.rand(N) > 0.5).astype(float)  # y_true:(N,) etiquetas binarias en {0,1}
assert y_true.shape == (N,)  # Assert shape: contrato de y_true


# ======== (2) FORWARD ========
z = X @ w + b  # z:(N,) porque (N,D)@(D,)=(N,) y luego +b se broadcast
assert z.shape == (N,)  # Assert shape: contrato de z

a = sigmoid(z)  # a:(N,) activación sigmoide por muestra
assert a.shape == (N,)  # Assert shape: contrato de a

loss = float(np.mean((a - y_true) ** 2))  # L: escalar (MSE promedio sobre el batch)


# ======== (3) BACKWARD (CHAIN RULE) ========
# Objetivo: dL/dw y dL/db. Para llegar ahí pasamos por dL/da, da/dz, dz/dw, dz/db.

dL_da = (2.0 / N) * (a - y_true)  # dL/da:(N,) derivada del MSE promedio respecto a a
assert dL_da.shape == (N,)  # Assert shape: dL/da

da_dz = a * (1.0 - a)  # da/dz:(N,) derivada de la sigmoide por elemento (σ(z)(1-σ(z)))
assert da_dz.shape == (N,)  # Assert shape: da/dz

dL_dz = dL_da * da_dz  # dL/dz:(N,) regla de la cadena: dL/dz = dL/da * da/dz
assert dL_dz.shape == (N,)  # Assert shape: dL/dz

# z = X@w + b => para cada muestra i: z_i = sum_j X[i,j]*w[j] + b
# Por eso:
# - dz/dw se acumula sumando sobre el batch
# - dz/db es 1 para cada muestra (y luego sumamos)

dL_dw = X.T @ dL_dz  # dL/dw:(D,) porque (D,N)@(N,)=(D,) (acumula contribuciones del batch)
assert dL_dw.shape == (D,)  # Assert shape: dL/dw

dL_db = float(np.sum(dL_dz))  # dL/db: escalar; suma porque b afecta a todos los z_i con derivada 1


# ======== (4) DEBUGGING INVARIANTS ========
assert np.isfinite(loss)  # La pérdida debe ser finita (si hay NaN/inf, hay bug numérico)
assert np.all(np.isfinite(dL_dw))  # Los gradientes deben ser finitos
assert np.isfinite(dL_db)  # El gradiente del bias debe ser finito
```

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

- Refuerzo diario de shapes: `Herramientas_Estudio/DRILL_DIMENSIONES_NUMPY.md`
- Simulacros: `Herramientas_Estudio/SIMULACRO_EXAMEN_TEORICO.md`
- Evaluación (rúbrica): [Herramientas_Estudio/RUBRICA_v1.md](../../Herramientas_Estudio/RUBRICA_v1.md) (scope `M02` en `rubrica.csv`)
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V5_ESTRATEGICO.md)

### Recursos (cuándo usarlos)

| Prioridad | Recurso | Cuándo usarlo en este módulo | Para qué |
|----------|---------|------------------------------|----------|
| **Obligatorio** | `Herramientas_Estudio/DRILL_DIMENSIONES_NUMPY.md` | Cada vez que una multiplicación/proyección te cambie el shape de forma inesperada | Evitar bugs silenciosos por shapes |
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
# Ejemplos típicos de cómo aparece el álgebra lineal en ML (forma compacta)
# Nota: ŷ representa la predicción; σ suele ser una función no lineal (p. ej., sigmoid/ReLU)
Regresión Lineal:     ŷ = Xθ                 (multiplicación matriz-vector: features X, pesos θ)
Logistic Regression:  ŷ = σ(Xθ)              (mismo Xθ, pero pasando por activación σ)
Neural Network:       ŷ = σ(W₃σ(W₂σ(W₁x)))   (composición de capas: multiplicaciones + activaciones)
PCA:                  X_reduced = XV         (proyección de X sobre eigenvectors V)
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
import numpy as np  # NumPy para representar vectores como arrays y generar datos aleatorios
import matplotlib.pyplot as plt  # Matplotlib para visualizar vectores en 2D

# Un vector es una lista ordenada de números
# Geométricamente: flecha con dirección y magnitud

# Vector en R² (2 dimensiones)
v = np.array([3, 4])  # Vector 2D: sus componentes son (x=3, y=4)

# Vector en R³ (3 dimensiones)
w = np.array([1, 2, 3])  # Vector 3D: (x=1, y=2, z=3)

# Vector en R^n (n dimensiones) - común en ML
# Ejemplo: imagen 28x28 = 784 dimensiones
image_vector = np.random.randn(784)  # Simula un "vector de features" de una imagen aplanada (flatten)

# Visualización 2D
def plot_vector(v, origin=[0, 0], color='blue', label=None):  # Dibuja v desde un origen, con color y etiqueta
    """Dibuja un vector desde el origen."""
    plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color=color, label=label)  # Flecha 2D

plt.figure(figsize=(8, 8))  # Crea una figura cuadrada para ver bien la geometría
plot_vector(np.array([3, 4]), color='blue', label='v = [3, 4]')  # Vector v dibujado desde (0,0)
plot_vector(np.array([2, 1]), color='red', label='w = [2, 1]')  # Otro vector para comparar dirección/magnitud
plt.xlim(-1, 5)  # Límite del eje X
plt.ylim(-1, 5)  # Límite del eje Y
plt.grid(True)  # Rejilla para facilitar lectura de componentes
plt.axhline(y=0, color='k', linewidth=0.5)  # Dibuja eje horizontal (y=0)
plt.axvline(x=0, color='k', linewidth=0.5)  # Dibuja eje vertical (x=0)
plt.legend()  # Muestra leyenda con etiquetas
plt.title('Vectores en R²')  # Título del gráfico
plt.show()  # Renderiza la figura
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 1.1: Definición Geométrica y Algebraica</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Vectores como “flechas” y como arrays: `shape`, notación y visualización
- **ID (opcional):** `M02-T01_1`
- **Duración estimada:** 45–75 min
- **Nivel:** Intro/Intermedio
- **Dependencias:** M01 (NumPy: `ndarray`, `shape`, `dtype`)

#### 2) Objetivo(s) de aprendizaje (medibles)
- Al terminar, el estudiante podrá **representar** un vector en ℝ²/ℝ³/ℝⁿ como `np.ndarray` y **verificar** su `shape`.
- Al terminar, el estudiante podrá **dibujar** un vector 2D (con `quiver`) y **explicar** qué significa dirección vs magnitud.

#### 3) Relevancia y contexto
- En ML, casi todo se reduce a vectores: features, pesos, gradientes y embeddings. Dominar “qué es un vector” evita errores de interpretación y de `shape`.

#### 4) Mapa conceptual / lista de conceptos clave
- Vector (ℝⁿ) ↔ array 1D
- Componentes ↔ coordenadas
- Dirección y magnitud
- `shape` como contrato

#### 5) Definiciones, notación y fórmulas esenciales
- Vector: `v = (v₁, …, vₙ)`.
- En NumPy (convención común): `v.shape == (n,)`.

#### 6) Explicación didáctica (2 niveles)
- **Intuición:** un vector es una instrucción de movimiento: “camina x en X, y en Y”.
- **Implementación:** en NumPy, un vector suele ser 1D; si lo conviertes a 2D (`(n,1)`), cambian reglas de `@` y broadcasting.

#### 7) Ejemplos modelados
- `v = np.array([3,4])` representa un vector en ℝ².
- `image_vector.shape == (784,)` representa un punto en un espacio de 784 dimensiones.

#### 8) Práctica guiada
- Crea `v2 = np.array([2, -1])`, `v3 = np.array([1,2,3])` y escribe `assert v2.ndim == 1`.
- Cambia `v2` a `v2_col = v2.reshape(-1,1)` y observa qué cambia al hacer `v2_col.T @ v2_col`.

#### 9) Práctica independiente / transferencia
- Toma un vector de 784 features y responde: ¿qué significa “una dimensión” en ese caso?

#### 10) Evaluación (formativa)
- ¿Cuál es la diferencia práctica entre `shape (n,)` y `shape (n,1)`?

#### 11) Errores comunes
- Confundir vector columna `(n,1)` con vector 1D `(n,)` y obtener resultados inesperados al multiplicar.

#### 12) Retención (spaced)
- (día 2) define vector en 1 frase y escribe su `shape` típico en NumPy.
- (día 7) explica por qué una imagen 28×28 se representa como vector de 784.

#### 13) Diferenciación
- **Avanzado:** explica por qué `v @ v` no es lo mismo que `v[:,None] @ v[None,:]`.

#### 14) Recursos
- 3Blue1Brown (álgebra lineal) + documentación de NumPy sobre `ndarray`.

#### 15) Nota para el facilitador
- Refuerza el hábito: antes de operar, pedir el `shape` esperado y escribirlo.
</details>

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
import numpy as np  # NumPy para operaciones vectorizadas y producto punto

# Vectores de ejemplo
a = np.array([1, 2, 3])  # Vector a (p. ej., features)
b = np.array([4, 5, 6])  # Vector b (p. ej., pesos)

# === SUMA DE VECTORES ===
# (a + b)ᵢ = aᵢ + bᵢ
suma = a + b  # Suma elemento a elemento
print(f"a + b = {suma}")  # [5, 7, 9]

# === RESTA DE VECTORES ===
resta = a - b  # Resta elemento a elemento
print(f"a - b = {resta}")  # [-3, -3, -3]

# === MULTIPLICACIÓN POR ESCALAR ===
# (c·a)ᵢ = c·aᵢ
escalar = 2 * a  # Multiplica cada elemento por 2
print(f"2·a = {escalar}")  # [2, 4, 6]

# === PRODUCTO PUNTO (DOT PRODUCT) ===
# a·b = Σᵢ aᵢ·bᵢ
# Resultado: escalar
dot = np.dot(a, b)  # Calcula producto punto: Σ(a_i * b_i)
print(f"a·b = {dot}")  # 1*4 + 2*5 + 3*6 = 32

# Alternativamente:
dot_alt = a @ b  # Operador @: producto punto para vectores 1D (equivalente a np.dot)
dot_sum = np.sum(a * b)  # Multiplicación elemento a elemento y suma manual
print(f"Verificación: {dot_alt}, {dot_sum}")  # Comprueba que las 3 implementaciones coinciden
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 1.2: Operaciones con Vectores</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Suma, resta, escalado y producto punto (base de `ŷ = Xθ`)
- **ID (opcional):** `M02-T01_2`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 1.1, M01 (`@`, `np.dot`, `shape`)

#### 2) Objetivo(s) de aprendizaje (medibles)
- Al terminar, el estudiante podrá **implementar** suma/resta/escalado y **verificar** resultados con `assert`.
- Al terminar, el estudiante podrá **calcular** `a·b` de 3 formas (`np.dot`, `@`, `np.sum(a*b)`) y **explicar** por qué coinciden.

#### 3) Relevancia y contexto
- `w·x` es una suma ponderada: es el núcleo de regresión lineal, logística y del forward pass de capas densas.

#### 4) Mapa conceptual / lista de conceptos clave
- operaciones elemento a elemento vs operación de reducción (dot)
- suma ponderada
- contrato de shapes: `(n,)·(n,) → escalar`

#### 5) Definiciones, notación y fórmulas esenciales
- `a·b = Σᵢ aᵢ bᵢ`.
- Para `X:(N,D)` y `w:(D,)`, `X@w:(N,)`.

#### 6) Explicación didáctica (2 niveles)
- **Intuición:** el dot mide alineación (qué tanto `a` “apunta” como `b`).
- **Implementación:** valida shapes antes de operar; si no coinciden, falla temprano.

#### 7) Ejemplos modelados
- Reproducir `a=[1,2,3]`, `b=[4,5,6]` y confirmar `a·b = 32`.
- Mostrar que `a*b` no es dot: produce vector, no escalar.

#### 8) Práctica guiada
- Escribe `dot_manual(a,b)` con un loop y compáralo con `np.dot`.
- Agrega `assert a.shape == b.shape` para forzar el contrato.

#### 9) Práctica independiente / transferencia
- Implementa una neurona: `z = w @ x + b`, con `assert x.shape == w.shape`.

#### 10) Evaluación
- ¿Qué devuelve `(D,) @ (D,)` vs `(N,D) @ (D,)`?

#### 11) Errores comunes
- usar `*` cuando se quería `@` (y viceversa).
- confundir `(n,)` con `(n,1)` y que el resultado cambie por broadcasting.

#### 12) Retención
- (día 2) predice el shape de 6 operaciones con `@`.
- (día 7) explica en 3 líneas por qué `w·x` es una suma ponderada.

#### 13) Diferenciación
- Avanzado: implementa similitud coseno y maneja el caso del vector cero.

#### 14) Recursos
- Glosario: dot product / matrix multiplication; NumPy docs: `dot`, `matmul`.

#### 15) Nota docente
- Hábito: “anota shapes” antes de multiplicar; la intuición llega después.
</details>

### 1.3 Interpretación Geométrica del Producto Punto

```python
import numpy as np  # NumPy para dot/norm/arccos y trabajar con arrays

def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> float:  # Definir función para calcular ángulo entre vectores
    """
    Calcula el ángulo entre dos vectores.

    cos(θ) = (a·b) / (||a|| ||b||)
    """
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # cos(θ) = (a·b)/(||a|| ||b||)
    # Clip para evitar errores numéricos fuera de [-1, 1]
    cos_theta = np.clip(cos_theta, -1, 1)  # Asegura dominio válido para arccos (robustez numérica)
    theta_rad = np.arccos(cos_theta)  # Convierte coseno en ángulo (radianes)
    theta_deg = np.degrees(theta_rad)  # Convierte radianes a grados (más interpretable)
    return theta_deg  # Devuelve el ángulo final

# Ejemplos
v1 = np.array([1, 0])  # Eje x
v2 = np.array([0, 1])  # Eje y (ortogonal a x)
v3 = np.array([1, 1])  # Diagonal (45° respecto a x)
v4 = np.array([-1, 0])  # Dirección opuesta a x (180°)

print(f"Ángulo entre [1,0] y [0,1]: {angle_between_vectors(v1, v2):.0f}°")  # 90°
print(f"Ángulo entre [1,0] y [1,1]: {angle_between_vectors(v1, v3):.0f}°")  # 45°
print(f"Ángulo entre [1,0] y [-1,0]: {angle_between_vectors(v1, v4):.0f}°") # 180°

# Interpretación para ML:
# - Producto punto alto → vectores similares (mismo "sentido")
# - Producto punto ≈ 0 → vectores ortogonales (independientes)
# - Producto punto negativo → vectores opuestos
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 1.3: Interpretación Geométrica del Producto Punto</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Dot product ↔ ángulo ↔ similitud (con estabilidad numérica)
- **ID (opcional):** `M02-T01_3`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 1.2

#### 2) Objetivo(s) de aprendizaje (medibles)
- Al terminar, el estudiante podrá **calcular** el ángulo entre dos vectores usando `arccos` y **validarlo** con casos 0°/90°/180°.
- Al terminar, el estudiante podrá **explicar** por qué `np.clip(cosθ, -1, 1)` previene errores numéricos.

#### 3) Relevancia y contexto
- Similaridad coseno aparece en embeddings, clustering, recuperación de información y métricas de similitud.

#### 4) Mapa conceptual / conceptos clave
- `a·b` → coseno → ángulo
- norma como normalizador
- estabilidad numérica (clip)

#### 5) Definiciones, notación y fórmulas esenciales
- `cos(θ) = (a·b) / (||a|| ||b||)`.

#### 6) Explicación didáctica (2 niveles)
- **Intuición:** el ángulo te dice “qué tan alineados” están.
- **Implementación:** primero compute dot y normas; luego normaliza; luego `arccos` con `clip`.

#### 7) Ejemplos modelados
- ortogonales → 90°
- opuestos → 180°

#### 8) Práctica guiada
- agrega guardas para vector cero y decide política (raise vs return).

#### 9) Transferencia
- implementa `cosine_similarity(a,b)` y rankea 10 vectores por similitud a un query.

#### 10) Evaluación
- ¿Por qué puede salir `cosθ=1.00000002` en float?

#### 11) Errores comunes
- división por 0 cuando alguna norma es 0.

#### 12) Retención
- (día 2) define ortogonalidad y da ejemplo.
- (día 7) explica por qué coseno sirve mejor que dot si magnitudes cambian.

#### 13) Diferenciación
- avanzado: compara coseno vs euclidiana en 2 datasets simples.

#### 14) Recursos
- 3Blue1Brown + glosario del proyecto.

#### 15) Nota docente
- pedir predicción cualitativa del ángulo antes de calcularlo.
</details>

### 1.4 Proyección de Vectores

```python
import numpy as np  # NumPy para producto punto y operar con vectores como arrays

def project(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # Definir función de proyección vectorial
    """
    Proyecta el vector a sobre el vector b.

    proj_b(a) = (a·b / b·b) · b

    Útil para: PCA, regresión, descomposición de señales
    """
    scalar = np.dot(a, b) / np.dot(b, b)  # Escalar de proyección: (a·b)/(b·b)
    return scalar * b  # Reconstruye el vector proyectado en la dirección de b

# Ejemplo
a = np.array([3, 4])  # Vector a a proyectar
b = np.array([1, 0])  # Vector unitario en x (dirección de proyección)

proyeccion = project(a, b)  # Calcula la proyección de a sobre b
print(f"Proyección de {a} sobre {b}: {proyeccion}")  # [3, 0]

# La proyección nos da "cuánto" de a está en la dirección de b
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 1.4: Proyección de Vectores</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Proyección como “sombra” (puente directo a PCA)
- **ID (opcional):** `M02-T01_4`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 1.2–1.3

#### 2) Objetivo(s) de aprendizaje (medibles)
- Al terminar, el estudiante podrá **implementar** `proj_b(a)` y **verificar** el resultado en un caso controlado.
- Al terminar, el estudiante podrá **explicar** por qué aparece el término `b·b` en el denominador.

#### 3) Relevancia y contexto
- PCA y mínimos cuadrados se entienden como proyecciones sobre subespacios.

#### 4) Mapa conceptual / conceptos clave
- componente paralela
- dirección `b`
- escala `(a·b)/(b·b)`

#### 5) Definiciones, notación y fórmulas esenciales
- `proj_b(a) = ((a·b)/(b·b)) b`.

#### 6) Explicación didáctica (2 niveles)
- **Intuición:** “cuánta parte de `a` vive en la dirección de `b`”.
- **Implementación:** calcula escalar → multiplica por `b` → verifica shape.

#### 7) Ejemplos modelados
- proyección sobre eje X en 2D.

#### 8) Práctica guiada
- agrega `assert np.dot(b,b) != 0`.

#### 9) Transferencia
- proyecta un conjunto de puntos sobre un vector unitario y grafica antes/después.

#### 10) Evaluación
- ¿Qué pasa si `b` no es unitario?

#### 11) Errores comunes
- proyectar sobre vector cero.

#### 12) Retención
- (día 2) memoriza la fórmula y explica el denominador.

#### 13) Diferenciación
- avanzado: proyección sobre una base ortonormal `V` (matriz).

#### 14) Recursos
- notas de PCA del módulo + 3Blue1Brown.

#### 15) Nota docente
- insistir en “PCA = proyección” antes de hablar de SVD.
</details>

---

## 💻 Parte 2: Normas y Distancias

### 2.1 Norma L2 (Euclidiana)

```python
import numpy as np  # NumPy para operaciones vectorizadas y norma (linalg.norm)

def l2_norm(x: np.ndarray) -> float:  # Definir función de norma L2 (euclidiana)
    """
    Norma L2 (Euclidiana): longitud del vector.

    ||x||₂ = √(Σᵢ xᵢ²)

    Uso en ML:
    - Regularización Ridge
    - Normalización de vectores
    - Distancia euclidiana
    """
    return np.sqrt(np.sum(x ** 2))  # sqrt(sum(x_i^2)): eleva al cuadrado, suma y saca raíz

# Equivalente en NumPy
x = np.array([3, 4])  # Vector de ejemplo (triángulo 3-4-5)
print(f"||x||₂ = {l2_norm(x)}")           # 5.0  # Llama a nuestra implementación
print(f"NumPy:  {np.linalg.norm(x)}")     # 5.0  # Implementación interna de NumPy (por defecto L2)
print(f"NumPy:  {np.linalg.norm(x, 2)}")  # 5.0 (especificando ord=2)  # Misma norma, pero explícita

# Vector unitario (normalizado)
def normalize(x: np.ndarray) -> np.ndarray:  # Definir función de normalización vectorial
    """Convierte vector a longitud 1."""
    return x / np.linalg.norm(x)  # Divide el vector por su norma para que ||x|| = 1

x_unit = normalize(x)  # Normaliza x para obtener un vector unitario
print(f"Unitario: {x_unit}")  # [0.6, 0.8]  # Componentes escaladas manteniendo dirección
print(f"Norma del unitario: {np.linalg.norm(x_unit)}")  # 1.0  # Verifica que ahora la norma es 1
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 2.1: Norma L2 (Euclidiana)</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Norma L2 como longitud y como regularización (Ridge)
- **ID (opcional):** `M02-T02_1`
- **Duración estimada:** 60–90 min
- **Nivel:** Intermedio
- **Dependencias:** Parte 1 (vectores + dot), M01 (NumPy, `shape`)

#### 2) Objetivo(s) de aprendizaje (medibles)
- Al terminar, el estudiante podrá **calcular** `||x||₂` con una implementación propia y con `np.linalg.norm` y **compararlas** con `assert`.
- Al terminar, el estudiante podrá **normalizar** un vector y **explicar** qué garantiza `||x_unit||₂ = 1`.

#### 3) Relevancia y contexto
- La L2 aparece en:
  - distancias (KNN/K-Means)
  - normalización (coseno similitud)
  - regularización L2 (Ridge / weight decay)

#### 4) Mapa conceptual / lista de conceptos clave
- magnitud (longitud)
- normalización
- invariantes numéricos (`np.isfinite`, norma > 0)

#### 5) Definiciones, notación y fórmulas esenciales
- `||x||₂ = sqrt(Σᵢ xᵢ²)`
- normalización: `x_unit = x / ||x||₂` (si `||x||₂ != 0`)

#### 6) Explicación didáctica (2 niveles)
- **Intuición:** la norma es “qué tan largo” es el vector.
- **Operativa:** calcula cuadrados → suma → raíz; valida casos especiales (vector cero).

#### 7) Ejemplos modelados
- `x=[3,4]` tiene `||x||₂ = 5`.
- `x_unit = x/5` tiene norma 1.

#### 8) Práctica guiada
- Implementa `l2_norm` y verifica:
  - `assert abs(l2_norm(np.array([3,4])) - 5.0) < 1e-9`
  - `assert abs(np.linalg.norm(x_unit) - 1.0) < 1e-9`

#### 9) Práctica independiente / transferencia
- Implementa `safe_normalize(x, eps=1e-12)` que evita dividir por 0.

#### 10) Evaluación
- ¿Qué devuelve `np.linalg.norm(X, axis=1)` si `X.shape==(N,D)`?

#### 11) Errores comunes
- dividir por 0 al normalizar el vector cero.
- confundir L2 (raíz) con L2^2 (sin raíz) en fórmulas.

#### 12) Retención
- (día 2) calcula la norma de 3 vectores a mano.
- (día 7) explica cuándo usarías L2^2 en lugar de L2 (optimización).

#### 13) Diferenciación
- avanzado: deriva por qué L2 penaliza más valores grandes que L1.

#### 14) Recursos
- Glosario: L2 norm; NumPy `linalg.norm`.

#### 15) Nota docente
- Enfatizar el patrón: **definir contrato → implementar → validar con asserts**.
</details>

### 2.2 Norma L1 (Manhattan)

```python
import numpy as np  # NumPy para abs/sum y cálculo de normas

def l1_norm(x: np.ndarray) -> float:  # Definir función de norma L1 (Manhattan)
    """
    Norma L1 (Manhattan): suma de valores absolutos.

    ||x||₁ = Σᵢ |xᵢ|

    Uso en ML:
    - Regularización Lasso (promueve sparsity)
    - Robustez a outliers
    """
    return np.sum(np.abs(x))  # Suma de valores absolutos: Σ|x_i|

x = np.array([3, -4, 5])  # Vector con signo mixto (para ver el efecto del abs)
print(f"||x||₁ = {l1_norm(x)}")                  # 12  # |3|+|−4|+|5| = 12
print(f"NumPy:  {np.linalg.norm(x, 1)}")         # 12.0  # Validación con NumPy

# Comparación L1 vs L2
# L1 penaliza todos los valores igualmente
# L2 penaliza más los valores grandes (cuadrado)
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 2.2: Norma L1 (Manhattan)</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Norma L1 como suma de magnitudes y puente a Lasso
- **ID (opcional):** `M02-T02_2`
- **Duración estimada:** 45–75 min
- **Nivel:** Intermedio
- **Dependencias:** 2.1

#### 2) Objetivo(s) de aprendizaje (medibles)
- Calcular `||x||₁` con `np.sum(np.abs(x))` y validar con `np.linalg.norm(x, 1)`.
- Explicar (intuición) por qué L1 se asocia a sparsity frente a L2.

#### 3) Relevancia y contexto
- Regularización L1 (Lasso), métricas Manhattan y robustez relativa a outliers.

#### 4) Conceptos clave
- `abs`, suma, sparsity

#### 5) Fórmula esencial
- `||x||₁ = Σᵢ |xᵢ|`.

#### 6) Didáctica
- `abs` → `sum` y validar con NumPy.

#### 7) Ejemplo
- `x=[3,-4,5]` ⇒ `||x||₁=12`.

#### 8) Práctica guiada
- Añade `assert abs(l1_norm(x) - np.linalg.norm(x, 1)) < 1e-9`.

#### 9) Transferencia
- Comparar L1 y L2 en un vector con outlier y discutir penalización.

#### 10) Evaluación
- ¿Qué norma usarías si quieres penalizar menos los valores grandes que L2?

#### 11) Errores comunes
- Olvidar `abs`.

#### 12) Retención
- (día 2) calcula L1 de 3 vectores.

#### 13) Diferenciación
- Avanzado: L1 como distancia Manhattan entre puntos.

#### 14) Recursos
- Glosario: L1 norm.

#### 15) Nota docente
- Pedir siempre comparación L1 vs L2 en el mismo ejemplo.
</details>

### 2.3 Norma L∞ (Máximo)

```python
import numpy as np  # NumPy para abs/max y norma infinito

def linf_norm(x: np.ndarray) -> float:  # Definir función de norma L∞ (máximo)
    """
    Norma L∞: máximo valor absoluto.

    ||x||∞ = max(|xᵢ|)
    """
    return np.max(np.abs(x))  # max(|x_i|): toma el mayor valor absoluto

x = np.array([3, -7, 5])  # Vector donde el valor dominante es -7 (en valor absoluto)
print(f"||x||∞ = {linf_norm(x)}")            # 7  # max(|3|,|−7|,|5|) = 7
print(f"NumPy:  {np.linalg.norm(x, np.inf)}") # 7.0  # Validación usando np.inf como orden
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 2.3: Norma L∞ (Máximo)</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Norma infinito como criterio de “peor caso” por componente
- **ID (opcional):** `M02-T02_3`
- **Duración estimada:** 30–60 min
- **Nivel:** Intermedio
- **Dependencias:** 2.1

#### 2) Objetivo(s) de aprendizaje (medibles)
- Calcular `||x||∞` con `np.max(np.abs(x))` y validar con `np.linalg.norm(x, np.inf)`.
- Explicar cuándo L∞ es la métrica adecuada (máximo error permitido).

#### 3) Relevancia y contexto
- Control de tolerancias y errores por componente; debugging numérico.

#### 4) Conceptos clave
- máximo absoluto, “peor caso”

#### 5) Fórmula esencial
- `||x||∞ = maxᵢ |xᵢ|`.

#### 6) Didáctica
- `abs` → `max`.

#### 7) Ejemplo
- `x=[3,-7,5]` ⇒ `||x||∞=7`.

#### 8) Práctica guiada
- Añade `assert linf_norm(x) == np.linalg.norm(x, np.inf)`.

#### 9) Transferencia
- Detecta outliers por feature con `np.max(np.abs(X), axis=0)`.

#### 10) Evaluación
- ¿Qué norma usarías si solo importa el mayor desvío en cualquier componente?

#### 11) Errores comunes
- olvidar `abs`.

#### 12) Retención
- (día 2) da un ejemplo donde L∞ detecta un outlier claro.

#### 13) Diferenciación
- Avanzado: conectar L∞ con bounds/tolerancias en optimización.

#### 14) Recursos
- NumPy `linalg.norm`.

#### 15) Nota docente
- Conectar con “tolerancia máxima” en validación.
</details>

### 2.4 Distancia Euclidiana

```python
import numpy as np  # NumPy para normas/distancias y operaciones vectorizadas

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:  # Definir función de distancia euclidiana
    """
    Distancia Euclidiana entre dos puntos.

    d(a, b) = ||a - b||₂ = √(Σᵢ (aᵢ - bᵢ)²)

    Uso en ML:
    - KNN (k-nearest neighbors)
    - K-Means (asignación a clusters)
    - Evaluación de similaridad
    """
    return np.linalg.norm(a - b)  # ||a-b||: resta vectorial y norma L2

# Ejemplo
p1 = np.array([0, 0])  # Punto 1 (origen)
p2 = np.array([3, 4])  # Punto 2 (triángulo 3-4-5)
print(f"Distancia: {euclidean_distance(p1, p2)}")  # 5.0  # √(3^2 + 4^2) = 5

# Para múltiples puntos (eficiente)
def pairwise_distances(X: np.ndarray) -> np.ndarray:  # Definir función de matriz de distancias
    """
    Calcula matriz de distancias entre todos los puntos.
    X: matriz (n_samples, n_features)
    Retorna: matriz (n_samples, n_samples)
    """
    # Usando broadcasting
    # ||a - b||² = ||a||² + ||b||² - 2(a·b)
    sq_norms = np.sum(X ** 2, axis=1)  # ||x_i||^2 por fila (shape (n_samples,))
    distances_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T  # ||a-b||^2 = ||a||^2+||b||^2-2a·b
    distances_sq = np.maximum(distances_sq, 0)  # Evitar negativos por errores numéricos (redondeo)
    return np.sqrt(distances_sq)  # Raíz elemento a elemento => distancias euclidianas

# Test
X = np.array([[0, 0], [3, 4], [1, 1]])  # 3 puntos en 2D
D = pairwise_distances(X)  # Matriz de distancias entre pares
print("Matriz de distancias:")  # Imprimir encabezado
print(D)  # Imprimir matriz de distancias
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 2.4: Distancia Euclidiana</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Distancias (L2) y matriz de distancias (broadcasting + estabilidad)
- **ID (opcional):** `M02-T02_4`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 2.1 (norma), 1.2 (dot), M01 (broadcasting)

#### 2) Objetivo(s) de aprendizaje (medibles)
- Calcular `d(a,b)=||a-b||₂` y **validarlo** con el caso 3-4-5.
- Construir `D:(N,N)` y **verificar** el `shape` con `assert`.

#### 3) Relevancia y contexto
- Base para KNN, K-Means (asignación) y clustering.

#### 4) Mapa conceptual / lista de conceptos clave
- resta → norma
- identidad `||a-b||² = ||a||² + ||b||² - 2a·b`
- estabilidad: clamp de valores negativos por redondeo

#### 5) Definiciones, notación y fórmulas esenciales
- `d(a,b)=||a-b||₂`.

#### 6) Explicación didáctica (2 niveles)
- **Intuición:** distancia = separación.
- **Implementación:** para pares masivos, usa identidad + broadcasting.

#### 7) Ejemplos modelados
- `d([0,0],[3,4])=5`.

#### 8) Práctica guiada
- Añade `assert D.shape == (X.shape[0], X.shape[0])`.

#### 9) Transferencia
- KNN trivial: para cada fila, el índice del vecino más cercano (excluyendo diagonal).

#### 10) Evaluación
- ¿Por qué `distances_sq` puede ser levemente negativo?

#### 11) Errores comunes
- olvidar `axis` al sumar cuadrados.

#### 12) Retención
- (día 2) calcula 3 distancias a mano.

#### 13) Diferenciación
- Avanzado: cuándo NO conviene construir `D` completa.

#### 14) Recursos
- Glosario: L2 norm, dot product.

#### 15) Nota docente
- Pedir que el alumno escriba shapes antes de programar.
</details>

### 2.5 Similitud Coseno

```python
import numpy as np  # NumPy para dot y norma (linalg.norm)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:  # Definir función de similitud coseno
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
    dot_product = np.dot(a, b)  # Producto punto (alineación)
    norm_a = np.linalg.norm(a)  # Magnitud de a
    norm_b = np.linalg.norm(b)  # Magnitud de b

    if norm_a == 0 or norm_b == 0:  # Caso borde: vector cero => evita división por 0
        return 0.0  # Convención: similitud 0 si no hay dirección definida

    return dot_product / (norm_a * norm_b)  # cos(θ) = (a·b)/(||a|| ||b||)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:  # Definir función de distancia coseno
    """Distancia coseno = 1 - similitud coseno."""
    return 1 - cosine_similarity(a, b)  # Convierte similitud (alto=parecido) en distancia

# Ejemplos
v1 = np.array([1, 0, 0])  # Vector base
v2 = np.array([1, 0, 0])  # Idéntico a v1
v3 = np.array([0, 1, 0])  # Ortogonal a v1
v4 = np.array([-1, 0, 0])  # Opuesto a v1

print(f"Similitud (idénticos):  {cosine_similarity(v1, v2)}")   # 1.0
print(f"Similitud (ortogonales): {cosine_similarity(v1, v3)}")  # 0.0
print(f"Similitud (opuestos):    {cosine_similarity(v1, v4)}")  # -1.0
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 2.5: Similitud Coseno</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Similitud por ángulo (normalización + caso vector cero)
- **ID (opcional):** `M02-T02_5`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 1.2–1.3, 2.1

#### 2) Objetivo(s) de aprendizaje (medibles)
- Implementar `cosine_similarity` y **manejar** el caso `||a||=0` o `||b||=0`.
- Explicar por qué coseno se centra en dirección y no en magnitud.

#### 3) Relevancia y contexto
- Embeddings (NLP/visión), ranking, recomendadores.

#### 4) Conceptos clave
- dot + norma
- normalización
- política para vector cero

#### 5) Fórmulas
- `sim(a,b)=(a·b)/(||a|| ||b||)`.
- `dist=1-sim`.

#### 6) Didáctica
- dot → normas → división (con guardas).

#### 7) Ejemplos
- idénticos 1, ortogonales 0, opuestos -1.

#### 8) Práctica guiada
- Escribe 4 `assert` para los casos base y para vector cero.

#### 9) Transferencia
- Rankea una lista de vectores por similitud a un query.

#### 10) Evaluación
- ¿Por qué coseno es preferible al dot para comparar documentos de distinta longitud?

#### 11) Errores comunes
- no controlar vector cero.

#### 12) Retención
- (día 2) escribe la fórmula de memoria.

#### 13) Diferenciación
- Avanzado: vectorizar para `X:(N,D)` vs `q:(D,)`.

#### 14) Recursos
- Glosario: cosine similarity.

#### 15) Nota docente
- Conectar con embeddings y métricas de similitud reales.
</details>

---

## 💻 Parte 3: Matrices

### 3.1 Operaciones Básicas

```python
import numpy as np  # NumPy para crear matrices (arrays 2D) y operar con ellas

# Crear matrices
A = np.array([  # Crear matriz A
    [1, 2, 3],  # Primera fila
    [4, 5, 6]  # Segunda fila
])  # Shape: (2, 3)  # 2 filas, 3 columnas

B = np.array([  # Crear matriz B
    [7, 8],  # Primera fila
    [9, 10],  # Segunda fila
    [11, 12]  # Tercera fila
])  # Shape: (3, 2)  # 3 filas, 2 columnas

# === SUMA Y RESTA ===
# Solo para matrices del mismo shape
C = np.array([[1, 2, 3], [4, 5, 6]])  # Crear matriz C del mismo shape que A
print(f"A + C =\n{A + C}")  # Suma elemento a elemento (solo si shapes coinciden)

# === MULTIPLICACIÓN POR ESCALAR ===
print(f"2·A =\n{2 * A}")  # Escala cada elemento de A por 2

# === PRODUCTO MATRICIAL ===
# (m×n) @ (n×p) = (m×p)
# A(2×3) @ B(3×2) = (2×2)
AB = A @ B  # Calcular producto matricial
print(f"A @ B =\n{AB}")  # Producto matricial: combina filas de A con columnas de B
# [[58, 64],
#  [139, 154]]

# Verificación manual del elemento [0,0]:
# 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58 ✓

# === TRANSPUESTA ===
print(f"A^T =\n{A.T}")  # Transpuesta: intercambia filas por columnas (shape (3,2))
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 3.1: Operaciones Básicas (Matrices)</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** `shape` en matrices: suma, transpose y producto matricial
- **ID (opcional):** `M02-T03_1`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** M01 (NumPy y `shape`), Parte 1 (vectores)

#### 2) Objetivo(s) de aprendizaje (medibles)
- Predecir y **verificar** shapes en `A+C`, `2*A`, `A@B` y `A.T`.
- Identificar incompatibilidades de shapes antes de ejecutar (falla temprana).

#### 3) Relevancia y contexto
- Las matrices modelan transformaciones lineales y el cómputo central de modelos (`XW`, `W@x`).

#### 4) Mapa conceptual / lista de conceptos clave
- `A.shape=(m,n)`
- elemento-a-elemento vs `@`
- transpuesta

#### 5) Definiciones, notación y fórmulas esenciales
- `(m×n) @ (n×p) = (m×p)`.

#### 6) Explicación didáctica (2 niveles)
- **Intuición:** `@` combina filas con columnas.
- **Operativa:** escribe shapes de entrada → calcula shape de salida → valida con `assert`.

#### 7) Ejemplos modelados
- `A(2,3) @ B(3,2) → (2,2)`.

#### 8) Práctica guiada
- Añade `assert A.shape == (2,3)` y `assert (A@B).shape == (2,2)`.

#### 9) Práctica independiente / transferencia
- Dado `X:(N,D)` y `W:(D,K)`, implementa `Y = X@W` y verifica `Y:(N,K)`.

#### 10) Evaluación
- ¿Por qué `A+C` requiere mismo `shape` pero `A@B` no?

#### 11) Errores comunes
- Confundir `*` con `@`.

#### 12) Retención
- (día 2) predice shapes de 10 productos.

#### 13) Diferenciación
- Avanzado: `X@W` vs `W@X` y cuándo conviene transponer.

#### 14) Recursos
- Glosario: Matrix Multiplication.

#### 15) Nota docente
- Exigir “contrato de shapes” escrito antes de correr el código.
</details>

### 3.2 Matriz por Vector (Transformación Lineal)

```python
import numpy as np  # NumPy para trigonometría y multiplicación matricial

# La multiplicación matriz-vector es una TRANSFORMACIÓN LINEAL
# y = Ax transforma el vector x al espacio de y

# Ejemplo: Rotación 90° en R²
theta = np.pi / 2  # 90 grados (en radianes)
R = np.array([  # Crear matriz de rotación
    [np.cos(theta), -np.sin(theta)],  # Primera fila: cos y -sin
    [np.sin(theta),  np.cos(theta)]  # Segunda fila: sin y cos
])  # Matriz de rotación 2x2

x = np.array([1, 0])  # Vector original sobre el eje x
y = R @ x  # Aplica la transformación lineal (rotación)
print(f"Rotar [1,0] 90°: {y}")  # [0, 1]  # Resultado esperado: pasa a apuntar al eje y

# En ML: y = Wx + b (capa de red neuronal)
W = np.random.randn(10, 784)  # Pesos: 784 entradas → 10 salidas
b = np.random.randn(10)         # Bias (uno por neurona de salida)
x = np.random.randn(784)        # Input (imagen aplanada)

y = W @ x + b  # Output de la capa: (10,784) @ (784,) + (10,) => (10,)
print(f"Shape de y: {y.shape}")  # (10,)  # 10 activaciones de salida
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 3.2: Matriz por Vector (Transformación Lineal)</strong></summary>

#### 1) Metadatos
- **Título:** Transformaciones lineales y el patrón `W@x + b`
- **ID (opcional):** `M02-T03_2`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 3.1, 1.2

#### 2) Objetivos
- Razonar shapes en `(K,D)@(D,)` y en batch `(N,D)@(D,K)`.
- Conectar `W@x + b` con una capa densa.

#### 3) Relevancia
- Es el “forward” más común en ML (proyección + sesgo).

#### 4) Conceptos clave
- matriz como función
- broadcasting del bias

#### 5) Fórmulas
- `y = Wx + b`.

#### 6) Didáctica
- Tabla: entrada `x` → `W` → salida `y` (con shapes).

#### 7) Ejemplos
- Rotación 2D con `R`.

#### 8) Práctica guiada
- Añade `assert W.shape == (10,784)` y `assert x.shape == (784,)`.

#### 9) Transferencia
- Implementa `Y = X@W + b` para `X:(N,D)`.

#### 10) Evaluación
- ¿Por qué `b:(K,)` se suma sin loop?

#### 11) Errores comunes
- mezclar `x@W` con `W@x`.

#### 12) Retención
- (día 2) predice shapes para 5 capas.

#### 13) Diferenciación
- Avanzado: vectorizar el forward para batch.

#### 14) Recursos
- M01 (broadcasting) + glosario.

#### 15) Nota docente
- Exigir “shapes escritos” en cada ejercicio.
</details>

### 3.3 Matriz Inversa

```python
import numpy as np  # NumPy para invertir matrices y manejar errores de álgebra lineal

def safe_inverse(A: np.ndarray) -> np.ndarray:  # Definir función para calcular inversa de forma segura
    """
    Calcula la inversa de A si existe.
    A @ A⁻¹ = A⁻¹ @ A = I

    Uso en ML:
    - Solución cerrada de regresión lineal: θ = (X^T X)⁻¹ X^T y
    - Whitening en PCA
    """
    try:  # Intentar calcular inversa
        return np.linalg.inv(A)  # Calcula A^{-1} si existe (A debe ser cuadrada y no singular)
    except np.linalg.LinAlgError:  # Capturar error de álgebra lineal
        print("Matriz no invertible (singular)")  # Mensaje informativo si det(A)=0 (o numéricamente singular)
        return None  # Devuelve None para indicar que no hay inversa

# Ejemplo
A = np.array([  # Crear matriz A 2x2
    [4, 7],  # Primera fila
    [2, 6]  # Segunda fila
])  # Matriz invertible (determinante ≠ 0)

A_inv = safe_inverse(A)  # Calcular inversa de forma segura
print(f"A⁻¹ =\n{A_inv}")  # Imprime la inversa (si existe)

# Verificar: A @ A⁻¹ = I
identity = A @ A_inv  # Calcular producto de matriz con su inversa
print(f"A @ A⁻¹ ≈ I:\n{np.round(identity, 10)}")  # Redondea para ver la identidad pese a errores numéricos

# NOTA: En ML, evita calcular inversas cuando sea posible
# Usa np.linalg.solve() en su lugar (más estable numéricamente)
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 3.3: Matriz Inversa</strong></summary>

#### 1) Metadatos
- **Título:** Inversa, singularidad y por qué preferimos `solve`
- **ID (opcional):** `M02-T03_3`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 3.1

#### 2) Objetivos
- Entender cuándo existe inversa y cómo validar `A@A⁻¹ ≈ I`.
- Explicar por qué `inv` suele ser peor que `solve`.

#### 3) Relevancia
- Se conecta con soluciones cerradas (regresión) y estabilidad numérica.

#### 4) Conceptos clave
- matriz singular
- `LinAlgError`
- `allclose`/redondeo

#### 5) Fórmula
- `A A^{-1} = I`.

#### 6) Didáctica
- Verifica siempre (no confiar en “que salió”).

#### 7) Ejemplos
- `np.round(A @ A_inv, 10)` para ver `I`.

#### 8) Práctica guiada
- Prueba una matriz singular y observa el error.

#### 9) Transferencia
- Reescribe una solución `inv(A)@b` como `solve(A,b)`.

#### 10) Evaluación
- ¿Qué significa “singular” geométricamente?

#### 11) Errores comunes
- comparar floats con igualdad exacta.

#### 12) Retención
- (día 7) explica “singular” en una frase.

#### 13) Diferenciación
- Avanzado: noción de condición numérica (conceptual).

#### 14) Recursos
- NumPy: `linalg.solve`.

#### 15) Nota docente
- Regla: “`solve` antes que `inv`”.
</details>

### 3.4 Solución de Sistemas Lineales

```python
import numpy as np  # NumPy para resolver sistemas lineales con solve

# Sistema: Ax = b
# Encontrar x

A = np.array([  # Crear matriz de coeficientes
    [3, 1],  # Primera fila
    [1, 2]  # Segunda fila
])  # Matriz de coeficientes (2x2)
b = np.array([9, 8])  # Vector de términos independientes (2,)

# Método 1: Inversa (NO RECOMENDADO)
x_inv = np.linalg.inv(A) @ b  # Funciona, pero suele ser menos estable/eficiente que solve

# Método 2: solve (RECOMENDADO - más estable)
x_solve = np.linalg.solve(A, b)  # Resuelve Ax=b directamente (mejor práctica numérica)

print(f"Solución: x = {x_solve}")  # [2, 3]

# Verificar
print(f"A @ x = {A @ x_solve}")    # [9, 8] ✓  # Comprueba que Ax reproduce b
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 3.4: Sistemas Lineales (`solve`)</strong></summary>

#### 1) Metadatos
- **Título:** Resolver `Ax=b` de forma estable y verificar `A@x ≈ b`
- **ID (opcional):** `M02-T03_4`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 3.3

#### 2) Objetivos
- Usar `np.linalg.solve` y validar con `np.allclose`.
- Diferenciar “resolver” vs “invertir”.

#### 3) Relevancia
- Base de mínimos cuadrados y muchas piezas de ML clásico.

#### 4) Conceptos clave
- sistema lineal
- verificación numérica

#### 5) Fórmula
- `Ax=b`.

#### 6) Didáctica
- Paso 1: resolver. Paso 2: verificar.

#### 7) Ejemplo
- `A@x_solve` reproduce `b`.

#### 8) Práctica guiada
- Añade `assert np.allclose(A @ x_solve, b)`.

#### 9) Transferencia
- Resolver múltiples RHS: `A:(D,D)` y `B:(D,K)`.

#### 10) Evaluación
- ¿Qué pasa si `A` es singular?

#### 11) Errores comunes
- igualdad exacta con floats.

#### 12) Retención
- (día 2) explica por qué verificar es obligatorio.

#### 13) Diferenciación
- Avanzado: conectar con `linalg.lstsq` (conceptual).

#### 14) Recursos
- NumPy: `linalg.solve`, `linalg.lstsq`.

#### 15) Nota docente
- Insistir en `allclose` como estándar.
</details>

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

- [`viz_transformations.py`](../Laboratorios_Interactivos/viz_transformations.py)

Ejecución:

```bash
# Ejecuta el script que dibuja una rejilla y muestra cómo la transforma una matriz
python3 M02_Algebra_Lineal/Laboratorios_Interactivos/viz_transformations.py  # Corre el archivo (requiere librerías como matplotlib)
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
import numpy as np  # NumPy para álgebra lineal (eig) y operaciones con matrices

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
A = np.array([  # Matriz simétrica 2x2 (caso típico donde eigendecomposition es estable)
    [2, 1],  # Primera fila
    [1, 2]  # Segunda fila
])  # Matriz con eigenvalues 3 y 1

# Calcular eigenvalues y eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)  # Devuelve (λ, V) tal que A @ V = V @ diag(λ)

print(f"Eigenvalues: {eigenvalues}")     # Autovalores (factores de escala)
print(f"Eigenvectors:\n{eigenvectors}")  # Autovectores (columnas): direcciones que no rotan

# Verificar: Av = λv
v1 = eigenvectors[:, 0]  # Primer eigenvector (columna 0)
lambda1 = eigenvalues[0]  # Primer eigenvalue asociado a v1

Av = A @ v1  # Aplica la transformación A al eigenvector
lambda_v = lambda1 * v1  # Escala v1 por su eigenvalue (debería coincidir con Av)

print(f"\nVerificación Av = λv:")  # Imprimir encabezado de verificación
print(f"Av     = {Av}")  # Resultado de aplicar A al eigenvector
print(f"λv     = {lambda_v}")  # Resultado de escalar el eigenvector por su eigenvalue
print(f"¿Iguales? {np.allclose(Av, lambda_v)}")  # allclose tolera pequeños errores numéricos (float)
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 4.1: Eigenvalues y Eigenvectors (Concepto)</strong></summary>

#### 1) Metadatos (1–2 líneas)
- **Título:** Direcciones invariantes: `Av = λv` y su lectura geométrica
- **ID (opcional):** `M02-T04_1`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** Parte 3 (matrices), Parte 1 (vectores)

#### 2) Objetivo(s) de aprendizaje (medibles)
- Verificar computacionalmente `Av ≈ λv` y **explicar** qué significa “misma dirección”.
- Distinguir eigenvector (dirección) vs eigenvalue (escala).

#### 3) Relevancia y contexto
- PCA, estabilidad de sistemas, PageRank, y “ejes” naturales de una transformación.

#### 4) Conceptos clave
- transformación lineal
- dirección invariante
- verificación numérica con `allclose`

#### 5) Definiciones y fórmulas
- `Av = λv`.

#### 6) Didáctica
- Siempre: (1) calcular (2) verificar (3) interpretar.

#### 7) Ejemplos
- matriz simétrica 2×2 con eigenvalues {3,1}.

#### 8) Práctica guiada
- Cambia `A` y observa si hay eigenvectors reales.

#### 9) Transferencia
- Conecta con PCA: eigenvectors de covarianza.

#### 10) Evaluación
- ¿Por qué el signo de un eigenvector puede cambiar (`v` vs `-v`)?

#### 11) Errores comunes
- esperar igualdad exacta en floats.

#### 12) Retención
- (día 2) escribe `Av=λv` de memoria y define cada símbolo.

#### 13) Diferenciación
- Avanzado: matrices no simétricas (eigenvalues complejos) (conceptual).

#### 14) Recursos
- 3Blue1Brown: eigenvectors.

#### 15) Nota docente
- Pide interpretación geométrica antes de “correr eig”.
</details>

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
import numpy as np  # NumPy para centrar datos, covarianza y eigendecomposition

def pca_via_eigen(X: np.ndarray, n_components: int) -> tuple:  # Definir función PCA usando eigendecomposition
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
    X_centered = X - np.mean(X, axis=0)  # Centra por columnas (features) para eliminar el sesgo (offset)

    # 2. Calcular matriz de covarianza
    # Cov = (1/n) X^T X
    n_samples = X.shape[0]  # Número de muestras (filas)
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)  # Σ ≈ (1/(n-1)) X^T X

    # 3. Calcular eigenvalues y eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # Autovalores ~ varianzas, autovectores ~ direcciones principales

    # 4. Ordenar por eigenvalue (mayor a menor)
    idx = np.argsort(eigenvalues)[::-1]  # Ordena índices de mayor a menor
    eigenvalues = eigenvalues[idx]  # Reordena eigenvalues
    eigenvectors = eigenvectors[:, idx]  # Reordena columnas de eigenvectors para alinear con eigenvalues

    # 5. Seleccionar top n_components
    components = eigenvectors[:, :n_components].real  # Toma las primeras direcciones (y parte real por estabilidad)

    # 6. Proyectar datos
    X_transformed = X_centered @ components  # Proyección: (n_samples,n_features)@(n_features,n_components)

    # 7. Calcular varianza explicada
    total_variance = np.sum(eigenvalues)  # Suma total de varianza (suma de eigenvalues)
    explained_variance = eigenvalues[:n_components].real / total_variance  # Porcentaje de varianza por componente

    return X_transformed, components, explained_variance  # Devolver datos transformados, componentes y varianza explicada

# Demo
np.random.seed(42)  # Fija semilla para reproducibilidad
X = np.random.randn(100, 5)  # 100 muestras, 5 features (dataset sintético)

X_pca, components, var_explained = pca_via_eigen(X, n_components=2)  # Aplicar PCA para reducir a 2 componentes

print(f"Shape original: {X.shape}")  # Dimensión antes de reducir: (n_samples, n_features)
print(f"Shape reducido: {X_pca.shape}")  # Dimensión después de PCA: (n_samples, n_components)
print(f"Varianza explicada: {var_explained}")  # Proporción por componente (suma <= 1)
print(f"Varianza total explicada: {np.sum(var_explained):.2%}")  # Porcentaje total capturado por las componentes
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 4.2: Eigenvalues para PCA</strong></summary>

#### 1) Metadatos
- **Título:** PCA como eigen de la covarianza (shapes + varianza explicada)
- **ID (opcional):** `M02-T04_2`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** 4.1, 3.1–3.4

#### 2) Objetivos
- Implementar PCA via eig y **verificar** shapes en proyección `(n_samples,n_components)`.
- Interpretar eigenvalues como varianza explicada.

#### 3) Relevancia
- Base conceptual de reducción dimensional y preparación para M06.

#### 4) Conceptos clave
- centrado
- covarianza
- ordenamiento por varianza

#### 5) Fórmulas
- `Σ ≈ (1/(n-1)) Xc^T Xc`.

#### 6) Didáctica
- pipeline: centrar → covarianza → eig → ordenar → proyectar.

#### 7) Ejemplos
- dataset sintético 100×5, reduce a 2.

#### 8) Práctica guiada
- añade `assert X_transformed.shape == (n_samples, n_components)`.

#### 9) Transferencia
- compara con SVD (adelanto 5.2).

#### 10) Evaluación
- ¿por qué se centra `X` antes de PCA?

#### 11) Errores comunes
- no centrar → PCA incorrecto.

#### 12) Retención
- (día 7) describe PCA en 3 pasos.

#### 13) Diferenciación
- Avanzado: explicar por qué `eig` puede dar complejos y por qué se usa `.real`.

#### 14) Recursos
- Glosario: PCA, covariance.

#### 15) Nota docente
- Enfatizar shapes como contrato.
</details>

---

## 💻 Parte 5: SVD (Singular Value Decomposition)

### 5.1 Concepto

```python
import numpy as np  # NumPy para SVD (linalg.svd) y reconstrucción

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
A = np.array([  # Matriz no-cuadrada 3x2 (SVD funciona aunque no sea cuadrada)
    [1, 2],  # Primera fila
    [3, 4],  # Segunda fila
    [5, 6]  # Tercera fila
])  # 3×2

U, S, Vt = np.linalg.svd(A, full_matrices=False)  # full_matrices=False => formas "compactas" (economy SVD)

print(f"U shape: {U.shape}")   # (3, 2)  # U tiene m filas (muestras) y k columnas (k=min(m,n))
print(f"S shape: {S.shape}")   # (2,)    # S es un vector de k valores singulares (σ1 ≥ σ2 ≥ ...)
print(f"Vt shape: {Vt.shape}") # (2, 2)  # Vt tiene k filas y n columnas (direcciones en espacio de features)

# Reconstruir A
A_reconstructed = U @ np.diag(S) @ Vt  # U·Σ·V^T (Σ se construye con diag(S))
print(f"\n¿A ≈ U Σ V^T? {np.allclose(A, A_reconstructed)}")  # Comprueba reconstrucción (debe ser True)
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 5.1: SVD (Concepto)</strong></summary>

#### 1) Metadatos
- **Título:** `A = UΣVᵀ` y por qué SVD es “la navaja suiza”
- **ID (opcional):** `M02-T05_1`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** 3.1–3.4

#### 2) Objetivos
- Interpretar shapes de `U`, `S`, `Vt` y **reconstruir** `A`.
- Explicar por qué SVD aplica a matrices no cuadradas.

#### 3) Relevancia
- PCA estable, compresión, factorization y regularización.

#### 4) Conceptos clave
- valores singulares no negativos
- truncation
- reconstrucción

#### 5) Fórmulas
- `A = UΣVᵀ`.

#### 6) Didáctica
- “descomponer → interpretar shapes → reconstruir y validar”.

#### 7) Ejemplos
- matriz 3×2 con economy SVD.

#### 8) Práctica guiada
- añade `assert np.allclose(A, U@np.diag(S)@Vt)`.

#### 9) Transferencia
- conecta `S` con energía/varianza.

#### 10) Evaluación
- ¿Qué significa `full_matrices=False`?

#### 11) Errores comunes
- confundir `V` con `Vᵀ`.

#### 12) Retención
- (día 2) escribe `A=UΣVᵀ` de memoria.

#### 13) Diferenciación
- Avanzado: relación con eigen de `AᵀA` (conceptual).

#### 14) Recursos
- 3Blue1Brown: SVD.

#### 15) Nota docente
- Insistir en validar con reconstrucción.
</details>

### 5.2 PCA via SVD (Método Preferido)

```python
import numpy as np  # NumPy para centrar datos y aplicar SVD

def pca_via_svd(X: np.ndarray, n_components: int) -> tuple:  # Definir función PCA usando SVD
    """
    PCA usando SVD (más estable que eigendecomposition).

    La relación: si X = UΣV^T, entonces:
    - V contiene las componentes principales
    - Σ²/(n-1) son las varianzas (eigenvalues de X^TX)
    """
    # 1. Centrar datos
    X_centered = X - np.mean(X, axis=0)  # Centra por columnas para que PCA capture varianza y no la media

    # 2. SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # Descompone X_centered = U·diag(S)·Vt

    # 3. Componentes principales (filas de Vt)
    components = Vt[:n_components]  # Cada fila es una componente (dirección en espacio de features)

    # 4. Proyectar datos
    X_transformed = X_centered @ components.T  # (n_samples,n_features)@(n_features,n_components)

    # 5. Varianza explicada
    variance = (S ** 2) / (X.shape[0] - 1)  # S^2/(n-1) ~ eigenvalues de la covarianza
    explained_variance_ratio = variance[:n_components] / np.sum(variance)  # Porcentaje de varianza por componente

    return X_transformed, components, explained_variance_ratio  # Devolver datos transformados, componentes y varianza explicada

# Demo
np.random.seed(42)  # Semilla fija para reproducibilidad del ejemplo
X = np.random.randn(100, 10)  # Dataset sintético: 100 muestras, 10 features

X_pca, components, var_ratio = pca_via_svd(X, n_components=3)  # Reduce a 3 componentes principales

print(f"Varianza explicada por componente: {var_ratio}")  # Vector con proporciones por componente
print(f"Varianza total explicada: {np.sum(var_ratio):.2%}")  # Suma de proporciones (qué tanto se conserva)
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 5.2: PCA via SVD</strong></summary>

#### 1) Metadatos
- **Título:** PCA estable: SVD sobre datos centrados
- **ID (opcional):** `M02-T05_2`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** 5.1, 4.2

#### 2) Objetivos
- Implementar PCA via SVD y **verificar** shapes de proyección.
- Explicar la relación `S^2/(n-1)` con varianzas.

#### 3) Relevancia
- Es la forma recomendada de PCA por estabilidad numérica.

#### 4) Conceptos clave
- centrado
- `Vt` como componentes
- varianza explicada

#### 5) Fórmulas
- `variance = S^2/(n-1)`.

#### 6) Didáctica
- centrar → SVD → componentes → proyectar → varianza.

#### 7) Ejemplos
- dataset 100×10 reducido a 3.

#### 8) Práctica guiada
- añade `assert X_transformed.shape == (100,3)`.

#### 9) Transferencia
- compara con eig (4.2) y discute estabilidad.

#### 10) Evaluación
- ¿por qué SVD puede ser más estable que eig?

#### 11) Errores comunes
- olvidar centrar.

#### 12) Retención
- (día 7) explica PCA via SVD en 4 pasos.

#### 13) Diferenciación
- Avanzado: elegir `n_components` por varianza acumulada.

#### 14) Recursos
- Docs NumPy `linalg.svd`.

#### 15) Nota docente
- Enfatizar validación por `shape` y varianza.
</details>

### 5.3 Compresión de Imágenes con SVD

```python
import numpy as np  # NumPy para SVD, reconstrucción y manipulación de imágenes como arrays

def compress_image_svd(image: np.ndarray, k: int) -> np.ndarray:  # Definir función de compresión de imagen con SVD
    """
    Comprime una imagen usando truncated SVD.

    Args:
        image: matriz 2D (grayscale) o 3D (RGB)
        k: número de valores singulares a retener

    Returns:
        imagen comprimida
    """
    if len(image.shape) == 2:  # Caso 2D: imagen en escala de grises (matriz m×n)
        # Grayscale
        U, S, Vt = np.linalg.svd(image, full_matrices=False)  # SVD de la imagen como matriz
        compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]  # Truncated SVD: conserva solo k componentes
        return np.clip(compressed, 0, 255).astype(np.uint8)  # Recorta a rango válido de píxeles y castea a uint8
    else:  # Caso 3D: imagen a color (RGB)
        # RGB: comprimir cada canal
        compressed = np.zeros_like(image)  # Reserva salida con misma forma (alto, ancho, 3)
        for i in range(3):  # Itera canales: 0=R, 1=G, 2=B
            compressed[:, :, i] = compress_image_svd(image[:, :, i], k)  # Aplica SVD por canal (recursión al caso 2D)
        return compressed  # Devuelve imagen RGB comprimida

def compression_ratio(original_shape: tuple, k: int) -> float:  # Definir función para calcular ratio de compresión
    """Calcula ratio de compresión."""
    m, n = original_shape[:2]  # Alto (m) y ancho (n) de la imagen
    original_size = m * n  # Número de valores en la imagen original (por canal)
    compressed_size = k * (m + n + 1)  # Parámetros aproximados: U(m×k) + S(k) + Vt(k×n)
    return compressed_size / original_size  # Ratio < 1 => compresión (menos parámetros que píxeles)

# Demo (sin cargar imagen real)
# Simular imagen 100x100
image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)  # Matriz de píxeles enteros [0,255]

for k in [5, 10, 20, 50]:  # Probar diferentes números de componentes (k)
    compressed = compress_image_svd(image, k)  # Reconstrucción aproximada con k valores singulares
    ratio = compression_ratio(image.shape, k)  # Estima cuánto se reduce el número de parámetros
    print(f"k={k}: ratio={ratio:.2%}")  # Muestra el ratio (más bajo => más compresión)
```

<details>
<summary><strong>📌 Complemento pedagógico — Tema 5.3: Compresión con SVD</strong></summary>

#### 1) Metadatos
- **Título:** Truncated SVD: trade-off compresión vs error
- **ID (opcional):** `M02-T05_3`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 5.1

#### 2) Objetivos
- Implementar compresión con `k` valores singulares y **explicar** el trade-off.
- Interpretar el ratio como “parámetros guardados”.

#### 3) Relevancia
- Compresión, denoising, aproximaciones de baja-rango.

#### 4) Conceptos clave
- rango efectivo
- truncation
- clipping y dtype

#### 5) Fórmulas
- aproximación: `U_k Σ_k V_kᵀ`.

#### 6) Didáctica
- variar `k` y observar ratio.

#### 7) Ejemplos
- `k` pequeño = más compresión, más error.

#### 8) Práctica guiada
- mide error `||A-A_k||` (si aplica) y relación con `k`.

#### 9) Transferencia
- conecta con embeddings/matrix factorization.

#### 10) Evaluación
- ¿por qué casteamos a `uint8` y hacemos `clip`?

#### 11) Errores comunes
- olvidar `clip` y producir overflow.

#### 12) Retención
- (día 7) explica en 3 líneas qué hace truncated SVD.

#### 13) Diferenciación
- Avanzado: aplicar a imagen RGB por canal.

#### 14) Recursos
- Lectura: low-rank approximation.

#### 15) Nota docente
- Pedir reporte de ratio + ejemplo de salida.
</details>

---

## 🎯 Ejercicios progresivos por tema + soluciones

Reglas:

- **Intenta primero** sin ver soluciones.
- **Tiempo sugerido:** 15–25 min por ejercicio.
- **Éxito mínimo:** tu solución debe pasar los `assert`.

---

### Ejercicio 2.1: Vectores - operaciones básicas y shapes

#### Enunciado

1) **Básico**

- Crea dos vectores `a` y `b` en `R^3`.
- Calcula `a + b`, `a - b` y `3*a`.

2) **Intermedio**

- Verifica con `assert` que la suma es conmutativa: `a + b == b + a`.

3) **Avanzado**

- Convierte un vector 1D `x` con shape `(3,)` en un vector columna `(3, 1)` y verifica shapes.

#### Solución

```python
import numpy as np  # NumPy para arrays, operaciones vectorizadas y comparaciones numéricas

a = np.array([1.0, 2.0, 3.0])  # Define el vector a en R^3
b = np.array([4.0, 5.0, 6.0])  # Define el vector b en R^3

s = a + b  # Suma elemento a elemento (suma vectorial)
d = a - b  # Resta elemento a elemento (diferencia vectorial)
scaled = 3 * a  # Multiplicación por escalar (escala cada componente)

assert np.allclose(a + b, b + a)  # Verifica conmutatividad de la suma

x = np.array([7.0, 8.0, 9.0])  # Vector 1D con shape (3,)
x_col = x.reshape(-1, 1)  # Convierte a vector columna con shape (3, 1)
assert x.shape == (3,)  # Confirma shape original (vector 1D)
assert x_col.shape == (3, 1)  # Confirma shape del vector columna
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.1: Vectores, operaciones y shapes</strong></summary>

#### 1) Idea clave
- En NumPy, un vector 1D tiene shape `(d,)`; un vector columna es 2D con shape `(d, 1)`.
- Muchos “bugs” en ML vienen de mezclar estos dos casos y confiar en broadcasting sin querer.

#### 2) Errores comunes
- Esperar que `(d,)` se comporte siempre como columna.
- Hacer `x.T` sobre un vector 1D: **no cambia** el shape.
</details>

---

### Ejercicio 2.2: Producto punto, ángulo y proyección

#### Enunciado

1) **Básico**

- Calcula `a·b` de 3 formas: `np.dot(a,b)`, `a @ b` y `np.sum(a*b)`.

2) **Intermedio**

- Implementa `cos_theta = (a·b)/(||a|| ||b||)` y verifica que esté en `[-1, 1]`.

3) **Avanzado**

- Implementa la proyección `proj_b(a) = (a·b)/(b·b) * b`.
- Verifica que el residual `r = a - proj_b(a)` sea ortogonal a `b` (`r·b ≈ 0`).

#### Solución

```python
import numpy as np  # NumPy para producto punto, norma, clipping y tests numéricos

a = np.array([1.0, 2.0, 3.0])  # Define vector a
b = np.array([4.0, 5.0, 6.0])  # Define vector b

d1 = np.dot(a, b)  # Producto punto usando np.dot
d2 = a @ b  # Producto punto usando @ (1D @ 1D)
d3 = np.sum(a * b)  # Producto punto como suma de productos elemento a elemento
assert np.isclose(d1, d2) and np.isclose(d2, d3)  # Las tres formas deben coincidir

cos_theta = d1 / (np.linalg.norm(a) * np.linalg.norm(b))  # cos(θ) = (a·b)/(||a|| ||b||)
cos_theta = float(np.clip(cos_theta, -1.0, 1.0))  # Asegura dominio válido para arccos y castea a float
assert -1.0 <= cos_theta <= 1.0  # cos(θ) debe estar en [-1, 1]

proj = (np.dot(a, b) / np.dot(b, b)) * b  # Proyección de a sobre b
r = a - proj  # Residual (debe ser ortogonal a b)
assert np.isclose(np.dot(r, b), 0.0, atol=1e-10)  # Verifica ortogonalidad: r·b ≈ 0
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.2: Dot, coseno y proyección</strong></summary>

#### 1) Idea clave
- `cos(θ) = (a·b)/(||a|| ||b||)` puede salirse de `[-1,1]` por error numérico: por eso se usa `np.clip`.
- La proyección separa `a` en parte paralela a `b` y residual ortogonal `r`.

#### 2) Errores comunes
- No controlar el caso `||a||=0` o `||b||=0` si generalizas.
- Verificar ortogonalidad sin tolerancia (usar `atol`).
</details>

---

### Ejercicio 2.3: Normas L1/L2/L∞ (intuición + verificación)

#### Enunciado

1) **Básico**

- Calcula `||x||_1`, `||x||_2`, `||x||_∞` para `x = [3, -4, 12]`.

2) **Intermedio**

- Verifica que coincidan con `np.linalg.norm(x, ord=...)`.

3) **Avanzado**

- Verifica la desigualdad `||x||_∞ <= ||x||_2 <= ||x||_1`.

#### Solución

```python
import numpy as np  # NumPy para abs/sum/max, sqrt y normas de referencia

x = np.array([3.0, -4.0, 12.0])  # Vector de ejemplo con signos mixtos

n1 = np.sum(np.abs(x))  # Norma L1: suma de valores absolutos
n2 = np.sqrt(np.sum(x * x))  # Norma L2: raíz de la suma de cuadrados
ninf = np.max(np.abs(x))  # Norma L∞: máximo valor absoluto

assert np.isclose(n1, np.linalg.norm(x, 1))  # Valida contra NumPy (L1)
assert np.isclose(n2, np.linalg.norm(x, 2))  # Valida contra NumPy (L2)
assert np.isclose(ninf, np.linalg.norm(x, np.inf))  # Valida contra NumPy (L∞)

assert ninf <= n2 + 1e-12  # Desigualdad: ||x||∞ <= ||x||2
assert n2 <= n1 + 1e-12  # Desigualdad: ||x||2 <= ||x||1
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.3: Normas L1/L2/L∞ y desigualdades</strong></summary>

#### 1) Idea clave
- `||x||_1` suma magnitudes, `||x||_2` mide energía/longitud, `||x||_∞` toma el “peor caso” por componente.
- La cadena `||x||_∞ <= ||x||_2 <= ||x||_1` se cumple en dimensión finita.

#### 2) Errores comunes
- Olvidar `abs` en L1/L∞.
- Comparar floats sin tolerancias (usar `np.isclose` o un epsilon pequeño).
</details>

---

### Ejercicio 2.4: Distancias (euclidiana y manhattan) + matriz de distancias

#### Enunciado

1) **Básico**

- Calcula la distancia euclidiana entre `p1=[0,0]` y `p2=[3,4]`.

2) **Intermedio**

- Calcula la distancia Manhattan para los mismos puntos.

3) **Avanzado**

- Dada una matriz `X` con 3 puntos, construye una matriz de distancias euclidianas `D` de shape `3x3`.
- Verifica que `D` sea simétrica y tenga ceros en la diagonal.

#### Solución

```python
import numpy as np  # NumPy para arrays, norma, broadcasting y asserts

p1 = np.array([0.0, 0.0])  # Punto origen
p2 = np.array([3.0, 4.0])  # Punto a distancia 5 del origen

d2 = np.linalg.norm(p2 - p1)  # Distancia euclidiana (norma L2)
d1 = np.sum(np.abs(p2 - p1))  # Distancia Manhattan (norma L1)

assert np.isclose(d2, 5.0)  # Triángulo 3-4-5
assert np.isclose(d1, 7.0)  # |3| + |4| = 7

X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])  # 3 puntos (n=3) en 2D
sq_norms = np.sum(X ** 2, axis=1)  # Normas al cuadrado ||x_i||^2 por fila (shape: (n,))
D_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * (X @ X.T)  # Usa identidad: ||a-b||^2 = ||a||^2+||b||^2-2a·b
D_sq = np.maximum(D_sq, 0.0)  # Corrige negativos por error numérico
D = np.sqrt(D_sq)  # Matriz de distancias euclidianas

assert D.shape == (3, 3)  # La matriz de distancias debe ser n×n
assert np.allclose(D, D.T)  # Las distancias son simétricas
assert np.allclose(np.diag(D), 0.0)  # Distancia a sí mismo = 0
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.4: Matriz de distancias (vectorización)</strong></summary>

#### 1) Idea clave
- La identidad `||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b` permite construir `D` sin bucles.
- Por redondeo, `D_sq` puede quedar levemente negativo; por eso se hace `np.maximum(D_sq, 0.0)` antes de `sqrt`.

#### 2) Errores comunes
- No corregir negativos numéricos y obtener `nan` en `sqrt`.
- No validar simetría y diagonal cero (invariantes de una matriz de distancias).
</details>

---

### Ejercicio 2.5: Similitud coseno (y el caso del vector cero)

#### Enunciado

1) **Básico**

- Verifica que vectores idénticos tengan similitud coseno ≈ 1.

2) **Intermedio**

- Verifica que vectores ortogonales tengan similitud coseno ≈ 0.

3) **Avanzado**

- Define qué hacer cuando uno de los vectores es cero, evitando división por cero.

#### Solución

```python
import numpy as np  # NumPy para norma, producto punto y pruebas con asserts

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:  # Calcula similitud coseno como dot normalizado
    na = np.linalg.norm(a)  # ||a||_2: magnitud de a
    nb = np.linalg.norm(b)  # ||b||_2: magnitud de b
    if na == 0.0 or nb == 0.0:  # Si hay vector cero, la dirección no está definida
        return 0.0  # Convención: devolver 0 para evitar división por cero
    return float(np.dot(a, b) / (na * nb))  # cos(θ) = (a·b)/(||a|| ||b||)

v1 = np.array([1.0, 2.0, 3.0])  # Vector de referencia
v2 = np.array([1.0, 2.0, 3.0])  # Idéntico a v1
v3 = np.array([1.0, 0.0, 0.0])  # Unitario en eje x
v4 = np.array([0.0, 1.0, 0.0])  # Unitario en eje y (ortogonal a v3)
z = np.array([0.0, 0.0, 0.0])  # Vector cero

assert np.isclose(cosine_similarity(v1, v2), 1.0)  # Misma dirección => similitud ≈ 1
assert np.isclose(cosine_similarity(v3, v4), 0.0)  # Ortogonales => similitud ≈ 0
assert cosine_similarity(v1, z) == 0.0  # Convención del vector cero
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.5: Similitud coseno y vector cero</strong></summary>

#### 1) Idea clave
- La similitud coseno compara **direcciones**, no magnitudes: normaliza por `||a||` y `||b||`.
- Si alguno es vector cero, la dirección no existe: hay que definir una convención (aquí: devolver `0.0`).

#### 2) Errores comunes
- Dividir entre 0 por no checar `||a||` o `||b||`.
- Confundir similitud con distancia (si quieres “distancia coseno”: `1 - cos`).
</details>

---

### Ejercicio 2.6: Multiplicación matricial y razonamiento de shapes

#### Enunciado

1) **Básico**

- Calcula `A @ B` donde `A` es `(2,3)` y `B` es `(3,2)`.

2) **Intermedio**

- Para un dataset `X` con shape `(n,d)`, verifica:
  - `X.T @ X` tiene shape `(d,d)`
  - `X @ X.T` tiene shape `(n,n)`

3) **Avanzado**

- Implementa `y_hat = X @ w + b` con `w` de shape `(d,)` y `b` escalar.

#### Solución

```python
import numpy as np  # NumPy para multiplicación matricial (@) y generación de datos aleatorios

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Matriz A con shape (2, 3)
B = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # Matriz B con shape (3, 2)
C = A @ B  # Producto matricial => shape (2, 2)
assert C.shape == (2, 2)  # Regla: (2,3)@(3,2)=(2,2)

n, d = 7, 4  # n samples (filas), d features (columnas)
X = np.random.randn(n, d)  # Matriz de datos con shape (n, d)
assert (X.T @ X).shape == (d, d)  # Gram de features: (d,n)@(n,d)=(d,d)
assert (X @ X.T).shape == (n, n)  # Gram de samples: (n,d)@(d,n)=(n,n)

w = np.random.randn(d)  # Vector de pesos con shape (d,)
b = 0.25  # Bias escalar
y_hat = X @ w + b  # Predicción lineal (b se “broadcastea” a (n,))
assert y_hat.shape == (n,)  # Debe haber 1 predicción por sample
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.6: Multiplicación matricial y contratos de shape</strong></summary>

#### 1) Idea clave
- `@` sigue la regla `(m,k)@(k,n)->(m,n)`.
- En ML, `X:(n,d)` y `w:(d,)` producen `X @ w:(n,)` (una predicción por muestra).

#### 2) Errores comunes
- Confundir `*` (Hadamard/broadcasting) con `@` (álgebra lineal).
- Usar `w` como `(d,1)` y sorprenderse con salida `(n,1)`.
</details>

---

### Ejercicio 2.7: Sistemas lineales: `solve` vs inversa (estabilidad)

#### Enunciado

1) **Básico**

- Resuelve `Ax=b` usando `np.linalg.solve`.

2) **Intermedio**

- Resuelve con `np.linalg.inv(A) @ b` y compara resultados.

3) **Avanzado**

- Construye una matriz singular y verifica que `np.linalg.solve` lance error.

#### Solución

```python
import numpy as np  # NumPy para solve/inv y asserts numéricos

A = np.array([[3.0, 1.0], [1.0, 2.0]])  # Matriz de coeficientes 2x2
b = np.array([9.0, 8.0])  # Vector del lado derecho (2,)

x_solve = np.linalg.solve(A, b)  # Método preferido: resuelve Ax=b sin invertir A
x_inv = np.linalg.inv(A) @ b  # Alternativa (menos estable): x = A^{-1} b

assert np.allclose(A @ x_solve, b)  # La solución debe satisfacer Ax=b
assert np.allclose(x_solve, x_inv)  # Para matrices bien condicionadas, deben coincidir cerca

S = np.array([[1.0, 2.0], [2.0, 4.0]])  # Matriz singular (fila 2 = 2x fila 1)
try:  # Intentar resolver sistema singular (debe fallar)
    np.linalg.solve(S, np.array([1.0, 1.0]))  # Debe fallar: no hay solución única
    raise AssertionError("Se esperaba LinAlgError para matriz singular")  # Si no falla, el test debe fallar
except np.linalg.LinAlgError:  # Capturar error de álgebra lineal
    pass  # Camino esperado
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.7: `solve` vs inversa (estabilidad numérica)</strong></summary>

#### 1) Idea clave
- `np.linalg.solve(A, b)` es preferible a `np.linalg.inv(A) @ b`:
  - evita formar explícitamente la inversa
  - suele ser más estable y eficiente.
- Si `A` es singular (o casi singular), `solve` debe fallar (no hay solución única).

#### 2) Errores comunes
- Usar la inversa “por costumbre” en vez de `solve`.
- No testear el caso singular y asumir que siempre habrá solución.
</details>

---

### Ejercicio 2.8: Eigenvalues/eigenvectors (verificar Av=λv)

#### Enunciado

1) **Básico**

- Calcula eigenvalues/eigenvectors de una matriz simétrica 2x2.

2) **Intermedio**

- Verifica numéricamente `A @ v ≈ λ v` para cada par.

3) **Avanzado**

- Para matriz simétrica, verifica que los eigenvectors sean ortogonales.

#### Solución

```python
import numpy as np  # NumPy para eigendecomposition, dot y asserts

A = np.array([[2.0, 1.0], [1.0, 2.0]])  # Matriz simétrica (sus eigenvectors deben ser ortogonales)
vals, vecs = np.linalg.eig(A)  # Calcula autovalores/autovectores (pueden venir como complejos)

for i in range(2):  # Iterar sobre los dos eigenvectors
    v = vecs[:, i]  # i-ésimo eigenvector
    lam = vals[i]  # eigenvalue correspondiente
    assert np.allclose(A @ v, lam * v)  # Verifica Av = λv

v0 = vecs[:, 0]  # Primer eigenvector
v1 = vecs[:, 1]  # Segundo eigenvector
assert np.isclose(np.dot(v0, v1), 0.0, atol=1e-10)  # Ortogonalidad (para A simétrica)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.8: Verificación `Av = λv`</strong></summary>

#### 1) Idea clave
- Un eigenvector `v` mantiene su dirección bajo la transformación `A`, solo se escala por `λ`.
- En matrices simétricas, los eigenvectors asociados a eigenvalues distintos son ortogonales.

#### 2) Errores comunes
- Olvidar tolerancia numérica al verificar `A @ v ≈ λ v`.
- No considerar que NumPy puede devolver complejos (por eso a veces se usa `.real`).
</details>

---

### Ejercicio 2.9: PCA (eigen vs SVD) - consistencia de shapes

#### Enunciado

1) **Básico**

- Genera un dataset `X` con shape `(200, 3)` con features correlacionadas.

2) **Intermedio**

- Implementa PCA vía eigendecomposition de la covarianza y reduce a 2D.

3) **Avanzado**

- Implementa PCA vía SVD y verifica:
  - mismos shapes de salida
  - varianza explicada ordenada descendentemente

#### Solución

```python
import numpy as np  # NumPy para datos aleatorios, eig/SVD y validación de shapes

np.random.seed(0)  # Reproducibilidad
n, d = 200, 3  # n muestras, d features
X = np.random.randn(n, d)  # Matriz de datos con shape (n, d)
w_true = np.array([0.5, -1.2, 2.0])  # Pesos reales
noise = 0.1 * np.random.randn(n)  # Ruido aditivo
y = X @ w_true + noise  # Targets: modelo lineal con ruido

Xc = X - X.mean(axis=0)  # Centra features (PCA asume media 0)
cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)  # Matriz de covarianza muestral (3x3)
vals, vecs = np.linalg.eig(cov)  # vals ~ varianzas; vecs ~ direcciones principales
idx = np.argsort(vals)[::-1]  # Ordena por varianza descendente
vals = vals[idx].real  # Seguridad numérica: parte real
vecs = vecs[:, idx].real  # Reordena eigenvectors
comps = vecs[:, :2]  # Toma top-2 componentes (3x2)
Xk = Xc @ comps  # Proyecta datos centrados (n x 2)
ratio = vals[:2] / np.sum(vals)  # Varianza explicada por componente

def pca_svd(X: np.ndarray, k: int):  # PCA usando SVD (más estable en práctica)
    Xc = X - X.mean(axis=0)  # Centra datos
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Xc = U diag(S) Vt
    comps = Vt[:k].T  # Top-k vectores singulares derechos => direcciones principales (3xk)
    Xk = Xc @ comps  # Proyección (n x k)
    var = (S ** 2) / (Xc.shape[0] - 1)  # S^2/(n-1) ~ eigenvalues de covarianza
    ratio = var[:k] / np.sum(var)  # Varianza explicada
    return Xk, comps, ratio  # Devolver datos proyectados, componentes y varianza explicada

X_s, C_s, r_s = pca_svd(X, 2)  # PCA por SVD

assert Xk.shape == (n, 2)  # Shape de datos reducidos
assert X_s.shape == (n, 2)  # Shape de datos reducidos por SVD
assert comps.shape == (3, 2)  # Shape de componentes: (n_features, k)
assert C_s.shape == (3, 2)  # Shape de componentes por SVD
assert ratio.shape == (2,)  # Ratios de varianza explicada
assert r_s.shape == (2,)  # Ratios de varianza explicada por SVD
assert ratio[0] >= ratio[1]  # Debe estar ordenado descendentemente
assert r_s[0] >= r_s[1]  # Verificar orden descendente en SVD también
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.9: PCA (eigen vs SVD)</strong></summary>

#### 1) Idea clave
- PCA busca direcciones de máxima varianza: eigenvectors de la covarianza.
- SVD sobre datos centrados suele ser más estable numéricamente y evita formar `cov` explícita.

#### 2) Errores comunes
- No centrar `X` antes de PCA.
- Confundir shapes: componentes deben ser `(n_features, k)` y proyección `(n_samples, k)`.
</details>

---

### Ejercicio 2.10: SVD - reconstrucción y error (truncated SVD)

#### Enunciado

1) **Básico**

- Calcula la SVD de una matriz `A`.

2) **Intermedio**

- Reconstruye `A` exactamente y verifica `A ≈ U Σ V^T`.

3) **Avanzado**

- Reconstruye con rango `k=1` y `k=2` y verifica que el error disminuya.

#### Solución

```python
import numpy as np  # NumPy para SVD, reconstrucción y normas

A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Matriz ejemplo no-cuadrada (3x2)
U, S, Vt = np.linalg.svd(A, full_matrices=False)  # SVD económica: A = U diag(S) Vt

A_full = U @ np.diag(S) @ Vt  # Reconstrucción usando todos los valores singulares
assert np.allclose(A, A_full)  # Debe coincidir con A (tolerancia numérica)

def trunc(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int):  # Reconstrucción rank-k
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]  # Aproximación de rango k (truncated SVD)

A1 = trunc(U, S, Vt, 1)  # Mejor aproximación de rango 1
A2 = trunc(U, S, Vt, 2)  # Mejor aproximación de rango 2 (aquí: rango completo para 3x2)

err1 = np.linalg.norm(A - A1)  # Error de reconstrucción
err2 = np.linalg.norm(A - A2)  # Error con mayor rango
assert err2 <= err1 + 1e-12  # El error debe bajar al aumentar el rango
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.10: Truncated SVD y trade-off</strong></summary>

#### 1) Idea clave
- Truncated SVD da la **mejor aproximación** de rango `k` (en norma Frobenius).
- A mayor `k`, menor error de reconstrucción.

#### 2) Errores comunes
- Reconstruir usando shapes incorrectos (`U[:, :k]`, `S[:k]`, `Vt[:k, :]`).
- Esperar que `k=2` siempre sea “aproximación”: si el rango ya es completo, reconstruye casi exacto.
</details>

---

### (Bonus) Ejercicio 2.11: De álgebra lineal a ML - regresión cerrada

#### Enunciado

- Genera `X` y `y` para un modelo lineal `y = Xw + noise`.
- Estima `w_hat` con la ecuación normal usando `solve`: `(X^T X) w = X^T y`.
- Verifica que `w_hat` sea cercano a `w_true`.

#### Solución

```python
import numpy as np  # NumPy para datos aleatorios y resolución de sistemas lineales

np.random.seed(1)  # Reproducibilidad
n, d = 300, 3  # n muestras, d features
X = np.random.randn(n, d)  # Matriz de diseño
w_true = np.array([0.5, -1.2, 2.0])  # Pesos reales
noise = 0.1 * np.random.randn(n)  # Ruido aditivo
y = X @ w_true + noise  # Targets: modelo lineal con ruido

XtX = X.T @ X  # Lado izquierdo de ecuación normal (d x d)
Xty = X.T @ y  # Lado derecho (d,)
w_hat = np.linalg.solve(XtX, Xty)  # Resuelve (X^T X) w = X^T y

assert w_hat.shape == (d,)  # Shape esperado del vector de pesos
assert np.linalg.norm(w_hat - w_true) < 0.2  # Debe recuperar pesos razonablemente
```

<details open>
<summary><strong>📌 Complemento pedagógico — Ejercicio 2.11: Ecuación normal y por qué usar `solve`</strong></summary>

#### 1) Idea clave
- La ecuación normal es un sistema lineal: `(XᵀX) w = Xᵀy`.
- `solve(XtX, Xty)` es la forma estándar (evita calcular `inv(XtX)`).

#### 2) Errores comunes
- Usar `inv(XtX) @ Xty` (menos estable).
- Asumir que `XᵀX` siempre es invertible: si hay colinealidad fuerte puede ser singular o mal condicionada.
</details>

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

import numpy as np  # NumPy para arrays, operaciones vectorizadas y álgebra lineal
from typing import Tuple, Optional  # Tipos para anotar retornos (tuplas) y valores opcionales


# ============================================================
# PARTE 1: OPERACIONES CON VECTORES
# ============================================================

def dot_product(a: np.ndarray, b: np.ndarray) -> float:  # Definir función de producto punto con type hints
    """
    Producto punto de dos vectores.

    a·b = Σᵢ aᵢ·bᵢ
    """
    assert a.shape == b.shape, "Vectores deben tener mismo shape"  # Validación: misma dimensión para multiplicar por componentes
    return float(np.sum(a * b))  # Multiplica elemento a elemento y suma: Σ(a_i * b_i) (cast a float nativo)


def vector_angle(a: np.ndarray, b: np.ndarray) -> float:  # Definir función para calcular ángulo entre vectores
    """
    Ángulo entre dos vectores en grados.

    cos(θ) = (a·b) / (||a|| ||b||)
    """
    cos_theta = dot_product(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # Calcula cos(θ) usando dot y magnitudes
    cos_theta = np.clip(cos_theta, -1, 1)  # Recorta por estabilidad numérica (evita valores fuera de [-1, 1])
    return float(np.degrees(np.arccos(cos_theta)))  # arccos => radianes; degrees => grados; cast a float


def project_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # Definir función de proyección vectorial
    """
    Proyección del vector a sobre el vector b.

    proj_b(a) = (a·b / b·b) · b
    """
    scalar = dot_product(a, b) / dot_product(b, b)  # Calcula el escalar (a·b)/(b·b)
    return scalar * b  # Devuelve el vector proyectado: escalar * b (misma dirección que b)


# ============================================================
# PARTE 2: NORMAS
# ============================================================

def l1_norm(x: np.ndarray) -> float:  # Definir función de norma L1 (Manhattan)
    """Norma L1 (Manhattan): ||x||₁ = Σ|xᵢ|"""
    return float(np.sum(np.abs(x)))  # abs => |x_i|; sum => Σ|x_i|; cast a float


def l2_norm(x: np.ndarray) -> float:  # Definir función de norma L2 (Euclidiana)
    """Norma L2 (Euclidiana): ||x||₂ = √(Σxᵢ²)"""
    return float(np.sqrt(np.sum(x ** 2)))  # x**2 => x_i^2; sum => Σx_i^2; sqrt => raíz cuadrada


def linf_norm(x: np.ndarray) -> float:  # Definir función de norma L∞ (Máximo)
    """Norma L∞ (Máximo): ||x||∞ = max|xᵢ|"""
    return float(np.max(np.abs(x)))  # abs => |x_i|; max => máximo valor absoluto


def normalize(x: np.ndarray, ord: int = 2) -> np.ndarray:  # Definir función de normalización vectorial
    """Normaliza vector a norma 1."""
    norm = np.linalg.norm(x, ord=ord)  # Calcula la norma indicada (por defecto L2)
    if norm == 0:  # Caso borde: el vector cero no tiene dirección (evita dividir entre 0)
        return x  # Devuelve tal cual (alternativa común: devolver ceros)
    return x / norm  # Escala para que ||x|| = 1, preservando la dirección


# ============================================================
# PARTE 3: DISTANCIAS
# ============================================================

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:  # Definir función de distancia euclidiana
    """Distancia Euclidiana: d(a,b) = ||a-b||₂"""
    return l2_norm(a - b)  # Resta punto a punto y calcula norma L2 del vector diferencia


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:  # Definir función de distancia Manhattan
    """Distancia Manhattan: d(a,b) = ||a-b||₁"""
    return l1_norm(a - b)  # Resta punto a punto y suma valores absolutos (L1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:  # Definir función de similitud coseno
    """
    Similitud coseno: sim(a,b) = (a·b) / (||a|| ||b||)
    Rango: [-1, 1]
    """
    norm_a = l2_norm(a)  # ||a||: magnitud del vector a
    norm_b = l2_norm(b)  # ||b||: magnitud del vector b
    if norm_a == 0 or norm_b == 0:  # Si algún vector es cero, no hay dirección definida
        return 0.0  # Convención: similitud 0 para evitar división por 0
    return dot_product(a, b) / (norm_a * norm_b)  # (a·b)/(||a||||b||) => cos(θ)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:  # Definir función de distancia coseno
    """Distancia coseno: 1 - similitud_coseno"""
    return 1 - cosine_similarity(a, b)  # Convierte similitud (alto=parecido) en distancia (bajo=parecido)


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:  # Definir función de matriz de distancias euclidianas
    """
    Matriz de distancias euclidianas entre todos los pares.

    Args:
        X: matriz (n_samples, n_features)
    Returns:
        D: matriz (n_samples, n_samples) de distancias
    """
    sq_norms = np.sum(X ** 2, axis=1)  # Calcula ||x_i||^2 por fila (shape: (n_samples,))
    D_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * X @ X.T  # Usa identidad: ||a-b||^2 = ||a||^2+||b||^2-2a·b
    D_sq = np.maximum(D_sq, 0)  # Corrige posibles negativos por redondeo numérico
    return np.sqrt(D_sq)  # Raíz elemento a elemento => distancias euclidianas


# ============================================================
# PARTE 4: EIGENVALUES Y PCA
# ============================================================

def eigendecomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # Definir función de descomposición en eigenvalores
    """
    Calcula eigenvalues y eigenvectors, ordenados por eigenvalue descendente.

    Returns:
        eigenvalues: array de eigenvalues (ordenados)
        eigenvectors: matriz donde columna i es el eigenvector i
    """
    eigenvalues, eigenvectors = np.linalg.eig(A)  # Calcula autovalores/autovectores (pueden venir como complejos)

    # Ordenar por eigenvalue descendente
    idx = np.argsort(eigenvalues)[::-1]  # argsort devuelve índices en orden ascendente; [::-1] invierte a descendente
    eigenvalues = eigenvalues[idx].real  # Reordena eigenvalues y toma parte real (para matrices reales típicas)
    eigenvectors = eigenvectors[:, idx].real  # Reordena columnas de eigenvectors para que coincidan con eigenvalues

    return eigenvalues, eigenvectors  # Devuelve (λ, V) con λ ordenados y V alineado


def pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # Definir función PCA con type hints
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
    X_centered = X - np.mean(X, axis=0)  # Resta la media por feature (columna) para que PCA capture varianza, no offset

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)  # Descompone X = U·diag(S)·Vt (S: valores singulares)

    # Componentes principales
    components = Vt[:n_components]  # Toma las primeras n_components filas de Vt (direcciones principales)

    # Proyectar
    X_transformed = X_centered @ components.T  # Proyección de los datos al subespacio de componentes principales

    # Varianza explicada
    variance = (S ** 2) / (X.shape[0] - 1)  # Varianza por componente: relaciona S^2 con eigenvalues de la covarianza
    explained_variance_ratio = variance[:n_components] / np.sum(variance)  # Proporción de varianza capturada por cada componente

    return X_transformed, components, explained_variance_ratio  # Devolver datos transformados, componentes y varianza explicada


# ============================================================
# TESTS
# ============================================================

def run_tests():  # Definir función de tests para verificar implementaciones
    """Ejecuta tests básicos."""
    print("Ejecutando tests...")  # Imprimir mensaje de inicio

    # Test producto punto
    a = np.array([1, 2, 3])  # Vector a de prueba
    b = np.array([4, 5, 6])  # Vector b de prueba
    assert abs(dot_product(a, b) - 32) < 1e-10  # 1*4 + 2*5 + 3*6 = 32
    print("✓ dot_product")  # Confirmar que el test de producto punto pasó

    # Test normas
    x = np.array([3, 4])  # Vector 3-4-5 para validar L2 y L1
    assert abs(l2_norm(x) - 5) < 1e-10  # L2: √(3^2+4^2)=5
    assert abs(l1_norm(x) - 7) < 1e-10  # L1: |3|+|4|=7
    print("✓ normas")  # Confirmar que los tests de normas pasaron

    # Test distancias
    p1 = np.array([0, 0])  # Punto origen
    p2 = np.array([3, 4])  # Punto a distancia 5 del origen
    assert abs(euclidean_distance(p1, p2) - 5) < 1e-10  # Distancia euclidiana esperada: 5
    print("✓ distancias")  # Confirmar que el test de distancias pasó

    # Test similitud coseno
    v1 = np.array([1, 0])  # Eje x
    v2 = np.array([1, 0])  # Misma dirección que v1
    v3 = np.array([0, 1])  # Eje y (ortogonal a x)
    assert abs(cosine_similarity(v1, v2) - 1) < 1e-10  # Misma dirección => similitud 1
    assert abs(cosine_similarity(v1, v3)) < 1e-10  # Ortogonales => similitud 0
    print("✓ cosine_similarity")  # Confirmar que el test de similitud coseno pasó

    # Test PCA
    np.random.seed(42)  # Fija semilla para reproducibilidad
    X = np.random.randn(50, 10)  # Dataset sintético: 50 muestras, 10 features
    X_pca, _, var_ratio = pca(X, 3)  # Reduce a 3 componentes
    assert X_pca.shape == (50, 3)  # Debe devolver (n_samples, n_components)
    assert np.sum(var_ratio) <= 1.0  # La varianza explicada total no puede exceder 1
    print("✓ PCA")  # Confirmar que el test de PCA pasó

    print("\n¡Todos los tests pasaron!")  # Mensaje de éxito


if __name__ == "__main__":  # Verificar si se ejecuta como script principal
    run_tests()  # Ejecutar tests automáticamente
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
- Registra hallazgos en `Herramientas_Estudio/DIARIO_ERRORES.md`.
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](../../Recursos_Adicionales/Planes_Estrategicos/PLAN_V5_ESTRATEGICO.md)

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

# 📘 Extensión Académica: Nivel MS-AI (University of Colorado Boulder Pathway)

> Esta sección complementa el contenido práctico con rigor matemático formal, contexto histórico y conexiones teóricas profundas requeridas para programas de posgrado en IA.

---

## A.1 Contexto Histórico y Motivación Académica

### El Salto Conceptual del Siglo XIX

El concepto moderno de **espacio vectorial** emergió de tres fuentes independientes:

1. **Hermann Grassmann (1844):** *Die Lineale Ausdehnungslehre* — Primera axiomatización de espacios de dimensión arbitraria
2. **William Rowan Hamilton (1843):** Cuaterniones — Extensión de los números complejos
3. **Josiah Willard Gibbs (1881):** Análisis vectorial para física

**Insight fundamental:** Las propiedades algebraicas de las "flechas" en $\mathbb{R}^3$ son las mismas que las de polinomios, funciones, o cualquier estructura donde puedas "sumar" y "escalar".

### Por Qué Esto Importa para ML de Alto Nivel

En ML moderno, trabajamos con espacios de dimensión $d$ donde:
- $d = 784$ (imagen MNIST aplanada)
- $d = 768$ (embedding BERT-base)
- $d = 12,288$ (embedding GPT-3)

**La intuición geométrica de $\mathbb{R}^2$ y $\mathbb{R}^3$ se extiende matemáticamente a $\mathbb{R}^d$.**

---

## A.2 Analogía de Alto Impacto: El GPS Multidimensional

Imagina un GPS con **784 coordenadas** en lugar de 2. Cada coordenada representa "qué tan cerca estás" de una característica particular.

En el espacio de imágenes MNIST:
- Un "7" y un "1" son puntos **cercanos** (ambos tienen trazos verticales)
- Un "7" y un "0" están **lejos** (geometrías muy diferentes)

> **Machine Learning es, fundamentalmente, geometría en espacios de alta dimensión.**

---

## A.3 Definición Axiomática Formal: Espacio Vectorial

Un **espacio vectorial** $V$ sobre el campo $\mathbb{R}$ es un conjunto equipado con dos operaciones que satisfacen los siguientes axiomas:

### Axiomas de Suma

| Axioma | Expresión Formal |
|--------|------------------|
| Conmutatividad | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| Asociatividad | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| Elemento neutro | $\exists \mathbf{0} \in V : \mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| Inverso aditivo | $\forall \mathbf{v} \in V, \exists (-\mathbf{v}) : \mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |

### Axiomas de Multiplicación Escalar

| Axioma | Expresión Formal |
|--------|------------------|
| Distributividad (escalar) | $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$ |
| Distributividad (vector) | $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$ |
| Asociatividad | $a(b\mathbf{v}) = (ab)\mathbf{v}$ |
| Identidad | $1 \cdot \mathbf{v} = \mathbf{v}$ |

---

## A.4 Producto Interno y Geometría Formal

### El Producto Punto Euclidiano

En $\mathbb{R}^n$, el producto interno estándar es:

$$
\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T \mathbf{y} = \sum_{i=1}^{n} x_i y_i
$$

**Interpretación geométrica rigurosa:**

$$
\mathbf{x} \cdot \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos(\theta)
$$

donde $\theta$ es el ángulo entre los vectores.

### Proyección Ortogonal sobre Subespacio

Para proyectar $\mathbf{b}$ sobre el espacio columna de $A$:

$$
\text{proj}_{\text{col}(A)}(\mathbf{b}) = A(A^T A)^{-1} A^T \mathbf{b}
$$

La matriz $P = A(A^T A)^{-1} A^T$ es la **matriz de proyección ortogonal**.

---

## A.5 Transformaciones Lineales: Matrices como Funciones

### Definición Formal

Una **transformación lineal** $T : V \to W$ satisface:

1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. $T(c\mathbf{v}) = cT(\mathbf{v})$

**Teorema:** Toda transformación lineal de $\mathbb{R}^n$ a $\mathbb{R}^m$ puede representarse como multiplicación por una matriz $A \in \mathbb{R}^{m \times n}$.

### El Determinante como Cambio de Volumen

Para $A \in \mathbb{R}^{n \times n}$:

$$
|\det(A)| = \text{factor de escala del volumen n-dimensional}
$$

- $\det(A) > 0$: Preserva orientación
- $\det(A) < 0$: Invierte orientación
- $\det(A) = 0$: Colapsa a dimensión menor (singular)

---

## A.6 Familia de Normas $L^p$ (Teoría Completa)

### Definición General

Para $p \geq 1$, la **norma $L^p$**:

$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}
$$

### Casos Límite

$$
\|\mathbf{x}\|_\infty = \lim_{p \to \infty} \|\mathbf{x}\|_p = \max_i |x_i|
$$

### Propiedades (Axiomas de Norma)

1. **Positividad:** $\|\mathbf{x}\| \geq 0$, y $\|\mathbf{x}\| = 0 \Leftrightarrow \mathbf{x} = \mathbf{0}$
2. **Homogeneidad:** $\|c\mathbf{x}\| = |c| \|\mathbf{x}\|$
3. **Desigualdad triangular:** $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

---

## A.7 Eigendescomposición: Teoría Espectral

### Definición

Un escalar $\lambda$ es **eigenvalue** de $A$ si existe $\mathbf{v} \neq \mathbf{0}$ tal que:

$$
A\mathbf{v} = \lambda \mathbf{v}
$$

### Teorema Espectral (Matrices Simétricas)

Si $A = A^T$, entonces:

$$
A = Q \Lambda Q^T
$$

donde:
- $Q$: matriz ortogonal de eigenvectores ($Q^T Q = I$)
- $\Lambda$: matriz diagonal de eigenvalues

**Consecuencia crucial:** Los eigenvectores de matrices simétricas son ortogonales entre sí.

---

## A.8 SVD: El Teorema Fundamental

### Teorema de Descomposición en Valores Singulares

Para **cualquier** matriz $A \in \mathbb{R}^{m \times n}$:

$$
A = U \Sigma V^T
$$

donde:
- $U \in \mathbb{R}^{m \times m}$: vectores singulares izquierdos (ortonormal)
- $\Sigma \in \mathbb{R}^{m \times n}$: valores singulares $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{n \times n}$: vectores singulares derechos (ortonormal)

### Interpretación Geométrica

Cualquier transformación lineal = **Rotación → Escalado → Rotación**

$$
A\mathbf{x} = U(\Sigma(V^T\mathbf{x}))
$$

### Teorema de Eckart-Young (Aproximación Óptima)

La mejor aproximación de rango $k$ a $A$ (en norma Frobenius o espectral):

$$
A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

minimiza $\|A - B\|$ sobre todas las matrices $B$ de rango $\leq k$.

---

## A.9 Condicionamiento Numérico

### Número de Condición

$$
\kappa(A) = \|A\| \|A^{-1}\| = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}
$$

### Interpretación

- $\kappa \approx 1$: Sistema bien condicionado
- $\kappa \gg 1$: Sistema mal condicionado (pequeños errores en input → grandes errores en output)
- $\kappa = \infty$: Matriz singular

**Regla práctica:** Se pierden aproximadamente $\log_{10}(\kappa)$ dígitos de precisión al resolver $Ax = b$.

---

## A.10 Conexiones con Cursos del MS-AI Pathway

| Concepto de Este Módulo | Curso del Pathway | Aplicación Directa |
|-------------------------|-------------------|-------------------|
| Producto punto / Similitud coseno | DTSA 5509 (Supervised Learning) | Kernels lineales, predicciones |
| Normas L1/L2 | DTSA 5509 | Regularización Ridge/Lasso |
| Eigenvalues/Eigenvectors | DTSA 5510 (Unsupervised Learning) | PCA, reducción dimensional |
| Multiplicación matricial | DTSA 5511 (Deep Learning) | Forward pass en redes |
| SVD | DTSA 5510 | Compresión, sistemas de recomendación |
| Condicionamiento | Todos | Estabilidad numérica |

---

## A.11 Referencias Académicas

1. **Strang, G. (2016).** *Introduction to Linear Algebra*, 5th ed. Wellesley-Cambridge Press.
2. **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*, Chapter 2. MIT Press.
3. **Trefethen, L.N., & Bau, D. (1997).** *Numerical Linear Algebra*. SIAM.
4. **3Blue1Brown - Essence of Linear Algebra:** [youtube.com/playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

---

*Extensión académica desarrollada siguiendo el curriculum del MS-AI Pathway de la University of Colorado Boulder.*

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [M01_Fundamentos_Python](../../M01_Fundamentos_Python/) | [README](../../README.md) | [M03_Calculo_Optimizacion](../../M03_Calculo_Optimizacion/) |
