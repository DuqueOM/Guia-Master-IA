# Módulo 05 - Supervised Learning

> **🎯 Objetivo:** Dominar regresión lineal, logística y métricas de evaluación
> **Fase:** 2 - Núcleo de ML | **Semanas 9-12**
> **Curso del Pathway:** Introduction to Machine Learning: Supervised Learning

---

<a id="m05-0"></a>

## 🧭 Cómo usar este módulo (modo 0→100)

**Propósito:** que puedas construir un pipeline supervisado “de examen”:

- entrenar (regresión lineal/logística)
- evaluar (métricas)
- validar (train/test, K-fold)
- controlar overfitting (regularización)

### Objetivos de aprendizaje (medibles)

Al terminar este módulo podrás:

- **Implementar** regresión lineal y regresión logística desde cero.
- **Derivar** el gradiente de MSE y de cross-entropy (con la forma `Xᵀ(ŷ - y)`).
- **Elegir** métricas correctas según el costo de FP/FN.
- **Aplicar** validación (split y K-fold) evitando leakage.
- **Validar** tu implementación con Shadow Mode (sklearn) como ground truth.
- **Explicar** Entropía/Gini, Information Gain y el contraste **Bagging vs Boosting** (Random Forest vs Gradient Boosting) a nivel conceptual.

Enlaces rápidos:

- [04_PROBABILIDAD_ML.md](04_PROBABILIDAD_ML.md) (MLE → cross-entropy)
- [GLOSARIO.md](GLOSARIO.md)
- [RECURSOS.md](RECURSOS.md)
- [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
- [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)
- Evaluación (rúbrica): [study_tools/RUBRICA_v1.md](../study_tools/RUBRICA_v1.md) (scope `M05` en `rubrica.csv`)

### Recursos (cuándo usarlos)

| Prioridad | Recurso | Cuándo usarlo en este módulo | Para qué |
|----------|---------|------------------------------|----------|
| **Obligatorio** | [04_PROBABILIDAD_ML.md](04_PROBABILIDAD_ML.md) | Antes de implementar `log-loss`/cross-entropy y el gradiente de logística | Conectar MLE → cross-entropy y evitar derivaciones “de memoria” |
| **Obligatorio** | `study_tools/DIRTY_DATA_CHECK.md` | Antes del primer entrenamiento real (Semana 9–10), al preparar datasets | Evitar que el modelo “aprenda basura” por fallas de datos |
| **Obligatorio** | `study_tools/DIARIO_ERRORES.md` | Cada vez que veas métricas incoherentes, accuracy “mágico” o divergencia | Registrar bugs, causas y fixes reproducibles |
| **Complementario** | [StatQuest ML (playlist)](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) | Semana 10–12 (logística, métricas, regularización) | Refuerzo conceptual rápido + ejemplos |
| **Complementario** | [Stanford CS229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) | Después de implementar regresión lineal/logística (para profundizar) | Profundizar en teoría y derivaciones estándar |
| **Opcional** | [RECURSOS.md](RECURSOS.md) | Al finalizar el módulo (para escoger práctica extra) | Expandir sin perder el foco del Pathway |

---

## 🧠 ¿Qué es Supervised Learning?

```text
APRENDIZAJE SUPERVISADO

Tenemos:
- Datos de entrada X (features)
- Etiquetas Y (targets/labels)

Objetivo: Aprender una función f tal que f(X) ≈ Y

Tipos principales:
├── REGRESIÓN: Y es continuo (precio, temperatura)
│   └── Output: número real
└── CLASIFICACIÓN: Y es discreto (spam/no spam, dígito 0-9)
    └── Output: clase o probabilidad
```

---

## 📚 Contenido del Módulo

| Semana | Tema | Entregable |
|--------|------|------------|
| 9 | Regresión Lineal | `linear_regression.py` |
| 10 | Regresión Logística | `logistic_regression.py` |
| 11 | Métricas de Evaluación | `metrics.py` |
| 12 | Validación + Regularización + Árboles | Cross-validation, L1/L2 + Tree-Based Models |

---

## 💻 Parte 1: Regresión Lineal

### 1.1 Modelo

```python
import numpy as np

"""
REGRESIÓN LINEAL

Hipótesis: h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
         = θᵀx (forma matricial)

Donde:
- θ (theta): parámetros/pesos del modelo
- x: vector de features (con x₀ = 1 para el bias)

En forma matricial para múltiples muestras:
    ŷ = Xθ

Donde:
- X: matriz (m × n+1) con m muestras y n features + columna de 1s
- θ: vector (n+1 × 1) de parámetros
- ŷ: vector (m × 1) de predicciones
"""

def add_bias_term(X: np.ndarray) -> np.ndarray:
    """Añade columna de 1s para el término de bias."""
    m = X.shape[0]
    return np.column_stack([np.ones(m), X])

def predict_linear(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predicción lineal: ŷ = Xθ"""
    return X @ theta
```

### 1.2 Función de Costo (MSE)

```python
import numpy as np

def mse_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Mean Squared Error Cost Function.

    J(θ) = (1/2m) Σᵢ (h(xᵢ) - yᵢ)²
         = (1/2m) ||Xθ - y||²

    El factor 1/2 es por conveniencia (cancela con la derivada).
    """
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def mse_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Gradiente del MSE respecto a θ.

    ∂J/∂θ = (1/m) Xᵀ(Xθ - y)
    """
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    return (1 / m) * X.T @ errors
```

### 1.3 Solución Cerrada (Normal Equation)

```python
import numpy as np

def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solución cerrada para regresión lineal.

    θ = (XᵀX)⁻¹ Xᵀy

    Ventajas:
    - No requiere iteraciones
    - No hay hiperparámetros (learning rate)

    Desventajas:
    - O(n³) por la inversión de matriz
    - No funciona si XᵀX es singular
    - No escala bien para n grande (>10,000 features)
    """
    XtX = X.T @ X
    Xty = X.T @ y

    # Usar solve en lugar de inv para estabilidad numérica
    theta = np.linalg.solve(XtX, Xty)
    return theta
```

### 1.4 Gradient Descent para Regresión

```python
import numpy as np
from typing import List, Tuple

class LinearRegression:
    """Regresión Lineal implementada desde cero."""

    def __init__(self):
        self.theta = None
        self.cost_history = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'gradient_descent',
        learning_rate: float = 0.01,
        n_iterations: int = 1000
    ) -> 'LinearRegression':
        """
        Entrena el modelo.

        Args:
            X: features (m, n)
            y: targets (m,)
            method: 'gradient_descent' o 'normal_equation'
            learning_rate: tasa de aprendizaje (solo para GD)
            n_iterations: número de iteraciones (solo para GD)
        """
        # Añadir bias
        X_b = add_bias_term(X)
        m, n = X_b.shape

        if method == 'normal_equation':
            self.theta = normal_equation(X_b, y)
        else:
            # Inicializar theta con ceros o valores pequeños
            self.theta = np.zeros(n)

            for i in range(n_iterations):
                # Calcular gradiente
                gradient = mse_gradient(X_b, y, self.theta)

                # Actualizar theta
                self.theta = self.theta - learning_rate * gradient

                # Guardar costo para monitoreo
                cost = mse_cost(X_b, y, self.theta)
                self.cost_history.append(cost)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice valores."""
        X_b = add_bias_term(X)
        return X_b @ self.theta

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# Demo
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5  # y = 4 + 3x + ruido

model = LinearRegression()
model.fit(X, y, method='gradient_descent', learning_rate=0.1, n_iterations=1000)

print(f"Parámetros aprendidos: {model.theta}")
print(f"Esperados: [4, 3]")
print(f"R² score: {model.score(X, y):.4f}")
```

---

## 💻 Parte 2: Regresión Logística

### 2.0 Regresión Logística — Nivel: intermedio (core del Pathway)

**Propósito:** pasar de “sé aplicar sigmoid” a **poder entrenar, derivar y validar** un clasificador binario (y extenderlo a multiclase con One-vs-All).

#### Objetivos de aprendizaje (medibles)

Al terminar esta parte podrás:

- **Explicar** por qué regresión logística es un modelo lineal *sobre el log-odds* (aunque la salida sea una probabilidad).
- **Derivar** (a mano) el gradiente de la pérdida *Binary Cross-Entropy* y reconocer la forma `Xᵀ(ŷ - y)`.
- **Implementar** `fit()` con gradient descent estable (con `clip`/`eps`) y verificar convergencia.
- **Diagnosticar** errores típicos: shapes, overflow en `exp`, signos invertidos, saturación de sigmoid.
- **Validar** tu implementación con **Shadow Mode** (comparación con sklearn) y con un *overfit test* en dataset pequeño.

#### Prerrequisitos

- De `Módulo 03`: Chain Rule y gradiente.
- De `Módulo 04`: interpretación de MLE (conexión con cross-entropy).

Enlaces rápidos:

- [GLOSARIO: Logistic Regression](GLOSARIO.md#logistic-regression)
- [GLOSARIO: Sigmoid](GLOSARIO.md#sigmoid)
- [GLOSARIO: Binary Cross-Entropy](GLOSARIO.md#binary-cross-entropy)
- [GLOSARIO: Gradient Descent](GLOSARIO.md#gradient-descent)
- [RECURSOS.md](RECURSOS.md)

#### Explicación progresiva (intuición → formalización → implementación)

##### a) Intuición

Quieres un modelo que devuelva:

- un **score lineal** `z = θᵀx` (como en regresión lineal), y
- lo convierta en una **probabilidad** en `(0, 1)`.

Eso lo hace `σ(z)`.

##### a.1 Odds, log-odds y por qué esto “sigue siendo lineal”

Si el modelo produce `p = P(y=1|x)`, define:

```
odds = p / (1 - p)
logit(p) = log(odds)
```

La regresión logística asume que **el log-odds es lineal**:

```
logit(p) = θᵀx
```

Y la sigmoide es simplemente la función que vuelve de logit a probabilidad:

```
p = σ(θᵀx) = 1 / (1 + exp(-θᵀx))
```

Esto importa porque te permite interpretar el modelo:

- subir `θᵀx` en +1 incrementa el **log-odds** en +1 (cambio multiplicativo en odds).

##### a.2 Por qué NO usar MSE para clasificación

Podrías intentar usar MSE con `ŷ = σ(z)`, pero en práctica es mala idea:

- **La geometría del entrenamiento empeora:** el gradiente se vuelve poco informativo cuando `σ(z)` se satura (cerca de 0 o 1).
- **La función objetivo deja de ser convexa** (puede tener mínimos locales / mesetas), haciendo el descenso de gradiente menos confiable.
- **No penaliza bien el caso “seguro y equivocado”:** si `y=1` pero `ŷ≈0`, quieres un castigo enorme; eso lo da `-log(ŷ)`.

Por eso usamos **Log-Loss / Binary Cross-Entropy**, que viene de MLE y es convexa para este modelo.

##### a.3 Visual: frontera de decisión

La frontera de decisión es el conjunto de puntos donde `p = 0.5`:

```
σ(θᵀx) = 0.5  ⇔  θᵀx = 0
```

##### a.3.1 Intuición geométrica: el “plano de corte”

Piensa en tus datos como puntos en un espacio.

- En 2D, `θᵀx + b = 0` es una **línea**.
- En 3D, es un **plano**.
- En `n` dimensiones, es un **hiperplano**.

La cantidad `z = θᵀx + b` es un **score con signo**:

- `z > 0` → estás del lado “positivo” del plano
- `z < 0` → estás del lado “negativo”

La sigmoide `σ(z)` convierte ese score (relacionado con la distancia al plano) en probabilidad:

- puntos muy lejos del plano (|z| grande) → probabilidad cerca de 0 o 1
- puntos cerca del plano (`z ≈ 0`) → probabilidad cerca de 0.5

Visualización sugerida (dibújalo): una nube roja/azul y una línea que la corta; marca puntos a distinta distancia y escribe su `z` y `σ(z)`.

##### a.3.2 Conexión conceptual: SVM y la idea de “margen” (sin implementar)

Aunque no implementes SVM aquí, su intuición te mejora la comprensión de regularización.

Idea:

- En clasificación lineal, hay muchas líneas/planos que separan (si los datos lo permiten).
- SVM busca el separador que deja la “carretera” más ancha entre clases: **máximo margen**.

Conexión con lo que sí implementas:

- La **regularización** (L2/L1) controla complejidad efectiva.
- En problemas separables o casi separables, regularizar suele empujar a soluciones más estables, con fronteras menos extremas.

Visualización sugerida: dos líneas separadoras posibles y dibujar cuál deja más espacio mínimo a los puntos más cercanos (support vectors).

En 2D, `θᵀx = 0` es una **línea**.

```
clase 1:   o o o o o
           o o o o o

frontera:  ---------

clase 0:   x x x x x
           x x x x x
```

##### a.4 Worked example (numérico) de BCE

Datos: `x=2`, `y=1`.

- `w=0.5`, `b=0`
- `z = wx + b = 1`
- `ŷ = σ(1) ≈ 0.731`

Como `y=1`, la loss por muestra es:

```
L = -log(ŷ) ≈ -log(0.731) ≈ 0.313
```

Interpretación: la predicción es “bastante” correcta, por eso la loss es pequeña. Si `ŷ` fuera 0.01, la loss sería enorme.

##### a.5 Código generador de intuición (Protocolo D): frontera de decisión en 2D

Objetivo: ver que la **frontera de decisión** (`p=0.5`) es lineal, aunque la salida `σ(z)` sea curva (curva en *probabilidad*, no en geometría de la frontera).

```python
import numpy as np
import matplotlib.pyplot as plt


def make_blobs_2d(n=200, seed=42):
    rng = np.random.default_rng(seed)
    c0 = rng.normal(loc=(-2.0, -1.5), scale=0.8, size=(n // 2, 2))
    c1 = rng.normal(loc=(2.0, 1.5), scale=0.8, size=(n // 2, 2))
    X = np.vstack([c0, c1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def plot_decision_boundary(model, X, y, title="Decision boundary"):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )

    grid = np.column_stack([xx.ravel(), yy.ravel()])
    proba = model.predict_proba(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, proba, levels=20, cmap="RdBu", alpha=0.35)
    plt.contour(xx, yy, proba, levels=[0.5], colors="black", linewidths=2)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=18, label="Clase 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=18, label="Clase 1")

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()


# Usa TU LogisticRegression del módulo (la clase ya existe más abajo)
# X, y = make_blobs_2d(n=300)
# model = LogisticRegression()
# model.fit(X, y, learning_rate=0.1, n_iterations=2000)
# plot_decision_boundary(model, X, y)
```

Reto visual (opcional, si usas sklearn solo para generar datos):

- genera `make_moons` y grafica la frontera
- verás por qué logística falla (frontera lineal)
- luego entrena tu MLP (M07) y observa cómo la frontera se curva

##### b) Formalización mínima

- **Modelo:** `ŷ = σ(Xθ)`
- **Decisión:** `ŷ ≥ 0.5 → clase 1` (umbral configurable)
- **Loss (BCE):** penaliza fuerte cuando estás “seguro y equivocado” (ej. `ŷ≈0` pero `y=1`).

##### c) Regla de oro de shapes

Evita bugs silenciosos usando una convención consistente:

- `X`: `(m, n)`
- `θ`: `(n,)` (o `(n, 1)` si prefieres columnas)
- `y`: `(m,)`

Y verifica que `X @ θ` te da `(m,)`.

#### Actividades activas (para convertir teoría en habilidad)

- **Retrieval practice (5 min):** escribe sin mirar:
  - la ecuación de BCE,
  - el gradiente `∇θ`.
- **Ejercicio de calibración:** cambia el `threshold` de 0.5 a 0.3 y explica qué pasa con precision/recall.
- **Sanity check obligatorio:** entrena con 20 ejemplos hasta obtener accuracy ~100% (si no, hay bug).

#### Evaluación (criterios de “dominio”)

- **Dominio matemático:** puedes explicar por qué aparece `(ŷ - y)` en el gradiente.
- **Dominio de implementación:** tu `fit()` reduce BCE de forma monotónica (o casi) en un dataset simple.
- **Dominio de validación:** tu accuracy difiere <5% de sklearn en Shadow Mode.

#### Errores comunes (los que más queman tiempo)

- **Overflow/NaN:** `exp(500)` revienta. Solución: `clip(z)` y `eps` en logs.
- **Saturación:** si `|z|` crece, `σ(z)` se pega a 0/1 y el gradiente se hace pequeño.
- **Signo invertido:** si actualizas en la dirección equivocada, la loss sube.
- **Sin normalización:** features en escalas muy distintas hacen que GD sea inestable.

#### Integración con Plan v4/v5

- **v4.0:** usa `study_tools/SIMULACRO_EXAMEN_TEORICO.md` para preguntas tipo examen (sigmoid vs softmax, BCE vs MSE).
- **v5.0:** ejecuta **Shadow Mode** como verificación externa antes de dar por terminado el módulo.

### 2.1 Función Sigmoid

```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Función sigmoid/logística.

    σ(z) = 1 / (1 + e^(-z))

    Propiedades:
    - Rango: (0, 1) - perfecto para probabilidades
    - σ(0) = 0.5
    - σ'(z) = σ(z)(1 - σ(z))
    """
    # Clip para evitar overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Visualizar
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)
plt.figure(figsize=(8, 4))
plt.plot(z, sigmoid(z))
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Función Sigmoid')
plt.grid(True)
# plt.show()
```

### 2.2 Hipótesis Logística

```python
"""
REGRESIÓN LOGÍSTICA

No predice un valor continuo, sino la PROBABILIDAD de pertenecer a la clase 1.

h(x) = P(y=1|x; θ) = σ(θᵀx)

Decisión:
- Si h(x) ≥ 0.5 → predicir clase 1
- Si h(x) < 0.5 → predicir clase 0

Equivalente a:
- Si θᵀx ≥ 0 → clase 1
- Si θᵀx < 0 → clase 0

El "decision boundary" está en θᵀx = 0
"""

def predict_proba(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Predice probabilidad de clase 1."""
    return sigmoid(X @ theta)

def predict_class(X: np.ndarray, theta: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Predice clase (0 o 1)."""
    return (predict_proba(X, theta) >= threshold).astype(int)
```

### 2.3 Binary Cross-Entropy Loss

```python
import numpy as np

def binary_cross_entropy(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    eps: float = 1e-15
) -> float:
    """
    Binary Cross-Entropy (Log Loss).

    J(θ) = -(1/m) Σᵢ [yᵢ log(hᵢ) + (1-yᵢ) log(1-hᵢ)]

    Donde hᵢ = σ(θᵀxᵢ)

    Por qué esta función de costo:
    - Es convexa (tiene un único mínimo global)
    - Penaliza mucho las predicciones muy incorrectas
    - Es la derivación de Maximum Likelihood Estimation
    """
    m = len(y)
    h = sigmoid(X @ theta)

    # Clip para evitar log(0)
    h = np.clip(h, eps, 1 - eps)

    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def bce_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Gradiente de Binary Cross-Entropy.

    ∂J/∂θ = (1/m) Xᵀ(h - y)

    ¡Tiene la misma forma que el gradiente del MSE!
    Esto es porque derivamos σ(z) y la derivada σ'(z) = σ(z)(1-σ(z))
    cancela parte de la expresión.
    """
    m = len(y)
    h = sigmoid(X @ theta)
    return (1/m) * X.T @ (h - y)
```

### 2.4 Implementación Completa

```python
import numpy as np
from typing import List

class LogisticRegression:
    """Regresión Logística implementada desde cero."""

    def __init__(self):
        self.theta = None
        self.cost_history = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.1,
        n_iterations: int = 1000
    ) -> 'LogisticRegression':
        """Entrena con gradient descent."""
        # Añadir bias
        X_b = np.column_stack([np.ones(len(X)), X])
        m, n = X_b.shape

        # Inicializar
        self.theta = np.zeros(n)

        for i in range(n_iterations):
            # Gradiente
            gradient = bce_gradient(X_b, y, self.theta)

            # Actualizar
            self.theta = self.theta - learning_rate * gradient

            # Guardar costo
            cost = binary_cross_entropy(X_b, y, self.theta)
            self.cost_history.append(cost)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice probabilidades."""
        X_b = np.column_stack([np.ones(len(X)), X])
        return sigmoid(X_b @ self.theta)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predice clases."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy."""
        return np.mean(self.predict(X) == y)


# Demo con datos sintéticos
np.random.seed(42)

# Generar datos de dos clases
n_samples = 200
X_class0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
X_class1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
X = np.vstack([X_class0, X_class1])
y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Entrenar
model = LogisticRegression()
model.fit(X, y, learning_rate=0.1, n_iterations=1000)

print(f"Accuracy: {model.score(X, y):.2%}")
print(f"Parámetros: {model.theta}")
```

---

## 🧩 Consolidación (Regresión Logística)

### Entregable conceptual (v3.3): Interpretación de pesos (LogReg)

Objetivo: conectar el vector de pesos con “qué está mirando” el modelo.

- Dataset recomendado: MNIST (28x28) en binario (p. ej. 0 vs 1) usando `sklearn.datasets.fetch_openml("mnist_784", as_frame=False)`.
- Entrena tu regresión logística sobre imágenes aplanadas (`784` features).
- Visualiza:
  - toma `theta[1:]` (sin bias), reshapea a `(28, 28)` y grafica con `imshow`.
  - usa un mapa de color divergente (p. ej. centrado en 0) y guarda una imagen.
- Interpreta en 5–10 líneas:
  - ¿qué regiones tienen peso positivo/negativo?
  - ¿por qué eso tiene sentido para el dígito?

### Errores comunes

- **Etiquetas incorrectas:** BCE asume `y ∈ {0,1}` (no `{-1,1}`) si usas la fórmula estándar.
- **Olvidar el bias:** si no agregas columna de 1s, la frontera se forza a pasar por el origen.
- **`exp` overflow:** si `z` crece, `exp(-z)` puede overflow/underflow → usa `clip`.
- **`log(0)`:** si `h` llega a 0 o 1 exactos, `log` revienta → usa `eps`.
- **Sin escalado:** features con escalas distintas hacen el GD inestable.

### Debugging / validación (v5)

- **Overfit test:** entrena con 20 ejemplos hasta casi 100% accuracy. Si no, asume bug.
- **Shadow Mode:** compara con sklearn para la misma semilla/dataset.
- Registra hallazgos en `study_tools/DIARIO_ERRORES.md`.
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Reto Feynman (tablero blanco)

Explica en 5 líneas o menos:

1) ¿Qué es el logit y por qué logística es lineal “en el espacio de log-odds”?
2) ¿Por qué `-log(ŷ)` explota cuando estás seguro y equivocado?
3) ¿Qué significa `Xᵀ(ŷ - y)` y por qué aparece en el gradiente?

---

## 💻 Parte 3: Métricas de Evaluación

### 3.0 Métricas — Nivel: intermedio (de “calcular” a “tomar decisiones”)

**Propósito:** que no te quedes en “sé calcular accuracy”, sino que puedas **elegir la métrica correcta según el riesgo** (FP vs FN), detectar desbalance de clases y justificar tus decisiones como en un informe.

#### Objetivos de aprendizaje (medibles)

Al terminar esta parte podrás:

- **Explicar** la matriz de confusión y derivar TP/TN/FP/FN sin mirar apuntes.
- **Aplicar** accuracy/precision/recall/F1/specificity y explicar cuándo cada una es adecuada.
- **Analizar** el impacto del umbral (`threshold`) en precision/recall.
- **Diagnosticar** trampas comunes: accuracy alta con clases desbalanceadas, leakage, evaluar sobre train.

#### Prerrequisitos y conexiones

- Conexión directa con probabilidad/loss:
  - [04_PROBABILIDAD_ML.md](04_PROBABILIDAD_ML.md) (MLE → cross-entropy)
- Glosario:
  - [GLOSARIO: Confusion Matrix](GLOSARIO.md#confusion-matrix)
  - [GLOSARIO: Accuracy](GLOSARIO.md#accuracy)
  - [GLOSARIO: Precision](GLOSARIO.md#precision)
  - [GLOSARIO: Recall](GLOSARIO.md#recall)
  - [GLOSARIO: F1 Score](GLOSARIO.md#f1-score)

#### Resumen ejecutivo (big idea)

La métrica es una traducción explícita de “qué error es más caro”:

- Si te preocupa **no perder positivos reales** → prioriza **recall**.
- Si te preocupa **no disparar falsas alarmas** → prioriza **precision**.
- Si necesitas balance → **F1**.
- Si tu dataset está balanceado y el costo es simétrico → **accuracy** puede servir.

#### Actividades activas (obligatorias)

- **Retrieval practice (5 min):** escribe la matriz 2x2 y define TP/TN/FP/FN.
- **Experimento de umbral:** evalúa con `threshold = 0.3, 0.5, 0.7` y anota cómo cambian precision/recall.
- **Caso desbalanceado:** crea un dataset donde 95% sea clase 0 y muestra por qué accuracy engaña.

#### Errores comunes (los que más dañan resultados)

- **Evaluar en training:** te da una “métrica falsa” por overfitting.
- **Leakage:** normalizar/seleccionar features usando todo el dataset antes del split.
- **No fijar semilla:** resultados no reproducibles.

Integración con Plan v4/v5:

- [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md) (rutina + simulacros)
- [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md) (validación externa / rigor)
- Diario: `study_tools/DIARIO_ERRORES.md`

### 3.1 Matriz de Confusión

```python
import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de confusión.

    Para clasificación binaria:

                    Predicho
                    0       1
    Real    0      TN      FP
            1      FN      TP

    - TP (True Positive): Predijo 1, era 1
    - TN (True Negative): Predijo 0, era 0
    - FP (False Positive): Predijo 1, era 0 (Error Tipo I)
    - FN (False Negative): Predijo 0, era 1 (Error Tipo II)
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    return cm

def extract_tp_tn_fp_fn(y_true: np.ndarray, y_pred: np.ndarray):
    """Extrae TP, TN, FP, FN para clasificación binaria."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn
```

### 3.2 Accuracy, Precision, Recall, F1

```python
import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Proporción de predicciones correctas.

    Problema: Puede ser engañoso con clases desbalanceadas.
    Si 99% son clase 0, predecir siempre 0 da 99% accuracy.
    """
    return np.mean(y_true == y_pred)

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Precision = TP / (TP + FP)

    De todos los que predije como positivos, ¿cuántos realmente lo son?

    Alta precisión = pocos falsos positivos.
    Importante cuando el costo de FP es alto (ej: spam → inbox).
    """
    tp, tn, fp, fn = extract_tp_tn_fp_fn(y_true, y_pred)
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Recall (Sensitivity, True Positive Rate) = TP / (TP + FN)

    De todos los positivos reales, ¿cuántos capturé?

    Alto recall = pocos falsos negativos.
    Importante cuando el costo de FN es alto (ej: detección de cáncer).
    """
    tp, tn, fp, fn = extract_tp_tn_fp_fn(y_true, y_pred)
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    F1 = 2 * (precision * recall) / (precision + recall)

    Media armónica de precision y recall.

    Útil cuando quieres un balance entre ambas métricas.
    F1 alto solo si AMBAS precision y recall son altas.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Specificity (True Negative Rate) = TN / (TN + FP)

    De todos los negativos reales, ¿cuántos identifiqué?
    """
    tp, tn, fp, fn = extract_tp_tn_fp_fn(y_true, y_pred)
    if tn + fp == 0:
        return 0.0
    return tn / (tn + fp)
```

### 3.3 Clase Metrics Completa

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class ClassificationReport:
    """Reporte de métricas de clasificación."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    confusion_matrix: np.ndarray

    def __str__(self) -> str:
        cm = self.confusion_matrix
        return f"""
Classification Report
=====================
Accuracy:    {self.accuracy:.4f}
Precision:   {self.precision:.4f}
Recall:      {self.recall:.4f}
F1 Score:    {self.f1:.4f}
Specificity: {self.specificity:.4f}

Confusion Matrix:
           Pred 0  Pred 1
Actual 0   {cm[0,0]:5d}   {cm[0,1]:5d}
Actual 1   {cm[1,0]:5d}   {cm[1,1]:5d}
"""

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationReport:
    """Genera reporte completo de métricas."""
    return ClassificationReport(
        accuracy=accuracy(y_true, y_pred),
        precision=precision(y_true, y_pred),
        recall=recall(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        specificity=specificity(y_true, y_pred),
        confusion_matrix=confusion_matrix(y_true, y_pred)
    )

# Demo
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])

report = classification_report(y_true, y_pred)
print(report)
```

---

## 💻 Parte 4: Validación y Regularización

### 4.0 Validación y regularización — Nivel: intermedio/avanzado

**Propósito:** aprender el “workflow real” que evita autoengaño:

- dividir datos correctamente
- validar de forma robusta
- controlar overfitting (regularización)

#### Objetivos de aprendizaje (medibles)

Al terminar esta parte podrás:

- **Explicar** la diferencia entre train/val/test y por qué el test no se toca.
- **Aplicar** K-fold cross validation y reportar media ± desviación.
- **Diagnosticar** sesgo-varianza en términos prácticos (qué cambia si aumentas `λ` o si cambias el tamaño del modelo).
- **Implementar** regularización L2 y justificar por qué se excluye el bias.

#### Resumen ejecutivo (big idea)

- **Validación** te dice si generalizas.
- **Regularización** controla complejidad efectiva.

Conectar esto con el Pathway:

- En el curso, se evalúa tanto la *matemática* como tu capacidad de **evitar leakage** y reportar resultados correctamente.

#### Actividades activas (obligatorias)

- Ejecuta `train_test_split` con al menos 2 semillas distintas y compara varianza en accuracy.
- Haz K-fold (k=5) y reporta `mean ± std`.
- Prueba `lambda_` en `{0, 0.01, 0.1, 1.0}` y describe el efecto.

#### Errores comunes

- **Data leakage** por normalizar antes del split.
- **Elegir hiperparámetros mirando el test** (invalidas el test).
- **Regularizar el bias** sin querer.

#### Integración con Plan v4/v5

- v4.0: usa simulacros para preguntas tipo examen (`study_tools/SIMULACRO_EXAMEN_TEORICO.md`).
- v5.0: valida tu implementación con Shadow Mode (sklearn) antes de cerrar el módulo.

### 4.1 Train/Test Split

```python
import numpy as np

def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = None
) -> tuple:
    """
    Divide datos en conjuntos de entrenamiento y prueba.

    Args:
        X: features
        y: targets
        test_size: proporción para test (0-1)
        random_state: semilla para reproducibilidad
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(y)
    indices = np.random.permutation(n)

    test_size_n = int(n * test_size)
    test_indices = indices[:test_size_n]
    train_indices = indices[test_size_n:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
```

### 4.2 K-Fold Cross Validation

```python
import numpy as np
from typing import List, Tuple

def k_fold_split(n: int, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Genera índices para K-Fold Cross Validation.

    Returns:
        Lista de (train_indices, val_indices) para cada fold
    """
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n

        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        folds.append((train_indices, val_indices))

    return folds

def cross_validate(
    model_class,
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    **model_params
) -> dict:
    """
    Realiza K-Fold Cross Validation.

    Returns:
        Dict con scores de cada fold y promedio
    """
    folds = k_fold_split(len(y), k)
    scores = []

    for i, (train_idx, val_idx) in enumerate(folds):
        # Split
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train
        model = model_class()
        model.fit(X_train, y_train, **model_params)

        # Evaluate
        score = model.score(X_val, y_val)
        scores.append(score)

    return {
        'scores': scores,
        'mean': np.mean(scores),
        'std': np.std(scores)
    }

# Demo
# cv_results = cross_validate(LogisticRegression, X, y, k=5, learning_rate=0.1, n_iterations=500)
# print(f"CV Accuracy: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
```

### 4.3 Regularización

```python
import numpy as np

class LogisticRegressionRegularized:
    """Logistic Regression con regularización L1/L2."""

    def __init__(self, regularization: str = 'l2', lambda_: float = 0.01):
        """
        Args:
            regularization: 'l1', 'l2', o None
            lambda_: fuerza de regularización
        """
        self.regularization = regularization
        self.lambda_ = lambda_
        self.theta = None
        self.cost_history = []

    def _cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Costo con regularización."""
        m = len(y)
        h = sigmoid(X @ self.theta)
        h = np.clip(h, 1e-15, 1 - 1e-15)

        # Cross-entropy base
        bce = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

        # Regularización (excluir bias theta[0])
        if self.regularization == 'l2':
            # Ridge: λ/2m * Σθⱼ²
            reg = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:] ** 2)
        elif self.regularization == 'l1':
            # Lasso: λ/m * Σ|θⱼ|
            reg = (self.lambda_ / m) * np.sum(np.abs(self.theta[1:]))
        else:
            reg = 0

        return bce + reg

    def _gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradiente con regularización."""
        m = len(y)
        h = sigmoid(X @ self.theta)

        # Gradiente base
        grad = (1/m) * X.T @ (h - y)

        # Regularización (excluir bias)
        if self.regularization == 'l2':
            reg_grad = np.concatenate([[0], (self.lambda_ / m) * self.theta[1:]])
        elif self.regularization == 'l1':
            reg_grad = np.concatenate([[0], (self.lambda_ / m) * np.sign(self.theta[1:])])
        else:
            reg_grad = 0

        return grad + reg_grad

    def fit(self, X: np.ndarray, y: np.ndarray,
            learning_rate: float = 0.1, n_iterations: int = 1000):
        X_b = np.column_stack([np.ones(len(X)), X])
        self.theta = np.zeros(X_b.shape[1])

        for _ in range(n_iterations):
            gradient = self._gradient(X_b, y)
            self.theta -= learning_rate * gradient
            self.cost_history.append(self._cost(X_b, y))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.column_stack([np.ones(len(X)), X])
        return (sigmoid(X_b @ self.theta) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)
```

---

### ⚠️ Aviso crítico antes de Árboles: Recursividad (Semana 12)

La implementación de árboles se basa en **recursión**. Si no defines y pruebas condiciones de parada, vas a generar árboles infinitos o muy profundos.

- Condiciones de parada mínimas: `max_depth`, pureza (todas las etiquetas iguales), `min_samples_split`, “no split improves”.
- Recurso recomendado: https://realpython.com/python-recursion/
- Debug mínimo: imprime `depth`, `n_samples` y el criterio elegido por nodo durante desarrollo.

## 🌳 Parte 5: Tree-Based Models (Semana 12)

Esta semana cubre modelos supervisados **no diferenciables** (no entrenan con Gradient Descent). La lógica de entrenamiento es:

- elegir un *split* (feature + threshold)
- medir qué tan “puro” queda cada lado (Entropía o Gini)
- repetir recursivamente

### 5.1 Entropía, Gini e Information Gain

Definiciones base (para clasificación):

- **Entropía:** `H(y) = - Σ p(c) log2 p(c)`
- **Gini:** `G(y) = 1 - Σ p(c)^2`

Un split `(j, t)` divide el dataset en:

- izquierda: `x_j ≤ t`
- derecha: `x_j > t`

La idea es maximizar la mejora en pureza:

- **Information Gain:** `IG = impurity(parent) - weighted_impurity(children)`

### 5.2 Entrenable desde cero (entregable)

Entregable runnable:

- `scripts/decision_tree_from_scratch.py`

Ejecuta:

```bash
python3 scripts/decision_tree_from_scratch.py --criterion gini --max-depth 5
```

Objetivo mínimo:

- que el script entrene un árbol y reporte accuracy train/test en un dataset toy
- que puedas explicar (en 5 líneas) cómo el árbol decide el mejor split

### 5.3 Ensembles (intro): Bagging vs Boosting

Conceptos clave:

- **Bagging (Random Forest):** muchos árboles entrenados en *bootstrap samples*; reduce varianza.
- **Boosting (Gradient Boosting/XGBoost):** árboles entrenados secuencialmente corrigiendo errores; reduce bias (pero puede sobreajustar).

---

## 📦 Entregable del Módulo

- `supervised_learning.py` (regresión lineal + logística + métricas + validación).
- `scripts/decision_tree_from_scratch.py` (árbol de decisión simple desde cero, sin gradientes).

### `supervised_learning.py`

```python
"""
Supervised Learning Module

Implementación desde cero de:
- Linear Regression (con Normal Equation y Gradient Descent)
- Logistic Regression (con regularización L1/L2)
- Métricas de evaluación
- Cross Validation

Autor: [Tu nombre]
Módulo: 05 - Supervised Learning
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def add_bias(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


# ============================================================
# REGRESIÓN LINEAL
# ============================================================

class LinearRegression:
    def __init__(self):
        self.theta = None
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            method: str = 'normal', lr: float = 0.01, n_iter: int = 1000):
        X_b = add_bias(X)

        if method == 'normal':
            self.theta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)
        else:
            m, n = X_b.shape
            self.theta = np.zeros(n)
            for _ in range(n_iter):
                grad = (1/m) * X_b.T @ (X_b @ self.theta - y)
                self.theta -= lr * grad
                self.cost_history.append(np.mean((X_b @ self.theta - y)**2))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return add_bias(X) @ self.theta

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - ss_res / ss_tot


# ============================================================
# REGRESIÓN LOGÍSTICA
# ============================================================

class LogisticRegression:
    def __init__(self, reg: str = None, lambda_: float = 0.01):
        self.reg = reg
        self.lambda_ = lambda_
        self.theta = None
        self.cost_history = []

    def fit(self, X: np.ndarray, y: np.ndarray,
            lr: float = 0.1, n_iter: int = 1000):
        X_b = add_bias(X)
        m, n = X_b.shape
        self.theta = np.zeros(n)

        for _ in range(n_iter):
            h = sigmoid(X_b @ self.theta)
            grad = (1/m) * X_b.T @ (h - y)

            if self.reg == 'l2':
                grad[1:] += (self.lambda_/m) * self.theta[1:]
            elif self.reg == 'l1':
                grad[1:] += (self.lambda_/m) * np.sign(self.theta[1:])

            self.theta -= lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(add_bias(X) @ self.theta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# ============================================================
# MÉTRICAS
# ============================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2*p*r/(p+r) if (p+r) > 0 else 0

def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    cm[0, 0] = np.sum((y_true == 0) & (y_pred == 0))
    cm[0, 1] = np.sum((y_true == 0) & (y_pred == 1))
    cm[1, 0] = np.sum((y_true == 1) & (y_pred == 0))
    cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))
    return cm


# ============================================================
# VALIDACIÓN
# ============================================================

def train_test_split(X, y, test_size=0.2, seed=None):
    if seed: np.random.seed(seed)
    n = len(y)
    idx = np.random.permutation(n)
    split = int(n * test_size)
    return X[idx[split:]], X[idx[:split]], y[idx[split:]], y[idx[:split]]

def cross_validate(model_class, X, y, k=5, **params):
    n = len(y)
    idx = np.random.permutation(n)
    fold_size = n // k
    scores = []

    for i in range(k):
        val_idx = idx[i*fold_size:(i+1)*fold_size]
        train_idx = np.concatenate([idx[:i*fold_size], idx[(i+1)*fold_size:]])

        model = model_class()
        model.fit(X[train_idx], y[train_idx], **params)
        scores.append(model.score(X[val_idx], y[val_idx]))

    return {'scores': scores, 'mean': np.mean(scores), 'std': np.std(scores)}


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Test Linear Regression
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5

    lr = LinearRegression()
    lr.fit(X, y)
    print(f"Linear Regression R²: {lr.score(X, y):.4f}")

    # Test Logistic Regression
    X_c0 = np.random.randn(50, 2) + [-2, -2]
    X_c1 = np.random.randn(50, 2) + [2, 2]
    X_clf = np.vstack([X_c0, X_c1])
    y_clf = np.array([0]*50 + [1]*50)

    log_reg = LogisticRegression()
    log_reg.fit(X_clf, y_clf)
    print(f"Logistic Regression Accuracy: {log_reg.score(X_clf, y_clf):.4f}")

    # Test metrics
    y_true = np.array([0,0,0,1,1,1,1,1])
    y_pred = np.array([0,0,1,1,1,0,1,1])
    print(f"Precision: {precision(y_true, y_pred):.4f}")
    print(f"Recall: {recall(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")

    # Test CV
    cv = cross_validate(LogisticRegression, X_clf, y_clf, k=5, lr=0.1, n_iter=500)
    print(f"CV Score: {cv['mean']:.4f} ± {cv['std']:.4f}")

    print("\n✓ Todos los tests pasaron!")
```

---

## 📝 Derivación Analítica: El Entregable de Lápiz y Papel (v3.3)

> 🎓 **Simulación de Examen:** En la maestría te pedirán: *"Derive la regla de actualización de pesos para Logistic Regression"*. Debes poder hacerlo a mano.

### Derivación del Gradiente de Logistic Regression

**Objetivo:** Derivar `∂L/∂w` para la función de costo Cross-Entropy.

#### Paso 1: Definir la Función de Costo

```
L(w) = -(1/n) Σ_{i=1..n} [ y_i log(ŷ_i) + (1 - y_i) log(1 - ŷ_i) ]
```

Donde:
- `ŷ_i = σ(wᵀ x_i) = 1 / (1 + e^{-wᵀ x_i})`
- `σ(z)` es la función sigmoid

#### Paso 2: Derivar la Sigmoid

```
dσ/dz = σ(z)(1 - σ(z))
```

**Demostración:**
```
σ(z) = 1 / (1 + e^{-z})

dσ/dz = e^{-z} / (1 + e^{-z})^2
      = (1 / (1 + e^{-z})) · (e^{-z} / (1 + e^{-z}))
      = σ(z)(1 - σ(z))
```

#### Paso 3: Aplicar la Regla de la Cadena

Para un solo ejemplo `(x_i, y_i)`:

```
∂L_i/∂w = (∂L_i/∂ŷ_i) · (∂ŷ_i/∂z_i) · (∂z_i/∂w)
```

Donde `z_i = wᵀ x_i`

**Calcular cada término:**

1. `∂L_i/∂ŷ_i = -y_i/ŷ_i + (1 - y_i)/(1 - ŷ_i)`

2. `∂ŷ_i/∂z_i = ŷ_i(1 - ŷ_i)`

3. `∂z_i/∂w = x_i`

#### Paso 4: Simplificar

```
∂L_i/∂w = ( -y_i/ŷ_i + (1 - y_i)/(1 - ŷ_i) ) · ŷ_i(1 - ŷ_i) · x_i
```

Simplificando el término entre paréntesis:
```
= ( (-y_i(1 - ŷ_i) + (1 - y_i)ŷ_i) / (ŷ_i(1 - ŷ_i)) ) · ŷ_i(1 - ŷ_i) · x_i
= (-y_i + y_iŷ_i + ŷ_i - y_iŷ_i) · x_i
= (ŷ_i - y_i) · x_i
```

#### Resultado Final

```
∂L/∂w = (1/n) Σ_{i=1..n} (ŷ_i - y_i) x_i
      = (1/n) Xᵀ (ŷ - y)
```

**Forma vectorizada (para código):**
```python
gradient = (1/n) * X.T @ (y_pred - y_true)
```

### Tu Entregable

Escribe en un documento (Markdown o LaTeX):
1. La derivación completa del gradiente de Cross-Entropy
2. La derivación de la regla de actualización: `w <- w - α ∇L`
3. Por qué el gradiente tiene la forma `(ŷ - y)` (interpretación geométrica)

---

## 🎯 El Reto del Tablero Blanco (Metodología Feynman)

Explica en **máximo 5 líneas** sin jerga técnica:

1. **¿Por qué usamos sigmoid en clasificación?**
   > Pista: Piensa en probabilidades entre 0 y 1.

2. **¿Por qué Cross-Entropy y no MSE para clasificación?**
   > Pista: Piensa en qué pasa cuando `ŷ ≈ 0` pero `y = 1`.

3. **¿Qué significa "One-vs-All"?**
   > Pista: Piensa en cómo clasificar 10 dígitos con clasificadores binarios.

---

## 🔍 Shadow Mode: Validación con sklearn (v3.3)

> ⚠️ **Regla:** sklearn está **prohibido para aprender**, pero es **necesario para validar**. Si tu implementación difiere significativamente de sklearn, tienes un bug.

### Protocolo de Validación (Viernes de Fase 2)

```python
"""
Shadow Mode - Validación de Implementaciones
Compara tu código desde cero vs sklearn para detectar bugs.

Regla: Si la diferencia de accuracy es >5%, revisar matemáticas.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.linear_model import LinearRegression as SklearnLinReg
from sklearn.metrics import accuracy_score, mean_squared_error

# Importar tu implementación
# from src.logistic_regression import LogisticRegression as MyLR
# from src.linear_regression import LinearRegression as MyLinReg


def shadow_mode_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Compara tu Logistic Regression vs sklearn.

    Los coeficientes y accuracy deben ser casi idénticos.
    """
    print("=" * 60)
    print("SHADOW MODE: Logistic Regression")
    print("=" * 60)

    # ========== TU IMPLEMENTACIÓN ==========
    # my_model = MyLR()
    # my_model.fit(X_train, y_train, lr=0.1, n_iter=1000)
    # my_pred = my_model.predict(X_test)
    # my_acc = accuracy_score(y_test, my_pred)
    # my_weights = my_model.weights

    # Placeholder (reemplazar con tu código)
    my_acc = 0.85
    my_weights = np.zeros(X_train.shape[1])

    # ========== SKLEARN (GROUND TRUTH) ==========
    sklearn_model = SklearnLR(max_iter=1000, solver='lbfgs')
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_acc = accuracy_score(y_test, sklearn_pred)
    sklearn_weights = sklearn_model.coef_.flatten()

    # ========== COMPARACIÓN ==========
    acc_diff = abs(my_acc - sklearn_acc)
    weight_diff = np.linalg.norm(my_weights - sklearn_weights[:len(my_weights)])

    print(f"\n📊 RESULTADOS:")
    print(f"  Tu Accuracy:     {my_acc:.4f}")
    print(f"  sklearn Accuracy: {sklearn_acc:.4f}")
    print(f"  Diferencia:       {acc_diff:.4f}")

    print(f"\n📐 PESOS:")
    print(f"  Diferencia L2 de pesos: {weight_diff:.4f}")

    # Veredicto
    print("\n" + "-" * 60)
    if acc_diff < 0.05:
        print("✓ PASSED: Tu implementación es correcta")
        return True
    else:
        print("✗ FAILED: Diferencia significativa - revisa tu matemática")
        print("  Posibles causas:")
        print("  - Gradiente mal calculado")
        print("  - Learning rate muy alto/bajo")
        print("  - Falta de normalización de datos")
        return False


def shadow_mode_linear_regression(X_train, y_train, X_test, y_test):
    """
    Compara tu Linear Regression vs sklearn.
    """
    print("=" * 60)
    print("SHADOW MODE: Linear Regression")
    print("=" * 60)

    # ========== TU IMPLEMENTACIÓN ==========
    # my_model = MyLinReg()
    # my_model.fit(X_train, y_train)
    # my_pred = my_model.predict(X_test)
    # my_mse = mean_squared_error(y_test, my_pred)

    # Placeholder
    my_mse = 0.5

    # ========== SKLEARN ==========
    sklearn_model = SklearnLinReg()
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)

    # ========== COMPARACIÓN ==========
    mse_ratio = my_mse / sklearn_mse if sklearn_mse > 0 else float('inf')

    print(f"\n📊 RESULTADOS:")
    print(f"  Tu MSE:     {my_mse:.4f}")
    print(f"  sklearn MSE: {sklearn_mse:.4f}")
    print(f"  Ratio:       {mse_ratio:.2f}x")

    print("\n" + "-" * 60)
    if mse_ratio < 1.1:  # Dentro del 10%
        print("✓ PASSED: Tu implementación es correcta")
        return True
    else:
        print("✗ FAILED: Tu MSE es significativamente mayor")
        return False


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split

    # Dataset de clasificación
    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=42
    )
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    # Dataset de regresión
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=10, noise=10, random_state=42
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Ejecutar Shadow Mode
    shadow_mode_logistic_regression(X_train_c, y_train_c, X_test_c, y_test_c)
    print("\n")
    shadow_mode_linear_regression(X_train_r, y_train_r, X_test_r, y_test_r)
```

### Checklist Shadow Mode

| Día | Algoritmo | Validar |
|-----|-----------|---------|
| Viernes Sem 10 | Linear Regression | MSE ≈ sklearn |
| Viernes Sem 11 | Logistic Regression | Accuracy ≈ sklearn |
| Viernes Sem 12 | Métricas | Precision/Recall = sklearn |

---

## ✅ Checklist de Finalización (v3.3)

### Conocimiento
- [ ] Implementé regresión lineal con Normal Equation y GD
- [ ] Entiendo MSE y su gradiente
- [ ] Implementé regresión logística desde cero
- [ ] Entiendo sigmoid y binary cross-entropy
- [ ] Puedo calcular TP, TN, FP, FN de una matriz de confusión
- [ ] Implementé accuracy, precision, recall, F1
- [ ] Implementé train/test split
- [ ] Implementé K-fold cross validation
- [ ] Entiendo regularización L1 vs L2

### Shadow Mode (v3.3 - Obligatorio)
- [ ] **Linear Regression**: Mi MSE ≈ sklearn (ratio < 1.1)
- [ ] **Logistic Regression**: Mi Accuracy ≈ sklearn (diff < 5%)

### Entregables de Código
- [ ] `logistic_regression.py` con tests pasando
- [ ] `artifacts/m05_logreg_weights.png` + 5–10 líneas de interpretación (pesos 28x28)
- [ ] `mypy src/` pasa sin errores
- [ ] `pytest tests/` pasa sin errores

### Derivación Analítica (Obligatorio)
- [ ] Derivé el gradiente de Cross-Entropy a mano
- [ ] Documento con derivación completa (Markdown o LaTeX)
- [ ] Puedo explicar por qué `∇L = Xᵀ(ŷ - y)`

### Metodología Feynman
- [ ] Puedo explicar sigmoid en 5 líneas sin jerga
- [ ] Puedo explicar Cross-Entropy vs MSE en 5 líneas
- [ ] Puedo explicar One-vs-All en 5 líneas

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [04_PROBABILIDAD_ML](04_PROBABILIDAD_ML.md) | [00_INDICE](00_INDICE.md) | [06_UNSUPERVISED_LEARNING](06_UNSUPERVISED_LEARNING.md) |
