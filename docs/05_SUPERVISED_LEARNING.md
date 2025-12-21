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

### 🧪 Ver para Entender (Laboratorios Interactivos)

- Guía central: [INTERACTIVE_LABS.md](INTERACTIVE_LABS.md)
- Regresión lineal (táctil):
  - `streamlit run interactive_labs/m05_supervised/visualizacion_regresion.py`
- Overfitting / Bias–Variance (grado 1→20):
  - `streamlit run interactive_labs/m05_supervised/overfitting_bias_variance_app.py`

### Cápsula (obligatoria): Vectorización extrema (prohibido usar loops)

Regla práctica para todo el módulo:

- **Prohibido** iterar con `for` sobre muestras (`N`) o features (`D`) para computar predicciones, pérdidas o gradientes.
- **Permitido** iterar sobre iteraciones de entrenamiento (`for step in range(...)`) o épocas.

Objetivo: que el *core* de ML sea una composición de operaciones tipo:

- `logits = X @ W`
- `grad = X.T @ something`

Ejemplos canónicos (con **disciplina de shapes** y sin loops):

```python
import numpy as np  # NumPy: álgebra lineal y vectorización


# ============================================================
# 1) Forward multiclase: logits = X @ W
# ============================================================
N = 5  # N: número de muestras
D = 4  # D: número de features
K = 3  # K: número de clases

X = np.random.randn(N, D).astype(float)  # X:(N,D) batch de entrada
assert X.shape == (N, D)  # Contrato de shape para X

W = np.random.randn(D, K).astype(float)  # W:(D,K) pesos por clase
assert W.shape == (D, K)  # Contrato de shape para W

logits = X @ W  # logits:(N,K) porque (N,D)@(D,K)=(N,K)
assert logits.shape == (N, K)  # Contrato: logits debe ser 2D (batch x clases)


# ============================================================
# 2) Logística binaria: gradiente vectorizado ∇w = (1/N) X^T(ŷ - y)
# ============================================================
w = np.random.randn(D).astype(float)  # w:(D,) pesos binarios (una clase)
assert w.shape == (D,)  # Contrato de shape para w

y = (np.random.rand(N) > 0.5).astype(float)  # y:(N,) etiquetas binarias en {0,1}
assert y.shape == (N,)  # Contrato de shape para y

z = X @ w  # z:(N,) logits binarios
assert z.shape == (N,)  # Contrato de shape para z

y_hat = 1.0 / (1.0 + np.exp(-z))  # sigmoid(z) vectorizada (sin loops)
assert y_hat.shape == (N,)  # Contrato de shape para ŷ

grad_w = (X.T @ (y_hat - y)) / N  # (D,N)@(N,)=(D,) (forma de examen)
assert grad_w.shape == (D,)  # Contrato: gradiente debe tener el shape de w
# ============================================================
# 3) Distancias pairwise sin loops (kNN / clustering):
#    dist2[i,j] = ||X_query[i] - X_train[j]||^2
# ============================================================
M = 6  # M: número de queries
X_train = np.random.randn(N, D).astype(float)  # X_train:(N,D)
X_query = np.random.randn(M, D).astype(float)  # X_query:(M,D)
assert X_train.shape == (N, D)  # Shape correcto para broadcasting
assert X_query.shape == (M, D)  # Shape correcto para broadcasting

# Trick algebraico: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
q_norm2 = np.sum(X_query ** 2, axis=1, keepdims=True)  # (M,1) ||q_i||^2
t_norm2 = np.sum(X_train ** 2, axis=1, keepdims=True).T  # (1,N) ||t_j||^2
cross = X_query @ X_train.T  # (M,N) producto punto entre cada par (q_i, t_j)

dist2 = q_norm2 + t_norm2 - 2.0 * cross  # (M,N) distancias cuadradas
dist2 = np.maximum(dist2, 0.0)  # Evita negativos por error numérico (float)
assert dist2.shape == (M, N)  # Shape correcto de matriz de distancias
```

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
import numpy as np  # Importa NumPy para arrays y operaciones numéricas

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

def add_bias_term(X: np.ndarray) -> np.ndarray:  # añade término de bias (columna de 1s)
    """Añade columna de 1s para el término de bias."""
    m = X.shape[0]  # número de muestras (filas) en X
    return np.column_stack([np.ones(m), X])  # concatena columna de 1s (bias) con X

def predict_linear(X: np.ndarray, theta: np.ndarray) -> np.ndarray:  # predicción lineal: y_hat = Xθ
    """Predicción lineal: ŷ = Xθ"""
    return X @ theta  # producto matricial para obtener las predicciones
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 1.1: Modelo (Regresión Lineal)</strong></summary>

#### 1) Metadatos
- **Título:** De la hipótesis `ŷ=Xθ` a un contrato de shapes (y por qué el bias es “feature 0”)
- **ID (opcional):** `M05-T01_1`
- **Duración estimada:** 60–120 min
- **Nivel:** Básico–Intermedio
- **Dependencias:** Álgebra lineal mínima (producto matriz-vector), noción de dataset tabular

#### 2) Objetivos
- Escribir la hipótesis en forma escalar y matricial y explicar qué representa cada símbolo.
- Usar una convención de shapes sin ambigüedad: `X:(m,n)` y `θ:(n+1,)` tras agregar bias.
- Verificar rápidamente si una implementación está “bien cableada” (shape checks).

#### 3) Relevancia
- Todo el resto del módulo (logística, métricas, regularización) depende de tener claro el *forward* `X @ θ`.
- La mayoría de bugs “misteriosos” en ML-from-scratch son bugs de shapes, no de matemáticas.

#### 4) Mapa conceptual mínimo
- **Datos** `X` (features) + **parámetros** `θ` → **predicción** `ŷ`.
- **Bias** → se implementa como `x₀=1` y `θ₀`.

#### 5) Definiciones esenciales
- `m`: número de muestras.
- `n`: número de features (sin bias).
- `θ₀`: intercepto/bias.

#### 6) Explicación didáctica
- Trátalo como “contrato”: si `add_bias_term(X)` devuelve `(m,n+1)`, entonces `θ` debe tener longitud `n+1`.

#### 7) Ejemplo modelado
- Dataset 1D (`n=1`): `X:(m,1)` → con bias `X_b:(m,2)` y `θ:(2,)`.

#### 8) Práctica guiada
- Escribe 3 asserts: shapes de `X_b`, `θ`, `X_b @ θ`.

#### 9) Práctica independiente
- Convierte un dataset con 3 features a `X_b` y verifica que el forward funciona sin loops.

#### 10) Autoevaluación
- ¿Por qué `x₀=1` hace que el intercepto sea un peso más?

#### 11) Errores comunes
- Duplicar bias (agregar columna de 1s dos veces).
- Usar `θ` como columna `(n+1,1)` y luego mezclar con `(n+1,)` sin querer.

#### 12) Retención
- Mantra: `ŷ = X_b @ θ` y el bias es `x₀=1`.

#### 13) Diferenciación
- Avanzado: generaliza a multiclase `logits = X @ W`.

#### 14) Recursos
- Cheatsheet de shapes y producto matricial.

#### 15) Nota docente
- Pide que el alumno “debuggee en voz alta” un error de shape típico (ej. `(m,n)@(n+1,)`).
</details>

### 1.2 Función de Costo (MSE)

```python
import numpy as np  # Importa NumPy para arrays y operaciones numéricas

def mse_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:  # calcula el costo MSE para θ
    """
    Mean Squared Error Cost Function.

    J(θ) = (1/2m) Σᵢ (h(xᵢ) - yᵢ)²
         = (1/2m) ||Xθ - y||²

    El factor 1/2 es por conveniencia (cancela con la derivada).
    """
    m = len(y)  # número de muestras
    predictions = X @ theta  # predicciones del modelo (ŷ)
    errors = predictions - y  # residuos (ŷ - y)
    return (1 / (2 * m)) * np.sum(errors ** 2)  # costo MSE con factor 1/2m

def mse_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:  # gradiente del MSE respecto a θ
    """
    Gradiente del MSE respecto a θ.

    ∂J/∂θ = (1/m) Xᵀ(Xθ - y)
    """
    m = len(y)  # número de muestras
    predictions = X @ theta  # predicciones del modelo (ŷ)
    errors = predictions - y  # residuos (ŷ - y)
    return (1 / m) * X.T @ errors  # gradiente vectorizado: (1/m) Xᵀ(ŷ-y)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 1.2: Función de Costo (MSE)</strong></summary>

#### 1) Metadatos
- **Título:** MSE como “penalización cuadrática” y como `||Xθ-y||²` (con lectura geométrica)
- **ID (opcional):** `M05-T01_2`
- **Duración estimada:** 60–120 min
- **Nivel:** Básico–Intermedio
- **Dependencias:** 1.1, suma de cuadrados, producto `X.T @ v`

#### 2) Objetivos
- Explicar el MSE en lenguaje natural (errores grandes se penalizan más).
- Reconocer la forma vectorizada del gradiente `∇θ = (1/m) Xᵀ(Xθ - y)`.
- Entender por qué aparece `Xᵀ` (proyección del error hacia parámetros).

#### 3) Relevancia
- Este patrón de gradiente `Xᵀ(ŷ-y)` reaparece en logística (BCE) y en softmax (CCE).

#### 4) Mapa conceptual mínimo
- **Predicción** `ŷ` → **residuo** `(ŷ-y)` → **gradiente** `Xᵀ(residuo)`.

#### 5) Definiciones esenciales
- **Residuo**: `r = ŷ - y`.
- **Costo**: promedio (o suma) de `r²`.

#### 6) Explicación didáctica
- El factor `1/2` en el costo suele usarse para simplificar derivadas; el mínimo no cambia.

#### 7) Ejemplo modelado
- Si duplicas un error (de 2 a 4), la contribución al costo se cuadruplica (4→16).

#### 8) Práctica guiada
- Implementa un test: si `theta` es perfecto (`X@theta==y`), entonces `mse_cost==0` y `mse_gradient==0`.

#### 9) Práctica independiente
- Compara `mse_cost` con `np.mean((X@theta - y)**2)` y explica la diferencia del `1/2`.

#### 10) Autoevaluación
- ¿Qué significa que el gradiente apunte hacia donde el costo sube más rápido?

#### 11) Errores comunes
- Confundir shapes: `y` como `(m,1)` vs `(m,)`.
- Olvidar el promedio por `m` (magnitud del gradiente depende del batch size).

#### 12) Retención
- Fórmula clave: `∇θ MSE = (1/m) Xᵀ(Xθ-y)`.

#### 13) Diferenciación
- Avanzado: conecta con mínimos cuadrados y proyecciones (subespacios).

#### 14) Recursos
- Notas de least squares, interpretación geométrica.

#### 15) Nota docente
- Pide que el alumno derive la forma vectorizada desde la forma sumatoria (una vez, con calma).
</details>

### 1.3 Solución Cerrada (Normal Equation)

```python
import numpy as np  # Importa NumPy para arrays y operaciones numéricas

def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:  # calcula θ por ecuación normal
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
    XtX = X.T @ X  # calcula XᵀX
    Xty = X.T @ y  # calcula Xᵀy

    # Usar solve en lugar de inv para estabilidad numérica
    theta = np.linalg.solve(XtX, Xty)  # resuelve el sistema lineal XtX·θ = Xty
    return theta  # devuelve el vector de parámetros θ
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 1.3: Solución Cerrada (Normal Equation)</strong></summary>

#### 1) Metadatos
- **Título:** Normal Equation: cuándo sirve, cuándo falla y por qué `solve` es mejor que `inv`
- **ID (opcional):** `M05-T01_3`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 1.1–1.2, noción de matriz singular/condicionamiento

#### 2) Objetivos
- Implementar `θ = argmin ||Xθ-y||²` vía ecuaciones normales.
- Explicar por qué `XᵀX` puede ser singular o mal condicionada.
- Preferir `np.linalg.solve` sobre `inv` por estabilidad.

#### 3) Relevancia
- Te da un “baseline” para validar GD: si ambos dan resultados parecidos (cuando aplica), tu GD está bien.

#### 4) Mapa conceptual mínimo
- Minimizar SSE → derivada = 0 → `XᵀXθ = Xᵀy`.

#### 5) Definiciones esenciales
- **Singular**: no invertible.
- **Condicionamiento**: sensibilidad numérica a perturbaciones.

#### 6) Explicación didáctica
- En alta dimensión o con colinealidad fuerte, `XᵀX` puede “romperse” numéricamente.

#### 7) Ejemplo modelado
- Si una feature es combinación lineal de otra (duplicada), `XᵀX` tiende a singular.

#### 8) Práctica guiada
- Crea una feature duplicada en `X` y observa qué ocurre con `np.linalg.solve`.

#### 9) Práctica independiente
- Implementa Ridge cerrada: `θ=(XᵀX+λI)^{-1}Xᵀy` (solo conceptual aquí).

#### 10) Autoevaluación
- ¿Por qué la complejidad crece como `O(n³)`?

#### 11) Errores comunes
- Usar `inv` por costumbre.
- Olvidar agregar bias antes de la ecuación normal.

#### 12) Retención
- Regla: si puedes usar closed-form, úsala para validar GD (no necesariamente para producción).

#### 13) Diferenciación
- Avanzado: `np.linalg.lstsq` y pseudo-inversa (SVD) como alternativa estable.

#### 14) Recursos
- Documentación NumPy: `solve`, `lstsq`, conceptos de singularidad.

#### 15) Nota docente
- Pedir un “diagnóstico” cuando falla: ¿singularidad real o numérica?
</details>

### 1.4 Gradient Descent para Regresión

```python
import numpy as np  # Importa NumPy para operaciones matemáticas
from typing import List, Tuple  # Importa tipos para anotaciones

class LinearRegression:  # modelo de regresión lineal con GD o ecuación normal
    """Regresión Lineal implementada desde cero."""

    def __init__(self):  # inicializa parámetros e historial
        self.theta = None  # Parámetros del modelo (pesos + bias)
        self.cost_history = []  # Historial de costos para monitoreo

    def fit(  # entrena el modelo usando GD o ecuación normal
        self,  # instancia del modelo
        X: np.ndarray,  # features (m, n)
        y: np.ndarray,  # targets (m,)
        method: str = 'gradient_descent',  # método de entrenamiento
        learning_rate: float = 0.01,  # tasa de aprendizaje (solo GD)
        n_iterations: int = 1000  # número de iteraciones (solo GD)
    ) -> 'LinearRegression':  # retorna self para estilo sklearn
        """
        Entrena el modelo.

        Args:
            X: features (m, n)
            y: targets (m,)
            method: 'gradient_descent' o 'normal_equation'
            learning_rate: tasa de aprendizaje (solo para GD)
            n_iterations: número de iteraciones (solo para GD)
        """
        # Añadir bias a las features
        X_b = add_bias_term(X)  # construye X con término de bias
        m, n = X_b.shape  # m: muestras, n: features + bias

        if method == 'normal_equation':  # entrena con solución cerrada
            self.theta = normal_equation(X_b, y)  # Solución analítica directa
        else:  # entrena con descenso de gradiente
            # Inicializar theta con ceros o valores pequeños
            self.theta = np.zeros(n)  # inicializa parámetros

            for i in range(n_iterations):  # iteraciones de GD
                # Calcular gradiente del MSE
                gradient = mse_gradient(X_b, y, self.theta)  # gradiente del coste MSE

                # Actualizar theta usando gradient descent
                self.theta = self.theta - learning_rate * gradient  # update: θ ← θ − α∇θ

                # Guardar costo para monitoreo de convergencia
                cost = mse_cost(X_b, y, self.theta)  # calcula coste actual
                self.cost_history.append(cost)  # guarda histórico de coste

        return self  # permite chaining

    def predict(self, X: np.ndarray) -> np.ndarray:  # predice y_hat para nuevas muestras
        """Predice valores."""
        X_b = add_bias_term(X)  # Añade bias para predicción
        return X_b @ self.theta  # Predicción lineal: y = X·θ

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # calcula R² como métrica de ajuste
        """R² score."""
        y_pred = self.predict(X)  # Predicciones del modelo
        ss_res = np.sum((y - y_pred) ** 2)  # Suma de residuos al cuadrado
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Suma total de cuadrados
        return 1 - (ss_res / ss_tot)  # R² = 1 - (residuos/total)


# Demo de regresión lineal
np.random.seed(42)  # Fija semilla para reproducibilidad
X = 2 * np.random.rand(100, 1)  # 100 puntos entre 0 y 2
y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5  # y = 4 + 3x + ruido gaussiano

model = LinearRegression()  # Crea instancia del modelo
model.fit(X, y, method='gradient_descent', learning_rate=0.1, n_iterations=1000)  # Entrena

print(f"Parámetros aprendidos: {model.theta}")  # Muestra θ aprendido
print(f"Esperados: [4, 3]")  # Valores teóricos (bias=4, pendiente=3)
print(f"R² score: {model.score(X, y):.4f}")  # Calidad del ajuste
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 1.4: Gradient Descent para Regresión</strong></summary>

#### 1) Metadatos
- **Título:** Gradient Descent “de examen”: convergence, learning rate, y checks de sanidad
- **ID (opcional):** `M05-T01_4`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** 1.1–1.3, gradiente MSE, noción de iteración/épocas

#### 2) Objetivos
- Entrenar regresión lineal por GD con un `learning_rate` razonable.
- Leer el `cost_history` y detectar divergencia o estancamiento.
- Entender por qué la vectorización es obligatoria (performance + claridad).

#### 3) Relevancia
- GD es la base del entrenamiento de modelos más grandes (logística, MLP). Aquí practicas el ciclo “forward → loss → grad → update”.

#### 4) Mapa conceptual mínimo
- Inicializar `θ` → repetir: `grad = Xᵀ(ŷ-y)/m` → `θ ← θ - α grad`.

#### 5) Definiciones esenciales
- **Learning rate (α)**: tamaño del paso.
- **Divergencia**: el costo sube o se vuelve NaN/inf.
- **Convergencia**: el costo baja y se estabiliza.

#### 6) Explicación didáctica
- Si `α` es muy grande: saltas el mínimo y explota.
- Si `α` es muy pequeño: entrenas “para siempre”.

#### 7) Ejemplo modelado
- En el demo, la solución esperada es ~`[4,3]` (con ruido). Si sale lejísimos, revisa shapes, bias y `α`.

#### 8) Práctica guiada
- Imprime cada 100 iteraciones: costo actual. Debe decrecer (aprox).

#### 9) Práctica independiente
- Implementa early stopping: si la mejora del costo < `tol` por varias iteraciones, detén.

#### 10) Autoevaluación
- ¿Qué pasa si omites el bias? ¿Cómo cambia la recta aprendida?

#### 11) Errores comunes
- No normalizar features → GD lento o inestable.
- Mezclar `X` con `X_b` en gradiente/predicción.
- Reportar R² en train y creer que generaliza (falta split).

#### 12) Retención
- Checklist: bias, shapes, costo decrece, no NaNs, params razonables.

#### 13) Diferenciación
- Avanzado: batch vs mini-batch vs SGD (conceptual) y efecto en el ruido del gradiente.

#### 14) Recursos
- Notas de optimización básica, escalado de features.

#### 15) Nota docente
- Pide un “protocolo de debugging”: 1) overfit test en dataset pequeño, 2) comparar con normal equation.
</details>

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
import numpy as np  # arrays, álgebra lineal y generación de números aleatorios
import matplotlib.pyplot as plt  # visualización de datos y frontera de decisión


def make_blobs_2d(n=200, seed=42):  # genera dos clusters 2D para clasificación binaria
    rng = np.random.default_rng(seed)  # inicializa RNG reproducible
    c0 = rng.normal(loc=(-2.0, -1.5), scale=0.8, size=(n // 2, 2))  # clase 0: gaussiana 2D
    c1 = rng.normal(loc=(2.0, 1.5), scale=0.8, size=(n // 2, 2))  # clase 1: gaussiana 2D
    X = np.vstack([c0, c1])  # concatena muestras de ambas clases (features)
    y = np.array([0] * (n // 2) + [1] * (n // 2))  # vector de etiquetas 0/1
    return X, y  # retorna dataset sintético


def sigmoid(z):  # sigmoide para mapear logits a (0,1)
    z = np.clip(z, -500, 500)  # evita overflow numérico en exp para |z| grande
    return 1 / (1 + np.exp(-z))  # aplica la función sigmoide elemento a elemento


def add_bias(X):  # añade columna de 1s para intercepto
    return np.column_stack([np.ones(len(X)), X])  # añade columna de 1s para el término bias


def plot_decision_boundary(model, X, y, title="Decision boundary"):  # grafica probabilidad y frontera p=0.5 en 2D
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0  # rango en x con margen
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0  # rango en y con margen

    xx, yy = np.meshgrid(  # crea una malla regular para evaluar el modelo
        np.linspace(x_min, x_max, 250),  # muestrea coordenadas x
        np.linspace(y_min, y_max, 250),  # muestrea coordenadas y
    )  # retorna matrices (xx, yy) con coordenadas del grid

    grid = np.column_stack([xx.ravel(), yy.ravel()])  # aplana la malla a lista de puntos (N, 2)
    proba = model.predict_proba(grid).reshape(xx.shape)  # evalúa p(y=1|x) y re-forma a la malla

    plt.figure(figsize=(7, 6))  # crea figura
    plt.contourf(xx, yy, proba, levels=20, cmap="RdBu", alpha=0.35)  # mapa de calor de probabilidad
    plt.contour(xx, yy, proba, levels=[0.5], colors="black", linewidths=2)  # línea de decisión p=0.5

    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=18, label="Clase 0")  # puntos de la clase 0
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=18, label="Clase 1")  # puntos de la clase 1

    plt.title(title)  # título del gráfico
    plt.legend()  # leyenda de clases
    plt.grid(True, alpha=0.2)  # rejilla suave para lectura
    plt.show()  # renderiza la figura


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

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 2.0: Regresión Logística (marco mental completo)</strong></summary>

#### 1) Metadatos
- **Título:** Qué estás construyendo realmente: probabilidades, decisión, loss y gradiente (una sola historia)
- **ID (opcional):** `M05-T02_0`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio (core)
- **Dependencias:** M03 (chain rule), M04 (MLE → cross-entropy)

#### 2) Objetivos
- Unificar en una frase el pipeline: `z = Xθ` → `p = σ(z)` → `loss = BCE(p,y)` → `grad = Xᵀ(p-y)/m` → update.
- Explicar por qué logística es “lineal” en la frontera, aunque `σ` sea no lineal.
- Saber qué debes observar cuando algo falla (NaNs, saturación, signos, shapes).

#### 3) Relevancia
- Esta sección es el puente directo a MLP/softmax (M07): cambia `σ` por softmax y `BCE` por CCE, pero el esqueleto es el mismo.

#### 4) Mapa conceptual mínimo
- **Modelo:** `p(y=1|x) = σ(θᵀx)`.
- **Decisión:** `p ≥ threshold`.
- **Entrenamiento (MLE):** minimizar NLL = BCE.
- **Gradiente vectorizado:** siempre termina en `Xᵀ(something)`.

#### 5) Definiciones esenciales
- **Logit:** `z = θᵀx` (score sin acotar).
- **Probabilidad:** `p = σ(z)`.
- **Loss BCE:** castiga fuerte “seguro y equivocado”.

#### 6) Explicación didáctica
- Lo más importante no es memorizar fórmulas, sino saber qué variable inspeccionar:
  - si `p` es 0/1 exacto → `log(0)` rompe → `eps`.
  - si `|z|` es enorme → saturación → gradiente pequeño.

#### 7) Ejemplo modelado
- Si tu modelo predice `p=0.01` cuando `y=1`, BCE es grande; eso fuerza una corrección fuerte del gradiente.

#### 8) Práctica guiada
- Haz un “overfit test” con 20 ejemplos y confirma que BCE cae y accuracy sube.

#### 9) Práctica independiente
- Cambia `threshold` y observa el tradeoff precision/recall (lo conectarás con métricas en Parte 3).

#### 10) Autoevaluación
- ¿Cuál es la única pieza que convierte un score lineal en probabilidad? (respuesta: `σ`).

#### 11) Errores comunes
- Entrenar con `y∈{-1,1}` usando BCE estándar.
- Olvidar bias.
- Mezclar `X` con `X_b` (con bias) en distintas funciones.

#### 12) Retención
- Recita el mantra: `z→σ(z)→BCE→Xᵀ(p-y)`.

#### 13) Diferenciación
- Avanzado: interpreta `θ` como dirección normal al hiperplano; magnitud controla “confianza”.

#### 14) Recursos
- M04 (MLE/cross-entropy) y glosario de sigmoid/logistic regression.

#### 15) Nota docente
- Pide al alumno un diagrama de flujo con shapes: `X:(m,n)`, `θ:(n,)`, `z:(m,)`, `p:(m,)`, `grad:(n,)`.
</details>

### 2.1 Función Sigmoid

```python
import numpy as np  # Importa NumPy para operaciones matemáticas

def sigmoid(z: np.ndarray) -> np.ndarray:  # aplica la función logística elemento a elemento
    """
    Función sigmoid/logística.

    σ(z) = 1 / (1 + e^(-z))

    Propiedades:
    - Rango: (0, 1) - perfecto para probabilidades
    - σ(0) = 0.5
    - σ'(z) = σ(z)(1 - σ(z))
    """
    # Clip para evitar overflow en exp() con valores extremos
    z = np.clip(z, -500, 500)  # limita z para evitar overflow/underflow en exp()
    return 1 / (1 + np.exp(-z))  # Fórmula matemática de la sigmoide

# Visualizar la función sigmoid
import matplotlib.pyplot as plt  # Importa matplotlib para gráficos

z = np.linspace(-10, 10, 100)  # Valores de prueba de -10 a 10
plt.figure(figsize=(8, 4))  # Crea figura de 8x4 pulgadas
plt.plot(z, sigmoid(z))  # Grafica sigmoid(z)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)  # Línea horizontal en y=0.5
plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)  # Línea vertical en x=0
plt.xlabel('z')  # Etiqueta eje x
plt.ylabel('σ(z)')  # Etiqueta eje y
plt.title('Función Sigmoid')  # Título del gráfico
plt.grid(True)  # Activa cuadrícula
# plt.show()  # Descomentar para mostrar gráfico
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 2.1: Sigmoid (intuición + estabilidad numérica)</strong></summary>

#### 1) Metadatos
- **Título:** Sigmoid como “puerta” a probabilidades y por qué hay que hacer `clip`
- **ID (opcional):** `M05-T02_1`
- **Duración estimada:** 45–90 min
- **Nivel:** Básico–Intermedio
- **Dependencias:** Exponencial/log, overflow/underflow

#### 2) Objetivos
- Entender que `σ(z)` solo reescala el score a `(0,1)`; no hace la frontera no lineal.
- Reconocer saturación: `z>>0 → σ≈1` y `z<<0 → σ≈0`.
- Justificar `clip(z)` como protección numérica.

#### 3) Relevancia
- Si no controlas overflow/saturación, tu BCE se vuelve NaN y el entrenamiento colapsa.

#### 4) Mapa conceptual mínimo
- `z` crece → `exp(-z)` puede underflow; `z` muy negativo → `exp(-z)` overflow.

#### 5) Definiciones esenciales
- `σ(0)=0.5`.
- `σ'(z)=σ(z)(1-σ(z))` (máxima en 0, mínima en extremos).

#### 6) Explicación didáctica
- Cuando `σ` se satura, el gradiente se vuelve pequeño: puede “aprender lento” aunque el error sea real.

#### 7) Ejemplo modelado
- Prueba `z=[-1000,0,1000]` y observa que sin `clip` puedes romper `exp`.

#### 8) Práctica guiada
- Escribe un test: `sigmoid(np.array([0.0]))==0.5` (aprox).

#### 9) Práctica independiente
- Implementa una sigmoid estable alternativa (log-sum-exp) y compara.

#### 10) Autoevaluación
- ¿Qué pasa con `σ'(z)` cuando `z` es muy grande en valor absoluto?

#### 11) Errores comunes
- Creer que sigmoid “hace no lineal” la frontera.

#### 12) Retención
- “Sigmoid curva la probabilidad, no la geometría del plano”.

#### 13) Diferenciación
- Avanzado: relación entre sigmoid y logit.

#### 14) Recursos
- Material de estabilidad numérica (overflow/underflow).

#### 15) Nota docente
- Pedir al alumno que explique por qué `clip` es un *guardrail* y no un “hack”.
</details>

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

def predict_proba(X: np.ndarray, theta: np.ndarray) -> np.ndarray:  # predice probabilidad P(y=1|x) usando σ(Xθ)
    """Predice probabilidad de clase 1."""
    return sigmoid(X @ theta)  # calcula p=σ(Xθ)

def predict_class(X: np.ndarray, theta: np.ndarray, threshold: float = 0.5) -> np.ndarray:  # predice clase usando umbral
    """Predice clase (0 o 1)."""
    return (predict_proba(X, theta) >= threshold).astype(int)  # convierte probas a clases 0/1
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 2.2: Hipótesis + umbral (qué significa predecir)</strong></summary>

#### 1) Metadatos
- **Título:** Probabilidad vs clase: `predict_proba` y `predict_class` no son lo mismo
- **ID (opcional):** `M05-T02_2`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 2.1

#### 2) Objetivos
- Separar claramente: score `z`, probabilidad `p`, decisión `ŷ`.
- Entender el papel del `threshold` como decisión de negocio (no matemática fija).

#### 3) Relevancia
- Cambiar `threshold` es una de las maneras más simples y potentes de controlar FP vs FN (verás esto en métricas).

#### 4) Mapa conceptual mínimo
- `predict_proba` te da un ranking de “confianza”.
- `predict_class` es una política: “si p≥t, digo 1”.

#### 5) Definiciones esenciales
- **Frontera:** `θᵀx=0` si `t=0.5`.

#### 6) Explicación didáctica
- `t=0.5` es convencional; si el costo de FN es alto, baja el umbral.

#### 7) Ejemplo modelado
- En spam: prefieres recall alto → `threshold` más bajo (aceptas más FP).

#### 8) Práctica guiada
- Evalúa el mismo modelo con `t=0.3,0.5,0.7` y registra cambios de precision/recall.

#### 9) Práctica independiente
- Encuentra un `threshold` que maximice F1 en un dataset de validación.

#### 10) Autoevaluación
- ¿Por qué dos modelos con igual accuracy pueden ser muy distintos cuando cambias `threshold`?

#### 11) Errores comunes
- Calcular métricas usando probabilidades como si fueran clases.

#### 12) Retención
- “Primero calibro y evalúo probabilidades; luego decido clases con un umbral”.

#### 13) Diferenciación
- Avanzado: curva ROC/PR (conceptual) como barrido de thresholds.

#### 14) Recursos
- Glosario: precision/recall y confusion matrix.

#### 15) Nota docente
- Pide que el alumno explique verbalmente qué significa: “predigo 1 si p≥0.3”.
</details>

### 2.3 Binary Cross-Entropy Loss

```python
import numpy as np  # arrays y operaciones numéricas

def binary_cross_entropy(  # BCE (log-loss) para regresión logística
    X: np.ndarray,  # matriz de features (m, d)
    y: np.ndarray,  # etiquetas binarias (m,)
    theta: np.ndarray,  # parámetros (d,)
    eps: float = 1e-15  # epsilon para estabilidad numérica
) -> float:  # retorna el costo promedio
    """
    Binary Cross-Entropy (Log Loss).

    J(θ) = -(1/m) Σᵢ [yᵢ log(hᵢ) + (1-yᵢ) log(1-hᵢ)]

    Donde hᵢ = σ(θᵀxᵢ)

    Por qué esta función de costo:
    - Es convexa (tiene un único mínimo global)
    - Penaliza mucho las predicciones muy incorrectas
    - Es la derivación de Maximum Likelihood Estimation
    """
    m = len(y)  # número de muestras
    h = sigmoid(X @ theta)  # probabilidades predichas p=σ(Xθ)

    # Clip para evitar log(0)
    h = np.clip(h, eps, 1 - eps)  # limita p para evitar log(0) y log(1)

    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))  # BCE promedio
    return cost  # retorna costo escalar

def bce_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:  # gradiente de BCE w.r.t. theta
    """
    Gradiente de Binary Cross-Entropy.

    ∂J/∂θ = (1/m) Xᵀ(h - y)

    ¡Tiene la misma forma que el gradiente del MSE!
    Esto es porque derivamos σ(z) y la derivada σ'(z) = σ(z)(1-σ(z))
    cancela parte de la expresión.
    """
    m = len(y)  # número de muestras
    h = sigmoid(X @ theta)  # probabilidades predichas p=σ(Xθ)
    return (1/m) * X.T @ (h - y)  # ∇θ = (1/m) Xᵀ(p - y)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 2.3: BCE + gradiente (lo que debes saber de memoria)</strong></summary>

#### 1) Metadatos
- **Título:** BCE como NLL (MLE) y por qué el gradiente termina en `Xᵀ(p-y)`
- **ID (opcional):** `M05-T02_3`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 2.1–2.2, logaritmos

#### 2) Objetivos
- Entender BCE como “castigo logarítmico” a la probabilidad asignada a la clase correcta.
- Memorizar la forma del gradiente vectorizado.
- Entender por qué `eps` evita `log(0)` sin cambiar el objetivo conceptual.

#### 3) Relevancia
- Esta es la pérdida estándar para binario y base de softmax cross-entropy en multiclase.

#### 4) Mapa conceptual mínimo
- Si `y=1`: loss = `-log(p)`.
- Si `y=0`: loss = `-log(1-p)`.

#### 5) Definiciones esenciales
- `p = σ(Xθ)`.
- `∇θ = (1/m) Xᵀ(p-y)`.

#### 6) Explicación didáctica
- El gradiente “mide error en probabilidad”: si `p>y`, empuja hacia abajo; si `p<y`, empuja hacia arriba.

#### 7) Ejemplo modelado
- Una sola muestra: si `y=1` y `p=0.1`, el error `(p-y)` es negativo y el update mueve `θ` para subir `z`.

#### 8) Práctica guiada
- Haz un gradient check numérico en 1 coordenada (diferencias centrales) con dataset pequeño.

#### 9) Práctica independiente
- Grafica BCE vs `p` para `y=1` y `y=0` y explica la asimetría.

#### 10) Autoevaluación
- ¿Por qué BCE penaliza más el caso “seguro y equivocado” que MSE?

#### 11) Errores comunes
- Usar `y` como int pero con shape `(m,1)` y romper broadcasting.
- No hacer `clip` en `p` antes del log.

#### 12) Retención
- Fórmula clave: `grad = Xᵀ(p-y)/m`.

#### 13) Diferenciación
- Avanzado: relación con la entropía cruzada y KL-divergence.

#### 14) Recursos
- M04 (MLE→cross-entropy), glosario BCE.

#### 15) Nota docente
- Pide que el alumno derive el gradiente una vez y luego lo trate como “patrón” reusable.
</details>

### 2.4 Implementación Completa

```python
import numpy as np  # Importa NumPy: se usa para arrays, operaciones vectorizadas, RNG y construcción de datasets sintéticos
from typing import List  # Importa List: soporte de tipado (anotaciones) para listas; no afecta el runtime

class LogisticRegression:  # Regresión logística binaria desde cero: aprende θ para predecir P(y=1|x)=σ(X_bθ)
    """Regresión Logística implementada desde cero."""

    def __init__(self):  # Inicializa el modelo: prepara parámetros y contenedor de costes
        self.theta = None  # Vector de parámetros (d+1,): incluye bias en theta[0]; se asigna en fit
        self.cost_history = []  # Historial de coste (BCE) por iteración: útil para evaluar convergencia

    def fit(  # Entrena por gradient descent: minimiza BCE actualizando θ iterativamente
        self,  # Referencia al objeto: permite almacenar theta/cost_history en la instancia
        X: np.ndarray,  # Matriz de features (n,d): datos de entrada sin bias (se agrega dentro)
        y: np.ndarray,  # Vector de labels (n,): valores 0/1 para clasificación binaria
        learning_rate: float = 0.1,  # Tasa de aprendizaje α: escala el paso del update θ ← θ − α·grad
        n_iterations: int = 1000  # Número de iteraciones: cuántos pasos de GD se ejecutan
    ) -> 'LogisticRegression':  # Retorna self tipado: estilo sklearn para encadenar llamadas
        """Entrena con gradient descent."""
        # Añadir bias
        X_b = np.column_stack([np.ones(len(X)), X])  # Construye X con bias: agrega columna de 1s para intercepto
        m, n = X_b.shape  # Extrae shapes: m=n muestras, n=d+1 parámetros (incluye bias)

        # Inicializar
        self.theta = np.zeros(n)  # Inicializa θ en cero: punto de partida estándar para GD

        for i in range(n_iterations):  # Loop de entrenamiento: repite updates de GD n_iterations veces
            # Gradiente
            gradient = bce_gradient(X_b, y, self.theta)  # Calcula ∂BCE/∂θ: gradiente del coste respecto a parámetros

            # Actualizar
            self.theta = self.theta - learning_rate * gradient  # Update GD: mueve θ en dirección opuesta al gradiente

            # Guardar costo
            cost = binary_cross_entropy(X_b, y, self.theta)  # Calcula BCE actual: mide qué tan bien ajusta el modelo en esta iteración
            self.cost_history.append(cost)  # Guarda coste: permite inspeccionar si baja y detectar divergencia/NaN

        return self  # Devuelve la instancia entrenada: habilita chaining (model.fit(...).predict(...))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # Predice probabilidades: devuelve P(y=1|x) en (0,1)
        """Predice probabilidades."""
        X_b = np.column_stack([np.ones(len(X)), X])  # Añade bias: alinea X con theta aprendida
        return sigmoid(X_b @ self.theta)  # Aplica σ a logits: σ(X_bθ) produce probabilidades por muestra

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:  # Predice clases: umbraliza probabilidades para obtener 0/1
        """Predice clases."""
        return (self.predict_proba(X) >= threshold).astype(int)  # Comparación booleana y cast: devuelve etiquetas enteras

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # Evalúa accuracy: fracción de predicciones correctas
        """Accuracy."""
        return np.mean(self.predict(X) == y)  # Promedio de aciertos: compara arrays (True/False) y promedia


# Demo con datos sintéticos
np.random.seed(42)  # Fija semilla global: hace reproducible el dataset sintético de la demo

# Generar datos de dos clases
n_samples = 200  # Número total de muestras: se divide en dos mitades (clase 0 y clase 1)
X_class0 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])  # Clase 0: nube gaussiana centrada en (-2,-2)
X_class1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])  # Clase 1: nube gaussiana centrada en (2,2)
X = np.vstack([X_class0, X_class1])  # Apila features: matriz final (n,2) con ambas clases
y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))  # Labels: 0 para clase 0 y 1 para clase 1 (balanceado)

# Entrenar
model = LogisticRegression()  # Crea modelo: instancia la implementación desde cero
model.fit(X, y, learning_rate=0.1, n_iterations=1000)  # Entrena por GD: ajusta theta usando el dataset sintético

print(f"Accuracy: {model.score(X, y):.2%}")  # Reporta accuracy en train: en dataset separable debería ser alta (sanity check)
print(f"Parámetros: {model.theta}")  # Imprime theta: permite inspeccionar signo/magnitud (incluye bias en theta[0])
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 2.4: Implementación completa (checklist de robustez)</strong></summary>

#### 1) Metadatos
- **Título:** Cómo saber que tu LogReg “funciona”: contratos, overfit test y Shadow Mode
- **ID (opcional):** `M05-T02_4`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** 2.1–2.3

#### 2) Objetivos
- Establecer invariantes: `theta` tamaño correcto, `cost_history` finito, `predict_proba` en `(0,1)`.
- Detectar rápido si GD diverge (cost sube/NaN).
- Ejecutar “overfit test” como prueba unitaria del entrenamiento.

#### 3) Relevancia
- Un modelo que “corre” no necesariamente aprende. Necesitas una batería mínima de checks.

#### 4) Mapa conceptual mínimo
- **Datos** → **bias** → **sigmoid** → **BCE** → **grad** → **update**.

#### 5) Definiciones esenciales
- `X_b = [1, X]`.
- `theta[0]` es bias.

#### 6) Explicación didáctica
- Si el costo no baja en un dataset fácil, asume bug antes de “tocar hiperparámetros”.

#### 7) Ejemplo modelado
- Con datos separables (dos gaussianas separadas), deberías obtener accuracy alta.

#### 8) Práctica guiada
- Imprime cada 100 iteraciones: `cost`. Debe caer en promedio.

#### 9) Práctica independiente
- Añade early stopping y guarda el mejor `theta` por costo.

#### 10) Autoevaluación
- ¿Qué síntoma te indica signo invertido en el update? (costo sube sistemáticamente).

#### 11) Errores comunes
- No escalar features.
- Confundir `predict_proba` con `predict` en métricas.

#### 12) Retención
- Checklist mínimo: `finite`, `monotonic-ish`, `overfit test`, `shadow mode`.

#### 13) Diferenciación
- Avanzado: regularización L2 (MAP) y su efecto en estabilidad.

#### 14) Recursos
- Plan v5: validación externa y rutina de checks.

#### 15) Nota docente
- Pide evidencia: captura de `cost_history` (inicio vs final) + comparación con sklearn.
</details>

---

## 🧩 Consolidación (Regresión Logística)

<details open>
<summary><strong>📌 Complemento pedagógico — Consolidación LogReg: interpretación y criterio de dominio</strong></summary>

#### 1) Metadatos
- **Título:** De “entrenar un modelo” a “entender qué aprendió” (pesos como explicación)
- **ID (opcional):** `M05-CONS-LOGREG`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 2.4

#### 2) Objetivos
- Interpretar el vector de pesos como “dirección” que favorece una clase.
- En imágenes (MNIST), mapear `theta[1:]` a 28×28 y explicar regiones importantes.

#### 3) Relevancia
- Esto te entrena para hacer informes: no solo reportar accuracy, sino justificar el comportamiento del modelo.

#### 4) Mapa conceptual mínimo
- Pesos positivos aumentan `z` → suben probabilidad de clase 1.
- Pesos negativos disminuyen `z` → bajan probabilidad.

#### 5) Definiciones esenciales
- `theta[0]`: bias.
- `theta[1:]`: pesos por feature.

#### 6) Explicación didáctica
- Interpretación correcta es “si sube esta feature, sube/baja el logit”, no “causa”.

#### 7) Ejemplo modelado
- Para 0 vs 1, pesos en trazos típicos del “1” deberían ser positivos (según cómo codifiques la clase).

#### 8) Práctica guiada
- Guarda el mapa de pesos y escribe 5 líneas de interpretación con hipótesis verificables.

#### 9) Práctica independiente
- Repite con otra pareja (3 vs 8) y discute por qué es más difícil.

#### 10) Autoevaluación
- ¿Cómo cambia la interpretación si inviertes qué clase es 1 y cuál es 0?

#### 11) Errores comunes
- Olvidar remover el bias antes del reshape.
- Interpretar magnitudes sin normalizar features.

#### 12) Retención
- “Pesos → logit → probabilidad”: siempre explica primero qué clase corresponde a `y=1`.

#### 13) Diferenciación
- Avanzado: inspeccionar errores (top confusiones) y correlacionarlos con regiones de peso.

#### 14) Recursos
- Herramientas de visualización y notas de interpretabilidad lineal.

#### 15) Nota docente
- Pide consistencia: la explicación debe predecir qué píxeles cambiarían la predicción.
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 3.0: Métricas (cómo elegir y no autoengañarte)</strong></summary>

#### 1) Metadatos
- **Título:** Métricas como decisión: qué optimizas depende del costo (FP vs FN) y del umbral
- **ID (opcional):** `M05-T03_0`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** LogReg (probabilidades + threshold), matriz de confusión

#### 2) Objetivos
- Pasar de “sé calcular” a “sé elegir” la métrica correcta según el problema.
- Entender cómo el `threshold` cambia precision/recall sin re-entrenar el modelo.
- Detectar el caso clásico de autoengaño: accuracy alta con dataset desbalanceado.

#### 3) Relevancia
- En proyectos reales, la métrica es parte del producto: define qué errores toleras.
- Métricas conectan directamente con tu política de decisión (umbral) y con el tipo de informe.

#### 4) Mapa conceptual mínimo
- **Modelo** produce `p(y=1|x)`.
- **Threshold** produce `ŷ`.
- `ŷ` + `y` → **confusion matrix** → métricas.

#### 5) Definiciones esenciales
- **TP/TN/FP/FN:** conteos base.
- **Precision:** de lo que dije “positivo”, cuánto era positivo.
- **Recall:** de lo positivo real, cuánto capturé.

#### 6) Explicación didáctica
- Si tu modelo solo da clases, ya tomaste una decisión de threshold (implícita). Mejor separar: proba → threshold → métricas.

#### 7) Ejemplo modelado
- Detección de cáncer: FN es caro → prioriza recall.
- Filtro de spam: FP es caro → prioriza precision.

#### 8) Práctica guiada
- Para un mismo modelo, evalúa `threshold` en `{0.3, 0.5, 0.7}` y anota cómo cambian precision/recall.

#### 9) Práctica independiente
- Crea un dataset con 95% clase 0 y muestra:
  - baseline “siempre 0” → accuracy alta, pero recall para clase 1 = 0.

#### 10) Autoevaluación
- ¿Por qué no puedes comparar modelos con thresholds distintos sin decir el threshold?

#### 11) Errores comunes
- Reportar solo accuracy.
- Evaluar sobre train y reportar métricas “perfectas”.

#### 12) Retención
- Regla: “métrica = costo implícito” (si no lo defines, el modelo decide por ti).

#### 13) Diferenciación
- Avanzado: curva PR/ROC como barrido de thresholds (sin cambiar el modelo).

#### 14) Recursos
- Glosario de confusion matrix/precision/recall/F1.

#### 15) Nota docente
- Pide que el alumno justifique una métrica con una frase de costo (“FN cuesta más que FP”).
</details>

### 3.1 Matriz de Confusión

```python
import numpy as np  # arrays y operaciones vectorizadas para conteos

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:  # matriz de confusión para clasificación binaria/multiclase
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
    classes = np.unique(np.concatenate([y_true, y_pred]))  # obtiene clases presentes en true y pred
    n_classes = len(classes)  # número de clases
    cm = np.zeros((n_classes, n_classes), dtype=int)  # inicializa matriz KxK de conteos

    for i, true_class in enumerate(classes):  # recorre clases reales (filas)
        for j, pred_class in enumerate(classes):  # recorre clases predichas (columnas)
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))  # cuenta ocurrencias (true=i, pred=j)

    return cm  # retorna matriz de confusión

def extract_tp_tn_fp_fn(y_true: np.ndarray, y_pred: np.ndarray):  # extrae TP/TN/FP/FN (asumiendo clase positiva=1)
    """Extrae TP, TN, FP, FN para clasificación binaria."""
    tp = np.sum((y_true == 1) & (y_pred == 1))  # true positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # true negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # false positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # false negatives
    return tp, tn, fp, fn  # retorna conteos
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 3.1: Matriz de Confusión (la base de todo)</strong></summary>

#### 1) Metadatos
- **Título:** TP/TN/FP/FN como diagnóstico (no como tabla)
- **ID (opcional):** `M05-T03_1`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** Definir explícitamente cuál es la clase positiva

#### 2) Objetivos
- Leer la matriz 2×2 sin confundirte entre FP y FN.
- Traducir el problema (spam, fraude, cáncer, etc.) a “qué error es más caro”.
- Usar la matriz para explicar cambios de precision/recall al mover el threshold.

#### 3) Relevancia
- Todas las métricas son funciones de estos cuatro números.
- Si FP/FN están invertidos, todo el análisis posterior queda inválido.

#### 4) Mapa conceptual mínimo
- `y_true` vs `y_pred` → conteos → métricas.
- Cambiar `threshold` mueve masa entre celdas (no crea magia).

#### 5) Definiciones esenciales
- **FP:** predije 1 pero era 0 (alarma falsa).
- **FN:** predije 0 pero era 1 (caso perdido).

#### 6) Explicación didáctica
- Subir threshold suele:
  - bajar FP (menos alarmas)
  - subir FN (pierdes positivos)

#### 7) Ejemplo modelado
- Si “positivo” = cáncer, FN suele ser más grave que FP.

#### 8) Práctica guiada
- Crea 10 pares (true,pred) y llena la matriz a mano.

#### 9) Práctica independiente
- Repite con una definición distinta de “positivo” y observa cómo cambia la interpretación.

#### 10) Autoevaluación
- ¿Qué celda corresponde a “dije 0 pero era 1”?

#### 11) Errores comunes
- No declarar clase positiva.
- Intercambiar FP/FN.

#### 12) Retención
- Atajo: FP = (pred 1, true 0), FN = (pred 0, true 1).

#### 13) Diferenciación
- Multiclase: matriz K×K, y cada clase se puede analizar como one-vs-rest.

#### 14) Recursos
- Glosario: Confusion Matrix.

#### 15) Nota docente
- Exige que el alumno explique un FP y un FN con un ejemplo de su dominio.
</details>

### 3.2 Accuracy, Precision, Recall, F1

```python
import numpy as np  # arrays y operaciones vectorizadas para métricas

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # accuracy binario: proporción de aciertos
    """
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Proporción de predicciones correctas.

    Problema: Puede ser engañoso con clases desbalanceadas.
    Si 99% son clase 0, predecir siempre 0 da 99% accuracy.
    """
    return np.mean(y_true == y_pred)  # promedio de booleanos True/False => accuracy en [0,1]

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # precision binario: confiabilidad de positivos
    """
    Precision = TP / (TP + FP)

    De todos los que predije como positivos, ¿cuántos realmente lo son?

    Alta precisión = pocos falsos positivos.
    Importante cuando el costo de FP es alto (ej: spam → inbox).
    """
    tp, tn, fp, fn = extract_tp_tn_fp_fn(y_true, y_pred)  # extrae conteos de confusión
    if tp + fp == 0:  # caso borde: no hay predicciones positivas
        return 0.0  # por convención, precision=0
    return tp / (tp + fp)  # TP/(TP+FP)

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # recall binario: cobertura de positivos reales
    """
    Recall (Sensitivity, True Positive Rate) = TP / (TP + FN)

    De todos los positivos reales, ¿cuántos capturé?

    Alto recall = pocos falsos negativos.
    Importante cuando el costo de FN es alto (ej: detección de cáncer).
    """
    tp, tn, fp, fn = extract_tp_tn_fp_fn(y_true, y_pred)  # extrae conteos de confusión
    if tp + fn == 0:  # caso borde: no hay positivos reales
        return 0.0  # por convención, recall=0
    return tp / (tp + fn)  # TP/(TP+FN)

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # F1 binario: balance entre precision y recall
    """
    F1 = 2 * (precision * recall) / (precision + recall)

    Media armónica de precision y recall.

    Útil cuando quieres un balance entre ambas métricas.
    F1 alto solo si AMBAS precision y recall son altas.
    """
    p = precision(y_true, y_pred)  # calcula precision
    r = recall(y_true, y_pred)  # calcula recall
    if p + r == 0:  # caso borde: ambas métricas en 0
        return 0.0  # por convención, F1=0
    return 2 * (p * r) / (p + r)  # 2PR/(P+R)

def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # especificidad: cobertura de negativos reales
    """
    Specificity (True Negative Rate) = TN / (TN + FP)

    De todos los negativos reales, ¿cuántos identifiqué?
    """
    tp, tn, fp, fn = extract_tp_tn_fp_fn(y_true, y_pred)  # extrae conteos de confusión
    if tn + fp == 0:  # caso borde: no hay negativos reales
        return 0.0  # por convención, specificity=0
    return tn / (tn + fp)  # TN/(TN+FP)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 3.2: Accuracy/Precision/Recall/F1 (cuándo usar cada una)</strong></summary>

#### 1) Metadatos
- **Título:** Elegir métrica = declarar costo (y reportar threshold)
- **ID (opcional):** `M05-T03_2`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 3.1

#### 2) Objetivos
- Identificar cuándo accuracy es engañosa (desbalance).
- Elegir precision vs recall según el costo de FP vs FN.
- Entender F1 como balance: cae si una de las dos es baja.

#### 3) Relevancia
- Aquí defines “qué significa que el modelo sea bueno”.

#### 4) Mapa conceptual mínimo
- accuracy: desempeño global.
- precision: control de FP.
- recall: control de FN.
- F1: balance precision/recall.

#### 5) Definiciones esenciales
- **Precision** responde: “si dije 1, ¿cuántas veces acerté?”
- **Recall** responde: “de los 1 reales, ¿cuántos encontré?”

#### 6) Explicación didáctica
- Al subir threshold, normalmente sube precision y baja recall.

#### 7) Ejemplo modelado
- Modelo conservador: predice pocos 1 → precision alta, recall baja.

#### 8) Práctica guiada
- Con el mismo conjunto, evalúa `threshold` en 0.3/0.5/0.7 y compara.

#### 9) Práctica independiente
- Busca un threshold que maximice F1 en validación y reporta (F1, threshold).

#### 10) Autoevaluación
- ¿Qué te falta para reproducir el mismo reporte mañana?

#### 11) Errores comunes
- Reportar métricas sin decir threshold.
- Optimizar F1 sin justificar el costo del error.

#### 12) Retención
- Regla: costo → métrica → threshold.

#### 13) Diferenciación
- Multiclase: macro vs micro (cuando las clases están desbalanceadas).

#### 14) Recursos
- Glosario: Precision/Recall/F1.

#### 15) Nota docente
- Obliga a que el alumno elija una métrica y la defienda con una frase de costo.
</details>

### 3.3 Clase Metrics Completa

```python
import numpy as np  # arrays y operaciones numéricas para métricas
from dataclasses import dataclass  # decorador para crear clases de datos

@dataclass  # genera __init__/__repr__/etc. para el reporte
class ClassificationReport:  # contenedor tipado para métricas de clasificación
    """Reporte de métricas de clasificación."""
    accuracy: float  # proporción de aciertos
    precision: float  # TP / (TP + FP)
    recall: float  # TP / (TP + FN)
    f1: float  # media armónica de precision/recall
    specificity: float  # TN / (TN + FP)
    confusion_matrix: np.ndarray  # matriz 2x2 de conteos

    def __str__(self) -> str:  # formatea el reporte como texto legible
        cm = self.confusion_matrix  # alias local para interpolar en el template
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

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationReport:  # empaqueta métricas en un objeto
    """Genera reporte completo de métricas."""
    return ClassificationReport(  # construye el reporte con métricas base
        accuracy=accuracy(y_true, y_pred),  # accuracy global
        precision=precision(y_true, y_pred),  # precision (clase positiva)
        recall=recall(y_true, y_pred),  # recall (clase positiva)
        f1=f1_score(y_true, y_pred),  # F1-score
        specificity=specificity(y_true, y_pred),  # especificidad (clase negativa)
        confusion_matrix=confusion_matrix(y_true, y_pred)  # matriz de confusión 2x2
    )  # retorna instancia de ClassificationReport

# Demo
y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # etiquetas reales
y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1])  # predicciones (ejemplo)

report = classification_report(y_true, y_pred)  # genera reporte de métricas
print(report)  # imprime reporte formateado
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 3.3: Reporte de métricas (de números a diagnóstico)</strong></summary>

#### 1) Metadatos
- **Título:** Del reporte a la acción: qué cambiar si precision/recall no cumplen
- **ID (opcional):** `M05-T03_3`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 3.1–3.2

#### 2) Objetivos
- Empaquetar métricas para comparar experimentos sin confusión.
- Interpretar el reporte como diagnóstico: qué tipo de error domina.
- Mantener reproducibilidad: mismo split/seed/threshold.

#### 3) Relevancia
- En un proyecto, el reporte es lo que justifica decisiones (no solo el código).

#### 4) Mapa conceptual mínimo
- confusion matrix → métricas → reporte → decisión (threshold/feature/modelo).

#### 5) Definiciones esenciales
- “Reporte” no es solo números: requiere contexto (dataset/split/threshold).

#### 6) Explicación didáctica
- Si el reporte no incluye contexto, es fácil autoengañarse con comparaciones inválidas.

#### 7) Ejemplo modelado
- Recall bajo: baja threshold o mejora features; Precision baja: sube threshold o reduce ruido.

#### 8) Práctica guiada
- Cambia 2 predicciones del demo y observa cómo cambian todas las métricas.

#### 9) Práctica independiente
- Extiende a macro-F1 en multiclase (one-vs-rest).

#### 10) Autoevaluación
- ¿Qué te falta para reproducir el mismo reporte mañana?

#### 11) Errores comunes
- Comparar reportes de datasets distintos.

#### 12) Retención
- “Métrica sin contexto = número sin significado”.

#### 13) Diferenciación
- Avanzado: incluir `mean±std` vía cross-validation.

#### 14) Recursos
- Plan v5: disciplina de validación y registro de resultados.

#### 15) Nota docente
- Pide una recomendación concreta basada en el reporte (threshold/features/datos).
</details>

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

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 4.0: Validación + Regularización (workflow anti-autoengaño)</strong></summary>

#### 1) Metadatos
- **Título:** Cómo saber si generalizas: split correcto, validación y control de overfitting
- **ID (opcional):** `M05-T04_0`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio/Avanzado
- **Dependencias:** Métricas (Parte 3), LogReg (Parte 2)

#### 2) Objetivos
- Explicar la diferencia entre train/val/test y por qué el test no se toca.
- Entender qué pregunta responde K-fold (variancia de performance).
- Entender regularización como control de complejidad efectiva (bias-varianza).

#### 3) Relevancia
- Sin validación, puedes “ganar” en train y fallar en producción.
- Regularización es una herramienta central para modelos lineales y redes.

#### 4) Mapa conceptual mínimo
- Entrenar en train.
- Elegir hiperparámetros con val (o CV).
- Reportar final en test una sola vez.

#### 5) Definiciones esenciales
- **Leakage:** usar info del test/val al entrenar.
- **Overfitting:** buen train, mal test.

#### 6) Explicación didáctica
- Si miras el test repetidamente, el test se convierte en “val” sin querer.

#### 7) Ejemplo modelado
- Dos seeds distintas → dos splits distintos → accuracy distinta: eso es varianza.

#### 8) Práctica guiada
- Ejecuta 2 splits con semillas diferentes y reporta ambas métricas.

#### 9) Práctica independiente
- Haz K-fold y reporta `mean ± std`.

#### 10) Autoevaluación
- ¿Cuál conjunto se usa para elegir `lambda_`?

#### 11) Errores comunes
- Normalizar usando todo el dataset antes del split.
- Elegir hiperparámetros “viendo” el test.

#### 12) Retención
- Regla: test se usa una vez, al final.

#### 13) Diferenciación
- Avanzado: nested CV (conceptual) para selección + evaluación robusta.

#### 14) Recursos
- Plan v5: Shadow Mode para validar implementaciones.

#### 15) Nota docente
- Exigir que el alumno declare explícitamente qué datos usó para cada decisión.
</details>

### 4.1 Train/Test Split

```python
import numpy as np  # arrays y utilidades de aleatoriedad

def train_test_split(  # divide X/y en train y test de forma reproducible
    X: np.ndarray,  # features
    y: np.ndarray,  # targets
    test_size: float = 0.2,  # proporción para test
    random_state: int = None  # semilla opcional
) -> tuple:  # retorna (X_train, X_test, y_train, y_test)
    """
    Divide datos en conjuntos de entrenamiento y prueba.

    Args:
        X: features
        y: targets
        test_size: proporción para test (0-1)
        random_state: semilla para reproducibilidad
    """
    if random_state is not None:  # si se pide reproducibilidad
        np.random.seed(random_state)  # fija semilla global de numpy

    n = len(y)  # número de muestras
    indices = np.random.permutation(n)  # permuta índices de 0..n-1

    test_size_n = int(n * test_size)  # tamaño del test en número de muestras
    test_indices = indices[:test_size_n]  # índices para test
    train_indices = indices[test_size_n:]  # índices para train

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]  # indexa y retorna splits
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 4.1: Train/Test Split (contratos y fugas)</strong></summary>

#### 1) Metadatos
- **Título:** Split reproducible: qué debe ser aleatorio y qué debe ser determinista
- **ID (opcional):** `M05-T04_1`
- **Duración estimada:** 45–90 min
- **Nivel:** Intermedio
- **Dependencias:** 4.0

#### 2) Objetivos
- Verificar invariantes: tamaños de split, alineación X/y, sin duplicados.
- Entender el rol de `random_state`.

#### 3) Relevancia
- Si tu split está mal, todo el benchmark se vuelve irrelevante.

#### 4) Mapa conceptual mínimo
- Permutar índices → cortar → indexar X/y.

#### 5) Definiciones esenciales
- **Reproducibilidad:** misma semilla → mismo split.

#### 6) Explicación didáctica
- Split debe hacerse antes de normalizar/seleccionar features (evita leakage).

#### 7) Ejemplo modelado
- Verifica que `len(train)+len(test)=n`.

#### 8) Práctica guiada
- Imprime tamaños y distribuciones de clase por split.

#### 9) Práctica independiente
- Implementa split estratificado (conceptual) para clasificación desbalanceada.

#### 10) Autoevaluación
- ¿Qué se rompe si `X` y `y` se permutan con índices distintos?

#### 11) Errores comunes
- Reusar el test para “ajustar” el modelo.

#### 12) Retención
- Regla: split primero; transformaciones después (fit en train).

#### 13) Diferenciación
- Avanzado: train/val/test + pipelines.

#### 14) Recursos
- Plan v4/v5: disciplina de evaluación.

#### 15) Nota docente
- Pide que el alumno identifique 2 formas de leakage y cómo evitarlas.
</details>

### 4.2 K-Fold Cross Validation

```python
import numpy as np  # Importa NumPy: se usa para generar índices, barajar, concatenar y calcular estadísticas
from typing import List, Tuple  # Importa tipos: anota listas de folds y tuplas (train_idx, val_idx) para claridad

def k_fold_split(n: int, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:  # Genera folds: devuelve pares (train_indices, val_indices)
    """
    Genera índices para K-Fold Cross Validation.

    Returns:
        Lista de (train_indices, val_indices) para cada fold
    """
    indices = np.arange(n)  # Crea índices 0..n-1: base para particionar el dataset en folds
    np.random.shuffle(indices)  # Baraja in-place: randomiza el orden para evitar sesgos por ordenación

    fold_size = n // k  # Tamaño de cada fold: división entera (si sobra, el último fold se ajusta)
    folds = []  # Lista acumuladora: contendrá k tuplas (train_idx, val_idx)

    for i in range(k):  # Itera k folds: cada i define un segmento distinto para validación
        start = i * fold_size  # Inicio del fold i dentro del vector de índices barajados
        end = start + fold_size if i < k - 1 else n  # Fin del fold: último fold captura el remanente para cubrir todos

        val_indices = indices[start:end]  # Índices de validación: subarray del fold i
        train_indices = np.concatenate([indices[:start], indices[end:]])  # Índices de entrenamiento: todos los índices fuera del fold i

        folds.append((train_indices, val_indices))  # Guarda fold: par (train,val) para consumirlo en cross_validate

    return folds  # Devuelve lista de folds: cada elemento permite indexar X e y con train/val

def cross_validate(  # ejecuta validación cruzada K-Fold para una clase de modelo
    model_class,  # Clase del modelo a evaluar: debe poder instanciarse como model_class() y tener fit/score
    X: np.ndarray,  # Matriz de features (n,d): datos completos a partir de los cuales se generan folds
    y: np.ndarray,  # Vector de etiquetas (n,): targets alineados con X
    k: int = 5,  # Número de folds: más folds => menos sesgo pero más costo computacional
    **model_params  # Hiperparámetros: se pasan a model.fit(...) en cada fold
) -> dict:  # Retorna un dict con scores por fold y agregados (mean/std)
    """
    Realiza K-Fold Cross Validation.

    Returns:
        Dict con scores de cada fold y promedio
    """
    folds = k_fold_split(len(y), k)  # Genera folds: lista de pares (train_idx, val_idx) para iterar
    scores = []  # Acumula scores por fold: se agregan para media y desviación

    for i, (train_idx, val_idx) in enumerate(folds):  # Recorre folds: cada iteración entrena en train y evalúa en val
        # Split
        X_train, X_val = X[train_idx], X[val_idx]  # Subconjuntos de features: train/val para el fold actual
        y_train, y_val = y[train_idx], y[val_idx]  # Subconjuntos de labels: targets alineados con los índices

        # Train
        model = model_class()  # Instancia un modelo nuevo: evita contaminación de estado entre folds
        model.fit(X_train, y_train, **model_params)  # Entrena en train: aplica hiperparámetros pasados por **model_params

        # Evaluate
        score = model.score(X_val, y_val)  # Evalúa desempeño en validación: métrica provista por el modelo (p.ej., accuracy)
        scores.append(score)  # Guarda score del fold: se usará para estadística final

    return {  # Devuelve resumen de CV: permite reportar distribución y estabilidad del rendimiento
        'scores': scores,  # Lista de scores por fold: muestra variación por partición
        'mean': np.mean(scores),  # Promedio de scores: estimación puntual del rendimiento
        'std': np.std(scores)  # Desviación estándar: cuantifica sensibilidad al split (inestabilidad)
    }  # Cierra dict: estructura final de resultados para logging/decisiones

# Demo
# cv_results = cross_validate(LogisticRegression, X, y, k=5, learning_rate=0.1, n_iterations=500)
# print(f"CV Accuracy: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 4.2: K-Fold (qué estima y qué no)</strong></summary>

#### 1) Metadatos
- **Título:** K-fold como estimador de varianza y robustez (no como “mejorar accuracy”)
- **ID (opcional):** `M05-T04_2`
- **Duración estimada:** 60–120 min
- **Nivel:** Intermedio
- **Dependencias:** 4.1

#### 2) Objetivos
- Entender que K-fold produce una distribución de scores.
- Reportar `mean ± std`.
- Evitar errores de leakage en CV (fit transforms dentro de cada fold).

#### 3) Relevancia
- Te da confianza en la estabilidad del modelo.

#### 4) Mapa conceptual mínimo
- Repartir índices en folds → entrenar k veces → evaluar k veces.

#### 5) Definiciones esenciales
- **Fold:** partición usada como validación.

#### 6) Explicación didáctica
- Si la std es alta, tu rendimiento depende demasiado del split.

#### 7) Ejemplo modelado
- `k=5` → 5 scores; promedia y reporta dispersión.

#### 8) Práctica guiada
- Ejecuta CV con 2 seeds y compara la std.

#### 9) Práctica independiente
- Implementa un “grid” pequeño sobre `learning_rate` y compara medias.

#### 10) Autoevaluación
- ¿Por qué CV no reemplaza el test final?

#### 11) Errores comunes
- Elegir hiperparámetros y evaluar todo en el mismo CV sin un test final (sobreajuste de selección).

#### 12) Retención
- Regla: CV para selección/estimación; test para cierre.

#### 13) Diferenciación
- Avanzado: nested CV (conceptual).

#### 14) Recursos
- Notas de validación y bias-varianza.

#### 15) Nota docente
- Pide que el alumno explique qué significa “std alta” con una analogía.
</details>

### 4.3 Regularización

```python
import numpy as np  # Importa NumPy: se usa para álgebra vectorizada, concatenación, abs/sign y operaciones en coste/gradiente

class LogisticRegressionRegularized:  # Logistic Regression con regularización: añade penalización L1/L2 para controlar complejidad
    """Logistic Regression con regularización L1/L2."""

    def __init__(self, regularization: str = 'l2', lambda_: float = 0.01):  # Inicializa configuración de regularización (tipo y fuerza)
        """
        Args:
            regularization: 'l1', 'l2', o None
            lambda_: fuerza de regularización
        """
        self.regularization = regularization  # Tipo de penalización: 'l2' (ridge), 'l1' (lasso) o None (sin regularizar)
        self.lambda_ = lambda_  # Parámetro λ: escala cuánto pesa la penalización respecto a la loss base
        self.theta = None  # Parámetros del modelo (d+1,): incluye bias en theta[0]; se inicializa al entrenar
        self.cost_history = []  # Historial del coste total por iteración: útil para ver convergencia y efecto de λ

    def _cost(self, X: np.ndarray, y: np.ndarray) -> float:  # calcula el coste BCE + término de regularización
        """Costo con regularización."""
        m = len(y)  # Número de muestras: se usa para promediar la loss y escalar el término de regularización
        h = sigmoid(X @ self.theta)  # Probabilidades predichas: aplica σ a logits Xθ (X ya incluye bias)
        h = np.clip(h, 1e-15, 1 - 1e-15)  # Evita log(0): recorta h para estabilidad numérica en cross-entropy

        # Cross-entropy base
        bce = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))  # BCE promedio: log-loss binaria (cross-entropy)

        # Regularización (excluir bias theta[0])
        if self.regularization == 'l2':  # L2/Ridge: penaliza suma de cuadrados de pesos (excluyendo el bias)
            # Ridge: λ/2m * Σθⱼ²
            reg = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:] ** 2)  # Término L2: (λ/2m)||θ_{1:}||²
        elif self.regularization == 'l1':  # L1/Lasso: penaliza suma de valores absolutos (promueve sparsity)
            # Lasso: λ/m * Σ|θⱼ|
            reg = (self.lambda_ / m) * np.sum(np.abs(self.theta[1:]))  # Término L1: (λ/m)||θ_{1:}||₁
        else:  # Sin regularización: no se añade penalización al coste
            reg = 0  # Penalización nula: coste total coincide con BCE

        return bce + reg  # Coste total: loss base + penalización (controla overfitting)

    def _gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:  # calcula gradiente BCE + contribución de regularización
        """Gradiente con regularización."""
        m = len(y)  # Número de muestras: normaliza el gradiente (promedio)
        h = sigmoid(X @ self.theta)  # Probabilidades: σ(Xθ) para construir el gradiente de la loss

        # Gradiente base
        grad = (1/m) * X.T @ (h - y)  # Gradiente BCE: (1/m) Xᵀ(h−y) para logística (sin regularización)

        # Regularización (excluir bias)
        if self.regularization == 'l2':  # Gradiente L2: proporcional a θ (excepto bias)
            reg_grad = np.concatenate([[0], (self.lambda_ / m) * self.theta[1:]])  # [0, (λ/m)θ₁:]: excluye bias en la penalización
        elif self.regularization == 'l1':  # Gradiente L1: usa signo (subgradiente) para empujar a 0
            reg_grad = np.concatenate([[0], (self.lambda_ / m) * np.sign(self.theta[1:])])  # [0, (λ/m)sign(θ₁:)]: excluye bias
        else:  # Sin regularización: no se suma gradiente adicional
            reg_grad = 0  # Penalización nula: gradiente total coincide con grad base

        return grad + reg_grad  # Gradiente total: grad base + contribución de regularización

    def fit(self, X: np.ndarray, y: np.ndarray,  # entrena el modelo con GD (firma multi-línea)
            learning_rate: float = 0.1, n_iterations: int = 1000):  # Entrena por GD: actualiza θ usando gradiente con regularización
        X_b = np.column_stack([np.ones(len(X)), X])  # Añade bias: crea X_b (n,d+1) con columna de 1s al inicio
        self.theta = np.zeros(X_b.shape[1])  # Inicializa θ: un parámetro por columna de X_b (incluye bias)

        for _ in range(n_iterations):  # Loop de entrenamiento: repite actualizaciones de GD
            gradient = self._gradient(X_b, y)  # Calcula gradiente total en el punto actual (incluye regularización)
            self.theta -= learning_rate * gradient  # Actualiza θ: θ ← θ − α·grad
            self.cost_history.append(self._cost(X_b, y))  # Registra coste: permite inspeccionar convergencia y overfitting

        return self  # Devuelve self: permite chaining estilo sklearn

    def predict(self, X: np.ndarray) -> np.ndarray:  # predice etiquetas binarias aplicando σ y umbral 0.5
        X_b = np.column_stack([np.ones(len(X)), X])  # Añade bias: alinea features con theta aprendida (incluye intercepto)
        return (sigmoid(X_b @ self.theta) >= 0.5).astype(int)  # Umbraliza probas: devuelve etiquetas 0/1 según prob>=0.5

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # calcula accuracy promedio sobre (X,y)
        return np.mean(self.predict(X) == y)  # Accuracy: promedio de aciertos del clasificador sobre (X,y)
```

<details open>
<summary><strong>📌 Complemento pedagógico — Sección 4.3: Regularización (L1/L2 y por qué se excluye el bias)</strong></summary>

#### 1) Metadatos
- **Título:** Regularización como control de complejidad: L2 (suaviza) vs L1 (sparse)
- **ID (opcional):** `M05-T04_3`
- **Duración estimada:** 90–150 min
- **Nivel:** Intermedio
- **Dependencias:** 2.3 (BCE/gradiente), 4.0

#### 2) Objetivos
- Entender qué término se agrega al costo y cómo afecta el gradiente.
- Justificar por qué el bias no se regulariza.
- Relacionar `lambda_` con bias-varianza.

#### 3) Relevancia
- Regularización suele ser la diferencia entre generalizar o sobreajustar en modelos lineales.

#### 4) Mapa conceptual mínimo
- Loss base + penalización a pesos → update más “conservador”.

#### 5) Definiciones esenciales
- **L2:** penaliza cuadrados (shrink continuo).
- **L1:** penaliza valores absolutos (promueve sparsity).

#### 6) Explicación didáctica
- Regularizar el bias puede desplazar la frontera innecesariamente.

#### 7) Ejemplo modelado
- Si `lambda_` sube, típicamente bajan magnitudes de `theta[1:]`.

#### 8) Práctica guiada
- Prueba `lambda_` en `{0,0.01,0.1,1.0}` y observa train/test.

#### 9) Práctica independiente
- Grafica norma de `theta` vs `lambda_`.

#### 10) Autoevaluación
- ¿Qué efecto esperas en el gap train-test cuando aumenta `lambda_`?

#### 11) Errores comunes
- Regularizar también `theta[0]`.
- Olvidar ajustar el gradiente con el término de regularización.

#### 12) Retención
- Regla: penaliza pesos, no el bias.

#### 13) Diferenciación
- Avanzado: conexión con MAP (prior gaussiano / laplaciano).

#### 14) Recursos
- Notas de Ridge/Lasso y sesgo-varianza.

#### 15) Nota docente
- Pide que el alumno explique por qué L1 puede hacer pesos exactamente 0.
</details>

---

### ⚠️ Aviso crítico antes de Árboles: Recursividad (Semana 12)

La implementación de árboles se basa en **recursión**. Si no defines y pruebas condiciones de parada, vas a generar árboles infinitos o muy profundos.

- Condiciones de parada mínimas: `max_depth`, pureza (todas las etiquetas iguales), `min_samples_split`, “no split improves”.
- Recurso recomendado: https://realpython.com/python-recursion/
- Debug mínimo: imprime `depth`, `n_samples` y el criterio elegido por nodo durante desarrollo.

### Micro-sprint (15 minutos): recursividad mínima para árboles

Dos reglas que debes internalizar:

- **Caso base:** el caso más pequeño que puedes responder inmediatamente (aquí se detiene la recursión).
- **Paso recursivo:** reduces el problema a una versión más pequeña de sí mismo.

Si no puedes decir el caso base en 1 línea, tu implementación del árbol probablemente recursará para siempre.

#### Ejemplo: suma recursiva (practica el modelo mental)

```python
from typing import Sequence  # Importa Sequence para anotar entradas indexables (listas/tuplas)

def sum_recursive(xs: Sequence[float]) -> float:  # suma recursiva de una secuencia de números
    # Caso base: la suma de una lista vacía es 0
    if len(xs) == 0:  # condición de parada: secuencia vacía
        return 0.0  # retorna 0 como elemento neutro de la suma

    # Paso recursivo: reduces el problema quitando el primer elemento
    return float(xs[0]) + sum_recursive(xs[1:])  # suma primer elemento y recurre con el resto


assert sum_recursive([]) == 0.0  # test: caso base (lista vacía)
assert sum_recursive([3.0]) == 3.0  # test: un solo elemento
assert sum_recursive([3.0, 2.0, 5.0]) == 10.0  # test: suma de varios elementos
```

#### Pila de llamadas (lo que Python está haciendo)

```text
sum_recursive([3, 2, 5])
= 3 + sum_recursive([2, 5])
    = 2 + sum_recursive([5])
        = 5 + sum_recursive([])
            = 0
```

#### Conexión con Decision Trees: condiciones de parada = casos base

Al construir un nodo, tu caso base debería dispararse cuando:

- `depth >= max_depth`
- el nodo es **puro** (todas las etiquetas son iguales)
- `n_samples < min_samples_split`
- ningún split mejora impureza (information gain <= 0)

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

## 🎯 Ejercicios por tema (progresivos) + Soluciones

Reglas:

- **Intenta primero** sin mirar la solución.
- **Timebox sugerido:** 20–45 min por ejercicio.
- **Éxito mínimo:** tu solución debe pasar los `assert`.

---

### Ejercicio 5.1: Regresión lineal (Normal Equation) + recuperación de pesos

#### Enunciado

1) **Básico**

- Genera un dataset sintético: `y = Xw + noise`.

2) **Intermedio**

- Estima `w_hat` usando la ecuación normal con `np.linalg.solve`.

3) **Avanzado**

- Verifica que `w_hat` se aproxima a `w_true` y que el MSE es pequeño.

#### Solución

```python
import numpy as np  # arrays y operaciones numéricas

np.random.seed(0)  # fija semilla para reproducibilidad
n, d = 500, 3  # número de muestras y dimensión
X = np.random.randn(n, d)  # features gaussianas
w_true = np.array([0.7, -1.5, 2.0])  # pesos "verdaderos" para generar y
noise = 0.05 * np.random.randn(n)  # ruido gaussiano aditivo
y = X @ w_true + noise  # targets con ruido

# Normal equation: (X^T X) w = X^T y
XtX = X.T @ X  # matriz X^T X
Xty = X.T @ y  # vector X^T y
w_hat = np.linalg.solve(XtX, Xty)  # estima w por ecuación normal

mse = np.mean((X @ w_hat - y) ** 2)  # error cuadrático medio

assert w_hat.shape == (d,)  # sanity check: shape de w_hat
assert np.linalg.norm(w_hat - w_true) < 0.15  # sanity check: recupera pesos
assert mse < 0.01  # sanity check: MSE pequeño
```

---

### Ejercicio 5.2: Regresión lineal (Gradient Descent) + comparación con Normal Equation

#### Enunciado

1) **Básico**

- Implementa GD para minimizar MSE: `w <- w - α (1/n) X^T (Xw - y)`.

2) **Intermedio**

- Compara `w_gd` contra `w_ne` (normal equation).

3) **Avanzado**

- Verifica que el loss disminuye (al menos al final es menor que al inicio).

#### Solución

```python
import numpy as np  # arrays y operaciones numéricas

np.random.seed(1)  # fija semilla para reproducibilidad
n, d = 400, 4  # número de muestras y dimensión
X = np.random.randn(n, d)  # features gaussianas
w_true = np.array([1.0, -2.0, 0.5, 3.0])  # pesos "verdaderos" para generar y
y = X @ w_true + 0.1 * np.random.randn(n)  # targets con ruido gaussiano

XtX = X.T @ X  # matriz X^T X
Xty = X.T @ y  # vector X^T y
w_ne = np.linalg.solve(XtX, Xty)  # solución cerrada (normal equation)

w = np.zeros(d)  # inicializa pesos para GD
alpha = 0.05  # learning rate
losses = []  # historial de loss (MSE) para verificar descenso
for _ in range(3000):  # iteraciones de GD
    r = X @ w - y  # residuo (pred - real)
    grad = (X.T @ r) / n  # gradiente de MSE: (1/n) X^T (Xw - y)
    w = w - alpha * grad  # actualización: w <- w - alpha * grad
    losses.append(float(np.mean(r**2)))  # guarda MSE actual

w_gd = w  # pesos aprendidos por GD

assert losses[-1] <= losses[0]  # sanity check: el loss baja
assert np.linalg.norm(w_gd - w_ne) < 0.2  # sanity check: GD se aproxima a normal equation
```

---

### Ejercicio 5.3: Métricas desde una matriz de confusión (TP/TN/FP/FN)

#### Enunciado

1) **Básico**

- Implementa una función que compute TP/TN/FP/FN para un problema binario.

2) **Intermedio**

- Implementa accuracy, precision, recall, F1.

3) **Avanzado**

- Valida con un caso conocido y `assert`.

#### Solución

```python
import numpy as np  # arrays y operaciones vectorizadas

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):  # calcula TP/TN/FP/FN para binario
    y_true = np.asarray(y_true).astype(int)  # normaliza a ndarray de enteros
    y_pred = np.asarray(y_pred).astype(int)  # normaliza a ndarray de enteros
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))  # true positives
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))  # true negatives
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))  # false positives
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))  # false negatives
    return tp, tn, fp, fn  # retorna conteos


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray):  # calcula métricas binarias desde conteos
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)  # obtiene TP/TN/FP/FN
    eps = 1e-12  # estabilizador para evitar división por cero
    acc = (tp + tn) / (tp + tn + fp + fn + eps)  # accuracy
    prec = tp / (tp + fp + eps)  # precision
    rec = tp / (tp + fn + eps)  # recall
    f1 = 2 * prec * rec / (prec + rec + eps)  # F1-score
    return float(acc), float(prec), float(rec), float(f1)  # retorna métricas como floats


y_true = np.array([1, 1, 1, 0, 0, 0])  # etiquetas verdaderas (binarias)
y_pred = np.array([1, 0, 1, 0, 1, 0])  # predicciones (binarias)
tp, tn, fp, fn = confusion_counts(y_true, y_pred)  # computa conteos de confusión

assert (tp, tn, fp, fn) == (2, 2, 1, 1)  # valida conteos esperados

acc, prec, rec, f1 = precision_recall_f1(y_true, y_pred)  # calcula métricas
assert np.isclose(acc, 4/6)  # valida accuracy
assert np.isclose(prec, 2/3)  # valida precision
assert np.isclose(rec, 2/3)  # valida recall
assert np.isclose(f1, 2/3)  # valida F1
```

---

### Ejercicio 5.4: Logistic Regression - sigmoid + BCE estable

#### Enunciado

1) **Básico**

- Implementa `sigmoid(z)` con `np.clip` para evitar overflow.

2) **Intermedio**

- Implementa Binary Cross-Entropy estable (con `clip`).

3) **Avanzado**

- Verifica:
  - BCE cerca de 0 para predicciones casi perfectas.
  - BCE ≈ `-log(0.9)` cuando `y=1` y `p=0.9`.

#### Solución

```python
import numpy as np  # arrays y operaciones numéricas

def sigmoid(z: np.ndarray) -> np.ndarray:  # sigmoide estable para logits
    z = np.asarray(z, dtype=float)  # asegura ndarray float
    z = np.clip(z, -500, 500)  # evita overflow en exp para |z| grande
    return 1.0 / (1.0 + np.exp(-z))  # σ(z)=1/(1+e^{-z})


def bce(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:  # BCE estable con clipping
    y_true = np.asarray(y_true, dtype=float)  # normaliza etiquetas a float
    y_pred = np.asarray(y_pred, dtype=float)  # normaliza predicciones a float
    y_pred = np.clip(y_pred, eps, 1.0 - eps)  # evita log(0) y log(1)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))  # BCE promedio


y_true = np.array([1.0, 0.0, 1.0, 0.0])  # etiquetas reales
y_pred_good = np.array([0.999, 0.001, 0.999, 0.001])  # predicciones casi perfectas
assert bce(y_true, y_pred_good) < 0.01  # BCE debe ser cercana a 0
assert np.isclose(bce(np.array([1.0]), np.array([0.9])), -np.log(0.9), atol=1e-12)  # caso y=1, p=0.9
```

---

### Ejercicio 5.5: Gradiente de Logistic Regression (verificación numérica)

#### Enunciado

1) **Básico**

- Implementa el gradiente de BCE para Logistic Regression:
  - `ŷ = sigmoid(Xw)`
  - `∇w = (1/n) X^T (ŷ - y)`

2) **Intermedio**

- Implementa una función de pérdida `L(w)`.

3) **Avanzado**

- Verifica 1 coordenada del gradiente con diferencias centrales.

#### Solución

```python
import numpy as np  # arrays, álgebra lineal y utilidades numéricas

def sigmoid(z: np.ndarray) -> np.ndarray:  # función logística para mapear logits a probabilidades
    z = np.asarray(z, dtype=float)  # asegura ndarray float para operaciones estables
    z = np.clip(z, -500, 500)  # evita overflow en exp para valores extremos
    return 1.0 / (1.0 + np.exp(-z))  # aplica sigmoide elemento a elemento


def bce_from_logits(X: np.ndarray, y: np.ndarray, w: np.ndarray, eps: float = 1e-15) -> float:  # BCE dada X,y,w
    logits = X @ w  # calcula logits z = Xw
    y_hat = sigmoid(logits)  # convierte logits a probabilidades
    y_hat = np.clip(y_hat, eps, 1.0 - eps)  # evita log(0) en la BCE
    return float(-np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))  # BCE promedio


def grad_bce(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:  # gradiente de BCE para logística
    y_hat = sigmoid(X @ w)  # predicción probabilística ŷ
    return (X.T @ (y_hat - y)) / X.shape[0]  # ∇w = (1/n) X^T (ŷ - y)


np.random.seed(2)  # fija semilla global para reproducibilidad
n, d = 200, 3  # número de muestras y dimensión
X = np.random.randn(n, d)  # features gaussianas
w0 = np.array([0.3, -0.7, 1.2])  # pesos "verdaderos" para generar etiquetas
probs = sigmoid(X @ w0)  # probabilidades base p(y=1|x) según w0
y = (np.random.rand(n) < probs).astype(float)  # samplea y ~ Bernoulli(probs)

w = np.random.randn(d)  # inicializa pesos para evaluar gradiente
g = grad_bce(X, y, w)  # gradiente analítico

idx = 1  # coordenada del gradiente a verificar
h = 1e-6  # tamaño de paso para diferencias centrales
e = np.zeros(d)  # vector base para perturbar una sola coordenada
e[idx] = 1.0  # activa la coordenada idx
L_plus = bce_from_logits(X, y, w + h * e)  # pérdida con w[idx] + h
L_minus = bce_from_logits(X, y, w - h * e)  # pérdida con w[idx] - h
g_num = (L_plus - L_minus) / (2.0 * h)  # aproximación numérica del gradiente (central)

assert np.isclose(g[idx], g_num, rtol=1e-4, atol=1e-6)  # valida gradiente analítico vs numérico
```

---

### Ejercicio 5.6: Umbral (threshold) y trade-off precision/recall

#### Enunciado

1) **Básico**

- Dadas probabilidades `p` y etiquetas `y`, construye predicciones con umbral `t`.

2) **Intermedio**

- Calcula precision/recall para `t=0.5` y `t=0.3`.

3) **Avanzado**

- Verifica que al bajar el umbral típicamente sube el recall (en el mismo dataset).

#### Solución

```python
import numpy as np  # arrays y operaciones vectorizadas

def predict_threshold(p: np.ndarray, t: float) -> np.ndarray:  # aplica umbral t a probabilidades
    return (np.asarray(p) >= t).astype(int)  # boolean -> int (0/1)


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray):  # calcula precision/recall desde etiquetas
    y_true = np.asarray(y_true).astype(int)  # normaliza etiquetas reales
    y_pred = np.asarray(y_pred).astype(int)  # normaliza etiquetas predichas
    tp = np.sum((y_true == 1) & (y_pred == 1))  # true positives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # false positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # false negatives
    eps = 1e-12  # estabilizador numérico
    prec = tp / (tp + fp + eps)  # precision = TP/(TP+FP)
    rec = tp / (tp + fn + eps)  # recall = TP/(TP+FN)
    return float(prec), float(rec)  # retorna métricas


np.random.seed(3)  # semilla del demo (no estrictamente necesaria aquí)
y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # etiquetas reales
p = np.array([0.9, 0.6, 0.55, 0.52, 0.4, 0.35, 0.2, 0.1])  # probabilidades predichas

pred_05 = predict_threshold(p, 0.5)  # predicciones con umbral 0.5
pred_03 = predict_threshold(p, 0.3)  # predicciones con umbral 0.3

prec05, rec05 = precision_recall(y_true, pred_05)  # métricas con t=0.5
prec03, rec03 = precision_recall(y_true, pred_03)  # métricas con t=0.3

assert rec03 >= rec05  # al bajar umbral suele subir recall (capturas más positivos)
```

---

### Ejercicio 5.7: Regularización L2 (Ridge) y norma de pesos

#### Enunciado

1) **Básico**

- Implementa Ridge Regression: `(X^T X + λI) w = X^T y`.

2) **Intermedio**

- Compara `||w_ridge||` contra `||w_ols||`.

3) **Avanzado**

- Verifica que para `λ>0`, típicamente `||w_ridge|| <= ||w_ols||`.

#### Solución

```python
import numpy as np  # arrays y álgebra lineal

np.random.seed(4)  # fija semilla para reproducibilidad
n, d = 300, 5  # número de muestras y dimensión
X = np.random.randn(n, d)  # features gaussianas
w_true = np.array([2.0, -1.0, 0.5, 0.0, 3.0])  # pesos "verdaderos" para generar y
y = X @ w_true + 0.2 * np.random.randn(n)  # targets con ruido

XtX = X.T @ X  # matriz X^T X
Xty = X.T @ y  # vector X^T y
w_ols = np.linalg.solve(XtX, Xty)  # solución OLS (lambda=0)

lam = 10.0  # fuerza de regularización L2
w_ridge = np.linalg.solve(XtX + lam * np.eye(d), Xty)  # solución ridge: (XtX + λI)w = Xty

assert np.linalg.norm(w_ridge) <= np.linalg.norm(w_ols) + 1e-8  # ridge tiende a reducir la norma de w
```

---

### Ejercicio 5.8: Train/Test split reproducible (semilla)

#### Enunciado

1) **Básico**

- Implementa `train_test_split(X,y,test_size,seed)`.

2) **Intermedio**

- Verifica que con la misma semilla el split es idéntico.

3) **Avanzado**

- Verifica que no se pierden muestras y que shapes son correctos.

#### Solución

```python
import numpy as np  # arrays y RNG para generar datos y permutar índices

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 0):  # split reproducible
    X = np.asarray(X)  # asegura ndarray para indexación
    y = np.asarray(y)  # asegura ndarray para indexación
    n = X.shape[0]  # número de muestras
    rng = np.random.default_rng(seed)  # RNG reproducible
    idx = np.arange(n)  # índices 0..n-1
    rng.shuffle(idx)  # permuta índices in-place
    n_test = int(round(n * test_size))  # tamaño del test (redondeo)
    test_idx = idx[:n_test]  # índices del split de test
    train_idx = idx[n_test:]  # índices del split de train
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]  # retorna (X_train, X_test, y_train, y_test)


np.random.seed(0)  # fija semilla global para reproducibilidad del demo
X = np.random.randn(100, 2)  # dataset sintético (100,2)
y = (np.random.rand(100) < 0.5).astype(int)  # etiquetas binarias aleatorias

Xtr1, Xte1, ytr1, yte1 = train_test_split(X, y, test_size=0.25, seed=42)  # primer split
Xtr2, Xte2, ytr2, yte2 = train_test_split(X, y, test_size=0.25, seed=42)  # segundo split (misma seed)

assert np.allclose(Xtr1, Xtr2)  # verifica reproducibilidad en train
assert np.allclose(Xte1, Xte2)  # verifica reproducibilidad en test
assert np.all(ytr1 == ytr2)  # verifica reproducibilidad de y_train
assert np.all(yte1 == yte2)  # verifica reproducibilidad de y_test
assert Xtr1.shape[0] + Xte1.shape[0] == 100  # verifica que no se pierden muestras
```

---

### Ejercicio 5.9: K-Fold cross-validation (partición correcta)

#### Enunciado

1) **Básico**

- Implementa un generador de folds (índices train/val).

2) **Intermedio**

- Verifica que cada índice aparece exactamente una vez en validación.

3) **Avanzado**

- Verifica que train/val no se solapan.

#### Solución

```python
import numpy as np  # arrays y RNG para particiones

def kfold_indices(n: int, k: int, seed: int = 0):  # genera índices (train, val) para K-fold
    rng = np.random.default_rng(seed)  # RNG reproducible
    idx = np.arange(n)  # índices 0..n-1
    rng.shuffle(idx)  # permuta índices
    folds = np.array_split(idx, k)  # divide en k folds (tamaños casi iguales)
    for i in range(k):  # itera folds como validación
        val_idx = folds[i]  # fold i como validación
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])  # resto como train
        yield train_idx, val_idx  # retorna par (train_idx, val_idx)


n = 23  # número de muestras
k = 5  # número de folds
seen = np.zeros(n, dtype=int)  # contador de apariciones en validación
for tr, va in kfold_indices(n, k, seed=123):  # recorre folds
    assert len(np.intersect1d(tr, va)) == 0  # train/val no se solapan
    seen[va] += 1  # marca que estos índices aparecieron en validación
assert np.all(seen == 1)  # cada índice aparece exactamente una vez en validación
```

---

### Ejercicio 5.10: Árboles - Gini e Information Gain (split 1D)

#### Enunciado

1) **Básico**

- Implementa impurity Gini para etiquetas binarias.

2) **Intermedio**

- Para un feature 1D y un umbral `t`, computa el Information Gain.

3) **Avanzado**

- Encuentra el mejor umbral entre varios candidatos y verifica el resultado.

#### Solución

```python
import numpy as np  # arrays y operaciones numéricas

def gini(y: np.ndarray) -> float:  # impurity Gini para etiquetas binarias
    y = np.asarray(y).astype(int)  # normaliza a ndarray de enteros
    if y.size == 0:  # caso borde: sin muestras
        return 0.0  # por convención, impurity nula
    p1 = np.mean(y == 1)  # proporción de la clase 1
    p0 = 1.0 - p1  # proporción de la clase 0
    return float(1.0 - (p0**2 + p1**2))  # Gini = 1 - Σ_k p_k^2


def info_gain_gini(x: np.ndarray, y: np.ndarray, t: float) -> float:  # information gain usando Gini como impurity
    x = np.asarray(x, dtype=float)  # normaliza feature 1D
    y = np.asarray(y, dtype=int)  # normaliza etiquetas
    parent = gini(y)  # impurity del nodo padre
    left = y[x <= t]  # etiquetas que caen a la izquierda del umbral
    right = y[x > t]  # etiquetas que caen a la derecha del umbral
    w_left = left.size / y.size  # peso (fracción) del hijo izquierdo
    w_right = right.size / y.size  # peso (fracción) del hijo derecho
    child = w_left * gini(left) + w_right * gini(right)  # impurity ponderada de hijos
    return float(parent - child)  # gain = impurity_padre - impurity_hijos


x = np.array([0.1, 0.2, 0.25, 0.8, 0.85, 0.9])  # feature 1D de juguete
y = np.array([0, 0, 0, 1, 1, 1])  # etiquetas binarias asociadas

candidates = [0.2, 0.25, 0.8]  # umbrales candidatos a evaluar
gains = [info_gain_gini(x, y, t) for t in candidates]  # ganancia por cada umbral
best_t = candidates[int(np.argmax(gains))]  # selecciona umbral con mayor gain

assert best_t in [0.25, 0.8]  # valida que el mejor umbral separa las clases
assert max(gains) > 0.0  # valida que hay ganancia positiva
```

---

### (Bonus) Ejercicio 5.11: Shadow Mode - comparar contra solución cerrada en mini-dataset

#### Enunciado

- Entrena regresión lineal por GD y compara predicción con solución cerrada en un conjunto pequeño.

#### Solución

```python
import numpy as np  # arrays y operaciones numéricas

np.random.seed(5)  # fija semilla para reproducibilidad
n, d = 30, 2  # número de muestras y dimensión
X = np.random.randn(n, d)  # features sintéticas
w_true = np.array([1.2, -0.4])  # pesos "verdaderos" para generar y
y = X @ w_true + 0.01 * np.random.randn(n)  # targets con ruido pequeño

w_ne = np.linalg.solve(X.T @ X, X.T @ y)  # solución cerrada (normal equation)

w = np.zeros(d)  # inicializa pesos para GD
alpha = 0.1  # learning rate
for _ in range(2000):  # iteraciones de GD
    grad = (X.T @ (X @ w - y)) / n  # gradiente MSE: (1/n) Xᵀ(Xw - y)
    w = w - alpha * grad  # actualización: w <- w - alpha * grad

y_ne = X @ w_ne  # predicción usando solución cerrada
y_gd = X @ w  # predicción usando pesos de GD

assert np.mean((y_ne - y_gd) ** 2) < 1e-4  # verifica que GD aproxima la solución cerrada
```


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

import numpy as np  # Importa NumPy: base para álgebra lineal, vectorización, RNG y operaciones usadas en regresión y métricas
from typing import Tuple, List, Optional  # Importa tipos: anota firmas para claridad/herramientas (no afecta el runtime)
from dataclasses import dataclass  # Importa dataclass: permite definir contenedores de datos simples si se usan en el módulo


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:  # Sigmoide: transforma logits reales a probabilidades (0,1), usada en regresión logística
    z = np.clip(z, -500, 500)  # Estabilidad numérica: evita overflow en exp() con valores muy grandes en magnitud
    return 1 / (1 + np.exp(-z))  # σ(z)=1/(1+e^{-z}): cálculo vectorizado para arrays

def add_bias(X: np.ndarray) -> np.ndarray:  # Añade bias: inserta columna de 1s para modelar intercepto como theta[0]
    return np.column_stack([np.ones(len(X)), X])  # Construye X_b (n,d+1): concatena 1s con features para usar X_b @ theta


# ============================================================
# REGRESIÓN LINEAL
# ============================================================

class LinearRegression:  # Regresión lineal: aprende parámetros θ para predecir y≈Xθ (incluye intercepto via bias)
    def __init__(self):  # Inicializa el modelo: prepara parámetros y un historial opcional de coste
        self.theta = None  # Parámetros del modelo (d+1,): se aprenden en fit y se usan en predict
        self.cost_history = []  # Historial de MSE por iteración: útil si se entrena con descenso de gradiente

    def fit(self, X: np.ndarray, y: np.ndarray,  # Ajusta θ: elige solución cerrada (normal) o iterativa (GD)
            method: str = 'normal', lr: float = 0.01, n_iter: int = 1000):  # Hiperparámetros: método, learning rate y número de iteraciones
        X_b = add_bias(X)  # Añade columna de 1s: permite aprender el intercepto en theta[0]

        if method == 'normal':  # Ecuación normal: obtiene θ óptimo para MSE sin iterar (mínimos cuadrados)
            self.theta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)  # Resuelve (XᵀX)θ=Xᵀy (más estable que invertir XᵀX)
        else:  # Descenso de gradiente batch: optimiza MSE iterativamente
            m, n = X_b.shape  # m=n muestras (normaliza gradiente), n=d+1 parámetros (incluye bias)
            self.theta = np.zeros(n)  # Inicializa θ: vector de ceros como punto de partida de GD
            for _ in range(n_iter):  # Itera n_iter pasos de GD: cada paso reduce (idealmente) el coste
                grad = (1/m) * X_b.T @ (X_b @ self.theta - y)  # Gradiente MSE: (1/m) Xᵀ(Xθ−y)
                self.theta -= lr * grad  # Actualización: θ ← θ − α·grad
                self.cost_history.append(np.mean((X_b @ self.theta - y)**2))  # Guarda MSE actual: permite trazar convergencia
        return self  # Devuelve self: permite chaining estilo sklearn (model.fit(...).predict(...))

    def predict(self, X: np.ndarray) -> np.ndarray:  # Predice valores continuos: y_hat = X_b θ
        return add_bias(X) @ self.theta  # Multiplicación matriz-vector: produce predicciones (n,)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # R²: mide varianza explicada por el modelo (1 es perfecto)
        y_pred = self.predict(X)  # Predicción sobre X: base para medir error vs y
        ss_res = np.sum((y - y_pred)**2)  # SSE/SS_res: suma de cuadrados residual (error no explicado)
        ss_tot = np.sum((y - np.mean(y))**2)  # SST/SS_tot: suma de cuadrados total (variación respecto a la media)
        return 1 - ss_res / ss_tot  # R² = 1 - SSE/SST: compara el modelo contra baseline de predecir la media


# ============================================================
# REGRESIÓN LOGÍSTICA
# ============================================================

class LogisticRegression:  # Regresión logística binaria: modela P(y=1|x)=σ(X_b θ) y aprende θ por descenso de gradiente
    def __init__(self, reg: str = None, lambda_: float = 0.01):  # Inicializa configuración de regularización (opcional)
        self.reg = reg  # Tipo de regularización: None, 'l1' o 'l2' (define penalización aplicada a los pesos)
        self.lambda_ = lambda_  # Fuerza de regularización λ: escala el término de penalización en el gradiente
        self.theta = None  # Parámetros del modelo (d+1,): incluye bias; se inicializan en fit
        self.cost_history = []  # Historial de coste (si se registra): útil para debug/convergencia (aquí no se llena)

    def fit(self, X: np.ndarray, y: np.ndarray,  # Entrena el clasificador: optimiza cross-entropy implícita vía GD
            lr: float = 0.1, n_iter: int = 1000):  # Hiperparámetros: learning rate y número de iteraciones
        X_b = add_bias(X)  # Añade bias: permite intercepto (theta[0]) en el modelo lineal antes de la sigmoide
        m, n = X_b.shape  # Extrae shapes: m=n muestras (normaliza gradiente), n=d+1 parámetros (incluye bias)
        self.theta = np.zeros(n)  # Inicializa parámetros: vector de ceros como punto de partida del descenso de gradiente

        for _ in range(n_iter):  # Itera GD batch: actualiza θ para reducir el error de clasificación
            h = sigmoid(X_b @ self.theta)  # Probabilidades predichas: aplica σ a los logits X_bθ
            grad = (1/m) * X_b.T @ (h - y)  # Gradiente base: deriva de log-loss para logistic regression sin regularización

            if self.reg == 'l2':  # L2 (ridge): penaliza magnitudes cuadráticas, favorece pesos pequeños (shrinkage)
                grad[1:] += (self.lambda_/m) * self.theta[1:]  # No regulariza bias (índice 0): solo aplica a pesos de features
            elif self.reg == 'l1':  # L1 (lasso): penaliza norma L1, induce sparsity (pesos exactamente 0)
                grad[1:] += (self.lambda_/m) * np.sign(self.theta[1:])  # Subgradiente L1: signo por componente (maneja no diferenciabilidad en 0)

            self.theta -= lr * grad  # Actualización de GD: θ ← θ − α·grad
        return self  # Devuelve self: estilo sklearn para encadenar llamadas

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # Devuelve probabilidad de clase positiva (y=1) para cada muestra
        return sigmoid(add_bias(X) @ self.theta)  # Calcula σ(X_bθ): salida en (0,1) interpretada como probabilidad

    def predict(self, X: np.ndarray) -> np.ndarray:  # Devuelve etiqueta binaria {0,1} aplicando un umbral a predict_proba
        return (self.predict_proba(X) >= 0.5).astype(int)  # Umbral 0.5: clasifica como 1 si prob>=0.5; castea booleanos a int

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # Accuracy: proporción de aciertos (métrica simple de clasificación)
        return np.mean(self.predict(X) == y)  # Promedio de comparaciones: True=1, False=0 => accuracy en [0,1]


# ============================================================
# MÉTRICAS
# ============================================================

def accuracy(y_true, y_pred):  # Accuracy: proporción de aciertos (clasificación) = mean(y_true == y_pred)
    return np.mean(y_true == y_pred)  # Calcula promedio de booleanos: True=1, False=0 => valor en [0,1]

def precision(y_true, y_pred):  # Precision: TP/(TP+FP), qué tan confiables son los positivos predichos
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives: positivos reales predichos como positivos
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives: negativos reales predichos como positivos
    return tp / (tp + fp) if (tp + fp) > 0 else 0  # Evita división por cero: si no hay predicciones positivas, precision=0

def recall(y_true, y_pred):  # Recall: TP/(TP+FN), cobertura de positivos reales detectados
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives: positivos reales capturados por el modelo
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives: positivos reales que el modelo se perdió
    return tp / (tp + fn) if (tp + fn) > 0 else 0  # Evita división por cero: si no hay positivos reales, recall=0

def f1_score(y_true, y_pred):  # F1: media armónica de precision y recall, balancea ambos cuando hay desbalance de clases
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)  # Calcula precision y recall: ingredientes del F1
    return 2*p*r/(p+r) if (p+r) > 0 else 0  # Evita división por cero: si p+r=0, no hay señal => F1=0

def confusion_matrix(y_true, y_pred):  # Matriz 2x2: filas=clase real, columnas=clase predicha (útil para analizar errores)
    cm = np.zeros((2, 2), dtype=int)  # Inicializa conteos: [[TN, FP],[FN, TP]] con dtype int
    cm[0, 0] = np.sum((y_true == 0) & (y_pred == 0))  # TN: negativos reales predichos como negativos
    cm[0, 1] = np.sum((y_true == 0) & (y_pred == 1))  # FP: negativos reales predichos como positivos
    cm[1, 0] = np.sum((y_true == 1) & (y_pred == 0))  # FN: positivos reales predichos como negativos
    cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))  # TP: positivos reales predichos como positivos
    return cm  # Devuelve matriz: permite derivar precision/recall y ver tradeoffs por tipo de error


# ============================================================
# VALIDACIÓN
# ============================================================

def train_test_split(X, y, test_size=0.2, seed=None):  # Divide en train/test: baraja índices y separa según proporción `test_size`
    if seed: np.random.seed(seed)  # Si hay semilla, fija RNG global: hace reproducible la partición
    n = len(y)  # Número de muestras: define el rango de índices a permutar
    idx = np.random.permutation(n)  # Permutación aleatoria de índices: mezcla el dataset antes de partir
    split = int(n * test_size)  # Tamaño del test: cantidad de índices que irán al conjunto de evaluación
    return X[idx[split:]], X[idx[:split]], y[idx[split:]], y[idx[:split]]  # Retorna X_train, X_test, y_train, y_test (según índices permutados)

def cross_validate(model_class, X, y, k=5, **params):  # K-fold CV: evalúa el modelo entrenando k veces y promediando el score
    n = len(y)  # Número de muestras: base para construir folds
    idx = np.random.permutation(n)  # Baraja índices: evita sesgo por orden de los datos
    fold_size = n // k  # Tamaño del fold: división entera (si sobra, algunos ejemplos pueden quedar fuera)
    scores = []  # Scores por fold: se agregan para calcular media y desviación

    for i in range(k):  # Itera folds: en cada i, un segmento es validación y el resto es entrenamiento
        val_idx = idx[i*fold_size:(i+1)*fold_size]  # Índices de validación: slice del fold i
        train_idx = np.concatenate([idx[:i*fold_size], idx[(i+1)*fold_size:]])  # Índices de train: concatena todos los demás folds

        model = model_class()  # Instancia modelo nuevo por fold: evita reutilizar estado/θ entre entrenamientos
        model.fit(X[train_idx], y[train_idx], **params)  # Entrena en el subset de train: pasa hiperparámetros por **params
        scores.append(model.score(X[val_idx], y[val_idx]))  # Evalúa en validación: guarda score para estadística final

    return {'scores': scores, 'mean': np.mean(scores), 'std': np.std(scores)}  # Resume CV: lista de scores + media + desviación estándar


# ============================================================
# TESTS
# ============================================================

if __name__ == "__main__":  # Entry point: ejecuta pruebas rápidas cuando se corre como script
    np.random.seed(42)  # Fija semilla global: hace reproducibles los datos sintéticos

    # Test Linear Regression
    X = 2 * np.random.rand(100, 1)  # Features: 100 valores uniformes en [0,2) con 1 feature (shape (100,1))
    y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5  # Target: relación lineal y=4+3x + ruido gaussiano (simula datos reales)

    lr = LinearRegression()  # Instancia modelo de regresión lineal
    lr.fit(X, y)  # Ajusta θ con el método por defecto (ecuación normal)
    print(f"Linear Regression R²: {lr.score(X, y):.4f}")  # Imprime R²: sanity check de que el ajuste explica bien la varianza

    # Test Logistic Regression
    X_c0 = np.random.randn(50, 2) + [-2, -2]  # Clase 0: 50 puntos alrededor de (-2,-2) (distribución normal)
    X_c1 = np.random.randn(50, 2) + [2, 2]  # Clase 1: 50 puntos alrededor de (2,2)
    X_clf = np.vstack([X_c0, X_c1])  # Junta features: (100,2) concatenando ambas nubes
    y_clf = np.array([0]*50 + [1]*50)  # Labels: 0 para clase 0 y 1 para clase 1 (shape (100,))

    log_reg = LogisticRegression()  # Instancia regresión logística (sin regularización por defecto)
    log_reg.fit(X_clf, y_clf)  # Entrena θ por GD sobre el dataset sintético
    print(f"Logistic Regression Accuracy: {log_reg.score(X_clf, y_clf):.4f}")  # Imprime accuracy: verifica que clasifica razonablemente

    # Test metrics
    y_true = np.array([0,0,0,1,1,1,1,1])  # Labels reales: ejemplo pequeño para comprobar métricas manualmente
    y_pred = np.array([0,0,1,1,1,0,1,1])  # Predicciones: incluye FP y FN para que precision/recall no sean triviales
    print(f"Precision: {precision(y_true, y_pred):.4f}")  # Calcula e imprime precision del ejemplo
    print(f"Recall: {recall(y_true, y_pred):.4f}")  # Calcula e imprime recall del ejemplo
    print(f"F1: {f1_score(y_true, y_pred):.4f}")  # Calcula e imprime F1 del ejemplo

    # Test CV
    cv = cross_validate(LogisticRegression, X_clf, y_clf, k=5, lr=0.1, n_iter=500)  # Ejecuta CV 5-fold pasando hiperparámetros al fit
    print(f"CV Score: {cv['mean']:.4f} ± {cv['std']:.4f}")  # Reporta media±std: estima estabilidad del rendimiento

    print("\n✓ Todos los tests pasaron!")  # Mensaje final: indica que se ejecutaron pruebas sin errores
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
gradient = (1/n) * X.T @ (y_pred - y_true)  # gradiente vectorizado: (1/n) Xᵀ(ŷ - y)
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
import numpy as np  # Importa NumPy: se usa para zeros, norma L2, abs y operaciones numéricas en la comparación
from sklearn.linear_model import LogisticRegression as SklearnLR  # Importa LogisticRegression de sklearn como referencia (“ground truth”)
from sklearn.linear_model import LinearRegression as SklearnLinReg  # Importa LinearRegression de sklearn para validar MSE contra tu implementación
from sklearn.metrics import accuracy_score, mean_squared_error  # Importa métricas: accuracy y MSE para comparar resultados

# Importar tu implementación
# from src.logistic_regression import LogisticRegression as MyLR
# from src.linear_regression import LinearRegression as MyLinReg


def shadow_mode_logistic_regression(X_train, y_train, X_test, y_test):  # Valida tu Logistic Regression contra sklearn y detecta desviaciones
    """
    Compara tu Logistic Regression vs sklearn.

    Los coeficientes y accuracy deben ser casi idénticos.
    """
    print("=" * 60)  # Separador visual: hace más legible la salida en consola
    print("SHADOW MODE: Logistic Regression")  # Título del bloque: indica qué algoritmo se está validando
    print("=" * 60)  # Separador visual: cierra el encabezado

    # ========== TU IMPLEMENTACIÓN ==========
    # my_model = MyLR()
    # my_model.fit(X_train, y_train, lr=0.1, n_iter=1000)
    # my_pred = my_model.predict(X_test)
    # my_acc = accuracy_score(y_test, my_pred)
    # my_weights = my_model.weights

    # Placeholder (reemplazar con tu código)
    my_acc = 0.85  # Placeholder: accuracy dummy (reemplazar por accuracy real de tu modelo)
    my_weights = np.zeros(X_train.shape[1])  # Placeholder: vector de pesos dummy con dimensión d (reemplazar por pesos reales)

    # ========== SKLEARN (GROUND TRUTH) ==========
    sklearn_model = SklearnLR(max_iter=1000, solver='lbfgs')  # Crea modelo sklearn: baseline estable para comparar (mismo tipo de modelo)
    sklearn_model.fit(X_train, y_train)  # Entrena sklearn en el mismo train: produce coeficientes y predicciones de referencia
    sklearn_pred = sklearn_model.predict(X_test)  # Predice en test: etiquetas binarias para medir accuracy
    sklearn_acc = accuracy_score(y_test, sklearn_pred)  # Accuracy de referencia: se compara contra tu accuracy
    sklearn_weights = sklearn_model.coef_.flatten()  # Extrae pesos: coef_ tiene shape (1,d) en binario; flatten -> (d,)

    # ========== COMPARACIÓN ==========
    acc_diff = abs(my_acc - sklearn_acc)  # Diferencia absoluta de accuracy: métrica simple para el veredicto
    weight_diff = np.linalg.norm(my_weights - sklearn_weights[:len(my_weights)])  # Diferencia L2 de pesos: norma de la resta (cuanto menor, más parecido)

    print(f"\n📊 RESULTADOS:")  # Encabezado de resultados numéricos
    print(f"  Tu Accuracy:     {my_acc:.4f}")  # Muestra tu accuracy (debería venir de tu implementación real)
    print(f"  sklearn Accuracy: {sklearn_acc:.4f}")  # Muestra accuracy sklearn (baseline)
    print(f"  Diferencia:       {acc_diff:.4f}")  # Reporta diferencia: si es grande, hay sospecha de bug

    print(f"\n📐 PESOS:")  # Encabezado de comparación de parámetros
    print(f"  Diferencia L2 de pesos: {weight_diff:.4f}")  # Norma L2: cuantifica cuánto difieren los coeficientes

    # Veredicto
    print("\n" + "-" * 60)  # Separador antes del veredicto: delimita la sección final
    if acc_diff < 0.05:  # Criterio: diferencia <5% se considera aceptable (según la regla indicada)
        print("✓ PASSED: Tu implementación es correcta")  # Mensaje OK: tu modelo se alinea razonablemente con sklearn
        return True  # Devuelve True: útil para automatizar checks en pipelines/QA
    else:  # Caso problemático: la diferencia excede el umbral y se sugiere revisar
        print("✗ FAILED: Diferencia significativa - revisa tu matemática")  # Mensaje FAIL: indica que hay discrepancia relevante
        print("  Posibles causas:")  # Lista de hipótesis típicas que causan divergencia
        print("  - Gradiente mal calculado")  # Error común: derivada/gradiente incorrecto en la función de coste
        print("  - Learning rate muy alto/bajo")  # Hiperparámetro: puede causar no convergencia o convergencia lenta
        print("  - Falta de normalización de datos")  # Preprocesamiento: escalas distintas pueden dificultar el entrenamiento
        return False  # Devuelve False: permite fallar la validación automáticamente


def shadow_mode_linear_regression(X_train, y_train, X_test, y_test):  # Valida tu Linear Regression contra sklearn comparando MSE
    """
    Compara tu Linear Regression vs sklearn.
    """
    print("=" * 60)  # Separador visual: salida más legible
    print("SHADOW MODE: Linear Regression")  # Título del bloque: indica validación de regresión lineal
    print("=" * 60)  # Separador visual

    # ========== TU IMPLEMENTACIÓN ==========
    # my_model = MyLinReg()
    # my_model.fit(X_train, y_train)
    # my_pred = my_model.predict(X_test)
    # my_mse = mean_squared_error(y_test, my_pred)

    # Placeholder
    my_mse = 0.5  # Placeholder: MSE dummy (reemplazar por MSE real de tu implementación)

    # ========== SKLEARN ==========
    sklearn_model = SklearnLinReg()  # Crea baseline sklearn: regresión lineal por mínimos cuadrados
    sklearn_model.fit(X_train, y_train)  # Ajusta baseline: obtiene coeficientes óptimos en closed-form
    sklearn_pred = sklearn_model.predict(X_test)  # Predice en test: valores continuos
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)  # MSE sklearn: referencia para comparar tu MSE

    # ========== COMPARACIÓN ==========
    mse_ratio = my_mse / sklearn_mse if sklearn_mse > 0 else float('inf')  # Ratio de errores: >1 implica peor que sklearn; evita dividir por 0

    print(f"\n📊 RESULTADOS:")  # Encabezado de resultados
    print(f"  Tu MSE:     {my_mse:.4f}")  # Muestra MSE de tu implementación (placeholder en este snippet)
    print(f"  sklearn MSE: {sklearn_mse:.4f}")  # Muestra MSE del baseline
    print(f"  Ratio:       {mse_ratio:.2f}x")  # Reporta cuántas veces es tu error respecto al baseline

    print("\n" + "-" * 60)  # Separador antes del veredicto
    if mse_ratio < 1.1:  # Dentro del 10%: tolerancia razonable por diferencias numéricas/implementación
        print("✓ PASSED: Tu implementación es correcta")  # OK: tu MSE está alineado con el baseline
        return True  # Devuelve True: validación aprobada
    else:  # Caso fallo: tu MSE es mucho peor que sklearn
        print("✗ FAILED: Tu MSE es significativamente mayor")  # FAIL: sugiere bug o mala convergencia
        return False  # Devuelve False: validación fallida


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":  # entrypoint: ejecuta la demo solo si se corre como script
    # Ejecuta la demo de validación: compara tu implementación con sklearn
    from sklearn.datasets import make_classification, make_regression  # Importa generadores: crean datasets sintéticos para demo/pruebas
    from sklearn.model_selection import train_test_split  # Importa split sklearn: genera particiones consistentes para la demo

    # Dataset de clasificación
    X_clf, y_clf = make_classification(  # Crea dataset binario: features y labels para probar logistic regression
        n_samples=1000, n_features=10, n_classes=2, random_state=42  # Parámetros: tamaño, dimensionalidad, nº clases y semilla
    )  # Devuelve X (n,d) e y (n,) para clasificación
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(  # Divide dataset de clasificación: train/test para validar sin leakage
        X_clf, y_clf, test_size=0.2, random_state=42  # 20% test: semilla para reproducibilidad
    )  # Retorna splits: X_train/X_test/y_train/y_test

    # Dataset de regresión
    X_reg, y_reg = make_regression(  # Crea dataset de regresión: targets continuos con ruido
        n_samples=1000, n_features=10, noise=10, random_state=42  # Parámetros: tamaño, features, nivel de ruido y semilla
    )  # Devuelve X (n,d) e y (n,) para regresión
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(  # Divide dataset de regresión: entrenamiento y evaluación
        X_reg, y_reg, test_size=0.2, random_state=42  # 20% test y semilla fija para reproducibilidad
    )  # Retorna splits para regresión

    # Ejecutar Shadow Mode
    shadow_mode_logistic_regression(X_train_c, y_train_c, X_test_c, y_test_c)  # Ejecuta comparación logistic: devuelve True/False según el umbral
    print("\n")  # Línea en blanco: separa salidas entre validación de clasificación y regresión
    shadow_mode_linear_regression(X_train_r, y_train_r, X_test_r, y_test_r)  # Ejecuta comparación lineal: chequea ratio de MSE vs sklearn
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
