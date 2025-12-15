# Módulo 04: Probabilidad Esencial para Machine Learning

> **Semana 8 | Prerequisito para entender Loss Functions y GMM**
> **Filosofía: Solo la probabilidad que necesitas para la Línea 1**

---

## 🎯 Objetivo del Módulo

Dominar los **conceptos mínimos de probabilidad** necesarios para:

1. Entender **Logistic Regression** como modelo probabilístico
2. Comprender **Cross-Entropy Loss** y por qué funciona
3. Prepararte para **Gaussian Mixture Models (GMM)** en Unsupervised
4. Entender **Softmax** como distribución de probabilidad

> ⚠️ **Nota:** Este NO es el curso completo de Probabilidad (Línea 2). Es solo lo esencial para ML.

---

<a id="m04-0"></a>

## 🧭 Cómo usar este módulo (modo 0→100)

**Propósito:** conectar probabilidad con lo que realmente usarás en el Pathway:

- pérdidas (cross-entropy) como *negative log-likelihood*
- clasificación probabilística (logistic/softmax)
- gaussianas como base de modelos generativos (GMM)
- estabilidad numérica (evitar `NaN`)

### Objetivos de aprendizaje (medibles)

Al terminar el módulo podrás:

- **Explicar** `P(A|B)` y el teorema de Bayes con un ejemplo de clasificación.
- **Aplicar** el punto de vista de MLE: “elegir parámetros que hacen los datos más probables”.
- **Derivar** por qué minimizar cross-entropy equivale a maximizar log-likelihood (binaria y multiclase).
- **Implementar** softmax y log-softmax de forma numéricamente estable (log-sum-exp).
- **Diagnosticar** fallos típicos: `log(0)`, overflow/underflow, probabilidades que no suman 1.

### Prerrequisitos

- De `Módulo 01`: NumPy (vectorización, `axis`, broadcasting).
- De `Módulo 03`: Chain Rule y gradiente (para entender el salto a `Módulo 05/07`).

Enlaces rápidos:

- [RECURSOS.md](RECURSOS.md)
- [GLOSARIO: Binary Cross-Entropy](GLOSARIO.md#binary-cross-entropy)
- [GLOSARIO: Softmax](GLOSARIO.md#softmax)
- [GLOSARIO: Chain Rule](GLOSARIO.md#chain-rule)

### Integración con Plan v4/v5

- [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
- [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)
- Registro de errores: `study_tools/DIARIO_ERRORES.md`
- Evaluación (rúbrica): [study_tools/RUBRICA_v1.md](../study_tools/RUBRICA_v1.md) (scope `M04` en `rubrica.csv`; incluye PB-8)

### Recursos (cuándo usarlos)

| Prioridad | Recurso | Cuándo usarlo en este módulo | Para qué |
|----------|---------|------------------------------|----------|
| **Obligatorio** | `study_tools/DIARIO_ERRORES.md` | Cada vez que aparezca `NaN`, `inf`, `log(0)` u overflow/underflow | Registrar el caso y crear un “fix” reproducible |
| **Obligatorio** | [StatQuest - Maximum Likelihood](https://www.youtube.com/watch?v=XepXtl9YKwc) | Antes (o durante) la sección de MLE y cross-entropy | Alinear intuición de “maximizar verosimilitud” |
| **Complementario** | [3Blue1Brown - Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) | Cuando Bayes se sienta “fórmula sin sentido” (día 3-4) | Visualizar prior/likelihood/posterior |
| **Complementario** | [Mathematics for ML (book)](https://mml-book.github.io/) | Al implementar Gaussiana multivariada y covarianza | Refuerzo de notación y derivaciones |
| **Opcional** | [RECURSOS.md](RECURSOS.md) | Al terminar el módulo (para planificar Línea 2 o profundizar) | Elegir rutas de estudio sin romper el foco de Línea 1 |

### Mapa conceptual (qué conecta con qué)

- **MLE → Cross-Entropy:** sustenta Logistic Regression (Módulo 05) y BCE/CCE en Deep Learning (Módulo 07).
- **Gaussiana multivariada:** es el “átomo” de GMM (Módulo 06).
- **Softmax + Log-Sum-Exp:** evita inestabilidad numérica en clasificación multiclase (Módulo 05/07).

---

## 📚 Contenido

### Día 1-2: Fundamentos de Probabilidad

#### 1.1 Probabilidad Básica

```text
P(A) = casos favorables / casos totales

Propiedades:
- 0 ≤ P(A) ≤ 1
- P(Ω) = 1 (espacio muestral)
- P(∅) = 0 (evento imposible)
```

#### 1.2 Probabilidad Condicional

```text
P(A|B) = P(A ∩ B) / P(B)

"Probabilidad de A dado que B ocurrió"
```

**Ejemplo en ML:**
- P(spam | contiene "gratis") = ¿Qué tan probable es spam si el email dice "gratis"?

#### 1.3 Independencia

```text
A y B son independientes si:
P(A ∩ B) = P(A) · P(B)

Equivalente a:
P(A|B) = P(A)
```

---

### Día 3-4: Teorema de Bayes (Crítico para ML)

#### 2.1 La Fórmula

```text
            P(B|A) · P(A)
P(A|B) = ─────────────────
               P(B)

Donde:
- P(A|B) = Posterior (lo que queremos calcular)
- P(B|A) = Likelihood (verosimilitud)
- P(A)   = Prior (conocimiento previo)
- P(B)   = Evidence (normalizador)
```

#### 2.2 Interpretación para ML

```text
              P(datos|clase) · P(clase)
P(clase|datos) = ─────────────────────────
                      P(datos)

Ejemplo: Clasificación de spam
- P(spam|palabras) = P(palabras|spam) · P(spam) / P(palabras)
```

#### 2.3 Implementación en Python

```python
import numpy as np

def bayes_classifier(x: np.ndarray,
                     likelihood_spam: float,
                     likelihood_ham: float,
                     prior_spam: float = 0.3) -> str:
    """
    Clasificador Bayesiano simple.

    Args:
        x: Características del email (simplificado)
        likelihood_spam: P(x|spam)
        likelihood_ham: P(x|ham)
        prior_spam: P(spam) - conocimiento previo

    Returns:
        'spam' o 'ham'
    """
    prior_ham = 1 - prior_spam

    # Posterior (sin normalizar, solo comparamos)
    posterior_spam = likelihood_spam * prior_spam
    posterior_ham = likelihood_ham * prior_ham

    return 'spam' if posterior_spam > posterior_ham else 'ham'


# Ejemplo: Email con palabra "gratis"
# P("gratis"|spam) = 0.8, P("gratis"|ham) = 0.1
result = bayes_classifier(
    x=None,  # simplificado
    likelihood_spam=0.8,
    likelihood_ham=0.1,
    prior_spam=0.3
)
print(f"Clasificación: {result}")  # spam
```

#### 2.4 Naive Bayes (Conexión con Supervised Learning)

```python
def naive_bayes_predict(X: np.ndarray,
                        class_priors: np.ndarray,
                        feature_probs: dict) -> np.ndarray:
    """
    Naive Bayes asume independencia entre features:
    P(x1, x2, ..., xn | clase) = P(x1|clase) · P(x2|clase) · ... · P(xn|clase)

    Esta "ingenuidad" simplifica mucho el cálculo.
    """
    n_samples = X.shape[0]
    n_classes = len(class_priors)

    log_posteriors = np.zeros((n_samples, n_classes))

    for c in range(n_classes):
        # Log para evitar underflow con muchas features
        log_prior = np.log(class_priors[c])
        log_likelihood = np.sum(np.log(feature_probs[c][X]), axis=1)
        log_posteriors[:, c] = log_prior + log_likelihood

    return np.argmax(log_posteriors, axis=1)
```

---

## 🧩 Micro-Capítulo Maestro: Maximum Likelihood Estimation (MLE) — Nivel: Avanzado

### 1) Intuición (la metáfora del detective)

Imagina que eres un detective que llega a una escena del crimen (tus **datos** `X`).

- Tienes una lista de sospechosos (tus **modelos**).
- Cada sospechoso tiene un comportamiento ajustable por perillas (tus **parámetros** `θ`).

MLE pregunta:

> **¿Qué valores de `θ` hacen MÁS PROBABLE que estos datos específicos hayan ocurrido?**

Importante:

- No estamos diciendo “qué parámetro es más probable” (eso sería un enfoque Bayesiano).
- Estamos diciendo “qué parámetro le da la mayor probabilidad a los datos que YA vimos”.

### 2) Formalización (likelihood y log-likelihood)

Sea `X = {x1, x2, ..., xn}` un conjunto de datos i.i.d.

La **likelihood** es:

`L(θ | X) = P(X | θ) = Π_{i=1}^{n} P(x_i | θ)`

Como multiplicar muchos números pequeños causa underflow, usamos log:

`ℓ(θ) = log L(θ|X) = Σ_{i=1}^{n} log P(x_i | θ)`

Como `log` es monótona creciente, maximizar `L` y maximizar `ℓ` es equivalente:

`θ_MLE = argmax_θ ℓ(θ)`

### 3) Derivación clave: de MLE a MSE (Regresión Lineal)

La idea conceptual: cuando usas **MSE**, estás asumiendo implícitamente un modelo de ruido.

Supón que tu regresión lineal es:

`y = Xβ + ε` con `ε ~ N(0, σ² I)`

Entonces la probabilidad de observar `y` dado `β` es Gaussiana:

`P(y | X, β) ∝ exp( - (1/(2σ²)) ||y - Xβ||² )`

Tomando log-likelihood y tirando constantes que no dependen de `β`:

`ℓ(β) = const - (1/(2σ²)) ||y - Xβ||²`

Maximizar `ℓ(β)` equivale a minimizar `||y - Xβ||²`.

Conclusión:

- Minimizar **SSE/MSE** es exactamente hacer **MLE** bajo ruido Gaussiano.
- Esta conexión es el puente directo hacia **Statistical Estimation** (Línea 2).

### 4) Conexión Línea 2: estimadores, sesgo y varianza (intuición)

En Línea 2, la palabra clave es **estimador**: una regla que convierte datos en un parámetro.

- Un **estimador** es una función: `\hat{θ} = g(X)`.
- **Sesgo (bias):** si `E[\hat{θ}]` no coincide con el valor real `θ`.
- **Varianza:** cuánto cambia `\hat{θ}` si repites el muestreo.

Regla mental:

- **Más bias** suele dar **menos varianza**.
- **Menos bias** suele dar **más varianza**.

Esto reaparece en ML como *bias-variance tradeoff*.

### 5) Teoría de Estimadores (lo que te evalúan en proyectos/examen)

Aquí pasamos de la intuición a una formalización que aparece mucho en evaluación.

#### 5.1 Sesgo, varianza y MSE (descomposición clave)

Si quieres estimar un parámetro real `θ` con un estimador `\hat{θ}`, el error cuadrático medio es:

`MSE(\hat{θ}) = E[(\hat{θ} - θ)^2]`

La identidad importante es:

`MSE(\hat{θ}) = Var(\hat{θ}) + Bias(\hat{θ})^2`

Donde:

- `Bias(\hat{θ}) = E[\hat{θ}] - θ`
- `Var(\hat{θ}) = E[(\hat{θ} - E[\hat{θ}])^2]`

Lectura mental:

- Puedes reducir MSE bajando varianza, aunque suba un poco el sesgo.
- O puedes “perseguir cero sesgo” y pagar con alta varianza.

Esto es exactamente el *bias-variance trade-off* en ML (por ejemplo, regularizar o simplificar modelos).

#### 5.2 Unbiased vs consistente (2 propiedades distintas)

- **Unbiased (insesgado):** `E[\hat{θ}] = θ`.
- **Consistente:** cuando `n → ∞`, `\hat{θ} → θ` (en un sentido probabilístico).

Un estimador puede ser sesgado y aun así consistente (y a veces es preferible si reduce varianza para `n` finito).

#### 5.3 Conexión directa con regularización (puente a ML)

Ejemplo mental:

- **Ridge / L2** introduce sesgo (empuja coeficientes hacia 0).
- A cambio suele reducir varianza (solución más estable ante ruido y colinealidad).

En términos de la descomposición:

- sube `Bias^2`
- baja `Var`

Si el total baja, mejora el `MSE` esperado fuera de muestra.

## 🧩 Micro-Capítulo Maestro: Introducción a Markov Chains — Nivel: Intermedio

### 1) Concepto

Una cadena de Markov es un sistema que salta entre estados.

Propiedad de Markov (“falta de memoria”):

`P(S_{t+1} | S_t, S_{t-1}, ...) = P(S_{t+1} | S_t)`

### 2) Representación matricial (puente con Álgebra Lineal)

Si tienes 3 estados (Sol, Nube, Lluvia), defines una matriz de transición `P` (3×3) donde cada fila suma 1.

Si `π_t` es un vector fila (1×3) con la distribución “hoy”, entonces:

`π_{t+1} = π_t P`

Y en `k` pasos:

`π_{t+k} = π_t P^k`

### 3) Reto mental: estacionariedad = eigenvector

Si repites multiplicaciones, muchas cadenas convergen a una distribución estacionaria `π*` tal que:

`π* = π* P`

Eso significa (en la perspectiva correcta) que `π*` es un **eigenvector** asociado al **eigenvalue 1**.

---

### Día 5: Distribución Gaussiana (Normal)

#### 3.1 La Distribución Más Importante en ML

```text
                    1              (x - μ)²
f(x) = ───────────────── · exp(- ─────────)
       σ · √(2π)                   2σ²

Parámetros:
- μ (mu): Media (centro de la campana)
- σ (sigma): Desviación estándar (ancho)
- σ² (sigma²): Varianza
```

#### 3.2 Por Qué es Importante

1. **Muchos fenómenos naturales** siguen esta distribución
2. **Teorema del Límite Central:** promedios de cualquier distribución → Normal
3. **GMM usa Gaussianas** para modelar clusters
4. **Inicialización de pesos** en redes neuronales

#### 3.3 Implementación

```python
import numpy as np

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Probability Density Function de la Gaussiana.

    Args:
        x: Puntos donde evaluar
        mu: Media
        sigma: Desviación estándar

    Returns:
        Densidad de probabilidad en cada punto
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * np.exp(exponent)


# Visualización
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)

# Diferentes Gaussianas
plt.figure(figsize=(10, 6))
plt.plot(x, gaussian_pdf(x, mu=0, sigma=1), label='μ=0, σ=1 (estándar)')
plt.plot(x, gaussian_pdf(x, mu=0, sigma=2), label='μ=0, σ=2 (más ancha)')
plt.plot(x, gaussian_pdf(x, mu=2, sigma=1), label='μ=2, σ=1 (desplazada)')
plt.legend()
plt.title('Distribuciones Gaussianas')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.savefig('gaussian_distributions.png')
```

#### 3.4 Gaussiana Multivariada (Para GMM)

```python
def multivariate_gaussian_pdf(x: np.ndarray,
                               mu: np.ndarray,
                               cov: np.ndarray) -> float:
    """
    Gaussiana multivariada para vectores.

    Args:
        x: Vector de características (d,)
        mu: Vector de medias (d,)
        cov: Matriz de covarianza (d, d)

    Returns:
        Densidad de probabilidad
    """
    d = len(mu)
    diff = x - mu

    # Determinante e inversa de la covarianza
    det_cov = np.linalg.det(cov)
    inv_cov = np.linalg.inv(cov)

    # Coeficiente de normalización
    coefficient = 1 / (np.sqrt((2 * np.pi) ** d * det_cov))

    # Exponente (forma cuadrática)
    exponent = -0.5 * diff.T @ inv_cov @ diff

    return coefficient * np.exp(exponent)


# Ejemplo 2D
mu = np.array([0, 0])
cov = np.array([[1, 0.5],
                [0.5, 1]])  # Correlación positiva

x = np.array([0.5, 0.5])
prob = multivariate_gaussian_pdf(x, mu, cov)
print(f"P(x=[0.5, 0.5]) = {prob:.4f}")
```

---

### Día 6: Maximum Likelihood Estimation (MLE)

#### 4.0 MLE → Cross-Entropy (la conexión que te piden en exámenes)

**Idea:** si un modelo produce probabilidades `P(y|x, θ)`, entrenar por MLE significa:

- maximizar `Πᵢ P(yᵢ|xᵢ, θ)`

Por estabilidad numérica y conveniencia, trabajamos con log:

- maximizar `Σᵢ log P(yᵢ|xᵢ, θ)`

Y como optimizadores minimizan, entrenamos minimizando:

- `-Σᵢ log P(yᵢ|xᵢ, θ)`  (negative log-likelihood)

Ese término es exactamente la **cross-entropy** que usas en:

- Logistic Regression (BCE) en `Módulo 05`
- clasificación multiclase (CCE) en `Módulo 07`

**Cheat sheet:**

- **MLE:** maximizar likelihood
- **Entrenamiento:** minimizar negative log-likelihood
- **En clasificación:** eso se llama cross-entropy

---

### Extensión Estratégica (Línea 2): Statistical Estimation

#### MLE como filosofía: “ajustar perillas”

MLE no es solo una fórmula: es una forma de pensar.

- Tienes un modelo con parámetros `θ` (las “perillas”).
- Ya viste datos `D`.
- Pregunta: ¿qué valores de `θ` hacen que `D` sea lo más probable posible?

Formalmente:

```text
θ_MLE = argmax_θ P(D | θ)
```

Como `P(D|θ)` suele ser un producto grande, usamos log:

```text
θ_MLE = argmax_θ log P(D | θ)
```

Esto es el puente directo a **Statistical Estimation** (Línea 2): estimadores, sesgo, varianza, y por qué “promedio” aparece en tantos lados.

#### Worked example: Moneda (Bernoulli) → estimador MLE

Modelo:

- `X_i ~ Bernoulli(p)` donde `p = P(cara)`.

Datos:

- `D = {x_1, ..., x_n}` con `x_i ∈ {0,1}`.

Likelihood:

```text
P(D | p) = Π_i p^{x_i} (1-p)^{(1-x_i)}
```

Log-likelihood:

```text
ℓ(p) = Σ_i [x_i log p + (1-x_i) log(1-p)]
```

Derivar y hacer 0 (intuición: el máximo ocurre cuando la “probabilidad del modelo” coincide con la frecuencia observada):

```text
dℓ/dp = Σ_i [x_i/p - (1-x_i)/(1-p)] = 0
```

Solución:

```text
p_MLE = (1/n) Σ_i x_i
```

Interpretación: el MLE de `p` es simplemente la **proporción de caras**. Este patrón (media muestral) reaparece en gaussianas y en muchos estimadores.

#### 4.1 La Idea Central

```text
MLE: Encontrar los parámetros θ que maximizan la probabilidad
     de observar los datos que tenemos.

θ_MLE = argmax P(datos | θ)
            θ
```

#### 4.2 Por Qué es Fundamental

- **Logistic Regression** usa MLE para encontrar los pesos
- **Cross-Entropy Loss** viene de maximizar likelihood
- **GMM** usa MLE (via EM algorithm)

#### 4.3 MLE para Gaussiana

```python
def mle_gaussian(data: np.ndarray) -> tuple[float, float]:
    """
    Estimar parámetros de Gaussiana con MLE.

    Para una Gaussiana, los estimadores MLE son:
    - μ_MLE = media muestral
    - σ²_MLE = varianza muestral (con n, no n-1)

    Args:
        data: Muestras observadas

    Returns:
        (mu_mle, sigma_mle)
    """
    n = len(data)

    # MLE de la media
    mu_mle = np.mean(data)

    # MLE de la varianza (dividir por n, no n-1)
    sigma_squared_mle = np.sum((data - mu_mle) ** 2) / n
    sigma_mle = np.sqrt(sigma_squared_mle)

    return mu_mle, sigma_mle


# Ejemplo: Generar datos y estimar
np.random.seed(42)
true_mu, true_sigma = 5.0, 2.0
samples = np.random.normal(true_mu, true_sigma, size=1000)

estimated_mu, estimated_sigma = mle_gaussian(samples)
print(f"Parámetros reales: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimados:     μ={estimated_mu:.3f}, σ={estimated_sigma:.3f}")
```

#### 4.4 Conexión con Cross-Entropy Loss

#### 4.5 MLE para multiclase (Softmax + Categorical Cross-Entropy)

Para `K` clases, `y` es one-hot y el modelo produce probabilidades con softmax:

- `p = softmax(z)` donde `z = XW` son logits

Likelihood (por muestra):

- `P(y|x) = Π_k p_k^{y_k}`

Log-likelihood:

- `log P(y|x) = Σ_k y_k log(p_k)`

Negative log-likelihood promedio:

- `L = -(1/m) Σᵢ Σ_k y_{ik} log(p_{ik})`

Eso es exactamente **Categorical Cross-Entropy**.

```python
def cross_entropy_from_mle():
    """
    Demostración de que Cross-Entropy viene de MLE.

    Para clasificación binaria con Bernoulli:
    P(y|x, θ) = p^y · (1-p)^(1-y)

    Donde p = σ(θᵀx) (predicción del modelo)

    Log-likelihood:
    log P(y|x, θ) = y·log(p) + (1-y)·log(1-p)

    Maximizar likelihood = Minimizar negative log-likelihood
    = Minimizar Cross-Entropy!
    """
    # Ejemplo numérico
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.2])  # Probabilidades

    # Cross-Entropy (negative log-likelihood promedio)
    epsilon = 1e-15  # Para evitar log(0)
    ce = -np.mean(
        y_true * np.log(y_pred + epsilon) +
        (1 - y_true) * np.log(1 - y_pred + epsilon)
    )

    print(f"Cross-Entropy Loss: {ce:.4f}")
    return ce

cross_entropy_from_mle()
```

---

## 🌱 Extensión Estratégica (Línea 2): Markov Chains (intro conceptual)

> Esta sección es conceptual: no vas a implementar Markov Chains en Línea 1, pero sí necesitas que la idea te resulte familiar cuando entres al curso de **Discrete-Time Markov Chains and Monte Carlo Methods**.

### Idea central: estados y transiciones

Una cadena de Markov modela un sistema que “salta” entre **estados**.

- Hoy estás en un estado `S_t`.
- Mañana estás en `S_{t+1}`.
- Lo importante: `P(S_{t+1} | S_t)` depende solo del estado actual (memoria de 1 paso).

### Matriz de transición (conexión con Álgebra Lineal)

Definimos una matriz `P` donde:

- `P[i, j] = P(estado j | estado i)`
- Cada fila suma 1 (matriz estocástica por filas)

Si `π_t` es un vector fila con la distribución de probabilidad sobre estados en el tiempo `t`, entonces:

```text
π_{t+1} = π_t P
```

Esto conecta directamente con `Módulo 02`: es **multiplicación de matrices** aplicada a probabilidades.

### Ejemplo mínimo (2 estados)

Estados: `A` y `B`.

```text
P = [[0.9, 0.1],
     [0.2, 0.8]]
```

Interpretación:

- Si estás en `A`, te quedas en `A` con 0.9, pasas a `B` con 0.1.
- Si estás en `B`, pasas a `A` con 0.2, te quedas en `B` con 0.8.

### Estacionariedad (semilla para Línea 2)

Una distribución estacionaria `π*` satisface:

```text
π* = π* P
```

En otras palabras: es un **autovector** (eigenvector) asociado al eigenvalue `1` (visto desde la perspectiva correcta). Esto vuelve a conectar Markov Chains con eigenvalues/eigenvectors.

---

### Día 7: Softmax como Distribución de Probabilidad

#### 5.1 De Logits a Probabilidades

```text
                    exp(zᵢ)
softmax(z)ᵢ = ─────────────────
              Σⱼ exp(zⱼ)

Propiedades:
- Cada salida ∈ (0, 1)
- Suma de salidas = 1 (distribución válida)
- Preserva el orden (mayor logit → mayor probabilidad)
```

#### 5.2 El Problema de Estabilidad Numérica (v3.3)

```text
⚠️ PROBLEMA: exp() puede causar overflow/underflow

Ejemplo peligroso:
    z = [1000, 1001, 1002]
    exp(z) = [inf, inf, inf]  → NaN en softmax!

Ejemplo underflow:
    z = [-1000, -1001, -1002]
    exp(z) = [0, 0, 0]  → 0/0 = NaN!
```

#### 5.3 Log-Sum-Exp Trick (Estabilidad Numérica)

```text
TRUCO: softmax(z) = softmax(z - max(z))

Demostración:
    softmax(z - c)ᵢ = exp(zᵢ - c) / Σⱼ exp(zⱼ - c)
                    = exp(zᵢ)·exp(-c) / Σⱼ exp(zⱼ)·exp(-c)
                    = exp(zᵢ) / Σⱼ exp(zⱼ)
                    = softmax(z)ᵢ

Al restar max(z), todos los exponentes son ≤ 0, evitando overflow.
```

#### 5.4 Implementación Numéricamente Estable

> Regla práctica: si vas a calcular cross-entropy, prefiere **log-softmax** estable en vez de `np.log(softmax(z))`.

```python
def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax numéricamente estable usando Log-Sum-Exp trick.

    Truco: Restar el máximo para evitar overflow en exp()
    softmax(z) = softmax(z - max(z))

    Args:
        z: Logits (scores antes de activación)

    Returns:
        Probabilidades que suman 1
    """
    # Log-Sum-Exp trick: restar el máximo
    z_stable = z - np.max(z, axis=-1, keepdims=True)

    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """
    Log-Softmax estable (útil para Cross-Entropy).

    log(softmax(z)) calculado de forma estable.
    Evita calcular softmax primero y luego log (pierde precisión).
    """
    z_stable = z - np.max(z, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(z_stable), axis=-1, keepdims=True))
    return z_stable - log_sum_exp


def categorical_cross_entropy_from_logits(y_true: np.ndarray, logits: np.ndarray) -> float:
    """
    Cross-entropy estable usando logits directamente.

    Evita calcular softmax explícito.
    Útil cuando entrenas modelos y quieres estabilidad.
    """
    log_probs = log_softmax(logits)
    return -np.mean(np.sum(y_true * log_probs, axis=1))


# ============================================================
# DEMOSTRACIÓN: Por qué el trick es necesario
# ============================================================

def demo_numerical_stability():
    """Muestra por qué necesitamos el Log-Sum-Exp trick."""

    # Caso peligroso: logits muy grandes
    z_dangerous = np.array([1000.0, 1001.0, 1002.0])

    # Sin el trick (INCORRECTO)
    def softmax_naive(z):
        exp_z = np.exp(z)  # ¡Overflow!
        return exp_z / np.sum(exp_z)

    # Con el trick (CORRECTO)
    def softmax_stable(z):
        z_stable = z - np.max(z)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z)

    print("Logits peligrosos:", z_dangerous)
    print()

    # Naive (falla)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_naive = softmax_naive(z_dangerous)
        print(f"Softmax NAIVE: {result_naive}")
        print(f"  → Suma: {np.sum(result_naive)} (debería ser 1.0)")

    # Estable (funciona)
    result_stable = softmax_stable(z_dangerous)
    print(f"\nSoftmax ESTABLE: {result_stable}")
    print(f"  → Suma: {np.sum(result_stable):.6f} ✓")

demo_numerical_stability()


# Ejemplo: Clasificación multiclase (dígitos 0-9)
logits = np.array([2.0, 1.0, 0.1, -1.0, 3.0, 0.5, -0.5, 1.5, 0.0, -2.0])
probs = softmax(logits)

print("\nLogits → Probabilidades:")
for i, (l, p) in enumerate(zip(logits, probs)):
    print(f"  Clase {i}: logit={l:+.1f} → prob={p:.3f}")
print(f"\nSuma de probabilidades: {np.sum(probs):.6f}")
print(f"Clase predicha: {np.argmax(probs)}")
```

#### 5.3 Categorical Cross-Entropy (Multiclase)

```python
def categorical_cross_entropy(y_true: np.ndarray,
                               y_pred: np.ndarray) -> float:
    """
    Loss para clasificación multiclase.

    Args:
        y_true: One-hot encoded labels (n_samples, n_classes)
        y_pred: Probabilidades softmax (n_samples, n_classes)

    Returns:
        Loss promedio
    """
    epsilon = 1e-15
    # Solo cuenta la clase correcta (donde y_true=1)
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))


# Ejemplo
y_true = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Clase 4
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Clase 0
])

y_pred = np.array([
    softmax(np.array([0, 0, 0, 0, 5, 0, 0, 0, 0, 0])),  # Confiado en 4
    softmax(np.array([3, 1, 0, 0, 0, 0, 0, 0, 0, 0])),  # Confiado en 0
])

loss = categorical_cross_entropy(y_true, y_pred)
print(f"Categorical Cross-Entropy: {loss:.4f}")
```

---

## 🎯 Ejercicios por tema (progresivos) + Soluciones

Reglas:

- **Intenta primero** sin mirar la solución.
- **Timebox sugerido:** 15–30 min por ejercicio.
- **Éxito mínimo:** tu solución debe pasar los `assert`.

---

### Ejercicio 4.1: Probabilidad condicional (P(A|B)) y consistencia

#### Enunciado

1) **Básico**

- Dado un conjunto de conteos de eventos, calcula `P(A)`, `P(B)` y `P(A ∩ B)`.

2) **Intermedio**

- Calcula `P(A|B) = P(A∩B)/P(B)` y verifica que está en `[0,1]`.

3) **Avanzado**

- Verifica que `P(A∩B) = P(A|B)·P(B)`.

#### Solución

```python
import numpy as np

# Simulación con conteos (dataset pequeño)
n = 100
count_A = 40
count_B = 50
count_A_and_B = 20

P_A = count_A / n
P_B = count_B / n
P_A_and_B = count_A_and_B / n

P_A_given_B = P_A_and_B / P_B

assert 0.0 <= P_A <= 1.0
assert 0.0 <= P_B <= 1.0
assert 0.0 <= P_A_given_B <= 1.0
assert np.isclose(P_A_and_B, P_A_given_B * P_B)
```

---

### Ejercicio 4.2: Bayes en modo clasificador (posterior sin normalizar)

#### Enunciado

1) **Básico**

- Implementa el cálculo de posterior sin normalizar:
  - `score_spam = P(x|spam)·P(spam)`
  - `score_ham = P(x|ham)·P(ham)`

2) **Intermedio**

- Normaliza y obtén `P(spam|x)` y `P(ham|x)`.

3) **Avanzado**

- Verifica que las probabilidades normalizadas suman 1.

#### Solución

```python
import numpy as np

P_spam = 0.3
P_ham = 1.0 - P_spam

P_x_given_spam = 0.8
P_x_given_ham = 0.1

score_spam = P_x_given_spam * P_spam
score_ham = P_x_given_ham * P_ham

Z = score_spam + score_ham
P_spam_given_x = score_spam / Z
P_ham_given_x = score_ham / Z

assert np.isclose(P_spam_given_x + P_ham_given_x, 1.0)
assert P_spam_given_x > P_ham_given_x
```

---

### Ejercicio 4.3: Independencia (test empírico)

#### Enunciado

1) **Básico**

- Simula dos variables binarias independientes `A` y `B`.

2) **Intermedio**

- Estima `P(A)`, `P(B)`, `P(A∩B)` y verifica `P(A∩B) ≈ P(A)P(B)`.

3) **Avanzado**

- Simula un caso dependiente y verifica que la igualdad se rompe.

#### Solución

```python
import numpy as np

np.random.seed(0)
n = 20000

# Independientes
A = (np.random.rand(n) < 0.4)
B = (np.random.rand(n) < 0.5)

P_A = A.mean()
P_B = B.mean()
P_A_and_B = (A & B).mean()

assert abs(P_A_and_B - (P_A * P_B)) < 0.01

# Dependientes: B es casi A
B_dep = (A | (np.random.rand(n) < 0.05))
P_B_dep = B_dep.mean()
P_A_and_B_dep = (A & B_dep).mean()

assert abs(P_A_and_B_dep - (P_A * P_B_dep)) > 0.02
```

---

### Ejercicio 4.4: MLE de Bernoulli ("fracción de heads")

#### Enunciado

1) **Básico**

- Genera muestras Bernoulli con `p_true`.

2) **Intermedio**

- Implementa el estimador MLE `p_hat = mean(x)`.

3) **Avanzado**

- Verifica que `p_hat` se aproxima a `p_true` con suficientes muestras.

#### Solución

```python
import numpy as np

np.random.seed(1)
p_true = 0.7
n = 5000
x = (np.random.rand(n) < p_true).astype(float)

p_hat = float(np.mean(x))
assert abs(p_hat - p_true) < 0.02
```

---

### Ejercicio 4.5: PDF Gaussiana univariada (sanity check)

#### Enunciado

1) **Básico**

- Implementa la PDF de una normal `N(μ,σ²)`.

2) **Intermedio**

- Verifica que para `N(0,1)` en `x=0` la densidad ≈ `0.39894228`.

3) **Avanzado**

- Verifica que `pdf(x)` es simétrica: `pdf(a) == pdf(-a)` cuando `μ=0`.

#### Solución

```python
import numpy as np

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sigma = float(sigma)
    assert sigma > 0
    z = (x - mu) / sigma
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * z**2)


val0 = gaussian_pdf(np.array([0.0]), mu=0.0, sigma=1.0)[0]
assert np.isclose(val0, 0.39894228, atol=1e-4)

a = 1.7
assert np.isclose(
    gaussian_pdf(np.array([a]), 0.0, 1.0)[0],
    gaussian_pdf(np.array([-a]), 0.0, 1.0)[0],
    rtol=1e-12,
    atol=1e-12,
)
```

---

### Ejercicio 4.6: Gaussiana multivariada (2D) + covarianza válida

#### Enunciado

1) **Básico**

- Implementa la densidad `N(μ, Σ)` en 2D.

2) **Intermedio**

- Para `μ=0` y `Σ=I`, verifica que `pdf(0) = 1/(2π)`.

3) **Avanzado**

- Verifica que `Σ` es definida positiva (eigenvalores > 0) antes de invertir.

#### Solución

```python
import numpy as np

def multivariate_gaussian_pdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    d = x.shape[0]

    assert mu.shape == (d,)
    assert cov.shape == (d, d)
    assert np.allclose(cov, cov.T)
    eigvals = np.linalg.eigvals(cov)
    assert np.all(eigvals > 0)

    diff = x - mu
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    norm = 1.0 / (np.sqrt(((2.0 * np.pi) ** d) * det))
    expo = -0.5 * float(diff.T @ inv @ diff)
    return float(norm * np.exp(expo))


mu = np.array([0.0, 0.0])
cov = np.eye(2)
pdf0 = multivariate_gaussian_pdf(np.array([0.0, 0.0]), mu, cov)
assert np.isclose(pdf0, 1.0 / (2.0 * np.pi), atol=1e-6)
assert pdf0 > 0.0
```

---

### Ejercicio 4.7: Log-Sum-Exp y log-softmax estable

#### Enunciado

1) **Básico**

- Implementa `logsumexp(z)` de forma estable (restando `max(z)`).

2) **Intermedio**

- Implementa `log_softmax(z) = z - logsumexp(z)`.

3) **Avanzado**

- Verifica que `sum(exp(log_softmax(z))) == 1` y que no hay `inf` con logits grandes.

#### Solución

```python
import numpy as np

def logsumexp(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    m = np.max(z)
    return float(m + np.log(np.sum(np.exp(z - m))))


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return z - logsumexp(z)


z = np.array([1000.0, 0.0, -1000.0])
lsm = log_softmax(z)
probs = np.exp(lsm)
assert np.isfinite(lsm).all()
assert np.isfinite(probs).all()
assert np.isclose(np.sum(probs), 1.0)
```

---

### Ejercicio 4.8: Softmax estable (invariancia a constantes)

#### Enunciado

1) **Básico**

- Implementa softmax estable: `exp(z-max)/sum(exp(z-max))`.

2) **Intermedio**

- Verifica que suma 1.

3) **Avanzado**

- Verifica invariancia: `softmax(z) == softmax(z + c)`.

#### Solución

```python
import numpy as np

def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z_shift = z - np.max(z)
    expz = np.exp(z_shift)
    return expz / np.sum(expz)


z = np.array([2.0, 1.0, 0.0])
p = softmax(z)
assert np.isclose(np.sum(p), 1.0)

c = 100.0
p2 = softmax(z + c)
assert np.allclose(p, p2)
assert np.argmax(p) == np.argmax(z)
```

---

### Ejercicio 4.9: Binary Cross-Entropy estable (evitar log(0))

#### Enunciado

1) **Básico**

- Implementa BCE: `-mean(y log(p) + (1-y) log(1-p))`.

2) **Intermedio**

- Usa `clip`/`epsilon` para evitar `log(0)`.

3) **Avanzado**

- Verifica:
  - BCE cerca de 0 para predicciones casi perfectas.
  - BCE ≈ `-log(0.9)` cuando `y=1` y `p=0.9`.

#### Solución

```python
import numpy as np

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


y_true = np.array([1.0, 0.0, 1.0, 0.0])
y_pred_good = np.array([0.999, 0.001, 0.999, 0.001])
assert binary_cross_entropy(y_true, y_pred_good) < 0.01

assert np.isclose(binary_cross_entropy(np.array([1.0]), np.array([0.9])), -np.log(0.9), atol=1e-12)
```

---

### Ejercicio 4.10: Categorical Cross-Entropy (multiclase) + one-hot

#### Enunciado

1) **Básico**

- Implementa CCE: `-mean(sum(y_true * log(y_pred)))`.

2) **Intermedio**

- Asegura que `y_pred` no contiene ceros (epsilon).

3) **Avanzado**

- Verifica que el loss baja cuando aumenta la probabilidad de la clase correcta.

#### Solución

```python
import numpy as np

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, eps, 1.0)
    return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))


y_true = np.array([[0, 1, 0], [1, 0, 0]], dtype=float)
y_pred_bad = np.array([[0.34, 0.33, 0.33], [0.34, 0.33, 0.33]], dtype=float)
y_pred_good = np.array([[0.05, 0.90, 0.05], [0.90, 0.05, 0.05]], dtype=float)

loss_bad = categorical_cross_entropy(y_true, y_pred_bad)
loss_good = categorical_cross_entropy(y_true, y_pred_good)
assert loss_good < loss_bad
```

---

### (Bonus) Ejercicio 4.11: Cadena de Markov (matriz de transición)

#### Enunciado

1) **Básico**

- Define una matriz de transición `P` (filas suman 1).

2) **Intermedio**

- Propaga una distribución `π_{t+1} = π_t P` y verifica que sigue siendo distribución.

3) **Avanzado**

- Encuentra una distribución estacionaria aproximada iterando muchas veces y verifica `π ≈ πP`.

#### Solución

```python
import numpy as np

P = np.array([
    [0.9, 0.1],
    [0.2, 0.8],
], dtype=float)
assert np.allclose(P.sum(axis=1), 1.0)

pi = np.array([1.0, 0.0])
for _ in range(50):
    pi = pi @ P
    assert np.isclose(np.sum(pi), 1.0)
    assert np.all(pi >= 0)

pi_star = pi.copy()
assert np.allclose(pi_star, pi_star @ P, atol=1e-6)
```

## 🔨 Entregables del Módulo

### E1: `probability.py`

```python
"""
Módulo de probabilidad esencial para ML.
Implementaciones desde cero con NumPy.
"""

import numpy as np
from typing import Tuple

def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Densidad de probabilidad Gaussiana univariada."""
    pass

def multivariate_gaussian_pdf(x: np.ndarray,
                               mu: np.ndarray,
                               cov: np.ndarray) -> float:
    """Densidad de probabilidad Gaussiana multivariada."""
    pass

def mle_gaussian(data: np.ndarray) -> Tuple[float, float]:
    """Estimación MLE de parámetros de Gaussiana."""
    pass

def softmax(z: np.ndarray) -> np.ndarray:
    """Función softmax numéricamente estable."""
    pass

def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy loss."""
    pass

def categorical_cross_entropy(y_true: np.ndarray,
                               y_pred: np.ndarray) -> float:
    """Categorical cross-entropy loss para multiclase."""
    pass
```

### E2: Tests

```python
# tests/test_probability.py
import numpy as np
import pytest
from src.probability import (
    gaussian_pdf, mle_gaussian, softmax,
    cross_entropy, categorical_cross_entropy
)

def test_gaussian_pdf_standard():
    """PDF de Gaussiana estándar en x=0 debe ser ~0.3989."""
    result = gaussian_pdf(np.array([0.0]), mu=0, sigma=1)
    expected = 1 / np.sqrt(2 * np.pi)  # ~0.3989
    assert np.isclose(result[0], expected, rtol=1e-5)

def test_softmax_sums_to_one():
    """Softmax debe sumar 1."""
    z = np.random.randn(10)
    probs = softmax(z)
    assert np.isclose(np.sum(probs), 1.0)

def test_softmax_preserves_order():
    """Mayor logit → mayor probabilidad."""
    z = np.array([1.0, 2.0, 3.0])
    probs = softmax(z)
    assert probs[2] > probs[1] > probs[0]

def test_mle_gaussian_accuracy():
    """MLE debe recuperar parámetros con suficientes datos."""
    np.random.seed(42)
    true_mu, true_sigma = 10.0, 3.0
    data = np.random.normal(true_mu, true_sigma, size=10000)

    est_mu, est_sigma = mle_gaussian(data)

    assert np.isclose(est_mu, true_mu, rtol=0.05)
    assert np.isclose(est_sigma, true_sigma, rtol=0.05)

def test_cross_entropy_perfect_prediction():
    """CE debe ser ~0 para predicciones perfectas."""
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.999, 0.001, 0.999])

    loss = cross_entropy(y_true, y_pred)
    assert loss < 0.01
```

---

## 📊 Resumen Visual

```text
┌─────────────────────────────────────────────────────────────────┐
│  PROBABILIDAD PARA ML - MAPA CONCEPTUAL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TEOREMA DE BAYES                                               │
│       │                                                         │
│       ├──► Naive Bayes Classifier (Módulo 05)                   │
│       └──► Intuición de posterior vs prior                      │
│                                                                 │
│  DISTRIBUCIÓN GAUSSIANA                                         │
│       │                                                         │
│       ├──► GMM en Unsupervised (Módulo 06)                      │
│       ├──► Inicialización de pesos en DL (Módulo 07)            │
│       └──► Normalización de datos                               │
│                                                                 │
│  MAXIMUM LIKELIHOOD (MLE)                                       │
│       │                                                         │
│       ├──► Cross-Entropy Loss (Logistic Regression)             │
│       ├──► Categorical CE (Softmax + Multiclase)                │
│       └──► EM Algorithm en GMM                                  │
│                                                                 │
│  SOFTMAX                                                        │
│       │                                                         │
│       └──► Capa de salida en clasificación multiclase           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔗 Conexiones con Otros Módulos

| Concepto | Dónde se usa |
|----------|--------------|
| Teorema de Bayes | Naive Bayes en Módulo 05 |
| Gaussiana | GMM en Módulo 06, inicialización en Módulo 07 |
| MLE | Derivación de Cross-Entropy en Módulo 05 |
| Softmax | Capa de salida en Módulo 07 |
| Cross-Entropy | Loss function principal en Módulo 05 y 07 |

---

## 🧩 Consolidación (errores comunes + debugging v5 + reto Feynman)

### Errores comunes

- **Confundir PDF con probabilidad:** en continuas, `f(x)` es densidad; la probabilidad requiere integrar en un intervalo.
- **`log(0)` en cross-entropy:** siempre usa `epsilon` o `np.clip`.
- **Overflow/underflow en `exp`:** aplica log-sum-exp / log-softmax.
- **MLE “mágico”:** si no puedes explicar por qué aparece la media, repite el worked example Bernoulli.

### Debugging / validación (v5)

- Cuando algo explote con `nan/inf`, revisa:
  - `np.log` sobre valores 0
  - `np.exp` sobre logits grandes
  - normalización incorrecta en probabilidades (que no suman 1)
- Registra hallazgos en `study_tools/DIARIO_ERRORES.md`.
- Protocolos completos:
  - [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)
  - [PLAN_V5_ESTRATEGICO.md](PLAN_V5_ESTRATEGICO.md)

### Reto Feynman (tablero blanco)

Explica en 5 líneas o menos:

1) ¿Por qué maximizar likelihood es equivalente a minimizar negative log-likelihood?
2) ¿Por qué el MLE de una moneda es “proporción de caras”?
3) ¿Qué significa `π_{t+1} = π_t P` y por qué es álgebra lineal?

## ✅ Checklist del Módulo

- [ ] Puedo explicar el Teorema de Bayes con un ejemplo
- [ ] Sé calcular la PDF de una Gaussiana a mano
- [ ] Entiendo por qué MLE da Cross-Entropy como loss
- [ ] Implementé softmax numéricamente estable
- [ ] Puedo derivar el MLE de una Bernoulli (moneda) y explicarlo
- [ ] Puedo explicar qué es una Markov Chain y qué representa una matriz de transición
- [ ] Los tests de `probability.py` pasan

---

## 📖 Recursos Adicionales

### Videos
- [3Blue1Brown - Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM)
- [StatQuest - Maximum Likelihood](https://www.youtube.com/watch?v=XepXtl9YKwc)
- [StatQuest - Gaussian Distribution](https://www.youtube.com/watch?v=rzFX5NWojp0)

### Lecturas
- Mathematics for ML, Cap. 6 (Probability)
- Pattern Recognition and ML (Bishop), Cap. 1-2

---

> 💡 **Nota Final:** Este módulo sigue siendo compacto comparado con un curso completo de probabilidad/estadística, pero aquí ya tienes el núcleo de Línea 1 y una “semilla” intencional para Línea 2 (estimación y Markov Chains).

---

**[← Módulo 03: Cálculo](03_CALCULO_MULTIVARIANTE.md)** | **[Módulo 05: Supervised Learning →](05_SUPERVISED_LEARNING.md)**
