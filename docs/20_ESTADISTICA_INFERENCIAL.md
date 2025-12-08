# Módulo 05 - Estadística Inferencial para IA

> **🎯 Objetivo:** Dominar estimación, pruebas de hipótesis e inferencia  
> **⭐ PATHWAY LÍNEA 2:** Statistical Estimation for Data Science and AI

---

## 🧠 Analogía: Inferir la Población desde la Muestra

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   INFERENCIA ESTADÍSTICA                                                    │
│   ──────────────────────                                                    │
│                                                                             │
│   POBLACIÓN (desconocida) ────────────────────────────────────────────────  │
│   • Parámetros verdaderos: μ, σ², θ                                         │
│   • Imposible medir todos los individuos                                    │
│                                                                             │
│               ↓ Muestreo                                                    │
│                                                                             │
│   MUESTRA (observada) ───────────────────────────────────────────────────   │
│   • Estadísticos: x̄, s², θ̂                                                  │
│   • n observaciones                                                         │
│                                                                             │
│               ↓ Inferencia                                                  │
│                                                                             │
│   ESTIMACIÓN ────────────────────────────────────────────────────────────   │
│   • Punto: θ̂ ≈ θ                                                            │
│   • Intervalo: [θ̂ - error, θ̂ + error] contiene θ con 95% confianza          │
│                                                                             │
│   APLICACIÓN EN ML:                                                         │
│   • Train set = muestra                                                     │
│   • Performance en test = estimación de performance real                    │
│   • Cross-validation = reducir varianza de la estimación                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Estimación Puntual](#1-estimacion-puntual)
2. [Maximum Likelihood Estimation (MLE)](#2-mle)
3. [Maximum A Posteriori (MAP)](#3-map)
4. [Intervalos de Confianza](#4-intervalos)
5. [Pruebas de Hipótesis](#5-hipotesis)
6. [Regresión Estadística](#6-regresion)

---

## 1. Estimación Puntual {#1-estimacion-puntual}

### 1.1 Propiedades de Buenos Estimadores

```
PROPIEDADES DESEABLES:
────────────────────────

1. INSESGADO (Unbiased):
   E[θ̂] = θ
   El estimador acierta "en promedio"

2. CONSISTENTE:
   θ̂ → θ cuando n → ∞
   Mejora con más datos

3. EFICIENTE:
   Mínima varianza entre estimadores insesgados
   Menor incertidumbre

SESGO-VARIANZA TRADE-OFF (crucial para ML):
• Sesgo alto → underfitting (modelo muy simple)
• Varianza alta → overfitting (modelo muy complejo)
• Objetivo: minimizar error total = sesgo² + varianza
```

### 1.2 Estimadores Comunes

```python
from typing import List
import math

def sample_mean(data: List[float]) -> float:
    """Unbiased estimator of population mean.
    
    E[x̄] = μ
    """
    return sum(data) / len(data)


def sample_variance_unbiased(data: List[float]) -> float:
    """Unbiased estimator of population variance.
    
    Uses n-1 (Bessel's correction) for unbiasedness.
    E[s²] = σ²
    """
    n = len(data)
    mean = sample_mean(data)
    return sum((x - mean) ** 2 for x in data) / (n - 1)


def sample_variance_mle(data: List[float]) -> float:
    """MLE estimator of variance (biased but consistent).
    
    Uses n instead of n-1.
    """
    n = len(data)
    mean = sample_mean(data)
    return sum((x - mean) ** 2 for x in data) / n


def standard_error(data: List[float]) -> float:
    """Standard error of the mean: SE = s / √n.
    
    Measures uncertainty in our estimate of the mean.
    """
    return math.sqrt(sample_variance_unbiased(data) / len(data))
```

---

## 2. Maximum Likelihood Estimation (MLE) {#2-mle}

### 2.1 Concepto

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   MAXIMUM LIKELIHOOD ESTIMATION                                             │
│   ─────────────────────────────                                             │
│                                                                             │
│   Pregunta: ¿Qué parámetros θ hacen MÁS PROBABLE los datos observados?      │
│                                                                             │
│   Likelihood: L(θ|data) = P(data|θ)                                         │
│                                                                             │
│   MLE: θ̂_MLE = argmax L(θ|data)                                             │
│                    θ                                                        │
│                                                                             │
│   Práctica: maximizar log-likelihood (más estable numéricamente):           │
│   θ̂_MLE = argmax log L(θ|data)                                              │
│                    θ                                                        │
│                                                                             │
│   EJEMPLO - Moneda sesgada:                                                 │
│   Datos: 7 caras en 10 lanzamientos                                         │
│   L(p) = C(10,7) × p^7 × (1-p)^3                                            │
│   MLE: p̂ = 7/10 = 0.7                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 MLE para Distribuciones Comunes

```python
def mle_bernoulli(successes: int, trials: int) -> float:
    """MLE for Bernoulli parameter p.
    
    p̂_MLE = number of successes / number of trials
    
    Example:
        >>> mle_bernoulli(7, 10)
        0.7
    """
    return successes / trials


def mle_normal_mean(data: List[float]) -> float:
    """MLE for normal distribution mean.
    
    μ̂_MLE = sample mean
    """
    return sample_mean(data)


def mle_normal_variance(data: List[float]) -> float:
    """MLE for normal distribution variance.
    
    Note: This is BIASED (uses n, not n-1).
    σ̂²_MLE = (1/n) Σ(xᵢ - x̄)²
    """
    return sample_variance_mle(data)


def mle_poisson(data: List[int]) -> float:
    """MLE for Poisson rate parameter λ.
    
    λ̂_MLE = sample mean
    """
    return sum(data) / len(data)
```

### 2.3 MLE con Gradient Descent (Logistic Regression)

```python
def sigmoid(z: float) -> float:
    """Logistic sigmoid function."""
    if z < -500:  # Prevent overflow
        return 0.0
    elif z > 500:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def log_likelihood_logistic(
    X: List[List[float]], 
    y: List[int], 
    weights: List[float]
) -> float:
    """Log-likelihood for logistic regression.
    
    ℓ(w) = Σ [yᵢ log(σ(wᵀxᵢ)) + (1-yᵢ) log(1-σ(wᵀxᵢ))]
    """
    ll = 0.0
    for xi, yi in zip(X, y):
        z = sum(w * x for w, x in zip(weights, xi))
        p = sigmoid(z)
        # Avoid log(0)
        p = max(min(p, 1 - 1e-15), 1e-15)
        ll += yi * math.log(p) + (1 - yi) * math.log(1 - p)
    return ll


def gradient_log_likelihood(
    X: List[List[float]], 
    y: List[int], 
    weights: List[float]
) -> List[float]:
    """Gradient of log-likelihood for logistic regression.
    
    ∂ℓ/∂wⱼ = Σ (yᵢ - σ(wᵀxᵢ)) × xᵢⱼ
    """
    n_features = len(weights)
    gradient = [0.0] * n_features
    
    for xi, yi in zip(X, y):
        z = sum(w * x for w, x in zip(weights, xi))
        p = sigmoid(z)
        error = yi - p
        for j in range(n_features):
            gradient[j] += error * xi[j]
    
    return gradient
```

---

## 3. Maximum A Posteriori (MAP) {#3-map}

### 3.1 Concepto: MLE + Prior

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   MAXIMUM A POSTERIORI (MAP)                                                │
│   ──────────────────────────                                                │
│                                                                             │
│   MLE: Solo usa datos                                                       │
│   θ̂_MLE = argmax P(data|θ)                                                  │
│                                                                             │
│   MAP: Incorpora conocimiento previo (prior)                                │
│   θ̂_MAP = argmax P(θ|data) = argmax P(data|θ) × P(θ)                        │
│                                                                             │
│   Usando Bayes:                                                             │
│   P(θ|data) ∝ P(data|θ) × P(θ)                                              │
│   posterior ∝ likelihood × prior                                            │
│                                                                             │
│   RELACIÓN CON REGULARIZACIÓN:                                              │
│   • Prior Gaussiano → L2 regularization (Ridge)                             │
│   • Prior Laplaciano → L1 regularization (Lasso)                            │
│                                                                             │
│   ¿CUÁNDO USAR MAP vs MLE?                                                  │
│   • Datos abundantes: MLE ≈ MAP (prior se vuelve irrelevante)               │
│   • Datos escasos: MAP más estable (prior regulariza)                       │
│   • Conocimiento previo: MAP permite incorporarlo                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Ejemplo: MAP con Prior Gaussiano

```python
def map_with_gaussian_prior(
    data: List[float], 
    prior_mean: float, 
    prior_variance: float,
    likelihood_variance: float
) -> float:
    """MAP estimate for normal mean with Gaussian prior.
    
    Conjugate prior: Normal prior + Normal likelihood = Normal posterior
    
    θ̂_MAP = (n/σ² × x̄ + 1/τ² × μ₀) / (n/σ² + 1/τ²)
    
    where:
    - x̄: sample mean
    - n: sample size
    - σ²: likelihood variance
    - μ₀: prior mean
    - τ²: prior variance
    """
    n = len(data)
    sample_mean_val = sample_mean(data)
    
    precision_likelihood = n / likelihood_variance
    precision_prior = 1 / prior_variance
    
    numerator = precision_likelihood * sample_mean_val + precision_prior * prior_mean
    denominator = precision_likelihood + precision_prior
    
    return numerator / denominator
```

---

## 4. Intervalos de Confianza {#4-intervalos}

### 4.1 Concepto

```
INTERVALO DE CONFIANZA:
───────────────────────

Un intervalo [a, b] tal que:
P(a ≤ θ ≤ b) = 1 - α

Para 95% de confianza: α = 0.05

INTERPRETACIÓN CORRECTA:
Si repitiéramos el experimento muchas veces,
95% de los intervalos construidos contendrían θ.

INTERPRETACIÓN INCORRECTA:
"Hay 95% de probabilidad de que θ esté en [a,b]"
(θ es fijo, no aleatorio en estadística frecuentista)
```

### 4.2 Intervalo para la Media

```python
def confidence_interval_mean(
    data: List[float], 
    confidence: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for population mean.
    
    Assumes large sample (n > 30) using Normal approximation.
    For small samples, use t-distribution.
    
    CI = x̄ ± z* × (s / √n)
    
    Example:
        >>> data = [23, 25, 27, 29, 31]
        >>> confidence_interval_mean(data, 0.95)
        (23.5..., 30.5...)
    """
    n = len(data)
    mean = sample_mean(data)
    se = standard_error(data)
    
    # z* values for common confidence levels
    z_values = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    z = z_values.get(confidence, 1.96)
    
    margin = z * se
    return (mean - margin, mean + margin)


def confidence_interval_proportion(
    successes: int, 
    trials: int, 
    confidence: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for population proportion.
    
    Uses normal approximation (valid when np > 5 and n(1-p) > 5).
    
    CI = p̂ ± z* × √(p̂(1-p̂)/n)
    """
    p_hat = successes / trials
    se = math.sqrt(p_hat * (1 - p_hat) / trials)
    
    z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_values.get(confidence, 1.96)
    
    margin = z * se
    return (p_hat - margin, p_hat + margin)
```

---

## 5. Pruebas de Hipótesis {#5-hipotesis}

### 5.1 Framework

```
ESTRUCTURA DE UNA PRUEBA:
─────────────────────────

1. HIPÓTESIS NULA (H₀): Lo que asumimos es verdad
   "No hay efecto" / "No hay diferencia"

2. HIPÓTESIS ALTERNATIVA (H₁): Lo que queremos probar
   "Hay efecto" / "Hay diferencia"

3. ESTADÍSTICO DE PRUEBA: Resume los datos

4. P-VALUE: P(observar datos tan extremos | H₀ es verdad)
   p < α → Rechazar H₀

5. DECISIÓN:
   • p < 0.05 → "Estadísticamente significativo"
   • p ≥ 0.05 → "No hay evidencia suficiente"

TIPOS DE ERROR:
• Tipo I (α): Rechazar H₀ cuando es verdadera (falso positivo)
• Tipo II (β): No rechazar H₀ cuando es falsa (falso negativo)
• Power = 1 - β: Probabilidad de detectar efecto real
```

### 5.2 Z-Test para la Media

```python
def z_test_one_sample(
    data: List[float], 
    population_mean: float, 
    population_std: float,
    alternative: str = "two-sided"
) -> tuple[float, float]:
    """One-sample Z-test for population mean.
    
    H₀: μ = μ₀
    H₁: μ ≠ μ₀ (two-sided) / μ > μ₀ (greater) / μ < μ₀ (less)
    
    Returns:
        z_statistic, p_value
    """
    n = len(data)
    x_bar = sample_mean(data)
    
    z = (x_bar - population_mean) / (population_std / math.sqrt(n))
    
    # Calculate p-value using standard normal CDF approximation
    if alternative == "two-sided":
        p_value = 2 * (1 - standard_normal_cdf_approx(abs(z)))
    elif alternative == "greater":
        p_value = 1 - standard_normal_cdf_approx(z)
    else:  # less
        p_value = standard_normal_cdf_approx(z)
    
    return z, p_value


def standard_normal_cdf_approx(x: float) -> float:
    """Approximation of standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
```

### 5.3 T-Test (Muestras Pequeñas)

```python
def t_test_two_sample(
    group1: List[float], 
    group2: List[float]
) -> tuple[float, float]:
    """Two-sample t-test (Welch's t-test).
    
    Tests if two groups have different means.
    Does not assume equal variances.
    
    H₀: μ₁ = μ₂
    H₁: μ₁ ≠ μ₂
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = sample_mean(group1), sample_mean(group2)
    var1 = sample_variance_unbiased(group1)
    var2 = sample_variance_unbiased(group2)
    
    # Welch's t-statistic
    se = math.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / se
    
    # Welch-Satterthwaite degrees of freedom
    num = (var1/n1 + var2/n2) ** 2
    denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    df = num / denom
    
    # Approximate p-value (would need t-distribution for exact)
    # For large df, t approaches normal
    p_value = 2 * (1 - standard_normal_cdf_approx(abs(t_stat)))
    
    return t_stat, p_value
```

### 5.4 Chi-Square Test (Datos Categóricos)

```python
def chi_square_test(
    observed: List[int], 
    expected: List[float]
) -> tuple[float, int]:
    """Chi-square goodness of fit test.
    
    Tests if observed frequencies match expected.
    
    χ² = Σ (O - E)² / E
    
    Returns:
        chi_square_statistic, degrees_of_freedom
    """
    chi_sq = sum(
        (o - e) ** 2 / e 
        for o, e in zip(observed, expected)
    )
    df = len(observed) - 1
    
    return chi_sq, df
```

---

## 6. Regresión Estadística {#6-regresion}

### 6.1 Regresión Lineal Simple

```python
def linear_regression_ols(
    X: List[float], 
    y: List[float]
) -> tuple[float, float]:
    """Ordinary Least Squares linear regression.
    
    y = β₀ + β₁x + ε
    
    Minimizes Σ(yᵢ - ŷᵢ)²
    
    Returns:
        intercept (β₀), slope (β₁)
    """
    n = len(X)
    mean_x = sum(X) / n
    mean_y = sum(y) / n
    
    # Slope: β₁ = Cov(x,y) / Var(x)
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(X, y))
    denominator = sum((xi - mean_x) ** 2 for xi in X)
    
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    
    return intercept, slope


def r_squared(y_true: List[float], y_pred: List[float]) -> float:
    """Coefficient of determination (R²).
    
    R² = 1 - SS_res / SS_tot
    
    Proportion of variance explained by the model.
    0 ≤ R² ≤ 1 (for linear regression with intercept)
    """
    mean_y = sum(y_true) / len(y_true)
    
    ss_tot = sum((yi - mean_y) ** 2 for yi in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    
    return 1 - ss_res / ss_tot
```

### 6.2 Regresión Lineal Múltiple (Forma Matricial)

```python
def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication A × B."""
    rows_a, cols_a = len(A), len(A[0])
    cols_b = len(B[0])
    
    result = [[0.0] * cols_b for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += A[i][k] * B[k][j]
    
    return result


def transpose(A: List[List[float]]) -> List[List[float]]:
    """Matrix transpose."""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


# Note: Full OLS requires matrix inversion
# β = (XᵀX)⁻¹ Xᵀy
# In practice, use numerical libraries (numpy.linalg.lstsq)
```

---

## ⚠️ Conexión con Machine Learning

```
ESTADÍSTICA → MACHINE LEARNING:
─────────────────────────────────

• MLE → Training neural networks (minimize cross-entropy)
• MAP → Regularization (L1/L2 penalties)
• Hypothesis testing → Model comparison, A/B testing
• Confidence intervals → Uncertainty quantification
• Bias-variance → Model selection, regularization tuning

DIFERENCIAS DE ENFOQUE:
• Estadística: explicar, inferir sobre parámetros
• ML: predecir, generalizar a nuevos datos

Pero los fundamentos matemáticos son los MISMOS.
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 20.1: MLE para Datos Reales
Dado un dataset de tiempos de respuesta, estimar λ de distribución exponencial.

### Ejercicio 20.2: A/B Testing
Implementar prueba de proporciones para comparar dos versiones.

### Ejercicio 20.3: Regresión con Regularización
Comparar OLS vs Ridge (MAP con prior gaussiano).

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Statistical Learning](https://www.statlearning.com/) | Libro (gratis) | 🔴 Obligatorio |
| [Seeing Theory](https://seeing-theory.brown.edu/) | Interactivo | 🟡 Recomendado |
| [StatQuest: Statistics](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) | Videos | 🔴 Obligatorio |

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [19_PROBABILIDAD_FUNDAMENTOS](19_PROBABILIDAD_FUNDAMENTOS.md) | [00_INDICE](00_INDICE.md) | [21_CADENAS_MARKOV_MONTECARLO](21_CADENAS_MARKOV_MONTECARLO.md) |
