# Módulo 04 - Fundamentos de Probabilidad para IA

> **🎯 Objetivo:** Dominar los conceptos probabilísticos esenciales para ML/IA  
> **⭐ PATHWAY LÍNEA 2:** Probability Fundamentals for Data Science and AI

---

## 🧠 Analogía: La Probabilidad como Lenguaje de la Incertidumbre

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   LA IA VIVE EN LA INCERTIDUMBRE                                            │
│   ─────────────────────────────                                             │
│                                                                             │
│   Determinístico (Algoritmos clásicos):                                     │
│   if x > 5: return "grande"  → SIEMPRE la misma respuesta                   │
│                                                                             │
│   Probabilístico (Machine Learning):                                        │
│   P(spam | email) = 0.87     → "Probablemente spam, 87% seguro"             │
│                                                                             │
│   ¿POR QUÉ PROBABILIDAD?                                                    │
│   • Datos ruidosos e incompletos                                            │
│   • Predicciones sobre el futuro                                            │
│   • Cuantificar confianza en decisiones                                     │
│   • Generalizar de muestras a poblaciones                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Fundamentos de Probabilidad](#1-fundamentos)
2. [Probabilidad Condicional y Bayes](#2-bayes)
3. [Variables Aleatorias](#3-variables-aleatorias)
4. [Distribuciones de Probabilidad](#4-distribuciones)
5. [Esperanza, Varianza y Momentos](#5-momentos)

---

## 1. Fundamentos de Probabilidad {#1-fundamentos}

### 1.1 Espacio Muestral y Eventos

```python
from typing import Set, Dict
import math

# Espacio muestral: todos los resultados posibles
# Evento: subconjunto del espacio muestral

def probability_basic(favorable: int, total: int) -> float:
    """Basic probability: P(A) = favorable outcomes / total outcomes.
    
    Example:
        >>> probability_basic(1, 6)  # Sacar un 6 en un dado
        0.16666666666666666
    """
    if total == 0:
        raise ValueError("Total outcomes cannot be zero")
    return favorable / total


def complement_probability(p_a: float) -> float:
    """P(not A) = 1 - P(A).
    
    Example:
        >>> complement_probability(0.3)  # P(no llueve) si P(llueve) = 0.3
        0.7
    """
    return 1.0 - p_a
```

### 1.2 Axiomas de Kolmogorov

```
AXIOMAS DE PROBABILIDAD:
─────────────────────────
1. P(A) ≥ 0          (No negativas)
2. P(Ω) = 1          (Espacio muestral tiene prob. 1)
3. P(A ∪ B) = P(A) + P(B)   si A ∩ B = ∅  (Aditividad)

PROPIEDADES DERIVADAS:
─────────────────────────
• P(∅) = 0
• P(A') = 1 - P(A)
• P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
• Si A ⊆ B, entonces P(A) ≤ P(B)
```

### 1.3 Operaciones con Eventos

```python
def union_probability(p_a: float, p_b: float, p_intersection: float) -> float:
    """P(A ∪ B) = P(A) + P(B) - P(A ∩ B).
    
    Inclusion-exclusion principle.
    
    Example:
        >>> # P(rey o corazón) en baraja
        >>> union_probability(4/52, 13/52, 1/52)
        0.3076923076923077
    """
    return p_a + p_b - p_intersection


def intersection_independent(p_a: float, p_b: float) -> float:
    """P(A ∩ B) = P(A) × P(B) for independent events.
    
    Example:
        >>> # Dos monedas, ambas cara
        >>> intersection_independent(0.5, 0.5)
        0.25
    """
    return p_a * p_b
```

---

## 2. Probabilidad Condicional y Bayes {#2-bayes}

### 2.1 Probabilidad Condicional

```python
def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """P(A|B) = P(A ∩ B) / P(B).
    
    Probability of A given B has occurred.
    
    Example:
        >>> # P(llueve | nublado)
        >>> conditional_probability(0.3, 0.4)
        0.75
    """
    if p_b == 0:
        raise ValueError("P(B) cannot be zero")
    return p_a_and_b / p_b
```

### 2.2 Teorema de Bayes ⭐ FUNDAMENTAL PARA ML

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   TEOREMA DE BAYES                                                          │
│   ─────────────────                                                         │
│                                                                             │
│                    P(B|A) × P(A)                                            │
│   P(A|B) = ─────────────────────────                                        │
│                     P(B)                                                    │
│                                                                             │
│   Donde:                                                                    │
│   • P(A|B) = POSTERIOR (lo que queremos saber)                              │
│   • P(B|A) = LIKELIHOOD (evidencia dado la hipótesis)                       │
│   • P(A)   = PRIOR (creencia inicial)                                       │
│   • P(B)   = EVIDENCE (normalizador)                                        │
│                                                                             │
│   EJEMPLO SPAM:                                                             │
│   P(spam | "gratis") = P("gratis"|spam) × P(spam) / P("gratis")             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
def bayes_theorem(
    p_b_given_a: float,  # Likelihood
    p_a: float,          # Prior
    p_b: float           # Evidence
) -> float:
    """Bayes' Theorem: P(A|B) = P(B|A) × P(A) / P(B).
    
    The foundation of probabilistic machine learning.
    
    Example:
        >>> # Test médico: P(enfermo | test positivo)
        >>> # P(test+|enfermo) = 0.99, P(enfermo) = 0.01, P(test+) = 0.02
        >>> bayes_theorem(0.99, 0.01, 0.02)
        0.495
    """
    return (p_b_given_a * p_a) / p_b


def bayes_with_total_probability(
    p_b_given_a: float,
    p_a: float,
    p_b_given_not_a: float
) -> float:
    """Bayes with P(B) calculated via total probability.
    
    P(B) = P(B|A)P(A) + P(B|¬A)P(¬A)
    
    Example:
        >>> # Spam classifier
        >>> # P("free"|spam)=0.7, P(spam)=0.3, P("free"|not spam)=0.1
        >>> bayes_with_total_probability(0.7, 0.3, 0.1)
        0.75
    """
    p_not_a = 1 - p_a
    p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
    return (p_b_given_a * p_a) / p_b
```

### 2.3 Aplicación: Clasificador Naive Bayes

```python
from collections import defaultdict
from typing import List, Tuple

class NaiveBayesClassifier:
    """Simple Naive Bayes for text classification.
    
    Assumes features are conditionally independent given class.
    
    P(class|features) ∝ P(class) × ∏ P(feature|class)
    """
    
    def __init__(self) -> None:
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.feature_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_samples: int = 0
    
    def fit(self, X: List[List[str]], y: List[str]) -> None:
        """Train the classifier.
        
        Args:
            X: List of feature lists (e.g., words in documents)
            y: List of class labels
        """
        for features, label in zip(X, y):
            self.class_counts[label] += 1
            self.total_samples += 1
            for feature in features:
                self.feature_counts[label][feature] += 1
    
    def predict(self, features: List[str]) -> str:
        """Predict class for given features.
        
        Returns class with highest posterior probability.
        """
        best_class = None
        best_score = float('-inf')
        
        for cls in self.class_counts:
            # Log probability to avoid underflow
            score = math.log(self.class_counts[cls] / self.total_samples)
            
            total_features_in_class = sum(self.feature_counts[cls].values())
            vocab_size = len(set(
                f for counts in self.feature_counts.values() 
                for f in counts
            ))
            
            for feature in features:
                # Laplace smoothing
                count = self.feature_counts[cls].get(feature, 0) + 1
                prob = count / (total_features_in_class + vocab_size)
                score += math.log(prob)
            
            if score > best_score:
                best_score = score
                best_class = cls
        
        return best_class
```

---

## 3. Variables Aleatorias {#3-variables-aleatorias}

### 3.1 Discretas vs Continuas

```
VARIABLES ALEATORIAS:
────────────────────────

DISCRETAS: Valores contables
• Número de emails spam
• Cara o cruz
• Clasificación (0, 1, 2, ...)

CONTINUAS: Valores en un rango
• Temperatura
• Altura
• Probabilidad predicha

FUNCIÓN DE PROBABILIDAD:
• Discreta: PMF (Probability Mass Function)
  P(X = x)

• Continua: PDF (Probability Density Function)
  P(a ≤ X ≤ b) = ∫[a,b] f(x)dx
```

### 3.2 Función de Distribución Acumulativa (CDF)

```python
def cdf_from_pmf(pmf: Dict[int, float], x: int) -> float:
    """CDF: F(x) = P(X ≤ x) = Σ P(X = k) for k ≤ x.
    
    Example:
        >>> pmf = {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
        >>> cdf_from_pmf(pmf, 3)
        0.5
    """
    return sum(prob for val, prob in pmf.items() if val <= x)
```

---

## 4. Distribuciones de Probabilidad {#4-distribuciones}

### 4.1 Distribución Bernoulli

```python
def bernoulli_pmf(k: int, p: float) -> float:
    """Bernoulli: single trial with success probability p.
    
    P(X = k) = p^k × (1-p)^(1-k) for k ∈ {0, 1}
    
    Example:
        >>> bernoulli_pmf(1, 0.7)  # P(success) with p=0.7
        0.7
    """
    if k == 1:
        return p
    elif k == 0:
        return 1 - p
    else:
        return 0.0
```

### 4.2 Distribución Binomial

```python
def factorial(n: int) -> int:
    """Calculate n! iteratively."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def binomial_coefficient(n: int, k: int) -> int:
    """C(n, k) = n! / (k! × (n-k)!)."""
    return factorial(n) // (factorial(k) * factorial(n - k))


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Binomial: k successes in n independent trials.
    
    P(X = k) = C(n,k) × p^k × (1-p)^(n-k)
    
    Example:
        >>> # P(3 caras en 5 lanzamientos)
        >>> binomial_pmf(3, 5, 0.5)
        0.3125
    """
    return binomial_coefficient(n, k) * (p ** k) * ((1 - p) ** (n - k))
```

### 4.3 Distribución Normal (Gaussiana) ⭐ FUNDAMENTAL

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   DISTRIBUCIÓN NORMAL                                                       │
│   ───────────────────                                                       │
│                                                                             │
│                    1              (x - μ)²                                  │
│   f(x) = ───────────────── × exp(- ─────────)                               │
│          σ × √(2π)                  2σ²                                     │
│                                                                             │
│   Parámetros:                                                               │
│   • μ (mu) = media (centro de la campana)                                   │
│   • σ (sigma) = desviación estándar (ancho)                                 │
│                                                                             │
│              .---.                                                          │
│            .'     '.            68% dentro de 1σ                            │
│           /    μ    \           95% dentro de 2σ                            │
│         _/           \_         99.7% dentro de 3σ                          │
│   ─────/───────────────\─────                                               │
│       μ-2σ  μ-σ  μ  μ+σ  μ+2σ                                               │
│                                                                             │
│   ¿POR QUÉ ES TAN IMPORTANTE?                                               │
│   • Teorema del Límite Central                                              │
│   • Muchos fenómenos naturales                                              │
│   • Base de modelos lineales                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Probability density function of normal distribution.
    
    Example:
        >>> normal_pdf(0, 0, 1)  # Standard normal at mean
        0.3989422804014327
    """
    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma ** 2)
    return coefficient * math.exp(exponent)


def standard_normal_cdf_approx(x: float) -> float:
    """Approximation of standard normal CDF using error function.
    
    Uses the relationship: Φ(x) = 0.5 × (1 + erf(x/√2))
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def z_score(x: float, mu: float, sigma: float) -> float:
    """Standardize a value: z = (x - μ) / σ.
    
    Converts any normal distribution to standard normal.
    
    Example:
        >>> z_score(85, 70, 10)  # Score of 85 with mean 70, std 10
        1.5
    """
    return (x - mu) / sigma
```

### 4.4 Otras Distribuciones Importantes

```python
def poisson_pmf(k: int, lam: float) -> float:
    """Poisson: events in fixed interval.
    
    P(X = k) = (λ^k × e^(-λ)) / k!
    
    Used for: emails per hour, arrivals per minute.
    
    Example:
        >>> poisson_pmf(3, 2.5)  # 3 events when average is 2.5
        0.21376...
    """
    return (lam ** k * math.exp(-lam)) / factorial(k)


def exponential_pdf(x: float, lam: float) -> float:
    """Exponential: time between Poisson events.
    
    f(x) = λ × e^(-λx) for x ≥ 0
    
    Used for: time until next event.
    """
    if x < 0:
        return 0.0
    return lam * math.exp(-lam * x)
```

---

## 5. Esperanza, Varianza y Momentos {#5-momentos}

### 5.1 Valor Esperado (Media)

```python
def expected_value_discrete(pmf: Dict[float, float]) -> float:
    """E[X] = Σ x × P(X = x).
    
    The "center of mass" of the distribution.
    
    Example:
        >>> pmf = {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
        >>> expected_value_discrete(pmf)
        3.5
    """
    return sum(x * prob for x, prob in pmf.items())


def expected_value_sample(data: List[float]) -> float:
    """Sample mean as estimate of E[X].
    
    x̄ = (1/n) × Σ xᵢ
    """
    return sum(data) / len(data)
```

### 5.2 Varianza y Desviación Estándar

```python
def variance_discrete(pmf: Dict[float, float]) -> float:
    """Var(X) = E[(X - μ)²] = E[X²] - (E[X])².
    
    Measures spread around the mean.
    """
    mu = expected_value_discrete(pmf)
    return sum((x - mu) ** 2 * prob for x, prob in pmf.items())


def variance_sample(data: List[float]) -> float:
    """Sample variance (unbiased estimator).
    
    s² = (1/(n-1)) × Σ (xᵢ - x̄)²
    """
    n = len(data)
    mean = expected_value_sample(data)
    return sum((x - mean) ** 2 for x in data) / (n - 1)


def std_dev_sample(data: List[float]) -> float:
    """Sample standard deviation."""
    return math.sqrt(variance_sample(data))
```

### 5.3 Covarianza y Correlación

```python
def covariance(x: List[float], y: List[float]) -> float:
    """Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)].
    
    Measures linear relationship between variables.
    • Cov > 0: positive relationship
    • Cov < 0: negative relationship
    • Cov = 0: no linear relationship (not necessarily independent!)
    """
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    return sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)


def correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation: ρ = Cov(X,Y) / (σₓ × σᵧ).
    
    Normalized to [-1, 1].
    • ρ = 1: perfect positive linear relationship
    • ρ = -1: perfect negative linear relationship
    • ρ = 0: no linear relationship
    
    IMPORTANT for ML: Correlation ≠ Causation!
    """
    cov = covariance(x, y)
    std_x = std_dev_sample(x)
    std_y = std_dev_sample(y)
    
    if std_x == 0 or std_y == 0:
        return 0.0
    
    return cov / (std_x * std_y)
```

---

## ⚠️ Conceptos Clave para ML

### Independence vs Conditional Independence

```
INDEPENDENCIA:
P(A ∩ B) = P(A) × P(B)
P(A|B) = P(A)  (conocer B no cambia A)

INDEPENDENCIA CONDICIONAL (crucial para Naive Bayes):
P(A ∩ B | C) = P(A|C) × P(B|C)

Aunque A y B no sean independientes, pueden serlo dado C.
Naive Bayes ASUME que features son independientes dado la clase.
```

### Law of Large Numbers

```
A medida que n → ∞:
• Sample mean → True mean
• Sample variance → True variance

Justifica usar estadísticas muestrales como estimadores.
```

### Central Limit Theorem ⭐

```
La suma/promedio de muchas variables aleatorias independientes
tiende a una distribución NORMAL, sin importar la distribución original.

IMPLICACIÓN PARA ML:
• Muchos errores se distribuyen normalmente
• Justifica asumir normalidad en muchos modelos
• Base teórica de muchos métodos estadísticos
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 19.1: Bayes para Diagnóstico
Un test tiene 99% sensibilidad, 95% especificidad. La enfermedad afecta al 1% de la población. ¿Cuál es P(enfermo | test+)?

### Ejercicio 19.2: Distribución Binomial
Si 30% de emails son spam, ¿cuál es la probabilidad de recibir exactamente 4 spam en 10 emails?

### Ejercicio 19.3: Naive Bayes
Implementar clasificador de sentimiento usando el código de ejemplo.

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Probability for Data Science](https://www.coursera.org/learn/machine-learning-probability-and-statistics) | Curso | 🔴 Obligatorio |
| [3Blue1Brown: Bayes](https://www.youtube.com/watch?v=HZGCoVF3YvM) | Video | 🔴 Obligatorio |
| [Khan Academy: Statistics](https://www.khanacademy.org/math/statistics-probability) | Curso | 🟡 Recomendado |

---

## 🔗 Referencias del Glosario

- [Probabilidad Condicional](GLOSARIO.md#probabilidad-condicional)
- [Teorema de Bayes](GLOSARIO.md#teorema-de-bayes)
- [Distribución Normal](GLOSARIO.md#distribucion-normal)
- [Esperanza Matemática](GLOSARIO.md#esperanza-matematica)
- [Varianza](GLOSARIO.md#varianza)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [18_HEAPS](18_HEAPS.md) | [00_INDICE](00_INDICE.md) | [20_ESTADISTICA_INFERENCIAL](20_ESTADISTICA_INFERENCIAL.md) |
