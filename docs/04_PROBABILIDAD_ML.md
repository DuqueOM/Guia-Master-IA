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

## 📚 Contenido

### Día 1-2: Fundamentos de Probabilidad

#### 1.1 Probabilidad Básica

```
P(A) = casos favorables / casos totales

Propiedades:
- 0 ≤ P(A) ≤ 1
- P(Ω) = 1 (espacio muestral)
- P(∅) = 0 (evento imposible)
```

#### 1.2 Probabilidad Condicional

```
P(A|B) = P(A ∩ B) / P(B)

"Probabilidad de A dado que B ocurrió"
```

**Ejemplo en ML:**
- P(spam | contiene "gratis") = ¿Qué tan probable es spam si el email dice "gratis"?

#### 1.3 Independencia

```
A y B son independientes si:
P(A ∩ B) = P(A) · P(B)

Equivalente a:
P(A|B) = P(A)
```

---

### Día 3-4: Teorema de Bayes (Crítico para ML)

#### 2.1 La Fórmula

```
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

```
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

### Día 5: Distribución Gaussiana (Normal)

#### 3.1 La Distribución Más Importante en ML

```
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

#### 4.1 La Idea Central

```
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

### Día 7: Softmax como Distribución de Probabilidad

#### 5.1 De Logits a Probabilidades

```
                    exp(zᵢ)
softmax(z)ᵢ = ─────────────────
              Σⱼ exp(zⱼ)

Propiedades:
- Cada salida ∈ (0, 1)
- Suma de salidas = 1 (distribución válida)
- Preserva el orden (mayor logit → mayor probabilidad)
```

#### 5.2 El Problema de Estabilidad Numérica (v3.3)

```
⚠️ PROBLEMA: exp() puede causar overflow/underflow

Ejemplo peligroso:
    z = [1000, 1001, 1002]
    exp(z) = [inf, inf, inf]  → NaN en softmax!

Ejemplo underflow:
    z = [-1000, -1001, -1002]
    exp(z) = [0, 0, 0]  → 0/0 = NaN!
```

#### 5.3 Log-Sum-Exp Trick (Estabilidad Numérica)

```
TRUCO: softmax(z) = softmax(z - max(z))

Demostración:
    softmax(z - c)ᵢ = exp(zᵢ - c) / Σⱼ exp(zⱼ - c)
                    = exp(zᵢ)·exp(-c) / Σⱼ exp(zⱼ)·exp(-c)
                    = exp(zᵢ) / Σⱼ exp(zⱼ)
                    = softmax(z)ᵢ

Al restar max(z), todos los exponentes son ≤ 0, evitando overflow.
```

#### 5.4 Implementación Numéricamente Estable

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

```
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

## ✅ Checklist del Módulo

- [ ] Puedo explicar el Teorema de Bayes con un ejemplo
- [ ] Sé calcular la PDF de una Gaussiana a mano
- [ ] Entiendo por qué MLE da Cross-Entropy como loss
- [ ] Implementé softmax numéricamente estable
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

> 💡 **Nota Final:** Este módulo es deliberadamente corto (1 semana). No necesitas ser experto en probabilidad para la Línea 1, pero estos conceptos son el "pegamento" que conecta las matemáticas con las funciones de pérdida que usarás en los siguientes módulos.

---

**[← Módulo 03: Cálculo](03_CALCULO_MULTIVARIANTE.md)** | **[Módulo 05: Supervised Learning →](05_SUPERVISED_LEARNING.md)**
