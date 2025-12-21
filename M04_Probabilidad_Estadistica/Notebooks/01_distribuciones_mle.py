#!/usr/bin/env python3
"""
Notebook M04: Distribuciones y Maximum Likelihood Estimation
=============================================================
Implementación práctica de conceptos de probabilidad para ML.

Ejecutar: python 01_distribuciones_mle.py
"""
from __future__ import annotations

import numpy as np

rng = np.random.default_rng(seed=42)  # Reproducibilidad controlada

# =============================================================================
# PARTE 1: Distribuciones de Probabilidad
# =============================================================================

print("=" * 60)
print("PARTE 1: Distribuciones de Probabilidad")
print("=" * 60)

# --- 1.1 Distribución Gaussiana (Normal) ---
print("\n--- 1.1 Distribución Gaussiana ---")


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Función de densidad de probabilidad Gaussiana.

    p(x) = (1 / sqrt(2πσ²)) * exp(-(x-μ)² / 2σ²)
    """
    coef = 1.0 / (sigma * np.sqrt(2 * np.pi))  # Coeficiente de normalización
    exponent = -((x - mu) ** 2) / (2 * sigma**2)  # Exponente
    return coef * np.exp(exponent)  # PDF completa


# Ejemplo: N(0, 1) - Gaussiana estándar
x = np.linspace(-4, 4, 100)  # Dominio
pdf_standard = gaussian_pdf(x, mu=0.0, sigma=1.0)  # PDF evaluada
print(f"PDF en x=0: {gaussian_pdf(np.array([0.0]), 0.0, 1.0)[0]:.4f}")
print(f"PDF en x=1: {gaussian_pdf(np.array([1.0]), 0.0, 1.0)[0]:.4f}")

# --- 1.2 Distribución Bernoulli ---
print("\n--- 1.2 Distribución Bernoulli ---")


def bernoulli_pmf(k: np.ndarray, p: float) -> np.ndarray:
    """
    Función de masa de probabilidad Bernoulli.

    P(X=k) = p^k * (1-p)^(1-k)  para k ∈ {0, 1}
    """
    return (p**k) * ((1 - p) ** (1 - k))  # PMF


p = 0.7  # Probabilidad de éxito
print(f"P(X=1) con p={p}: {bernoulli_pmf(np.array([1]), p)[0]:.4f}")
print(f"P(X=0) con p={p}: {bernoulli_pmf(np.array([0]), p)[0]:.4f}")

# --- 1.3 Softmax (Distribución Categórica) ---
print("\n--- 1.3 Softmax Estable ---")


def softmax_stable(z: np.ndarray) -> np.ndarray:
    """
    Softmax numéricamente estable.

    softmax(z)_i = exp(z_i - max(z)) / Σ exp(z_j - max(z))

    El truco de restar max(z) evita overflow en exp().
    """
    z_shifted = z - np.max(z)  # Estabilidad numérica: evita exp(grande)
    exp_z = np.exp(z_shifted)  # Exponencial de cada elemento
    return exp_z / np.sum(exp_z)  # Normalización para que sume 1


logits = np.array([2.0, 1.0, 0.1])  # Logits (scores sin normalizar)
probs = softmax_stable(logits)  # Probabilidades
print(f"Logits: {logits}")
print(f"Softmax: {probs}")
print(f"Suma: {probs.sum():.4f}")  # Debe ser 1.0

# =============================================================================
# PARTE 2: Maximum Likelihood Estimation (MLE)
# =============================================================================

print("\n" + "=" * 60)
print("PARTE 2: Maximum Likelihood Estimation")
print("=" * 60)

# --- 2.1 MLE para Gaussiana ---
print("\n--- 2.1 MLE para Gaussiana ---")

# Datos observados (simulamos de una Gaussiana conocida)
true_mu, true_sigma = 5.0, 2.0  # Parámetros verdaderos (desconocidos en práctica)
data = rng.normal(true_mu, true_sigma, size=1000)  # 1000 muestras


def mle_gaussian(data: np.ndarray) -> tuple[float, float]:
    """
    MLE para parámetros de Gaussiana.

    μ_MLE = (1/n) Σ x_i  (media muestral)
    σ²_MLE = (1/n) Σ (x_i - μ_MLE)²  (varianza muestral)
    """
    n = len(data)  # Número de muestras
    mu_mle = np.sum(data) / n  # Media MLE
    sigma2_mle = np.sum((data - mu_mle) ** 2) / n  # Varianza MLE
    sigma_mle = np.sqrt(sigma2_mle)  # Desviación estándar MLE
    return mu_mle, sigma_mle


mu_est, sigma_est = mle_gaussian(data)  # Estimación
print(f"Parámetros verdaderos: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimados: μ={mu_est:.4f}, σ={sigma_est:.4f}")

# --- 2.2 MLE para Bernoulli ---
print("\n--- 2.2 MLE para Bernoulli ---")

# Datos binarios
true_p = 0.3  # Probabilidad verdadera
coin_flips = rng.binomial(1, true_p, size=500)  # 500 lanzamientos


def mle_bernoulli(data: np.ndarray) -> float:
    """
    MLE para parámetro p de Bernoulli.

    p_MLE = (1/n) Σ x_i = número de éxitos / total
    """
    return float(np.sum(data)) / float(len(data))  # Proporción de éxitos


p_est = mle_bernoulli(coin_flips)  # Estimación
print(f"p verdadero: {true_p}")
print(f"p MLE: {p_est:.4f}")

# =============================================================================
# PARTE 3: Teorema de Bayes
# =============================================================================

print("\n" + "=" * 60)
print("PARTE 3: Teorema de Bayes")
print("=" * 60)

# --- 3.1 Bayes Básico ---
print("\n--- 3.1 Ejemplo Clásico: Test Médico ---")

# Escenario: Test para una enfermedad rara
P_enfermedad = 0.001  # P(D) = 1 en 1000 tiene la enfermedad (prior)
P_positivo_dado_enfermo = 0.99  # P(+|D) = sensibilidad
P_positivo_dado_sano = 0.05  # P(+|¬D) = tasa de falsos positivos

# Teorema de Bayes: P(D|+) = P(+|D) * P(D) / P(+)
# donde P(+) = P(+|D)*P(D) + P(+|¬D)*P(¬D)

P_sano = 1 - P_enfermedad  # P(¬D)
P_positivo = (
    P_positivo_dado_enfermo * P_enfermedad + P_positivo_dado_sano * P_sano
)  # Probabilidad total
P_enfermo_dado_positivo = (P_positivo_dado_enfermo * P_enfermedad) / P_positivo  # Bayes

print(f"P(Enfermo) = {P_enfermedad:.4f}")
print(f"P(+|Enfermo) = {P_positivo_dado_enfermo:.4f}")
print(f"P(+|Sano) = {P_positivo_dado_sano:.4f}")
print(
    f"P(Enfermo|+) = {P_enfermo_dado_positivo:.4f}"
)  # ~2% aunque test sea 99% preciso!

# --- 3.2 Bayes en Clasificación ---
print("\n--- 3.2 Naive Bayes Simplificado ---")


def naive_bayes_predict(
    x: np.ndarray,  # Features del ejemplo a clasificar
    class_priors: np.ndarray,  # P(C_k) para cada clase
    class_means: np.ndarray,  # μ_k para cada clase (asumiendo Gaussiana)
    class_stds: np.ndarray,  # σ_k para cada clase
) -> int:
    """
    Clasificador Naive Bayes Gaussiano simplificado.

    P(C_k|x) ∝ P(C_k) * Π P(x_i|C_k)

    Asumimos features independientes (naive assumption).
    """
    n_classes = len(class_priors)  # Número de clases
    log_posteriors = np.zeros(n_classes)  # Log-probabilidades posteriores

    for k in range(n_classes):
        # Log-prior
        log_prior = np.log(class_priors[k])  # log P(C_k)

        # Log-likelihood (suma de log de Gaussianas independientes)
        log_likelihood = np.sum(
            -0.5 * np.log(2 * np.pi * class_stds[k] ** 2)
            - 0.5 * ((x - class_means[k]) / class_stds[k]) ** 2
        )  # Σ log P(x_i|C_k)

        log_posteriors[k] = log_prior + log_likelihood  # Log-posterior

    return int(np.argmax(log_posteriors))  # Clase con mayor posterior


# Ejemplo simple con 2 clases
class_priors = np.array([0.5, 0.5])  # Priors uniformes
class_means = np.array([0.0, 3.0])  # Medias de cada clase
class_stds = np.array([1.0, 1.0])  # Stds iguales

test_point = np.array([1.0])  # Punto a clasificar
prediction = naive_bayes_predict(test_point, class_priors, class_means, class_stds)
print(f"Punto x={test_point[0]} → Clase {prediction}")

# =============================================================================
# PARTE 4: Conexión con Loss Functions en ML
# =============================================================================

print("\n" + "=" * 60)
print("PARTE 4: MLE y Loss Functions")
print("=" * 60)

print("\n--- Conexión MLE ↔ Cross-Entropy ---")
print(
    """
En clasificación binaria con Logistic Regression:

- Modelo: P(y=1|x) = σ(w·x + b)
- Likelihood: L(w,b) = Π P(y_i|x_i)
- Log-Likelihood: ℓ(w,b) = Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]

Maximizar Log-Likelihood = Minimizar Cross-Entropy Loss:

  Loss_CE = -ℓ(w,b) = -Σ [y_i log(p_i) + (1-y_i) log(1-p_i)]

¡Son equivalentes!
"""
)


def cross_entropy_loss(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15
) -> float:
    """
    Binary Cross-Entropy Loss.

    L = -(1/n) Σ [y log(p) + (1-y) log(1-p)]

    eps evita log(0).
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Estabilidad numérica
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))  # BCE
    return float(loss)


# Ejemplo
y_true = np.array([1, 0, 1, 1, 0])  # Labels verdaderos
y_pred_good = np.array([0.9, 0.1, 0.8, 0.95, 0.2])  # Predicciones buenas
y_pred_bad = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Predicciones malas

print(f"CE Loss (predicciones buenas): {cross_entropy_loss(y_true, y_pred_good):.4f}")
print(f"CE Loss (predicciones malas): {cross_entropy_loss(y_true, y_pred_bad):.4f}")

print("\n" + "=" * 60)
print("✅ Notebook M04 completado")
print("=" * 60)
