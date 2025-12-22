#!/usr/bin/env python3
"""
Lab 1: MLE/MAP y Estimadores Estadísticos
==========================================

Módulo: M04 - Probabilidad y Estadística
Tiempo Estimado: 2-3 horas
Prerequisitos: Cálculo diferencial, probabilidad básica

Objetivos de Aprendizaje:
-------------------------
1. Derivar MLE para distribuciones comunes (Bernoulli, Gaussiana, Poisson)
2. Implementar MLE numéricamente con scipy.optimize
3. Comparar MLE vs MAP con diferentes priors
4. Visualizar el trade-off sesgo-varianza

Referencias:
------------
- Murphy, "ML: A Probabilistic Perspective", Cap. 3
- Bishop, "Pattern Recognition and ML", Cap. 2
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

rng = np.random.default_rng(42)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %% [markdown]
# # Lab 1: Maximum Likelihood y Maximum A Posteriori
#
# ## Introducción
#
# | Método | Fórmula | Interpretación |
# |--------|---------|----------------|
# | **MLE** | θ̂ = argmax P(D\|θ) | Maximiza verosimilitud |
# | **MAP** | θ̂ = argmax P(θ\|D) = argmax P(D\|θ)P(θ) | Incluye prior |
#
# MAP = MLE cuando el prior es uniforme (no informativo).

# %% [markdown]
# ## Parte 1: MLE Analítico (45 min)
#
# ### 1.1 MLE para Distribución de Bernoulli
#
# Datos: x₁, x₂, ..., xₙ ∈ {0, 1}
#
# Likelihood: L(θ) = ∏ θ^xᵢ (1-θ)^(1-xᵢ)
#
# Log-likelihood: ℓ(θ) = Σxᵢ log(θ) + Σ(1-xᵢ) log(1-θ)
#
# Derivando y igualando a 0: θ̂_MLE = (1/n) Σxᵢ = x̄


# %%
def mle_bernoulli(data: NDArray[np.int64]) -> float:
    """
    MLE para parámetro θ de Bernoulli.

    θ̂_MLE = media muestral = número de éxitos / n
    """
    return float(np.mean(data))


# Demostración
print("=== MLE para Bernoulli ===\n")
true_theta = 0.7
n_samples = 100
data_bernoulli = rng.binomial(1, true_theta, n_samples)

theta_mle = mle_bernoulli(data_bernoulli)
print(f"θ verdadero:  {true_theta}")
print(f"θ̂_MLE:        {theta_mle:.4f}")
print(f"Datos: {sum(data_bernoulli)} éxitos de {n_samples} intentos")


# %% [markdown]
# ### 1.2 MLE para Distribución Gaussiana
#
# Datos: x₁, x₂, ..., xₙ ~ N(μ, σ²)
#
# Log-likelihood:
# ℓ(μ, σ²) = -n/2 log(2π) - n/2 log(σ²) - 1/(2σ²) Σ(xᵢ - μ)²
#
# Derivando:
# - μ̂_MLE = (1/n) Σxᵢ = x̄
# - σ̂²_MLE = (1/n) Σ(xᵢ - x̄)² (¡sesgado!)


# %%
def mle_gaussian(data: NDArray[np.float64]) -> tuple[float, float]:
    """
    MLE para parámetros μ y σ² de Gaussiana.

    Retorna
    -------
    tuple[float, float]
        - μ̂_MLE (media muestral)
        - σ̂²_MLE (varianza muestral sesgada, dividiendo por n)
    """
    n = len(data)
    mu_mle = np.mean(data)
    sigma2_mle = np.sum((data - mu_mle) ** 2) / n  # Sesgado (divide por n)
    return mu_mle, sigma2_mle


# Demostración
print("\n=== MLE para Gaussiana ===\n")
true_mu, true_sigma2 = 5.0, 4.0
n_samples = 50
data_gaussian = rng.normal(true_mu, np.sqrt(true_sigma2), n_samples)

mu_mle, sigma2_mle = mle_gaussian(data_gaussian)
print(f"Parámetros verdaderos: μ = {true_mu}, σ² = {true_sigma2}")
print(f"MLE: μ̂ = {mu_mle:.4f}, σ̂² = {sigma2_mle:.4f}")
print("\n⚠️ Nota: σ̂²_MLE está sesgado. Estimador insesgado divide por (n-1).")
print(f"   σ̂² insesgado = {np.var(data_gaussian, ddof=1):.4f}")


# %% [markdown]
# ### 1.3 MLE para Distribución de Poisson
#
# Datos: x₁, x₂, ..., xₙ ~ Poisson(λ)
#
# Likelihood: L(λ) = ∏ (λ^xᵢ e^(-λ)) / xᵢ!
#
# Log-likelihood: ℓ(λ) = Σxᵢ log(λ) - nλ - Σlog(xᵢ!)
#
# Derivando: λ̂_MLE = (1/n) Σxᵢ = x̄


# %%
def mle_poisson(data: NDArray[np.int64]) -> float:
    """MLE para parámetro λ de Poisson."""
    return float(np.mean(data))


print("\n=== MLE para Poisson ===\n")
true_lambda = 3.5
data_poisson = rng.poisson(true_lambda, 100)

lambda_mle = mle_poisson(data_poisson)
print(f"λ verdadero:  {true_lambda}")
print(f"λ̂_MLE:        {lambda_mle:.4f}")


# %% [markdown]
# ## Parte 2: MLE Numérico (30 min)
#
# Cuando no hay solución analítica, usamos optimización numérica.


# %%
def negative_log_likelihood_gaussian(
    params: tuple[float, float],
    data: NDArray[np.float64],
) -> float:
    """
    Negative log-likelihood para Gaussiana (para minimizar).

    params: (μ, log_σ²) - usamos log para evitar σ² < 0
    """
    mu, log_sigma2 = params
    sigma2 = np.exp(log_sigma2)
    n = len(data)

    nll = 0.5 * n * np.log(2 * np.pi * sigma2) + np.sum((data - mu) ** 2) / (2 * sigma2)
    return float(nll)


def mle_gaussian_numeric(data: NDArray[np.float64]) -> tuple[float, float]:
    """MLE numérico para Gaussiana usando scipy.optimize."""
    # Inicialización
    mu_init = np.mean(data)
    sigma2_init = np.var(data)

    result = optimize.minimize(
        negative_log_likelihood_gaussian,
        x0=[mu_init, np.log(sigma2_init)],
        args=(data,),
        method="BFGS",
    )

    mu_mle = result.x[0]
    sigma2_mle = np.exp(result.x[1])

    return mu_mle, sigma2_mle


# Comparar analítico vs numérico
print("\n=== Comparación MLE Analítico vs Numérico ===\n")
mu_analytic, sigma2_analytic = mle_gaussian(data_gaussian)
mu_numeric, sigma2_numeric = mle_gaussian_numeric(data_gaussian)

print(f"Analítico: μ̂ = {mu_analytic:.6f}, σ̂² = {sigma2_analytic:.6f}")
print(f"Numérico:  μ̂ = {mu_numeric:.6f}, σ̂² = {sigma2_numeric:.6f}")


# %% [markdown]
# ## Parte 3: MAP con Prior Conjugado (45 min)
#
# ### Beta-Binomial: Prior conjugado para Bernoulli
#
# Prior: θ ~ Beta(α, β)
# Likelihood: x₁, ..., xₙ | θ ~ Bernoulli(θ)
# Posterior: θ | x ~ Beta(α + Σxᵢ, β + n - Σxᵢ)
#
# θ̂_MAP = (α + Σxᵢ - 1) / (α + β + n - 2)
#
# Para α = β = 1 (prior uniforme): θ̂_MAP = θ̂_MLE


# %%
def map_bernoulli(data: NDArray[np.int64], alpha: float, beta: float) -> float:
    """
    MAP para Bernoulli con prior Beta(α, β).

    θ̂_MAP = (α + k - 1) / (α + β + n - 2)

    donde k = número de éxitos, n = tamaño muestral.
    """
    n = len(data)
    k = np.sum(data)

    # Modo de Beta(α + k, β + n - k)
    alpha_post = alpha + k
    beta_post = beta + n - k

    if alpha_post > 1 and beta_post > 1:
        theta_map = (alpha_post - 1) / (alpha_post + beta_post - 2)
    else:
        # Si posterior no tiene modo bien definido, usar media
        theta_map = alpha_post / (alpha_post + beta_post)

    return float(theta_map)


# %% Comparar MLE vs MAP con diferentes priors
print("\n=== MLE vs MAP con Prior Beta ===\n")

# Escenario: pocos datos, θ real = 0.8
true_theta = 0.8
n_small = 10
data_small = rng.binomial(1, true_theta, n_small)
k = sum(data_small)
print(f"Datos: {k} éxitos de {n_small} intentos (θ verdadero = {true_theta})\n")

# MLE
theta_mle = mle_bernoulli(data_small)
print(f"MLE:                        θ̂ = {theta_mle:.4f}")

# MAP con diferentes priors
priors = [
    (1, 1, "Uniforme (α=1, β=1)"),
    (2, 2, "Débil centrado en 0.5 (α=2, β=2)"),
    (5, 5, "Fuerte centrado en 0.5 (α=5, β=5)"),
    (10, 2, "Fuerte sesgado hacia 0.8 (α=10, β=2)"),
]

for alpha, beta, desc in priors:
    theta_map = map_bernoulli(data_small, alpha, beta)
    print(f"MAP {desc}: θ̂ = {theta_map:.4f}")


# %% Visualizar efecto del prior
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
theta_range = np.linspace(0.001, 0.999, 200)

# Prior débil
alpha, beta = 2, 2
k, n = sum(data_small), len(data_small)
prior = stats.beta.pdf(theta_range, alpha, beta)
likelihood = stats.binom.pmf(k, n, theta_range)
posterior = stats.beta.pdf(theta_range, alpha + k, beta + n - k)

axes[0].plot(theta_range, prior / max(prior), "b--", label="Prior")
axes[0].plot(theta_range, likelihood / max(likelihood), "g-.", label="Likelihood")
axes[0].plot(
    theta_range, posterior / max(posterior), "r-", linewidth=2, label="Posterior"
)
axes[0].axvline(
    mle_bernoulli(data_small),
    color="green",
    linestyle=":",
    label=f"MLE={mle_bernoulli(data_small):.2f}",
)
axes[0].axvline(
    map_bernoulli(data_small, alpha, beta),
    color="red",
    linestyle=":",
    label=f"MAP={map_bernoulli(data_small, alpha, beta):.2f}",
)
axes[0].set_title(f"Prior débil: Beta({alpha}, {beta})")
axes[0].set_xlabel("θ")
axes[0].legend()

# Prior fuerte centrado
alpha, beta = 10, 10
posterior = stats.beta.pdf(theta_range, alpha + k, beta + n - k)
prior = stats.beta.pdf(theta_range, alpha, beta)

axes[1].plot(theta_range, prior / max(prior), "b--", label="Prior")
axes[1].plot(theta_range, likelihood / max(likelihood), "g-.", label="Likelihood")
axes[1].plot(
    theta_range, posterior / max(posterior), "r-", linewidth=2, label="Posterior"
)
axes[1].axvline(
    map_bernoulli(data_small, alpha, beta),
    color="red",
    linestyle=":",
    label=f"MAP={map_bernoulli(data_small, alpha, beta):.2f}",
)
axes[1].set_title(f"Prior fuerte: Beta({alpha}, {beta})")
axes[1].set_xlabel("θ")
axes[1].legend()

# Prior informativo correcto
alpha, beta = 8, 2
posterior = stats.beta.pdf(theta_range, alpha + k, beta + n - k)
prior = stats.beta.pdf(theta_range, alpha, beta)

axes[2].plot(theta_range, prior / max(prior), "b--", label="Prior")
axes[2].plot(theta_range, likelihood / max(likelihood), "g-.", label="Likelihood")
axes[2].plot(
    theta_range, posterior / max(posterior), "r-", linewidth=2, label="Posterior"
)
axes[2].axvline(
    map_bernoulli(data_small, alpha, beta),
    color="red",
    linestyle=":",
    label=f"MAP={map_bernoulli(data_small, alpha, beta):.2f}",
)
axes[2].set_title(f"Prior informativo: Beta({alpha}, {beta})")
axes[2].set_xlabel("θ")
axes[2].legend()

plt.tight_layout()
plt.savefig("../assets/mle_vs_map.png", dpi=150)
plt.show()


# %% [markdown]
# ## Parte 4: Sesgo-Varianza (30 min)
#
# ### Trade-off fundamental
#
# MSE(θ̂) = Bias(θ̂)² + Var(θ̂)
#
# - MLE: bajo sesgo, alta varianza (con pocos datos)
# - MAP: puede tener sesgo pero menor varianza


# %%
def simulate_estimator_bias_variance(
    true_theta: float,
    n_samples: int,
    n_simulations: int,
    alpha: float,
    beta: float,
) -> tuple[float, float, float, float]:
    """
    Simula múltiples datasets para calcular sesgo y varianza de MLE y MAP.
    """
    mle_estimates = []
    map_estimates = []

    for _ in range(n_simulations):
        data = rng.binomial(1, true_theta, n_samples)
        mle_estimates.append(mle_bernoulli(data))
        map_estimates.append(map_bernoulli(data, alpha, beta))

    mle_estimates = np.array(mle_estimates)
    map_estimates = np.array(map_estimates)

    # Sesgo
    bias_mle = np.mean(mle_estimates) - true_theta
    bias_map = np.mean(map_estimates) - true_theta

    # Varianza
    var_mle = np.var(mle_estimates)
    var_map = np.var(map_estimates)

    return bias_mle, var_mle, bias_map, var_map


# %%
print("\n=== Análisis Sesgo-Varianza ===\n")

true_theta = 0.7
n_simulations = 1000
alpha_prior, beta_prior = 5, 5  # Prior centrado en 0.5

print(f"θ verdadero = {true_theta}, Prior = Beta({alpha_prior}, {beta_prior})\n")
print(
    f"{'n':>5} | {'Bias MLE':>10} | {'Var MLE':>10} | {'MSE MLE':>10} | "
    f"{'Bias MAP':>10} | {'Var MAP':>10} | {'MSE MAP':>10}"
)
print("-" * 80)

sample_sizes = [5, 10, 20, 50, 100, 500]
mse_mle_list = []
mse_map_list = []

for n in sample_sizes:
    bias_mle, var_mle, bias_map, var_map = simulate_estimator_bias_variance(
        true_theta, n, n_simulations, alpha_prior, beta_prior
    )
    mse_mle = bias_mle**2 + var_mle
    mse_map = bias_map**2 + var_map
    mse_mle_list.append(mse_mle)
    mse_map_list.append(mse_map)

    print(
        f"{n:>5} | {bias_mle:>10.4f} | {var_mle:>10.4f} | {mse_mle:>10.4f} | "
        f"{bias_map:>10.4f} | {var_map:>10.4f} | {mse_map:>10.4f}"
    )

# Visualizar
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sample_sizes, mse_mle_list, "o-", label="MSE (MLE)", linewidth=2)
ax.plot(sample_sizes, mse_map_list, "s-", label="MSE (MAP)", linewidth=2)
ax.set_xlabel("Tamaño de muestra (n)")
ax.set_ylabel("MSE")
ax.set_title("MSE de MLE vs MAP en función del tamaño de muestra")
ax.legend()
ax.set_xscale("log")
plt.tight_layout()
plt.savefig("../assets/bias_variance_mle_map.png", dpi=150)
plt.show()

print("\n✅ Observación: MAP tiene menor MSE con pocos datos (el prior ayuda),")
print("   pero converge a MLE cuando n es grande (los datos dominan).")


# %% [markdown]
# ## Ejercicios para el Estudiante
#
# ### Ejercicio 1: MLE para Exponencial
# Deriva el MLE para λ en una distribución Exponencial(λ).
# Pista: f(x|λ) = λ exp(-λx)
#
# ### Ejercicio 2: MAP para Gaussiana con prior Gaussiano
# Si x ~ N(μ, σ²) con σ² conocido y prior μ ~ N(μ₀, σ₀²),
# deriva la fórmula del posterior y el MAP.
#
# ### Ejercicio 3: Regularización como MAP
# Demuestra que Ridge Regression (L2) corresponde a MLE
# con prior Gaussiano en los pesos.

# %%
print("\n" + "=" * 70)
print("FIN DEL LAB 1: MLE/MAP Y ESTIMADORES")
print("=" * 70)
