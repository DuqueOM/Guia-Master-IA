#!/usr/bin/env python3
"""
Lab 2: Monte Carlo y MCMC (Markov Chain Monte Carlo)
=====================================================

Módulo: M04 - Probabilidad y Estadística
Tiempo Estimado: 3-4 horas
Prerequisitos: Probabilidad básica, distribuciones

Objetivos de Aprendizaje:
-------------------------
1. Entender el método de Monte Carlo para integración
2. Implementar Metropolis-Hastings desde cero
3. Implementar Gibbs Sampling para distribución bivariada
4. Diagnosticar convergencia con trace plots y R-hat

Referencias:
------------
- Murphy, "ML: A Probabilistic Perspective", Cap. 24
- Bishop, "Pattern Recognition and ML", Cap. 11
"""
from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import multivariate_normal

# %% [markdown]
# # Lab 2: Monte Carlo y MCMC
#
# ## Introducción
#
# Los métodos de Monte Carlo nos permiten aproximar integrales y muestrear
# de distribuciones complejas. MCMC extiende esto para distribuciones donde
# no podemos muestrear directamente.
#
# ### ¿Por qué es importante?
#
# | Aplicación | Uso de MCMC |
# |------------|-------------|
# | Inferencia Bayesiana | Muestrear del posterior P(θ|D) |
# | Modelos Gráficos | Inferencia en redes bayesianas |
# | Deep Learning | Dropout (aproximación Monte Carlo) |
# | Física Estadística | Simulación de sistemas complejos |
# %% Imports

rng = np.random.default_rng(42)

# Configuración de plots
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %% [markdown]
# ## Parte 1: Monte Carlo Simple (30 min)
#
# ### 1.1 Estimación de π
#
# Usamos el hecho de que un círculo de radio 1 inscrito en un cuadrado
# de lado 2 tiene área = π.
#
# Si lanzamos puntos aleatorios uniformemente en el cuadrado [-1,1]×[-1,1],
# la proporción que cae dentro del círculo es π/4.


# %%
def estimate_pi_monte_carlo(n_samples: int) -> tuple[float, float]:
    """
    Estima π usando Monte Carlo.

    Parámetros
    ----------
    n_samples : int
        Número de puntos aleatorios.

    Retorna
    -------
    tuple[float, float]
        - Estimación de π
        - Error estándar de la estimación
    """
    # Generar puntos uniformes en [-1, 1] × [-1, 1]
    x = rng.uniform(-1, 1, n_samples)
    y = rng.uniform(-1, 1, n_samples)

    # Contar puntos dentro del círculo unitario
    inside_circle = (x**2 + y**2) <= 1

    # Proporción × 4 = estimación de π
    pi_estimate = 4 * np.mean(inside_circle)

    # Error estándar: SE = σ / √n
    std_error = 4 * np.std(inside_circle) / np.sqrt(n_samples)

    return pi_estimate, std_error


# Demostración
print("=== Estimación de π con Monte Carlo ===\n")
for n in [100, 1000, 10000, 100000]:
    pi_est, se = estimate_pi_monte_carlo(n)
    error = abs(pi_est - np.pi)
    print(f"n={n:>6}: π ≈ {pi_est:.6f} ± {se:.6f} (error = {error:.6f})")

print(f"\nValor real: π = {np.pi:.6f}")


# %% [markdown]
# ### 1.2 Integración Monte Carlo
#
# Para calcular ∫f(x)dx sobre un dominio, usamos:
#
# ∫f(x)dx ≈ (Volumen del dominio) × (1/n) × Σf(xᵢ)
#
# donde xᵢ son muestras uniformes del dominio.


# %%
def monte_carlo_integration(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    a: float,
    b: float,
    n_samples: int,
) -> tuple[float, float]:
    """
    Integra f(x) de a hasta b usando Monte Carlo.

    Parámetros
    ----------
    f : Callable
        Función a integrar.
    a, b : float
        Límites de integración.
    n_samples : int
        Número de muestras.

    Retorna
    -------
    tuple[float, float]
        - Estimación de la integral
        - Error estándar
    """
    # Muestrear uniformemente en [a, b]
    x = rng.uniform(a, b, n_samples)

    # Evaluar función
    fx = f(x)

    # Integral = (b-a) × promedio de f(x)
    integral = (b - a) * np.mean(fx)
    std_error = (b - a) * np.std(fx) / np.sqrt(n_samples)

    return integral, std_error


# Ejemplo: ∫₀¹ x² dx = 1/3
def f_cuadrado(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x**2


print("\n=== Integración Monte Carlo ===\n")
print("∫₀¹ x² dx = 1/3 ≈ 0.3333...\n")

for n in [100, 1000, 10000]:
    integral, se = monte_carlo_integration(f_cuadrado, 0, 1, n)
    print(f"n={n:>5}: Integral ≈ {integral:.6f} ± {se:.6f}")


# %% [markdown]
# ## Parte 2: Metropolis-Hastings (60 min)
#
# ### 2.1 El Algoritmo
#
# Metropolis-Hastings nos permite muestrear de una distribución objetivo
# π(x) cuando solo conocemos una función proporcional a ella.
#
# **Algoritmo:**
# 1. Inicializar x₀
# 2. Para t = 1, 2, ..., T:
#    a. Proponer x' ~ q(x'|xₜ₋₁)
#    b. Calcular ratio de aceptación: α = min(1, π(x')q(xₜ₋₁|x') / π(xₜ₋₁)q(x'|xₜ₋₁))
#    c. Aceptar x' con probabilidad α, sino mantener xₜ₋₁
#
# Para proposal simétrico (q(x'|x) = q(x|x')):
# α = min(1, π(x') / π(xₜ₋₁))


# %%
def metropolis_hastings(
    log_target: Callable[[float], float],
    proposal_std: float,
    n_samples: int,
    x_init: float = 0.0,
    burn_in: int = 1000,
) -> tuple[NDArray[np.float64], float]:
    """
    Implementa Metropolis-Hastings con proposal Gaussiano.

    Parámetros
    ----------
    log_target : Callable
        Log de la distribución objetivo (hasta constante).
    proposal_std : float
        Desviación estándar del proposal Gaussiano.
    n_samples : int
        Número de muestras a generar (después de burn-in).
    x_init : float
        Valor inicial.
    burn_in : int
        Muestras a descartar al inicio.

    Retorna
    -------
    tuple[NDArray, float]
        - Muestras de la distribución objetivo
        - Tasa de aceptación
    """
    total_samples = n_samples + burn_in
    samples: NDArray[np.float64] = np.zeros(total_samples, dtype=np.float64)
    samples[0] = x_init

    n_accepted = 0

    for t in range(1, total_samples):
        # Estado actual
        x_current = samples[t - 1]

        # Proponer nuevo estado (proposal Gaussiano simétrico)
        x_proposed = x_current + rng.normal(0, proposal_std)

        # Calcular log-ratio de aceptación
        log_alpha = log_target(x_proposed) - log_target(x_current)

        # Aceptar con probabilidad min(1, α)
        if np.log(rng.uniform()) < log_alpha:
            samples[t] = x_proposed
            n_accepted += 1
        else:
            samples[t] = x_current

    acceptance_rate = n_accepted / total_samples

    return samples[burn_in:], acceptance_rate


# %% [markdown]
# ### 2.2 Ejemplo: Muestrear de una Gaussiana
#
# Objetivo: N(μ=3, σ²=2)

# %%
# Definir log de la distribución objetivo
mu_target = 3.0
sigma_target = np.sqrt(2.0)


def log_normal(x: float) -> float:
    """Log de N(3, 2)."""
    return float(-0.5 * ((x - mu_target) / sigma_target) ** 2)


# Ejecutar M-H
print("\n=== Metropolis-Hastings: Muestreo de N(3, 2) ===\n")

samples_mh, acceptance = metropolis_hastings(
    log_target=log_normal,
    proposal_std=1.5,
    n_samples=10000,
    x_init=0.0,
    burn_in=1000,
)

print(f"Tasa de aceptación: {acceptance:.2%}")
print(f"Media muestral:     {np.mean(samples_mh):.4f} (teórico: {mu_target})")
print(f"Std muestral:       {np.std(samples_mh):.4f} (teórico: {sigma_target:.4f})")

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Trace plot
axes[0].plot(samples_mh[:1000], alpha=0.7)
axes[0].axhline(mu_target, color="red", linestyle="--", label="μ teórico")
axes[0].set_xlabel("Iteración")
axes[0].set_ylabel("x")
axes[0].set_title("Trace Plot (primeras 1000 muestras)")
axes[0].legend()

# Histograma vs distribución teórica
x_range = np.linspace(-2, 8, 200)
axes[1].hist(samples_mh, bins=50, density=True, alpha=0.7, label="Muestras M-H")
axes[1].plot(
    x_range,
    stats.norm.pdf(x_range, mu_target, sigma_target),
    "r-",
    linewidth=2,
    label="N(3, √2) teórica",
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("Densidad")
axes[1].set_title("Histograma vs Distribución Objetivo")
axes[1].legend()

plt.tight_layout()
plt.savefig("../assets/mh_gaussian_sampling.png", dpi=150)
plt.show()


# %% [markdown]
# ### 2.3 Efecto del Proposal Width
#
# El ancho del proposal afecta la eficiencia:
# - Muy pequeño: alta aceptación pero exploración lenta
# - Muy grande: baja aceptación, muchos rechazos
# - Óptimo: ~23-44% de aceptación (teoría para distribuciones unimodales)

# %%
print("\n=== Efecto del Proposal Width ===\n")

proposal_widths = [0.1, 0.5, 1.5, 5.0, 20.0]
results = []

for width in proposal_widths:
    samples, acc = metropolis_hastings(log_normal, width, n_samples=5000, burn_in=500)
    results.append((width, acc, np.mean(samples), np.std(samples)))
    print(
        f"σ_proposal = {width:>4.1f}: Aceptación = {acc:>5.1%}, "
        f"μ̂ = {np.mean(samples):>5.2f}, σ̂ = {np.std(samples):>4.2f}"
    )

print("\n⚠️ Regla práctica: buscar aceptación entre 20-50%")


# %% [markdown]
# ## Parte 3: Gibbs Sampling (45 min)
#
# Gibbs Sampling es un caso especial de M-H donde siempre aceptamos.
# Funciona cuando podemos muestrear de las distribuciones condicionales.
#
# Para (x, y) ~ π(x, y):
# 1. Muestrear x ~ π(x|y)
# 2. Muestrear y ~ π(y|x)
# 3. Repetir


# %%
def gibbs_sampling_bivariate_normal(
    mu: NDArray[np.float64],
    sigma: NDArray[np.float64],
    n_samples: int,
    burn_in: int = 500,
) -> NDArray[np.float64]:
    """
    Gibbs Sampling para distribución normal bivariada.

    Parámetros
    ----------
    mu : NDArray
        Vector de medias [μ₁, μ₂].
    sigma : NDArray
        Matriz de covarianza 2×2.
    n_samples : int
        Número de muestras.
    burn_in : int
        Muestras a descartar.

    Retorna
    -------
    NDArray
        Muestras (n_samples, 2).
    """
    total = n_samples + burn_in
    samples = np.zeros((total, 2))

    # Extraer parámetros
    mu1, mu2 = mu
    sigma1 = np.sqrt(sigma[0, 0])
    sigma2 = np.sqrt(sigma[1, 1])
    rho = sigma[0, 1] / (sigma1 * sigma2)

    # Inicializar
    x, y = 0.0, 0.0

    for t in range(total):
        # Muestrear x | y
        mu_x_given_y = mu1 + rho * (sigma1 / sigma2) * (y - mu2)
        std_x_given_y = sigma1 * np.sqrt(1 - rho**2)
        x = rng.normal(mu_x_given_y, std_x_given_y)

        # Muestrear y | x
        mu_y_given_x = mu2 + rho * (sigma2 / sigma1) * (x - mu1)
        std_y_given_x = sigma2 * np.sqrt(1 - rho**2)
        y = rng.normal(mu_y_given_x, std_y_given_x)

        samples[t] = [x, y]

    return samples[burn_in:]


# %%
print("\n=== Gibbs Sampling: Gaussiana Bivariada ===\n")

# Definir distribución objetivo
mu_biv = np.array([2.0, 5.0])
sigma_biv = np.array([[1.0, 0.8], [0.8, 2.0]])

print("Objetivo: N(μ=[2, 5], Σ=[[1, 0.8], [0.8, 2]])\n")

# Ejecutar Gibbs
samples_gibbs = gibbs_sampling_bivariate_normal(mu_biv, sigma_biv, n_samples=5000)

print(f"Media muestral:  {samples_gibbs.mean(axis=0)}")
print(f"Media teórica:   {mu_biv}")
print(f"\nCov muestral:\n{np.cov(samples_gibbs.T)}")
print(f"\nCov teórica:\n{sigma_biv}")

# Visualizar
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Trace plots
axes[0].plot(samples_gibbs[:500, 0], label="x₁")
axes[0].plot(samples_gibbs[:500, 1], label="x₂")
axes[0].set_xlabel("Iteración")
axes[0].set_ylabel("Valor")
axes[0].set_title("Trace Plots")
axes[0].legend()

# Scatter plot
axes[1].scatter(samples_gibbs[:, 0], samples_gibbs[:, 1], alpha=0.3, s=5)
axes[1].set_xlabel("x₁")
axes[1].set_ylabel("x₂")
axes[1].set_title("Muestras Gibbs (scatter)")

# Contour con muestras
x_grid = np.linspace(-1, 5, 100)
y_grid = np.linspace(1, 9, 100)
X, Y = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X, Y))
rv = multivariate_normal(mu_biv, sigma_biv)
Z = rv.pdf(pos)

axes[2].contour(X, Y, Z, levels=10, cmap="viridis")
axes[2].scatter(
    samples_gibbs[::10, 0],
    samples_gibbs[::10, 1],
    alpha=0.3,
    s=5,
    c="red",
    label="Muestras",
)
axes[2].set_xlabel("x₁")
axes[2].set_ylabel("x₂")
axes[2].set_title("Contornos + Muestras")

plt.tight_layout()
plt.savefig("../assets/gibbs_bivariate.png", dpi=150)
plt.show()


# %% [markdown]
# ## Parte 4: Diagnósticos de Convergencia (30 min)
#
# ### 4.1 Effective Sample Size (ESS)
#
# ESS mide cuántas muestras independientes equivalen a nuestras muestras correlacionadas.
#
# ESS = n / (1 + 2*Σₖ ρₖ)
#
# donde ρₖ es la autocorrelación en lag k.


# %%


def effective_sample_size(samples: NDArray[np.float64]) -> float:
    """
    Calcula el Effective Sample Size.

    Parámetros
    ----------
    samples : NDArray
        Muestras 1D de la cadena MCMC.

    Retorna
    -------
    float
        ESS estimado.
    """
    n = len(samples)
    if n < 2:
        return float(n)

    # Calcular autocorrelación
    mean = np.mean(samples)
    var = np.var(samples)
    if var == 0:
        return float(n)

    # Autocorrelación usando FFT (más eficiente)
    samples_centered = samples - mean
    fft_result = np.fft.fft(samples_centered, n=2 * n)
    acf = np.fft.ifft(fft_result * np.conj(fft_result))[:n].real
    acf = acf / acf[0]

    # Sumar autocorrelaciones hasta que sean insignificantes
    sum_acf = 0.0
    for k in range(1, n):
        if acf[k] < 0.05:
            break
        sum_acf += acf[k]

    ess = n / (1 + 2 * sum_acf)
    return float(max(1.0, ess))


# %%
print("\n=== Effective Sample Size ===\n")

# Comparar diferentes proposal widths
for width in [0.1, 1.5, 10.0]:
    samples, acc = metropolis_hastings(log_normal, width, n_samples=5000, burn_in=500)
    ess = effective_sample_size(samples)
    print(
        f"σ_proposal = {width:>4.1f}: ESS = {ess:>7.1f} / 5000 "
        f"({100*ess/5000:>5.1f}%), Aceptación = {acc:>5.1%}"
    )


# %% [markdown]
# ### 4.2 R-hat (Gelman-Rubin Diagnostic)
#
# R-hat compara la varianza dentro de cadenas vs entre cadenas.
# - R-hat ≈ 1: cadenas han convergido
# - R-hat > 1.1: posibles problemas de convergencia


# %%
def gelman_rubin_rhat(chains: list[NDArray[np.float64]]) -> float:
    """
    Calcula el diagnóstico R-hat de Gelman-Rubin.

    Parámetros
    ----------
    chains : list[NDArray]
        Lista de cadenas MCMC (cada una es 1D array).

    Retorna
    -------
    float
        Valor de R-hat.
    """
    n = min(len(c) for c in chains)  # longitud de cada cadena

    # Medias de cada cadena
    chain_means = np.array([np.mean(c[:n]) for c in chains])

    # Varianzas de cada cadena
    chain_vars = np.array([np.var(c[:n], ddof=1) for c in chains])

    # Media global
    _ = np.mean(chain_means)  # grand_mean for reference

    # Varianza entre cadenas (B)
    B = n * np.var(chain_means, ddof=1)

    # Varianza dentro de cadenas (W)
    W = np.mean(chain_vars)

    # Estimación de varianza
    var_plus = ((n - 1) / n) * W + (1 / n) * B

    # R-hat
    rhat = np.sqrt(var_plus / W) if W > 0 else 1.0

    return float(rhat)


# %%
print("\n=== R-hat Diagnostic ===\n")

# Ejecutar múltiples cadenas con diferentes inicializaciones
n_chains = 4
chains: list[NDArray[np.float64]] = []

for _ in range(n_chains):
    x_init_value = float(rng.uniform(-10, 10))  # Inicialización aleatoria
    samples, _acceptance_rate = metropolis_hastings(
        log_normal, 1.5, n_samples=2000, x_init=x_init_value, burn_in=500
    )
    chains.append(samples)

rhat = gelman_rubin_rhat(chains)
print(f"R-hat = {rhat:.4f}")

if rhat < 1.1:
    print("✅ Cadenas han convergido (R-hat < 1.1)")
else:
    print("⚠️ Posible problema de convergencia (R-hat > 1.1)")


# %% [markdown]
# ## Ejercicios para el Estudiante
#
# ### Ejercicio 1: Muestrear de distribución bimodal
# Implementa M-H para muestrear de una mezcla de Gaussianas:
# π(x) = 0.5*N(-3, 1) + 0.5*N(3, 1)
#
# ### Ejercicio 2: Gibbs para modelo jerárquico
# Implementa Gibbs Sampling para:
# μ ~ N(0, 10)
# xᵢ ~ N(μ, 1) para i = 1, ..., n
# Dado datos x, muestrea del posterior de μ.
#
# ### Ejercicio 3: Comparar proposal distributions
# Compara M-H con:
# - Proposal Gaussiano
# - Proposal Uniforme
# ¿Cuál es más eficiente para una Gaussiana objetivo?

# %% [markdown]
# ## Resumen
#
# | Método | Cuándo usar | Ventajas | Desventajas |
# |--------|-------------|----------|-------------|
# | **Monte Carlo Simple** | Integración numérica | Simple, paralelo | Requiere muestreo directo |
# | **Metropolis-Hastings** | Distribución arbitraria | Flexible | Requiere tuning de proposal |
# | **Gibbs Sampling** | Condicionales conocidas | Sin rechazo | Solo si condicionales fáciles |
#
# ### Diagnósticos clave:
# - **Trace plots**: visual, buscar "mezcla" estable
# - **ESS**: eficiencia del muestreo
# - **R-hat**: convergencia de múltiples cadenas

# %%
print("\n" + "=" * 70)
print("FIN DEL LAB 2: MONTE CARLO Y MCMC")
print("=" * 70)
