#!/usr/bin/env python3
"""
Lab 3: Cadenas de Markov (Discrete-Time Markov Chains)
=======================================================

Módulo: M04 - Probabilidad y Estadística
Tiempo Estimado: 2-3 horas
Prerequisitos: Álgebra lineal (eigenvalues), probabilidad

Objetivos de Aprendizaje:
-------------------------
1. Construir matrices de transición desde datos
2. Calcular distribuciones estacionarias (analítica y numéricamente)
3. Simular random walks y verificar convergencia
4. Estimar mixing time empíricamente

Referencias:
------------
- Levin & Peres, "Markov Chains and Mixing Times", Cap. 1-4
- Murphy, "ML: A Probabilistic Perspective", Cap. 17
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, matrix_power
from numpy.typing import NDArray

rng = np.random.default_rng(42)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# %% [markdown]
# # Lab 3: Discrete-Time Markov Chains (DTMC)
#
# ## Introducción
#
# Una **Cadena de Markov** es un proceso estocástico donde el estado futuro
# depende solo del estado presente (propiedad de Markov):
#
# P(Xₜ₊₁ = j | Xₜ = i, Xₜ₋₁, ..., X₀) = P(Xₜ₊₁ = j | Xₜ = i) = Pᵢⱼ
#
# ### Aplicaciones en ML
#
# | Aplicación | Uso de Markov Chains |
# |------------|---------------------|
# | **PageRank** | Web como grafo, importancia = distribución estacionaria |
# | **HMM** | Modelos ocultos de Markov para secuencias |
# | **MCMC** | Muestreo de distribuciones complejas |
# | **NLP** | Modelos de lenguaje n-gram |
# | **Reinforcement Learning** | MDP = Markov Decision Process |

# %% [markdown]
# ## Parte 1: Matrices de Transición (30 min)
#
# La matriz de transición P tiene elementos Pᵢⱼ = P(Xₜ₊₁ = j | Xₜ = i).
#
# **Propiedades:**
# - Pᵢⱼ ≥ 0 (no negativas)
# - Σⱼ Pᵢⱼ = 1 (filas suman 1)


# %%
def create_transition_matrix(
    n_states: int, transition_probs: dict | None = None
) -> NDArray[np.float64]:
    """
    Crea una matriz de transición.

    Parámetros
    ----------
    n_states : int
        Número de estados.
    transition_probs : dict, optional
        Diccionario {(i, j): prob}. Si None, genera aleatoria.

    Retorna
    -------
    NDArray
        Matriz de transición (n_states, n_states).
    """
    if transition_probs is None:
        # Generar matriz aleatoria y normalizar filas
        P = rng.random((n_states, n_states))
        P = P / P.sum(axis=1, keepdims=True)
    else:
        P = np.zeros((n_states, n_states))
        for (i, j), prob in transition_probs.items():
            P[i, j] = prob
        # Verificar que filas suman 1
        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1):
            raise ValueError(f"Filas no suman 1: {row_sums}")

    return P


def verify_stochastic_matrix(P: NDArray[np.float64]) -> bool:
    """Verifica que P es una matriz estocástica válida."""
    is_non_negative = bool(np.all(P >= 0))
    rows_sum_to_one = bool(np.allclose(P.sum(axis=1), 1))
    return is_non_negative and rows_sum_to_one


# %% Ejemplo: Modelo del Clima
print("=== Ejemplo: Modelo del Clima ===\n")

# Estados: 0 = Soleado, 1 = Nublado, 2 = Lluvioso
states = ["Soleado", "Nublado", "Lluvioso"]

# Matriz de transición (basada en patrones climáticos típicos)
weather_transitions = {
    (0, 0): 0.7,
    (0, 1): 0.2,
    (0, 2): 0.1,  # De Soleado
    (1, 0): 0.3,
    (1, 1): 0.4,
    (1, 2): 0.3,  # De Nublado
    (2, 0): 0.2,
    (2, 1): 0.3,
    (2, 2): 0.5,  # De Lluvioso
}

P_weather = create_transition_matrix(3, weather_transitions)

print("Matriz de Transición del Clima:")
print(f"{'':>10} {states[0]:>10} {states[1]:>10} {states[2]:>10}")
for i, row in enumerate(P_weather):
    print(f"{states[i]:>10} {row[0]:>10.2f} {row[1]:>10.2f} {row[2]:>10.2f}")

print(f"\n¿Es matriz estocástica válida? {verify_stochastic_matrix(P_weather)}")


# %% [markdown]
# ## Parte 2: Simulación de la Cadena (30 min)


# %%
def simulate_markov_chain(
    P: NDArray[np.float64],
    initial_state: int,
    n_steps: int,
) -> NDArray[np.int64]:
    """
    Simula una cadena de Markov.

    Parámetros
    ----------
    P : NDArray
        Matriz de transición.
    initial_state : int
        Estado inicial.
    n_steps : int
        Número de pasos a simular.

    Retorna
    -------
    NDArray
        Secuencia de estados visitados.
    """
    n_states = P.shape[0]
    states_sequence = np.zeros(n_steps + 1, dtype=np.int64)
    states_sequence[0] = initial_state

    current_state = initial_state
    for t in range(n_steps):
        # Muestrear siguiente estado según probabilidades de transición
        next_state = rng.choice(n_states, p=P[current_state])
        states_sequence[t + 1] = next_state
        current_state = next_state

    return states_sequence


# %% Simular el clima
print("\n=== Simulación del Clima (30 días) ===\n")

trajectory = simulate_markov_chain(P_weather, initial_state=0, n_steps=30)

print("Secuencia de estados:")
print(" → ".join([states[s] for s in trajectory[:15]]))
print(" → ".join([states[s] for s in trajectory[15:]]))

# Contar frecuencias
unique, counts = np.unique(trajectory, return_counts=True)
print("\nFrecuencias empíricas:")
for state, count in zip(unique, counts, strict=True):
    print(f"  {states[state]}: {count/len(trajectory):.2%}")


# %% [markdown]
# ## Parte 3: Distribución Estacionaria (45 min)
#
# La **distribución estacionaria** π satisface:
#
# π = π P  (π es eigenvector izquierdo con eigenvalue 1)
#
# Equivalentemente: πᵀ = Pᵀ πᵀ (eigenvector derecho de Pᵀ)
#
# ### Teorema Ergódico
# Si la cadena es **irreducible** y **aperiódica**, entonces:
# - Existe una única distribución estacionaria π
# - Para cualquier estado inicial: lim_{n→∞} P^n = 1π (convergencia a π)


# %%
def stationary_distribution_analytical(P: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calcula la distribución estacionaria usando eigenvalues.

    π es el eigenvector izquierdo de P con eigenvalue 1,
    equivalente al eigenvector derecho de P.T con eigenvalue 1.
    """
    # Eigenvectors de P.T
    eigenvalues, eigenvectors = eig(P.T)

    # Encontrar eigenvector con eigenvalue ≈ 1
    idx = np.argmin(np.abs(eigenvalues - 1))

    # Eigenvector correspondiente (normalizado)
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()  # Normalizar para que sume 1

    return np.asarray(pi, dtype=np.float64)


def stationary_distribution_power(
    P: NDArray[np.float64],
    n_iterations: int = 100,
) -> NDArray[np.float64]:
    """
    Calcula distribución estacionaria usando iteración de potencias.

    lim_{n→∞} P^n converge a matriz donde todas las filas son π.
    """
    P_power = matrix_power(P, n_iterations)
    # Cualquier fila de P^n es aproximadamente π
    return np.asarray(P_power[0], dtype=np.float64)


def stationary_distribution_simulation(
    P: NDArray[np.float64],
    n_steps: int = 10000,
    initial_state: int = 0,
) -> NDArray[np.float64]:
    """
    Estima distribución estacionaria por simulación.
    """
    trajectory = simulate_markov_chain(P, initial_state, n_steps)

    # Contar frecuencias
    n_states = P.shape[0]
    counts = np.bincount(trajectory, minlength=n_states)
    pi = counts / len(trajectory)

    return pi


# %% Calcular distribución estacionaria del clima
print("\n=== Distribución Estacionaria del Clima ===\n")

pi_analytical = stationary_distribution_analytical(P_weather)
pi_power = stationary_distribution_power(P_weather, n_iterations=100)
pi_simulation = stationary_distribution_simulation(P_weather, n_steps=100000)

print(f"{'Método':>15} | {states[0]:>10} | {states[1]:>10} | {states[2]:>10}")
print("-" * 55)
print(
    f"{'Analítico':>15} | {pi_analytical[0]:>10.4f} | {pi_analytical[1]:>10.4f} | {pi_analytical[2]:>10.4f}"
)
print(
    f"{'Potencias':>15} | {pi_power[0]:>10.4f} | {pi_power[1]:>10.4f} | {pi_power[2]:>10.4f}"
)
print(
    f"{'Simulación':>15} | {pi_simulation[0]:>10.4f} | {pi_simulation[1]:>10.4f} | {pi_simulation[2]:>10.4f}"
)

# Verificar que π = πP
print(
    f"\n✅ Verificación πP = π: {np.allclose(pi_analytical @ P_weather, pi_analytical)}"
)


# %% [markdown]
# ## Parte 4: Convergencia y Mixing Time (45 min)
#
# ### Mixing Time
#
# El **mixing time** es el número de pasos necesarios para que la cadena
# esté "cerca" de la distribución estacionaria.
#
# t_mix(ε) = min{t : ||μₜ - π||_TV ≤ ε}
#
# donde ||·||_TV es la distancia de variación total.


# %%
def total_variation_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """
    Distancia de variación total entre dos distribuciones.

    TV(p, q) = (1/2) Σ |p_i - q_i| = (1/2) ||p - q||_1
    """
    return float(0.5 * np.sum(np.abs(p - q)))


def estimate_mixing_time(
    P: NDArray[np.float64],
    epsilon: float = 0.01,
    max_steps: int = 1000,
) -> tuple[int, list[float]]:
    """
    Estima el mixing time de una cadena de Markov.

    Parámetros
    ----------
    P : NDArray
        Matriz de transición.
    epsilon : float
        Tolerancia para convergencia.
    max_steps : int
        Máximo de pasos a simular.

    Retorna
    -------
    tuple[int, list[float]]
        - Mixing time estimado
        - Lista de distancias TV en cada paso
    """
    pi = stationary_distribution_analytical(P)
    n_states = P.shape[0]

    # Empezar desde distribución concentrada en estado 0
    mu = np.zeros(n_states)
    mu[0] = 1.0

    distances: list[float] = []
    mixing_time = max_steps

    for t in range(max_steps):
        mu = mu @ P  # Evolucionar distribución
        tv_dist = float(total_variation_distance(mu, pi))
        distances.append(tv_dist)

        if tv_dist <= epsilon and mixing_time == max_steps:
            mixing_time = t + 1

    return mixing_time, distances


# %% Analizar convergencia del modelo del clima
print("\n=== Análisis de Convergencia ===\n")

mixing_time, tv_distances = estimate_mixing_time(P_weather, epsilon=0.01)
print(f"Mixing time (ε=0.01): {mixing_time} pasos")

# Visualizar convergencia
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Evolución de la distribución
P_power = np.eye(3)
distributions_list: list[NDArray[np.float64]] = [P_power[0].copy()]
for _ in range(20):
    P_power = P_power @ P_weather
    distributions_list.append(P_power[0].copy())

distributions: NDArray[np.float64] = np.array(distributions_list, dtype=np.float64)

for i, state in enumerate(states):
    axes[0].plot(distributions[:, i], "o-", label=state)
axes[0].axhline(pi_analytical[0], color="blue", linestyle="--", alpha=0.5)
axes[0].axhline(pi_analytical[1], color="orange", linestyle="--", alpha=0.5)
axes[0].axhline(pi_analytical[2], color="green", linestyle="--", alpha=0.5)
axes[0].set_xlabel("Paso t")
axes[0].set_ylabel("P(Xₜ = estado | X₀ = Soleado)")
axes[0].set_title("Convergencia a Distribución Estacionaria")
axes[0].legend()

# Distancia TV
axes[1].semilogy(tv_distances[:50], "b-", linewidth=2)
axes[1].axhline(0.01, color="red", linestyle="--", label="ε = 0.01")
axes[1].axvline(
    mixing_time, color="green", linestyle="--", label=f"t_mix = {mixing_time}"
)
axes[1].set_xlabel("Paso t")
axes[1].set_ylabel("Distancia de Variación Total")
axes[1].set_title("Convergencia en Distancia TV")
axes[1].legend()

plt.tight_layout()
plt.savefig("../assets/markov_convergence.png", dpi=150)
plt.show()


# %% [markdown]
# ## Parte 5: Propiedades de la Cadena (30 min)
#
# ### Irreducibilidad
# Una cadena es **irreducible** si es posible llegar de cualquier estado
# a cualquier otro en un número finito de pasos.
#
# ### Aperiodicidad
# Un estado tiene período d si solo puede volver a sí mismo en
# múltiplos de d pasos. La cadena es **aperiódica** si todos los
# estados tienen período 1.


# %%
def is_irreducible(P: NDArray[np.float64], max_power: int = 100) -> bool:
    """
    Verifica si la cadena es irreducible.

    Una cadena es irreducible si (I + P)^n > 0 para algún n grande.
    """
    n = P.shape[0]
    # (I + P)^n tiene todas las entradas positivas si es irreducible
    Q = np.eye(n) + P
    Q_power = matrix_power(Q, max_power)
    return bool(np.all(Q_power > 0))


def estimate_period(P: NDArray[np.float64], state: int, max_steps: int = 100) -> int:
    """
    Estima el período de un estado.

    El período es el GCD de todos los tiempos de retorno posibles.
    """
    from functools import reduce
    from math import gcd

    return_times = []

    for t in range(1, max_steps + 1):
        P_t = matrix_power(P, t)
        if P_t[state, state] > 0:
            return_times.append(t)

    if not return_times:
        return 0  # No puede volver

    return int(reduce(gcd, return_times))


# %% Verificar propiedades
print("\n=== Propiedades de la Cadena del Clima ===\n")

print(f"¿Es irreducible? {is_irreducible(P_weather)}")
for i, state in enumerate(states):
    period = estimate_period(P_weather, i)
    print(f"Período de '{state}': {period}")

print(
    "\n✅ La cadena es irreducible y aperiódica → existe π única y la cadena converge."
)


# %% [markdown]
# ## Parte 6: Ejemplo Avanzado - PageRank (30 min)
#
# PageRank modela la web como una cadena de Markov donde los estados
# son páginas y las transiciones son links.


# %%
def pagerank(
    adjacency_matrix: NDArray[np.int64],
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Calcula PageRank usando power iteration.

    Parámetros
    ----------
    adjacency_matrix : NDArray
        Matriz de adyacencia (A[i,j] = 1 si hay link de i a j).
    damping : float
        Factor de amortiguamiento (típicamente 0.85).
    max_iter : int
        Máximo de iteraciones.
    tol : float
        Tolerancia para convergencia.

    Retorna
    -------
    NDArray
        Vector de PageRank normalizado.
    """
    n_pages = adjacency_matrix.shape[0]

    # Crear matriz de transición
    out_degree = adjacency_matrix.sum(axis=1)
    # Manejar nodos sin links salientes (dangling nodes)
    out_degree[out_degree == 0] = 1
    P = adjacency_matrix / out_degree[:, np.newaxis]

    # Matriz de Google: G = d*P + (1-d)/n * 1
    # Esto asegura que la cadena sea irreducible y aperiódica
    teleport = np.ones((n_pages, n_pages)) / n_pages
    G = damping * P + (1 - damping) * teleport

    # Power iteration
    rank = np.ones(n_pages) / n_pages

    for _ in range(max_iter):
        new_rank = rank @ G
        if np.linalg.norm(new_rank - rank) < tol:
            break
        rank = new_rank

    return np.asarray(rank / rank.sum(), dtype=np.float64)


# %% Ejemplo de PageRank
print("\n=== Ejemplo: PageRank Simplificado ===\n")

# Pequeña red de 4 páginas
# A → B, A → C
# B → C
# C → A
# D → B, D → C

pages = ["A", "B", "C", "D"]
adj = np.array(
    [
        [0, 1, 1, 0],  # A links to B, C
        [0, 0, 1, 0],  # B links to C
        [1, 0, 0, 0],  # C links to A
        [0, 1, 1, 0],  # D links to B, C
    ]
)

ranks = pagerank(adj, damping=0.85)

print("Estructura de links:")
print("  A → B, C")
print("  B → C")
print("  C → A")
print("  D → B, C")

print("\nPageRank:")
for page, rank_val in sorted(zip(pages, ranks, strict=True), key=lambda x: -x[1]):
    print(f"  {page}: {rank_val:.4f}")

print("\n💡 Nota: A tiene el mayor rank porque C (popular) apunta solo a A.")


# %% [markdown]
# ## Ejercicios para el Estudiante
#
# ### Ejercicio 1: Cadena Periódica
# Crea una cadena de 2 estados que sea periódica (período 2).
# Verifica que NO converge a una distribución estacionaria única.
#
# ### Ejercicio 2: Tiempo de Primera Pasada
# Calcula el tiempo esperado de primera llegada al estado "Lluvioso"
# empezando desde "Soleado" en el modelo del clima.
#
# ### Ejercicio 3: Cadena Absorbente
# Modifica el modelo del clima para que "Lluvioso" sea un estado absorbente
# (una vez lluvioso, siempre lluvioso). Calcula la probabilidad de absorción.

# %%
print("\n" + "=" * 70)
print("FIN DEL LAB 3: CADENAS DE MARKOV")
print("=" * 70)
