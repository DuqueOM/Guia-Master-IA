# Módulo 06 - Cadenas de Markov y Métodos Monte Carlo

> **🎯 Objetivo:** Dominar procesos estocásticos y métodos de muestreo  
> **⭐ PATHWAY LÍNEA 2:** Discrete-Time Markov Chains and Monte Carlo Methods

---

## 🧠 Analogía: Random Walks y Dados Infinitos

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   CADENA DE MARKOV = "El Borracho Caminando"                                │
│   ──────────────────────────────────────────                                │
│                                                                             │
│   Un borracho camina por la calle. En cada esquina:                         │
│   • 50% probabilidad de ir a la izquierda                                   │
│   • 50% probabilidad de ir a la derecha                                     │
│                                                                             │
│   PROPIEDAD CLAVE (Markov):                                                 │
│   Dónde irá SOLO depende de dónde está AHORA.                               │
│   No importa cómo llegó ahí.                                                │
│                                                                             │
│           P=0.5        P=0.5        P=0.5                                   │
│       A ◄──────────▶ B ◄──────────▶ C ◄──────────▶ D                      │
│                                                                             │
│   MONTE CARLO = "Tirar Dados para Calcular π"                               │
│   ────────────────────────────────────────────                              │
│                                                                             │
│   Problema: Calcular área de círculo en un cuadrado                         │
│   Solución: Tirar puntos al azar, contar cuántos caen dentro                │
│                                                                             │
│   ┌───────────┐                                                             │
│   │  • •  ○   │   ○ = dentro del círculo                                    │
│   │ ○  ○○ •   │   • = fuera del círculo                                     │
│   │  ○   ○    │                                                             │
│   │ •  ○  •   │   π ≈ 4 × (puntos dentro / total puntos)                    │
│   └───────────┘                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Cadenas de Markov Discretas](#1-markov-discretas)
2. [Propiedades y Distribución Estacionaria](#2-propiedades)
3. [Algoritmos Basados en Markov](#3-algoritmos)
4. [Métodos Monte Carlo](#4-monte-carlo)
5. [Markov Chain Monte Carlo (MCMC)](#5-mcmc)
6. [Aplicaciones en ML](#6-aplicaciones)

---

## 1. Cadenas de Markov Discretas {#1-markov-discretas}

### 1.1 Definición y Matriz de Transición

```
CADENA DE MARKOV:
─────────────────

Secuencia de estados X₀, X₁, X₂, ... donde:

P(Xₙ₊₁ = j | Xₙ = i, Xₙ₋₁, ..., X₀) = P(Xₙ₊₁ = j | Xₙ = i)

"El futuro solo depende del presente, no del pasado"

MATRIZ DE TRANSICIÓN P:
Pᵢⱼ = P(ir a estado j | estoy en estado i)

• Cada fila suma 1 (probabilidades)
• Pᵢⱼ ≥ 0
```

```python
from typing import List, Dict, Tuple
import random
import math


class MarkovChain:
    """Discrete-time Markov chain implementation.
    
    Example - Weather:
        states = ["sunny", "rainy"]
        transitions = {
            "sunny": {"sunny": 0.8, "rainy": 0.2},
            "rainy": {"sunny": 0.4, "rainy": 0.6}
        }
    """
    
    def __init__(
        self, 
        states: List[str], 
        transition_matrix: Dict[str, Dict[str, float]]
    ) -> None:
        """Initialize Markov chain.
        
        Args:
            states: List of state names
            transition_matrix: P[from_state][to_state] = probability
        """
        self.states = states
        self.transitions = transition_matrix
        self._validate()
    
    def _validate(self) -> None:
        """Validate that rows sum to 1."""
        for state in self.states:
            row_sum = sum(self.transitions[state].values())
            if abs(row_sum - 1.0) > 1e-10:
                raise ValueError(f"Row for state {state} sums to {row_sum}, not 1")
    
    def step(self, current_state: str) -> str:
        """Take one step from current state.
        
        Returns next state sampled from transition probabilities.
        """
        probs = self.transitions[current_state]
        r = random.random()
        
        cumulative = 0.0
        for state, prob in probs.items():
            cumulative += prob
            if r <= cumulative:
                return state
        
        return list(probs.keys())[-1]  # Edge case
    
    def simulate(self, start_state: str, n_steps: int) -> List[str]:
        """Simulate n steps of the Markov chain.
        
        Returns list of states visited.
        """
        trajectory = [start_state]
        current = start_state
        
        for _ in range(n_steps):
            current = self.step(current)
            trajectory.append(current)
        
        return trajectory
    
    def get_matrix(self) -> List[List[float]]:
        """Return transition matrix as 2D list."""
        n = len(self.states)
        matrix = [[0.0] * n for _ in range(n)]
        
        state_to_idx = {s: i for i, s in enumerate(self.states)}
        
        for from_state, to_probs in self.transitions.items():
            i = state_to_idx[from_state]
            for to_state, prob in to_probs.items():
                j = state_to_idx[to_state]
                matrix[i][j] = prob
        
        return matrix


# Example: Weather model
weather_chain = MarkovChain(
    states=["sunny", "rainy"],
    transition_matrix={
        "sunny": {"sunny": 0.8, "rainy": 0.2},
        "rainy": {"sunny": 0.4, "rainy": 0.6}
    }
)
```

### 1.2 Distribución después de n pasos

```python
def matrix_power(P: List[List[float]], n: int) -> List[List[float]]:
    """Compute P^n (transition matrix to the nth power).
    
    P^n[i][j] = probability of being in state j after n steps,
                starting from state i.
    """
    size = len(P)
    
    # Start with identity matrix
    result = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    
    # Matrix multiplication n times
    current = [row[:] for row in P]  # Copy P
    
    while n > 0:
        if n % 2 == 1:
            result = matrix_multiply(result, current)
        current = matrix_multiply(current, current)
        n //= 2
    
    return result


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices."""
    n = len(A)
    result = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    
    return result


def distribution_after_n_steps(
    initial_dist: List[float], 
    P: List[List[float]], 
    n: int
) -> List[float]:
    """Compute probability distribution after n steps.
    
    π(n) = π(0) × P^n
    
    Args:
        initial_dist: Initial probability for each state
        P: Transition matrix
        n: Number of steps
    
    Returns:
        Probability distribution after n steps
    """
    P_n = matrix_power(P, n)
    
    result = [0.0] * len(initial_dist)
    for j in range(len(initial_dist)):
        for i in range(len(initial_dist)):
            result[j] += initial_dist[i] * P_n[i][j]
    
    return result
```

---

## 2. Propiedades y Distribución Estacionaria {#2-propiedades}

### 2.1 Propiedades Importantes

```
PROPIEDADES DE CADENAS DE MARKOV:
─────────────────────────────────

IRREDUCIBLE:
• Desde cualquier estado se puede llegar a cualquier otro
• Un solo componente comunicante

APERIÓDICA:
• El GCD de los ciclos de retorno es 1
• No hay ciclos determinísticos

ERGÓDICA = Irreducible + Aperiódica + Finita
• Tiene distribución estacionaria ÚNICA
• Converge a estacionaria desde cualquier inicio
```

### 2.2 Distribución Estacionaria

```
DISTRIBUCIÓN ESTACIONARIA π:
────────────────────────────

Una distribución π tal que:
π = π × P

"Si empiezo con distribución π, después de un paso
sigo teniendo distribución π"

TEOREMA ERGÓDICO:
Para cadenas ergódicas, sin importar el estado inicial:
lim(n→∞) P^n converge a matriz con filas iguales a π

APLICACIÓN:
• PageRank: π = importancia de cada página web
• Física estadística: π = distribución de equilibrio
```

```python
def find_stationary_power_iteration(
    P: List[List[float]], 
    tolerance: float = 1e-10,
    max_iterations: int = 1000
) -> List[float]:
    """Find stationary distribution using power iteration.
    
    Iteratively multiply by P until convergence.
    
    π × P = π means π is left eigenvector with eigenvalue 1.
    """
    n = len(P)
    
    # Start with uniform distribution
    pi = [1.0 / n] * n
    
    for iteration in range(max_iterations):
        # Compute π × P
        new_pi = [0.0] * n
        for j in range(n):
            for i in range(n):
                new_pi[j] += pi[i] * P[i][j]
        
        # Check convergence
        diff = sum(abs(new_pi[i] - pi[i]) for i in range(n))
        if diff < tolerance:
            return new_pi
        
        pi = new_pi
    
    return pi  # May not have converged


def verify_stationary(pi: List[float], P: List[List[float]]) -> bool:
    """Verify that π is stationary: π × P ≈ π."""
    n = len(pi)
    result = [0.0] * n
    
    for j in range(n):
        for i in range(n):
            result[j] += pi[i] * P[i][j]
    
    diff = sum(abs(result[i] - pi[i]) for i in range(n))
    return diff < 1e-6
```

### 2.3 Tiempo de Mezcla (Mixing Time)

```python
def estimate_mixing_time(
    chain: MarkovChain, 
    epsilon: float = 0.01,
    samples: int = 1000
) -> int:
    """Estimate how many steps until distribution is close to stationary.
    
    Uses empirical simulation.
    """
    P = chain.get_matrix()
    pi = find_stationary_power_iteration(P)
    
    for n in range(1, 1000):
        P_n = matrix_power(P, n)
        
        # Total variation distance from stationary
        max_dist = 0.0
        for i in range(len(chain.states)):
            dist = 0.5 * sum(abs(P_n[i][j] - pi[j]) for j in range(len(pi)))
            max_dist = max(max_dist, dist)
        
        if max_dist < epsilon:
            return n
    
    return -1  # Did not converge
```

---

## 3. Algoritmos Basados en Markov {#3-algoritmos}

### 3.1 PageRank (Google)

```python
def pagerank(
    graph: Dict[str, List[str]], 
    damping: float = 0.85,
    iterations: int = 100
) -> Dict[str, float]:
    """PageRank algorithm - a Markov chain on the web graph.
    
    Random surfer model:
    - With probability d, follow a random outgoing link
    - With probability 1-d, jump to a random page
    
    Args:
        graph: Adjacency list (page -> list of linked pages)
        damping: Probability of following a link (typically 0.85)
        iterations: Number of power iterations
    
    Returns:
        PageRank score for each page
    """
    pages = list(graph.keys())
    n = len(pages)
    page_to_idx = {p: i for i, p in enumerate(pages)}
    
    # Initialize uniform
    rank = {page: 1.0 / n for page in pages}
    
    for _ in range(iterations):
        new_rank = {}
        
        for page in pages:
            # Teleport contribution (random jump)
            score = (1 - damping) / n
            
            # Link contribution
            for other_page in pages:
                if page in graph.get(other_page, []):
                    out_degree = len(graph[other_page])
                    if out_degree > 0:
                        score += damping * rank[other_page] / out_degree
            
            new_rank[page] = score
        
        rank = new_rank
    
    return rank


# Example
web_graph = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C"]
}
# scores = pagerank(web_graph)
```

### 3.2 Random Walk para Búsqueda en Grafos

```python
def random_walk_similarity(
    graph: Dict[str, List[str]], 
    node1: str, 
    node2: str,
    walk_length: int = 100,
    num_walks: int = 1000
) -> float:
    """Estimate similarity between nodes via random walks.
    
    Probability of reaching node2 from node1 via random walk.
    """
    hits = 0
    
    for _ in range(num_walks):
        current = node1
        
        for _ in range(walk_length):
            neighbors = graph.get(current, [])
            if not neighbors:
                break
            current = random.choice(neighbors)
            
            if current == node2:
                hits += 1
                break
    
    return hits / num_walks
```

---

## 4. Métodos Monte Carlo {#4-monte-carlo}

### 4.1 Integración Monte Carlo

```python
def monte_carlo_integration(
    f: callable, 
    a: float, 
    b: float, 
    n_samples: int = 10000
) -> float:
    """Estimate ∫[a,b] f(x) dx using Monte Carlo.
    
    Estimate = (b-a) × (1/n) × Σ f(xᵢ)
    
    where xᵢ ~ Uniform(a, b)
    """
    total = 0.0
    
    for _ in range(n_samples):
        x = random.uniform(a, b)
        total += f(x)
    
    return (b - a) * total / n_samples


def estimate_pi(n_samples: int = 100000) -> float:
    """Estimate π using Monte Carlo.
    
    Ratio of points inside quarter circle to total points.
    Area of quarter circle = πr²/4, area of square = r²
    π = 4 × (points inside) / (total points)
    """
    inside = 0
    
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        
        if x*x + y*y <= 1:
            inside += 1
    
    return 4 * inside / n_samples
```

### 4.2 Muestreo por Importancia (Importance Sampling)

```python
def importance_sampling(
    f: callable,           # Function to integrate
    p: callable,           # Proposal distribution density
    sample_p: callable,    # Function to sample from p
    n_samples: int = 10000
) -> float:
    """Importance sampling for E_q[f(x)] where q is hard to sample.
    
    E_q[f(x)] = E_p[f(x) × q(x)/p(x)]
    
    where p is easy to sample from.
    """
    total = 0.0
    
    for _ in range(n_samples):
        x = sample_p()
        # Weight = q(x) / p(x), here we assume q is the target
        weight = 1.0  # Simplified; actual requires q(x)/p(x) ratio
        total += f(x) * weight
    
    return total / n_samples
```

---

## 5. Markov Chain Monte Carlo (MCMC) {#5-mcmc}

### 5.1 Por Qué MCMC

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   EL PROBLEMA DE MUESTREO                                                   │
│   ───────────────────────                                                   │
│                                                                             │
│   Queremos muestras de una distribución P(x)                                │
│   pero:                                                                     │
│   • No podemos calcular P(x) directamente                                   │
│   • Solo conocemos P(x) ∝ f(x) (hasta una constante)                        │
│   • El espacio es de alta dimensión                                         │
│                                                                             │
│   SOLUCIÓN: MCMC                                                            │
│   ───────────────                                                           │
│   Construir una cadena de Markov cuya distribución estacionaria             │
│   sea exactamente la distribución que queremos muestrear.                   │
│                                                                             │
│   Después de "burn-in", las muestras de la cadena                           │
│   son muestras aproximadas de P(x).                                         │
│                                                                             │
│   APLICACIONES:                                                             │
│   • Inferencia Bayesiana (posterior sampling)                               │
│   • Modelos generativos                                                     │
│   • Física estadística                                                      │
│   • Optimización estocástica                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Algoritmo Metropolis-Hastings

```python
def metropolis_hastings(
    target_log_prob: callable,  # Log of target distribution (up to constant)
    proposal_sample: callable,  # Sample from proposal q(x'|x)
    initial_state: List[float],
    n_samples: int = 10000,
    burn_in: int = 1000
) -> List[List[float]]:
    """Metropolis-Hastings MCMC sampler.
    
    1. Propose x' ~ q(x'|x)
    2. Accept with probability min(1, [P(x')/P(x)] × [q(x|x')/q(x'|x)])
    3. If symmetric proposal (q(x|x') = q(x'|x)), acceptance = min(1, P(x')/P(x))
    
    Args:
        target_log_prob: Log probability of target (can be unnormalized)
        proposal_sample: Function that proposes new state given current
        initial_state: Starting point
        n_samples: Number of samples to collect (after burn-in)
        burn_in: Number of initial samples to discard
    
    Returns:
        List of samples from target distribution
    """
    samples = []
    current = initial_state
    current_log_prob = target_log_prob(current)
    
    total_iterations = n_samples + burn_in
    accepted = 0
    
    for i in range(total_iterations):
        # Propose new state
        proposed = proposal_sample(current)
        proposed_log_prob = target_log_prob(proposed)
        
        # Acceptance ratio (log scale for numerical stability)
        # Assuming symmetric proposal
        log_acceptance = proposed_log_prob - current_log_prob
        
        # Accept or reject
        if math.log(random.random()) < log_acceptance:
            current = proposed
            current_log_prob = proposed_log_prob
            accepted += 1
        
        # Collect sample (after burn-in)
        if i >= burn_in:
            samples.append(current[:])  # Copy
    
    acceptance_rate = accepted / total_iterations
    print(f"Acceptance rate: {acceptance_rate:.2%}")
    
    return samples


# Example: Sample from 2D Gaussian
def gaussian_log_prob(x: List[float]) -> float:
    """Log probability of standard 2D Gaussian."""
    return -0.5 * (x[0]**2 + x[1]**2)


def gaussian_proposal(x: List[float]) -> List[float]:
    """Random walk proposal."""
    step_size = 0.5
    return [xi + random.gauss(0, step_size) for xi in x]


# samples = metropolis_hastings(gaussian_log_prob, gaussian_proposal, [0.0, 0.0])
```

### 5.3 Gibbs Sampling

```python
def gibbs_sampling_2d(
    conditional_x: callable,  # Sample x given y
    conditional_y: callable,  # Sample y given x
    initial: Tuple[float, float],
    n_samples: int = 10000,
    burn_in: int = 1000
) -> List[Tuple[float, float]]:
    """Gibbs sampling for 2D distribution.
    
    Instead of proposing both coordinates at once,
    sample each coordinate from its conditional distribution.
    
    Always accepts! No rejection step.
    
    Requirements:
    - Must be able to sample from P(x|y) and P(y|x)
    """
    samples = []
    x, y = initial
    
    for i in range(n_samples + burn_in):
        # Sample x given current y
        x = conditional_x(y)
        
        # Sample y given new x
        y = conditional_y(x)
        
        if i >= burn_in:
            samples.append((x, y))
    
    return samples


# Example: Bivariate normal with correlation
def sample_x_given_y(y: float, rho: float = 0.8) -> float:
    """Sample x | y for bivariate normal with correlation rho."""
    mean = rho * y
    std = math.sqrt(1 - rho**2)
    return random.gauss(mean, std)


def sample_y_given_x(x: float, rho: float = 0.8) -> float:
    """Sample y | x for bivariate normal with correlation rho."""
    mean = rho * x
    std = math.sqrt(1 - rho**2)
    return random.gauss(mean, std)
```

---

## 6. Aplicaciones en ML {#6-aplicaciones}

### 6.1 Inferencia Bayesiana con MCMC

```python
def bayesian_linear_regression_mcmc(
    X: List[List[float]], 
    y: List[float],
    n_samples: int = 5000
) -> List[List[float]]:
    """Bayesian linear regression using MCMC.
    
    Prior: weights ~ Normal(0, 1)
    Likelihood: y ~ Normal(Xw, σ²)
    
    Sample from posterior P(w|X,y).
    """
    n_features = len(X[0])
    
    def log_posterior(weights: List[float]) -> float:
        # Log prior: N(0, 1) for each weight
        log_prior = -0.5 * sum(w**2 for w in weights)
        
        # Log likelihood
        sigma = 1.0  # Assume known for simplicity
        log_likelihood = 0.0
        for xi, yi in zip(X, y):
            pred = sum(w * x for w, x in zip(weights, xi))
            log_likelihood += -0.5 * ((yi - pred) / sigma) ** 2
        
        return log_prior + log_likelihood
    
    def proposal(weights: List[float]) -> List[float]:
        return [w + random.gauss(0, 0.1) for w in weights]
    
    initial = [0.0] * n_features
    
    return metropolis_hastings(log_posterior, proposal, initial, n_samples)
```

### 6.2 Hidden Markov Models (HMM)

```python
class HiddenMarkovModel:
    """Simple Hidden Markov Model.
    
    Hidden states follow Markov chain.
    Observations depend only on current hidden state.
    
    Used for: speech recognition, sequence labeling, financial modeling.
    """
    
    def __init__(
        self,
        states: List[str],
        observations: List[str],
        transition_probs: Dict[str, Dict[str, float]],
        emission_probs: Dict[str, Dict[str, float]],
        initial_probs: Dict[str, float]
    ) -> None:
        self.states = states
        self.observations = observations
        self.A = transition_probs    # P(next_state | current_state)
        self.B = emission_probs      # P(observation | state)
        self.pi = initial_probs      # P(initial_state)
    
    def generate_sequence(self, length: int) -> Tuple[List[str], List[str]]:
        """Generate hidden states and observations."""
        hidden = []
        observed = []
        
        # Initial state
        state = self._sample_from_dist(self.pi)
        
        for _ in range(length):
            hidden.append(state)
            
            # Emit observation
            obs = self._sample_from_dist(self.B[state])
            observed.append(obs)
            
            # Transition to next state
            state = self._sample_from_dist(self.A[state])
        
        return hidden, observed
    
    def _sample_from_dist(self, dist: Dict[str, float]) -> str:
        """Sample from a discrete distribution."""
        r = random.random()
        cumulative = 0.0
        for item, prob in dist.items():
            cumulative += prob
            if r <= cumulative:
                return item
        return list(dist.keys())[-1]
    
    def forward_algorithm(self, observations: List[str]) -> float:
        """Compute P(observations | model) using forward algorithm.
        
        Dynamic programming to avoid exponential computation.
        """
        T = len(observations)
        
        # Alpha[t][state] = P(O₁...Oₜ, Sₜ = state)
        alpha = [{} for _ in range(T)]
        
        # Initialize
        for state in self.states:
            alpha[0][state] = self.pi[state] * self.B[state][observations[0]]
        
        # Forward pass
        for t in range(1, T):
            for state in self.states:
                prob = sum(
                    alpha[t-1][prev_state] * self.A[prev_state][state]
                    for prev_state in self.states
                )
                alpha[t][state] = prob * self.B[state][observations[t]]
        
        # Total probability
        return sum(alpha[T-1][state] for state in self.states)
```

---

## ⚠️ Conceptos Clave

### Convergence Diagnostics

```
¿CÓMO SABER SI MCMC CONVERGIÓ?
────────────────────────────────

1. TRACE PLOTS: Visualizar muestras vs iteración
   - Debe parecer "ruido estacionario"
   - No tendencias ni patrones

2. MULTIPLE CHAINS: Correr varias cadenas desde diferentes inicios
   - Deben mezclarse (coincidir)

3. AUTOCORRELATION: Muestras consecutivas correlacionadas
   - Usar "thinning" (tomar cada k-ésima muestra)

4. EFFECTIVE SAMPLE SIZE: Muestras independientes efectivas
   - ESS << N indica alta autocorrelación
```

### Burn-in y Thinning

```python
def process_mcmc_samples(
    raw_samples: List[List[float]],
    burn_in_fraction: float = 0.2,
    thinning: int = 10
) -> List[List[float]]:
    """Post-process MCMC samples.
    
    1. Discard burn-in (initial samples before convergence)
    2. Thin (reduce autocorrelation)
    """
    n_samples = len(raw_samples)
    burn_in = int(n_samples * burn_in_fraction)
    
    # Remove burn-in and thin
    processed = raw_samples[burn_in::thinning]
    
    return processed
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 21.1: Weather Markov Chain
Simular 365 días de clima con la cadena sunny/rainy.

### Ejercicio 21.2: Estimate π
Usar Monte Carlo para estimar π con 1M puntos.

### Ejercicio 21.3: Sample from Mixture
Usar Metropolis-Hastings para muestrear de mezcla de Gaussianas.

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [MCMC for ML](https://www.coursera.org/learn/bayesian-statistics) | Curso | 🔴 Obligatorio |
| [PageRank Explained](https://www.youtube.com/watch?v=P8Kt6Abq_rM) | Video | 🟡 Recomendado |
| [Markov Chains (3B1B)](https://www.youtube.com/watch?v=i3AkTO9HLXo) | Video | 🔴 Obligatorio |

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [20_ESTADISTICA_INFERENCIAL](20_ESTADISTICA_INFERENCIAL.md) | [00_INDICE](00_INDICE.md) | [22_ML_SUPERVISADO](22_ML_SUPERVISADO.md) |
