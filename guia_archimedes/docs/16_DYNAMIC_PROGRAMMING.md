# Anexo DSA - Dynamic Programming

> **⚠️ MÓDULO OPCIONAL:** Este módulo NO es requerido para el Pathway. Es útil para entrevistas técnicas.  
> **🎯 Objetivo:** Dominar la técnica de DP para resolver problemas de optimización.

---

## 🧠 Analogía: No Calcular lo Mismo Dos Veces

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   DYNAMIC PROGRAMMING = Recordar para no recalcular                         │
│   ─────────────────────────────────────────────────                         │
│                                                                             │
│   FIBONACCI NAIVE:                                                          │
│                fib(5)                                                       │
│               /      \                                                      │
│          fib(4)      fib(3)          ← fib(3) se calcula 2 veces!           │
│          /    \       /   \                                                 │
│      fib(3) fib(2) fib(2) fib(1)     ← fib(2) se calcula 3 veces!           │
│       ...                                                                   │
│                                                                             │
│   DYNAMIC PROGRAMMING:                                                      │
│   ┌───────────────────────────────┐                                         │
│   │ Ya calculé fib(3)? → Buscar   │                                         │
│   │ No calculado? → Calcular y    │                                         │
│   │                 GUARDAR       │                                         │
│   └───────────────────────────────┘                                         │
│                                                                             │
│   REQUISITOS para usar DP:                                                  │
│   1. OPTIMAL SUBSTRUCTURE: Solución óptima usa soluciones óptimas de        │
│      subproblemas                                                           │
│   2. OVERLAPPING SUBPROBLEMS: Los mismos subproblemas se repiten            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Conceptos Fundamentales](#1-conceptos)
2. [Top-Down (Memoization)](#2-top-down)
3. [Bottom-Up (Tabulation)](#3-bottom-up)
4. [Problemas Clásicos](#4-clasicos)
5. [Framework para Resolver DP](#5-framework)

---

## 1. Conceptos Fundamentales {#1-conceptos}

### 1.1 ¿Qué es Dynamic Programming?

```
DP = Técnica de optimización que:
1. Divide problema en subproblemas
2. Resuelve cada subproblema UNA SOLA VEZ
3. Guarda resultados para reusar

NO es:
- Un algoritmo específico
- Solo memorización
- Aplicable a cualquier problema
```

### 1.2 Dos Enfoques

| Top-Down (Memoization) | Bottom-Up (Tabulation) |
|------------------------|------------------------|
| Recursivo + cache | Iterativo + tabla |
| Empieza del problema grande | Empieza de casos base |
| Solo calcula lo necesario | Calcula todo |
| Más intuitivo | Más eficiente (no call stack) |

---

## 2. Top-Down (Memoization) {#2-top-down}

### 2.1 Fibonacci con Memoization

```python
def fibonacci_memo(n: int, memo: dict[int, int] | None = None) -> int:
    """Fibonacci with memoization (top-down DP).
    
    Time: O(n) - each value computed once
    Space: O(n) - for memo dict + call stack
    
    Example:
        >>> fibonacci_memo(10)
        55
    """
    if memo is None:
        memo = {}
    
    # Check cache first
    if n in memo:
        return memo[n]
    
    # Base cases
    if n <= 1:
        return n
    
    # Compute and cache
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


# Con decorator
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_lru(n: int) -> int:
    """Fibonacci with automatic memoization."""
    if n <= 1:
        return n
    return fibonacci_lru(n - 1) + fibonacci_lru(n - 2)
```

### 2.2 Template Top-Down

```python
def solve_top_down(problem):
    memo = {}
    
    def dp(state):
        # 1. Check cache
        if state in memo:
            return memo[state]
        
        # 2. Base case
        if is_base_case(state):
            return base_value
        
        # 3. Recurrence relation
        result = combine(dp(smaller_states))
        
        # 4. Cache and return
        memo[state] = result
        return result
    
    return dp(initial_state)
```

---

## 3. Bottom-Up (Tabulation) {#3-bottom-up}

### 3.1 Fibonacci Bottom-Up

```python
def fibonacci_bottom_up(n: int) -> int:
    """Fibonacci with tabulation (bottom-up DP).
    
    Builds solution from base cases up.
    
    Time: O(n)
    Space: O(n) for the table
    
    Example:
        >>> fibonacci_bottom_up(10)
        55
    """
    if n <= 1:
        return n
    
    # Table to store computed values
    dp = [0] * (n + 1)
    
    # Base cases
    dp[0] = 0
    dp[1] = 1
    
    # Fill table from base cases up
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]


def fibonacci_optimized(n: int) -> int:
    """Fibonacci with O(1) space.
    
    Only need previous two values.
    """
    if n <= 1:
        return n
    
    prev2 = 0  # fib(i-2)
    prev1 = 1  # fib(i-1)
    
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

### 3.2 Template Bottom-Up

```python
def solve_bottom_up(problem):
    # 1. Define table size and initialize
    dp = initialize_table(problem_size)
    
    # 2. Set base cases
    dp[base_indices] = base_values
    
    # 3. Fill table iteratively
    for state in all_states_in_order:
        dp[state] = combine(dp[smaller_states])
    
    # 4. Return final answer
    return dp[final_state]
```

---

## 4. Problemas Clásicos {#4-clasicos}

### 4.1 Climbing Stairs

```python
def climb_stairs(n: int) -> int:
    """Number of ways to climb n stairs (1 or 2 steps at a time).
    
    Recurrence: ways(n) = ways(n-1) + ways(n-2)
    (Same as Fibonacci!)
    
    Example:
        >>> climb_stairs(4)
        5  # [1,1,1,1], [1,1,2], [1,2,1], [2,1,1], [2,2]
    """
    if n <= 2:
        return n
    
    prev2 = 1  # ways to reach step 1
    prev1 = 2  # ways to reach step 2
    
    for _ in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

### 4.2 Coin Change (Minimum Coins)

```python
def coin_change(coins: list[int], amount: int) -> int:
    """Find minimum coins needed to make amount.
    
    Classic DP problem.
    
    Recurrence:
        dp[a] = min(dp[a - coin] + 1) for all coins where coin <= a
    
    Example:
        >>> coin_change([1, 2, 5], 11)
        3  # 5 + 5 + 1
    
    Time: O(amount * len(coins))
    Space: O(amount)
    """
    # dp[i] = minimum coins to make amount i
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 0 coins to make amount 0
    
    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a and dp[a - coin] != float('inf'):
                dp[a] = min(dp[a], dp[a - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

### 4.3 Longest Common Subsequence (LCS)

```python
def lcs(text1: str, text2: str) -> int:
    """Find length of longest common subsequence.
    
    Subsequence: characters in same order but not necessarily contiguous.
    
    Example:
        >>> lcs("abcde", "ace")
        3  # "ace"
    
    Recurrence:
        If text1[i] == text2[j]: dp[i][j] = dp[i-1][j-1] + 1
        Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    Time: O(m * n)
    Space: O(m * n)
    """
    m, n = len(text1), len(text2)
    
    # dp[i][j] = LCS of text1[:i] and text2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]
```

### 4.4 0/1 Knapsack

```python
def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    """0/1 Knapsack: maximize value within weight capacity.
    
    Each item can be taken at most once.
    
    Example:
        >>> knapsack([1, 2, 3], [6, 10, 12], 5)
        22  # items with weight 2 and 3
    
    Recurrence:
        dp[i][w] = max(
            dp[i-1][w],                        # don't take item i
            dp[i-1][w-weight[i]] + value[i]    # take item i
        )
    
    Time: O(n * capacity)
    Space: O(n * capacity) or O(capacity) optimized
    """
    n = len(weights)
    
    # dp[i][w] = max value using first i items with capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i
            dp[i][w] = dp[i - 1][w]
            
            # Take item i (if it fits)
            if weights[i - 1] <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i - 1][w - weights[i - 1]] + values[i - 1]
                )
    
    return dp[n][capacity]
```

### 4.5 Maximum Subarray (Kadane's Algorithm)

```python
def max_subarray(nums: list[int]) -> int:
    """Find contiguous subarray with maximum sum.
    
    Kadane's Algorithm - clever DP.
    
    Recurrence:
        max_ending_here = max(num, max_ending_here + num)
    
    Example:
        >>> max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4])
        6  # [4, -1, 2, 1]
    
    Time: O(n)
    Space: O(1)
    """
    max_sum = nums[0]
    current_sum = nums[0]
    
    for num in nums[1:]:
        # Either extend current subarray or start new
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum
```

---

## 5. Framework para Resolver DP {#5-framework}

### 5.1 Pasos para Resolver

```
1. IDENTIFICAR: ¿Es un problema de DP?
   - ¿Tiene optimal substructure?
   - ¿Hay overlapping subproblems?
   - ¿Pide optimizar algo o contar combinaciones?

2. DEFINIR ESTADO:
   - ¿Qué representa dp[i] o dp[i][j]?
   - ¿Qué información necesito para resolver el problema?

3. ENCONTRAR RECURRENCIA:
   - ¿Cómo se relaciona dp[i] con estados anteriores?
   - Escribir la fórmula matemática

4. IDENTIFICAR CASOS BASE:
   - ¿Cuáles son los subproblemas triviales?
   - ¿Qué valores conozco directamente?

5. DETERMINAR ORDEN DE CÁLCULO:
   - ¿En qué orden llenar la tabla?
   - Asegurar que dependencias ya estén calculadas

6. IMPLEMENTAR:
   - Top-down (más intuitivo) o
   - Bottom-up (más eficiente)
```

### 5.2 Señales de que es DP

```
KEYWORDS que indican DP:
- "minimum/maximum"
- "count the number of ways"
- "is it possible"
- "longest/shortest"
- "optimal"

PATRONES comunes:
- Secuencias/arrays: dp[i] = f(dp[i-1], dp[i-2], ...)
- Dos secuencias: dp[i][j] = f(dp[i-1][j], dp[i][j-1], ...)
- Intervalos: dp[i][j] = f(dp[i+1][j], dp[i][j-1])
- Capacidad: dp[i][w] = f(dp[i-1][w], dp[i-1][w-item])
```

---

## ⚠️ Errores Comunes

### Error 1: Recurrencia incorrecta

```python
# ❌ Mal: no considera todos los casos
dp[i] = dp[i-1] + something

# ✅ Asegurar que considera TODAS las opciones
dp[i] = max/min over ALL valid transitions
```

### Error 2: Orden de cálculo incorrecto

```python
# ❌ Usar valores no calculados aún
for i in range(n):
    dp[i] = dp[i+1] + ...  # dp[i+1] no existe!

# ✅ Calcular dependencias primero
for i in range(n-1, -1, -1):  # Reverse
    dp[i] = dp[i+1] + ...
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 16.1: Fibonacci con memo y tabulation
### Ejercicio 16.2: Coin Change
### Ejercicio 16.3: Longest Common Subsequence
### Ejercicio 16.4: 0/1 Knapsack

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [MIT DP Lecture](https://www.youtube.com/watch?v=OQ5jsbhAv_M) | Video | 🔴 Obligatorio |
| [Dynamic Programming Patterns](https://leetcode.com/discuss/general-discussion/458695/dynamic-programming-patterns) | Guía | 🔴 Obligatorio |

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [15_GRAPHS](15_GRAPHS.md) | [00_INDICE](00_INDICE.md) | [17_GREEDY](17_GREEDY.md) |
