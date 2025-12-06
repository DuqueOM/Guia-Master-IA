# 17 - Algoritmos Greedy

> **🎯 Objetivo:** Entender cuándo y cómo aplicar la estrategia greedy correctamente.

---

## 🧠 Analogía: El Cambio de Monedas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   GREEDY = Tomar la MEJOR opción local en cada paso                         │
│   ─────────────────────────────────────────────────                         │
│                                                                             │
│   DAR CAMBIO DE $36:                                                        │
│   Monedas disponibles: $25, $10, $5, $1                                     │
│                                                                             │
│   ESTRATEGIA GREEDY:                                                        │
│   1. ¿Cabe $25? Sí → Tomar (quedan $11)                                     │
│   2. ¿Cabe $25? No. ¿$10? Sí → Tomar (quedan $1)                            │
│   3. ¿Cabe $10? No. ¿$5? No. ¿$1? Sí → Tomar (quedan $0)                    │
│   Total: 3 monedas ($25 + $10 + $1)                                         │
│                                                                             │
│   ✅ FUNCIONA porque estas monedas tienen "greedy choice property"          │
│                                                                             │
│   CONTRAEJEMPLO (monedas $1, $3, $4):                                       │
│   Dar cambio de $6:                                                         │
│   - Greedy: $4 + $1 + $1 = 3 monedas                                        │
│   - Óptimo: $3 + $3 = 2 monedas                                             │
│   ❌ Greedy NO siempre da óptimo                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [¿Qué es Greedy?](#1-que-es)
2. [Greedy vs Dynamic Programming](#2-vs-dp)
3. [Problemas Clásicos Greedy](#3-clasicos)
4. [Cómo Probar Correctitud](#4-probar)

---

## 1. ¿Qué es Greedy? {#1-que-es}

### 1.1 Definición

```
GREEDY ALGORITHM:
1. En cada paso, toma la decisión que parece MEJOR en ese momento
2. Una vez tomada, NUNCA reconsiderar
3. Esperar que las decisiones locales lleven al óptimo global

REQUISITOS para que funcione:
1. GREEDY CHOICE PROPERTY: Una solución óptima puede construirse
   tomando decisiones greedy
2. OPTIMAL SUBSTRUCTURE: Solución óptima contiene soluciones
   óptimas de subproblemas
```

### 1.2 Cuándo Usar Greedy

```
SEÑALES de que greedy puede funcionar:
- Problema pide máximo/mínimo
- Elementos pueden ordenarse por algún criterio
- Tomar una decisión no afecta decisiones futuras
- Hay "greedy choice" obvio

SEÑALES de que greedy NO funcionará:
- Decisiones actuales afectan opciones futuras
- Necesitas "deshacer" decisiones
- No hay criterio claro para ordenar
→ Probablemente necesitas DP
```

---

## 2. Greedy vs Dynamic Programming {#2-vs-dp}

### 2.1 Comparación

| Aspecto | Greedy | Dynamic Programming |
|---------|--------|---------------------|
| Decisiones | Una vez, definitiva | Considera todas |
| Subproblemas | No recalcula | Guarda y reusa |
| Complejidad | Generalmente menor | Mayor pero garantizado |
| Correctitud | Requiere demostración | Siempre correcto |
| Implementación | Más simple | Más compleja |

### 2.2 Coin Change: Greedy vs DP

```python
# GREEDY - Simple pero no siempre correcto
def coin_change_greedy(coins: list[int], amount: int) -> int:
    """May NOT give optimal solution for all coin systems."""
    coins_sorted = sorted(coins, reverse=True)
    count = 0
    
    for coin in coins_sorted:
        while amount >= coin:
            amount -= coin
            count += 1
    
    return count if amount == 0 else -1


# DP - Siempre correcto
def coin_change_dp(coins: list[int], amount: int) -> int:
    """Always gives optimal solution."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a and dp[a - coin] != float('inf'):
                dp[a] = min(dp[a], dp[a - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1


# Test
coins = [1, 3, 4]
amount = 6
print(coin_change_greedy(coins, amount))  # 3 (4+1+1) ❌
print(coin_change_dp(coins, amount))      # 2 (3+3) ✅
```

---

## 3. Problemas Clásicos Greedy {#3-clasicos}

### 3.1 Activity Selection

```python
def activity_selection(
    start: list[int],
    end: list[int]
) -> list[int]:
    """Select maximum non-overlapping activities.
    
    GREEDY: Always pick activity that ends earliest.
    
    Example:
        >>> activity_selection([1, 3, 0, 5, 8, 5], [2, 4, 6, 7, 9, 9])
        [0, 1, 3, 4]  # Activities 0, 1, 3, 4 (indices)
    
    Time: O(n log n) for sorting
    """
    n = len(start)
    
    # Create activities with indices
    activities = [(start[i], end[i], i) for i in range(n)]
    
    # Sort by end time (greedy criterion)
    activities.sort(key=lambda x: x[1])
    
    selected = [activities[0][2]]  # Select first (earliest ending)
    last_end = activities[0][1]
    
    for s, e, idx in activities[1:]:
        if s >= last_end:  # Doesn't overlap
            selected.append(idx)
            last_end = e
    
    return selected
```

### 3.2 Fractional Knapsack

```python
def fractional_knapsack(
    weights: list[float],
    values: list[float],
    capacity: float
) -> float:
    """Fractional Knapsack - can take fractions of items.
    
    GREEDY: Take items with best value/weight ratio first.
    
    Note: Unlike 0/1 knapsack, greedy works here because
    we can take fractions.
    
    Time: O(n log n)
    """
    n = len(weights)
    
    # Calculate value per unit weight
    ratios = [(values[i] / weights[i], weights[i], values[i]) 
              for i in range(n)]
    
    # Sort by ratio (descending)
    ratios.sort(reverse=True)
    
    total_value = 0.0
    remaining = capacity
    
    for ratio, weight, value in ratios:
        if remaining >= weight:
            # Take whole item
            total_value += value
            remaining -= weight
        else:
            # Take fraction
            total_value += ratio * remaining
            break
    
    return total_value
```

### 3.3 Huffman Coding (Preview)

```python
import heapq
from typing import Optional


class HuffmanNode:
    def __init__(self, char: Optional[str], freq: int):
        self.char = char
        self.freq = freq
        self.left: Optional[HuffmanNode] = None
        self.right: Optional[HuffmanNode] = None
    
    def __lt__(self, other):
        return self.freq < other.freq


def huffman_tree(freq: dict[str, int]) -> HuffmanNode:
    """Build Huffman tree using greedy algorithm.
    
    GREEDY: Always merge two nodes with lowest frequency.
    
    Used for optimal prefix-free encoding.
    """
    # Create leaf nodes and add to min-heap
    heap = [HuffmanNode(char, f) for char, f in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        # Take two smallest
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Merge
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        heapq.heappush(heap, merged)
    
    return heap[0]
```

### 3.4 Jump Game

```python
def can_jump(nums: list[int]) -> bool:
    """Can reach the last index from first?
    
    nums[i] = max jump length from position i.
    
    GREEDY: Track the farthest reachable position.
    
    Example:
        >>> can_jump([2, 3, 1, 1, 4])
        True
        >>> can_jump([3, 2, 1, 0, 4])
        False
    """
    farthest = 0
    
    for i, jump in enumerate(nums):
        if i > farthest:
            # Can't reach this position
            return False
        farthest = max(farthest, i + jump)
        if farthest >= len(nums) - 1:
            return True
    
    return True
```

### 3.5 Interval Scheduling

```python
def min_meeting_rooms(intervals: list[tuple[int, int]]) -> int:
    """Minimum meeting rooms needed.
    
    GREEDY: Process events in order, track active meetings.
    """
    if not intervals:
        return 0
    
    # Create events: (time, is_start)
    events = []
    for start, end in intervals:
        events.append((start, 1))   # Meeting starts
        events.append((end, -1))    # Meeting ends
    
    # Sort by time, ends before starts if same time
    events.sort(key=lambda x: (x[0], x[1]))
    
    rooms_needed = 0
    current_rooms = 0
    
    for time, delta in events:
        current_rooms += delta
        rooms_needed = max(rooms_needed, current_rooms)
    
    return rooms_needed
```

---

## 4. Cómo Probar Correctitud {#4-probar}

### 4.1 Técnicas de Demostración

```
EXCHANGE ARGUMENT:
1. Suponer que existe solución óptima diferente de la greedy
2. Mostrar que podemos "intercambiar" para llegar a solución greedy
3. Sin empeorar la calidad
4. Por tanto, greedy también es óptimo

STAYING AHEAD:
1. Mostrar que después de cada paso, greedy está "adelante"
2. Greedy es al menos tan bueno como cualquier otra opción
3. Por inducción, greedy termina óptimo
```

### 4.2 Ejemplo: Activity Selection

```
CLAIM: Greedy (elegir por end time) es óptimo.

PROOF (Exchange):
1. Sea OPT solución óptima, G solución greedy
2. Si OPT ≠ G, existe primera actividad diferente
3. Sea a₁ la primera actividad de G (menor end time)
4. Sea b₁ la primera de OPT (end time ≥ a₁)
5. Podemos reemplazar b₁ por a₁ en OPT:
   - No crea conflictos (a₁ termina antes)
   - Misma cantidad de actividades
6. Repetir hasta OPT = G
7. Por tanto, G es óptimo ✓
```

---

## ⚠️ Errores Comunes

### Error 1: Asumir que greedy siempre funciona

```python
# ❌ Greedy no siempre da óptimo
# Siempre verificar si el problema tiene greedy choice property
```

### Error 2: Criterio de ordenamiento incorrecto

```python
# Activity Selection
# ❌ Ordenar por duración
activities.sort(key=lambda x: x[1] - x[0])  # Incorrecto

# ✅ Ordenar por tiempo de fin
activities.sort(key=lambda x: x[1])  # Correcto
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 17.1: Activity Selection
### Ejercicio 17.2: Fractional Knapsack
### Ejercicio 17.3: Jump Game
### Ejercicio 17.4: Minimum Meeting Rooms

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Greedy Algorithms - MIT](https://www.youtube.com/watch?v=2P-yW7LQr08) | Video | 🔴 Obligatorio |
| [When Greedy Works](https://www.geeksforgeeks.org/greedy-algorithms/) | Guía | 🟡 Recomendado |

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [16_DYNAMIC_PROGRAMMING](16_DYNAMIC_PROGRAMMING.md) | [00_INDICE](00_INDICE.md) | [18_HEAPS](18_HEAPS.md) |
