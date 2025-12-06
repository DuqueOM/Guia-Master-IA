# 09 - Búsqueda Binaria

> **🎯 Objetivo:** Implementar Binary Search sin errores off-by-one, entendiendo cuándo y cómo aplicarla.

---

## 🧠 Analogía: El Juego de las 20 Preguntas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   BÚSQUEDA LINEAL = Adivinar número del 1 al 100                            │
│   ────────────────────────────────────────────────                          │
│   "¿Es el 1?" No. "¿Es el 2?" No. "¿Es el 3?" No...                         │
│   Peor caso: 100 intentos → O(n)                                            │
│                                                                             │
│   BÚSQUEDA BINARIA = El juego inteligente                                   │
│   ───────────────────────────────────────                                   │
│   "¿Es mayor que 50?" Sí. → Descartar 1-50                                  │
│   "¿Es mayor que 75?" No. → Descartar 76-100                                │
│   "¿Es mayor que 62?" Sí. → Descartar 51-62                                 │
│   ...                                                                       │
│   Máximo: 7 intentos para 100 números → O(log n)                            │
│                                                                             │
│   REQUISITO: Los datos deben estar ORDENADOS                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [El Algoritmo Fundamental](#1-algoritmo)
2. [Implementación Sin Errores](#2-implementacion)
3. [Off-by-One: El Error Clásico](#3-off-by-one)
4. [Variantes Importantes](#4-variantes)
5. [Aplicación en Archimedes](#5-aplicacion)

---

## 1. El Algoritmo Fundamental {#1-algoritmo}

### 1.1 Idea Central

```
┌─────────────────────────────────────────────────────────────────┐
│  BINARY SEARCH: Dividir el espacio de búsqueda a la mitad       │
│                                                                 │
│  Buscar 7 en [1, 3, 5, 7, 9, 11, 13]                            │
│                                                                 │
│  Paso 1: low=0, high=6, mid=3 → items[3]=7                      │
│          [1, 3, 5, 7, 9, 11, 13]                                │
│                   ↑                                             │
│          ¡Encontrado! Retornar 3                                │
│                                                                 │
│  Buscar 9:                                                      │
│  Paso 1: mid=3 → items[3]=7 < 9 → buscar en derecha             │
│  Paso 2: low=4, high=6, mid=5 → items[5]=11 > 9 → izquierda     │
│  Paso 3: low=4, high=4, mid=4 → items[4]=9 → ¡Encontrado!       │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Complejidad

```
n elementos → máximo log₂(n) comparaciones

n = 1,000      → 10 comparaciones
n = 1,000,000  → 20 comparaciones
n = 1,000,000,000 → 30 comparaciones

¡Increíblemente eficiente!
```

---

## 2. Implementación Sin Errores {#2-implementacion}

### 2.1 Versión Iterativa (Recomendada)

```python
def binary_search(items: list[int], target: int) -> int:
    """Find target in sorted list using binary search.
    
    Args:
        items: Sorted list of integers.
        target: Value to find.
    
    Returns:
        Index of target if found, -1 otherwise.
    
    Complexity:
        Time: O(log n)
        Space: O(1)
    
    Example:
        >>> binary_search([1, 3, 5, 7, 9], 5)
        2
        >>> binary_search([1, 3, 5, 7, 9], 4)
        -1
    """
    left = 0
    right = len(items) - 1
    
    while left <= right:
        # Prevent integer overflow (not an issue in Python, but good practice)
        mid = left + (right - left) // 2
        
        if items[mid] == target:
            return mid
        elif items[mid] < target:
            left = mid + 1  # Target is in right half
        else:
            right = mid - 1  # Target is in left half
    
    return -1  # Not found
```

### 2.2 Versión Recursiva

```python
def binary_search_recursive(
    items: list[int],
    target: int,
    left: int = 0,
    right: int | None = None
) -> int:
    """Binary search using recursion.
    
    Complexity:
        Time: O(log n)
        Space: O(log n) for call stack
    """
    if right is None:
        right = len(items) - 1
    
    # Base case: search space exhausted
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if items[mid] == target:
        return mid
    elif items[mid] < target:
        return binary_search_recursive(items, target, mid + 1, right)
    else:
        return binary_search_recursive(items, target, left, mid - 1)
```

### 2.3 Visualización Paso a Paso

```python
def binary_search_verbose(items: list[int], target: int) -> int:
    """Binary search with debug output."""
    left = 0
    right = len(items) - 1
    step = 0
    
    while left <= right:
        mid = left + (right - left) // 2
        step += 1
        
        print(f"Step {step}: left={left}, right={right}, mid={mid}")
        print(f"  Checking items[{mid}] = {items[mid]}")
        
        if items[mid] == target:
            print(f"  Found {target} at index {mid}")
            return mid
        elif items[mid] < target:
            print(f"  {items[mid]} < {target}, searching right half")
            left = mid + 1
        else:
            print(f"  {items[mid]} > {target}, searching left half")
            right = mid - 1
    
    print(f"  {target} not found after {step} steps")
    return -1

# binary_search_verbose([1, 3, 5, 7, 9, 11, 13], 9)
# Step 1: left=0, right=6, mid=3
#   Checking items[3] = 7
#   7 < 9, searching right half
# Step 2: left=4, right=6, mid=5
#   Checking items[5] = 11
#   11 > 9, searching left half
# Step 3: left=4, right=4, mid=4
#   Checking items[4] = 9
#   Found 9 at index 4
```

---

## 3. Off-by-One: El Error Clásico {#3-off-by-one}

### 3.1 Los Puntos Críticos

```
┌─────────────────────────────────────────────────────────────────┐
│  ERRORES COMUNES:                                               │
│                                                                 │
│  1. while left < right vs while left <= right                   │
│     → left < right: puede no revisar último elemento            │
│     → left <= right: CORRECTO, revisa cuando left == right      │
│                                                                 │
│  2. right = mid vs right = mid - 1                              │
│     → right = mid: puede causar loop infinito                   │
│     → right = mid - 1: CORRECTO, excluye mid ya revisado        │
│                                                                 │
│  3. left = mid vs left = mid + 1                                │
│     → left = mid: puede causar loop infinito                    │
│     → left = mid + 1: CORRECTO, excluye mid ya revisado         │
│                                                                 │
│  4. mid = (left + right) / 2                                    │
│     → Overflow en otros lenguajes (no en Python)                │
│     → Mejor: mid = left + (right - left) // 2                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Ejemplo del Loop Infinito

```python
# ❌ Bug: loop infinito
def binary_search_buggy(items, target):
    left, right = 0, len(items) - 1
    while left < right:  # Bug 1: debería ser <=
        mid = (left + right) // 2
        if items[mid] < target:
            left = mid  # Bug 2: debería ser mid + 1
        else:
            right = mid
    return left if items[left] == target else -1

# Con items=[1, 3], target=3:
# left=0, right=1, mid=0
# items[0]=1 < 3, left=0 (no cambia!)
# Loop infinito porque left nunca avanza
```

### 3.3 Template a Prueba de Errores

```python
def binary_search_template(items: list[int], target: int) -> int:
    """Template seguro para binary search.
    
    Reglas de oro:
    1. while left <= right (incluir caso left == right)
    2. left = mid + 1 (excluir mid de búsqueda futura)
    3. right = mid - 1 (excluir mid de búsqueda futura)
    4. mid = left + (right - left) // 2 (evitar overflow)
    """
    left = 0
    right = len(items) - 1
    
    while left <= right:  # Regla 1: incluir igualdad
        mid = left + (right - left) // 2  # Regla 4: evitar overflow
        
        if items[mid] == target:
            return mid
        elif items[mid] < target:
            left = mid + 1  # Regla 2: excluir mid
        else:
            right = mid - 1  # Regla 3: excluir mid
    
    return -1
```

---

## 4. Variantes Importantes {#4-variantes}

### 4.1 Encontrar Primera Ocurrencia

```python
def find_first(items: list[int], target: int) -> int:
    """Find index of first occurrence of target.
    
    Returns -1 if target not found.
    
    Example:
        >>> find_first([1, 2, 2, 2, 3], 2)
        1  # First 2 is at index 1
    """
    left = 0
    right = len(items) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if items[mid] == target:
            result = mid  # Guardar y seguir buscando a la izquierda
            right = mid - 1
        elif items[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### 4.2 Encontrar Última Ocurrencia

```python
def find_last(items: list[int], target: int) -> int:
    """Find index of last occurrence of target.
    
    Example:
        >>> find_last([1, 2, 2, 2, 3], 2)
        3  # Last 2 is at index 3
    """
    left = 0
    right = len(items) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if items[mid] == target:
            result = mid  # Guardar y seguir buscando a la derecha
            left = mid + 1
        elif items[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### 4.3 Encontrar Punto de Inserción

```python
def find_insert_position(items: list[int], target: int) -> int:
    """Find index where target should be inserted to maintain sorted order.
    
    Equivalent to bisect_left from bisect module.
    
    Example:
        >>> find_insert_position([1, 3, 5, 7], 4)
        2  # Insert 4 at index 2 → [1, 3, 4, 5, 7]
        >>> find_insert_position([1, 3, 5, 7], 5)
        2  # Insert before existing 5
    """
    left = 0
    right = len(items)  # Note: len(items), not len(items) - 1
    
    while left < right:  # Note: <, not <=
        mid = left + (right - left) // 2
        
        if items[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left
```

### 4.4 Contar Ocurrencias

```python
def count_occurrences(items: list[int], target: int) -> int:
    """Count how many times target appears in sorted list.
    
    Uses find_first and find_last.
    
    Complexity: O(log n)
    
    Example:
        >>> count_occurrences([1, 2, 2, 2, 3], 2)
        3
    """
    first = find_first(items, target)
    
    if first == -1:
        return 0
    
    last = find_last(items, target)
    return last - first + 1
```

### 4.5 Búsqueda en Rango

```python
def search_in_range(
    items: list[int],
    low_target: int,
    high_target: int
) -> list[int]:
    """Find all elements in [low_target, high_target] range.
    
    Returns list of indices.
    
    Example:
        >>> search_in_range([1, 3, 5, 7, 9, 11], 4, 8)
        [2, 3]  # Indices of 5 and 7
    """
    # Find first element >= low_target
    start = find_insert_position(items, low_target)
    
    # Find first element > high_target
    end = find_insert_position(items, high_target + 1)
    
    return list(range(start, end))
```

---

## 5. Aplicación en Archimedes {#5-aplicacion}

### 5.1 Búsqueda en Posting Lists Ordenadas

```python
def search_in_posting_list(
    posting_list: list[int],
    doc_id: int
) -> bool:
    """Check if doc_id exists in sorted posting list.
    
    Posting lists in inverted index are often sorted by doc_id.
    Binary search is perfect for checking membership.
    """
    return binary_search(posting_list, doc_id) != -1
```

### 5.2 Intersección de Posting Lists

```python
def intersect_posting_lists(
    list1: list[int],
    list2: list[int]
) -> list[int]:
    """Intersect two sorted posting lists.
    
    Uses binary search for smaller list against larger list.
    
    Complexity: O(m log n) where m < n
    """
    # Ensure list1 is smaller
    if len(list1) > len(list2):
        list1, list2 = list2, list1
    
    result = []
    for doc_id in list1:
        if binary_search(list2, doc_id) != -1:
            result.append(doc_id)
    
    return result
```

### 5.3 Búsqueda de Umbral de Score

```python
from typing import NamedTuple


class ScoredDocument(NamedTuple):
    doc_id: int
    score: float


def find_docs_above_threshold(
    ranked_docs: list[ScoredDocument],
    min_score: float
) -> list[ScoredDocument]:
    """Find documents with score >= min_score.
    
    Assumes ranked_docs is sorted by score DESCENDING.
    
    Example:
        >>> docs = [ScoredDocument(1, 0.9), ScoredDocument(2, 0.7), 
        ...         ScoredDocument(3, 0.5), ScoredDocument(4, 0.3)]
        >>> find_docs_above_threshold(docs, 0.6)
        [ScoredDocument(1, 0.9), ScoredDocument(2, 0.7)]
    """
    if not ranked_docs:
        return []
    
    # Binary search for last document with score >= min_score
    left = 0
    right = len(ranked_docs) - 1
    result_end = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if ranked_docs[mid].score >= min_score:
            result_end = mid
            left = mid + 1  # Buscar más a la derecha
        else:
            right = mid - 1
    
    if result_end == -1:
        return []
    
    return ranked_docs[:result_end + 1]
```

---

## ⚠️ Errores Comunes

### Error 1: Olvidar que la lista debe estar ordenada

```python
# ❌ Binary search en lista no ordenada
items = [3, 1, 4, 1, 5, 9, 2, 6]
binary_search(items, 5)  # ¡Resultado incorrecto o no encontrado!

# ✅ Ordenar primero (o usar estructura ya ordenada)
items = sorted(items)
binary_search(items, 5)  # Correcto
```

### Error 2: Usar >= en lugar de > (o viceversa)

```python
# ❌ Condición incorrecta para primera ocurrencia
if items[mid] >= target:  # >= encontrará cualquier ocurrencia
    right = mid - 1

# ✅ Para primera ocurrencia, guardar y seguir buscando
if items[mid] == target:
    result = mid
    right = mid - 1  # Seguir buscando a la izquierda
```

### Error 3: Índices fuera de rango

```python
# ❌ Acceder sin verificar
def search_bad(items, target):
    mid = len(items) // 2
    return items[mid] == target  # IndexError si lista vacía!

# ✅ Verificar primero
def search_good(items, target):
    if not items:
        return False
    # ... binary search
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 9.1: Binary Search Básica
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-91)

### Ejercicio 9.2: Primera y Última Ocurrencia
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-92)

### Ejercicio 9.3: Búsqueda en Archimedes
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-93)

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Binary Search Visualization](https://www.cs.usfca.edu/~galles/visualization/Search.html) | Herramienta | 🔴 Obligatorio |
| [LeetCode Binary Search](https://leetcode.com/tag/binary-search/) | Práctica | 🔴 Obligatorio |
| [bisect Module](https://docs.python.org/3/library/bisect.html) | Docs | 🟡 Recomendado |

---

## 🔗 Referencias del Glosario

- [Binary Search](GLOSARIO.md#binary-search)
- [Off-by-One Error](GLOSARIO.md#off-by-one)
- [Logarithmic Complexity](GLOSARIO.md#logarithmic)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [08_SORTING](08_SORTING.md) | [00_INDICE](00_INDICE.md) | [10_ALGEBRA_LINEAL](10_ALGEBRA_LINEAL.md) |
