# Anexo DSA - Algoritmos de Ordenamiento

> **⚠️ MÓDULO OPCIONAL:** Este módulo NO es requerido para el Pathway. Es útil para entrevistas técnicas.  
> **🎯 Objetivo:** Implementar QuickSort y MergeSort desde cero.

---

## 🧠 Analogía: Ordenando Cartas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   QUICKSORT = El método del "pivote"                                        │
│   ──────────────────────────────────                                        │
│                                                                             │
│   1. Elige una carta (pivote): por ejemplo, el 7                            │
│   2. Separa: menores a la izquierda, mayores a la derecha                   │
│   3. Ahora el 7 está en su lugar correcto                                   │
│   4. Repite con cada grupo                                                  │
│                                                                             │
│   [3, 8, 2, 7, 1, 9, 4]  → pivote = 7                                       │
│   [3, 2, 1, 4] [7] [8, 9]  → 7 en su lugar                                  │
│   Repetir para [3,2,1,4] y [8,9]                                            │
│                                                                             │
│   MERGESORT = El método de "dividir y fusionar"                             │
│   ─────────────────────────────────────────────                             │
│                                                                             │
│   1. Divide el mazo en dos mitades                                          │
│   2. Ordena cada mitad (recursivamente)                                     │
│   3. Fusiona las dos mitades ordenadas                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Por Qué Importan los Algoritmos de Sorting](#1-importancia)
2. [QuickSort: El Favorito en la Práctica](#2-quicksort)
3. [MergeSort: Estable y Predecible](#3-mergesort)
4. [Comparación y Cuándo Usar Cada Uno](#4-comparacion)
5. [Análisis de Complejidad Detallado](#5-analisis)

---

## 1. Por Qué Importan los Algoritmos de Sorting {#1-importancia}

### 1.1 Sorting es Fundamental

```
┌─────────────────────────────────────────────────────────────────┐
│  APLICACIONES DE SORTING                                        │
│                                                                 │
│  • Búsqueda binaria: requiere datos ordenados                   │
│  • Ranking de resultados: ordenar por relevancia                │
│  • Eliminación de duplicados: ordenar + recorrer                │
│  • Mediana, percentiles: ordenar + acceder por índice           │
│  • Sistemas de bases de datos: índices ordenados                │
│                                                                 │
│  EN ARCHIMEDES INDEXER:                                         │
│  Ordenar resultados de búsqueda por score de relevancia         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Complejidades de Referencia

| Algoritmo | Mejor | Promedio | Peor | Espacio |
|-----------|-------|----------|------|---------|
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) |
| Selection Sort | O(n²) | O(n²) | O(n²) | O(1) |
| Insertion Sort | O(n) | O(n²) | O(n²) | O(1) |
| **QuickSort** | O(n log n) | O(n log n) | O(n²) | O(log n) |
| **MergeSort** | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Python's Timsort | O(n) | O(n log n) | O(n log n) | O(n) |

---

## 2. QuickSort: El Favorito en la Práctica {#2-quicksort}

### 2.1 El Algoritmo

```
┌─────────────────────────────────────────────────────────────────┐
│  QUICKSORT - Pasos:                                             │
│                                                                 │
│  1. Si la lista tiene 0 o 1 elementos, ya está ordenada         │
│  2. Elegir un PIVOTE (elemento de referencia)                   │
│  3. PARTICIONAR: reorganizar para que:                          │
│     - Elementos < pivote queden a la izquierda                  │
│     - Elementos >= pivote queden a la derecha                   │
│  4. Recursivamente ordenar izquierda y derecha                  │
│  5. Concatenar: izquierda + pivote + derecha                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementación Básica (Fácil de Entender)

```python
def quicksort_simple(items: list[int]) -> list[int]:
    """QuickSort with simple partitioning.
    
    This version creates new lists (not in-place).
    Easier to understand but uses more memory.
    
    Complexity:
        Time: O(n log n) average, O(n²) worst
        Space: O(n) for new lists
    
    Example:
        >>> quicksort_simple([3, 1, 4, 1, 5, 9, 2, 6])
        [1, 1, 2, 3, 4, 5, 6, 9]
    """
    # Base case: already sorted
    if len(items) <= 1:
        return items
    
    # Choose pivot (last element for simplicity)
    pivot = items[-1]
    
    # Partition into three groups
    less = [x for x in items[:-1] if x < pivot]
    equal = [x for x in items if x == pivot]
    greater = [x for x in items[:-1] if x > pivot]
    
    # Recursively sort and concatenate
    return quicksort_simple(less) + equal + quicksort_simple(greater)
```

### 2.3 Implementación In-Place (Eficiente en Memoria)

```python
def quicksort(items: list[int]) -> list[int]:
    """QuickSort with in-place partitioning.
    
    Modifies the original list.
    
    Returns:
        The same list, now sorted.
    """
    _quicksort_helper(items, 0, len(items) - 1)
    return items


def _quicksort_helper(items: list[int], low: int, high: int) -> None:
    """Recursive helper for in-place quicksort."""
    if low < high:
        # Partition and get pivot position
        pivot_index = _partition(items, low, high)
        
        # Recursively sort elements before and after partition
        _quicksort_helper(items, low, pivot_index - 1)
        _quicksort_helper(items, pivot_index + 1, high)


def _partition(items: list[int], low: int, high: int) -> int:
    """Partition array around pivot (last element).
    
    Lomuto partition scheme.
    
    Returns:
        Final position of pivot.
    """
    pivot = items[high]
    i = low - 1  # Index of smaller element
    
    for j in range(low, high):
        if items[j] < pivot:
            i += 1
            items[i], items[j] = items[j], items[i]
    
    # Place pivot in correct position
    items[i + 1], items[high] = items[high], items[i + 1]
    return i + 1
```

### 2.4 Visualización de Partición

```
Inicial: [8, 3, 1, 7, 0, 10, 2]  (pivot = 2)

j=0: 8 < 2? NO  → [8, 3, 1, 7, 0, 10, 2]  i=-1
j=1: 3 < 2? NO  → [8, 3, 1, 7, 0, 10, 2]  i=-1
j=2: 1 < 2? SÍ  → [1, 3, 8, 7, 0, 10, 2]  i=0 (swap 8↔1)
j=3: 7 < 2? NO  → [1, 3, 8, 7, 0, 10, 2]  i=0
j=4: 0 < 2? SÍ  → [1, 0, 8, 7, 3, 10, 2]  i=1 (swap 3↔0)
j=5: 10< 2? NO  → [1, 0, 8, 7, 3, 10, 2]  i=1

Final: colocar pivot en i+1=2
       [1, 0, 2, 7, 3, 10, 8]
              ↑ pivot en posición correcta

Izquierda: [1, 0] (todos < 2)
Derecha:   [7, 3, 10, 8] (todos > 2)
```

### 2.5 Random Pivot (Evitar O(n²))

```python
import random


def quicksort_random(items: list[int]) -> list[int]:
    """QuickSort with random pivot selection.
    
    Random pivot prevents worst case O(n²) on sorted input.
    """
    _quicksort_random_helper(items, 0, len(items) - 1)
    return items


def _quicksort_random_helper(items: list[int], low: int, high: int) -> None:
    if low < high:
        pivot_index = _partition_random(items, low, high)
        _quicksort_random_helper(items, low, pivot_index - 1)
        _quicksort_random_helper(items, pivot_index + 1, high)


def _partition_random(items: list[int], low: int, high: int) -> int:
    """Partition with random pivot."""
    # Choose random pivot and swap to end
    random_index = random.randint(low, high)
    items[random_index], items[high] = items[high], items[random_index]
    
    return _partition(items, low, high)
```

---

## 3. MergeSort: Estable y Predecible {#3-mergesort}

### 3.1 El Algoritmo

```
┌─────────────────────────────────────────────────────────────────┐
│  MERGESORT - Pasos:                                             │
│                                                                 │
│  1. Si la lista tiene 0 o 1 elementos, ya está ordenada         │
│  2. DIVIDIR: partir la lista en dos mitades                     │
│  3. CONQUISTAR: ordenar cada mitad recursivamente               │
│  4. COMBINAR: fusionar las dos mitades ordenadas                │
│                                                                 │
│  La "magia" está en el paso de MERGE:                           │
│  - Dos listas ordenadas se pueden fusionar en O(n)              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementación Completa

```python
def mergesort(items: list[int]) -> list[int]:
    """Sort list using merge sort algorithm.
    
    Creates new lists (not in-place).
    
    Complexity:
        Time: O(n log n) always
        Space: O(n) for temporary arrays
    
    Example:
        >>> mergesort([3, 1, 4, 1, 5, 9, 2, 6])
        [1, 1, 2, 3, 4, 5, 6, 9]
    """
    # Base case
    if len(items) <= 1:
        return items.copy()
    
    # Divide
    mid = len(items) // 2
    left = items[:mid]
    right = items[mid:]
    
    # Conquer (recursively sort)
    left_sorted = mergesort(left)
    right_sorted = mergesort(right)
    
    # Combine (merge)
    return _merge(left_sorted, right_sorted)


def _merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted lists into one sorted list.
    
    Uses two-pointer technique.
    
    Complexity: O(n + m) where n, m are list lengths
    """
    result = []
    i = j = 0
    
    # Compare elements from both lists
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # <= makes it stable
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements (one list is exhausted)
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result
```

### 3.3 Visualización de Merge

```
Fusionar [1, 3, 5] con [2, 4, 6]:

i=0, j=0: 1 vs 2 → tomar 1    result=[1]
i=1, j=0: 3 vs 2 → tomar 2    result=[1, 2]
i=1, j=1: 3 vs 4 → tomar 3    result=[1, 2, 3]
i=2, j=1: 5 vs 4 → tomar 4    result=[1, 2, 3, 4]
i=2, j=2: 5 vs 6 → tomar 5    result=[1, 2, 3, 4, 5]
i=3, j=2: (left agotada)      result=[1, 2, 3, 4, 5, 6]

Final: [1, 2, 3, 4, 5, 6]
```

### 3.4 MergeSort In-Place (Opcional, Más Complejo)

```python
def mergesort_inplace(items: list[int]) -> list[int]:
    """In-place merge sort using auxiliary array.
    
    More memory efficient than creating many small lists.
    """
    aux = items.copy()
    _mergesort_inplace_helper(items, aux, 0, len(items) - 1)
    return items


def _mergesort_inplace_helper(
    items: list[int],
    aux: list[int],
    low: int,
    high: int
) -> None:
    if low >= high:
        return
    
    mid = (low + high) // 2
    _mergesort_inplace_helper(items, aux, low, mid)
    _mergesort_inplace_helper(items, aux, mid + 1, high)
    _merge_inplace(items, aux, low, mid, high)


def _merge_inplace(
    items: list[int],
    aux: list[int],
    low: int,
    mid: int,
    high: int
) -> None:
    # Copy to auxiliary array
    for k in range(low, high + 1):
        aux[k] = items[k]
    
    i = low
    j = mid + 1
    
    for k in range(low, high + 1):
        if i > mid:
            items[k] = aux[j]
            j += 1
        elif j > high:
            items[k] = aux[i]
            i += 1
        elif aux[j] < aux[i]:
            items[k] = aux[j]
            j += 1
        else:
            items[k] = aux[i]
            i += 1
```

---

## 4. Comparación y Cuándo Usar Cada Uno {#4-comparacion}

### 4.1 Tabla Comparativa

| Aspecto | QuickSort | MergeSort |
|---------|-----------|-----------|
| **Complejidad promedio** | O(n log n) | O(n log n) |
| **Peor caso** | O(n²) | O(n log n) |
| **Espacio** | O(log n) | O(n) |
| **Estable** | ❌ No | ✅ Sí |
| **In-place** | ✅ Sí | ❌ No (típicamente) |
| **Cache-friendly** | ✅ Mejor | ❌ Peor |

### 4.2 ¿Qué Significa "Estable"?

```python
# Elementos con mismo valor mantienen orden relativo

data = [("Alice", 25), ("Bob", 30), ("Carol", 25)]

# Ordenar por edad
# ESTABLE: Alice antes de Carol (original order preserved)
# sorted_stable = [("Alice", 25), ("Carol", 25), ("Bob", 30)]

# NO ESTABLE: Carol podría quedar antes de Alice
# sorted_unstable = [("Carol", 25), ("Alice", 25), ("Bob", 30)]
```

### 4.3 Cuándo Usar Cada Uno

```
┌─────────────────────────────────────────────────────────────────┐
│  USA QUICKSORT cuando:                                          │
│  • Memoria es limitada (in-place)                               │
│  • No necesitas estabilidad                                     │
│  • Datos son aleatorios (no ya ordenados)                       │
│  • Quieres mejor rendimiento promedio en práctica               │
│                                                                 │
│  USA MERGESORT cuando:                                          │
│  • Necesitas garantía O(n log n) siempre                        │
│  • Necesitas ordenamiento estable                               │
│  • Memoria no es problema                                       │
│  • Datos podrían estar casi ordenados                           │
│                                                                 │
│  EN ARCHIMEDES:                                                 │
│  Usaremos QuickSort para ordenar resultados por score           │
│  porque raramente están pre-ordenados y queremos velocidad      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Análisis de Complejidad Detallado {#5-analisis}

### 5.1 QuickSort: Por Qué O(n log n) Promedio

```
┌─────────────────────────────────────────────────────────────────┐
│  MEJOR CASO: Pivote divide perfectamente por la mitad           │
│                                                                 │
│  Nivel 0: 1 problema de tamaño n                                │
│  Nivel 1: 2 problemas de tamaño n/2                             │
│  Nivel 2: 4 problemas de tamaño n/4                             │
│  ...                                                            │
│  Nivel log n: n problemas de tamaño 1                           │
│                                                                 │
│  Trabajo por nivel: O(n) (partición)                            │
│  Número de niveles: O(log n)                                    │
│  Total: O(n) × O(log n) = O(n log n)                            │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 QuickSort: Por Qué O(n²) Peor Caso

```
┌─────────────────────────────────────────────────────────────────┐
│  PEOR CASO: Lista ya ordenada + pivot siempre el último         │
│                                                                 │
│  [1, 2, 3, 4, 5]  pivot=5 → [1,2,3,4] [] + [5]                  │
│  [1, 2, 3, 4]     pivot=4 → [1,2,3]   [] + [4]                  │
│  [1, 2, 3]        pivot=3 → [1,2]     [] + [3]                  │
│  [1, 2]           pivot=2 → [1]       [] + [2]                  │
│                                                                 │
│  Cada nivel quita solo 1 elemento → n niveles                   │
│  Trabajo por nivel: O(n), O(n-1), O(n-2), ...                   │
│  Total: n + (n-1) + ... + 1 = n(n+1)/2 = O(n²)                  │
│                                                                 │
│  SOLUCIÓN: Random pivot evita esto en la práctica               │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 MergeSort: Siempre O(n log n)

```
┌─────────────────────────────────────────────────────────────────┐
│  SIEMPRE divide exactamente por la mitad                        │
│                                                                 │
│  T(n) = 2×T(n/2) + O(n)                                         │
│         ↑         ↑                                             │
│    2 subproblemas  merge                                        │
│    de tamaño n/2                                                │
│                                                                 │
│  Por Master Theorem:                                            │
│  T(n) = O(n log n)                                              │
│                                                                 │
│  No hay peor caso porque la división es siempre balanceada      │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Análisis de Espacio

```python
# QuickSort: O(log n) espacio para call stack
# - Cada llamada recursiva usa espacio constante
# - Profundidad máxima: log n (caso promedio)
# - Profundidad máxima: n (peor caso)

# MergeSort: O(n) espacio para arrays temporales
# - Cada merge crea nuevo array
# - El array más grande es de tamaño n
# - Plus O(log n) para call stack
```

---

## ⚠️ Errores Comunes

### Error 1: Off-by-one en partition

```python
# ❌ Error común: incluir pivote en recursión
_quicksort_helper(items, low, pivot_index)  # Incluye pivote
_quicksort_helper(items, pivot_index, high)  # Pivote otra vez!

# ✅ Correcto: excluir pivote (ya está en su lugar)
_quicksort_helper(items, low, pivot_index - 1)
_quicksort_helper(items, pivot_index + 1, high)
```

### Error 2: No manejar lista vacía

```python
# ❌ Falla con lista vacía
def quicksort_bad(items):
    pivot = items[-1]  # IndexError!

# ✅ Manejar caso base
def quicksort_good(items):
    if len(items) <= 1:
        return items
    pivot = items[-1]
```

### Error 3: Modificar lista durante iteración

```python
# ❌ Confuso y propenso a errores
for i, item in enumerate(items):
    items[i], items[j] = ...  # Modifica mientras itera

# ✅ Usar índices explícitos
for j in range(low, high):
    if items[j] < pivot:
        i += 1
        items[i], items[j] = items[j], items[i]
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 8.1: Implementar QuickSort
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-81)

### Ejercicio 8.2: Implementar MergeSort
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-82)

### Ejercicio 8.3: Ordenar por Score
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-83) - Aplicar al ranking de Archimedes

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Visualgo Sorting](https://visualgo.net/en/sorting) | Visualización | 🔴 Obligatorio |
| [Grokking Algorithms Ch.4](https://www.manning.com/books/grokking-algorithms) | Libro | 🔴 Obligatorio |
| [QuickSort Analysis](https://www.youtube.com/watch?v=uXBnyYuwPe8) | Video | 🟡 Recomendado |

---

## 🔗 Referencias del Glosario

- [QuickSort](GLOSARIO.md#quicksort)
- [MergeSort](GLOSARIO.md#mergesort)
- [Partition](GLOSARIO.md#partition)
- [Estabilidad](GLOSARIO.md#estabilidad)
- [In-Place](GLOSARIO.md#in-place)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [07_RECURSION](07_RECURSION.md) | [00_INDICE](00_INDICE.md) | [09_BINARY_SEARCH](09_BINARY_SEARCH.md) |
