# Anexo DSA - Recursión y Divide & Conquer

> **⚠️ MÓDULO OPCIONAL:** Este módulo NO es requerido para el Pathway. Es útil para entrevistas técnicas.  
> **🎯 Objetivo:** Dominar el pensamiento recursivo.

---

## 🧠 Analogía: Las Muñecas Rusas (Matryoshkas)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   RECURSIÓN = Resolver un problema resolviéndolo para una versión menor     │
│   ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│   Muñecas Rusas:                                                            │
│   ┌─────────────────┐                                                       │
│   │ ┌─────────────┐ │                                                       │
│   │ │ ┌─────────┐ │ │                                                       │
│   │ │ │ ┌─────┐ │ │ │                                                       │
│   │ │ │ │ ●   │ │ │ │  ← Caso base: la muñeca más pequeña (sólida)          │
│   │ │ │ └─────┘ │ │ │                                                       │
│   │ │ └─────────┘ │ │  ← Cada muñeca "contiene" una versión menor           │
│   │ └─────────────┘ │                                                       │
│   └─────────────────┘                                                       │
│                                                                             │
│   Para abrir TODAS las muñecas:                                             │
│   1. ¿Es la muñeca sólida? → PARAR (caso base)                              │
│   2. Si no, abrir esta muñeca y REPETIR con la de adentro                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [¿Qué es Recursión?](#1-que-es)
2. [Caso Base y Caso Recursivo](#2-casos)
3. [El Call Stack](#3-call-stack)
4. [Ejemplos Clásicos](#4-ejemplos)
5. [Divide & Conquer](#5-divide-conquer)
6. [Optimización con Memoization](#6-memoization)

---

## 1. ¿Qué es Recursión? {#1-que-es}

### 1.1 Definición

```
┌─────────────────────────────────────────────────────────────────┐
│  RECURSIÓN: Una función que se llama a sí misma                 │
│                                                                 │
│  def funcion():                                                 │
│      ...                                                        │
│      funcion()  ← Se llama a sí misma                           │
│      ...                                                        │
│                                                                 │
│  ⚠️ Sin condición de parada → recursión infinita → crash        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 ¿Por Qué Usar Recursión?

```
PROBLEMAS NATURALMENTE RECURSIVOS:
─────────────────────────────────

1. Estructuras de datos recursivas
   - Árboles: un nodo tiene hijos que son árboles
   - Listas enlazadas: una lista es un nodo + otra lista
   - Sistemas de archivos: carpetas contienen carpetas

2. Problemas que se reducen a versiones menores
   - Factorial: n! = n × (n-1)!
   - Fibonacci: fib(n) = fib(n-1) + fib(n-2)
   - Ordenamiento: ordenar lista = ordenar sublistas + combinar
```

---

## 2. Caso Base y Caso Recursivo {#2-casos}

### 2.1 Los Dos Ingredientes Esenciales

```python
def recursive_function(problem):
    # 1. CASO BASE: problema tan pequeño que se resuelve directamente
    if problem_is_trivial(problem):
        return trivial_solution
    
    # 2. CASO RECURSIVO: reducir el problema y llamar recursivamente
    smaller_problem = reduce(problem)
    return combine(recursive_function(smaller_problem))
```

### 2.2 Ejemplo: Factorial

```python
def factorial(n: int) -> int:
    """Calculate n! = n × (n-1) × (n-2) × ... × 1
    
    Base case: 0! = 1
    Recursive: n! = n × (n-1)!
    
    Example:
        >>> factorial(5)
        120  # 5 × 4 × 3 × 2 × 1
    """
    # Caso base
    if n <= 1:
        return 1
    
    # Caso recursivo
    return n * factorial(n - 1)


# Traza de ejecución:
# factorial(4)
#   → 4 * factorial(3)
#       → 3 * factorial(2)
#           → 2 * factorial(1)
#               → 1  (caso base)
#           → 2 * 1 = 2
#       → 3 * 2 = 6
#   → 4 * 6 = 24
```

### 2.3 Ejemplo: Suma de Lista

```python
def sum_list(numbers: list[int]) -> int:
    """Sum all numbers in list using recursion.
    
    Base case: empty list → 0
    Recursive: sum = first + sum(rest)
    
    Example:
        >>> sum_list([1, 2, 3, 4])
        10
    """
    # Caso base: lista vacía
    if not numbers:
        return 0
    
    # Caso recursivo: primer elemento + suma del resto
    return numbers[0] + sum_list(numbers[1:])


# Alternativa más eficiente (evita crear sublistas)
def sum_list_efficient(numbers: list[int], index: int = 0) -> int:
    """Sum using index instead of slicing."""
    # Caso base: índice fuera de rango
    if index >= len(numbers):
        return 0
    
    # Caso recursivo
    return numbers[index] + sum_list_efficient(numbers, index + 1)
```

---

## 3. El Call Stack {#3-call-stack}

### 3.1 Visualización del Stack

```
┌─────────────────────────────────────────────────────────────────┐
│  CALL STACK: Pila de llamadas a funciones                       │
│                                                                 │
│  Cada llamada recursiva agrega un "frame" al stack              │
│  Cuando termina, se "desapila" y retorna al anterior            │
│                                                                 │
│  factorial(4):                                                  │
│                                                                 │
│  LLAMANDO (stack crece →)          RETORNANDO (stack decrece ←) │
│                                                                 │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │ factorial(1) = 1 │ ←base       │ factorial(1) = 1 │ →return  │
│  ├──────────────────┤              ├──────────────────┤         │
│  │ factorial(2)     │              │ factorial(2) = 2 │ →return │
│  ├──────────────────┤              ├──────────────────┤         │
│  │ factorial(3)     │              │ factorial(3) = 6 │ →return │
│  ├──────────────────┤              ├──────────────────┤         │
│  │ factorial(4)     │              │ factorial(4) = 24│ →return │
│  └──────────────────┘              └──────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Límite de Recursión

```python
import sys

# Python tiene un límite por defecto
print(sys.getrecursionlimit())  # 1000 (típicamente)

# Excederlo causa RecursionError
def infinite_recursion():
    return infinite_recursion()

# infinite_recursion()  # RecursionError: maximum recursion depth exceeded

# Puedes aumentar el límite (con cuidado)
sys.setrecursionlimit(2000)
```

### 3.3 Visualizar la Recursión

```python
def factorial_verbose(n: int, depth: int = 0) -> int:
    """Factorial with execution trace."""
    indent = "  " * depth
    print(f"{indent}factorial({n})")
    
    if n <= 1:
        print(f"{indent}→ returning 1 (base case)")
        return 1
    
    result = n * factorial_verbose(n - 1, depth + 1)
    print(f"{indent}→ returning {n} * ... = {result}")
    return result

# factorial_verbose(4) muestra:
# factorial(4)
#   factorial(3)
#     factorial(2)
#       factorial(1)
#       → returning 1 (base case)
#     → returning 2 * ... = 2
#   → returning 3 * ... = 6
# → returning 4 * ... = 24
```

---

## 4. Ejemplos Clásicos {#4-ejemplos}

### 4.1 Fibonacci

```python
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number.
    
    Sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...
    
    Base cases: fib(0) = 0, fib(1) = 1
    Recursive: fib(n) = fib(n-1) + fib(n-2)
    
    ⚠️ This naive version is O(2^n) - very slow!
    See memoization section for optimization.
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    return fibonacci(n - 1) + fibonacci(n - 2)
```

### 4.2 Búsqueda en Lista

```python
def search_recursive(
    items: list[any],
    target: any,
    index: int = 0
) -> int:
    """Search for target in list, return index or -1.
    
    Base cases:
    - Index out of bounds → not found (-1)
    - Found target → return index
    
    Recursive: check next index
    """
    # Caso base: fin de lista
    if index >= len(items):
        return -1
    
    # Caso base: encontrado
    if items[index] == target:
        return index
    
    # Caso recursivo: buscar en el resto
    return search_recursive(items, target, index + 1)
```

### 4.3 Contar Ocurrencias

```python
def count_occurrences(items: list[any], target: any) -> int:
    """Count how many times target appears in list.
    
    Base case: empty list → 0
    Recursive: (1 if first matches else 0) + count(rest)
    """
    if not items:
        return 0
    
    first_match = 1 if items[0] == target else 0
    return first_match + count_occurrences(items[1:], target)
```

### 4.4 Invertir String

```python
def reverse_string(s: str) -> str:
    """Reverse a string recursively.
    
    Base case: empty or single char → return as is
    Recursive: last char + reverse(rest)
    
    Example:
        >>> reverse_string("hello")
        'olleh'
    """
    if len(s) <= 1:
        return s
    
    return s[-1] + reverse_string(s[:-1])
```

### 4.5 Palíndromo

```python
def is_palindrome(s: str) -> bool:
    """Check if string is a palindrome.
    
    Base cases:
    - Length 0 or 1 → True
    - First != Last → False
    
    Recursive: check first == last, then inner string
    
    Example:
        >>> is_palindrome("radar")
        True
    """
    # Normalizar: quitar espacios y minúsculas
    s = s.lower().replace(" ", "")
    
    if len(s) <= 1:
        return True
    
    if s[0] != s[-1]:
        return False
    
    return is_palindrome(s[1:-1])
```

---

## 5. Divide & Conquer {#5-divide-conquer}

### 5.1 El Patrón

```
┌─────────────────────────────────────────────────────────────────┐
│  DIVIDE & CONQUER (Divide y Vencerás)                           │
│                                                                 │
│  1. DIVIDIR: Partir el problema en subproblemas más pequeños    │
│  2. CONQUISTAR: Resolver cada subproblema (recursivamente)      │
│  3. COMBINAR: Unir las soluciones parciales                     │
│                                                                 │
│  Ejemplos clásicos:                                             │
│  - MergeSort: dividir lista, ordenar mitades, combinar          │
│  - QuickSort: particionar, ordenar particiones                  │
│  - Binary Search: buscar en mitad correcta                      │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Merge Sort (Ejemplo Perfecto)

```python
def merge_sort(items: list[int]) -> list[int]:
    """Sort list using merge sort algorithm.
    
    Divide: split list in half
    Conquer: recursively sort each half
    Combine: merge sorted halves
    
    Complexity: O(n log n) always
    """
    # Base case: 0 or 1 elements already sorted
    if len(items) <= 1:
        return items
    
    # DIVIDE: split in half
    mid = len(items) // 2
    left = items[:mid]
    right = items[mid:]
    
    # CONQUER: sort each half recursively
    left_sorted = merge_sort(left)
    right_sorted = merge_sort(right)
    
    # COMBINE: merge sorted halves
    return merge(left_sorted, right_sorted)


def merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted lists into one sorted list.
    
    Uses two-pointer technique.
    Complexity: O(n + m)
    """
    result = []
    i = j = 0
    
    # Compare elements from both lists
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Add remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result
```

### 5.3 Visualización de Merge Sort

```
┌─────────────────────────────────────────────────────────────────┐
│  merge_sort([38, 27, 43, 3, 9, 82, 10])                         │
│                                                                 │
│  DIVIDIR:                                                       │
│                    [38, 27, 43, 3, 9, 82, 10]                   │
│                           /            \                        │
│               [38, 27, 43]              [3, 9, 82, 10]          │
│                /       \                  /        \            │
│           [38, 27]    [43]           [3, 9]    [82, 10]         │
│            /    \                     /   \      /    \         │
│          [38]  [27]               [3]   [9]  [82]  [10]         │
│                                                                 │
│  COMBINAR (merge):                                              │
│          [27, 38] ← merge [38],[27]   [3, 9] [10, 82]           │
│                \    /                    \    /                 │
│             [27, 38, 43]            [3, 9, 10, 82]              │
│                     \                  /                        │
│                [3, 9, 10, 27, 38, 43, 82]                       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Máximo de Lista (Divide & Conquer)

```python
def find_max_dc(items: list[int]) -> int:
    """Find maximum using divide and conquer.
    
    Base cases:
    - Single element → that element
    - Two elements → larger of the two
    
    Recursive: max of (max left half, max right half)
    """
    if len(items) == 0:
        raise ValueError("Cannot find max of empty list")
    
    if len(items) == 1:
        return items[0]
    
    if len(items) == 2:
        return items[0] if items[0] > items[1] else items[1]
    
    mid = len(items) // 2
    left_max = find_max_dc(items[:mid])
    right_max = find_max_dc(items[mid:])
    
    return left_max if left_max > right_max else right_max
```

---

## 6. Optimización con Memoization {#6-memoization}

### 6.1 El Problema con Fibonacci Naive

```
┌─────────────────────────────────────────────────────────────────┐
│  fib(5) calcula fib(3) DOS veces, fib(2) TRES veces, etc.       │
│                                                                 │
│                      fib(5)                                     │
│                    /        \                                   │
│               fib(4)        fib(3)        ← fib(3) calculado 2x │
│              /     \        /    \                              │
│          fib(3)  fib(2)  fib(2) fib(1)   ← fib(2) calculado 3x  │
│          /   \                                                  │
│      fib(2) fib(1)                                              │
│                                                                 │
│  Complejidad: O(2^n) - ¡Exponencial!                            │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Memoization: Recordar Resultados

```python
def fibonacci_memo(n: int, cache: dict[int, int] | None = None) -> int:
    """Fibonacci with memoization.
    
    Cache stores already computed values to avoid redundant work.
    
    Complexity: O(n) time, O(n) space
    """
    if cache is None:
        cache = {}
    
    # Check cache first
    if n in cache:
        return cache[n]
    
    # Base cases
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    # Compute and cache
    result = fibonacci_memo(n - 1, cache) + fibonacci_memo(n - 2, cache)
    cache[n] = result
    
    return result


# Comparación de tiempos:
# fibonacci(35)      → ~3 segundos
# fibonacci_memo(35) → <0.001 segundos
```

### 6.3 Usando functools.lru_cache

```python
from functools import lru_cache

@lru_cache(maxsize=None)  # Cache ilimitado
def fibonacci_cached(n: int) -> int:
    """Fibonacci with automatic memoization."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)


# Ver estadísticas del cache
print(fibonacci_cached.cache_info())
# CacheInfo(hits=48, misses=51, maxsize=None, currsize=51)

# Limpiar cache
fibonacci_cached.cache_clear()
```

---

## ⚠️ Errores Comunes

### Error 1: Olvidar el caso base

```python
# ❌ Sin caso base → RecursionError
def countdown_bad(n):
    print(n)
    countdown_bad(n - 1)  # Nunca termina

# ✅ Con caso base
def countdown_good(n):
    if n <= 0:
        print("Done!")
        return
    print(n)
    countdown_good(n - 1)
```

### Error 2: No reducir el problema

```python
# ❌ El problema no se reduce
def broken_sum(items):
    if not items:
        return 0
    return items[0] + broken_sum(items)  # Misma lista!

# ✅ Reducir correctamente
def working_sum(items):
    if not items:
        return 0
    return items[0] + working_sum(items[1:])  # Lista más corta
```

### Error 3: Crear copias innecesarias

```python
# ❌ Ineficiente: crea nueva lista cada vez
def sum_slow(items):
    if not items:
        return 0
    return items[0] + sum_slow(items[1:])  # items[1:] crea copia

# ✅ Eficiente: usar índice
def sum_fast(items, index=0):
    if index >= len(items):
        return 0
    return items[index] + sum_fast(items, index + 1)
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 7.1: Factorial y Fibonacci
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-71)

### Ejercicio 7.2: Suma y Máximo Recursivos
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-72)

### Ejercicio 7.3: Merge de Listas Ordenadas
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-73)

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Grokking Algorithms Ch.3-4](https://www.manning.com/books/grokking-algorithms) | Libro | 🔴 Obligatorio |
| [Recursion Visualizer](https://recursion.vercel.app/) | Herramienta | 🟡 Recomendado |
| [MIT Divide & Conquer](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/) | Curso | 🟢 Complementario |

---

## 🔗 Referencias del Glosario

- [Recursión](GLOSARIO.md#recursion)
- [Caso Base](GLOSARIO.md#caso-base)
- [Call Stack](GLOSARIO.md#call-stack)
- [Divide & Conquer](GLOSARIO.md#divide-conquer)
- [Memoization](GLOSARIO.md#memoization)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [06_INVERTED_INDEX](06_INVERTED_INDEX.md) | [00_INDICE](00_INDICE.md) | [08_SORTING](08_SORTING.md) |
