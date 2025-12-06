# 03 - Lógica y Matemáticas Discretas

> **🎯 Objetivo:** Dominar la teoría de conjuntos, lógica proposicional y la notación Big O para analizar algoritmos.

---

## 🧠 Analogía: El Lenguaje de las Computadoras

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   MATEMÁTICAS DISCRETAS = EL IDIOMA NATIVO DE LAS COMPUTADORAS              │
│   ─────────────────────────────────────────────────────────────             │
│                                                                             │
│   Las computadoras no entienden "más o menos" ni "aproximadamente"          │
│   Solo entienden: SÍ/NO, 0/1, VERDADERO/FALSO                               │
│                                                                             │
│   CONJUNTOS → Colecciones sin duplicados (sets en Python)                   │
│   LÓGICA    → Condiciones y decisiones (if/and/or)                          │
│   BIG O     → "¿Cuánto tarda?" sin medir con cronómetro                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Teoría de Conjuntos](#1-conjuntos)
2. [Operaciones de Conjuntos](#2-operaciones)
3. [Lógica Proposicional](#3-logica)
4. [Introducción a Big O](#4-big-o)
5. [Complejidad de Estructuras Python](#5-complejidad-python)

---

## 1. Teoría de Conjuntos {#1-conjuntos}

### 1.1 ¿Qué es un Conjunto?

```
┌─────────────────────────────────────────────────────────────────┐
│  CONJUNTO = Colección de elementos ÚNICOS sin orden             │
│                                                                 │
│  Lista:    [1, 2, 2, 3, 1]  → Permite duplicados, tiene orden   │
│  Conjunto: {1, 2, 3}        → Sin duplicados, sin orden         │
│                                                                 │
│  APLICACIÓN EN ARCHIMEDES:                                      │
│  • Stop words: {"the", "and", "or", "a", "an"}                  │
│  • Palabras únicas de un documento                              │
│  • Documentos que contienen una palabra                         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Sets en Python

```python
# Crear sets
stop_words: set[str] = {"the", "and", "or", "a", "an", "is", "are"}
empty_set: set[str] = set()  # No usar {} (eso es dict vacío)

# Crear set desde lista (elimina duplicados)
words = ["hello", "world", "hello", "python"]
unique_words = set(words)  # {"hello", "world", "python"}

# Verificar pertenencia: O(1) promedio
if "hello" in unique_words:
    print("Found!")

# Agregar y eliminar
unique_words.add("new")
unique_words.remove("hello")  # KeyError si no existe
unique_words.discard("missing")  # No error si no existe
```

### 1.3 frozenset: Conjuntos Inmutables

```python
# frozenset no se puede modificar
STOP_WORDS: frozenset[str] = frozenset({"the", "and", "or", "a", "an"})

# Útil como clave de diccionario o en otros sets
document_signatures: set[frozenset[str]] = set()
doc1_words = frozenset({"hello", "world"})
document_signatures.add(doc1_words)  # OK con frozenset

# Con set normal no funciona:
# document_signatures.add({"hello", "world"})  # TypeError: unhashable type: 'set'
```

---

## 2. Operaciones de Conjuntos {#2-operaciones}

### 2.1 Operaciones Fundamentales

```
┌─────────────────────────────────────────────────────────────────┐
│  A = {1, 2, 3}    B = {2, 3, 4}                                 │
│                                                                 │
│  UNIÓN (A ∪ B)         = {1, 2, 3, 4}   # Todos los elementos   │
│  INTERSECCIÓN (A ∩ B)  = {2, 3}         # Elementos comunes     │
│  DIFERENCIA (A - B)    = {1}            # En A pero no en B     │
│  DIFERENCIA SIMÉTRICA  = {1, 4}         # En uno pero no ambos  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 En Python

```python
A: set[int] = {1, 2, 3}
B: set[int] = {2, 3, 4}

# Unión
union = A | B                    # {1, 2, 3, 4}
union = A.union(B)               # Equivalente

# Intersección
intersection = A & B             # {2, 3}
intersection = A.intersection(B) # Equivalente

# Diferencia
difference = A - B               # {1}
difference = A.difference(B)     # Equivalente

# Diferencia simétrica
sym_diff = A ^ B                 # {1, 4}
sym_diff = A.symmetric_difference(B)
```

### 2.3 Aplicación: Búsqueda AND/OR

```python
def search_and(index: dict[str, set[int]], words: list[str]) -> set[int]:
    """Find documents containing ALL words (AND logic).
    
    Uses set intersection to find common documents.
    
    Example:
        >>> index = {"hello": {1, 2}, "world": {2, 3}}
        >>> search_and(index, ["hello", "world"])
        {2}  # Only doc 2 contains both words
    """
    if not words:
        return set()
    
    # Start with all docs containing first word
    result = index.get(words[0], set()).copy()
    
    # Intersect with docs containing each subsequent word
    for word in words[1:]:
        result &= index.get(word, set())
    
    return result


def search_or(index: dict[str, set[int]], words: list[str]) -> set[int]:
    """Find documents containing ANY word (OR logic).
    
    Uses set union to combine all matching documents.
    
    Example:
        >>> index = {"hello": {1, 2}, "world": {2, 3}}
        >>> search_or(index, ["hello", "world"])
        {1, 2, 3}  # Docs containing hello OR world
    """
    result: set[int] = set()
    
    for word in words:
        result |= index.get(word, set())
    
    return result
```

### 2.4 Subconjuntos y Superconjuntos

```python
A = {1, 2}
B = {1, 2, 3, 4}

A.issubset(B)    # True: A ⊆ B
B.issuperset(A)  # True: B ⊇ A
A < B            # True: A es subconjunto propio (A ⊂ B)
A.isdisjoint({5, 6})  # True: sin elementos en común
```

---

## 3. Lógica Proposicional {#3-logica}

### 3.1 Operadores Lógicos

```
┌─────────────────────────────────────────────────────────────────┐
│  OPERADOR    SÍMBOLO    PYTHON    SIGNIFICADO                   │
│  ─────────   ───────    ──────    ───────────                   │
│  AND         ∧          and       Ambos verdaderos              │
│  OR          ∨          or        Al menos uno verdadero        │
│  NOT         ¬          not       Negación                      │
│  IMPLICACIÓN →          if/then   Si A entonces B               │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Tablas de Verdad

```python
# AND: ambos deben ser True
#  A     B     A and B
# True  True   True
# True  False  False
# False True   False
# False False  False

# OR: al menos uno True
#  A     B     A or B
# True  True   True
# True  False  True
# False True   True
# False False  False

# NOT: invierte
#  A      not A
# True    False
# False   True
```

### 3.3 Expresiones Complejas en Python

```python
def is_valid_document(doc: Document) -> bool:
    """Check if document meets all validation criteria."""
    has_content = len(doc.content) > 0
    has_valid_id = doc.doc_id >= 0
    is_not_too_long = len(doc.content) < 1_000_000
    
    # AND: todas las condiciones
    return has_content and has_valid_id and is_not_too_long


def should_index_word(word: str, stop_words: set[str]) -> bool:
    """Determine if word should be indexed.
    
    Index if:
    - Word is not a stop word, AND
    - Word has at least 2 characters, AND
    - Word is alphanumeric
    """
    is_not_stopword = word not in stop_words
    is_long_enough = len(word) >= 2
    is_alphanumeric = word.isalnum()
    
    return is_not_stopword and is_long_enough and is_alphanumeric
```

### 3.4 Leyes de De Morgan

```
┌─────────────────────────────────────────────────────────────────┐
│  LEY DE DE MORGAN                                               │
│                                                                 │
│  not (A and B) = (not A) or (not B)                             │
│  not (A or B)  = (not A) and (not B)                            │
│                                                                 │
│  ÚTIL PARA SIMPLIFICAR CONDICIONES                              │
└─────────────────────────────────────────────────────────────────┘
```

```python
# Ejemplo: "no indexar si es stop word O es muy corta"
# Versión original
if not (word in stop_words or len(word) < 2):
    index_word(word)

# Aplicando De Morgan: equivalente
if word not in stop_words and len(word) >= 2:
    index_word(word)
```

---

## 4. Introducción a Big O {#4-big-o}

### 4.1 ¿Qué es Big O?

```
┌─────────────────────────────────────────────────────────────────┐
│  BIG O = Cómo crece el tiempo cuando crece la entrada           │
│                                                                 │
│  NO mide segundos exactos                                       │
│  SÍ mide: "¿Cuánto peor se pone con más datos?"                 │
│                                                                 │
│  Analogía: Enviar un paquete                                    │
│  ──────────────────────────                                     │
│  • O(1): Email (instantáneo, sin importar tamaño)               │
│  • O(n): Caminar (tiempo proporcional a distancia)              │
│  • O(n²): Revisar todas las combinaciones de n personas         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Complejidades Comunes

```
┌─────────────────────────────────────────────────────────────────┐
│  COMPLEJIDAD    NOMBRE          EJEMPLO                         │
│  ───────────    ──────          ───────                         │
│  O(1)           Constante       Acceso a dict por clave         │
│  O(log n)       Logarítmica     Binary search                   │
│  O(n)           Lineal          Recorrer una lista              │
│  O(n log n)     Linearítmica    QuickSort, MergeSort            │
│  O(n²)          Cuadrática      Dos loops anidados              │
│  O(2^n)         Exponencial     Subconjuntos de n elementos     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Visualización del Crecimiento

```
Tiempo ▲
       │                                          O(n²)
       │                                      ●
       │                                  ●
       │                              ●
       │                          ●                    O(n)
       │                      ●               ●────────●
       │                  ●           ●───────
       │              ●       ●───────        O(log n)
       │          ●   ●───────        ●───────●───────●
       │      ●───                    
       │  ●───────────────────────────────────────────  O(1)
       └──────────────────────────────────────────────► n (elementos)
            10   20   30   40   50   60   70   80
```

### 4.4 Cómo Determinar Big O

```python
# O(1) - Constante: no depende del tamaño de entrada
def get_first(items: list) -> any:
    return items[0]

# O(n) - Lineal: un loop sobre n elementos
def find_max(items: list[int]) -> int:
    max_val = items[0]
    for item in items:  # n iteraciones
        if item > max_val:
            max_val = item
    return max_val

# O(n²) - Cuadrática: loops anidados
def has_duplicate(items: list) -> bool:
    for i in range(len(items)):      # n
        for j in range(len(items)):  # × n
            if i != j and items[i] == items[j]:
                return True
    return False

# O(n) - Mejor versión con set
def has_duplicate_fast(items: list) -> bool:
    seen: set = set()
    for item in items:  # n iteraciones
        if item in seen:  # O(1) lookup
            return True
        seen.add(item)
    return False
```

### 4.5 Reglas para Calcular Big O

```
┌─────────────────────────────────────────────────────────────────┐
│  REGLA 1: Ignorar constantes                                    │
│  O(2n) → O(n)                                                   │
│  O(n + 100) → O(n)                                              │
│                                                                 │
│  REGLA 2: Tomar el término dominante                            │
│  O(n² + n) → O(n²)                                              │
│  O(n³ + n² + n) → O(n³)                                         │
│                                                                 │
│  REGLA 3: Operaciones en secuencia se suman                     │
│  f() de O(n) + g() de O(n²) → O(n + n²) → O(n²)                 │
│                                                                 │
│  REGLA 4: Loops anidados se multiplican                         │
│  for i in range(n):  # O(n)                                     │
│      for j in range(m):  # O(m)                                 │
│          ...             # → O(n × m)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Complejidad de Estructuras Python {#5-complejidad-python}

### 5.1 Tabla de Complejidades

| Operación | list | dict | set |
|-----------|------|------|-----|
| Acceso por índice | O(1) | - | - |
| Buscar elemento | O(n) | O(1)* | O(1)* |
| Insertar al final | O(1)* | O(1)* | O(1)* |
| Insertar al inicio | O(n) | - | - |
| Eliminar por valor | O(n) | O(1)* | O(1)* |
| Iterar todo | O(n) | O(n) | O(n) |

*Amortizado: en promedio, aunque casos raros pueden ser peores.

### 5.2 Por Qué Esto Importa

```python
# ❌ O(n) por cada búsqueda → O(n × m) total
def remove_stopwords_slow(tokens: list[str], stopwords: list[str]) -> list[str]:
    """Slow: O(n × m) where n=tokens, m=stopwords."""
    return [t for t in tokens if t not in stopwords]  # 'in' es O(m) en lista

# ✅ O(1) por cada búsqueda → O(n) total
def remove_stopwords_fast(tokens: list[str], stopwords: set[str]) -> list[str]:
    """Fast: O(n) where n=tokens."""
    return [t for t in tokens if t not in stopwords]  # 'in' es O(1) en set
```

### 5.3 Benchmark Real

```python
import time

# Crear datos de prueba
tokens = ["word" + str(i) for i in range(10000)]
stopwords_list = ["word" + str(i) for i in range(1000)]
stopwords_set = set(stopwords_list)

# Benchmark lista
start = time.time()
result = [t for t in tokens if t not in stopwords_list]
list_time = time.time() - start

# Benchmark set
start = time.time()
result = [t for t in tokens if t not in stopwords_set]
set_time = time.time() - start

print(f"List: {list_time:.4f}s")  # ~0.5s
print(f"Set:  {set_time:.4f}s")   # ~0.001s
print(f"Set is {list_time/set_time:.0f}x faster")  # ~500x
```

---

## ⚠️ Errores Comunes

### Error 1: Usar lista cuando set es mejor

```python
# ❌ Lento para búsquedas frecuentes
stop_words = ["the", "and", "or"]
if word in stop_words:  # O(n) cada vez
    pass

# ✅ Rápido
stop_words = {"the", "and", "or"}
if word in stop_words:  # O(1) cada vez
    pass
```

### Error 2: No considerar el tamaño de entrada

```python
# Parece simple, pero es O(n²)
def get_duplicates(items: list) -> list:
    duplicates = []
    for item in items:
        if items.count(item) > 1:  # count() es O(n)
            duplicates.append(item)
    return list(set(duplicates))

# Mejor: O(n)
from collections import Counter
def get_duplicates_fast(items: list) -> list:
    counts = Counter(items)  # O(n)
    return [item for item, count in counts.items() if count > 1]
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 3.1: Stop Words como Set
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-31)

### Ejercicio 3.2: Operaciones de Conjuntos
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-32)

### Ejercicio 3.3: Analizar Complejidad
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-33)

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Big O Cheat Sheet](https://www.bigocheatsheet.com/) | Referencia | 🔴 Obligatorio |
| [Python Time Complexity](https://wiki.python.org/moin/TimeComplexity) | Documentación | 🔴 Obligatorio |
| [Grokking Algorithms Ch.1](https://www.manning.com/books/grokking-algorithms) | Libro | 🟡 Recomendado |

---

## 🔗 Referencias del Glosario

- [Conjunto (Set)](GLOSARIO.md#set)
- [Big O Notation](GLOSARIO.md#big-o)
- [Complejidad Temporal](GLOSARIO.md#complejidad-temporal)
- [Hash Table](GLOSARIO.md#hash-table)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [02_OOP_DESDE_CERO](02_OOP_DESDE_CERO.md) | [00_INDICE](00_INDICE.md) | [04_ARRAYS_STRINGS](04_ARRAYS_STRINGS.md) |
