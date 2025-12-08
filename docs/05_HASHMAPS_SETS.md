# Anexo DSA - Hash Maps y Sets

> **⚠️ MÓDULO OPCIONAL:** Este módulo NO es requerido para el Pathway. Es útil para entrevistas técnicas.  
> **🎯 Objetivo:** Dominar diccionarios y sets en Python.

---

## 🧠 Analogía: El Índice de un Libro vs Leer Página por Página

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   LISTA = LIBRO SIN ÍNDICE                                                  │
│   ────────────────────────                                                  │
│   Para encontrar "recursión" debes leer página por página → O(n)            │
│                                                                             │
│   DICCIONARIO = LIBRO CON ÍNDICE ALFABÉTICO                                 │
│   ──────────────────────────────────────────                                │
│   Buscas "recursión" en el índice → página 142 → directo → O(1)             │
│                                                                             │
│   ¿CÓMO FUNCIONA EL "ÍNDICE"?                                               │
│   ────────────────────────────                                              │
│   HASH FUNCTION: Convierte "recursión" → número → posición en memoria       │
│   "recursión" → hash() → 7293847 → slot 47 en el array interno              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Cómo Funciona un Hash Map](#1-como-funciona)
2. [Diccionarios en Python](#2-diccionarios)
3. [Sets: Conjuntos con Hash](#3-sets)
4. [Colisiones y Resolución](#4-colisiones)
5. [Aplicación: Contador de Frecuencias](#5-aplicacion)

---

## 1. Cómo Funciona un Hash Map {#1-como-funciona}

### 1.1 La Función Hash

```
┌─────────────────────────────────────────────────────────────────┐
│  HASH FUNCTION: Convierte cualquier dato en un número           │
│                                                                 │
│  "hello" → hash("hello") → 2314058222102390712                  │
│  "world" → hash("world") → 6736076307280336625                  │
│                                                                 │
│  PROPIEDADES IMPORTANTES:                                       │
│  ✅ Mismo input → siempre mismo output (determinista)          │
│  ✅ Rápido de calcular                                         │
│  ✅ Distribuye bien los valores (pocos "choques")              │
│  ❌ Diferente input puede dar mismo output (colisión)          │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Del Hash a la Posición

```python
# Internamente, un diccionario es un array
# El hash determina dónde guardar el valor

def simplified_hash_position(key: str, array_size: int) -> int:
    """Simplified example of how position is calculated.
    
    Real implementation is more complex.
    """
    hash_value = hash(key)
    position = hash_value % array_size  # Módulo para que quepa
    return position

# Ejemplo conceptual (NO es implementación real)
# dict con 8 slots internos:
# "hello" → hash → 2314058... → 2314058 % 8 = 2 → slot[2]
# "world" → hash → 6736076... → 6736076 % 8 = 1 → slot[1]
```

### 1.3 Por Qué es O(1)

```
┌─────────────────────────────────────────────────────────────────┐
│  LISTA: Buscar "hello" en ["world", "python", "hello", ...]     │
│  ─────────────────────────────────────────────────────────────  │
│  1. Comparar con "world" → NO                                   │
│  2. Comparar con "python" → NO                                  │
│  3. Comparar con "hello" → SÍ                                   │
│  → Peor caso: revisar TODOS los n elementos → O(n)              │
│                                                                 │
│  DICCIONARIO: Buscar "hello"                                    │
│  ────────────────────────────                                   │
│  1. Calcular hash("hello") → 2314058                            │
│  2. Ir directo a slot[2314058 % size]                           │
│  3. Verificar que la clave coincide                             │
│  → Siempre ~3 pasos, sin importar tamaño → O(1)                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Diccionarios en Python {#2-diccionarios}

### 2.1 Creación y Acceso Básico

```python
# Crear diccionarios
word_counts: dict[str, int] = {"hello": 5, "world": 3}
empty: dict[str, int] = {}
from_pairs = dict([("a", 1), ("b", 2)])

# Acceso: O(1)
count = word_counts["hello"]  # 5
# word_counts["missing"]  # KeyError!

# Acceso seguro: O(1)
count = word_counts.get("hello")      # 5
count = word_counts.get("missing")    # None
count = word_counts.get("missing", 0) # 0 (default)

# Verificar existencia: O(1)
if "hello" in word_counts:
    print("Found!")

# Asignar: O(1)
word_counts["new"] = 10
word_counts["hello"] = 6  # Sobrescribe
```

### 2.2 Métodos Importantes

```python
word_counts = {"hello": 5, "world": 3, "python": 7}

# Obtener claves, valores, pares
keys = word_counts.keys()       # dict_keys(['hello', 'world', 'python'])
values = word_counts.values()   # dict_values([5, 3, 7])
items = word_counts.items()     # dict_items([('hello', 5), ...])

# Iterar
for word in word_counts:        # Itera sobre claves
    print(word)

for word, count in word_counts.items():
    print(f"{word}: {count}")

# Eliminar: O(1)
del word_counts["hello"]
count = word_counts.pop("world")  # Retorna valor y elimina
count = word_counts.pop("missing", 0)  # Default si no existe

# Actualizar con otro diccionario
word_counts.update({"new": 1, "python": 10})

# setdefault: obtener o insertar default
word_counts.setdefault("java", 0)  # Inserta "java": 0 si no existe
```

### 2.3 defaultdict: Diccionario con Default Automático

```python
from collections import defaultdict

# ❌ Con dict normal, necesitas verificar existencia
word_counts: dict[str, int] = {}
for word in ["a", "b", "a", "c", "a"]:
    if word not in word_counts:
        word_counts[word] = 0
    word_counts[word] += 1

# ✅ Con defaultdict, el default se crea automáticamente
word_counts: defaultdict[str, int] = defaultdict(int)  # int() = 0
for word in ["a", "b", "a", "c", "a"]:
    word_counts[word] += 1  # Si no existe, crea con valor 0

print(dict(word_counts))  # {'a': 3, 'b': 1, 'c': 1}

# defaultdict con lista
index: defaultdict[str, list[int]] = defaultdict(list)
index["hello"].append(1)  # Crea lista vacía si no existe
index["hello"].append(5)
print(dict(index))  # {'hello': [1, 5]}
```

### 2.4 Counter: Diccionario para Contar

```python
from collections import Counter

words = ["apple", "banana", "apple", "cherry", "banana", "apple"]

# Contar frecuencias
counts = Counter(words)
print(counts)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})

# Acceso como diccionario
print(counts["apple"])   # 3
print(counts["missing"]) # 0 (no KeyError!)

# Métodos útiles
print(counts.most_common(2))  # [('apple', 3), ('banana', 2)]

# Operaciones matemáticas
more_words = Counter(["apple", "date"])
total = counts + more_words  # Suma conteos
```

---

## 3. Sets: Conjuntos con Hash {#3-sets}

### 3.1 Internamente, un Set es un Dict sin Valores

```
┌─────────────────────────────────────────────────────────────────┐
│  SET: Solo almacena las claves, sin valores asociados           │
│                                                                 │
│  Internamente:                                                  │
│  set({"a", "b", "c"}) ≈ {"a": None, "b": None, "c": None}       │
│                                                                 │
│  Por eso tiene las mismas complejidades O(1) que dict           │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Operaciones y Complejidad

```python
words: set[str] = {"hello", "world"}

# Agregar: O(1)
words.add("python")

# Verificar: O(1) - ¡Esta es la operación clave!
if "hello" in words:
    print("Found!")

# Eliminar: O(1)
words.remove("hello")     # KeyError si no existe
words.discard("missing")  # No error si no existe

# Operaciones de conjuntos: O(min(len(a), len(b)))
a = {1, 2, 3}
b = {2, 3, 4}
union = a | b          # {1, 2, 3, 4}
intersection = a & b   # {2, 3}
difference = a - b     # {1}
```

### 3.3 Cuándo Usar Set vs List

| Operación | List | Set | Usar Set cuando... |
|-----------|------|-----|-------------------|
| `x in collection` | O(n) | O(1) | Muchas búsquedas |
| Mantener orden | ✅ | ❌ | Orden no importa |
| Permitir duplicados | ✅ | ❌ | Solo necesitas únicos |
| Acceso por índice | ✅ | ❌ | No necesitas índices |

```python
# ❌ Lento: verificar stop words en lista
stop_words_list = ["the", "a", "an", "and", "or", "but", ...]

def is_stopword_slow(word: str) -> bool:
    return word in stop_words_list  # O(n) cada vez

# ✅ Rápido: verificar en set
stop_words_set = {"the", "a", "an", "and", "or", "but", ...}

def is_stopword_fast(word: str) -> bool:
    return word in stop_words_set  # O(1) cada vez
```

---

## 4. Colisiones y Resolución {#4-colisiones}

### 4.1 ¿Qué es una Colisión?

```
┌─────────────────────────────────────────────────────────────────┐
│  COLISIÓN: Dos claves diferentes → mismo slot                   │
│                                                                 │
│  "hello" → hash → 47293 % 8 = 5 → slot[5]                       │
│  "world" → hash → 82645 % 8 = 5 → slot[5]  ← ¡MISMO SLOT!       │
│                                                                 │
│  Python resuelve esto con "open addressing":                    │
│  Si slot[5] está ocupado, busca slot[6], slot[7], etc.          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Por Qué Sigue Siendo O(1)

```
Python mantiene el diccionario "poco lleno" (load factor < 2/3)
Cuando se llena demasiado, lo hace más grande y redistribuye

Con buen factor de carga:
- Promedio: 1-2 comparaciones por búsqueda → O(1) amortizado
- Peor caso (muy raro): O(n) si todas las claves colisionan
```

### 4.3 Qué Puede Ser Clave de Diccionario

```python
# ✅ HASHABLE: tipos inmutables
d = {}
d["string"] = 1        # str: OK
d[42] = 2              # int: OK
d[3.14] = 3            # float: OK
d[(1, 2, 3)] = 4       # tuple: OK
d[frozenset({1,2})] = 5  # frozenset: OK

# ❌ NO HASHABLE: tipos mutables
# d[[1, 2, 3]] = 6     # TypeError: unhashable type: 'list'
# d[{1, 2}] = 7        # TypeError: unhashable type: 'set'
# d[{"a": 1}] = 8      # TypeError: unhashable type: 'dict'

# ¿Por qué? Si el objeto cambia, su hash cambiaría
# y no lo encontraríamos donde lo guardamos
```

---

## 5. Aplicación: Contador de Frecuencias {#5-aplicacion}

### 5.1 Contador Manual

```python
def count_word_frequencies(tokens: list[str]) -> dict[str, int]:
    """Count frequency of each word in token list.
    
    Args:
        tokens: List of words to count.
    
    Returns:
        Dictionary mapping words to their counts.
    
    Complexity:
        O(n) where n = len(tokens)
    
    Example:
        >>> count_word_frequencies(["a", "b", "a"])
        {'a': 2, 'b': 1}
    """
    frequencies: dict[str, int] = {}
    
    for token in tokens:
        # O(1) lookup + O(1) assignment
        frequencies[token] = frequencies.get(token, 0) + 1
    
    return frequencies
```

### 5.2 Con defaultdict

```python
from collections import defaultdict

def count_frequencies_defaultdict(tokens: list[str]) -> dict[str, int]:
    """Count frequencies using defaultdict.
    
    Cleaner than manual .get() approach.
    """
    frequencies: defaultdict[str, int] = defaultdict(int)
    
    for token in tokens:
        frequencies[token] += 1
    
    return dict(frequencies)
```

### 5.3 Con Counter (Una Línea)

```python
from collections import Counter

def count_frequencies_counter(tokens: list[str]) -> dict[str, int]:
    """Count frequencies using Counter.
    
    Most Pythonic approach.
    """
    return dict(Counter(tokens))
```

### 5.4 Benchmark Comparativo

```python
import time
from collections import Counter, defaultdict

def benchmark_frequency_counters(tokens: list[str]) -> None:
    """Compare performance of different counting methods."""
    
    # Method 1: Manual with .get()
    start = time.time()
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    manual_time = time.time() - start
    
    # Method 2: defaultdict
    start = time.time()
    freq = defaultdict(int)
    for t in tokens:
        freq[t] += 1
    defaultdict_time = time.time() - start
    
    # Method 3: Counter
    start = time.time()
    freq = Counter(tokens)
    counter_time = time.time() - start
    
    print(f"Manual:      {manual_time:.4f}s")
    print(f"defaultdict: {defaultdict_time:.4f}s")
    print(f"Counter:     {counter_time:.4f}s")

# Con 1,000,000 tokens:
# Manual:      0.0800s
# defaultdict: 0.0750s
# Counter:     0.0650s  ← Más rápido (implementado en C)
```

### 5.5 Construyendo hacia el Índice Invertido

```python
from collections import defaultdict

def build_term_document_map(
    documents: list[tuple[int, list[str]]]
) -> dict[str, set[int]]:
    """Build mapping from terms to document IDs.
    
    This is the core of an inverted index.
    
    Args:
        documents: List of (doc_id, tokens) pairs.
    
    Returns:
        Dictionary mapping each term to set of doc IDs containing it.
    
    Example:
        >>> docs = [(1, ["hello", "world"]), (2, ["hello", "python"])]
        >>> build_term_document_map(docs)
        {'hello': {1, 2}, 'world': {1}, 'python': {2}}
    """
    term_to_docs: defaultdict[str, set[int]] = defaultdict(set)
    
    for doc_id, tokens in documents:
        for token in tokens:
            term_to_docs[token].add(doc_id)
    
    return dict(term_to_docs)
```

---

## ⚠️ Errores Comunes

### Error 1: Modificar dict mientras iteras

```python
# ❌ RuntimeError: dictionary changed size during iteration
word_counts = {"a": 1, "b": 2, "c": 3}
for word in word_counts:
    if word_counts[word] < 2:
        del word_counts[word]

# ✅ Iterar sobre copia de claves
for word in list(word_counts.keys()):
    if word_counts[word] < 2:
        del word_counts[word]

# ✅ O crear nuevo diccionario
word_counts = {w: c for w, c in word_counts.items() if c >= 2}
```

### Error 2: Asumir orden en versiones antiguas

```python
# Python 3.7+: dict mantiene orden de inserción
# Python < 3.7: NO garantiza orden

# Si necesitas orden garantizado, usa:
from collections import OrderedDict
```

### Error 3: Usar objeto mutable como clave

```python
# ❌ TypeError
cache = {}
cache[[1, 2, 3]] = "result"  # Lista no es hashable

# ✅ Convertir a tupla
cache[tuple([1, 2, 3])] = "result"
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 5.1: Contador de Frecuencias
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-51)

### Ejercicio 5.2: Benchmark List vs Set
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-52)

### Ejercicio 5.3: Term-Document Map
Ver [EJERCICIOS.md](EJERCICIOS.md#ejercicio-53)

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Python Dict Implementation](https://www.youtube.com/watch?v=npw4s1QTmPg) | Video | 🟡 Recomendado |
| [Time Complexity](https://wiki.python.org/moin/TimeComplexity) | Wiki | 🔴 Obligatorio |
| [collections Module](https://docs.python.org/3/library/collections.html) | Docs | 🟡 Recomendado |

---

## 🔗 Referencias del Glosario

- [Hash Map](GLOSARIO.md#hash-map)
- [Hash Function](GLOSARIO.md#hash-function)
- [Colisión](GLOSARIO.md#colision)
- [Set](GLOSARIO.md#set)
- [O(1) Amortizado](GLOSARIO.md#amortizado)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [04_ARRAYS_STRINGS](04_ARRAYS_STRINGS.md) | [00_INDICE](00_INDICE.md) | [06_INVERTED_INDEX](06_INVERTED_INDEX.md) |
