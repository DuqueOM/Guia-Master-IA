# 📝 Ejercicios Prácticos

> Ejercicios organizados por módulo con dificultad progresiva.

---

## Índice de Ejercicios

| Módulo | Tema | Dificultad | # Ejercicios |
|--------|------|------------|--------------|
| 01 | Python Profesional | 🟢 Básico | 4 |
| 02 | OOP | 🟢 Básico | 5 |
| 03 | Lógica y Big O | 🟡 Intermedio | 3 |
| 04 | Arrays y Strings | 🟢 Básico | 3 |
| 05 | Hash Maps | 🟡 Intermedio | 3 |
| 06 | Índice Invertido | 🟡 Intermedio | 3 |
| 07 | Recursión | 🟡 Intermedio | 3 |
| 08 | Sorting | 🔴 Avanzado | 3 |
| 09 | Binary Search | 🟡 Intermedio | 3 |
| 10 | Álgebra Lineal | 🟡 Intermedio | 3 |
| 11 | TF-IDF | 🔴 Avanzado | 3 |
| 13 | Linked Lists, Stacks, Queues | 🟡 Intermedio | 4 |
| 14 | Trees y BST | 🔴 Avanzado | 5 |
| 15 | Graphs, BFS, DFS | 🔴 Avanzado | 5 |
| 16 | Dynamic Programming | 🔴 Avanzado | 5 |
| 17 | Greedy Algorithms | 🟡 Intermedio | 4 |
| 18 | Heaps | 🔴 Avanzado | 4 |

---

## Módulo 01: Python Profesional

### Ejercicio 1.1: Type Hints Básicos
**Objetivo:** Agregar type hints a funciones existentes.

```python
# Agregar type hints a estas funciones:

def clean_text(text):
    return text.lower().strip()

def count_words(text):
    return len(text.split())

def get_unique_words(words):
    return list(set(words))
```

### Ejercicio 1.2: Función Pura
**Objetivo:** Convertir función impura a pura.

```python
# Convertir a función pura (sin modificar estado externo):
results = []

def add_to_results(item):
    results.append(item)
    return len(results)
```

### Ejercicio 1.3: Docstrings
**Objetivo:** Escribir docstrings estilo Google.

```python
# Agregar docstring completo con Args, Returns, Example:
def tokenize(text, min_length=2):
    words = text.lower().split()
    return [w for w in words if len(w) >= min_length]
```

### Ejercicio 1.4: Configurar Linters
**Objetivo:** Crear `pyproject.toml` con mypy y ruff configurados.

---

## Módulo 02: OOP

### Ejercicio 2.1: Clase Document Básica
**Objetivo:** Crear clase Document con `__init__`, atributos tipados.

```python
# Crear clase Document con:
# - doc_id: int
# - content: str
# - tokens: list[str] (vacía inicialmente)
# - Método tokenize() que llena tokens
```

### Ejercicio 2.2: Métodos Mágicos
**Objetivo:** Implementar `__repr__`, `__str__`, `__eq__`, `__len__`.

### Ejercicio 2.3: Properties
**Objetivo:** Agregar validación con properties para `doc_id` (>= 0) y `content` (no vacío).

### Ejercicio 2.4: Clase Corpus
**Objetivo:** Crear Corpus que contenga Documents con métodos add, get, remove.

### Ejercicio 2.5: SOLID
**Objetivo:** Refactorizar una clase "Dios" que hace todo en clases separadas.

---

## Módulo 03: Lógica y Big O

### Ejercicio 3.1: Stop Words como Set
**Objetivo:** Implementar filtrado de stop words usando set para O(1) lookup.

```python
# Dado:
stop_words_list = ["the", "a", "an", "is", "are"]
tokens = ["the", "quick", "brown", "fox", "is", "fast"]

# Implementar filter_stopwords() que sea O(n) no O(n×m)
```

### Ejercicio 3.2: Operaciones de Conjuntos
**Objetivo:** Implementar búsqueda AND y OR usando set operations.

### Ejercicio 3.3: Analizar Complejidad
**Objetivo:** Determinar Big O de 5 fragmentos de código dados.

```python
# ¿Cuál es la complejidad de cada uno?

# A
for i in range(n):
    print(i)

# B
for i in range(n):
    for j in range(n):
        print(i, j)

# C
for i in range(n):
    for j in range(i):
        print(i, j)

# D
i = n
while i > 0:
    print(i)
    i = i // 2

# E
def recursive(n):
    if n <= 1:
        return
    recursive(n - 1)
    recursive(n - 1)
```

---

## Módulo 04: Arrays y Strings

### Ejercicio 4.1: Manipulación de Listas
**Objetivo:** Implementar rotate_left(list, k) sin usar slicing.

### Ejercicio 4.2: Tokenizador
**Objetivo:** Implementar tokenizador completo con:
- Eliminar puntuación
- Convertir a minúsculas
- Filtrar por longitud mínima

### Ejercicio 4.3: Análisis de Complejidad
**Objetivo:** Comparar dos implementaciones de reverse y explicar cuál es mejor.

---

## Módulo 05: Hash Maps

### Ejercicio 5.1: Contador de Frecuencias
**Objetivo:** Implementar word_frequencies(tokens) → dict[str, int].

### Ejercicio 5.2: Benchmark List vs Set
**Objetivo:** Escribir script que mide tiempo de búsqueda en list vs set.

### Ejercicio 5.3: Term-Document Map
**Objetivo:** Construir diccionario term → set[doc_id].

---

## Módulo 06: Índice Invertido

### Ejercicio 6.1: Índice Básico
**Objetivo:** Implementar InvertedIndex con add_document() y search().

### Ejercicio 6.2: Búsqueda AND/OR
**Objetivo:** Agregar search_and() y search_or() al índice.

### Ejercicio 6.3: Índice con Frecuencias
**Objetivo:** Modificar índice para guardar frecuencia de cada término por documento.

---

## Módulo 07: Recursión

### Ejercicio 7.1: Factorial y Fibonacci
**Objetivo:** Implementar ambos recursivamente con casos base correctos.

### Ejercicio 7.2: Suma y Máximo
**Objetivo:** Implementar sum_list() y find_max() recursivamente.

### Ejercicio 7.3: Merge de Listas
**Objetivo:** Implementar merge(list1, list2) que fusiona dos listas ordenadas.

---

## Módulo 08: Sorting

### Ejercicio 8.1: QuickSort
**Objetivo:** Implementar quicksort() con partición Lomuto.

### Ejercicio 8.2: MergeSort
**Objetivo:** Implementar mergesort() con función merge() auxiliar.

### Ejercicio 8.3: Ordenar por Score
**Objetivo:** Ordenar lista de (doc_id, score) por score descendente usando tu quicksort.

---

## Módulo 09: Binary Search

### Ejercicio 9.1: Binary Search Básica
**Objetivo:** Implementar binary_search() iterativo sin errores off-by-one.

### Ejercicio 9.2: Primera y Última Ocurrencia
**Objetivo:** Implementar find_first() y find_last() para elementos repetidos.

### Ejercicio 9.3: Búsqueda de Umbral
**Objetivo:** Encontrar todos los documentos con score >= threshold en lista ordenada.

---

## Módulo 10: Álgebra Lineal

### Ejercicio 10.1: Operaciones Vectoriales
**Objetivo:** Implementar add_vectors(), subtract_vectors(), scalar_multiply().

### Ejercicio 10.2: Producto Punto y Norma
**Objetivo:** Implementar dot_product() y magnitude().

### Ejercicio 10.3: Similitud de Coseno
**Objetivo:** Implementar cosine_similarity() usando las funciones anteriores.

---

## Módulo 11: TF-IDF

### Ejercicio 11.1: Term Frequency
**Objetivo:** Implementar compute_tf(term, document).

### Ejercicio 11.2: Inverse Document Frequency
**Objetivo:** Implementar compute_idf(term, corpus).

### Ejercicio 11.3: Sistema de Ranking
**Objetivo:** Implementar rank_documents() que ordena por similitud de coseno.

---

## Módulo 13: Linked Lists, Stacks, Queues

### Ejercicio 13.1: Implementar Stack
**Objetivo:** Crear clase Stack con push, pop, peek, is_empty.

### Ejercicio 13.2: Paréntesis Balanceados
**Objetivo:** Verificar si string tiene paréntesis `()[]{}` balanceados usando Stack.

### Ejercicio 13.3: Implementar Queue
**Objetivo:** Crear clase Queue con enqueue, dequeue usando deque.

### Ejercicio 13.4: Reverse Linked List
**Objetivo:** Invertir una linked list iterativamente.

---

## Módulo 14: Trees y BST

### Ejercicio 14.1: Implementar BST
**Objetivo:** Crear clase BST con insert y search.

### Ejercicio 14.2: Tree Traversals
**Objetivo:** Implementar inorder, preorder, postorder (recursivo e iterativo).

### Ejercicio 14.3: Validar BST
**Objetivo:** Verificar si un árbol cumple la propiedad BST.

### Ejercicio 14.4: Altura del Árbol
**Objetivo:** Calcular altura de un árbol binario.

### Ejercicio 14.5: Level Order Traversal
**Objetivo:** Recorrer árbol por niveles usando Queue.

---

## Módulo 15: Graphs, BFS, DFS

### Ejercicio 15.1: Implementar Graph
**Objetivo:** Crear clase Graph con adjacency list.

### Ejercicio 15.2: BFS
**Objetivo:** Implementar Breadth-First Search.

### Ejercicio 15.3: DFS
**Objetivo:** Implementar Depth-First Search (recursivo e iterativo).

### Ejercicio 15.4: Shortest Path (Unweighted)
**Objetivo:** Encontrar camino más corto usando BFS.

### Ejercicio 15.5: Detectar Ciclo
**Objetivo:** Detectar si un grafo tiene ciclo usando DFS.

---

## Módulo 16: Dynamic Programming

### Ejercicio 16.1: Fibonacci con DP
**Objetivo:** Implementar con memoization y tabulation.

### Ejercicio 16.2: Climbing Stairs
**Objetivo:** Contar formas de subir n escaleras (1 o 2 pasos).

### Ejercicio 16.3: Coin Change
**Objetivo:** Mínimas monedas para un amount.

### Ejercicio 16.4: Longest Common Subsequence
**Objetivo:** Encontrar LCS de dos strings.

### Ejercicio 16.5: 0/1 Knapsack
**Objetivo:** Maximizar valor con capacidad limitada.

---

## Módulo 17: Greedy Algorithms

### Ejercicio 17.1: Activity Selection
**Objetivo:** Seleccionar máximas actividades no superpuestas.

### Ejercicio 17.2: Fractional Knapsack
**Objetivo:** Maximizar valor tomando fracciones de items.

### Ejercicio 17.3: Jump Game
**Objetivo:** Determinar si puedes llegar al final del array.

### Ejercicio 17.4: Minimum Meeting Rooms
**Objetivo:** Mínimas salas para todas las reuniones.

---

## Módulo 18: Heaps

### Ejercicio 18.1: Implementar MinHeap
**Objetivo:** Crear clase MinHeap con push, pop, peek.

### Ejercicio 18.2: K Largest Elements
**Objetivo:** Encontrar los k elementos más grandes.

### Ejercicio 18.3: Top K Frequent
**Objetivo:** Encontrar los k elementos más frecuentes.

### Ejercicio 18.4: Merge K Sorted Lists
**Objetivo:** Fusionar k listas ordenadas.

---

## 📚 Soluciones

Ver [EJERCICIOS_SOLUCIONES.md](EJERCICIOS_SOLUCIONES.md) para soluciones detalladas.

---

## 💡 Consejos

1. **Intenta primero:** No mires las soluciones hasta intentar al menos 30 minutos.
2. **Escribe tests:** Antes de implementar, escribe casos de prueba.
3. **Analiza complejidad:** Para cada solución, determina su Big O.
4. **Compara:** Después de resolver, compara con la solución oficial.
