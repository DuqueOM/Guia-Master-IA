# 15 - Grafos, BFS y DFS

> **🎯 Objetivo:** Dominar grafos y sus algoritmos de recorrido - **tema CRÍTICO del Pathway**.

---

## 🧠 Analogía: Mapa de Ciudades y Carreteras

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   GRAFO = RED DE CONEXIONES                                                 │
│   ─────────────────────────                                                 │
│                                                                             │
│   Ciudades = NODOS (vertices)                                               │
│   Carreteras = ARISTAS (edges)                                              │
│                                                                             │
│       [A]───────[B]                                                         │
│        │ \       │                                                          │
│        │  \      │                                                          │
│        │   \     │                                                          │
│       [C]───[D]──[E]                                                        │
│                                                                             │
│   TIPOS:                                                                    │
│   • Dirigido: calles de un solo sentido (A→B no implica B→A)                │
│   • No dirigido: calles de dos sentidos (A↔B)                               │
│   • Ponderado: carreteras con distancias/costos                             │
│   • No ponderado: todas las conexiones iguales                              │
│                                                                             │
│   BFS = Explorar por NIVELES (círculos concéntricos)                        │
│   DFS = Explorar PROFUNDO primero (ir hasta el fondo, luego volver)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Representación de Grafos](#1-representacion)
2. [BFS (Breadth-First Search)](#2-bfs)
3. [DFS (Depth-First Search)](#3-dfs)
4. [Aplicaciones Comunes](#4-aplicaciones)
5. [Comparación BFS vs DFS](#5-comparacion)

---

## 1. Representación de Grafos {#1-representacion}

### 1.1 Adjacency List (Lista de Adyacencia)

```python
from collections import defaultdict
from typing import TypeVar, Generic

T = TypeVar('T')


class Graph(Generic[T]):
    """Unweighted graph using adjacency list.
    
    Most common representation. Good for sparse graphs.
    Space: O(V + E)
    """
    
    def __init__(self, directed: bool = False) -> None:
        self.adjacency: dict[T, list[T]] = defaultdict(list)
        self.directed = directed
    
    def add_vertex(self, vertex: T) -> None:
        """Add vertex without edges."""
        if vertex not in self.adjacency:
            self.adjacency[vertex] = []
    
    def add_edge(self, source: T, destination: T) -> None:
        """Add edge between vertices.
        
        For undirected graph, adds both directions.
        """
        self.adjacency[source].append(destination)
        
        if not self.directed:
            self.adjacency[destination].append(source)
    
    def get_neighbors(self, vertex: T) -> list[T]:
        """Get all neighbors of a vertex."""
        return self.adjacency.get(vertex, [])
    
    def get_vertices(self) -> list[T]:
        """Get all vertices."""
        return list(self.adjacency.keys())
    
    def __repr__(self) -> str:
        return f"Graph({dict(self.adjacency)})"


# Ejemplo de uso
graph = Graph[str](directed=False)
graph.add_edge("A", "B")
graph.add_edge("A", "C")
graph.add_edge("B", "D")
graph.add_edge("C", "D")
print(graph.get_neighbors("A"))  # ['B', 'C']
```

### 1.2 Adjacency Matrix (Matriz de Adyacencia)

```python
class GraphMatrix:
    """Graph using adjacency matrix.
    
    Good for dense graphs or when need O(1) edge lookup.
    Space: O(V²)
    """
    
    def __init__(self, num_vertices: int) -> None:
        self.num_vertices = num_vertices
        # matrix[i][j] = 1 if edge from i to j
        self.matrix: list[list[int]] = [
            [0] * num_vertices for _ in range(num_vertices)
        ]
    
    def add_edge(self, source: int, dest: int) -> None:
        """Add edge (undirected)."""
        self.matrix[source][dest] = 1
        self.matrix[dest][source] = 1
    
    def has_edge(self, source: int, dest: int) -> bool:
        """Check if edge exists. O(1)"""
        return self.matrix[source][dest] == 1
    
    def get_neighbors(self, vertex: int) -> list[int]:
        """Get neighbors. O(V)"""
        return [i for i, val in enumerate(self.matrix[vertex]) if val == 1]
```

### 1.3 Cuándo Usar Cada Representación

| Operación | Adj List | Adj Matrix |
|-----------|----------|------------|
| Space | O(V + E) | O(V²) |
| Check edge | O(degree) | O(1) |
| Get neighbors | O(1) | O(V) |
| Add edge | O(1) | O(1) |
| **Mejor para** | Sparse graphs | Dense graphs |

---

## 2. BFS (Breadth-First Search) {#2-bfs}

### 2.1 Concepto

```
┌─────────────────────────────────────────────────────────────────┐
│  BFS = Buscar por NIVELES (como ondas en el agua)               │
│                                                                 │
│  Desde A:                                                       │
│       [A]───[B]───[D]                                           │
│        │     │                                                  │
│       [C]───[E]                                                 │
│                                                                 │
│  Nivel 0: A                                                     │
│  Nivel 1: B, C (vecinos de A)                                   │
│  Nivel 2: D, E (vecinos de B y C no visitados)                  │
│                                                                 │
│  ORDEN DE VISITA: A → B → C → D → E                             │
│                                                                 │
│  USA QUEUE (FIFO) para procesar en orden de llegada             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementación

```python
from collections import deque


def bfs(graph: Graph[T], start: T) -> list[T]:
    """Breadth-First Search traversal.
    
    Visits nodes level by level using a queue.
    
    Args:
        graph: The graph to traverse.
        start: Starting vertex.
    
    Returns:
        List of vertices in BFS order.
    
    Time: O(V + E)
    Space: O(V) for visited set and queue
    """
    visited: set[T] = set()
    result: list[T] = []
    queue: deque[T] = deque([start])
    
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()  # FIFO
        result.append(vertex)
        
        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result


def bfs_with_levels(graph: Graph[T], start: T) -> list[list[T]]:
    """BFS that returns nodes grouped by level."""
    visited: set[T] = set()
    levels: list[list[T]] = []
    queue: deque[T] = deque([start])
    
    visited.add(start)
    
    while queue:
        level_size = len(queue)
        current_level: list[T] = []
        
        for _ in range(level_size):
            vertex = queue.popleft()
            current_level.append(vertex)
            
            for neighbor in graph.get_neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        levels.append(current_level)
    
    return levels
```

### 2.3 Shortest Path (Unweighted)

```python
def shortest_path_bfs(
    graph: Graph[T],
    start: T,
    end: T
) -> list[T] | None:
    """Find shortest path in unweighted graph.
    
    BFS guarantees shortest path in unweighted graphs
    because it explores level by level.
    
    Returns:
        List of vertices from start to end, or None if no path.
    """
    if start == end:
        return [start]
    
    visited: set[T] = set()
    queue: deque[tuple[T, list[T]]] = deque([(start, [start])])
    visited.add(start)
    
    while queue:
        vertex, path = queue.popleft()
        
        for neighbor in graph.get_neighbors(vertex):
            if neighbor == end:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None  # No path found
```

---

## 3. DFS (Depth-First Search) {#3-dfs}

### 3.1 Concepto

```
┌─────────────────────────────────────────────────────────────────┐
│  DFS = Ir lo más PROFUNDO posible, luego retroceder             │
│                                                                 │
│  Desde A:                                                       │
│       [A]───[B]───[D]                                           │
│        │     │                                                  │
│       [C]───[E]                                                 │
│                                                                 │
│  Camino: A → B → D (fondo!) → back → E → back → C               │
│                                                                 │
│  ORDEN DE VISITA: A → B → D → E → C                             │
│  (puede variar según orden de vecinos)                          │
│                                                                 │
│  USA STACK (LIFO) o RECURSIÓN                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementación Recursiva

```python
def dfs_recursive(graph: Graph[T], start: T) -> list[T]:
    """Depth-First Search using recursion.
    
    Time: O(V + E)
    Space: O(V) for visited + O(V) for call stack
    """
    visited: set[T] = set()
    result: list[T] = []
    
    def _dfs(vertex: T) -> None:
        visited.add(vertex)
        result.append(vertex)
        
        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                _dfs(neighbor)
    
    _dfs(start)
    return result
```

### 3.3 Implementación Iterativa (con Stack)

```python
def dfs_iterative(graph: Graph[T], start: T) -> list[T]:
    """Depth-First Search using explicit stack.
    
    Avoids recursion limit issues for large graphs.
    
    Note: Order may differ slightly from recursive
    due to stack vs recursion mechanics.
    """
    visited: set[T] = set()
    result: list[T] = []
    stack: list[T] = [start]
    
    while stack:
        vertex = stack.pop()  # LIFO
        
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            
            # Add neighbors to stack (reverse for same order as recursive)
            for neighbor in reversed(graph.get_neighbors(vertex)):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result
```

### 3.4 Detectar Ciclos con DFS

```python
def has_cycle_undirected(graph: Graph[T]) -> bool:
    """Detect cycle in undirected graph using DFS."""
    visited: set[T] = set()
    
    def _dfs(vertex: T, parent: T | None) -> bool:
        visited.add(vertex)
        
        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                if _dfs(neighbor, vertex):
                    return True
            elif neighbor != parent:
                # Found visited node that's not parent = cycle
                return True
        
        return False
    
    # Check all components (graph may be disconnected)
    for vertex in graph.get_vertices():
        if vertex not in visited:
            if _dfs(vertex, None):
                return True
    
    return False
```

---

## 4. Aplicaciones Comunes {#4-aplicaciones}

### 4.1 Encontrar Todos los Caminos

```python
def find_all_paths(
    graph: Graph[T],
    start: T,
    end: T
) -> list[list[T]]:
    """Find all paths from start to end using DFS."""
    all_paths: list[list[T]] = []
    
    def _dfs(vertex: T, path: list[T]) -> None:
        if vertex == end:
            all_paths.append(path.copy())
            return
        
        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in path:  # Avoid cycles
                path.append(neighbor)
                _dfs(neighbor, path)
                path.pop()  # Backtrack
    
    _dfs(start, [start])
    return all_paths
```

### 4.2 Componentes Conexos

```python
def count_connected_components(graph: Graph[T]) -> int:
    """Count number of connected components."""
    visited: set[T] = set()
    count = 0
    
    for vertex in graph.get_vertices():
        if vertex not in visited:
            # BFS/DFS from this vertex marks all reachable
            bfs_mark_visited(graph, vertex, visited)
            count += 1
    
    return count


def bfs_mark_visited(
    graph: Graph[T],
    start: T,
    visited: set[T]
) -> None:
    """Mark all reachable vertices as visited."""
    queue: deque[T] = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        for neighbor in graph.get_neighbors(vertex):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

---

## 5. Comparación BFS vs DFS {#5-comparacion}

### 5.1 Cuándo Usar Cada Uno

| Aspecto | BFS | DFS |
|---------|-----|-----|
| Estructura | Queue | Stack/Recursión |
| Explora | Por niveles | Por profundidad |
| Shortest Path | ✅ Garantizado* | ❌ No garantizado |
| Memoria | O(ancho del grafo) | O(profundidad) |
| Grafos anchos | ❌ Mucha memoria | ✅ Mejor |
| Grafos profundos | ✅ Mejor | ❌ Stack overflow |

\* Solo para grafos no ponderados

### 5.2 Resumen de Uso

```
USA BFS cuando:
• Necesitas shortest path (no ponderado)
• Explorar por niveles
• Grafos muy profundos (evita stack overflow)

USA DFS cuando:
• Necesitas explorar todos los caminos
• Detectar ciclos
• Topological sort
• Grafos muy anchos (menos memoria)
```

---

## ⚠️ Errores Comunes

### Error 1: Olvidar marcar como visitado ANTES de agregar a queue/stack

```python
# ❌ Puede agregar mismo nodo múltiples veces
if neighbor not in visited:
    queue.append(neighbor)
    # visited.add(neighbor)  # ¡Falta!

# ✅ Marcar inmediatamente
if neighbor not in visited:
    visited.add(neighbor)  # Antes de agregar
    queue.append(neighbor)
```

### Error 2: No manejar grafos desconectados

```python
# ❌ Solo visita un componente
def bfs_bad(graph, start):
    # Solo desde start...

# ✅ Iterar sobre todos los vértices
def bfs_all(graph):
    visited = set()
    for vertex in graph.get_vertices():
        if vertex not in visited:
            bfs(graph, vertex)  # Visita este componente
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 15.1: Implementar BFS
### Ejercicio 15.2: Implementar DFS recursivo e iterativo
### Ejercicio 15.3: Shortest path con BFS
### Ejercicio 15.4: Detectar ciclo en grafo

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Visualgo Graph](https://visualgo.net/en/dfsbfs) | Visual | 🔴 Obligatorio |
| [Abdul Bari BFS/DFS](https://www.youtube.com/watch?v=pcKY4hjDrxk) | Video | 🔴 Obligatorio |

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [14_TREES](14_TREES.md) | [00_INDICE](00_INDICE.md) | [16_DYNAMIC_PROGRAMMING](16_DYNAMIC_PROGRAMMING.md) |
