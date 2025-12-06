# 18 - Heaps y Priority Queues

> **🎯 Objetivo:** Dominar heaps para problemas de "top K" y scheduling.

---

## 🧠 Analogía: La Sala de Emergencias

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PRIORITY QUEUE = Sala de emergencias del hospital                         │
│   ─────────────────────────────────────────────────                         │
│                                                                             │
│   No es "primero en llegar, primero en atender" (FIFO)                      │
│   Es "más urgente primero"                                                  │
│                                                                             │
│   Pacientes: [dolor cabeza, infarto, gripe, accidente]                      │
│   Orden de atención: infarto → accidente → gripe → dolor cabeza             │
│                                                                             │
│   HEAP = Estructura eficiente para implementar Priority Queue               │
│                                                                             │
│   MIN-HEAP: El elemento más pequeño siempre arriba                          │
│   MAX-HEAP: El elemento más grande siempre arriba                           │
│                                                                             │
│                 1 (min)                    9 (max)                          │
│                / \                        / \                               │
│               3   2                      7   8                              │
│              / \                        / \                                 │
│             5   4                      3   5                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Estructura del Heap](#1-estructura)
2. [Operaciones Básicas](#2-operaciones)
3. [heapq en Python](#3-heapq)
4. [Problemas Clásicos](#4-problemas)

---

## 1. Estructura del Heap {#1-estructura}

### 1.1 Propiedades del Heap

```
HEAP PROPERTY (Min-Heap):
- Cada nodo es MENOR o igual que sus hijos
- El mínimo siempre está en la raíz

COMPLETE BINARY TREE:
- Todos los niveles llenos excepto el último
- Último nivel lleno de izquierda a derecha

REPRESENTACIÓN EN ARRAY:
Para nodo en índice i:
- Parent: (i - 1) // 2
- Left child: 2*i + 1
- Right child: 2*i + 2

Array: [1, 3, 2, 5, 4]

        1 (idx 0)
       / \
      3   2 (idx 1, 2)
     / \
    5   4 (idx 3, 4)
```

### 1.2 Implementación desde Cero

```python
class MinHeap:
    """Min-heap implementation from scratch.
    
    Smallest element always at root (index 0).
    """
    
    def __init__(self) -> None:
        self._heap: list[int] = []
    
    def _parent(self, i: int) -> int:
        return (i - 1) // 2
    
    def _left_child(self, i: int) -> int:
        return 2 * i + 1
    
    def _right_child(self, i: int) -> int:
        return 2 * i + 2
    
    def _swap(self, i: int, j: int) -> None:
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
    
    def _sift_up(self, i: int) -> None:
        """Move element up until heap property restored."""
        while i > 0:
            parent = self._parent(i)
            if self._heap[i] < self._heap[parent]:
                self._swap(i, parent)
                i = parent
            else:
                break
    
    def _sift_down(self, i: int) -> None:
        """Move element down until heap property restored."""
        size = len(self._heap)
        while True:
            smallest = i
            left = self._left_child(i)
            right = self._right_child(i)
            
            if left < size and self._heap[left] < self._heap[smallest]:
                smallest = left
            if right < size and self._heap[right] < self._heap[smallest]:
                smallest = right
            
            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break
    
    def push(self, value: int) -> None:
        """Add element to heap. O(log n)"""
        self._heap.append(value)
        self._sift_up(len(self._heap) - 1)
    
    def pop(self) -> int:
        """Remove and return minimum. O(log n)"""
        if not self._heap:
            raise IndexError("Pop from empty heap")
        
        min_val = self._heap[0]
        
        # Move last element to root
        self._heap[0] = self._heap[-1]
        self._heap.pop()
        
        # Restore heap property
        if self._heap:
            self._sift_down(0)
        
        return min_val
    
    def peek(self) -> int:
        """Return minimum without removing. O(1)"""
        if not self._heap:
            raise IndexError("Peek at empty heap")
        return self._heap[0]
    
    def __len__(self) -> int:
        return len(self._heap)
    
    def __bool__(self) -> bool:
        return len(self._heap) > 0
```

---

## 2. Operaciones Básicas {#2-operaciones}

### 2.1 Complejidades

| Operación | Complejidad | Descripción |
|-----------|-------------|-------------|
| push | O(log n) | Agregar elemento |
| pop | O(log n) | Extraer mínimo/máximo |
| peek | O(1) | Ver mínimo/máximo |
| heapify | O(n) | Convertir array a heap |

### 2.2 Heapify (Convertir Array a Heap)

```python
def heapify(arr: list[int]) -> None:
    """Convert array to min-heap in-place. O(n)
    
    Start from last non-leaf node and sift down.
    """
    n = len(arr)
    
    # Start from last non-leaf node
    for i in range(n // 2 - 1, -1, -1):
        _sift_down_arr(arr, n, i)


def _sift_down_arr(arr: list[int], n: int, i: int) -> None:
    """Sift down for array."""
    while True:
        smallest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and arr[left] < arr[smallest]:
            smallest = left
        if right < n and arr[right] < arr[smallest]:
            smallest = right
        
        if smallest != i:
            arr[i], arr[smallest] = arr[smallest], arr[i]
            i = smallest
        else:
            break
```

---

## 3. heapq en Python {#3-heapq}

### 3.1 Uso Básico

```python
import heapq

# MIN-HEAP (default en Python)
heap: list[int] = []

heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
heapq.heappush(heap, 1)

print(heap)  # [1, 1, 4, 3] - estructura interna

print(heapq.heappop(heap))  # 1 (mínimo)
print(heapq.heappop(heap))  # 1
print(heapq.heappop(heap))  # 3

# Peek without removing
print(heap[0])  # 4 (mínimo actual)

# Convert existing list to heap
nums = [5, 3, 8, 1, 2]
heapq.heapify(nums)  # O(n) - modifica in-place
print(nums)  # [1, 2, 8, 5, 3]
```

### 3.2 Max-Heap Trick

```python
# Python solo tiene min-heap
# Para max-heap: negar los valores

import heapq

nums = [3, 1, 4, 1, 5]
max_heap = [-x for x in nums]
heapq.heapify(max_heap)

# Extraer máximo
max_val = -heapq.heappop(max_heap)  # 5

# Insertar
heapq.heappush(max_heap, -10)  # Insertar 10
```

### 3.3 Heap con Tuplas (Prioridad Custom)

```python
import heapq

# (priority, data) - ordena por primer elemento
tasks = []
heapq.heappush(tasks, (3, "low priority"))
heapq.heappush(tasks, (1, "high priority"))
heapq.heappush(tasks, (2, "medium priority"))

while tasks:
    priority, task = heapq.heappop(tasks)
    print(f"Processing: {task}")
# high priority → medium priority → low priority
```

---

## 4. Problemas Clásicos {#4-problemas}

### 4.1 K Largest Elements

```python
import heapq


def k_largest(nums: list[int], k: int) -> list[int]:
    """Find k largest elements.
    
    Use min-heap of size k.
    
    Time: O(n log k)
    Space: O(k)
    """
    # Min-heap keeps k largest seen so far
    heap: list[int] = []
    
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heapreplace(heap, num)  # pop + push
    
    return sorted(heap, reverse=True)


# Alternative using nlargest
def k_largest_simple(nums: list[int], k: int) -> list[int]:
    return heapq.nlargest(k, nums)
```

### 4.2 Merge K Sorted Lists

```python
import heapq
from typing import Optional


class ListNode:
    def __init__(self, val: int = 0, next: 'ListNode' = None):
        self.val = val
        self.next = next


def merge_k_lists(lists: list[Optional[ListNode]]) -> Optional[ListNode]:
    """Merge k sorted linked lists.
    
    Use heap to always get smallest current element.
    
    Time: O(N log k) where N = total elements
    """
    heap: list[tuple[int, int, ListNode]] = []
    
    # Add first node of each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    
    dummy = ListNode()
    current = dummy
    
    while heap:
        val, idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next
        
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))
    
    return dummy.next
```

### 4.3 Kth Smallest in Matrix

```python
import heapq


def kth_smallest(matrix: list[list[int]], k: int) -> int:
    """Find kth smallest in row/col sorted matrix.
    
    Use heap to explore in sorted order.
    """
    n = len(matrix)
    # (value, row, col)
    heap = [(matrix[0][0], 0, 0)]
    visited = {(0, 0)}
    
    for _ in range(k):
        val, r, c = heapq.heappop(heap)
        
        # Add right neighbor
        if c + 1 < n and (r, c + 1) not in visited:
            visited.add((r, c + 1))
            heapq.heappush(heap, (matrix[r][c + 1], r, c + 1))
        
        # Add bottom neighbor
        if r + 1 < n and (r + 1, c) not in visited:
            visited.add((r + 1, c))
            heapq.heappush(heap, (matrix[r + 1][c], r + 1, c))
    
    return val
```

### 4.4 Top K Frequent Elements

```python
import heapq
from collections import Counter


def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """Find k most frequent elements.
    
    Time: O(n log k)
    """
    count = Counter(nums)
    
    # Min-heap of (frequency, element)
    heap: list[tuple[int, int]] = []
    
    for num, freq in count.items():
        if len(heap) < k:
            heapq.heappush(heap, (freq, num))
        elif freq > heap[0][0]:
            heapq.heapreplace(heap, (freq, num))
    
    return [num for freq, num in heap]
```

### 4.5 Median from Data Stream

```python
import heapq


class MedianFinder:
    """Find median from continuous data stream.
    
    Use two heaps:
    - max_heap: smaller half
    - min_heap: larger half
    """
    
    def __init__(self):
        self.small: list[int] = []  # max-heap (negated)
        self.large: list[int] = []  # min-heap
    
    def add_num(self, num: int) -> None:
        """Add number to stream. O(log n)"""
        # Add to max-heap (small)
        heapq.heappush(self.small, -num)
        
        # Balance: move largest from small to large
        heapq.heappush(self.large, -heapq.heappop(self.small))
        
        # Ensure small has >= elements than large
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def find_median(self) -> float:
        """Get current median. O(1)"""
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

---

## ⚠️ Errores Comunes

### Error 1: Olvidar que heapq es min-heap

```python
# ❌ Esperar max-heap
import heapq
heap = [3, 1, 4]
heapq.heapify(heap)
print(heapq.heappop(heap))  # 1, no 4!

# ✅ Para max: negar valores
max_heap = [-x for x in [3, 1, 4]]
heapq.heapify(max_heap)
print(-heapq.heappop(max_heap))  # 4
```

### Error 2: Modificar heap directamente

```python
# ❌ Rompe la propiedad del heap
heap[0] = 100  # ¡No hacer!

# ✅ Usar operaciones del heap
heapq.heapreplace(heap, new_value)  # pop + push
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 18.1: Implementar MinHeap desde cero
### Ejercicio 18.2: K Largest Elements
### Ejercicio 18.3: Top K Frequent
### Ejercicio 18.4: Merge K Sorted Lists

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Visualgo Heap](https://visualgo.net/en/heap) | Visual | 🔴 Obligatorio |
| [heapq Documentation](https://docs.python.org/3/library/heapq.html) | Docs | 🔴 Obligatorio |

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [17_GREEDY](17_GREEDY.md) | [00_INDICE](00_INDICE.md) | [12_PROYECTO_INTEGRADOR](12_PROYECTO_INTEGRADOR.md) |
