# ✅ Soluciones de Ejercicios

> Soluciones detalladas con explicaciones.

---

## Módulo 01: Python Profesional

### Solución 1.1: Type Hints

```python
def clean_text(text: str) -> str:
    return text.lower().strip()

def count_words(text: str) -> int:
    return len(text.split())

def get_unique_words(words: list[str]) -> list[str]:
    return list(set(words))
```

### Solución 1.2: Función Pura

```python
# ❌ Impura (modifica estado externo)
results = []
def add_to_results_impure(item):
    results.append(item)
    return len(results)

# ✅ Pura (no modifica estado externo)
def add_to_results_pure(results: list, item) -> tuple[list, int]:
    new_results = results + [item]
    return new_results, len(new_results)

# Uso:
my_results = []
my_results, count = add_to_results_pure(my_results, "item1")
```

### Solución 1.3: Docstrings

```python
def tokenize(text: str, min_length: int = 2) -> list[str]:
    """Tokenize text into words above minimum length.
    
    Args:
        text: Input text to tokenize.
        min_length: Minimum word length to include.
    
    Returns:
        List of lowercase tokens meeting length requirement.
    
    Example:
        >>> tokenize("Hello World", min_length=4)
        ['hello', 'world']
    """
    words = text.lower().split()
    return [w for w in words if len(w) >= min_length]
```

---

## Módulo 02: OOP

### Solución 2.1: Clase Document

```python
class Document:
    def __init__(self, doc_id: int, content: str) -> None:
        self.doc_id: int = doc_id
        self.content: str = content
        self.tokens: list[str] = []
    
    def tokenize(self) -> list[str]:
        self.tokens = self.content.lower().split()
        return self.tokens
```

### Solución 2.2: Métodos Mágicos

```python
class Document:
    def __init__(self, doc_id: int, content: str) -> None:
        self.doc_id = doc_id
        self.content = content
        self.tokens: list[str] = []
    
    def __repr__(self) -> str:
        return f"Document(doc_id={self.doc_id})"
    
    def __str__(self) -> str:
        return f"Doc #{self.doc_id}: {self.content[:30]}..."
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return NotImplemented
        return self.doc_id == other.doc_id
    
    def __len__(self) -> int:
        return len(self.tokens)
```

---

## Módulo 03: Lógica y Big O

### Solución 3.1: Stop Words con Set

```python
def filter_stopwords(tokens: list[str], stop_words: list[str]) -> list[str]:
    # Convertir a set para O(1) lookup
    stop_set = set(stop_words)  # O(m)
    # Filtrar en O(n) total
    return [t for t in tokens if t not in stop_set]

# Complejidad: O(n + m) en lugar de O(n × m)
```

### Solución 3.3: Analizar Complejidad

```python
# A: O(n) - un loop simple
# B: O(n²) - dos loops anidados, ambos van hasta n
# C: O(n²) - suma de 0+1+2+...+(n-1) = n(n-1)/2 = O(n²)
# D: O(log n) - divide por 2 cada iteración
# E: O(2^n) - árbol de llamadas crece exponencialmente
```

---

## Módulo 05: Hash Maps

### Solución 5.1: Contador de Frecuencias

```python
def word_frequencies(tokens: list[str]) -> dict[str, int]:
    """Count word frequencies. O(n) time."""
    frequencies: dict[str, int] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies

# Alternativa con Counter:
from collections import Counter
def word_frequencies_v2(tokens: list[str]) -> dict[str, int]:
    return dict(Counter(tokens))
```

---

## Módulo 06: Índice Invertido

### Solución 6.1: Índice Básico

```python
from collections import defaultdict

class InvertedIndex:
    def __init__(self) -> None:
        self._index: defaultdict[str, set[int]] = defaultdict(set)
    
    def add_document(self, doc_id: int, tokens: list[str]) -> None:
        for token in tokens:
            self._index[token].add(doc_id)
    
    def search(self, term: str) -> set[int]:
        return self._index.get(term, set()).copy()
```

---

## Módulo 07: Recursión

### Solución 7.1: Factorial y Fibonacci

```python
def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n: int) -> int:
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
```

### Solución 7.3: Merge

```python
def merge(left: list[int], right: list[int]) -> list[int]:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

---

## Módulo 08: Sorting

### Solución 8.1: QuickSort

```python
def quicksort(items: list[int]) -> list[int]:
    if len(items) <= 1:
        return items
    
    pivot = items[-1]
    less = [x for x in items[:-1] if x < pivot]
    equal = [x for x in items if x == pivot]
    greater = [x for x in items[:-1] if x > pivot]
    
    return quicksort(less) + equal + quicksort(greater)
```

### Solución 8.2: MergeSort

```python
def mergesort(items: list[int]) -> list[int]:
    if len(items) <= 1:
        return items.copy()
    
    mid = len(items) // 2
    left = mergesort(items[:mid])
    right = mergesort(items[mid:])
    
    return merge(left, right)
```

---

## Módulo 09: Binary Search

### Solución 9.1: Binary Search

```python
def binary_search(items: list[int], target: int) -> int:
    left, right = 0, len(items) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if items[mid] == target:
            return mid
        elif items[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

---

## Módulo 10: Álgebra Lineal

### Solución 10.3: Similitud de Coseno

```python
import math

def dot_product(v1: list[float], v2: list[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))

def magnitude(v: list[float]) -> float:
    return math.sqrt(sum(x ** 2 for x in v))

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = dot_product(v1, v2)
    mag1, mag2 = magnitude(v1), magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)
```

---

## Módulo 11: TF-IDF

### Solución 11.1-11.2: TF e IDF

```python
import math

def compute_tf(term: str, document: list[str]) -> float:
    if not document:
        return 0.0
    return document.count(term) / len(document)

def compute_idf(term: str, corpus: list[list[str]]) -> float:
    if not corpus:
        return 0.0
    docs_with_term = sum(1 for doc in corpus if term in doc)
    if docs_with_term == 0:
        return 0.0
    return math.log(len(corpus) / docs_with_term)
```

---

## Módulo 13: Linked Lists, Stacks, Queues

### Solución 13.1: Implementar Stack

```python
from typing import Generic, TypeVar

T = TypeVar('T')

class Stack(Generic[T]):
    """Stack LIFO implementation. All operations O(1)."""
    
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop()
    
    def peek(self) -> T:
        if self.is_empty():
            raise IndexError("Peek at empty stack")
        return self._items[-1]
    
    def is_empty(self) -> bool:
        return len(self._items) == 0
    
    def __len__(self) -> int:
        return len(self._items)
```

### Solución 13.2: Paréntesis Balanceados

```python
def is_balanced(expression: str) -> bool:
    """Check if parentheses are balanced. O(n)"""
    stack: list[str] = []
    matching = {')': '(', ']': '[', '}': '{'}
    
    for char in expression:
        if char in '([{':
            stack.append(char)
        elif char in ')]}':
            if not stack or stack.pop() != matching[char]:
                return False
    
    return len(stack) == 0

# Tests
assert is_balanced("()[]{}") == True
assert is_balanced("([{}])") == True
assert is_balanced("([)]") == False
assert is_balanced("((") == False
```

### Solución 13.4: Reverse Linked List

```python
class ListNode:
    def __init__(self, val: int = 0):
        self.val = val
        self.next: ListNode | None = None

def reverse_list(head: ListNode | None) -> ListNode | None:
    """Reverse linked list iteratively. O(n) time, O(1) space."""
    prev = None
    current = head
    
    while current:
        next_node = current.next  # Save next
        current.next = prev       # Reverse pointer
        prev = current            # Move prev forward
        current = next_node       # Move current forward
    
    return prev
```

---

## Módulo 14: Trees y BST

### Solución 14.1: BST con insert y search

```python
class TreeNode:
    def __init__(self, val: int):
        self.val = val
        self.left: TreeNode | None = None
        self.right: TreeNode | None = None

class BST:
    def __init__(self) -> None:
        self.root: TreeNode | None = None
    
    def insert(self, val: int) -> None:
        """Insert value. O(log n) average, O(n) worst."""
        if not self.root:
            self.root = TreeNode(val)
            return
        
        current = self.root
        while True:
            if val < current.val:
                if current.left is None:
                    current.left = TreeNode(val)
                    return
                current = current.left
            else:
                if current.right is None:
                    current.right = TreeNode(val)
                    return
                current = current.right
    
    def search(self, val: int) -> bool:
        """Search for value. O(log n) average."""
        current = self.root
        while current:
            if val == current.val:
                return True
            elif val < current.val:
                current = current.left
            else:
                current = current.right
        return False
```

### Solución 14.2: Tree Traversals

```python
def inorder(root: TreeNode | None) -> list[int]:
    """Left, Root, Right. Returns sorted for BST."""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root: TreeNode | None) -> list[int]:
    """Root, Left, Right."""
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def postorder(root: TreeNode | None) -> list[int]:
    """Left, Right, Root."""
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

### Solución 14.4: Altura del Árbol

```python
def tree_height(root: TreeNode | None) -> int:
    """Calculate tree height. O(n)"""
    if not root:
        return -1  # Empty tree has height -1
    return 1 + max(tree_height(root.left), tree_height(root.right))
```

---

## Módulo 15: Graphs

### Solución 15.2: BFS

```python
from collections import deque

def bfs(graph: dict[str, list[str]], start: str) -> list[str]:
    """BFS traversal. O(V + E)"""
    visited = set()
    result = []
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result
```

### Solución 15.3: DFS

```python
def dfs_recursive(graph: dict[str, list[str]], start: str) -> list[str]:
    """DFS recursive. O(V + E)"""
    visited = set()
    result = []
    
    def dfs(vertex: str) -> None:
        visited.add(vertex)
        result.append(vertex)
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(start)
    return result

def dfs_iterative(graph: dict[str, list[str]], start: str) -> list[str]:
    """DFS iterative with stack. O(V + E)"""
    visited = set()
    result = []
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            for neighbor in reversed(graph.get(vertex, [])):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result
```

### Solución 15.4: Shortest Path BFS

```python
def shortest_path(graph: dict[str, list[str]], start: str, end: str) -> list[str] | None:
    """Find shortest path in unweighted graph. O(V + E)"""
    if start == end:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        vertex, path = queue.popleft()
        
        for neighbor in graph.get(vertex, []):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None
```

---

## Módulo 16: Dynamic Programming

### Solución 16.2: Climbing Stairs

```python
def climb_stairs(n: int) -> int:
    """Count ways to climb n stairs (1 or 2 steps). O(n) time, O(1) space."""
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1
```

### Solución 16.3: Coin Change

```python
def coin_change(coins: list[int], amount: int) -> int:
    """Minimum coins for amount. O(amount * len(coins))"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a and dp[a - coin] != float('inf'):
                dp[a] = min(dp[a], dp[a - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
```

### Solución 16.4: Longest Common Subsequence

```python
def lcs(text1: str, text2: str) -> int:
    """Find LCS length. O(m * n)"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

---

## Módulo 17: Greedy

### Solución 17.1: Activity Selection

```python
def activity_selection(start: list[int], end: list[int]) -> list[int]:
    """Select max non-overlapping activities. O(n log n)"""
    activities = sorted(zip(start, end, range(len(start))), key=lambda x: x[1])
    
    selected = [activities[0][2]]
    last_end = activities[0][1]
    
    for s, e, idx in activities[1:]:
        if s >= last_end:
            selected.append(idx)
            last_end = e
    
    return selected
```

### Solución 17.3: Jump Game

```python
def can_jump(nums: list[int]) -> bool:
    """Can reach last index? O(n)"""
    farthest = 0
    for i, jump in enumerate(nums):
        if i > farthest:
            return False
        farthest = max(farthest, i + jump)
        if farthest >= len(nums) - 1:
            return True
    return True
```

---

## Módulo 18: Heaps

### Solución 18.2: K Largest Elements

```python
import heapq

def k_largest(nums: list[int], k: int) -> list[int]:
    """Find k largest elements. O(n log k)"""
    heap: list[int] = []
    
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heapreplace(heap, num)
    
    return sorted(heap, reverse=True)
```

### Solución 18.3: Top K Frequent

```python
import heapq
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """Find k most frequent elements. O(n log k)"""
    count = Counter(nums)
    heap: list[tuple[int, int]] = []
    
    for num, freq in count.items():
        if len(heap) < k:
            heapq.heappush(heap, (freq, num))
        elif freq > heap[0][0]:
            heapq.heapreplace(heap, (freq, num))
    
    return [num for freq, num in heap]
```

---

## 💡 Notas

- Cada solución incluye la complejidad óptima
- Verifica tus soluciones comparando con estas
- Si tu solución es diferente pero correcta, ¡está bien!
