# Anexo DSA - Árboles y Binary Search Trees

> **⚠️ MÓDULO OPCIONAL:** Este módulo NO es requerido para el Pathway. Es útil para entrevistas técnicas.  
> **🎯 Objetivo:** Dominar árboles binarios, BST y sus traversals.

---

## 🧠 Analogía: El Árbol Genealógico

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ÁRBOL = Estructura jerárquica como árbol genealógico                      │
│   ─────────────────────────────────────────────────────                     │
│                                                                             │
│                        [Abuelo]           ← ROOT (raíz)                     │
│                        /      \                                             │
│                   [Padre]    [Tío]        ← INTERNAL NODES                  │
│                   /    \        \                                           │
│               [Hijo1] [Hijo2]  [Primo]    ← LEAVES (hojas)                  │
│                                                                             │
│   TÉRMINOS:                                                                 │
│   • Root: Nodo sin padre (el de arriba)                                     │
│   • Parent/Child: Relación directa                                          │
│   • Siblings: Nodos con mismo padre                                         │
│   • Leaf: Nodo sin hijos                                                    │
│   • Height: Distancia máxima desde root a hoja                              │
│   • Depth: Distancia desde root a un nodo                                   │
│                                                                             │
│   BINARY TREE = Cada nodo tiene máximo 2 hijos (left, right)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Binary Tree Básico](#1-binary-tree)
2. [Traversals (Recorridos)](#2-traversals)
3. [Binary Search Tree (BST)](#3-bst)
4. [Operaciones en BST](#4-operaciones)
5. [Análisis de Complejidad](#5-analisis)

---

## 1. Binary Tree Básico {#1-binary-tree}

### 1.1 Estructura del Nodo

```python
from typing import Generic, TypeVar, Optional

T = TypeVar('T')


class TreeNode(Generic[T]):
    """A node in a binary tree.
    
    Attributes:
        value: Data stored in this node.
        left: Reference to left child (or None).
        right: Reference to right child (or None).
    """
    
    def __init__(self, value: T) -> None:
        self.value: T = value
        self.left: Optional[TreeNode[T]] = None
        self.right: Optional[TreeNode[T]] = None
    
    def __repr__(self) -> str:
        return f"TreeNode({self.value})"
    
    def is_leaf(self) -> bool:
        """Check if node has no children."""
        return self.left is None and self.right is None


class BinaryTree(Generic[T]):
    """Basic binary tree structure."""
    
    def __init__(self) -> None:
        self.root: Optional[TreeNode[T]] = None
    
    def is_empty(self) -> bool:
        return self.root is None
```

### 1.2 Construir un Árbol Manualmente

```python
#        10
#       /  \
#      5    15
#     / \   / \
#    3   7 12  20

root = TreeNode(10)
root.left = TreeNode(5)
root.right = TreeNode(15)
root.left.left = TreeNode(3)
root.left.right = TreeNode(7)
root.right.left = TreeNode(12)
root.right.right = TreeNode(20)
```

---

## 2. Traversals (Recorridos) {#2-traversals}

### 2.1 Los Tres Traversals DFS

```
┌─────────────────────────────────────────────────────────────────┐
│  TRES FORMAS DE RECORRER UN ÁRBOL (DFS)                         │
│                                                                 │
│        1                                                        │
│       / \                                                       │
│      2   3                                                      │
│     / \                                                         │
│    4   5                                                        │
│                                                                 │
│  INORDER (Left, Root, Right):   4, 2, 5, 1, 3                   │
│  → En BST: ¡sale ORDENADO!                                      │
│                                                                 │
│  PREORDER (Root, Left, Right):  1, 2, 4, 5, 3                   │
│  → Útil para copiar/serializar árbol                            │
│                                                                 │
│  POSTORDER (Left, Right, Root): 4, 5, 2, 3, 1                   │
│  → Útil para eliminar árbol (hijos antes que padre)             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Implementación Recursiva

```python
def inorder_recursive(node: Optional[TreeNode[T]]) -> list[T]:
    """Inorder traversal: Left, Root, Right.
    
    For BST, returns elements in sorted order.
    
    Time: O(n) - visit each node once
    Space: O(h) - recursion stack, h = height
    """
    if node is None:
        return []
    
    result = []
    result.extend(inorder_recursive(node.left))
    result.append(node.value)
    result.extend(inorder_recursive(node.right))
    return result


def preorder_recursive(node: Optional[TreeNode[T]]) -> list[T]:
    """Preorder traversal: Root, Left, Right."""
    if node is None:
        return []
    
    result = [node.value]
    result.extend(preorder_recursive(node.left))
    result.extend(preorder_recursive(node.right))
    return result


def postorder_recursive(node: Optional[TreeNode[T]]) -> list[T]:
    """Postorder traversal: Left, Right, Root."""
    if node is None:
        return []
    
    result = []
    result.extend(postorder_recursive(node.left))
    result.extend(postorder_recursive(node.right))
    result.append(node.value)
    return result
```

### 2.3 Implementación Iterativa (con Stack)

```python
def inorder_iterative(root: Optional[TreeNode[T]]) -> list[T]:
    """Inorder using explicit stack instead of recursion.
    
    Important: Shows how recursion uses the call stack.
    """
    result = []
    stack: list[TreeNode[T]] = []
    current = root
    
    while current is not None or stack:
        # Go as far left as possible
        while current is not None:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        result.append(current.value)
        
        # Move to right subtree
        current = current.right
    
    return result


def preorder_iterative(root: Optional[TreeNode[T]]) -> list[T]:
    """Preorder using stack."""
    if root is None:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.value)
        
        # Push right first so left is processed first (LIFO)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result
```

### 2.4 Level Order (BFS)

```python
from collections import deque


def level_order(root: Optional[TreeNode[T]]) -> list[list[T]]:
    """Level order traversal using queue (BFS).
    
    Returns nodes level by level.
    
    Example:
        [1]
        [2, 3]
        [4, 5]
    """
    if root is None:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.value)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

---

## 3. Binary Search Tree (BST) {#3-bst}

### 3.1 Propiedad del BST

```
┌─────────────────────────────────────────────────────────────────┐
│  BST PROPERTY:                                                  │
│  Para cada nodo:                                                │
│  • Todos los valores en subárbol izquierdo < valor del nodo     │
│  • Todos los valores en subárbol derecho > valor del nodo       │
│                                                                 │
│  VÁLIDO BST:           INVÁLIDO BST:                            │
│        10                    10                                 │
│       /  \                  /  \                                │
│      5    15               5    15                              │
│     / \                   / \                                   │
│    3   7                 3   12  ← 12 > 10 pero está en left!   │
│                                                                 │
│  BENEFICIO: Búsqueda O(log n) en promedio                       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Implementación de BST

```python
class BST(Generic[T]):
    """Binary Search Tree implementation.
    
    Maintains BST property: left < root < right.
    
    Average case complexities (balanced):
        - search: O(log n)
        - insert: O(log n)
        - delete: O(log n)
    
    Worst case (unbalanced/skewed):
        - All operations: O(n)
    """
    
    def __init__(self) -> None:
        self.root: Optional[TreeNode[T]] = None
        self._size: int = 0
    
    def __len__(self) -> int:
        return self._size
    
    def is_empty(self) -> bool:
        return self.root is None
    
    def insert(self, value: T) -> None:
        """Insert value maintaining BST property. O(h)"""
        self.root = self._insert_recursive(self.root, value)
        self._size += 1
    
    def _insert_recursive(
        self,
        node: Optional[TreeNode[T]],
        value: T
    ) -> TreeNode[T]:
        """Recursive helper for insert."""
        if node is None:
            return TreeNode(value)
        
        if value < node.value:
            node.left = self._insert_recursive(node.left, value)
        elif value > node.value:
            node.right = self._insert_recursive(node.right, value)
        # If equal, we don't insert (no duplicates)
        
        return node
    
    def search(self, value: T) -> bool:
        """Search for value in BST. O(h)"""
        return self._search_recursive(self.root, value)
    
    def _search_recursive(
        self,
        node: Optional[TreeNode[T]],
        value: T
    ) -> bool:
        """Recursive helper for search."""
        if node is None:
            return False
        
        if value == node.value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def search_iterative(self, value: T) -> bool:
        """Iterative search - often preferred."""
        current = self.root
        
        while current is not None:
            if value == current.value:
                return True
            elif value < current.value:
                current = current.left
            else:
                current = current.right
        
        return False
    
    def find_min(self) -> Optional[T]:
        """Find minimum value. O(h)"""
        if self.root is None:
            return None
        
        current = self.root
        while current.left is not None:
            current = current.left
        return current.value
    
    def find_max(self) -> Optional[T]:
        """Find maximum value. O(h)"""
        if self.root is None:
            return None
        
        current = self.root
        while current.right is not None:
            current = current.right
        return current.value
    
    def inorder(self) -> list[T]:
        """Return sorted list of all values."""
        return inorder_recursive(self.root)
```

---

## 4. Operaciones en BST {#4-operaciones}

### 4.1 Delete (La Más Compleja)

```python
def delete(self, value: T) -> None:
    """Delete value from BST. O(h)
    
    Three cases:
    1. Leaf node: just remove
    2. One child: replace with child
    3. Two children: replace with inorder successor
    """
    self.root = self._delete_recursive(self.root, value)

def _delete_recursive(
    self,
    node: Optional[TreeNode[T]],
    value: T
) -> Optional[TreeNode[T]]:
    """Recursive helper for delete."""
    if node is None:
        return None
    
    if value < node.value:
        node.left = self._delete_recursive(node.left, value)
    elif value > node.value:
        node.right = self._delete_recursive(node.right, value)
    else:
        # Found node to delete
        
        # Case 1: Leaf node
        if node.left is None and node.right is None:
            self._size -= 1
            return None
        
        # Case 2: One child
        if node.left is None:
            self._size -= 1
            return node.right
        if node.right is None:
            self._size -= 1
            return node.left
        
        # Case 3: Two children
        # Find inorder successor (smallest in right subtree)
        successor = self._find_min_node(node.right)
        node.value = successor.value
        node.right = self._delete_recursive(node.right, successor.value)
    
    return node

def _find_min_node(self, node: TreeNode[T]) -> TreeNode[T]:
    """Find node with minimum value in subtree."""
    current = node
    while current.left is not None:
        current = current.left
    return current
```

### 4.2 Validar si es BST

```python
def is_valid_bst(root: Optional[TreeNode[int]]) -> bool:
    """Check if tree is valid BST.
    
    Uses inorder traversal: should be sorted.
    """
    def inorder(node: Optional[TreeNode[int]]) -> list[int]:
        if node is None:
            return []
        return inorder(node.left) + [node.value] + inorder(node.right)
    
    values = inorder(root)
    
    # Check if sorted
    for i in range(len(values) - 1):
        if values[i] >= values[i + 1]:
            return False
    return True


def is_valid_bst_efficient(
    root: Optional[TreeNode[int]],
    min_val: float = float('-inf'),
    max_val: float = float('inf')
) -> bool:
    """Check BST validity with range checking. O(n) time, O(h) space."""
    if root is None:
        return True
    
    if root.value <= min_val or root.value >= max_val:
        return False
    
    return (
        is_valid_bst_efficient(root.left, min_val, root.value) and
        is_valid_bst_efficient(root.right, root.value, max_val)
    )
```

---

## 5. Análisis de Complejidad {#5-analisis}

### 5.1 Complejidades

| Operación | Balanced BST | Skewed BST |
|-----------|--------------|------------|
| Search | O(log n) | O(n) |
| Insert | O(log n) | O(n) |
| Delete | O(log n) | O(n) |
| Traversal | O(n) | O(n) |
| Min/Max | O(log n) | O(n) |

### 5.2 Por Qué se Desbalancea

```
Insertar 1, 2, 3, 4, 5 en orden:

    1
     \
      2
       \
        3
         \
          4
           \
            5

→ Se convierte en linked list → O(n) para todo
→ Solución: Árboles balanceados (AVL, Red-Black)
```

---

## ⚠️ Errores Comunes

### Error 1: Confundir traversals

```python
# Memorizar: "Inorder = In order" (para BST)
# Inorder de BST siempre da elementos ORDENADOS
```

### Error 2: No manejar caso vacío

```python
# ❌
def find_min(root):
    while root.left:  # AttributeError si root es None
        root = root.left

# ✅
def find_min(root):
    if root is None:
        return None
    while root.left:
        root = root.left
    return root.value
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 14.1: Implementar BST con insert y search
### Ejercicio 14.2: Implementar los 3 traversals
### Ejercicio 14.3: Validar si árbol es BST
### Ejercicio 14.4: Encontrar altura del árbol

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Visualgo BST](https://visualgo.net/en/bst) | Visual | 🔴 Obligatorio |
| [Abdul Bari Trees](https://www.youtube.com/watch?v=qH6yxkw0u78) | Video | 🔴 Obligatorio |

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [13_LINKED_LISTS](13_LINKED_LISTS_STACKS_QUEUES.md) | [00_INDICE](00_INDICE.md) | [15_GRAPHS](15_GRAPHS.md) |
