# 13 - Linked Lists, Stacks y Queues

> **🎯 Objetivo:** Dominar estructuras de datos lineales fundamentales que son base para Trees y Graphs.

---

## 🧠 Analogía: El Tren, la Pila de Platos y la Fila del Banco

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   LINKED LIST = UN TREN                                                     │
│   ─────────────────────                                                     │
│   Cada vagón (nodo) tiene:                                                  │
│   • Pasajeros (datos)                                                       │
│   • Enganche al siguiente vagón (pointer)                                   │
│                                                                             │
│   [HEAD] → [A|→] → [B|→] → [C|→] → [D|∅]                                    │
│                                                                             │
│   STACK = PILA DE PLATOS                                                    │
│   ──────────────────────                                                    │
│   LIFO: Last In, First Out                                                  │
│   Solo puedes sacar el plato de arriba                                      │
│                                                                             │
│     ┌───┐                                                                   │
│     │ C │ ← top (último en entrar, primero en salir)                        │
│     ├───┤                                                                   │
│     │ B │                                                                   │
│     ├───┤                                                                   │
│     │ A │ ← bottom                                                          │
│     └───┘                                                                   │
│                                                                             │
│   QUEUE = FILA DEL BANCO                                                    │
│   ──────────────────────                                                    │
│   FIFO: First In, First Out                                                 │
│   El primero en llegar es el primero en ser atendido                        │
│                                                                             │
│   [A] → [B] → [C] → [D]                                                     │
│    ↑                  ↑                                                     │
│   front             rear                                                    │
│   (sale)           (entra)                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Contenido

1. [Linked Lists](#1-linked-lists)
2. [Stacks](#2-stacks)
3. [Queues](#3-queues)
4. [Comparación y Cuándo Usar](#4-comparacion)

---

## 1. Linked Lists {#1-linked-lists}

### 1.1 Node y Linked List

```python
from typing import Generic, TypeVar, Optional

T = TypeVar('T')


class Node(Generic[T]):
    """A node in a linked list.
    
    Attributes:
        data: The value stored in this node.
        next: Reference to the next node (or None).
    """
    
    def __init__(self, data: T) -> None:
        self.data: T = data
        self.next: Optional[Node[T]] = None
    
    def __repr__(self) -> str:
        return f"Node({self.data})"


class LinkedList(Generic[T]):
    """Singly linked list implementation.
    
    Time Complexities:
        - append: O(n) without tail, O(1) with tail
        - prepend: O(1)
        - search: O(n)
        - delete: O(n)
        - access by index: O(n)
    """
    
    def __init__(self) -> None:
        self.head: Optional[Node[T]] = None
        self._size: int = 0
    
    def is_empty(self) -> bool:
        """Check if list is empty. O(1)"""
        return self.head is None
    
    def __len__(self) -> int:
        """Return number of elements. O(1)"""
        return self._size
    
    def prepend(self, data: T) -> None:
        """Add element at the beginning. O(1)
        
        Args:
            data: Value to add.
        """
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self._size += 1
    
    def append(self, data: T) -> None:
        """Add element at the end. O(n)
        
        Args:
            data: Value to add.
        """
        new_node = Node(data)
        
        if self.is_empty():
            self.head = new_node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = new_node
        
        self._size += 1
    
    def search(self, data: T) -> Optional[Node[T]]:
        """Find node containing data. O(n)
        
        Returns:
            Node if found, None otherwise.
        """
        current = self.head
        while current is not None:
            if current.data == data:
                return current
            current = current.next
        return None
    
    def delete(self, data: T) -> bool:
        """Delete first node with given data. O(n)
        
        Returns:
            True if deleted, False if not found.
        """
        if self.is_empty():
            return False
        
        # Special case: delete head
        if self.head.data == data:
            self.head = self.head.next
            self._size -= 1
            return True
        
        # Search for node before the one to delete
        current = self.head
        while current.next is not None:
            if current.next.data == data:
                current.next = current.next.next
                self._size -= 1
                return True
            current = current.next
        
        return False
    
    def to_list(self) -> list[T]:
        """Convert to Python list. O(n)"""
        result = []
        current = self.head
        while current is not None:
            result.append(current.data)
            current = current.next
        return result
    
    def __repr__(self) -> str:
        return f"LinkedList({self.to_list()})"
    
    def __iter__(self):
        """Allow iteration over list."""
        current = self.head
        while current is not None:
            yield current.data
            current = current.next
```

### 1.2 Doubly Linked List

```python
class DNode(Generic[T]):
    """Node for doubly linked list."""
    
    def __init__(self, data: T) -> None:
        self.data: T = data
        self.prev: Optional[DNode[T]] = None
        self.next: Optional[DNode[T]] = None


class DoublyLinkedList(Generic[T]):
    """Doubly linked list with head and tail pointers.
    
    Advantages over singly linked:
    - O(1) append (with tail pointer)
    - O(1) delete from end
    - Can traverse backwards
    """
    
    def __init__(self) -> None:
        self.head: Optional[DNode[T]] = None
        self.tail: Optional[DNode[T]] = None
        self._size: int = 0
    
    def append(self, data: T) -> None:
        """Add at end. O(1) with tail pointer."""
        new_node = DNode(data)
        
        if self.tail is None:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        
        self._size += 1
    
    def prepend(self, data: T) -> None:
        """Add at beginning. O(1)"""
        new_node = DNode(data)
        
        if self.head is None:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        
        self._size += 1
    
    def pop_last(self) -> Optional[T]:
        """Remove and return last element. O(1)"""
        if self.tail is None:
            return None
        
        data = self.tail.data
        
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        
        self._size -= 1
        return data
```

### 1.3 List vs Linked List

| Operación | Python list | Linked List |
|-----------|-------------|-------------|
| Access [i] | O(1) | O(n) |
| Append | O(1) amort | O(n) o O(1)* |
| Prepend | O(n) | O(1) |
| Insert middle | O(n) | O(n) search + O(1) insert |
| Delete first | O(n) | O(1) |
| Delete last | O(1) | O(n) o O(1)** |
| Search | O(n) | O(n) |

\* O(1) si guardamos tail pointer  
\*\* O(1) con doubly linked list

---

## 2. Stacks {#2-stacks}

### 2.1 Implementación con Lista

```python
class Stack(Generic[T]):
    """Stack (LIFO) implementation using list.
    
    All operations are O(1).
    
    Example:
        >>> s = Stack()
        >>> s.push(1)
        >>> s.push(2)
        >>> s.pop()
        2
        >>> s.peek()
        1
    """
    
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def is_empty(self) -> bool:
        """Check if stack is empty. O(1)"""
        return len(self._items) == 0
    
    def push(self, item: T) -> None:
        """Add item to top. O(1)"""
        self._items.append(item)
    
    def pop(self) -> T:
        """Remove and return top item. O(1)
        
        Raises:
            IndexError: If stack is empty.
        """
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop()
    
    def peek(self) -> T:
        """Return top item without removing. O(1)
        
        Raises:
            IndexError: If stack is empty.
        """
        if self.is_empty():
            raise IndexError("Peek at empty stack")
        return self._items[-1]
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __repr__(self) -> str:
        return f"Stack({self._items})"
```

### 2.2 Aplicaciones de Stack

```python
def is_balanced_parentheses(expression: str) -> bool:
    """Check if parentheses are balanced.
    
    Example:
        >>> is_balanced_parentheses("((()))")
        True
        >>> is_balanced_parentheses("(()")
        False
    """
    stack: Stack[str] = Stack()
    matching = {')': '(', ']': '[', '}': '{'}
    
    for char in expression:
        if char in '([{':
            stack.push(char)
        elif char in ')]}':
            if stack.is_empty():
                return False
            if stack.pop() != matching[char]:
                return False
    
    return stack.is_empty()


def reverse_string_with_stack(s: str) -> str:
    """Reverse string using stack.
    
    Demonstrates LIFO property.
    """
    stack: Stack[str] = Stack()
    
    for char in s:
        stack.push(char)
    
    result = []
    while not stack.is_empty():
        result.append(stack.pop())
    
    return ''.join(result)
```

### 2.3 Call Stack (Contexto de Recursión)

```
┌─────────────────────────────────────────────────────────────────┐
│  EL CALL STACK ES UN STACK                                      │
│                                                                 │
│  factorial(3):                                                  │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │ factorial(1)=1  │ ← top (se resuelve primero)               │
│  ├─────────────────┤                                            │
│  │ factorial(2)    │ waiting for factorial(1)                  │
│  ├─────────────────┤                                            │
│  │ factorial(3)    │ waiting for factorial(2)                  │
│  └─────────────────┘                                            │
│                                                                 │
│  Por eso recursión infinita causa "Stack Overflow"             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Queues {#3-queues}

### 3.1 Implementación con Deque

```python
from collections import deque


class Queue(Generic[T]):
    """Queue (FIFO) implementation using deque.
    
    Using deque for O(1) operations at both ends.
    Using list would make dequeue O(n).
    
    Example:
        >>> q = Queue()
        >>> q.enqueue(1)
        >>> q.enqueue(2)
        >>> q.dequeue()
        1
    """
    
    def __init__(self) -> None:
        self._items: deque[T] = deque()
    
    def is_empty(self) -> bool:
        """Check if queue is empty. O(1)"""
        return len(self._items) == 0
    
    def enqueue(self, item: T) -> None:
        """Add item to rear. O(1)"""
        self._items.append(item)
    
    def dequeue(self) -> T:
        """Remove and return front item. O(1)
        
        Raises:
            IndexError: If queue is empty.
        """
        if self.is_empty():
            raise IndexError("Dequeue from empty queue")
        return self._items.popleft()
    
    def front(self) -> T:
        """Return front item without removing. O(1)"""
        if self.is_empty():
            raise IndexError("Front of empty queue")
        return self._items[0]
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __repr__(self) -> str:
        return f"Queue({list(self._items)})"
```

### 3.2 Queue con Lista (Ineficiente)

```python
class QueueWithList(Generic[T]):
    """Queue using list - INEFFICIENT for demonstration.
    
    dequeue is O(n) because list.pop(0) shifts all elements.
    """
    
    def __init__(self) -> None:
        self._items: list[T] = []
    
    def enqueue(self, item: T) -> None:
        """O(1)"""
        self._items.append(item)
    
    def dequeue(self) -> T:
        """O(n) - BAD! All elements shift."""
        if not self._items:
            raise IndexError("Dequeue from empty queue")
        return self._items.pop(0)  # O(n)!
```

### 3.3 Aplicaciones de Queue

```python
def bfs_preview(graph: dict, start: str) -> list[str]:
    """BFS uses a queue - preview for Graphs module.
    
    Visit nodes level by level.
    """
    visited = []
    queue: Queue[str] = Queue()
    queue.enqueue(start)
    
    while not queue.is_empty():
        node = queue.dequeue()
        if node not in visited:
            visited.append(node)
            for neighbor in graph.get(node, []):
                queue.enqueue(neighbor)
    
    return visited
```

---

## 4. Comparación y Cuándo Usar {#4-comparacion}

### 4.1 Tabla Resumen

| Estructura | Orden | Operación Principal | Uso Típico |
|------------|-------|---------------------|------------|
| Stack | LIFO | push/pop | Undo, parsing, DFS |
| Queue | FIFO | enqueue/dequeue | BFS, scheduling |
| Linked List | Insertion order | insert/delete | Cuando muchas inserciones/deletes |

### 4.2 Cuándo Usar Cada Una

```
USA STACK cuando:
• Necesitas deshacer operaciones (undo)
• Parsear expresiones (paréntesis balanceados)
• Implementar DFS
• Convertir recursión a iteración

USA QUEUE cuando:
• Procesar en orden de llegada
• Implementar BFS
• Buffer de datos (producer-consumer)
• Scheduling de tareas

USA LINKED LIST cuando:
• Muchas inserciones/eliminaciones al inicio
• No necesitas acceso por índice
• Tamaño muy variable
• Implementar otras estructuras
```

---

## ⚠️ Errores Comunes

### Error 1: Usar list para Queue

```python
# ❌ O(n) para dequeue
queue = []
queue.append(item)      # O(1)
item = queue.pop(0)     # O(n)!

# ✅ O(1) con deque
from collections import deque
queue = deque()
queue.append(item)      # O(1)
item = queue.popleft()  # O(1)
```

### Error 2: No verificar vacío antes de pop/dequeue

```python
# ❌ Error si está vacía
def bad_pop(stack):
    return stack.pop()  # IndexError!

# ✅ Verificar primero
def good_pop(stack):
    if stack.is_empty():
        raise IndexError("Stack is empty")
    return stack.pop()
```

---

## 🔧 Ejercicios Prácticos

### Ejercicio 13.1: Implementar Stack
Implementar Stack con operaciones push, pop, peek, is_empty.

### Ejercicio 13.2: Paréntesis Balanceados
Usar stack para verificar `()[]{}` balanceados.

### Ejercicio 13.3: Implementar Queue
Implementar Queue con deque.

---

## 📚 Recursos Externos

| Recurso | Tipo | Prioridad |
|---------|------|-----------|
| [Visualgo Linked List](https://visualgo.net/en/list) | Visual | 🔴 Obligatorio |
| [Stack vs Queue](https://www.youtube.com/watch?v=wjI1WNcIntg) | Video | 🟡 Recomendado |

---

## 🔗 Referencias del Glosario

- [Linked List](GLOSARIO.md#linked-list)
- [Stack](GLOSARIO.md#stack)
- [Queue](GLOSARIO.md#queue)
- [LIFO](GLOSARIO.md#lifo)
- [FIFO](GLOSARIO.md#fifo)

---

## 🧭 Navegación

| ← Anterior | Índice | Siguiente → |
|------------|--------|-------------|
| [12_PROYECTO_INTEGRADOR](12_PROYECTO_INTEGRADOR.md) | [00_INDICE](00_INDICE.md) | [14_TREES](14_TREES.md) |
