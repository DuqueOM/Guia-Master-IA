# 📖 Glosario Técnico

> Definiciones A-Z de términos usados en la guía.

---

## A

### Adjacency List
**Definición:** Representación de grafo donde cada vértice tiene lista de vecinos.
**Espacio:** O(V + E)
**Uso:** Grafos sparse (pocos edges).

### Adjacency Matrix
**Definición:** Matriz donde M[i][j] = 1 si hay edge de i a j.
**Espacio:** O(V²)
**Uso:** Grafos dense, verificar edge en O(1).

### Algoritmo
**Definición:** Secuencia finita de pasos para resolver un problema.
**Analogía:** Una receta de cocina: ingredientes (input) → pasos → plato (output).

### Amortizado
**Definición:** Complejidad promedio sobre muchas operaciones.
**Ejemplo:** `list.append()` es O(1) amortizado aunque ocasionalmente sea O(n).

### Array
**Definición:** Estructura de datos con elementos en posiciones contiguas de memoria.
**En Python:** Las `list` son arrays dinámicos.

---

## B

### Big O Notation
**Definición:** Notación para describir el crecimiento del tiempo/espacio con el tamaño de entrada.
**Común:** O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2^n)

### BFS (Breadth-First Search)
**Definición:** Algoritmo de recorrido de grafos que explora por niveles.
**Estructura:** Usa Queue (FIFO).
**Uso:** Shortest path en grafos no ponderados.
**Complejidad:** O(V + E)

### Binary Search
**Definición:** Algoritmo que encuentra un elemento en lista ordenada dividiendo el espacio a la mitad.
**Complejidad:** O(log n)
**Requisito:** Lista debe estar ordenada.

### Binary Search Tree (BST)
**Definición:** Árbol binario donde left < root < right para cada nodo.
**Operaciones:** O(log n) promedio, O(n) peor caso.
**Uso:** Búsqueda, inserción y eliminación eficientes.

### Bottom-Up (DP)
**Definición:** Enfoque de DP que resuelve subproblemas desde los más pequeños.
**Sinónimo:** Tabulation.
**Ventaja:** No usa call stack, más eficiente en memoria.

---

## C

### Caso Base
**Definición:** Condición que termina la recursión sin más llamadas recursivas.
**Ejemplo:** En factorial, `if n <= 1: return 1`.

### Clase
**Definición:** Plantilla para crear objetos con atributos y métodos.
**Analogía:** El plano de una casa; los objetos son las casas construidas.

### Colisión (Hash)
**Definición:** Cuando dos claves diferentes producen el mismo hash.
**Resolución:** Python usa "open addressing" para encontrar otro slot.

### Complejidad Temporal
**Definición:** Cuánto tiempo toma un algoritmo en función del tamaño de entrada.

### Cycle (Grafo)
**Definición:** Camino que comienza y termina en el mismo vértice.
**Detección:** DFS puede detectar ciclos en O(V + E).

### Cosine Similarity
**Definición:** Medida de similitud entre vectores basada en el ángulo entre ellos.
**Fórmula:** cos(θ) = (A·B) / (||A|| × ||B||)
**Rango:** 0 (perpendiculares) a 1 (paralelos) para vectores TF-IDF.

---

## D

### DFS (Depth-First Search)
**Definición:** Algoritmo de recorrido que explora lo más profundo posible antes de retroceder.
**Estructura:** Usa Stack o recursión.
**Uso:** Detectar ciclos, encontrar caminos, topological sort.
**Complejidad:** O(V + E)

### Divide & Conquer
**Definición:** Estrategia de dividir problema en subproblemas, resolverlos y combinar.
**Ejemplos:** MergeSort, QuickSort, Binary Search.

### Document Frequency (DF)
**Definición:** Número de documentos que contienen un término.
**Uso:** Para calcular IDF.

### Docstring
**Definición:** String de documentación al inicio de función/clase/módulo.
**Formato:** Google style, NumPy style, o reStructuredText.

### Dynamic Programming (DP)
**Definición:** Técnica de optimización que guarda resultados de subproblemas.
**Requisitos:** Optimal substructure + overlapping subproblems.
**Enfoques:** Top-down (memoization) y Bottom-up (tabulation).

---

## F

### FIFO (First In, First Out)
**Definición:** Orden donde el primero en entrar es el primero en salir.
**Estructura:** Queue.
**Analogía:** Fila del supermercado.

---

## G

### Graph (Grafo)
**Definición:** Estructura de nodos (vértices) conectados por aristas (edges).
**Tipos:** Dirigido/no dirigido, ponderado/no ponderado.
**Representación:** Adjacency list o matrix.

### Greedy Algorithm
**Definición:** Estrategia que toma la mejor opción local en cada paso.
**Requisito:** Greedy choice property para garantizar óptimo.
**Ejemplos:** Activity selection, Huffman coding.

---

## H

### Heap
**Definición:** Árbol binario completo con propiedad de heap (parent <= children para min-heap).
**Operaciones:** Insert O(log n), extract-min O(log n), peek O(1).
**Uso:** Priority queues, heapsort, top-K problems.

### Hash Function
**Definición:** Función que convierte cualquier dato en un número (hash).
**Propiedades:** Determinista, rápida, distribución uniforme.

### Hash Map / Hash Table
**Definición:** Estructura que mapea claves a valores usando hashing.
**En Python:** `dict`.
**Complejidad:** O(1) promedio para get/set/delete.

---

## I

### IDF (Inverse Document Frequency)
**Definición:** Medida de qué tan raro es un término en el corpus.
**Fórmula:** IDF(t) = log(N / df(t)) donde N = total docs, df = doc frequency.
**Intuición:** Palabras raras tienen IDF alto.

### Índice Invertido
**Definición:** Estructura que mapea términos a documentos que los contienen.
**Estructura:** `{término: [lista de doc_ids]}`
**Uso:** Corazón de los motores de búsqueda.

### Inmutabilidad
**Definición:** Propiedad de objetos que no pueden modificarse después de crearse.
**En Python:** str, tuple, frozenset son inmutables.

### In-Place
**Definición:** Algoritmo que modifica la estructura original sin crear copia.
**Ejemplo:** QuickSort in-place usa O(log n) espacio extra.

---

## I

### Inorder Traversal
**Definición:** Recorrido de árbol: Left, Root, Right.
**Propiedad:** En BST, da elementos en orden ascendente.

---

## L

### Leaf Node
**Definición:** Nodo de árbol sin hijos.
**Identificación:** node.left == None and node.right == None

### LIFO (Last In, First Out)
**Definición:** Orden donde el último en entrar es el primero en salir.
**Estructura:** Stack.
**Analogía:** Pila de platos.

### Linked List
**Definición:** Estructura de nodos donde cada nodo apunta al siguiente.
**Tipos:** Singly (un puntero), Doubly (dos punteros).
**Ventaja:** O(1) insert/delete al inicio.

### Linter
**Definición:** Herramienta que analiza código para detectar errores y problemas de estilo.
**Ejemplos:** ruff, flake8, pylint.

### Logarítmico
**Definición:** Complejidad O(log n) - crece muy lentamente.
**Ejemplo:** Binary search en 1 billón de elementos = ~30 pasos.

---

## M

### Matriz
**Definición:** Array bidimensional de números.
**En Python puro:** Lista de listas: `[[1,2], [3,4]]`.

### Memoization
**Definición:** Técnica de cachear resultados de funciones para evitar recálculo.
**Uso:** Optimizar recursión (ej: Fibonacci).

### MergeSort
**Definición:** Algoritmo de ordenamiento divide & conquer.
**Complejidad:** O(n log n) siempre.
**Propiedad:** Estable.

---

## N

### Norma (Vector)
**Definición:** Longitud/magnitud de un vector.
**Fórmula:** ||v|| = √(v₁² + v₂² + ... + vₙ²)

---

## O

### Optimal Substructure
**Definición:** Propiedad donde solución óptima contiene soluciones óptimas de subproblemas.
**Requisito:** Necesario para aplicar DP o Greedy.

### Overlapping Subproblems
**Definición:** Cuando los mismos subproblemas se resuelven múltiples veces.
**Requisito:** Necesario para que DP sea beneficioso.

### Off-by-One Error
**Definición:** Error donde un índice está desplazado por 1.
**Común en:** Loops, binary search, slicing.

### OOP (Object-Oriented Programming)
**Definición:** Paradigma que organiza código en objetos con datos y comportamiento.
**Pilares:** Encapsulamiento, herencia, polimorfismo.

---

## P

### Postorder Traversal
**Definición:** Recorrido de árbol: Left, Right, Root.
**Uso:** Eliminar árbol (hijos antes que padre), evaluar expresiones.

### Preorder Traversal
**Definición:** Recorrido de árbol: Root, Left, Right.
**Uso:** Copiar/serializar árbol.

### Priority Queue
**Definición:** Cola donde elementos salen según prioridad, no orden de llegada.
**Implementación:** Típicamente con Heap.
**Operaciones:** Insert O(log n), extract O(log n).

### Partition
**Definición:** En QuickSort, reorganizar array para que elementos < pivot estén antes.
**Resultado:** Pivot queda en su posición final.

### PEP8
**Definición:** Guía de estilo oficial de Python.
**Puntos clave:** 4 espacios, 79-88 chars línea, snake_case.

### Producto Punto (Dot Product)
**Definición:** Suma de productos de componentes correspondientes.
**Fórmula:** a·b = a₁b₁ + a₂b₂ + ... + aₙbₙ

### Property
**Definición:** Mecanismo para controlar acceso a atributos con getters/setters.
**Uso:** Validación, cálculo dinámico, encapsulamiento.

---

## Q

### Queue
**Definición:** Estructura de datos FIFO (First In, First Out).
**Operaciones:** enqueue O(1), dequeue O(1).
**Uso:** BFS, scheduling, buffers.

### QuickSort
**Definición:** Algoritmo de ordenamiento basado en partición.
**Complejidad:** O(n log n) promedio, O(n²) peor caso.
**Ventaja:** In-place, cache-friendly.

---

## R

### Recursión
**Definición:** Técnica donde una función se llama a sí misma.
**Componentes:** Caso base + caso recursivo.

---

## S

### Stack
**Definición:** Estructura de datos LIFO (Last In, First Out).
**Operaciones:** push O(1), pop O(1), peek O(1).
**Uso:** Call stack, DFS, undo, parsing.

### Set
**Definición:** Colección de elementos únicos sin orden.
**Operaciones O(1):** add, remove, contains.

### SOLID
**Definición:** 5 principios de diseño orientado a objetos.
- **S**ingle Responsibility
- **O**pen/Closed
- **L**iskov Substitution
- **I**nterface Segregation
- **D**ependency Inversion

### Stable Sort
**Definición:** Ordenamiento que mantiene orden relativo de elementos iguales.
**Ejemplo:** MergeSort es estable, QuickSort no.

---

## T

### Tabulation
**Definición:** Enfoque de DP que llena tabla iterativamente desde casos base.
**Sinónimo:** Bottom-up DP.
**Ventaja:** No usa call stack.

### Top-Down (DP)
**Definición:** Enfoque de DP recursivo con memoization.
**Ventaja:** Solo calcula subproblemas necesarios.

### Tree (Árbol)
**Definición:** Estructura jerárquica de nodos sin ciclos.
**Términos:** Root, parent, child, leaf, height, depth.
**Tipos:** Binary tree, BST, AVL, etc.

### Tree Traversal
**Definición:** Visitar todos los nodos de un árbol.
**DFS:** Inorder, Preorder, Postorder.
**BFS:** Level-order.

### Term Frequency (TF)
**Definición:** Frecuencia de un término en un documento.
**Fórmula:** TF(t,d) = count(t,d) / total_terms(d)

### TF-IDF
**Definición:** Producto de Term Frequency × Inverse Document Frequency.
**Uso:** Medir importancia de término en documento dentro de corpus.

### Tokenización
**Definición:** Proceso de dividir texto en unidades (tokens).
**Ejemplo:** "Hello, World!" → ["hello", "world"]

### Type Hint
**Definición:** Anotación que indica el tipo esperado de variable/parámetro/retorno.
**Ejemplo:** `def greet(name: str) -> str:`

---

## V

### Vector
**Definición:** Lista ordenada de números que representa punto/dirección en espacio.
**En Python puro:** `list[float]`
**Uso en IR:** Representar documentos en espacio de términos.

### Vertex (Vértice)
**Definición:** Nodo en un grafo.
**Plural:** Vertices.
**Notación:** V = número de vértices.

### Vocabulario
**Definición:** Conjunto de todos los términos únicos en un corpus.
**Tamaño:** Determina dimensión de vectores TF-IDF.

---

## Siglas Comunes

| Sigla | Significado |
|-------|-------------|
| BST | Binary Search Tree |
| BFS | Breadth-First Search |
| DFS | Depth-First Search |
| DP | Dynamic Programming |
| FIFO | First In, First Out |
| LIFO | Last In, First Out |
| OOP | Object-Oriented Programming |
| TF | Term Frequency |
| IDF | Inverse Document Frequency |
