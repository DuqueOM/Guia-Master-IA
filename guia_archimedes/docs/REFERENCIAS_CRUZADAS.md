# 🔗 Mapa de Referencias Cruzadas

> Navegación completa entre todos los documentos de la guía.

---

## 📊 Matriz de Dependencias de Módulos

```
ORDEN ÓPTIMO DE ESTUDIO (Flujo de Dependencias):

01 Python Profesional
 ↓
02 OOP desde Cero
 ↓
03 Lógica Discreta ──────────────────────┐
 ↓                                        │
04 Arrays y Strings                       │ Fundamentos
 ↓                                        │ de Big O
05 Hash Maps y Sets ←─────────────────────┘
 ↓
06 Índice Invertido
 ↓
┌──────────────────────────────────────────┐
│ BLOQUE DSA AVANZADO (Pathway Critical)   │
├──────────────────────────────────────────┤
│ 13 Linked Lists, Stacks, Queues          │
│  ↓                                       │
│ 14 Trees y BST                           │
│  ↓                                       │
│ 15 Graphs, BFS, DFS                      │
└──────────────────────────────────────────┘
 ↓
┌──────────────────────────────────────────┐
│ BLOQUE ALGORITMOS (Pathway Critical)     │
├──────────────────────────────────────────┤
│ 07 Recursión ─────────────┐              │
│  ↓                        ↓              │
│ 08 Sorting           16 Dynamic Prog     │
│  ↓                        ↓              │
│ 09 Binary Search     17 Greedy           │
│                           ↓              │
│                      18 Heaps            │
└──────────────────────────────────────────┘
 ↓
10 Álgebra Lineal
 ↓
11 TF-IDF y Coseno
 ↓
12 PROYECTO INTEGRADOR
```

---

## 📖 Referencias por Módulo

### 01_PYTHON_PROFESIONAL.md
| Referencia a | Tipo |
|--------------|------|
| [GLOSARIO.md#type-hint](GLOSARIO.md#type-hint) | Término |
| [GLOSARIO.md#pep8](GLOSARIO.md#pep8) | Término |
| [EJERCICIOS.md#módulo-01](EJERCICIOS.md#módulo-01-python-profesional) | Ejercicios |

### 02_OOP_DESDE_CERO.md
| Referencia a | Tipo |
|--------------|------|
| 01_PYTHON_PROFESIONAL.md | Prerrequisito |
| [GLOSARIO.md#clase](GLOSARIO.md#clase) | Término |
| [GLOSARIO.md#oop](GLOSARIO.md#oop-object-oriented-programming) | Término |
| [GLOSARIO.md#solid](GLOSARIO.md#solid) | Término |

### 03_LOGICA_DISCRETA.md
| Referencia a | Tipo |
|--------------|------|
| [GLOSARIO.md#big-o-notation](GLOSARIO.md#big-o-notation) | Término |
| [GLOSARIO.md#set](GLOSARIO.md#set) | Término |

### 04_ARRAYS_STRINGS.md
| Referencia a | Tipo |
|--------------|------|
| [GLOSARIO.md#array](GLOSARIO.md#array) | Término |
| 03_LOGICA_DISCRETA.md (Big O) | Prerrequisito |

### 05_HASHMAPS_SETS.md
| Referencia a | Tipo |
|--------------|------|
| [GLOSARIO.md#hash-map](GLOSARIO.md#hash-map--hash-table) | Término |
| [GLOSARIO.md#colision](GLOSARIO.md#colisión-hash) | Término |

### 06_INVERTED_INDEX.md
| Referencia a | Tipo |
|--------------|------|
| 05_HASHMAPS_SETS.md | Prerrequisito |
| [GLOSARIO.md#indice-invertido](GLOSARIO.md#índice-invertido) | Término |
| 12_PROYECTO_INTEGRADOR.md | Uso en proyecto |

### 07_RECURSION.md
| Referencia a | Tipo |
|--------------|------|
| [GLOSARIO.md#recursion](GLOSARIO.md#recursión) | Término |
| [GLOSARIO.md#caso-base](GLOSARIO.md#caso-base) | Término |
| [GLOSARIO.md#memoization](GLOSARIO.md#memoization) | Término |
| 13_LINKED_LISTS.md (call stack) | Concepto relacionado |

### 08_SORTING.md
| Referencia a | Tipo |
|--------------|------|
| 07_RECURSION.md | Prerrequisito |
| [GLOSARIO.md#quicksort](GLOSARIO.md#quicksort) | Término |
| [GLOSARIO.md#divide-conquer](GLOSARIO.md#divide--conquer) | Término |
| 12_PROYECTO_INTEGRADOR.md | Uso en proyecto |

### 09_BINARY_SEARCH.md
| Referencia a | Tipo |
|--------------|------|
| 08_SORTING.md | Prerrequisito (datos ordenados) |
| [GLOSARIO.md#binary-search](GLOSARIO.md#binary-search) | Término |
| [GLOSARIO.md#off-by-one](GLOSARIO.md#off-by-one-error) | Término |

### 10_ALGEBRA_LINEAL.md
| Referencia a | Tipo |
|--------------|------|
| [GLOSARIO.md#vector](GLOSARIO.md#vector) | Término |
| [GLOSARIO.md#producto-punto](GLOSARIO.md#producto-punto-dot-product) | Término |
| [GLOSARIO.md#norma](GLOSARIO.md#norma-vector) | Término |
| 11_TFIDF_COSENO.md | Siguiente |

### 11_TFIDF_COSENO.md
| Referencia a | Tipo |
|--------------|------|
| 10_ALGEBRA_LINEAL.md | Prerrequisito |
| [GLOSARIO.md#tf](GLOSARIO.md#term-frequency-tf) | Término |
| [GLOSARIO.md#idf](GLOSARIO.md#idf-inverse-document-frequency) | Término |
| [GLOSARIO.md#cosine-similarity](GLOSARIO.md#cosine-similarity) | Término |
| 12_PROYECTO_INTEGRADOR.md | Uso en proyecto |

### 12_PROYECTO_INTEGRADOR.md
| Referencia a | Tipo |
|--------------|------|
| Todos los módulos 01-11 | Prerrequisitos |
| CHECKLIST.md | Verificación |
| RUBRICA_EVALUACION.md | Evaluación |

### 13_LINKED_LISTS_STACKS_QUEUES.md
| Referencia a | Tipo |
|--------------|------|
| [GLOSARIO.md#linked-list](GLOSARIO.md#linked-list) | Término |
| [GLOSARIO.md#stack](GLOSARIO.md#stack) | Término |
| [GLOSARIO.md#queue](GLOSARIO.md#queue) | Término |
| [GLOSARIO.md#lifo](GLOSARIO.md#lifo-last-in-first-out) | Término |
| [GLOSARIO.md#fifo](GLOSARIO.md#fifo-first-in-first-out) | Término |
| 14_TREES.md | Siguiente |
| 15_GRAPHS.md | Siguiente (Queue para BFS) |

### 14_TREES.md
| Referencia a | Tipo |
|--------------|------|
| 13_LINKED_LISTS.md | Prerrequisito (nodos, punteros) |
| [GLOSARIO.md#tree](GLOSARIO.md#tree-árbol) | Término |
| [GLOSARIO.md#bst](GLOSARIO.md#binary-search-tree-bst) | Término |
| [GLOSARIO.md#inorder](GLOSARIO.md#inorder-traversal) | Término |
| 15_GRAPHS.md | Siguiente |

### 15_GRAPHS.md
| Referencia a | Tipo |
|--------------|------|
| 13_LINKED_LISTS.md (Queue, Stack) | Prerrequisito |
| 14_TREES.md (conceptos de nodos) | Prerrequisito |
| [GLOSARIO.md#graph](GLOSARIO.md#graph-grafo) | Término |
| [GLOSARIO.md#bfs](GLOSARIO.md#bfs-breadth-first-search) | Término |
| [GLOSARIO.md#dfs](GLOSARIO.md#dfs-depth-first-search) | Término |

### 16_DYNAMIC_PROGRAMMING.md
| Referencia a | Tipo |
|--------------|------|
| 07_RECURSION.md | Prerrequisito |
| [GLOSARIO.md#dynamic-programming](GLOSARIO.md#dynamic-programming-dp) | Término |
| [GLOSARIO.md#memoization](GLOSARIO.md#memoization) | Término |
| [GLOSARIO.md#tabulation](GLOSARIO.md#tabulation) | Término |
| [GLOSARIO.md#optimal-substructure](GLOSARIO.md#optimal-substructure) | Término |

### 17_GREEDY.md
| Referencia a | Tipo |
|--------------|------|
| 16_DYNAMIC_PROGRAMMING.md (comparación) | Relacionado |
| [GLOSARIO.md#greedy](GLOSARIO.md#greedy-algorithm) | Término |
| 18_HEAPS.md (Huffman usa heap) | Siguiente |

### 18_HEAPS.md
| Referencia a | Tipo |
|--------------|------|
| 14_TREES.md (árbol binario completo) | Prerrequisito |
| [GLOSARIO.md#heap](GLOSARIO.md#heap) | Término |
| [GLOSARIO.md#priority-queue](GLOSARIO.md#priority-queue) | Término |

---

## 📚 Referencias en Documentos Auxiliares

### EJERCICIOS.md
- Enlaza a cada módulo para los ejercicios correspondientes
- Enlaza a EJERCICIOS_SOLUCIONES.md para respuestas

### EJERCICIOS_SOLUCIONES.md
- Enlaza de vuelta a EJERCICIOS.md
- Referencias a módulos para contexto

### GLOSARIO.md
- Referenciado desde todos los módulos
- Organizado A-Z con siglas al final

### SIMULACRO_ENTREVISTA.md
- Referencias a módulos por sección temática
- Enlaza a RECURSOS.md para más práctica

### RECURSOS.md
- Organizado por tema (Matemáticas, DSA, Python)
- URLs a cursos del Pathway

### CHECKLIST.md
- Referencias a todos los componentes del proyecto
- Enlaza a RUBRICA_EVALUACION.md

### RUBRICA_EVALUACION.md
- Criterios que mapean a módulos específicos

---

## ✅ Verificación de Enlaces

### Comandos para Verificar

```bash
# Verificar enlaces internos rotos
grep -r "\[.*\](.*\.md)" guia_archimedes/*.md | \
  grep -v "http" | \
  while read line; do
    file=$(echo "$line" | cut -d: -f1)
    link=$(echo "$line" | grep -oP '\(.*?\.md\)' | tr -d '()')
    if [[ ! -z "$link" && ! -f "guia_archimedes/$link" ]]; then
      echo "BROKEN: $file -> $link"
    fi
  done
```

---

## 🗺️ Flujo de Navegación Recomendado

### Para Principiante (Ruta Completa)
```
index.md → 00_INDICE.md → 01 → 02 → 03 → 04 → 05 → 06 → 
13 → 14 → 15 → 07 → 08 → 09 → 16 → 17 → 18 → 10 → 11 → 12
```

### Para Pathway (Solo DSA)
```
00_INDICE.md → 04 → 05 → 13 → 14 → 15 → 07 → 08 → 09 → 16 → 17 → 18 → 
SIMULACRO_ENTREVISTA.md
```

### Para Referencia Rápida
```
GLOSARIO.md (términos) → Módulo específico → EJERCICIOS.md
```
