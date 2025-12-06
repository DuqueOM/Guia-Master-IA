# 📋 Syllabus - Archimedes Indexer

> **Programa de Formación: De Python Básico a Candidato MS in AI**

---

## 🎯 Objetivos del Programa

Al completar este programa, el estudiante será capaz de:

1. **Diseñar** sistemas de software usando principios OOP y SOLID
2. **Implementar** estructuras de datos fundamentales (Hash Maps, Índices) desde cero
3. **Codificar** algoritmos clásicos (QuickSort, Binary Search) sin librerías
4. **Aplicar** álgebra lineal para ranking de documentos (TF-IDF, Similitud de Coseno)
5. **Analizar** la complejidad algorítmica usando notación Big O
6. **Defender** decisiones técnicas en inglés a nivel técnico

---

## 📊 Estructura del Programa

### Macro-Módulos

| # | Macro-Módulo | Duración | Mini-Proyecto Asociado |
|---|--------------|----------|----------------------|
| I | Fundamentos de Python Profesional | 4 semanas | Clases `Document` y `Corpus` |
| II | Estructuras de Datos Core | 6 semanas | `InvertedIndex` funcional |
| III | Algoritmos Clásicos | 4 semanas | `sorting.py` y `searching.py` |
| IV | Matemáticas Aplicadas | 4 semanas | `vectorizer.py` + `similarity.py` |
| V | Integración y Defensa | 4 semanas | Motor de búsqueda completo |

**Total: 22 semanas** (con margen para repaso = 6 meses)

---

## 📚 Mapeo Macro-Módulos → Módulos → Código

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ MACRO-MÓDULO I: FUNDAMENTOS                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Módulos: 01, 02, 03                                                         │
│ Código:  src/document.py, src/corpus.py                                     │
│ Tests:   tests/test_document.py                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MACRO-MÓDULO II: ESTRUCTURAS DE DATOS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Módulos: 04, 05, 06                                                         │
│ Código:  src/tokenizer.py, src/inverted_index.py                            │
│ Tests:   tests/test_tokenizer.py, tests/test_index.py                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MACRO-MÓDULO III: ALGORITMOS                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Módulos: 07, 08, 09                                                         │
│ Código:  src/sorting.py, src/searching.py                                   │
│ Tests:   tests/test_sorting.py, tests/test_searching.py                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MACRO-MÓDULO IV: MATEMÁTICAS APLICADAS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Módulos: 10, 11                                                             │
│ Código:  src/vectorizer.py, src/similarity.py                               │
│ Tests:   tests/test_vectorizer.py, tests/test_similarity.py                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ MACRO-MÓDULO V: INTEGRACIÓN                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Módulos: 12                                                                 │
│ Código:  src/search_engine.py                                               │
│ Docs:    docs/COMPLEXITY_ANALYSIS.md, README.md                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📖 Detalle por Módulo

### Módulo 01: Python Profesional

| Contenido | Entregable |
|-----------|------------|
| Type hints y anotaciones | Código tipado con `mypy` pasando |
| Funciones puras vs impuras | Funciones sin side effects |
| PEP8 y estilo consistente | Código que pasa `ruff` o `flake8` |
| Docstrings y documentación | Cada función documentada |

**Mini-proyecto:** Función `clean_text(text: str) -> str` tipada y documentada.

**Validación:** `mypy src/ && ruff check src/`

---

### Módulo 02: OOP desde Cero

| Contenido | Entregable |
|-----------|------------|
| Clases y objetos | Clase `Document` con atributos |
| `__init__`, `__repr__`, `__str__` | Métodos mágicos implementados |
| Encapsulamiento | Properties y validación |
| Composición vs Herencia | Clase `Corpus` que contiene `Document`s |
| Principios SOLID básicos | Single Responsibility aplicado |

**Mini-proyecto:** Clases `Document` y `Corpus` funcionales.

**Validación:** `python -m pytest tests/test_document.py -v`

---

### Módulo 03: Lógica y Matemáticas Discretas

| Contenido | Entregable |
|-----------|------------|
| Teoría de conjuntos | Uso correcto de `set` en Python |
| Lógica proposicional | Expresiones booleanas complejas |
| Notación Big O (introducción) | Explicar O(1), O(n), O(n²) |
| Demostraciones simples | Documentar "por qué funciona" |

**Mini-proyecto:** Lista de stop words como `set` con análisis de complejidad.

**Validación:** Documento explicando complejidad de operaciones `in` en `list` vs `set`.

---

### Módulo 04: Arrays, Strings y Memoria

| Contenido | Entregable |
|-----------|------------|
| Listas en Python (bajo nivel) | Entender slicing y copia |
| Manipulación de strings | Tokenización básica |
| Complejidad de operaciones | Tabla de O() para list |
| Inmutabilidad vs mutabilidad | Evitar bugs de referencia |

**Mini-proyecto:** Tokenizador básico que separa texto en palabras.

**Validación:** `python -m pytest tests/test_tokenizer.py -v`

---

### Módulo 05: Hash Maps y Sets

| Contenido | Entregable |
|-----------|------------|
| Cómo funciona un diccionario | Entender hashing |
| Colisiones y resolución | Saber que existen, no implementar |
| Complejidad O(1) amortizada | Explicar cuándo y por qué |
| Sets para búsqueda rápida | Stop words como `frozenset` |

**Mini-proyecto:** Diccionario de frecuencia de palabras.

**Validación:** Benchmark `list` vs `set` para búsqueda (script incluido).

---

### Módulo 06: Índice Invertido

| Contenido | Entregable |
|-----------|------------|
| Qué es un índice invertido | Diagrama y explicación |
| Estructura `{palabra: [doc_ids]}` | Clase `InvertedIndex` |
| Agregar documentos al índice | Método `add_document()` |
| Buscar documentos por palabra | Método `search(query)` |

**Mini-proyecto:** `InvertedIndex` que indexa y busca en corpus de prueba.

**Validación:** `python -m pytest tests/test_index.py -v`

**Análisis requerido:** ¿Cuál es la complejidad de `add_document()`? ¿Y de `search()`?

---

### Módulo 07: Recursión y Divide & Conquer

| Contenido | Entregable |
|-----------|------------|
| Pensamiento recursivo | Funciones recursivas simples |
| Caso base y caso recursivo | Identificar en ejemplos |
| Call stack y límites | Entender `RecursionError` |
| Divide & Conquer pattern | Factorial, Fibonacci, suma de lista |

**Mini-proyecto:** `factorial()`, `fibonacci()`, `sum_list()` recursivos.

**Validación:** Tests que verifican casos base y casos grandes.

---

### Módulo 08: Algoritmos de Ordenamiento

| Contenido | Entregable |
|-----------|------------|
| QuickSort desde cero | Implementación funcional |
| Pivot selection | Random pivot para evitar O(n²) |
| MergeSort (opcional) | Implementación alternativa |
| Análisis de complejidad | O(n log n) promedio, O(n²) peor |

**Mini-proyecto:** `quicksort()` y `mergesort()` en `sorting.py`.

**Validación:** `python -m pytest tests/test_sorting.py -v`

**Análisis requerido:** Documento explicando cuándo QuickSort es O(n²).

---

### Módulo 09: Búsqueda Binaria

| Contenido | Entregable |
|-----------|------------|
| Binary Search clásica | Implementación sin errores |
| Off-by-one errors | Cómo evitarlos sistemáticamente |
| Variantes | Buscar primer/último elemento |
| Cuándo aplicar | Lista ordenada, O(log n) |

**Mini-proyecto:** `binary_search()` con variantes en `searching.py`.

**Validación:** `python -m pytest tests/test_searching.py -v`

---

### Módulo 10: Álgebra Lineal sin NumPy

| Contenido | Entregable |
|-----------|------------|
| Vectores como listas | Representación básica |
| Suma de vectores | `add_vectors(v1, v2)` |
| Producto punto | `dot_product(v1, v2)` |
| Norma de un vector | `magnitude(v)` |
| Matrices como listas de listas | Representación 2D |

**Mini-proyecto:** Módulo `linear_algebra.py` con operaciones básicas.

**Validación:** Tests que verifican matemáticamente cada operación.

---

### Módulo 11: TF-IDF y Similitud de Coseno

| Contenido | Entregable |
|-----------|------------|
| Term Frequency (TF) | Función `compute_tf()` |
| Inverse Document Frequency (IDF) | Función `compute_idf()` |
| TF-IDF combinado | Función `compute_tfidf()` |
| Similitud de coseno | Función `cosine_similarity()` |
| Vectorización de documentos | Cada doc como vector TF-IDF |

**Mini-proyecto:** Sistema de ranking por relevancia.

**Validación:** Tests + comparación manual con resultados conocidos.

---

### Módulo 12: Proyecto Integrador

| Contenido | Entregable |
|-----------|------------|
| Ensamblaje de componentes | `SearchEngine` que usa todo |
| API de búsqueda | Método `search(query, top_k)` |
| Análisis Big O completo | `COMPLEXITY_ANALYSIS.md` |
| README profesional | Documentación de uso |
| Tests de integración | `test_engine.py` |

**Entregable final:**
1. Motor de búsqueda funcional
2. Análisis de complejidad de cada operación
3. README en inglés
4. Suite de tests con >80% coverage

**Validación:** Demo en vivo + defensa del análisis Big O.

---

## 📊 Rúbrica General (100 puntos)

| Dimensión | Puntos | Criterio |
|-----------|--------|----------|
| **Funcionalidad** | 30 | El motor busca y rankea correctamente |
| **Código limpio** | 20 | PEP8, type hints, docstrings |
| **Tests** | 20 | Cobertura >80%, casos edge |
| **Análisis Big O** | 20 | Documento completo y correcto |
| **Documentación** | 10 | README claro, en inglés |

### Niveles

| Puntuación | Nivel |
|------------|-------|
| 90-100 | Listo para Pathway + entrevistas técnicas |
| 75-89 | Buen nivel, reforzar áreas débiles |
| 60-74 | Necesita más práctica antes de Pathway |
| <60 | Revisar módulos fundamentales |

---

## 🎯 Preparación para Pathway

El curso de entrada típico del Pathway es **"Algorithms for Searching, Sorting, and Indexing"**.

Este programa cubre directamente:
- ✅ Sorting (QuickSort, MergeSort)
- ✅ Searching (Binary Search)
- ✅ Indexing (Inverted Index)
- ✅ Análisis de complejidad (Big O)
- ✅ Python profesional

### Alineación con el Pathway

| Tema del Pathway | Módulo de esta Guía |
|------------------|---------------------|
| Algorithm Analysis | 03, 08, 09, 12 |
| Sorting Algorithms | 08 |
| Binary Search | 09 |
| Hash Tables | 05, 06 |
| Basic Data Structures | 04, 05 |

---

## 📅 Cronograma Sugerido

Ver [PLAN_ESTUDIOS.md](PLAN_ESTUDIOS.md) para el cronograma día a día.

---

## ✅ Checklist de Finalización del Programa

- [ ] Todos los módulos completados
- [ ] Proyecto `archimedes-indexer` funcional
- [ ] Tests pasando con >80% coverage
- [ ] `COMPLEXITY_ANALYSIS.md` completo
- [ ] README en inglés
- [ ] Simulacro de entrevista completado
- [ ] Capaz de explicar el proyecto en inglés (5 min)

---

> 💡 **Recuerda:** El objetivo no es solo construir el motor, sino poder *defenderlo* técnicamente. Practica explicar cada decisión.
