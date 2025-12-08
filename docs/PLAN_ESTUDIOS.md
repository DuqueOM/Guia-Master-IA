# 📅 Plan de Estudios - 10 Módulos Obligatorios

> **6 Meses | 6 horas/día | Lunes a Sábado** - Preparación para MS in AI Pathway

---

## 🗓️ Vista General: 26 Semanas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SEMANAS 1-6       │ SEMANAS 7-14      │ SEMANAS 15-22    │ SEMANAS 23-26    │
│ FUNDAMENTOS       │ PROB/STAT ⭐      │ MACHINE L. ⭐    │ PROYECTO        │
│ Módulos 01-03     │ Módulos 04-06     │ Módulos 07-09    │ Módulo 10        │
│ Python + Álgebra  │ PATHWAY LÍNEA 2   │ PATHWAY LÍNEA 1  │ INTEGRACIÓN      │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Dedicación total:** 36 horas/semana × 26 semanas = **~936 horas**

### Los 10 Módulos Obligatorios

| Semanas | Módulo | Tema | Curso del Pathway |
|---------|--------|------|-------------------|
| 1-2 | 01 | Python Profesional | - |
| 3-4 | 02 | OOP desde Cero | - |
| 5-6 | 03 | Álgebra Lineal para ML | - |
| 7-9 | 04 | Fundamentos de Probabilidad | Probability Fundamentals |
| 10-12 | 05 | Estadística Inferencial | Statistical Estimation |
| 13-14 | 06 | Markov y Monte Carlo | Markov Chains & MC |
| 15-17 | 07 | ML Supervisado | Supervised Learning |
| 18-19 | 08 | ML No Supervisado | Unsupervised Algorithms |
| 20-22 | 09 | Deep Learning | Intro to Deep Learning |
| 23-26 | 10 | Proyecto Final | - |

---

> **Nota:** Las secciones de semanas detalladas que siguen corresponden al plan original basado en motor de búsqueda y DSA.
> Para el **Pathway**, los únicos módulos obligatorios son los 10 de la tabla anterior.
> Los contenidos de Arrays, Hash Maps, Recursión, Sorting, Trees, Graphs y DP deben considerarse **Anexos DSA opcionales**.

---

## 📌 Distribución Diaria Típica

| Bloque | Horario | Actividad | Duración |
|--------|---------|-----------|----------|
| 🌅 Mañana | 08:00 - 10:30 | Estudio teórico (lectura del módulo) | 2.5 h |
| ☕ Pausa | 10:30 - 11:00 | Descanso | 30 min |
| 🌇 Mediodía | 11:00 - 13:30 | Implementación (código) | 2.5 h |
| 🌙 Tarde | 15:00 - 16:00 | Ejercicios + repaso | 1 h |

---

## 🗓️ SEMANA 1: Python Profesional (Parte 1)

**Módulo:** 01 - Python Profesional
**Objetivo:** Escribir código Python con estándares profesionales

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Type hints básicos | Tipar funciones existentes | Ejercicio 1.1 |
| M | Type hints avanzados | Tipar clases simples | Ejercicio 1.2 |
| X | Funciones puras | Refactorizar a puras | Ejercicio 1.3 |
| J | PEP8 y linters | Configurar `ruff` | Corregir warnings |
| V | Docstrings | Documentar módulo | Revisar con `pydoc` |
| S | **Repaso semanal** | Mini-proyecto: `clean_text()` | Autoevaluación |

**Entregable:** Función `clean_text()` tipada, documentada, pasando linters.

**Recursos:**
- [Real Python: Type Hints](https://realpython.com/python-type-checking/)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)

---

## 🗓️ SEMANA 2: Python Profesional (Parte 2) + OOP Inicio

**Módulo:** 01 (cierre) + 02 (inicio)
**Objetivo:** Dominar type hints complejos, iniciar OOP

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Generics, Optional, Union | Tipar estructuras complejas | Ejercicio 1.4 |
| M | `mypy` en profundidad | Corregir errores de mypy | Config `pyproject.toml` |
| X | Clases: `__init__` | Clase `Document` básica | Ejercicio 2.1 |
| J | `__repr__`, `__str__` | Métodos mágicos en Document | Ejercicio 2.2 |
| V | Properties | Validación en properties | Ejercicio 2.3 |
| S | **Repaso** | Clase `Document` completa | Test manual |

**Entregable:** Clase `Document` con type hints y métodos mágicos.

---

## 🗓️ SEMANA 3: OOP Avanzado

**Módulo:** 02 - OOP desde Cero
**Objetivo:** Composición, herencia básica, SOLID

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Composición vs Herencia | Clase `Corpus` con lista de docs | Ejercicio 2.4 |
| M | Single Responsibility | Refactorizar clases grandes | Ejercicio 2.5 |
| X | Open/Closed (básico) | Extensibilidad sin modificar | Diagrama de clases |
| J | Dataclasses | Migrar `Document` a dataclass | Comparar código |
| V | Testing de clases | `test_document.py` | pytest básico |
| S | **Repaso** | `Corpus` + tests | Simulacro módulo |

**Entregable:** `Document`, `Corpus` con tests pasando.

---

## 🗓️ SEMANA 4: Lógica y Matemáticas Discretas

**Módulo:** 03 - Lógica y Matemáticas Discretas
**Objetivo:** Fundamentos de lógica y notación Big O básica

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Teoría de conjuntos | `set` vs `list` en Python | Ejercicio 3.1 |
| M | Operaciones de conjuntos | Unión, intersección, diferencia | Ejercicio 3.2 |
| X | Lógica proposicional | Expresiones booleanas complejas | Ejercicio 3.3 |
| J | Intro a Big O | O(1), O(n), O(n²) | Analizar loops |
| V | Big O de estructuras | Tabla de complejidades | Documento análisis |
| S | **Checkpoint Fase I** | Simulacro Fundamentos | Autoevaluación |

**Entregable:** Lista de stop words como `set` + análisis de complejidad.

**Checkpoint:** [SIMULACRO_FUNDAMENTOS.md](SIMULACRO_FUNDAMENTOS.md)

---

## 🗓️ SEMANA 5-6: Arrays, Strings y Tokenización

**Módulo:** 04 - Arrays, Strings y Memoria
**Objetivo:** Manipulación eficiente de secuencias, tokenizador básico

### Semana 5

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Listas en Python (internals) | Slicing, copia profunda | Ejercicio 4.1 |
| M | Complejidad de list | append, insert, pop | Tabla de O() |
| X | Strings: inmutabilidad | Manipulación eficiente | Ejercicio 4.2 |
| J | Tokenización básica | `split()`, `lower()`, `strip()` | Tokenizador v1 |
| V | Eliminar puntuación | Regex básico o manual | Tokenizador v2 |
| S | **Repaso** | Tests del tokenizador | Benchmark |

### Semana 6

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Stop words | Filtrar palabras comunes | Tokenizador v3 |
| M | Stemming (concepto) | Stemming básico manual | Opcional |
| X | Normalización | Acentos, mayúsculas | Tokenizador final |
| J | Testing exhaustivo | Casos edge (vacío, solo símbolos) | test_tokenizer.py |
| V | Documentación | Docstrings completos | README del módulo |
| S | **Repaso** | Tokenizador completo | Benchmark final |

**Entregable:** `tokenizer.py` con tests y documentación.

---

## 🗓️ SEMANA 7-8: Hash Maps y Sets

**Módulo:** 05 - Hash Maps y Sets
**Objetivo:** Entender y usar eficientemente diccionarios y sets

### Semana 7

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Cómo funciona un hash | Concepto de hashing | Ejercicio 5.1 |
| M | Diccionarios Python | get, setdefault, defaultdict | Ejercicio 5.2 |
| X | Colisiones (concepto) | No implementar, solo entender | Lectura |
| J | O(1) amortizado | Cuándo y por qué | Documento |
| V | Sets: operaciones | in, add, remove, intersection | Ejercicio 5.3 |
| S | **Repaso** | Frecuencia de palabras v1 | Test manual |

### Semana 8

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | frozenset | Cuándo usar inmutable | Stop words optimizado |
| M | Counter de collections | Alternativa a dict manual | Comparar |
| X | Benchmark list vs set | Script de medición | Gráfica de tiempos |
| J | Aplicación: word count | Contador de palabras completo | test_word_count.py |
| V | Documentación | Análisis de complejidad | Documento |
| S | **Repaso** | Módulo hashmaps completo | Autoevaluación |

**Entregable:** Contador de frecuencias + benchmark + análisis.

---

## 🗓️ SEMANA 9-11: Índice Invertido

**Módulo:** 06 - Índice Invertido
**Objetivo:** Construir el núcleo del motor de búsqueda

### Semana 9

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Qué es un índice invertido | Diagrama palabra→docs | Ejercicio 6.1 |
| M | Estructura de datos | `{word: [doc_id, ...]}` | Clase `InvertedIndex` |
| X | Método `add_document()` | Tokenizar + indexar | Implementación |
| J | Método `search(word)` | Buscar palabra simple | Implementación |
| V | Testing básico | Casos simples | test_index.py v1 |
| S | **Repaso** | Índice funcional básico | Demo |

### Semana 10

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Búsqueda multi-palabra | AND lógico (intersección) | Implementación |
| M | OR lógico | Unión de resultados | Implementación |
| X | Frecuencia en índice | `{word: [(doc_id, freq), ...]}` | Upgrade estructura |
| J | Posiciones (opcional) | Índice posicional | Lectura |
| V | Testing avanzado | Casos edge | test_index.py v2 |
| S | **Repaso** | Índice con AND/OR | Demo |

### Semana 11

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Análisis de complejidad | O() de add, search | Documento |
| M | Persistencia (opcional) | Guardar/cargar índice | JSON simple |
| X | Corpus de prueba | Crear 10-20 docs de test | data/sample_corpus/ |
| J | Demo completa | Indexar corpus, buscar | Script demo |
| V | Documentación | README del módulo | Docstrings |
| S | **Checkpoint Fase II** | Simulacro Estructuras | Autoevaluación |

**Entregable:** `InvertedIndex` completo con análisis de complejidad.

**Checkpoint:** [SIMULACRO_ESTRUCTURAS.md](SIMULACRO_ESTRUCTURAS.md)

---

## 🗓️ SEMANA 12-13: Recursión

**Módulo:** 07 - Recursión y Divide & Conquer
**Objetivo:** Dominar el pensamiento recursivo

### Semana 12

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Qué es recursión | Analogía de espejos | Ejercicio 7.1 |
| M | Caso base y recursivo | Identificar en ejemplos | Factorial |
| X | Call stack | Visualizar con prints | Fibonacci |
| J | RecursionError | Límites y cómo evitarlo | sys.setrecursionlimit |
| V | Suma de lista recursiva | `sum_list()` | Ejercicio 7.2 |
| S | **Repaso** | Funciones recursivas básicas | Test |

### Semana 13

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Divide & Conquer | Patrón general | Diagrama |
| M | Merge de listas | Fusionar ordenadas | Implementación |
| X | Búsqueda recursiva | Buscar en lista | Ejercicio 7.3 |
| J | Optimización (memoization) | Concepto básico | Fibonacci optimizado |
| V | Testing recursivo | Casos base y grandes | test_recursion.py |
| S | **Repaso** | Módulo recursión completo | Autoevaluación |

**Entregable:** Funciones recursivas con tests.

---

## 🗓️ SEMANA 14-15: Algoritmos de Ordenamiento

**Módulo:** 08 - Algoritmos de Ordenamiento
**Objetivo:** Implementar QuickSort y MergeSort desde cero

### Semana 14

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | QuickSort: concepto | Pivot, partición | Diagrama |
| M | QuickSort: partición | Implementar partition() | Ejercicio 8.1 |
| X | QuickSort: recursión | Implementar quicksort() | Implementación |
| J | Pivot selection | Random vs fijo | Comparar |
| V | Análisis de complejidad | O(n log n) vs O(n²) | Documento |
| S | **Repaso** | QuickSort funcional | Test básico |

### Semana 15

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | MergeSort: concepto | Divide, merge | Diagrama |
| M | MergeSort: merge | Implementar merge() | Ejercicio 8.2 |
| X | MergeSort: recursión | Implementar mergesort() | Implementación |
| J | Comparación Quick vs Merge | Cuándo usar cada uno | Tabla comparativa |
| V | Testing exhaustivo | Casos edge, estabilidad | test_sorting.py |
| S | **Repaso** | sorting.py completo | Benchmark |

**Entregable:** `sorting.py` con QuickSort, MergeSort, análisis.

---

## 🗓️ SEMANA 16: Búsqueda Binaria

**Módulo:** 09 - Búsqueda Binaria
**Objetivo:** Implementación perfecta sin errores off-by-one

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Binary Search: concepto | Dividir espacio a la mitad | Diagrama |
| M | Implementación clásica | `binary_search()` | Ejercicio 9.1 |
| X | Off-by-one errors | Cómo evitarlos | Debug común |
| J | Variante: primer elemento | `find_first()` | Implementación |
| V | Variante: último elemento | `find_last()` | Implementación |
| S | **Checkpoint Fase III** | Simulacro Algoritmos | Autoevaluación |

**Entregable:** `searching.py` con variantes de binary search.

**Checkpoint:** [SIMULACRO_ALGORITMOS.md](SIMULACRO_ALGORITMOS.md)

---

## 🗓️ SEMANA 17-18: Álgebra Lineal sin NumPy

**Módulo:** 10 - Álgebra Lineal sin NumPy
**Objetivo:** Operaciones vectoriales y matriciales desde cero

### Semana 17

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Vectores como listas | Representación | Ejercicio 10.1 |
| M | Suma de vectores | `add_vectors()` | Implementación |
| X | Producto escalar | Multiplicar por escalar | Ejercicio 10.2 |
| J | Producto punto | `dot_product()` | Implementación |
| V | Norma/magnitud | `magnitude()` | Implementación |
| S | **Repaso** | Operaciones vectoriales | Test |

### Semana 18

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Matrices como listas de listas | Representación 2D | Ejercicio 10.3 |
| M | Suma de matrices | `add_matrices()` | Implementación |
| X | Transpuesta | `transpose()` | Implementación |
| J | Producto matriz-vector | `matrix_vector_mult()` | Implementación |
| V | Testing matemático | Verificar con cálculos | test_linear_algebra.py |
| S | **Repaso** | linear_algebra.py completo | Autoevaluación |

**Entregable:** `linear_algebra.py` con operaciones vectoriales/matriciales.

---

## 🗓️ SEMANA 19-20: TF-IDF y Similitud de Coseno

**Módulo:** 11 - TF-IDF y Similitud de Coseno
**Objetivo:** Sistema de ranking por relevancia

### Semana 19

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Term Frequency (TF) | Fórmula y concepto | Ejercicio 11.1 |
| M | Implementar TF | `compute_tf()` | Implementación |
| X | Inverse Document Frequency | Fórmula y concepto | Ejercicio 11.2 |
| J | Implementar IDF | `compute_idf()` | Implementación |
| V | TF-IDF combinado | `compute_tfidf()` | Implementación |
| S | **Repaso** | Vectores TF-IDF | Test manual |

### Semana 20

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Similitud de Coseno | Fórmula y geometría | Diagrama |
| M | Implementar coseno | `cosine_similarity()` | Ejercicio 11.3 |
| X | Ranking de documentos | Ordenar por similitud | Implementación |
| J | Integrar con QuickSort | Ordenar resultados | Implementación |
| V | Testing completo | Verificar rankings | test_similarity.py |
| S | **Checkpoint Fase IV** | Simulacro Matemáticas | Autoevaluación |

**Entregable:** `vectorizer.py` + `similarity.py` + tests.

**Checkpoint:** [SIMULACRO_MATEMATICAS.md](SIMULACRO_MATEMATICAS.md)

---

## 🗓️ SEMANA 21-24: Proyecto Integrador

**Módulo:** 12 - Proyecto Integrador
**Objetivo:** Motor de búsqueda completo + defensa técnica

### Semana 21: Ensamblaje

| Día | Actividad |
|-----|-----------|
| L | Diseñar clase `SearchEngine` |
| M | Integrar `Corpus` + `InvertedIndex` |
| X | Integrar `Tokenizer` |
| J | Integrar `Vectorizer` + `Similarity` |
| V | Método `search(query, top_k)` |
| S | Demo básica funcionando |

### Semana 22: Refinamiento

| Día | Actividad |
|-----|-----------|
| L | Integrar `QuickSort` para ranking |
| M | Optimizar performance |
| X | Tests de integración |
| J | Casos edge y errores |
| V | Cobertura >80% |
| S | Refactorización |

### Semana 23: Documentación y Análisis

| Día | Actividad |
|-----|-----------|
| L | Análisis Big O: agregar documento |
| M | Análisis Big O: búsqueda |
| X | Análisis Big O: ranking |
| J | Escribir COMPLEXITY_ANALYSIS.md |
| V | README.md profesional (inglés) |
| S | Revisar documentación |

### Semana 24: Defensa y Preparación

| Día | Actividad |
|-----|-----------|
| L | Preparar presentación (5 min) |
| M | Practicar explicación en inglés |
| X | Simulacro de entrevista |
| J | Ajustes finales |
| V | **Demo final grabada** |
| S | **Autoevaluación final** |

**Entregable Prerrequisitos:** Motor de búsqueda funcional con TF-IDF

---

# ⭐ FASE PATHWAY (Semanas 25-35)

## 🗓️ Semanas 25-28: Probabilidad y Estadística [PATHWAY LÍNEA 2]

### Semana 25-26: Fundamentos de Probabilidad (Módulo 19)

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Probabilidad básica, axiomas | Simular dados, monedas | Ejercicio 19.1 |
| M | Probabilidad condicional | Implementar P(A|B) | Ejercicio 19.2 |
| X | Teorema de Bayes | Bayes desde cero | Ejercicio 19.3 |
| J | Variables aleatorias | Distribuciones discretas | Ejercicio 19.4 |
| V | Distribuciones continuas | Normal, exponencial | Ejercicio 19.5 |
| S | **Repaso** | Naive Bayes simple | Simulacro Prob |

### Semana 27-28: Estadística Inferencial (Módulo 20)

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Estimación puntual | MLE Bernoulli | Ejercicio 20.1 |
| M | MLE general | MLE Normal | Ejercicio 20.2 |
| X | Intervalos de confianza | CI desde cero | Ejercicio 20.3 |
| J | Tests de hipótesis | Z-test, T-test | Ejercicio 20.4 |
| V | Cross-validation | K-fold desde cero | Ejercicio 20.5 |
| S | **Repaso** | Bootstrap | Simulacro Estadística |

### Semana 29: Cadenas de Markov (Módulo 21)

| Día | Actividad |
|-----|-----------|
| L | Teoría: Cadenas de Markov, matrices de transición |
| M | Código: Construir matriz de transición |
| X | Teoría: Distribución estacionaria, PageRank |
| J | Código: PageRank desde cero |
| V | Teoría: Monte Carlo, MCMC |
| S | Código: Metropolis-Hastings, generador de texto |

**Checkpoint:** Simulacro Probabilidad (20 preguntas)

---

## 🗓️ Semanas 30-33: Machine Learning [PATHWAY LÍNEA 1]

### Semana 30-31: ML Supervisado (Módulo 22)

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Pipeline de ML, bias-variance | Regresión lineal | Ejercicio 22.1 |
| M | Gradiente descendente | Regresión con GD | Ejercicio 22.2 |
| X | Clasificación logística | Logistic regression | Ejercicio 22.3 |
| J | Árboles de decisión | Decision tree | Ejercicio 22.4 |
| V | KNN | KNN desde cero | Ejercicio 22.5 |
| S | **Repaso** | Métricas de evaluación | Simulacro ML |

### Semana 32: ML No Supervisado (Módulo 23)

| Día | Actividad |
|-----|-----------|
| L | Teoría: Clustering, K-Means |
| M | Código: K-Means desde cero |
| X | Teoría: PCA, reducción de dimensionalidad |
| J | Código: PCA desde cero |
| V | Teoría: Detección de anomalías |
| S | Código: LOF, evaluación de clusters |

### Semana 33: Deep Learning (Módulo 24)

| Día | Actividad |
|-----|-----------|
| L | Teoría: Perceptrón, neurona artificial |
| M | Código: Perceptrón, funciones de activación |
| X | Teoría: MLP, backpropagation |
| J | Código: MLP que resuelve XOR |
| V | Teoría: SGD, Adam, regularización |
| S | Código: Red neuronal completa |

**Checkpoint:** Simulacro Machine Learning (20 preguntas)

---

## 🗓️ Semanas 34-35: Proyecto Final e Integración

### Semana 34: ML Pipeline (Módulo 12)

| Día | Actividad |
|-----|-----------|
| L | Integrar NaiveBayesClassifier |
| M | Integrar KMeans |
| X | Integrar NeuralNetwork |
| J | Integrar MarkovTextGenerator |
| V | Evaluación estadística con CI |
| S | Tests de integración |

### Semana 35: Defensa y Preparación Final

| Día | Actividad |
|-----|-----------|
| L | Comparación estadística de modelos |
| M | Documentar MODEL_COMPARISON.md |
| X | README.md profesional en inglés |
| J | Simulacro completo (120 preguntas) |
| V | **Demo final del pipeline** |
| S | **Autoevaluación + Preparar auditar cursos del Pathway** |

---

## ✅ Checklist de Finalización

### Prerrequisitos (Módulos 01-18)
- [ ] Módulos 01-11 completados
- [ ] Módulos 13-18 (DSA) completados
- [ ] Motor de búsqueda TF-IDF funcional

### ⭐ Pathway Línea 2: Probabilidad (Módulos 19-21)
- [ ] Módulo 19: Fundamentos de Probabilidad
- [ ] Módulo 20: Estadística Inferencial
- [ ] Módulo 21: Cadenas de Markov

### ⭐ Pathway Línea 1: Machine Learning (Módulos 22-24)
- [ ] Módulo 22: ML Supervisado
- [ ] Módulo 23: ML No Supervisado
- [ ] Módulo 24: Deep Learning

### Proyecto Final (Módulo 12)
- [ ] ML Pipeline completo funcionando
- [ ] Comparación estadística de modelos
- [ ] README en inglés
- [ ] Tests con >80% coverage

### Preparación Final
- [ ] Simulacro completo aprobado (100+ preguntas)
- [ ] Capaz de explicar cada modelo en inglés
- [ ] Listo para auditar cursos del Pathway

---

## 📚 Recursos

Ver **[RECURSOS.md](RECURSOS.md)** para la lista completa de cursos, libros y videos.

---

> 💡 **Tip:** Prioriza los módulos 19-24 y el proyecto 12. Son el **foco del Pathway**. Los prerrequisitos (01-18) son base, no el objetivo.
