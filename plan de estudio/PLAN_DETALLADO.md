# Plan de Formación Detallado (6 Meses)

**Objetivo:** Preparación para el Pathway MS in AI (CU Boulder).
**Intensidad:** 6 horas/día (Lunes-Sábado) = 36 horas/semana.
**Idioma:** Todo el material técnico debe consumirse en **INGLÉS** (con subtítulos en inglés) para entrenar el oído y el vocabulario técnico.

---

## 🗓️ Estrategia de Inglés (Transversal)
*Nivel Actual: B1 -> Objetivo: B2+/C1 Técnico*

1.  **Inmersión Total:** Configura tus dispositivos, IDE y documentación en inglés.
2.  **Regla de Oro:** Cursos en Coursera/Video **SIEMPRE** con audio en inglés.
    *   *Mes 1-2:* Subtítulos en Inglés (para asociar sonido con texto).
    *   *Mes 3-4:* Intentar sin subtítulos la primera vez, activar si es necesario.
    *   *Mes 5-6:* Sin subtítulos.
3.  **Glosario Activo:** Crea un documento `ENGLISH_GLOSSARY.md` y anota cada término técnico nuevo (e.g., *eigenvector*, *gradient descent*, *linked list*, *heap*).

---

## MES 1: Matemáticas y Lógica (El Despertar)
**Objetivo:** Reactivar el cerebro matemático, limpiar el código y acostumbrarse al inglés técnico básico.

### 🌅 Mañana: Mathematics for Machine Learning: Linear Algebra
*   **Foco:** Vectores, matrices, proyecciones, eigenvalores/eigenvectores.
*   **Inglés:** Presta atención a términos como *span*, *basis*, *linear combination*.
*   **Entregable:** Notebooks con ejercicios resueltos explicados en inglés (comentarios en el código).

### 🌇 Tarde: Python "Hardcore" (Sin Librerías)
*   **Tarea:** Implementar operaciones matriciales (Suma, Producto Punto, Transpuesta, Inversa simple) usando solo listas de Python puros.
*   **Prohibido:** `import numpy`, `import pandas`.
*   **Por qué:** Entenderás la complejidad computacional de iterar sobre arrays anidados.

### 🌙 Noche: Discrete Mathematics (Lógica)
*   **Tema:** Introducción a la lógica matemática, demostraciones (proofs), teoría de conjuntos.
*   **Importancia:** La base para entender algoritmos y bases de datos.

---

## MES 2: Cálculo y Probabilidad (El Motor de la IA)
**Objetivo:** Dominar las matemáticas de la incertidumbre y la optimización. La IA es básicamente estadística computacional.

### 🌅 Mañana: Mathematics for Machine Learning: Multivariate Calculus
*   **Foco:** Derivadas parciales, gradientes, regla de la cadena.
*   **Aplicación:** Entender el "Backpropagation" en redes neuronales. Sin esto, la IA es magia negra.

### 🌇 Tarde: Probability & Statistics for Machine Learning
*   **Curso:** *Probability & Statistics for Machine Learning & Data Science* (DeepLearning.AI).
*   **Temas Clave:** Teorema de Bayes, Distribuciones (Normal, Binomial), Esperanza Matemática, Varianza.
*   **Por qué:** Sustituye a "Arquitectura" porque para entrar a IA, es infinitamente más valioso saber probabilidad que construir un chip.

### 🏖️ Fin de Semana: Machine Learning Specialization (Andrew Ng)
*   **Curso:** Supervised Machine Learning.
*   **Conexión:** Ahora entenderás que el "Costo" es una función de cálculo y que las "Predicciones" son probabilísticas.

---

## MES 3: Estructuras de Datos I (La Caja de Herramientas)
**Objetivo:** Salir del scripting y entrar a la ingeniería de software seria. Preparación directa para entrevistas técnicas.

### 🔨 Foco Total: Algoritmos y Estructuras de Datos
*   **Libro Guía:** *Grokking Algorithms* (Lectura ligera y visual para conceptos).
*   **Plataforma:** LeetCode (Empieza con nivel Easy).

### Temario Crítico:
1.  **Arrays & Strings:** Manipulación de memoria contigua.
2.  **Linked Lists:** Punteros y referencias.
3.  **Stacks & Queues:** LIFO vs FIFO (Vital para búsquedas BFS/DFS).
4.  **Hash Maps:** La estructura de datos más importante en la práctica (Diccionarios).

### Inglés:
*   Lee los enunciados de los problemas en LeetCode en voz alta.
*   Trata de explicar tu solución en inglés (Rubber Duck Debugging).

---

## MES 4: Matemáticas Discretas II y Algoritmos II (El Filtro)
**Objetivo:** Dominar la complejidad y las estructuras no lineales.

### 🌅 Mañana: Discrete Mathematics (Grafos y Combinatoria)
*   **Temas:** Teoría de Grafos (Nodos, Aristas, Caminos), Árboles, Probabilidad básica.
*   **Por qué:** Los grafos modelan redes sociales, rutas de GPS, y dependencias de software.

### 🌇 Tarde: LeetCode (Trees & Graphs)
*   **Temas:** Binary Trees, BST, DFS (Depth-First Search), BFS (Breadth-First Search).
*   **Advertencia:** Esta es la barrera de entrada. Si entiendes recursión y grafos, estás del otro lado.
*   **Restricción:** NO USAR IA (ChatGPT/Copilot) para resolver los problemas. Sufre el problema.

---

## MES 5: Algoritmos de Ordenamiento y Búsqueda (El Pathway)
**Objetivo:** Preparación específica para el examen de admisión (que suele basarse en esto).

### 🔍 Foco: Sorting & Searching
1.  **Sorting:** Merge Sort, Quick Sort, Heap Sort.
    *   *Análisis:* ¿Por qué Quick Sort es O(n log n)? ¿Cuándo es O(n^2)?
2.  **Searching:** Binary Search (Implementación perfecta sin errores "off-by-one").

### 🔗 Integración de Conocimientos
*   Usa **Discretas** para demostrar la eficiencia (Big O Notation).
*   Usa **Arquitectura** para explicar por qué un Array es más rápido que una Linked List (Caché locality).
*   Usa **Python** para implementar desde cero.

---

## MES 6: Simulación y Repaso Final
**Objetivo:** Simulacro de examen y pulido final.

### 🕵️ Auditoría y Simulacros
1.  **Revisión Pathway Boulder:** Entra a Coursera y audita (ver videos gratis) los cursos específicos del Pathway (e.g., "Algorithms for Searching, Sorting, and Indexing").
2.  **LeetCode Medium:** Resuelve 3 problemas diarios de nivel medio en menos de 45 minutos cada uno.
3.  **Mock Interviews:** Grábate explicando la solución de un algoritmo en inglés.

### ✅ Checklist de Salida
- [ ] Puedo implementar un QuickSort de memoria en Python.
- [ ] Entiendo qué es un Gradiente y cómo se calcula.
- [ ] Puedo leer un paper técnico básico en inglés y entender el 80%.
- [ ] Tengo mi entorno de desarrollo local configurado profesionalmente.
