# 🎯 Simulacro de Entrevista - MS in AI Pathway

> 120+ preguntas con respuestas detalladas para las **2 líneas del Pathway**

---

## 📋 Estructura del Simulacro

| Sección | Categoría | Preguntas | Tiempo |
|---------|-----------|-----------|--------|
| 1. Python y OOP | [PRERREQUISITO] | 10 | 15 min |
| 2. Estructuras de Datos | [PRERREQUISITO] | 15 | 25 min |
| 3. Trees y Graphs | [PRERREQUISITO] | 15 | 30 min |
| 4. Algoritmos y DP | [PRERREQUISITO] | 20 | 40 min |
| 5. Matemáticas y Big O | [PRERREQUISITO] | 20 | 30 min |
| **6. Probabilidad y Estadística** | ⭐ [PATHWAY LÍNEA 2] | 20 | 30 min |
| **7. Machine Learning** | ⭐ [PATHWAY LÍNEA 1] | 20 | 35 min |

**Total:** 120+ preguntas, ~205 minutos

---

## ✅ Checklist Mínimo Pathway

Si tienes poco tiempo, **prioriza las secciones 6 y 7**:
- [ ] Sección 6: 20 preguntas de Probabilidad/Estadística
- [ ] Sección 7: 20 preguntas de Machine Learning

---

## Sección 1: Python y OOP [PRERREQUISITO]

### P1: ¿Qué son los type hints y por qué usarlos?
**R:** Anotaciones que indican tipos esperados. Beneficios: documentación viva, detección de errores con mypy, mejor autocompletado.

```python
def greet(name: str) -> str:
    return f"Hello, {name}"
```

### P2: ¿Cuál es la diferencia entre `list` y `tuple`?
**R:** 
- `list`: mutable, se puede modificar
- `tuple`: inmutable, no se puede cambiar después de crear
- `tuple` es hashable (puede ser clave de dict), `list` no

### P3: ¿Qué significa que Python sea "pass by object reference"?
**R:** Se pasa referencia al objeto. Si el objeto es mutable, cambios dentro de la función afectan al original. Si es inmutable, se crea nuevo objeto.

### P4: ¿Para qué sirve `__init__`?
**R:** Inicializar atributos de instancia cuando se crea un objeto. Es el constructor de la clase.

### P5: ¿Cuál es la diferencia entre `__str__` y `__repr__`?
**R:** 
- `__str__`: para usuarios, legible
- `__repr__`: para desarrolladores, sin ambigüedad, idealmente eval-able

### P6: ¿Qué es un property en Python?
**R:** Mecanismo para controlar acceso a atributos con getter/setter, manteniendo sintaxis de atributo.

### P7: ¿Qué significa "composición sobre herencia"?
**R:** Preferir contener objetos de otra clase (has-a) sobre heredar (is-a). Más flexible y menos acoplado.

### P8: ¿Qué es una función pura?
**R:** Función que siempre retorna mismo output para mismo input y no tiene efectos secundarios.

### P9: ¿Para qué sirve `@dataclass`?
**R:** Genera automáticamente `__init__`, `__repr__`, `__eq__` para clases que principalmente almacenan datos.

### P10: ¿Cómo harías una clase inmutable?
**R:** Usar `@dataclass(frozen=True)` o definir `__setattr__` para prevenir modificaciones.

---

## Sección 2: Estructuras de Datos [PRERREQUISITO]

### P11: ¿Cuál es la complejidad de buscar en una lista vs en un set?
**R:** Lista: O(n), Set: O(1) promedio. Set usa hashing.

### P12: ¿Cómo funciona internamente un diccionario?
**R:** Hash table. La clave se hashea para determinar posición en array interno. Colisiones se resuelven con probing.

### P13: ¿Por qué `dict` es O(1) para acceso?
**R:** Hash de la clave da posición directa. No necesita buscar secuencialmente.

### P14: ¿Qué es una colisión en hash table?
**R:** Cuando dos claves diferentes producen el mismo hash. Se resuelve buscando siguiente slot disponible.

### P15: ¿Qué puede ser clave de diccionario?
**R:** Solo objetos hashables (inmutables): str, int, float, tuple, frozenset. No: list, set, dict.

### P16: ¿Cuál es la diferencia entre `set` y `frozenset`?
**R:** `set` es mutable, `frozenset` inmutable. frozenset puede ser clave de dict o elemento de otro set.

### P17: ¿Qué es un índice invertido?
**R:** Estructura que mapea términos a documentos que los contienen. `{"word": [doc1, doc2, ...]}`. Base de motores de búsqueda.

### P18: ¿Por qué usarías un set para stop words?
**R:** Búsqueda O(1). Si son 50 stop words y 1000 tokens, con lista sería O(50×1000)=O(50000), con set O(1000).

### P19: ¿Cuál es la complejidad de `list.append()` vs `list.insert(0, x)`?
**R:** 
- append: O(1) amortizado
- insert(0): O(n) porque mueve todos los elementos

### P20: ¿Qué estructura usarías para un contador de frecuencias?
**R:** `dict` o `collections.Counter`. Mapea elemento a conteo, acceso O(1).

### P21: ¿Cómo implementarías búsqueda AND con sets?
**R:** Intersección: `set1 & set2`. Retorna elementos en ambos.

### P22: ¿Cómo implementarías búsqueda OR con sets?
**R:** Unión: `set1 | set2`. Retorna elementos en cualquiera.

### P23: ¿Qué es Document Frequency?
**R:** Número de documentos que contienen un término. Usado para calcular IDF.

### P24: ¿Cuándo usarías `defaultdict`?
**R:** Cuando quieres valores por defecto automáticos. Ej: `defaultdict(list)` crea listas vacías para claves nuevas.

### P25: ¿Qué es un posting list?
**R:** Lista de documentos que contienen un término, almacenada en índice invertido.

---

## Sección 3: Trees y Graphs [PRERREQUISITO]

### P26: ¿Qué es un Binary Tree?
**R:** Árbol donde cada nodo tiene máximo 2 hijos (left y right).

### P27: ¿Cuál es la diferencia entre Binary Tree y BST?
**R:** 
- Binary Tree: cualquier árbol con máx 2 hijos
- BST: Binary tree donde left < root < right

### P28: ¿Cuáles son los tres traversals DFS de un árbol?
**R:** 
- Inorder: Left, Root, Right (en BST da orden ascendente)
- Preorder: Root, Left, Right
- Postorder: Left, Right, Root

### P29: ¿Cómo implementarías level-order traversal?
**R:** Usar Queue (BFS). Agregar root, luego procesar nivel por nivel.

### P30: ¿Cuál es la complejidad de search en BST?
**R:** O(log n) promedio, O(n) peor caso (árbol desbalanceado/lineal).

### P31: ¿Qué es un grafo dirigido vs no dirigido?
**R:** 
- Dirigido: edges tienen dirección (A→B no implica B→A)
- No dirigido: conexión bidireccional (A↔B)

### P32: ¿Cuáles son las dos formas de representar un grafo?
**R:** 
- Adjacency List: dict de listas, O(V+E) espacio
- Adjacency Matrix: matriz V×V, O(V²) espacio

### P33: ¿Cuál es la diferencia entre BFS y DFS?
**R:** 
- BFS: explora por niveles, usa Queue, encuentra shortest path
- DFS: explora en profundidad, usa Stack/recursión

### P34: ¿Cuándo usar BFS vs DFS?
**R:** 
- BFS: shortest path (no ponderado), nivel por nivel
- DFS: detectar ciclos, caminos, backtracking

### P35: ¿Cómo detectar un ciclo en un grafo?
**R:** DFS marcando nodos como "en progreso" y "visitado". Si encuentras nodo "en progreso", hay ciclo.

### P36: ¿Qué es un DAG?
**R:** Directed Acyclic Graph. Grafo dirigido sin ciclos. Permite topological sort.

### P37: ¿Cuál es la complejidad de BFS/DFS?
**R:** O(V + E) donde V = vértices, E = edges.

### P38: ¿Qué estructura usa BFS y cuál DFS?
**R:** 
- BFS: Queue (FIFO)
- DFS: Stack (LIFO) o recursión

### P39: ¿Por qué BFS garantiza shortest path en grafos no ponderados?
**R:** Porque explora todos los nodos a distancia k antes de los de distancia k+1.

### P40: ¿Cómo encontrarías camino más corto en grafo ponderado?
**R:** Dijkstra's algorithm (no cubierto en detalle, pero saber que existe).

---

## Sección 4: Algoritmos y DP [PRERREQUISITO]

### P41: Explica cómo funciona QuickSort.
**R:** 
1. Elegir pivote
2. Particionar: menores a izquierda, mayores a derecha
3. Recursivamente ordenar cada partición
Complejidad: O(n log n) promedio, O(n²) peor caso.

### P42: ¿Por qué QuickSort puede ser O(n²)?
**R:** Si el pivote siempre es el mínimo o máximo. Ej: lista ya ordenada con pivote fijo al final. Cada partición solo reduce en 1.

### P28: ¿Cómo evitar el peor caso de QuickSort?
**R:** Random pivot selection. Aleatoriza la elección del pivote.

### P29: Explica MergeSort.
**R:**
1. Dividir lista en dos mitades
2. Ordenar cada mitad recursivamente
3. Fusionar las mitades ordenadas
Complejidad: O(n log n) siempre.

### P30: ¿Cuál es la diferencia entre QuickSort y MergeSort?
**R:**
- QuickSort: in-place, O(log n) espacio, no estable
- MergeSort: O(n) espacio, estable, siempre O(n log n)

### P31: ¿Qué significa que un sort sea "estable"?
**R:** Elementos iguales mantienen su orden relativo original.

### P32: Explica Binary Search.
**R:** En lista ordenada, comparar con elemento medio. Si menor, buscar en mitad izquierda; si mayor, en derecha. Complejidad: O(log n).

### P33: ¿Cuál es el error off-by-one más común en binary search?
**R:** Usar `while left < right` en lugar de `left <= right`, o no ajustar correctamente mid+1/mid-1.

### P34: ¿Qué es recursión?
**R:** Función que se llama a sí misma. Requiere caso base (termina) y caso recursivo (se llama con input menor).

### P35: ¿Qué es el call stack?
**R:** Pila que guarda estado de cada llamada a función. Cada llamada recursiva agrega un frame.

### P36: ¿Qué es memoization?
**R:** Cachear resultados de funciones para evitar recálculo. Útil en recursión con subproblemas repetidos.

### P37: ¿Por qué Fibonacci naive es O(2^n)?
**R:** Cada llamada hace dos llamadas. Árbol de llamadas crece exponencialmente. fib(n) se recalcula muchas veces.

### P38: ¿Cómo optimizar Fibonacci a O(n)?
**R:** Memoization: guardar resultados en dict/cache. Cada valor se calcula solo una vez.

### P39: ¿Qué es Divide & Conquer?
**R:** Patrón que divide problema en subproblemas, resuelve cada uno, y combina soluciones. Ej: MergeSort, QuickSort.

### P43: ¿Cómo fusionarías dos listas ordenadas?
**R:** Two pointers: comparar elementos actuales de ambas, agregar el menor al resultado, avanzar ese puntero. O(n+m).

### P44: ¿Qué es Dynamic Programming?
**R:** Técnica que guarda resultados de subproblemas para evitar recálculo. Requiere optimal substructure + overlapping subproblems.

### P45: ¿Cuáles son los dos enfoques de DP?
**R:** 
- Top-down: Recursivo con memoization
- Bottom-up: Iterativo con tabulation

### P46: ¿Qué es la recurrencia de Coin Change?
**R:** dp[amount] = min(dp[amount - coin] + 1) para todas las monedas válidas.

### P47: ¿Cuándo usar Greedy vs DP?
**R:** 
- Greedy: Si la mejor opción local lleva al óptimo global
- DP: Si necesitas explorar todas las opciones

### P48: ¿Qué es "greedy choice property"?
**R:** Propiedad donde elegir el óptimo local en cada paso lleva al óptimo global.

### P49: ¿Cómo funciona Activity Selection greedy?
**R:** Ordenar por tiempo de fin, siempre elegir la que termina primero y no se superpone.

### P50: ¿Qué es un Heap?
**R:** Árbol binario completo con propiedad heap (parent <= children para min-heap).

### P51: ¿Cuáles son las complejidades de operaciones en Heap?
**R:** Insert: O(log n), Extract-min: O(log n), Peek: O(1), Heapify: O(n).

### P52: ¿Cómo encontrar los K elementos más grandes?
**R:** Usar min-heap de tamaño k. Para cada elemento, si es mayor que el mínimo del heap, reemplazar.

### P53: ¿Por qué usar min-heap para K largest?
**R:** Min-heap mantiene el k-ésimo más grande en la raíz. Elementos más grandes que la raíz entran al heap.

### P54: ¿Qué es Priority Queue?
**R:** Cola donde elementos salen por prioridad, no por orden de llegada. Se implementa con Heap.

### P55: ¿Cómo hacer max-heap en Python?
**R:** heapq es min-heap. Para max-heap, negar los valores al insertar y al extraer.

---

## Sección 5: Matemáticas y Big O [PRERREQUISITO]

### P56: ¿Qué significa O(n)?
**R:** El tiempo crece linealmente con el tamaño de entrada. Duplicar n duplica el tiempo.

### P57: Ordena de menor a mayor: O(n²), O(1), O(n log n), O(log n), O(n)
**R:** O(1) < O(log n) < O(n) < O(n log n) < O(n²)

### P58: ¿Cuántas comparaciones hace binary search en 1 millón de elementos?
**R:** log₂(1,000,000) ≈ 20 comparaciones.

### P59: ¿Qué es el producto punto?
**R:** Suma de productos de componentes correspondientes: a·b = a₁b₁ + a₂b₂ + ... Resultado es escalar.

### P45: ¿Qué es la norma de un vector?
**R:** Su longitud/magnitud. ||v|| = √(v₁² + v₂² + ...). Distancia del origen al punto.

### P46: ¿Qué mide la similitud de coseno?
**R:** El coseno del ángulo entre vectores. 1 = misma dirección, 0 = perpendiculares. Mide similitud ignorando magnitud.

### P47: ¿Qué es TF (Term Frequency)?
**R:** Frecuencia de un término en un documento, normalizada por longitud. TF = count/total_terms.

### P48: ¿Qué es IDF (Inverse Document Frequency)?
**R:** Mide qué tan raro es un término. IDF = log(N/df). Términos raros tienen IDF alto.

### P49: ¿Por qué usamos TF-IDF en lugar de solo TF?
**R:** TF solo mide frecuencia local. IDF penaliza palabras comunes ("the", "is"). TF-IDF balancea ambos.

### P50: ¿Cuál es la complejidad de calcular similitud de coseno?
**R:** O(V) donde V es la dimensión del vector (tamaño del vocabulario). Hay que recorrer todos los componentes.

---

---

## Sección 6: Probabilidad y Estadística ⭐ [PATHWAY LÍNEA 2]

### P60: ¿Qué es el Teorema de Bayes y para qué se usa en ML?
**R:** P(A|B) = P(B|A) × P(A) / P(B). Permite actualizar creencias (prior) dado nueva evidencia (likelihood). Base de clasificadores Naive Bayes y modelos probabilísticos.

### P61: ¿Cuál es la diferencia entre probabilidad y likelihood?
**R:** 
- Probabilidad: P(data|params) - probabilidad de datos dados parámetros fijos
- Likelihood: L(params|data) - qué tan probables son los parámetros dados los datos

### P62: ¿Qué es MLE (Maximum Likelihood Estimation)?
**R:** Encontrar los parámetros θ que maximizan la probabilidad de observar los datos: θ̂ = argmax P(data|θ). Es cómo se entrenan la mayoría de modelos de ML.

### P63: ¿Qué es MAP y cómo se relaciona con regularización?
**R:** Maximum A Posteriori incorpora un prior: θ̂ = argmax P(θ|data) ∝ P(data|θ) × P(θ). Prior gaussiano → L2 regularization. Prior laplaciano → L1 regularization.

### P64: ¿Qué es la distribución normal y por qué es importante?
**R:** Distribución "campana de Gauss". Importante por el Teorema del Límite Central: la suma de muchas variables independientes tiende a normal. Muchos errores en ML se asumen normales.

### P65: ¿Qué es esperanza y varianza?
**R:** 
- E[X] = Σ x × P(x) = "valor promedio esperado"
- Var(X) = E[(X - μ)²] = "spread" alrededor de la media

### P66: ¿Qué es una cadena de Markov?
**R:** Proceso estocástico donde el futuro solo depende del estado actual, no del pasado: P(Xₙ₊₁|Xₙ, Xₙ₋₁, ...) = P(Xₙ₊₁|Xₙ). Usado en PageRank, modelos de lenguaje.

### P67: ¿Qué es la distribución estacionaria de una cadena de Markov?
**R:** Distribución π tal que π = πP. Después de muchos pasos, la cadena converge a esta distribución sin importar el estado inicial.

### P68: ¿Qué es MCMC y para qué se usa?
**R:** Markov Chain Monte Carlo. Técnica para muestrear de distribuciones complejas construyendo una cadena de Markov cuya distribución estacionaria es la distribución objetivo.

### P69: Explica el algoritmo Metropolis-Hastings.
**R:**
1. Proponer nuevo estado x' desde distribución q(x'|x)
2. Aceptar con probabilidad min(1, P(x')/P(x))
3. Si acepta, mover a x'; si no, quedarse en x
4. Repetir

### P70: ¿Qué es un intervalo de confianza?
**R:** Rango [a,b] tal que si repitiéramos el experimento muchas veces, el parámetro real estaría dentro del intervalo en (1-α)% de las veces (ej: 95%).

### P71: ¿Cuál es la diferencia entre error Tipo I y Tipo II?
**R:**
- Tipo I (α): Rechazar H₀ cuando es verdadera (falso positivo)
- Tipo II (β): No rechazar H₀ cuando es falsa (falso negativo)

### P72: ¿Qué es covarianza y correlación?
**R:**
- Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] - Relación lineal, no normalizada
- Correlation = Cov(X,Y)/(σₓσᵧ) - Normalizada a [-1, 1]

### P73: ¿Qué es la distribución Bernoulli y Binomial?
**R:**
- Bernoulli: Un solo experimento con prob p de éxito
- Binomial: k éxitos en n experimentos Bernoulli independientes

### P74: ¿Por qué usamos log-likelihood en lugar de likelihood?
**R:** Producto de probabilidades pequeñas → underflow. Logaritmo convierte productos en sumas, numéricamente más estable.

### P75: ¿Qué es el Teorema del Límite Central?
**R:** La distribución de la media muestral tiende a una normal cuando n → ∞, sin importar la distribución original. Justifica asumir normalidad en muchos contextos.

### P76: ¿Qué es independencia condicional?
**R:** P(A,B|C) = P(A|C) × P(B|C). A y B son independientes dado C. Base de Naive Bayes: features son independientes dado la clase.

### P77: ¿Qué es estimador insesgado?
**R:** Estimador cuyo valor esperado es igual al parámetro real: E[θ̂] = θ. Ejemplo: media muestral es insesgada para la media poblacional.

### P78: ¿Qué es bootstrap?
**R:** Técnica de remuestreo: crear muchas muestras tomando con reemplazo de los datos originales. Usado para estimar varianza de estimadores.

### P79: ¿Qué es el p-value?
**R:** Probabilidad de observar resultados tan extremos como los observados, asumiendo que H₀ es verdadera. Si p < α, rechazamos H₀.

---

## Sección 7: Machine Learning ⭐ [PATHWAY LÍNEA 1]

### P80: ¿Cuál es la diferencia entre aprendizaje supervisado y no supervisado?
**R:**
- Supervisado: Datos etiquetados (X, y). Objetivo: predecir y dado X.
- No supervisado: Solo datos X. Objetivo: encontrar estructura (clusters, dimensiones).

### P81: ¿Qué es el bias-variance tradeoff?
**R:** Error = Bias² + Variance + Ruido irreducible.
- Bias alto → underfitting (modelo muy simple)
- Variance alta → overfitting (modelo muy complejo)
Objetivo: encontrar el balance óptimo.

### P82: ¿Qué es overfitting y cómo detectarlo?
**R:** Modelo aprende ruido del train set y no generaliza. Se detecta cuando train accuracy >> test accuracy. Soluciones: más datos, regularización, menos complejidad.

### P83: ¿Qué es cross-validation y para qué sirve?
**R:** Dividir datos en k folds, entrenar en k-1 y validar en 1, rotar. Da estimación más robusta del rendimiento que un solo train/test split.

### P84: Explica gradient descent.
**R:** Algoritmo de optimización: w = w - lr × ∂L/∂w. Sigue la dirección de máxima pendiente descendente para minimizar la función de pérdida.

### P85: ¿Cuál es la diferencia entre batch, mini-batch y SGD?
**R:**
- Batch: Usa todos los datos para cada update
- Mini-batch: Usa subconjunto (ej: 32 samples)
- SGD: Usa 1 sample por update
Mini-batch es el más común: balance entre estabilidad y velocidad.

### P86: ¿Qué es regularización L1 y L2?
**R:**
- L1 (Lasso): Suma de |w|. Produce sparsity (pesos = 0).
- L2 (Ridge): Suma de w². Shrinks pesos pero no a cero.
Ambas previenen overfitting al penalizar pesos grandes.

### P87: Explica regresión logística.
**R:** Clasificador lineal: P(y=1|x) = σ(wᵀx + b). Usa sigmoid para mapear a [0,1]. Se entrena minimizando binary cross-entropy con gradient descent.

### P88: ¿Cómo funciona un árbol de decisión?
**R:** Divide recursivamente los datos según el feature que maximiza ganancia de información (o minimiza Gini). Hojas contienen predicciones. Fácil de interpretar, propenso a overfitting.

### P89: ¿Qué es Random Forest?
**R:** Ensemble de árboles de decisión. Cada árbol entrena en bootstrap sample con subset de features aleatorio. Predicción final = promedio/voto mayoritario. Reduce variance.

### P90: Explica K-Nearest Neighbors.
**R:** Predice según el voto de los k vecinos más cercanos. No-paramétrico (no entrena). Complejidad O(n×d) por predicción. Sensible a escala de features.

### P91: ¿Qué es SVM y cuál es la idea del kernel trick?
**R:** SVM encuentra hiperplano con máximo margen entre clases. Kernel trick: proyectar a dimensión superior donde datos son linealmente separables, sin calcular la proyección explícita.

### P92: ¿Qué métricas usarías para clasificación desbalanceada?
**R:** Accuracy engaña. Mejor usar:
- Precision: TP/(TP+FP) - de los predichos +, cuántos son +
- Recall: TP/(TP+FN) - de los reales +, cuántos encontramos
- F1: armonic mean de precision y recall
- AUC-ROC

### P93: Explica K-Means clustering.
**R:**
1. Inicializar k centroides aleatorios
2. Asignar cada punto al centroide más cercano
3. Actualizar centroides al promedio de sus puntos
4. Repetir hasta convergencia
Requiere especificar k. Sensible a inicialización.

### P94: ¿Cómo elegir el número de clusters en K-Means?
**R:**
- Elbow method: graficar inertia vs k, buscar "codo"
- Silhouette score: mide cohesión vs separación
- Domain knowledge

### P95: ¿Qué es PCA y para qué sirve?
**R:** Principal Component Analysis. Reduce dimensionalidad proyectando a direcciones de máxima varianza (eigenvectors de la matriz de covarianza). Usado para visualización, compresión, preprocesamiento.

### P96: ¿Qué es una red neuronal?
**R:** Composición de funciones: y = f(W₃ × f(W₂ × f(W₁x + b₁) + b₂) + b₃). Cada capa es transformación lineal + activación no lineal. Aprende features automáticamente.

### P97: Explica backpropagation.
**R:** Algoritmo para calcular gradientes en redes neuronales usando la regla de la cadena. Forward pass calcula output, backward pass propaga gradientes desde el loss hacia atrás.

### P98: ¿Qué son funciones de activación y cuáles conoces?
**R:** Funciones no lineales entre capas.
- Sigmoid: (0,1), problemas de vanishing gradient
- ReLU: max(0,x), estándar para capas ocultas
- Softmax: para output de clasificación multiclase

### P99: ¿Qué es una CNN y para qué se usa?
**R:** Convolutional Neural Network. Capas de convolución extraen features espaciales. Usadas para imágenes. Ventaja: comparten parámetros, detectan patrones independiente de posición.

### P100: ¿Qué es una RNN y cuál es el problema del vanishing gradient?
**R:** Recurrent Neural Network. Estado oculto depende del anterior, captura secuencias. Vanishing gradient: gradientes se vuelven muy pequeños en secuencias largas. Solución: LSTM, GRU.

---

## 🎯 Autoevaluación

| Respuestas Correctas | Nivel |
|---------------------|-------|
| 100-120 | 🏆 Listo para Pathway - Ambas líneas |
| 80-99 | ✅ Buen nivel, reforzar gaps |
| 60-79 | ⚠️ Necesita más estudio |
| <60 | ❌ Revisar módulos |

---

## 💡 Tips para la Entrevista Real

1. **Explica tu pensamiento:** Verbaliza mientras resuelves
2. **Empieza simple:** Primero solución bruta, luego optimiza
3. **Pregunta si dudas:** Clarifica requisitos
4. **Analiza Big O:** Siempre menciona complejidad
5. **Practica en inglés:** Todo el Pathway es en inglés
6. **Conecta conceptos:** ML usa probabilidad, DL usa álgebra lineal
7. **Implementa desde cero:** Demuestra que entiendes, no solo usas sklearn
