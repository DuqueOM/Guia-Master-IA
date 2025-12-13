# 📅 Plan de Estudios - ML SPECIALIST v3.3

> **24 Semanas | 6 horas/día | Lunes a Sábado**
> **Preparación para MS in AI Pathway - Línea 1: Machine Learning**

**Idioma:** Español | [English →](en/PLAN_ESTUDIOS.md)

---

## 🗓️ Vista General: 24 Semanas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SEMANAS 1-8        │ SEMANAS 9-20       │ SEMANAS 21-24                     │
│ FUNDAMENTOS        │ ML CORE ⭐         │ PROYECTO MNIST                    │
│ Python + Mate      │ PATHWAY LÍNEA 1    │ INTEGRACIÓN                       │
│ + Probabilidad     │ Supervised +       │ Pipeline End-to-End               │
│ Módulos 01-04      │ Unsupervised + DL  │ Módulo 08                         │
│                    │ Módulos 05-07      │ 4 semanas intensivas              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Dedicación total:** 36 horas/semana × 24 semanas = **~864 horas**

### Los 8 Módulos Obligatorios

### ⚡ Enlaces rápidos (bloques 0→100)

Estos atajos te llevan directo a la sección **"Cómo usar este módulo (modo 0→100)"** dentro de cada módulo:

| Módulo | Atajo |
|--------|-------|
| 01 | [M01 → Cómo usar](01_PYTHON_CIENTIFICO.md#m01-0) |
| 02 | [M02 → Cómo usar](02_ALGEBRA_LINEAL_ML.md#m02-0) |
| 03 | [M03 → Cómo usar](03_CALCULO_MULTIVARIANTE.md#m03-0) |
| 04 | [M04 → Cómo usar](04_PROBABILIDAD_ML.md#m04-0) |
| 05 | [M05 → Cómo usar](05_SUPERVISED_LEARNING.md#m05-0) |
| 06 | [M06 → Cómo usar](06_UNSUPERVISED_LEARNING.md#m06-0) |
| 07 | [M07 → Cómo usar](07_DEEP_LEARNING.md#m07-0) |

| Semanas | Módulo | Tema | Curso del Pathway |
|---------|--------|------|-------------------|
| 1-2 | 01 | Python + Pandas + NumPy | - (Fundamento) |
| 3-5 | 02 | Álgebra Lineal para ML | - (Fundamento) |
| 6-7 | 03 | Cálculo Multivariante | - (Fundamento) |
| 8 | 04 | Probabilidad para ML | - (Fundamento) |
| 9-12 | 05 | Supervised Learning | Introduction to ML: Supervised Learning |
| 13-16 | 06 | Unsupervised Learning | Unsupervised Algorithms in ML |
| 17-20 | 07 | Deep Learning + CNNs | Introduction to Deep Learning |
| 21-24 | 08 | Proyecto MNIST Analyst | Integración de las 3 materias |

---

> **Filosofía v3.3:** "Matemáticas Aplicadas a Código". Pandas para datos, NumPy para matemáticas, probabilidad para loss functions.

---

## 📌 Distribución Diaria Típica

| Bloque | Horario | Actividad | Duración |
|--------|---------|-----------|----------|
| 🌅 Mañana | 08:00 - 10:30 | Teoría matemática + notación | 2.5 h |
| ☕ Pausa | 10:30 - 11:00 | Descanso | 30 min |
| 🌇 Mediodía | 11:00 - 13:30 | Implementación en NumPy | 2.5 h |
| 🌙 Tarde | 15:00 - 16:00 | Ejercicios + visualización | 1 h |

Para ver el **protocolo diario detallado**, simulacros de examen y ajustes por semana (versión estratégica v4.0), consulta también:

- [PLAN_V4_ESTRATEGICO.md](PLAN_V4_ESTRATEGICO.md)

---

## 🧠 Protocolo E (v5.1): Rescate Cognitivo + Metacognición + Puente Teoría↔Código

Bloques fijos (para reducir fatiga y mejorar retención):

- **Diario (5 min):** `study_tools/DIARIO_METACOGNITIVO.md`
- **Semanal (20–30 min):** `study_tools/TEORIA_CODIGO_BRIDGE.md`
- **Sábado (1 hora):** `study_tools/CIERRE_SEMANAL.md`
- **Badges por módulo:** `study_tools/BADGES_CHECKPOINTS.md`
- **Simulacros performance-based (PB):** `study_tools/SIMULACRO_PERFORMANCE_BASED.md` (Semanas **8, 16, 23**)
- **Rúbrica (scoring semanal + checkpoints):** `study_tools/RUBRICA_v1.md` + `rubrica.csv`

---

## 🗓️ SEMANA 0: Preparación (Setup + Rúbrica)

**Objetivo:** dejar el repo listo para ejecución + calibrar evaluación.

- Crear plantilla y pesos de rúbrica: `study_tools/RUBRICA_v1.md` + `rubrica.csv`.
- Definir roles de evaluación: auto (estudiante), IA/pareja (AI Code Reviewer), mentor externo (si existe).
- Test rápido: aplicar la rúbrica a 1 entregable pequeño (p.ej. drill NumPy) y ajustar descriptores/pesos.

---

# 🔷 FASE 1: FUNDAMENTOS MATEMÁTICOS (Semanas 1-8)

*Objetivo: Leer notación matemática y traducirla a Python/NumPy*

---

## 🗓️ SEMANA 1-2: Python + Pandas + NumPy (Módulo 01)

**Objetivo:** Dominar Pandas para datos reales + NumPy para matemáticas
**Por qué:** En el mundo real, los datos vienen en CSVs sucios. Pandas es esencial para cargar y limpiar datos antes de aplicar ML.

### Semana 1: Pandas + NumPy Básico

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Pandas: DataFrame y Series | `pd.read_csv()`, `.head()`, `.info()` | Ejercicio 1.1 |
| M | Limpieza de datos | `dropna()`, `fillna()`, dtypes | Ejercicio 1.2 |
| X | Selección y filtrado | `.loc[]`, `.iloc[]`, condiciones | Ejercicio 1.3 |
| J | NumPy: Arrays vs listas | Crear arrays, dtypes | Ejercicio 1.4 |
| V | NumPy: Indexing y slicing | Extraer submatrices | Ejercicio 1.5 |
| S | **Checkpoint** | Pandas → NumPy: `.to_numpy()` | Pipeline de carga |

### Semana 2: NumPy Vectorizado

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Broadcasting (concepto) | Operaciones elemento a elemento | Ejercicio 1.6 |
| M | np.dot, np.matmul, @ | Producto matricial | Ejercicio 1.7 |
| X | Reshape, flatten, transpose | Manipulación de formas | Ejercicio 1.8 |
| J | Agregaciones y ejes | sum, mean, std con axis | Ejercicio 1.9 |
| V | Random en NumPy | Generación de datos sintéticos | Ejercicio 1.10 |
| S | **Checkpoint** | Pipeline completo CSV → NumPy | Entregable |

**Entregable:** Script que carga CSV con Pandas, limpia datos, y convierte a NumPy para análisis.

**Extensión v5.0 – Dirty Data Check (Módulo 01):**
Además del script, documenta al menos **5 problemas reales** del CSV (nulos, outliers, tipos incorrectos, codificación rara, duplicados) y tus decisiones de limpieza en:
`study_tools/DIRTY_DATA_CHECK.md` (Caso 1).

**Evaluación (rúbrica):**

- Scoring rápido semanal en `study_tools/CIERRE_SEMANAL.md` usando `study_tools/RUBRICA_v1.md`.
- Registrar errores relevantes en `study_tools/DIARIO_ERRORES.md` y, si aplica, enlazarlos a un `criterion_id` de `rubrica.csv`.

**Recursos:**
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)

---

## 🗓️ SEMANA 3-5: Álgebra Lineal para ML (Módulo 02)

**Objetivo:** Vectores, matrices, normas, autovectores
**Conexión con Pathway:** Vital para Unsupervised Learning (PCA requiere Eigenvalues) y Deep Learning (multiplicaciones de matrices)

### Semana 3: Vectores y Operaciones Básicas

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Vectores: definición geométrica | Crear vectores en NumPy | Ejercicio 2.1 |
| M | Suma, resta de vectores | Implementar operaciones | Visualizar con matplotlib |
| X | Producto escalar (scalar mult) | `c * v` en NumPy | Ejercicio 2.2 |
| J | Producto punto (dot product) | Fórmula $\vec{a} \cdot \vec{b}$ | np.dot() |
| V | Interpretación geométrica | Proyección, ángulo | Diagrama |
| S | **Repaso** | Funciones vectoriales | Test |

### Semana 4: Normas y Distancias

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Norma L2 (Euclidiana) | $\|x\|_2 = \sqrt{\sum x_i^2}$ | np.linalg.norm() |
| M | Norma L1 (Manhattan) | $\|x\|_1 = \sum |x_i|$ | Implementar manual |
| X | Distancia Euclidiana | $d(a,b) = \|a - b\|_2$ | Función distancia |
| J | Distancia coseno | $1 - \cos(\theta)$ | Similitud coseno |
| V | Aplicación: KNN concepto | Vecino más cercano | Demo simple |
| S | **Repaso** | Librería de distancias | Test |

### Semana 5: Matrices y Descomposición

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Matrices: suma, multiplicación | np.matmul, @ | Ejercicio 2.3 |
| M | Transpuesta, inversa | A.T, np.linalg.inv() | Ejercicio 2.4 |
| X | Autovalores/Autovectores (intro) | Qué son y por qué importan | np.linalg.eig() |
| J | SVD (concepto) | Descomposición de matrices | np.linalg.svd() |
| V | Aplicación: PCA preview | Reducción dimensional | Demo visual |
| S | **Checkpoint** | `linear_algebra.py` completo | Entregable |

**Entregable:** Librería `linear_algebra.py` que implementa:
- Producto punto, normas L1/L2
- Distancia euclidiana y coseno
- Proyección de vectores
- Wrapper para eigenvalues

**Evaluación (rúbrica):**

- Scoring rápido semanal en `study_tools/CIERRE_SEMANAL.md` usando `study_tools/RUBRICA_v1.md`.
- Al checkpoint (Semana 5): scoring parcial del módulo (scope M02 en `rubrica.csv`).

---

## 🗓️ SEMANA 6-7: Cálculo Multivariante (Módulo 03) [CRÍTICO]

**Objetivo:** Derivadas, gradiente, Chain Rule
**Conexión con Pathway:** Es el lenguaje del Deep Learning. Sin la Regla de la Cadena, no entenderás Backpropagation.

### Semana 6: Derivadas, Gradiente y GD

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Derivada: tasa de cambio | Calcular derivadas simples | Ejercicio 3.1 |
| M | Derivadas parciales | $\frac{\partial f}{\partial x}$ para $f(x,y)$ | Ejercicio 3.2 |
| X | Gradiente: vector de parciales | $\nabla f$ | Implementar |
| J | Gradient Descent (concepto) | Algoritmo básico | Pseudocódigo |
| V | Gradient Descent (código) | Minimizar $f(x,y) = x^2 + y^2$ | Implementar |
| S | **Repaso** | Learning rate y convergencia | Visualización |

### Semana 7: Chain Rule

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Regla de la Cadena (1D) | $\frac{d}{dx}f(g(x))$ | Ejercicio 3.3 |
| M | Regla de la Cadena (multi) | Composición de funciones | Ejercicio 3.4 |
| X | Aplicación: función de pérdida | Derivar MSE | Ejercicio 3.5 |
| J | Preview Backpropagation | Cómo fluyen gradientes | Diagrama |
| V | Derivar Cross-Entropy | Preparación para logística | Ejercicio 3.6 |
| S | **Checkpoint** | GD + Chain Rule documentado | Entregable |

**Entregable:** Gradient Descent manual con visualización de trayectoria.

**Evaluación (rúbrica):**

- Scoring rápido semanal en `study_tools/CIERRE_SEMANAL.md` usando `study_tools/RUBRICA_v1.md`.
- Al checkpoint (Semana 7): scoring parcial del módulo (scope M03 en `rubrica.csv`).

---

## 🗓️ SEMANA 8: Probabilidad para ML (Módulo 04)

**Objetivo:** Bayes, Gaussiana, MLE - lo mínimo para entender loss functions
**Conexión con Pathway:** Cross-Entropy viene de MLE. GMM usa Gaussianas.

### Semana 8: Probabilidad Esencial

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Probabilidad básica | P(A), P(A\|B) | Ejercicios |
| M | Teorema de Bayes | Prior, Likelihood, Posterior | Implementar |
| X | Distribución Gaussiana | PDF, μ, σ | `gaussian_pdf()` |
| J | Gaussiana multivariada | Matriz de covarianza | Implementar |
| V | MLE (Maximum Likelihood) | Por qué da Cross-Entropy | Demostración |
| S | **Checkpoint Fase 1** | `probability.py` completo | Entregable |

**Entregable:** Librería `probability.py` con Gaussiana, MLE y softmax.

**Extensión Protocolo E:**

- **Simulacro PB-8 (90 min):** `study_tools/SIMULACRO_PERFORMANCE_BASED.md`
- **Cierre semanal (1 hora):** `study_tools/CIERRE_SEMANAL.md`
- **Evaluación con rúbrica:** aplicar scoring de PB-8 y hacer scoring completo al cierre del módulo (Semana 8)

---

# 🔷 FASE 2: NÚCLEO DE MACHINE LEARNING (Semanas 9-20)

*Objetivo: Implementar desde cero los algoritmos exactos de los 3 cursos del Pathway*

---

## 🗓️ SEMANA 9-12: Supervised Learning (Módulo 05)

**Materia:** Introduction to Machine Learning: Supervised Learning

### Semana 9: Regresión Lineal

Teorema de la semana (concepto-guía): mínimos cuadrados.

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Regresión: concepto | Línea de mejor ajuste | Visualizar datos |
| M | Mínimos cuadrados | Fórmula cerrada: $(X^TX)^{-1}X^Ty$ | Implementar |
| X | MSE como función de costo | $J(\theta) = \frac{1}{n}\sum(y - \hat{y})^2$ | Calcular MSE |
| J | GD para regresión | Derivar gradiente de MSE | Implementar |
| V | Regresión múltiple | Más de una feature | Extender código |
| S | **Repaso** | `linear_regression.py` v1 | Test |

### Semana 10: Regresión Logística

Teorema de la semana (concepto-guía): MLE → cross-entropy.

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Clasificación binaria | 0/1, sí/no | Dataset simple |
| M | Función sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | Implementar |
| X | Hipótesis logística | $h_\theta(x) = \sigma(\theta^T x)$ | Implementar |
| J | Cross-Entropy Loss | $-[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$ | Implementar |
| V | GD para logística | Derivar gradiente | Implementar |
| S | **Repaso** | `logistic_regression.py` | Test |

### Semana 11: Evaluación y Métricas

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Train/Test split | Por qué separar datos | Implementar split |
| M | Accuracy | Porcentaje correcto | Implementar |
| X | Precision y Recall | TP, FP, FN, TN | Implementar |
| J | F1-Score | Media armónica | Implementar |
| V | Matriz de confusión | Visualización | matplotlib |
| S | **Repaso** | `metrics.py` completo | Test |

### Semana 12: Validación Cruzada y Regularización

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Overfitting vs Underfitting | Bias-Variance tradeoff | Diagrama |
| M | K-Fold Cross Validation | Validación robusta | Implementar |
| X | **Tree-Based Models** | Entropía/Gini + Information Gain | Ejercicios |
| J | Árbol de Decisión (CART/ID3) | Implementación recursiva (sin gradientes) | Código |
| V | Ensembles (intro) | Bagging vs Boosting (Random Forest vs Gradient Boosting) | Comparar fronteras |
| S | **Checkpoint** | Supervisado completo | Entregable |

**Entregable:** `logistic_regression.py` desde cero usando NumPy para clasificar datos simples, con métricas y cross-validation **+** `scripts/decision_tree_from_scratch.py` (Árbol de Decisión simple desde cero).

**Extensión v5.0 – Dirty Data Check (Módulo 05):**
Para el dataset supervisado usado en regresión logística:
- Incluir variables categóricas (One-Hot Encoding manual).
- Incluir variables numéricas que requieran escalado (MinMax/Standard manual).
- Documentar al menos **5 decisiones clave** de limpieza y preprocesamiento en `study_tools/DIRTY_DATA_CHECK.md` (Caso 2).

**Evaluación (rúbrica):**

- Al checkpoint (Semana 12): ejecutar **rúbrica completa** (auto + IA/pareja) y registrar acciones correctivas.
- Objetivo de calibración: PB-8 y/o scoring global ≥ 75 para marcar checkpoint como sólido.

---

## 🗓️ SEMANA 13-16: Unsupervised Learning (Módulo 06)

**Materia:** Unsupervised Algorithms in Machine Learning

### Semana 13: K-Means Clustering

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Clustering: concepto | Agrupar sin etiquetas | Visualizar clusters |
| M | Algoritmo Lloyd (K-Means) | Asignar, actualizar, repetir | Pseudocódigo |
| X | Implementar K-Means | Versión básica | Código |
| J | K-Means++ inicialización | Mejor selección de centroides | Implementar |
| V | Criterio de parada | Convergencia | Implementar |
| S | **Repaso** | `kmeans.py` funcional | Test |

### Semana 14: Evaluación de Clusters

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Inercia (within-cluster) | Suma de distancias al centroide | Implementar |
| M | Método del codo | Elegir K óptimo | Visualizar |
| X | Silhouette Score | Calidad de clusters | Implementar |
| J | Limitaciones K-Means | Clusters no esféricos | Ejemplos |
| V | Generar datos sintéticos | make_blobs equivalente | Función propia |
| S | **Repaso** | Evaluación completa | Documento |

### Semana 15: PCA (Principal Component Analysis)

Teorema de la semana (concepto-guía): PCA como maximización de varianza.

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Reducción dimensional | Por qué reducir | Visualizar |
| M | PCA: concepto | Dirección de máxima varianza | Diagrama |
| X | PCA con eigenvalues | Autovectores de covarianza | np.linalg.eig() |
| J | PCA con SVD | Más estable numéricamente | np.linalg.svd() |
| V | Varianza explicada | Cuánta info se pierde | Implementar |
| S | **Repaso** | `pca.py` v1 | Test |

### Semana 16: PCA Aplicado y GMM

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Reconstrucción desde PCA | Proyectar y reconstruir | Implementar |
| M | Compresión de imágenes | PCA para reducir | Demo visual |
| X | GMM (concepto) | Mezcla de Gaussianas | Teoría |
| J | EM Algorithm (intro) | Expectation-Maximization | Pseudocódigo |
| V | Detección de anomalías | Outliers con GMM | Concepto |
| S | **Checkpoint** | No supervisado completo | Entregable |

**Entregable:** `kmeans.py` y `pca.py`. Usar PCA para comprimir una imagen y visualizar cuánta varianza se pierde con diferentes números de componentes.

**Extensión Protocolo E:**

- **Simulacro PB-16 (90 min):** `study_tools/SIMULACRO_PERFORMANCE_BASED.md`
- **Cierre semanal (1 hora):** `study_tools/CIERRE_SEMANAL.md`
- **Evaluación con rúbrica:** aplicar scoring de PB-16 y hacer scoring completo al cierre del módulo (Semana 16)

---

## 🗓️ SEMANA 17-20: Deep Learning + CNNs (Módulo 07)

**Materia:** Introduction to Deep Learning

### Semana 17: Perceptrón y Fundamentos

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Neurona artificial | Analogía biológica | Diagrama |
| M | Perceptrón simple | $y = \text{sign}(w \cdot x + b)$ | Implementar |
| X | Funciones de activación | Sigmoid, ReLU, Tanh | Implementar todas |
| J | Limitación del perceptrón | No puede resolver XOR | Demostrar |
| V | Necesidad de capas | Redes multicapa | Diagrama |
| S | **Repaso** | `activations.py` | Test |

### Semana 18: Forward Propagation

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | MLP: arquitectura | Capas ocultas | Diagrama |
| M | Forward pass | Propagación hacia adelante | Pseudocódigo |
| X | Implementar forward | Clase NeuralNetwork | Código |
| J | Función de pérdida DL | Cross-entropy para multiclase | Softmax |
| V | Inicialización de pesos | Xavier, He | Implementar |
| S | **Repaso** | Forward funcional | Test |

### Semana 19: CNNs - Teoría + Forward Pass (NumPy)

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | Por qué CNNs para imágenes | Problema de MLP con imágenes | Diagrama |
| M | Operación de convolución | Kernel, filtro, feature map | Demo visual |
| X | Stride, padding, pooling | Cálculo de dimensiones output | Ejercicios |
| J | **CNN Forward Pass (NumPy)** | Convolución + pooling (forward) | Código |
| V | Arquitectura tipo LeNet | Dimensiones end-to-end | Quiz de shapes |
| S | **Repaso** | Quiz de dimensiones CNN | Test teórico |

### Semana 20: Optimizadores y Entrenamiento

| Día | Mañana (Teoría) | Mediodía (Código) | Tarde (Práctica) |
|-----|-----------------|-------------------|------------------|
| L | SGD (Stochastic GD) | Mini-batches | Implementar |
| M | Momentum / Adam | Acelerar convergencia | Implementar |
| X | **Intro a PyTorch** | Tensores, `nn.Module`, `DataLoader` | Primer forward |
| J | **CNN Training con PyTorch** | Entrenar una CNN (sin backward manual) | Código |
| V | Comparación | CNN NumPy (forward) vs CNN PyTorch (training) | Notas |
| S | **Checkpoint** | MLP resuelve XOR | Entregable |

**Entregable:** `neural_net.py` - Una red neuronal que resuelve el problema XOR implementando `backward()` manualmente **+** `scripts/train_cnn_pytorch.py` (entrenamiento CNN con PyTorch).

**Evaluación (rúbrica):**

- Al checkpoint (Semana 20): ejecutar **rúbrica completa** (auto + IA/pareja) y registrar acciones correctivas.

---

# 🔷 FASE 3: PROYECTO FINAL "MNIST ANALYST" (Semanas 21-24)

*Objetivo: Un proyecto intensivo de 4 semanas que demuestra competencia en las 3 áreas*

**Dataset:** MNIST (dígitos, 28×28) / **Fashion-MNIST** (alternativo, mismo formato)

> 💡 **v3.3:** MNIST es un dataset simple (solo 10 clases, imágenes pequeñas). 4 semanas son suficientes.

---

## 🗓️ SEMANA 21: EDA + No Supervisado

**Materia demostrada:** Unsupervised Algorithms in Machine Learning

| Día | Actividad |
|-----|-----------|
| L | Cargar MNIST, entender estructura (784 dimensiones) |
| M | Implementar PCA desde cero, reducir a 2-3 componentes |
| X | Visualizar dígitos en gráfico 2D |
| J | Implementar K-Means, agrupar dígitos SIN etiquetas |
| V | Visualizar centroides como imágenes 28x28 |
| S | **Checkpoint:** Notebook PCA + K-Means |

**Entregable:** Jupyter notebook con PCA 2D y K-Means clustering.

---

## 🗓️ SEMANA 22: Clasificación Supervisada

**Materia demostrada:** Introduction to ML: Supervised Learning

| Día | Actividad |
|-----|-----------|
| L | Train/test split, normalización |
| M | Implementar Logistic Regression One-vs-All (10 clasificadores) |
| X | Entrenar y medir Accuracy global |
| J | Precision, Recall, F1, matriz de confusión |
| V | Visualizar errores (imágenes mal clasificadas) |
| S | **Checkpoint:** Logistic Regression completo |

**Entregable:** `logistic_mnist.py` con métricas completas.

---

## 🗓️ SEMANA 23: Deep Learning

**Materia demostrada:** Introduction to Deep Learning

| Día | Actividad |
|-----|-----------|
| L | Diseñar arquitectura MLP (784→128→64→10) |
| M | Implementar forward pass + softmax |
| X | Implementar backprop con cross-entropy |
| J | Training loop con mini-batches |
| V | Entrenar y ajustar hiperparámetros |
| S | **Checkpoint:** MLP funcional >90% accuracy |

**Entregable:** `neural_network_mnist.py` con backprop manual.

**Extensión v5.0 – Examen de Admisión Simulado:**
En las semanas 22 y 23, realizar los simulacros definidos en `study_tools/EXAMEN_ADMISION_SIMULADO.md` (2 horas, sin IDE ni internet, 40% pseudocódigo, 60% teoría). El simulacro de la semana 23 debe alcanzar ≥ 80/100 como métrica de "listo para admisión".

**Evaluación (rúbrica):**

- **Simulacro PB-23:** se considera cubierto por el examen final de `study_tools/EXAMEN_ADMISION_SIMULADO.md`.
- **Evaluación con rúbrica:** PB-23 ≥ 80/100 es requisito duro para marcar "Listo para admisión"

---

## 🗓️ SEMANA 24: Benchmark + Informe Final

**Objetivo:** Comparar modelos y documentar

| Día | Actividad |
|-----|-----------|
| L | Comparar rendimiento: Logistic vs MLP |
| M | Benchmark alternativo (recomendado): **Fashion-MNIST** |
| X | Dirty Data Check: generar dataset corrupto (`scripts/corrupt_mnist.py`) + limpieza |
| J | Deployment mínimo: guardar checkpoint + `scripts/predict.py` |
| V | Escribir MODEL_COMPARISON.md + crear README.md profesional (inglés) |
| S | **Entrega final + Autoevaluación** |

**Entregable Final:**

```
mnist-analyst/
├── src/
│   ├── data_loader.py
│   ├── linear_algebra.py
│   ├── probability.py
│   ├── pca.py
│   ├── kmeans.py
│   ├── logistic_regression.py
│   ├── neural_network.py
│   └── mnist_pipeline.py
├── notebooks/
│   ├── 01_eda_pca_kmeans.ipynb
│   ├── 02_logistic_classification.ipynb
│   └── 03_neural_network_benchmark.ipynb
├── docs/
│   └── MODEL_COMPARISON.md
├── tests/
│   └── test_*.py
└── README.md
```

---

## ✅ Checklist de Finalización - ML SPECIALIST v3.3

### Fase 1: Fundamentos (Módulos 01-04)
- [ ] Python + Pandas + NumPy dominado
- [ ] Álgebra lineal: normas, distancias, SVD, eigenvalues
- [ ] Cálculo: gradientes, chain rule, gradient descent
- [ ] Probabilidad: Bayes, Gaussiana, MLE, softmax

### Fase 2: ML Core (Módulos 05-07) ⭐ PATHWAY
- [ ] **Supervised (05):** Logistic Regression con métricas
- [ ] **Unsupervised (06):** K-Means y PCA desde cero
- [ ] **Deep Learning (07):** MLP con backprop + teoría CNNs

### Fase 3: Proyecto MNIST (Módulo 08)
- [ ] PCA reduce MNIST a 2D con visualización
- [ ] K-Means agrupa dígitos sin etiquetas
- [ ] Logistic Regression clasifica con >85% accuracy
- [ ] MLP supera a Logistic con >90% accuracy
- [ ] MODEL_COMPARISON.md explica matemáticamente las diferencias
- [ ] README.md profesional en inglés

### Verificación Final
- [ ] Puedo explicar matemáticamente por qué funciona cada algoritmo
- [ ] Puedo derivar las fórmulas de gradiente a mano
- [ ] Puedo implementar desde cero sin copiar código
- [ ] Entiendo convolución, stride, padding, pooling para CNNs
- [ ] Listo para los 3 cursos del Pathway Línea 1

---

## 📚 Recursos Recomendados

### Matemáticas
- [3Blue1Brown: Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown: Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

### Machine Learning
- [Stanford CS229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
- [Coursera: Machine Learning (Andrew Ng)](https://www.coursera.org/learn/machine-learning)

### Deep Learning
- [3Blue1Brown: Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)

---

> 💡 **Filosofía v3.3:** Esta guía te lleva de Python básico a candidato competitivo del MS in AI en exactamente 6 meses (24 semanas). Si puedes implementar PCA, K-Means, Logistic Regression y un MLP desde cero sobre MNIST, y entiendes la teoría de CNNs, **dominas la Línea 1 del Pathway**.
