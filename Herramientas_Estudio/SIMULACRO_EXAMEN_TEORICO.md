# 📝 Simulacro de Examen Teórico - Sábados

> El Pathway de CU Boulder tiene exámenes teóricos rigurosos, no solo código.
> Este documento entrena la resolución de problemas con lápiz y papel bajo presión de tiempo.

---

## 📋 Protocolo del Simulacro

### Reglas Estrictas
- ⏱️ **1 hora máximo** por simulacro
- 📵 **Sin IDE, sin internet**
- 📝 **Solo lápiz y papel**
- 🧮 **Calculadora básica permitida** (no científica)

### Formato
- 5-7 preguntas por simulacro
- Mezcla de cálculo, álgebra lineal, probabilidad y conceptos ML
- Puntuación: 100 puntos total

---

## 📚 Banco de Preguntas por Fase

---

### FASE 1: Fundamentos Matemáticos (Semanas 1-8)

#### Simulacro 1A: NumPy y Álgebra Lineal Básica

**Pregunta 1 (15 pts)** - Operaciones con Matrices

Dadas las matrices:
```
A = [[1, 2],    B = [[5, 6],
     [3, 4]]         [7, 8]]
```

Calcula a mano:
a) A + B
b) A @ B (producto matricial)
c) A * B (Hadamard product)
d) A.T (transpuesta de A)

---

**Pregunta 2 (20 pts)** - Dimensiones y Broadcasting

Sin usar código, determina el shape resultante o indica si hay error:

a) `(3, 4) @ (4, 5)` = ?
b) `(3, 4) + (4,)` = ?
c) `(3, 4) @ (3, 4)` = ?
d) `(2, 3, 4) * (3, 1)` = ?
e) `np.sum((5, 4, 3), axis=1, keepdims=True)` = ?

---

**Pregunta 3 (15 pts)** - Determinantes e Inversas

Para la matriz:
```
A = [[2, 1],
     [5, 3]]
```

a) Calcula det(A)
b) Calcula A⁻¹
c) Verifica que A @ A⁻¹ = I

---

**Pregunta 4 (20 pts)** - Eigenvalores

Para la matriz:
```
A = [[4, 2],
     [1, 3]]
```

a) Plantea la ecuación característica det(A - λI) = 0
b) Encuentra los eigenvalores
c) Para cada eigenvalor, encuentra el eigenvector correspondiente

---

**Pregunta 5 (15 pts)** - Conceptual

Responde brevemente:

a) ¿Por qué es importante que una matriz sea invertible en regresión lineal?
b) ¿Qué significa geométricamente que el determinante sea cero?
c) ¿Cuál es la diferencia entre norma L1 y L2? ¿Cuándo usar cada una?

---

**Pregunta 6 (15 pts)** - Aplicación

Tienes un sistema de ecuaciones:
```
2x + 3y = 8
4x + 5y = 14
```

a) Escríbelo en forma matricial Ax = b
b) Resuelve usando la inversa de A

---

#### Simulacro 1B: Cálculo Multivariante

**Pregunta 1 (20 pts)** - Gradientes

Para la función f(x, y) = x²y + 3xy² - 2x + 5

a) Calcula ∂f/∂x
b) Calcula ∂f/∂y
c) Evalúa ∇f en el punto (1, 2)
d) ¿En qué dirección crece más rápido f en ese punto?

---

**Pregunta 2 (20 pts)** - Regla de la Cadena

Sea z = f(u, v) donde u = x² + y y v = xy

Si f(u, v) = u²v, calcula ∂z/∂x y ∂z/∂y

---

**Pregunta 3 (20 pts)** - Optimización

Para f(x, y) = x² + y² - 2x - 4y + 5

a) Encuentra los puntos críticos (∇f = 0)
b) Calcula la matriz Hessiana
c) Determina si el punto crítico es mínimo, máximo o punto silla

---

**Pregunta 4 (20 pts)** - Gradient Descent

Tienes f(x) = x² - 4x + 4

a) Calcula f'(x)
b) Si empiezas en x₀ = 0 con learning rate α = 0.1, calcula x₁, x₂, x₃
c) ¿Hacia qué valor converge x?
d) Si α = 2, ¿qué pasa? Explica.

---

**Pregunta 5 (20 pts)** - Conceptual

a) ¿Por qué el gradiente apunta en la dirección de máximo crecimiento?
b) ¿Qué representa geométricamente la Hessiana?
c) ¿Qué es un punto silla y por qué es problemático en optimización?
d) Dibuja una superficie con un mínimo local que no sea global

---

### FASE 2: Probabilidad y Estadística (Semanas 9-12)

#### Simulacro 2A: Probabilidad

**Pregunta 1 (20 pts)** - Bayes

Un test médico tiene:
- Sensibilidad (true positive rate): 95%
- Especificidad (true negative rate): 90%
- Prevalencia de la enfermedad: 1%

Si una persona da positivo, ¿cuál es la probabilidad de que realmente tenga la enfermedad?

---

**Pregunta 2 (20 pts)** - Distribuciones

a) Si X ~ N(5, 4), ¿cuál es P(X > 7)?
b) Si X ~ Bernoulli(0.3), ¿cuál es E[X] y Var[X]?
c) Si X₁, X₂, ..., X₁₀₀ son i.i.d. con E[Xᵢ] = 10, ¿cuál es E[X̄]?

---

**Pregunta 3 (20 pts)** - MLE

Tienes datos: [2, 4, 6, 8, 10]

Asumiendo que vienen de una distribución normal N(μ, σ²):
a) Escribe la función de verosimilitud L(μ, σ²)
b) Deriva los estimadores MLE para μ y σ²
c) Calcula los valores numéricos

---

**Pregunta 4 (20 pts)** - Conceptual

a) ¿Cuál es la diferencia entre probabilidad frecuentista y bayesiana?
b) ¿Por qué usamos log-likelihood en lugar de likelihood?
c) ¿Qué dice el Teorema Central del Límite y por qué es importante en ML?

---

**Pregunta 5 (20 pts)** - Aplicación ML

En clasificación binaria:
a) Define Precision y Recall
b) Si tienes 100 muestras: 80 TN, 10 TP, 5 FP, 5 FN
   - Calcula Accuracy
   - Calcula Precision
   - Calcula Recall
   - Calcula F1-score
c) ¿Cuándo es mejor optimizar Recall que Precision?

---

### FASE 3: Machine Learning (Semanas 13-18)

#### Simulacro 3A: Supervised Learning

**Pregunta 1 (25 pts)** - Regresión Lineal

Tienes datos:
| x | y |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |
| 4 | 4 |
| 5 | 5 |

a) Calcula los coeficientes de regresión lineal y = β₀ + β₁x usando las fórmulas:
   - β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
   - β₀ = ȳ - β₁x̄

b) ¿Cuál es la predicción para x = 6?

---

**Pregunta 2 (25 pts)** - Regularización

a) Escribe la función de costo para Ridge Regression
b) Escribe la función de costo para Lasso Regression
c) ¿Cuál de las dos produce "sparsity" (coeficientes exactamente cero)? ¿Por qué?
d) Si λ → ∞, ¿qué pasa con los coeficientes en cada caso?

---

**Pregunta 3 (25 pts)** - SVM

a) ¿Qué es el margen en SVM y por qué queremos maximizarlo?
b) Escribe la formulación del problema de optimización para SVM lineal
c) ¿Qué son los vectores de soporte?
d) ¿Cómo permite el "kernel trick" clasificar datos no linealmente separables?

---

**Pregunta 4 (25 pts)** - Conceptual

a) Explica el trade-off bias-variance
b) ¿Qué es overfitting? ¿Cómo lo detectas? ¿Cómo lo previenes?
c) ¿Por qué necesitamos un conjunto de validación además de train y test?
d) Si tu modelo tiene alto bias, ¿qué harías? ¿Y si tiene alta varianza?

---

### FASE 4: Deep Learning (Semanas 19-24)

#### Simulacro 4A: Redes Neuronales

**Pregunta 1 (25 pts)** - Forward Pass

Red neuronal simple:
- Input: x = [1, 2] (1x2)
- W₁ = [[0.5, -0.5], [0.3, 0.7]] (2x2)
- b₁ = [0.1, 0.2] (1x2)
- Activación: ReLU
- W₂ = [[0.4], [0.6]] (2x1)
- b₂ = [0.1] (1x1)

Calcula paso a paso:
a) z₁ = xW₁ + b₁
b) a₁ = ReLU(z₁)
c) z₂ = a₁W₂ + b₂
d) ŷ = z₂

---

**Pregunta 2 (25 pts)** - Backpropagation

Continuando del ejercicio anterior:
- y_true = 1
- Loss = (ŷ - y)²

a) Calcula ∂L/∂ŷ
b) Calcula ∂L/∂W₂
c) Calcula ∂L/∂a₁
d) Explica cómo calcularías ∂L/∂W₁ (no necesitas el valor numérico)

---

**Pregunta 3 (25 pts)** - Conceptual

a) ¿Por qué necesitamos funciones de activación no lineales?
b) ¿Qué es el problema del vanishing gradient y cómo lo resuelve ReLU?
c) ¿Cuál es la diferencia entre SGD, Momentum y Adam?
d) ¿Por qué usamos mini-batches en lugar de todo el dataset?

---

**Pregunta 4 (25 pts)** - Diseño

a) Si duplico el learning rate en una superficie convexa, ¿qué pasa con la convergencia?
b) ¿Cómo elegirías la arquitectura (número de capas, neuronas) para un problema nuevo?
c) ¿Qué es Dropout y por qué funciona como regularización?
d) Dibuja un grafo computacional para: L = (wx + b - y)²

---

## 📊 Plantilla de Puntuación

| Simulacro | Fecha | Puntuación | Tiempo | Temas Débiles |
|-----------|-------|------------|--------|---------------|
| 1A | | /100 | min | |
| 1B | | /100 | min | |
| 2A | | /100 | min | |
| 3A | | /100 | min | |
| 4A | | /100 | min | |

---

## 🎯 Criterios de Aprobación

- **< 60 puntos**: Revisar el tema a fondo
- **60-75 puntos**: Competente, seguir practicando
- **75-90 puntos**: Buen nivel, listo para examen real
- **> 90 puntos**: Excelente, avanzar al siguiente tema

---

## 📅 Calendario de Simulacros

| Semana | Simulacro Recomendado |
|--------|----------------------|
| 4 | 1A: Álgebra Lineal |
| 8 | 1B: Cálculo |
| 12 | 2A: Probabilidad |
| 16 | 3A: ML Supervisado |
| 22 | 4A: Deep Learning |
| 24 | Simulacro Final (todos los temas) |
