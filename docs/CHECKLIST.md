# ☑️ Checklist Final - ML Specialist v3.3

> Verificación completa del programa de **24 semanas** con validación matemática rigurosa.

---

## 📏 Rúbrica (evaluación formal)

- `study_tools/RUBRICA_v1.md`
- `rubrica.csv`

Uso recomendado:

- **Cierre semanal:** scoring rápido (auto).
- **Cierres de módulo (Semanas 12, 16, 20):** scoring completo (auto + IA/pareja).
- **PB-8 / PB-16 / PB-23:** scoring de simulacro + post-mortem.

## 📚 Fase 1: Fundamentos (Semanas 1-8)

### Módulo 01: Python + Pandas + NumPy (Semanas 1-2)

#### Conocimiento
- [ ] Pandas: cargar CSV con `read_csv()`
- [ ] Pandas: limpiar datos con `dropna()`, `fillna()`
- [ ] Pandas: selección con `.loc[]`, `.iloc[]`
- [ ] Pandas → NumPy: `.to_numpy()`
- [ ] NumPy: creación de arrays (1D, 2D, 3D)
- [ ] NumPy: indexing, slicing, broadcasting
- [ ] NumPy: agregaciones por eje (axis=0, axis=1)
- [ ] **Conozco los 5 errores comunes de NumPy y sus soluciones**

#### Estándares Profesionales (v3.2)
- [ ] `mypy src/` pasa sin errores
- [ ] `ruff check src/` pasa sin errores
- [ ] Al menos 3 tests con `pytest` pasando

#### Metodología Feynman
- [ ] Puedo explicar broadcasting en 5 líneas sin jerga

### Módulo 02: Álgebra Lineal para ML (Semanas 3-5)
- [ ] Producto punto y significado geométrico
- [ ] Normas L1, L2, L∞ implementadas
- [ ] Distancia euclidiana y similitud coseno
- [ ] Multiplicación de matrices con `@`
- [ ] Eigenvalues/eigenvectors con `np.linalg.eig()`
- [ ] SVD con `np.linalg.svd()`
- [ ] `linear_algebra.py` con tests pasando

### Módulo 03: Cálculo Multivariante (Semanas 6-7)
- [ ] Derivadas parciales calculadas
- [ ] Gradiente de funciones multivariables
- [ ] Gradient Descent implementado desde cero
- [ ] Efecto del learning rate entendido
- [ ] Chain Rule aplicada a funciones compuestas
- [ ] `calculus.py` con Gradient Descent funcional

#### Gradient Checking (v3.3 - Obligatorio)
- [ ] **`grad_check.py` implementado**
- [ ] **Validé derivadas de MSE, sigmoid y capa lineal**
- [ ] Error relativo < 10⁻⁷ en todos los tests

### Módulo 04: Probabilidad para ML (Semana 8)
- [ ] Teorema de Bayes explicado con ejemplo
- [ ] Gaussiana univariada: PDF implementada
- [ ] Gaussiana multivariada: concepto entendido
- [ ] MLE: conexión con Cross-Entropy explicada
- [ ] **Softmax con Log-Sum-Exp trick implementado (v3.3)**
- [ ] `probability.py` con tests pasando

#### Evaluación (PB-8)
- [ ] **PB-8 ≥ 75/100** y evaluado con la rúbrica
- [ ] Post-mortem: 3 fallos registrados en `study_tools/DIARIO_ERRORES.md`

---

## 🤖 Fase 2: Núcleo de ML (Semanas 9-20) ⭐ PATHWAY

### Módulo 05: Supervised Learning (Semanas 9-12)

#### Conocimiento
- [ ] Regresión lineal (Normal Equation + GD)
- [ ] MSE y su gradiente derivado
- [ ] Regresión logística desde cero
- [ ] Sigmoid y binary cross-entropy
- [ ] Matriz de confusión (TP, TN, FP, FN)
- [ ] Accuracy, Precision, Recall, F1 implementados
- [ ] Train/test split manual
- [ ] K-fold cross validation
- [ ] Regularización L2 (Ridge)

#### Derivación Analítica (v3.2 - Obligatorio)
- [ ] **Derivé el gradiente de Cross-Entropy a mano**
- [ ] **Documento con derivación completa (Markdown o LaTeX)**

#### Metodología Feynman
- [ ] Puedo explicar sigmoid vs softmax en 5 líneas

### Módulo 06: Unsupervised Learning (Semanas 13-16)
- [ ] K-Means con K-Means++ initialization
- [ ] Algoritmo de Lloyd (asignar-actualizar-repetir)
- [ ] Inercia y método del codo
- [ ] PCA usando SVD (`np.linalg.svd()`)
- [ ] Varianza explicada y elección de n_components
- [ ] Reconstrucción desde componentes principales
- [ ] `kmeans.py` y `pca.py` con tests pasando

#### Evaluación (PB-16)
- [ ] **PB-16 ≥ 75/100** y evaluado con la rúbrica
- [ ] Cierre de módulo: rúbrica completa aplicada (auto + IA/pareja)

### Módulo 07: Deep Learning + CNNs (Semanas 17-20)

#### Conocimiento
- [ ] Neurona artificial y perceptrón
- [ ] Sigmoid, ReLU, tanh, softmax + derivadas
- [ ] Problema XOR y su no-linealidad
- [ ] Forward pass para MLP
- [ ] Backpropagation con Chain Rule
- [ ] SGD, Momentum, Adam implementados
- [ ] Red resuelve problema XOR
- [ ] **CNNs (teoría):** convolución, stride, padding, pooling

#### Derivación Analítica (v3.2 - Obligatorio)
- [ ] **Derivé las ecuaciones de backprop para red de 2 capas**
- [ ] **Diagrama de grafo computacional**

#### Metodología Feynman
- [ ] Puedo explicar backpropagation en 5 líneas sin jerga

#### Cierre de módulo (Semana 20)
- [ ] Rúbrica completa aplicada (auto + IA/pareja)

---

## 🎯 Fase 3: Proyecto MNIST Analyst (Semanas 21-24)

### Semana 21: EDA + No Supervisado
- [ ] MNIST cargado y normalizado
- [ ] PCA reduce a 2D con visualización
- [ ] Varianza explicada analizada
- [ ] K-Means agrupa dígitos sin etiquetas
- [ ] Centroides visualizados como imágenes 28x28

### Semana 22: Clasificación Supervisada
- [ ] Logistic Regression One-vs-All implementado
- [ ] Accuracy > 85% en test set
- [ ] Precision, Recall, F1 por clase
- [ ] Matriz de confusión analizada
- [ ] Errores visualizados (imágenes mal clasificadas)

### Semana 23: Deep Learning
- [ ] MLP 784→128→64→10 implementado
- [ ] Forward y backward pass funcionales
- [ ] Mini-batch SGD funcionando
- [ ] Accuracy > 90% en test set

#### Evaluación (PB-23 / Examen de admisión simulado)
- [ ] **PB-23 ≥ 80/100** (requisito duro) y evaluado con la rúbrica

### Semana 24: Benchmark + Informe
- [ ] Comparación MLP vs Logistic Regression
- [ ] `MODEL_COMPARISON.md` explicando diferencias
- [ ] `README.md` profesional en inglés
- [ ] Demo notebook completo

### Requisitos v3.2 (Obligatorios)
- [ ] **Análisis Bias-Variance** con experimento práctico (3 tamaños de MLP)
- [ ] **Notebook en formato Paper** (Abstract, Methods, Results, Discussion)
- [ ] `mypy src/` pasa sin errores en todo el proyecto
- [ ] `pytest tests/` con cobertura significativa

### Metodología Feynman
- [ ] Puedo explicar Bias vs Variance en 5 líneas
- [ ] Puedo explicar por qué MLP supera a Logistic en 5 líneas

---

## 💻 Código

### Estructura del Proyecto MNIST
```
mnist-analyst/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── linear_algebra.py
│   ├── probability.py
│   ├── pca.py
│   ├── kmeans.py
│   ├── logistic_regression.py
│   ├── neural_network.py
│   ├── metrics.py
│   └── pipeline.py
├── notebooks/
│   ├── 01_eda_pca_kmeans.ipynb
│   ├── 02_logistic_classification.ipynb
│   └── 03_neural_network_benchmark.ipynb
├── tests/
│   └── test_*.py
├── docs/
│   └── MODEL_COMPARISON.md
├── README.md
└── requirements.txt
```

### Calidad de Código
- [ ] Type hints en todas las funciones
- [ ] Docstrings con Args, Returns
- [ ] `mypy src/` pasa sin errores
- [ ] Código vectorizado (sin loops innecesarios)

### Tests
- [ ] Tests unitarios para cada módulo
- [ ] Tests para edge cases
- [ ] Todos los tests pasan

---

## 📝 Documentación

### README.md del Proyecto
- [ ] Descripción del proyecto
- [ ] Instrucciones de instalación
- [ ] Ejemplo de uso
- [ ] Resultados y métricas
- [ ] Escrito en inglés

### MODEL_COMPARISON.md
- [ ] Tabla comparativa de modelos
- [ ] Explicación matemática de diferencias
- [ ] Análisis de PCA
- [ ] Análisis de K-Means
- [ ] Conclusiones

---

## 🚀 Verificación Final

```bash
# 1. Tests
python -m pytest tests/ -v

# 2. Pipeline completo
python -c "
from src.pipeline import run_mnist_pipeline
# Ejecutar pipeline demo
"

# 3. Verificar accuracy
# Logistic Regression: > 85%
# Neural Network: > 90%
```

---

## ✅ Declaración de Completitud

### Por Fase

- [ ] **Fase 1:** Fundamentos matemáticos dominados
- [ ] **Fase 2:** Algoritmos ML implementados desde cero
- [ ] **Fase 3:** Proyecto MNIST completo

### Por Curso del Pathway

- [ ] **Supervised Learning:** Regresión + Clasificación
- [ ] **Unsupervised Learning:** K-Means + PCA
- [ ] **Deep Learning:** MLP con Backpropagation

### Métricas Finales

| Métrica | Objetivo | Logrado |
|---------|----------|---------|
| Logistic Regression Accuracy | >85% | ___% |
| Neural Network Accuracy | >90% | ___% |
| Módulos completados | 8/8 | ___/8 |
| Tests pasando | 100% | ___% |

**Fecha de completitud:** _______________

**Listo para el MS in AI Pathway - Línea 1:** ☐ Sí ☐ No
