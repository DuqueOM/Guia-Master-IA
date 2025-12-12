# 📖 Glosario Técnico - ML Specialist v3.3

> Definiciones A-Z de términos de Machine Learning usados en la guía.

---

## A

### Activation Function
**Definición:** Función no lineal aplicada a la salida de una neurona.
**Ejemplos:** ReLU, Sigmoid, Tanh, Softmax.
**Por qué:** Sin activaciones, una red sería solo transformaciones lineales.

### Adam
**Definición:** Adaptive Moment Estimation - optimizador que combina Momentum y RMSprop.
**Parámetros:** lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8
**Uso:** Default moderno para entrenar redes neuronales.

### Accuracy
**Definición:** Proporción de predicciones correctas.
**Fórmula:** (TP + TN) / (TP + TN + FP + FN)
**Limitación:** Engañoso con clases desbalanceadas.

---

## B

### Backpropagation
**Definición:** Algoritmo para calcular gradientes en redes neuronales usando la Chain Rule.
**Proceso:** Forward pass → calcular loss → backward pass → actualizar pesos.
**Base matemática:** ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w

### Batch Size
**Definición:** Número de muestras procesadas antes de actualizar pesos.
**Trade-off:** Grande = estable pero lento; pequeño = ruidoso pero rápido.
**Común:** 32, 64, 128, 256.

### Bias (parámetro)
**Definición:** Término constante en z = Wx + b que permite desplazar la función.
**Analogía:** El intercepto en una recta y = mx + b.

### Binary Cross-Entropy
**Definición:** Función de pérdida para clasificación binaria.
**Fórmula:** L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
**Uso:** Salida sigmoid, predicción de probabilidad.

### Broadcasting
**Definición:** Expansión automática de arrays para operaciones elemento a elemento.
**Ejemplo:** array(3,1) + array(1,4) → array(3,4)
**Regla:** Dimensiones deben ser iguales o una debe ser 1.

---

## C

### Centroid
**Definición:** Punto central de un cluster (promedio de sus puntos).
**En K-Means:** Se actualiza iterativamente hasta convergencia.

### Chain Rule
**Definición:** Regla para derivar funciones compuestas.
**Fórmula:** d/dx f(g(x)) = f'(g(x)) · g'(x)
**Importancia:** Base matemática de Backpropagation.

### Classification
**Definición:** Tarea de predecir una categoría discreta.
**Binaria:** 2 clases (spam/no spam).
**Multiclase:** >2 clases (dígitos 0-9).

### Clustering
**Definición:** Agrupar puntos similares sin etiquetas supervisadas.
**Algoritmos:** K-Means, DBSCAN, Hierarchical.

### Confusion Matrix
**Definición:** Tabla que muestra predicciones vs valores reales.
**Componentes:** TP, TN, FP, FN.

### Convergence
**Definición:** Cuando el algoritmo deja de mejorar significativamente.
**Criterio:** Cambio en loss < tolerancia, o gradiente ≈ 0.

### Cosine Similarity
**Definición:** Similitud basada en el ángulo entre vectores.
**Fórmula:** cos(θ) = (a·b) / (||a|| ||b||)
**Rango:** [-1, 1], donde 1 = idénticos.

### Cross-Validation
**Definición:** Técnica para evaluar modelo dividiendo datos en K folds.
**K-Fold:** Entrenar K veces, cada vez con diferente fold como validación.
**Uso:** Estimar rendimiento real, evitar overfitting.

---

## D

### Deep Learning
**Definición:** ML con redes neuronales de múltiples capas ocultas.
**Ventaja:** Aprende features automáticamente.
**Requisito:** Muchos datos y compute.

### Derivative
**Definición:** Tasa de cambio instantánea de una función.
**Notación:** f'(x), df/dx, ∂f/∂x (parcial).

### Dimensionality Reduction
**Definición:** Reducir número de features preservando información.
**Métodos:** PCA, t-SNE, UMAP.
**Uso:** Visualización, eliminar ruido, acelerar entrenamiento.

### Dot Product
**Definición:** Suma de productos elemento a elemento.
**Fórmula:** a·b = Σ aᵢbᵢ
**Uso:** Similitud, proyecciones, capas de red neuronal.

---

## E

### Eigenvalue / Eigenvector
**Definición:** Para matriz A, Av = λv donde v es eigenvector y λ es eigenvalue.
**Interpretación:** Direcciones principales de la transformación.
**Uso en ML:** PCA usa eigenvectors de la matriz de covarianza.

### Epoch
**Definición:** Una pasada completa por todo el dataset de entrenamiento.
**Típico:** 10-100 epochs dependiendo del problema.

### Euclidean Distance
**Definición:** Distancia en línea recta entre dos puntos.
**Fórmula:** d(a,b) = √Σ(aᵢ - bᵢ)²
**Uso:** K-Means, KNN.

---

## F

### F1 Score
**Definición:** Media armónica de Precision y Recall.
**Fórmula:** F1 = 2 · (P · R) / (P + R)
**Uso:** Balance entre precision y recall.

### Feature
**Definición:** Variable de entrada (columna) en un dataset.
**Ejemplo:** En MNIST, cada píxel es un feature (784 total).

### Forward Pass
**Definición:** Propagación de input a través de la red para obtener output.
**Cálculo:** z = Wx + b, a = activation(z), repetir por capa.

---

## G

### Gradient
**Definición:** Vector de derivadas parciales.
**Notación:** ∇f = [∂f/∂x₁, ∂f/∂x₂, ...]
**Propiedad:** Apunta en dirección de máximo ascenso.

### Gradient Descent
**Definición:** Algoritmo de optimización que sigue el gradiente negativo.
**Update:** θ = θ - α · ∇L(θ)
**Variantes:** Batch, Mini-batch, Stochastic (SGD).

---

## H

### Hidden Layer
**Definición:** Capa entre input y output en una red neuronal.
**Función:** Aprende representaciones intermedias.

### Hyperparameter
**Definición:** Parámetro configurado antes del entrenamiento (no aprendido).
**Ejemplos:** Learning rate, número de capas, batch size.

---

## I

### Inertia
**Definición:** Suma de distancias cuadradas de puntos a sus centroides.
**En K-Means:** Métrica a minimizar.
**Uso:** Método del codo para elegir K.

---

## K

### K-Means
**Definición:** Algoritmo de clustering que particiona en K grupos.
**Pasos:** 1) Inicializar centroides 2) Asignar puntos 3) Actualizar centroides 4) Repetir.
**Complejidad:** O(n · k · i · d) donde i=iteraciones, d=dimensiones.

### K-Means++
**Definición:** Inicialización inteligente para K-Means.
**Método:** Elegir centroides iniciales lejos entre sí.
**Ventaja:** Mejor convergencia, evita mínimos locales.

---

## L

### L1 Norm (Manhattan)
**Definición:** Suma de valores absolutos.
**Fórmula:** ||x||₁ = Σ|xᵢ|
**Uso:** Regularización Lasso, promueve sparsity.

### L2 Norm (Euclidean)
**Definición:** Raíz de suma de cuadrados (longitud del vector).
**Fórmula:** ||x||₂ = √Σxᵢ²
**Uso:** Regularización Ridge, normalización.

### Learning Rate
**Definición:** Tamaño del paso en Gradient Descent.
**Símbolo:** α (alpha) o lr.
**Trade-off:** Grande = rápido pero inestable; pequeño = estable pero lento.

### Linear Regression
**Definición:** Modelo que predice valor continuo con combinación lineal.
**Fórmula:** ŷ = Xθ
**Loss:** MSE (Mean Squared Error).

### Logistic Regression
**Definición:** Modelo de clasificación binaria usando sigmoid.
**Fórmula:** P(y=1) = σ(Xθ)
**Loss:** Binary Cross-Entropy.

### Loss Function
**Definición:** Función que mide error entre predicción y valor real.
**Ejemplos:** MSE (regresión), Cross-Entropy (clasificación).
**Objetivo:** Minimizar durante entrenamiento.

---

## M

### Matrix Multiplication
**Definición:** Operación (m×n) @ (n×p) → (m×p).
**Elemento:** C[i,j] = Σₖ A[i,k] · B[k,j]
**Uso:** Transformaciones lineales, capas de red.

### Mini-batch
**Definición:** Subconjunto de datos usado en una iteración de SGD.
**Ventaja:** Balance entre eficiencia y estabilidad.

### MLP (Multilayer Perceptron)
**Definición:** Red neuronal fully-connected con capas ocultas.
**Arquitectura:** Input → Hidden(s) → Output.

### MNIST
**Definición:** Dataset de dígitos escritos a mano (28×28 píxeles).
**Tamaño:** 60k train, 10k test.
**Uso:** Benchmark clásico de clasificación de imágenes.

### MSE (Mean Squared Error)
**Definición:** Promedio de errores al cuadrado.
**Fórmula:** MSE = (1/n) Σ(y - ŷ)²
**Uso:** Loss para regresión.

### Momentum
**Definición:** Técnica que acelera SGD acumulando gradientes pasados.
**Fórmula:** v = β·v + (1-β)·∇L; θ = θ - α·v
**Ventaja:** Escapa mínimos locales, reduce oscilaciones.

---

## N

### Normalization
**Definición:** Escalar datos a un rango estándar.
**Min-Max:** x' = (x - min) / (max - min) → [0, 1]
**Z-score:** x' = (x - μ) / σ → media 0, std 1.

### NumPy
**Definición:** Librería de Python para computación numérica eficiente.
**Ventaja:** Operaciones vectorizadas (evita loops).
**Objeto principal:** ndarray (n-dimensional array).

---

## O

### One-Hot Encoding
**Definición:** Representar categoría como vector binario.
**Ejemplo:** clase 3 de 5 → [0, 0, 0, 1, 0]
**Uso:** Labels para clasificación multiclase.

### Overfitting
**Definición:** Modelo que memoriza training data pero no generaliza.
**Síntoma:** Train loss bajo, test loss alto.
**Soluciones:** Más datos, regularización, dropout, early stopping.

---

## P

### Partial Derivative
**Definición:** Derivada respecto a una variable, tratando otras como constantes.
**Notación:** ∂f/∂x
**Uso:** Calcular gradientes en funciones multivariable.

### PCA (Principal Component Analysis)
**Definición:** Reducción dimensional que preserva máxima varianza.
**Método:** Proyectar datos en eigenvectors principales.
**Output:** Componentes principales ordenados por varianza explicada.

### Precision
**Definición:** De los predichos positivos, ¿cuántos son correctos?
**Fórmula:** TP / (TP + FP)
**Importancia:** Cuando FP es costoso.

### Projection
**Definición:** Mapear un punto a un subespacio (línea, plano).
**En PCA:** Proyectar datos al espacio de componentes principales.

---

## R

### Recall
**Definición:** De los positivos reales, ¿cuántos capturé?
**Fórmula:** TP / (TP + FN)
**Importancia:** Cuando FN es costoso.

### Regression
**Definición:** Predecir un valor continuo.
**Ejemplos:** Precio de casa, temperatura.

### Regularization
**Definición:** Técnica para prevenir overfitting penalizando complejidad.
**L1 (Lasso):** Añade λ·||θ||₁ al loss.
**L2 (Ridge):** Añade λ·||θ||₂² al loss.

### ReLU (Rectified Linear Unit)
**Definición:** f(x) = max(0, x)
**Derivada:** 1 si x > 0, 0 si x ≤ 0.
**Ventaja:** Simple, evita vanishing gradient.

---

## S

### SGD (Stochastic Gradient Descent)
**Definición:** Gradient descent con una muestra (o mini-batch) por update.
**Ventaja:** Más rápido, escapa mínimos locales.
**Desventaja:** Updates ruidosos.

### Sigmoid
**Definición:** σ(x) = 1 / (1 + e⁻ˣ)
**Rango:** (0, 1)
**Uso:** Clasificación binaria, probabilidades.
**Derivada:** σ(x) · (1 - σ(x))

### Silhouette Score
**Definición:** Métrica de calidad de clustering.
**Rango:** [-1, 1], mayor es mejor.
**Cálculo:** Basado en cohesión intra-cluster y separación inter-cluster.

### Softmax
**Definición:** Convierte vector a distribución de probabilidad.
**Fórmula:** softmax(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ
**Uso:** Capa de salida para clasificación multiclase.

### Supervised Learning
**Definición:** Aprender de datos con etiquetas (X, y).
**Tareas:** Clasificación, Regresión.

### SVD (Singular Value Decomposition)
**Definición:** Factorización A = UΣVᵀ.
**Uso:** PCA (más estable), compresión, sistemas de recomendación.

---

## T

### Tanh
**Definición:** Tangente hiperbólica, similar a sigmoid pero centrada en 0.
**Rango:** (-1, 1)
**Derivada:** 1 - tanh²(x)

### Test Set
**Definición:** Datos reservados para evaluación final del modelo.
**Regla:** NUNCA usar para entrenar o seleccionar hiperparámetros.

### Training Set
**Definición:** Datos usados para entrenar el modelo.
**Típico:** 70-80% del dataset total.

### Transpose
**Definición:** Intercambiar filas y columnas de una matriz.
**Notación:** Aᵀ
**Propiedad:** (AB)ᵀ = BᵀAᵀ

---

## U

### Underfitting
**Definición:** Modelo demasiado simple que no captura patrones.
**Síntoma:** Train loss alto, test loss alto.
**Soluciones:** Modelo más complejo, más features, más entrenamiento.

### Unsupervised Learning
**Definición:** Aprender de datos sin etiquetas.
**Tareas:** Clustering, reducción dimensional, detección de anomalías.

---

## V

### Validation Set
**Definición:** Datos para ajustar hiperparámetros y detectar overfitting.
**Típico:** 10-20% del training data.

### Variance (estadística)
**Definición:** Medida de dispersión de los datos.
**Fórmula:** Var(X) = E[(X - μ)²]

### Variance (ML)
**Definición:** Error por sensibilidad a fluctuaciones en training data.
**Alta varianza:** Overfitting.

### Vectorization
**Definición:** Reemplazar loops por operaciones de arrays.
**Ventaja:** 10-100x más rápido con NumPy.
**Ejemplo:** `np.dot(a, b)` en lugar de `sum(a[i]*b[i] for i in range(n))`

---

## W

### Weight
**Definición:** Parámetro aprendido que determina importancia de input.
**En redes:** Matriz W en z = Wx + b.

---

## X

### Xavier Initialization
**Definición:** Inicializar pesos con varianza 1/n_inputs.
**Fórmula:** W ~ N(0, 1/n_in) o U(-√(1/n_in), √(1/n_in))
**Uso:** Capas con tanh/sigmoid.

### XOR Problem
**Definición:** Problema no linealmente separable clásico.
**Importancia:** Demuestra necesidad de capas ocultas en redes neuronales.
**Solución:** MLP con al menos una capa oculta.
