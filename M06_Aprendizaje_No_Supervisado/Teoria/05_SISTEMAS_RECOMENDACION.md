# Módulo 6.5: Sistemas de Recomendación

> **Semana:** 15 | **Curso Alineado:** CSCA 5632 - Unsupervised Learning
> **Prerequisitos:** SVD, PCA, Álgebra Lineal

---

## 🎯 Objetivos de Aprendizaje

Al finalizar este módulo serás capaz de:

1. **Distinguir** entre filtrado colaborativo y basado en contenido
2. **Implementar** factorización de matrices con SVD para recomendaciones
3. **Construir** un recomendador funcional con el dataset MovieLens
4. **Evaluar** sistemas de recomendación con métricas apropiadas
5. **Comprender** el problema del cold-start y estrategias de mitigación

---

## 📚 Tabla de Contenidos

1. [Introducción a Sistemas de Recomendación](#1-introducción)
2. [Taxonomía de Métodos](#2-taxonomía-de-métodos)
3. [Filtrado Colaborativo](#3-filtrado-colaborativo)
4. [Factorización de Matrices](#4-factorización-de-matrices)
5. [SVD para Recomendaciones](#5-svd-para-recomendaciones)
6. [Implementación Práctica: MovieLens](#6-implementación-práctica)
7. [Métricas de Evaluación](#7-métricas-de-evaluación)
8. [Problemas y Soluciones](#8-problemas-y-soluciones)
9. [Ejercicios](#9-ejercicios)

---

## 1. Introducción a Sistemas de Recomendación

### 1.1 Motivación: El Problema de la Sobrecarga de Información

```
┌─────────────────────────────────────────────────────────────────┐
│                    EL PROBLEMA DE ESCALA                        │
├─────────────────────────────────────────────────────────────────┤
│  Netflix:     ~15,000 títulos                                   │
│  Amazon:      ~350 millones de productos                        │
│  Spotify:     ~100 millones de canciones                        │
│  YouTube:     500 horas de video subidas por MINUTO             │
├─────────────────────────────────────────────────────────────────┤
│  Pregunta: ¿Cómo encuentra el usuario lo que le interesa?       │
│  Respuesta: SISTEMAS DE RECOMENDACIÓN                           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Impacto Económico

| Empresa | Métrica | Fuente |
|---------|---------|--------|
| **Netflix** | 80% del contenido visto proviene de recomendaciones | Netflix Tech Blog |
| **Amazon** | 35% de ventas provienen de recomendaciones | McKinsey |
| **YouTube** | 70% del tiempo de visualización | YouTube Creator Academy |
| **Spotify** | Discover Weekly: 40M usuarios activos semanales | Spotify |

### 1.3 Formalización del Problema

**Definición Formal:**

Dado:
- Conjunto de usuarios U = {u₁, u₂, ..., uₘ}
- Conjunto de items I = {i₁, i₂, ..., iₙ}
- Matriz de ratings R ∈ ℝᵐˣⁿ (parcialmente observada)

Objetivo:
- Predecir los ratings faltantes R̂ᵤᵢ para (u,i) no observados
- Generar lista ordenada de top-K recomendaciones para cada usuario

```
         Items
         i₁  i₂  i₃  i₄  i₅
       ┌───┬───┬───┬───┬───┐
   u₁  │ 5 │ ? │ 3 │ ? │ 1 │
       ├───┼───┼───┼───┼───┤
U  u₂  │ ? │ 4 │ ? │ 2 │ ? │   R = Matriz de Ratings
s      ├───┼───┼───┼───┼───┤       (sparse)
e  u₃  │ 4 │ ? │ ? │ 5 │ ? │
r      ├───┼───┼───┼───┼───┤
s  u₄  │ ? │ 3 │ 4 │ ? │ 5 │
       └───┴───┴───┴───┴───┘

       ? = valores a predecir
```

---

## 2. Taxonomía de Métodos

### 2.1 Clasificación Principal

```
                    Sistemas de Recomendación
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌───────────┐      ┌───────────┐      ┌───────────┐
    │ Content   │      │Collaborative│     │  Hybrid   │
    │  Based    │      │ Filtering  │      │           │
    └───────────┘      └───────────┘      └───────────┘
          │                   │
          │           ┌───────┴───────┐
          │           │               │
          ▼           ▼               ▼
    ┌───────────┐ ┌───────┐    ┌───────────┐
    │ Atributos │ │Memory │    │  Model    │
    │ de items  │ │ Based │    │  Based    │
    └───────────┘ └───────┘    └───────────┘
                      │               │
                ┌─────┴─────┐   ┌─────┴─────┐
                │           │   │           │
             User-       Item- │Matrix     │Deep
             Based       Based │Factor.    │Learning
```

### 2.2 Comparación de Enfoques

| Aspecto | Content-Based | Collaborative Filtering |
|---------|---------------|------------------------|
| **Datos requeridos** | Atributos de items | Solo ratings |
| **Cold-start usuarios** | Sí (si hay perfil) | Problemático |
| **Cold-start items** | Sí (si hay atributos) | Problemático |
| **Serendipity** | Baja (burbuja de filtro) | Alta |
| **Escalabilidad** | Alta | Media (memory-based) |
| **Explicabilidad** | Alta | Media-Baja |

---

## 3. Filtrado Colaborativo

### 3.1 Intuición Fundamental

> "Si a usuarios similares les gustaron items similares en el pasado,
> probablemente les gustarán items similares en el futuro."

**Analogía del Cine:**
```
Tú viste y te gustaron: Matrix, Inception, Interstellar
Tu amigo vio: Matrix, Inception, Interstellar, Arrival
→ Recomendación: Arrival (porque tu amigo tiene gustos similares)
```

### 3.2 User-Based Collaborative Filtering

**Algoritmo:**
1. Encontrar usuarios similares al usuario objetivo
2. Agregar ratings de usuarios similares para items no vistos
3. Recomendar items con mayor rating predicho

```python
def predict_rating_user_based(user_u, item_i, ratings_matrix, k=10):
    """
    Predicción basada en usuarios similares.

    r̂(u,i) = r̄ᵤ + Σ sim(u,v) * (rᵥᵢ - r̄ᵥ) / Σ |sim(u,v)|
    """
    # Encontrar usuarios que han calificado item_i
    users_who_rated_i = ratings_matrix[:, item_i].nonzero()[0]

    # Calcular similaridad con cada usuario
    similarities = []
    for v in users_who_rated_i:
        if v != user_u:
            sim = cosine_similarity(ratings_matrix[user_u], ratings_matrix[v])
            similarities.append((v, sim))

    # Tomar top-k más similares
    top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    # Predicción ponderada
    user_mean = ratings_matrix[user_u].mean()
    numerator = sum(sim * (ratings_matrix[v, item_i] - ratings_matrix[v].mean())
                    for v, sim in top_k)
    denominator = sum(abs(sim) for _, sim in top_k)

    return user_mean + numerator / denominator if denominator > 0 else user_mean
```

### 3.3 Item-Based Collaborative Filtering

**Diferencia clave:** En lugar de buscar usuarios similares, buscamos items similares.

**Ventaja:** La similaridad entre items es más estable que entre usuarios (los gustos de usuarios cambian más frecuentemente).

```python
def predict_rating_item_based(user_u, item_i, ratings_matrix, k=10):
    """
    Predicción basada en items similares.

    r̂(u,i) = Σ sim(i,j) * rᵤⱼ / Σ |sim(i,j)|

    donde j son items calificados por usuario u similares a item i
    """
    # Items calificados por usuario u
    items_rated_by_u = ratings_matrix[user_u].nonzero()[0]

    # Calcular similaridad con item_i
    similarities = []
    for j in items_rated_by_u:
        if j != item_i:
            sim = cosine_similarity(ratings_matrix[:, item_i], ratings_matrix[:, j])
            similarities.append((j, sim))

    # Top-k items más similares
    top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]

    # Predicción ponderada
    numerator = sum(sim * ratings_matrix[user_u, j] for j, sim in top_k)
    denominator = sum(abs(sim) for _, sim in top_k)

    return numerator / denominator if denominator > 0 else 0
```

### 3.4 Métricas de Similaridad

| Métrica | Fórmula | Uso |
|---------|---------|-----|
| **Cosine** | cos(u,v) = u·v / (‖u‖‖v‖) | Ratings implícitos |
| **Pearson** | ρ(u,v) = cov(u,v) / (σᵤσᵥ) | Ratings explícitos (considera bias) |
| **Jaccard** | J(u,v) = \|u∩v\| / \|u∪v\| | Datos binarios (compró/no compró) |

---

## 4. Factorización de Matrices

### 4.1 La Gran Idea

> **Hipótesis de Baja Dimensionalidad:**
> Los gustos de usuarios y características de items pueden representarse
> en un espacio latente de dimensión k << min(m, n).

```
Matriz R (m×n)           ≈    P (m×k)    ×    Q (k×n)
┌─────────────┐             ┌───────┐      ┌─────────┐
│             │             │       │      │         │
│  Ratings    │      ≈      │ User  │   ×  │  Item   │
│  Observados │             │Factors│      │ Factors │
│             │             │       │      │         │
└─────────────┘             └───────┘      └─────────┘
  (sparse)                    (dense)        (dense)

Ejemplo: k = 20 factores latentes
- Factor 1: ¿Es película de acción?
- Factor 2: ¿Tiene romance?
- Factor 3: ¿Es para adultos?
- ... (la mayoría no son interpretables)
```

### 4.2 Representación Matemática

Para predecir el rating del usuario u para el item i:

```
r̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤᵀ · qᵢ

donde:
- μ  = media global de ratings
- bᵤ = bias del usuario u (¿califica generalmente alto/bajo?)
- bᵢ = bias del item i (¿es generalmente bien/mal calificado?)
- pᵤ = vector latente del usuario (k dimensiones)
- qᵢ = vector latente del item (k dimensiones)
```

### 4.3 Función de Pérdida

Minimizar el error de reconstrucción con regularización:

```
L = Σ (rᵤᵢ - r̂ᵤᵢ)² + λ(‖pᵤ‖² + ‖qᵢ‖² + bᵤ² + bᵢ²)
    (u,i)∈Ω

donde Ω = conjunto de ratings observados
      λ = parámetro de regularización
```

### 4.4 Optimización: SGD vs ALS

**Stochastic Gradient Descent (SGD):**
```python
def sgd_update(u, i, rating, P, Q, b_u, b_i, mu, lr=0.01, reg=0.1):
    """Una actualización de SGD"""
    # Predicción actual
    pred = mu + b_u[u] + b_i[i] + np.dot(P[u], Q[i])
    error = rating - pred

    # Actualizar biases
    b_u[u] += lr * (error - reg * b_u[u])
    b_i[i] += lr * (error - reg * b_i[i])

    # Actualizar factores latentes
    P[u] += lr * (error * Q[i] - reg * P[u])
    Q[i] += lr * (error * P[u] - reg * Q[i])
```

**Alternating Least Squares (ALS):**
- Fijar Q, optimizar P (problema de mínimos cuadrados)
- Fijar P, optimizar Q
- Repetir hasta convergencia
- **Ventaja:** Paralelizable, usado por Spark MLlib

---

## 5. SVD para Recomendaciones

### 5.1 SVD Clásico vs SVD para RecSys

**SVD Clásico** (Álgebra Lineal):
```
A = UΣVᵀ

donde:
- U: vectores singulares izquierdos (m×m)
- Σ: valores singulares (diagonal, m×n)
- V: vectores singulares derechos (n×n)
```

**Problema:** SVD clásico requiere matriz completa (sin valores faltantes).

**SVD para RecSys:** Técnicamente es **factorización de matrices** (no SVD puro), pero se llama "SVD" por convención en la literatura de recomendación.

### 5.2 Truncated SVD para Reducción de Dimensionalidad

```python
import numpy as np
from scipy.sparse.linalg import svds

def truncated_svd_recommendation(ratings_matrix, k=50):
    """
    Aproximación de baja dimensión usando SVD truncado.
    Solo funciona si la matriz es densa (se rellenan valores faltantes).
    """
    # Rellenar valores faltantes con media (simple)
    ratings_filled = ratings_matrix.copy()
    ratings_filled[ratings_filled == 0] = ratings_matrix[ratings_matrix > 0].mean()

    # SVD truncado
    U, sigma, Vt = svds(ratings_filled, k=k)

    # Reconstruir matriz aproximada
    sigma_diag = np.diag(sigma)
    predictions = U @ sigma_diag @ Vt

    return predictions
```

### 5.3 Algoritmo SVD++ (Koren, 2008)

**Mejora:** Incorporar feedback implícito (qué items ha visto el usuario, aunque no los haya calificado).

```
r̂ᵤᵢ = μ + bᵤ + bᵢ + qᵢᵀ · (pᵤ + |N(u)|^(-1/2) Σⱼ∈N(u) yⱼ)

donde:
- N(u) = conjunto de items que usuario u ha interactuado
- yⱼ = vector de feedback implícito para item j
```

### 5.4 Implementación con Surprise Library

```python
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split

# Cargar datos MovieLens
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_builtin('ml-100k')

# Dividir datos
trainset, testset = train_test_split(data, test_size=0.2)

# Entrenar SVD
algo = SVD(
    n_factors=100,      # Dimensionalidad del espacio latente
    n_epochs=20,        # Número de épocas
    lr_all=0.005,       # Learning rate
    reg_all=0.02,       # Regularización
    biased=True         # Incluir biases
)
algo.fit(trainset)

# Evaluar
predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse:.4f}")

# Hacer una predicción
user_id = '196'
item_id = '302'
pred = algo.predict(user_id, item_id)
print(f"Predicción para user {user_id}, item {item_id}: {pred.est:.2f}")
```

---

## 6. Implementación Práctica: MovieLens

### 6.1 Descripción del Dataset

| Versión | Ratings | Usuarios | Películas | Densidad |
|---------|---------|----------|-----------|----------|
| ml-100k | 100,000 | 943 | 1,682 | 6.3% |
| ml-1m | 1,000,000 | 6,040 | 3,706 | 4.5% |
| ml-10m | 10,000,000 | 71,567 | 10,681 | 1.3% |
| ml-25m | 25,000,000 | 162,541 | 62,423 | 0.2% |

### 6.2 Exploración Inicial

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar MovieLens 100k
ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                      names=['user_id', 'item_id', 'rating', 'timestamp'])

# Estadísticas básicas
print(f"Total ratings: {len(ratings):,}")
print(f"Usuarios únicos: {ratings.user_id.nunique()}")
print(f"Items únicos: {ratings.item_id.nunique()}")
print(f"Rating promedio: {ratings.rating.mean():.2f}")
print(f"Densidad: {len(ratings) / (ratings.user_id.nunique() * ratings.item_id.nunique()) * 100:.2f}%")

# Distribución de ratings
ratings.rating.value_counts().sort_index().plot(kind='bar')
plt.title('Distribución de Ratings')
plt.xlabel('Rating')
plt.ylabel('Frecuencia')
plt.show()

# Long-tail de popularidad
item_counts = ratings.groupby('item_id').size().sort_values(ascending=False)
plt.figure(figsize=(12, 4))
plt.plot(range(len(item_counts)), item_counts.values)
plt.xlabel('Item (ordenado por popularidad)')
plt.ylabel('Número de ratings')
plt.title('Long-tail: Distribución de popularidad de items')
plt.yscale('log')
plt.show()
```

### 6.3 Pipeline Completo de Recomendación

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

class MatrixFactorizationRecommender:
    """
    Recomendador basado en factorización de matrices con SGD.
    """

    def __init__(self, n_factors=50, n_epochs=20, lr=0.01, reg=0.1):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

    def fit(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        """Entrenar el modelo."""
        # Crear mapeos de IDs
        self.user_ids = ratings_df[user_col].unique()
        self.item_ids = ratings_df[item_col].unique()
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {i: j for j, i in enumerate(self.item_ids)}

        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)

        # Inicializar parámetros
        self.global_mean = ratings_df[rating_col].mean()
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        self.P = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.n_factors))

        # Convertir a arrays numpy
        users = ratings_df[user_col].map(self.user_to_idx).values
        items = ratings_df[item_col].map(self.item_to_idx).values
        ratings = ratings_df[rating_col].values

        # Entrenamiento con SGD
        for epoch in range(self.n_epochs):
            # Shuffle
            indices = np.random.permutation(len(ratings))
            total_loss = 0

            for idx in indices:
                u, i, r = users[idx], items[idx], ratings[idx]

                # Predicción
                pred = self.global_mean + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
                error = r - pred
                total_loss += error ** 2

                # Actualizar parámetros
                self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])

                P_u_old = self.P[u].copy()
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * P_u_old - self.reg * self.Q[i])

            rmse = np.sqrt(total_loss / len(ratings))
            print(f"Epoch {epoch+1}/{self.n_epochs}, RMSE: {rmse:.4f}")

        return self

    def predict(self, user_id, item_id):
        """Predecir rating para un usuario e item."""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean

        u = self.user_to_idx[user_id]
        i = self.item_to_idx[item_id]

        pred = self.global_mean + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
        return np.clip(pred, 1, 5)

    def recommend(self, user_id, n=10, exclude_seen=True, seen_items=None):
        """Generar top-N recomendaciones para un usuario."""
        if user_id not in self.user_to_idx:
            return []

        u = self.user_to_idx[user_id]

        # Predecir todos los items
        predictions = []
        for item_id in self.item_ids:
            if exclude_seen and seen_items and item_id in seen_items:
                continue
            predictions.append((item_id, self.predict(user_id, item_id)))

        # Ordenar por predicción
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

# Uso
model = MatrixFactorizationRecommender(n_factors=50, n_epochs=20)
model.fit(ratings)

# Recomendaciones para usuario 1
seen = set(ratings[ratings.user_id == 1].item_id)
recommendations = model.recommend(user_id=1, n=10, seen_items=seen)
print("Top 10 recomendaciones para usuario 1:")
for item_id, score in recommendations:
    print(f"  Item {item_id}: {score:.2f}")
```

---

## 7. Métricas de Evaluación

### 7.1 Métricas de Rating Prediction

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **RMSE** | √(Σ(rᵤᵢ - r̂ᵤᵢ)²/N) | Error cuadrático medio (penaliza errores grandes) |
| **MAE** | Σ\|rᵤᵢ - r̂ᵤᵢ\|/N | Error absoluto medio |

### 7.2 Métricas de Ranking (Top-K)

| Métrica | Descripción | Fórmula |
|---------|-------------|---------|
| **Precision@K** | Proporción de items relevantes en top-K | \|Rec ∩ Rel\| / K |
| **Recall@K** | Proporción de items relevantes recuperados | \|Rec ∩ Rel\| / \|Rel\| |
| **NDCG@K** | Normalized Discounted Cumulative Gain | DCG / IDCG |
| **MAP** | Mean Average Precision | Promedio de AP sobre usuarios |
| **Hit Rate** | Proporción de usuarios con al menos 1 hit | |

### 7.3 Implementación de Métricas

```python
def precision_at_k(recommended, relevant, k):
    """Precision@K"""
    recommended_k = set(recommended[:k])
    relevant_set = set(relevant)
    return len(recommended_k & relevant_set) / k

def recall_at_k(recommended, relevant, k):
    """Recall@K"""
    recommended_k = set(recommended[:k])
    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        return 0
    return len(recommended_k & relevant_set) / len(relevant_set)

def ndcg_at_k(recommended, relevant, k):
    """NDCG@K"""
    def dcg(scores, k):
        return sum(s / np.log2(i + 2) for i, s in enumerate(scores[:k]))

    # Relevance scores (1 if in relevant, 0 otherwise)
    scores = [1 if item in relevant else 0 for item in recommended[:k]]
    ideal_scores = sorted(scores, reverse=True)

    dcg_val = dcg(scores, k)
    idcg_val = dcg(ideal_scores, k)

    return dcg_val / idcg_val if idcg_val > 0 else 0

# Ejemplo de uso
recommended = ['item1', 'item2', 'item3', 'item4', 'item5']
relevant = ['item2', 'item5', 'item7']

print(f"Precision@5: {precision_at_k(recommended, relevant, 5):.2f}")
print(f"Recall@5: {recall_at_k(recommended, relevant, 5):.2f}")
print(f"NDCG@5: {ndcg_at_k(recommended, relevant, 5):.2f}")
```

---

## 8. Problemas y Soluciones

### 8.1 El Problema del Cold-Start

| Tipo | Problema | Soluciones |
|------|----------|------------|
| **Nuevo usuario** | No hay ratings históricos | Content-based, demografía, preguntas iniciales |
| **Nuevo item** | Nadie lo ha calificado | Content-based, item attributes |
| **Nuevo sistema** | Pocos datos totales | Híbrido, exploración activa |

### 8.2 Sparsity

**Problema:** En sistemas reales, la matriz de ratings tiene >99% de valores faltantes.

**Soluciones:**
- Regularización fuerte
- Factorización de matrices (reduce dimensionalidad)
- Incorporar datos auxiliares (grafos sociales, atributos)

### 8.3 Scalability

```
           Complejidad Computacional

Method              Time           Space
─────────────────────────────────────────
User-Based CF       O(m²n)         O(m²)
Item-Based CF       O(mn²)         O(n²)
Matrix Fact.        O(nnz·k·T)     O((m+n)k)

m = usuarios, n = items, k = factores
nnz = número de ratings, T = épocas
```

### 8.4 Burbuja de Filtro (Filter Bubble)

**Problema:** Sistema solo recomienda items similares a los ya vistos.

**Soluciones:**
- Diversificación explícita en recomendaciones
- Exploración (ε-greedy, Thompson Sampling)
- Métricas de diversidad (ILS, coverage)

---

## 9. Ejercicios

### Ejercicio 1: Implementar Similaridad Coseno

```python
"""
Implementar la función de similaridad coseno para vectores sparse.
Usarla para encontrar los 5 items más similares a un item dado.
"""
def cosine_similarity_sparse(vec1, vec2):
    # Tu implementación
    pass
```

### Ejercicio 2: Comparar User-Based vs Item-Based

```python
"""
1. Cargar MovieLens 100k
2. Implementar ambos métodos (user-based e item-based)
3. Comparar RMSE en conjunto de test
4. Analizar tiempos de predicción
"""
```

### Ejercicio 3: Tune SVD con Grid Search

```python
"""
Usar Surprise para hacer grid search sobre:
- n_factors: [20, 50, 100, 200]
- n_epochs: [10, 20, 30]
- reg_all: [0.01, 0.02, 0.1]

Reportar mejor configuración y RMSE.
"""
from surprise.model_selection import GridSearchCV
# Tu código
```

### Ejercicio 4: Implementar NDCG desde cero

```python
"""
Implementar NDCG@K con relevancia binaria y graduada.
Validar contra implementación de sklearn.
"""
```

---

## 10. Resumen

| Concepto | Punto Clave |
|----------|-------------|
| **Filtrado Colaborativo** | Usuarios similares → items similares |
| **Factorización** | R ≈ P × Qᵀ (espacio latente) |
| **SVD** | Técnica fundamental, Netflix Prize winner |
| **Evaluación** | RMSE para ratings, NDCG para ranking |
| **Cold-Start** | Híbrido content + collaborative |

---

## 11. Lecturas Recomendadas

1. **"Matrix Factorization Techniques for Recommender Systems"** (Koren et al., IEEE 2009) - Paper fundamental

2. **"The BellKor Solution to the Netflix Prize"** (2009) - Caso de estudio detallado

3. **Surprise Library Documentation** - https://surprise.readthedocs.io/

4. **"Recommender Systems Handbook"** (Ricci et al., 2015) - Referencia completa

---

*Material desarrollado para el MS-AI Pathway - University of Colorado Boulder*
*Semana 15 - CSCA 5632: Unsupervised Learning*
