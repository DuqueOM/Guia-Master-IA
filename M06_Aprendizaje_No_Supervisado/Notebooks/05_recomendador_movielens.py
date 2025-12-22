"""
Notebook 05: Sistema de Recomendación con SVD - MovieLens
==========================================================

Módulo 6 - Semana 15: Sistemas de Recomendación
Curso Alineado: CSCA 5632 - Unsupervised Learning

Objetivos:
1. Implementar factorización de matrices desde cero
2. Usar la librería Surprise para SVD
3. Construir un recomendador funcional con MovieLens
4. Evaluar con métricas apropiadas (RMSE, Precision@K)

Dataset: MovieLens 100K
    https://grouplens.org/datasets/movielens/

Dependencias:
    pip install surprise pandas numpy matplotlib seaborn

Ejecutar como script o convertir a notebook con jupytext.
"""

# %% [markdown]
# # Sistema de Recomendación con Factorización de Matrices
#
# En este notebook implementaremos un sistema de recomendación de películas
# usando el dataset MovieLens y la técnica de **Factorización de Matrices (SVD)**.

# %%
# Imports
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Verificar Surprise
try:
    from surprise import NMF, SVD, Dataset, KNNBasic, SVDpp, accuracy
    from surprise.model_selection import GridSearchCV, cross_validate, train_test_split

    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("⚠️ Surprise no instalado. Ejecutar: pip install scikit-surprise")

print("✅ Imports completados")

# %% [markdown]
# ## 1. Cargar y Explorar MovieLens Dataset
#
# MovieLens es el dataset estándar para evaluación de sistemas de recomendación.
# Usaremos la versión 100K (100,000 ratings).

# %%
if SURPRISE_AVAILABLE:
    # Cargar dataset builtin de Surprise
    print("📥 Cargando MovieLens 100K...")
    data = Dataset.load_builtin("ml-100k")

    # Obtener dataframe para exploración
    trainset = data.build_full_trainset()

    print("\n📊 Estadísticas del Dataset:")
    print(f"   Usuarios: {trainset.n_users:,}")
    print(f"   Items (películas): {trainset.n_items:,}")
    print(f"   Ratings: {trainset.n_ratings:,}")
    print(
        f"   Densidad: {trainset.n_ratings / (trainset.n_users * trainset.n_items) * 100:.2f}%"
    )
    print(f"   Rango de ratings: {trainset.rating_scale}")

# %%
if SURPRISE_AVAILABLE:
    # Convertir a DataFrame para análisis
    ratings_list = [
        (trainset.to_raw_uid(u), trainset.to_raw_iid(i), r)
        for u, i, r in trainset.all_ratings()
    ]

    df_ratings = pd.DataFrame(ratings_list, columns=["user_id", "item_id", "rating"])

    print("\n📋 Muestra de ratings:")
    print(df_ratings.head(10))

    print("\n📈 Distribución de ratings:")
    print(df_ratings["rating"].value_counts().sort_index())

# %%
if SURPRISE_AVAILABLE:
    # Visualizar distribución
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Distribución de ratings
    axes[0].hist(df_ratings["rating"], bins=5, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Distribución de Ratings")

    # Ratings por usuario
    ratings_per_user = df_ratings.groupby("user_id").size()
    axes[1].hist(ratings_per_user, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Número de ratings")
    axes[1].set_ylabel("Número de usuarios")
    axes[1].set_title("Ratings por Usuario")
    axes[1].set_xlim(0, 500)

    # Ratings por película (long-tail)
    ratings_per_item = df_ratings.groupby("item_id").size().sort_values(ascending=False)
    axes[2].plot(range(len(ratings_per_item)), ratings_per_item.values)
    axes[2].set_xlabel("Película (ordenada por popularidad)")
    axes[2].set_ylabel("Número de ratings")
    axes[2].set_title("Long-tail de Popularidad")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 1.5 Visualización de Sparsity (Matriz Usuario × Item)
#
# > **💡 Conexión con M02 - Álgebra Lineal**: La matriz de ratings R es una matriz
# > **dispersa (sparse)** donde la mayoría de entradas son desconocidas.
# > En M02 estudiamos matrices densas vs dispersas. Aquí, el 98%+ está vacía.

# %%
if SURPRISE_AVAILABLE:
    # Crear matriz para visualización (subset pequeño)
    n_users_sample, n_items_sample = 50, 100
    unique_users = df_ratings["user_id"].unique()[:n_users_sample]
    unique_items = df_ratings["item_id"].unique()[:n_items_sample]

    # Matriz de ratings (NaN para valores faltantes)
    rating_matrix = np.full((n_users_sample, n_items_sample), np.nan)
    user_map = {u: i for i, u in enumerate(unique_users)}
    item_map = {it: i for i, it in enumerate(unique_items)}

    for _, row in df_ratings.iterrows():
        if row["user_id"] in user_map and row["item_id"] in item_map:
            rating_matrix[user_map[row["user_id"]], item_map[row["item_id"]]] = row[
                "rating"
            ]

    # Visualizar sparsity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico 1: Patrón de sparsity
    mask = ~np.isnan(rating_matrix)
    axes[0].imshow(mask, cmap="Blues", aspect="auto")
    axes[0].set_xlabel("Items (películas)")
    axes[0].set_ylabel("Usuarios")
    axes[0].set_title("Patrón de Sparsity (Azul = Rating conocido)")
    sparsity = 1 - np.sum(mask) / mask.size
    axes[0].text(
        0.02,
        0.98,
        f"Sparsity: {sparsity*100:.1f}%",
        transform=axes[0].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white"},
    )

    # Gráfico 2: Heatmap de ratings
    rating_display = np.ma.masked_where(np.isnan(rating_matrix), rating_matrix)
    im = axes[1].imshow(rating_display, cmap="YlOrRd", aspect="auto", vmin=1, vmax=5)
    axes[1].set_xlabel("Items (películas)")
    axes[1].set_ylabel("Usuarios")
    axes[1].set_title("Matriz Usuario × Item (valores conocidos)")
    plt.colorbar(im, ax=axes[1], label="Rating")

    plt.tight_layout()
    plt.show()

    print("\n📊 Estadísticas de Sparsity:")
    print(f"   Matriz completa: {trainset.n_users * trainset.n_items:,} entradas")
    print(f"   Ratings conocidos: {trainset.n_ratings:,}")
    print(
        f"   Sparsity global: {(1 - trainset.n_ratings / (trainset.n_users * trainset.n_items)) * 100:.2f}%"
    )

# %% [markdown]
# ## 2. Matrix Factorization: Teoría vs Práctica
#
# ### 2.1 Teoría Matemática
#
# La idea es descomponer la matriz de ratings R (usuarios × items) en dos matrices:
# - P (usuarios × factores): preferencias latentes de usuarios
# - Q (factores × items): características latentes de items
#
# $$ R \approx P \times Q^T $$
#
# Además, usamos biases para capturar tendencias:
# $$ \hat{r}_{ui} = \mu + b_u + b_i + p_u \cdot q_i $$
#
# ### 2.2 💡 SVD Clásico vs SVD para Recomendación
#
# > **⚠️ DISTINCIÓN CRÍTICA**:
# >
# > | Aspecto | SVD Clásico (M02) | TruncatedSVD (sklearn) | SVD Recomendación |
# > |---------|-------------------|------------------------|-------------------|
# > | Fórmula | $A = U\Sigma V^T$ | Aproximación low-rank | $R \approx PQ^T$ |
# > | Valores faltantes | ❌ Requiere matriz completa | ❌ Trata NaN como 0 | ✅ Los ignora |
# > | Uso típico | Álgebra lineal | Reducción dimensionalidad | Sistemas recomendación |
# > | Implementación | `np.linalg.svd()` | `sklearn.TruncatedSVD` | Surprise, LightFM |
#
# **¿Por qué NO usar TruncatedSVD de sklearn para recomendación?**
# - Trata valores faltantes como 0 (un rating muy bajo)
# - Esto sesga las predicciones hacia items populares

# %%
# Demo: Comparación TruncatedSVD vs SVD para Recomendación
if SURPRISE_AVAILABLE:
    from sklearn.decomposition import TruncatedSVD

    # TruncatedSVD trata NaN como 0 (INCORRECTO para recomendación)
    rating_matrix_filled = np.nan_to_num(rating_matrix, nan=0.0)
    truncated = TruncatedSVD(n_components=10, random_state=42)
    U_trunc = truncated.fit_transform(rating_matrix_filled)

    print("⚠️ TruncatedSVD (sklearn) NO es apropiado para recomendación:")
    print(f"   - Trata {int(np.sum(np.isnan(rating_matrix)))} valores NaN como 0")
    print(f"   - Varianza explicada: {truncated.explained_variance_ratio_.sum():.1%}")
    print(
        "\n✅ En cambio, SVD de Surprise IGNORA valores faltantes durante el entrenamiento."
    )


# %%
class MatrixFactorizationSGD:
    """
    Factorización de Matrices con Stochastic Gradient Descent.

    Implementación educativa desde cero.

    Parámetros:
    -----------
    n_factors : int
        Número de factores latentes (dimensión del embedding).
    n_epochs : int
        Número de épocas de entrenamiento.
    lr : float
        Learning rate.
    reg : float
        Parámetro de regularización L2.
    verbose : bool
        Si True, imprime progreso.
    """

    def __init__(
        self,
        n_factors=50,
        n_epochs=20,
        lr=0.005,
        reg=0.02,
        verbose=True,
        random_state: int | None = None,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.verbose = verbose
        self.random_state = random_state

    def fit(
        self, ratings_df, user_col="user_id", item_col="item_id", rating_col="rating"
    ):
        """
        Entrenar el modelo.

        Parámetros:
        -----------
        ratings_df : pd.DataFrame
            DataFrame con columnas user, item, rating.
        """
        # Crear mapeos de IDs
        self.user_ids = ratings_df[user_col].unique()
        self.item_ids = ratings_df[item_col].unique()
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {i: j for j, i in enumerate(self.item_ids)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.idx_to_item = {j: i for i, j in self.item_to_idx.items()}

        self.n_users = len(self.user_ids)
        self.n_items = len(self.item_ids)

        rng = np.random.default_rng(self.random_state)

        # Inicializar parámetros
        self.global_mean = ratings_df[rating_col].mean()
        self.b_u = np.zeros(self.n_users)  # User bias
        self.b_i = np.zeros(self.n_items)  # Item bias
        self.P = rng.normal(0, 0.1, (self.n_users, self.n_factors))  # User factors
        self.Q = rng.normal(0, 0.1, (self.n_items, self.n_factors))  # Item factors

        # Convertir a arrays
        users = ratings_df[user_col].map(self.user_to_idx).values
        items = ratings_df[item_col].map(self.item_to_idx).values
        ratings = ratings_df[rating_col].values

        # Historial de entrenamiento
        self.history: dict[str, list[float]] = {"rmse": []}

        # Entrenamiento SGD
        for epoch in range(self.n_epochs):
            # Shuffle
            indices = rng.permutation(len(ratings))
            total_error = 0

            for idx in indices:
                u, i, r = users[idx], items[idx], ratings[idx]

                # Predicción actual
                pred = (
                    self.global_mean
                    + self.b_u[u]
                    + self.b_i[i]
                    + np.dot(self.P[u], self.Q[i])
                )
                error = r - pred
                total_error += error**2

                # Actualizar biases
                self.b_u[u] += self.lr * (error - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (error - self.reg * self.b_i[i])

                # Actualizar factores
                P_u_old = self.P[u].copy()
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * P_u_old - self.reg * self.Q[i])

            rmse = np.sqrt(total_error / len(ratings))
            self.history["rmse"].append(rmse)

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.n_epochs} - RMSE: {rmse:.4f}")

        return self

    def predict(self, user_id, item_id):
        """Predecir rating para un usuario e item."""
        if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
            return self.global_mean

        u = self.user_to_idx[user_id]
        i = self.item_to_idx[item_id]

        pred = (
            self.global_mean + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
        )
        return np.clip(pred, 1, 5)

    def recommend(self, user_id, n=10, exclude_seen=True, seen_items=None):
        """
        Generar top-N recomendaciones para un usuario.

        Retorna lista de (item_id, predicted_rating).
        """
        if user_id not in self.user_to_idx:
            return []

        predictions = []
        for item_id in self.item_ids:
            if exclude_seen and seen_items and item_id in seen_items:
                continue
            pred = self.predict(user_id, item_id)
            predictions.append((item_id, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    def plot_training(self):
        """Visualizar curva de entrenamiento."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history["rmse"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.title("Training Progress")
        plt.grid(True, alpha=0.3)
        plt.show()


# %%
# Entrenar modelo desde cero
print("🔬 Entrenando modelo de factorización (desde cero)...")

mf_model = MatrixFactorizationSGD(
    n_factors=50, n_epochs=20, lr=0.005, reg=0.02, verbose=True
)

mf_model.fit(df_ratings)

# %%
# Visualizar entrenamiento
mf_model.plot_training()

# %%
# Probar recomendaciones
user_test = df_ratings["user_id"].iloc[0]
seen_movies = set(df_ratings[df_ratings["user_id"] == user_test]["item_id"])

print(f"\n🎬 Recomendaciones para usuario '{user_test}':")
print(f"   Películas ya vistas: {len(seen_movies)}")

recommendations = mf_model.recommend(user_test, n=10, seen_items=seen_movies)
print("\n   Top 10 recomendaciones:")
for i, (item_id, score) in enumerate(recommendations, 1):
    print(f"   {i}. Item {item_id}: {score:.2f}")

# %% [markdown]
# ## 3. SVD con Surprise Library
#
# Ahora usaremos la librería **Surprise** que implementa SVD optimizado
# y otras variantes como SVD++ y NMF.

# %%
if SURPRISE_AVAILABLE:
    # Split train/test
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    print("📊 Split:")
    print(f"   Train: {trainset.n_ratings:,} ratings")
    print(f"   Test: {len(testset):,} ratings")

# %%
if SURPRISE_AVAILABLE:
    # Entrenar SVD
    print("\n🔬 Entrenando SVD (Surprise)...")

    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)

    svd.fit(trainset)

    # Evaluar
    predictions = svd.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    print("\n📈 Resultados:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")

# %%
if SURPRISE_AVAILABLE:
    # Cross-validation para comparar algoritmos
    print("\n🔬 Comparando algoritmos (5-fold CV)...")

    algorithms = {
        "SVD": SVD(random_state=42),
        "SVD++": SVDpp(random_state=42),
        "NMF": NMF(random_state=42),
        "KNN (user-based)": KNNBasic(sim_options={"user_based": True}),
    }

    results = {}

    for name, algo in algorithms.items():
        print(f"\n   Evaluando {name}...")
        cv_results = cross_validate(
            algo, data, measures=["RMSE", "MAE"], cv=5, verbose=False
        )
        results[name] = {
            "RMSE": cv_results["test_rmse"].mean(),
            "MAE": cv_results["test_mae"].mean(),
            "Fit Time": cv_results["fit_time"].mean(),
        }
        print(f"   RMSE: {results[name]['RMSE']:.4f} | MAE: {results[name]['MAE']:.4f}")

    # Tabla comparativa
    results_df = pd.DataFrame(results).T
    print("\n📊 Tabla Comparativa:")
    print(results_df.to_string())

# %%
if SURPRISE_AVAILABLE:
    # Visualizar comparación
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(results))
    width = 0.35

    rmse_vals = [results[name]["RMSE"] for name in results]
    mae_vals = [results[name]["MAE"] for name in results]

    ax.bar(x - width / 2, rmse_vals, width, label="RMSE", color="steelblue")
    ax.bar(x + width / 2, mae_vals, width, label="MAE", color="coral")

    ax.set_ylabel("Error")
    ax.set_title("Comparación de Algoritmos de Recomendación")
    ax.set_xticks(x)
    ax.set_xticklabels(results.keys(), rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Grid Search para Optimizar Hiperparámetros

# %%
if SURPRISE_AVAILABLE:
    print("🔍 Grid Search para SVD...")

    param_grid = {
        "n_factors": [50, 100],
        "n_epochs": [20, 30],
        "lr_all": [0.002, 0.005],
        "reg_all": [0.02, 0.1],
    }

    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=-1)
    gs.fit(data)

    print("\n📈 Mejores resultados:")
    print(f"   Mejor RMSE: {gs.best_score['rmse']:.4f}")
    print(f"   Mejores parámetros: {gs.best_params['rmse']}")

# %% [markdown]
# ## 5. Métricas de Ranking: Precision@K y NDCG


# %%
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    """
    Calcular Precision y Recall @K.

    Considera como "relevante" un item con rating >= threshold.
    """
    # Agrupar predicciones por usuario
    user_est_true = defaultdict(list)
    for uid, _iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = []
    recalls = []

    for _uid, user_ratings in user_est_true.items():
        # Ordenar por rating predicho
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Top K predicciones
        top_k = user_ratings[:k]

        # Número de items relevantes en top K
        n_relevant_in_k = sum(1 for (_, true_r) in top_k if true_r >= threshold)

        # Total de items relevantes
        n_relevant_total = sum(1 for (_, true_r) in user_ratings if true_r >= threshold)

        # Precision@K
        precision = n_relevant_in_k / k
        precisions.append(precision)

        # Recall@K
        recall = n_relevant_in_k / n_relevant_total if n_relevant_total > 0 else 0
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)


# %%
if SURPRISE_AVAILABLE:
    # Calcular métricas de ranking
    print("📊 Métricas de Ranking:")

    for k in [5, 10, 20]:
        precision, recall = precision_recall_at_k(predictions, k=k, threshold=4.0)
        print(f"   @{k}: Precision={precision:.4f}, Recall={recall:.4f}")

# %% [markdown]
# ## 6. Generar Recomendaciones Personalizadas

# %%
if SURPRISE_AVAILABLE:
    # Reentrenar en todo el dataset
    full_trainset = data.build_full_trainset()
    svd_final = SVD(n_factors=100, n_epochs=30, random_state=42)
    svd_final.fit(full_trainset)

    def get_top_n_recommendations(model, trainset, user_id, n=10):
        """
        Obtener top-N recomendaciones para un usuario.
        """
        # Items que el usuario ya ha calificado
        try:
            inner_uid = trainset.to_inner_uid(user_id)
            rated_items = set(trainset.ur[inner_uid])
            rated_items = {trainset.to_raw_iid(iid) for iid, _ in rated_items}
        except ValueError:
            rated_items = set()

        # Predecir para todos los items no vistos
        predictions = []
        for inner_iid in range(trainset.n_items):
            raw_iid = trainset.to_raw_iid(inner_iid)
            if raw_iid not in rated_items:
                pred = model.predict(user_id, raw_iid)
                predictions.append((raw_iid, pred.est))

        # Ordenar y retornar top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n]

    # Ejemplo
    test_user = "196"
    recommendations = get_top_n_recommendations(
        svd_final, full_trainset, test_user, n=10
    )

    print(f"\n🎬 Top 10 recomendaciones para usuario '{test_user}':")
    for i, (item_id, score) in enumerate(recommendations, 1):
        print(f"   {i}. Película {item_id}: {score:.2f} ⭐")

# %% [markdown]
# ## 7. Análisis de Factores Latentes

# %%
if SURPRISE_AVAILABLE:
    # Explorar factores latentes
    print("🔬 Análisis de Factores Latentes")

    # Obtener matrices de factores
    P = svd_final.pu  # User factors
    Q = svd_final.qi  # Item factors

    print(f"\n   Shape de P (usuarios): {P.shape}")
    print(f"   Shape de Q (items): {Q.shape}")

    # Visualizar primeros 2 factores de items
    plt.figure(figsize=(10, 8))

    # Seleccionar subset de items
    n_items_plot = 100
    plt.scatter(Q[:n_items_plot, 0], Q[:n_items_plot, 1], alpha=0.6)

    plt.xlabel("Factor 1")
    plt.ylabel("Factor 2")
    plt.title("Proyección de Items en Espacio Latente (primeros 2 factores)")
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n   Los factores latentes capturan características 'ocultas'")
    print("   como género, época, estilo, que correlacionan con preferencias.")

# %% [markdown]
# ## 8. Ejercicios

# %%
print(
    """
📝 EJERCICIOS

1. COLD-START:
   - Simular usuario nuevo (sin ratings)
   - ¿Cómo podemos recomendar? Implementar estrategia basada en popularidad

2. EVALUACIÓN OFFLINE:
   - Implementar NDCG@K desde cero
   - Comparar con Precision@K: ¿cuál penaliza más errores en top posiciones?

3. HÍBRIDO:
   - Combinar predicciones de SVD y KNN (weighted average)
   - ¿Mejora el RMSE?

4. VISUALIZACIÓN:
   - Usar t-SNE para visualizar items en 2D basado en factores latentes
   - ¿Se agrupan por género?

5. IMPLICIT FEEDBACK:
   - Convertir ratings a feedback implícito (1 si rating >= 4, 0 sino)
   - Entrenar modelo ALS para implicit feedback
"""
)

# %% [markdown]
# ## 9. Resumen

# %%
print(
    """
╔══════════════════════════════════════════════════════════════════╗
║                        RESUMEN                                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  FACTORIZACIÓN DE MATRICES:                                      ║
║  - Descompone R ≈ P × Qᵀ en espacio latente                      ║
║  - Captura patrones no observables directamente                  ║
║  - Base del Netflix Prize (2009)                                 ║
║                                                                  ║
║  ALGORITMOS:                                                     ║
║  - SVD: Factorización básica con biases                          ║
║  - SVD++: Incorpora feedback implícito                           ║
║  - NMF: Factores no negativos (más interpretables)               ║
║  - ALS: Paralelizable (usado en Spark)                           ║
║                                                                  ║
║  MÉTRICAS:                                                       ║
║  - RMSE/MAE: Predicción de rating                                ║
║  - Precision/Recall@K: Calidad de ranking                        ║
║  - NDCG: Penaliza errores en top posiciones                      ║
║                                                                  ║
║  PROBLEMAS:                                                      ║
║  - Cold-start: Usuarios/items nuevos sin datos                   ║
║  - Sparsity: >99% de la matriz vacía                             ║
║  - Scalability: Millones de usuarios/items                       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
)

# %%
print("\n✅ Notebook completado!")
print("   Este módulo es CRÍTICO para CSCA 5632.")
print("   Asegúrate de entender la matemática detrás de SVD.")
