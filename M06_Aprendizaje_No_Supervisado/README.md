# Módulo 06: Aprendizaje No Supervisado

> **Semanas:** 12-15 | **Fase:** ML Core ⭐ | **Curso Alineado:** CSCA 5632

---

## 📁 Estructura

```
M06_Aprendizaje_No_Supervisado/
├── Teoria/
│   ├── 01_clustering_kmeans.md
│   ├── 02_pca_svd.md
│   ├── 03_gmm_em.md
│   ├── 04_tsne_umap.md
│   └── 05_sistemas_recomendacion.md       # NUEVO: Filtrado Colaborativo
├── Notebooks/
│   ├── 01_kmeans_scratch.ipynb
│   ├── 01b_kmeans_sklearn.ipynb
│   ├── 02_pca_scratch.ipynb
│   ├── 02b_pca_sklearn.ipynb
│   ├── 03_gmm_em_scratch.ipynb
│   ├── 04_tsne_umap_visualizacion.ipynb
│   ├── 05_svd_factorizacion_matrices.ipynb # NUEVO
│   └── 05b_recomendador_movielens.ipynb    # NUEVO: Proyecto MovieLens
├── Laboratorios_Interactivos/
│   ├── pca_rotation_plotly_app.py
│   ├── kmeans_clustering_app.py
│   └── movie_recommender_app.py           # NUEVO
├── datasets/
│   └── README_movielens.md                # Instrucciones descarga MovieLens
└── assets/
```

---

## 🎯 Objetivos de Aprendizaje

### Semana 12: Clustering

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Implementar K-Means desde cero | Lloyd's algorithm + K-Means++ init |
| Selección óptima de K | Método del codo + Silhouette Score |
| Usar `sklearn.cluster` | Comparar implementación manual vs sklearn |

### Semana 13: Reducción de Dimensionalidad

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Implementar PCA desde cero | Usando eigendecomposition y SVD |
| Entender varianza explicada | Seleccionar componentes óptimos |
| Visualizar MNIST en 2D | t-SNE y UMAP funcionando |

### Semana 14: Modelos Generativos

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Implementar GMM con EM | Algoritmo EM convergiendo |
| Entender latent variables | Conexión con clustering suave |
| Comparar GMM vs K-Means | Análisis de ventajas/desventajas |

### Semana 15: Sistemas de Recomendación 🆕 (CRÍTICO para CSCA 5632)

| Objetivo | Criterio de Éxito |
|----------|-------------------|
| Entender Filtrado Colaborativo | User-based vs Item-based |
| **Implementar SVD para recomendaciones** | Factorización de matriz de ratings |
| **Proyecto MovieLens Small** | Recomendador funcional con RMSE < 1.0 |
| Evaluar sistemas de recomendación | Precisión@K, Recall@K, NDCG |

---

## 📚 Recursos Semana 15 - Sistemas de Recomendación

### Dataset
- **MovieLens Small (100K)**: https://grouplens.org/datasets/movielens/
- Descargar `ml-latest-small.zip` → extraer en `datasets/`

### Lecturas
1. **"Matrix Factorization Techniques for Recommender Systems"** (Koren et al., IEEE 2009)
2. **Surprise Library Documentation** - https://surprise.readthedocs.io/
3. **Netflix Prize Paper** - Entender el contexto histórico

---

## ⚡ Inicio Rápido

```bash
# Semana 12: Clustering
jupyter notebook Notebooks/01_kmeans_scratch.ipynb
streamlit run Laboratorios_Interactivos/kmeans_clustering_app.py

# Semana 13: PCA
jupyter notebook Notebooks/02_pca_scratch.ipynb
streamlit run Laboratorios_Interactivos/pca_rotation_plotly_app.py

# Semana 14: GMM
jupyter notebook Notebooks/03_gmm_em_scratch.ipynb

# Semana 15: Sistemas de Recomendación (CRÍTICO)
jupyter notebook Notebooks/05_svd_factorizacion_matrices.ipynb
jupyter notebook Notebooks/05b_recomendador_movielens.ipynb
streamlit run Laboratorios_Interactivos/movie_recommender_app.py
```

---

## ✅ Entregables del Módulo

- [ ] `kmeans.py` con tests (from scratch)
- [ ] `pca.py` con tests (from scratch)
- [ ] `gmm.py` con algoritmo EM (from scratch)
- [ ] Visualización t-SNE/UMAP de MNIST
- [ ] **`movie_recommender.py` usando SVD** (CRÍTICO)
- [ ] **Análisis completo MovieLens con métricas de evaluación**

---

## ⚠️ Nota Importante

> El módulo de **Sistemas de Recomendación** es frecuentemente evaluado en CSCA 5632.
> La factorización de matrices (SVD) es un tema central que conecta álgebra lineal
> con aplicaciones prácticas de ML. No saltar esta sección.

---

## 🔗 Navegación

| Anterior | Índice | Siguiente |
|----------|--------|-----------|
| [M05 Supervisado](../M05_Aprendizaje_Supervisado/) | [README](../README.md) | [M07 Deep Learning →](../M07_Deep_Learning/) |
