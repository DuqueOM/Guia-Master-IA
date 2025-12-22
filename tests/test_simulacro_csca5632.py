"""
Tests para Simulacro CSCA 5632 - Unsupervised Learning
Ejecutar: pytest tests/test_simulacro_csca5632.py -v
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "Herramientas_Estudio/Simulacros")


class TestKMeans:
    """Tests para K-Means."""

    def test_kmeans_labels_range(self) -> None:
        """Labels deben estar en [0, k-1]."""
        from simulacro_CSCA5632 import kmeans

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 2))

        _, labels = kmeans(X, k=3)
        assert np.all(labels >= 0)
        assert np.all(labels < 3)

    def test_kmeans_centroids_shape(self) -> None:
        """Centroides deben tener shape (k, n_features)."""
        from simulacro_CSCA5632 import kmeans

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 4))

        centroids, _ = kmeans(X, k=5)
        assert centroids.shape == (5, 4)

    def test_kmeans_finds_clusters(self) -> None:
        """K-Means debe encontrar clusters separados."""
        from simulacro_CSCA5632 import kmeans

        rng = np.random.default_rng(42)
        cluster1 = rng.standard_normal((30, 2)) + np.array([0, 0])
        cluster2 = rng.standard_normal((30, 2)) + np.array([10, 10])
        X = np.vstack([cluster1, cluster2])

        _, labels = kmeans(X, k=2)
        assert len(np.unique(labels)) == 2


class TestPCA:
    """Tests para PCA."""

    def test_pca_output_shape(self) -> None:
        """Datos transformados deben tener n_components columnas."""
        from simulacro_CSCA5632 import pca

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))

        X_t, components, var_ratio = pca(X, n_components=3)
        assert X_t.shape == (100, 3)
        assert components.shape == (10, 3)
        assert len(var_ratio) == 3

    def test_pca_variance_positive(self) -> None:
        """Varianza explicada debe ser positiva."""
        from simulacro_CSCA5632 import pca

        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))

        _, _, var_ratio = pca(X, n_components=2)
        assert np.all(var_ratio >= 0)


class TestMatrixFactorization:
    """Tests para Matrix Factorization."""

    def test_mf_shapes(self) -> None:
        """P y Q deben tener shapes correctos."""
        from simulacro_CSCA5632 import matrix_factorization_sgd

        R = np.array(
            [
                [5, 3, np.nan],
                [4, np.nan, 2],
                [np.nan, 1, 5],
            ]
        )

        P, Q = matrix_factorization_sgd(R, k=2, n_epochs=10)
        assert P.shape == (3, 2)
        assert Q.shape == (3, 2)

    def test_predict_rating_scalar(self) -> None:
        """predict_rating debe retornar un escalar."""
        from simulacro_CSCA5632 import predict_rating

        P = np.array([[1, 2], [3, 4]])
        Q = np.array([[0.5, 0.5], [1, 1]])

        rating = predict_rating(P, Q, user_idx=0, item_idx=1)
        assert isinstance(rating, (int, float) | np.floating)
