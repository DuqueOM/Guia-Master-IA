"""
Tests para Simulacro CSCA 5622 - Supervised Learning
Ejecutar: pytest tests/test_simulacro_csca5622.py -v
"""

from __future__ import annotations

# Importar funciones del simulacro
import sys

import numpy as np
import pytest

sys.path.insert(0, "Herramientas_Estudio/Simulacros")


class TestLinearRegression:
    """Tests para regresión lineal."""

    def test_gradient_descent_convergence(self) -> None:
        """Verifica que GD converge a los pesos correctos."""
        from simulacro_CSCA5622 import linear_regression_gradient_descent

        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(100), rng.standard_normal((100, 2))])
        true_weights = np.array([1.0, 2.0, -1.0])
        y = X @ true_weights

        weights = linear_regression_gradient_descent(
            X, y, learning_rate=0.1, n_iterations=1000
        )
        np.testing.assert_allclose(weights, true_weights, atol=0.1)

    def test_weights_shape(self) -> None:
        """Verifica shape correcto de pesos."""
        from simulacro_CSCA5622 import linear_regression_gradient_descent

        rng = np.random.default_rng(42)
        X = np.column_stack([np.ones(50), rng.standard_normal((50, 3))])
        y = rng.standard_normal(50)

        weights = linear_regression_gradient_descent(X, y)
        assert weights.shape == (4,)


class TestLogisticRegression:
    """Tests para regresión logística."""

    def test_sigmoid_bounds(self) -> None:
        """Verifica que sigmoid está en [0, 1]."""
        from simulacro_CSCA5622 import sigmoid

        z = np.array([-100, -1, 0, 1, 100])
        result = sigmoid(z)

        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_sigmoid_center(self) -> None:
        """Verifica que sigmoid(0) = 0.5."""
        from simulacro_CSCA5622 import sigmoid

        result = sigmoid(np.array([0.0]))
        np.testing.assert_allclose(result, [0.5], atol=1e-10)

    def test_predict_binary(self) -> None:
        """Verifica que predict retorna 0 o 1."""
        from simulacro_CSCA5622 import predict

        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        weights = rng.standard_normal(3)

        preds = predict(X, weights)
        assert set(np.unique(preds)).issubset({0, 1})


class TestGiniImpurity:
    """Tests para Gini impurity."""

    def test_gini_pure(self) -> None:
        """Gini de clase pura debe ser 0."""
        from simulacro_CSCA5622 import gini_impurity

        y = np.array([1, 1, 1, 1, 1])
        assert gini_impurity(y) == pytest.approx(0.0, abs=1e-10)

    def test_gini_balanced(self) -> None:
        """Gini de 50/50 debe ser 0.5."""
        from simulacro_CSCA5622 import gini_impurity

        y = np.array([0, 0, 1, 1])
        assert gini_impurity(y) == pytest.approx(0.5, abs=1e-10)

    def test_gini_empty(self) -> None:
        """Gini de array vacío debe ser 0."""
        from simulacro_CSCA5622 import gini_impurity

        y = np.array([], dtype=np.int64)
        assert gini_impurity(y) == 0.0

    def test_information_gain_positive(self) -> None:
        """Information gain debe ser >= 0."""
        from simulacro_CSCA5622 import information_gain

        y_parent = np.array([0, 0, 1, 1, 1, 1])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1, 1, 1])

        ig = information_gain(y_parent, y_left, y_right)
        assert ig >= 0
