"""
Tests para Simulacro CSCA 5642 - Deep Learning
Ejecutar: pytest tests/test_simulacro_csca5642.py -v
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "Herramientas_Estudio/Simulacros")


class TestActivations:
    """Tests para funciones de activación."""

    def test_relu_positive(self) -> None:
        """ReLU de positivos debe ser identidad."""
        from simulacro_CSCA5642 import relu

        z = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(z), z)

    def test_relu_negative(self) -> None:
        """ReLU de negativos debe ser 0."""
        from simulacro_CSCA5642 import relu

        z = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(relu(z), np.zeros(3))

    def test_softmax_sums_to_one(self) -> None:
        """Softmax debe sumar 1."""
        from simulacro_CSCA5642 import softmax

        z = np.array([[1, 2, 3], [4, 5, 6]])
        result = softmax(z)
        np.testing.assert_allclose(result.sum(axis=1), [1, 1], atol=1e-10)

    def test_softmax_positive(self) -> None:
        """Softmax debe ser positivo."""
        from simulacro_CSCA5642 import softmax

        z = np.array([[-100, 0, 100]])
        result = softmax(z)
        assert np.all(result >= 0)


class TestMLPForward:
    """Tests para MLP forward pass."""

    def test_forward_shapes(self) -> None:
        """Shapes de activaciones deben ser correctos."""
        from simulacro_CSCA5642 import mlp_forward

        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 4))
        W1 = rng.standard_normal((4, 8))
        b1 = np.zeros(8)
        W2 = rng.standard_normal((8, 3))
        b2 = np.zeros(3)

        activations, _ = mlp_forward(X, [W1, W2], [b1, b2])

        assert activations[0].shape == (10, 4)  # input
        assert activations[1].shape == (10, 8)  # hidden
        assert activations[2].shape == (10, 3)  # output

    def test_forward_softmax_output(self) -> None:
        """Salida de forward debe ser probabilidades válidas."""
        from simulacro_CSCA5642 import mlp_forward

        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 3))
        W1 = rng.standard_normal((3, 4))
        b1 = np.zeros(4)
        W2 = rng.standard_normal((4, 2))
        b2 = np.zeros(2)

        activations, _ = mlp_forward(X, [W1, W2], [b1, b2])
        output = activations[-1]

        np.testing.assert_allclose(output.sum(axis=1), np.ones(5), atol=1e-10)


class TestCrossEntropy:
    """Tests para Cross-Entropy loss."""

    def test_ce_loss_positive(self) -> None:
        """Cross-entropy debe ser positivo."""
        from simulacro_CSCA5642 import cross_entropy_loss

        y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
        y_true = np.array([0, 1])

        loss = cross_entropy_loss(y_pred, y_true)
        assert loss > 0

    def test_ce_loss_perfect_prediction(self) -> None:
        """CE loss con predicción perfecta debe ser ~0."""
        from simulacro_CSCA5642 import cross_entropy_loss

        y_pred = np.array([[0.999, 0.001], [0.001, 0.999]])
        y_true = np.array([0, 1])

        loss = cross_entropy_loss(y_pred, y_true)
        assert loss < 0.01


class TestConv2D:
    """Tests para convolución 2D."""

    def test_conv2d_output_shape(self) -> None:
        """Shape de salida de conv2d debe ser correcto."""
        from simulacro_CSCA5642 import conv2d_forward

        rng = np.random.default_rng(42)
        X = rng.standard_normal((2, 8, 8, 3))
        W = rng.standard_normal((3, 3, 3, 16))
        b = np.zeros(16)

        out = conv2d_forward(X, W, b, stride=1, padding=0)
        assert out.shape == (2, 6, 6, 16)

    def test_conv2d_with_padding(self) -> None:
        """Conv2d con padding=1 debe mantener tamaño."""
        from simulacro_CSCA5642 import conv2d_forward

        rng = np.random.default_rng(42)
        X = rng.standard_normal((1, 5, 5, 1))
        W = rng.standard_normal((3, 3, 1, 4))
        b = np.zeros(4)

        out = conv2d_forward(X, W, b, stride=1, padding=1)
        assert out.shape == (1, 5, 5, 4)

    def test_maxpool_output_shape(self) -> None:
        """Shape de max pooling debe ser correcto."""
        from simulacro_CSCA5642 import max_pool2d

        rng = np.random.default_rng(42)
        X = rng.standard_normal((2, 8, 8, 4))

        out = max_pool2d(X, pool_size=2, stride=2)
        assert out.shape == (2, 4, 4, 4)
