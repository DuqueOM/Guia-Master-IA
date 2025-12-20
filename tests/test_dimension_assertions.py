"""
Tests para validar dimensiones de matrices en operaciones de ML.
Ejecutar: pytest tests/test_dimension_assertions.py -v
"""

import numpy as np
import pytest


class TestDimensionAssertions:
    """Unit tests agresivos para dimensiones - el 90% de errores en DL son de shapes."""

    def test_matrix_multiplication_dimensions(self):
        """Verifica que matmul produce las dimensiones esperadas."""
        rng = np.random.default_rng(0)
        A = rng.standard_normal((3, 4))
        B = rng.standard_normal((4, 5))
        C = A @ B
        assert C.shape == (3, 5), f"Expected (3, 5), got {C.shape}"

    def test_broadcasting_addition(self):
        """Verifica broadcasting en suma."""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((3, 4))
        b = rng.standard_normal(4)  # Vector 1D
        C = A + b
        assert C.shape == (3, 4), f"Expected (3, 4), got {C.shape}"

    def test_broadcasting_column_vector(self):
        """Verifica broadcasting con vector columna."""
        rng = np.random.default_rng(2)
        A = rng.standard_normal((3, 4))
        b = rng.standard_normal((3, 1))  # Vector columna
        C = A + b
        assert C.shape == (3, 4), f"Expected (3, 4), got {C.shape}"

    def test_forward_pass_dimensions(self):
        """Verifica dimensiones en forward pass de red neuronal."""
        batch_size = 32
        input_size = 784
        hidden_size = 128
        output_size = 10

        # Datos
        rng = np.random.default_rng(3)
        X = rng.standard_normal((batch_size, input_size))

        # Pesos
        W1 = rng.standard_normal((input_size, hidden_size))
        b1 = np.zeros((1, hidden_size))
        W2 = rng.standard_normal((hidden_size, output_size))
        b2 = np.zeros((1, output_size))

        # Forward pass
        Z1 = X @ W1 + b1
        assert Z1.shape == (
            batch_size,
            hidden_size,
        ), f"Z1: Expected {(batch_size, hidden_size)}, got {Z1.shape}"

        A1 = np.maximum(0, Z1)  # ReLU
        assert A1.shape == Z1.shape, "A1 shape mismatch"

        Z2 = A1 @ W2 + b2
        assert Z2.shape == (
            batch_size,
            output_size,
        ), f"Z2: Expected {(batch_size, output_size)}, got {Z2.shape}"

    def test_backward_pass_dimensions(self):
        """Verifica que gradientes tienen misma shape que parámetros."""
        batch_size = 32
        input_size = 784
        hidden_size = 128
        output_size = 10

        # Forward (simplificado)
        rng = np.random.default_rng(4)
        X = rng.standard_normal((batch_size, input_size))
        W1 = rng.standard_normal((input_size, hidden_size))
        W2 = rng.standard_normal((hidden_size, output_size))

        Z1 = X @ W1
        A1 = np.maximum(0, Z1)
        _ = A1 @ W2

        # Backward
        dZ2 = rng.standard_normal((batch_size, output_size))  # Gradiente de loss

        # Gradientes
        dW2 = A1.T @ dZ2
        assert dW2.shape == W2.shape, f"dW2: Expected {W2.shape}, got {dW2.shape}"

        dA1 = dZ2 @ W2.T
        assert dA1.shape == A1.shape, f"dA1: Expected {A1.shape}, got {dA1.shape}"

        dZ1 = dA1 * (Z1 > 0)  # ReLU derivative
        assert dZ1.shape == Z1.shape, "dZ1 shape mismatch"

        dW1 = X.T @ dZ1
        assert dW1.shape == W1.shape, f"dW1: Expected {W1.shape}, got {dW1.shape}"

    def test_softmax_dimensions(self):
        """Verifica que softmax mantiene dimensiones."""
        batch_size = 32
        num_classes = 10

        rng = np.random.default_rng(5)
        logits = rng.standard_normal((batch_size, num_classes))

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        assert probs.shape == logits.shape, "Softmax changed shape"
        assert np.allclose(probs.sum(axis=1), 1.0), "Softmax rows don't sum to 1"

    def test_reduction_with_keepdims(self):
        """Verifica comportamiento de keepdims."""
        rng = np.random.default_rng(6)
        A = rng.standard_normal((3, 4, 5))

        # Sin keepdims
        s1 = np.sum(A, axis=1)
        assert s1.shape == (3, 5), f"Expected (3, 5), got {s1.shape}"

        # Con keepdims
        s2 = np.sum(A, axis=1, keepdims=True)
        assert s2.shape == (3, 1, 5), f"Expected (3, 1, 5), got {s2.shape}"

    def test_reshape_consistency(self):
        """Verifica que reshape preserva elementos."""
        A = np.arange(24)

        B = A.reshape(4, 6)
        assert B.shape == (4, 6)
        assert B.size == A.size

        C = A.reshape(2, 3, 4)
        assert C.shape == (2, 3, 4)
        assert C.size == A.size

        D = A.reshape(-1, 6)
        assert D.shape == (4, 6)

    def test_batch_gradient_dimensions(self):
        """Verifica dimensiones de gradientes con batch."""
        batch_size = 64
        features = 100

        rng = np.random.default_rng(7)
        X = rng.standard_normal((batch_size, features))
        W = rng.standard_normal((features, 1))
        b = np.zeros((1, 1))

        # Forward
        y_pred = X @ W + b
        assert y_pred.shape == (batch_size, 1)

        # Loss gradient (asumiendo MSE)
        y_true = rng.standard_normal((batch_size, 1))
        dL_dy = 2 * (y_pred - y_true) / batch_size
        assert dL_dy.shape == (batch_size, 1)

        # Gradiente de W
        dW = X.T @ dL_dy
        assert dW.shape == W.shape, f"dW: Expected {W.shape}, got {dW.shape}"

        # Gradiente de b (suma sobre batch)
        db = np.sum(dL_dy, axis=0, keepdims=True)
        assert db.shape == b.shape, f"db: Expected {b.shape}, got {db.shape}"


class TestNumericalGradient:
    """Verificación numérica de gradientes."""

    def test_numerical_gradient_simple(self):
        """Verifica gradiente con diferencias finitas."""

        def f(x):
            return x**2

        def df(x):
            return 2 * x

        x = 3.0
        eps = 1e-5

        numerical_grad = (f(x + eps) - f(x - eps)) / (2 * eps)
        analytical_grad = df(x)

        assert (
            abs(numerical_grad - analytical_grad) < 1e-4
        ), f"Gradients don't match: numerical={numerical_grad}, analytical={analytical_grad}"

    def test_numerical_gradient_multivariate(self):
        """Verifica gradiente multivariado."""

        def f(x, y):
            return x**2 + x * y + y**2

        def grad_f(x, y):
            return np.array([2 * x + y, x + 2 * y])

        x, y = 2.0, 3.0
        eps = 1e-5

        # Gradiente numérico
        num_grad_x = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)
        num_grad_y = (f(x, y + eps) - f(x, y - eps)) / (2 * eps)
        numerical_grad = np.array([num_grad_x, num_grad_y])

        # Gradiente analítico
        analytical_grad = grad_f(x, y)

        assert np.allclose(
            numerical_grad, analytical_grad, atol=1e-4
        ), "Gradients don't match"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
