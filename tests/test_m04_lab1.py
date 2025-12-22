#!/usr/bin/env python3
"""
Tests para M04 Lab 1: MLE y MAP
================================

Verifica implementaciones de:
- MLE para Bernoulli y Gaussiana
- MAP con priors conjugados (Beta-Binomial)
- Análisis sesgo-varianza
"""
from __future__ import annotations

import numpy as np
import pytest

from M04_Probabilidad_Estadistica.Notebooks.Lab1_MLE_MAP import (
    map_bernoulli,
    mle_bernoulli,
    mle_gaussian,
    simulate_estimator_bias_variance,
)


class TestMLEBernoulli:
    """Tests para MLE de distribución Bernoulli."""

    def test_mle_all_ones(self) -> None:
        """MLE de datos todos 1 debe ser 1.0."""
        data = np.array([1, 1, 1, 1, 1])
        assert mle_bernoulli(data) == pytest.approx(1.0)

    def test_mle_all_zeros(self) -> None:
        """MLE de datos todos 0 debe ser 0.0."""
        data = np.array([0, 0, 0, 0, 0])
        assert mle_bernoulli(data) == pytest.approx(0.0)

    def test_mle_mixed(self) -> None:
        """MLE debe ser la proporción de 1s."""
        data = np.array([1, 1, 0, 1, 0])  # 3/5 = 0.6
        assert mle_bernoulli(data) == pytest.approx(0.6)

    def test_mle_single_observation(self) -> None:
        """MLE con un solo dato."""
        assert mle_bernoulli(np.array([1])) == pytest.approx(1.0)
        assert mle_bernoulli(np.array([0])) == pytest.approx(0.0)

    def test_mle_large_sample(self) -> None:
        """MLE converge al parámetro verdadero con n grande."""
        rng = np.random.default_rng(42)
        true_theta = 0.7
        data = rng.binomial(1, true_theta, 10000)
        estimated = mle_bernoulli(data)
        assert estimated == pytest.approx(true_theta, abs=0.02)


class TestMLEGaussian:
    """Tests para MLE de distribución Gaussiana."""

    def test_mle_gaussian_mean(self) -> None:
        """MLE de media debe ser la media muestral."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu, _ = mle_gaussian(data)
        assert mu == pytest.approx(3.0)

    def test_mle_gaussian_variance(self) -> None:
        """MLE de varianza debe ser varianza muestral (sesgada, /n)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, sigma2 = mle_gaussian(data)
        # Varianza poblacional: sum((x - mean)^2) / n
        expected_var = np.var(data, ddof=0)
        assert sigma2 == pytest.approx(expected_var)

    def test_mle_gaussian_convergence(self) -> None:
        """MLE converge a parámetros verdaderos con n grande."""
        rng = np.random.default_rng(42)
        true_mu, true_sigma2 = 5.0, 4.0
        data = rng.normal(true_mu, np.sqrt(true_sigma2), 10000)
        mu, sigma2 = mle_gaussian(data)
        assert mu == pytest.approx(true_mu, abs=0.1)
        assert sigma2 == pytest.approx(true_sigma2, abs=0.2)


class TestMAPBernoulli:
    """Tests para MAP de distribución Bernoulli con prior Beta."""

    def test_map_uniform_prior(self) -> None:
        """MAP con prior uniforme (Beta(1,1)) es cercano a MLE."""
        data = np.array([1, 1, 0, 1, 0])  # 3/5
        map_est = map_bernoulli(data, alpha=1, beta=1)
        # MAP = (k + alpha - 1) / (n + alpha + beta - 2) para prior Beta
        # Con prior uniforme y datos suficientes, MAP ≈ MLE
        assert map_est == pytest.approx(0.6, abs=0.1)

    def test_map_strong_prior(self) -> None:
        """MAP con prior fuerte se acerca al prior."""
        data = np.array([1, 1, 1])  # MLE = 1.0
        # Prior fuerte centrado en 0.5: Beta(100, 100)
        map_est = map_bernoulli(data, alpha=100, beta=100)
        # MAP debe estar más cerca de 0.5 que de 1.0
        assert map_est < 0.6
        assert map_est > 0.4

    def test_map_no_data_returns_prior_mode(self) -> None:
        """MAP sin datos debe devolver la moda del prior."""
        data = np.array([], dtype=int)
        # Prior Beta(5, 3): moda = (5-1)/(5+3-2) = 4/6 ≈ 0.667
        map_est = map_bernoulli(data, alpha=5, beta=3)
        expected_mode = (5 - 1) / (5 + 3 - 2)
        assert map_est == pytest.approx(expected_mode, abs=0.01)

    def test_map_regularization(self) -> None:
        """MAP con prior actúa como regularización."""
        # Datos extremos: todos 1
        data = np.ones(5, dtype=int)
        mle = mle_bernoulli(data)  # = 1.0
        map_est = map_bernoulli(data, alpha=2, beta=2)  # Prior empuja hacia 0.5
        assert map_est < mle  # MAP debe ser menor que 1.0


class TestNegativeLogLikelihood:
    """Tests para negative log-likelihood Gaussiana (verificación conceptual)."""

    def test_nll_concept(self) -> None:
        """Verificar que MLE minimiza NLL implícitamente."""
        rng = np.random.default_rng(42)
        true_mu, true_sigma2 = 5.0, 4.0
        data = rng.normal(true_mu, np.sqrt(true_sigma2), 100)

        # MLE debe aproximar parámetros verdaderos
        mu_mle, sigma2_mle = mle_gaussian(data)
        assert mu_mle == pytest.approx(true_mu, abs=0.5)
        assert sigma2_mle == pytest.approx(true_sigma2, abs=1.0)


class TestBiasVarianceSimulation:
    """Tests para simulación de sesgo-varianza."""

    def test_mle_unbiased(self) -> None:
        """MLE de Bernoulli es insesgado."""
        bias_mle, _, _, _ = simulate_estimator_bias_variance(
            true_theta=0.5,
            n_samples=100,
            n_simulations=1000,
            alpha=2,
            beta=2,
        )
        # Sesgo de MLE debe ser cercano a 0
        assert abs(bias_mle) < 0.02

    def test_map_has_bias_with_prior(self) -> None:
        """MAP tiene sesgo cuando prior difiere de theta verdadero."""
        # theta = 0.8, prior centrado en 0.5
        _, _, bias_map, _ = simulate_estimator_bias_variance(
            true_theta=0.8,
            n_samples=20,
            n_simulations=1000,
            alpha=10,
            beta=10,
        )
        # MAP debe tener sesgo negativo (hacia 0.5)
        assert bias_map < 0

    def test_map_lower_variance(self) -> None:
        """MAP generalmente tiene menor varianza que MLE."""
        _, var_mle, _, var_map = simulate_estimator_bias_variance(
            true_theta=0.5,
            n_samples=20,
            n_simulations=1000,
            alpha=5,
            beta=5,
        )
        # Con prior informativo, MAP tiene menor varianza
        assert var_map <= var_mle


class TestReturnTypes:
    """Tests para verificar tipos de retorno correctos."""

    def test_mle_bernoulli_returns_float(self) -> None:
        """mle_bernoulli debe retornar float."""
        result = mle_bernoulli(np.array([1, 0, 1]))
        assert isinstance(result, float)

    def test_mle_gaussian_returns_floats(self) -> None:
        """mle_gaussian debe retornar tupla de floats."""
        mu, sigma2 = mle_gaussian(np.array([1.0, 2.0, 3.0]))
        assert isinstance(mu, float)
        assert isinstance(sigma2, float)

    def test_map_bernoulli_returns_float(self) -> None:
        """map_bernoulli debe retornar float."""
        result = map_bernoulli(np.array([1, 0, 1]), alpha=2, beta=2)
        assert isinstance(result, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
