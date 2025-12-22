#!/usr/bin/env python3
"""
Tests para M04 Lab 2: Monte Carlo y MCMC
=========================================

Verifica implementaciones de:
- Integración Monte Carlo
- Metropolis-Hastings
- Gibbs Sampling
- Diagnósticos de convergencia (ESS, R-hat)
"""
from __future__ import annotations

import numpy as np
import pytest

from M04_Probabilidad_Estadistica.Notebooks.Lab2_MonteCarlo_MCMC import (
    effective_sample_size,
    gelman_rubin_rhat,
    gibbs_sampling_bivariate_normal,
    metropolis_hastings,
    monte_carlo_integration,
)


def func_constant(x: np.ndarray) -> np.ndarray:
    """f(x) = 1."""
    return np.ones_like(x)


def func_linear(x: np.ndarray) -> np.ndarray:
    """f(x) = x."""
    return x


def func_quadratic(x: np.ndarray) -> np.ndarray:
    """f(x) = x^2."""
    return x**2


def make_covariance_matrix(std1: float, std2: float, rho: float) -> np.ndarray:
    cov12 = rho * std1 * std2
    return np.array([[std1**2, cov12], [cov12, std2**2]])


class TestMonteCarloIntegration:
    """Tests para integración Monte Carlo."""

    def test_uniform_integration(self) -> None:
        """Integral de f(x) = 1 en [0, 1] debe ser 1."""
        result, _ = monte_carlo_integration(
            f=func_constant,
            a=0,
            b=1,
            n_samples=10000,
        )
        assert result == pytest.approx(1.0, abs=0.05)

    def test_linear_integration(self) -> None:
        """Integral de f(x) = x en [0, 1] debe ser 0.5."""
        result, _ = monte_carlo_integration(
            f=func_linear,
            a=0,
            b=1,
            n_samples=10000,
        )
        assert result == pytest.approx(0.5, abs=0.05)

    def test_quadratic_integration(self) -> None:
        """Integral de f(x) = x² en [0, 1] debe ser 1/3."""
        result, _ = monte_carlo_integration(
            f=func_quadratic,
            a=0,
            b=1,
            n_samples=10000,
        )
        assert result == pytest.approx(1 / 3, abs=0.05)

    def test_sine_integration(self) -> None:
        """Integral de sin(x) en [0, π] debe ser 2."""
        result, _ = monte_carlo_integration(
            f=np.sin,
            a=0,
            b=np.pi,
            n_samples=10000,
        )
        assert result == pytest.approx(2.0, abs=0.1)

    def test_returns_std_error(self) -> None:
        """Debe retornar error estándar positivo."""
        _, std_err = monte_carlo_integration(
            f=func_linear,
            a=0,
            b=1,
            n_samples=1000,
        )
        assert std_err > 0

    def test_std_error_decreases_with_n(self) -> None:
        """Error estándar debe disminuir con más muestras."""
        _, std_err_small = monte_carlo_integration(
            f=func_quadratic,
            a=0,
            b=1,
            n_samples=100,
        )
        _, std_err_large = monte_carlo_integration(
            f=func_quadratic,
            a=0,
            b=1,
            n_samples=10000,
        )
        assert std_err_large < std_err_small


def log_standard_normal(x: float) -> float:
    """Log de N(0, 1) para tests."""
    return -0.5 * x**2


class TestMetropolisHastings:
    """Tests para algoritmo Metropolis-Hastings."""

    def test_samples_shape(self) -> None:
        """Debe retornar número correcto de muestras."""
        samples, _ = metropolis_hastings(
            log_standard_normal,
            proposal_std=1.0,
            n_samples=500,
            x_init=0.0,
            burn_in=100,
        )
        assert len(samples) == 500

    def test_acceptance_rate_in_range(self) -> None:
        """Tasa de aceptación debe estar en [0, 1]."""
        _, acceptance_rate = metropolis_hastings(
            log_standard_normal,
            proposal_std=1.0,
            n_samples=1000,
            x_init=0.0,
            burn_in=100,
        )
        assert 0 <= acceptance_rate <= 1

    def test_samples_gaussian_mean(self) -> None:
        """Muestras de N(0, 1) deben tener media ≈ 0."""
        samples, _ = metropolis_hastings(
            log_standard_normal,
            proposal_std=1.0,
            n_samples=5000,
            x_init=0.0,
            burn_in=500,
        )
        assert np.mean(samples) == pytest.approx(0.0, abs=0.15)

    def test_samples_gaussian_std(self) -> None:
        """Muestras de N(0, 1) deben tener std ≈ 1."""
        samples, _ = metropolis_hastings(
            log_standard_normal,
            proposal_std=1.0,
            n_samples=5000,
            x_init=0.0,
            burn_in=500,
        )
        assert np.std(samples) == pytest.approx(1.0, abs=0.15)

    def test_proposal_std_affects_acceptance(self) -> None:
        """Proposal std grande debe reducir tasa de aceptación."""
        _, acc_small = metropolis_hastings(
            log_standard_normal,
            proposal_std=0.5,
            n_samples=1000,
            x_init=0.0,
            burn_in=100,
        )
        _, acc_large = metropolis_hastings(
            log_standard_normal,
            proposal_std=5.0,
            n_samples=1000,
            x_init=0.0,
            burn_in=100,
        )

        # Proposal más grande generalmente tiene menor aceptación
        assert acc_large < acc_small

    def test_burn_in_removes_initial_samples(self) -> None:
        """Burn-in debe eliminar muestras iniciales."""
        samples, _ = metropolis_hastings(
            log_standard_normal,
            proposal_std=1.0,
            n_samples=100,
            x_init=10.0,  # Inicio lejos de la moda
            burn_in=500,
        )
        # Después de burn-in, muestras no deben estar en 10
        assert np.mean(samples) < 5


class TestGibbsSampling:
    """Tests para Gibbs Sampling bivariado."""

    def test_samples_shape(self) -> None:
        """Debe retornar forma correcta (n_samples, 2)."""
        samples = gibbs_sampling_bivariate_normal(
            mu=np.array([0.0, 0.0]),
            sigma=make_covariance_matrix(1.0, 1.0, rho=0.5),
            n_samples=500,
            burn_in=100,
        )
        assert samples.shape == (500, 2)

    def test_marginal_means(self) -> None:
        """Medias marginales deben aproximar mu."""
        samples = gibbs_sampling_bivariate_normal(
            mu=np.array([2.0, -1.0]),
            sigma=make_covariance_matrix(1.0, 1.0, rho=0.5),
            n_samples=5000,
            burn_in=500,
        )
        assert np.mean(samples[:, 0]) == pytest.approx(2.0, abs=0.15)
        assert np.mean(samples[:, 1]) == pytest.approx(-1.0, abs=0.15)

    def test_marginal_stds(self) -> None:
        """Desviaciones estándar marginales deben aproximar sigma."""
        samples = gibbs_sampling_bivariate_normal(
            mu=np.array([0.0, 0.0]),
            sigma=make_covariance_matrix(2.0, 0.5, rho=0.0),
            n_samples=5000,
            burn_in=500,
        )
        assert np.std(samples[:, 0]) == pytest.approx(2.0, abs=0.2)
        assert np.std(samples[:, 1]) == pytest.approx(0.5, abs=0.1)

    def test_correlation(self) -> None:
        """Correlación muestral debe aproximar rho."""
        rho_true = 0.7
        samples = gibbs_sampling_bivariate_normal(
            mu=np.array([0.0, 0.0]),
            sigma=make_covariance_matrix(1.0, 1.0, rho=rho_true),
            n_samples=5000,
            burn_in=500,
        )
        rho_sample = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        assert rho_sample == pytest.approx(rho_true, abs=0.1)


class TestEffectiveSampleSize:
    """Tests para Effective Sample Size (ESS)."""

    def test_ess_independent_samples(self) -> None:
        """ESS de muestras independientes ≈ n."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 1000)
        ess = effective_sample_size(samples)
        # ESS debe ser cercano a n para muestras iid
        assert ess > 800  # Al menos 80% de n

    def test_ess_correlated_samples(self) -> None:
        """ESS de muestras correlacionadas < n."""
        # Crear muestras autocorrelacionadas (random walk)
        rng = np.random.default_rng(42)
        n = 1000
        samples = np.zeros(n)
        samples[0] = 0
        for i in range(1, n):
            samples[i] = 0.95 * samples[i - 1] + rng.normal(0, 0.1)

        ess = effective_sample_size(samples)
        assert ess < n * 0.5  # ESS debe ser significativamente menor que n

    def test_ess_positive(self) -> None:
        """ESS debe ser positivo."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ess = effective_sample_size(samples)
        assert ess > 0

    def test_ess_bounded_by_n(self) -> None:
        """ESS no debe exceder n."""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 100)
        ess = effective_sample_size(samples)
        assert ess <= 100


class TestGelmanRubinRhat:
    """Tests para diagnóstico Gelman-Rubin R-hat."""

    def test_rhat_converged_chains(self) -> None:
        """R-hat de cadenas convergidas debe ser ≈ 1."""
        rng = np.random.default_rng(42)
        # 4 cadenas de N(0, 1) - ya convergidas
        chains = [rng.normal(0, 1, 1000) for _ in range(4)]
        rhat = gelman_rubin_rhat(chains)
        assert rhat == pytest.approx(1.0, abs=0.05)

    def test_rhat_non_converged_chains(self) -> None:
        """R-hat de cadenas no convergidas debe ser > 1."""
        rng = np.random.default_rng(42)
        # Cadenas con medias diferentes (no convergidas)
        chains = [
            rng.normal(0, 1, 500),
            rng.normal(3, 1, 500),
            rng.normal(-3, 1, 500),
            rng.normal(6, 1, 500),
        ]
        rhat = gelman_rubin_rhat(chains)
        assert rhat > 1.1

    def test_rhat_returns_float(self) -> None:
        """gelman_rubin_rhat debe retornar float."""
        rng = np.random.default_rng(42)
        chains = [rng.normal(0, 1, 100) for _ in range(4)]
        rhat = gelman_rubin_rhat(chains)
        assert isinstance(rhat, float)


def identity_func(x: np.ndarray) -> np.ndarray:
    """Función identidad para tests."""
    return x


class TestReturnTypes:
    """Tests para verificar tipos de retorno."""

    def test_monte_carlo_returns_floats(self) -> None:
        """monte_carlo_integration debe retornar tupla de floats."""
        result, std_err = monte_carlo_integration(
            f=identity_func, a=0, b=1, n_samples=100
        )
        assert isinstance(result, float)
        assert isinstance(std_err, float)

    def test_mh_returns_array_and_float(self) -> None:
        """metropolis_hastings debe retornar array y float."""
        samples, acc_rate = metropolis_hastings(
            log_standard_normal, 1.0, n_samples=100, x_init=0.0, burn_in=10
        )
        assert isinstance(samples, np.ndarray)
        assert isinstance(acc_rate, float)

    def test_gibbs_returns_array(self) -> None:
        """gibbs_sampling_bivariate_normal debe retornar NDArray."""
        samples = gibbs_sampling_bivariate_normal(
            mu=np.array([0.0, 0.0]),
            sigma=make_covariance_matrix(1.0, 1.0, rho=0.5),
            n_samples=100,
            burn_in=10,
        )
        assert isinstance(samples, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
