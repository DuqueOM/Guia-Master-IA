#!/usr/bin/env python3
"""
Tests para M04 Lab 3: Cadenas de Markov
========================================

Verifica implementaciones de:
- Matrices estocásticas
- Distribución estacionaria (eigenvalores, potencia, simulación)
- Irreducibilidad y aperiodicidad
- Distancia de variación total y tiempo de mezcla
- PageRank
"""
from __future__ import annotations

import numpy as np
import pytest

from M04_Probabilidad_Estadistica.Notebooks.Lab3_MarkovChains import (
    estimate_mixing_time,
    is_irreducible,
    pagerank,
    simulate_markov_chain,
    stationary_distribution_analytical,
    stationary_distribution_power,
    stationary_distribution_simulation,
    total_variation_distance,
    verify_stochastic_matrix,
)


class TestStochasticMatrix:
    """Tests para verificación de matrices estocásticas."""

    def test_valid_stochastic_matrix(self) -> None:
        """Matriz estocástica válida debe pasar verificación."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        assert verify_stochastic_matrix(P) is True

    def test_invalid_negative_entries(self) -> None:
        """Matriz con entradas negativas no es estocástica."""
        P = np.array([[0.5, 0.5], [-0.1, 1.1]])
        assert verify_stochastic_matrix(P) is False

    def test_invalid_rows_not_sum_one(self) -> None:
        """Matriz cuyas filas no suman 1 no es estocástica."""
        P = np.array([[0.5, 0.4], [0.3, 0.7]])
        assert verify_stochastic_matrix(P) is False

    def test_identity_matrix(self) -> None:
        """Matriz identidad es estocástica (cadenas absorbentes)."""
        P = np.eye(3)
        assert verify_stochastic_matrix(P) is True

    def test_uniform_matrix(self) -> None:
        """Matriz uniforme (1/n) es estocástica."""
        n = 4
        P = np.ones((n, n)) / n
        assert verify_stochastic_matrix(P) is True


class TestStationaryDistributionEigen:
    """Tests para distribución estacionaria via eigenvalores."""

    def test_stationary_sums_to_one(self) -> None:
        """Distribución estacionaria debe sumar 1."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        pi = stationary_distribution_analytical(P)
        assert np.sum(pi) == pytest.approx(1.0)

    def test_stationary_non_negative(self) -> None:
        """Distribución estacionaria debe ser no negativa."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        pi = stationary_distribution_analytical(P)
        assert np.all(pi >= 0)

    def test_stationary_is_fixed_point(self) -> None:
        """πP = π para distribución estacionaria."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        pi = stationary_distribution_analytical(P)
        assert np.allclose(pi @ P, pi)

    def test_uniform_chain_stationary(self) -> None:
        """Cadena uniforme tiene distribución estacionaria uniforme."""
        n = 4
        P = np.ones((n, n)) / n
        pi = stationary_distribution_analytical(P)
        expected = np.ones(n) / n
        assert np.allclose(pi, expected, atol=1e-6)

    def test_doubly_stochastic_uniform(self) -> None:
        """Matriz doblemente estocástica tiene π uniforme."""
        P = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5]])
        pi = stationary_distribution_analytical(P)
        expected = np.ones(3) / 3
        assert np.allclose(pi, expected, atol=1e-6)


class TestStationaryDistributionPower:
    """Tests para distribución estacionaria via potencias."""

    def test_power_matches_eigen(self) -> None:
        """Método de potencia debe dar mismo resultado que eigenvalores."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        pi_eigen = stationary_distribution_analytical(P)
        pi_power = stationary_distribution_power(P, n_iterations=100)
        assert np.allclose(pi_eigen, pi_power, atol=1e-4)

    def test_power_converges(self) -> None:
        """Más iteraciones deben dar mejor aproximación."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        pi_eigen = stationary_distribution_analytical(P)
        pi_few = stationary_distribution_power(P, n_iterations=10)
        pi_many = stationary_distribution_power(P, n_iterations=100)

        err_few = np.linalg.norm(pi_few - pi_eigen)
        err_many = np.linalg.norm(pi_many - pi_eigen)
        assert err_many <= err_few


class TestStationaryDistributionSimulation:
    """Tests para distribución estacionaria via simulación."""

    def test_simulation_approximates_stationary(self) -> None:
        """Simulación larga debe aproximar distribución estacionaria."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        pi_eigen = stationary_distribution_analytical(P)
        pi_sim = stationary_distribution_simulation(P, n_steps=50000, initial_state=0)
        assert np.allclose(pi_eigen, pi_sim, atol=0.05)

    def test_simulation_sums_to_one(self) -> None:
        """Distribución simulada debe sumar 1."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        pi_sim = stationary_distribution_simulation(P, n_steps=1000, initial_state=0)
        assert np.sum(pi_sim) == pytest.approx(1.0)


class TestSimulateMarkovChain:
    """Tests para simulación de cadenas de Markov."""

    def test_chain_length(self) -> None:
        """Cadena debe tener longitud correcta."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        chain = simulate_markov_chain(P, initial_state=0, n_steps=100)
        assert len(chain) == 101  # n_steps + 1 (incluye estado inicial)

    def test_chain_starts_at_initial(self) -> None:
        """Cadena debe empezar en estado inicial."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        chain = simulate_markov_chain(P, initial_state=1, n_steps=10)
        assert chain[0] == 1

    def test_chain_valid_states(self) -> None:
        """Todos los estados deben ser válidos."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        chain = simulate_markov_chain(P, initial_state=0, n_steps=100)
        assert np.all((chain >= 0) & (chain < 2))

    def test_absorbing_state(self) -> None:
        """Estado absorbente debe permanecer absorbido."""
        P = np.array([[1.0, 0.0], [0.5, 0.5]])  # Estado 0 es absorbente
        chain = simulate_markov_chain(P, initial_state=0, n_steps=10)
        assert np.all(chain == 0)


class TestIrreducibility:
    """Tests para verificación de irreducibilidad."""

    def test_irreducible_chain(self) -> None:
        """Cadena completamente conectada es irreducible."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        assert is_irreducible(P) is True

    def test_reducible_chain(self) -> None:
        """Cadena con estados absorbentes es reducible."""
        P = np.array([[1.0, 0.0], [0.0, 1.0]])  # Dos absorbentes
        assert is_irreducible(P) is False

    def test_cyclic_irreducible(self) -> None:
        """Cadena cíclica es irreducible."""
        P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        assert is_irreducible(P) is True


class TestTotalVariationDistance:
    """Tests para distancia de variación total."""

    def test_same_distribution(self) -> None:
        """TV entre distribuciones iguales es 0."""
        p = np.array([0.3, 0.7])
        tv = total_variation_distance(p, p)
        assert tv == pytest.approx(0.0)

    def test_disjoint_distributions(self) -> None:
        """TV entre distribuciones disjuntas es 1."""
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        tv = total_variation_distance(p, q)
        assert tv == pytest.approx(1.0)

    def test_tv_symmetric(self) -> None:
        """TV es simétrica: TV(p, q) = TV(q, p)."""
        p = np.array([0.3, 0.7])
        q = np.array([0.5, 0.5])
        assert total_variation_distance(p, q) == pytest.approx(
            total_variation_distance(q, p)
        )

    def test_tv_bounded(self) -> None:
        """TV está en [0, 1]."""
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.4, 0.4, 0.2])
        tv = total_variation_distance(p, q)
        assert 0 <= tv <= 1


class TestMixingTime:
    """Tests para tiempo de mezcla."""

    def test_mixing_time_positive(self) -> None:
        """Tiempo de mezcla debe ser positivo."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        mixing_time, _ = estimate_mixing_time(P, epsilon=0.01)
        assert mixing_time > 0

    def test_fast_mixing_chain(self) -> None:
        """Cadena rápida (uniforme) debe mezclar rápido."""
        n = 3
        P = np.ones((n, n)) / n  # Mezcla en un paso
        mixing_time, _ = estimate_mixing_time(P, epsilon=0.01)
        assert mixing_time <= 2

    def test_returns_distances(self) -> None:
        """Debe retornar lista de distancias."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        _, distances = estimate_mixing_time(P, epsilon=0.01, max_steps=50)
        assert len(distances) == 50
        assert all(d >= 0 for d in distances)


class TestPageRank:
    """Tests para algoritmo PageRank."""

    def test_pagerank_sums_to_one(self) -> None:
        """PageRank debe sumar 1."""
        # Red simple: A → B, B → A
        adj = np.array([[0, 1], [1, 0]])
        ranks = pagerank(adj, damping=0.85)
        assert np.sum(ranks) == pytest.approx(1.0)

    def test_pagerank_symmetric_graph(self) -> None:
        """Grafo simétrico debe tener PageRank uniforme."""
        # Grafo completo de 3 nodos
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        ranks = pagerank(adj, damping=0.85)
        expected = np.ones(3) / 3
        assert np.allclose(ranks, expected, atol=0.01)

    def test_pagerank_sink_handled(self) -> None:
        """Nodos sink (sin salidas) deben manejarse."""
        # A → B, B es sink
        adj = np.array([[0, 1], [0, 0]])
        ranks = pagerank(adj, damping=0.85)
        assert np.sum(ranks) == pytest.approx(1.0)
        assert np.all(ranks > 0)

    def test_pagerank_hub_has_low_rank(self) -> None:
        """Nodo hub (muchas salidas, pocas entradas) tiene menor rank."""
        # Hub A apunta a B, C, D; todos apuntan de vuelta a A
        adj = np.array(
            [
                [0, 1, 1, 1],  # A → B, C, D
                [1, 0, 0, 0],  # B → A
                [1, 0, 0, 0],  # C → A
                [1, 0, 0, 0],  # D → A
            ]
        )
        ranks = pagerank(adj, damping=0.85)
        # A debe tener el mayor rank (recibe 3 enlaces)
        assert ranks[0] == max(ranks)


class TestReturnTypes:
    """Tests para verificar tipos de retorno."""

    def test_verify_stochastic_returns_bool(self) -> None:
        """verify_stochastic_matrix debe retornar bool."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        result = verify_stochastic_matrix(P)
        assert isinstance(result, bool)

    def test_stationary_returns_ndarray(self) -> None:
        """Funciones de distribución estacionaria deben retornar NDArray."""
        P = np.array([[0.5, 0.5], [0.3, 0.7]])
        assert isinstance(stationary_distribution_analytical(P), np.ndarray)
        assert isinstance(stationary_distribution_power(P), np.ndarray)
        assert isinstance(stationary_distribution_simulation(P, 100, 0), np.ndarray)

    def test_tv_returns_float(self) -> None:
        """total_variation_distance debe retornar float."""
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.7])
        result = total_variation_distance(p, q)
        assert isinstance(result, float)

    def test_pagerank_returns_ndarray(self) -> None:
        """pagerank debe retornar NDArray."""
        adj = np.array([[0, 1], [1, 0]])
        result = pagerank(adj)
        assert isinstance(result, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
