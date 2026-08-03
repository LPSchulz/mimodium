import numpy as np
import pytest
from mimodium.rng import factor_correlation_matrix, get_rng
from mimodium.scenario.ap import CfgPoissonNumAPs
from mimodium.scenario.cpu import CfgPoissonNumCPUs
from mimodium.scenario.ue import CfgUniformRandomUEPositions


def test_get_rng_is_stable_for_same_seed_and_task_config():
    task = CfgPoissonNumAPs(density=10.0)

    values_a = get_rng(1234, task).random(8)
    values_b = get_rng(1234, task).random(8)

    assert np.array_equal(values_a, values_b)


def test_get_rng_changes_for_different_task_configs():
    values_a = get_rng(1234, CfgPoissonNumAPs(density=10.0)).random(8)
    values_b = get_rng(1234, CfgPoissonNumAPs(density=11.0)).random(8)

    assert not np.array_equal(values_a, values_b)


def test_get_rng_changes_for_different_task_classes():
    values_a = get_rng(1234, CfgPoissonNumAPs(density=10.0)).random(8)
    values_b = get_rng(1234, CfgPoissonNumCPUs(density=10.0)).random(8)

    assert not np.array_equal(values_a, values_b)


def test_seed_override_is_reproducible_independent_of_global_seed():
    task = CfgUniformRandomUEPositions(seed_override=99)

    positions_a = task(K=4, sqrt_A=100.0, seed=1)
    positions_b = task(K=4, sqrt_A=100.0, seed=2)

    assert np.array_equal(positions_a, positions_b)


def test_factor_correlation_matrix_uses_cholesky_for_positive_definite_matrix():
    correlation = np.array([[1.0, 0.25], [0.25, 1.0]])

    factor = factor_correlation_matrix(correlation)

    np.testing.assert_allclose(factor @ factor.conj().T, correlation)
    np.testing.assert_allclose(factor, np.linalg.cholesky(correlation))


def test_factor_correlation_matrix_supports_singular_complex_matrix():
    steering = np.array([1.0, 1.0j])
    correlation = np.outer(steering, steering.conj())

    factor = factor_correlation_matrix(correlation)

    np.testing.assert_allclose(
        factor @ factor.conj().T,
        correlation,
        atol=1e-15,
    )


def test_factor_correlation_matrix_rejects_indefinite_matrix():
    with pytest.raises(ValueError, match="positive semidefinite"):
        factor_correlation_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_factor_correlation_matrix_supports_batched_leading_dimensions():
    positive_definite = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=complex)
    steering = np.array([1.0, 1.0j])
    singular = np.outer(steering, steering.conj())
    correlations = np.array(
        [
            [positive_definite, singular],
            [singular.conj(), positive_definite],
        ]
    )

    factors = factor_correlation_matrix(correlations)

    reconstructed = factors @ factors.swapaxes(-1, -2).conj()
    np.testing.assert_allclose(reconstructed, correlations, atol=1e-15)


def test_factor_correlation_matrix_rejects_indefinite_matrix_in_batch():
    correlations = np.array(
        [
            [[1.0, 0.25], [0.25, 1.0]],
            [[1.0, 2.0], [2.0, 1.0]],
        ]
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        factor_correlation_matrix(correlations)
