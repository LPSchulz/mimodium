import numpy as np
import pytest
from mimodium.scenario import cpu
from mimodium.warnings import ScenarioSizeWarning


def test_explicit_num_cpus_returns_configured_value():
    assert cpu.CfgExplicitNumCPUs(2)() == 2


def test_same_as_num_aps_returns_ap_count():
    assert cpu.CfgSameAsNumAPs()(L=5) == 5


def test_poisson_num_cpus_tracks_area_over_many_draws():
    task = cpu.CfgPoissonNumCPUs(density=10.0)

    small_area_counts = np.array(
        [task(A=10_000_000.0, seed=seed) for seed in range(100)]
    )
    large_area_counts = np.array(
        [task(A=100_000_000.0, seed=seed) for seed in range(100)]
    )

    assert np.all(small_area_counts >= 1)
    assert np.all(large_area_counts >= 1)
    assert 90.0 < np.mean(small_area_counts) < 110.0
    assert 970.0 < np.mean(large_area_counts) < 1030.0
    assert np.mean(large_area_counts) > np.mean(small_area_counts)
    assert task(A=10_000_000.0, seed=7) == task(A=10_000_000.0, seed=7)


def test_poisson_num_cpus_seed_override_ignores_global_seed():
    task = cpu.CfgPoissonNumCPUs(density=10.0, seed_override=23)

    assert task(A=10_000_000.0, seed=1) == task(A=10_000_000.0, seed=2)


def test_poisson_num_cpus_warns_and_clamps_zero_draw_to_one():
    with pytest.warns(ScenarioSizeWarning, match="scenario area may be too small"):
        assert cpu.CfgPoissonNumCPUs(density=0.0)(A=1, seed=1) == 1


def test_explicit_cpu_positions_return_configured_array():
    positions = np.array([[1.0, 2.0], [3.0, 4.0]])

    task = cpu.CfgExplicitCPUPositions(positions)

    np.testing.assert_array_equal(task(), positions)


def test_evenly_spaced_cpu_positions_place_points_at_correct_positions():
    positions = cpu.CfgEvenlySpacedCPUPositions()(J=4, sqrt_A=100.0)

    np.testing.assert_allclose(
        positions,
        np.array([[25.0, 25.0], [75.0, 25.0], [25.0, 75.0], [75.0, 75.0]]),
    )


def test_evenly_spaced_cpu_positions_reject_non_square_counts():
    with pytest.raises(ValueError, match="perfect square"):
        cpu.CfgEvenlySpacedCPUPositions()(J=3, sqrt_A=100.0)


def test_uniform_random_cpu_positions_seed_override_ignores_global_seed():
    task = cpu.CfgUniformRandomCPUPositions(seed_override=7)

    first = task(J=3, sqrt_A=100.0, seed=1)
    second = task(J=3, sqrt_A=100.0, seed=2)

    np.testing.assert_array_equal(first, second)


def test_uniform_random_cpu_positions_stay_inside_area():
    task = cpu.CfgUniformRandomCPUPositions()

    for area_length in (1.0, 100.0):
        positions = np.stack(
            [task(J=32, sqrt_A=area_length, seed=seed) for seed in range(100)]
        )

        assert positions.shape == (100, 32, 2)
        assert np.all((0.0 <= positions) & (positions < area_length))

    np.testing.assert_array_equal(
        task(J=32, sqrt_A=100.0, seed=7),
        task(J=32, sqrt_A=100.0, seed=7),
    )


def test_uniform_random_cpu_positions_can_force_first_position():
    forced_position = np.array([5.0, 6.0])

    positions = cpu.CfgUniformRandomCPUPositions(
        forced_first_cpu_position=forced_position
    )(J=3, sqrt_A=100.0, seed=7)

    np.testing.assert_array_equal(positions[0], forced_position)


def test_same_as_ap_positions_returns_copy():
    ap_positions = np.array([[1.0, 2.0], [3.0, 4.0]])

    cpu_positions = cpu.CfgSameAsAPPositions()(ap_positions)

    np.testing.assert_array_equal(cpu_positions, ap_positions)
    assert cpu_positions is not ap_positions
