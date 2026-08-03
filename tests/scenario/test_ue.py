import numpy as np
import pytest
from mimodium.scenario import ue
from mimodium.warnings import ScenarioSizeWarning


def test_explicit_num_ues_returns_configured_value():
    assert ue.CfgExplicitNumUEs(5)() == 5


def test_poisson_num_ues_tracks_area_over_many_draws():
    task = ue.CfgPoissonNumUEs(density=10.0)

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


def test_poisson_num_ues_seed_override_ignores_global_seed():
    task = ue.CfgPoissonNumUEs(density=10.0, seed_override=23)

    assert task(A=10_000_000.0, seed=1) == task(A=10_000_000.0, seed=2)


def test_poisson_num_ues_warns_and_clamps_zero_draw_to_one():
    with pytest.warns(ScenarioSizeWarning, match="scenario area may be too small"):
        assert ue.CfgPoissonNumUEs(density=0.0)(A=1.0, seed=1) == 1


def test_explicit_ue_positions_return_configured_array():
    positions = np.array([[1.0, 2.0], [3.0, 4.0]])

    task = ue.CfgExplicitUEPositions(positions)

    np.testing.assert_array_equal(task(), positions)


def test_uniform_random_ue_positions_seed_override_ignores_global_seed():
    task = ue.CfgUniformRandomUEPositions(seed_override=7)

    first = task(K=3, sqrt_A=100.0, seed=1)
    second = task(K=3, sqrt_A=100.0, seed=2)

    np.testing.assert_array_equal(first, second)


def test_uniform_random_ue_positions_stay_inside_area():
    task = ue.CfgUniformRandomUEPositions()

    for area_length in (1.0, 100.0):
        positions = np.stack(
            [task(K=32, sqrt_A=area_length, seed=seed) for seed in range(100)]
        )

        assert positions.shape == (100, 32, 2)
        assert np.all((0.0 <= positions) & (positions < area_length))

    np.testing.assert_array_equal(
        task(K=32, sqrt_A=100.0, seed=7),
        task(K=32, sqrt_A=100.0, seed=7),
    )


def test_uniform_random_ue_positions_can_force_first_position():
    forced_position = np.array([5.0, 6.0])

    positions = ue.CfgUniformRandomUEPositions(
        forced_first_ue_position=forced_position
    )(K=3, sqrt_A=100.0, seed=7)

    np.testing.assert_array_equal(positions[0], forced_position)


def test_ue_heights_repeat_configured_height():
    heights = ue.CfgUEHeights(height=1.5)(K=3)

    np.testing.assert_allclose(heights, np.array([1.5, 1.5, 1.5]))


def test_compute_ue_locations_forms_3d_coordinates():
    positions = np.array([[1.0, 2.0], [3.0, 4.0]])
    heights = np.array([1.5, 2.0])

    locations = ue.ComputeUELocations()(positions, heights)

    np.testing.assert_allclose(locations, np.array([[1.0, 2.0, 1.5], [3.0, 4.0, 2.0]]))
