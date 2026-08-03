import numpy as np
import pytest
from mimodium.scenario import ap
from mimodium.warnings import ScenarioSizeWarning


def test_explicit_num_aps_returns_configured_value():
    assert ap.CfgExplicitNumAPs(3)() == 3


def test_num_antennas_returns_configured_value():
    assert ap.CfgNumAntennas(4)() == 4


def test_antenna_spacing_returns_configured_value():
    assert ap.CfgAntennaSpacing(0.5)() == 0.5


def test_compute_total_number_antennas_multiplies_aps_and_antennas():
    assert ap.ComputeTotalNumberAntennas()(L=3, N=4) == 12


def test_aligned_array_orientations_repeats_azimuth():
    orientations = ap.CfgAlignedArrayOrientations(azimuth=1.2)(L=3)

    np.testing.assert_allclose(orientations, np.array([1.2, 1.2, 1.2]))


def test_random_azimuth_array_orientations_stay_in_valid_range():
    task = ap.CfgRandomAzimuthArrayOrientations()

    orientations = np.stack([task(L=32, seed=seed) for seed in range(100)])

    assert orientations.shape == (100, 32)
    assert np.all((0.0 <= orientations) & (orientations < 2 * np.pi))
    np.testing.assert_array_equal(task(L=32, seed=7), task(L=32, seed=7))


def test_random_azimuth_array_orientations_seed_override_ignores_global_seed():
    task = ap.CfgRandomAzimuthArrayOrientations(seed_override=99)

    first = task(L=4, seed=1)
    second = task(L=4, seed=2)

    np.testing.assert_array_equal(first, second)


def test_poisson_num_aps_tracks_area_over_many_draws():
    task = ap.CfgPoissonNumAPs(density=10.0)

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


def test_poisson_num_aps_seed_override_ignores_global_seed():
    task = ap.CfgPoissonNumAPs(density=10.0, seed_override=23)

    assert task(A=10_000_000.0, seed=1) == task(A=10_000_000.0, seed=2)


def test_poisson_num_aps_warns_and_clamps_zero_draw_to_one():
    with pytest.warns(ScenarioSizeWarning, match="scenario area may be too small"):
        assert ap.CfgPoissonNumAPs(density=0.0)(A=1.0, seed=1) == 1


def test_explicit_ap_positions_return_configured_array():
    positions = np.array([[1.0, 2.0], [3.0, 4.0]])

    task = ap.CfgExplicitAPPositions(positions)

    np.testing.assert_array_equal(task(), positions)


def test_evenly_spaced_ap_positions_place_points_at_correct_positions():
    positions = ap.CfgEvenlySpacedAPPositions()(L=4, sqrt_A=100.0)

    np.testing.assert_allclose(
        positions,
        np.array([[25.0, 25.0], [75.0, 25.0], [25.0, 75.0], [75.0, 75.0]]),
    )


def test_evenly_spaced_ap_positions_reject_non_square_counts():
    with pytest.raises(ValueError, match="perfect square"):
        ap.CfgEvenlySpacedAPPositions()(L=3, sqrt_A=100.0)


def test_uniform_random_ap_positions_seed_override_ignores_global_seed():
    task = ap.CfgUniformRandomAPPositions(seed_override=7)

    first = task(L=3, sqrt_A=100.0, seed=1)
    second = task(L=3, sqrt_A=100.0, seed=2)

    np.testing.assert_array_equal(first, second)


def test_uniform_random_ap_positions_stay_inside_area():
    task = ap.CfgUniformRandomAPPositions()

    for area_length in (1.0, 100.0):
        positions = np.stack(
            [task(L=32, sqrt_A=area_length, seed=seed) for seed in range(100)]
        )

        assert positions.shape == (100, 32, 2)
        assert np.all((0.0 <= positions) & (positions < area_length))

    np.testing.assert_array_equal(
        task(L=32, sqrt_A=100.0, seed=7),
        task(L=32, sqrt_A=100.0, seed=7),
    )


def test_uniform_random_ap_positions_can_force_first_position():
    forced_position = np.array([5.0, 6.0])

    positions = ap.CfgUniformRandomAPPositions(
        forced_first_ap_position=forced_position
    )(L=3, sqrt_A=100.0, seed=7)

    np.testing.assert_array_equal(positions[0], forced_position)


def test_ap_heights_repeat_configured_height():
    heights = ap.CfgAPHeights(height=12.0)(L=2)

    np.testing.assert_allclose(heights, np.array([12.0, 12.0]))


def test_compute_ap_locations_forms_3d_coordinates():
    heights = np.array([12.0, 12.0])
    positions = np.array([[1.0, 2.0], [3.0, 4.0]])

    locations = ap.ComputeAPLocations()(positions, heights)

    np.testing.assert_allclose(
        locations, np.array([[1.0, 2.0, 12.0], [3.0, 4.0, 12.0]])
    )
