import numpy as np
import pytest
from mimodium.propagation import channel_generation as cg
from mimodium.warnings import ApplicabilityWarning
from scipy.linalg import toeplitz


def approximate_spatial_correlations_loop_reference(
    *,
    K,
    L,
    N,
    azimuths,
    d_H,
    orientations,
    asd_in_degrees,
):
    asd_in_rads = asd_in_degrees * np.pi / 180
    correlations = np.zeros((K, L, N, N), dtype=complex)
    for k in range(K):
        for l in range(L):
            first_column = np.zeros(N, dtype=complex)
            nominal_angle = azimuths[k, l] - orientations[l]
            for l_minus_m in range(N):
                first_column[l_minus_m] = np.exp(
                    1j * 2 * np.pi * d_H * l_minus_m * np.sin(nominal_angle)
                ) * np.exp(
                    (-(asd_in_rads**2) / 2)
                    * (2 * np.pi * d_H * l_minus_m * np.cos(nominal_angle)) ** 2
                )
            correlations[k, l] = toeplitz(first_column)
    return correlations


def test_num_realizations_returns_configured_count():
    assert cg.CfgNumRealizations(8)() == 8


def test_num_realizations_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        cg.CfgNumRealizations(0)


def test_positive_shadow_fading_increases_large_scale_fading_gain():
    path_loss_db = np.array([[30.0, 40.0]])
    shadow_fading_db = np.array([[0.0, 10.0]])

    beta = cg.ComputeLargeScaleFading()(
        path_loss_db=path_loss_db, shadow_fading_db=shadow_fading_db
    )

    np.testing.assert_allclose(beta, np.array([[1e-3, 1e-3]]))


def test_uncorrelated_spatial_correlations_tile_identity_matrix():
    correlations = cg.CfgUncorrelatedSpatialCorrelations()(K=2, L=1, N=2)

    expected = np.tile(np.eye(2)[np.newaxis, np.newaxis, :, :], (2, 1, 1, 1))
    np.testing.assert_allclose(correlations, expected)


def test_approximate_spatial_correlations_reduce_to_rank_one_for_zero_asd_broadside():
    correlations = cg.CfgApproximateSpatialCorrelations(asd_in_degrees=0.0)(
        K=1,
        L=1,
        N=3,
        azimuths=np.array([[0.0]]),
        d_H=0.5,
        orientations=np.array([0.0]),
    )

    np.testing.assert_allclose(correlations[0, 0], np.ones((3, 3)))


def test_approximate_spatial_correlations_warn_above_small_angle_range():
    with pytest.warns(ApplicabilityWarning, match="at most 15 degrees"):
        cg.CfgApproximateSpatialCorrelations(asd_in_degrees=15.1)


@pytest.mark.parametrize("asd_in_degrees", [-0.1, np.nan, np.inf])
def test_approximate_spatial_correlations_reject_invalid_angular_spread(
    asd_in_degrees,
):
    with pytest.raises(ValueError, match="finite and non-negative"):
        cg.CfgApproximateSpatialCorrelations(asd_in_degrees=asd_in_degrees)


def test_approximate_spatial_correlations_match_loop_reference():
    K, L, N = 3, 4, 7
    azimuths = np.linspace(-np.pi, np.pi, K * L).reshape(K, L)
    orientations = np.array([0.0, 0.3, np.pi / 2, 1.7])
    asd_in_degrees = 10.0
    d_H = 0.5

    expected = approximate_spatial_correlations_loop_reference(
        K=K,
        L=L,
        N=N,
        azimuths=azimuths,
        d_H=d_H,
        orientations=orientations,
        asd_in_degrees=asd_in_degrees,
    )
    actual = cg.CfgApproximateSpatialCorrelations(asd_in_degrees=asd_in_degrees)(
        K=K,
        L=L,
        N=N,
        azimuths=azimuths,
        d_H=d_H,
        orientations=orientations,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_rayleigh_channels_seed_override_is_reproducible():
    task = cg.CfgRayleighChannels(seed_override=12)
    R = np.ones((1, 1, 1, 1), dtype=complex)
    beta = np.array([[2.0]])

    first = task(K=1, L=1, N=1, O=3, R=R, beta=beta, seed=1)
    second = task(K=1, L=1, N=1, O=3, R=R, beta=beta, seed=2)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (1, 1, 1, 3)


def test_rayleigh_channels_support_singular_spatial_correlations():
    channels = cg.CfgRayleighChannels(seed_override=12)(
        K=1,
        L=1,
        N=2,
        O=3,
        R=np.ones((1, 1, 2, 2), dtype=complex),
        beta=np.ones((1, 1)),
        seed=1,
    )

    np.testing.assert_allclose(channels[0, 0, 0], channels[0, 0, 1])


def test_rayleigh_channels_reject_indefinite_spatial_correlations():
    correlation = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=complex)

    with pytest.raises(ValueError, match="positive semidefinite"):
        cg.CfgRayleighChannels()(
            K=1,
            L=1,
            N=2,
            O=1,
            R=correlation[np.newaxis, np.newaxis],
            beta=np.ones((1, 1)),
            seed=1,
        )


def test_rayleigh_channels_follow_configured_complex_gaussian_distribution():
    correlation = np.array(
        [
            [1.0, 0.3 + 0.2j],
            [0.3 - 0.2j, 1.0],
        ]
    )
    beta = 2.5

    channels = cg.CfgRayleighChannels(seed_override=12)(
        K=1,
        L=1,
        N=2,
        O=100_000,
        R=correlation[np.newaxis, np.newaxis],
        beta=np.array([[beta]]),
        seed=1,
    )[0, 0]

    sample_mean = np.mean(channels, axis=1)
    sample_covariance = channels @ channels.conj().T / channels.shape[1]
    sample_pseudo_covariance = channels @ channels.T / channels.shape[1]

    np.testing.assert_allclose(sample_mean, 0.0, atol=0.02)
    np.testing.assert_allclose(
        sample_covariance,
        beta * correlation,
        rtol=0.015,
        atol=0.015,
    )
    np.testing.assert_allclose(sample_pseudo_covariance, 0.0, atol=0.025)
