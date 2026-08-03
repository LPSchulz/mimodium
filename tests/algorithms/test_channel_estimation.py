import numpy as np
from mimodium.algorithms import channel_estimation as ce


def test_max_pilot_power_returns_full_power_vector():
    np.testing.assert_array_equal(
        ce.CfgAllocateFullPilotPower()(K=3, pilot_max_power=np.full(3, 2.0)),
        np.full(3, 2.0),
    )


def test_perfect_channel_estimation_uses_realized_expected_or_zero_channels():
    h = np.array(
        [
            [
                [[1.0 + 1.0j, 2.0], [3.0, 4.0 - 1.0j]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]
        ]
    )
    exp_h = np.array([[[100.0 + 2.0j, 200.0], [300.0, 400.0], [500.0, 600.0]]])
    estimated = np.array([[True, False, False]])
    measured = np.array([[True, True, False]])

    h_hat = ce.CfgPerfectChannelEstimation()(
        h=h,
        exp_h=exp_h,
        estimated=estimated,
        measured=measured,
    )

    expected = np.array(
        [
            [
                [[1.0 + 1.0j, 2.0], [3.0, 4.0 - 1.0j]],
                [[300.0, 300.0], [400.0, 400.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )
    np.testing.assert_array_equal(h_hat, expected)


def test_analytical_channel_expectation_is_complex_zero():
    expected_h = ce.CfgAnalyticalChannelExpectation()(K=2, L=3, N=4)

    assert expected_h.shape == (2, 3, 4)
    assert np.issubdtype(expected_h.dtype, np.complexfloating)
    np.testing.assert_array_equal(expected_h, 0.0)


def test_measured_channel_expectation_matches_complex_sample_mean():
    h = np.array(
        [
            [
                [
                    [1.0 + 1.0j, 3.0 + 2.0j],
                    [2.0 - 1.0j, 4.0 + 3.0j],
                ]
            ]
        ]
    )

    expected_h = ce.CfgMeasuredChannelExpectation()(h)

    np.testing.assert_allclose(expected_h, [[[2.0 + 1.5j, 3.0 + 1.0j]]])


def test_analytical_received_signals_correlation_matches_scalar_reference_case():
    psi = ce.CfgAnalyticalReceivedPilotsCorrelation()(
        t=np.array([0, 1]),
        eta=np.array([10.0, 20.0]),
        sigma2_ul=0.5,
        L=1,
        N=1,
        tau_p=2,
        R=np.ones((2, 1, 1, 1), dtype=complex),
        beta=np.array([[2.0], [3.0]]),
    )

    np.testing.assert_allclose(psi[:, 0, 0, 0], np.array([40.5, 120.5]))


def test_measured_received_signals_correlation_matches_complex_reference_case():
    y = np.array(
        [
            [
                [
                    [1.0 + 1.0j, 3.0],
                    [2.0, 4.0 - 2.0j],
                ]
            ]
        ]
    )

    psi = ce.CfgMeasuredReceivedPilotsCorrelation()(y)

    expected = np.array([[[[5.5, 7.0 + 4.0j], [7.0 - 4.0j, 12.0]]]])
    np.testing.assert_allclose(psi, expected)


def test_analytical_estimation_error_correlation_matches_scalar_mmse_formula():
    C = ce.CfgAnalyticalEstimationErrorCorrelation()(
        K=1,
        etas=np.array([1.0]),
        tau_p=1,
        t=np.array([0]),
        R=np.ones((1, 1, 1, 1), dtype=complex),
        beta=np.array([[2.0]]),
        Psi=np.array([[[[3.0]]]], dtype=complex),
        estimated=np.array([[True]]),
        measured=np.array([[True]]),
    )

    np.testing.assert_allclose(C[0, 0, 0, 0], 2.0 - 4.0 / 3.0)


def test_perfect_analytical_error_correlation_respects_link_sets():
    C = ce.CfgAnalyticalEstimationErrorCorrelation(is_perfect=True)(
        K=1,
        etas=np.ones(1),
        tau_p=1,
        t=np.zeros(1, dtype=int),
        R=np.array([[[[2.0]], [[3.0]], [[4.0]]]], dtype=complex),
        beta=np.array([[5.0, 6.0, 7.0]]),
        Psi=np.zeros((1, 3, 1, 1), dtype=complex),
        estimated=np.array([[True, False, False]]),
        measured=np.array([[True, True, False]]),
    )

    np.testing.assert_array_equal(C[:, :, 0, 0], [[0.0, 18.0, 0.0]])


def test_measured_estimation_error_correlation_zeros_unmeasured_links():
    h = np.array([[[[2.0, 4.0]]]], dtype=complex)
    h_hat = np.array([[[[1.0, 1.0]]]], dtype=complex)

    C = ce.CfgMeasuredEstimationErrorCorrelation()(
        h=np.concatenate([h, h], axis=1),
        h_hat=np.concatenate([h_hat, h_hat], axis=1),
        measured=np.array([[True, False]]),
    )

    np.testing.assert_allclose(C[0, 0, 0, 0], 5.0)
    np.testing.assert_allclose(C[0, 1, 0, 0], 0.0)


def test_received_signals_seed_override_is_reproducible():
    task = ce.ComputeReceivedSignals(seed_override=13)
    L = 1
    N = 1
    O = 2
    t = np.array([0])
    etas = np.array([1.0])
    h = np.ones((1, 1, 1, 2), dtype=complex)
    sigma2_ul = 1.0
    tau_p = 1

    first = task(
        L=L, N=N, O=O, t=t, etas=etas, h=h, sigma2_ul=sigma2_ul, tau_p=tau_p, seed=1
    )
    second = task(
        L=L, N=N, O=O, t=t, etas=etas, h=h, sigma2_ul=sigma2_ul, tau_p=tau_p, seed=2
    )

    np.testing.assert_array_equal(first, second)


def test_pilot_based_mmse_channel_estimation_matches_scalar_reference_case():
    h_hat = ce.CfgPilotBasedMMSEChannelEstimation()(
        K=1,
        L=1,
        N=1,
        O=1,
        y=np.array([[[[3.0]]]], dtype=complex),
        Psi=np.array([[[[2.0]]]], dtype=complex),
        R=np.ones((1, 1, 1, 1), dtype=complex),
        beta=np.array([[2.0]]),
        etas=np.array([1.0]),
        tau_p=1,
        t=np.array([0]),
        exp_h=np.zeros((1, 1, 1), dtype=complex),
        estimated=np.array([[True]]),
        measured=np.array([[True]]),
    )

    np.testing.assert_allclose(h_hat[0, 0, 0, 0], 3.0)


def _multi_link_mmse_inputs():
    generator = np.random.default_rng(42)
    K, L, N, O, tau_p = 3, 2, 2, 3, 2

    correlation_factors = generator.normal(size=(K, L, N, N)) + 1.0j * generator.normal(
        size=(K, L, N, N)
    )
    R = correlation_factors @ correlation_factors.conj().swapaxes(-1, -2)

    observation_factors = generator.normal(
        size=(tau_p, L, N, N)
    ) + 1.0j * generator.normal(size=(tau_p, L, N, N))
    Psi = observation_factors @ observation_factors.conj().swapaxes(-1, -2)
    Psi += np.eye(N)[np.newaxis, np.newaxis]

    y = generator.normal(size=(tau_p, L, N, O)) + 1.0j * generator.normal(
        size=(tau_p, L, N, O)
    )
    return {
        "K": K,
        "L": L,
        "N": N,
        "O": O,
        "y": y,
        "Psi": Psi,
        "R": R,
        "beta": np.array([[0.7, 1.2], [0.9, 0.4], [1.5, 0.8]]),
        "etas": np.array([0.5, 1.0, 1.3]),
        "tau_p": tau_p,
        "t": np.array([0, 1, 0]),
    }


def test_pilot_based_mmse_uses_estimates_expectations_and_zeros_by_link_set():
    inputs = _multi_link_mmse_inputs()
    estimated = np.array([[True, False], [False, True], [False, False]])
    measured = np.array([[True, True], [False, True], [True, False]])
    exp_h = np.arange(12).reshape(3, 2, 2) + 1.0j

    scaled_R = inputs["beta"][..., np.newaxis, np.newaxis] * inputs["R"]
    x_kl = np.linalg.solve(
        inputs["Psi"][inputs["t"]],
        inputs["y"][inputs["t"]],
    )
    mmse_estimates = np.sqrt(
        inputs["etas"][:, np.newaxis, np.newaxis, np.newaxis] * inputs["tau_p"]
    ) * (scaled_R @ x_kl)

    expected = np.zeros_like(mmse_estimates)
    expected[measured] = exp_h[measured][..., np.newaxis]
    expected[estimated] = mmse_estimates[estimated]

    actual = ce.CfgPilotBasedMMSEChannelEstimation()(
        **inputs,
        exp_h=exp_h,
        estimated=estimated,
        measured=measured,
    )

    np.testing.assert_allclose(actual, expected)


def test_pilot_based_mmse_uses_expectations_when_no_links_are_estimated():
    exp_h = np.arange(12).reshape(3, 2, 2) + 1.0j
    measured = np.array([[True, False], [False, True], [True, False]])

    actual = ce.CfgPilotBasedMMSEChannelEstimation()(
        **_multi_link_mmse_inputs(),
        exp_h=exp_h,
        estimated=np.zeros((3, 2), dtype=bool),
        measured=measured,
    )

    expected = np.zeros((3, 2, 2, 3), dtype=complex)
    expected[measured] = exp_h[measured][..., np.newaxis]
    np.testing.assert_array_equal(actual, expected)
