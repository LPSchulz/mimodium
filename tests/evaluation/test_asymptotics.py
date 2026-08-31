import numpy as np
from mimodium.evaluation import asymptotics


def normalized_variance(samples):
    expected = np.mean(samples, axis=-1)
    return np.mean(np.abs(samples - expected[..., np.newaxis]) ** 2, axis=-1) / (
        np.abs(expected) ** 2
    )


def normalized_second_moments(effective_channels):
    expected_desired = np.array(
        [np.mean(effective_channels[k, k]) for k in range(effective_channels.shape[0])]
    )
    return np.mean(np.abs(effective_channels) ** 2, axis=-1) / (
        np.abs(expected_desired[:, np.newaxis]) ** 2
    )


def symmetric_normalized_second_moments(channel_inner_products):
    channel_powers = np.real(
        np.mean(
            np.diagonal(channel_inner_products, axis1=0, axis2=1),
            axis=0,
        )
    )
    return np.mean(np.abs(channel_inner_products) ** 2, axis=-1) / (
        channel_powers[:, np.newaxis] * channel_powers[np.newaxis, :]
    )


def test_channel_hardening_is_zero_for_realization_invariant_norms():
    channels = np.ones((2, 1, 2, 3), dtype=complex)

    values = asymptotics.ComputeChannelHardening()(K=2, L=1, N=2, O=3, h=channels)

    np.testing.assert_allclose(values, np.zeros(2))


def test_effective_uplink_hardening_coherently_adds_serving_cpus():
    g = [
        np.array(
            [
                [
                    [1 + 1j, 2 + 1j, 3 + 1j],
                    [3 - 1j, 2 - 1j, 1 - 1j],
                ],
                [
                    [2 + 0.5j, 3 + 0.5j, 4 + 0.5j],
                    [1 - 0.5j, 1 - 0.5j, 2 - 0.5j],
                ],
            ]
        ),
        np.array(
            [
                [[0.5 + 0.2j, 1 - 0.3j, 1.5 + 0.1j]],
                [[1 + 1j, 2 - 1j, 4]],
            ]
        ),
    ]

    values = asymptotics.ComputeEffectiveUplinkChannelHardening()(g)

    desired = np.stack((np.sum(g[0][0], axis=0), np.sum(g[1][1], axis=0)))
    np.testing.assert_allclose(values, normalized_variance(desired))
    assert values[0] == 0


def test_effective_downlink_hardening_coherently_adds_stream_cpus():
    f = [
        [
            np.array(
                [
                    [1 + 1j, 2 + 1j, 3 + 1j],
                    [3 - 1j, 2 - 1j, 1 - 1j],
                ]
            ),
            np.array([[2 + 0.5j, 3 + 0.5j, 4 + 0.5j]]),
        ],
        [
            np.array(
                [
                    [0.5 + 0.2j, 1 - 0.3j, 1.5 + 0.1j],
                    [0.2 - 0.1j, 0.4 + 0.2j, 0.1 + 0.3j],
                ]
            ),
            np.array([[1 + 1j, 2 - 1j, 4]]),
        ],
    ]

    values = asymptotics.ComputeEffectiveDownlinkChannelHardening()(f)

    desired = np.stack((np.sum(f[0][0], axis=0), np.sum(f[1][1], axis=0)))
    np.testing.assert_allclose(values, normalized_variance(desired))
    assert values[0] == 0


def test_effective_uplink_favorable_propagation_uses_second_moments():
    g = [
        np.array(
            [
                [
                    [1, 2, 3],
                    [3, 2, 1],
                ],
                [
                    [1, 1, 2],
                    [1, 2, 2],
                ],
            ],
            dtype=complex,
        ),
        np.array(
            [
                [[1j, 2j, 3j]],
                [[2 + 1j, 3 - 1j, 4]],
            ]
        ),
    ]

    values = asymptotics.ComputeEffectiveUplinkFavorablePropagation()(g)

    effective_channels = np.stack((np.sum(g[0], axis=1), np.sum(g[1], axis=1)))
    np.testing.assert_allclose(
        values,
        normalized_second_moments(effective_channels),
    )
    interference_variance_only = np.var(effective_channels[0, 1]) / np.abs(4) ** 2
    assert values[0, 1] > interference_variance_only


def test_effective_downlink_favorable_propagation_uses_transmitting_cpu_set():
    f = [
        [
            np.array([[1, 2, 3], [3, 2, 1]], dtype=complex),
            np.array([[1, 2, 2]], dtype=complex),
        ],
        [
            np.array([[1j, 2j, 3j], [0.5j, 0.25j, 0.75j]]),
            np.array([[2 + 1j, 3 - 1j, 4]]),
        ],
    ]

    values = asymptotics.ComputeEffectiveDownlinkFavorablePropagation()(f)

    effective_channels = np.array(
        [
            [np.sum(f[0][0], axis=0), np.sum(f[0][1], axis=0)],
            [np.sum(f[1][0], axis=0), np.sum(f[1][1], axis=0)],
        ]
    )
    np.testing.assert_allclose(
        values,
        normalized_second_moments(effective_channels),
    )


def test_perfect_csi_maximum_ratio_recovers_effective_metrics():
    samples = np.arange(1, 2 * 1 * 2 * 4 + 1).reshape(2, 1, 2, 4)
    h = samples + 1j * np.flip(samples, axis=-1) / 3
    mr_effective_channels = np.einsum("klno,ilno->kio", np.conj(h), h)
    g = [mr_effective_channels[k, :, np.newaxis] for k in range(2)]
    f = [[mr_effective_channels[k, i, np.newaxis] for i in range(2)] for k in range(2)]

    classical_hardening = asymptotics.ComputeChannelHardening()(
        K=2,
        L=1,
        N=2,
        O=4,
        h=h,
    )
    np.testing.assert_allclose(
        asymptotics.ComputeEffectiveUplinkChannelHardening()(g),
        classical_hardening,
    )
    np.testing.assert_allclose(
        asymptotics.ComputeEffectiveDownlinkChannelHardening()(f),
        classical_hardening,
    )
    np.testing.assert_allclose(
        asymptotics.ComputeEffectiveUplinkFavorablePropagation()(g),
        normalized_second_moments(mr_effective_channels),
    )
    np.testing.assert_allclose(
        asymptotics.ComputeEffectiveDownlinkFavorablePropagation()(f),
        normalized_second_moments(mr_effective_channels),
    )


def test_classical_favorable_propagation_is_symmetric_and_gain_invariant():
    h = np.zeros((2, 1, 2, 2), dtype=complex)
    h[0, 0, 0] = 1
    h[1] = 2 * h[0]

    values = asymptotics.ComputeFavorablePropagation()(h)

    np.testing.assert_allclose(values, np.ones((2, 2)))
    np.testing.assert_allclose(values, values.T)


def test_classical_favorable_propagation_uses_symmetric_second_moments():
    samples = np.arange(1, 2 * 1 * 2 * 4 + 1).reshape(2, 1, 2, 4)
    h = samples + 1j * np.flip(samples, axis=-1) / 3
    channel_inner_products = np.einsum("klno,ilno->kio", np.conj(h), h)

    values = asymptotics.ComputeFavorablePropagation()(h)

    np.testing.assert_allclose(
        values,
        symmetric_normalized_second_moments(channel_inner_products),
    )
    np.testing.assert_allclose(values, values.T)
    np.testing.assert_allclose(
        np.diag(values),
        1 + asymptotics.ComputeChannelHardening()(K=2, L=1, N=2, O=4, h=h),
    )


def test_metrics_are_undefined_when_expected_desired_channel_is_zero():
    g = [np.zeros((1, 0, 3), dtype=complex)]
    f = [[np.zeros((0, 3), dtype=complex)]]

    assert np.isnan(asymptotics.ComputeEffectiveUplinkChannelHardening()(g)[0])
    assert np.isnan(asymptotics.ComputeEffectiveDownlinkChannelHardening()(f)[0])
    assert np.isnan(asymptotics.ComputeEffectiveUplinkFavorablePropagation()(g)[0, 0])
    assert np.isnan(asymptotics.ComputeEffectiveDownlinkFavorablePropagation()(f)[0, 0])
