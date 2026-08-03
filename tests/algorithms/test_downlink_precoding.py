import numpy as np
import pytest
from mimodium.algorithms import downlink_precoding


def test_effective_dl_channels_for_complex_multi_cpu_network():
    K = 3
    O = 2
    used = np.array(
        [
            [True, True, False],
            [False, True, True],
            [False, False, False],
        ]
    )
    fronthaul = np.array(
        [
            [True, False],
            [True, False],
            [False, True],
        ]
    )
    real = np.arange(1, K * 3 * 2 * O + 1).reshape(K, 3, 2, O)
    h = real + 1j * np.flip(real, axis=-1)
    w = [
        [
            np.array(
                [
                    [1 + 1j, 2],
                    [2 - 1j, 1j],
                    [1, -2j],
                    [-1j, 1 + 1j],
                ]
            )
        ],
        [
            np.array([[1 - 1j, 2j], [2, 1 + 1j]]),
            np.array([[2 + 1j, -1j], [1, 2 - 1j]]),
        ],
        [],
    ]

    f = downlink_precoding.ComputeEffectiveDLChannels()(
        K=K,
        O=O,
        h=h,
        w=w,
        used=used,
        fronthaul=fronthaul,
    )

    h_stream_0 = h[:, [0, 1]].reshape(K, 4, O)
    expected_stream_0 = np.einsum("kno,no->ko", np.conj(h_stream_0), w[0][0])
    expected_stream_1 = np.stack(
        (
            np.einsum("kno,no->ko", np.conj(h[:, 1]), w[1][0]),
            np.einsum("kno,no->ko", np.conj(h[:, 2]), w[1][1]),
        ),
        axis=1,
    )

    assert len(f) == K
    assert all(len(f_k) == K for f_k in f)
    np.testing.assert_allclose(f[0][0], expected_stream_0[0, np.newaxis])
    np.testing.assert_allclose(f[1][0], expected_stream_0[1, np.newaxis])
    np.testing.assert_allclose(f[2][0], expected_stream_0[2, np.newaxis])
    np.testing.assert_allclose(f[0][1], expected_stream_1[0])
    np.testing.assert_allclose(f[1][1], expected_stream_1[1])
    np.testing.assert_allclose(f[2][1], expected_stream_1[2])
    assert f[0][2].shape == (0, O)
    assert f[1][2].shape == (0, O)
    assert f[2][2].shape == (0, O)


def test_expected_effective_dl_channel_moments_for_complex_channels():
    f_00 = np.array(
        [
            [1 + 1j, 2 - 1j, -1 + 2j],
            [2, -1j, 3 + 1j],
        ]
    )
    f_01 = np.array([[2 + 1j, -1 + 2j, 3]])
    f_10 = np.array(
        [
            [-1j, 1 + 3j, 2],
            [2 - 2j, -1, 1j],
        ]
    )
    f_11 = np.array([[1 - 1j, 2j, -2 + 1j]])
    f = [[f_00, f_01], [f_10, f_11]]

    outers = downlink_precoding.ComputeExpectedEffectiveDLChannelOuters()(f)
    desired = downlink_precoding.ComputeExpectedDesiredEffectiveDLChannels()(f)

    np.testing.assert_allclose(outers[0][0], f_00 @ np.conj(f_00.T) / 3)
    np.testing.assert_allclose(outers[0][1], f_01 @ np.conj(f_01.T) / 3)
    np.testing.assert_allclose(outers[1][0], f_10 @ np.conj(f_10.T) / 3)
    np.testing.assert_allclose(outers[1][1], f_11 @ np.conj(f_11.T) / 3)
    np.testing.assert_allclose(desired[0], np.mean(f_00, axis=1))
    np.testing.assert_allclose(desired[1], np.mean(f_11, axis=1))


def test_uplink_direction_precoding_normalizes_each_ue_cpu_pair():
    v_00 = np.array(
        [
            [1 + 1j, 2, -1j],
            [2 - 1j, 1j, 1 + 2j],
        ]
    )
    v_01 = np.array([[3, 4j, 1 - 1j]])
    v_20 = np.array(
        [
            [1j, 2 - 1j, -2],
            [1, -1j, 3 + 1j],
            [2 + 2j, 1, -1 + 2j],
        ]
    )
    v = [[v_00, v_01], [], [v_20]]

    w = downlink_precoding.CfgUplinkDirectionPrecoding()(v)

    expected_norm_00 = np.mean(np.sum(np.abs(v_00) ** 2, axis=0))
    expected_norm_01 = np.mean(np.sum(np.abs(v_01) ** 2, axis=0))
    expected_norm_20 = np.mean(np.sum(np.abs(v_20) ** 2, axis=0))
    np.testing.assert_allclose(w[0][0], v_00 / np.sqrt(expected_norm_00))
    np.testing.assert_allclose(w[0][1], v_01 / np.sqrt(expected_norm_01))
    np.testing.assert_allclose(w[2][0], v_20 / np.sqrt(expected_norm_20))
    assert w[1] == []
    np.testing.assert_allclose(np.mean(np.sum(np.abs(w[0][0]) ** 2, axis=0)), 1)
    np.testing.assert_allclose(np.mean(np.sum(np.abs(w[0][1]) ** 2, axis=0)), 1)
    np.testing.assert_allclose(np.mean(np.sum(np.abs(w[2][0]) ** 2, axis=0)), 1)


def test_uplink_direction_precoding_rejects_zero_norm():
    v = [[np.zeros((2, 3), dtype=complex)]]

    with pytest.raises(AssertionError, match="nonzero expected squared norm"):
        downlink_precoding.CfgUplinkDirectionPrecoding()(v)


def test_expected_precoding_norm_squares_for_complex_multi_cpu_precoders():
    used = np.array(
        [
            [True, True, True],
            [False, True, False],
            [False, False, False],
        ]
    )
    fronthaul = np.array(
        [
            [True, False],
            [True, False],
            [False, True],
        ]
    )
    w = [
        [
            np.array(
                [
                    [0.5, 0],
                    [0, 0.5j],
                    [np.sqrt(0.75), 0],
                    [0, np.sqrt(0.75) * 1j],
                ]
            ),
            np.array([[1, 0], [0, 1j]]),
        ],
        [np.array([[1j, 0], [0, -1]])],
        [],
    ]

    exp_w_H_w = downlink_precoding.ComputeExpectedPrecodingNormSquares()(
        K=3,
        L=3,
        N=2,
        O=2,
        w=w,
        used=used,
        fronthaul=fronthaul,
    )

    np.testing.assert_allclose(
        exp_w_H_w,
        np.array(
            [
                [0.25, 0.75, 1],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
    )
