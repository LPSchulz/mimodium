import numpy as np
from mimodium.algorithms import uplink_combining


def test_considered_interferer_tasks_return_expected_sets():
    fronthaul = np.array(
        [
            [True, False],
            [True, False],
            [False, True],
            [False, True],
        ]
    )
    used = np.array(
        [
            [True, True, True, False],
            [False, True, False, False],
            [False, False, True, True],
        ]
    )
    measured = np.array(
        [
            [True, True, True, False],
            [True, True, False, False],
            [False, True, True, True],
        ]
    )
    estimated = np.array(
        [
            [True, True, True, False],
            [False, True, True, False],
            [True, False, True, True],
        ]
    )

    all_sets = uplink_combining.CfgConsiderAllUEs()(K=3, used=used, fronthaul=fronthaul)
    none_sets = uplink_combining.CfgConsiderNoUEs()(K=3, used=used, fronthaul=fronthaul)
    measured_sets = uplink_combining.CfgConsiderUEsWithMeasuredStatistics()(
        K=3,
        measured=measured,
        used=used,
        fronthaul=fronthaul,
    )
    estimated_sets = uplink_combining.CfgConsiderUEsWithEstimatedChannels()(
        K=3,
        estimated=estimated,
        used=used,
        fronthaul=fronthaul,
    )
    served_sets = uplink_combining.CfgConsiderUEsServedBySameAPs()(
        K=3, used=used, fronthaul=fronthaul
    )

    assert [[K_jk.tolist() for K_jk in K_jk_sets] for K_jk_sets in all_sets] == [
        [[0, 1, 2], [0, 1, 2]],
        [[0, 1, 2]],
        [[0, 1, 2]],
    ]
    assert [[K_jk.tolist() for K_jk in K_jk_sets] for K_jk_sets in none_sets] == [
        [[], []],
        [[]],
        [[]],
    ]
    assert [[K_jk.tolist() for K_jk in K_jk_sets] for K_jk_sets in measured_sets] == [
        [[0, 1, 2], [0, 2]],
        [[0, 1, 2]],
        [[0, 2]],
    ]
    assert [[K_jk.tolist() for K_jk in K_jk_sets] for K_jk_sets in estimated_sets] == [
        [[0, 1, 2], [0, 1, 2]],
        [[0, 1]],
        [[0, 1, 2]],
    ]
    assert [[K_jk.tolist() for K_jk in K_jk_sets] for K_jk_sets in served_sets] == [
        [[0, 1], [0, 2]],
        [[0, 1]],
        [[0, 2]],
    ]


def test_maximum_ratio_combining_uses_estimated_channels_on_serving_aps():
    K, L, N, O = 3, 4, 2, 2
    fronthaul = np.array(
        [
            [True, False],
            [True, False],
            [False, True],
            [False, True],
        ]
    )
    used = np.array(
        [
            [True, True, True, False],
            [False, True, False, False],
            [False, False, True, True],
        ]
    )
    h_hat = np.arange(K * L * N * O).reshape(K, L, N, O).astype(complex)

    vectors = uplink_combining.CfgMaximumRatioCombining()(
        K=K, h_hat=h_hat, used=used, fronthaul=fronthaul
    )

    assert [len(v_k) for v_k in vectors] == [2, 1, 1]
    np.testing.assert_array_equal(
        vectors[0][0], np.concatenate((h_hat[0, 0], h_hat[0, 1]))
    )
    np.testing.assert_array_equal(vectors[0][1], h_hat[0, 2])
    np.testing.assert_array_equal(vectors[1][0], h_hat[1, 1])
    np.testing.assert_array_equal(
        vectors[2][0], np.concatenate((h_hat[2, 2], h_hat[2, 3]))
    )
    assert not np.shares_memory(vectors[0][0], h_hat)
    assert not np.shares_memory(vectors[0][1], h_hat)
    assert not np.shares_memory(vectors[1][0], h_hat)
    assert not np.shares_memory(vectors[2][0], h_hat)


def test_combining_norm_squares_and_expectations_match_vector_norms():
    vectors = [
        [
            np.array([[3.0 + 4.0j, 1.0], [2.0j, 2.0 - 1.0j]]),
            np.array([[1.0 - 1.0j, 2.0 + 2.0j]]),
        ],
        [np.array([[2.0, 2.0], [1.0 + 2.0j, 3.0j]])],
    ]

    norms = uplink_combining.ComputeCombiningNormSquares()(O=2, v=vectors)
    expected = uplink_combining.ComputeExpectedCombiningNormSquares()(norms)

    np.testing.assert_allclose(norms[0], [[29.0, 6.0], [2.0, 8.0]])
    np.testing.assert_allclose(norms[1], [[9.0, 13.0]])
    np.testing.assert_allclose(expected[0], [17.5, 5.0])
    np.testing.assert_allclose(expected[1], [11.0])
