import numpy as np
import pytest
from mimodium.algorithms import uplink_fusion


def test_effective_ul_channels_use_complex_channels_from_each_serving_cpu():
    K, L, N, O = 2, 3, 2, 3
    used = np.array(
        [
            [True, True, True],
            [False, True, False],
        ]
    )
    fronthaul = np.array(
        [
            [True, False],
            [True, False],
            [False, True],
        ]
    )
    samples = np.arange(K * L * N * O).reshape(K, L, N, O)
    h = samples + 1j * (samples[::-1] + 1) / 10
    h_hat = 0.7 * h + (0.5 - 0.25j)
    v = [
        [
            np.array(
                [
                    [1.0 + 0.5j, 0.5 - 0.2j, 1.2j],
                    [0.3j, 1.0 + 0.1j, -0.4j],
                    [0.8 - 0.2j, -0.5j, 0.7 + 0.3j],
                    [0.1 + 0.4j, 1.3, -0.2 + 0.1j],
                ]
            ),
            np.array(
                [
                    [0.6 + 0.2j, 1.1j, -0.3 + 0.4j],
                    [1.0 - 0.1j, 0.2 + 0.5j, 0.9],
                ]
            ),
        ],
        [
            np.array(
                [
                    [1.2 - 0.3j, 0.4 + 0.6j, -0.5j],
                    [0.7j, 1.1 - 0.2j, 0.8 + 0.1j],
                ]
            )
        ],
    ]

    g = uplink_fusion.ComputeEffectiveULChannels()(
        K=K, O=O, h=h, v=v, used=used, fronthaul=fronthaul
    )
    g_hat = uplink_fusion.ComputeEstimatedEffectiveULChannels()(
        K=K,
        O=O,
        h_hat=h_hat,
        v=v,
        used=used,
        fronthaul=fronthaul,
    )

    expected_g_0 = np.stack(
        (
            np.einsum(
                "ao,kao->ko",
                np.conj(v[0][0]),
                h[:, [0, 1]].reshape(K, 2 * N, O),
            ),
            np.einsum("ao,kao->ko", np.conj(v[0][1]), h[:, 2]),
        ),
        axis=1,
    )
    expected_g_1 = np.einsum("ao,kao->ko", np.conj(v[1][0]), h[:, 1])[:, np.newaxis]
    expected_g_hat_0 = np.stack(
        (
            np.einsum(
                "ao,kao->ko",
                np.conj(v[0][0]),
                h_hat[:, [0, 1]].reshape(K, 2 * N, O),
            ),
            np.einsum("ao,kao->ko", np.conj(v[0][1]), h_hat[:, 2]),
        ),
        axis=1,
    )
    expected_g_hat_1 = np.einsum("ao,kao->ko", np.conj(v[1][0]), h_hat[:, 1])[
        :, np.newaxis
    ]

    np.testing.assert_allclose(g[0], expected_g_0)
    np.testing.assert_allclose(g[1], expected_g_1)
    np.testing.assert_allclose(g_hat[0], expected_g_hat_0)
    np.testing.assert_allclose(g_hat[1], expected_g_hat_1)


def test_effective_ul_channel_moments_average_complex_realizations():
    g = [
        np.array(
            [
                [
                    [1.0 + 1.0j, 2.0 - 1.0j, -1.0 + 2.0j],
                    [0.5 - 0.2j, 1.5 + 0.4j, 2.0 - 0.5j],
                ],
                [
                    [0.2 + 0.3j, -0.4 + 1.2j, 1.1 - 0.7j],
                    [2.0 + 0.1j, -1.0 - 0.6j, 0.3 + 0.9j],
                ],
            ]
        ),
        np.array(
            [
                [[0.4 + 0.1j, 1.2 - 0.3j, -0.2 + 0.8j]],
                [[1.0j, 0.5 + 0.5j, 1.5 - 0.2j]],
            ]
        ),
    ]

    exp_g = uplink_fusion.ComputeExpectedEffectiveULChannels()(g)
    exp_g_g_H = uplink_fusion.ComputeExpectedEffectiveULChannelOuters()(g)

    np.testing.assert_allclose(exp_g[0], np.mean(g[0], axis=2))
    np.testing.assert_allclose(exp_g[1], np.mean(g[1], axis=2))
    for k in range(2):
        for i in range(2):
            expected_outer = np.zeros((g[k].shape[1], g[k].shape[1]), dtype=complex)
            for o in range(g[k].shape[2]):
                expected_outer += np.outer(g[k][i, :, o], np.conj(g[k][i, :, o]))
            expected_outer /= g[k].shape[2]
            np.testing.assert_allclose(exp_g_g_H[k][i], expected_outer)
            np.testing.assert_allclose(exp_g_g_H[k][i], np.conj(exp_g_g_H[k][i].T))


def test_known_and_unknown_effective_channels_split_master_cpu_information():
    K, O = 3, 2
    used = np.array(
        [
            [True, True],
            [True, True],
            [False, False],
        ]
    )
    fronthaul = np.array([[True, False], [False, True]])
    measured = np.array(
        [
            [True, True],
            [True, True],
            [True, False],
        ]
    )
    g_hat = [
        np.arange(12).reshape(K, 2, O) + 1j * np.arange(12, 24).reshape(K, 2, O),
        np.arange(24, 36).reshape(K, 2, O) + 1j * np.arange(36, 48).reshape(K, 2, O),
        np.zeros((K, 0, O), dtype=complex),
    ]
    exp_g = [
        np.array(
            [
                [1.0 + 0.2j, 2.0 - 0.1j],
                [3.0 + 0.4j, 4.0 + 0.3j],
                [5.0 - 0.2j, 6.0 + 0.5j],
            ]
        ),
        np.array(
            [
                [0.5 + 0.1j, 1.5 - 0.2j],
                [2.5 + 0.3j, 3.5 + 0.4j],
                [4.5 - 0.5j, 5.5 + 0.6j],
            ]
        ),
        np.zeros((K, 0), dtype=complex),
    ]
    g = [
        g_hat[0] + np.array([[[0.2, -0.1], [0.4j, -0.3j]]]),
        g_hat[1] + np.array([[[0.1j, -0.2j], [0.3, -0.4]]]),
        np.zeros((K, 0, O), dtype=complex),
    ]

    instantaneous = uplink_fusion.CfgMasterCPUInstantaneousEffectiveULKnowledge()(
        K=K,
        used=used,
        fronthaul=fronthaul,
        master_cpu=np.array([1, 0, 0]),
    )
    known = uplink_fusion.ComputeKnownEffectiveULChannels()(
        K=K,
        O=O,
        g_hat=g_hat,
        exp_g=exp_g,
        measured=measured,
        used=used,
        fronthaul=fronthaul,
        instantaneous=instantaneous,
    )
    unknown = uplink_fusion.ComputeUnknownEffectiveULChannels()(g, known)

    expected_known_0 = np.repeat(exp_g[0][..., np.newaxis], O, axis=2)
    expected_known_0[:, 1] = g_hat[0][:, 1]
    expected_known_1 = np.repeat(exp_g[1][..., np.newaxis], O, axis=2)
    expected_known_1[2, 1] = 0
    expected_known_1[:, 0] = g_hat[1][:, 0]
    np.testing.assert_allclose(known[0], expected_known_0)
    np.testing.assert_allclose(known[1], expected_known_1)
    assert known[2].shape == (K, 0, O)
    np.testing.assert_allclose(unknown[0], g[0] - expected_known_0)
    np.testing.assert_allclose(unknown[1], g[1] - expected_known_1)
    assert unknown[2].shape == (K, 0, O)

    all_instantaneous = (
        uplink_fusion.CfgAllServingCPUsInstantaneousEffectiveULKnowledge()(
            K=K,
            used=used,
            fronthaul=fronthaul,
        )
    )
    fully_known = uplink_fusion.ComputeKnownEffectiveULChannels()(
        K=K,
        O=O,
        g_hat=g_hat,
        exp_g=exp_g,
        measured=measured,
        used=used,
        fronthaul=fronthaul,
        instantaneous=all_instantaneous,
    )
    for fully_known_k, g_hat_k in zip(fully_known, g_hat):
        np.testing.assert_array_equal(fully_known_k, g_hat_k)


def test_conditional_unknown_effective_channel_outers_use_c_for_known_cpus():
    K, O = 2, 2
    used = np.array([[True, True], [False, False]])
    fronthaul = np.array([[True, False], [False, True]])
    v = [
        [
            np.array([[1.0, 2.0]]),
            np.array([[1j, 3.0]]),
        ],
        [],
    ]
    C = np.array(
        [
            [[[2.0]], [[3.0]]],
            [[[4.0]], [[5.0]]],
        ],
        dtype=complex,
    )
    g_unknown_0 = np.array(
        [
            [[1.0, 2.0], [1.0, 3.0]],
            [[2.0, 1.0], [2.0, 4.0]],
        ],
        dtype=complex,
    )
    g_unknown = [g_unknown_0, np.zeros((K, 0, O), dtype=complex)]

    master_instantaneous = (
        uplink_fusion.CfgMasterCPUInstantaneousEffectiveULKnowledge()(
            K=K,
            used=used,
            fronthaul=fronthaul,
            master_cpu=np.array([0, 0]),
        )
    )
    partial = uplink_fusion.ComputeConditionalUnknownEffectiveULChannelOuters()(
        K=K,
        O=O,
        g_unknown=g_unknown,
        C=C,
        v=v,
        used=used,
        fronthaul=fronthaul,
        instantaneous=master_instantaneous,
    )

    expected_partial = np.zeros((K, 2, 2, O), dtype=complex)
    expected_partial[0, 1, 1] = 5.0
    expected_partial[1, 1, 1] = 10.0
    expected_partial[0, 0, 0] = np.array([2.0, 8.0])
    expected_partial[1, 0, 0] = np.array([4.0, 16.0])
    np.testing.assert_allclose(partial[0], expected_partial)
    assert partial[1].shape == (K, 0, 0, O)

    all_instantaneous = (
        uplink_fusion.CfgAllServingCPUsInstantaneousEffectiveULKnowledge()(
            K=K,
            used=used,
            fronthaul=fronthaul,
        )
    )
    fully_instantaneous = (
        uplink_fusion.ComputeConditionalUnknownEffectiveULChannelOuters()(
            K=K,
            O=O,
            g_unknown=g_unknown,
            C=C,
            v=v,
            used=used,
            fronthaul=fronthaul,
            instantaneous=all_instantaneous,
        )
    )

    expected_full = np.zeros((K, 2, 2, O), dtype=complex)
    expected_full[0, 0, 0] = np.array([2.0, 8.0])
    expected_full[1, 0, 0] = np.array([4.0, 16.0])
    expected_full[0, 1, 1] = np.array([3.0, 27.0])
    expected_full[1, 1, 1] = np.array([5.0, 45.0])
    np.testing.assert_allclose(fully_instantaneous[0], expected_full)


def test_fusion_design_ue_tasks_return_expected_sets():
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
            [False, False, False, False],
        ]
    )
    measured = np.array(
        [
            [True, True, True, False],
            [True, True, False, False],
            [False, True, True, True],
            [True, False, False, False],
        ]
    )
    estimated = np.array(
        [
            [True, True, True, False],
            [False, True, True, False],
            [True, False, True, True],
            [False, False, False, False],
        ]
    )

    all_sets = uplink_fusion.CfgConsiderAllUEs()(K=4, used=used, fronthaul=fronthaul)
    none_sets = uplink_fusion.CfgConsiderNoUEs()(K=4, used=used, fronthaul=fronthaul)
    measured_sets = uplink_fusion.CfgConsiderUEsWithMeasuredStatistics()(
        K=4,
        measured=measured,
        used=used,
        fronthaul=fronthaul,
    )
    estimated_sets = uplink_fusion.CfgConsiderUEsWithEstimatedChannels()(
        K=4,
        estimated=estimated,
        used=used,
        fronthaul=fronthaul,
    )
    served_sets = uplink_fusion.CfgConsiderUEsServedBySameAPs()(
        K=4, used=used, fronthaul=fronthaul
    )

    assert [K_k.tolist() for K_k in all_sets] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [],
    ]
    assert [K_k.tolist() for K_k in none_sets] == [[], [], [], []]
    assert [K_k.tolist() for K_k in measured_sets] == [
        [0, 1, 2, 3],
        [0, 1, 2],
        [0, 2],
        [],
    ]
    assert [K_k.tolist() for K_k in estimated_sets] == [
        [0, 1, 2],
        [0, 1],
        [0, 1, 2],
        [],
    ]
    assert [K_k.tolist() for K_k in served_sets] == [
        [0, 1, 2],
        [0, 1],
        [0, 2],
        [],
    ]


def test_equal_fusion_weights_match_cpu_and_realization_dimensions():
    known_g = [np.zeros((2, 2, 3)), np.zeros((2, 0, 3))]
    exp_g = [np.zeros((2, 2)), np.zeros((2, 0))]

    ssfd = uplink_fusion.CfgEqualSSFDWeights()(g_known=known_g, O=3)
    lsfd = uplink_fusion.CfgEqualLSFDWeights()(exp_g=exp_g)

    np.testing.assert_array_equal(ssfd[0], np.ones((2, 3)))
    assert ssfd[1].shape == (0, 3)
    np.testing.assert_array_equal(lsfd[0], np.ones(2))
    assert lsfd[1].shape == (0,)


def test_optimal_lsfd_weights_satisfy_complex_matrix_systems():
    p_max = np.array([0.7, 1.3])
    sigma2_ul = 0.4
    exp_g = [
        np.array([[1.0 + 0.2j, 0.5 - 0.1j], [0.3 + 0.4j, 1.1 - 0.2j]]),
        np.array([[0.6 - 0.3j, 0.2 + 0.5j], [1.2 + 0.1j, 0.7 - 0.4j]]),
    ]
    exp_gg_H = [
        np.array(
            [
                [[2.0, 0.3 + 0.1j], [0.3 - 0.1j, 1.5]],
                [[1.2, -0.2 + 0.05j], [-0.2 - 0.05j, 1.8]],
            ]
        ),
        np.array(
            [
                [[1.7, 0.1 - 0.2j], [0.1 + 0.2j, 1.4]],
                [[2.2, 0.4 + 0.15j], [0.4 - 0.15j, 1.6]],
            ]
        ),
    ]
    exp_v_H_v = [np.array([1.1, 0.8]), np.array([0.9, 1.2])]
    fusion_ues = [np.array([0, 1]), np.array([1])]

    weights = uplink_fusion.CfgOptimalLSFDWeights()(
        exp_gg_H=exp_gg_H,
        exp_g=exp_g,
        exp_v_H_v=exp_v_H_v,
        p_max=p_max,
        sigma2_ul=sigma2_ul,
        fusion_ues=fusion_ues,
    )

    for k in range(2):
        system_matrix = sigma2_ul * np.diag(exp_v_H_v[k]).astype(complex)
        for i in fusion_ues[k]:
            system_matrix += p_max[i] * exp_gg_H[k][i]
        np.testing.assert_allclose(system_matrix @ weights[k], p_max[k] * exp_g[k][k])
        assert weights[k].shape == (2,)


def test_optimal_ssfd_weights_satisfy_each_complex_realization_system():
    K, O = 2, 3
    p_max = np.array([0.8, 1.4])
    sigma2_ul = 0.3
    known_g = [
        np.array(
            [
                [
                    [1.0 + 0.2j, 0.7 - 0.1j, 1.1 + 0.3j],
                    [0.4 - 0.2j, 0.9 + 0.5j, 0.6 - 0.4j],
                ],
                [
                    [0.3 + 0.6j, 1.2 - 0.3j, 0.5 + 0.1j],
                    [1.0 - 0.1j, 0.2 + 0.4j, 1.3 + 0.2j],
                ],
            ]
        ),
        np.array(
            [
                [
                    [0.8 - 0.2j, 0.4 + 0.3j, 1.0 + 0.1j],
                    [0.6 + 0.5j, 1.1 - 0.2j, 0.3 + 0.7j],
                ],
                [
                    [1.2 + 0.1j, 0.5 - 0.4j, 0.9 + 0.2j],
                    [0.2 - 0.3j, 1.0 + 0.6j, 0.7 - 0.1j],
                ],
            ]
        ),
    ]
    exp_unknown = [
        np.repeat(
            np.array(
                [
                    [[0.4, 0.05 + 0.02j], [0.05 - 0.02j, 0.3]],
                    [[0.2, -0.03j], [0.03j, 0.5]],
                ]
            )[..., np.newaxis],
            O,
            axis=3,
        ),
        np.repeat(
            np.array(
                [
                    [[0.3, 0.04 - 0.01j], [0.04 + 0.01j, 0.25]],
                    [[0.45, 0.02 + 0.03j], [0.02 - 0.03j, 0.35]],
                ]
            )[..., np.newaxis],
            O,
            axis=3,
        ),
    ]
    v_H_v = [
        np.repeat(np.array([[1.0], [0.7]]), O, axis=1),
        np.repeat(np.array([[0.8], [1.1]]), O, axis=1),
    ]
    fusion_ues = [np.array([0, 1]), np.array([0, 1])]

    weights = uplink_fusion.CfgOptimalSSFDWeights()(
        K=K,
        O=O,
        known_g=known_g,
        exp_g_g_H_unknown=exp_unknown,
        p_max=p_max,
        v_H_v=v_H_v,
        sigma2_ul=sigma2_ul,
        fusion_ues=fusion_ues,
    )

    for k in range(K):
        assert weights[k].shape == (2, O)
        for o in range(O):
            system_matrix = sigma2_ul * np.diag(v_H_v[k][:, o]).astype(complex)
            for i in fusion_ues[k]:
                system_matrix += p_max[i] * (
                    np.outer(known_g[k][i, :, o], np.conj(known_g[k][i, :, o]))
                    + exp_unknown[k][i, :, :, o]
                )
            np.testing.assert_allclose(
                system_matrix @ weights[k][:, o],
                np.sqrt(p_max[k]) * known_g[k][k, :, o],
            )


def test_fully_instantaneous_mmse_fusion_satisfies_reduced_dimension_systems():
    rng = np.random.default_rng(1234)
    K, O = 2, 4
    p_max = np.array([0.8, 1.4])
    sigma2_ul = 0.3
    g = [
        rng.normal(size=(K, 2, O)) + 1j * rng.normal(size=(K, 2, O)),
        rng.normal(size=(K, 1, O)) + 1j * rng.normal(size=(K, 1, O)),
    ]
    v_H_v = [
        rng.uniform(0.5, 1.5, size=(2, O)),
        rng.uniform(0.5, 1.5, size=(1, O)),
    ]
    fusion_ues = [np.array([0, 1]), np.array([0, 1])]

    weights = uplink_fusion.CfgFullyInstantaneousMMSEFusionWeights()(
        K=K,
        O=O,
        g=g,
        p_max=p_max,
        v_H_v=v_H_v,
        sigma2_ul=sigma2_ul,
        fusion_ues=fusion_ues,
    )

    for k in range(K):
        assert weights[k].shape == (g[k].shape[1], O)
        for o in range(O):
            system_matrix = sigma2_ul * np.diag(v_H_v[k][:, o]).astype(complex)
            for i in fusion_ues[k]:
                system_matrix += p_max[i] * np.outer(
                    g[k][i, :, o], np.conj(g[k][i, :, o])
                )
            np.testing.assert_allclose(
                system_matrix @ weights[k][:, o],
                np.sqrt(p_max[k]) * g[k][k, :, o],
            )


def test_effective_ul_channel_tasks_reject_mismatched_combining_vectors():
    used = np.array([[True]])
    fronthaul = np.array([[True]])
    h = np.zeros((1, 1, 1, 1), dtype=complex)

    with pytest.raises(AssertionError):
        uplink_fusion.ComputeEffectiveULChannels()(
            K=1, O=1, h=h, v=[[]], used=used, fronthaul=fronthaul
        )
    with pytest.raises(AssertionError):
        uplink_fusion.ComputeEstimatedEffectiveULChannels()(
            K=1,
            O=1,
            h_hat=h,
            v=[[]],
            used=used,
            fronthaul=fronthaul,
        )


def test_known_effective_ul_channels_reject_missing_master_cpu():
    with pytest.raises(AssertionError):
        uplink_fusion.CfgMasterCPUInstantaneousEffectiveULKnowledge()(
            K=1,
            used=np.array([[True]]),
            fronthaul=np.array([[True]]),
            master_cpu=np.array([1]),
        )


def test_fusion_tasks_reject_mismatched_parallel_lists():
    with pytest.raises(AssertionError):
        uplink_fusion.ComputeUnknownEffectiveULChannels()(
            g=[np.zeros((1, 1, 1))],
            g_known=[],
        )
    with pytest.raises(AssertionError):
        uplink_fusion.CfgOptimalLSFDWeights()(
            exp_gg_H=[np.zeros((1, 1, 1))],
            exp_g=[],
            exp_v_H_v=[np.ones(1)],
            p_max=np.ones(1),
            sigma2_ul=1.0,
            fusion_ues=[np.array([0])],
        )
    with pytest.raises(AssertionError):
        uplink_fusion.CfgOptimalSSFDWeights()(
            K=1,
            O=1,
            known_g=[np.zeros((1, 1, 1))],
            exp_g_g_H_unknown=[],
            p_max=np.ones(1),
            v_H_v=[np.ones((1, 1))],
            sigma2_ul=1.0,
            fusion_ues=[np.array([0])],
        )
