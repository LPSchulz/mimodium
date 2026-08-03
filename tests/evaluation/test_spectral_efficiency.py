import numpy as np
from mimodium.evaluation import spectral_efficiency


def test_uplink_standard_bound_uses_conditional_multicpu_uncertainty():
    g_known_0 = np.array(
        [
            [[2 + 1j, 1 - 1j], [0.5, 1j]],
            [[1j, 2], [1 - 0.5j, 0.25]],
        ]
    )
    conditional_0 = np.array(
        [
            [
                [[0.4, 0.8], [0.1j, 0.0]],
                [[-0.1j, 0.0], [0.3, 0.5]],
            ],
            [
                [[0.2, 0.6], [0.05, -0.1j]],
                [[0.05, 0.1j], [0.7, 0.4]],
            ],
        ],
        dtype=complex,
    )
    a_0 = np.array([[1.0, 0.5 + 0.5j], [0.25j, 1.0]])
    p = np.array([2.0, 1.5])
    v_H_v_0 = np.array([[1.0, 2.0], [0.5, 1.25]])
    sigma2_ul = 0.5

    ses = spectral_efficiency.CfgUplinkStandardBound()(
        K=2,
        g_known=[g_known_0, np.zeros((2, 0, 2), dtype=complex)],
        exp_g_g_H_unknown=[
            conditional_0,
            np.zeros((2, 0, 0, 2), dtype=complex),
        ],
        p=p,
        a=[a_0, np.zeros((0, 2), dtype=complex)],
        sigma2_ul=sigma2_ul,
        v_H_v=[v_H_v_0, np.zeros((0, 2))],
        tau_u=2,
        tau_c=5,
    )

    effective = np.sum(np.conj(a_0)[np.newaxis, :, :] * g_known_0, axis=1)
    desired = p[0] * np.abs(effective[0]) ** 2
    received = np.sum(p[:, np.newaxis] * np.abs(effective) ** 2, axis=0)
    conditional_sum = np.sum(
        p[:, np.newaxis, np.newaxis, np.newaxis] * conditional_0,
        axis=0,
    )
    uncertainty = np.array(
        [
            np.real(np.conj(a_0[:, 0]) @ conditional_sum[:, :, 0] @ a_0[:, 0]),
            np.real(np.conj(a_0[:, 1]) @ conditional_sum[:, :, 1] @ a_0[:, 1]),
        ]
    )
    noise = sigma2_ul * np.sum(np.abs(a_0) ** 2 * v_H_v_0, axis=0)
    expected_0 = 0.4 * np.mean(
        np.log2(1 + desired / (received - desired + uncertainty + noise))
    )
    np.testing.assert_allclose(ses, np.array([expected_0, 0.0]))


def test_uplink_genie_aided_bound_handles_complex_multicpu_fusion():
    g_0 = np.array(
        [
            [[1 + 1j, 2], [0.5, 1 - 0.5j]],
            [[0.5j, 1], [1, 0.25j]],
            [[0.2, 0.3j], [0.4, 0.1]],
        ]
    )
    g_1 = np.array(
        [
            [[0.25, 0.5j]],
            [[1 - 1j, 2 + 0.5j]],
            [[0.1j, 0.2]],
        ]
    )
    g_2 = np.zeros((3, 0, 2), dtype=complex)
    a_0 = np.array([1 + 0.5j, 0.75 - 0.25j])
    a_1 = np.array([1j])
    a_2 = np.empty(0, dtype=complex)
    p = np.array([2.0, 1.5, 0.75])
    v_H_v_0 = np.array([[1.0, 1.5], [0.5, 1.0]])
    v_H_v_1 = np.array([[2.0, 0.75]])
    sigma2_ul = 0.4

    ses = spectral_efficiency.CfgUplinkGenieAidedBound()(
        K=3,
        g=[g_0, g_1, g_2],
        a=[a_0, a_1, a_2],
        p=p,
        v_H_v=[v_H_v_0, v_H_v_1, np.zeros((0, 2))],
        sigma2_ul=sigma2_ul,
        tau_u=2,
        tau_c=5,
    )

    effective_0 = np.einsum("j,ijo->io", np.conj(a_0), g_0)
    desired_0 = p[0] * np.abs(effective_0[0]) ** 2
    received_0 = np.sum(p[:, np.newaxis] * np.abs(effective_0) ** 2, axis=0)
    noise_0 = sigma2_ul * np.sum(np.abs(a_0[:, np.newaxis]) ** 2 * v_H_v_0, axis=0)
    effective_1 = np.einsum("j,ijo->io", np.conj(a_1), g_1)
    desired_1 = p[1] * np.abs(effective_1[1]) ** 2
    received_1 = np.sum(p[:, np.newaxis] * np.abs(effective_1) ** 2, axis=0)
    noise_1 = sigma2_ul * np.sum(np.abs(a_1[:, np.newaxis]) ** 2 * v_H_v_1, axis=0)
    expected = 0.4 * np.array(
        [
            np.mean(np.log2(1 + desired_0 / (received_0 - desired_0 + noise_0))),
            np.mean(np.log2(1 + desired_1 / (received_1 - desired_1 + noise_1))),
            0.0,
        ]
    )
    np.testing.assert_allclose(ses, expected)


def test_uplink_uatf_bound_handles_complex_multicpu_moments():
    exp_g_0 = np.array(
        [
            [1 + 1j, 0.5],
            [0.25j, 1],
            [0.2, 0.5j],
        ]
    )
    exp_g_g_H_0 = np.stack(
        [
            np.outer(exp_g_0[0], np.conj(exp_g_0[0])) + 0.3 * np.eye(2),
            np.outer(exp_g_0[1], np.conj(exp_g_0[1])) + 0.4 * np.eye(2),
            np.outer(exp_g_0[2], np.conj(exp_g_0[2])) + 0.2 * np.eye(2),
        ]
    )
    exp_g_1 = np.array([[0.25j], [1 - 0.5j], [0.2]])
    exp_g_g_H_1 = np.array(
        [
            [[np.abs(exp_g_1[0, 0]) ** 2 + 0.1]],
            [[np.abs(exp_g_1[1, 0]) ** 2 + 0.25]],
            [[np.abs(exp_g_1[2, 0]) ** 2 + 0.15]],
        ]
    )
    a_0 = np.array([1 + 0.5j, 0.75 - 0.25j])
    a_1 = np.array([1 + 1j])
    a_2 = np.empty(0, dtype=complex)
    p = np.array([2.0, 1.5, 0.75])
    exp_v_H_v_0 = np.array([1.0, 0.5])
    exp_v_H_v_1 = np.array([1.25])
    sigma2_ul = 0.4

    ses = spectral_efficiency.CfgUplinkUatfBound()(
        K=3,
        exp_g_g_H=[
            exp_g_g_H_0,
            exp_g_g_H_1,
            np.zeros((3, 0, 0), dtype=complex),
        ],
        exp_g=[exp_g_0, exp_g_1, np.zeros((3, 0), dtype=complex)],
        exp_v_H_v=[exp_v_H_v_0, exp_v_H_v_1, np.zeros(0)],
        p=p,
        a=[a_0, a_1, a_2],
        sigma2_ul=sigma2_ul,
        tau_u=2,
        tau_c=5,
    )

    desired_0 = p[0] * np.abs(np.conj(a_0) @ exp_g_0[0]) ** 2
    received_0 = np.real(
        np.sum(p * np.einsum("j,ijm,m->i", np.conj(a_0), exp_g_g_H_0, a_0))
    )
    noise_0 = sigma2_ul * np.abs(a_0) ** 2 @ exp_v_H_v_0
    desired_1 = p[1] * np.abs(np.conj(a_1) @ exp_g_1[1]) ** 2
    received_1 = np.real(
        np.sum(p * np.einsum("j,ijm,m->i", np.conj(a_1), exp_g_g_H_1, a_1))
    )
    noise_1 = sigma2_ul * np.abs(a_1) ** 2 @ exp_v_H_v_1
    expected = 0.4 * np.array(
        [
            np.log2(1 + desired_0 / (received_0 - desired_0 + noise_0)),
            np.log2(1 + desired_1 / (received_1 - desired_1 + noise_1)),
            0.0,
        ]
    )
    np.testing.assert_allclose(ses, expected)


def test_downlink_uatf_bound_handles_multicpu_coherence_and_unserved_ues():
    used = np.array(
        [
            [True, True, True],
            [True, False, False],
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
    rho = np.array(
        [
            [1.0, 3.0, 4.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    exp_f_0 = np.array([1 + 1j, 0.5])
    exp_f_1 = np.array([2 - 1j])
    exp_f_2 = np.empty(0, dtype=complex)
    exp_f_f_H = [
        [
            np.outer(exp_f_0, np.conj(exp_f_0)) + 0.5 * np.eye(2),
            np.array([[1.5]]),
            np.zeros((0, 0)),
        ],
        [
            np.array([[1.0, 0.2j], [-0.2j, 2.0]]),
            np.array([[np.abs(exp_f_1[0]) ** 2 + 0.4]]),
            np.zeros((0, 0)),
        ],
        [
            np.array([[0.5, 0.0], [0.0, 0.25]]),
            np.array([[0.2]]),
            np.zeros((0, 0)),
        ],
    ]
    sigma2_dl = 0.75

    ses = spectral_efficiency.CfgDLUatFBound()(
        K=3,
        used=used,
        fronthaul=fronthaul,
        tau_d=3,
        tau_c=5,
        exp_f=[exp_f_0, exp_f_1, exp_f_2],
        exp_f_f_H=exp_f_f_H,
        rho=rho,
        sigma2_dl=sigma2_dl,
    )

    sqrt_rho_0 = np.sqrt(np.array([4.0, 4.0]))
    sqrt_rho_1 = np.sqrt(np.array([2.0]))
    desired_0 = np.abs(sqrt_rho_0 @ exp_f_0) ** 2
    received_0 = np.real(sqrt_rho_0 @ exp_f_f_H[0][0] @ sqrt_rho_0) + np.real(
        sqrt_rho_1 @ exp_f_f_H[0][1] @ sqrt_rho_1
    )
    desired_1 = np.abs(sqrt_rho_1 @ exp_f_1) ** 2
    received_1 = np.real(sqrt_rho_0 @ exp_f_f_H[1][0] @ sqrt_rho_0) + np.real(
        sqrt_rho_1 @ exp_f_f_H[1][1] @ sqrt_rho_1
    )
    expected = 0.6 * np.array(
        [
            np.log2(1 + desired_0 / (received_0 - desired_0 + sigma2_dl)),
            np.log2(1 + desired_1 / (received_1 - desired_1 + sigma2_dl)),
            0.0,
        ]
    )
    np.testing.assert_allclose(ses, expected)
