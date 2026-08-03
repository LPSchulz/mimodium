import numpy as np
from mimodium.algorithms import power_control


def test_max_uplink_power_returns_independent_copy_of_power_limits():
    power_limits = np.array([1.0, 2.0, 4.0])

    powers = power_control.CfgMaxUplinkPower()(K=3, ue_max_power=power_limits)

    np.testing.assert_array_equal(powers, power_limits)
    assert not np.shares_memory(powers, power_limits)


def test_equal_power_importance_weights_are_one_only_at_serving_cpus():
    importance = power_control.CfgEqualPowerImportanceWeights()(
        K=3,
        J=3,
        used=np.array(
            [
                [True, False, True],
                [False, True, False],
                [False, False, False],
            ]
        ),
        fronthaul=np.eye(3, dtype=bool),
    )

    np.testing.assert_array_equal(
        importance,
        np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def test_lsfd_power_importance_uses_normalized_complex_weight_energy():
    used = np.array(
        [
            [True, False, True],
            [False, True, False],
            [False, False, False],
        ]
    )
    fronthaul = np.eye(3, dtype=bool)
    a = [
        np.array([1 + 1j, 2j]),
        np.array([-3j]),
        np.array([], dtype=complex),
    ]

    importance = power_control.CfgLSFDBasedPowerImportanceWeights()(
        K=3,
        J=3,
        used=used,
        fronthaul=fronthaul,
        a=a,
    )

    np.testing.assert_allclose(
        importance,
        np.array(
            [
                [1 / 3, 0, 2 / 3],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
    )


def test_lsfd_power_importance_is_zero_for_zero_weights():
    importance = power_control.CfgLSFDBasedPowerImportanceWeights()(
        K=1,
        J=2,
        used=np.array([[True, True]]),
        fronthaul=np.eye(2, dtype=bool),
        a=[np.zeros(2, dtype=complex)],
    )

    np.testing.assert_array_equal(importance, np.zeros((1, 2)))


def test_ssfd_power_importance_averages_normalized_energy_per_realization():
    used = np.array(
        [
            [True, False, True],
            [False, True, False],
            [False, False, False],
        ]
    )
    fronthaul = np.eye(3, dtype=bool)
    a = [
        np.array(
            [
                [1, -3j, 0],
                [2, -6j, 5],
            ]
        ),
        np.array([[1 + 1j, -4, 2j]]),
        np.empty((0, 3), dtype=complex),
    ]

    importance = power_control.CfgSSFDBasedPowerImportanceWeights()(
        K=3,
        J=3,
        used=used,
        fronthaul=fronthaul,
        a=a,
    )

    np.testing.assert_allclose(
        importance,
        np.array(
            [
                [2 / 15, 0, 13 / 15],
                [0, 1, 0],
                [0, 0, 0],
            ]
        ),
    )
    np.testing.assert_allclose(np.sum(importance[:2], axis=1), np.ones(2))


def test_ssfd_power_importance_is_zero_for_zero_weight_realizations():
    importance = power_control.CfgSSFDBasedPowerImportanceWeights()(
        K=1,
        J=2,
        used=np.array([[True, True]]),
        fronthaul=np.eye(2, dtype=bool),
        a=[np.array([[0, 1], [0, 2]], dtype=complex)],
    )

    np.testing.assert_allclose(importance, np.array([[0.1, 0.4]]))


def test_ssfd_power_importance_is_invariant_to_per_realization_phase_and_scale():
    used = np.array([[True, False, True]])
    fronthaul = np.eye(3, dtype=bool)
    a = np.array(
        [
            [1 + 1j, 2 - 1j, -1j],
            [2, -1 + 2j, 3],
        ]
    )
    realization_scales = np.array([2j, 0.1, -3 + 4j])

    original = power_control.CfgSSFDBasedPowerImportanceWeights()(
        K=1,
        J=3,
        used=used,
        fronthaul=fronthaul,
        a=[a],
    )
    rescaled = power_control.CfgSSFDBasedPowerImportanceWeights()(
        K=1,
        J=3,
        used=used,
        fronthaul=fronthaul,
        a=[a * realization_scales],
    )

    np.testing.assert_allclose(rescaled, original)


def test_largest_feasible_power_scale_uses_limiting_active_ap():
    scale = power_control.largest_feasible_power_scale(
        loads=np.array([0.0, 2.0, 4.0]),
        limits=np.array([1.0, 10.0, 12.0]),
    )

    assert scale == 3.0


def test_largest_feasible_power_scale_is_zero_without_active_load():
    scale = power_control.largest_feasible_power_scale(
        loads=np.zeros(3),
        limits=np.array([1.0, 2.0, 3.0]),
    )

    assert scale == 0.0


def test_uplink_proportional_downlink_power_splits_and_globally_scales_power():
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
    exp_w_H_w = np.array(
        [
            [0.25, 0.75, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    a = [
        np.array([np.sqrt(0.8), np.sqrt(0.2)]),
        np.array([1.0]),
    ]

    rho = power_control.CfgUplinkProportionalDownlinkPower()(
        K=2,
        J=2,
        L=3,
        rho_max=np.array([2.0, 5.0, 4.0]),
        exp_w_H_w=exp_w_H_w,
        used=used,
        fronthaul=fronthaul,
        p=np.array([4.0, 2.0]),
        a=a,
    )

    global_scale = 5.0 / 4.4
    np.testing.assert_allclose(
        rho,
        np.array(
            [
                [0.8, 2.4, 0.8],
                [0, 2.0, 0],
            ]
        )
        * global_scale,
    )
    np.testing.assert_allclose(
        np.sum(rho, axis=0), np.array([0.8, 4.4, 0.8]) * global_scale
    )


def test_uplink_proportional_downlink_power_is_zero_for_zero_lsfd_weights():
    rho = power_control.CfgUplinkProportionalDownlinkPower()(
        K=1,
        J=1,
        L=1,
        rho_max=np.array([2.0]),
        exp_w_H_w=np.ones((1, 1)),
        used=np.ones((1, 1), dtype=bool),
        fronthaul=np.ones((1, 1), dtype=bool),
        p=np.zeros(1),
        a=[np.zeros(1, dtype=complex)],
    )

    np.testing.assert_array_equal(rho, np.zeros((1, 1)))


def test_equal_downlink_power_respects_importance_and_heterogeneous_ap_limits():
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
    exp_w_H_w = np.array(
        [
            [0.25, 0.75, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    importance = np.array(
        [
            [0.5, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )

    rho = power_control.CfgEqualDownlinkPower(interference_control=False)(
        K=3,
        J=2,
        L=3,
        rho_max=np.array([1.0, 5.0, 4.0]),
        exp_w_H_w=exp_w_H_w,
        used=used,
        fronthaul=fronthaul,
        importance=importance,
    )

    np.testing.assert_allclose(
        rho,
        np.array(
            [
                [5 / 11, 15 / 11, 4],
                [0, 40 / 11, 0],
                [0, 0, 0],
            ]
        ),
    )
    np.testing.assert_allclose(np.sum(rho, axis=0), np.array([5 / 11, 5, 4]))


def test_equal_downlink_power_caps_less_important_cpu_without_increasing_power():
    rho = power_control.CfgEqualDownlinkPower(interference_control=True)(
        K=1,
        J=2,
        L=2,
        rho_max=np.array([10.0, 10.0]),
        exp_w_H_w=np.ones((1, 2)),
        used=np.array([[True, True]]),
        fronthaul=np.eye(2, dtype=bool),
        importance=np.array([[1.0, 0.25]]),
    )

    np.testing.assert_allclose(rho, np.array([[10.0, 2.5]]))


def test_interference_control_selects_reference_only_from_serving_cpus():
    rho = power_control.CfgEqualDownlinkPower(interference_control=True)(
        K=2,
        J=2,
        L=2,
        rho_max=np.array([10.0, 10.0]),
        exp_w_H_w=np.array([[0.0, 1.0], [0.0, 0.0]]),
        used=np.array([[False, True], [False, False]]),
        fronthaul=np.eye(2, dtype=bool),
        importance=np.ones((2, 2)),
    )

    np.testing.assert_allclose(rho, np.array([[0.0, 10.0], [0.0, 0.0]]))
