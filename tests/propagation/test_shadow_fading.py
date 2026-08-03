import numpy as np
import pytest
from dagreon import Workflow
from mimodium import rng
from mimodium import scenario as scn
from mimodium.propagation import path_loss
from mimodium.propagation import shadow_fading as sf


def _inputs(K=2, L=3):
    ue_positions = np.column_stack((np.arange(K, dtype=float), np.zeros(K)))
    ue_distances = np.linalg.norm(
        ue_positions[:, np.newaxis, :] - ue_positions[np.newaxis, :, :], axis=2
    )
    return {
        "los_states": np.zeros((K, L), dtype=bool),
        "ue_distances": ue_distances,
    }


def _simple_inputs(K=2, L=3):
    inputs = _inputs(K, L)
    inputs.pop("los_states")
    inputs.update(K=K, L=L)
    return inputs


def test_zero_shadow_fading_returns_zero_matrix():
    np.testing.assert_array_equal(sf.CfgZeroShadowFading()(K=2, L=3), np.zeros((2, 3)))


def test_zero_decorrelation_distance_samples_independently():
    task = sf.CfgLogNormalShadowFading(
        sigma_db=1.0,
        decorrelation_distance_m=0.0,
        seed_override=5,
    )

    values = task(**_simple_inputs(K=3, L=40_000), seed=1)

    correlations = np.corrcoef(values)
    np.testing.assert_allclose(correlations, np.eye(3), atol=0.015)


def test_seed_override_ignores_global_seed():
    task = sf.CfgLogNormalShadowFading(sigma_db=1.0, seed_override=5)

    first = task(**_simple_inputs(), seed=1)
    second = task(**_simple_inputs(), seed=2)

    np.testing.assert_array_equal(first, second)


def test_shadow_fading_task_resolves_its_own_sampling_dependencies_in_workflow():
    workflow = Workflow(
        [
            rng.CfgSeed(5),
            scn.area.CfgAreaLength(100.0),
            scn.area.CfgWrapAround(False),
            scn.ap.CfgExplicitNumAPs(2),
            scn.ue.CfgExplicitNumUEs(2),
            scn.ue.CfgExplicitUEPositions(np.array([[0.0, 0.0], [10.0, 0.0]])),
            scn.geometry.ComputeUEtoUE2DDistances(),
            path_loss.ComputeAlwaysNonLineOfSight(),
            sf.CfgLogNormalShadowFading(
                sigma_db=1.0,
                decorrelation_distance_m=10.0,
            ),
        ]
    )

    values = workflow.run(sf.ShadowFadingdB)

    assert values.shape == (2, 2)


def test_correlation_is_exp_minus_one_at_the_decorrelation_distance():
    task = sf.CfgLogNormalShadowFading(
        sigma_db=1.0,
        decorrelation_distance_m=10.0,
        seed_override=5,
    )
    inputs = _simple_inputs(K=2, L=50_000)
    inputs["ue_distances"] = np.array([[0.0, 10.0], [10.0, 0.0]])

    values = task(**inputs, seed=1)

    assert np.corrcoef(values)[0, 1] == pytest.approx(np.exp(-1.0), abs=0.015)


def test_correlation_uses_the_supplied_ue_distance_matrix():
    task = sf.CfgLogNormalShadowFading(
        sigma_db=1.0,
        decorrelation_distance_m=2.0,
        seed_override=5,
    )
    inputs = _simple_inputs(K=2, L=50_000)
    inputs["ue_distances"] = np.array([[0.0, 2.0], [2.0, 0.0]])

    values = task(**inputs, seed=1)

    assert np.corrcoef(values)[0, 1] == pytest.approx(np.exp(-1.0), abs=0.015)


def test_physical_ap_sites_receive_independent_fields():
    task = sf.CfgLogNormalShadowFading(
        sigma_db=1.0,
        decorrelation_distance_m=10.0,
        seed_override=5,
    )

    # With one UE, the columns are independent physical-site realizations. Pairing a
    # large number of them provides repeated samples for checking their independence.
    values = task(**_simple_inputs(K=1, L=40_000), seed=1).reshape(-1, 2)

    assert np.corrcoef(values.T)[0, 1] == pytest.approx(0.0, abs=0.015)


def test_zero_decorrelation_distance_is_the_default():
    task = sf.CfgLogNormalShadowFading(sigma_db=1.0)

    assert task.decorrelation_distance_m == 0.0


def test_spatial_correlation_supports_co_located_ues():
    task = sf.CfgLogNormalShadowFading(
        sigma_db=1.0,
        decorrelation_distance_m=10.0,
        seed_override=5,
    )
    inputs = _simple_inputs(K=2, L=3)
    inputs["ue_distances"] = np.zeros((2, 2))

    values = task(**inputs, seed=1)

    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[0], values[1])


@pytest.mark.parametrize("sigma_db", [-1.0, np.nan, np.inf])
def test_log_normal_shadow_fading_rejects_invalid_sigma_db(sigma_db):
    with pytest.raises(ValueError, match="sigma_db"):
        sf.CfgLogNormalShadowFading(sigma_db=sigma_db)


@pytest.mark.parametrize("decorrelation_distance_m", [-10.0, np.nan, np.inf])
def test_log_normal_shadow_fading_rejects_invalid_decorrelation_distance(
    decorrelation_distance_m,
):
    with pytest.raises(ValueError, match="decorrelation_distance_m"):
        sf.CfgLogNormalShadowFading(
            sigma_db=1.0,
            decorrelation_distance_m=decorrelation_distance_m,
        )


@pytest.mark.parametrize(
    ("task", "sigma_los", "sigma_nlos"),
    [
        (
            sf.CfgUrbanMacro3GPPShadowFading(seed_override=7),
            4.0,
            6.0,
        ),
        (
            sf.CfgUrbanMicro3GPPShadowFading(seed_override=7),
            4.0,
            7.82,
        ),
        (
            sf.CfgIndoorOffice3GPPShadowFading(seed_override=7),
            3.0,
            8.03,
        ),
        (
            sf.CfgUrbanMacro3GPPShadowFading(use_optional=True, seed_override=7),
            4.0,
            7.8,
        ),
        (
            sf.CfgUrbanMicro3GPPShadowFading(use_optional=True, seed_override=7),
            4.0,
            8.2,
        ),
        (
            sf.CfgIndoorOffice3GPPShadowFading(use_optional=True, seed_override=7),
            3.0,
            8.29,
        ),
        (
            sf.CfgIndoorFactory3GPPShadowFading(subscenario="SL", seed_override=7),
            4.3,
            5.7,
        ),
        (
            sf.CfgIndoorFactory3GPPShadowFading(subscenario="DL", seed_override=7),
            4.3,
            7.2,
        ),
        (
            sf.CfgIndoorFactory3GPPShadowFading(subscenario="SH", seed_override=7),
            4.3,
            5.9,
        ),
        (
            sf.CfgIndoorFactory3GPPShadowFading(subscenario="DH", seed_override=7),
            4.3,
            4.0,
        ),
    ],
)
def test_3gpp_shadow_fading_tasks_use_documented_sigmas(task, sigma_los, sigma_nlos):
    inputs = _inputs(K=1, L=2)
    inputs["los_states"] = np.array([[True, False]])
    generator = sf.get_rng(7, task)
    standard_samples_los = generator.normal(0.0, 1.0, (1, 2))
    standard_samples_nlos = generator.normal(0.0, 1.0, (1, 2))

    values = task(**inputs, seed=1)

    np.testing.assert_allclose(
        values,
        np.array(
            [
                [
                    sigma_los * standard_samples_los[0, 0],
                    sigma_nlos * standard_samples_nlos[0, 1],
                ]
            ]
        ),
    )


def test_rural_macro_3gpp_shadow_fading_uses_distance_dependent_sigmas():
    task = sf.CfgRuralMacro3GPPShadowFading(seed_override=7)
    los_states = np.array([[True, True, False]])
    generator = sf.get_rng(7, task)
    standard_samples_los = generator.normal(0.0, 1.0, los_states.shape)
    standard_samples_nlos = generator.normal(0.0, 1.0, los_states.shape)

    values = task(
        los_states=los_states,
        ue_distances=np.zeros((1, 1)),
        distances_2d=np.array([[100.0, 3_000.0, 100.0]]),
        ue_heights=np.array([1.5]),
        ap_heights=np.array([35.0, 35.0, 35.0]),
        fc=2e9,
        seed=1,
    )

    np.testing.assert_allclose(
        values,
        np.array(
            [
                [
                    4.0 * standard_samples_los[0, 0],
                    6.0 * standard_samples_los[0, 1],
                    8.0 * standard_samples_nlos[0, 2],
                ]
            ]
        ),
    )


def test_rural_macro_3gpp_uses_condition_specific_correlation_distances():
    task = sf.CfgRuralMacro3GPPShadowFading(seed_override=7)
    ue_distances = np.array([[0.0, 5.0], [5.0, 0.0]])
    los_states = np.array([[True, False], [True, False]])
    generator = sf.get_rng(7, task)
    standard_samples_los = generator.normal(0.0, 1.0, los_states.shape)
    standard_samples_nlos = generator.normal(0.0, 1.0, los_states.shape)
    correlation_los = np.exp(-ue_distances / 37.0)
    correlation_nlos = np.exp(-ue_distances / 120.0)

    values = task(
        los_states=los_states,
        ue_distances=ue_distances,
        distances_2d=np.full((2, 2), 100.0),
        ue_heights=np.full(2, 1.5),
        ap_heights=np.full(2, 35.0),
        fc=2e9,
        seed=1,
    )

    expected_los = np.linalg.cholesky(correlation_los) @ standard_samples_los
    expected_nlos = np.linalg.cholesky(correlation_nlos) @ standard_samples_nlos
    expected = np.column_stack((4.0 * expected_los[:, 0], 8.0 * expected_nlos[:, 1]))
    np.testing.assert_allclose(values, expected)


def test_suburban_macro_3gpp_shadow_fading_uses_distance_dependent_sigmas():
    task = sf.CfgSuburbanMacro3GPPShadowFading(seed_override=7)
    los_states = np.array([[True, True, False]])
    generator = sf.get_rng(7, task)
    standard_samples_los = generator.normal(0.0, 1.0, los_states.shape)
    standard_samples_nlos = generator.normal(0.0, 1.0, los_states.shape)

    values = task(
        los_states=los_states,
        ue_distances=np.zeros((1, 1)),
        distances_2d=np.array([[100.0, 3_000.0, 100.0]]),
        ue_heights=np.array([1.5]),
        ap_heights=np.array([30.0, 30.0, 30.0]),
        fc=2e9,
        seed=1,
    )

    np.testing.assert_allclose(
        values,
        np.array(
            [
                [
                    4.0 * standard_samples_los[0, 0],
                    6.0 * standard_samples_los[0, 1],
                    8.0 * standard_samples_nlos[0, 2],
                ]
            ]
        ),
    )


def test_suburban_macro_3gpp_uses_condition_specific_correlation_distances():
    task = sf.CfgSuburbanMacro3GPPShadowFading(seed_override=7)
    ue_distances = np.array([[0.0, 5.0], [5.0, 0.0]])
    los_states = np.array([[True, False], [True, False]])
    generator = sf.get_rng(7, task)
    standard_samples_los = generator.normal(0.0, 1.0, los_states.shape)
    standard_samples_nlos = generator.normal(0.0, 1.0, los_states.shape)
    correlation_los = np.exp(-ue_distances / 40.0)
    correlation_nlos = np.exp(-ue_distances / 50.0)

    values = task(
        los_states=los_states,
        ue_distances=ue_distances,
        distances_2d=np.full((2, 2), 100.0),
        ue_heights=np.full(2, 1.5),
        ap_heights=np.full(2, 30.0),
        fc=2e9,
        seed=1,
    )

    expected_los = np.linalg.cholesky(correlation_los) @ standard_samples_los
    expected_nlos = np.linalg.cholesky(correlation_nlos) @ standard_samples_nlos
    expected = np.column_stack((4.0 * expected_los[:, 0], 8.0 * expected_nlos[:, 1]))
    np.testing.assert_allclose(values, expected)


@pytest.mark.parametrize(
    ("task", "sigma_los", "sigma_nlos", "los_distance_m", "nlos_distance_m"),
    [
        (sf.CfgUrbanMacro3GPPShadowFading(seed_override=7), 4.0, 6.0, 37.0, 50.0),
        (
            sf.CfgUrbanMicro3GPPShadowFading(seed_override=7),
            4.0,
            7.82,
            10.0,
            13.0,
        ),
        (
            sf.CfgIndoorOffice3GPPShadowFading(seed_override=7),
            3.0,
            8.03,
            10.0,
            6.0,
        ),
        (
            sf.CfgIndoorFactory3GPPShadowFading(seed_override=7),
            4.3,
            5.7,
            10.0,
            10.0,
        ),
    ],
)
def test_3gpp_tasks_use_condition_specific_correlation_distances(
    task, sigma_los, sigma_nlos, los_distance_m, nlos_distance_m
):
    inputs = _inputs(K=2, L=2)
    inputs["ue_distances"] = np.array([[0.0, 5.0], [5.0, 0.0]])
    inputs["los_states"] = np.array([[True, False], [True, False]])
    generator = sf.get_rng(7, task)
    standard_samples_los = generator.normal(0.0, 1.0, (2, 2))
    standard_samples_nlos = generator.normal(0.0, 1.0, (2, 2))
    correlation_los = np.exp(-inputs["ue_distances"] / los_distance_m)
    correlation_nlos = np.exp(-inputs["ue_distances"] / nlos_distance_m)

    values = task(**inputs, seed=1)

    expected_los = np.linalg.cholesky(correlation_los) @ standard_samples_los
    expected_nlos = np.linalg.cholesky(correlation_nlos) @ standard_samples_nlos
    expected = np.column_stack(
        (sigma_los * expected_los[:, 0], sigma_nlos * expected_nlos[:, 1])
    )
    np.testing.assert_allclose(values, expected)


def test_indoor_factory_rejects_unknown_subscenario():
    with pytest.raises(ValueError, match="subscenario"):
        sf.CfgIndoorFactory3GPPShadowFading(subscenario="HH")  # type: ignore
