import numpy as np
import pytest
from dagreon import Workflow
from mimodium import scenario as scn
from mimodium.propagation import path_loss, radio_parameters


def test_exponential_path_loss_matches_log_distance_reference_case():
    distances = np.array([[1.0, 10.0]])

    values = path_loss.CfgExponentialPathLoss(
        reference_path_loss_db=30.0, pathloss_exponent=2.0
    )(distances)

    np.testing.assert_allclose(values, np.array([[30.0, 50.0]]))


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("reference_path_loss_db", np.nan),
        ("pathloss_exponent", 0.0),
        ("pathloss_exponent", np.inf),
    ],
)
def test_exponential_path_loss_rejects_invalid_parameters(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        path_loss.CfgExponentialPathLoss(**{parameter: value})


def test_path_loss_consumes_shared_carrier_frequency_in_workflow():
    workflow = Workflow(
        [
            scn.area.CfgAreaLength(100.0),
            scn.area.CfgWrapAround(False),
            scn.ap.CfgExplicitNumAPs(1),
            scn.ap.CfgExplicitAPPositions(np.array([[10.0, 0.0]])),
            scn.ap.CfgAPHeights(10.0),
            scn.ap.ComputeAPLocations(),
            scn.ue.CfgExplicitNumUEs(1),
            scn.ue.CfgExplicitUEPositions(np.array([[0.0, 0.0]])),
            scn.ue.CfgUEHeights(1.5),
            scn.ue.ComputeUELocations(),
            scn.geometry.ComputeUEtoAPDifferences(),
            scn.geometry.ComputeUEtoAP2DDistances(),
            scn.geometry.ComputeUEtoAP3DDistances(),
            radio_parameters.CfgCarrierFrequency(2e9),
            path_loss.ComputeAlwaysNonLineOfSight(),
            path_loss.CfgUrbanMicro3GPPPathLoss(),
        ]
    )

    values = workflow.run(path_loss.PathLossdB)

    assert values.shape == (1, 1)
    assert np.isfinite(values[0, 0])


def test_rural_macro_warns_outside_environment_parameter_applicability():
    applicability_inputs = {
        "ue_heights": np.array([1.5]),
        "ap_heights": np.array([35.0]),
        "distances_2d": np.array([[100.0]]),
        "distances_3d": np.array([[105.0]]),
        "los_states": np.array([[True]]),
        "fc": 2e9,
    }

    with pytest.warns(path_loss.ApplicabilityWarning, match="h"):
        path_loss.CfgRuralMacro3GPPPathLoss(h=4.0).check_applicability(
            **applicability_inputs
        )

    with pytest.warns(path_loss.ApplicabilityWarning, match="w"):
        path_loss.CfgRuralMacro3GPPPathLoss(w=60.0).check_applicability(
            **applicability_inputs
        )


@pytest.mark.parametrize(("parameter", "value"), [("h", 0.0), ("w", np.nan)])
def test_rural_macro_rejects_invalid_environment_parameters(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        path_loss.CfgRuralMacro3GPPPathLoss(**{parameter: value})


@pytest.mark.parametrize(
    ("task", "ap_height", "expected"),
    [
        (
            path_loss.CfgRuralMacro3GPPPathLoss(),
            35.0,
            [79.33765375396655, 114.67394403992665, 125.56354510244701],
        ),
        (
            path_loss.CfgSuburbanMacro3GPPPathLoss(),
            30.0,
            [79.87588788038525, 118.65944944380344, 132.07818284081262],
        ),
    ],
)
def test_rural_and_suburban_macro_path_loss_reference_points(task, ap_height, expected):
    distances_2d = np.array([[100.0, 3_000.0, 1_000.0]])
    distances_3d = np.sqrt(distances_2d**2 + (ap_height - 1.5) ** 2)

    values = task(
        K=1,
        L=3,
        ue_heights=np.array([1.5]),
        ap_heights=np.full(3, ap_height),
        distances_2d=distances_2d,
        distances_3d=distances_3d,
        los_states=np.array([[True, True, False]]),
        fc=2e9,
    )

    np.testing.assert_allclose(values[0], expected)


@pytest.mark.parametrize(
    ("task", "ap_height", "distances_2d", "expected"),
    [
        (
            path_loss.CfgUrbanMacro3GPPPathLoss(seed_override=1),
            25.0,
            [100.0, 1_000.0, 100.0],
            [78.27739570312649, 108.91167278954842, 98.17676261633486],
        ),
        (
            path_loss.CfgUrbanMicro3GPPPathLoss(),
            10.0,
            [50.0, 500.0, 50.0],
            [74.22888826017217, 106.85681741312263, 89.00396646218121],
        ),
    ],
)
def test_urban_path_loss_default_reference_points(
    task, ap_height, distances_2d, expected
):
    distances_2d = np.array([distances_2d])
    distances_3d = np.sqrt(distances_2d**2 + (ap_height - 1.5) ** 2)
    kwargs = (
        {"seed": 2} if isinstance(task, path_loss.CfgUrbanMacro3GPPPathLoss) else {}
    )

    values = task(
        K=1,
        L=3,
        ue_heights=np.array([1.5]),
        ap_heights=np.full(3, ap_height),
        distances_2d=distances_2d,
        distances_3d=distances_3d,
        los_states=np.array([[True, True, False]]),
        fc=2e9,
        **kwargs,
    )

    np.testing.assert_allclose(values[0], expected)


def test_urban_micro_path_loss_selects_los_or_maximum_nlos_formula():
    task = path_loss.CfgUrbanMicro3GPPPathLoss()
    distances_2d = np.array([[10.0, 10.0]])
    distances_3d = np.array([[10.0, 10.0]])
    los_states = np.array([[True, False]])

    values = task(
        K=1,
        L=2,
        ue_heights=np.array([1.5]),
        ap_heights=np.array([10.0, 10.0]),
        distances_2d=distances_2d,
        distances_3d=distances_3d,
        los_states=los_states,
        fc=2e9,
    )

    los_value = task.pl1(10.0, 2e9)
    nlos_value = max(los_value, task.pl_umi_nlos_prime(10.0, 1.5, 2e9))
    np.testing.assert_allclose(values, np.array([[los_value, nlos_value]]))


@pytest.mark.parametrize(
    ("task", "ap_height", "expected_nlos"),
    [
        (
            path_loss.CfgUrbanMacro3GPPPathLoss(use_optional=True, seed_override=1),
            25.0,
            98.42059991327963,
        ),
        (
            path_loss.CfgUrbanMicro3GPPPathLoss(use_optional=True),
            10.0,
            102.22059991327962,
        ),
    ],
)
def test_optional_outdoor_nlos_path_loss_matches_3gpp(task, ap_height, expected_nlos):
    values = task(
        K=1,
        L=1,
        ue_heights=np.array([1.5]),
        ap_heights=np.array([ap_height]),
        distances_2d=np.array([[100.0]]),
        distances_3d=np.array([[100.0]]),
        los_states=np.array([[False]]),
        fc=2e9,
        **(
            {"seed": 2} if isinstance(task, path_loss.CfgUrbanMacro3GPPPathLoss) else {}
        ),
    )

    np.testing.assert_allclose(values, [[expected_nlos]])


def test_indoor_office_warns_outside_documented_distance_range():
    task = path_loss.CfgIndoorOffice3GPPPathLoss()

    with pytest.warns(path_loss.ApplicabilityWarning, match="distances_3d"):
        values = task(
            K=1,
            L=1,
            distances_3d=np.array([[0.5]]),
            los_states=np.array([[True]]),
            fc=2e9,
        )

    np.testing.assert_allclose(values, np.array([[task.pl_inh_los(0.5, 2e9)]]))


def test_indoor_office_optional_nlos_path_loss_matches_3gpp():
    task = path_loss.CfgIndoorOffice3GPPPathLoss(use_optional=True)

    values = task(
        K=1,
        L=1,
        distances_3d=np.array([[100.0]]),
        los_states=np.array([[False]]),
        fc=2e9,
    )

    np.testing.assert_allclose(values, [[102.22059991327962]])


@pytest.mark.parametrize(
    ("subscenario", "expected_nlos"),
    [
        ("SL", 90.02059991327963),
        ("DL", 96.02059991327963),
        ("SH", 84.42059991327963),
        ("DH", 83.45059991327963),
    ],
)
def test_indoor_factory_path_loss_matches_3gpp(subscenario, expected_nlos):
    task = path_loss.CfgIndoorFactory3GPPPathLoss(subscenario=subscenario)

    values = task(
        K=1,
        L=2,
        distances_3d=np.array([[100.0, 100.0]]),
        los_states=np.array([[True, False]]),
        fc=2e9,
    )

    np.testing.assert_allclose(values[0, 0], 80.55956991761564)
    np.testing.assert_allclose(values[0, 1], expected_nlos)


def test_indoor_factory_rejects_unknown_subscenario():
    with pytest.raises(ValueError, match="subscenario"):
        path_loss.CfgIndoorFactory3GPPPathLoss(subscenario="HH")  # type: ignore


def test_indoor_factory_dense_low_is_not_below_sparse_low():
    task = path_loss.CfgIndoorFactory3GPPPathLoss(subscenario="DL")

    values = task(
        K=1,
        L=1,
        distances_3d=np.array([[1.0]]),
        los_states=np.array([[False]]),
        fc=2e9,
    )

    np.testing.assert_allclose(values, [[39.020599913279625]])


def test_suburban_macro_path_loss_selects_los_or_maximum_nlos_formula():
    task = path_loss.CfgSuburbanMacro3GPPPathLoss()

    values = task(
        K=1,
        L=2,
        ue_heights=np.array([1.5]),
        ap_heights=np.array([30.0, 30.0]),
        distances_2d=np.array([[100.0, 100.0]]),
        distances_3d=np.array([[104.0, 104.0]]),
        los_states=np.array([[True, False]]),
        fc=2e9,
    )

    np.testing.assert_allclose(values[0, 0], 79.87754849231443)
    np.testing.assert_allclose(values[0, 1], 93.89200147167062)


@pytest.mark.parametrize(
    ("task", "applicability_inputs", "fc"),
    [
        (
            path_loss.CfgRuralMacro3GPPPathLoss(),
            {
                "ue_heights": np.array([1.5]),
                "ap_heights": np.array([35.0]),
                "distances_2d": np.array([[100.0]]),
                "distances_3d": np.array([[105.0]]),
                "los_states": np.array([[True]]),
            },
            30e9,
        ),
        (
            path_loss.CfgSuburbanMacro3GPPPathLoss(),
            {
                "ue_heights": np.array([1.5]),
                "ap_heights": np.array([30.0]),
                "distances_2d": np.array([[100.0]]),
                "distances_3d": np.array([[104.0]]),
            },
            37e9,
        ),
        (
            path_loss.CfgUrbanMacro3GPPPathLoss(),
            {
                "ue_heights": np.array([1.5]),
                "ap_heights": np.array([25.0]),
                "distances_2d": np.array([[100.0]]),
                "distances_3d": np.array([[103.0]]),
            },
            100e9,
        ),
        (
            path_loss.CfgUrbanMicro3GPPPathLoss(),
            {
                "ue_heights": np.array([1.5]),
                "ap_heights": np.array([10.0]),
                "distances_2d": np.array([[100.0]]),
                "distances_3d": np.array([[101.0]]),
            },
            100e9,
        ),
        (
            path_loss.CfgIndoorOffice3GPPPathLoss(),
            {"distances_3d": np.array([[100.0]])},
            100e9,
        ),
        (
            path_loss.CfgIndoorFactory3GPPPathLoss(),
            {"distances_3d": np.array([[100.0]])},
            100e9,
        ),
    ],
)
def test_each_3gpp_task_checks_its_carrier_frequency_applicability(
    task, applicability_inputs, fc
):
    with pytest.warns(path_loss.ApplicabilityWarning, match="fc"):
        task.check_applicability(fc=fc, **applicability_inputs)


def test_uma_environment_height_choices_stop_below_ue_height():
    class ChooseMaximum:
        def binomial(self, *args):
            return 0

        def choice(self, values):
            return values[-1]

    task = path_loss.CfgUrbanMacro3GPPPathLoss()

    assert task.calculate_h_e(300.0, 15.0, ChooseMaximum()) == 12.0  # type: ignore


def test_outdoor_models_warn_outside_documented_distance_range():
    task = path_loss.CfgUrbanMicro3GPPPathLoss()

    with pytest.warns(path_loss.ApplicabilityWarning, match="distances_2d"):
        task(
            K=1,
            L=1,
            ue_heights=np.array([1.5]),
            ap_heights=np.array([10.0]),
            distances_2d=np.array([[5.0]]),
            distances_3d=np.array([[10.0]]),
            los_states=np.array([[True]]),
            fc=2e9,
        )


def test_hata_cost231_path_loss_uses_flat_near_field_below_d0():
    task = path_loss.CfgHataCost231PathLoss(d0=10.0, d1=50.0)

    values = task(
        K=1,
        L=2,
        ue_heights=np.array([1.5]),
        ap_heights=np.array([10.0, 10.0]),
        distances_2d=np.array([[5.0, 10.0]]),
        fc=2e9,
    )

    np.testing.assert_allclose(values[0, 0], values[0, 1])


def test_hata_cost231_rejects_invalid_breakpoints():
    with pytest.raises(ValueError, match="d0"):
        path_loss.CfgHataCost231PathLoss(d0=0.0)

    with pytest.raises(ValueError, match="d0 must not exceed d1"):
        path_loss.CfgHataCost231PathLoss(d0=60.0, d1=50.0)
