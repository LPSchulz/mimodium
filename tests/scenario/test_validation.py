import numpy as np
import pytest

from mimodium.scenario import ap, area, cpu, midhaul, power, ue


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: area.CfgAreaLength(0.0), "positive"),
        (lambda: area.CfgAreaLength(np.nan), "finite"),
        (lambda: ap.CfgExplicitNumAPs(0), "positive"),
        (lambda: ap.CfgNumAntennas(-1), "positive"),
        (lambda: ap.CfgAntennaSpacing(0.0), "positive"),
        (lambda: ap.CfgAlignedArrayOrientations(-0.1), r"\[0, 2π\)"),
        (lambda: ap.CfgAlignedArrayOrientations(2 * np.pi), r"\[0, 2π\)"),
        (lambda: ap.CfgPoissonNumAPs(-1.0), "non-negative"),
        (lambda: ap.CfgAPHeights(0.0), "positive"),
        (lambda: cpu.CfgExplicitNumCPUs(0), "positive"),
        (lambda: cpu.CfgPoissonNumCPUs(np.inf), "finite"),
        (lambda: ue.CfgExplicitNumUEs(-1), "positive"),
        (lambda: ue.CfgPoissonNumUEs(-1.0), "non-negative"),
        (lambda: ue.CfgUEHeights(np.nan), "finite"),
        (lambda: midhaul.CfgFixedRadiusMidhaulLinks(-1.0), "non-negative"),
        (lambda: power.CfgPilotMaxPower(-1.0), "non-negative"),
        (lambda: power.CfgUEMaxPower(np.inf), "finite"),
        (lambda: power.CfgAPMaxPower(-1.0), "non-negative"),
    ],
)
def test_scalar_configuration_rejects_invalid_values(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda positions: ap.CfgExplicitAPPositions(positions),
        lambda positions: cpu.CfgExplicitCPUPositions(positions),
        lambda positions: ue.CfgExplicitUEPositions(positions),
    ],
)
@pytest.mark.parametrize(
    "positions",
    [
        np.array([]),
        np.array([1.0, 2.0]),
        np.array([[1.0, np.nan]]),
        np.ones((2, 3)),
    ],
)
def test_explicit_positions_reject_invalid_arrays(factory, positions):
    with pytest.raises(ValueError):
        factory(positions)


@pytest.mark.parametrize(
    "factory",
    [
        lambda position: ap.CfgUniformRandomAPPositions(
            forced_first_ap_position=position
        ),
        lambda position: cpu.CfgUniformRandomCPUPositions(
            forced_first_cpu_position=position
        ),
        lambda position: ue.CfgUniformRandomUEPositions(
            forced_first_ue_position=position
        ),
    ],
)
@pytest.mark.parametrize(
    "position",
    [
        np.array([1.0]),
        np.array([1.0, np.nan]),
        np.ones((1, 2)),
    ],
)
def test_forced_positions_reject_invalid_arrays(factory, position):
    with pytest.raises(ValueError):
        factory(position)
