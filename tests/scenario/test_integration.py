import numpy as np
from dagreon import Workflow
from mimodium import rng
from mimodium.scenario import ap, area, cpu, fronthaul, geometry, midhaul, power, ue


def test_wrap_around_connects_nodes_across_opposite_area_edges():
    workflow = Workflow(
        [
            area.CfgAreaLength(100.0),
            area.CfgWrapAround(True),
            ap.CfgExplicitNumAPs(2),
            ap.CfgExplicitAPPositions(np.array([[1.0, 50.0], [51.0, 50.0]])),
            ap.CfgAPHeights(10.0),
            ap.ComputeAPLocations(),
            cpu.CfgExplicitNumCPUs(3),
            cpu.CfgExplicitCPUPositions(
                np.array([[99.0, 50.0], [49.0, 50.0], [3.0, 50.0]])
            ),
            ue.CfgExplicitNumUEs(1),
            ue.CfgExplicitUEPositions(np.array([[99.0, 50.0]])),
            ue.CfgUEHeights(1.5),
            ue.ComputeUELocations(),
            geometry.ComputeUEtoAPDifferences(),
            geometry.ComputeUEtoAP2DDistances(),
            geometry.ComputeAPtoCPUDifferences(),
            geometry.ComputeAPtoCPU2DDistances(),
            geometry.ComputeCPUtoCPUDifferences(),
            geometry.ComputeCPUtoCPU2DDistances(),
            fronthaul.CfgClosestFronthaulLinks(),
            midhaul.CfgFixedRadiusMidhaulLinks(radius=5.0),
        ]
    )

    np.testing.assert_allclose(
        workflow.run(geometry.UEtoAP2DDistances), np.array([[2.0, 48.0]])
    )
    np.testing.assert_allclose(
        workflow.run(geometry.APtoCPU2DDistances),
        np.array([[2.0, 48.0, 2.0], [48.0, 2.0, 48.0]]),
    )
    np.testing.assert_array_equal(
        workflow.run(fronthaul.FronthaulLinks),
        np.array([[True, False, False], [False, True, False]]),
    )
    np.testing.assert_array_equal(
        workflow.run(midhaul.MidhaulLinks),
        np.array(
            [
                [False, False, True],
                [False, False, False],
                [True, False, False],
            ]
        ),
    )


def test_complete_scenario_produces_consistent_output_shapes_and_topologies():
    workflow = Workflow(
        [
            area.CfgAreaLength(100.0),
            area.ComputeAreaSize(),
            area.CfgWrapAround(False),
            ap.CfgExplicitNumAPs(4),
            ap.CfgNumAntennas(2),
            ap.ComputeTotalNumberAntennas(),
            ap.CfgEvenlySpacedAPPositions(),
            ap.CfgAPHeights(10.0),
            ap.ComputeAPLocations(),
            cpu.CfgExplicitNumCPUs(2),
            cpu.CfgExplicitCPUPositions(np.array([[25.0, 50.0], [75.0, 50.0]])),
            ue.CfgExplicitNumUEs(3),
            ue.CfgExplicitUEPositions(
                np.array([[10.0, 10.0], [50.0, 50.0], [90.0, 90.0]])
            ),
            ue.CfgUEHeights(1.5),
            ue.ComputeUELocations(),
            geometry.ComputeUEtoAPDifferences(),
            geometry.ComputeUEtoAP2DDistances(),
            geometry.ComputeUEtoAP3DDistances(),
            geometry.ComputeUEtoUE2DDistances(),
            geometry.ComputeAPtoCPUDifferences(),
            geometry.ComputeAPtoCPU2DDistances(),
            geometry.ComputeCPUtoCPUDifferences(),
            geometry.ComputeCPUtoCPU2DDistances(),
            fronthaul.CfgClosestFronthaulLinks(),
            midhaul.CfgFullMidhaulLinks(),
            power.CfgPilotMaxPower(100.0),
            power.CfgUEMaxPower(200.0),
            power.CfgAPMaxPower(1_000.0),
        ]
    )

    assert workflow.run(area.AreaSize) == 10_000.0
    assert workflow.run(ap.TotalNumAPAntennas) == 8
    assert workflow.run(ap.APPositions).shape == (4, 2)
    assert workflow.run(ap.APLocations).shape == (4, 3)
    assert workflow.run(cpu.CPUPositions).shape == (2, 2)
    assert workflow.run(ue.UEPositions).shape == (3, 2)
    assert workflow.run(ue.UELocations).shape == (3, 3)
    assert workflow.run(geometry.UEtoAP2DDistances).shape == (3, 4)
    assert workflow.run(geometry.UEtoAP3DDistances).shape == (3, 4)
    assert workflow.run(geometry.UEtoUE2DDistances).shape == (3, 3)
    assert workflow.run(geometry.APtoCPU2DDistances).shape == (4, 2)
    assert workflow.run(geometry.CPUtoCPU2DDistances).shape == (2, 2)
    assert workflow.run(power.PilotMaxPower).shape == (3,)
    assert workflow.run(power.UEMaxPower).shape == (3,)
    assert workflow.run(power.APMaxPower).shape == (4,)

    fronthaul_links = workflow.run(fronthaul.FronthaulLinks)
    midhaul_links = workflow.run(midhaul.MidhaulLinks)

    assert fronthaul_links.shape == (4, 2)
    np.testing.assert_array_equal(np.sum(fronthaul_links, axis=1), np.ones(4))
    assert midhaul_links.shape == (2, 2)
    np.testing.assert_array_equal(midhaul_links, midhaul_links.T)
    np.testing.assert_array_equal(np.diag(midhaul_links), np.zeros(2, dtype=bool))


def test_seeded_random_scenario_is_reproducible():
    workflow = Workflow(
        [
            area.CfgAreaLength(100.0),
            area.CfgWrapAround(False),
            ap.CfgExplicitNumAPs(8),
            ap.CfgUniformRandomAPPositions(),
            ap.CfgRandomAzimuthArrayOrientations(),
            cpu.CfgExplicitNumCPUs(3),
            cpu.CfgUniformRandomCPUPositions(),
            ue.CfgExplicitNumUEs(12),
            ue.CfgUniformRandomUEPositions(),
            geometry.ComputeAPtoCPUDifferences(),
            geometry.ComputeAPtoCPU2DDistances(),
            fronthaul.CfgClosestFronthaulLinks(),
        ]
    )

    def run_scenario(seed: int) -> tuple[np.ndarray, ...]:
        seed_override = [rng.CfgSeed(seed)]
        return (
            workflow.run(ap.APPositions, overrides=seed_override),
            workflow.run(ap.APArrayOrientations, overrides=seed_override),
            workflow.run(cpu.CPUPositions, overrides=seed_override),
            workflow.run(ue.UEPositions, overrides=seed_override),
            workflow.run(fronthaul.FronthaulLinks, overrides=seed_override),
        )

    first = run_scenario(seed=7)
    repeated = run_scenario(seed=7)
    different = run_scenario(seed=8)

    for first_value, repeated_value in zip(first, repeated, strict=True):
        np.testing.assert_array_equal(first_value, repeated_value)

    for first_value, different_value in zip(first[:4], different[:4], strict=True):
        assert not np.array_equal(first_value, different_value)
