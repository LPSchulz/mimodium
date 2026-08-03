import numpy as np
from dagreon import Workflow
from mimodium.scenario import ap, area, cpu, fronthaul, geometry


def test_closest_fronthaul_links_connect_each_ap_to_nearest_cpu():
    distances = np.array([[3.0, 1.0], [2.0, 5.0], [4.0, 0.5]])

    links = fronthaul.CfgClosestFronthaulLinks()(L=3, J=2, distances_2d=distances)

    np.testing.assert_array_equal(
        links,
        np.array([[False, True], [True, False], [False, True]]),
    )
    np.testing.assert_array_equal(
        fronthaul.get_set_of_aps_connected_to_cpu_j(1, links), [0, 2]
    )
    assert fronthaul.get_cpu_connected_to_ap_l(1, links) == 0


def test_closest_fronthaul_links_use_configured_positions_and_computed_distances():
    workflow = Workflow(
        [
            area.CfgAreaLength(20.0),
            area.CfgWrapAround(False),
            ap.CfgExplicitNumAPs(3),
            cpu.CfgExplicitNumCPUs(2),
            ap.CfgExplicitAPPositions(np.array([[0.0, 0.0], [9.0, 0.0], [8.0, 8.0]])),
            cpu.CfgExplicitCPUPositions(np.array([[0.0, 1.0], [10.0, 1.0]])),
            geometry.ComputeAPtoCPUDifferences(),
            geometry.ComputeAPtoCPU2DDistances(),
            fronthaul.CfgClosestFronthaulLinks(),
        ]
    )

    links = workflow.run(fronthaul.FronthaulLinks)

    np.testing.assert_array_equal(
        links,
        np.array([[True, False], [False, True], [False, True]]),
    )


def test_colocated_ap_and_cpu_tasks_produce_one_to_one_fronthaul_links():
    workflow = Workflow(
        [
            area.CfgAreaLength(100.0),
            area.CfgWrapAround(False),
            ap.CfgExplicitNumAPs(4),
            cpu.CfgSameAsNumAPs(),
            ap.CfgEvenlySpacedAPPositions(),
            cpu.CfgSameAsAPPositions(),
            geometry.ComputeAPtoCPUDifferences(),
            geometry.ComputeAPtoCPU2DDistances(),
            fronthaul.CfgClosestFronthaulLinks(),
        ]
    )

    links = workflow.run(fronthaul.FronthaulLinks)

    np.testing.assert_array_equal(links, np.eye(4, dtype=bool))
