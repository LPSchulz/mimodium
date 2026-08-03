import numpy as np
from dagreon import Workflow
from mimodium.scenario import area, cpu, geometry, midhaul


def test_full_midhaul_links_connect_all_distinct_cpus():
    links = midhaul.CfgFullMidhaulLinks()(J=3)

    np.testing.assert_array_equal(
        links,
        np.array(
            [
                [False, True, True],
                [True, False, True],
                [True, True, False],
            ]
        ),
    )


def test_fixed_radius_midhaul_links_exclude_self_and_distant_cpus():
    distances = np.array([[0.0, 5.0, 11.0], [5.0, 0.0, 8.0], [11.0, 8.0, 0.0]])

    links = midhaul.CfgFixedRadiusMidhaulLinks(radius=10.0)(J=3, distances_2d=distances)

    np.testing.assert_array_equal(
        links,
        np.array(
            [
                [False, True, False],
                [True, False, True],
                [False, True, False],
            ]
        ),
    )
    np.testing.assert_array_equal(
        midhaul.get_set_of_cpus_neighboring_cpu_j(1, links), [0, 2]
    )


def test_delaunay_midhaul_links_fully_connect_less_than_four_cpus():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    links = midhaul.CfgDelaunayMidhaulLinks()(
        J=3, cpu_pos=positions, wrap_around=False, sqrt_A=1.0
    )

    np.testing.assert_array_equal(links, midhaul.CfgFullMidhaulLinks()(J=3))


def test_wrapped_delaunay_midhaul_links_do_not_connect_cpus_to_themselves():
    positions = np.array([[1.0, 1.0], [9.0, 1.0], [1.0, 9.0], [7.0, 7.0]])

    links = midhaul.CfgDelaunayMidhaulLinks()(
        J=4,
        cpu_pos=positions,
        wrap_around=True,
        sqrt_A=10.0,
    )

    np.testing.assert_array_equal(np.diag(links), np.zeros(4, dtype=bool))


def test_no_midhaul_links_are_all_false():
    np.testing.assert_array_equal(
        midhaul.CfgNoMidhaulLinks()(J=2), np.zeros((2, 2), dtype=bool)
    )


def test_midhaul_configurations_produce_expected_links_for_cpu_scenario():
    cpu_positions = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [5.0, 5.0]])
    workflow = Workflow(
        [
            area.CfgAreaLength(10.0),
            area.CfgWrapAround(False),
            cpu.CfgExplicitNumCPUs(4),
            cpu.CfgExplicitCPUPositions(cpu_positions),
            geometry.ComputeCPUtoCPUDifferences(),
            geometry.ComputeCPUtoCPU2DDistances(),
        ]
    )

    full_links = workflow.run(
        midhaul.MidhaulLinks,
        overrides=[midhaul.CfgFullMidhaulLinks()],
    )
    fixed_radius_links = workflow.run(
        midhaul.MidhaulLinks,
        overrides=[midhaul.CfgFixedRadiusMidhaulLinks(radius=4.5)],
    )
    delaunay_links = workflow.run(
        midhaul.MidhaulLinks,
        overrides=[midhaul.CfgDelaunayMidhaulLinks()],
    )
    no_links = workflow.run(
        midhaul.MidhaulLinks,
        overrides=[midhaul.CfgNoMidhaulLinks()],
    )

    np.testing.assert_array_equal(
        full_links,
        np.array(
            [
                [False, True, True, True],
                [True, False, True, True],
                [True, True, False, True],
                [True, True, True, False],
            ]
        ),
    )
    np.testing.assert_array_equal(
        fixed_radius_links,
        np.array(
            [
                [False, True, True, False],
                [True, False, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ]
        ),
    )
    np.testing.assert_array_equal(
        delaunay_links,
        np.array(
            [
                [False, True, True, False],
                [True, False, True, True],
                [True, True, False, True],
                [False, True, True, False],
            ]
        ),
    )
    np.testing.assert_array_equal(no_links, np.zeros((4, 4), dtype=bool))
