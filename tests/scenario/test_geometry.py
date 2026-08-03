import numpy as np
from mimodium.scenario import geometry


def test_compute_wrapped_diff_mutates_large_offsets_to_shorter_wraparound():
    diff = np.array([[[60.0, -70.0, 2.0]]])

    geometry.compute_wrapped_diff(diff, g=100.0)

    np.testing.assert_allclose(diff, np.array([[[-40.0, 30.0, 2.0]]]))


def test_compute_ue_to_ap_differences_applies_wraparound_per_axis():
    ap_locations = np.array([[90.0, 10.0, 5.0], [20.0, 30.0, 6.0]])
    ue_locations = np.array([[10.0, 90.0, 1.0]])

    diff = geometry.ComputeUEtoAPDifferences()(
        K=1,
        L=2,
        ap_loc=ap_locations,
        ue_loc=ue_locations,
        sqrt_A=100.0,
        wrap_around=True,
    )

    np.testing.assert_allclose(
        diff, np.array([[[-20.0, 20.0, 4.0], [10.0, 40.0, 5.0]]])
    )


def test_compute_ap_to_cpu_differences_applies_wraparound_per_axis():
    ap_positions = np.array([[90.0, 10.0], [20.0, 30.0]])
    cpu_positions = np.array([[10.0, 90.0], [70.0, 80.0]])

    diff = geometry.ComputeAPtoCPUDifferences()(
        L=2,
        J=2,
        ap_pos=ap_positions,
        cpu_pos=cpu_positions,
        sqrt_A=100.0,
        wrap_around=True,
    )

    np.testing.assert_allclose(
        diff,
        np.array(
            [
                [[20.0, -20.0], [-20.0, -30.0]],
                [[-10.0, -40.0], [50.0, 50.0]],
            ]
        ),
    )


def test_compute_cpu_to_cpu_differences_applies_wraparound_per_axis():
    cpu_positions = np.array([[10.0, 90.0], [70.0, 80.0]])

    diff = geometry.ComputeCPUtoCPUDifferences()(
        J=2,
        cpu_pos=cpu_positions,
        sqrt_A=100.0,
        wrap_around=True,
    )

    np.testing.assert_allclose(
        diff,
        np.array(
            [
                [[0.0, 0.0], [-40.0, -10.0]],
                [[40.0, 10.0], [0.0, 0.0]],
            ]
        ),
    )


def test_ap_to_cpu_and_cpu_to_cpu_differences_match_coordinate_subtraction():
    ap_positions = np.array([[0.0, 0.0], [10.0, 0.0]])
    cpu_positions = np.array([[3.0, 4.0], [13.0, 4.0]])

    ap_cpu = geometry.ComputeAPtoCPUDifferences()(
        L=2,
        J=2,
        ap_pos=ap_positions,
        cpu_pos=cpu_positions,
        sqrt_A=100.0,
        wrap_around=False,
    )
    cpu_cpu = geometry.ComputeCPUtoCPUDifferences()(
        J=2,
        cpu_pos=cpu_positions,
        sqrt_A=100.0,
        wrap_around=False,
    )

    np.testing.assert_allclose(ap_cpu[0, 0], [3.0, 4.0])
    np.testing.assert_allclose(ap_cpu[1, 0], [-7.0, 4.0])
    np.testing.assert_allclose(cpu_cpu[0, 1], [10.0, 0.0])
    np.testing.assert_allclose(cpu_cpu[1, 0], [-10.0, 0.0])


def test_distance_helpers_match_pythagorean_reference_cases():
    diff_2d = np.array([[[3.0, 4.0], [5.0, 12.0]]])
    diff_3d = np.array([[[3.0, 4.0, 12.0], [1.0, 2.0, 2.0]]])

    np.testing.assert_allclose(geometry._compute_2d_distances(diff_2d), [[5.0, 13.0]])
    np.testing.assert_allclose(geometry._compute_3d_distances(diff_3d), [[13.0, 3.0]])
    np.testing.assert_allclose(
        geometry.ComputeUEtoAP2DDistances()(diff_2d), [[5.0, 13.0]]
    )
    np.testing.assert_allclose(
        geometry.ComputeAPtoCPU2DDistances()(diff_2d), [[5.0, 13.0]]
    )
    np.testing.assert_allclose(
        geometry.ComputeCPUtoCPU2DDistances()(diff_2d), [[5.0, 13.0]]
    )
    np.testing.assert_allclose(
        geometry.ComputeUEtoAP3DDistances()(diff_3d), [[13.0, 3.0]]
    )


def test_compute_ue_to_ue_2d_distances_applies_wraparound_per_axis():
    ue_positions = np.array([[10.0, 90.0], [70.0, 80.0]])

    distances = geometry.ComputeUEtoUE2DDistances()(
        ue_pos=ue_positions,
        sqrt_A=100.0,
        wrap_around=True,
    )

    np.testing.assert_allclose(
        distances,
        np.array(
            [
                [0.0, np.sqrt(40.0**2 + 10.0**2)],
                [np.sqrt(40.0**2 + 10.0**2), 0.0],
            ]
        ),
    )


def test_compute_ue_to_ue_2d_distances_matches_coordinate_subtraction():
    ue_positions = np.array([[0.0, 0.0], [3.0, 4.0]])

    distances = geometry.ComputeUEtoUE2DDistances()(
        ue_pos=ue_positions,
        sqrt_A=100.0,
        wrap_around=False,
    )

    np.testing.assert_allclose(distances, np.array([[0.0, 5.0], [5.0, 0.0]]))


def test_angle_helpers_match_axis_aligned_reference_cases():
    diff = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]])

    np.testing.assert_allclose(
        geometry.compute_azimuth_angles(diff), [[0.0, np.pi / 2]]
    )
    np.testing.assert_allclose(
        geometry.ComputeUEtoAPAzimuthAngles()(diff), [[0.0, np.pi / 2]]
    )
    np.testing.assert_allclose(
        geometry.ComputeAPtoCPUAzimuthAngles()(diff), [[0.0, np.pi / 2]]
    )
    np.testing.assert_allclose(
        geometry.ComputeCPUtoCPUAzimuthAngles()(diff), [[0.0, np.pi / 2]]
    )
