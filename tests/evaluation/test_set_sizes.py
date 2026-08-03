import numpy as np
from mimodium.evaluation import set_sizes


def test_association_set_size_tasks_count_each_link_type_by_axis():
    measured = np.array([[True, True], [True, True], [False, True]])
    candidate = np.array([[True, False], [True, True], [False, False]])
    used = np.array([[True, False], [False, True], [False, False]])
    estimated = np.array([[True, False], [False, True], [False, True]])

    np.testing.assert_array_equal(
        set_sizes.ComputeNumMeasuredUEsPerAP()(measured), np.array([2, 3])
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumMeasuringAPsPerUE()(measured), np.array([2, 2, 1])
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumCandidateUEsPerAP()(candidate), np.array([2, 1])
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumCandidateAPsPerUE()(candidate), np.array([1, 2, 0])
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumServedUEsPerAP()(used), np.array([1, 1])
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumServingAPsPerUE()(used), np.array([1, 1, 0])
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumEstimatedUEsPerAP()(estimated), np.array([1, 2])
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumEstimatingAPsPerUE()(estimated), np.array([1, 1, 1])
    )


def test_num_aps_per_cpu_counts_fronthaul_links_by_cpu():
    fronthaul = np.array([[True, False], [False, True], [False, True]])

    np.testing.assert_array_equal(
        set_sizes.ComputeNumAPsPerCPU()(fronthaul), np.array([1, 2])
    )


def test_association_set_size_tasks_count_each_cpu_relation_once():
    measured = np.array(
        [[True, True, True], [True, False, False], [False, False, True]]
    )
    candidate = np.array(
        [[True, False, False], [False, True, True], [False, False, False]]
    )
    used = np.array([[True, True, False], [False, True, True], [False, False, False]])
    estimated = np.array(
        [[True, False, True], [False, True, False], [False, True, True]]
    )
    fronthaul = np.array([[True, False], [True, False], [False, True]])

    np.testing.assert_array_equal(
        set_sizes.ComputeNumMeasuredUEsPerCPU()(measured, fronthaul),
        np.array([2, 2]),
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumMeasuringCPUsPerUE()(measured, fronthaul),
        np.array([2, 1, 1]),
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumCandidateUEsPerCPU()(candidate, fronthaul),
        np.array([2, 1]),
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumCandidateCPUsPerUE()(candidate, fronthaul),
        np.array([1, 2, 0]),
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumServedUEsPerCPU()(used, fronthaul),
        np.array([2, 1]),
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumServingCPUsPerUE()(used, fronthaul),
        np.array([1, 2, 0]),
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumEstimatedUEsPerCPU()(estimated, fronthaul),
        np.array([3, 2]),
    )
    np.testing.assert_array_equal(
        set_sizes.ComputeNumEstimatingCPUsPerUE()(estimated, fronthaul),
        np.array([2, 1, 2]),
    )
