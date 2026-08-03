import numpy as np
import pytest
from mimodium.algorithms import access


def test_master_ap_selects_largest_large_scale_fading_per_ue():
    beta = np.array([[1.0, 3.0], [4.0, 2.0]])

    np.testing.assert_array_equal(access.ComputeMasterAP()(beta), np.array([1, 0]))


def test_master_cpu_follows_fronthaul_link_of_master_ap():
    fronthaul = np.array([[True, False], [False, True]])

    master_cpu = access.ComputeMasterCPU()(
        K=2, master_ap=np.array([1, 0]), fronthaul=fronthaul
    )

    np.testing.assert_array_equal(master_cpu, np.array([1, 0]))


def test_measure_from_master_only_marks_aps_at_master_cpu_above_threshold():
    beta = np.array([[1.0, 0.01], [0.01, 1.0]])
    fronthaul = np.array([[True, False], [False, True]])
    master_cpu = np.array([0, 1])

    measured = access.CfgMeasureFromMaster(min_beta_db=-10.0)(
        K=2,
        L=2,
        beta=beta,
        fronthaul=fronthaul,
        master_cpu=master_cpu,
    )

    np.testing.assert_array_equal(measured, np.array([[True, False], [False, True]]))


def test_measure_from_neighbors_includes_neighbor_cpu_aps_above_threshold():
    beta = np.array([[1.0, 0.5]])
    fronthaul = np.array([[True, False], [False, True]])
    midhaul = np.array([[False, True], [True, False]])

    measured = access.CfgMeasureFromNeighbors(min_beta_db=-10.0)(
        K=1,
        L=2,
        beta=beta,
        fronthaul=fronthaul,
        midhaul=midhaul,
        master_cpu=np.array([0]),
    )

    np.testing.assert_array_equal(measured, np.array([[True, True]]))


def test_measure_all_respects_minimum_large_scale_fading_threshold():
    beta = np.array([[1.0, 0.1, 0.001]])

    np.testing.assert_array_equal(
        access.CfgMeasureAll(min_beta_db=-10.0)(K=1, L=3, beta=beta),
        np.array([[True, True, False]]),
    )


def test_candidate_tasks_intersect_measurement_topology_and_threshold():
    beta = np.array([[1.0, 0.01]])
    measured = np.array([[True, True]])
    fronthaul = np.array([[True, False], [False, True]])

    direct = access.CfgCandidateMeasured(min_beta_db=-10.0)(
        K=1, L=2, measured=measured, beta=beta
    )
    master = access.CfgCandidateFromMaster(min_beta_db=-10.0)(
        K=1,
        L=2,
        beta=beta,
        fronthaul=fronthaul,
        master_cpu=np.array([0]),
        measured=measured,
    )

    np.testing.assert_array_equal(direct, np.array([[True, False]]))
    np.testing.assert_array_equal(master, np.array([[True, False]]))


def test_candidate_from_neighbors_includes_neighbor_cpu_aps():
    beta = np.array([[1.0, 0.5]])
    measured = np.array([[True, True]])
    fronthaul = np.array([[True, False], [False, True]])
    midhaul = np.array([[False, True], [True, False]])

    candidate = access.CfgCandidateFromNeighbors(min_beta_db=-10.0)(
        K=1,
        L=2,
        beta=beta,
        fronthaul=fronthaul,
        midhaul=midhaul,
        master_cpu=np.array([0]),
        measured=measured,
    )

    np.testing.assert_array_equal(candidate, np.array([[True, True]]))


def test_candidate_policies_accept_empty_rows_and_return_subsets_of_measured():
    beta = np.array([[1.0, 0.5, 0.2], [0.001, 0.001, 0.001]])
    fronthaul = np.array([[True, False], [False, True], [False, True]], dtype=bool)
    midhaul = np.array([[False, True], [True, False]], dtype=bool)
    master_cpu = np.array([0, 1])
    measured = access.CfgMeasureAll(min_beta_db=-10.0)(K=2, L=3, beta=beta)

    pilot_ids = access.CfgBestPilotToMasterAP(seed_override=1)(
        K=2,
        tau_p=2,
        beta=beta,
        measured=measured,
        master_ap=np.array([0, 0]),
        seed=0,
    )
    candidates = (
        access.CfgCandidateMeasured()(K=2, L=3, measured=measured, beta=beta),
        access.CfgCandidateFromMaster()(
            K=2,
            L=3,
            beta=beta,
            fronthaul=fronthaul,
            master_cpu=master_cpu,
            measured=measured,
        ),
        access.CfgCandidateFromNeighbors()(
            K=2,
            L=3,
            beta=beta,
            fronthaul=fronthaul,
            midhaul=midhaul,
            master_cpu=master_cpu,
            measured=measured,
        ),
    )

    assert not np.any(measured[1])
    assert np.all((0 <= pilot_ids) & (pilot_ids < 2))
    for candidate in candidates:
        assert not np.any(candidate & ~measured)
        assert not np.any(candidate[1])


def test_initial_access_set_helpers_return_sorted_indices():
    master_ap = np.array([0, 1, 0])
    master_cpu = np.array([1, 1, 0])
    measured = np.array([[False, True], [True, True], [True, True]])
    candidate = np.array([[False, True], [True, False], [True, True]])

    np.testing.assert_array_equal(
        access.get_set_of_ues_with_master_ap_l(0, master_ap), [0, 2]
    )
    np.testing.assert_array_equal(
        access.get_set_of_ues_with_master_cpu_j(1, master_cpu), [0, 1]
    )
    np.testing.assert_array_equal(
        access.get_set_of_ues_measured_by_ap_l(1, measured), [0, 1, 2]
    )
    np.testing.assert_array_equal(
        access.get_set_of_aps_measuring_ue_k(1, measured), [0, 1]
    )
    np.testing.assert_array_equal(
        access.get_set_of_ues_candidate_to_ap_l(0, candidate), [1, 2]
    )
    np.testing.assert_array_equal(
        access.get_set_of_aps_with_ue_k_as_candidate(2, candidate), [0, 1]
    )


def test_best_pilot_to_master_ap_seed_override_is_reproducible():
    task = access.CfgBestPilotToMasterAP(seed_override=3)
    K = 4
    tau_p = 2
    beta = np.ones((4, 1))
    measured = np.ones((4, 1), dtype=bool)
    master_ap = np.zeros(4, dtype=int)

    first = task(
        K=K, tau_p=tau_p, beta=beta, measured=measured, master_ap=master_ap, seed=1
    )
    second = task(
        K=K, tau_p=tau_p, beta=beta, measured=measured, master_ap=master_ap, seed=2
    )

    np.testing.assert_array_equal(first, second)
    assert np.all((0 <= first) & (first < 2))


def test_random_pilots_seed_override_is_reproducible_and_in_range():
    task = access.CfgRandomPilots(seed_override=11)

    first = task(tau_p=3, K=5, seed=1)
    second = task(tau_p=3, K=5, seed=2)

    np.testing.assert_array_equal(first, second)
    assert np.all((0 <= first) & (first < 3))


def test_serve_all_candidates_returns_candidate_links():
    candidates = np.array([[True, False], [False, True]])

    used = access.CfgServeAllCandidates()(candidates, master_ap=np.array([0, 1]))

    np.testing.assert_array_equal(used, candidates)
    assert used.dtype == bool


def test_ap_greedy_serves_best_candidate_per_ap_and_pilot():
    used = access.CfgServeAPGreedy()(
        K=3,
        L=1,
        tau_p=1,
        pilot_ids=np.zeros(3, dtype=int),
        beta=np.array([[1.0], [3.0], [2.0]]),
        candidate=np.ones((3, 1), dtype=bool),
        master_ap=np.zeros(3, dtype=int),
    )

    np.testing.assert_array_equal(used, np.array([[False], [True], [False]]))


def test_ue_greedy_adds_strongest_candidate_aps_until_threshold():
    used = access.CfgServeUEGreedy(rel_threshold=0.75, max_aps=2)(
        K=1,
        L=3,
        beta=np.array([[4.0, 2.0, 1.0]]),
        candidate=np.ones((1, 3), dtype=bool),
        master_ap=np.array([0]),
    )

    np.testing.assert_array_equal(used, np.array([[True, True, False]]))


@pytest.mark.parametrize(
    "task_type", [access.CfgServeUEGreedy, access.CfgServeCombinedGreedy]
)
@pytest.mark.parametrize("rel_threshold", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_greedy_serving_rejects_invalid_relative_thresholds(task_type, rel_threshold):
    with pytest.raises(ValueError, match="rel_threshold"):
        task_type(rel_threshold=rel_threshold, max_aps=1)


@pytest.mark.parametrize(
    "task_type", [access.CfgServeUEGreedy, access.CfgServeCombinedGreedy]
)
@pytest.mark.parametrize("max_aps", [0, -1])
def test_greedy_serving_rejects_nonpositive_max_aps(task_type, max_aps):
    with pytest.raises(ValueError, match="max_aps"):
        task_type(rel_threshold=1.0, max_aps=max_aps)


def test_ue_greedy_respects_one_ap_limit_at_threshold_boundary():
    used = access.CfgServeUEGreedy(rel_threshold=1.0, max_aps=1)(
        K=1,
        L=3,
        beta=np.array([[1.0, 3.0, 2.0]]),
        candidate=np.ones((1, 3), dtype=bool),
        master_ap=np.array([1]),
    )

    np.testing.assert_array_equal(used, np.array([[False, True, False]]))


def test_combined_greedy_applies_ue_candidate_reduction_then_ap_greedy():
    used = access.CfgServeCombinedGreedy(rel_threshold=1.0, max_aps=2)(
        K=2,
        L=2,
        tau_p=1,
        pilot_ids=np.zeros(2, dtype=int),
        beta=np.array([[4.0, 1.0], [3.0, 2.0]]),
        candidate=np.ones((2, 2), dtype=bool),
        master_ap=np.array([0, 1]),
    )

    np.testing.assert_array_equal(used, np.array([[True, False], [False, True]]))


def test_serving_policies_return_candidate_subsets_and_require_the_master_link():
    candidates = np.array(
        [[False, True, False], [True, True, False], [False, False, False]]
    )
    master_ap = np.array([0, 0, 2])
    beta = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    pilot_ids = np.array([0, 1, 0])
    used_sets = (
        access.CfgServeAllCandidates()(candidates, master_ap),
        access.CfgServeAPGreedy()(
            K=3,
            L=3,
            tau_p=2,
            pilot_ids=pilot_ids,
            beta=beta,
            candidate=candidates,
            master_ap=master_ap,
        ),
        access.CfgServeUEGreedy(rel_threshold=1.0, max_aps=3)(
            K=3,
            L=3,
            beta=beta,
            candidate=candidates,
            master_ap=master_ap,
        ),
        access.CfgServeCombinedGreedy(rel_threshold=1.0, max_aps=3)(
            K=3,
            L=3,
            tau_p=2,
            pilot_ids=pilot_ids,
            beta=beta,
            candidate=candidates,
            master_ap=master_ap,
        ),
    )

    for used in used_sets:
        assert not np.any(used & ~candidates)
        assert not np.any(used[0])
        assert not np.any(used[2])
        assert used[1, master_ap[1]]


def test_estimation_link_tasks_copy_their_reference_link_sets():
    measured = np.array([[True, False], [True, True]])
    candidate = np.array([[True, False], [True, True]])
    used = np.array([[True, False], [False, True]])
    beta = np.array([[1.0, 1.0], [0.01, 1.0]])

    np.testing.assert_array_equal(access.CfgEstimateMeasured()(measured), measured)
    np.testing.assert_array_equal(access.CfgEstimateCandidates()(candidate), candidate)
    np.testing.assert_array_equal(access.CfgEstimateServed()(used), used)
    np.testing.assert_array_equal(
        access.CfgEstimateMeasuredWithThreshold(min_beta_db=-10.0)(
            beta, measured, used
        ),
        np.array([[True, False], [False, True]]),
    )


def test_estimation_policies_contain_used_and_remain_within_measured():
    measured = np.array([[True, True, True], [False, False, False]])
    candidate = np.array([[True, True, False], [False, False, False]])
    used = np.array([[True, False, False], [False, False, False]])
    beta = np.array([[0.001, 1.0, 1.0], [1.0, 1.0, 1.0]])
    estimated_from_measured = access.CfgEstimateMeasured()(measured)
    estimated_from_served = access.CfgEstimateServed()(used)
    estimated_sets = (
        estimated_from_measured,
        access.CfgEstimateMeasuredWithThreshold(min_beta_db=-10.0)(
            beta, measured, used
        ),
        access.CfgEstimateCandidates()(candidate),
        estimated_from_served,
    )

    for estimated in estimated_sets:
        assert not np.any(used & ~estimated)
        assert not np.any(estimated & ~measured)
        assert not np.any(estimated[1])

    assert np.any(candidate & ~estimated_from_served)
    assert np.any(estimated_from_measured & ~candidate)


def test_clustering_set_helpers_return_serving_relationships():
    used = np.array([[True, False], [True, True], [False, True]])
    fronthaul = np.array([[True, False], [False, True]])

    np.testing.assert_array_equal(access.get_set_of_ues_served_by_ap_l(0, used), [0, 1])
    np.testing.assert_array_equal(access.get_set_of_aps_serving_ue_k(1, used), [0, 1])
    np.testing.assert_array_equal(
        access.get_set_of_aps_connected_to_cpu_j_serving_ue_k(1, 1, used, fronthaul),
        [1],
    )
    np.testing.assert_array_equal(
        access.get_set_of_ues_served_by_cpu_j(0, used, fronthaul), [0, 1]
    )
    np.testing.assert_array_equal(
        access.get_set_of_cpus_serving_ue_k(1, used, fronthaul), [0, 1]
    )
