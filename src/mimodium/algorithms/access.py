"""Configure UE access, pilot assignment, clustering, and channel estimation links.

Access proceeds through four UE--AP link sets. For a measured link, the AP stores
large-scale channel statistics. Candidate links are measured links that may be selected
for service. Used links are the candidates selected during clustering. Estimated links
are measured links for which the AP also estimates the small-scale channel. Every used
link must be both a candidate and estimated, giving
``measured ⊇ candidate ⊇ used`` and ``measured ⊇ estimated ⊇ used``.

Candidate and estimated links do not contain one another. A candidate that is not
selected for service may not need a channel estimate, while estimating a non-candidate
link can provide interference information. Estimation can therefore be selected after
clustering to avoid unnecessary channel estimates.

Access begins by assigning each UE a master AP and CPU. The measured sets are formed
next, followed by pilot assignment and formation of the candidate sets. Finally, the
used and estimated links are selected.
"""

import numpy as np
from dagreon import task

from ..propagation import LargeScaleFadingCoefficients, NumPilots
from ..rng import Seed, get_rng
from ..scenario import FronthaulLinks, MidhaulLinks, NumAPs, NumUEs
from ..scenario.fronthaul import (
    get_cpu_connected_to_ap_l,
    get_set_of_aps_connected_to_cpu_j,
)
from ..scenario.midhaul import get_set_of_cpus_neighboring_cpu_j

#: Master AP index for each UE.
#:
#: :shape: ``(K,)``
#: :dtype: ``int``
type MasterAP = np.ndarray
#: Master CPU index for each UE.
#:
#: :shape: ``(K,)``
#: :dtype: ``int``
type MasterCPU = np.ndarray
#: Whether an AP measures or stores large-scale statistics for a UE--AP link.
#:
#: :shape: ``(K, L)``
#: :dtype: ``bool``
type MeasuredStatisticLinks = np.ndarray
#: Whether a UE--AP link is considered as a serving candidate.
#:
#: :shape: ``(K, L)``
#: :dtype: ``bool``
type CandidateWirelessLinks = np.ndarray
#: Pilot index assigned to each UE.
#:
#: :shape: ``(K,)``
#: :dtype: ``int``
type AssignedPilotIDs = np.ndarray
#: Whether a UE--AP link is used for serving after clustering.
#:
#: :shape: ``(K, L)``
#: :dtype: ``bool``
type UsedWirelessLinks = np.ndarray
#: Whether an AP estimates the small-scale channel of a UE--AP link.
#:
#: :shape: ``(K, L)``
#: :dtype: ``bool``
type EstimatedChannelLinks = np.ndarray


@task
class ComputeMasterAP:
    """Assign each UE to the AP with the largest large-scale fading coefficient."""

    def __call__(self, beta: LargeScaleFadingCoefficients) -> MasterAP:
        return np.argmax(beta, axis=1)


@task
class ComputeMasterCPU:
    """Assign each UE to the CPU connected to its :type:`MasterAP`."""

    def __call__(
        self, K: NumUEs, master_ap: MasterAP, fronthaul: FronthaulLinks
    ) -> MasterCPU:
        master_cpu = np.zeros_like(master_ap)
        for k in range(K):
            master_cpu[k] = get_cpu_connected_to_ap_l(master_ap[k], fronthaul)
        return master_cpu


@task
class CfgMeasureAll:
    """Measure every UE--AP link whose large-scale fading is at least
    :code:`min_beta_db` decibels."""

    min_beta_db: float = -np.inf

    def __call__(
        self, K: NumUEs, L: NumAPs, beta: LargeScaleFadingCoefficients
    ) -> MeasuredStatisticLinks:
        measured = np.zeros((K, L), dtype=bool)
        for k in range(K):
            for l in range(L):
                if 10 * np.log10(beta[k, l]) >= self.min_beta_db:
                    measured[k, l] = True
        return measured


@task
class CfgMeasureFromNeighbors:
    """Measure UE--AP statistics at APs attached to the master CPU or its neighbors.

    For every UE, include APs connected to its :type:`MasterCPU` and to CPUs adjacent
    over :type:`MidhaulLinks`. Links with large-scale fading below
    :code:`min_beta_db` decibels are excluded.
    """

    min_beta_db: float = -np.inf

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        beta: LargeScaleFadingCoefficients,
        fronthaul: FronthaulLinks,
        midhaul: MidhaulLinks,
        master_cpu: MasterCPU,
    ) -> MeasuredStatisticLinks:
        measured = np.zeros((K, L), dtype=bool)
        for k in range(K):
            neighbor_js = get_set_of_cpus_neighboring_cpu_j(master_cpu[k], midhaul)
            for j in np.concatenate((neighbor_js, [master_cpu[k]])):
                L_j = get_set_of_aps_connected_to_cpu_j(j, fronthaul)
                for l in L_j:
                    if 10 * np.log10(beta[k, l]) >= self.min_beta_db:
                        measured[k, l] = True
        return measured


@task
class CfgMeasureFromMaster:
    """Measure UE--AP statistics only at APs attached to each UE's master CPU.

    Links with large-scale fading below :code:`min_beta_db` decibels are excluded.
    """

    min_beta_db: float = -np.inf

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        beta: LargeScaleFadingCoefficients,
        fronthaul: FronthaulLinks,
        master_cpu: MasterCPU,
    ) -> MeasuredStatisticLinks:
        measured = np.zeros((K, L), dtype=bool)
        for k in range(K):
            for l in get_set_of_aps_connected_to_cpu_j(master_cpu[k], fronthaul):
                if 10 * np.log10(beta[k, l]) >= self.min_beta_db:
                    measured[k, l] = True
        return measured


@task
class CfgCandidateMeasured:
    """Use measured links as serving candidates after an optional gain threshold.

    A link is a candidate only if it is measured and its large-scale fading is at least
    :code:`min_beta_db` decibels.
    """

    min_beta_db: float = -np.inf

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        measured: MeasuredStatisticLinks,
        beta: LargeScaleFadingCoefficients,
    ) -> CandidateWirelessLinks:
        candidate = np.zeros((K, L), dtype=bool)
        for k in range(K):
            for l in range(L):
                if measured[k, l] and 10 * np.log10(beta[k, l]) >= self.min_beta_db:
                    candidate[k, l] = True
        return candidate


@task
class CfgCandidateFromNeighbors:
    """Use measured links from the master CPU neighborhood as serving candidates.

    For every UE, candidate APs must be connected to the master CPU or a neighboring
    CPU, already have measured statistics, and satisfy the :code:`min_beta_db`
    large-scale-fading threshold.
    """

    min_beta_db: float = -np.inf

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        beta: LargeScaleFadingCoefficients,
        fronthaul: FronthaulLinks,
        midhaul: MidhaulLinks,
        master_cpu: MasterCPU,
        measured: MeasuredStatisticLinks,
    ) -> CandidateWirelessLinks:
        candidate = np.zeros((K, L), dtype=bool)
        for k in range(K):
            neighbor_js = get_set_of_cpus_neighboring_cpu_j(master_cpu[k], midhaul)
            for j in np.concatenate((neighbor_js, [master_cpu[k]])):
                L_j = get_set_of_aps_connected_to_cpu_j(j, fronthaul)
                for l in L_j:
                    if measured[k, l] and 10 * np.log10(beta[k, l]) >= self.min_beta_db:
                        candidate[k, l] = True
        return candidate


@task
class CfgCandidateFromMaster:
    """Use measured links from each UE's master CPU as serving candidates.

    Candidate APs must be attached to the UE's :type:`MasterCPU`, already have measured
    statistics, and satisfy the :code:`min_beta_db` large-scale-fading threshold.
    """

    min_beta_db: float = -np.inf

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        beta: LargeScaleFadingCoefficients,
        fronthaul: FronthaulLinks,
        master_cpu: MasterCPU,
        measured: MeasuredStatisticLinks,
    ) -> CandidateWirelessLinks:
        candidate = np.zeros((K, L), dtype=bool)
        for k in range(K):
            for l in get_set_of_aps_connected_to_cpu_j(master_cpu[k], fronthaul):
                if measured[k, l] and 10 * np.log10(beta[k, l]) >= self.min_beta_db:
                    candidate[k, l] = True
        return candidate


@task
class CfgBestPilotToMasterAP:
    """Assign pilots sequentially by minimizing contamination at the master AP.

    For each UE, choose the pilot whose already-assigned measured UEs have the smallest
    large-scale-fading sum at the UE's :type:`MasterAP`. If several pilots tie, one is
    chosen uniformly at random. The :type:`Seed` for the random number generator can be
    overridden with :code:`seed_override`.
    """

    seed_override: int | None = None

    # algorithm 4.1 from CF mMIMO book, page 288, adjusted so that each AP only
    # considers the UEs that it measures the statistics of
    # it is not really "joint", lines 2-9 do the pilot assignment, lines 10-15 do the
    # clustering
    def __call__(
        self,
        K: NumUEs,
        tau_p: NumPilots,
        beta: LargeScaleFadingCoefficients,
        measured: MeasuredStatisticLinks,
        master_ap: MasterAP,
        seed: Seed,
    ) -> AssignedPilotIDs:
        pilot_ids = np.full(K, fill_value=-1, dtype=int)
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        for k in range(K):
            beta_sums = np.zeros(tau_p, dtype=float)
            for t in range(tau_p):
                # out of the UEs that use this pilot index
                ues_with_pilot_t = np.argwhere(pilot_ids == t)
                # and are known to this AP (measured stats)
                known_ues_with_pilot_t = np.intersect1d(
                    ues_with_pilot_t, np.argwhere(measured[:, master_ap[k]])
                )
                beta_sums[t] = np.sum(beta[known_ues_with_pilot_t, master_ap[k]])
            # if there are multiple pilots with the same sum, choose one at random
            min_sum = np.min(beta_sums)
            pilot_ids[k] = rng.choice(np.argwhere(beta_sums == min_sum).flatten())
        return pilot_ids


@task
class CfgRandomPilots:
    """Draw each UE's pilot index uniformly from the available pilot symbols.

    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`.
    """

    seed_override: int | None = None

    def __call__(self, tau_p: NumPilots, K: NumUEs, seed: Seed) -> AssignedPilotIDs:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        return rng.integers(0, tau_p, K)


@task
class CfgServeAllCandidates:
    """Serve every candidate link for UEs whose master AP is a candidate."""

    def __call__(
        self, candidate: CandidateWirelessLinks, master_ap: MasterAP
    ) -> UsedWirelessLinks:
        return _clear_ues_without_master_link(candidate.copy(), master_ap)


@task
class CfgServeAPGreedy:
    """Serve at most one candidate UE per pilot at each AP.

    For every AP and pilot index, select the candidate UE with the largest large-scale
    fading coefficient among UEs assigned to that pilot. This is the clustering part
    of Algorithm 4.1 from the cell-free massive MIMO book. A UE is left completely
    unserved if its master AP does not select it.
    """

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        tau_p: NumPilots,
        pilot_ids: AssignedPilotIDs,
        beta: LargeScaleFadingCoefficients,
        candidate: CandidateWirelessLinks,
        master_ap: MasterAP,
    ) -> UsedWirelessLinks:
        used = np.zeros((K, L), dtype=bool)
        # each AP
        for l in range(L):
            # UEs that could be served by this AP
            K_l_candidate = np.argwhere(candidate[:, l])
            # for each pilot index
            for t in range(tau_p):
                # UEs that use this pilot index
                K_pilot_t = np.argwhere(pilot_ids == t)
                # and are candidates
                K_pilot_t_candidate = np.intersect1d(K_pilot_t, K_l_candidate)

                # the UE with the best channel
                if len(K_pilot_t_candidate) > 0:
                    best_ue_for_pilot_t = K_pilot_t_candidate[
                        np.argmax(beta[K_pilot_t_candidate, l])
                    ]
                    used[best_ue_for_pilot_t, l] = True
        return _clear_ues_without_master_link(used, master_ap)


@task
class CfgServeUEGreedy:
    """Let each UE greedily select APs with the strongest candidate links.

    APs are considered in decreasing large-scale fading order. UE :math:`k` keeps
    adding candidate APs until their cumulative share of the candidate-link power
    reaches :code:`rel_threshold` in the range ``(0, 1]``, or until the positive
    integer :code:`max_aps` has been selected. This is sometimes called a
    largest-large-scale-fading (LLSF) rule. A UE is left completely unserved if its
    master AP is not selected.
    """

    rel_threshold: float
    max_aps: int

    def __post_init__(self):
        if not np.isfinite(self.rel_threshold) or not 0 < self.rel_threshold <= 1:
            raise ValueError("rel_threshold must be finite and in the range (0, 1]")
        if self.max_aps <= 0:
            raise ValueError("max_aps must be positive")

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        beta: LargeScaleFadingCoefficients,
        candidate: CandidateWirelessLinks,
        master_ap: MasterAP,
    ) -> UsedWirelessLinks:
        used = np.zeros((K, L), dtype=bool)
        # each UE selects
        for k in range(K):
            num_aps = 0
            total_power = np.sum(beta[k, candidate[k]])
            rel_power = 0.0
            # loop through the APs in decreasing order of channel power
            for l in np.flip(np.argsort(beta[k])):
                if candidate[k, l]:
                    used[k, l] = True
                    num_aps += 1
                    rel_power += beta[k, l] / total_power
                    # as long as the relative power and max_aps are not reached
                    if rel_power >= self.rel_threshold or num_aps >= self.max_aps:
                        break
        return _clear_ues_without_master_link(used, master_ap)


@task
class CfgServeCombinedGreedy:
    """Combine UE-greedy candidate pruning with AP-greedy pilot separation.

    First, each UE keeps its strongest candidate APs according to
    :code:`rel_threshold` in the range ``(0, 1]`` and the positive integer
    :code:`max_aps`. Then each AP serves at most one of the remaining candidate UEs per
    pilot, choosing the strongest large-scale fading link. A UE is left completely
    unserved if its master AP does not select it.
    """

    rel_threshold: float
    max_aps: int

    def __post_init__(self):
        if not np.isfinite(self.rel_threshold) or not 0 < self.rel_threshold <= 1:
            raise ValueError("rel_threshold must be finite and in the range (0, 1]")
        if self.max_aps <= 0:
            raise ValueError("max_aps must be positive")

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        tau_p: NumPilots,
        pilot_ids: AssignedPilotIDs,
        beta: LargeScaleFadingCoefficients,
        candidate: CandidateWirelessLinks,
        master_ap: MasterAP,
    ) -> UsedWirelessLinks:
        ue_greedy_candidate = np.zeros((K, L), dtype=bool)
        # first do UE greedy
        for k in range(K):
            num_aps = 0
            total_power = np.sum(beta[k, candidate[k]])
            rel_power = 0.0
            for l in np.flip(np.argsort(beta[k])):
                if candidate[k, l]:
                    ue_greedy_candidate[k, l] = True
                    num_aps += 1
                    rel_power += beta[k, l] / total_power
                    if rel_power >= self.rel_threshold or num_aps >= self.max_aps:
                        break
        used = np.zeros((K, L), dtype=bool)
        # then do AP greedy for the remaining candidates
        for l in range(L):
            K_l_candidate = np.argwhere(ue_greedy_candidate[:, l])
            for t in range(tau_p):
                K_pilot_t = np.argwhere(pilot_ids == t)
                K_pilot_t_candidate = np.intersect1d(K_pilot_t, K_l_candidate)
                if len(K_pilot_t_candidate) > 0:
                    best_ue_for_pilot_t = K_pilot_t_candidate[
                        np.argmax(beta[K_pilot_t_candidate, l])
                    ]
                    used[best_ue_for_pilot_t, l] = True
        return _clear_ues_without_master_link(used, master_ap)


@task
class CfgEstimateMeasured:
    """Estimate the channel on every link with measured large-scale statistics."""

    def __call__(self, measured: MeasuredStatisticLinks) -> EstimatedChannelLinks:
        return np.copy(measured)


@task
class CfgEstimateMeasuredWithThreshold:
    """Estimate measured links whose large-scale fading exceeds a threshold.

    Measured links are retained if their large-scale fading is at least
    :code:`min_beta_db` decibels. Used links are always retained.
    """

    min_beta_db: float = -np.inf

    def __call__(
        self,
        beta: LargeScaleFadingCoefficients,
        measured: MeasuredStatisticLinks,
        used: UsedWirelessLinks,
    ) -> EstimatedChannelLinks:
        estimated = np.copy(measured)
        estimated[measured] = 10 * np.log10(beta[measured]) >= self.min_beta_db
        estimated[used] = True
        return estimated


@task
class CfgEstimateCandidates:
    """Estimate the channel on every candidate UE--AP link."""

    def __call__(self, candidate: CandidateWirelessLinks) -> EstimatedChannelLinks:
        return np.copy(candidate)


@task
class CfgEstimateServed:
    """Estimate the channel on every served UE--AP link."""

    def __call__(self, used: UsedWirelessLinks) -> EstimatedChannelLinks:
        return np.copy(used)


def get_set_of_ues_with_master_ap_l(l: int, master_ap: MasterAP) -> np.ndarray:
    """Return the UE indices whose master AP is AP :math:`l`."""

    return np.argwhere(master_ap == l).flatten()


def get_set_of_ues_with_master_cpu_j(j: int, master_cpu: MasterCPU) -> np.ndarray:
    """Return the UE indices whose master CPU is CPU :math:`j`."""

    return np.argwhere(master_cpu == j).flatten()


def get_set_of_ues_measured_by_ap_l(
    l: int, measured: MeasuredStatisticLinks
) -> np.ndarray:
    """Return the UE indices measured by AP :math:`l`."""

    return np.argwhere(measured[:, l]).flatten()


def get_set_of_aps_measuring_ue_k(
    k: int, measured: MeasuredStatisticLinks
) -> np.ndarray:
    """Return the AP indices that measure UE :math:`k`."""

    return np.argwhere(measured[k, :]).flatten()


def get_set_of_ues_candidate_to_ap_l(
    l: int, candidate: CandidateWirelessLinks
) -> np.ndarray:
    """Return the UE indices for which AP :math:`l` is a serving candidate."""

    return np.argwhere(candidate[:, l]).flatten()


def get_set_of_aps_with_ue_k_as_candidate(
    k: int, candidate: CandidateWirelessLinks
) -> np.ndarray:
    """Return the candidate AP indices for UE :math:`k`."""

    return np.argwhere(candidate[k, :]).flatten()


def _clear_ues_without_master_link(
    used: UsedWirelessLinks, master_ap: MasterAP
) -> UsedWirelessLinks:
    served_by_master = used[np.arange(len(master_ap)), master_ap]
    used[~served_by_master] = False
    return used


def get_set_of_ues_served_by_ap_l(l: int, used: UsedWirelessLinks) -> np.ndarray:
    """Return the UE indices served by AP :math:`l`."""

    return np.argwhere(used[:, l]).flatten()


def get_set_of_aps_serving_ue_k(k: int, used: UsedWirelessLinks) -> np.ndarray:
    """Return the AP indices serving UE :math:`k`."""

    return np.argwhere(used[k, :]).flatten()


def get_set_of_aps_connected_to_cpu_j_serving_ue_k(
    j: int, k: int, used: UsedWirelessLinks, fronthaul: FronthaulLinks
) -> np.ndarray:
    """Return APs that are connected to CPU :math:`j` and serve UE :math:`k`."""

    L_j_set = get_set_of_aps_connected_to_cpu_j(j, fronthaul)
    L_k_set = get_set_of_aps_serving_ue_k(k, used)
    return np.intersect1d(L_j_set, L_k_set)


def get_set_of_ues_served_by_cpu_j(
    j: int, used: UsedWirelessLinks, fronthaul: FronthaulLinks
) -> np.ndarray:
    """Return the UE indices served by at least one AP connected to CPU :math:`j`."""

    L_j_set = get_set_of_aps_connected_to_cpu_j(j, fronthaul)
    # index is 2D (but we are only interested in the first dimension)
    return np.unique(np.argwhere(used[:, L_j_set])[:, 0])


def get_set_of_cpus_serving_ue_k(
    k: int, used: UsedWirelessLinks, fronthaul: FronthaulLinks
) -> np.ndarray:
    """Return CPUs connected to at least one AP that serves UE :math:`k`."""

    L_k_set = get_set_of_aps_serving_ue_k(k, used)
    # index is 2D (but we are only interested in the second dimension)
    return np.unique(np.argwhere(fronthaul[L_k_set, :])[:, 1])
