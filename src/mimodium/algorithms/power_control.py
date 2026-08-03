r"""Configure uplink and downlink transmit powers.

Notation
--------
Let :math:`\mathcal{J}_k` contain the CPUs serving UE :math:`k`, and let
:math:`\mathcal{L}_{jk}` contain the APs connected to CPU :math:`j` that serve
that UE. Uplink power control assigns UE :math:`k` the data power :math:`p_k`,
subject to

.. math::

   0 \leq p_k \leq P_k^{\mathrm{ul}}.

Downlink power control assigns :math:`\rho_{kl}` to the signal intended for UE
:math:`k` at AP :math:`l`. Every AP must satisfy

.. math::

   \sum_{k=1}^{K}\rho_{kl}
   \leq
   P_l^{\mathrm{ap}}.

Precoder power shares
---------------------
The precoder :math:`\mathbf{w}_{jk}^{(o)}` is normalized jointly over the APs
in :math:`\mathcal{L}_{jk}`. Partitioning it into AP-sized blocks, define

.. math::

   \eta_{kl}
   =
   \mathbb{E}\left\{
     \left\|\mathbf{w}_{jk,l}\right\|^2
   \right\},
   \qquad l\in\mathcal{L}_{jk}.

For every UE--CPU pair, these shares obey

.. math::

   \sum_{l\in\mathcal{L}_{jk}}\eta_{kl}=1.

Consequently, a UE served by multiple CPUs has one independently normalized
precoder per CPU and :math:`\sum_l\eta_{kl}=|\mathcal{J}_k|`.
:type:`ExpectedPrecodingNormSquares` stores the dense :math:`K\times L` matrix
of these expected squared-norm shares, with zeros on unused UE--AP links.

CPU importance weights
----------------------
:type:`PowerImportanceWeights` is a dense :math:`K\times J` representation
whose entry :math:`\omega_{kj}` controls the relative downlink allocation from
CPU :math:`j` to UE :math:`k`. All configurations set non-serving CPU entries
to zero, so an unserved UE has an all-zero importance row. Equal weights use
:math:`\omega_{kj}=1` on serving CPUs. LSFD-based weights use the normalized
squared magnitudes

.. math::

   \omega_{kj}
   =
   \frac{|a_{k,j}|^2}{\|\mathbf{a}_k\|^2}

on serving CPUs. SSFD-based weights instead normalize the squared magnitudes
within every realization and then average the resulting fractions:

.. math::

   \omega_{kj}
   =
   \mathbb{E}_o\left\{
     \frac{|a_{k,j}^{(o)}|^2}
          {\|\mathbf{a}_k^{(o)}\|^2}
   \right\}.

This makes SSFD importance invariant to a realization-dependent nonzero phase
and scale applied to the complete fusion vector.

Downlink allocation policies
----------------------------
:class:`CfgUplinkProportionalDownlinkPower` first distributes :math:`p_k` across
serving CPUs according to normalized LSFD squared magnitudes and then across
their APs according to :math:`\eta_{kl}`. If
:math:`\tilde{\rho}_{kl}` denotes this initial allocation, one common factor

.. math::

   s
   =
   \min_{l:\,\sum_k\tilde{\rho}_{kl}>0}
   \frac{P_l^{\mathrm{ap}}}{\sum_k\tilde{\rho}_{kl}}

is applied to every entry. Thus, the uplink powers determine relative
downlink powers; the final absolute powers can be scaled either up or down.
Computing :math:`s` requires network-wide AP loads and power limits, so this
policy assumes centralized global information.

:class:`CfgEqualDownlinkPower` instead allocates power independently at each
CPU. For CPU :math:`j`, it uses

.. math::

   q_j
   =
   \min_{l\in\mathcal{L}_j:\,\sum_k\omega_{kj}\eta_{kl}>0}
   \frac{P_l^{\mathrm{ap}}}
        {\sum_k\omega_{kj}\eta_{kl}},
   \qquad
   \rho_{kl}=q_j\omega_{kj}\eta_{kl}.

Hence, every UE starts with the same CPU-local scale before importance
weighting. The total allocation from a CPU is independent of how many of its
APs serve the UE, so UEs served by fewer APs can receive a larger share at
each participating AP.

Multi-CPU coordination
----------------------
Independent CPU-local allocation raises two coordination concerns. First,
additional power from one CPU can be wasteful when another CPU already
provides most of a UE's useful signal. The importance weights express this
relative usefulness. Second, allocating substantial power through a poorly
estimated channel can create interference; addressing that concern requires
information about channel quality and allocations at other CPUs.

The optional interference-control stage implements a limited coordination
step for the first concern. It treats the CPU with the largest importance
entry among the serving CPUs as the reference and only decreases allocations
from the other serving CPUs, never increases them. This corresponds to the
broader design of first performing CPU-local allocation, then sharing the
reference allocation, and finally allowing other CPUs to reduce their powers.
"""

import numpy as np
from dagreon import task

from ..scenario import (
    APMaxPower,
    FronthaulLinks,
    NumAPs,
    NumCPUs,
    NumUEs,
    UEMaxPower,
)
from ..scenario.fronthaul import get_set_of_aps_connected_to_cpu_j
from .access import (
    UsedWirelessLinks,
    get_set_of_aps_connected_to_cpu_j_serving_ue_k,
    get_set_of_cpus_serving_ue_k,
)
from .downlink_precoding import ExpectedPrecodingNormSquares
from .uplink_fusion import LSFDWeights, SSFDWeights

#: Uplink data transmit power :math:`p_k` of each UE in watts.
#:
#: :shape: ``(K,)``
#: :dtype: ``float``
type UplinkPowers = np.ndarray
#: Relative importance of each serving CPU to each UE.
#:
#: :shape: ``(K, J)``
#: :dtype: ``float``
type PowerImportanceWeights = np.ndarray
#: Downlink transmit power :math:`\rho_{kl}` allocated by AP :math:`l` to UE
#: :math:`k` in watts.
#:
#: :shape: ``(K, L)``
#: :dtype: ``float``
type DownlinkPowers = np.ndarray


@task
class CfgEqualPowerImportanceWeights:
    """Assign unit importance to serving CPUs and zero to all other CPUs."""

    def __call__(
        self,
        K: NumUEs,
        J: NumCPUs,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> PowerImportanceWeights:
        importance = np.zeros((K, J), dtype=float)
        for k in range(K):
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            importance[k, J_k_set] = 1
        return importance


@task
class CfgLSFDBasedPowerImportanceWeights:
    """Derive CPU importance from normalized LSFD weight magnitudes.

    For each UE, the squared magnitudes sum to one over its serving CPUs. Entries for
    CPUs that do not serve the UE remain zero. A zero LSFD vector produces zero
    importance weights.
    """

    def __call__(
        self,
        K: NumUEs,
        J: NumCPUs,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        a: LSFDWeights,
    ) -> PowerImportanceWeights:
        importance = np.zeros((K, J), dtype=float)
        for k, a_k in enumerate(a):
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            if J_k_set.size == 0:
                continue
            a_energy_k = np.abs(a_k) ** 2
            total_energy_k = np.sum(a_energy_k)
            a_frac_k = np.divide(
                a_energy_k,
                total_energy_k,
                out=np.zeros_like(a_energy_k, dtype=float),
                where=total_energy_k > 0,
            )
            importance[k, J_k_set] = a_frac_k
        return importance


@task
class CfgSSFDBasedPowerImportanceWeights:
    """Average per-realization normalized SSFD weight energies.

    Each realization is normalized separately before its squared magnitudes are
    averaged. The result is invariant to a realization-dependent nonzero phase and
    scale applied to the complete SSFD vector. A zero SSFD vector in a realization
    contributes zero importance weights for that realization.
    """

    def __call__(
        self,
        K: NumUEs,
        J: NumCPUs,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        a: SSFDWeights,
    ) -> PowerImportanceWeights:
        importance = np.zeros((K, J), dtype=float)
        for k, a_k in enumerate(a):
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            if J_k_set.size == 0:
                continue
            a_energy_k = np.abs(a_k) ** 2
            total_energy_k = np.sum(a_energy_k, axis=0, keepdims=True)
            a_frac_k = np.divide(
                a_energy_k,
                total_energy_k,
                out=np.zeros_like(a_energy_k, dtype=float),
                where=total_energy_k > 0,
            )
            importance[k, J_k_set] = np.mean(a_frac_k, axis=1)
        return importance


@task
class CfgMaxUplinkPower:
    """Use the maximum configured uplink data power for every UE."""

    def __call__(self, K: NumUEs, ue_max_power: UEMaxPower) -> UplinkPowers:
        return ue_max_power.copy()


def largest_feasible_power_scale(loads: np.ndarray, limits: APMaxPower) -> float:
    """Return the largest common multiplier satisfying per-AP power limits."""

    active = loads > 0
    if not np.any(active):
        return 0.0
    return float(np.min(limits[active] / loads[active]))


@task
class CfgUplinkProportionalDownlinkPower:
    """Allocate downlink power proportionally to the uplink powers.

    The power is split first across serving CPUs according to normalized
    :type:`LSFDWeights`, then across their APs according to expected precoder norms.
    Finally, one network-wide scale factor fills the limiting AP power constraint.
    This preserves the relative uplink-based allocation but not its absolute scale
    and assumes centralized knowledge of all AP loads and limits.
    """

    def __call__(
        self,
        K: NumUEs,
        J: NumCPUs,
        L: NumAPs,
        rho_max: APMaxPower,
        exp_w_H_w: ExpectedPrecodingNormSquares,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        p: UplinkPowers,
        a: LSFDWeights,
    ) -> DownlinkPowers:
        rho = np.zeros((K, L), dtype=float)
        # Split each UE's uplink power across CPUs according to its LSFD weights,
        # then across the participating APs according to the precoder power shares.
        for k, (a_k, p_k, exp_w_k_H_w_k) in enumerate(zip(a, p, exp_w_H_w)):
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            a_energy_k = np.abs(a_k) ** 2
            total_energy_k = np.sum(a_energy_k)
            a_frac_k = np.divide(
                a_energy_k,
                total_energy_k,
                out=np.zeros_like(a_energy_k, dtype=float),
                where=total_energy_k > 0,
            )
            for j, a_frac_kj in zip(J_k_set, a_frac_k):
                L_jk_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                rho[k, L_jk_set] = p_k * exp_w_k_H_w_k[L_jk_set] * a_frac_kj
        # This common scale requires global knowledge of all AP loads and limits.
        rho = rho * largest_feasible_power_scale(np.sum(rho, axis=0), rho_max)
        assert np.all(np.sum(rho, axis=0) <= rho_max * 1.005), (
            "Power constraints not satisfied"
        )
        return rho


@task
class CfgEqualDownlinkPower:
    """Allocate equal total downlink power to UEs served by the same CPU.

    :type:`PowerImportanceWeights` distribute the CPU allocation across UEs, while
    expected precoder norms distribute each UE's allocation across APs. If
    :code:`interference_control` is enabled, allocations from less important CPUs are
    capped relative to the UE's most important serving CPU.

    Because each UE receives the same CPU-local scale before importance weighting,
    a UE served through fewer APs can receive a larger share at each of those APs.
    """

    interference_control: bool = True

    def __call__(
        self,
        K: NumUEs,
        J: NumCPUs,
        L: NumAPs,
        rho_max: APMaxPower,
        exp_w_H_w: ExpectedPrecodingNormSquares,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        importance: PowerImportanceWeights,
    ) -> DownlinkPowers:
        rho = np.zeros((K, L), dtype=float)
        # Each CPU computes one common power scale for the UEs it serves.
        for j in range(J):
            L_j_set = get_set_of_aps_connected_to_cpu_j(j, fronthaul)
            if L_j_set.size == 0:
                continue
            # Each UE--CPU precoder is normalized to one but may be split across APs.
            exp_w_Lj_H_w_Lj = exp_w_H_w[:, L_j_set] * importance[:, j][:, np.newaxis]
            # The AP with the smallest limit-to-load ratio limits this CPU's scale.
            unit_loads = np.sum(exp_w_Lj_H_w_Lj, axis=0)
            rho_j = largest_feasible_power_scale(unit_loads, rho_max[L_j_set])
            rho[:, L_j_set] = exp_w_Lj_H_w_Lj * rho_j

        if self.interference_control:
            for k in range(K):
                J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
                if J_k_set.size == 0:
                    continue
                j_star = int(J_k_set[np.argmax(importance[k, J_k_set])])
                importance_kj_star = importance[k, j_star]
                L_j_star_set = get_set_of_aps_connected_to_cpu_j(j_star, fronthaul)
                rho_kj_star = np.sum(rho[k, L_j_star_set])
                for j in J_k_set:
                    L_j_set = get_set_of_aps_connected_to_cpu_j(j, fronthaul)
                    rho_kj = np.sum(rho[k, L_j_set])
                    rho_kj_target = importance[k, j] / importance_kj_star * rho_kj_star
                    # Only decrease power; increasing it could violate an AP constraint.
                    if rho_kj > rho_kj_target:
                        rho[k, L_j_set] *= rho_kj_target / rho_kj

        assert np.all(np.sum(rho, axis=0) <= rho_max * 1.005), (
            "Power constraints not satisfied"
        )
        return rho
