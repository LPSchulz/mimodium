r"""Configure local uplink combining at the CPUs serving each UE.

Notation
--------
For CPU :math:`j` detecting UE :math:`k`, let :math:`\mathcal{L}_{jk}` contain
the APs that are connected to CPU :math:`j` and serve UE :math:`k`. The channel
estimate of UE :math:`i`, concatenated across these APs, is

.. math::

   \hat{\mathbf{h}}_{i,\mathcal{L}_{jk}}^{(o)}
   =
   \operatorname{col}\left(
     \left\{\hat{\mathbf{h}}_{il}^{(o)}
     : l\in\mathcal{L}_{jk}\right\}
   \right),

and its block-diagonal estimation-error correlation is

.. math::

   \mathbf{C}_{i,\mathcal{L}_{jk}}
   =
   \operatorname{blockdiag}\left(
     \left\{\mathbf{C}_{il}:l\in\mathcal{L}_{jk}\right\}
   \right).

Both have spatial dimension :math:`N|\mathcal{L}_{jk}|`.
:math:`\mathcal{K}_{jk}^{\mathrm{comb}}` denotes the UEs whose available channel
information is used to design the local combining vector
:math:`\mathbf{v}_{jk}^{(o)}`. Depending on its configuration, this set can
include the desired UE :math:`k` as well as interfering UEs.

Relationship between combining schemes
--------------------------------------
The power-dependent combiners obtain :math:`\mathbf{v}_{jk}^{(o)}` from a
linear system

.. math::

   \mathbf{A}_{jk}^{(o)}\mathbf{v}_{jk}^{(o)}
   =
   \sqrt{p_k}\hat{\mathbf{h}}_{k,\mathcal{L}_{jk}}^{(o)}.

For MMSE combining, the system matrix is the sum of three contributions:

.. math::

   \mathbf{A}_{jk}^{(o)}
   =
   \underbrace{
     \sum_{i\in\mathcal{K}_{jk}^{\mathrm{comb}}}
     p_i
     \hat{\mathbf{h}}_{i,\mathcal{L}_{jk}}^{(o)}
     \hat{\mathbf{h}}_{i,\mathcal{L}_{jk}}^{(o)\mathrm{H}}
   }_{\text{instantaneous estimated-channel correlations}}
   +
   \underbrace{
     \sum_{i\in\mathcal{K}_{jk}^{\mathrm{comb}}}
     p_i\mathbf{C}_{i,\mathcal{L}_{jk}}
   }_{\text{estimation-error correlations}}
   +
   \underbrace{
     \sigma_{\mathrm{ul}}^2\mathbf{I}
   }_{\text{receiver noise}}.

RZF omits the estimation-error term, while ZF also omits the receiver-noise
term. Consequently, the MMSE and RZF matrices are positive definite when the
receiver-noise power is positive. The ZF matrix is only positive semidefinite
and may be singular, so ZF uses a least-squares solution instead of requiring
an inverse. MR does not form or solve a system: it uses the desired UE's
concatenated channel estimate directly.

Maximum-power design convention
-------------------------------
ZF, RZF, and MMSE combiners depend on UE transmit powers. They are designed here
using each UE's configured maximum uplink power,
:math:`p_i=P_i^{\mathrm{ul}}`, supplied through :type:`UEMaxPower`. These are
design powers: subsequent power control may assign lower operational powers
without redesigning the combining vectors. This convention keeps combiner design
independent of the later power-control result; no fixed-point iteration between
power allocation and combining is performed.
"""

import numpy as np
from dagreon import task

from ..array_ops import (
    concatenate_all_channels_of_aps,
    concatenate_all_correlations_of_aps,
    outer_product_with_self,
)
from ..propagation import NumRealizations, ULNoisePower
from ..scenario import FronthaulLinks, NumUEs, UEMaxPower
from .access import (
    EstimatedChannelLinks,
    MeasuredStatisticLinks,
    UsedWirelessLinks,
    get_set_of_aps_connected_to_cpu_j_serving_ue_k,
    get_set_of_cpus_serving_ue_k,
)
from .channel_estimation import ChannelEstimates, EstimationErrorCorrelations

#: Squared norms of local combining vectors for every serving CPU and realization.
#:
#: :shape: ``(J_k, O)``
#: :dtype: ``float``
type CombiningNormSquares = list[np.ndarray]
#: Expected squared norms of local combining vectors.
#:
#: :shape: ``(J_k,)``
#: :dtype: ``float``
type ExpectedCombiningNormSquares = list[np.ndarray]
#: Local uplink combining vectors. Each nested entry stores
#: :math:`\mathbf{v}_{jk}^{(o)}` across all realizations for one CPU :math:`j`
#: serving UE :math:`k`.
#:
#: :shape: ``(N |L_jk|, O)``
#: :dtype: ``complex``
type CombiningVectors = list[list[np.ndarray]]
#: UE indices in :math:`\mathcal{K}_{jk}^{\mathrm{comb}}` whose available
#: channel information is used to design each local combining vector. The desired
#: UE :math:`k` may be included.
#:
#: :shape: ``(|K_jk^comb|,)``
#: :dtype: ``int``
type CombinerDesignUEs = list[list[np.ndarray]]


@task
class ComputeCombiningNormSquares:
    r"""Compute :math:`\|\mathbf{v}_{jk}^{(o)}\|^2` for every local combiner."""

    def __call__(self, O: NumRealizations, v: CombiningVectors) -> CombiningNormSquares:
        v_H_v: list[np.ndarray] = []
        for v_k in v:
            v_k_H_v_k = np.zeros((len(v_k), O), dtype=float)
            for j, v_jk in enumerate(v_k):
                v_k_H_v_k[j] = np.real(np.vecdot(v_jk.T, v_jk.T))
            v_H_v.append(v_k_H_v_k)
        return v_H_v


@task
class ComputeExpectedCombiningNormSquares:
    """Average local combining-vector norm squares over channel realizations."""

    def __call__(self, v_H_v: CombiningNormSquares) -> ExpectedCombiningNormSquares:
        exp_v_H_v: list[np.ndarray] = []
        for v_k_H_v_k in v_H_v:
            exp_v_H_v.append(np.mean(v_k_H_v_k, axis=1))
        return exp_v_H_v


@task
class CfgConsiderAllUEs:
    """Include every UE when designing every local combining vector."""

    def __call__(
        self, K: NumUEs, used: UsedWirelessLinks, fronthaul: FronthaulLinks
    ) -> CombinerDesignUEs:
        combiner_ues: list[list[np.ndarray]] = []
        for k in range(K):
            K_jk_sets: list[np.ndarray] = []
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            for _ in J_k:
                K_jk_sets.append(np.arange(K))
            combiner_ues.append(K_jk_sets)
        return combiner_ues


@task
class CfgConsiderNoUEs:
    """Use an empty UE set when designing every local combining vector."""

    def __call__(
        self, K: NumUEs, used: UsedWirelessLinks, fronthaul: FronthaulLinks
    ) -> CombinerDesignUEs:
        combiner_ues: list[list[np.ndarray]] = []
        for k in range(K):
            K_jk_sets: list[np.ndarray] = []
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            for _ in J_k:
                K_jk_sets.append(np.array([], dtype=int))
            combiner_ues.append(K_jk_sets)
        return combiner_ues


@task
class CfgConsiderUEsWithMeasuredStatistics:
    """Consider UEs measured by APs participating in each local combiner.

    For CPU :math:`j` detecting UE :math:`k`, take the union of UEs whose large-scale
    statistics are measured by the APs of that CPU that serve UE :math:`k`.
    """

    def __call__(
        self,
        K: NumUEs,
        measured: MeasuredStatisticLinks,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> CombinerDesignUEs:
        combiner_ues: list[list[np.ndarray]] = []
        for k in range(K):
            K_jk_sets: list[np.ndarray] = []
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            for j in J_k:
                K_jk_parts: list[np.ndarray] = []
                # we only care about the APs connected to this CPU that serve this UE
                # all other APs have no impact in the combining step at this CPU
                L_jk = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                for l in L_jk:
                    K_jk_parts.append(np.flatnonzero(measured[:, l]))
                K_jk_sets.append(np.unique(np.concatenate(K_jk_parts)))
            combiner_ues.append(K_jk_sets)
        return combiner_ues


@task
class CfgConsiderUEsWithEstimatedChannels:
    """Consider UEs estimated by APs participating in each local combiner.

    For CPU :math:`j` detecting UE :math:`k`, take the union of UEs whose channels are
    estimated by the APs of that CPU that serve UE :math:`k`.
    """

    def __call__(
        self,
        K: NumUEs,
        estimated: EstimatedChannelLinks,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> CombinerDesignUEs:
        combiner_ues: list[list[np.ndarray]] = []
        for k in range(K):
            K_jk_sets: list[np.ndarray] = []
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            for j in J_k:
                K_jk_parts: list[np.ndarray] = []
                L_jk = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                for l in L_jk:
                    K_jk_parts.append(np.flatnonzero(estimated[:, l]))
                K_jk_sets.append(np.unique(np.concatenate(K_jk_parts)))
            combiner_ues.append(K_jk_sets)
        return combiner_ues


@task
class CfgConsiderUEsServedBySameAPs:
    """Consider UEs served by any AP participating in each local combiner."""

    def __call__(
        self,
        K: NumUEs,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> CombinerDesignUEs:
        combiner_ues: list[list[np.ndarray]] = []
        for k in range(K):
            K_jk_sets: list[np.ndarray] = []
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            for j in J_k:
                K_jk_parts: list[np.ndarray] = []
                L_jk = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                for l in L_jk:
                    K_jk_parts.append(np.flatnonzero(used[:, l]))
                K_jk_sets.append(np.unique(np.concatenate(K_jk_parts)))
            combiner_ues.append(K_jk_sets)
        return combiner_ues


@task
class CfgMaximumRatioCombining:
    r"""Configure local maximum-ratio (MR) combining.

    MR uses the desired UE's concatenated channel estimate directly:

    .. math::

       \mathbf{v}_{jk}^{(o)}
       = \hat{\mathbf{h}}_{k,\mathcal{L}_{jk}}^{(o)}.

    It therefore requires neither a considered-UE set nor design powers.
    """

    def __call__(
        self,
        K: NumUEs,
        h_hat: ChannelEstimates,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> CombiningVectors:
        v: list[list[np.ndarray]] = []
        for k in range(K):
            v_k: list[np.ndarray] = []
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            for j in J_k_set:
                L_jk_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                h_hat_jk = concatenate_all_channels_of_aps(h_hat, L_jk_set)[k]
                v_k.append(h_hat_jk.copy())
            v.append(v_k)
        return v


@task
class CfgZeroForcingCombining:
    r"""Configure local zero-forcing (ZF) combining.

    For every channel realization, compute

    .. math::

       \mathbf{v}_{jk}^{(o)}
       =
       \sqrt{p_k}
       \left(
         \sum_{i\in\mathcal{K}_{jk}^{\mathrm{comb}}}
         p_i
         \hat{\mathbf{h}}_{i,\mathcal{L}_{jk}}^{(o)}
         \hat{\mathbf{h}}_{i,\mathcal{L}_{jk}}^{(o)\mathrm{H}}
       \right)^\dagger
       \hat{\mathbf{h}}_{k,\mathcal{L}_{jk}}^{(o)},

    where :math:`(\cdot)^\dagger` is the Moore--Penrose pseudoinverse. The
    implementation obtains the corresponding minimum-norm solution by least
    squares, so the considered-channel covariance may be singular.

    The powers :math:`p_i` are the maximum-power design values described in the
    module-level convention.
    """

    def __call__(
        self,
        O: NumRealizations,
        p_max: UEMaxPower,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        h_hat: ChannelEstimates,
        combiner_ues: CombinerDesignUEs,
    ) -> CombiningVectors:
        v: list[list[np.ndarray]] = []
        for k, K_jk_sets in enumerate(combiner_ues):
            v_k: list[np.ndarray] = []
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            assert len(J_k_set) == len(K_jk_sets)
            for j, K_jk in zip(J_k_set, K_jk_sets):
                L_jk_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                h_hat_jk = concatenate_all_channels_of_aps(h_hat, L_jk_set)
                N_jk = h_hat_jk.shape[1]
                known_corr = outer_product_with_self(h_hat_jk[K_jk], axis=1)
                p_known_corr_sum = np.linalg.vecdot(p_max[K_jk], known_corr, axis=0)  # type: ignore
                # matrix is not guaranteed to be invertible
                # np.lstsq requires 2D input
                v_jk = np.zeros((N_jk, O), dtype=complex)
                for o in range(O):
                    x, _, _, _ = np.linalg.lstsq(
                        p_known_corr_sum[:, :, o], h_hat_jk[k, :, o]
                    )
                    v_jk[:, o] = np.sqrt(p_max[k]) * x
                v_k.append(v_jk)
            v.append(v_k)
        return v


@task
class CfgRegularizedZeroForcingCombining:
    r"""Configure local regularized zero-forcing (RZF) combining.

    For every channel realization, compute

    .. math::

       \mathbf{v}_{jk}^{(o)}
       =
       \sqrt{p_k}
       \left(
         \sum_{i\in\mathcal{K}_{jk}^{\mathrm{comb}}}
         p_i
         \hat{\mathbf{h}}_{i,\mathcal{L}_{jk}}^{(o)}
         \hat{\mathbf{h}}_{i,\mathcal{L}_{jk}}^{(o)\mathrm{H}}
         + \sigma_{\mathrm{ul}}^2\mathbf{I}
       \right)^{-1}
       \hat{\mathbf{h}}_{k,\mathcal{L}_{jk}}^{(o)}.

    The uplink receiver-noise covariance provides the regularization.
    Estimation-error correlations are not included. The powers :math:`p_i` are
    the maximum-power design values described in the module-level convention.
    """

    def __call__(
        self,
        p_max: UEMaxPower,
        sigma2_ul: ULNoisePower,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        h_hat: ChannelEstimates,
        combiner_ues: CombinerDesignUEs,
    ) -> CombiningVectors:
        v: list[list[np.ndarray]] = []
        for k, K_jk_sets in enumerate(combiner_ues):
            v_k: list[np.ndarray] = []
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            assert len(J_k_set) == len(K_jk_sets)
            for j, K_jk in zip(J_k_set, K_jk_sets):
                L_jk_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                h_hat_jk = concatenate_all_channels_of_aps(h_hat, L_jk_set)
                N_jk = h_hat_jk.shape[1]
                known_corr = outer_product_with_self(h_hat_jk[K_jk], axis=1)
                p_known_corr_sum = np.linalg.vecdot(p_max[K_jk], known_corr, axis=0)  # type: ignore

                noise_power = sigma2_ul * np.eye(N_jk, dtype=complex)

                v_jk = (
                    np.sqrt(p_max[k])
                    * (
                        np.linalg.solve(
                            np.transpose(p_known_corr_sum, (2, 0, 1))
                            + noise_power[np.newaxis, :, :],
                            np.transpose(h_hat_jk[k, :, np.newaxis, :], (2, 0, 1)),
                        )
                    )[:, :, 0].T
                )
                v_k.append(v_jk)
            v.append(v_k)
        return v


@task
class CfgMMSECombining:
    r"""Configure classical local minimum mean-square error (MMSE) combining.

    For every channel realization, compute

    .. math::

       \mathbf{v}_{jk}^{(o)}
       =
       \sqrt{p_k}
       \left(
         \sum_{i\in\mathcal{K}_{jk}^{\mathrm{comb}}} p_i
         \left(
           \hat{\mathbf{h}}_{i,\mathcal{L}_{jk}}^{(o)}
           \hat{\mathbf{h}}_{i,\mathcal{L}_{jk}}^{(o)\mathrm{H}}
           + \mathbf{C}_{i,\mathcal{L}_{jk}}
         \right)
         + \sigma_{\mathrm{ul}}^2\mathbf{I}
       \right)^{-1}
       \hat{\mathbf{h}}_{k,\mathcal{L}_{jk}}^{(o)}.

    Thus, each local system matrix contains instantaneous estimated-channel
    correlations, statistical estimation-error correlations, and receiver noise.
    The powers :math:`p_i` are the maximum-power design values described in the
    module-level convention.
    """

    def __call__(
        self,
        O: NumRealizations,
        p_max: UEMaxPower,
        sigma2_ul: ULNoisePower,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        h_hat: ChannelEstimates,
        C: EstimationErrorCorrelations,
        combiner_ues: CombinerDesignUEs,
    ) -> CombiningVectors:
        v: list[list[np.ndarray]] = []
        for k, K_jk_sets in enumerate(combiner_ues):
            v_k: list[np.ndarray] = []
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            assert len(J_k_set) == len(K_jk_sets)
            for j, K_jk in zip(J_k_set, K_jk_sets):
                L_jk_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                h_hat_jk = concatenate_all_channels_of_aps(h_hat, L_jk_set)
                N_jk = h_hat_jk.shape[1]
                mat_sum = (
                    np.zeros((N_jk, N_jk, O), dtype=complex)
                    + sigma2_ul * np.eye(N_jk, dtype=complex)[:, :, np.newaxis]
                )
                C_jk = concatenate_all_correlations_of_aps(C, L_jk_set)
                for i in K_jk:
                    mat_sum += p_max[i] * (
                        outer_product_with_self(h_hat_jk[i], axis=0)
                        + C_jk[i][:, :, np.newaxis]
                    )

                v_jk = (
                    np.sqrt(p_max[k])
                    * (
                        np.linalg.solve(
                            np.transpose(mat_sum, (2, 0, 1)),
                            np.transpose(h_hat_jk[k, :, np.newaxis, :], (2, 0, 1)),
                        )
                    )[:, :, 0].T
                )
                v_k.append(v_jk)
            v.append(v_k)
        return v
