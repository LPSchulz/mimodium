r"""Configure the fusion of locally combined uplink signals across CPUs.

Notation
--------
Let :math:`\mathcal{J}_k` contain the CPUs serving UE :math:`k`. After CPU
:math:`j\in\mathcal{J}_k` applies its local combining vector
:math:`\mathbf{v}_{jk}^{(o)}`, the effective channel from transmitting UE
:math:`i` to the local detector for UE :math:`k` is

.. math::

   g_{ki,j}^{(o)}
   =
   \mathbf{v}_{jk}^{(o)\mathrm{H}}\mathbf{h}_{i,\mathcal{L}_{jk}}^{(o)},

where :math:`\mathbf{h}_{i,\mathcal{L}_{jk}}^{(o)}` concatenates UE
:math:`i`'s channels to the APs connected to CPU :math:`j` that serve UE
:math:`k`. Collecting the scalar effective channels from all serving CPUs gives

.. math::

   \mathbf{g}_{ki}^{(o)}
   =
   \operatorname{col}\left(
     \left\{g_{ki,j}^{(o)}:j\in\mathcal{J}_k\right\}
   \right).

The corresponding estimated effective channel
:math:`\hat{\mathbf{g}}_{ki}^{(o)}` is obtained by replacing the physical
channels with their estimates.

Effective-channel knowledge
---------------------------
Let :math:`\mathcal J_k^{\mathrm{inst}}\subseteq\mathcal J_k` contain the
serving CPUs whose instantaneous effective-channel estimates are available at
the detector. The configured alternatives use either only the master CPU or
all serving CPUs. The set :math:`\mathcal{K}_{jk}^{\mathrm{meas}}` contains
the UEs measured by at least one AP participating in CPU :math:`j`'s local
detector for UE :math:`k`. The known effective channel is

.. math::

   [\mathbf{g}_{ki,\mathrm{known}}^{(o)}]_j
   =
   \begin{cases}
     \hat{g}_{ki,j}^{(o)}, & j\in\mathcal J_k^{\mathrm{inst}},\\
     \mathbb{E}\{g_{ki,j}\},
       & j\notin\mathcal J_k^{\mathrm{inst}},
         \ i\in\mathcal{K}_{jk}^{\mathrm{meas}},\\
     0, & \text{otherwise}.
   \end{cases}

The residual is the difference between the realized and known effective
channels. :class:`ComputeConditionalUnknownEffectiveULChannelOuters` retains
the dependence of its correlation on the available instantaneous CSI. For
:math:`j\in\mathcal J_k^{\mathrm{inst}}`, the conditional variance is

.. math::

   \mathbf v_{jk}^{(o)\mathrm H}
   \mathbf C_{i,j}
   \mathbf v_{jk}^{(o)}.

LSFD and SSFD fusion
--------------------
Both fusion stages linearly combine the locally detected signals from the CPUs
in :math:`\mathcal{J}_k`. Large-scale fading decoding (LSFD) uses a single
statistical weight vector :math:`\mathbf{a}_k`, whereas small-scale fading
decoding (SSFD) uses a realization-dependent vector
:math:`\mathbf{a}_k^{(o)}` based on the effective-channel knowledge available
at the master CPU.

Let :math:`\mathcal{K}_k^{\mathrm{fuse}}` contain the UEs included when
designing the fusion vector, and define the statistical local-combining noise
matrix

.. math::

   \mathbf{D}_k
   =
   \operatorname{diag}\left(
     \left\{\mathbb{E}\{\|\mathbf{v}_{jk}\|^2\}
     :j\in\mathcal{J}_k\right\}
   \right).

The optimal LSFD implementation solves

.. math::

   \mathbf{A}_k^{\mathrm{LSFD}}\mathbf{a}_k
   =
   p_k\mathbb{E}\{\mathbf{g}_{kk}\},

with

.. math::

   \mathbf{A}_k^{\mathrm{LSFD}}
   =
   \sum_{i\in\mathcal{K}_k^{\mathrm{fuse}}}
   p_i\mathbb{E}\left\{
     \mathbf{g}_{ki}\mathbf{g}_{ki}^{\mathrm{H}}
   \right\}
   +
   \sigma_{\mathrm{ul}}^2\mathbf{D}_k.

For each realization, the optimal SSFD implementation instead solves

.. math::

   \mathbf{A}_k^{(o),\mathrm{SSFD}}\mathbf{a}_k^{(o)}
   =
   \sqrt{p_k}\mathbf{g}_{kk,\mathrm{known}}^{(o)},

where

.. math::

   \mathbf{A}_k^{(o),\mathrm{SSFD}}
   =
   \sum_{i\in\mathcal{K}_k^{\mathrm{fuse}}}
   p_i\left(
     \mathbf{g}_{ki,\mathrm{known}}^{(o)}
     \mathbf{g}_{ki,\mathrm{known}}^{(o)\mathrm{H}}
     +
     \mathbb E\left\{
       \left.
       \mathbf{g}_{ki,\mathrm{unknown}}^{(o)}
       \mathbf{g}_{ki,\mathrm{unknown}}^{(o)\mathrm H}
       \right|
       \mathcal I_k
     \right\}
   \right)
   +
   \sigma_{\mathrm{ul}}^2
   \operatorname{diag}\left(
     \left\{\|\mathbf v_{jk}^{(o)}\|^2
     :j\in\mathcal J_k\right\}
   \right).

Maximum-power design convention
-------------------------------
The optimal LSFD and SSFD vectors are designed using each UE's configured
maximum uplink power, :math:`p_i=P_i^{\mathrm{ul}}`, supplied through
:type:`UEMaxPower`. Subsequent power control may assign lower operational
powers without redesigning the fusion vectors. In particular, power control
itself can depend on the LSFD vector, so using the maximum-power vector for
LSFD design breaks this circular dependency. No fixed-point iteration between
LSFD design and power control is performed.

Multiplying a fusion vector by any nonzero scalar does not change the resulting
SINR. The factors :math:`p_k` in the LSFD right-hand side and
:math:`\sqrt{p_k}` in the SSFD right-hand side therefore select a normalization
only; they do not change the optimal fusion directions or spectral
efficiencies.
"""

import numpy as np
from dagreon import task

from ..array_ops import (
    concatenate_all_channels_of_aps,
    concatenate_all_correlations_of_aps,
    outer_product_with_self,
)
from ..propagation import ChannelRealizations, NumRealizations, ULNoisePower
from ..scenario import FronthaulLinks, NumUEs, UEMaxPower
from .access import (
    EstimatedChannelLinks,
    MasterCPU,
    MeasuredStatisticLinks,
    UsedWirelessLinks,
    get_set_of_aps_connected_to_cpu_j_serving_ue_k,
    get_set_of_cpus_serving_ue_k,
)
from .channel_estimation import ChannelEstimates, EstimationErrorCorrelations
from .uplink_combining import (
    CombiningNormSquares,
    CombiningVectors,
    ExpectedCombiningNormSquares,
)

#: Effective uplink channels after local combining. Entry ``g[k]`` has one row per
#: transmitting UE and one column per CPU serving UE :math:`k`.
#:
#: :shape: ``(K, J_k, O)``
#: :dtype: ``complex``
type EffectiveUplinkChannels = list[np.ndarray]
#: Effective uplink channels computed from estimated physical channels.
#:
#: :shape: ``(K, J_k, O)``
#: :dtype: ``complex``
type EstimatedEffectiveUplinkChannels = list[np.ndarray]
#: Realization average of :type:`EffectiveUplinkChannels`.
#:
#: :shape: ``(K, J_k)``
#: :dtype: ``complex``
type ExpectedEffectiveUplinkChannels = list[np.ndarray]
#: Expected outer products of effective uplink channels over serving CPUs.
#:
#: :shape: ``(K, J_k, J_k)``
#: :dtype: ``complex``
type ExpectedEffectiveUplinkChannelOuters = list[np.ndarray]
#: For each detected UE, mask of serving CPUs whose instantaneous estimated
#: effective channels are available at the detector.
#:
#: :shape: ``(J_k,)``
#: :dtype: ``bool``
type InstantaneousEffectiveUplinkChannelKnowledge = list[np.ndarray]
#: Effective uplink channel knowledge available at the detector.
#:
#: :shape: ``(K, J_k, O)``
#: :dtype: ``complex``
type KnownEffectiveUplinkChannels = list[np.ndarray]
#: Residual effective uplink channels not available at the detector.
#:
#: :shape: ``(K, J_k, O)``
#: :dtype: ``complex``
type UnknownEffectiveUplinkChannels = list[np.ndarray]
#: Expected outer products of unknown effective uplink channels, conditioned on
#: the instantaneous CSI available at the detector. The last dimension retains
#: the resulting realization dependence.
#:
#: :shape: ``(K, J_k, J_k, O)``
#: :dtype: ``complex``
type ExpectedUnknownEffectiveUplinkChannelOuters = list[np.ndarray]
#: UE indices in :math:`\mathcal{K}_k^{\mathrm{fuse}}` whose available
#: effective-channel information is used to design the fusion vector for UE
#: :math:`k`. The desired UE :math:`k` may be included.
#:
#: :shape: ``(|K_k^fuse|,)``
#: :dtype: ``int``
type FusionDesignUEs = list[np.ndarray]
#: Large-scale fading decoding weights over the CPUs serving each UE.
#:
#: :shape: ``(J_k,)``
#: :dtype: ``complex``
type LSFDWeights = list[np.ndarray]
#: Small-scale fading decoding weights over serving CPUs and realizations.
#:
#: :shape: ``(J_k, O)``
#: :dtype: ``complex``
type SSFDWeights = list[np.ndarray]


@task
class ComputeEffectiveULChannels:
    r"""Compute uplink channels after local combining at every serving CPU.

    For detector UE :math:`k`, transmitting UE :math:`i`, and serving CPU :math:`j`,
    compute :math:`g_{ki,j}^{(o)} =
    \mathbf{v}_{jk}^{\mathrm{H}}\mathbf{h}_{ij}^{(o)}`.
    """

    def __call__(
        self,
        K: NumUEs,
        O: NumRealizations,
        h: ChannelRealizations,
        v: CombiningVectors,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> EffectiveUplinkChannels:
        g: list[np.ndarray] = []
        assert len(v) == K
        for k in range(K):
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            assert len(J_k_set) == len(v[k])
            g_k = np.zeros((K, len(J_k_set), O), dtype=complex)
            for j_index, (j, v_jk) in enumerate(zip(J_k_set, v[k])):
                L_jk_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                h_jk = concatenate_all_channels_of_aps(h, L_jk_set)
                g_k[:, j_index] = np.vecdot(v_jk.T, np.transpose(h_jk, (0, 2, 1)))
            g.append(g_k)
        return g


@task
class ComputeExpectedEffectiveULChannels:
    """Average effective uplink channels over channel realizations."""

    def __call__(self, g: EffectiveUplinkChannels) -> ExpectedEffectiveUplinkChannels:
        exp_g: list[np.ndarray] = []
        for g_k in g:
            exp_g.append(np.mean(g_k, axis=2))
        return exp_g


@task
class ComputeExpectedEffectiveULChannelOuters:
    r"""Compute :math:`\mathbb{E}\{\mathbf{g}_{ki}\mathbf{g}_{ki}^{\mathrm{H}}\}`
    over channel realizations."""

    def __call__(
        self, g: EffectiveUplinkChannels
    ) -> ExpectedEffectiveUplinkChannelOuters:
        exp_g_g_H: list[np.ndarray] = []
        for g_k in g:
            exp_g_g_H.append(
                np.mean(outer_product_with_self(np.transpose(g_k, (0, 2, 1))), axis=1)
            )
        return exp_g_g_H


@task
class ComputeEstimatedEffectiveULChannels:
    r"""Compute effective uplink channels from estimated physical channels.

    This applies each local combining vector to :type:`ChannelEstimates` instead of
    the realized channels.
    """

    def __call__(
        self,
        K: NumUEs,
        O: NumRealizations,
        h_hat: ChannelEstimates,
        v: CombiningVectors,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> EstimatedEffectiveUplinkChannels:
        g_hat: list[np.ndarray] = []
        assert len(v) == K
        for k in range(K):
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            assert len(J_k_set) == len(v[k])
            g_hat_k = np.zeros((K, len(J_k_set), O), dtype=complex)
            for j_index, (j, v_jk) in enumerate(zip(J_k_set, v[k])):
                L_jk_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                h_hat_jk = concatenate_all_channels_of_aps(h_hat, L_jk_set)
                g_hat_k[:, j_index] = np.vecdot(
                    v_jk.T, np.transpose(h_hat_jk, (0, 2, 1))
                )
            g_hat.append(g_hat_k)
        return g_hat


@task
class CfgMasterCPUInstantaneousEffectiveULKnowledge:
    """Make only the master CPU's effective-channel estimates instantaneous."""

    def __call__(
        self,
        K: NumUEs,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        master_cpu: MasterCPU,
    ) -> InstantaneousEffectiveUplinkChannelKnowledge:
        instantaneous: list[np.ndarray] = []
        assert len(master_cpu) == K
        for k in range(K):
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            instantaneous_k = J_k == master_cpu[k]
            assert len(J_k) == 0 or np.count_nonzero(instantaneous_k) == 1
            instantaneous.append(instantaneous_k)
        return instantaneous


@task
class CfgAllServingCPUsInstantaneousEffectiveULKnowledge:
    """Make every serving CPU's effective-channel estimates instantaneous."""

    def __call__(
        self,
        K: NumUEs,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> InstantaneousEffectiveUplinkChannelKnowledge:
        instantaneous: list[np.ndarray] = []
        for k in range(K):
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            instantaneous.append(np.ones(len(J_k), dtype=bool))
        return instantaneous


@task
class ComputeKnownEffectiveULChannels:
    """Construct the effective-channel knowledge available for fusion.

    For each UE, use instantaneous estimated effective channels from the CPUs
    selected by :type:`InstantaneousEffectiveUplinkChannelKnowledge` and available
    statistical mean channels from every other serving CPU. A statistical effective
    channel is available when at least one AP participating in the corresponding
    local detector measures that transmitting UE.
    """

    def __call__(
        self,
        K: NumUEs,
        O: NumRealizations,
        g_hat: EstimatedEffectiveUplinkChannels,
        exp_g: ExpectedEffectiveUplinkChannels,
        measured: MeasuredStatisticLinks,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        instantaneous: InstantaneousEffectiveUplinkChannelKnowledge,
    ) -> KnownEffectiveUplinkChannels:
        g_known: list[np.ndarray] = []
        assert len(g_hat) == len(exp_g) == len(instantaneous) == K
        for k, (g_hat_k, exp_g_k, instantaneous_k) in enumerate(
            zip(g_hat, exp_g, instantaneous)
        ):
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            assert g_hat_k.shape[1] == exp_g_k.shape[1] == len(J_k_set)
            assert instantaneous_k.shape == (len(J_k_set),)
            g_known_k = np.tile(exp_g_k[:, :, np.newaxis], (1, 1, O))
            for j_index, j in enumerate(J_k_set):
                L_jk_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                measured_by_cpu = np.any(measured[:, L_jk_set], axis=1)
                g_known_k[~measured_by_cpu, j_index] = 0
            g_known_k[:, instantaneous_k] = g_hat_k[:, instantaneous_k]
            g_known.append(g_known_k)
        return g_known


@task
class ComputeUnknownEffectiveULChannels:
    """Subtract known effective channels from the realized effective channels."""

    def __call__(
        self, g: EffectiveUplinkChannels, g_known: KnownEffectiveUplinkChannels
    ) -> UnknownEffectiveUplinkChannels:
        g_unknown: list[np.ndarray] = []
        assert len(g) == len(g_known)
        for g_k, g_known_k in zip(g, g_known):
            g_unknown.append(g_k - g_known_k)
        return g_unknown


@task
class ComputeConditionalUnknownEffectiveULChannelOuters:
    r"""Condition effective-channel uncertainty on the available instantaneous CSI.

    Unknown components without instantaneous CSI retain their unconditional
    covariance. For every instantaneously known CPU component, independence
    between CPUs makes its cross-covariances zero and its conditional variance is

    .. math::

       \mathbf{v}_{kj}^{\mathrm H}\mathbf{C}_{k'j}\mathbf{v}_{kj}.
    """

    def __call__(
        self,
        K: NumUEs,
        O: NumRealizations,
        g_unknown: UnknownEffectiveUplinkChannels,
        C: EstimationErrorCorrelations,
        v: CombiningVectors,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        instantaneous: InstantaneousEffectiveUplinkChannelKnowledge,
    ) -> ExpectedUnknownEffectiveUplinkChannelOuters:
        conditional: list[np.ndarray] = []
        assert len(g_unknown) == len(v) == len(instantaneous) == K
        for k, (
            g_unknown_k,
            v_k,
            instantaneous_k,
        ) in enumerate(zip(g_unknown, v, instantaneous)):
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            assert len(J_k_set) == len(v_k) == len(instantaneous_k)
            unconditional_k = np.mean(
                outer_product_with_self(np.transpose(g_unknown_k, (0, 2, 1))),
                axis=1,
            )
            conditional_k = np.repeat(unconditional_k[..., np.newaxis], O, axis=3)
            for j_index in np.flatnonzero(instantaneous_k):
                j = J_k_set[j_index]
                L_jk_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                C_jk = concatenate_all_correlations_of_aps(C, L_jk_set)
                v_jk = v_k[j_index]
                conditional_k[:, j_index, :, :] = 0
                conditional_k[:, :, j_index, :] = 0
                conditional_k[:, j_index, j_index, :] = np.real(
                    np.vecdot(v_jk.T, np.transpose(C_jk @ v_jk, (0, 2, 1)))
                )
            conditional.append(conditional_k)
        return conditional


@task
class CfgConsiderAllUEs:
    """Include every UE when designing each served UE's fusion vector."""

    def __call__(
        self,
        K: NumUEs,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> FusionDesignUEs:
        fusion_ues: list[np.ndarray] = []
        for k in range(K):
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            if len(J_k) == 0:
                fusion_ues.append(np.array([], dtype=int))
            else:
                fusion_ues.append(np.arange(K))
        return fusion_ues


@task
class CfgConsiderNoUEs:
    """Use an empty UE set when designing every fusion vector."""

    def __call__(
        self,
        K: NumUEs,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> FusionDesignUEs:
        return [np.array([], dtype=int) for _ in range(K)]


@task
class CfgConsiderUEsWithMeasuredStatistics:
    """Consider UEs measured by APs participating in each fusion stage.

    For detected UE :math:`k`, take the union of UEs whose large-scale
    statistics are measured by any AP contributing a local detector to the
    fusion stage.
    """

    def __call__(
        self,
        K: NumUEs,
        measured: MeasuredStatisticLinks,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> FusionDesignUEs:
        fusion_ues: list[np.ndarray] = []
        for k in range(K):
            K_k_parts: list[np.ndarray] = []
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            for j in J_k:
                L_jk = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                for l in L_jk:
                    K_k_parts.append(np.flatnonzero(measured[:, l]))
            if K_k_parts:
                fusion_ues.append(np.unique(np.concatenate(K_k_parts)))
            else:
                fusion_ues.append(np.array([], dtype=int))
        return fusion_ues


@task
class CfgConsiderUEsWithEstimatedChannels:
    """Consider UEs estimated by APs participating in each fusion stage.

    For detected UE :math:`k`, take the union of UEs whose channels are
    estimated by any AP contributing a local detector to the fusion stage.
    """

    def __call__(
        self,
        K: NumUEs,
        estimated: EstimatedChannelLinks,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> FusionDesignUEs:
        fusion_ues: list[np.ndarray] = []
        for k in range(K):
            K_k_parts: list[np.ndarray] = []
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            for j in J_k:
                L_jk = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                for l in L_jk:
                    K_k_parts.append(np.flatnonzero(estimated[:, l]))
            if K_k_parts:
                fusion_ues.append(np.unique(np.concatenate(K_k_parts)))
            else:
                fusion_ues.append(np.array([], dtype=int))
        return fusion_ues


@task
class CfgConsiderUEsServedBySameAPs:
    """Consider UEs served by APs participating in each fusion stage."""

    def __call__(
        self,
        K: NumUEs,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> FusionDesignUEs:
        fusion_ues: list[np.ndarray] = []
        for k in range(K):
            K_k_parts: list[np.ndarray] = []
            J_k = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            for j in J_k:
                L_jk = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, k, used, fronthaul
                )
                for l in L_jk:
                    K_k_parts.append(np.flatnonzero(used[:, l]))
            if K_k_parts:
                fusion_ues.append(np.unique(np.concatenate(K_k_parts)))
            else:
                fusion_ues.append(np.array([], dtype=int))
        return fusion_ues


@task
class CfgEqualLSFDWeights:
    """Use unit statistical-fusion weights at every serving CPU."""

    def __call__(self, exp_g: ExpectedEffectiveUplinkChannels) -> LSFDWeights:
        a: list[np.ndarray] = []
        for exp_g_k in exp_g:
            J_k = exp_g_k.shape[1]
            a.append(np.ones(J_k))
        return a


@task
class CfgOptimalLSFDWeights:
    r"""Compute large-scale fading decoding weights from channel statistics.

    For each UE, solve the statistical interference-plus-noise system using its
    configured :type:`FusionDesignUEs`. The design uses :type:`UEMaxPower`
    because subsequent power control can itself depend on these LSFD weights;
    the weights and operational powers are not jointly iterated.
    """

    def __call__(
        self,
        exp_gg_H: ExpectedEffectiveUplinkChannelOuters,
        exp_g: ExpectedEffectiveUplinkChannels,
        exp_v_H_v: ExpectedCombiningNormSquares,
        p_max: UEMaxPower,
        sigma2_ul: ULNoisePower,
        fusion_ues: FusionDesignUEs,
    ) -> LSFDWeights:
        a: list[np.ndarray] = []
        assert len(exp_g) == len(exp_gg_H) == len(exp_v_H_v) == len(fusion_ues)
        for k, (exp_g_k, exp_gk_gk_H, exp_v_H_v_k, K_k_fuse) in enumerate(
            zip(exp_g, exp_gg_H, exp_v_H_v, fusion_ues)
        ):
            mat_sum = (
                np.sum(
                    p_max[K_k_fuse, np.newaxis, np.newaxis] * exp_gk_gk_H[K_k_fuse],
                    axis=0,
                )
                + np.diag(exp_v_H_v_k) * sigma2_ul
            )
            # lstsq is about 10x slower even when the matrix is regular
            a.append(p_max[k] * np.linalg.solve(mat_sum, exp_g_k[k]))
            # x, _, _, _ = np.linalg.lstsq(mat_sum, exp_g_k[k])
            # a.append(p_max[k] * x)
            assert not np.any(np.isnan(a[-1]))
        return a


@task
class CfgEqualSSFDWeights:
    """Use unit instantaneous-fusion weights at every serving CPU."""

    def __call__(
        self, g_known: KnownEffectiveUplinkChannels, O: NumRealizations
    ) -> SSFDWeights:
        a: list[np.ndarray] = []
        for g_k in g_known:
            J_k = g_k.shape[1]
            a.append(np.ones((J_k, O)))
        return a


@task
class CfgOptimalSSFDWeights:
    r"""Compute realization-dependent MMSE fusion weights.

    The interference covariance combines known instantaneous effective channels,
    conditional unknown-channel correlations, and instantaneous local-combining
    noise amplification. All UEs are assumed to transmit with :type:`UEMaxPower`.
    """

    def __call__(
        self,
        K: NumUEs,
        O: NumRealizations,
        known_g: KnownEffectiveUplinkChannels,
        exp_g_g_H_unknown: ExpectedUnknownEffectiveUplinkChannelOuters,
        p_max: UEMaxPower,
        v_H_v: CombiningNormSquares,
        sigma2_ul: ULNoisePower,
        fusion_ues: FusionDesignUEs,
    ) -> SSFDWeights:
        a: list[np.ndarray] = []
        assert (
            len(known_g) == len(exp_g_g_H_unknown) == len(v_H_v) == len(fusion_ues) == K
        )
        for k, (known_g_k, exp_g_k_g_k_H_unknown, K_k_fuse) in enumerate(
            zip(known_g, exp_g_g_H_unknown, fusion_ues)
        ):
            J_k = known_g_k.shape[1]
            mat_sum = (
                sigma2_ul
                * np.eye(J_k, dtype=complex)[:, :, np.newaxis]
                * v_H_v[k][:, np.newaxis, :]
            )
            for kk in K_k_fuse:
                mat_sum += p_max[kk] * (
                    outer_product_with_self(known_g_k[kk], axis=0)
                    + exp_g_k_g_k_H_unknown[kk]
                )
            a_k = (
                np.sqrt(p_max[k])
                * (
                    np.linalg.solve(
                        np.transpose(mat_sum, (2, 0, 1)),
                        np.transpose(known_g_k[k, :, np.newaxis, :], (2, 0, 1)),
                    )
                )[:, :, 0].T
            )
            a.append(a_k)
        return a


@task
class CfgFullyInstantaneousMMSEFusionWeights:
    r"""Compute an ideal fully instantaneous second MMSE combining stage.

    After local combining, this task treats the scalar outputs of the
    :math:`J_k` serving CPUs as a reduced-dimensional received signal. Assuming
    that every realized effective channel :math:`\mathbf{g}_{ki}^{(o)}` and
    every realized local-combiner norm are available at the fusion CPU, it
    solves

    .. math::

       \left(
         \sum_{i\in\mathcal{K}_k^{\mathrm{fuse}}}
         p_i\mathbf{g}_{ki}^{(o)}
         \mathbf{g}_{ki}^{(o)\mathrm{H}}
         +
         \sigma_{\mathrm{ul}}^2
         \operatorname{diag}\left(
           \left\{\|\mathbf{v}_{jk}^{(o)}\|^2
           :j\in\mathcal{J}_k\right\}
         \right)
       \right)
       \mathbf{a}_k^{(o)}
       =
       \sqrt{p_k}\mathbf{g}_{kk}^{(o)}.

    This is MMSE combining in the effective-channel domain after the first
    local combining stage. Its systems have dimension :math:`J_k`, rather than
    the total number of antennas used by a fully centralized receiver. The
    earlier local combining stage has already discarded spatial information,
    so the result is not generally equivalent to antenna-level centralized
    MMSE combining.

    The task models ideal, unquantized transport. Implementing this scheme
    requires the serving CPUs to make their locally combined signals,
    instantaneous effective channels, and combiner norms available at the
    fusion CPU. Quantization distortion and the resulting fronthaul load are
    not modeled here.

    The powers are the maximum-power design values described in the
    module-level convention.
    """

    def __call__(
        self,
        K: NumUEs,
        O: NumRealizations,
        g: EffectiveUplinkChannels,
        p_max: UEMaxPower,
        v_H_v: CombiningNormSquares,
        sigma2_ul: ULNoisePower,
        fusion_ues: FusionDesignUEs,
    ) -> SSFDWeights:
        a: list[np.ndarray] = []
        assert len(g) == len(v_H_v) == len(fusion_ues) == K
        for k, (g_k, v_k_H_v_k, K_k_fuse) in enumerate(zip(g, v_H_v, fusion_ues)):
            J_k = g_k.shape[1]
            assert v_k_H_v_k.shape == (J_k, O)
            mat_sum = (
                sigma2_ul
                * np.eye(J_k, dtype=complex)[:, :, np.newaxis]
                * v_k_H_v_k[:, np.newaxis, :]
            )
            mat_sum += np.sum(
                p_max[K_k_fuse, np.newaxis, np.newaxis, np.newaxis]
                * outer_product_with_self(g_k[K_k_fuse], axis=1),
                axis=0,
            )
            a_k = (
                np.sqrt(p_max[k])
                * np.linalg.solve(
                    np.transpose(mat_sum, (2, 0, 1)),
                    np.transpose(g_k[k, :, np.newaxis, :], (2, 0, 1)),
                )[:, :, 0].T
            )
            a.append(a_k)
        return a
