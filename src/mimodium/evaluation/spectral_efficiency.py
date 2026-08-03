r"""Evaluate uplink and downlink spectral-efficiency bounds.

Notation
--------
Let :math:`\mathcal{J}_k` contain the CPUs serving UE :math:`k`. CPU
:math:`j\in\mathcal{J}_k` uses the local combining vector
:math:`\mathbf{v}_{jk}^{(o)}` in channel realization :math:`o`. The resulting
effective uplink channel from transmitting UE :math:`i` to the local detector
for UE :math:`k` is

.. math::

   g_{ki,j}^{(o)}
   =
   \mathbf{v}_{jk}^{(o)\mathrm{H}}
   \mathbf{h}_{i,\mathcal{L}_{jk}}^{(o)}.

Collecting the effective channels from the serving CPUs gives
:math:`\mathbf{g}_{ki}^{(o)}`. A fusion vector
:math:`\mathbf{a}_k^{(o)}` produces the scalar effective channel

.. math::

   \gamma_{ki}^{(o)}
   =
   \mathbf{a}_k^{(o)\mathrm{H}}\mathbf{g}_{ki}^{(o)}.

For LSFD, :math:`\mathbf{a}_k` is constant across realizations. For SSFD,
:math:`\mathbf{a}_k^{(o)}` may depend on the available instantaneous
effective-channel information.

The post-combining noise contribution from CPU :math:`j` is proportional to
:math:`\|\mathbf{v}_{jk}^{(o)}\|^2`. Consequently, after fusion its power is

.. math::

   \sigma_{\mathrm{ul}}^2
   \sum_{j\in\mathcal{J}_k}
   |a_{k,j}^{(o)}|^2
   \|\mathbf{v}_{jk}^{(o)}\|^2.

Every uplink bound includes the prelog factor
:math:`\tau_{\mathrm{u}}/\tau_{\mathrm{c}}`; the downlink bound analogously
uses :math:`\tau_{\mathrm{d}}/\tau_{\mathrm{c}}`.

Uplink standard bound
---------------------
Let :math:`\mathcal I_k` contain the instantaneous effective-channel
information available to the detector and define

.. math::

   \mathbf{g}_{ki,\mathrm{known}}^{(o)}
   =
   \mathbb E\left\{
     \left.\mathbf g_{ki}^{(o)}\right|\mathcal I_k
   \right\},
   \qquad
   \mathbf{g}_{ki,\mathrm{unknown}}^{(o)}
   =
   \mathbf g_{ki}^{(o)}
   -
   \mathbf{g}_{ki,\mathrm{known}}^{(o)}.

The conditional residual correlation is

.. math::

   \mathbb E\left\{
     \left.
     \mathbf g_{ki,\mathrm{unknown}}^{(o)}
     \mathbf g_{ki,\mathrm{unknown}}^{(o)\mathrm H}
     \right|
     \mathcal I_k
   \right\}.

For an instantaneously known CPU component, this correlation contains

.. math::

   \mathbf v_{jk}^{(o)\mathrm H}
   \mathbf C_{i,j}
   \mathbf v_{jk}^{(o)}.

The conditional standard bound supports both partial instantaneous CSI and
instantaneous CSI from every serving CPU. Its SINR is

.. math::

   \mathrm{SINR}_{k,\mathrm{standard}}^{(o)}
   =
   \frac{
     p_k
     \left|
       \mathbf a_k^{(o)\mathrm H}
       \mathbf g_{kk,\mathrm{known}}^{(o)}
     \right|^2
   }{
     \sum_{i\ne k}
       p_i
       \left|
         \mathbf a_k^{(o)\mathrm H}
         \mathbf g_{ki,\mathrm{known}}^{(o)}
       \right|^2
     +
     \sum_i p_i
       \mathbf a_k^{(o)\mathrm H}
       \mathbb E\left\{
         \left.
         \mathbf g_{ki,\mathrm{unknown}}^{(o)}
         \mathbf g_{ki,\mathrm{unknown}}^{(o)\mathrm H}
         \right|
         \mathcal I_k
       \right\}
       \mathbf a_k^{(o)}
     +\sigma_{\mathrm{ul}}^2
      \sum_j
        |a_{k,j}^{(o)}|^2
        \|\mathbf v_{jk}^{(o)}\|^2
   }.

The logarithm is averaged over the channel realizations before applying the
uplink prelog factor.

Uplink genie-aided bound
------------------------
The genie-aided bound evaluates every realization using the true effective
channels. With LSFD vector :math:`\mathbf{a}_k`, its SINR is

.. math::

   \mathrm{SINR}_{k,\mathrm{genie}}^{(o)}
   =
   \frac{
     p_k|\mathbf{a}_k^{\mathrm{H}}\mathbf{g}_{kk}^{(o)}|^2
   }{
     \sum_{i\ne k}
       p_i|\mathbf{a}_k^{\mathrm{H}}\mathbf{g}_{ki}^{(o)}|^2
     +
     \sigma_{\mathrm{ul}}^2
     \sum_{j\in\mathcal{J}_k}
       |a_{k,j}|^2\|\mathbf{v}_{jk}^{(o)}\|^2
   }.

The bound averages :math:`\log_2(1+\mathrm{SINR})` over the channel
realizations and applies the uplink prelog factor.

Uplink use-and-then-forget bound
--------------------------------
Define the deterministic effective-channel moments

.. math::

   \boldsymbol{\mu}_{ki}
   =
   \mathbb{E}\{\mathbf{g}_{ki}\},
   \qquad
   \mathbf{R}_{ki}
   =
   \mathbb{E}\{
     \mathbf{g}_{ki}\mathbf{g}_{ki}^{\mathrm{H}}
   \},

and the statistical local-combining noise matrix

.. math::

   \mathbf{D}_k
   =
   \operatorname{diag}\left(
     \left\{
       \mathbb{E}\{\|\mathbf{v}_{jk}\|^2\}
       :j\in\mathcal{J}_k
     \right\}
   \right).

The use-and-then-forget SINR is

.. math::

   \mathrm{SINR}_{k,\mathrm{UatF}}
   =
   \frac{
     p_k|\mathbf{a}_k^{\mathrm{H}}\boldsymbol{\mu}_{kk}|^2
   }{
     \sum_{i=1}^{K}
       p_i\mathbf{a}_k^{\mathrm{H}}\mathbf{R}_{ki}\mathbf{a}_k
     -
     p_k|\mathbf{a}_k^{\mathrm{H}}\boldsymbol{\mu}_{kk}|^2
     +
     \sigma_{\mathrm{ul}}^2
       \mathbf{a}_k^{\mathrm{H}}\mathbf{D}_k\mathbf{a}_k
   }.

Unlike the instantaneous bounds, the logarithm is evaluated once from these
deterministic moments.

Downlink use-and-then-forget bound
----------------------------------
For stream :math:`i`, let :math:`\mathcal{J}_i` contain its serving CPUs and
let :math:`\mathbf{f}_{ki}` collect the effective downlink channels from those
CPUs to receiving UE :math:`k`. Define

.. math::

   \boldsymbol{\nu}_{kk}
   =
   \mathbb{E}\{\mathbf{f}_{kk}\},
   \qquad
   \mathbf{Q}_{ki}
   =
   \mathbb{E}\{
     \mathbf{f}_{ki}\mathbf{f}_{ki}^{\mathrm{H}}
   \}.

The AP-level downlink powers are aggregated per serving CPU:

.. math::

   q_{ij}
   =
   \sum_{l\in\mathcal{L}_j}\rho_{il},
   \qquad j\in\mathcal{J}_i,

and :math:`\sqrt{\mathbf{q}_i}` contains their nonnegative square roots. Since
unused UE--AP power entries are zero, summing over all APs connected to CPU
:math:`j` is equivalent to summing over the APs transmitting stream
:math:`i`. The downlink UatF SINR is

.. math::

   \mathrm{SINR}_{k,\mathrm{dl}}
   =
   \frac{
     |\sqrt{\mathbf{q}_k}^{\mathrm{T}}\boldsymbol{\nu}_{kk}|^2
   }{
     \sum_{i=1}^{K}
       \sqrt{\mathbf{q}_i}^{\mathrm{T}}
       \mathbf{Q}_{ki}
       \sqrt{\mathbf{q}_i}
     -
     |\sqrt{\mathbf{q}_k}^{\mathrm{T}}\boldsymbol{\nu}_{kk}|^2
     +
     \sigma_{\mathrm{dl}}^2
   }.

All spectral-efficiency configurations return zero for UEs without a serving
CPU.
"""

import numpy as np
from dagreon import task

from ..algorithms import (
    CombiningNormSquares,
    DownlinkPowers,
    EffectiveUplinkChannels,
    ExpectedCombiningNormSquares,
    ExpectedDesiredEffectiveDownlinkChannels,
    ExpectedEffectiveDownlinkChannelOuters,
    ExpectedEffectiveUplinkChannelOuters,
    ExpectedEffectiveUplinkChannels,
    ExpectedUnknownEffectiveUplinkChannelOuters,
    KnownEffectiveUplinkChannels,
    LSFDWeights,
    SSFDWeights,
    UplinkPowers,
    UsedWirelessLinks,
)
from ..algorithms.access import (
    get_set_of_cpus_serving_ue_k,
)
from ..propagation import (
    DLNoisePower,
    NumCoherenceSymbols,
    NumDownlinkSymbols,
    NumUplinkSymbols,
    ULNoisePower,
)
from ..scenario import FronthaulLinks, NumUEs
from ..scenario.fronthaul import get_set_of_aps_connected_to_cpu_j

#: Achievable uplink spectral efficiency of every UE in bit/s/Hz.
#:
#: :shape: ``(K,)``
#: :dtype: ``float``
type UplinkSpectralEfficiencies = np.ndarray  # shape=(K,), dtype=float
#: Achievable downlink spectral efficiency of every UE in bit/s/Hz.
#:
#: :shape: ``(K,)``
#: :dtype: ``float``
type DownlinkSpectralEfficiencies = np.ndarray  # shape=(K,), dtype=float


@task
class CfgUplinkGenieAidedBound:
    """Evaluate the uplink bound using the true instantaneous effective channels.

    LSFD weights fuse the local detector outputs, and the logarithmic rate is
    averaged over channel realizations.
    """

    def __call__(
        self,
        K: NumUEs,
        g: EffectiveUplinkChannels,
        a: LSFDWeights,
        p: UplinkPowers,
        v_H_v: CombiningNormSquares,
        sigma2_ul: ULNoisePower,
        tau_u: NumUplinkSymbols,
        tau_c: NumCoherenceSymbols,
    ) -> UplinkSpectralEfficiencies:
        SEs = np.zeros(K)
        assert len(g) == len(a) == len(v_H_v) == K
        for k, (g_k, a_k, v_k_H_v_k) in enumerate(zip(g, a, v_H_v)):
            # avoid computation for UEs that are not served to avoid dividing by 0
            if a_k.size == 0:
                SEs[k] = 0.0
                continue
            a_k_H = np.conj(np.transpose(a_k))
            desired = (
                p[np.newaxis, k] * np.abs(np.conjugate(np.transpose(g_k[k])) @ a_k) ** 2
            )
            interference = np.sum(
                p[np.newaxis, :] * np.abs(a_k_H @ np.transpose(g_k, (2, 1, 0))) ** 2,
                axis=1,
            )
            noise = sigma2_ul * np.sum(
                np.transpose(v_k_H_v_k) * (np.abs(a_k) ** 2)[np.newaxis, :], axis=1
            )
            SEs[k] += (tau_u / tau_c) * np.mean(
                np.log2(1 + desired / (interference - desired + noise))
            )
        assert np.all(SEs >= 0)
        return SEs


@task
class CfgUplinkUatfBound:
    """Evaluate the uplink use-and-then-forget bound from channel moments.

    The effective-channel means provide the coherent desired signal, while their
    second moments contain interference and channel uncertainty.
    """

    def __call__(
        self,
        K: NumUEs,
        exp_g_g_H: ExpectedEffectiveUplinkChannelOuters,
        exp_g: ExpectedEffectiveUplinkChannels,
        exp_v_H_v: ExpectedCombiningNormSquares,
        p: UplinkPowers,
        a: LSFDWeights,
        sigma2_ul: ULNoisePower,
        tau_u: NumUplinkSymbols,
        tau_c: NumCoherenceSymbols,
    ) -> UplinkSpectralEfficiencies:
        SEs = np.zeros(K)
        for k, (exp_gk_gk_H, exp_g_k, exp_v_H_v_k, a_k) in enumerate(
            zip(exp_g_g_H, exp_g, exp_v_H_v, a)
        ):
            # avoid computation for UEs that are not served to avoid dividing by 0
            if a_k.size == 0:
                SEs[k] = 0.0
                continue
            a_k_H = np.conj(np.transpose(a_k))
            desired = p[k] * np.abs(a_k_H @ exp_g_k[k]) ** 2
            interference = np.real(np.sum(p * (a_k_H @ exp_gk_gk_H @ a_k)))
            noise = sigma2_ul * np.transpose(np.abs(a_k) ** 2) @ exp_v_H_v_k
            SEs[k] = (tau_u / tau_c) * np.log2(
                1 + desired / (interference - desired + noise)
            )
        assert np.all(SEs >= 0)
        return SEs


@task
class CfgUplinkStandardBound:
    r"""Evaluate the standard uplink bound conditioned on the available CSI.

    The known effective channels and SSFD weights may vary between realizations.
    The matching conditional unknown-channel correlations retain this realization
    dependence; instantaneously known components therefore contribute
    :math:`\mathbf{v}_{kj}^{\mathrm H}\mathbf{C}_{k'j}\mathbf{v}_{kj}`.
    """

    def __call__(
        self,
        K: NumUEs,
        g_known: KnownEffectiveUplinkChannels,
        exp_g_g_H_unknown: ExpectedUnknownEffectiveUplinkChannelOuters,
        p: UplinkPowers,
        a: SSFDWeights,
        sigma2_ul: ULNoisePower,
        v_H_v: CombiningNormSquares,
        tau_u: NumUplinkSymbols,
        tau_c: NumCoherenceSymbols,
    ) -> UplinkSpectralEfficiencies:
        SEs = np.zeros(K)
        for k, (
            g_k_known,
            exp_g_k_g_k_H_unknown,
            a_k,
            v_k_H_v_k,
        ) in enumerate(zip(g_known, exp_g_g_H_unknown, a, v_H_v)):
            if a_k.size == 0:
                SEs[k] = 0.0
                continue
            known_effective_channel = np.vecdot(
                a_k.T, np.transpose(g_k_known, (0, 2, 1))
            )
            known_received_power = np.sum(
                p[:, np.newaxis] * np.abs(known_effective_channel) ** 2, axis=0
            )
            desired = p[k] * np.abs(known_effective_channel[k]) ** 2
            exp_mat_sum = np.sum(
                p[:, np.newaxis, np.newaxis, np.newaxis] * exp_g_k_g_k_H_unknown,
                axis=0,
            )
            uncertainty_times_weights = np.matvec(
                np.transpose(exp_mat_sum, (2, 0, 1)),
                a_k.T,
            )
            combining_uncertainty = np.real(np.vecdot(a_k.T, uncertainty_times_weights))
            interference = known_received_power - desired + combining_uncertainty
            noise = sigma2_ul * np.vecdot(np.abs(a_k.T) ** 2, v_k_H_v_k.T)
            instant_sinrs_k = desired / (interference + noise)
            SEs[k] = (tau_u / tau_c) * np.mean(np.log2(1 + instant_sinrs_k))
        assert np.all(SEs >= 0), f"SEs: {SEs}"
        return SEs


@task
class CfgDLUatFBound:
    """Evaluate the downlink use-and-then-forget bound from channel moments.

    AP-level powers are first aggregated into the CPU ordering used by the
    effective downlink channels.
    """

    def __call__(
        self,
        K: NumUEs,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
        tau_d: NumDownlinkSymbols,
        tau_c: NumCoherenceSymbols,
        exp_f: ExpectedDesiredEffectiveDownlinkChannels,
        exp_f_f_H: ExpectedEffectiveDownlinkChannelOuters,
        rho: DownlinkPowers,
        sigma2_dl: DLNoisePower,
    ) -> DownlinkSpectralEfficiencies:
        # bring the powers into format of effective channels:
        rho_list: list[np.ndarray] = []  # shape = K x J_k (like LSFD weights)
        for k in range(K):
            J_k_set = get_set_of_cpus_serving_ue_k(k, used, fronthaul)
            rho_k = np.zeros(len(J_k_set), dtype=float)
            for j_ind, j in enumerate(J_k_set):
                L_j = get_set_of_aps_connected_to_cpu_j(j, fronthaul)
                rho_k[j_ind] = np.sum(rho[k][L_j])
            rho_list.append(rho_k)

        SEs = np.zeros(K)
        for k, (exp_fk_fk_H, exp_f_kk) in enumerate(zip(exp_f_f_H, exp_f)):
            desired = np.abs(np.sqrt(rho_list[k]).T @ exp_f_kk) ** 2
            interference = 0
            for kk in range(K):
                interference += np.real(
                    np.sqrt(rho_list[kk]).T @ exp_fk_fk_H[kk] @ np.sqrt(rho_list[kk])
                )
            noise = sigma2_dl
            sinr = desired / (interference - desired + noise)
            SEs[k] = (tau_d / tau_c) * np.log2(1 + sinr)

        assert np.all(SEs >= 0)
        return SEs
