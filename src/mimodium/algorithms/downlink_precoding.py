r"""Configure downlink precoding directions and compute effective channels.

Notation
--------
Let :math:`\mathcal{J}_i` contain the CPUs serving UE :math:`i`, and let
:math:`\mathcal{L}_{ji}` contain the APs connected to CPU :math:`j` that serve
UE :math:`i`. The channels from receiving UE :math:`k` to those APs are
concatenated as

.. math::

   \mathbf{h}_{k,\mathcal{L}_{ji}}^{(o)}
   =
   \operatorname{col}\left(
     \left\{\mathbf{h}_{kl}^{(o)}:
     l\in\mathcal{L}_{ji}\right\}
   \right).

For CPU :math:`j\in\mathcal{J}_i`, the downlink precoding direction for UE
:math:`i` is :math:`\mathbf{w}_{ji}^{(o)}`. It has the same antenna ordering
as :math:`\mathbf{h}_{k,\mathcal{L}_{ji}}^{(o)}`. The resulting effective
channel at receiving UE :math:`k` from stream :math:`i` is

.. math::

   f_{ki,j}^{(o)}
   =
   \mathbf{h}_{k,\mathcal{L}_{ji}}^{(o)\mathrm{H}}
   \mathbf{w}_{ji}^{(o)}.

Collecting the contributions from all CPUs serving UE :math:`i` gives

.. math::

   \mathbf{f}_{ki}^{(o)}
   =
   \operatorname{col}\left(
     \left\{f_{ki,j}^{(o)}:j\in\mathcal{J}_i\right\}
   \right).

Thus, ``f[k][i]`` stores :math:`\mathbf{f}_{ki}^{(o)}` across all channel
realizations. Its CPU dimension is determined by the transmitted stream
:math:`i`, not by the receiving UE :math:`k`.

Uplink-based precoding directions
---------------------------------
In a reciprocal channel, an uplink combining vector can be reused as a
downlink precoding direction. :class:`CfgUplinkDirectionPrecoding` normalizes
each UE--CPU pair according to

.. math::

   \mathbf{w}_{ji}^{(o)}
   =
   \frac{\mathbf{v}_{ji}^{(o)}}
   {\sqrt{\mathbb{E}\{\|\mathbf{v}_{ji}\|^2\}}},

so that
:math:`\mathbb{E}\{\|\mathbf{w}_{ji}\|^2\}=1`. Precoding vectors contain only
these normalized spatial directions. Downlink transmit powers are configured
and applied separately.

The effective-channel moments required for downlink use-and-then-forget bounds
are :math:`\mathbb{E}\{\mathbf{f}_{ki}\mathbf{f}_{ki}^{\mathrm{H}}\}` for
every receiver--stream pair and
:math:`\mathbb{E}\{\mathbf{f}_{kk}\}` for each desired stream.

:class:`ComputeExpectedPrecodingNormSquares` partitions every normalized
UE--CPU precoder into its AP-sized blocks and computes the expected squared
norm of each block. These values describe how the unit expected precoder norm
is distributed across the participating APs and are used by downlink power
control.
"""

import numpy as np
from dagreon import task

from ..array_ops import concatenate_all_channels_of_aps, outer_product_with_self
from ..propagation import ChannelRealizations, NumRealizations
from ..scenario import FronthaulLinks, NumAntennasPerAP, NumAPs, NumUEs
from .access import (
    UsedWirelessLinks,
    get_set_of_aps_connected_to_cpu_j_serving_ue_k,
    get_set_of_cpus_serving_ue_k,
)
from .uplink_combining import CombiningVectors

#: Downlink precoding directions. Entry ``w[i]`` contains one vector
#: :math:`\mathbf{w}_{ji}^{(o)}` for every CPU :math:`j\in\mathcal{J}_i`.
#:
#: :shape: ``(N |L_ji|, O)``
#: :dtype: ``complex``
type PrecodingVectors = list[list[np.ndarray]]
#: Expected squared norm of each AP block of the downlink precoders. Entries on
#: unused UE--AP links are zero.
#:
#: :shape: ``(K, L)``
#: :dtype: ``float``
type ExpectedPrecodingNormSquares = np.ndarray
#: Effective downlink channels. Entry ``f[k][i]`` contains
#: :math:`\mathbf{f}_{ki}^{(o)}` across the CPUs serving stream :math:`i`.
#:
#: :shape: ``(|J_i|, O)``
#: :dtype: ``complex``
type EffectiveDownlinkChannels = list[list[np.ndarray]]
#: Expected outer products
#: :math:`\mathbb{E}\{\mathbf{f}_{ki}\mathbf{f}_{ki}^{\mathrm{H}}\}`.
#:
#: :shape: ``(|J_i|, |J_i|)``
#: :dtype: ``complex``
type ExpectedEffectiveDownlinkChannelOuters = list[list[np.ndarray]]
#: Expected desired effective downlink channel
#: :math:`\mathbb{E}\{\mathbf{f}_{kk}\}` for each UE.
#:
#: :shape: ``(|J_k|,)``
#: :dtype: ``complex``
type ExpectedDesiredEffectiveDownlinkChannels = list[np.ndarray]


@task
class CfgUplinkDirectionPrecoding:
    """Normalize uplink combining vectors for use as downlink precoding directions.

    Every UE--CPU precoder is scaled to have unit expected squared norm over channel
    realizations.
    """

    # see e.g. mMIMO Networks: Spectral, Energy and Hardware Efficiency, p. 318
    def __call__(self, v: CombiningVectors) -> PrecodingVectors:
        w: list[list[np.ndarray]] = []
        # the precoding vectors should be normalized to unit norm per CPU per UE
        for v_k in v:
            w_k: list[np.ndarray] = []
            for v_jk in v_k:
                exp_norm_squared = np.mean(np.real(np.vecdot(v_jk.T, v_jk.T)))
                assert exp_norm_squared > 0, (
                    "Combining vectors must have nonzero expected squared norm."
                )
                w_k.append(v_jk / np.sqrt(exp_norm_squared))
            w.append(w_k)
        return w


@task
class ComputeExpectedPrecodingNormSquares:
    r"""Compute each AP's expected share of the normalized precoder power.

    For :math:`l\in\mathcal{L}_{ji}`, compute

    .. math::

       \eta_{il}
       =
       \mathbb{E}\left\{\|\mathbf{w}_{ji,l}\|^2\right\}.

    The shares sum to one over :math:`\mathcal{L}_{ji}` for every UE--CPU pair.
    Consequently, their sum over all APs is :math:`|\mathcal{J}_i|`.
    """

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        N: NumAntennasPerAP,
        O: NumRealizations,
        w: PrecodingVectors,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> ExpectedPrecodingNormSquares:
        exp_w_H_w = np.zeros((K, L), dtype=float)
        for i, w_i in enumerate(w):
            J_i_set = get_set_of_cpus_serving_ue_k(i, used, fronthaul)
            for j, w_ji in zip(J_i_set, w_i):
                L_ji_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, i, used, fronthaul
                )
                exp_w_ji_H_w_ji = np.mean(
                    np.real(
                        np.vecdot(
                            w_ji.T.reshape((O, len(L_ji_set), N)),
                            w_ji.T.reshape((O, len(L_ji_set), N)),
                        ),
                    ),
                    axis=0,
                )
                assert np.isclose(np.sum(exp_w_ji_H_w_ji), 1), (
                    "Precoding vector should have unit expected squared norm."
                )
                for l, exp_w_jil_H_w_jil in zip(L_ji_set, exp_w_ji_H_w_ji):
                    exp_w_H_w[i, l] = exp_w_jil_H_w_jil
            assert np.isclose(np.sum(exp_w_H_w[i]), len(J_i_set))
        return exp_w_H_w


@task
class ComputeEffectiveDLChannels:
    r"""Compute effective downlink channels after precoding.

    For receiving UE :math:`k`, precoded stream :math:`i`, and CPU :math:`j` serving
    that stream, compute :math:`f_{ki,j}^{(o)} =
    \mathbf{h}_{kj}^{\mathrm{H}}\mathbf{w}_{ji}^{(o)}`.
    """

    def __call__(
        self,
        K: NumUEs,
        O: NumRealizations,
        h: ChannelRealizations,
        w: PrecodingVectors,
        used: UsedWirelessLinks,
        fronthaul: FronthaulLinks,
    ) -> EffectiveDownlinkChannels:
        f: list[list[np.ndarray]] = [[] for _ in range(K)]
        for i in range(K):
            J_i_set = get_set_of_cpus_serving_ue_k(i, used, fronthaul)
            f_all_i = np.zeros((K, len(J_i_set), O), dtype=complex)
            for j_index, (j, w_ji) in enumerate(zip(J_i_set, w[i])):
                L_ji_set = get_set_of_aps_connected_to_cpu_j_serving_ue_k(
                    j, i, used, fronthaul
                )
                h_ji = concatenate_all_channels_of_aps(h, L_ji_set)
                f_all_i[:, j_index] = np.vecdot(np.transpose(h_ji, (0, 2, 1)), w_ji.T)
            for k in range(K):
                f[k].append(f_all_i[k])
        return f


@task
class ComputeExpectedDesiredEffectiveDLChannels:
    """Average each UE's desired effective downlink channel over realizations."""

    def __call__(
        self, f: EffectiveDownlinkChannels
    ) -> ExpectedDesiredEffectiveDownlinkChannels:
        exp_f: list[np.ndarray] = []
        for k, f_k in enumerate(f):
            exp_f.append(np.mean(f_k[k], axis=1))
        return exp_f


@task
class ComputeExpectedEffectiveDLChannelOuters:
    r"""Compute :math:`\mathbb{E}\{\mathbf{f}_{ki}\mathbf{f}_{ki}^{\mathrm{H}}\}`
    over channel realizations."""

    def __call__(
        self, f: EffectiveDownlinkChannels
    ) -> ExpectedEffectiveDownlinkChannelOuters:
        exp_f_f_H: list[list[np.ndarray]] = []
        for f_k in f:
            exp_f_f_H_k: list[np.ndarray] = []
            for f_k_kk in f_k:
                exp_f_f_H_k.append(np.mean(outer_product_with_self(f_k_kk.T), axis=0))
            exp_f_f_H.append(exp_f_f_H_k)
        return exp_f_f_H
