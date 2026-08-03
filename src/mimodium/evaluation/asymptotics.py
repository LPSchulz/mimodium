r"""Evaluate channel hardening and favorable propagation.

The definitions follow Section 2.6 of Demir, Björnson, and Sanguinetti,
*Foundations of User-Centric Cell-Free Massive MIMO* (2021).  For an effective
desired channel :math:`z_{kk}` and an effective interfering channel
:math:`z_{ki}`, the finite-dimensional quantities corresponding to the
definitions in the book are

.. math::

   \mathrm{CH}_k
   =
   \mathbb{E}\left\{
     \left|
       \frac{z_{kk}}{\mathbb{E}\{z_{kk}\}} - 1
     \right|^2
   \right\},

and

.. math::

   \mathrm{FP}_{ki}
   =
   \mathbb{E}\left\{
     \left|
       \frac{z_{ki}}{\mathbb{E}\{z_{kk}\}}
     \right|^2
   \right\}.

Small values indicate a high degree of channel hardening or favorable
propagation, respectively.  Only the off-diagonal entries of a favorable-
propagation matrix compare distinct UEs; entry :math:`[k,k]` equals
:math:`1+\mathrm{CH}_k`.  A metric is ``nan`` when the mean desired effective
channel is zero, since the normalization is then undefined.

Unlike the book's introductory derivation, the effective channels used here
are produced by Mimodium's configured combining/precoding directions.  The
contributions from the CPUs participating in a detector or transmitted stream
are added coherently.  Consequently, selecting maximum-ratio processing
recovers the classical quantities, while interference-suppressing processing
measures the properties of the channel that the selected algorithm actually
creates.
"""

import numpy as np
from dagreon import task

from ..algorithms import EffectiveDownlinkChannels, EffectiveUplinkChannels
from ..propagation import ChannelRealizations, NumRealizations
from ..scenario import NumAntennasPerAP, NumAPs, NumUEs

#: Pairwise favorable-propagation metric for physical channels.
#:
#: Entry ``[k, i]`` is UE :math:`k`'s metric with respect to UE :math:`i`.
#: The diagonal is the normalized second moment of the desired channel and is
#: therefore equal to one plus the corresponding channel-hardening metric.
#:
#: :shape: ``(K, K)``
#: :dtype: ``float``
type FavorablePropagationMetrics = np.ndarray
#: Channel-hardening metric for physical channels.
#:
#: :shape: ``(K,)``
#: :dtype: ``float``
type ChannelHardeningMetrics = np.ndarray
#: Channel-hardening metric after configured uplink combining.
#:
#: :shape: ``(K,)``
#: :dtype: ``float``
type EffectiveUplinkChannelHardeningMetrics = np.ndarray
#: Channel-hardening metric after configured downlink precoding.
#:
#: :shape: ``(K,)``
#: :dtype: ``float``
type EffectiveDownlinkChannelHardeningMetrics = np.ndarray
#: Pairwise favorable-propagation metric after configured uplink combining.
#:
#: :shape: ``(K, K)``
#: :dtype: ``float``
type EffectiveUplinkFavorablePropagationMetrics = np.ndarray
#: Pairwise favorable-propagation metric after configured downlink precoding.
#:
#: :shape: ``(K, K)``
#: :dtype: ``float``
type EffectiveDownlinkFavorablePropagationMetrics = np.ndarray


def _compute_channel_hardening(
    desired_effective_channels: np.ndarray,
) -> np.ndarray:
    """Compute normalized desired-channel variances over the last axis."""
    expected_desired = np.mean(desired_effective_channels, axis=-1)
    desired_variances = np.mean(
        np.abs(desired_effective_channels - expected_desired[..., np.newaxis]) ** 2,
        axis=-1,
    )
    denominator = np.abs(expected_desired) ** 2
    values = np.full(denominator.shape, np.nan, dtype=float)
    np.divide(desired_variances, denominator, out=values, where=denominator > 0)
    return values


def _compute_favorable_propagation(
    effective_channels: np.ndarray,
) -> np.ndarray:
    """Compute normalized effective-channel second moments over the last axis."""
    expected_desired = np.mean(
        np.diagonal(effective_channels, axis1=0, axis2=1),
        axis=0,
    )
    denominator = np.abs(expected_desired) ** 2
    second_moments = np.mean(np.abs(effective_channels) ** 2, axis=-1)
    values = np.full(second_moments.shape, np.nan, dtype=float)
    np.divide(
        second_moments,
        denominator[:, np.newaxis],
        out=values,
        where=denominator[:, np.newaxis] > 0,
    )
    return values


def _sum_effective_uplink_channels(g: EffectiveUplinkChannels) -> np.ndarray:
    """Coherently add serving-CPU contributions into shape ``(K, K, O)``."""
    return np.stack([np.sum(g_k, axis=1) for g_k in g])


def _sum_effective_downlink_channels(f: EffectiveDownlinkChannels) -> np.ndarray:
    """Coherently add serving-CPU contributions into shape ``(K, K, O)``."""
    return np.stack([np.stack([np.sum(f_ki, axis=0) for f_ki in f_k]) for f_k in f])


@task
class ComputeEffectiveUplinkChannelHardening:
    r"""Compute hardening after the configured uplink combining.

    For detector UE :math:`k`, the desired effective scalar channel is

    .. math::

       z_{kk}^{(o)} = \sum_{j\in\mathcal{J}_k} g_{kk,j}^{(o)}.

    The returned value is
    :math:`\mathbb{V}\{z_{kk}\}/|\mathbb{E}\{z_{kk}\}|^2`.
    """

    def __call__(
        self, g: EffectiveUplinkChannels
    ) -> EffectiveUplinkChannelHardeningMetrics:
        effective_channels = _sum_effective_uplink_channels(g)
        desired = np.diagonal(effective_channels, axis1=0, axis2=1).T
        return _compute_channel_hardening(desired)


@task
class ComputeEffectiveDownlinkChannelHardening:
    r"""Compute hardening after the configured downlink precoding.

    For receiving UE :math:`k`, the desired effective scalar channel is

    .. math::

       z_{kk}^{(o)} = \sum_{j\in\mathcal{J}_k} f_{kk,j}^{(o)}.

    The returned value is
    :math:`\mathbb{V}\{z_{kk}\}/|\mathbb{E}\{z_{kk}\}|^2`.
    """

    def __call__(
        self, f: EffectiveDownlinkChannels
    ) -> EffectiveDownlinkChannelHardeningMetrics:
        effective_channels = _sum_effective_downlink_channels(f)
        desired = np.diagonal(effective_channels, axis1=0, axis2=1).T
        return _compute_channel_hardening(desired)


@task
class ComputeEffectiveUplinkFavorablePropagation:
    r"""Compute favorable propagation after configured uplink combining.

    Entry ``[k, i]`` is

    .. math::

       \frac{
         \mathbb{E}\left\{
           \left|\sum_{j\in\mathcal{J}_k}g_{ki,j}\right|^2
         \right\}
       }{
         \left|
           \mathbb{E}\left\{
             \sum_{j\in\mathcal{J}_k}g_{kk,j}
           \right\}
         \right|^2
       }.

    The off-diagonal entries quantify favorable propagation.  A second moment,
    rather than only a variance, is used so that coherent interference created
    by channel estimation or the selected combiner is not discarded.
    """

    def __call__(
        self, g: EffectiveUplinkChannels
    ) -> EffectiveUplinkFavorablePropagationMetrics:
        return _compute_favorable_propagation(_sum_effective_uplink_channels(g))


@task
class ComputeEffectiveDownlinkFavorablePropagation:
    r"""Compute favorable propagation after configured downlink precoding.

    Entry ``[k, i]`` is

    .. math::

       \frac{
         \mathbb{E}\left\{
           \left|\sum_{j\in\mathcal{J}_i}f_{ki,j}\right|^2
         \right\}
       }{
         \left|
           \mathbb{E}\left\{
             \sum_{j\in\mathcal{J}_k}f_{kk,j}
           \right\}
         \right|^2
       }.

    The off-diagonal entries quantify favorable propagation.  The CPU set in
    the numerator belongs to transmitted stream :math:`i`, as encoded by
    :type:`EffectiveDownlinkChannels`.
    """

    def __call__(
        self, f: EffectiveDownlinkChannels
    ) -> EffectiveDownlinkFavorablePropagationMetrics:
        return _compute_favorable_propagation(_sum_effective_downlink_channels(f))


@task
class ComputeChannelHardening:
    r"""Compute the classical physical-channel hardening reference.

    This treats all AP antennas as one array and applies perfect-CSI
    maximum-ratio processing, so
    :math:`z_{kk}=\sum_l\|\mathbf{h}_{kl}\|^2`.
    """

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        N: NumAntennasPerAP,
        O: NumRealizations,
        h: ChannelRealizations,
    ) -> ChannelHardeningMetrics:
        h_flat = h.reshape(K, L * N, O)
        desired = np.sum(np.abs(h_flat) ** 2, axis=1)
        return _compute_channel_hardening(desired)


@task
class ComputeFavorablePropagation:
    r"""Compute the classical perfect-CSI MR favorable-propagation reference.

    Entry ``[k, i]`` uses
    :math:`z_{ki}=\sum_l\mathbf{h}_{kl}^{\mathrm{H}}\mathbf{h}_{il}`.
    Prefer :class:`ComputeEffectiveUplinkFavorablePropagation` or
    :class:`ComputeEffectiveDownlinkFavorablePropagation` when evaluating a
    configured combining scheme.
    """

    def __call__(self, h: ChannelRealizations) -> FavorablePropagationMetrics:
        effective_channels = np.einsum(
            "klno,ilno->kio",
            np.conj(h),
            h,
        )
        return _compute_favorable_propagation(effective_channels)
