r"""Evaluate channel hardening and favorable propagation.

Two complementary favorable-propagation metrics are provided.  The classical
physical-channel metric measures the symmetric, normalized overlap between UE
channel vectors,

.. math::

   \mathrm{FP}_{ki}
   =
   \frac{
     \mathbb{E}\{|\mathbf{h}_k^{\mathrm{H}}\mathbf{h}_i|^2\}
   }{
     \mathbb{E}\{\|\mathbf{h}_k\|^2\}
     \mathbb{E}\{\|\mathbf{h}_i\|^2\}
   }.

The effective metric instead evaluates the interference created after the
configured combining or precoding.  For an effective desired channel
:math:`z_{kk}` and an effective interfering channel :math:`z_{ki}`, it is

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

The effective channels are produced by Mimodium's configured combining or
precoding directions, and contributions from participating CPUs are added
coherently.  The effective metric is directional and includes both channel
overlap and relative effective-channel strength; the classical metric is
symmetric and invariant to separate deterministic scaling of the UE channels.
"""

import numpy as np
from dagreon import task

from ..algorithms import EffectiveDownlinkChannels, EffectiveUplinkChannels
from ..propagation import ChannelRealizations, NumRealizations
from ..scenario import NumAntennasPerAP, NumAPs, NumUEs

#: Symmetric favorable-propagation metric for physical UE channels.
#:
#: Entry ``[k, i]`` is their normalized mean-squared channel overlap.
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
        effective_channels = np.stack([np.sum(g_k, axis=1) for g_k in g])
        desired = np.diagonal(effective_channels, axis1=0, axis2=1).T
        expected_desired = np.mean(desired, axis=-1)
        desired_variances = np.mean(
            np.abs(desired - expected_desired[:, np.newaxis]) ** 2,
            axis=-1,
        )
        denominator = np.abs(expected_desired) ** 2
        values = np.full(denominator.shape, np.nan, dtype=float)
        np.divide(desired_variances, denominator, out=values, where=denominator > 0)
        return values


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
        effective_channels = np.stack(
            [np.stack([np.sum(f_ki, axis=0) for f_ki in f_k]) for f_k in f]
        )
        desired = np.diagonal(effective_channels, axis1=0, axis2=1).T
        expected_desired = np.mean(desired, axis=-1)
        desired_variances = np.mean(
            np.abs(desired - expected_desired[:, np.newaxis]) ** 2,
            axis=-1,
        )
        denominator = np.abs(expected_desired) ** 2
        values = np.full(denominator.shape, np.nan, dtype=float)
        np.divide(desired_variances, denominator, out=values, where=denominator > 0)
        return values


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

    This metric adapts the effective-channel definition in Section 2.6.2 of
    `Demir, Björnson, and Sanguinetti (2021)
    <https://doi.org/10.1561/2000000109>`_ to Mimodium's configured uplink
    combining and CPU fusion.
    """

    def __call__(
        self, g: EffectiveUplinkChannels
    ) -> EffectiveUplinkFavorablePropagationMetrics:
        effective_channels = np.stack([np.sum(g_k, axis=1) for g_k in g])
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

    This metric adapts Definition 2.6.4 of `Demir, Björnson, and Sanguinetti
    (2021) <https://doi.org/10.1561/2000000109>`_ by using the effective
    channels produced by Mimodium's configured downlink precoding.
    """

    def __call__(
        self, f: EffectiveDownlinkChannels
    ) -> EffectiveDownlinkFavorablePropagationMetrics:
        effective_channels = np.stack(
            [np.stack([np.sum(f_ki, axis=0) for f_ki in f_k]) for f_k in f]
        )
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
        expected_desired = np.mean(desired, axis=-1)
        desired_variances = np.mean(
            np.abs(desired - expected_desired[:, np.newaxis]) ** 2,
            axis=-1,
        )
        denominator = np.abs(expected_desired) ** 2
        values = np.full(denominator.shape, np.nan, dtype=float)
        np.divide(desired_variances, denominator, out=values, where=denominator > 0)
        return values


@task
class ComputeFavorablePropagation:
    r"""Compute classical favorable propagation between physical UE channels.

    Entry ``[k, i]`` is

    .. math::

       \frac{
         \mathbb{E}\{|\mathbf{h}_k^{\mathrm{H}}\mathbf{h}_i|^2\}
       }{
         \mathbb{E}\{\|\mathbf{h}_k\|^2\}
         \mathbb{E}\{\|\mathbf{h}_i\|^2\}
       }.

    This is a finite-dimensional, power-normalized measure of the pairwise
    channel orthogonality used to define favorable propagation by `Ngo,
    Larsson, and Marzetta (2014) <https://arxiv.org/abs/1403.3461>`_.  The
    symmetric normalization also corresponds to the finite-dimensional metric
    in Section 2.5.2 of `Björnson, Hoydis, and Sanguinetti (2017)
    <https://doi.org/10.1561/2000000093>`_.
    """

    def __call__(self, h: ChannelRealizations) -> FavorablePropagationMetrics:
        channel_inner_products = np.einsum(
            "klno,ilno->kio",
            np.conj(h),
            h,
        )
        channel_powers = np.real(
            np.mean(
                np.diagonal(channel_inner_products, axis1=0, axis2=1),
                axis=0,
            )
        )
        denominator = channel_powers[:, np.newaxis] * channel_powers[np.newaxis, :]
        second_moments = np.mean(np.abs(channel_inner_products) ** 2, axis=-1)
        values = np.full(second_moments.shape, np.nan, dtype=float)
        np.divide(second_moments, denominator, out=values, where=denominator > 0)
        return values
