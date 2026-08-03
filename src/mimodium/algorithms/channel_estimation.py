"""Configure pilot powers and channel estimation at APs."""

import numpy as np
from dagreon import task

from ..array_ops import outer_product_with_self
from ..propagation import (
    ChannelRealizations,
    LargeScaleFadingCoefficients,
    NumPilots,
    NumRealizations,
    SpatialCorrelationMatrices,
    ULNoisePower,
)
from ..rng import Seed, get_rng
from ..scenario import NumAntennasPerAP, NumAPs, NumUEs, PilotMaxPower
from .access import AssignedPilotIDs, EstimatedChannelLinks, MeasuredStatisticLinks

#: Pilot transmit power :math:`\eta_k` of each UE in watts.
#:
#: :shape: ``(K,)``
#: :dtype: ``float``
type PilotPowers = np.ndarray

#: Channel estimates :math:`\hat{\mathbf{h}}_{kl}^{(o)}` for every UE, AP, and
#: realization.
#:
#: :shape: ``(K, L, N, O)``
#: :dtype: ``complex``
type ChannelEstimates = np.ndarray
#: Expected channel vectors :math:`\mathbb{E}\{\mathbf{h}_{kl}\}` for every UE and AP.
#:
#: :shape: ``(K, L, N)``
#: :dtype: ``complex``
type ExpectedChannels = np.ndarray
#: Pilot-observation correlation matrices :math:`\mathbf{\Psi}_{tl}` for each pilot
#: and AP.
#:
#: :shape: ``(tau_p, L, N, N)``
#: :dtype: ``complex``
type ReceivedPilotCorrelations = np.ndarray
#: Estimation-error correlation matrices
#: :math:`\mathbf{C}_{kl} = \mathbb{E}\{\tilde{\mathbf{h}}_{kl}
#: \tilde{\mathbf{h}}_{kl}^{\mathrm{H}}\}`.
#:
#: :shape: ``(K, L, N, N)``
#: :dtype: ``complex``
type EstimationErrorCorrelations = np.ndarray
#: Received pilot signals at each AP after correlating with each pilot sequence.
#:
#: :shape: ``(tau_p, L, N, O)``
#: :dtype: ``complex``
type ReceivedSignals = np.ndarray


@task
class CfgAllocateFullPilotPower:
    """Use the maximum configured pilot transmit power for every UE."""

    def __call__(self, K: NumUEs, pilot_max_power: PilotMaxPower) -> PilotPowers:
        return pilot_max_power.copy()


@task
class CfgAnalyticalChannelExpectation:
    r"""Set the analytical expectation of zero-mean Rayleigh channels to zero.

    For every UE :math:`k` and AP :math:`l`,
    :math:`\mathbb{E}\{\mathbf{h}_{kl}\} = \mathbf{0}`.
    """

    def __call__(self, K: NumUEs, L: NumAPs, N: NumAntennasPerAP) -> ExpectedChannels:
        return np.zeros((K, L, N), dtype=complex)


@task
class CfgMeasuredChannelExpectation:
    r"""Estimate expected channel vectors by averaging channel realizations.

    For every UE :math:`k` and AP :math:`l`, estimate

    .. math::

       \widehat{\mathbb{E}\{\mathbf{h}_{kl}\}}
       = \frac{1}{O}\sum_{o=1}^{O}\mathbf{h}_{kl}^{(o)} .
    """

    def __call__(self, h: ChannelRealizations) -> ExpectedChannels:
        return np.mean(h, axis=-1)


@task
class CfgAnalyticalReceivedPilotsCorrelation:
    r"""Compute pilot-observation correlation matrices for MMSE estimation.

    For pilot :math:`t` and AP :math:`l`, compute

    .. math::

       \mathbf{\Psi}_{tl}
       = \tau_p \sum_{i:t_i=t}\eta_i\beta_{il}\mathbf{R}_{il}
         + \sigma_{\mathrm{ul}}^2\mathbf{I}_N .
    """

    def __call__(
        self,
        t: AssignedPilotIDs,
        eta: PilotPowers,
        sigma2_ul: ULNoisePower,
        L: NumAPs,
        N: NumAntennasPerAP,
        tau_p: NumPilots,
        R: SpatialCorrelationMatrices,
        beta: LargeScaleFadingCoefficients,
    ) -> ReceivedPilotCorrelations:
        Psi = np.zeros([tau_p, L, N, N], dtype=complex)
        scaled_R = beta[:, :, np.newaxis, np.newaxis] * R
        for tk in range(tau_p):
            ues_with_pilot_t = np.where(t == tk)[0]
            Psi[tk] = (
                tau_p
                * np.sum(
                    eta[ues_with_pilot_t, np.newaxis, np.newaxis, np.newaxis]
                    * scaled_R[ues_with_pilot_t],
                    axis=0,
                )
                + (sigma2_ul * np.eye(N))[np.newaxis, :, :]
            )
        assert not np.any(np.isnan(Psi))
        return Psi


@task
class CfgMeasuredReceivedPilotsCorrelation:
    r"""Estimate pilot-observation correlations from received pilot signals.

    For pilot :math:`t` and AP :math:`l`, estimate

    .. math::

       \hat{\mathbf{\Psi}}_{tl}
       = \frac{1}{O}\sum_{o=1}^{O}
         \mathbf{y}_{tl}^{(o)}\mathbf{y}_{tl}^{(o)\mathrm{H}} .

    This is the raw sample correlation appropriate for the zero-mean pilot
    observations assumed by the channel-estimation model.
    """

    def __call__(self, y: ReceivedSignals) -> ReceivedPilotCorrelations:
        return np.mean(outer_product_with_self(y, axis=2), axis=-1)


@task
class CfgAnalyticalEstimationErrorCorrelation:
    r"""Compute analytical MMSE estimation-error correlations for Rayleigh fading.

    For estimated Rayleigh links, compute
    :math:`\mathbf{C}_{kl} = \beta_{kl}\mathbf{R}_{kl}
    - \eta_k\tau_p \beta_{kl}\mathbf{R}_{kl}\mathbf{\Psi}_{t_k l}^{-1}
    \beta_{kl}\mathbf{R}_{kl}`. If :code:`is_perfect` is set, estimated links
    have zero estimation error. Unestimated measured links keep their full channel
    covariance as estimation error, while unmeasured links are set to zero.
    """

    is_perfect: bool = False

    def __call__(
        self,
        K: NumUEs,
        etas: PilotPowers,
        tau_p: NumPilots,
        t: AssignedPilotIDs,
        R: SpatialCorrelationMatrices,
        beta: LargeScaleFadingCoefficients,
        Psi: ReceivedPilotCorrelations,
        estimated: EstimatedChannelLinks,
        measured: MeasuredStatisticLinks,
    ) -> EstimationErrorCorrelations:
        C = np.zeros_like(R)
        scaled_R = beta[:, :, np.newaxis, np.newaxis] * R
        if not self.is_perfect:
            for k in range(K):
                C[k] = scaled_R[k] - etas[k] * tau_p * (
                    scaled_R[k] @ np.linalg.solve((Psi[t[k]]), scaled_R[k])
                )
        C[~estimated] = scaled_R[~estimated]
        C[~measured] = 0
        return C


@task
class CfgMeasuredEstimationErrorCorrelation:
    """Estimate error correlations empirically from channel realizations."""

    def __call__(
        self,
        h: ChannelRealizations,
        h_hat: ChannelEstimates,
        measured: MeasuredStatisticLinks,
    ) -> EstimationErrorCorrelations:
        C = np.mean(outer_product_with_self(h - h_hat, axis=2), axis=-1)
        C[~measured] = 0
        return C


@task
class ComputeReceivedSignals:
    r"""Simulate received pilot signals for orthogonal pilot sequences.

    For pilot :math:`t`, the received signal is the sum of channels from all UEs using
    that pilot, scaled by :math:`\sqrt{\eta_k\tau_p}`, plus complex Gaussian uplink
    receiver noise:

    .. math::

       \mathbf{y}_{tl}^{(o)}
       = \sum_{i:t_i=t}\sqrt{\eta_i\tau_p}\,\mathbf{h}_{il}^{(o)}
         + \mathbf{n}_{tl}^{(o)}.

    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`.
    """

    seed_override: int | None = None

    def __call__(
        self,
        L: NumAPs,
        N: NumAntennasPerAP,
        O: NumRealizations,
        t: AssignedPilotIDs,
        etas: PilotPowers,
        h: ChannelRealizations,
        sigma2_ul: ULNoisePower,
        tau_p: NumPilots,
        seed: Seed,
    ) -> ReceivedSignals:
        # we implement formula 4.4 from the book, this assumes that the used pilot
        # sequences are orthogonal
        y = np.zeros([tau_p, L, N, O], dtype=complex)
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        for tk in range(tau_p):
            # for l in range(L):
            ues_with_pilot_t = np.where(t == tk)[0]
            signals = np.sum(
                np.sqrt(
                    etas[ues_with_pilot_t, np.newaxis, np.newaxis, np.newaxis] * tau_p
                )
                * h[ues_with_pilot_t, :],
                axis=0,
            )
            noise = rng.normal(
                0, np.sqrt(sigma2_ul / 2), [L, N, O]
            ) + 1.0j * rng.normal(0, np.sqrt(sigma2_ul / 2), [L, N, O])
            y[tk, :] = signals + noise
        return y


@task
class CfgPerfectChannelEstimation:
    """Use realized channels as perfect estimates on estimated links.

    On measured but unestimated links, use :type:`ExpectedChannels` for every
    realization. Links not marked in :type:`MeasuredStatisticLinks` are set to zero.
    """

    def __call__(
        self,
        h: ChannelRealizations,
        exp_h: ExpectedChannels,
        estimated: EstimatedChannelLinks,
        measured: MeasuredStatisticLinks,
    ) -> ChannelEstimates:
        h_hat = np.zeros_like(h)
        h_hat[measured] = exp_h[measured][..., np.newaxis]
        h_hat[estimated] = h[estimated]
        return h_hat


@task
class CfgPilotBasedMMSEChannelEstimation:
    """Compute pilot-based MMSE channel estimates only on estimated links.

    On measured but unestimated links, use :type:`ExpectedChannels` for every
    realization. Links not marked in :type:`MeasuredStatisticLinks` are set to zero.
    The batched linear solve and matrix products are restricted to links marked in
    :type:`EstimatedChannelLinks`.
    """

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        N: NumAntennasPerAP,
        O: NumRealizations,
        y: ReceivedSignals,
        Psi: ReceivedPilotCorrelations,
        R: SpatialCorrelationMatrices,
        beta: LargeScaleFadingCoefficients,
        etas: PilotPowers,
        tau_p: NumPilots,
        t: AssignedPilotIDs,
        exp_h: ExpectedChannels,
        estimated: EstimatedChannelLinks,
        measured: MeasuredStatisticLinks,
    ) -> ChannelEstimates:
        h_hat = np.zeros((K, L, N, O), dtype=complex)
        h_hat[measured] = exp_h[measured][..., np.newaxis]
        ue_ids, ap_ids = np.nonzero(estimated)
        if ue_ids.size == 0:
            return h_hat

        x_kl = np.linalg.solve(
            Psi[t[ue_ids], ap_ids],
            y[t[ue_ids], ap_ids],
        )
        scaled_R = beta[ue_ids, ap_ids, np.newaxis, np.newaxis] * R[ue_ids, ap_ids]
        h_hat[ue_ids, ap_ids] = np.sqrt(
            etas[ue_ids, np.newaxis, np.newaxis] * tau_p
        ) * (scaled_R @ x_kl)
        assert not np.any(np.isnan(h_hat))
        return h_hat
