"""Configure the channel model including its small-scale realizations."""

import warnings

import numpy as np
from dagreon import task

from ..rng import Seed, factor_correlation_matrix, get_rng
from ..scenario import (
    APAntennaSpacing,
    APArrayOrientations,
    NumAntennasPerAP,
    NumAPs,
    NumUEs,
    UEtoAPAzimuthAngles,
)
from ..warnings import ApplicabilityWarning
from .path_loss import PathLossdB
from .shadow_fading import ShadowFadingdB

#: Number of independent small-scale channel realizations :math:`O`.
type NumRealizations = int
#: Dimensionless large-scale power gains :math:`\beta_{kl}` between every UE
#: :math:`k` and AP :math:`l`, combining :type:`PathLossdB` and
#: :type:`ShadowFadingdB`.
#:
#: :shape: ``(K, L)``
#: :dtype: ``float``
type LargeScaleFadingCoefficients = np.ndarray
#: Antenna-domain spatial correlation matrices :math:`\mathbf{R}_{kl}` between every
#: UE :math:`k` and AP :math:`l`.
#:
#: :shape: ``(K, L, N, N)``
#: :dtype: ``complex``
type SpatialCorrelationMatrices = np.ndarray
#: Small-scale channel vectors :math:`\mathbf{h}_{kl}^{(o)}` between every UE
#: :math:`k` and AP :math:`l` for every realization :math:`o`.
#:
#: :shape: ``(K, L, N, O)``
#: :dtype: ``complex``
type ChannelRealizations = np.ndarray


@task
class CfgNumRealizations:
    """Set the number of independent channel realizations :math:`O` explicitly to
    :code:`num_realizations`."""

    num_realizations: int

    def __post_init__(self):
        if self.num_realizations <= 0:
            raise ValueError("num_realizations must be positive")

    def __call__(self) -> NumRealizations:
        return self.num_realizations


@task
class ComputeLargeScaleFading:
    r"""Combine path loss and shadow fading into linear large-scale power gains.

    For every UE :math:`k` and AP :math:`l`, compute
    :math:`\beta_{kl} = 10^{(-PL_{kl} + SF_{kl})/10}`, where :math:`PL_{kl}` and
    :math:`SF_{kl}` are given in decibels. The resulting :math:`\beta_{kl}` is a
    dimensionless power ratio. Thus, a positive shadow-fading value increases the
    channel gain.
    """

    def __call__(
        self, path_loss_db: PathLossdB, shadow_fading_db: ShadowFadingdB
    ) -> LargeScaleFadingCoefficients:
        return 10 ** ((-path_loss_db + shadow_fading_db) / 10)


@task
class CfgUncorrelatedSpatialCorrelations:
    r"""Set the spatial correlation matrix of every UE--AP link to
    :math:`\mathbf{R}_{kl} = \mathbf{I}_N`.

    Consequently, the antenna elements experience independent small-scale fading and
    :math:`\mathbf{h}_{kl} \sim
    \mathcal{CN}(\mathbf{0}, \beta_{kl}\mathbf{I}_N)`.
    """

    def __call__(
        self, K: NumUEs, L: NumAPs, N: NumAntennasPerAP
    ) -> SpatialCorrelationMatrices:
        return np.tile(np.eye(N)[np.newaxis, np.newaxis, :, :], [K, L, 1, 1])


@task
class CfgApproximateSpatialCorrelations:
    r"""Compute spatial correlation matrices with a local-scattering approximation.

    Set the angular standard deviation in degrees explicitly to
    :code:`asd_in_degrees`. For every UE--AP link, the nominal angle of arrival is the
    UE-to-AP azimuth relative to the AP array orientation. Gaussian angular deviations
    around this angle yield a Hermitian Toeplitz correlation matrix for a uniform
    linear array.

    Specifically, with nominal angle :math:`\varphi_{kl}`, antenna spacing :math:`d_H`
    in wavelengths, angular standard deviation :math:`\sigma_\varphi` in radians, and
    antenna indices :math:`m,n`,

    .. math::

       [\mathbf{R}_{kl}]_{mn}
       = \exp\!\left(j2\pi d_H(m-n)\sin(\varphi_{kl})\right)
         \exp\!\left(
           -\frac{\sigma_\varphi^2}{2}
           \left(2\pi d_H(m-n)\cos(\varphi_{kl})\right)^2
         \right).

    Setting :code:`asd_in_degrees` to zero produces a rank-one correlation matrix.
    Values larger than 15 degrees issue an :class:`ApplicabilityWarning`, because the
    small-angle approximation is intended for angular standard deviations up to
    approximately 15 degrees.
    """

    asd_in_degrees: float

    def __post_init__(self):
        if not np.isfinite(self.asd_in_degrees) or self.asd_in_degrees < 0:
            raise ValueError("asd_in_degrees must be finite and non-negative")
        if self.asd_in_degrees > 15:
            warnings.warn(
                "asd_in_degrees is outside the small-angle approximation's "
                "applicability range of at most 15 degrees.",
                ApplicabilityWarning,
                stacklevel=2,
            )

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        N: NumAntennasPerAP,
        azimuths: UEtoAPAzimuthAngles,
        d_H: APAntennaSpacing,
        orientations: APArrayOrientations,
    ) -> SpatialCorrelationMatrices:
        asd_in_rads = self.asd_in_degrees * np.pi / 180
        nominal_angles = azimuths - orientations[np.newaxis, :]
        antenna_lags = np.arange(N)
        phase_shifts = (
            2
            * np.pi
            * d_H
            * antenna_lags[np.newaxis, np.newaxis, :]
            * np.sin(nominal_angles[..., np.newaxis])
        )
        angular_spread_attenuations = (
            2
            * np.pi
            * d_H
            * antenna_lags[np.newaxis, np.newaxis, :]
            * np.cos(nominal_angles[..., np.newaxis])
        )
        first_columns = np.exp(1j * phase_shifts) * np.exp(
            -(asd_in_rads**2) / 2 * angular_spread_attenuations**2
        )

        antenna_index_differences = (
            antenna_lags[:, np.newaxis] - antenna_lags[np.newaxis, :]
        )
        correlations = first_columns[..., np.abs(antenna_index_differences)]
        return np.where(
            antenna_index_differences >= 0,
            correlations,
            correlations.conj(),
        )


@task
class CfgRayleighChannels:
    r"""Draw spatially correlated Rayleigh channels for every UE--AP link.

    For every UE :math:`k`, AP :math:`l`, and realization :math:`o`, draw
    :math:`\mathbf{h}_{kl}^{(o)} \sim
    \mathcal{CN}(\mathbf{0},\beta_{kl}\mathbf{R}_{kl})`. The :type:`Seed` for the
    random number generator can be overridden with :code:`seed_override`.

    Every :math:`\mathbf{R}_{kl}` must be positive semidefinite. Singular correlation
    matrices are supported; an indefinite matrix raises a :class:`ValueError`.
    """

    seed_override: int | None = None

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        N: NumAntennasPerAP,
        O: NumRealizations,
        R: SpatialCorrelationMatrices,
        beta: LargeScaleFadingCoefficients,
        seed: Seed,
    ) -> ChannelRealizations:
        if self.seed_override is not None:
            seed = self.seed_override
        scaled_sqrt_R = np.sqrt(
            beta[..., np.newaxis, np.newaxis]
        ) * factor_correlation_matrix(R)

        rng = get_rng(seed, self)

        sample_shape = (K, L, N, O)
        rayleigh_samples = (
            rng.normal(0.0, 1.0, sample_shape)
            + 1.0j * rng.normal(0.0, 1.0, sample_shape)
        ) / np.sqrt(2)
        return scaled_sqrt_R @ rayleigh_samples
