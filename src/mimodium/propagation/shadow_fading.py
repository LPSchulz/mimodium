"""Configure the shadow fading model between UEs and APs."""

from typing import Literal

import numpy as np
from dagreon import task

from ..rng import Seed, factor_correlation_matrix, get_rng
from ..scenario import (
    APHeights,
    NumAPs,
    NumUEs,
    UEHeights,
    UEtoAP2DDistances,
    UEtoUE2DDistances,
)
from .path_loss import LineOfSightStates
from .radio_parameters import CarrierFrequency

#: Shadow-fading gain :math:`SF_{kl}` in decibels between every UE :math:`k` and AP
#: :math:`l`. Its linear power-gain factor is :math:`10^{SF_{kl}/10}`.
#:
#: :shape: ``(K, L)``
#: :dtype: ``float``
type ShadowFadingdB = np.ndarray


@task
class CfgZeroShadowFading:
    """Set the shadow-fading gain of every UE--AP link explicitly to zero decibels."""

    def __call__(self, K: NumUEs, L: NumAPs) -> ShadowFadingdB:
        return np.zeros((K, L), dtype=float)


@task
class CfgLogNormalShadowFading:
    r"""Draw log-normal shadow-fading gains for every UE--AP link.

    Set the standard deviation in decibels explicitly to :code:`sigma_db`. Set the
    decorrelation distance in meters explicitly to :code:`decorrelation_distance_m`.
    In the decibel domain, :math:`SF_{kl}` is zero-mean Gaussian with variance
    :math:`\mathtt{sigma\_db}^2`; therefore, :math:`10^{SF_{kl}/10}` is log-normal.
    For a positive decorrelation distance :math:`d_{\mathrm{cor}}`, shadow fading
    between UEs :math:`i` and :math:`j` has correlation
    :math:`\exp(-d_{ij}/d_{\mathrm{cor}})`. A decorrelation distance of zero makes
    shadow fading independent between UEs.

    Shadow fading is sampled independently for each AP. The :type:`Seed` for the random
    number generator can be overridden with :code:`seed_override`.
    """

    sigma_db: float
    decorrelation_distance_m: float = 0.0
    seed_override: int | None = None

    def __post_init__(self):
        self.check_applicability()

    def check_applicability(self) -> None:
        """Validate the shadow-fading standard deviation and decorrelation distance."""
        if not np.isfinite(self.sigma_db) or self.sigma_db < 0:
            raise ValueError("sigma_db must be finite and non-negative")
        if (
            not np.isfinite(self.decorrelation_distance_m)
            or self.decorrelation_distance_m < 0
        ):
            raise ValueError("decorrelation_distance_m must be finite and non-negative")

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        ue_distances: UEtoUE2DDistances,
        seed: Seed,
    ) -> ShadowFadingdB:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        sf_samples = rng.normal(0.0, self.sigma_db, (K, L))
        if self.decorrelation_distance_m:
            correlation = np.exp(-ue_distances / self.decorrelation_distance_m)
            sf_samples = factor_correlation_matrix(correlation) @ sf_samples
        return sf_samples


@task
class CfgRuralMacro3GPPShadowFading:
    """Draw spatially correlated 3GPP rural-macro (RMa) shadow fading.

    For each AP, separate zero-mean Gaussian fields represent LOS and NLOS shadow
    fading. Each field is spatially correlated across UEs, and the LOS state of each
    link selects the returned value.

    LOS links have a standard deviation of 4 dB up to the RMa breakpoint distance and
    6 dB beyond it. NLOS links have a standard deviation of 8 dB. The LOS and NLOS
    fields have decorrelation distances of 37 m and 120 m, respectively.

    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`.
    """

    seed_override: int | None = None

    def __call__(
        self,
        los_states: LineOfSightStates,
        ue_distances: UEtoUE2DDistances,
        distances_2d: UEtoAP2DDistances,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        fc: CarrierFrequency,
        seed: Seed,
    ) -> ShadowFadingdB:
        if self.seed_override is not None:
            seed = self.seed_override
        breakpoint_distances = (
            ue_heights[:, np.newaxis] * ap_heights[np.newaxis, :] * 2 * np.pi * fc / 3e8
        )
        rng = get_rng(seed, self)
        sf_std_los_before_breakpoint_db = 4.0
        sf_std_los_after_breakpoint_db = 6.0
        sf_std_los_db = np.where(
            distances_2d <= breakpoint_distances,
            sf_std_los_before_breakpoint_db,
            sf_std_los_after_breakpoint_db,
        )
        sf_samples_los = rng.normal(0.0, 1.0, los_states.shape)
        decorrelation_distance_los_m = 37.0
        correlation_los = np.exp(-ue_distances / decorrelation_distance_los_m)
        sf_samples_los = factor_correlation_matrix(correlation_los) @ sf_samples_los

        sf_std_nlos_db = 8.0
        sf_samples_nlos = rng.normal(0.0, sf_std_nlos_db, los_states.shape)
        decorrelation_distance_nlos_m = 120.0
        correlation_nlos = np.exp(-ue_distances / decorrelation_distance_nlos_m)
        sf_samples_nlos = factor_correlation_matrix(correlation_nlos) @ sf_samples_nlos

        return np.where(
            los_states,
            sf_std_los_db * sf_samples_los,
            sf_samples_nlos,
        )


@task
class CfgSuburbanMacro3GPPShadowFading:
    """Draw spatially correlated 3GPP suburban-macro (SMa) shadow fading.

    For each AP, separate zero-mean Gaussian fields represent LOS and NLOS shadow
    fading. Each field is spatially correlated across UEs, and the LOS state of each
    link selects the returned value.

    LOS links have a standard deviation of 4 dB below the SMa breakpoint distance and
    6 dB from the breakpoint onward. NLOS links have a standard deviation of 8 dB. The
    LOS and NLOS fields have decorrelation distances of 40 m and 50 m, respectively.

    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`.
    """

    seed_override: int | None = None

    def __call__(
        self,
        los_states: LineOfSightStates,
        ue_distances: UEtoUE2DDistances,
        distances_2d: UEtoAP2DDistances,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        fc: CarrierFrequency,
        seed: Seed,
    ) -> ShadowFadingdB:
        if self.seed_override is not None:
            seed = self.seed_override
        breakpoint_distances = (
            ue_heights[:, np.newaxis] * ap_heights[np.newaxis, :] * 2 * np.pi * fc / 3e8
        )
        sf_std_los_db = np.where(distances_2d < breakpoint_distances, 4.0, 6.0)

        rng = get_rng(seed, self)
        sf_samples_los = rng.normal(0.0, 1.0, los_states.shape)
        correlation_los = np.exp(-ue_distances / 40.0)
        sf_samples_los = factor_correlation_matrix(correlation_los) @ sf_samples_los

        sf_samples_nlos = rng.normal(0.0, 8.0, los_states.shape)
        correlation_nlos = np.exp(-ue_distances / 50.0)
        sf_samples_nlos = factor_correlation_matrix(correlation_nlos) @ sf_samples_nlos

        return np.where(
            los_states,
            sf_std_los_db * sf_samples_los,
            sf_samples_nlos,
        )


@task
class CfgUrbanMacro3GPPShadowFading:
    """Draw spatially correlated 3GPP urban-macro (UMa) shadow fading.

    For each AP, separate zero-mean Gaussian fields represent LOS and NLOS shadow
    fading. Each field is spatially correlated across UEs, and the LOS state of each
    link selects the returned value.

    LOS links have a standard deviation of 4 dB. NLOS links have a standard deviation
    of 6 dB, or 7.8 dB when :code:`use_optional` is set. The LOS and NLOS fields have
    decorrelation distances of 37 m and 50 m, respectively.

    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`.
    """

    use_optional: bool = False
    seed_override: int | None = None

    def __call__(
        self,
        los_states: LineOfSightStates,
        ue_distances: UEtoUE2DDistances,
        seed: Seed,
    ) -> ShadowFadingdB:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        sf_std_los_db = 4.0
        sf_samples_los = rng.normal(0.0, sf_std_los_db, los_states.shape)
        decorrelation_distance_los_m = 37.0
        correlation_los = np.exp(-ue_distances / decorrelation_distance_los_m)
        sf_samples_los = factor_correlation_matrix(correlation_los) @ sf_samples_los

        sf_std_nlos_db = 6.0
        if self.use_optional:
            sf_std_nlos_db = 7.8
        sf_samples_nlos = rng.normal(0.0, sf_std_nlos_db, los_states.shape)
        decorrelation_distance_nlos_m = 50.0
        correlation_nlos = np.exp(-ue_distances / decorrelation_distance_nlos_m)
        sf_samples_nlos = factor_correlation_matrix(correlation_nlos) @ sf_samples_nlos

        return np.where(los_states, sf_samples_los, sf_samples_nlos)


@task
class CfgUrbanMicro3GPPShadowFading:
    """Draw spatially correlated 3GPP urban-micro (UMi) shadow fading.

    For each AP, separate zero-mean Gaussian fields represent LOS and NLOS shadow
    fading. Each field is spatially correlated across UEs, and the LOS state of each
    link selects the returned value.

    LOS links have a standard deviation of 4 dB. NLOS links have a standard deviation
    of 7.82 dB, or 8.2 dB when :code:`use_optional` is set. The LOS and NLOS fields
    have decorrelation distances of 10 m and 13 m, respectively.

    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`.
    """

    use_optional: bool = False
    seed_override: int | None = None

    def __call__(
        self,
        los_states: LineOfSightStates,
        ue_distances: UEtoUE2DDistances,
        seed: Seed,
    ) -> ShadowFadingdB:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        sf_std_los_db = 4.0
        sf_samples_los = rng.normal(0.0, sf_std_los_db, los_states.shape)
        decorrelation_distance_los_m = 10.0
        correlation_los = np.exp(-ue_distances / decorrelation_distance_los_m)
        sf_samples_los = factor_correlation_matrix(correlation_los) @ sf_samples_los

        sf_std_nlos_db = 7.82
        if self.use_optional:
            sf_std_nlos_db = 8.2
        sf_samples_nlos = rng.normal(0.0, sf_std_nlos_db, los_states.shape)
        decorrelation_distance_nlos_m = 13.0
        correlation_nlos = np.exp(-ue_distances / decorrelation_distance_nlos_m)
        sf_samples_nlos = factor_correlation_matrix(correlation_nlos) @ sf_samples_nlos

        return np.where(los_states, sf_samples_los, sf_samples_nlos)


@task
class CfgIndoorOffice3GPPShadowFading:
    """Draw spatially correlated 3GPP indoor-office (InH) shadow fading.

    For each AP, separate zero-mean Gaussian fields represent LOS and NLOS shadow
    fading. Each field is spatially correlated across UEs, and the LOS state of each
    link selects the returned value.

    LOS links have a standard deviation of 3 dB. NLOS links have a standard deviation
    of 8.03 dB, or 8.29 dB when :code:`use_optional` is set. The LOS and NLOS fields
    have decorrelation distances of 10 m and 6 m, respectively.

    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`.
    """

    use_optional: bool = False
    seed_override: int | None = None

    def __call__(
        self,
        los_states: LineOfSightStates,
        ue_distances: UEtoUE2DDistances,
        seed: Seed,
    ) -> ShadowFadingdB:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        sf_std_los_db = 3.0
        sf_samples_los = rng.normal(0.0, sf_std_los_db, los_states.shape)
        decorrelation_distance_los_m = 10.0
        correlation_los = np.exp(-ue_distances / decorrelation_distance_los_m)
        sf_samples_los = factor_correlation_matrix(correlation_los) @ sf_samples_los

        sf_std_nlos_db = 8.03
        if self.use_optional:
            sf_std_nlos_db = 8.29
        sf_samples_nlos = rng.normal(0.0, sf_std_nlos_db, los_states.shape)
        decorrelation_distance_nlos_m = 6.0
        correlation_nlos = np.exp(-ue_distances / decorrelation_distance_nlos_m)
        sf_samples_nlos = factor_correlation_matrix(correlation_nlos) @ sf_samples_nlos

        return np.where(los_states, sf_samples_los, sf_samples_nlos)


@task
class CfgIndoorFactory3GPPShadowFading:
    """Draw spatially correlated 3GPP indoor-factory (InF) shadow fading.

    For each AP, separate zero-mean Gaussian fields represent LOS and NLOS shadow
    fading. Each field is spatially correlated across UEs, and the LOS state of each
    link selects the returned value.

    Set :code:`subscenario` to ``"SL"`` for sparse clutter with low APs, ``"DL"`` for
    dense clutter with low APs, ``"SH"`` for sparse clutter with high APs, or ``"DH"``
    for dense clutter with high APs. LOS links have a standard deviation of 4.3 dB.
    NLOS links have standard deviations of 5.7, 7.2, 5.9, and 4.0 dB for the SL, DL,
    SH, and DH subscenarios, respectively. The LOS and NLOS fields both have a
    decorrelation distance of 10 m.

    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`.
    """

    subscenario: Literal["SL", "DL", "SH", "DH"] = "SL"
    seed_override: int | None = None

    def __post_init__(self):
        if self.subscenario not in ("SL", "DL", "SH", "DH"):
            raise ValueError("subscenario must be one of 'SL', 'DL', 'SH', or 'DH'")

    def __call__(
        self,
        los_states: LineOfSightStates,
        ue_distances: UEtoUE2DDistances,
        seed: Seed,
    ) -> ShadowFadingdB:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        correlation = np.exp(-ue_distances / 10.0)
        correlation_factor = factor_correlation_matrix(correlation)

        sf_samples_los = rng.normal(0.0, 4.3, los_states.shape)
        sf_samples_los = correlation_factor @ sf_samples_los

        sf_std_nlos_db = {
            "SL": 5.7,
            "DL": 7.2,
            "SH": 5.9,
            "DH": 4.0,
        }[self.subscenario]
        sf_samples_nlos = rng.normal(0.0, sf_std_nlos_db, los_states.shape)
        sf_samples_nlos = correlation_factor @ sf_samples_nlos

        return np.where(los_states, sf_samples_los, sf_samples_nlos)
