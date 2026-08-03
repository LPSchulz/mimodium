"""Configure the path loss model between UEs and APs."""

import warnings
from typing import Literal

import numpy as np
from dagreon import task

from ..rng import Seed, get_rng
from ..scenario import (
    APHeights,
    NumAPs,
    NumUEs,
    UEHeights,
    UEtoAP2DDistances,
    UEtoAP3DDistances,
)
from ..warnings import ApplicabilityWarning
from .radio_parameters import CarrierFrequency

#: Whether the channel between every UE :math:`k` and AP :math:`l` is
#: line-of-sight.
#:
#: :shape: ``(K, L)``
#: :dtype: ``bool``
type LineOfSightStates = np.ndarray
#: Path loss :math:`PL_{kl}` in decibels between every UE :math:`k` and AP
#: :math:`l`. Larger values represent more attenuation.
#:
#: :shape: ``(K, L)``
#: :dtype: ``float``
type PathLossdB = np.ndarray


def _carrier_frequency_ghz(fc: CarrierFrequency) -> float:
    return fc / 1e9


@task
class ComputeAlwaysNonLineOfSight:
    """Mark every UE--AP link as non-line-of-sight.

    This deterministic state model is useful with NLOS-only path-loss models or when
    isolating path loss from LOS-state generation.
    """

    def __call__(self, K: NumUEs, L: NumAPs) -> LineOfSightStates:
        return np.zeros((K, L), dtype=bool)


@task
class CfgExponentialPathLoss:
    r"""Compute single-slope path loss
    :math:`PL_{kl} = PL_0 + 10\alpha\log_{10}(d_{kl}^{3D}/1\,\mathrm{m})`.

    Set the path loss :math:`PL_0` at one meter in decibels explicitly to
    :code:`reference_path_loss_db` and the path-loss exponent :math:`\alpha` explicitly
    to
    :code:`pathloss_exponent`. This model does not distinguish between LOS and NLOS
    links.

    The defaults follow the legacy 3GPP urban-micro NLOS model
    :math:`PL = 22.7 + 26\log_{10}(f_c) + 36.7\log_{10}(d/1\,\mathrm{m})`, where
    :math:`f_c` is given in GHz. At 2 GHz,
    :math:`22.7 + 26\log_{10}(2) \approx 30.5` and :math:`36.7 = 10 \cdot 3.67`.
    """

    # only use this for pure NLOS scenarios
    # default values are like UrbanMicro3GPP with 2 GHz carrier frequency
    reference_path_loss_db: float = 30.5
    pathloss_exponent: float = 3.67

    def __post_init__(self):
        if not np.isfinite(self.reference_path_loss_db):
            raise ValueError("reference_path_loss_db must be finite")
        if not np.isfinite(self.pathloss_exponent) or self.pathloss_exponent <= 0:
            raise ValueError("pathloss_exponent must be finite and positive")

    def __call__(self, d_3d: UEtoAP3DDistances) -> PathLossdB:
        return self.reference_path_loss_db + 10 * self.pathloss_exponent * np.log10(
            d_3d
        )


@task
class CfgRuralMacro3GPPPathLoss:
    """Compute 3GPP rural-macro (RMa) path loss for LOS and NLOS links.

    The equations and applicability conditions are from Table 7.4.1-1, "Pathloss
    models," of 3GPP TR 38.901. The LOS equation changes at its breakpoint distance.
    The NLOS path loss is the maximum of the RMa LOS and NLOS equations.

    Set the average building height in meters explicitly to :code:`h` and the average
    street width in meters explicitly to :code:`w`.
    """

    # applicability ranges:
    # 5m <= h <= 50m, 5m <= w <= 50m, 10m <= h_bs <= 150m, 1m <= h_ut <= 10m
    # default values:
    # h_bs = 35m, h_ut = 1.5m

    h: float = 5  # average building height in m
    w: float = 20  # average street width in m

    def __post_init__(self):
        if not np.isfinite(self.h) or self.h <= 0:
            raise ValueError("h must be finite and positive")
        if not np.isfinite(self.w) or self.w <= 0:
            raise ValueError("w must be finite and positive")

    def check_applicability(
        self,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        distances_2d: UEtoAP2DDistances,
        distances_3d: UEtoAP3DDistances,
        los_states: LineOfSightStates,
        fc: CarrierFrequency,
    ) -> None:
        """Issue warnings for inputs outside the 3GPP RMa applicability ranges."""
        if not (0.5 < _carrier_frequency_ghz(fc) < 30):
            warnings.warn(
                "fc is outside the RMa applicability range of 0.5--30 GHz (exclusive).",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if not (5 <= self.h <= 50):
            warnings.warn(
                "h is outside the RMa applicability range of 5--50 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if not (5 <= self.w <= 50):
            warnings.warn(
                "w is outside the RMa applicability range of 5--50 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if not (np.all(1 <= ue_heights) and np.all(ue_heights <= 10)):
            warnings.warn(
                "ue_heights are outside the RMa applicability range of 1--10 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if not (np.all(10 <= ap_heights) and np.all(ap_heights <= 150)):
            warnings.warn(
                "ap_heights are outside the RMa applicability range of 10--150 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )
        maximum_distances = np.where(los_states, 10_000.0, 5_000.0)
        if np.any((distances_2d < 10) | (distances_2d > maximum_distances)):
            warnings.warn(
                "distances_2d are outside the RMa applicability range: "
                "10--10000 m for LOS and 10--5000 m for NLOS.",
                ApplicabilityWarning,
                stacklevel=3,
            )

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        distances_2d: UEtoAP2DDistances,
        distances_3d: UEtoAP3DDistances,
        los_states: LineOfSightStates,
        fc: CarrierFrequency,
    ) -> PathLossdB:
        self.check_applicability(
            ue_heights,
            ap_heights,
            distances_2d,
            distances_3d,
            los_states,
            fc,
        )
        pathlosses = np.zeros((K, L))
        fc_times_2pi_over_c = 2 * np.pi * fc / 3e8
        for k in range(K):
            for l in range(L):
                d_2d = distances_2d[k, l]
                d_3d = distances_3d[k, l]
                h_bs = ap_heights[l]
                h_ut = ue_heights[k]
                is_line_of_sight = los_states[k, l]

                d_bp = h_bs * h_ut * fc_times_2pi_over_c
                if d_2d <= d_bp:
                    pl_rma_los = self.pl1(d_3d, fc)
                else:
                    pl_rma_los = self.pl1(d_bp, fc) + self.pl2(d_3d, d_bp)
                if is_line_of_sight:
                    pathlosses[k, l] = pl_rma_los
                else:
                    pathlosses[k, l] = max(
                        pl_rma_los,
                        self.pl_rma_nlos_prime(d_3d, h_bs, h_ut, fc),
                    )
        return pathlosses

    def pl1(self, d, fc: CarrierFrequency):
        """Compute RMa LOS path loss in decibels below the breakpoint distance."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return (
            20 * np.log10(40 * np.pi * d * fc_ghz / 3)
            + min(0.03 * self.h**1.72, 10) * np.log10(d)
            - min(0.044 * self.h**1.72, 14.77)
            + 0.002 * np.log10(self.h) * d
        )

    def pl2(self, d_3d, d_bp):
        """Compute the additional RMa LOS loss beyond the breakpoint in decibels."""

        return 40 * np.log10(d_3d / d_bp)

    def pl_rma_nlos_prime(self, d_3d, h_bs, h_ut, fc: CarrierFrequency):
        """Compute the RMa NLOS candidate path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return (
            161.04
            - 7.1 * np.log10(self.w)
            + 7.5 * np.log10(self.h)
            - (24.37 - 3.7 * (self.h / h_bs) ** 2) * np.log10(h_bs)
            + (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(d_3d) - 3)
            + 20 * np.log10(fc_ghz)
            - (3.2 * (np.log10(11.75 * h_ut)) ** 2 - 4.97)
        )


@task
class CfgSuburbanMacro3GPPPathLoss:
    """Compute 3GPP suburban-macro (SMa) path loss for LOS and NLOS links.

    The equations and applicability conditions are from Table 7.4.1-1, "Pathloss
    models," of 3GPP TR 38.901 Release 19. The LOS equation changes at its breakpoint
    distance. The NLOS path loss is the maximum of the SMa LOS and NLOS equations.

    Set the average building height in meters explicitly to :code:`h` and the average
    street width in meters explicitly to :code:`w`.
    """

    h: float = 10
    w: float = 10

    def __post_init__(self):
        if not np.isfinite(self.h) or self.h <= 0:
            raise ValueError("h must be finite and positive")
        if not np.isfinite(self.w) or self.w <= 0:
            raise ValueError("w must be finite and positive")

    def check_applicability(
        self,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        distances_2d: UEtoAP2DDistances,
        distances_3d: UEtoAP3DDistances,
        fc: CarrierFrequency,
    ) -> None:
        """Issue warnings for inputs outside the 3GPP SMa applicability ranges."""
        if not (0.5 < _carrier_frequency_ghz(fc) < 37):
            warnings.warn(
                "fc is outside the SMa applicability range of 0.5--37 GHz (exclusive).",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if not (5 <= self.h <= 50):
            warnings.warn(
                "h is outside the SMa applicability range of 5--50 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if not (5 <= self.w <= 50):
            warnings.warn(
                "w is outside the SMa applicability range of 5--50 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if not (np.all(1 < ue_heights) and np.all(ue_heights < 14)):
            warnings.warn(
                "ue_heights are outside the SMa applicability range of 1--14 m "
                "(exclusive).",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if not (np.all(25 < ap_heights) and np.all(ap_heights < 35)):
            warnings.warn(
                "ap_heights are outside the SMa applicability range of 25--35 m "
                "(exclusive).",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if np.any((distances_2d < 10) | (distances_2d > 5_000)):
            warnings.warn(
                "distances_2d are outside the SMa applicability range of 10--5000 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        distances_2d: UEtoAP2DDistances,
        distances_3d: UEtoAP3DDistances,
        los_states: LineOfSightStates,
        fc: CarrierFrequency,
    ) -> PathLossdB:
        self.check_applicability(
            ue_heights,
            ap_heights,
            distances_2d,
            distances_3d,
            fc,
        )

        pathlosses = np.zeros((K, L))
        fc_times_2pi_over_c = 2 * np.pi * fc / 3e8
        for k in range(K):
            for l in range(L):
                d_2d = distances_2d[k, l]
                d_3d = distances_3d[k, l]
                h_bs = ap_heights[l]
                h_ut = ue_heights[k]
                d_bp = h_bs * h_ut * fc_times_2pi_over_c
                if d_2d < d_bp:
                    los_pathloss = self.pl1(d_3d, fc)
                else:
                    los_pathloss = self.pl1(d_bp, fc) + self.pl2(d_3d, d_bp)
                if los_states[k, l]:
                    pathlosses[k, l] = los_pathloss
                else:
                    pathlosses[k, l] = max(
                        los_pathloss,
                        self.pl_sma_nlos_prime(d_3d, h_bs, h_ut, fc),
                    )
        return pathlosses

    def pl1(self, d_3d, fc: CarrierFrequency):
        """Compute SMa LOS path loss in decibels below the breakpoint distance."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return (
            20 * np.log10(40 * np.pi * d_3d * fc_ghz / 3)
            + min(0.03 * self.h**1.72, 10) * np.log10(d_3d)
            - min(0.044 * self.h**1.72, 14.77)
            + 0.002 * np.log10(self.h) * d_3d
        )

    def pl2(self, d_3d, d_bp):
        """Compute the additional SMa LOS loss beyond the breakpoint in decibels."""

        return 40 * np.log10(d_3d / d_bp)

    def pl_sma_nlos_prime(self, d_3d, h_bs, h_ut, fc: CarrierFrequency):
        """Compute the SMa NLOS candidate path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return (
            161.04
            - 7.1 * np.log10(self.w)
            + 7.5 * np.log10(self.h)
            - (24.37 - 3.7 * (self.h / h_bs) ** 2) * np.log10(h_bs)
            + (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(d_3d) - 3)
            + 20 * np.log10(fc_ghz)
            - (3.2 * (np.log10(11.75 * h_ut)) ** 2 - 4.97)
        )


@task
class CfgUrbanMacro3GPPPathLoss:
    """Compute 3GPP urban-macro (UMa) path loss for LOS and NLOS links.

    The equations and applicability conditions are from Table 7.4.1-1, "Pathloss
    models," of 3GPP TR 38.901. The LOS equation changes at its breakpoint distance.
    The NLOS path loss is the maximum of the UMa LOS equation and the selected NLOS
    equation.

    Set :code:`use_optional` to use the optional NLOS equation instead of the default
    UMa NLOS equation. The breakpoint depends on a randomly sampled effective
    environment height. The :type:`Seed` for this sampling can be overridden with
    :code:`seed_override`.
    """

    # applicability ranges:
    # 1.5m <= h_ut <= 22.5
    # default values:
    # h_bs = 25m
    use_optional: bool = False
    seed_override: int | None = None

    def check_applicability(
        self,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        distances_2d: UEtoAP2DDistances,
        distances_3d: UEtoAP3DDistances,
        fc: CarrierFrequency,
    ) -> None:
        """Issue warnings for inputs outside the 3GPP UMa applicability ranges."""
        if not (0.5 < _carrier_frequency_ghz(fc) < 100):
            warnings.warn(
                "fc is outside the UMa applicability range of 0.5--100 GHz "
                "(exclusive).",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if not (np.all(1.5 <= ue_heights) and np.all(ue_heights <= 22.5)):
            warnings.warn(
                "ue_heights are outside the UMa applicability range of 1.5--22.5 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if np.any((distances_2d < 10) | (distances_2d > 5_000)):
            warnings.warn(
                "distances_2d are outside the UMa applicability range of 10--5000 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        distances_2d: UEtoAP2DDistances,
        distances_3d: UEtoAP3DDistances,
        los_states: LineOfSightStates,
        fc: CarrierFrequency,
        seed: Seed,
    ) -> PathLossdB:
        self.check_applicability(
            ue_heights,
            ap_heights,
            distances_2d,
            distances_3d,
            fc,
        )
        if self.seed_override is not None:
            seed = self.seed_override
        pathlosses = np.zeros((K, L))
        fc_times_4_over_c = 4 * fc / 3e8
        rng = get_rng(seed, self)
        for k in range(K):
            for l in range(L):
                d_2d = distances_2d[k, l]
                d_3d = distances_3d[k, l]
                h_bs = ap_heights[l]
                h_ut = ue_heights[k]
                is_line_of_sight = los_states[k, l]

                h_e = self.calculate_h_e(d_2d, h_ut, rng)
                d_bp = (h_bs - h_e) * (h_ut - h_e) * fc_times_4_over_c
                if d_2d <= d_bp:
                    pathloss_db = self.pl1(d_3d, fc)
                else:
                    pathloss_db = self.pl2(d_3d, d_bp, h_bs, h_ut, fc)
                if is_line_of_sight:
                    pathlosses[k, l] = pathloss_db
                else:
                    nlos_pathloss = self.pl_uma_nlos_prime(d_3d, h_ut, fc)
                    if self.use_optional:
                        nlos_pathloss = self.pl_uma_nlos_optional(d_3d, fc)
                    pathlosses[k, l] = max(
                        pathloss_db,
                        nlos_pathloss,
                    )
        return pathlosses

    def pl1(self, d_3d, fc: CarrierFrequency):
        """Compute UMa LOS path loss in decibels below the breakpoint distance."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return 28.0 + 22 * np.log10(d_3d) + 20 * np.log10(fc_ghz)

    def pl2(self, d_3d, d_bp, h_bs, h_ut, fc: CarrierFrequency):
        """Compute UMa LOS path loss in decibels beyond the breakpoint distance."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return (
            28.0
            + 40 * np.log10(d_3d)
            + 20 * np.log10(fc_ghz)
            - 9 * np.log10(d_bp**2 + (h_bs - h_ut) ** 2)
        )

    def pl_uma_nlos_prime(self, d_3d, h_ut, fc: CarrierFrequency):
        """Compute the default UMa NLOS candidate path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return (
            13.54 + 39.08 * np.log10(d_3d) + 20 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
        )

    def pl_uma_nlos_optional(self, d_3d, fc: CarrierFrequency):
        """Compute the optional UMa NLOS candidate path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return 32.4 + 30 * np.log10(d_3d) + 20 * np.log10(fc_ghz)

    def calculate_h_e(self, d_2d: float, h_ut: float, rng: np.random.Generator):
        """Sample the effective environment height in meters for the UMa breakpoint."""

        if h_ut < 13:
            return 1.0
        else:
            if d_2d <= 18:
                g_of_d_2d = 0
            else:
                g_of_d_2d = 1.25 * (d_2d / 100) ** 3 * np.exp(-d_2d / 150)
            c_of_d_2d_and_h_ut = ((h_ut - 13) / 10) ** 1.5 * g_of_d_2d
            if rng.binomial(1, 1 / (1 + c_of_d_2d_and_h_ut)) > 0:
                return 1.0
            else:
                environment_heights = np.arange(12, h_ut - 1.5 + 1e-12, 3)
                if environment_heights.size == 0:
                    return 1.0
                return rng.choice(environment_heights)


@task
class CfgUrbanMicro3GPPPathLoss:
    """Compute 3GPP urban-micro street-canyon (UMi) path loss for LOS and NLOS links.

    The equations and applicability conditions are from Table 7.4.1-1, "Pathloss
    models," of 3GPP TR 38.901. The LOS equation changes at its breakpoint distance.
    The NLOS path loss is the maximum of the UMi LOS equation and the selected NLOS
    equation.

    Set :code:`use_optional` to use the optional NLOS equation instead of the default
    UMi NLOS equation. The breakpoint calculation uses an effective environment height
    of one meter.
    """

    # applicability ranges:
    # 1.5m <= h_ut <= 22.5
    # default values:
    # h_bs = 10m
    use_optional: bool = False

    def check_applicability(
        self,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        distances_2d: UEtoAP2DDistances,
        distances_3d: UEtoAP3DDistances,
        fc: CarrierFrequency,
    ) -> None:
        """Issue warnings for inputs outside the 3GPP UMi applicability ranges."""
        if not (0.5 < _carrier_frequency_ghz(fc) < 100):
            warnings.warn(
                "fc is outside the UMi applicability range of 0.5--100 GHz "
                "(exclusive).",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if not (np.all(1.5 <= ue_heights) and np.all(ue_heights <= 22.5)):
            warnings.warn(
                "ue_heights are outside the UMi applicability range of 1.5--22.5 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if np.any((distances_2d < 10) | (distances_2d > 5_000)):
            warnings.warn(
                "distances_2d are outside the UMi applicability range of 10--5000 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        distances_2d: UEtoAP2DDistances,
        distances_3d: UEtoAP3DDistances,
        los_states: LineOfSightStates,
        fc: CarrierFrequency,
    ) -> PathLossdB:
        self.check_applicability(
            ue_heights,
            ap_heights,
            distances_2d,
            distances_3d,
            fc,
        )
        pathlosses = np.zeros((K, L))
        fc_times_4_over_c = 4 * fc / 3e8
        for k in range(K):
            for l in range(L):
                d_2d = distances_2d[k, l]
                d_3d = distances_3d[k, l]
                h_bs = ap_heights[l]
                h_ut = ue_heights[k]
                is_line_of_sight = los_states[k, l]

                h_e = 1.0
                d_bp = (h_bs - h_e) * (h_ut - h_e) * fc_times_4_over_c
                if d_2d <= d_bp:
                    pathloss_db = self.pl1(d_3d, fc)
                else:
                    pathloss_db = self.pl2(d_3d, d_bp, h_bs, h_ut, fc)
                if is_line_of_sight:
                    pathlosses[k, l] = pathloss_db
                else:
                    nlos_pathloss = self.pl_umi_nlos_prime(d_3d, h_ut, fc)
                    if self.use_optional:
                        nlos_pathloss = self.pl_umi_nlos_optional(d_3d, fc)
                    pathlosses[k, l] = max(
                        pathloss_db,
                        nlos_pathloss,
                    )
        return pathlosses

    def pl1(self, d_3d, fc: CarrierFrequency):
        """Compute UMi LOS path loss in decibels below the breakpoint distance."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return 32.4 + 21 * np.log10(d_3d) + 20 * np.log10(fc_ghz)

    def pl2(self, d_3d, d_bp, h_bs, h_ut, fc: CarrierFrequency):
        """Compute UMi LOS path loss in decibels beyond the breakpoint distance."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return (
            32.4
            + 40 * np.log10(d_3d)
            + 20 * np.log10(fc_ghz)
            - 9.5 * np.log10(d_bp**2 + (h_bs - h_ut) ** 2)
        )

    def pl_umi_nlos_prime(self, d_3d, h_ut, fc: CarrierFrequency):
        """Compute the default UMi NLOS candidate path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return (
            35.3 * np.log10(d_3d) + 22.4 + 21.3 * np.log10(fc_ghz) - 0.3 * (h_ut - 1.5)
        )

    def pl_umi_nlos_optional(self, d_3d, fc: CarrierFrequency):
        """Compute the optional UMi NLOS candidate path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return 32.4 + 31.9 * np.log10(d_3d) + 20 * np.log10(fc_ghz)


@task
class CfgIndoorOffice3GPPPathLoss:
    """Compute 3GPP indoor-office (InH) path loss for LOS and NLOS links.

    The equations and applicability conditions are from Table 7.4.1-1, "Pathloss
    models," of 3GPP TR 38.901. The NLOS path loss is the maximum of the InH LOS
    equation and the selected NLOS equation.

    Set :code:`use_optional` to use the optional NLOS equation instead of the default
    InH NLOS equation.
    """

    use_optional: bool = False

    def check_applicability(
        self,
        distances_3d: UEtoAP3DDistances,
        fc: CarrierFrequency,
    ) -> None:
        """Issue warnings for inputs outside the 3GPP InH applicability ranges."""
        if not (0.5 < _carrier_frequency_ghz(fc) < 100):
            warnings.warn(
                "fc is outside the indoor-office applicability range of "
                "0.5--100 GHz (exclusive).",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if np.any((distances_3d < 1) | (distances_3d > 150)):
            warnings.warn(
                "distances_3d are outside the indoor-office applicability range of "
                "1--150 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        distances_3d: UEtoAP3DDistances,
        los_states: LineOfSightStates,
        fc: CarrierFrequency,
    ) -> PathLossdB:
        self.check_applicability(distances_3d, fc)
        pathlosses = np.zeros((K, L))
        for k in range(K):
            for l in range(L):
                d_3d = distances_3d[k, l]
                is_line_of_sight = los_states[k, l]

                pathloss_db = self.pl_inh_los(d_3d, fc)
                if is_line_of_sight:
                    pathlosses[k, l] = pathloss_db
                else:
                    nlos_pathloss = self.pl_inh_nlos_prime(d_3d, fc)
                    if self.use_optional:
                        nlos_pathloss = self.pl_inh_nlos_optional(d_3d, fc)
                    pathlosses[k, l] = max(pathloss_db, nlos_pathloss)
        return pathlosses

    def pl_inh_los(self, d_3d, fc: CarrierFrequency):
        """Compute indoor-office LOS path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return 32.4 + 17.3 * np.log10(d_3d) + 20 * np.log10(fc_ghz)

    def pl_inh_nlos_prime(self, d_3d, fc: CarrierFrequency):
        """Compute the default indoor-office NLOS candidate path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return 38.3 * np.log10(d_3d) + 17.30 + 24.9 * np.log10(fc_ghz)

    def pl_inh_nlos_optional(self, d_3d, fc: CarrierFrequency):
        """Compute the optional indoor-office NLOS candidate path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return 32.4 + 31.9 * np.log10(d_3d) + 20 * np.log10(fc_ghz)


@task
class CfgIndoorFactory3GPPPathLoss:
    """Compute 3GPP indoor-factory (InF) path loss for LOS and NLOS links.

    The equations and applicability conditions are from Table 7.4.1-1, "Pathloss
    models," of 3GPP TR 38.901. The NLOS path loss is at least the InF LOS path loss.

    Set :code:`subscenario` to ``"SL"`` for sparse clutter with low APs, ``"DL"`` for
    dense clutter with low APs, ``"SH"`` for sparse clutter with high APs, or ``"DH"``
    for dense clutter with high APs. For the dense-low subscenario, the NLOS path loss
    is additionally at least the sparse-low NLOS path loss.
    """

    subscenario: Literal["SL", "DL", "SH", "DH"] = "SL"

    def __post_init__(self):
        if self.subscenario not in ("SL", "DL", "SH", "DH"):
            raise ValueError("subscenario must be one of 'SL', 'DL', 'SH', or 'DH'")

    def check_applicability(
        self,
        distances_3d: UEtoAP3DDistances,
        fc: CarrierFrequency,
    ) -> None:
        """Issue warnings for inputs outside the 3GPP InF applicability ranges."""
        if not (0.5 < _carrier_frequency_ghz(fc) < 100):
            warnings.warn(
                "fc is outside the indoor-factory applicability range of "
                "0.5--100 GHz (exclusive).",
                ApplicabilityWarning,
                stacklevel=3,
            )
        if np.any((distances_3d < 1) | (distances_3d > 600)):
            warnings.warn(
                "distances_3d are outside the indoor-factory applicability range of "
                "1--600 m.",
                ApplicabilityWarning,
                stacklevel=3,
            )

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        distances_3d: UEtoAP3DDistances,
        los_states: LineOfSightStates,
        fc: CarrierFrequency,
    ) -> PathLossdB:
        self.check_applicability(distances_3d, fc)
        pathlosses = np.zeros((K, L))
        for k in range(K):
            for l in range(L):
                d_3d = distances_3d[k, l]
                los_pathloss = self.pl_inf_los(d_3d, fc)
                if los_states[k, l]:
                    pathlosses[k, l] = los_pathloss
                else:
                    nlos_pathlosses = [
                        los_pathloss,
                        self.pl_inf_nlos_prime(d_3d, fc),
                    ]
                    if self.subscenario == "DL":
                        nlos_pathlosses.append(self.pl_inf_sl(d_3d, fc))
                    pathlosses[k, l] = max(nlos_pathlosses)
        return pathlosses

    def pl_inf_los(self, d_3d, fc: CarrierFrequency):
        """Compute indoor-factory LOS path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return 31.84 + 21.5 * np.log10(d_3d) + 19 * np.log10(fc_ghz)

    def pl_inf_nlos_prime(self, d_3d, fc: CarrierFrequency):
        """Compute the selected indoor-factory NLOS candidate path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        constant, distance_coefficient = {
            "SL": (33.0, 25.5),
            "DL": (18.6, 35.7),
            "SH": (32.4, 23.0),
            "DH": (33.63, 21.9),
        }[self.subscenario]
        return constant + distance_coefficient * np.log10(d_3d) + 20 * np.log10(fc_ghz)

    def pl_inf_sl(self, d_3d, fc: CarrierFrequency):
        """Compute sparse-low indoor-factory NLOS path loss in decibels."""

        fc_ghz = _carrier_frequency_ghz(fc)
        return 33 + 25.5 * np.log10(d_3d) + 20 * np.log10(fc_ghz)


@task
class CfgHataCost231PathLoss:
    r"""Compute three-slope Hata-COST231 path loss for every UE--AP link.

    Set the lower breakpoint distance in meters explicitly to :code:`d0` and the upper
    breakpoint distance in meters explicitly to :code:`d1`. For 2D link distance
    :math:`d`, define :math:`F = \log_{10}(f_c/1\,\mathrm{MHz})` and

    .. math::

       A = 46.3 + 33.9F
           - 13.82\log_{10}(h_{\mathrm{AP}}/1\,\mathrm{m})
           - (1.1F - 0.7)h_{\mathrm{UE}}/1\,\mathrm{m}
           + 1.56F - 0.8.

    The resulting path loss is

    .. math::

       PL(d) =
       \begin{cases}
       A + 15\log_{10}(d_1) + 20\log_{10}(d_0), & d \le d_0,\\
       A + 15\log_{10}(d_1) + 20\log_{10}(d), & d_0 < d \le d_1,\\
       A + 35\log_{10}(d), & d > d_1,
       \end{cases}

    where :math:`d`, :math:`d_0`, and :math:`d_1` are expressed in kilometers inside
    the logarithms. Thus, path loss is constant up to :code:`d0`, has path-loss
    exponent 2 between the breakpoints, and has path-loss exponent 3.5 beyond
    :code:`d1`.

    The intercept depends on the carrier frequency as well as the AP and UE heights.
    This model does not distinguish between LOS and NLOS links. It is the model used in
    the *Cell-Free Massive MIMO versus Small Cells* reference scenario.
    """

    # three slope path loss model from "Cell-Free Massive MIMO vs Small Cells"
    d0: float = 10  # close reference distance in m
    d1: float = 50

    def __post_init__(self):
        if not np.isfinite(self.d0) or self.d0 <= 0:
            raise ValueError("d0 must be finite and positive")
        if not np.isfinite(self.d1) or self.d1 <= 0:
            raise ValueError("d1 must be finite and positive")
        if self.d0 > self.d1:
            raise ValueError("d0 must not exceed d1")

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        ue_heights: UEHeights,
        ap_heights: APHeights,
        distances_2d: UEtoAP2DDistances,
        fc: CarrierFrequency,
    ) -> PathLossdB:
        pathlosses = np.zeros((K, L))
        log10carrier_freq = np.log10(fc / 1e6)
        for k in range(K):
            for l in range(L):
                h_bs = ap_heights[l]
                h_ut = ue_heights[k]

                reference_gain = (
                    46.3
                    + 33.9 * log10carrier_freq
                    - 13.82 * np.log10(h_bs)
                    - (1.1 * log10carrier_freq - 0.7) * h_ut
                    + (1.56 * log10carrier_freq - 0.8)
                )
                d_2d_km = distances_2d[k, l] / 1000
                d1_km = self.d1 / 1000
                d0_km = self.d0 / 1000
                if d_2d_km > d1_km:
                    pathloss_db = reference_gain + 35 * np.log10(d_2d_km)
                elif d_2d_km > d0_km and d_2d_km <= d1_km:
                    pathloss_db = (
                        reference_gain + 15 * np.log10(d1_km) + 20 * np.log10(d_2d_km)
                    )
                else:
                    pathloss_db = (
                        reference_gain + 15 * np.log10(d1_km) + 20 * np.log10(d0_km)
                    )
                pathlosses[k, l] = pathloss_db
        return pathlosses
