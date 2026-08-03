"""Configure the number of APs, their antenna array, and their positions."""

import warnings

import numpy as np
from dagreon import task

from ..rng import Seed, get_rng
from ..warnings import ScenarioSizeWarning
from .area import AreaLength, AreaSize

#: Number of access points :math:`L` in the scenario.
type NumAPs = int
#: Access point ground positions.
#:
#: :shape: ``(L, 2)``
#: :dtype: ``float``
type APPositions = np.ndarray
#: Access point heights.
#:
#: :shape: ``(L,)``
#: :dtype: ``float``
type APHeights = np.ndarray
#: Access point 3D coordinates, combining :type:`APPositions` and :type:`APHeights`.
#:
#: :shape: ``(L, 3)``
#: :dtype: ``float``
type APLocations = np.ndarray
#: Number of antennas :math:`N` per access point.
type NumAntennasPerAP = int
#: Total number of access point antennas, multiplying :type:`NumAPs` and
#: :type:`NumAntennasPerAP`.
type TotalNumAPAntennas = int
#: Inter-element antenna spacing in wavelengths for the uniform linear array.
type APAntennaSpacing = float
#: Access point array azimuth angles.
#:
#: :shape: ``(L,)``
#: :dtype: ``float``
type APArrayOrientations = np.ndarray


@task
class CfgExplicitNumAPs:
    """Set the number of APs :math:`L` explicitly to :code:`num_aps`."""

    num_aps: int

    def __post_init__(self):
        if self.num_aps <= 0:
            raise ValueError("num_aps must be positive")

    def __call__(self) -> NumAPs:
        return self.num_aps


@task
class CfgNumAntennas:
    """Set the number of antennas :math:`N` per AP explicitly to
    :code:`num_antennas`."""

    num_antennas: int

    def __post_init__(self):
        if self.num_antennas <= 0:
            raise ValueError("num_antennas must be positive")

    def __call__(self) -> NumAntennasPerAP:
        return self.num_antennas


@task
class CfgAntennaSpacing:
    """Set the inter-element antenna spacing in wavelengths for the uniform linear array
    of each AP explicitly to :code:`spacing`."""

    spacing: float

    def __post_init__(self):
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("spacing must be finite and positive")

    def __call__(self) -> APAntennaSpacing:
        return self.spacing


@task
class ComputeTotalNumberAntennas:
    r"""Computes the total number of AP antennas :math:`L \cdot N`."""

    def __call__(self, L: NumAPs, N: NumAntennasPerAP) -> TotalNumAPAntennas:
        return L * N


@task
class CfgAlignedArrayOrientations:
    """Set the azimuth angle of every AP uniform linear array explicitly to
    :code:`azimuth`."""

    azimuth: float = 0.0

    def __post_init__(self):
        if not 0 <= self.azimuth < 2 * np.pi:
            raise ValueError("azimuth must be in the range [0, 2π)")

    def __call__(self, L: NumAPs) -> APArrayOrientations:
        angles = np.full(L, self.azimuth, dtype=float)
        return angles


@task
class CfgRandomAzimuthArrayOrientations:
    """Draw the azimuth angles of AP uniform linear arrays from a uniform distribution
    over.
    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`."""

    seed_override: int | None = None

    def __call__(self, L: NumAPs, seed: Seed) -> APArrayOrientations:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        return rng.random(L) * 2 * np.pi


@task
class CfgPoissonNumAPs:
    """Sample the number of APs :math:`L` from a Poisson distribution
    corresponding to a Poisson Point Process (PPP) with density :code:`density` per
    square kilometer.
    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`.
    """

    density: float
    seed_override: int | None = None

    def __post_init__(self):
        if not np.isfinite(self.density) or self.density < 0:
            raise ValueError("density must be finite and non-negative")

    def __call__(self, A: AreaSize, seed: Seed) -> NumAPs:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        L = int(rng.poisson(A * self.density / 1_000_000.0))
        if L == 0:
            warnings.warn(
                "Number of APs drawn from the Poisson distribution is 0; "
                "the scenario area may be too small for the configured AP density. "
                "Setting the number of APs to 1.",
                ScenarioSizeWarning,
                stacklevel=2,
            )
            L = 1
        return L


@task
class CfgExplicitAPPositions:
    """Set the ground positions of the APs explicitly to :code:`positions`."""

    positions: np.ndarray

    def __post_init__(self):
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (L, 2)")
        if not np.all(np.isfinite(self.positions)):
            raise ValueError("positions must contain only finite values")

    def __call__(self) -> APPositions:
        return self.positions


@task
class CfgEvenlySpacedAPPositions:
    """Place APs on a square lattice. Only works if the number of APs :math:`L` is a
    perfect square."""

    def __call__(self, L: NumAPs, sqrt_A: AreaLength) -> APPositions:
        if np.sqrt(L) % 1 != 0:
            raise ValueError("NumAPs must be a perfect square")
        num_per_dim = int(np.sqrt(L))
        dist = sqrt_A / num_per_dim
        x_pos = np.tile(np.arange(dist / 2, sqrt_A, dist), num_per_dim)
        y_pos = np.repeat(np.arange(dist / 2, sqrt_A, dist), num_per_dim)
        return np.vstack((x_pos, y_pos)).T


@task
class CfgUniformRandomAPPositions:
    """Draw the ground positions of APs uniformly at random inside a square area.
    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`. Additionally, one can force the first AP to be at a specific
    position by setting :code:`forced_first_ap_position`. This does not change the
    overall distribution of the AP positions, since all other AP positions are drawn
    independently."""

    seed_override: int | None = None
    forced_first_ap_position: np.ndarray | None = None

    def __post_init__(self):
        if self.forced_first_ap_position is not None:
            if self.forced_first_ap_position.shape != (2,):
                raise ValueError("forced_first_ap_position must have shape (2,)")
            if not np.all(np.isfinite(self.forced_first_ap_position)):
                raise ValueError(
                    "forced_first_ap_position must contain only finite values"
                )

    def __call__(self, L: NumAPs, sqrt_A: AreaLength, seed: Seed) -> APPositions:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        positions = sqrt_A * rng.random((L, 2))
        if self.forced_first_ap_position is not None:
            positions[0] = self.forced_first_ap_position
        return positions


@task
class CfgAPHeights:
    """Set the height of all APs explicitly to :code:`height`."""

    height: float

    def __post_init__(self):
        if not np.isfinite(self.height) or self.height <= 0:
            raise ValueError("height must be finite and positive")

    def __call__(self, L: NumAPs) -> APHeights:
        return np.full(L, self.height)


@task
class ComputeAPLocations:
    """Combine AP ground positions with antenna heights to a 3D location."""

    def __call__(self, positions: APPositions, heights: APHeights) -> APLocations:
        return np.hstack((positions, heights[:, np.newaxis]))
