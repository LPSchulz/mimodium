"""Configure the number of UEs and their positions."""

import warnings

import numpy as np
from dagreon import task

from ..rng import Seed, get_rng
from ..warnings import ScenarioSizeWarning
from .area import AreaLength, AreaSize

#: Number of user equipments :math:`K` in the scenario.
type NumUEs = int
#: User equipment ground positions.
#:
#: :shape: ``(K, 2)``
#: :dtype: ``float``
type UEPositions = np.ndarray
#: User equipment heights.
#:
#: :shape: ``(K,)``
#: :dtype: ``float``
type UEHeights = np.ndarray
#: User equipment 3D coordinates, combining :type:`UEPositions` and :type:`UEHeights`.
#:
#: :shape: ``(K, 3)``
#: :dtype: ``float``
type UELocations = np.ndarray


@task
class CfgExplicitNumUEs:
    """Set the number of UEs :math:`K` explicitly to :code:`num_ues`."""

    num_ues: int

    def __post_init__(self):
        if self.num_ues <= 0:
            raise ValueError("num_ues must be positive")

    def __call__(self) -> NumUEs:
        return self.num_ues


@task
class CfgPoissonNumUEs:
    """Sample the number of UEs :math:`K` from a Poisson distribution
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

    def __call__(self, A: AreaSize, seed: Seed) -> NumUEs:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        K = int(rng.poisson(A * self.density / 1_000_000.0))
        if K == 0:
            warnings.warn(
                "Number of UEs drawn from the Poisson distribution is 0; "
                "the scenario area may be too small for the configured UE density. "
                "Setting the number of UEs to 1.",
                ScenarioSizeWarning,
                stacklevel=2,
            )
            K = 1
        return K


@task
class CfgExplicitUEPositions:
    """Set the ground positions of the UEs explicitly to :code:`positions`."""

    positions: np.ndarray

    def __post_init__(self):
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (K, 2)")
        if not np.all(np.isfinite(self.positions)):
            raise ValueError("positions must contain only finite values")

    def __call__(self) -> UEPositions:
        return self.positions


@task
class CfgUniformRandomUEPositions:
    """Draw the ground positions of UEs uniformly at random inside a square area.
    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`. Additionally, one can force the first UE to be at a specific
    position by setting :code:`forced_first_ue_position`. This does not change the
    overall distribution of the UE positions, since all other UE positions are drawn
    independently."""

    seed_override: int | None = None
    forced_first_ue_position: np.ndarray | None = None

    def __post_init__(self):
        if self.forced_first_ue_position is not None:
            if self.forced_first_ue_position.shape != (2,):
                raise ValueError("forced_first_ue_position must have shape (2,)")
            if not np.all(np.isfinite(self.forced_first_ue_position)):
                raise ValueError(
                    "forced_first_ue_position must contain only finite values"
                )

    def __call__(self, K: NumUEs, sqrt_A: AreaLength, seed: Seed) -> UEPositions:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        positions = sqrt_A * rng.random((K, 2))
        if self.forced_first_ue_position is not None:
            positions[0] = self.forced_first_ue_position
        return positions


@task
class CfgUEHeights:
    """Set the height of all UEs explicitly to :code:`height`."""

    height: float

    def __post_init__(self):
        if not np.isfinite(self.height) or self.height <= 0:
            raise ValueError("height must be finite and positive")

    def __call__(self, K: NumUEs) -> UEHeights:
        return np.full(K, self.height)


@task
class ComputeUELocations:
    """Combine UE ground positions with antenna heights to a 3D location."""

    def __call__(self, positions: UEPositions, heights: UEHeights) -> UELocations:
        return np.hstack((positions, heights[:, np.newaxis]))
