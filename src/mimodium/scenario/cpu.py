"""Configure the number of CPUs and their positions."""

import warnings

import numpy as np
from dagreon import task

from ..rng import Seed, get_rng
from ..warnings import ScenarioSizeWarning
from .ap import APPositions, NumAPs
from .area import AreaLength, AreaSize

#: Number of central processing units :math:`J` in the scenario.
type NumCPUs = int
#: CPU ground positions.
#:
#: :shape: ``(J, 2)``
#: :dtype: ``float``
type CPUPositions = np.ndarray


@task
class CfgExplicitNumCPUs:
    """Set the number of CPUs :math:`J` explicitly to :code:`num_cpus`."""

    num_cpus: int

    def __post_init__(self):
        if self.num_cpus <= 0:
            raise ValueError("num_cpus must be positive")

    def __call__(self) -> NumCPUs:
        return self.num_cpus


@task
class CfgSameAsNumAPs:
    """Use the same number of CPUs as the number of APs. Useful for scenarios where
    each AP has a dedicated CPU."""

    def __call__(self, L: NumAPs) -> NumCPUs:
        return L


@task
class CfgPoissonNumCPUs:
    """Sample the number of CPUs :math:`J` from a Poisson distribution
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

    def __call__(self, A: AreaSize, seed: Seed) -> NumCPUs:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        J = int(rng.poisson(A * self.density / 1_000_000.0))
        if J == 0:
            warnings.warn(
                "Number of CPUs drawn from the Poisson distribution is 0; "
                "the scenario area may be too small for the configured CPU density. "
                "Setting the number of CPUs to 1.",
                ScenarioSizeWarning,
                stacklevel=2,
            )
            J = 1
        return J


@task
class CfgExplicitCPUPositions:
    """Set the ground positions of the CPUs explicitly to :code:`positions`."""

    positions: np.ndarray

    def __post_init__(self):
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (J, 2)")
        if not np.all(np.isfinite(self.positions)):
            raise ValueError("positions must contain only finite values")

    def __call__(self) -> CPUPositions:
        return self.positions


@task
class CfgEvenlySpacedCPUPositions:
    """Place CPUs on a square lattice. Only works if the number of CPUs :math:`J` is a
    perfect square."""

    def __call__(self, J: NumCPUs, sqrt_A: AreaLength) -> CPUPositions:
        if np.sqrt(J) % 1 != 0:
            raise ValueError("NumCPUs must be a perfect square")
        num_per_dim = int(np.sqrt(J))
        dist = sqrt_A / num_per_dim
        x_pos = np.tile(np.arange(dist / 2, sqrt_A, dist), num_per_dim)
        y_pos = np.repeat(np.arange(dist / 2, sqrt_A, dist), num_per_dim)
        return np.vstack((x_pos, y_pos)).T


@task
class CfgUniformRandomCPUPositions:
    """Draw the ground positions of CPUs uniformly at random inside a square area.
    The :type:`Seed` for the random number generator can be overridden with
    :code:`seed_override`. Additionally, one can force the first CPU to be at a specific
    position by setting :code:`forced_first_cpu_position`. This does not change the
    overall distribution of the CPU positions, since all other CPU positions are drawn
    independently."""

    seed_override: int | None = None
    forced_first_cpu_position: np.ndarray | None = None

    def __post_init__(self):
        if self.forced_first_cpu_position is not None:
            if self.forced_first_cpu_position.shape != (2,):
                raise ValueError("forced_first_cpu_position must have shape (2,)")
            if not np.all(np.isfinite(self.forced_first_cpu_position)):
                raise ValueError(
                    "forced_first_cpu_position must contain only finite values"
                )

    def __call__(self, J: NumCPUs, sqrt_A: AreaLength, seed: Seed) -> CPUPositions:
        if self.seed_override is not None:
            seed = self.seed_override
        rng = get_rng(seed, self)
        positions = sqrt_A * rng.random((J, 2))
        if self.forced_first_cpu_position is not None:
            positions[0] = self.forced_first_cpu_position
        return positions


@task
class CfgSameAsAPPositions:
    """Use the same positions for CPUs as the APs. Useful for scenarios where
    each AP has a dedicated CPU."""

    def __call__(self, ap_pos: APPositions) -> CPUPositions:
        return ap_pos.copy()
