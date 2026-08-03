"""Configure the midhaul network connecting CPUs."""

import numpy as np
from dagreon import task
from scipy.spatial import Delaunay

from .area import AreaLength, WrapAround
from .cpu import CPUPositions, NumCPUs
from .geometry import CPUtoCPU2DDistances

#: Adjacency matrix representing presence of midhaul links between CPUs. Every CPU can
#: be connected to any number of other CPUs.
#:
#: :shape: ``(J, J)``
#: :dtype: ``bool``
type MidhaulLinks = np.ndarray


@task
class CfgFullMidhaulLinks:
    """Connect every distinct pair of CPUs."""

    def __call__(self, J: NumCPUs) -> MidhaulLinks:
        return ~np.diag([True] * J)


@task
class CfgFixedRadiusMidhaulLinks:
    """Connect each CPU to other CPUs within a horizontal radius."""

    radius: float

    def __post_init__(self):
        if not np.isfinite(self.radius) or self.radius < 0:
            raise ValueError("radius must be finite and non-negative")

    def __call__(self, J: NumCPUs, distances_2d: CPUtoCPU2DDistances) -> MidhaulLinks:
        midhaul_links = np.zeros((J, J), dtype=bool)
        midhaul_links[distances_2d < self.radius] = True
        np.fill_diagonal(midhaul_links, False)
        return midhaul_links


@task
class CfgDelaunayMidhaulLinks:
    """Connect CPUs using planar Delaunay neighbors."""

    def __call__(
        self,
        J: NumCPUs,
        cpu_pos: CPUPositions,
        wrap_around: WrapAround,
        sqrt_A: AreaLength,
    ) -> MidhaulLinks:
        if J < 4:
            # everything is connected
            return ~np.diag([True] * J)
        midhaul_links = np.zeros((J, J), dtype=bool)
        if wrap_around:
            cpu_pos = np.concatenate(
                [
                    cpu_pos,
                    cpu_pos + np.array([sqrt_A, 0]),
                    cpu_pos + np.array([-sqrt_A, 0]),
                    cpu_pos + np.array([0, sqrt_A]),
                    cpu_pos + np.array([0, -sqrt_A]),
                    cpu_pos + np.array([sqrt_A, sqrt_A]),
                    cpu_pos + np.array([-sqrt_A, -sqrt_A]),
                    cpu_pos + np.array([sqrt_A, -sqrt_A]),
                    cpu_pos + np.array([-sqrt_A, sqrt_A]),
                ]
            )
        tri = Delaunay(cpu_pos)
        for simplex in tri.simplices:
            midhaul_links[simplex[0] % J, simplex[1] % J] = True
            midhaul_links[simplex[1] % J, simplex[0] % J] = True
            midhaul_links[simplex[1] % J, simplex[2] % J] = True
            midhaul_links[simplex[2] % J, simplex[1] % J] = True
            midhaul_links[simplex[2] % J, simplex[0] % J] = True
            midhaul_links[simplex[0] % J, simplex[2] % J] = True
        np.fill_diagonal(midhaul_links, False)
        return midhaul_links


@task
class CfgNoMidhaulLinks:
    """Connect no CPUs."""

    def __call__(self, J: NumCPUs) -> MidhaulLinks:
        return np.zeros((J, J), dtype=bool)


def get_set_of_cpus_neighboring_cpu_j(j: int, midhaul: MidhaulLinks) -> np.ndarray:
    """Helper function to get an array of the CPU indices that are connected to
    CPU j."""
    return np.argwhere(midhaul[j, :]).flatten()
