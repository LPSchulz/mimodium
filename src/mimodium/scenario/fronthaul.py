"""Configure the fronthaul network connecting APs and CPUs."""

import numpy as np
from dagreon import task

from .ap import NumAPs
from .cpu import NumCPUs
from .geometry import APtoCPU2DDistances

#: Adjacency matrix representing presence of fronthaul links between APs and CPUs. Every
#: AP is connected to exactly one CPU.
#:
#: :shape: ``(L, J)``
#: :dtype: ``bool``
type FronthaulLinks = np.ndarray


@task
class CfgClosestFronthaulLinks:
    """Connect each AP to its nearest CPU."""

    def __call__(
        self, L: NumAPs, J: NumCPUs, distances_2d: APtoCPU2DDistances
    ) -> FronthaulLinks:
        fronthaul_links = np.zeros((L, J), dtype=bool)
        for l, cpu_of_ap in enumerate(np.argmin(distances_2d, axis=1)):
            fronthaul_links[l, cpu_of_ap] = True
        return fronthaul_links


def get_set_of_aps_connected_to_cpu_j(j: int, fronthaul: FronthaulLinks) -> np.ndarray:
    """Helper function to get an array of the AP indices that are connected to CPU j."""
    return np.argwhere(fronthaul[:, j]).flatten()


def get_cpu_connected_to_ap_l(l: int, fronthaul: FronthaulLinks) -> int:
    """Helper function to get the CPU index that is connected to AP l."""
    return np.argwhere(fronthaul[l, :]).item()
