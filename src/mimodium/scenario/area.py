"""Configure the size of the simulation area with optional wrap-around."""

import numpy as np
from dagreon import task

#: Side length :math:`\sqrt{A}` of the square simulation area in meters.
type AreaLength = float
#: Size :math:`A` of the square simulation area in square meters.
type AreaSize = float
#: Whether horizontal geometry uses toroidal boundary conditions.
type WrapAround = bool


@task
class CfgAreaLength:
    r"""Set the side length :math:`\sqrt{A}` of the simulation area in meters explicitly
    to :code:`area_length`."""

    area_length: float

    def __post_init__(self):
        if not np.isfinite(self.area_length) or self.area_length <= 0:
            raise ValueError("area_length must be finite and positive")

    def __call__(self) -> AreaLength:
        return self.area_length


@task
class ComputeAreaSize:
    """Compute the area size :math:`A` from the side length in square meters."""

    def __call__(self, area_length: AreaLength) -> AreaSize:
        return area_length**2


@task
class CfgWrapAround:
    """Specify whether horizontal geometry uses toroidal boundary conditions explicitly
    to :code:`wrap_around`."""

    wrap_around: bool

    def __call__(self) -> WrapAround:
        return self.wrap_around
