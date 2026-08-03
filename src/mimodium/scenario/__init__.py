"""Configure the simulation scenario, including area size, entities, and power limits."""

from . import ap, area, cpu, fronthaul, geometry, midhaul, power, ue
from .ap import (
    APAntennaSpacing,
    APHeights,
    APLocations,
    APPositions,
    APArrayOrientations,
    NumAntennasPerAP,
    NumAPs,
    TotalNumAPAntennas,
)
from .area import AreaLength, AreaSize, WrapAround
from .cpu import CPUPositions, NumCPUs
from .fronthaul import FronthaulLinks
from .geometry import (
    APtoCPU2DDistances,
    APtoCPUAzimuthAngles,
    APtoCPUDifferences,
    CPUtoCPU2DDistances,
    CPUtoCPUAzimuthAngles,
    CPUtoCPUDifferences,
    UEtoAP2DDistances,
    UEtoAP3DDistances,
    UEtoAPAzimuthAngles,
    UEtoAPDifferences,
    UEtoUE2DDistances,
)
from .midhaul import MidhaulLinks
from .power import (
    APMaxPower,
    PilotMaxPower,
    UEMaxPower,
)
from .ue import NumUEs, UEHeights, UELocations, UEPositions

__all__ = (
    "APAntennaSpacing",
    "APHeights",
    "APLocations",
    "APPositions",
    "APArrayOrientations",
    "NumAntennasPerAP",
    "NumAPs",
    "TotalNumAPAntennas",
    "AreaLength",
    "AreaSize",
    "WrapAround",
    "CPUPositions",
    "NumCPUs",
    "FronthaulLinks",
    "APtoCPU2DDistances",
    "APtoCPUAzimuthAngles",
    "APtoCPUDifferences",
    "CPUtoCPU2DDistances",
    "CPUtoCPUAzimuthAngles",
    "CPUtoCPUDifferences",
    "UEtoAP2DDistances",
    "UEtoAP3DDistances",
    "UEtoAPAzimuthAngles",
    "UEtoAPDifferences",
    "UEtoUE2DDistances",
    "MidhaulLinks",
    "APMaxPower",
    "PilotMaxPower",
    "UEMaxPower",
    "NumUEs",
    "UEHeights",
    "UELocations",
    "UEPositions",
)
