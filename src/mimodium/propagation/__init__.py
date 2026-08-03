"""Configure propagation conditions, including large-scale fading, small-scale fading,
and radio parameters."""

from ..warnings import ApplicabilityWarning
from . import (
    channel_generation,
    path_loss,
    radio_parameters,
    shadow_fading,
)
from .channel_generation import (
    ChannelRealizations,
    LargeScaleFadingCoefficients,
    NumRealizations,
    SpatialCorrelationMatrices,
)
from .path_loss import LineOfSightStates, PathLossdB
from .radio_parameters import (
    Bandwidth,
    CarrierFrequency,
    DLNoisePower,
    NumCoherenceSymbols,
    NumDownlinkSymbols,
    NumPilots,
    NumUplinkSymbols,
    ULNoisePower,
)
from .shadow_fading import ShadowFadingdB

__all__ = (
    "ChannelRealizations",
    "LargeScaleFadingCoefficients",
    "NumRealizations",
    "SpatialCorrelationMatrices",
    "Bandwidth",
    "CarrierFrequency",
    "DLNoisePower",
    "NumCoherenceSymbols",
    "NumDownlinkSymbols",
    "NumPilots",
    "NumUplinkSymbols",
    "ULNoisePower",
    "ApplicabilityWarning",
    "LineOfSightStates",
    "PathLossdB",
    "ShadowFadingdB",
)
