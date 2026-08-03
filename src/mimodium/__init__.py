from importlib.metadata import PackageNotFoundError, version

from . import rng
from .warnings import ApplicabilityWarning, ScenarioSizeWarning

try:
    __version__ = version("mimodium")
except PackageNotFoundError:
    __version__ = "unknown"


def core_tasks() -> tuple:
    """Return fresh instances of Mimodium's deterministic core tasks."""
    from . import algorithms, evaluation, propagation, scenario

    return (
        scenario.ap.ComputeTotalNumberAntennas(),
        scenario.ap.ComputeAPLocations(),
        scenario.ue.ComputeUELocations(),
        scenario.area.ComputeAreaSize(),
        scenario.geometry.ComputeUEtoAPDifferences(),
        scenario.geometry.ComputeAPtoCPUDifferences(),
        scenario.geometry.ComputeCPUtoCPUDifferences(),
        scenario.geometry.ComputeUEtoAP2DDistances(),
        scenario.geometry.ComputeAPtoCPU2DDistances(),
        scenario.geometry.ComputeCPUtoCPU2DDistances(),
        scenario.geometry.ComputeUEtoAP3DDistances(),
        scenario.geometry.ComputeUEtoUE2DDistances(),
        scenario.geometry.ComputeUEtoAPAzimuthAngles(),
        scenario.geometry.ComputeAPtoCPUAzimuthAngles(),
        scenario.geometry.ComputeCPUtoCPUAzimuthAngles(),
        propagation.channel_generation.ComputeLargeScaleFading(),
        propagation.path_loss.ComputeAlwaysNonLineOfSight(),
        propagation.radio_parameters.ComputeNumCoherenceSymbols(),
        algorithms.channel_estimation.ComputeReceivedSignals(),
        algorithms.access.ComputeMasterAP(),
        algorithms.access.ComputeMasterCPU(),
        algorithms.uplink_combining.ComputeCombiningNormSquares(),
        algorithms.uplink_combining.ComputeExpectedCombiningNormSquares(),
        algorithms.uplink_fusion.ComputeEstimatedEffectiveULChannels(),
        algorithms.uplink_fusion.ComputeEffectiveULChannels(),
        algorithms.uplink_fusion.ComputeExpectedEffectiveULChannels(),
        algorithms.uplink_fusion.ComputeExpectedEffectiveULChannelOuters(),
        algorithms.uplink_fusion.ComputeKnownEffectiveULChannels(),
        algorithms.uplink_fusion.ComputeUnknownEffectiveULChannels(),
        algorithms.uplink_fusion.ComputeConditionalUnknownEffectiveULChannelOuters(),
        algorithms.downlink_precoding.ComputeExpectedPrecodingNormSquares(),
        algorithms.downlink_precoding.ComputeEffectiveDLChannels(),
        algorithms.downlink_precoding.ComputeExpectedEffectiveDLChannelOuters(),
        algorithms.downlink_precoding.ComputeExpectedDesiredEffectiveDLChannels(),
        evaluation.asymptotics.ComputeChannelHardening(),
        evaluation.asymptotics.ComputeFavorablePropagation(),
        evaluation.asymptotics.ComputeEffectiveUplinkChannelHardening(),
        evaluation.asymptotics.ComputeEffectiveDownlinkChannelHardening(),
        evaluation.asymptotics.ComputeEffectiveUplinkFavorablePropagation(),
        evaluation.asymptotics.ComputeEffectiveDownlinkFavorablePropagation(),
        evaluation.set_sizes.ComputeNumMeasuredUEsPerAP(),
        evaluation.set_sizes.ComputeNumMeasuringAPsPerUE(),
        evaluation.set_sizes.ComputeNumMeasuredUEsPerCPU(),
        evaluation.set_sizes.ComputeNumMeasuringCPUsPerUE(),
        evaluation.set_sizes.ComputeNumCandidateUEsPerAP(),
        evaluation.set_sizes.ComputeNumCandidateAPsPerUE(),
        evaluation.set_sizes.ComputeNumCandidateUEsPerCPU(),
        evaluation.set_sizes.ComputeNumCandidateCPUsPerUE(),
        evaluation.set_sizes.ComputeNumServedUEsPerAP(),
        evaluation.set_sizes.ComputeNumServingAPsPerUE(),
        evaluation.set_sizes.ComputeNumServedUEsPerCPU(),
        evaluation.set_sizes.ComputeNumServingCPUsPerUE(),
        evaluation.set_sizes.ComputeNumEstimatedUEsPerAP(),
        evaluation.set_sizes.ComputeNumEstimatingAPsPerUE(),
        evaluation.set_sizes.ComputeNumEstimatedUEsPerCPU(),
        evaluation.set_sizes.ComputeNumEstimatingCPUsPerUE(),
        evaluation.set_sizes.ComputeNumAPsPerCPU(),
    )
