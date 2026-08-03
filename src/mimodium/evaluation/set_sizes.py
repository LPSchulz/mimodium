"""Compute cardinalities of access-association and fronthaul sets."""

import numpy as np
from dagreon import task

from ..algorithms import (
    CandidateWirelessLinks,
    EstimatedChannelLinks,
    MeasuredStatisticLinks,
    UsedWirelessLinks,
)
from ..scenario import FronthaulLinks

#: Number of UEs measured by each AP.
type NumMeasuredUEsPerAP = np.ndarray
#: Number of APs measuring each UE.
type NumMeasuringAPsPerUE = np.ndarray
#: Number of UEs measured by APs connected to each CPU.
type NumMeasuredUEsPerCPU = np.ndarray
#: Number of CPUs connected to at least one AP measuring each UE.
type NumMeasuringCPUsPerUE = np.ndarray
#: Number of candidate UEs at each AP.
type NumCandidateUEsPerAP = np.ndarray
#: Number of candidate APs for each UE.
type NumCandidateAPsPerUE = np.ndarray
#: Number of candidate UEs at APs connected to each CPU.
type NumCandidateUEsPerCPU = np.ndarray
#: Number of CPUs connected to at least one candidate AP for each UE.
type NumCandidateCPUsPerUE = np.ndarray
#: Number of UEs served by each AP.
type NumServedUEsPerAP = np.ndarray
#: Number of APs serving each UE.
type NumServingAPsPerUE = np.ndarray
#: Number of UEs served by each CPU.
type NumServedUEsPerCPU = np.ndarray
#: Number of CPUs serving each UE.
type NumServingCPUsPerUE = np.ndarray
#: Number of UEs whose channels are estimated by each AP.
type NumEstimatedUEsPerAP = np.ndarray
#: Number of APs estimating each UE's channel.
type NumEstimatingAPsPerUE = np.ndarray
#: Number of UEs estimated by APs connected to each CPU.
type NumEstimatedUEsPerCPU = np.ndarray
#: Number of CPUs connected to at least one AP estimating each UE's channel.
type NumEstimatingCPUsPerUE = np.ndarray
#: Number of APs connected to each CPU.
type NumAPsPerCPU = np.ndarray


@task
class ComputeNumMeasuredUEsPerAP:
    """Count the UEs measured by each AP."""

    def __call__(self, measured: MeasuredStatisticLinks) -> NumMeasuredUEsPerAP:
        return np.sum(measured, axis=0)


@task
class ComputeNumMeasuringAPsPerUE:
    """Count the APs measuring each UE."""

    def __call__(self, measured: MeasuredStatisticLinks) -> NumMeasuringAPsPerUE:
        return np.sum(measured, axis=1)


@task
class ComputeNumMeasuredUEsPerCPU:
    """Count the UEs measured by at least one AP connected to each CPU."""

    def __call__(
        self, measured: MeasuredStatisticLinks, fronthaul: FronthaulLinks
    ) -> NumMeasuredUEsPerCPU:
        return np.sum(_get_associated_ue_cpu_links(measured, fronthaul), axis=0)


@task
class ComputeNumMeasuringCPUsPerUE:
    """Count the CPUs connected to at least one AP measuring each UE."""

    def __call__(
        self, measured: MeasuredStatisticLinks, fronthaul: FronthaulLinks
    ) -> NumMeasuringCPUsPerUE:
        return np.sum(_get_associated_ue_cpu_links(measured, fronthaul), axis=1)


@task
class ComputeNumCandidateUEsPerAP:
    """Count the candidate UEs at each AP."""

    def __call__(self, candidate: CandidateWirelessLinks) -> NumCandidateUEsPerAP:
        return np.sum(candidate, axis=0)


@task
class ComputeNumCandidateAPsPerUE:
    """Count the candidate APs for each UE."""

    def __call__(self, candidate: CandidateWirelessLinks) -> NumCandidateAPsPerUE:
        return np.sum(candidate, axis=1)


@task
class ComputeNumCandidateUEsPerCPU:
    """Count the candidate UEs at APs connected to each CPU."""

    def __call__(
        self, candidate: CandidateWirelessLinks, fronthaul: FronthaulLinks
    ) -> NumCandidateUEsPerCPU:
        return np.sum(_get_associated_ue_cpu_links(candidate, fronthaul), axis=0)


@task
class ComputeNumCandidateCPUsPerUE:
    """Count the CPUs connected to at least one candidate AP for each UE."""

    def __call__(
        self, candidate: CandidateWirelessLinks, fronthaul: FronthaulLinks
    ) -> NumCandidateCPUsPerUE:
        return np.sum(_get_associated_ue_cpu_links(candidate, fronthaul), axis=1)


@task
class ComputeNumServedUEsPerAP:
    """Count the UEs served by each AP."""

    def __call__(self, used: UsedWirelessLinks) -> NumServedUEsPerAP:
        return np.sum(used, axis=0)


@task
class ComputeNumServingAPsPerUE:
    """Count the APs serving each UE."""

    def __call__(self, used: UsedWirelessLinks) -> NumServingAPsPerUE:
        return np.sum(used, axis=1)


@task
class ComputeNumServedUEsPerCPU:
    """Count the UEs served by at least one AP connected to each CPU."""

    def __call__(
        self, used: UsedWirelessLinks, fronthaul: FronthaulLinks
    ) -> NumServedUEsPerCPU:
        return np.sum(_get_associated_ue_cpu_links(used, fronthaul), axis=0)


@task
class ComputeNumServingCPUsPerUE:
    """Count the CPUs connected to at least one AP serving each UE."""

    def __call__(
        self, used: UsedWirelessLinks, fronthaul: FronthaulLinks
    ) -> NumServingCPUsPerUE:
        return np.sum(_get_associated_ue_cpu_links(used, fronthaul), axis=1)


@task
class ComputeNumEstimatedUEsPerAP:
    """Count the UEs whose channels are estimated by each AP."""

    def __call__(self, estimated: EstimatedChannelLinks) -> NumEstimatedUEsPerAP:
        return np.sum(estimated, axis=0)


@task
class ComputeNumEstimatingAPsPerUE:
    """Count the APs estimating each UE's channel."""

    def __call__(self, estimated: EstimatedChannelLinks) -> NumEstimatingAPsPerUE:
        return np.sum(estimated, axis=1)


@task
class ComputeNumEstimatedUEsPerCPU:
    """Count the UEs estimated by at least one AP connected to each CPU."""

    def __call__(
        self, estimated: EstimatedChannelLinks, fronthaul: FronthaulLinks
    ) -> NumEstimatedUEsPerCPU:
        return np.sum(_get_associated_ue_cpu_links(estimated, fronthaul), axis=0)


@task
class ComputeNumEstimatingCPUsPerUE:
    """Count the CPUs connected to at least one AP estimating each UE."""

    def __call__(
        self, estimated: EstimatedChannelLinks, fronthaul: FronthaulLinks
    ) -> NumEstimatingCPUsPerUE:
        return np.sum(_get_associated_ue_cpu_links(estimated, fronthaul), axis=1)


@task
class ComputeNumAPsPerCPU:
    """Count the APs connected to each CPU."""

    def __call__(self, fronthaul: FronthaulLinks) -> NumAPsPerCPU:
        return np.sum(fronthaul, axis=0)


def _get_associated_ue_cpu_links(
    association: np.ndarray, fronthaul: FronthaulLinks
) -> np.ndarray:
    return association @ fronthaul > 0
