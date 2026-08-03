"""Computations of distances and angles, taking wrap-around into account."""

import numpy as np
from dagreon import task

from .ap import APLocations, APPositions, NumAPs
from .area import AreaLength, WrapAround
from .cpu import CPUPositions, NumCPUs
from .ue import NumUEs, UELocations, UEPositions

#: 3D differences between :type:`UELocations` and :type:`APLocations` considering
#: :type:`WrapAround`.
#:
#: :shape: ``(K, L, 3)``
#: :dtype: ``float``
type UEtoAPDifferences = np.ndarray
#: 2D differences between :type:`APPositions` and :type:`CPUPositions` considering
#: :type:`WrapAround`.
#:
#: :shape: ``(L, J, 2)``
#: :dtype: ``float``
type APtoCPUDifferences = np.ndarray
#: 2D differences between :type:`CPUPositions` considering :type:`WrapAround`.
#:
#: :shape: ``(J, J, 2)``
#: :dtype: ``float``
type CPUtoCPUDifferences = np.ndarray
#: 2D distances from each UE to each AP considering :type:`WrapAround`.
#:
#: :shape: ``(K, L)``
#: :dtype: ``float``
type UEtoAP2DDistances = np.ndarray
#: 2D distances from each AP to each CPU considering :type:`WrapAround`.
#:
#: :shape: ``(L, J)``
#: :dtype: ``float``
type APtoCPU2DDistances = np.ndarray
#: 2D distances between CPUs considering :type:`WrapAround`.
#:
#: :shape: ``(J, J)``
#: :dtype: ``float``
type CPUtoCPU2DDistances = np.ndarray
#: 3D distances from each UE to each AP considering :type:`WrapAround`.
#:
#: :shape: ``(K, L)``
#: :dtype: ``float``
type UEtoAP3DDistances = np.ndarray
#: Horizontal distances between UEs considering :type:`WrapAround`.
#:
#: :shape: ``(K, K)``
#: :dtype: ``float``
type UEtoUE2DDistances = np.ndarray
#: Azimuth angles from each UE to each AP considering :type:`WrapAround`.
#:
#: :shape: ``(K, L)``
#: :dtype: ``float``
type UEtoAPAzimuthAngles = np.ndarray
#: Azimuth angles from APs toward CPUs considering :type:`WrapAround`.
#:
#: :shape: ``(L, J)``
#: :dtype: ``float``
type APtoCPUAzimuthAngles = np.ndarray
#: Azimuth angles between CPUs considering :type:`WrapAround`.
#:
#: :shape: ``(J, J)``
#: :dtype: ``float``
type CPUtoCPUAzimuthAngles = np.ndarray


def compute_wrapped_diff(diff: np.ndarray, g: float) -> None:
    diff[:, :, 0] = np.where(
        np.abs(diff[:, :, 0]) > g / 2,  # if wrapping around is shorter
        diff[:, :, 0] - np.sign(diff[:, :, 0]) * g,  # then wrap around
        diff[:, :, 0],  # else don't wrap around
    )
    diff[:, :, 1] = np.where(
        np.abs(diff[:, :, 1]) > g / 2,
        diff[:, :, 1] - np.sign(diff[:, :, 1]) * g,
        diff[:, :, 1],
    )


@task
class ComputeUEtoAPDifferences:
    """Compute the 3D differences between locations of UEs and APs taking into account
    possible wrap-around in the horizontal plane. The differences are useful to compute
    distances and azimuth angles."""

    def __call__(
        self,
        K: NumUEs,
        L: NumAPs,
        ap_loc: APLocations,
        ue_loc: UELocations,
        sqrt_A: AreaLength,
        wrap_around: WrapAround,
    ) -> UEtoAPDifferences:
        ap_repeated = np.transpose(
            np.tile(ap_loc, K).reshape(L, K, 3),
            (1, 0, 2),
        )
        ue_repeated = np.tile(ue_loc, L).reshape(K, L, 3)
        diff = ap_repeated - ue_repeated
        if wrap_around:
            compute_wrapped_diff(diff, sqrt_A)
        return diff


@task
class ComputeAPtoCPUDifferences:
    """Compute the 2D differences between locations of APs and CPUs taking into account
    possible wrap-around in the horizontal plane. The differences are useful to compute
    distances and azimuth angles."""

    def __call__(
        self,
        L: NumAPs,
        J: NumCPUs,
        ap_pos: APPositions,
        cpu_pos: CPUPositions,
        sqrt_A: AreaLength,
        wrap_around: WrapAround,
    ) -> APtoCPUDifferences:
        cpu_repeated = np.transpose(
            np.tile(cpu_pos, L).reshape(J, L, 2),
            (1, 0, 2),
        )
        ap_repeated = np.tile(ap_pos, J).reshape(L, J, 2)
        diff = cpu_repeated - ap_repeated
        if wrap_around:
            compute_wrapped_diff(diff, sqrt_A)
        return diff


@task
class ComputeCPUtoCPUDifferences:
    """Compute the 2D differences between locations of CPUs taking into account
    possible wrap-around in the horizontal plane."""

    def __call__(
        self,
        J: NumCPUs,
        cpu_pos: CPUPositions,
        sqrt_A: AreaLength,
        wrap_around: WrapAround,
    ) -> CPUtoCPUDifferences:
        cpu_repeated1 = np.transpose(
            np.tile(cpu_pos, J).reshape(J, J, 2),
            (1, 0, 2),
        )
        cpu_repeated2 = np.tile(cpu_pos, J).reshape(J, J, 2)
        diff = cpu_repeated1 - cpu_repeated2
        if wrap_around:
            compute_wrapped_diff(diff, sqrt_A)
        return diff


def _compute_2d_distances(diff: np.ndarray) -> np.ndarray:
    x_diff = diff[:, :, 0]
    y_diff = diff[:, :, 1]
    return np.sqrt(x_diff**2 + y_diff**2)


@task
class ComputeUEtoAP2DDistances:
    """Compute the horizontal distances from each UE to each AP."""

    def __call__(self, diff: UEtoAPDifferences) -> UEtoAP2DDistances:
        return _compute_2d_distances(diff)


@task
class ComputeAPtoCPU2DDistances:
    """Compute the horizontal distances from each AP to each CPU."""

    def __call__(self, diff: APtoCPUDifferences) -> APtoCPU2DDistances:
        return _compute_2d_distances(diff)


@task
class ComputeCPUtoCPU2DDistances:
    """Compute the horizontal distances from each CPU to each other CPU."""

    def __call__(self, diff: CPUtoCPUDifferences) -> CPUtoCPU2DDistances:
        return _compute_2d_distances(diff)


def _compute_3d_distances(diff: np.ndarray) -> np.ndarray:
    x_diff = diff[:, :, 0]
    y_diff = diff[:, :, 1]
    z_diff = diff[:, :, 2]
    return np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)


@task
class ComputeUEtoAP3DDistances:
    """Compute the 3D distances from each UE to each AP."""

    def __call__(self, diff: UEtoAPDifferences) -> UEtoAP3DDistances:
        return _compute_3d_distances(diff)


@task
class ComputeUEtoUE2DDistances:
    """Compute the symmetric matrix of horizontal distances between every pair of UEs.

    Wrap-around, when enabled, uses the shortest horizontal displacement per axis.
    """

    def __call__(
        self,
        ue_pos: UEPositions,
        sqrt_A: AreaLength,
        wrap_around: WrapAround,
    ) -> UEtoUE2DDistances:
        differences = ue_pos[:, np.newaxis, :] - ue_pos[np.newaxis, :, :]
        if wrap_around:
            compute_wrapped_diff(differences, sqrt_A)
        return _compute_2d_distances(differences)


def compute_azimuth_angles(diff: np.ndarray) -> np.ndarray:
    y_diff = diff[:, :, 1]
    x_diff = diff[:, :, 0]
    return np.arctan2(y_diff, x_diff)


@task
class ComputeUEtoAPAzimuthAngles:
    """Compute the azimuth angles from each UE to each AP."""

    def __call__(self, diff: UEtoAPDifferences) -> UEtoAPAzimuthAngles:
        return compute_azimuth_angles(diff)


@task
class ComputeAPtoCPUAzimuthAngles:
    """Compute the azimuth angles from each AP to each CPU."""

    def __call__(self, diff: APtoCPUDifferences) -> APtoCPUAzimuthAngles:
        return compute_azimuth_angles(diff)


@task
class ComputeCPUtoCPUAzimuthAngles:
    """Compute the azimuth angles from each CPU to each other CPU."""

    def __call__(self, diff: CPUtoCPUDifferences) -> CPUtoCPUAzimuthAngles:
        return compute_azimuth_angles(diff)
