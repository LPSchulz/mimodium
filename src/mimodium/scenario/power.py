"""Configure maximum transmit powers from user inputs in milliwatts."""

import numpy as np
from dagreon import task

from .ap import NumAPs
from .ue import NumUEs

#: Maximum pilot transmit power :math:`P^{\text{pilot}}` per UE in watts.
type PilotMaxPower = np.ndarray  # shape=(K,), dtype=float
#: Maximum uplink transmit power :math:`P^{\text{ul}}` per UE in watts.
type UEMaxPower = np.ndarray  # shape=(K,), dtype=float
#: Maximum downlink transmit power :math:`P^{\text{ap}}` per AP in watts.
type APMaxPower = np.ndarray  # shape=(L,), dtype=float


@task
class CfgPilotMaxPower:
    r"""Set the maximum pilot transmit power :math:`P^{\text{pilot}}` from
    :code:`max_power_mw` in milliwatts for each UE. This is an average power constraint,
    meaning it ensures for each UE :math:`k` that the average power of the pilot signal
    :math:`\phi_k` satisfies
    :math:`\mathbb{E}\{|\phi_k|^2\} \leq P^{\text{pilot}}_k`.
    """

    max_power_mw: float

    def __post_init__(self):
        if not np.isfinite(self.max_power_mw) or self.max_power_mw < 0:
            raise ValueError("max_power_mw must be finite and non-negative")

    def __call__(self, K: NumUEs) -> PilotMaxPower:
        return np.full(K, self.max_power_mw / 1_000.0, dtype=float)


@task
class CfgUEMaxPower:
    r"""Set the maximum uplink transmit power :math:`P^{\text{ul}}` from
    :code:`max_power_mw` in milliwatts for each UE. This is an average power constraint,
    meaning it ensures for each UE :math:`k` that the average power of the uplink signal
    :math:`s_k` satisfies
    :math:`\mathbb{E}\{|s_k|^2\} \leq P^{\text{ul}}_k`.
    """

    max_power_mw: float

    def __post_init__(self):
        if not np.isfinite(self.max_power_mw) or self.max_power_mw < 0:
            raise ValueError("max_power_mw must be finite and non-negative")

    def __call__(self, K: NumUEs) -> UEMaxPower:
        return np.full(K, self.max_power_mw / 1_000.0, dtype=float)


@task
class CfgAPMaxPower:
    r"""Set the maximum downlink transmit power :math:`P^{\text{ap}}` from
    :code:`max_power_mw` in milliwatts for each AP. This is an average power constraint,
    meaning it ensures for each AP :math:`l` that the average power of the sent signal
    :math:`y_l` satisfies
    :math:`\mathbb{E}\{|y_l|^2\} \leq P^{\text{ap}}_l`. Note, that the downlink signal
    may be designed at a CPU. In this case, the CPU must ensure that the average power
    of the part of the signal sent by AP :math:`l` satisfies the constraint.
    """

    max_power_mw: float

    def __post_init__(self):
        if not np.isfinite(self.max_power_mw) or self.max_power_mw < 0:
            raise ValueError("max_power_mw must be finite and non-negative")

    def __call__(self, L: NumAPs) -> APMaxPower:
        return np.full(L, self.max_power_mw / 1_000.0, dtype=float)
