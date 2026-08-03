"""Configure the carrier frequency, bandwidth, receiver noise, and coherence block."""

import numpy as np
from dagreon import task

#: Number of pilot symbols :math:`\tau_p` in one coherence block.
type NumPilots = int
#: Number of uplink data symbols :math:`\tau_u` in one coherence block.
type NumUplinkSymbols = int
#: Number of downlink data symbols :math:`\tau_d` in one coherence block.
type NumDownlinkSymbols = int
#: Sum of :type:`NumPilots`, :type:`NumUplinkSymbols`, and
#: :type:`NumDownlinkSymbols`, denoted by :math:`\tau_c`.
type NumCoherenceSymbols = int

#: Carrier frequency :math:`f_c` in Hertz.
type CarrierFrequency = float
#: Communication bandwidth :math:`B` in Hertz.
type Bandwidth = float
#: Downlink receiver-noise power :math:`\sigma_{\mathrm{dl}}^2` over
#: :type:`Bandwidth`, in watts.
type DLNoisePower = float
#: Uplink receiver-noise power :math:`\sigma_{\mathrm{ul}}^2` over
#: :type:`Bandwidth`, in watts.
type ULNoisePower = float


@task
class CfgNumPilots:
    r"""Set the number of pilot symbols :math:`\tau_p` explicitly to
    :code:`tau_p`."""

    tau_p: int

    def __post_init__(self):
        if self.tau_p <= 0:
            raise ValueError("tau_p must be positive")

    def __call__(self) -> NumPilots:
        return self.tau_p


@task
class CfgNumUplinkSymbols:
    r"""Set the number of uplink data symbols :math:`\tau_u` explicitly to
    :code:`tau_u`."""

    tau_u: int

    def __post_init__(self):
        if self.tau_u < 0:
            raise ValueError("tau_u must be non-negative")

    def __call__(self) -> NumUplinkSymbols:
        return self.tau_u


@task
class CfgNumDownlinkSymbols:
    r"""Set the number of downlink data symbols :math:`\tau_d` explicitly to
    :code:`tau_d`."""

    tau_d: int

    def __post_init__(self):
        if self.tau_d < 0:
            raise ValueError("tau_d must be non-negative")

    def __call__(self) -> NumDownlinkSymbols:
        return self.tau_d


@task
class ComputeNumCoherenceSymbols:
    r"""Compute the coherence-block length
    :math:`\tau_c = \tau_p + \tau_u + \tau_d`."""

    def __call__(
        self, tau_u: NumUplinkSymbols, tau_d: NumDownlinkSymbols, tau_p: NumPilots
    ) -> NumCoherenceSymbols:
        return tau_u + tau_d + tau_p


@task
class CfgCarrierFrequency:
    """Set the carrier frequency :math:`f_c` in Hertz explicitly to :code:`f_c`."""

    f_c: float

    def __post_init__(self):
        if not np.isfinite(self.f_c) or self.f_c <= 0:
            raise ValueError("f_c must be finite and positive")

    def __call__(self) -> CarrierFrequency:
        return self.f_c


@task
class CfgBandwidth:
    """Set the communication bandwidth :math:`B` in Hertz explicitly to :code:`B`."""

    B: float

    def __post_init__(self):
        if not np.isfinite(self.B) or self.B <= 0:
            raise ValueError("B must be finite and positive")

    def __call__(self) -> Bandwidth:
        return self.B


@task
class CfgDLNoisePower:
    r"""Compute the downlink receiver-noise power
    :math:`\sigma_{\mathrm{dl}}^2` in watts.

    Set the receiver temperature :math:`T` in kelvin explicitly to
    :code:`temperature`. Set the receiver noise figure or any other constant loss in
    decibels explicitly to :code:`constant_offset_db`. With Boltzmann constant
    :math:`k_B`, the receiver-noise power is

    .. math::

       \sigma_{\mathrm{dl}}^2
       = k_B T B\,10^{\mathtt{constant\_offset\_db}/10}.

    With the default values :code:`temperature=300.0` and
    :code:`constant_offset_db=7.0`, a bandwidth of 20 MHz gives approximately
    :math:`-93.82\,\mathrm{dBm}`.
    """

    temperature: float = 300.0
    constant_offset_db: float = 7.0

    def __post_init__(self):
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(self.constant_offset_db):
            raise ValueError("constant_offset_db must be finite")

    def __call__(self, B: Bandwidth) -> DLNoisePower:
        k_B = 1.380649e-23
        thermal_noise_power = self.temperature * B * k_B
        return thermal_noise_power * 10 ** (self.constant_offset_db / 10)


@task
class CfgULNoisePower:
    r"""Compute the uplink receiver-noise power
    :math:`\sigma_{\mathrm{ul}}^2` in watts.

    Set the receiver temperature :math:`T` in kelvin explicitly to
    :code:`temperature`. Set the receiver noise figure or any other constant loss in
    decibels explicitly to :code:`constant_offset_db`. With Boltzmann constant
    :math:`k_B`, the receiver-noise power is

    .. math::

       \sigma_{\mathrm{ul}}^2
       = k_B T B\,10^{\mathtt{constant\_offset\_db}/10}.

    With the default values :code:`temperature=300.0` and
    :code:`constant_offset_db=7.0`, a bandwidth of 20 MHz gives approximately
    :math:`-93.82\,\mathrm{dBm}`.
    """

    temperature: float = 300.0
    constant_offset_db: float = 7.0

    def __post_init__(self):
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(self.constant_offset_db):
            raise ValueError("constant_offset_db must be finite")

    def __call__(self, B: Bandwidth) -> ULNoisePower:
        k_B = 1.380649e-23
        thermal_noise_power = self.temperature * B * k_B
        return thermal_noise_power * 10 ** (self.constant_offset_db / 10)
