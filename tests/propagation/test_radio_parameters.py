import numpy as np
import pytest
from mimodium import propagation
from mimodium.propagation import radio_parameters


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        (radio_parameters.CfgBandwidth(20e6), 20e6),
        (radio_parameters.CfgNumPilots(4), 4),
        (radio_parameters.CfgNumUplinkSymbols(100), 100),
        (radio_parameters.CfgNumDownlinkSymbols(50), 50),
    ],
)
def test_radio_parameter_config_tasks_return_configured_scalars(task, expected):
    assert task() == expected


def test_compute_num_coherence_symbols_sums_pilot_uplink_and_downlink_symbols():
    assert (
        radio_parameters.ComputeNumCoherenceSymbols()(tau_u=100, tau_d=50, tau_p=4)
        == 154
    )


@pytest.mark.parametrize(
    "task",
    [
        radio_parameters.CfgNumPilots,
        radio_parameters.CfgNumUplinkSymbols,
        radio_parameters.CfgNumDownlinkSymbols,
    ],
)
def test_symbol_counts_reject_values_below_their_minimum(task):
    invalid_value = 0 if task is radio_parameters.CfgNumPilots else -1
    with pytest.raises(ValueError):
        task(invalid_value)


@pytest.mark.parametrize(
    ("task", "value"),
    [
        (radio_parameters.CfgCarrierFrequency, 0.0),
        (radio_parameters.CfgCarrierFrequency, np.nan),
        (radio_parameters.CfgBandwidth, 0.0),
        (radio_parameters.CfgBandwidth, np.inf),
    ],
)
def test_positive_radio_parameters_reject_invalid_values(task, value):
    with pytest.raises(ValueError, match="finite and positive"):
        task(value)


@pytest.mark.parametrize(
    "task",
    [
        radio_parameters.CfgDLNoisePower,
        radio_parameters.CfgULNoisePower,
    ],
)
def test_noise_power_configuration_rejects_invalid_values(task):
    with pytest.raises(ValueError, match="temperature"):
        task(temperature=0.0)
    with pytest.raises(ValueError, match="constant_offset_db"):
        task(constant_offset_db=np.inf)


def test_carrier_frequency_is_exported_from_propagation():
    assert propagation.CarrierFrequency.__value__ is float


def test_noise_power_matches_thermal_noise_reference_case():
    bandwidth = 20e6
    expected = 300.0 * bandwidth * 1.380649e-23 * 10 ** (7.0 / 10)

    assert radio_parameters.CfgULNoisePower()(bandwidth) == pytest.approx(expected)
    assert radio_parameters.CfgDLNoisePower()(bandwidth) == pytest.approx(expected)
    assert 10 * np.log10(expected / 1e-3) == pytest.approx(-93.82, abs=0.01)
