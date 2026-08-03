import numpy as np
from mimodium.scenario import power


def test_pilot_power_config_task_returns_correct_array():
    K = 5
    task = power.CfgPilotMaxPower(max_power_mw=100.0)
    result = task(K)
    assert isinstance(result, np.ndarray)
    assert result.shape == (K,)
    assert np.all(result == 0.1)


def test_ue_max_power_config_task_returns_correct_array():
    K = 4
    task = power.CfgUEMaxPower(max_power_mw=200.0)
    result = task(K)
    assert isinstance(result, np.ndarray)
    assert result.shape == (K,)
    assert np.all(result == 0.2)


def test_ap_max_power_config_task_returns_correct_array():
    L = 3
    task = power.CfgAPMaxPower(max_power_mw=300.0)
    result = task(L)
    assert isinstance(result, np.ndarray)
    assert result.shape == (L,)
    assert np.all(result == 0.3)
