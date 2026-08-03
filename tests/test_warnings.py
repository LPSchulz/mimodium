import mimodium
from mimodium import propagation
from mimodium.propagation import path_loss
from mimodium.warnings import ApplicabilityWarning, ScenarioSizeWarning


def test_warning_classes_are_available_from_the_top_level_package():
    assert mimodium.ApplicabilityWarning is ApplicabilityWarning
    assert mimodium.ScenarioSizeWarning is ScenarioSizeWarning


def test_legacy_propagation_warning_exports_remain_compatible():
    assert propagation.ApplicabilityWarning is ApplicabilityWarning
    assert path_loss.ApplicabilityWarning is ApplicabilityWarning
