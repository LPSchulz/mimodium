from mimodium.scenario import area


def test_area_length_returns_configured_value():
    assert area.CfgAreaLength(250.0)() == 250.0


def test_area_size_computes_from_length():
    assert area.ComputeAreaSize()(1000.0) == 1_000_000.0


def test_wrap_around_returns_configured_value():
    assert area.CfgWrapAround(True)() is True
