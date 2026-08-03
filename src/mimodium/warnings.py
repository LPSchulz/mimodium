class ApplicabilityWarning(UserWarning):
    """A model is evaluated outside its documented applicability range."""


class ScenarioSizeWarning(UserWarning):
    """A scenario area may be too small for its configured node density."""
