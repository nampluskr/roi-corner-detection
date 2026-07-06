# src/models/base/base_postprocessor.py: base class for raw-output-to-corner conversion

class BasePostprocessor:
    """Base class converting a method-specific raw model output into standard (N, 4, 2) corners."""

    def __call__(self, raw_output):
        raise NotImplementedError
