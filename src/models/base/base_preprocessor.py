# src/models/base/base_preprocessor.py: base class for corner-to-training-target conversion

class BasePreprocessor:
    """Base class converting standard (N, 4, 2) corners into a method-specific training target."""

    def __call__(self, corners):
        raise NotImplementedError
