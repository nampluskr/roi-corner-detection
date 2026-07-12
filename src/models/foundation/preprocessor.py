# src/models/foundation/preprocessor.py: convert standard corners into foundation-adapter regression targets

from src.models.base.base_preprocessor import BasePreprocessor


class FoundationPreprocessor(BasePreprocessor):
    """Flattens (N, 4, 2) normalized corners into (N, 8) targets with no value transform."""

    def __call__(self, corners):
        return corners.reshape(corners.shape[0], 8)
