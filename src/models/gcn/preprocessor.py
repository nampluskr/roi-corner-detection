# src/models/gcn/preprocessor.py: pass standard corners through as GCN deep-supervision targets

from src.models.base.base_preprocessor import BasePreprocessor


class GCNPreprocessor(BasePreprocessor):
    """Returns (N, 4, 2) corners unchanged since every refinement step is supervised by the same target."""

    def __call__(self, corners):
        return corners
