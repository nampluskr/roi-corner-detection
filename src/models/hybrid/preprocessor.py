# src/models/hybrid/preprocessor.py: corners-to-mask target reused from seg for the hybrid method

from src.models.seg.preprocessor import SegPreprocessor


class HybridPreprocessor(SegPreprocessor):
    """Rasterizes (N, 4, 2) normalized corners as (N, 1, S, S) filled quad masks, reused from seg."""
