# src/models/torchseg/preprocessor.py: reuse the segmentation quad-mask target from seg

from src.models.seg.preprocessor import SegPreprocessor


class TorchsegPreprocessor(SegPreprocessor):
    """Reuses SegPreprocessor to rasterize corners into a filled quad mask target."""
