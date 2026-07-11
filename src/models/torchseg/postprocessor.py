# src/models/torchseg/postprocessor.py: reuse the mask-to-corner contour decoding from seg

from src.models.seg.postprocessor import SegPostprocessor


class TorchsegPostprocessor(SegPostprocessor):
    """Reuses SegPostprocessor to extract corners from a quad mask via findContours."""
