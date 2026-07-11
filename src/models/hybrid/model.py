# src/models/hybrid/model.py: mask model reused from seg for the DL + classical CV hybrid method

from src.models.seg.model import SegModel


class HybridModel(SegModel):
    """MobileNetV3/ResNet encoder with a UNet decoder producing a (N, 1, H, W) quad mask, reused from seg."""

    def __init__(self, backbone="unet_mobilenet_v3_large", pretrained=True):
        super().__init__(backbone=backbone, pretrained=pretrained)
