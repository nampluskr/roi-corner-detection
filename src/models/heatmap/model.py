# src/models/heatmap/model.py: ResNet backbone with deconv head producing per-corner heatmaps

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.base.base_model import BaseModel

BACKBONE_WEIGHTS = {
    "resnet18": "/mnt/d/backbones/resnet18-f37072fd.pth",
    "resnet34": "/mnt/d/backbones/resnet34-b627a593.pth",
    "resnet50": "/mnt/d/backbones/resnet50-0676ba61.pth",
}

BACKBONE_BUILDERS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}

HEATMAP_SIZE = 56
NUM_CORNERS = 4
DECONV_CHANNELS = 256


class HeatmapModel(BaseModel):
    """ResNet backbone with three transposed-conv layers producing (N, 4, H, W) corner heatmaps."""

    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        if backbone not in BACKBONE_BUILDERS:
            raise ValueError("Unknown backbone: %s" % backbone)

        net = BACKBONE_BUILDERS[backbone](weights=None)
        if pretrained:
            state_dict = torch.load(BACKBONE_WEIGHTS[backbone], map_location="cpu", weights_only=True)
            net.load_state_dict(state_dict)

        in_channels = net.fc.in_features
        self.backbone = nn.Sequential(*list(net.children())[:-2])
        self.head = nn.Sequential(
            nn.ConvTranspose2d(in_channels, DECONV_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DECONV_CHANNELS),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(DECONV_CHANNELS, DECONV_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DECONV_CHANNELS),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(DECONV_CHANNELS, DECONV_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DECONV_CHANNELS),
            nn.ReLU(inplace=True),
            nn.Conv2d(DECONV_CHANNELS, NUM_CORNERS, kernel_size=1),
        )

    def forward(self, images):
        features = self.backbone(images)
        return self.head(features)
