# src/models/doc/model.py: pretrained encoder with a spatial-preserving strided-conv head for document regression

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


class DocModel(BaseModel):
    """Pretrained ResNet encoder with a spatial-preserving strided-conv head for corner regression."""

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
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 8),
        )

    def forward(self, images):
        features = self.backbone(images)
        return self.head(features)
