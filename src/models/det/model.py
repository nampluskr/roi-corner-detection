# src/models/det/model.py: ResNet backbone with a 1x1 grid detection head producing (N, A, 9)

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

OBJ_BIAS_INIT = -4.0


class DetModel(BaseModel):
    """ResNet feature extractor with a 1x1 conv detection head producing (N, A, 9) grid predictions."""

    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        if backbone not in BACKBONE_BUILDERS:
            raise ValueError("Unknown backbone: %s" % backbone)

        net = BACKBONE_BUILDERS[backbone](weights=None)
        if pretrained:
            state_dict = torch.load(BACKBONE_WEIGHTS[backbone], map_location="cpu", weights_only=True)
            net.load_state_dict(state_dict)

        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.head = nn.Conv2d(self._feature_channels(), 9, kernel_size=1)
        nn.init.constant_(self.head.bias[0], OBJ_BIAS_INIT)

    def _feature_channels(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)
            return self.backbone(x).shape[1]

    def forward(self, images):
        features = self.backbone(images)
        grid = self.head(features)
        n, c, h, w = grid.shape
        return grid.permute(0, 2, 3, 1).reshape(n, h * w, c)
