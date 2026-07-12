# src/models/line/model.py: M-LSD (MobileNetV2 + FPN decoder) producing a 16-channel line tp-map

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.base.base_model import BaseModel

MLSD_WEIGHTS = {
    "mlsd_large": "/mnt/d/backbones/mlsd_large_512_fp32.pth",
}

FPN_SELECTED = [1, 3, 6, 10, 13]
MLSD_INPUT_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BlockTypeA(nn.Module):
    """Projects two feature levels to fixed widths, upsamples the coarse one, and concatenates."""

    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale=True):
        super().__init__()
        self.upscale = upscale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True),
        )

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        if self.upscale:
            b = F.interpolate(b, scale_factor=2.0, mode="bilinear", align_corners=True)
        return torch.cat([a, b], dim=1)


class BlockTypeB(nn.Module):
    """Residual 3x3 conv followed by a 3x3 conv that changes the channel width."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x) + x
        return self.conv2(x)


class BlockTypeC(nn.Module):
    """Dilated 3x3 conv, standard 3x3 conv, and a 1x1 projection to the output channels."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 feature extractor returning the five FPN levels at indices 1, 3, 6, 10, 13."""

    def __init__(self):
        super().__init__()
        net = models.mobilenet_v2(weights=None)
        first = net.features[0][0]
        net.features[0][0] = nn.Conv2d(4, first.out_channels, kernel_size=3,
                                       stride=2, padding=1, bias=False)
        self.features = net.features[:FPN_SELECTED[-1] + 1]

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in FPN_SELECTED:
                outputs.append(x)
        return outputs


class LineModel(BaseModel):
    """M-LSD MobileNetV2-FPN model producing a 16-channel line tp-map from a 4-channel RGBA input."""

    def __init__(self, backbone="mlsd_large", pretrained=True):
        super().__init__()
        if backbone not in MLSD_WEIGHTS:
            raise ValueError("Unknown backbone: %s" % backbone)

        self.backbone = MobileNetV2Backbone()
        self.block15 = BlockTypeA(64, 96, 64, 64, upscale=False)
        self.block16 = BlockTypeB(128, 64)
        self.block17 = BlockTypeA(32, 64, 64, 64, upscale=True)
        self.block18 = BlockTypeB(128, 64)
        self.block19 = BlockTypeA(24, 64, 64, 64, upscale=True)
        self.block20 = BlockTypeB(128, 64)
        self.block21 = BlockTypeA(16, 64, 64, 64, upscale=True)
        self.block22 = BlockTypeB(128, 64)
        self.block23 = BlockTypeC(64, 16)

        if pretrained:
            path = MLSD_WEIGHTS[backbone]
            if not os.path.exists(path):
                raise FileNotFoundError("M-LSD weights not found: %s" % path)
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict, strict=False)

    def forward(self, images):
        mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        x = (images * std + mean).clamp(0.0, 1.0)
        x = F.interpolate(x, size=(MLSD_INPUT_SIZE, MLSD_INPUT_SIZE),
                          mode="bilinear", align_corners=False)
        x = x * 2.0 - 1.0
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        c1, c2, c3, c4, c5 = self.backbone(x)

        x = self.block15(c4, c5)
        x = self.block16(x)
        x = self.block17(c3, x)
        x = self.block18(x)
        x = self.block19(c2, x)
        x = self.block20(x)
        x = self.block21(c1, x)
        x = self.block22(x)
        return self.block23(x)
