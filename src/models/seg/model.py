# src/models/seg/model.py: ResNet/MobileNetV3 encoder with a UNet decoder producing a single quad mask

import torch
import torch.nn as nn
import torchvision.models as models

from src.models.base.base_model import BaseModel

BACKBONE_WEIGHTS = {
    "unet_resnet18": "/mnt/d/backbones/resnet18-f37072fd.pth",
    "unet_resnet34": "/mnt/d/backbones/resnet34-b627a593.pth",
    "unet_resnet50": "/mnt/d/backbones/resnet50-0676ba61.pth",
    "deeplabv3_resnet50": "/mnt/d/backbones/deeplabv3_resnet50_coco-cd0a2569.pth",
    "fcn_resnet50": "/mnt/d/backbones/fcn_resnet50_coco-1167a1af.pth",
    "deeplabv3_mobilenet_v3_large": "/mnt/d/backbones/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
    "lraspp_mobilenet_v3_large": "/mnt/d/backbones/lraspp_mobilenet_v3_large-d234d4ea.pth",
    "unet_mobilenet_v3_large": "/mnt/d/backbones/mobilenet_v3_large-8738ca79.pth",
}

BACKBONE_BUILDERS = {
    "unet_resnet18": models.resnet18,
    "unet_resnet34": models.resnet34,
    "unet_resnet50": models.resnet50,
    "deeplabv3_resnet50": models.resnet50,
    "fcn_resnet50": models.resnet50,
    "deeplabv3_mobilenet_v3_large": models.mobilenet_v3_large,
    "lraspp_mobilenet_v3_large": models.mobilenet_v3_large,
    "unet_mobilenet_v3_large": models.mobilenet_v3_large,
}

DECODER_CHANNELS = [256, 128, 64, 32]


def _build_stages(net):
    """Split a ResNet or MobileNetV3 encoder into 5 stages, each halving spatial resolution."""
    if hasattr(net, "layer1"):
        return [
            nn.Sequential(net.conv1, net.bn1, net.relu),
            nn.Sequential(net.maxpool, net.layer1),
            net.layer2,
            net.layer3,
            net.layer4,
        ]
    features = net.features
    return [features[0:1], features[1:3], features[3:5], features[5:8], features[8:16]]


def _load_backbone_weights(net, path):
    """Load classification or torchvision-segmentation checkpoint weights into an encoder."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not any(k.startswith("backbone.") for k in state_dict):
        net.load_state_dict(state_dict)
        return
    state_dict = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    target = net if hasattr(net, "layer1") else net.features
    target.load_state_dict(state_dict, strict=False)


class UpBlock(nn.Module):
    """Transposed-conv upsample followed by skip concatenation and a double conv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegModel(BaseModel):
    """ResNet/MobileNetV3 encoder with a UNet decoder producing a (N, 1, H, W) quad mask at half input resolution."""

    def __init__(self, backbone="unet_resnet50", pretrained=True):
        super().__init__()
        if backbone not in BACKBONE_BUILDERS:
            raise ValueError("Unknown backbone: %s" % backbone)

        net = BACKBONE_BUILDERS[backbone](weights=None)
        if pretrained:
            _load_backbone_weights(net, BACKBONE_WEIGHTS[backbone])

        self.backbone = nn.ModuleList(_build_stages(net))
        c0, c1, c2, c3, c4 = self._encoder_channels()
        d0, d1, d2, d3 = DECODER_CHANNELS
        self.head = nn.ModuleList([
            UpBlock(c4, c3, d0),
            UpBlock(d0, c2, d1),
            UpBlock(d1, c1, d2),
            UpBlock(d2, c0, d3),
            nn.Conv2d(d3, 1, kernel_size=1),
        ])

    def _encoder_channels(self):
        with torch.no_grad():
            x = torch.zeros(1, 3, 64, 64)
            channels = []
            for stage in self.backbone:
                x = stage(x)
                channels.append(x.shape[1])
        return channels

    def forward(self, images):
        x0 = self.backbone[0](images)
        x1 = self.backbone[1](x0)
        x2 = self.backbone[2](x1)
        x3 = self.backbone[3](x2)
        x4 = self.backbone[4](x3)
        d = self.head[0](x4, x3)
        d = self.head[1](d, x2)
        d = self.head[2](d, x1)
        d = self.head[3](d, x0)
        return self.head[4](d)
