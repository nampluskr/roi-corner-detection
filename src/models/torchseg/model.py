# src/models/torchseg/model.py: torchvision segmentation model (DeepLabV3/FCN/LR-ASPP) reused end-to-end

import torch
import torchvision.models.segmentation as segmentation

from src.models.base.base_model import BaseModel

BACKBONE_WEIGHTS = {
    "deeplabv3_resnet50": "/mnt/d/backbones/deeplabv3_resnet50_coco-cd0a2569.pth",
    "fcn_resnet50": "/mnt/d/backbones/fcn_resnet50_coco-1167a1af.pth",
    "deeplabv3_mobilenet_v3_large": "/mnt/d/backbones/deeplabv3_mobilenet_v3_large-fc3c493d.pth",
    "lraspp_mobilenet_v3_large": "/mnt/d/backbones/lraspp_mobilenet_v3_large-d234d4ea.pth",
}

BACKBONE_BUILDERS = {
    "deeplabv3_resnet50": (segmentation.deeplabv3_resnet50, True),
    "fcn_resnet50": (segmentation.fcn_resnet50, True),
    "deeplabv3_mobilenet_v3_large": (segmentation.deeplabv3_mobilenet_v3_large, True),
    "lraspp_mobilenet_v3_large": (segmentation.lraspp_mobilenet_v3_large, False),
}

NUM_CLASSES = 1


def _load_seg_weights(net, path):
    """Load a COCO segmentation checkpoint into net, keeping only name/shape-matching params."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    target = net.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in target and v.shape == target[k].shape}
    target.update(matched)
    net.load_state_dict(target)


class TorchsegModel(BaseModel):
    """Wraps a torchvision segmentation model built for a single quad-mask channel reused end-to-end."""

    def __init__(self, backbone="deeplabv3_resnet50", pretrained=True):
        super().__init__()
        if backbone not in BACKBONE_BUILDERS:
            raise ValueError("Unknown backbone: %s" % backbone)

        builder, has_aux = BACKBONE_BUILDERS[backbone]
        kwargs = {"aux_loss": False} if has_aux else {}
        net = builder(weights=None, weights_backbone=None, num_classes=NUM_CLASSES, **kwargs)
        if pretrained:
            _load_seg_weights(net, BACKBONE_WEIGHTS[backbone])
        self.net = net

    def forward(self, images):
        return self.net(images)["out"]
