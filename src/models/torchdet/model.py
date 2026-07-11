# src/models/torchdet/model.py: torchvision detector (Faster R-CNN/RetinaNet/SSD) reused end-to-end

import torch
import torchvision.models.detection as detection

from src.models.base.base_model import BaseModel

BACKBONE_WEIGHTS = {
    "fasterrcnn": "/mnt/d/backbones/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "retinanet": "/mnt/d/backbones/retinanet_resnet50_fpn_coco-eeacb38b.pth",
    "ssd": "/mnt/d/backbones/ssd300_vgg16_coco-b556d3b4.pth",
}

BACKBONE_BUILDERS = {
    "fasterrcnn": detection.fasterrcnn_resnet50_fpn,
    "retinanet": detection.retinanet_resnet50_fpn,
    "ssd": detection.ssd300_vgg16,
}

NUM_CLASSES = 5


def _load_coco_weights(detector, path):
    """Load a COCO detection checkpoint into detector, keeping only name/shape-matching params."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    target = detector.state_dict()
    matched = {k: v for k, v in state_dict.items() if k in target and v.shape == target[k].shape}
    target.update(matched)
    detector.load_state_dict(target)


class TorchdetModel(BaseModel):
    """Wraps a torchvision detector built for 5 classes (background + 4 corners) reused end-to-end."""

    def __init__(self, backbone="fasterrcnn", pretrained=True):
        super().__init__()
        if backbone not in BACKBONE_BUILDERS:
            raise ValueError("Unknown backbone: %s" % backbone)

        detector = BACKBONE_BUILDERS[backbone](weights=None, weights_backbone=None,
                                               num_classes=NUM_CLASSES)
        if pretrained:
            _load_coco_weights(detector, BACKBONE_WEIGHTS[backbone])

        detector.transform.image_mean = [0.0, 0.0, 0.0]
        detector.transform.image_std = [1.0, 1.0, 1.0]
        self.detector = detector

    def forward(self, images, targets=None):
        if torch.is_tensor(images):
            images = [image for image in images]
        if targets is None:
            return self.detector(images)
        return self.detector(images, targets)
