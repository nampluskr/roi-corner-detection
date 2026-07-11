# src/models/torchseg/wrapper.py: composes TorchsegModel with seg mask target/postproc and BCE + Dice losses

import torch

from src.models.base.base_wrapper import BaseWrapper
from src.models.seg.wrapper import SegWrapper
from src.models.torchseg.model import TorchsegModel
from src.models.torchseg.preprocessor import TorchsegPreprocessor
from src.models.torchseg.postprocessor import TorchsegPostprocessor
from src.losses.bce_loss import BCELoss
from src.losses.dice_loss import DiceLoss
from src.metrics.polygon_iou import PolygonIoU


class TorchsegWrapper(SegWrapper):
    """Wraps a torchvision segmentation model, reusing SegWrapper train/eval logic with seg mask targets."""

    def __init__(self, backbone="deeplabv3_resnet50", optimizer=None, preprocessor=None,
                 postprocessor=None, losses=None, metrics=None, device=None):
        model = TorchsegModel(backbone=backbone, pretrained=True)
        BaseWrapper.__init__(self, model, optimizer=optimizer,
                             preprocessor=preprocessor, postprocessor=postprocessor,
                             losses=losses, metrics=metrics, device=device)
        param_groups = [
            {"params": self.model.net.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.net.classifier.parameters(), "lr": 1e-4},
        ]
        self.set_optimizer(self.optimizer or torch.optim.AdamW(param_groups))
        self.set_preprocessor(self.preprocessor or TorchsegPreprocessor())
        self.set_postprocessor(self.postprocessor or TorchsegPostprocessor())
        self.set_losses(self.losses or {"bce": BCELoss(), "dice": DiceLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})
