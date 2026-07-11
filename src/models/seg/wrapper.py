# src/models/seg/wrapper.py: composes SegModel/Preprocessor/Postprocessor and BCE + Dice losses

import torch

from src.models.base.base_wrapper import BaseWrapper
from src.models.seg.model import SegModel
from src.models.seg.preprocessor import SegPreprocessor
from src.models.seg.postprocessor import SegPostprocessor
from src.losses.bce_loss import BCELoss
from src.losses.dice_loss import DiceLoss
from src.metrics.polygon_iou import PolygonIoU


class SegWrapper(BaseWrapper):
    """Wraps SegModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet50", optimizer=None, losses=None,
                 metrics=None, device=None):
        model = SegModel(backbone=backbone, pretrained=True)
        preprocessor = SegPreprocessor()
        postprocessor = SegPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         losses=losses, metrics=metrics, device=device)
        param_groups = [
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.head.parameters(), "lr": 1e-4},
        ]
        self.set_optimizer(self.optimizer or torch.optim.AdamW(param_groups))
        self.set_losses(self.losses or {"bce": BCELoss(), "dice": DiceLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def compute_losses(self, raw_output, targets):
        target = self.preprocessor(targets, size=raw_output.shape[-1])
        return {name: loss_fn(raw_output, target) for name, loss_fn in self.losses.items()}
