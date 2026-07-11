# src/models/seg/wrapper.py: composes SegModel/Preprocessor/Postprocessor and BCE + Dice losses

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.seg.model import SegModel
from src.models.seg.preprocessor import SegPreprocessor
from src.models.seg.postprocessor import SegPostprocessor
from src.losses.bce_loss import BCELoss
from src.losses.dice_loss import DiceLoss
from src.metrics.polygon_iou import PolygonIoU


class SegWrapper(BaseWrapper):
    """Wraps SegModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet50", optimizer=None, scheduler=None,
                 preprocessor=None, postprocessor=None, losses=None, metrics=None, device=None):
        model = SegModel(backbone=backbone, pretrained=True)
        preprocessor = preprocessor or SegPreprocessor()
        postprocessor = postprocessor or SegPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device)
        self.set_optimizer(self.optimizer or AdamW([
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.head.parameters(), "lr": 1e-4},
        ]))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_losses(self.losses or {"bce": BCELoss(), "dice": DiceLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def compute_losses(self, raw_output, targets):
        target = self.preprocessor(targets, size=raw_output.shape[-1])
        return {name: loss_fn(raw_output, target) for name, loss_fn in self.losses.items()}
