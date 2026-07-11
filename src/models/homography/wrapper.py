# src/models/homography/wrapper.py: composes HomographyModel/Preprocessor/Postprocessor and SmoothL1Loss

import torch

from src.models.base.base_wrapper import BaseWrapper
from src.models.homography.model import HomographyModel, ALPHA
from src.models.homography.preprocessor import HomographyPreprocessor
from src.models.homography.postprocessor import HomographyPostprocessor
from src.losses.smooth_l1_loss import SmoothL1Loss
from src.metrics.polygon_iou import PolygonIoU


class HomographyWrapper(BaseWrapper):
    """Wraps HomographyModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet50", optimizer=None, losses=None,
                 metrics=None, device=None):
        model = HomographyModel(backbone=backbone, pretrained=True)
        preprocessor = HomographyPreprocessor()
        postprocessor = HomographyPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         losses=losses, metrics=metrics, device=device)
        param_groups = [
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.fc.parameters(), "lr": 1e-4},
        ]
        self.set_optimizer(self.optimizer or torch.optim.AdamW(param_groups))
        self.set_losses(self.losses or {"loss": SmoothL1Loss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def compute_losses(self, raw_output, targets):
        target = self.preprocessor(targets)
        offsets = ALPHA * torch.tanh(raw_output)
        return {name: loss_fn(offsets, target) for name, loss_fn in self.losses.items()}
