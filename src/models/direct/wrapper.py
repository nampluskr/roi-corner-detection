# src/models/direct/wrapper.py: composes DirectModel/DirectPreprocessor/DirectPostprocessor and WingLoss

import torch

from src.models.base.base_wrapper import BaseWrapper
from src.models.direct.model import DirectModel
from src.models.direct.preprocessor import DirectPreprocessor
from src.models.direct.postprocessor import DirectPostprocessor
from src.losses.wing_loss import WingLoss
from src.metrics.polygon_iou import PolygonIoU


class DirectWrapper(BaseWrapper):
    """Wraps DirectModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet50", optimizer=None, losses=None,
                 metrics=None, device=None):
        model = DirectModel(backbone=backbone, pretrained=True)
        preprocessor = DirectPreprocessor()
        postprocessor = DirectPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         losses=losses, metrics=metrics, device=device)
        param_groups = [
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.fc.parameters(), "lr": 1e-4},
        ]
        self.set_optimizer(self.optimizer or torch.optim.AdamW(param_groups))
        self.set_losses(self.losses or {"loss": WingLoss(apply_sigmoid=True)})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

