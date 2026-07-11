# src/models/direct/wrapper.py: composes DirectModel/DirectPreprocessor/DirectPostprocessor and WingLoss

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.direct.model import DirectModel
from src.models.direct.preprocessor import DirectPreprocessor
from src.models.direct.postprocessor import DirectPostprocessor
from src.losses.wing_loss import WingLoss
from src.metrics.polygon_iou import PolygonIoU


class DirectWrapper(BaseWrapper):
    """Wraps DirectModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet50", optimizer=None, scheduler=None,
                 preprocessor=None, postprocessor=None, losses=None, metrics=None, device=None):
        model = DirectModel(backbone=backbone, pretrained=True)
        preprocessor = preprocessor or DirectPreprocessor()
        postprocessor = postprocessor or DirectPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device)
        self.set_optimizer(self.optimizer or AdamW([
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.fc.parameters(), "lr": 1e-4},
        ]))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_losses(self.losses or {"loss": WingLoss(apply_sigmoid=True)})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

