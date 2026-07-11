# src/models/heatmap/wrapper.py: composes HeatmapModel/Preprocessor/Postprocessor and MSELoss

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.heatmap.model import HeatmapModel
from src.models.heatmap.preprocessor import HeatmapPreprocessor
from src.models.heatmap.postprocessor import HeatmapPostprocessor
from src.losses.mse_loss import MSELoss
from src.metrics.polygon_iou import PolygonIoU


class HeatmapWrapper(BaseWrapper):
    """Wraps HeatmapModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet50", optimizer=None, scheduler=None,
                 preprocessor=None, postprocessor=None, losses=None, metrics=None, device=None):
        model = HeatmapModel(backbone=backbone, pretrained=True)
        preprocessor = preprocessor or HeatmapPreprocessor()
        postprocessor = postprocessor or HeatmapPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device)
        self.set_optimizer(self.optimizer or AdamW([
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.head.parameters(), "lr": 1e-4},
        ]))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_losses(self.losses or {"loss": MSELoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

