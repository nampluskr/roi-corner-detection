# src/models/heatmap/wrapper.py: composes HeatmapModel/Preprocessor/Postprocessor and MSELoss

import torch

from src.models.base.base_wrapper import BaseWrapper
from src.models.heatmap.model import HeatmapModel
from src.models.heatmap.preprocessor import HeatmapPreprocessor
from src.models.heatmap.postprocessor import HeatmapPostprocessor
from src.losses.mse_loss import MSELoss
from src.metrics.polygon_iou import PolygonIoU


class HeatmapWrapper(BaseWrapper):
    """Wraps HeatmapModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet50", optimizer=None, losses=None,
                 metrics=None, device=None):
        model = HeatmapModel(backbone=backbone, pretrained=True)
        preprocessor = HeatmapPreprocessor()
        postprocessor = HeatmapPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         losses=losses, metrics=metrics, device=device)
        param_groups = [
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.head.parameters(), "lr": 1e-4},
        ]
        self.set_optimizer(self.optimizer or torch.optim.AdamW(param_groups))
        self.set_losses(self.losses or {"loss": MSELoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

