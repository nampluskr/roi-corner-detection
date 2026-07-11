# src/models/gcn/wrapper.py: composes GCNModel/Preprocessor/Postprocessor with deep-supervised SmoothL1

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.gcn.model import GCNModel
from src.models.gcn.preprocessor import GCNPreprocessor
from src.models.gcn.postprocessor import GCNPostprocessor
from src.losses.smooth_l1_loss import SmoothL1Loss
from src.metrics.polygon_iou import PolygonIoU


class GCNWrapper(BaseWrapper):
    """Wraps GCNModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet50", optimizer=None, scheduler=None,
                 preprocessor=None, postprocessor=None, losses=None, metrics=None, device=None):
        model = GCNModel(backbone=backbone, pretrained=True)
        preprocessor = preprocessor or GCNPreprocessor()
        postprocessor = postprocessor or GCNPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device)
        backbone_ids = {id(p) for p in self.model.backbone.parameters()}
        head_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        self.set_optimizer(self.optimizer or AdamW([
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": head_params, "lr": 1e-4},
        ]))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_losses(self.losses or {"loss": SmoothL1Loss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def compute_losses(self, raw_output, targets):
        target = self.preprocessor(targets)
        num_steps = raw_output.shape[1]
        stacked = raw_output.permute(1, 0, 2, 3).reshape(-1, target.shape[1], target.shape[2])
        repeated = target.repeat(num_steps, 1, 1)
        return {"loss": self.losses["loss"](stacked, repeated)}
