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

    def __init__(self, backbone="resnet50", optimizer=None, preprocessor=None,
                 postprocessor=None, losses=None, metrics=None, device=None):
        model = HeatmapModel(backbone=backbone, pretrained=True)
        super().__init__(model, optimizer=optimizer,
                         preprocessor=preprocessor, postprocessor=postprocessor,
                         losses=losses, metrics=metrics, device=device)
        param_groups = [
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.head.parameters(), "lr": 1e-4},
        ]
        self.set_optimizer(self.optimizer or torch.optim.AdamW(param_groups))
        self.set_preprocessor(self.preprocessor or HeatmapPreprocessor())
        self.set_postprocessor(self.postprocessor or HeatmapPostprocessor())
        self.set_losses(self.losses or {"loss": MSELoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def on_fit_start(self, max_epochs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        target = self.preprocessor(targets)

        self.optimizer.zero_grad()
        raw_output = self.model(images)
        loss = sum(loss_fn(raw_output, target) for loss_fn in self.losses.values())
        loss.backward()
        self.optimizer.step()

        if self.metrics:
            with torch.no_grad():
                preds = self.postprocessor(raw_output).cpu().numpy()
            self.update_metrics(preds, targets.cpu().numpy())

        return {**self.compute_losses(), **self.compute_metrics()}

    @torch.no_grad()
    def eval_step(self, images, targets):
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        target = self.preprocessor(targets)

        raw_output = self.model(images)
        for loss_fn in self.losses.values():
            loss_fn(raw_output, target)
        preds = self.postprocessor(raw_output)
        preds_np = preds.cpu().numpy()

        self.update_metrics(preds_np, targets.cpu().numpy())

        return {**self.compute_losses(), **self.compute_metrics()}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        raw_output = self.model(images.to(self.device, non_blocking=True))
        preds = self.postprocessor(raw_output)
        return preds.cpu().numpy()
