# src/models/hybrid/wrapper.py: composes HybridModel/Preprocessor/Postprocessor and BCE + Dice losses

import torch

from src.models.base.base_wrapper import BaseWrapper
from src.models.hybrid.model import HybridModel
from src.models.hybrid.preprocessor import HybridPreprocessor
from src.models.hybrid.postprocessor import HybridPostprocessor
from src.losses.bce_loss import BCELoss
from src.losses.dice_loss import DiceLoss
from src.metrics.polygon_iou import PolygonIoU


class HybridWrapper(BaseWrapper):
    """Wraps HybridModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="mobilenet_v3_large", optimizer=None, preprocessor=None,
                 postprocessor=None, losses=None, metrics=None, device=None):
        model = HybridModel(backbone=backbone, pretrained=True)
        super().__init__(model, optimizer=optimizer,
                         preprocessor=preprocessor, postprocessor=postprocessor,
                         losses=losses, metrics=metrics, device=device)
        param_groups = [
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.head.parameters(), "lr": 1e-4},
        ]
        self.set_optimizer(self.optimizer or torch.optim.AdamW(param_groups))
        self.set_preprocessor(self.preprocessor or HybridPreprocessor())
        self.set_postprocessor(self.postprocessor or HybridPostprocessor())
        self.set_losses(self.losses or {"bce": BCELoss(), "dice": DiceLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def on_fit_start(self, max_epochs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        raw_output = self.model(images)
        target = self.preprocessor(targets, size=raw_output.shape[-1])
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

        raw_output = self.model(images)
        target = self.preprocessor(targets, size=raw_output.shape[-1])
        for loss_fn in self.losses.values():
            loss_fn(raw_output, target)
        preds = self.postprocessor(raw_output).cpu().numpy()

        self.update_metrics(preds, targets.cpu().numpy())

        return {**self.compute_losses(), **self.compute_metrics()}
