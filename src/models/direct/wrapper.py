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

    def __init__(self, backbone="resnet50", optimizer=None, preprocessor=None,
                 postprocessor=None, losses=None, metrics=None, device=None):
        model = DirectModel(backbone=backbone, pretrained=True)
        super().__init__(model, optimizer=optimizer,
                         preprocessor=preprocessor, postprocessor=postprocessor,
                         losses=losses, metrics=metrics, device=device)
        param_groups = [
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.fc.parameters(), "lr": 1e-4},
        ]
        self.set_optimizer(self.optimizer or torch.optim.AdamW(param_groups))
        self.set_preprocessor(self.preprocessor or DirectPreprocessor())
        self.set_postprocessor(self.postprocessor or DirectPostprocessor())
        self.set_losses(self.losses or {"loss": WingLoss(apply_sigmoid=True)})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def on_fit_start(self, max_epochs):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs)

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
