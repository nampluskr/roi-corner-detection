# src/models/det/wrapper.py: composes DetModel/Preprocessor/Postprocessor and focal/box/class losses

import torch

from src.models.base.base_wrapper import BaseWrapper
from src.models.det.model import DetModel
from src.models.det.preprocessor import DetPreprocessor
from src.models.det.postprocessor import DetPostprocessor
from src.losses.focal_loss import FocalLoss
from src.losses.smooth_l1_loss import SmoothL1Loss
from src.losses.cross_entropy_loss import CrossEntropyLoss
from src.metrics.polygon_iou import PolygonIoU

LAMBDA_OBJ = 1.0
LAMBDA_BOX = 5.0
LAMBDA_CLS = 1.0


class DetWrapper(BaseWrapper):
    """Wraps DetModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet50", optimizer=None, preprocessor=None,
                 postprocessor=None, losses=None, metrics=None, device=None):
        model = DetModel(backbone=backbone, pretrained=True)
        super().__init__(model, optimizer=optimizer,
                         preprocessor=preprocessor, postprocessor=postprocessor,
                         losses=losses, metrics=metrics, device=device)
        param_groups = [
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.head.parameters(), "lr": 1e-4},
        ]
        self.set_optimizer(self.optimizer or torch.optim.AdamW(param_groups))
        self.set_preprocessor(self.preprocessor or DetPreprocessor())
        self.set_postprocessor(self.postprocessor or DetPostprocessor())
        self.set_losses(self.losses or {"obj": FocalLoss(), "box": SmoothL1Loss(),
                                        "cls": CrossEntropyLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def on_fit_start(self, max_epochs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def _compute_loss(self, raw_output, targets):
        size = int(round(raw_output.shape[1] ** 0.5))
        obj_t, box_t, cls_t, pos_mask = self.preprocessor(targets, size=size)
        obj_logits = raw_output[..., 0]
        box_pred = raw_output[..., 1:5]
        cls_logits = raw_output[..., 5:9]

        loss = LAMBDA_OBJ * self.losses["obj"](obj_logits, obj_t)
        if pos_mask.any():
            loss = loss + LAMBDA_BOX * self.losses["box"](box_pred[pos_mask], box_t[pos_mask])
            loss = loss + LAMBDA_CLS * self.losses["cls"](cls_logits[pos_mask], cls_t[pos_mask])
        return loss

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        raw_output = self.model(images)
        loss = self._compute_loss(raw_output, targets)
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
        self._compute_loss(raw_output, targets)
        preds = self.postprocessor(raw_output).cpu().numpy()

        self.update_metrics(preds, targets.cpu().numpy())

        return {**self.compute_losses(), **self.compute_metrics()}
