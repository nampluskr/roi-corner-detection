# src/models/det/wrapper.py: composes DetModel/Preprocessor/Postprocessor and focal/box/class losses

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.det.model import DetModel
from src.models.det.preprocessor import DetPreprocessor
from src.models.det.postprocessor import DetPostprocessor
from src.losses.focal_loss import FocalLoss
from src.losses.smooth_l1_loss import SmoothL1Loss
from src.losses.cross_entropy_loss import CrossEntropyLoss
from src.metrics.polygon_iou import PolygonIoU


class DetWrapper(BaseWrapper):
    """Wraps DetModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet50", optimizer=None, scheduler=None,
                 preprocessor=None, postprocessor=None, losses=None, metrics=None, device=None):
        model = DetModel(backbone=backbone, pretrained=True)
        preprocessor = preprocessor or DetPreprocessor()
        postprocessor = postprocessor or DetPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device)
        self.set_optimizer(self.optimizer or AdamW([
            {"params": self.model.backbone.parameters(), "lr": 1e-5},
            {"params": self.model.head.parameters(), "lr": 1e-4},
        ]))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_losses(self.losses or {"obj": FocalLoss(), "box": SmoothL1Loss(weight=5.0),
                                        "cls": CrossEntropyLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def compute_losses(self, raw_output, targets):
        size = int(round(raw_output.shape[1] ** 0.5))
        obj_t, box_t, cls_t, pos_mask = self.preprocessor(targets, size=size)
        obj_logits = raw_output[..., 0]
        box_pred = raw_output[..., 1:5]
        cls_logits = raw_output[..., 5:9]

        losses = {"obj": self.losses["obj"](obj_logits, obj_t)}
        if pos_mask.any():
            losses["box"] = self.losses["box"](box_pred[pos_mask], box_t[pos_mask])
            losses["cls"] = self.losses["cls"](cls_logits[pos_mask], cls_t[pos_mask])
        return losses
