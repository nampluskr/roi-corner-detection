# src/models/line/wrapper.py: composes LineModel with a two-stage warmup recipe (frozen encoder, then unfreeze)

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.line.model import LineModel
from src.models.line.preprocessor import LinePreprocessor
from src.models.line.postprocessor import LinePostprocessor, CENTER_CHANNEL, DISP_START
from src.losses.focal_loss import FocalLoss
from src.metrics.polygon_iou import PolygonIoU
from src.metrics.success_rate import SuccessRate

DISP_WEIGHT = 1.0
BACKBONE_LR = 1e-5
DECODER_LR = 1e-4
UNFREEZE_DECODER_LR = 5e-5


class LineWrapper(BaseWrapper):
    """Wraps LineModel with a warmup schedule: decoder-only training first, then encoder unfreeze at reset LRs."""

    def __init__(self, backbone="mlsd_large", warmup_epochs=30, optimizer=None, scheduler=None,
                 preprocessor=None, postprocessor=None, losses=None, metrics=None, device=None):
        model = LineModel(backbone=backbone, pretrained=True)
        preprocessor = preprocessor or LinePreprocessor()
        postprocessor = postprocessor or LinePostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device)
        backbone_ids = {id(p) for p in self.model.backbone.parameters()}
        decoder_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        self.set_optimizer(self.optimizer or AdamW([
            {"params": self.model.backbone.parameters(), "lr": BACKBONE_LR},
            {"params": decoder_params, "lr": DECODER_LR},
        ]))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_losses(self.losses or {"center": FocalLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU(), "sr": SuccessRate()})
        self.warmup_epochs = warmup_epochs
        self._epochs_done = 0

    def set_encoder_requires_grad(self, requires_grad):
        for p in self.model.backbone.parameters():
            p.requires_grad = requires_grad

    def on_fit_start(self, max_epochs):
        self._epochs_done = 0
        self.set_encoder_requires_grad(self._epochs_done >= self.warmup_epochs)

    def on_epoch_end(self, valid_score=None):
        super().on_epoch_end(valid_score)
        self._epochs_done += 1
        if self._epochs_done == self.warmup_epochs:
            self.set_encoder_requires_grad(True)
            self.optimizer.param_groups[0]["lr"] = BACKBONE_LR
            self.optimizer.param_groups[1]["lr"] = UNFREEZE_DECODER_LR

    def compute_losses(self, raw_output, targets):
        target = self.preprocessor(targets)
        center_logit = raw_output[:, CENTER_CHANNEL:CENTER_CHANNEL + 1]
        disp_pred = raw_output[:, DISP_START:DISP_START + 4]
        center_loss = self.losses["center"](center_logit, target["center"])
        mask = target["mask"]
        denom = mask.sum().clamp(min=1.0)
        disp_loss = (F.smooth_l1_loss(disp_pred, target["disp"], reduction="none")
                     * mask).sum() / (denom * disp_pred.shape[1])
        return {"center": center_loss + DISP_WEIGHT * disp_loss}
