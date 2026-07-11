# src/models/doc/wrapper.py: composes DocModel/Preprocessor/Postprocessor with an encoder-freezing warmup schedule

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.doc.model import DocModel
from src.models.doc.preprocessor import DocPreprocessor
from src.models.doc.postprocessor import DocPostprocessor
from src.losses.wing_loss import WingLoss
from src.metrics.polygon_iou import PolygonIoU


class DocWrapper(BaseWrapper):
    """Wraps DocModel with discriminative learning rates and a warmup encoder-freezing schedule."""

    def __init__(self, backbone="resnet50", warmup_epochs=3, optimizer=None, scheduler=None,
                 preprocessor=None, postprocessor=None, losses=None, metrics=None, device=None):
        model = DocModel(backbone=backbone, pretrained=True)
        preprocessor = preprocessor or DocPreprocessor()
        postprocessor = postprocessor or DocPostprocessor()
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
        self.set_losses(self.losses or {"loss": WingLoss(apply_sigmoid=True)})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})
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
