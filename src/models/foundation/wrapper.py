# src/models/foundation/wrapper.py: composes FoundationModel with a frozen backbone and a head-only optimizer

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.base.base_wrapper import BaseWrapper
from src.models.foundation.model import FoundationModel
from src.models.foundation.preprocessor import FoundationPreprocessor
from src.models.foundation.postprocessor import FoundationPostprocessor
from src.losses.wing_loss import WingLoss
from src.metrics.polygon_iou import PolygonIoU


class FoundationWrapper(BaseWrapper):
    """Wraps FoundationModel with a permanently frozen backbone, training only the lightweight head."""

    def __init__(self, backbone="vits14", optimizer=None, scheduler=None,
                 preprocessor=None, postprocessor=None, losses=None, metrics=None, device=None):
        model = FoundationModel(backbone=backbone, pretrained=True)
        preprocessor = preprocessor or FoundationPreprocessor()
        postprocessor = postprocessor or FoundationPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         scheduler=scheduler, losses=losses, metrics=metrics, device=device)
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        backbone_ids = {id(p) for p in self.model.backbone.parameters()}
        head_params = [p for p in self.model.parameters() if id(p) not in backbone_ids]
        self.set_optimizer(self.optimizer or AdamW(head_params, lr=1e-3))
        self.set_scheduler(self.scheduler or ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7))
        self.set_losses(self.losses or {"loss": WingLoss(apply_sigmoid=True)})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})
