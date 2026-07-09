# src/models/direct/wrapper.py: composes DirectModel/DirectPreprocessor/DirectPostprocessor/DirectLoss

import torch

from src.models.base.base_wrapper import BaseWrapper
from src.models.direct.model import DirectModel
from src.models.direct.preprocessor import DirectPreprocessor
from src.models.direct.postprocessor import DirectPostprocessor
from src.models.direct.loss import DirectLoss
from src.metrics.polygon_iou import PolygonIoU


class DirectWrapper(BaseWrapper):
    """Wraps DirectModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet18", optimizer=None, preprocessor=None,
                 postprocessor=None, loss_fn=None, metrics=None, device=None):
        model = DirectModel(backbone=backbone, pretrained=True)
        super().__init__(model, optimizer=optimizer, 
                         preprocessor=preprocessor, postprocessor=postprocessor, 
                         loss_fn=loss_fn, metrics=metrics, device=device)
        self.set_optimizer(self.optimizer or torch.optim.Adam(self.model.parameters(), lr=1e-3))
        self.set_preprocessor(self.preprocessor or DirectPreprocessor())
        self.set_postprocessor(self.postprocessor or DirectPostprocessor())
        self.set_loss_fn(self.loss_fn or DirectLoss())
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    def train_step(self, images, targets):
        self.model.train()
        images, targets = images.to(self.device), targets.to(self.device)
        target = self.preprocessor(targets)

        self.optimizer.zero_grad()
        raw_output = self.model(images)
        loss = self.loss_fn(raw_output, target)
        loss.backward()
        self.optimizer.step()

        if self.metrics:
            with torch.no_grad():
                preds = self.postprocessor(raw_output).cpu().numpy()
            self.update_metrics(preds, targets.cpu().numpy())

        return {"loss": loss.item()}

    @torch.no_grad()
    def eval_step(self, images, targets):
        self.model.eval()
        images, targets = images.to(self.device), targets.to(self.device)
        target = self.preprocessor(targets)

        raw_output = self.model(images)
        loss = self.loss_fn(raw_output, target)
        preds = self.postprocessor(raw_output)
        preds_np = preds.cpu().numpy()

        self.update_metrics(preds_np, targets.cpu().numpy())

        return {"loss": loss.item(), "preds": preds_np}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        raw_output = self.model(images.to(self.device))
        preds = self.postprocessor(raw_output)
        return preds.cpu().numpy()
