# src/models/torchdet/wrapper.py: composes TorchdetModel/Preprocessor/Postprocessor and detector losses

import torch

from src.models.base.base_wrapper import BaseWrapper
from src.models.torchdet.model import TorchdetModel
from src.models.torchdet.preprocessor import TorchdetPreprocessor
from src.models.torchdet.postprocessor import TorchdetPostprocessor
from src.losses.base_loss import BaseLoss
from src.metrics.polygon_iou import PolygonIoU


class TorchdetWrapper(BaseWrapper):
    """Wraps a torchvision detector behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="fasterrcnn", optimizer=None, losses=None,
                 metrics=None, device=None):
        model = TorchdetModel(backbone=backbone, pretrained=True)
        preprocessor = TorchdetPreprocessor()
        postprocessor = TorchdetPostprocessor()
        super().__init__(model, preprocessor, postprocessor, optimizer=optimizer,
                         losses=losses, metrics=metrics, device=device)
        self.set_optimizer(self.optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-4))
        self.set_losses(self.losses or {"loss": BaseLoss()})
        self.set_metrics(self.metrics or {"iou": PolygonIoU()})

    @torch.no_grad()
    def _predict(self, images, image_size):
        self.model.eval()
        raw_output = self.model(images)
        return self.postprocessor(raw_output, image_size).cpu().numpy()

    def train_step(self, images, targets):
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        image_size = images.shape[-1]
        target_list = self.preprocessor(targets, image_size)

        self.model.train()
        self.optimizer.zero_grad()
        loss_dict = self.model(images, target_list)
        loss = sum(loss_dict.values())
        loss.backward()
        self.optimizer.step()
        self.losses["loss"].update(loss.item(), len(images))

        if self.metrics:
            preds = self._predict(images, image_size)
            self.model.train()
            self.update_metrics(preds, targets.cpu().numpy())

        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def eval_step(self, images, targets):
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        image_size = images.shape[-1]
        target_list = self.preprocessor(targets, image_size)

        self.model.train()
        loss_dict = self.model(images, target_list)
        self.losses["loss"].update(sum(loss_dict.values()).item(), len(images))

        preds = self._predict(images, image_size)
        self.update_metrics(preds, targets.cpu().numpy())

        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def predict_step(self, images):
        images = images.to(self.device, non_blocking=True)
        return self._predict(images, images.shape[-1])
