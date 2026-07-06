# src/models/direct/wrapper.py: composes DirectModel/DirectPreprocessor/DirectPostprocessor/DirectLoss

import torch

from src.models.base.base_wrapper import BaseWrapper
from src.models.direct.model import DirectModel
from src.models.direct.preprocessor import DirectPreprocessor
from src.models.direct.postprocessor import DirectPostprocessor
from src.models.direct.loss import DirectLoss


class DirectWrapper(BaseWrapper):
    """Wraps DirectModel training/evaluation/inference behind the shared Trainer/Evaluator/Predictor interface."""

    def __init__(self, backbone="resnet18", lr=1e-3, optimizer=None, device=None):
        model = DirectModel(backbone=backbone, pretrained=True)
        super().__init__(model, optimizer=optimizer, device=device)
        self.optimizer = self.optimizer or torch.optim.Adam(self.model.parameters(), lr=lr)
        self.preprocessor = DirectPreprocessor()
        self.postprocessor = DirectPostprocessor()
        self.loss_fn = DirectLoss()

    def train_step(self, images, corners):
        self.model.train()
        images, corners = images.to(self.device), corners.to(self.device)
        target = self.preprocessor(corners)

        self.optimizer.zero_grad()
        raw_output = self.model(images)
        loss = self.loss_fn(raw_output, target)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    @torch.no_grad()
    def eval_step(self, images, corners):
        self.model.eval()
        images, corners = images.to(self.device), corners.to(self.device)
        target = self.preprocessor(corners)

        raw_output = self.model(images)
        loss = self.loss_fn(raw_output, target)
        corners_pred = self.postprocessor(raw_output)

        return {"loss": loss.item(), "corners_pred": corners_pred.cpu().numpy()}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        raw_output = self.model(images.to(self.device))
        corners_pred = self.postprocessor(raw_output)
        return corners_pred.cpu().numpy()
