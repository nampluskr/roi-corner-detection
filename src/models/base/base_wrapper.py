# src/models/base/base_wrapper.py: base class resolving device placement for training wrappers

import torch


class BaseWrapper:
    """Base class resolving device placement and exposing train/eval/predict step methods."""

    def __init__(self, model, optimizer=None, preprocessor=None, postprocessor=None,
                 losses=None, metrics=None, device=None):
        self.set_device(device)
        self.model = model.to(self.device)
        self.set_optimizer(optimizer)
        self.set_preprocessor(preprocessor)
        self.set_postprocessor(postprocessor)
        self.set_losses(losses)
        self.set_metrics(metrics)

    def set_device(self, device):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self, "model"):
            self.model = self.model.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def set_postprocessor(self, postprocessor):
        self.postprocessor = postprocessor

    def set_losses(self, losses=None):
        self.losses = losses or {}

    def set_metrics(self, metrics=None):
        self.metrics = metrics or {}

    def reset_losses(self):
        for loss_fn in self.losses.values():
            loss_fn.reset()

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def update_metrics(self, preds, targets):
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute_losses(self):
        return {name: loss_fn.compute() for name, loss_fn in self.losses.items()}

    def compute_metrics(self):
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def train_step(self, images, targets):
        raise NotImplementedError

    def eval_step(self, images, targets):
        raise NotImplementedError

    def predict_step(self, images):
        raise NotImplementedError
