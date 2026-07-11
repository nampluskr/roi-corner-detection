# src/models/base/base_wrapper.py: base class resolving device placement for training wrappers

import torch


class BaseWrapper:
    """Base class resolving device placement and exposing train/eval/predict step methods."""

    def __init__(self, model, preprocessor, postprocessor, optimizer=None,
                 losses=None, metrics=None, device=None):
        self.set_device(device)
        self.model = model.to(self.device)
        self.set_optimizer(optimizer)
        self.scheduler = None
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.set_losses(losses)
        self.set_metrics(metrics)

    def set_device(self, device):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(self, "model"):
            self.model = self.model.to(self.device)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

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

    def compute_losses(self, raw_output, targets):
        target = self.preprocessor(targets)
        return {name: loss_fn(raw_output, target) for name, loss_fn in self.losses.items()}

    @torch.no_grad()
    def compute_metrics(self, raw_output, targets):
        if not self.metrics:
            return
        preds = self.postprocessor(raw_output).cpu().numpy()
        self.update_metrics(preds, targets.cpu().numpy())

    def get_loss_results(self):
        return {name: loss_fn.compute() for name, loss_fn in self.losses.items()}

    def get_metric_results(self):
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def on_fit_start(self, max_epochs):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2,
            threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def on_epoch_end(self, valid_score=None):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if valid_score is not None:
                self.scheduler.step(valid_score)
        else:
            self.scheduler.step()

    def train_step(self, images, targets):
        self.model.train()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        raw_output = self.model(images)
        losses = self.compute_losses(raw_output, targets)
        loss = sum(losses.values())
        loss.backward()
        self.optimizer.step()

        self.compute_metrics(raw_output, targets)

        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def eval_step(self, images, targets):
        self.model.eval()
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        raw_output = self.model(images)
        self.compute_losses(raw_output, targets)
        self.compute_metrics(raw_output, targets)

        return {**self.get_loss_results(), **self.get_metric_results()}

    @torch.no_grad()
    def predict_step(self, images):
        self.model.eval()
        raw_output = self.model(images.to(self.device, non_blocking=True))
        preds = self.postprocessor(raw_output)
        return preds.cpu().numpy()
