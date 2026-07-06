# src/models/base/base_wrapper.py: base class resolving device placement for training wrappers

import torch


class BaseWrapper:
    """Base class resolving device placement and exposing train/eval/predict step methods."""

    def __init__(self, model, optimizer=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer

    def train_step(self, images, corners):
        raise NotImplementedError

    def eval_step(self, images, corners):
        raise NotImplementedError

    def predict_step(self, images):
        raise NotImplementedError
