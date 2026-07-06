# src/core/trainer.py: epoch-level training and evaluation loop for model wrappers

import os
import json
import numpy as np


def format_result(result):
    """Format a result dict as a space-separated key=value string."""
    return " ".join("%s=%.4f" % (k, v) for k, v in result.items())


class Trainer:
    """Epoch-level training and evaluation loop for a wrapper's train_step/eval_step."""

    def __init__(self, wrapper, output_dir=None):
        self.wrapper = wrapper
        self.output_dir = output_dir

    def train(self, dataloader):
        result = {}
        for images, corners in dataloader:
            batch = self.wrapper.train_step(images, corners)
            for k, v in batch.items():
                result.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in result.items()}

    def evaluate(self, dataloader):
        result = {}
        for images, corners in dataloader:
            batch = self.wrapper.eval_step(images, corners)
            for k, v in batch.items():
                if isinstance(v, (int, float)):
                    result.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in result.items()}

    def fit(self, train_loader, valid_loader=None, max_epochs=10):
        history = {"train": {}}
        if valid_loader is not None:
            history["valid"] = {}

        for epoch in range(1, max_epochs + 1):
            train_result = self.train(train_loader)
            for k, v in train_result.items():
                history["train"].setdefault(k, []).append(v)
            log = "[%2d/%d] %s" % (epoch, max_epochs, format_result(train_result))

            if valid_loader is not None:
                valid_result = self.evaluate(valid_loader)
                for k, v in valid_result.items():
                    history["valid"].setdefault(k, []).append(v)
                log += " | %s" % format_result(valid_result)
            print(log)
        return history

    def save(self, history, output_dir=None):
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
