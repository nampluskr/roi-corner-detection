# src/core/trainer.py: epoch-level training and evaluation loop for model wrappers

import os
import json
from tqdm import tqdm

from src.core.factory import get_logger


def format_result(result):
    """Format a result dict as a space-separated key=value string."""
    return " ".join("%s=%.3f" % (k, v) for k, v in result.items())


class Trainer:
    """Epoch-level training and evaluation loop for a wrapper's train_step/eval_step."""

    def __init__(self, wrapper, metrics=None, output_dir=None):
        self.wrapper = wrapper
        self.output_dir = output_dir
        self.logger = get_logger("trainer", output_dir)
        if metrics is not None:
            self.wrapper.set_metrics(metrics)

    def train(self, dataloader):
        self.wrapper.reset_losses()
        self.wrapper.reset_metrics()
        progress = tqdm(dataloader, desc="train", leave=False, ascii=True)
        for images, targets in progress:
            batch = self.wrapper.train_step(images, targets)
            progress.set_postfix_str(format_result(batch))
        result = self.wrapper.compute_losses()
        result.update(self.wrapper.compute_metrics())
        return result

    def evaluate(self, dataloader):
        self.wrapper.reset_losses()
        self.wrapper.reset_metrics()
        progress = tqdm(dataloader, desc="valid", leave=False, ascii=True)
        for images, targets in progress:
            batch = self.wrapper.eval_step(images, targets)
            progress.set_postfix_str(format_result(batch))
        result = self.wrapper.compute_losses()
        result.update(self.wrapper.compute_metrics())
        return result

    def fit(self, train_loader, valid_loader=None, max_epochs=10):
        history = {"train": {}}
        if valid_loader is not None:
            history["valid"] = {}

        self.wrapper.on_fit_start(max_epochs)
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
            self.logger.info(log)
            self.wrapper.on_epoch_end()
        return history

    def save(self, history, output_dir=None):
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
