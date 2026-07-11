# src/core/trainer.py: epoch-level training and evaluation loop for model wrappers

import os
import json
from tqdm import tqdm

from src.core.factory import get_logger

DEFAULT_MONITOR = "iou"


def is_improved(score, best_score, mode, min_delta):
    """Return whether score improves on best_score for the given mode and min_delta."""
    if best_score is None:
        return True
    if mode == "max":
        return score > best_score + min_delta
    return score < best_score - min_delta


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
        result = self.wrapper.get_loss_results()
        result.update(self.wrapper.get_metric_results())
        return result

    def evaluate(self, dataloader):
        self.wrapper.reset_losses()
        self.wrapper.reset_metrics()
        progress = tqdm(dataloader, desc="valid", leave=False, ascii=True)
        for images, targets in progress:
            batch = self.wrapper.eval_step(images, targets)
            progress.set_postfix_str(format_result(batch))
        result = self.wrapper.get_loss_results()
        result.update(self.wrapper.get_metric_results())
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

            score = None
            if valid_loader is not None:
                valid_result = self.evaluate(valid_loader)
                for k, v in valid_result.items():
                    history["valid"].setdefault(k, []).append(v)
                log += " | %s" % format_result(valid_result)
                score = valid_result.get(DEFAULT_MONITOR)
            self.logger.info(log + self.lr_suffix())
            self.wrapper.on_epoch_end(score)
        return history

    def fit_early_stop(self, train_loader, valid_loader, max_epochs=100, patience=10,
                       monitor="iou", mode="max", min_delta=1e-4):
        history = {"train": {}, "valid": {}}
        active = True
        best_score = None
        best_state = None
        best_epoch = 0
        wait = 0

        self.wrapper.on_fit_start(max_epochs)
        for epoch in range(1, max_epochs + 1):
            train_result = self.train(train_loader)
            for k, v in train_result.items():
                history["train"].setdefault(k, []).append(v)
            valid_result = self.evaluate(valid_loader)
            for k, v in valid_result.items():
                history["valid"].setdefault(k, []).append(v)
            self.logger.info("[%2d/%d] %s | %s%s" % (epoch, max_epochs,
                             format_result(train_result), format_result(valid_result),
                             self.lr_suffix()))
            self.wrapper.on_epoch_end(valid_result.get(monitor))

            if active and monitor not in valid_result:
                self.logger.info("early stopping disabled: monitor '%s' not in valid results" % monitor)
                active = False
            if not active:
                continue
            score = valid_result[monitor]
            if is_improved(score, best_score, mode, min_delta):
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.wrapper.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    self.logger.info("early stop at epoch %d (best %s=%.4f @ epoch %d)"
                                     % (epoch, monitor, best_score, best_epoch))
                    break

        if best_state is not None:
            self.wrapper.model.load_state_dict(best_state)
            self.logger.info("restored best weights from epoch %d (%s=%.4f)"
                             % (best_epoch, monitor, best_score))
        return history

    def lr_suffix(self):
        optimizer = self.wrapper.optimizer
        if optimizer is None:
            return ""
        return " | lr=%.1e" % optimizer.param_groups[-1]["lr"]

    def save(self, history, output_dir=None):
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
