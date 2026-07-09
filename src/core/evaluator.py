# src/core/evaluator.py: dataloader-level accuracy-metric evaluation for a wrapper

import os
import json
from tqdm import tqdm

from src.core.factory import get_logger
from src.core.trainer import format_result
from src.metrics.polygon_iou import PolygonIoU
from src.metrics.mcd import MCD
from src.metrics.max_cd import MaxCD
from src.metrics.reprojection_error import ReprojectionError
from src.metrics.pck import PCK
from src.metrics.success_rate import SuccessRate

DEFAULT_METRICS = {
    "iou": PolygonIoU(),
    "mcd": MCD(),
    "max_cd": MaxCD(),
    "reproj_error": ReprojectionError(),
    "pck": PCK(),
    "sr": SuccessRate(),
}


class Evaluator:
    """Dataloader-level accuracy evaluation (IoU, MCD, MaxCD, reprojection error, SR, PCK)."""

    def __init__(self, wrapper, metrics=None, output_dir=None):
        self.wrapper = wrapper
        self.output_dir = output_dir
        self.logger = get_logger("evaluator", output_dir)
        self.wrapper.set_metrics(metrics if metrics is not None else DEFAULT_METRICS)

    def evaluate(self, dataloader):
        self.wrapper.reset_losses()
        self.wrapper.reset_metrics()
        for images, targets in tqdm(dataloader, desc="eval", leave=False, ascii=True):
            self.wrapper.eval_step(images, targets)
        result = self.wrapper.compute_metrics()
        self.logger.info(format_result(result))
        return result

    def save(self, result, output_dir=None):
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "eval_result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
