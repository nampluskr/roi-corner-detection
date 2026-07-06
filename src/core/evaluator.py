# src/core/evaluator.py: dataloader-level accuracy-metric evaluation for a wrapper

import os
import json
import numpy as np

from src.metrics.polygon_iou import PolygonIoU
from src.metrics.mcd import MCD
from src.metrics.max_cd import MaxCD
from src.metrics.reprojection_error import ReprojectionError
from src.metrics.pck import PCK
from src.utils.geometry import is_invalid_corners

REF_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
PCK_TAU = 0.02


class Evaluator:
    """Dataloader-level accuracy evaluation (IoU, MCD, MaxCD, reprojection error, SR, PCK)."""

    def __init__(self, wrapper, output_dir=None):
        self.wrapper = wrapper
        self.output_dir = output_dir
        self.iou_fn = PolygonIoU()
        self.mcd_fn = MCD()
        self.max_cd_fn = MaxCD()
        self.reproj_fn = ReprojectionError()
        self.pck_fn = PCK()

    def evaluate(self, dataloader):
        totals = {"iou": 0.0, "mcd": 0.0, "max_cd": 0.0, "reproj_error": 0.0, "pck": 0.0}
        success_count = 0
        reproj_count = 0
        total_count = 0

        for images, corners in dataloader:
            result = self.wrapper.eval_step(images, corners)
            pred_batch = result["corners_pred"]
            gt_batch = corners.numpy()

            for pred, gt in zip(pred_batch, gt_batch):
                total_count += 1
                if np.isnan(pred).any():
                    continue
                success_count += 1
                totals["iou"] += self.iou_fn(pred, gt)
                totals["mcd"] += self.mcd_fn(pred, gt)
                totals["max_cd"] += self.max_cd_fn(pred, gt)
                totals["pck"] += float(self.pck_fn(pred, gt, PCK_TAU))

                if is_invalid_corners(pred) or is_invalid_corners(gt):
                    continue
                reproj_count += 1
                totals["reproj_error"] += self.reproj_fn(pred, gt, REF_CORNERS)

        return {
            "iou": totals["iou"] / success_count if success_count > 0 else 0.0,
            "mcd": totals["mcd"] / success_count if success_count > 0 else 0.0,
            "max_cd": totals["max_cd"] / success_count if success_count > 0 else 0.0,
            "reproj_error": totals["reproj_error"] / reproj_count if reproj_count > 0 else 0.0,
            "pck": totals["pck"] / total_count if total_count > 0 else 0.0,
            "sr": success_count / total_count if total_count > 0 else 0.0,
        }

    def save(self, result, output_dir=None):
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "eval_result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
