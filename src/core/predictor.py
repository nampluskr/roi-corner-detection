# src/core/predictor.py: dataloader-level inference producing pred_corners.csv rows

import os
import numpy as np
import pandas as pd

from src.core.factory import get_logger
from src.data.dataset import Subset

CORNER_COLUMNS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
PRED_COLUMNS = ["image_dir", "image_name"] + CORNER_COLUMNS


def resolve_image_ids(dataset):
    """Return (image_dir, image_name) for each sample in dataset order, unwrapping nested Subsets."""
    ids = []
    for i in range(len(dataset)):
        base, idx = dataset, i
        while isinstance(base, Subset):
            idx = base.indices[idx]
            base = base.dataset
        sample = base.samples[idx]
        path = sample[0] if isinstance(sample, tuple) else sample
        ids.append(os.path.split(path))
    return ids


class Predictor:
    """Runs a wrapper over a dataloader and collects standard corners into a pred_corners.csv DataFrame."""

    def __init__(self, wrapper, output_dir=None):
        self.wrapper = wrapper
        self.output_dir = output_dir
        self.logger = get_logger("predictor", output_dir)

    def predict(self, dataloader):
        image_ids = resolve_image_ids(dataloader.dataset)
        preds = []
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            preds.append(self.wrapper.predict_step(images))
        preds = np.concatenate(preds, axis=0).reshape(len(image_ids), 8)

        rows = []
        for (image_dir, image_name), corner in zip(image_ids, preds):
            rows.append([image_dir, image_name] + [float(v) for v in corner])
        result = pd.DataFrame(rows, columns=PRED_COLUMNS)
        self.logger.info("predicted %d samples" % len(result))
        return result

    def save(self, preds, output_dir=None):
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "pred_corners.csv")
        preds.to_csv(path, index=False, float_format="%.4f")
