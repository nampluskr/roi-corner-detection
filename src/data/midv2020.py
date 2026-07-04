# src/data/midv2020.py: extract corners from MIDV-2020 binary masks into gt_corners.csv

import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.geometry import mask_to_corners, order_corners, is_invalid_corners

CSV_HEADER = ["image_dir", "image_name", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def create_data(data_dir, output_path):
    """Parse MIDV-2020 binary masks under data_dir and write gt_corners.csv to output_path."""
    mask_dir = os.path.join(data_dir, "masks")
    image_dir = os.path.join(data_dir, "images")

    rows = []
    for mask_fname in tqdm(sorted(os.listdir(mask_dir)), desc="midv2020"):
        if not mask_fname.endswith(".png"):
            continue

        stem = os.path.splitext(mask_fname)[0]
        img_fname = stem + ".jpg"
        img_path = os.path.join(image_dir, img_fname)
        if not os.path.exists(img_path):
            continue

        mask = np.array(Image.open(os.path.join(mask_dir, mask_fname)).convert("L"))
        ordered = order_corners(mask_to_corners(mask))
        if ordered.sum() == 0 or is_invalid_corners(ordered):
            continue

        rows.append([os.path.abspath(image_dir), img_fname] + ["%.4f" % v for v in ordered.reshape(8)])

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
