# src/data/smartdoc.py: parse SmartDoc frame_data.csv into gt_corners.csv

import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.geometry import order_corners, is_invalid_corners

CSV_HEADER = ["image_dir", "image_name", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def create_data(data_dir, output_path):
    """Parse SmartDoc raw annotations under data_dir and write gt_corners.csv to output_path."""
    csv_path = os.path.join(data_dir, "frame_data.csv")
    image_dir = os.path.join(data_dir, "images")

    groups = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row["frame_filename"]
            groups.setdefault(fname, {})
            groups[fname].setdefault(row["name"], (float(row["x"]), float(row["y"])))

    rows = []
    for fname, corners in tqdm(sorted(groups.items()), desc="smartdoc"):
        img_path = os.path.join(image_dir, fname)
        if not os.path.exists(img_path):
            continue
        if not all(k in corners for k in ("tl", "tr", "br", "bl")):
            continue

        img_w, img_h = Image.open(img_path).size
        pts = np.array([
            corners["tl"][0] / img_w, corners["tl"][1] / img_h,
            corners["tr"][0] / img_w, corners["tr"][1] / img_h,
            corners["br"][0] / img_w, corners["br"][1] / img_h,
            corners["bl"][0] / img_w, corners["bl"][1] / img_h,
        ], dtype=np.float32).reshape(4, 2)
        ordered = order_corners(pts)
        if is_invalid_corners(ordered):
            continue
        rows.append([os.path.abspath(image_dir), fname] + ["%.4f" % v for v in ordered.reshape(8)])

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
