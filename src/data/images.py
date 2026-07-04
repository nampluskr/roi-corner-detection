# src/data/images.py: scan a directory of plain images into an image_dir,image_name CSV

import os
import csv
from tqdm import tqdm

CSV_HEADER = ["image_dir", "image_name"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def create_data(data_dir, output_path):
    """Scan image files under data_dir and write an image_dir,image_name CSV to output_path."""
    rows = []
    for image_name in tqdm(sorted(os.listdir(data_dir)), desc="images"):
        if not image_name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        rows.append([os.path.abspath(data_dir), image_name])

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
