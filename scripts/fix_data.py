# scripts/fix_data.py: reorder corners in existing gt_corners.csv files and remove degenerate samples

import os
import sys
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.geometry import order_corners, is_invalid_corners

CSV_HEADER = ["image_dir", "image_name", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def fix_csv(csv_path, min_dist=0.02):
    """Reorder corners in csv_path to TL,TR,BR,BL and drop degenerate rows, overwriting csv_path."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fixed = 0
    removed = 0
    out_rows = []
    for row in rows:
        corners = np.array([
            row["x1"], row["y1"], row["x2"], row["y2"],
            row["x3"], row["y3"], row["x4"], row["y4"],
        ], dtype=np.float32).reshape(4, 2)
        ordered = order_corners(corners)
        if is_invalid_corners(ordered, min_dist):
            removed += 1
            continue
        if not np.allclose(corners, ordered):
            fixed += 1

        flat = ordered.reshape(8)
        row["x1"], row["y1"], row["x2"], row["y2"] = ["%.6f" % v for v in flat[:4]]
        row["x3"], row["y3"], row["x4"], row["y4"] = ["%.6f" % v for v in flat[4:]]
        out_rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(out_rows)

    print("[OK] %s  total=%d  fixed=%d  removed=%d" % (csv_path, len(rows), fixed, removed))


def parse_args():
    parser = argparse.ArgumentParser(description="Fix corner order in gt_corners.csv files")
    parser.add_argument("csv_paths", nargs="+", help="Path(s) to gt_corners.csv")
    parser.add_argument("--min_dist", type=float, default=0.02,
                         help="Minimum distance between any two corners (default: 0.02)")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in args.csv_paths:
        if not os.path.exists(path):
            print("[SKIP] not found: %s" % path)
            continue
        fix_csv(path, min_dist=args.min_dist)


if __name__ == "__main__":
    main()
