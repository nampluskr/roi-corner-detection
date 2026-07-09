# scripts/config.py: shared settings, path helpers, and argument parser

import argparse

DEFAULTS = {
    "data_dir": "/mnt/d/datasets/roi-corner",
    # "csv_path": ["data/smartdoc/gt_corners.csv", "data/midv2020/gt_corners.csv"],
    "csv_path": ["data/smartdoc/gt_corners.csv"],
    "output_dir": "outputs",
    "seed": 42,
    "method": "direct",
    "input_size": 224,
    "batch_size": 16,
    "max_epochs": 3,
}


def get_experiment(cfg):
    """Return experiment name derived from method/batch_size/max_epochs."""
    return "%s_bs%d_ep%d" % (cfg["method"], cfg["batch_size"], cfg["max_epochs"])


def parse_args():
    """Return parsed arguments shared by train.py and evaluate.py."""
    parser = argparse.ArgumentParser()
    parser.set_defaults(**DEFAULTS)
    parser.add_argument("--method")
    parser.add_argument("--device")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir")
    return parser.parse_args()
