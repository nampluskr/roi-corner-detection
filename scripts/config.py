# scripts/config.py: shared settings, path helpers, and argument parser

import os
import argparse

DEFAULTS = {
    "data_dir": "/mnt/d/datasets/roi-corner",
    "csv_path": ["data/smartdoc/gt_corners.csv", "data/midv2020/gt_corners.csv"],
    # "csv_path": ["data/smartdoc/gt_corners.csv"],
    "seed": 42,
    "method": "direct",
    "image_size": 224,
    "batch_size": 4,
    "max_epochs": 50,
    "patience": 5,
    "num_workers": 4,
    "train_size": 2000,     # None - all train samples
    "valid_size": 1000,     # None - all valid samples
    "test_size": 1000,      # None - all test samples
}

DEFAULT_BACKBONES = {
    "seg": "resnet50",
    "hybrid": "mobilenet_v3_large",
    "torchseg": "deeplabv3_resnet50",
    "torchdet": "fasterrcnn",
}


def get_experiment(cfg):
    """Return experiment name derived from method/batch_size/max_epochs/backbone."""
    method = cfg["method"]
    backbone = cfg.get("backbone") or DEFAULT_BACKBONES.get(method, "resnet50")
    return "%s_bs%d_ep%d_%s" % (method, cfg["batch_size"], cfg["max_epochs"], backbone)


def get_output_dir(cfg, base="outputs"):
    """Return outputs/{method}/{exp_name} directory for a config dict."""
    return os.path.join(base, cfg["method"], get_experiment(cfg))


def get_wrapper_kwargs(args):
    """Return optional model kwargs (e.g. backbone) to pass through get_wrapper."""
    kwargs = {}
    if args.backbone is not None:
        kwargs["backbone"] = args.backbone
    return kwargs


def parse_args():
    """Return parsed arguments shared by train.py and evaluate.py."""
    parser = argparse.ArgumentParser()
    parser.set_defaults(**DEFAULTS)
    parser.add_argument("--method")
    parser.add_argument("--backbone")
    parser.add_argument("--device")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--train_size", type=int)
    parser.add_argument("--valid_size", type=int)
    parser.add_argument("--test_size", type=int)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output_dir")
    return parser.parse_args()
