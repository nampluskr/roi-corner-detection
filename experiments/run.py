# experiments/run.py: batch runner for train/evaluate/predict via subprocess

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import get_experiment
from experiments.configs import CONFIGS

MODES = ["train", "evaluate", "predict"]

PASS_KEYS = ["device", "batch_size", "max_epochs", "num_workers",
             "train_size", "valid_size", "test_size", "checkpoint", "output_dir"]


def get_cli_args(cfg):
    """Return CLI args for scripts/{mode}.py built from one config dict."""
    args = ["--method", cfg["method"], "--save"]
    for key in PASS_KEYS:
        if key in cfg:
            args += ["--%s" % key, str(cfg[key])]
    return args


def run(mode, configs):
    """Run scripts/{mode}.py for each config via subprocess and report a summary."""
    script = os.path.join("scripts", "%s.py" % mode)
    total = len(configs)
    results = []

    for i, cfg in enumerate(configs, 1):
        exp_name = get_experiment(cfg)
        print("\n[%d/%d] %s | %s" % (i, total, mode, exp_name))
        cmd = [sys.executable, script] + get_cli_args(cfg)
        try:
            subprocess.run(cmd, check=True)
            results.append({"exp_name": exp_name, "success": True, "error": None})
            print("[OK] %s" % exp_name)
        except subprocess.CalledProcessError as e:
            results.append({"exp_name": exp_name, "success": False, "error": str(e)})
            print("[FAIL] %s: %s" % (exp_name, e))

    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    print("\n[done] %s: %d/%d success, %d failed" % (mode, len(success), total, len(failed)))
    for r in failed:
        print("  [FAIL] %s: %s" % (r["exp_name"], r["error"]))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    args = parser.parse_args()

    modes = MODES if args.mode == "all" else [args.mode]
    for mode in modes:
        run(mode, CONFIGS)