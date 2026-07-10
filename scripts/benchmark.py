# scripts/benchmark.py: evaluate all trained methods on the same test set into a comparison table

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import parse_args, get_experiment, get_output_dir, CONFIGS
import torch
from src.utils.io import load_model
from src.utils.measure import measure_parameters, measure_size_mb, measure_latency
from src.core.factory import get_dataloader, get_wrapper, get_logger
from src.core.evaluator import Evaluator


def benchmark_config(cfg, args, logger):
    """Evaluate one config's checkpoint and measure its size/latency; return a result row or None."""
    method = cfg["method"]
    exp_name = get_experiment(cfg)
    output_dir = get_output_dir(cfg)
    checkpoint = os.path.join(output_dir, "model.pth")
    if not os.path.exists(checkpoint):
        logger.info("skip %s: no checkpoint at %s" % (exp_name, checkpoint))
        return None
    try:
        wrapper = get_wrapper(method, device=args.device)
    except NotImplementedError:
        logger.info("skip %s: wrapper not implemented" % exp_name)
        return None

    load_model(wrapper.model, checkpoint)
    test_loader = get_dataloader("test", args.csv_path, image_size=args.image_size,
                                 batch_size=args.batch_size, seed=args.seed,
                                 num_workers=args.num_workers, num_samples=args.test_size)
    metrics = Evaluator(wrapper, output_dir=output_dir).evaluate(test_loader)

    row = {"experiment": exp_name, "method": method}
    row.update(metrics)
    row["cpu_latency_ms"] = measure_latency(wrapper.model, "cpu", image_size=args.image_size)
    if torch.cuda.is_available():
        row["gpu_latency_ms"] = measure_latency(wrapper.model, "cuda", image_size=args.image_size)
    else:
        row["gpu_latency_ms"] = float("nan")
    row["params"] = measure_parameters(wrapper.model)
    row["size_mb"] = measure_size_mb(wrapper.model)
    return row


def main():
    args = parse_args()
    output_dir = os.path.join("outputs", "comparison")
    logger = get_logger("benchmark", output_dir)

    rows = []
    for cfg in CONFIGS:
        row = benchmark_config(cfg, args, logger)
        if row is not None:
            rows.append(row)

    if not rows:
        logger.info("no trained configs found; nothing to benchmark")
        return

    result = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "results.csv")
    result.to_csv(csv_path, index=False, float_format="%.4f")
    logger.info("wrote %d method(s) to %s" % (len(rows), csv_path))


if __name__ == "__main__":
    main()
