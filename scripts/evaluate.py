# scripts/evaluate.py: CLI script for evaluating a trained model

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import parse_args
from src.utils.io import load_model
from src.core.factory import get_dataloader, get_wrapper
from src.core.evaluator import Evaluator


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join("outputs", args.method)
    checkpoint = args.checkpoint or os.path.join(output_dir, "model.pth")

    wrapper = get_wrapper(args.method, device=args.device)
    load_model(wrapper.model, checkpoint)

    valid_loader = get_dataloader("valid", args.csv_path, image_size=args.input_size,
                                   batch_size=args.batch_size, seed=args.seed)

    evaluator = Evaluator(wrapper, output_dir=output_dir)
    result = evaluator.evaluate(valid_loader)

    if args.save:
        evaluator.save(result)


if __name__ == "__main__":
    main()
