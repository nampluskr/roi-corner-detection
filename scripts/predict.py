# scripts/predict.py: CLI script for running inference and writing pred_corners.csv

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import parse_args, get_output_dir, get_wrapper_kwargs
from src.utils.io import load_model
from src.core.factory import get_dataloader, get_wrapper
from src.core.predictor import Predictor


def main():
    args = parse_args()
    output_dir = args.output_dir or get_output_dir(vars(args))
    checkpoint = args.checkpoint or os.path.join(output_dir, "model.pth")

    wrapper = get_wrapper(args.method, device=args.device, **get_wrapper_kwargs(args))
    load_model(wrapper.model, checkpoint)

    test_loader = get_dataloader("test", args.csv_path, image_size=args.image_size,
                                 batch_size=args.batch_size, seed=args.seed,
                                 num_workers=args.num_workers, num_samples=args.test_size)

    predictor = Predictor(wrapper, output_dir=output_dir)
    result = predictor.predict(test_loader)

    if args.save:
        predictor.save(result)


if __name__ == "__main__":
    main()
