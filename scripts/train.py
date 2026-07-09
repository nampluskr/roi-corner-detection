# scripts/train.py: CLI script for training a model on the training set

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import parse_args
from src.utils.io import save_model
from src.core.factory import get_dataloader, get_wrapper
from src.core.trainer import Trainer


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join("outputs", args.method)
    checkpoint = args.checkpoint or os.path.join(output_dir, "model.pth")

    train_loader = get_dataloader("train", args.csv_path, image_size=args.input_size,
                                   batch_size=args.batch_size, seed=args.seed)
    valid_loader = get_dataloader("valid", args.csv_path, image_size=args.input_size,
                                   batch_size=args.batch_size, seed=args.seed)

    wrapper = get_wrapper(args.method, device=args.device)
    trainer = Trainer(wrapper, output_dir=output_dir)
    history = trainer.fit(train_loader, valid_loader, max_epochs=args.max_epochs)

    if args.save:
        save_model(wrapper.model, checkpoint)
        trainer.save(history)


if __name__ == "__main__":
    main()
