# scripts/create_data.py: CLI entry point for converting raw dataset annotations to gt_corners.csv

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import smartdoc, midv2020


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw dataset annotations to gt_corners.csv")
    parser.add_argument("--dataset", choices=["smartdoc", "midv2020"], required=True)
    parser.add_argument("--data_dir", required=True, help="Raw dataset directory")
    parser.add_argument("--output_path", required=True, help="Destination gt_corners.csv path")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset == "smartdoc":
        smartdoc.create_data(args.data_dir, args.output_path)
    elif args.dataset == "midv2020":
        midv2020.create_data(args.data_dir, args.output_path)


if __name__ == "__main__":
    main()
