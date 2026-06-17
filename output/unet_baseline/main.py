#!/usr/bin/env python3
import argparse
from pathlib import Path

import train
import evaluate
from utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    print("=== Training ===", flush=True)
    train.main(name=args.name)

    cfg = load_config()
    checkpoint = Path(cfg["paths"]["output_dir"]) / "model" / "best_model.pt"

    print()
    print("=== Evaluation ===", flush=True)
    evaluate.main(
        name=args.name,
        checkpoint=checkpoint,
    )


if __name__ == "__main__":
    main()