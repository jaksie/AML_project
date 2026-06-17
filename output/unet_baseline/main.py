#!/usr/bin/env python3
import argparse

import train
import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    print("=== Training ===", flush=True)
    train.main()

    print()
    print("=== Evaluation ===", flush=True)
    evaluate.main(name=args.name)


if __name__ == "__main__":
    main()