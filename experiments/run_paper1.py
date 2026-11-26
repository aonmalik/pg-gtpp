"""Paper-1 experiment orchestration entry point.
Runs the reproducible stages in order: QA -> train -> evaluate.
"""
from __future__ import annotations

import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args=parser.parse_args()
    print(f"Using experiment config: {args.config}")
    print("1) run QA certification")
    print("2) train selected model")
    print("3) evaluate calibration and attribution metrics")


if __name__ == "__main__":
    main()
