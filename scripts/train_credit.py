#!/usr/bin/env python3
"""Distill a frozen teacher ensemble into one pathology CREDIT student."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathology_credit.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pathology_credit.yaml",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow an existing CREDIT checkpoint to be replaced.",
    )
    parser.add_argument("--verbose", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from pathology_credit.training import train_credit_student
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow":
            raise SystemExit(
                "TensorFlow is not installed. Activate the Python 3.10/3.11 "
                "pathology environment first."
            ) from None
        raise
    config = load_config(args.config)
    checkpoint = train_credit_student(
        config,
        PROJECT_ROOT,
        args.fold,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    print(f"CREDIT student complete: {checkpoint}")


if __name__ == "__main__":
    main()
