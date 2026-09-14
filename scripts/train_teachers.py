#!/usr/bin/env python3
"""Train the five pathology fusion teachers for one patient-level fold."""

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
        help="Allow existing teacher checkpoints to be replaced.",
    )
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from pathology_credit.training import train_teacher_ensemble
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow":
            raise SystemExit(
                "TensorFlow is not installed. Activate the Python 3.10/3.11 "
                "pathology environment first."
            ) from None
        raise
    config = load_config(args.config)
    checkpoints = train_teacher_ensemble(
        config,
        PROJECT_ROOT,
        args.fold,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    print("Teacher ensemble complete:")
    for checkpoint in checkpoints:
        print(f"  {checkpoint}")


if __name__ == "__main__":
    main()
