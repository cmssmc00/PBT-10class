#!/usr/bin/env python3
"""Create stable patient-level cross-validation splits for pathology CREDIT."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathology_credit.config import load_config, project_path
from pathology_credit.data import (
    CLASS_NAMES,
    create_patient_splits,
    load_patient_records,
    save_patient_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create stratified patient-level train/eval/test folds "
            "without opening HDF5 feature files."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pathology_credit.yaml",
        help="Experiment YAML used for all unspecified options.",
    )
    parser.add_argument(
        "--patients-json",
        type=Path,
        default=None,
        help="Override the patient JSON configured in YAML.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the data root configured in YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Override the split JSON path configured in YAML."),
    )
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=None,
        help="Fraction of each outer non-test pool assigned to eval.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--ihc-policy",
        choices=("all", "mask_inferred"),
        default=None,
        help="IHC policy stored in loaded records; it does not affect labels.",
    )
    parser.add_argument(
        "--exclude-tumor-flag-zero",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Exclude all tumor_flag=0 records before splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    def resolved_path(override: Path | None, configured: str) -> Path:
        if override is None:
            return project_path(PROJECT_ROOT, configured)
        return override if override.is_absolute() else PROJECT_ROOT / override

    patients_json = resolved_path(args.patients_json, config.data.patients_json)
    data_root = resolved_path(args.data_root, config.data.data_root)
    output = resolved_path(args.output, config.data.split_file)
    n_splits = config.data.n_splits if args.n_splits is None else args.n_splits
    eval_fraction = (
        config.data.eval_fraction if args.eval_fraction is None else args.eval_fraction
    )
    seed = config.data.split_seed if args.seed is None else args.seed
    ihc_policy = config.data.ihc_policy if args.ihc_policy is None else args.ihc_policy
    exclude_tumor_flag_zero = (
        config.data.exclude_tumor_flag_zero
        if args.exclude_tumor_flag_zero is None
        else args.exclude_tumor_flag_zero
    )

    records = load_patient_records(
        patients_json=patients_json,
        data_root=data_root,
        ihc_policy=ihc_policy,
        exclude_tumor_flag_zero=exclude_tumor_flag_zero,
        clinical_json=(
            None
            if config.data.clinical_json is None
            else project_path(PROJECT_ROOT, config.data.clinical_json)
        ),
    )
    splits = create_patient_splits(
        records=records,
        n_splits=n_splits,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    destination = save_patient_splits(splits, output)

    counts = Counter(record.class_name for record in records)
    print(f"Wrote {len(records)} patients to {destination}")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {counts[class_name]}")
    for fold in splits["folds"]:
        print(
            f"  fold {fold['fold']}: train={len(fold['train'])}, "
            f"eval={len(fold['eval'])}, test={len(fold['test'])}"
        )


if __name__ == "__main__":
    main()
