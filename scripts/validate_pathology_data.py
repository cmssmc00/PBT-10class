#!/usr/bin/env python3
"""Validate patient JSON and CONCH HDF5 bags before starting training."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathology_credit.config import load_config, project_path
from pathology_credit.data import CLASS_NAMES, IHC_NAMES, load_patient_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pathology_credit.yaml",
    )
    parser.add_argument(
        "--check-values",
        action="store_true",
        help="Also stream through every feature and reject NaN/Inf.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the default data_validation.json output path.",
    )
    return parser.parse_args()


def _write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def main() -> None:
    args = parse_args()
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required; activate the pathology environment first"
        ) from exc

    config = load_config(args.config)
    records = load_patient_records(
        patients_json=project_path(PROJECT_ROOT, config.data.patients_json),
        data_root=project_path(PROJECT_ROOT, config.data.data_root),
        ihc_policy=config.data.ihc_policy,
        exclude_tumor_flag_zero=config.data.exclude_tumor_flag_zero,
        clinical_json=(
            None
            if config.data.clinical_json is None
            else project_path(PROJECT_ROOT, config.data.clinical_json)
        ),
    )
    patch_counts: list[int] = []
    dtypes: Counter[str] = Counter()
    for record in records:
        patient_patches = 0
        for h5_path in record.h5_paths:
            with h5py.File(h5_path, "r") as h5_file:
                if config.data.feature_key not in h5_file:
                    raise KeyError(
                        f"{record.case_id}: {h5_path} has no "
                        f"{config.data.feature_key!r} dataset"
                    )
                features = h5_file[config.data.feature_key]
                if features.ndim != 2 or features.shape[1] != config.data.feature_dim:
                    raise ValueError(
                        f"{record.case_id}: expected [N, "
                        f"{config.data.feature_dim}] in {h5_path}, got "
                        f"{features.shape}"
                    )
                if features.shape[0] == 0:
                    raise ValueError(f"{record.case_id}: empty bag in {h5_path}")
                if not np.issubdtype(features.dtype, np.floating):
                    raise TypeError(
                        f"{record.case_id}: non-floating features in "
                        f"{h5_path}: {features.dtype}"
                    )
                if args.check_values:
                    for start in range(0, int(features.shape[0]), 8192):
                        block = np.asarray(features[start : start + 8192])
                        if not np.all(np.isfinite(block)):
                            raise ValueError(
                                f"{record.case_id}: NaN/Inf features in {h5_path}"
                            )
                patient_patches += int(features.shape[0])
                dtypes[str(features.dtype)] += 1
        patch_counts.append(patient_patches)

    class_counts = Counter(record.class_name for record in records)
    inferred_counts = {
        marker: int(sum(record.ihc_inferred[index] for record in records))
        for index, marker in enumerate(IHC_NAMES)
    }
    summary = {
        "kind": "pathology_credit_data_validation",
        "num_patients": len(records),
        "num_wsi": int(sum(len(record.h5_paths) for record in records)),
        "feature_key": config.data.feature_key,
        "feature_dim": config.data.feature_dim,
        "feature_dtypes": dict(sorted(dtypes.items())),
        "total_patches": int(sum(patch_counts)),
        "min_patches_per_patient": int(min(patch_counts)),
        "median_patches_per_patient": float(np.median(patch_counts)),
        "max_patches_per_patient": int(max(patch_counts)),
        "class_counts": {name: int(class_counts.get(name, 0)) for name in CLASS_NAMES},
        "inferred_ihc_counts": inferred_counts,
        "checked_all_values_for_finiteness": bool(args.check_values),
    }
    if config.data.clinical_json is not None:
        summary["clinical_prior"] = {
            "clinical_json": str(
                project_path(PROJECT_ROOT, config.data.clinical_json)
            ),
            "age_available": int(sum(record.age_mask for record in records)),
            "location_available": int(
                sum(record.location_mask for record in records)
            ),
            "location_missing_or_unknown": int(
                sum(not record.location_mask for record in records)
            ),
        }
    output = args.output
    if output is None:
        output = (
            project_path(PROJECT_ROOT, config.output_dir)
            / config.name
            / "data_validation.json"
        )
    elif not output.is_absolute():
        output = PROJECT_ROOT / output
    _write_json(summary, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Validation: {output}")


if __name__ == "__main__":
    main()
