#!/usr/bin/env python3
"""Aggregate held-out predictions from every fold into one OOF evaluation.

The script intentionally consumes only ``evaluation_test`` artifacts.  It
checks their patient identity, ordering, labels, class mapping, and cohort
fingerprint against the configured split before computing any metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathology_credit.config import (
    STUDENT_EXECUTION_MODE,
    load_config,
    project_path,
)
from pathology_credit.data import (
    CLASS_NAMES,
    load_patient_records,
    load_patient_splits,
    validate_splits_against_records,
)
from pathology_credit.metrics import evaluate_credit, expected_calibration_error


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pathology_credit.yaml",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=_positive_int,
        default=2000,
        help="Number of patient-level bootstrap replicates (default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Bootstrap RNG seed. By default, use global_seed + 30000 from "
            "the configuration."
        ),
    )
    parser.add_argument(
        "--ece-bins",
        type=_positive_int,
        default=15,
        help="Number of equal-width bins for ECE (default: 15).",
    )
    return parser.parse_args()


def _assert_json_finite(value: Any, *, location: str = "root") -> None:
    """Reject non-standard NaN/Infinity values accepted by ``json.loads``."""

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite JSON number at {location}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _assert_json_finite(child, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_json_finite(child, location=f"{location}[{index}]")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Required evaluation summary not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    _assert_json_finite(payload, location=str(path))
    return payload


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    _assert_json_finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _decode_case_id(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_fold_summary(
    summary: Mapping[str, Any],
    *,
    path: Path,
    fold_index: int,
    expected_train_case_ids: Sequence[str],
    expected_eval_case_ids: Sequence[str],
    expected_test_case_ids: Sequence[str],
    dataset_fingerprint: str,
    split_file_sha256: str,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "kind": "pathology_credit_test_evaluation",
        "fold": fold_index,
        "partition": "test",
        "num_patients": len(expected_test_case_ids),
        "class_names": list(CLASS_NAMES),
        "split_dataset_fingerprint": dataset_fingerprint,
        "split_file_sha256": split_file_sha256,
        "student_execution_mode": STUDENT_EXECUTION_MODE,
    }
    for key, expected in checks.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"Fold {fold_index} summary field {key!r} does not match the "
                f"current split/config: observed={summary.get(key)!r}, "
                f"expected={expected!r} ({path})"
            )
    if summary.get("case_ids") != list(expected_test_case_ids):
        raise ValueError(
            f"Fold {fold_index} summary case_ids do not exactly match the "
            f"ordered split test IDs ({path})"
        )
    checkpoint_value = summary.get("checkpoint")
    checkpoint_hash = summary.get("checkpoint_sha256")
    if not isinstance(checkpoint_value, str) or not isinstance(checkpoint_hash, str):
        raise TypeError(f"Fold {fold_index} has invalid checkpoint provenance")
    checkpoint_path = Path(checkpoint_value)
    expected_checkpoint_path = path.parent.parent / "best.weights.h5"
    if not checkpoint_path.is_absolute():
        raise ValueError(f"Fold {fold_index} checkpoint path must be absolute")
    if checkpoint_path.resolve() != expected_checkpoint_path.resolve():
        raise ValueError(
            f"Fold {fold_index} summary points to the wrong student checkpoint: "
            f"observed={checkpoint_path}, expected={expected_checkpoint_path}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Fold {fold_index} checkpoint no longer exists: {checkpoint_path}"
        )
    observed_checkpoint_hash = _file_sha256(checkpoint_path)
    if observed_checkpoint_hash != checkpoint_hash:
        raise ValueError(f"Fold {fold_index} checkpoint SHA-256 mismatch")

    if summary.get("config") != dict(expected_config):
        raise ValueError(
            f"Fold {fold_index} evaluation was produced with a different "
            f"experiment configuration ({path})"
        )
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError(f"Fold {fold_index} summary has no metrics object: {path}")
    if metrics.get("num_samples") != len(expected_test_case_ids):
        raise ValueError(f"Fold {fold_index} summary metric sample count mismatch")
    if metrics.get("num_classes") != len(CLASS_NAMES):
        raise ValueError(f"Fold {fold_index} summary metric class count mismatch")

    metadata_path = checkpoint_path.parent / "metadata.json"
    metadata = _read_json(metadata_path)
    if summary.get("student_metadata") != str(metadata_path):
        raise ValueError(
            f"Fold {fold_index} summary points to the wrong student metadata"
        )
    observed_metadata_hash = _file_sha256(metadata_path)
    if summary.get("student_metadata_sha256") != observed_metadata_hash:
        raise ValueError(f"Fold {fold_index} student metadata SHA-256 mismatch")
    metadata_checks = {
        "kind": "pathology_credit_student",
        "fold": fold_index,
        "class_names": list(CLASS_NAMES),
        "split_dataset_fingerprint": dataset_fingerprint,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "execution_mode": STUDENT_EXECUTION_MODE,
        "train_case_ids": list(expected_train_case_ids),
        "eval_case_ids": list(expected_eval_case_ids),
        "test_case_ids": list(expected_test_case_ids),
        "config": dict(expected_config),
    }
    for key, expected in metadata_checks.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Fold {fold_index} student metadata field {key!r} does not "
                f"match the current split/config ({metadata_path})"
            )
    recipe = metadata.get("recipe")
    if not isinstance(recipe, Mapping):
        raise TypeError(f"Fold {fold_index} student metadata has no recipe")
    if recipe.get("execution_mode") != STUDENT_EXECUTION_MODE:
        raise ValueError(
            f"Fold {fold_index} student recipe was not produced by the required "
            "fixed-signature training path"
        )
    data_provenance = metadata.get("data_provenance")
    if not isinstance(data_provenance, Mapping):
        raise TypeError(f"Fold {fold_index} student metadata has no data provenance")
    if data_provenance.get("split_file_sha256") != split_file_sha256:
        raise ValueError(
            f"Fold {fold_index} student metadata belongs to a different split file"
        )

    teacher_ensemble = metadata.get("teacher_ensemble")
    training_config = expected_config.get("training")
    if not isinstance(training_config, Mapping):
        raise TypeError("Expected experiment config has no training section")
    teacher_seeds = training_config.get("teacher_seeds")
    if not isinstance(teacher_seeds, list):
        raise TypeError("Expected experiment config has no teacher seed list")
    expected_teacher_seeds = list(teacher_seeds)
    if not isinstance(teacher_ensemble, list) or len(teacher_ensemble) != len(
        expected_teacher_seeds
    ):
        raise ValueError(
            f"Fold {fold_index} student metadata does not contain the configured "
            "teacher ensemble"
        )
    teacher_sources: list[dict[str, Any]] = []
    for teacher_index, (member, seed) in enumerate(
        zip(teacher_ensemble, expected_teacher_seeds)
    ):
        if not isinstance(member, Mapping):
            raise TypeError(
                f"Fold {fold_index} teacher {teacher_index} provenance is invalid"
            )
        member_checks = {
            "teacher_index": teacher_index,
            "seed": seed,
            "split_dataset_fingerprint": dataset_fingerprint,
            "train_case_ids": list(expected_train_case_ids),
            "eval_case_ids": list(expected_eval_case_ids),
            "test_case_ids": list(expected_test_case_ids),
        }
        for key, expected in member_checks.items():
            if member.get(key) != expected:
                raise ValueError(
                    f"Fold {fold_index} teacher {teacher_index} metadata field "
                    f"{key!r} does not match the current three-way split"
                )
        teacher_checkpoint_value = member.get("checkpoint")
        teacher_checkpoint_hash = member.get("checkpoint_sha256")
        if not isinstance(teacher_checkpoint_value, str) or not isinstance(
            teacher_checkpoint_hash, str
        ):
            raise TypeError(
                f"Fold {fold_index} teacher {teacher_index} checkpoint provenance "
                "is invalid"
            )
        teacher_checkpoint = Path(teacher_checkpoint_value)
        expected_teacher_checkpoint = (
            checkpoint_path.parent.parent
            / "teachers"
            / f"teacher_{teacher_index}_seed_{seed}"
            / "best.weights.h5"
        )
        if not teacher_checkpoint.is_absolute():
            raise ValueError(
                f"Fold {fold_index} teacher {teacher_index} checkpoint path must "
                "be absolute"
            )
        if teacher_checkpoint.resolve() != expected_teacher_checkpoint.resolve():
            raise ValueError(
                f"Fold {fold_index} teacher {teacher_index} checkpoint is not at "
                "the expected fold-specific path"
            )
        if not teacher_checkpoint.is_file():
            raise FileNotFoundError(
                f"Fold {fold_index} teacher {teacher_index} checkpoint no longer "
                f"exists: {teacher_checkpoint}"
            )
        observed_teacher_hash = _file_sha256(teacher_checkpoint)
        if observed_teacher_hash != teacher_checkpoint_hash:
            raise ValueError(
                f"Fold {fold_index} teacher {teacher_index} checkpoint SHA-256 mismatch"
            )

        teacher_metadata_path = teacher_checkpoint.parent / "metadata.json"
        teacher_metadata = _read_json(teacher_metadata_path)
        teacher_metadata_checks = {
            "kind": "pathology_credit_teacher",
            "fold": fold_index,
            "teacher_index": teacher_index,
            "seed": seed,
            "class_names": list(CLASS_NAMES),
            "split_dataset_fingerprint": dataset_fingerprint,
            "checkpoint": str(teacher_checkpoint),
            "checkpoint_sha256": teacher_checkpoint_hash,
            "train_case_ids": list(expected_train_case_ids),
            "eval_case_ids": list(expected_eval_case_ids),
            "test_case_ids": list(expected_test_case_ids),
            "config": dict(expected_config),
        }
        for key, expected in teacher_metadata_checks.items():
            if teacher_metadata.get(key) != expected:
                raise ValueError(
                    f"Fold {fold_index} teacher {teacher_index} checkpoint "
                    f"metadata field {key!r} does not match the current split/config"
                )
        teacher_data_provenance = teacher_metadata.get("data_provenance")
        if not isinstance(teacher_data_provenance, Mapping):
            raise TypeError(
                f"Fold {fold_index} teacher {teacher_index} metadata has no data "
                "provenance"
            )
        if teacher_data_provenance.get("split_file_sha256") != split_file_sha256:
            raise ValueError(
                f"Fold {fold_index} teacher {teacher_index} metadata belongs to "
                "a different split file"
            )
        teacher_sources.append(
            {
                "teacher_index": teacher_index,
                "seed": int(seed),
                "metadata": str(teacher_metadata_path),
                "metadata_sha256": _file_sha256(teacher_metadata_path),
                "checkpoint": str(teacher_checkpoint),
                "checkpoint_sha256": observed_teacher_hash,
            }
        )

    return {
        "fold": fold_index,
        "num_patients": len(expected_test_case_ids),
        "summary": str(path),
        "student_metadata": str(metadata_path),
        "student_metadata_sha256": observed_metadata_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": observed_checkpoint_hash,
        "student_execution_mode": STUDENT_EXECUTION_MODE,
        "teacher_checkpoints": teacher_sources,
    }


def _load_and_validate_fold(
    *,
    evaluation_dir: Path,
    fold_index: int,
    expected_train_case_ids: Sequence[str],
    expected_eval_case_ids: Sequence[str],
    expected_test_case_ids: Sequence[str],
    expected_labels: Mapping[str, int],
    dataset_fingerprint: str,
    split_file_sha256: str,
    expected_config: Mapping[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, Any]]:
    summary_path = evaluation_dir / "summary.json"
    arrays_path = evaluation_dir / "per_case_arrays.npz"
    summary = _read_json(summary_path)
    source = _validate_fold_summary(
        summary,
        path=summary_path,
        fold_index=fold_index,
        expected_train_case_ids=expected_train_case_ids,
        expected_eval_case_ids=expected_eval_case_ids,
        expected_test_case_ids=expected_test_case_ids,
        dataset_fingerprint=dataset_fingerprint,
        split_file_sha256=split_file_sha256,
        expected_config=expected_config,
    )

    try:
        archive_context = np.load(arrays_path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Required fold prediction archive not found: {arrays_path}"
        ) from exc
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not read prediction archive {arrays_path}: {exc}"
        ) from exc

    with archive_context as archive:
        required = {"case_id", "y_true", "raw_credit_output"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"Fold {fold_index} archive is missing arrays: {sorted(missing)}"
            )
        case_id_array = np.asarray(archive["case_id"])
        y_true = np.asarray(archive["y_true"])
        raw_output = np.asarray(archive["raw_credit_output"], dtype=np.float64)

    if case_id_array.ndim != 1:
        raise ValueError(f"Fold {fold_index} case_id must be one-dimensional")
    case_ids = [_decode_case_id(value) for value in case_id_array.tolist()]
    if case_ids != list(expected_test_case_ids):
        observed_set = set(case_ids)
        expected_set = set(expected_test_case_ids)
        raise ValueError(
            f"Fold {fold_index} NPZ case_id does not exactly match the ordered "
            f"split test IDs; missing={sorted(expected_set - observed_set)[:5]}, "
            f"unexpected={sorted(observed_set - expected_set)[:5]}"
        )
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Fold {fold_index} NPZ contains duplicate case IDs")

    if y_true.ndim != 1 or y_true.shape[0] != len(case_ids):
        raise ValueError(
            f"Fold {fold_index} y_true must have shape ({len(case_ids)},), "
            f"got {y_true.shape}"
        )
    if y_true.dtype.kind not in "iu":
        raise ValueError(f"Fold {fold_index} y_true must have an integer dtype")
    y_true = y_true.astype(np.int64, copy=False)
    expected_y = np.asarray(
        [expected_labels[case_id] for case_id in case_ids], dtype=np.int64
    )
    if not np.array_equal(y_true, expected_y):
        mismatch = int(np.flatnonzero(y_true != expected_y)[0])
        raise ValueError(
            f"Fold {fold_index} label/class mapping mismatch for "
            f"{case_ids[mismatch]}: NPZ={int(y_true[mismatch])}, "
            f"cohort={int(expected_y[mismatch])}"
        )

    expected_width = 2 * len(CLASS_NAMES) + 1
    if raw_output.shape != (len(case_ids), expected_width):
        raise ValueError(
            f"Fold {fold_index} raw_credit_output must have shape "
            f"({len(case_ids)}, {expected_width}), got {raw_output.shape}"
        )
    if not np.all(np.isfinite(raw_output)):
        raise ValueError(f"Fold {fold_index} raw_credit_output contains NaN/Infinity")

    source["predictions"] = str(arrays_path)
    source["predictions_sha256"] = _file_sha256(arrays_path)
    return case_ids, y_true, raw_output, source


def _mean_finite(values: np.ndarray) -> float | None:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    return float(np.mean(values[finite]))


def _replicate_metrics(
    indices: np.ndarray,
    *,
    y_true: np.ndarray,
    p_star: np.ndarray,
    per_sample: Mapping[str, np.ndarray],
    ece_bins: int,
) -> tuple[dict[str, float | None], int]:
    labels = y_true[indices]
    probabilities = p_star[indices]
    predicted = np.argmax(probabilities, axis=1)
    num_classes = p_star.shape[1]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    supports = np.sum(confusion, axis=1)
    present = supports > 0
    true_positives = np.diag(confusion)
    predicted_counts = np.sum(confusion, axis=0)
    recall = np.divide(
        true_positives,
        supports,
        out=np.zeros(num_classes, dtype=np.float64),
        where=present,
    )
    precision = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_counts > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    ece, _ = expected_calibration_error(labels, probabilities, num_bins=ece_bins)

    metrics: dict[str, float | None] = {
        "accuracy": float(np.mean(predicted == labels)),
        "balanced_accuracy": float(np.mean(recall[present])),
        # Absent true classes are omitted in a bootstrap replicate. The full
        # OOF estimate still contains all classes and therefore is unchanged.
        "macro_f1": float(np.mean(f1[present])),
        "nll": float(np.mean(np.asarray(per_sample["nll"])[indices])),
        "brier": float(np.mean(np.asarray(per_sample["brier"])[indices])),
        "ece": float(ece),
        "mean_TU": _mean_finite(np.asarray(per_sample["TU"])[indices]),
        "mean_AU": _mean_finite(np.asarray(per_sample["AU"])[indices]),
        "mean_EU": _mean_finite(np.asarray(per_sample["EU"])[indices]),
    }
    return metrics, int(np.sum(present))


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _bootstrap_confidence_intervals(
    *,
    y_true: np.ndarray,
    per_sample: Mapping[str, np.ndarray],
    point_metrics: Mapping[str, Any],
    samples: int,
    seed: int,
    ece_bins: int,
) -> dict[str, Any]:
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "nll",
        "brier",
        "ece",
        "mean_TU",
        "mean_AU",
        "mean_EU",
    )
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    num_patients = int(y_true.shape[0])
    missing_class_replicates = 0
    p_star = np.asarray(per_sample["p_star"], dtype=np.float64)

    for _ in range(samples):
        indices = rng.integers(0, num_patients, size=num_patients)
        replicate, num_present_classes = _replicate_metrics(
            indices,
            y_true=y_true,
            p_star=p_star,
            per_sample=per_sample,
            ece_bins=ece_bins,
        )
        if num_present_classes < p_star.shape[1]:
            missing_class_replicates += 1
        for name in metric_names:
            value = replicate[name]
            if value is not None and math.isfinite(value):
                values[name].append(float(value))

    estimates = {
        "accuracy": point_metrics["accuracy"],
        "balanced_accuracy": point_metrics["balanced_accuracy"],
        "macro_f1": point_metrics["macro_f1"],
        "nll": point_metrics["nll"],
        "brier": point_metrics["brier"],
        "ece": point_metrics["ece"],
        "mean_TU": point_metrics["uncertainty"]["mean_TU"],
        "mean_AU": point_metrics["uncertainty"]["mean_AU"],
        "mean_EU": point_metrics["uncertainty"]["mean_EU"],
    }
    intervals: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        distribution = np.asarray(values[name], dtype=np.float64)
        if distribution.size:
            lower, upper = np.percentile(distribution, [2.5, 97.5])
            lower_value: float | None = float(lower)
            upper_value: float | None = float(upper)
        else:
            lower_value = None
            upper_value = None
        intervals[name] = {
            "estimate": _finite_or_none(estimates[name]),
            "lower_95": lower_value,
            "upper_95": upper_value,
            "bootstrap_samples_valid": int(distribution.size),
        }

    return {
        "method": "patient-level nonparametric bootstrap with replacement",
        "interval": "95% percentile",
        "confidence_level": 0.95,
        "bootstrap_samples_requested": samples,
        "seed": seed,
        "num_replicates_missing_one_or_more_classes": missing_class_replicates,
        "missing_class_handling": (
            "Balanced accuracy and macro F1 average over true classes present "
            "in each replicate; all other metrics remain patient-level means "
            "or top-label ECE. Uncertainty means ignore only patients whose "
            "entropy calculation is marked non-finite."
        ),
        "metrics": intervals,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bootstrap_seed = (
        int(args.seed) if args.seed is not None else int(config.global_seed + 30_000)
    )

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
    split_path = project_path(PROJECT_ROOT, config.data.split_file)
    splits = load_patient_splits(split_path)
    validate_splits_against_records(splits, records)
    if splits["class_names"] != list(CLASS_NAMES):
        raise ValueError("Split class mapping does not match pathology_credit.data")
    split_config_checks = {
        "n_splits": config.data.n_splits,
        "seed": config.data.split_seed,
        "eval_fraction": config.data.eval_fraction,
    }
    for key, expected in split_config_checks.items():
        if splits.get(key) != expected:
            raise ValueError(
                f"Split field {key!r} does not match the current config: "
                f"observed={splits.get(key)!r}, expected={expected!r}"
            )

    record_by_id = {record.case_id: record for record in records}
    expected_labels = {record.case_id: int(record.class_index) for record in records}
    experiment_dir = project_path(PROJECT_ROOT, config.output_dir) / config.name
    split_file_sha256 = _file_sha256(split_path)

    accumulated_case_ids: list[str] = []
    accumulated_folds: list[int] = []
    accumulated_labels: list[np.ndarray] = []
    accumulated_outputs: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    for fold_payload in splits["folds"]:
        fold_index = int(fold_payload["fold"])
        expected_test_case_ids = list(fold_payload["test"])
        evaluation_dir = (
            experiment_dir / f"fold_{fold_index}" / "student" / "evaluation_test"
        )
        case_ids, labels, outputs, source = _load_and_validate_fold(
            evaluation_dir=evaluation_dir,
            fold_index=fold_index,
            expected_train_case_ids=list(fold_payload["train"]),
            expected_eval_case_ids=list(fold_payload["eval"]),
            expected_test_case_ids=expected_test_case_ids,
            expected_labels=expected_labels,
            dataset_fingerprint=splits["dataset_fingerprint"],
            split_file_sha256=split_file_sha256,
            expected_config=config.to_dict(),
        )
        accumulated_case_ids.extend(case_ids)
        accumulated_folds.extend([fold_index] * len(case_ids))
        accumulated_labels.append(labels)
        accumulated_outputs.append(outputs)
        sources.append(source)

    membership = Counter(accumulated_case_ids)
    expected_ids = set(record_by_id)
    if set(membership) != expected_ids or any(
        count != 1 for count in membership.values()
    ):
        missing = sorted(expected_ids - set(membership))
        unexpected = sorted(set(membership) - expected_ids)
        duplicates = sorted(
            case_id for case_id, count in membership.items() if count != 1
        )
        raise ValueError(
            "OOF artifacts must contain every configured patient exactly once; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, "
            f"non_unique={duplicates[:5]}"
        )

    concatenated_labels = np.concatenate(accumulated_labels, axis=0)
    concatenated_outputs = np.concatenate(accumulated_outputs, axis=0)
    fold_array = np.asarray(accumulated_folds, dtype=np.int32)
    position = {case_id: index for index, case_id in enumerate(accumulated_case_ids)}
    ordered_case_ids = [record.case_id for record in records]
    reorder = np.asarray([position[case_id] for case_id in ordered_case_ids])
    y_true = concatenated_labels[reorder]
    raw_output = concatenated_outputs[reorder]
    fold_array = fold_array[reorder]

    expected_ordered_labels = np.asarray(
        [record.class_index for record in records], dtype=np.int64
    )
    if not np.array_equal(y_true, expected_ordered_labels):
        raise RuntimeError(
            "OOF labels changed while restoring authoritative patient order"
        )

    metrics, per_sample = evaluate_credit(
        y_true,
        raw_output,
        num_classes=len(CLASS_NAMES),
        output_mode="mixed",
        class_names=CLASS_NAMES,
        ece_bins=args.ece_bins,
    )
    bootstrap = _bootstrap_confidence_intervals(
        y_true=y_true,
        per_sample=per_sample,
        point_metrics=metrics,
        samples=args.bootstrap_samples,
        seed=bootstrap_seed,
        ece_bins=args.ece_bins,
    )

    output_dir = experiment_dir / "oof_evaluation"
    summary = {
        "kind": "pathology_credit_oof_evaluation",
        "partition": "outer_test_out_of_fold",
        "num_patients": len(ordered_case_ids),
        "num_folds": len(splits["folds"]),
        "class_names": list(CLASS_NAMES),
        "split_dataset_fingerprint": splits["dataset_fingerprint"],
        "split_file_sha256": split_file_sha256,
        "student_execution_mode": STUDENT_EXECUTION_MODE,
        "prediction_rule": "softmax of the raw CREDIT class logits",
        "ece_bins": args.ece_bins,
        "metrics": metrics,
        "bootstrap": bootstrap,
        "fold_sources": sources,
        "config": config.to_dict(),
    }

    cases: list[dict[str, Any]] = []
    for index, case_id in enumerate(ordered_case_ids):
        true_index = int(per_sample["y_true"][index])
        predicted_index = int(per_sample["y_pred"][index])
        cases.append(
            {
                "case_id": case_id,
                "fold": int(fold_array[index]),
                "true_class_index": true_index,
                "true_class_name": CLASS_NAMES[true_index],
                "predicted_class_index": predicted_index,
                "predicted_class_name": CLASS_NAMES[predicted_index],
                "correct": bool(per_sample["correct"][index]),
                "confidence": _finite_or_none(per_sample["confidence"][index]),
                "p_star": per_sample["p_star"][index].tolist(),
                "lower_probs": per_sample["lower_probs"][index].tolist(),
                "upper_probs": per_sample["upper_probs"][index].tolist(),
                "interval_lengths": per_sample["interval_lengths"][index].tolist(),
                "beta": _finite_or_none(per_sample["beta"][index]),
                "TU_bits": _finite_or_none(per_sample["TU"][index]),
                "AU_bits": _finite_or_none(per_sample["AU"][index]),
                "EU_bits": _finite_or_none(per_sample["EU"][index]),
                "entropy_valid": bool(per_sample["entropy_valid"][index]),
            }
        )

    # Write the completion marker (summary.json) last so a present summary
    # always refers to fully written case-level artifacts.
    _atomic_savez(
        output_dir / "oof_predictions.npz",
        case_id=np.asarray(ordered_case_ids),
        fold=fold_array,
        raw_credit_output=raw_output,
        **per_sample,
    )
    _atomic_write_json(
        {
            "kind": "pathology_credit_oof_cases",
            "num_patients": len(cases),
            "class_names": list(CLASS_NAMES),
            "cases": cases,
        },
        output_dir / "cases.json",
    )
    _atomic_write_json(summary, output_dir / "summary.json")

    print(f"OOF patients: {len(ordered_case_ids)} across {len(splits['folds'])} folds")
    print(f"OOF accuracy: {metrics['accuracy']:.4f}")
    print(f"OOF macro F1: {metrics['macro_f1']:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
