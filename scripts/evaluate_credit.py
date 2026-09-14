#!/usr/bin/env python3
"""Evaluate a pathology CREDIT checkpoint on one held-out test fold."""

from __future__ import annotations

import argparse
import sys
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
from pathology_credit.data import CLASS_NAMES, make_tf_dataset
from pathology_credit.metrics import evaluate_credit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pathology_credit.yaml",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def _finite_float(value: Any) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def _hierarchy_summary(y_true: np.ndarray, class_logits: np.ndarray) -> dict[str, Any]:
    probabilities = _softmax(class_logits)

    stage_1_true = y_true == 9
    stage_1_pred = probabilities[:, 9] >= 0.5
    stage_1_correct = stage_1_true == stage_1_pred

    stage_2_keep = y_true != 9
    stage_2_true = y_true[stage_2_keep] == 8
    stage_2_denominator = np.maximum(
        1.0 - probabilities[stage_2_keep, 9], np.finfo(np.float64).tiny
    )
    stage_2_conditional = probabilities[stage_2_keep, 8] / stage_2_denominator
    stage_2_correct = stage_2_true == (stage_2_conditional >= 0.5)

    stage_3_keep = y_true < 8
    stage_3_pred = np.argmax(probabilities[stage_3_keep, :8], axis=-1)
    stage_3_correct = y_true[stage_3_keep] == stage_3_pred

    def stage_payload(correct: np.ndarray) -> dict[str, Any]:
        return {
            "num_patients": int(correct.size),
            "correct": int(np.sum(correct)),
            "accuracy": float(np.mean(correct)),
        }

    return {
        "stage_1_c10_vs_c1_c9": stage_payload(stage_1_correct),
        "stage_2_c9_vs_c1_c8_among_true_c1_c9": stage_payload(stage_2_correct),
        "stage_3_c1_c8_among_true_c1_c8": stage_payload(stage_3_correct),
        "note": (
            "Stage 2 and 3 denominators follow the true path, so upstream "
            "routing errors are reported by the final ten-class metrics."
        ),
    }


def main() -> None:
    args = parse_args()
    try:
        from pathology_credit.training import (
            file_sha256,
            load_credit_student,
            load_fold_records,
            select_model_inputs,
            set_global_seed,
            student_checkpoint_path,
            student_run_dir,
            write_json,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow":
            raise SystemExit(
                "TensorFlow is not installed. Activate the Python 3.10/3.11 "
                "pathology environment first."
            ) from None
        raise
    config = load_config(args.config)
    (
        train_records,
        eval_records,
        test_records,
        splits,
    ) = load_fold_records(config, PROJECT_ROOT, args.fold)
    evaluation_seed = int((config.global_seed + 20_000 + args.fold) % (2**32))
    set_global_seed(evaluation_seed)
    student, checkpoint_metadata = load_credit_student(config, PROJECT_ROOT, args.fold)
    if checkpoint_metadata.get("execution_mode") != STUDENT_EXECUTION_MODE:
        raise ValueError(
            "CREDIT checkpoint was not produced by the required fixed-signature "
            "training path"
        )
    if (
        checkpoint_metadata.get("split_dataset_fingerprint")
        != splits["dataset_fingerprint"]
    ):
        raise ValueError("CREDIT checkpoint was trained on a different cohort")
    expected_partition_ids = {
        "train_case_ids": [record.case_id for record in train_records],
        "eval_case_ids": [record.case_id for record in eval_records],
        "test_case_ids": [record.case_id for record in test_records],
    }
    for metadata_key, expected_ids in expected_partition_ids.items():
        if checkpoint_metadata.get(metadata_key) != expected_ids:
            raise ValueError(
                f"CREDIT checkpoint {metadata_key} does not match the current split"
            )

    checkpoint = student_checkpoint_path(config, PROJECT_ROOT, args.fold)
    metadata_path = student_run_dir(config, PROJECT_ROOT, args.fold) / "metadata.json"
    split_path = project_path(PROJECT_ROOT, config.data.split_file)

    dataset = make_tf_dataset(
        records=test_records,
        batch_size=config.training.eval_batch_size,
        training=False,
        max_patches=(
            None if config.data.max_eval_patches == 0 else config.data.max_eval_patches
        ),
        seed=evaluation_seed,
        feature_dim=config.data.feature_dim,
        feature_key=config.data.feature_key,
        include_ihc=config.model.student_use_ihc,
        include_clinical=config.model.use_clinical_prior,
    )
    labels: list[np.ndarray] = []
    raw_outputs: list[np.ndarray] = []
    for inputs, batch_labels in dataset:
        model_inputs = select_model_inputs(
            inputs,
            config.model.student_use_ihc,
            config.model.use_clinical_prior,
        )
        batch_output = student(model_inputs, training=False)
        labels.append(np.asarray(batch_labels.numpy(), dtype=np.int64))
        raw_outputs.append(np.asarray(batch_output.numpy(), dtype=np.float64))
    y_true = np.concatenate(labels, axis=0)
    output = np.concatenate(raw_outputs, axis=0)
    if y_true.shape[0] != len(test_records):
        raise RuntimeError(
            f"Expected {len(test_records)} test patients, got {y_true.shape[0]}"
        )

    metrics, per_sample = evaluate_credit(
        y_true,
        output,
        num_classes=len(CLASS_NAMES),
        output_mode="mixed",
        class_names=CLASS_NAMES,
        ece_bins=args.ece_bins,
    )
    evaluation_dir = (
        student_run_dir(config, PROJECT_ROOT, args.fold) / "evaluation_test"
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "kind": "pathology_credit_test_evaluation",
        "fold": args.fold,
        "partition": "test",
        "num_patients": len(test_records),
        "case_ids": [record.case_id for record in test_records],
        "class_names": list(CLASS_NAMES),
        "split_dataset_fingerprint": splits["dataset_fingerprint"],
        "split_file_sha256": file_sha256(split_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_metadata["checkpoint_sha256"],
        "student_metadata": str(metadata_path),
        "student_metadata_sha256": file_sha256(metadata_path),
        "student_execution_mode": STUDENT_EXECUTION_MODE,
        "inference_rule": (
            "softmax(hierarchical CREDIT log-probabilities)"
            if config.model.use_hierarchy
            else (
                "softmax(WSI logits + learned strength * clinical-prior logits)"
                if config.model.use_clinical_prior
                else "softmax(raw CREDIT class logits)"
            )
        ),
        "metrics": metrics,
        "config": config.to_dict(),
    }
    if config.model.use_hierarchy:
        summary["hierarchy_metrics"] = _hierarchy_summary(
            y_true, output[:, : len(CLASS_NAMES)]
        )
    if config.model.use_clinical_prior:
        prior_layers = (
            {
                "c10": "c10_prior_strength",
                "c9": "c9_prior_strength",
                "c1_c8": "c1_c8_prior_strength",
            }
            if config.model.use_hierarchy
            else {"flat_10class": "flat_prior_strength"}
        )
        summary["learned_prior_strengths"] = {
            node: float(
                1.0
                / (
                    1.0
                    + np.exp(
                        -float(student.get_layer(layer_name).raw_strength.numpy())
                    )
                )
            )
            for node, layer_name in prior_layers.items()
        }
    write_json(summary, evaluation_dir / "summary.json")

    case_results: list[dict[str, Any]] = []
    for index, record in enumerate(test_records):
        true_index = int(per_sample["y_true"][index])
        predicted_index = int(per_sample["y_pred"][index])
        case_results.append(
            {
                "case_id": record.case_id,
                "true_class_index": true_index,
                "true_class_name": CLASS_NAMES[true_index],
                "predicted_class_index": predicted_index,
                "predicted_class_name": CLASS_NAMES[predicted_index],
                "correct": bool(per_sample["correct"][index]),
                "confidence": _finite_float(per_sample["confidence"][index]),
                "p_star": per_sample["p_star"][index].tolist(),
                "lower_probs": per_sample["lower_probs"][index].tolist(),
                "upper_probs": per_sample["upper_probs"][index].tolist(),
                "interval_lengths": per_sample["interval_lengths"][index].tolist(),
                "beta": _finite_float(per_sample["beta"][index]),
                "TU_bits": _finite_float(per_sample["TU"][index]),
                "AU_bits": _finite_float(per_sample["AU"][index]),
                "EU_bits": _finite_float(per_sample["EU"][index]),
            }
        )
    write_json({"cases": case_results}, evaluation_dir / "cases.json")

    np.savez_compressed(
        evaluation_dir / "per_case_arrays.npz",
        case_id=np.asarray([record.case_id for record in test_records]),
        raw_credit_output=output,
        **per_sample,
    )
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Results: {evaluation_dir}")


if __name__ == "__main__":
    main()
