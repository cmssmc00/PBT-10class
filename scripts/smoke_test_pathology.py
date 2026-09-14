#!/usr/bin/env python3
"""Run one temporary forward/backward/checkpoint pathology smoke test."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathology_credit.config import load_config, project_path
from pathology_credit.data import (
    CLASS_NAMES,
    load_patient_records,
    load_patient_splits,
    make_tf_dataset,
    select_records,
    validate_splits_against_records,
)


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
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--patients", type=_positive_int, default=2)
    parser.add_argument("--max-patches", type=_positive_int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import tensorflow as tf

        from pathology_credit.ced import ced_loss, stack_teacher_logits
        from pathology_credit.models import (
            build_credit_student,
            build_fusion_teacher,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow":
            raise SystemExit(
                "TensorFlow is not installed. Activate the Python 3.10/3.11 "
                "pathology environment first."
            ) from None
        raise
    if importlib.util.find_spec("h5py") is None:
        raise SystemExit(
            "h5py is not installed. Activate the pathology environment first."
        )

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
    splits = load_patient_splits(project_path(PROJECT_ROOT, config.data.split_file))
    validate_splits_against_records(splits, records)
    if not 0 <= args.fold < len(splits["folds"]):
        raise ValueError(f"fold must be in [0, {len(splits['folds']) - 1}]")
    selected_ids = splits["folds"][args.fold]["train"][: args.patients]
    selected = select_records(records, selected_ids)
    dataset = make_tf_dataset(
        records=selected,
        batch_size=len(selected),
        training=False,
        max_patches=args.max_patches,
        seed=config.global_seed,
        feature_dim=config.data.feature_dim,
        feature_key=config.data.feature_key,
        include_ihc=(
            config.model.teacher_use_ihc or config.model.student_use_ihc
        ),
        include_clinical=config.model.use_clinical_prior,
    )
    inputs, labels = next(iter(dataset))

    architecture = {
        "feature_dim": config.data.feature_dim,
        "num_classes": len(CLASS_NAMES),
        "projection_dim": config.model.projection_dim,
        "attention_dim": config.model.attention_dim,
        "ihc_hidden_dim": config.model.ihc_hidden_dim,
        "fusion_dim": config.model.fusion_dim,
        "patch_dropout": config.model.patch_dropout,
        "fusion_dropout": config.model.fusion_dropout,
        "l2": config.model.l2,
        "ihc_mask_as_feature": config.model.ihc_mask_as_feature,
    }
    if config.model.use_hierarchy or config.model.use_clinical_prior:
        architecture.update(
            {
                "use_hierarchy": config.model.use_hierarchy,
                "use_clinical_prior": config.model.use_clinical_prior,
                "age_hidden_dim": config.model.age_hidden_dim,
                "location_embedding_dim": config.model.location_embedding_dim,
                "clinical_hidden_dim": config.model.clinical_hidden_dim,
                "num_locations": config.model.num_locations,
                "age_scale_years": config.model.age_scale_years,
                "prior_strength_init": config.model.prior_strength_init,
            }
        )

    def model_inputs(use_ihc: bool) -> dict[str, tf.Tensor]:
        keys = list(
            ("patches", "patch_mask", "ihc", "ihc_mask")
            if use_ihc
            else ("patches", "patch_mask")
        )
        if config.model.use_clinical_prior:
            keys.extend(("age", "age_mask", "location", "location_mask"))
        return {key: inputs[key] for key in keys}

    def regularization(model: tf.keras.Model, dtype: tf.dtypes.DType) -> tf.Tensor:
        if not model.losses:
            return tf.zeros((), dtype=dtype)
        return tf.add_n([tf.cast(loss, dtype) for loss in model.losses])

    tf.keras.utils.set_random_seed(config.training.teacher_seeds[0])
    teacher_a = build_fusion_teacher(
        **architecture, use_ihc=config.model.teacher_use_ihc
    )
    teacher_optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-4)
    with tf.GradientTape() as tape:
        teacher_logits = teacher_a(
            model_inputs(config.model.teacher_use_ihc), training=True
        )
        teacher_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, teacher_logits, from_logits=True
            )
        ) + regularization(teacher_a, teacher_logits.dtype)
    teacher_gradients = tape.gradient(teacher_loss, teacher_a.trainable_variables)
    teacher_pairs = [
        (gradient, variable)
        for gradient, variable in zip(teacher_gradients, teacher_a.trainable_variables)
        if gradient is not None
    ]
    if not teacher_pairs:
        raise RuntimeError("Teacher smoke step produced no gradients")
    teacher_optimizer.apply_gradients(teacher_pairs)
    if not np.isfinite(float(teacher_loss)):
        raise FloatingPointError("Teacher smoke loss is NaN/Inf")
    teacher_probabilities = tf.nn.softmax(teacher_logits, axis=-1).numpy()
    if not np.allclose(
        teacher_probabilities.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError("Teacher class probabilities do not sum to one")

    tf.keras.utils.set_random_seed(config.training.teacher_seeds[1])
    teacher_b = build_fusion_teacher(
        **architecture, use_ihc=config.model.teacher_use_ihc
    )
    teacher_a.trainable = False
    teacher_b.trainable = False

    student_seed = int((config.global_seed + 10_000 + args.fold) % (2**32))
    tf.keras.utils.set_random_seed(student_seed)
    student = build_credit_student(**architecture, use_ihc=config.model.student_use_ihc)
    teacher_inputs = model_inputs(config.model.teacher_use_ihc)
    student_inputs = model_inputs(config.model.student_use_ihc)
    frozen_logits = tf.stop_gradient(
        stack_teacher_logits((teacher_a, teacher_b), teacher_inputs, training=False)
    )
    student_optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-4)
    with tf.GradientTape() as tape:
        student_output = student(student_inputs, training=True)
        distillation_loss = ced_loss(
            student_output,
            frozen_logits,
            distillation_temperature=config.training.distillation_temperature,
        )
        student_loss = distillation_loss + regularization(
            student, distillation_loss.dtype
        )
    student_gradients = tape.gradient(student_loss, student.trainable_variables)
    student_pairs = [
        (gradient, variable)
        for gradient, variable in zip(student_gradients, student.trainable_variables)
        if gradient is not None
    ]
    if not student_pairs:
        raise RuntimeError("CREDIT smoke step produced no gradients")
    student_optimizer.apply_gradients(student_pairs)
    if not np.isfinite(float(student_loss)):
        raise FloatingPointError("CREDIT smoke loss is NaN/Inf")
    reference_output = np.asarray(
        student(student_inputs, training=False).numpy(), dtype=np.float32
    )

    with tempfile.TemporaryDirectory(prefix="pathology-credit-smoke-") as temp_dir:
        checkpoint = Path(temp_dir) / "student.weights.h5"
        student.save_weights(checkpoint)
        restored = build_credit_student(
            **architecture, use_ihc=config.model.student_use_ihc
        )
        restored.load_weights(checkpoint)
        restored_output = np.asarray(
            restored(student_inputs, training=False).numpy(), dtype=np.float32
        )
    if reference_output.shape != (len(selected), 2 * len(CLASS_NAMES) + 1):
        raise RuntimeError(f"Unexpected CREDIT output shape: {reference_output.shape}")
    if not np.all(np.isfinite(reference_output)):
        raise FloatingPointError("CREDIT smoke output contains NaN/Inf")
    student_probabilities = tf.nn.softmax(
        reference_output[:, : len(CLASS_NAMES)], axis=-1
    ).numpy()
    if not np.allclose(
        student_probabilities.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6
    ):
        raise RuntimeError("Student class probabilities do not sum to one")
    if not np.allclose(reference_output, restored_output, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Restored CREDIT weights do not reproduce predictions")

    print(
        f"Smoke test passed: patients={len(selected)}, "
        f"patches<={args.max_patches}, teacher_loss={float(teacher_loss):.5f}, "
        f"student_loss={float(student_loss):.5f}, output={reference_output.shape}, "
        f"hierarchy={config.model.use_hierarchy}, "
        f"clinical_prior={config.model.use_clinical_prior}"
    )


if __name__ == "__main__":
    main()
