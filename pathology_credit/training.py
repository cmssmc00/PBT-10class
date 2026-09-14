"""Training orchestration for the pathology teacher ensemble and CREDIT.

The teacher members share one architecture and all hyperparameters.  Their
random seeds differ, which changes initialization, shuffled patient order,
patch subsampling, and dropout masks.  The CREDIT student is then optimized
only against the frozen ensemble's credal targets, following the paper's CED
objective.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from .ced import ced_loss, credit_probabilities, stack_teacher_logits
from .config import STUDENT_EXECUTION_MODE, ExperimentConfig, project_path
from .data import (
    CLASS_NAMES,
    IHC_NAMES,
    PatientRecord,
    load_patient_records,
    load_patient_splits,
    make_tf_dataset,
    select_records,
    validate_splits_against_records,
)
from .models import build_credit_student, build_fusion_teacher

NUM_CLASSES = len(CLASS_NAMES)
MODEL_INPUT_KEYS = ("patches", "patch_mask", "ihc", "ihc_mask")
IMAGE_INPUT_KEYS = ("patches", "patch_mask")
CLINICAL_INPUT_KEYS = ("age", "age_mask", "location", "location_mask")
StudentBatchStep = Callable[
    [Mapping[str, tf.Tensor], tf.Tensor],
    tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
]


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow for a reproducible run."""

    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    enable_determinism = getattr(tf.config.experimental, "enable_op_determinism", None)
    if enable_determinism is not None:
        enable_determinism()


def write_json(payload: Mapping[str, Any], destination: str | Path) -> Path:
    """Atomically write a JSON artifact."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
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
    os.replace(temporary, path)
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experiment_records(
    config: ExperimentConfig, project_root: str | Path
) -> list[PatientRecord]:
    """Load the cohort selected by the experiment configuration."""

    return load_patient_records(
        patients_json=project_path(project_root, config.data.patients_json),
        data_root=project_path(project_root, config.data.data_root),
        ihc_policy=config.data.ihc_policy,
        exclude_tumor_flag_zero=config.data.exclude_tumor_flag_zero,
        clinical_json=(
            None
            if config.data.clinical_json is None
            else project_path(project_root, config.data.clinical_json)
        ),
    )


def load_fold_records(
    config: ExperimentConfig,
    project_root: str | Path,
    fold: int,
) -> tuple[
    list[PatientRecord],
    list[PatientRecord],
    list[PatientRecord],
    dict[str, Any],
]:
    """Load one patient-level train/eval/test fold.

    ``train`` fits model weights, ``eval`` selects checkpoints, and ``test`` is
    held out for the final fold report.
    """

    records = load_experiment_records(config, project_root)
    splits = load_patient_splits(project_path(project_root, config.data.split_file))
    validate_splits_against_records(splits, records)
    if splits["seed"] != config.data.split_seed:
        raise ValueError(
            "Split seed does not match data.split_seed; regenerate splits or "
            "use the matching config"
        )
    if splits["n_splits"] != config.data.n_splits:
        raise ValueError("Split n_splits does not match data.n_splits")
    if not np.isclose(splits["eval_fraction"], config.data.eval_fraction):
        raise ValueError("Split eval_fraction does not match data.eval_fraction")
    if isinstance(fold, bool) or not 0 <= fold < len(splits["folds"]):
        raise ValueError(f"fold must be in [0, {len(splits['folds']) - 1}], got {fold}")
    fold_payload = splits["folds"][fold]
    train_records = select_records(records, fold_payload["train"])
    eval_records = select_records(records, fold_payload["eval"])
    test_records = select_records(records, fold_payload["test"])
    return train_records, eval_records, test_records, splits


def balanced_class_weights(
    records: Sequence[PatientRecord],
) -> dict[int, float]:
    """Compute sklearn-style balanced weights from the training partition."""

    counts = Counter(record.class_index for record in records)
    missing = [index for index in range(NUM_CLASSES) if counts[index] == 0]
    if missing:
        raise ValueError(f"Training partition is missing classes: {missing}")
    total = len(records)
    return {
        index: total / (NUM_CLASSES * counts[index]) for index in range(NUM_CLASSES)
    }


def _max_patches(value: int) -> int | None:
    return None if value == 0 else value


def _architecture(config: ExperimentConfig, *, use_ihc: bool) -> dict[str, Any]:
    architecture = {
        "feature_dim": config.data.feature_dim,
        "num_classes": NUM_CLASSES,
        "projection_dim": config.model.projection_dim,
        "attention_dim": config.model.attention_dim,
        "ihc_hidden_dim": config.model.ihc_hidden_dim,
        "fusion_dim": config.model.fusion_dim,
        "patch_dropout": config.model.patch_dropout,
        "fusion_dropout": config.model.fusion_dropout,
        "l2": config.model.l2,
        "use_ihc": use_ihc,
        "ihc_mask_as_feature": config.model.ihc_mask_as_feature,
    }
    # Keep the historical architecture dictionary byte-for-byte compatible
    # for old flat checkpoints; new keys exist only in clinical/hierarchical runs.
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
    return architecture


def _teacher_recipe(config: ExperimentConfig) -> dict[str, Any]:
    """Canonical settings that can change supervised teacher weights."""

    return {
        "architecture": _architecture(config, use_ihc=config.model.teacher_use_ihc),
        "ihc_policy": config.data.ihc_policy,
        "feature_key": config.data.feature_key,
        "max_train_patches": config.data.max_train_patches,
        "max_eval_patches": config.data.max_eval_patches,
        "split_seed": config.data.split_seed,
        "n_splits": config.data.n_splits,
        "eval_fraction": config.data.eval_fraction,
        "optimizer": "Adam",
        "learning_rate": config.training.teacher_learning_rate,
        "gradient_clip_norm": config.training.gradient_clip_norm,
        "class_weighting": config.training.class_weighting,
        "batch_size": config.training.batch_size,
        "eval_batch_size": config.training.eval_batch_size,
        "max_epochs": config.training.teacher_epochs,
        "patience": config.training.patience,
    }


def _student_recipe(config: ExperimentConfig) -> dict[str, Any]:
    """Canonical settings that can change CREDIT student weights."""

    return {
        "architecture": _architecture(config, use_ihc=config.model.student_use_ihc),
        "teacher_recipe": _teacher_recipe(config),
        "teacher_seeds": list(config.training.teacher_seeds),
        "ihc_policy": config.data.ihc_policy,
        "feature_key": config.data.feature_key,
        "max_train_patches": config.data.max_train_patches,
        "max_eval_patches": config.data.max_eval_patches,
        "split_seed": config.data.split_seed,
        "n_splits": config.data.n_splits,
        "eval_fraction": config.data.eval_fraction,
        "optimizer": "Adam",
        "execution_mode": STUDENT_EXECUTION_MODE,
        "learning_rate": config.training.student_learning_rate,
        "gradient_clip_norm": config.training.gradient_clip_norm,
        "distillation_temperature": config.training.distillation_temperature,
        "batch_size": config.training.batch_size,
        "eval_batch_size": config.training.eval_batch_size,
        "max_epochs": config.training.student_epochs,
        "patience": config.training.patience,
        "global_seed": config.global_seed,
    }


def _data_provenance(
    config: ExperimentConfig,
    project_root: str | Path,
    records: Sequence[PatientRecord],
) -> dict[str, Any]:
    """Fingerprint labels/IHC JSON and the resolved HDF5 file manifest."""

    patients_json = project_path(project_root, config.data.patients_json)
    split_file = project_path(project_root, config.data.split_file)
    manifest_rows: list[str] = []
    for record in records:
        for wsi_id, h5_path in zip(record.wsi_ids, record.h5_paths):
            resolved = h5_path.resolve()
            stat = resolved.stat()
            manifest_rows.append(
                "\t".join(
                    (
                        record.case_id,
                        wsi_id,
                        str(h5_path),
                        str(resolved),
                        str(stat.st_size),
                        str(stat.st_mtime_ns),
                    )
                )
            )
    manifest = "\n".join(sorted(manifest_rows)).encode()
    provenance = {
        "patients_json": str(patients_json),
        "patients_json_sha256": file_sha256(patients_json),
        "split_file": str(split_file),
        "split_file_sha256": file_sha256(split_file),
        "h5_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "num_h5_files": len(manifest_rows),
        "manifest_definition": ("case_id,wsi_id,link_path,resolved_path,size,mtime_ns"),
    }
    if config.data.clinical_json is not None:
        clinical_json = project_path(project_root, config.data.clinical_json)
        provenance.update(
            {
                "clinical_json": str(clinical_json),
                "clinical_json_sha256": file_sha256(clinical_json),
            }
        )
    return provenance


def _build_teacher(config: ExperimentConfig) -> tf.keras.Model:
    return build_fusion_teacher(
        **_architecture(config, use_ihc=config.model.teacher_use_ihc)
    )


def _build_student(config: ExperimentConfig) -> tf.keras.Model:
    return build_credit_student(
        **_architecture(config, use_ihc=config.model.student_use_ihc)
    )


def select_model_inputs(
    inputs: Mapping[str, tf.Tensor],
    use_ihc: bool,
    use_clinical_prior: bool = False,
) -> dict[str, tf.Tensor]:
    """Select the exact Keras input dictionary for one modality setting."""

    keys = list(MODEL_INPUT_KEYS if use_ihc else IMAGE_INPUT_KEYS)
    if use_clinical_prior:
        keys.extend(CLINICAL_INPUT_KEYS)
    missing = [key for key in keys if key not in inputs]
    if missing:
        raise KeyError(f"Dataset batch is missing model inputs: {missing}")
    return {key: inputs[key] for key in keys}


def _adapt_dataset_inputs(
    dataset: Any, use_ihc: bool, use_clinical_prior: bool = False
) -> Any:
    """Select only the modality keys accepted by a Keras model."""

    if len(dataset.element_spec) == 2:
        return dataset.map(
            lambda inputs, label: (
                select_model_inputs(inputs, use_ihc, use_clinical_prior),
                label,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if len(dataset.element_spec) == 3:
        return dataset.map(
            lambda inputs, label, sample_weight: (
                select_model_inputs(inputs, use_ihc, use_clinical_prior),
                label,
                sample_weight,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    raise ValueError("Unexpected dataset element structure")


def _make_dataset(
    config: ExperimentConfig,
    records: Sequence[PatientRecord],
    *,
    training: bool,
    seed: int,
    class_weights: Mapping[int, float] | None = None,
) -> Any:
    return make_tf_dataset(
        records=records,
        batch_size=(
            config.training.batch_size if training else config.training.eval_batch_size
        ),
        training=training,
        max_patches=_max_patches(
            config.data.max_train_patches if training else config.data.max_eval_patches
        ),
        seed=seed,
        feature_dim=config.data.feature_dim,
        feature_key=config.data.feature_key,
        class_weights=class_weights,
        include_ihc=(
            config.model.teacher_use_ihc or config.model.student_use_ihc
        ),
        include_clinical=config.model.use_clinical_prior,
    )


def fold_output_dir(
    config: ExperimentConfig, project_root: str | Path, fold: int
) -> Path:
    return project_path(project_root, config.output_dir) / config.name / f"fold_{fold}"


def teacher_run_dir(
    config: ExperimentConfig,
    project_root: str | Path,
    fold: int,
    teacher_index: int,
    seed: int,
) -> Path:
    return (
        fold_output_dir(config, project_root, fold)
        / "teachers"
        / f"teacher_{teacher_index}_seed_{seed}"
    )


def teacher_checkpoint_paths(
    config: ExperimentConfig, project_root: str | Path, fold: int
) -> list[Path]:
    return [
        teacher_run_dir(config, project_root, fold, index, seed) / "best.weights.h5"
        for index, seed in enumerate(config.training.teacher_seeds)
    ]


def student_run_dir(
    config: ExperimentConfig, project_root: str | Path, fold: int
) -> Path:
    return fold_output_dir(config, project_root, fold) / "student"


def student_checkpoint_path(
    config: ExperimentConfig, project_root: str | Path, fold: int
) -> Path:
    return student_run_dir(config, project_root, fold) / "best.weights.h5"


def _class_counts(records: Sequence[PatientRecord]) -> dict[str, int]:
    counts = Counter(record.class_name for record in records)
    return {name: counts.get(name, 0) for name in CLASS_NAMES}


def train_teacher_ensemble(
    config: ExperimentConfig,
    project_root: str | Path,
    fold: int,
    *,
    overwrite: bool = False,
    verbose: int = 2,
) -> list[Path]:
    """Train the same supervised fusion model under all configured seeds."""

    project_root = Path(project_root).resolve()
    (
        train_records,
        eval_records,
        test_records,
        splits,
    ) = load_fold_records(config, project_root, fold)
    class_weights = (
        balanced_class_weights(train_records)
        if config.training.class_weighting == "balanced"
        else None
    )
    data_provenance = _data_provenance(
        config,
        project_root,
        train_records + eval_records + test_records,
    )
    checkpoints = teacher_checkpoint_paths(config, project_root, fold)
    existing = [path for path in checkpoints if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Teacher checkpoint(s) already exist; pass --overwrite to retrain: "
            + ", ".join(str(path) for path in existing)
        )

    for teacher_index, (seed, checkpoint) in enumerate(
        zip(config.training.teacher_seeds, checkpoints)
    ):
        tf.keras.backend.clear_session()
        set_global_seed(seed)
        run_dir = checkpoint.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        run_token = uuid.uuid4().hex
        candidate_checkpoint = run_dir / f".candidate-{run_token}.weights.h5"
        candidate_history_csv = run_dir / f".candidate-{run_token}.history.csv"

        train_dataset = _adapt_dataset_inputs(
            _make_dataset(
                config,
                train_records,
                training=True,
                seed=seed,
                class_weights=class_weights,
            ),
            config.model.teacher_use_ihc,
            config.model.use_clinical_prior,
        )
        eval_dataset = _adapt_dataset_inputs(
            _make_dataset(
                config,
                eval_records,
                training=False,
                seed=seed,
            ),
            config.model.teacher_use_ihc,
            config.model.use_clinical_prior,
        )

        model = _build_teacher(config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.training.teacher_learning_rate,
                clipnorm=config.training.gradient_clip_norm,
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(candidate_checkpoint),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=True,
                verbose=1 if verbose else 0,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config.training.patience,
                restore_best_weights=True,
                verbose=1 if verbose else 0,
            ),
            tf.keras.callbacks.CSVLogger(str(candidate_history_csv)),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        history = model.fit(
            train_dataset,
            validation_data=eval_dataset,
            epochs=config.training.teacher_epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
        if not candidate_checkpoint.exists():
            raise RuntimeError(
                f"Teacher {teacher_index} did not produce a fresh checkpoint"
            )

        history_payload = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        eval_losses = history_payload.get("val_loss", [])
        if not eval_losses or not all(np.isfinite(value) for value in eval_losses):
            raise FloatingPointError(
                f"Teacher {teacher_index} has missing or non-finite eval loss"
            )
        for metric_name, values in history_payload.items():
            if not all(np.isfinite(value) for value in values):
                raise FloatingPointError(
                    f"Teacher {teacher_index} has non-finite {metric_name}"
                )
        best_epoch = int(np.argmin(eval_losses))
        os.replace(candidate_checkpoint, checkpoint)
        os.replace(candidate_history_csv, run_dir / "history.csv")
        write_json(history_payload, run_dir / "history.json")
        metadata = {
            "kind": "pathology_credit_teacher",
            "fold": fold,
            "teacher_index": teacher_index,
            "seed": seed,
            "architecture": _architecture(config, use_ihc=config.model.teacher_use_ihc),
            "recipe": _teacher_recipe(config),
            "data_provenance": data_provenance,
            "class_names": list(CLASS_NAMES),
            "ihc_policy": config.data.ihc_policy,
            "feature_key": config.data.feature_key,
            "class_weighting": config.training.class_weighting,
            "class_weights": (
                None
                if class_weights is None
                else {str(key): value for key, value in class_weights.items()}
            ),
            "train_case_ids": [record.case_id for record in train_records],
            "eval_case_ids": [record.case_id for record in eval_records],
            "test_case_ids": [record.case_id for record in test_records],
            "train_class_counts": _class_counts(train_records),
            "eval_class_counts": _class_counts(eval_records),
            "test_class_counts": _class_counts(test_records),
            "split_dataset_fingerprint": splits["dataset_fingerprint"],
            "best_epoch_zero_based": best_epoch,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "config": config.to_dict(),
        }
        write_json(metadata, run_dir / "metadata.json")

    return checkpoints


def _validate_checkpoint_metadata(
    metadata_path: Path,
    *,
    expected_kind: str,
    expected_fold: int,
    expected_architecture: Mapping[str, Any],
    expected_recipe: Mapping[str, Any],
    expected_data_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Checkpoint metadata not found: {metadata_path}"
        ) from exc
    if metadata.get("kind") != expected_kind:
        raise ValueError(f"Unexpected checkpoint kind in {metadata_path}")
    if metadata.get("fold") != expected_fold:
        raise ValueError(f"Checkpoint fold does not match fold {expected_fold}")
    if metadata.get("architecture") != dict(expected_architecture):
        raise ValueError(
            f"Checkpoint architecture does not match current config: {metadata_path}"
        )
    if metadata.get("recipe") != dict(expected_recipe):
        raise ValueError(
            f"Checkpoint training recipe does not match current config: {metadata_path}"
        )
    if metadata.get("data_provenance") != dict(expected_data_provenance):
        raise ValueError(
            f"Checkpoint data provenance does not match current artifacts: "
            f"{metadata_path}"
        )
    if metadata.get("class_names") != list(CLASS_NAMES):
        raise ValueError(f"Checkpoint class mapping mismatch: {metadata_path}")
    return metadata


def load_teacher_ensemble(
    config: ExperimentConfig, project_root: str | Path, fold: int
) -> tuple[list[tf.keras.Model], list[dict[str, Any]]]:
    """Rebuild, verify, and freeze every teacher checkpoint."""

    teachers: list[tf.keras.Model] = []
    provenance: list[dict[str, Any]] = []
    architecture = _architecture(config, use_ihc=config.model.teacher_use_ihc)
    records = load_experiment_records(config, project_root)
    data_provenance = _data_provenance(config, project_root, records)
    for teacher_index, (seed, checkpoint) in enumerate(
        zip(
            config.training.teacher_seeds,
            teacher_checkpoint_paths(config, project_root, fold),
        )
    ):
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing teacher {teacher_index} checkpoint: {checkpoint}"
            )
        metadata = _validate_checkpoint_metadata(
            checkpoint.parent / "metadata.json",
            expected_kind="pathology_credit_teacher",
            expected_fold=fold,
            expected_architecture=architecture,
            expected_recipe=_teacher_recipe(config),
            expected_data_provenance=data_provenance,
        )
        if metadata.get("ihc_policy") != config.data.ihc_policy:
            raise ValueError(f"Teacher IHC policy mismatch: {checkpoint}")
        if metadata.get("feature_key") != config.data.feature_key:
            raise ValueError(f"Teacher HDF5 feature key mismatch: {checkpoint}")
        if (
            metadata.get("teacher_index") != teacher_index
            or metadata.get("seed") != seed
        ):
            raise ValueError(
                f"Teacher index/seed mismatch in {checkpoint.parent / 'metadata.json'}"
            )
        observed_hash = file_sha256(checkpoint)
        if metadata.get("checkpoint_sha256") != observed_hash:
            raise ValueError(f"Teacher checkpoint hash mismatch: {checkpoint}")
        model = _build_teacher(config)
        model.load_weights(checkpoint)
        model.trainable = False
        teachers.append(model)
        provenance.append(
            {
                "teacher_index": teacher_index,
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": observed_hash,
                "split_dataset_fingerprint": metadata.get("split_dataset_fingerprint"),
                "train_case_ids": metadata.get("train_case_ids"),
                "eval_case_ids": metadata.get("eval_case_ids"),
                "test_case_ids": metadata.get("test_case_ids"),
            }
        )
    return teachers, provenance


def _regularization_loss(model: tf.keras.Model, dtype: tf.dtypes.DType) -> tf.Tensor:
    if not model.losses:
        return tf.zeros((), dtype=dtype)
    return tf.add_n([tf.cast(loss, dtype) for loss in model.losses])


def _build_student_batch_steps(
    student: tf.keras.Model,
    teachers: Sequence[tf.keras.Model],
    optimizer: tf.keras.optimizers.Optimizer,
    config: ExperimentConfig,
) -> tuple[StudentBatchStep, StudentBatchStep, tuple[str, ...]]:
    """Create the fixed-signature train/eval graphs once for one student run."""

    include_ihc = config.model.teacher_use_ihc or config.model.student_use_ihc
    include_clinical = config.model.use_clinical_prior
    input_keys = list(MODEL_INPUT_KEYS if include_ihc else IMAGE_INPUT_KEYS)
    if include_clinical:
        input_keys.extend(CLINICAL_INPUT_KEYS)
    input_signature: dict[str, tf.TensorSpec] = {
        "patches": tf.TensorSpec(
            shape=(None, None, config.data.feature_dim), dtype=tf.float32
        ),
        "patch_mask": tf.TensorSpec(shape=(None, None), dtype=tf.bool),
    }
    if include_ihc:
        input_signature.update(
            {
                "ihc": tf.TensorSpec(shape=(None, len(IHC_NAMES)), dtype=tf.float32),
                "ihc_mask": tf.TensorSpec(shape=(None, len(IHC_NAMES)), dtype=tf.bool),
            }
        )
    if include_clinical:
        input_signature.update(
            {
                "age": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                "age_mask": tf.TensorSpec(shape=(None, 1), dtype=tf.bool),
                "location": tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
                "location_mask": tf.TensorSpec(shape=(None, 1), dtype=tf.bool),
            }
        )
    label_signature = tf.TensorSpec(shape=(None,), dtype=tf.int32)
    teacher_models = tuple(teachers)
    distillation_temperature = config.training.distillation_temperature

    @tf.function(input_signature=(input_signature, label_signature))
    def train_batch(
        inputs: Mapping[str, tf.Tensor], labels: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        del labels
        teacher_inputs = select_model_inputs(
            inputs,
            config.model.teacher_use_ihc,
            config.model.use_clinical_prior,
        )
        student_inputs = select_model_inputs(
            inputs,
            config.model.student_use_ihc,
            config.model.use_clinical_prior,
        )
        teacher_logits = tf.stop_gradient(
            stack_teacher_logits(teacher_models, teacher_inputs, training=False)
        )
        with tf.GradientTape() as tape:
            student_output = student(student_inputs, training=True)
            distillation_loss = ced_loss(
                student_output,
                teacher_logits,
                distillation_temperature=distillation_temperature,
            )
            regularization_loss = _regularization_loss(student, distillation_loss.dtype)
            total_loss = distillation_loss + regularization_loss
        gradients = tape.gradient(total_loss, student.trainable_variables)
        gradient_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, student.trainable_variables)
            if gradient is not None
        ]
        if not gradient_pairs:
            raise RuntimeError("CREDIT student produced no trainable gradients")
        optimizer.apply_gradients(gradient_pairs)
        probabilities, _, _ = credit_probabilities(student_output)
        return total_loss, distillation_loss, regularization_loss, probabilities

    @tf.function(input_signature=(input_signature, label_signature))
    def eval_batch(
        inputs: Mapping[str, tf.Tensor], labels: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        del labels
        teacher_inputs = select_model_inputs(
            inputs,
            config.model.teacher_use_ihc,
            config.model.use_clinical_prior,
        )
        student_inputs = select_model_inputs(
            inputs,
            config.model.student_use_ihc,
            config.model.use_clinical_prior,
        )
        teacher_logits = tf.stop_gradient(
            stack_teacher_logits(teacher_models, teacher_inputs, training=False)
        )
        student_output = student(student_inputs, training=False)
        distillation_loss = ced_loss(
            student_output,
            teacher_logits,
            distillation_temperature=distillation_temperature,
        )
        regularization_loss = _regularization_loss(student, distillation_loss.dtype)
        total_loss = distillation_loss + regularization_loss
        probabilities, _, _ = credit_probabilities(student_output)
        return total_loss, distillation_loss, regularization_loss, probabilities

    return train_batch, eval_batch, tuple(input_keys)


def _run_student_epoch(
    dataset: Any,
    batch_step: StudentBatchStep,
    input_keys: Sequence[str],
) -> dict[str, float]:
    mean_total = tf.keras.metrics.Mean()
    mean_ced = tf.keras.metrics.Mean()
    mean_regularization = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_examples = 0

    for inputs, labels in dataset:
        step_inputs = {key: inputs[key] for key in input_keys}
        total_loss, distillation_loss, regularization_loss, probabilities = batch_step(
            step_inputs, labels
        )
        batch_size = int(tf.shape(labels)[0].numpy())
        num_examples += batch_size
        mean_total.update_state(total_loss, sample_weight=batch_size)
        mean_ced.update_state(distillation_loss, sample_weight=batch_size)
        mean_regularization.update_state(regularization_loss, sample_weight=batch_size)
        accuracy.update_state(labels, probabilities)

    if num_examples == 0:
        raise RuntimeError("Dataset yielded no patient batches")
    return {
        "loss": float(mean_total.result().numpy()),
        "ced_loss": float(mean_ced.result().numpy()),
        "regularization_loss": float(mean_regularization.result().numpy()),
        "accuracy": float(accuracy.result().numpy()),
        "num_examples": float(num_examples),
    }


def train_credit_student(
    config: ExperimentConfig,
    project_root: str | Path,
    fold: int,
    *,
    overwrite: bool = False,
    verbose: int = 1,
) -> Path:
    """Train one CREDIT student from the frozen five-member ensemble."""

    project_root = Path(project_root).resolve()
    checkpoint = student_checkpoint_path(config, project_root, fold)
    if checkpoint.exists() and not overwrite:
        raise FileExistsError(
            f"Student checkpoint already exists; pass --overwrite to retrain: {checkpoint}"
        )
    (
        train_records,
        eval_records,
        test_records,
        splits,
    ) = load_fold_records(config, project_root, fold)
    run_dir = checkpoint.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    run_token = uuid.uuid4().hex
    candidate_checkpoint = run_dir / f".candidate-{run_token}.weights.h5"
    candidate_history = run_dir / f".candidate-{run_token}.history.json"
    data_provenance = _data_provenance(
        config,
        project_root,
        train_records + eval_records + test_records,
    )

    tf.keras.backend.clear_session()
    student_seed = int((config.global_seed + 10_000 + fold) % (2**32))
    set_global_seed(student_seed)
    teachers, teacher_provenance = load_teacher_ensemble(config, project_root, fold)
    expected_train_ids = [record.case_id for record in train_records]
    expected_eval_ids = [record.case_id for record in eval_records]
    expected_test_ids = [record.case_id for record in test_records]
    for member in teacher_provenance:
        if member["split_dataset_fingerprint"] != splits["dataset_fingerprint"]:
            raise ValueError(
                f"Teacher {member['teacher_index']} was trained on a different cohort"
            )
        if (
            member["train_case_ids"] != expected_train_ids
            or member["eval_case_ids"] != expected_eval_ids
            or member["test_case_ids"] != expected_test_ids
        ):
            raise ValueError(
                f"Teacher {member['teacher_index']} three-way partition does "
                "not match the current fold"
            )
    # Teacher construction consumes random numbers even though their loaded
    # weights do not depend on them. Reset before student initialization.
    set_global_seed(student_seed)
    student = _build_student(config)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.training.student_learning_rate,
        clipnorm=config.training.gradient_clip_norm,
    )
    train_batch, eval_batch, step_input_keys = _build_student_batch_steps(
        student, teachers, optimizer, config
    )

    train_dataset = _make_dataset(
        config,
        train_records,
        training=True,
        seed=student_seed,
    )
    eval_dataset = _make_dataset(
        config,
        eval_records,
        training=False,
        seed=student_seed,
    )

    history: list[dict[str, Any]] = []
    best_eval_loss = float("inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    for epoch in range(config.training.student_epochs):
        train_metrics = _run_student_epoch(train_dataset, train_batch, step_input_keys)
        eval_metrics = _run_student_epoch(eval_dataset, eval_batch, step_input_keys)
        for partition_name, partition_metrics in (
            ("train", train_metrics),
            ("eval", eval_metrics),
        ):
            if not all(np.isfinite(value) for value in partition_metrics.values()):
                raise FloatingPointError(
                    f"Non-finite CREDIT {partition_name} metrics at epoch "
                    f"{epoch}: {partition_metrics}"
                )
        epoch_payload = {
            "epoch": epoch,
            "train": train_metrics,
            "eval": eval_metrics,
        }
        history.append(epoch_payload)
        write_json({"epochs": history}, candidate_history)

        eval_loss = eval_metrics["loss"]
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            student.save_weights(candidate_checkpoint)
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"epoch {epoch + 1}/{config.training.student_epochs} "
                f"loss={train_metrics['loss']:.5f} "
                f"acc={train_metrics['accuracy']:.4f} "
                f"eval_loss={eval_loss:.5f} "
                f"eval_acc={eval_metrics['accuracy']:.4f}"
            )
        if epochs_without_improvement >= config.training.patience:
            if verbose:
                print(
                    f"Early stopping after epoch {epoch + 1}; "
                    f"best epoch was {(best_epoch or 0) + 1}."
                )
            break

    if best_epoch is None or not candidate_checkpoint.exists():
        raise RuntimeError("CREDIT training did not produce a valid checkpoint")
    student.load_weights(candidate_checkpoint)
    os.replace(candidate_checkpoint, checkpoint)
    os.replace(candidate_history, run_dir / "history.json")
    metadata = {
        "kind": "pathology_credit_student",
        "fold": fold,
        "seed": student_seed,
        "architecture": _architecture(config, use_ihc=config.model.student_use_ihc),
        "recipe": _student_recipe(config),
        "data_provenance": data_provenance,
        "class_names": list(CLASS_NAMES),
        "ihc_policy": config.data.ihc_policy,
        "feature_key": config.data.feature_key,
        "execution_mode": STUDENT_EXECUTION_MODE,
        "distillation_temperature": config.training.distillation_temperature,
        "teacher_ensemble": teacher_provenance,
        "train_case_ids": [record.case_id for record in train_records],
        "eval_case_ids": [record.case_id for record in eval_records],
        "test_case_ids": [record.case_id for record in test_records],
        "train_class_counts": _class_counts(train_records),
        "eval_class_counts": _class_counts(eval_records),
        "test_class_counts": _class_counts(test_records),
        "split_dataset_fingerprint": splits["dataset_fingerprint"],
        "best_epoch_zero_based": best_epoch,
        "best_eval_loss": best_eval_loss,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config": config.to_dict(),
    }
    write_json(metadata, run_dir / "metadata.json")
    return checkpoint


def load_credit_student(
    config: ExperimentConfig, project_root: str | Path, fold: int
) -> tuple[tf.keras.Model, dict[str, Any]]:
    """Rebuild and verify the best CREDIT student for evaluation."""

    checkpoint = student_checkpoint_path(config, project_root, fold)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing CREDIT checkpoint: {checkpoint}")
    records = load_experiment_records(config, project_root)
    data_provenance = _data_provenance(config, project_root, records)
    metadata = _validate_checkpoint_metadata(
        checkpoint.parent / "metadata.json",
        expected_kind="pathology_credit_student",
        expected_fold=fold,
        expected_architecture=_architecture(
            config, use_ihc=config.model.student_use_ihc
        ),
        expected_recipe=_student_recipe(config),
        expected_data_provenance=data_provenance,
    )
    if metadata.get("execution_mode") != STUDENT_EXECUTION_MODE:
        raise ValueError(f"CREDIT execution mode mismatch: {checkpoint}")
    if metadata.get("ihc_policy") != config.data.ihc_policy:
        raise ValueError(f"CREDIT IHC policy mismatch: {checkpoint}")
    if metadata.get("feature_key") != config.data.feature_key:
        raise ValueError(f"CREDIT HDF5 feature key mismatch: {checkpoint}")
    observed_hash = file_sha256(checkpoint)
    if metadata.get("checkpoint_sha256") != observed_hash:
        raise ValueError(f"CREDIT checkpoint hash mismatch: {checkpoint}")
    student = _build_student(config)
    student.load_weights(checkpoint)
    student.trainable = False
    return student, metadata


__all__ = [
    "STUDENT_EXECUTION_MODE",
    "balanced_class_weights",
    "file_sha256",
    "fold_output_dir",
    "load_credit_student",
    "load_experiment_records",
    "load_fold_records",
    "load_teacher_ensemble",
    "select_model_inputs",
    "set_global_seed",
    "student_checkpoint_path",
    "student_run_dir",
    "teacher_checkpoint_paths",
    "teacher_run_dir",
    "train_credit_student",
    "train_teacher_ensemble",
    "write_json",
]
