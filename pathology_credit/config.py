"""Configuration loading for the pathology CREDIT pipeline.

The original repository keeps experiment settings directly in the training
scripts.  The pathology adaptation uses a small YAML configuration so every
teacher and student can be reproduced from the same settings.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Fixed code-provenance marker for the memory-stable CREDIT training path.  This
# is intentionally not configurable: checkpoints produced by a different
# execution path must not be mixed in one five-fold experiment.
STUDENT_EXECUTION_MODE = "tf_function_fixed_signature_v1"


@dataclass
class DataConfig:
    patients_json: str = "data/patients_206.json"
    clinical_json: str | None = None
    data_root: str = "data"
    feature_key: str = "features"
    feature_dim: int = 768
    ihc_policy: str = "all"
    exclude_tumor_flag_zero: bool = False
    split_file: str = "splits/pathology_5fold.json"
    n_splits: int = 5
    # Measured against the outer non-test pool. With five outer folds, 0.125 of
    # the remaining 80% is about 10% of the full cohort.
    eval_fraction: float = 0.125
    split_seed: int = 20260830
    max_train_patches: int = 2048
    max_eval_patches: int = 0


@dataclass
class ModelConfig:
    projection_dim: int = 256
    attention_dim: int = 128
    ihc_hidden_dim: int = 32
    fusion_dim: int = 128
    patch_dropout: float = 0.1
    fusion_dropout: float = 0.2
    l2: float = 1.0e-4
    ihc_mask_as_feature: bool = False
    teacher_use_ihc: bool = True
    student_use_ihc: bool = True
    use_hierarchy: bool = False
    use_clinical_prior: bool = False
    age_hidden_dim: int = 8
    location_embedding_dim: int = 8
    clinical_hidden_dim: int = 16
    num_locations: int = 11
    age_scale_years: float = 25.0
    prior_strength_init: float = 0.1


@dataclass
class TrainingConfig:
    teacher_seeds: list[int] = field(default_factory=lambda: [101, 102, 103, 104, 105])
    teacher_epochs: int = 100
    student_epochs: int = 100
    batch_size: int = 2
    eval_batch_size: int = 1
    teacher_learning_rate: float = 3.0e-4
    student_learning_rate: float = 3.0e-4
    distillation_temperature: float = 2.5
    patience: int = 15
    class_weighting: str = "balanced"
    gradient_clip_norm: float = 5.0


@dataclass
class ExperimentConfig:
    name: str = "conch_ihc_credit_206_10class"
    output_dir: str = "outputs/pathology_credit"
    global_seed: int = 20260830
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Old checkpoints and completed OOF summaries predate the optional
        # clinical/hierarchy fields. Omit disabled defaults so those artifacts
        # remain exactly loadable and auditable with their original configs.
        if self.data.clinical_json is None:
            payload["data"].pop("clinical_json")
        if not self.model.use_hierarchy and not self.model.use_clinical_prior:
            for key in (
                "use_hierarchy",
                "use_clinical_prior",
                "age_hidden_dim",
                "location_embedding_dim",
                "clinical_hidden_dim",
                "num_locations",
                "age_scale_years",
                "prior_strength_init",
            ):
                payload["model"].pop(key)
        return payload


def _update_dataclass(instance: Any, values: dict[str, Any], section: str) -> Any:
    known = set(instance.__dataclass_fields__)
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown configuration keys in {section}: {unknown}")
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


def validate_config(config: ExperimentConfig) -> None:
    def positive_integer(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    def finite_number(name: str, value: Any, *, minimum: float | None = None) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")

    def seed_integer(name: str, value: Any) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 2**32
        ):
            raise ValueError(f"{name} must be an integer in [0, 2**32)")

    positive_integer("data.feature_dim", config.data.feature_dim)
    positive_integer("data.n_splits", config.data.n_splits)
    if config.data.n_splits < 2:
        raise ValueError("data.n_splits must be at least 2")
    finite_number("data.eval_fraction", config.data.eval_fraction)
    if not 0.0 < config.data.eval_fraction < 0.5:
        raise ValueError("data.eval_fraction must be between 0 and 0.5")
    if config.data.ihc_policy not in {"all", "mask_inferred"}:
        raise ValueError("data.ihc_policy must be 'all' or 'mask_inferred'")
    if config.training.class_weighting not in {"balanced", "none"}:
        raise ValueError("training.class_weighting must be 'balanced' or 'none'")
    if (
        not isinstance(config.training.teacher_seeds, list)
        or len(config.training.teacher_seeds) != 5
    ):
        raise ValueError("CREDIT requires exactly five independently trained teachers")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
        for seed in config.training.teacher_seeds
    ):
        raise ValueError("training.teacher_seeds must contain integers in [0, 2**32)")
    if len(set(config.training.teacher_seeds)) != len(config.training.teacher_seeds):
        raise ValueError("training.teacher_seeds must be unique")
    finite_number(
        "training.distillation_temperature",
        config.training.distillation_temperature,
    )
    if config.training.distillation_temperature <= 0:
        raise ValueError("training.distillation_temperature must be positive")
    for name, value in {
        "data.max_train_patches": config.data.max_train_patches,
        "data.max_eval_patches": config.data.max_eval_patches,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name, value in {
        "model.projection_dim": config.model.projection_dim,
        "model.attention_dim": config.model.attention_dim,
        "model.ihc_hidden_dim": config.model.ihc_hidden_dim,
        "model.fusion_dim": config.model.fusion_dim,
        "model.age_hidden_dim": config.model.age_hidden_dim,
        "model.location_embedding_dim": config.model.location_embedding_dim,
        "model.clinical_hidden_dim": config.model.clinical_hidden_dim,
        "model.num_locations": config.model.num_locations,
        "training.teacher_epochs": config.training.teacher_epochs,
        "training.student_epochs": config.training.student_epochs,
        "training.batch_size": config.training.batch_size,
        "training.eval_batch_size": config.training.eval_batch_size,
        "training.patience": config.training.patience,
    }.items():
        positive_integer(name, value)
    for name, value in {
        "training.teacher_learning_rate": config.training.teacher_learning_rate,
        "training.student_learning_rate": config.training.student_learning_rate,
        "training.gradient_clip_norm": config.training.gradient_clip_norm,
    }.items():
        finite_number(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    finite_number("model.l2", config.model.l2)
    if config.model.l2 < 0:
        raise ValueError("model.l2 must be non-negative")
    finite_number("model.age_scale_years", config.model.age_scale_years)
    if config.model.age_scale_years <= 0:
        raise ValueError("model.age_scale_years must be positive")
    finite_number("model.prior_strength_init", config.model.prior_strength_init)
    if not 0.0 < config.model.prior_strength_init < 1.0:
        raise ValueError("model.prior_strength_init must be strictly between 0 and 1")
    if not isinstance(config.data.feature_key, str) or not config.data.feature_key:
        raise ValueError("data.feature_key must be a non-empty string")
    for name, value in {
        "model.patch_dropout": config.model.patch_dropout,
        "model.fusion_dropout": config.model.fusion_dropout,
    }.items():
        finite_number(name, value)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    for name, value in {
        "data.exclude_tumor_flag_zero": config.data.exclude_tumor_flag_zero,
        "model.ihc_mask_as_feature": config.model.ihc_mask_as_feature,
        "model.teacher_use_ihc": config.model.teacher_use_ihc,
        "model.student_use_ihc": config.model.student_use_ihc,
        "model.use_hierarchy": config.model.use_hierarchy,
        "model.use_clinical_prior": config.model.use_clinical_prior,
    }.items():
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    if config.model.use_hierarchy and len(config.training.teacher_seeds) != 5:
        raise ValueError("Hierarchical CREDIT requires the five-teacher ensemble")
    if config.model.use_hierarchy and config.model.teacher_use_ihc:
        raise ValueError("The hierarchical experiment is WSI/clinical only; disable teacher IHC")
    if config.model.use_hierarchy and config.model.student_use_ihc:
        raise ValueError("The hierarchical experiment is WSI/clinical only; disable student IHC")
    if config.model.use_clinical_prior and config.model.num_locations != 11:
        raise ValueError("model.num_locations must be 11 for the clinical sidecar")
    if config.model.use_clinical_prior and config.model.teacher_use_ihc:
        raise ValueError("The clinical-prior experiment must disable teacher IHC")
    if config.model.use_clinical_prior and config.model.student_use_ihc:
        raise ValueError("The clinical-prior experiment must disable student IHC")
    if config.model.use_clinical_prior and not config.data.clinical_json:
        raise ValueError("data.clinical_json is required when clinical prior is enabled")
    if config.data.clinical_json is not None and (
        not isinstance(config.data.clinical_json, str) or not config.data.clinical_json
    ):
        raise ValueError("data.clinical_json must be null or a non-empty path")
    for name, value in {
        "global_seed": config.global_seed,
        "data.split_seed": config.data.split_seed,
    }.items():
        seed_integer(name, value)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise TypeError("The configuration root must be a mapping")

    config = ExperimentConfig()
    root_values = {
        k: v for k, v in payload.items() if k not in {"data", "model", "training"}
    }
    _update_dataclass(config, root_values, "root")
    if "data" in payload:
        _update_dataclass(config.data, payload["data"] or {}, "data")
    if "model" in payload:
        _update_dataclass(config.model, payload["model"] or {}, "model")
    if "training" in payload:
        _update_dataclass(config.training, payload["training"] or {}, "training")
    validate_config(config)
    return config


def project_path(project_root: str | Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else Path(project_root) / path
