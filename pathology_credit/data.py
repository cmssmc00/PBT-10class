"""Patient-level data utilities for the pathology CREDIT experiments.

The authoritative cohort description is ``data/patients_206.json``.  HDF5
feature files are deliberately not opened while records or splits are built;
they are opened only when a dataset iterator requests a patient's patch bag.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# These orders define the model targets and input columns.  Do not derive them
# from JSON/dictionary iteration order.
CLASS_NAMES = (
    "Glial tumors, low-grade",
    "Glial tumors, high-grade",
    "Glio-neuronal tumors",
    "Embryonal tumors, high-grade",
    "Ependymal tumors",
    "Uncertain histiogenesis tumors (these are defined by genetics or methylation classs)",
    "Choroid plexus tumor",
    "Pineal tumors",
    "Non-glial tumor",
    "Not tumor",
)
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}

IHC_NAMES = ("GFAP", "Synaptophysin", "INI1", "H3K27M", "ALK1")
IHC_TO_INDEX = {name: index for index, name in enumerate(IHC_NAMES)}

IHC_POLICIES = ("all", "mask_inferred")
LOCATION_NAMES = (
    "Cerebral hemispheric/lobar",
    "Cerebellum",
    "Ventricular/periventricular",
    "Brainstem/tectum",
    "Thalamus/deep gray",
    "Cerebellopontine angle",
    "Posterior fossa, NOS",
    "Pineal region",
    "Sellar/suprasellar/optic",
    "Spinal cord",
    "Unknown",
)
LOCATION_TO_INDEX = {name: index for index, name in enumerate(LOCATION_NAMES)}
UNKNOWN_LOCATION_INDEX = LOCATION_TO_INDEX["Unknown"]
SPLIT_SCHEMA_VERSION = 4
SPLIT_FRACTION_REFERENCE = "outer_non_test_pool"
SPLIT_PARTITIONS = ("train", "eval", "test")


@dataclass(frozen=True)
class PatientRecord:
    """Validated patient-level inputs and label.

    ``ihc`` and ``ihc_mask`` are already processed according to
    ``ihc_policy``. With the primary ``all`` policy, both observed and inferred
    POS/NEG values are used and every mask entry is true. ``ihc_inferred``
    preserves the original italic/provenance flags for auditing only.
    """

    case_id: str
    class_name: str
    class_index: int
    tumor_flag: int
    h5_paths: tuple[Path, ...]
    wsi_ids: tuple[str, ...]
    ihc: tuple[float, ...]
    ihc_mask: tuple[bool, ...]
    ihc_inferred: tuple[bool, ...]
    ihc_policy: str
    age: float = 0.0
    age_mask: bool = False
    location_index: int = UNKNOWN_LOCATION_INDEX
    location_mask: bool = False

    @property
    def label(self) -> int:
        """Integer target alias used by training code."""

        return self.class_index

    @property
    def classification(self) -> str:
        """Original ten-class target name alias."""

        return self.class_name


def _require_mapping(value: Any, field: str, case_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{case_id}: {field!r} must be a JSON object")
    return value


def _require_sequence(value: Any, field: str, case_id: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{case_id}: {field!r} must be a JSON list")
    return value


def _load_clinical_metadata(
    clinical_json: str | os.PathLike[str],
) -> dict[str, dict[str, Any]]:
    path = Path(clinical_json).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Clinical JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid clinical JSON at {path}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a top-level JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported clinical schema_version")
    if payload.get("age_unit") != "years":
        raise ValueError(f"{path}: age_unit must be 'years'")
    if payload.get("location_vocabulary") != list(LOCATION_NAMES):
        raise ValueError(f"{path}: location_vocabulary does not match the code")
    patients = payload.get("patients")
    if not isinstance(patients, list):
        raise TypeError(f"{path}: patients must be a JSON list")

    result: dict[str, dict[str, Any]] = {}
    for position, item in enumerate(patients):
        if not isinstance(item, Mapping):
            raise TypeError(f"{path}: clinical patient {position} must be an object")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{path}: clinical patient {position} has invalid case_id")
        case_id = case_id.strip()
        if case_id in result:
            raise ValueError(f"{path}: duplicate clinical case_id {case_id}")

        age_mask = item.get("age_mask")
        if not isinstance(age_mask, bool):
            raise TypeError(f"{case_id}: age_mask must be boolean")
        age_value = item.get("age")
        if age_mask:
            if isinstance(age_value, bool) or not isinstance(age_value, (int, float)):
                raise TypeError(f"{case_id}: available age must be numeric")
            age = float(age_value)
            if not math.isfinite(age) or not 0.0 <= age <= 120.0:
                raise ValueError(f"{case_id}: age must be finite and in [0, 120]")
        else:
            if age_value is not None:
                raise ValueError(f"{case_id}: unavailable age must be null")
            age = 0.0

        location_mask = item.get("location_mask")
        if not isinstance(location_mask, bool):
            raise TypeError(f"{case_id}: location_mask must be boolean")
        location_index = item.get("location_index")
        if (
            isinstance(location_index, bool)
            or not isinstance(location_index, int)
            or not 0 <= location_index < len(LOCATION_NAMES)
        ):
            raise ValueError(f"{case_id}: invalid location_index")
        expected_group = LOCATION_NAMES[location_index]
        if item.get("location_group") != expected_group:
            raise ValueError(f"{case_id}: location_group/index mismatch")
        if location_mask == (location_index == UNKNOWN_LOCATION_INDEX):
            raise ValueError(
                f"{case_id}: Unknown must be masked and known locations unmasked"
            )
        result[case_id] = {
            "age": age,
            "age_mask": age_mask,
            "location_index": location_index,
            "location_mask": location_mask,
        }
    return result


def load_patient_records(
    patients_json: str | os.PathLike[str],
    data_root: str | os.PathLike[str],
    ihc_policy: str = "all",
    exclude_tumor_flag_zero: bool = False,
    clinical_json: str | os.PathLike[str] | None = None,
) -> list[PatientRecord]:
    """Load and validate patient records without opening any HDF5 file.

    Args:
        patients_json: Path to the authoritative ``patients_206.json`` file.
        data_root: Directory against which ``relative_conch_paths`` resolve.
            For the supplied cohort this is the project's ``data`` directory.
        ihc_policy: ``all`` uses every supplied IHC value. ``mask_inferred``
            zeroes inferred values and marks them unavailable in ``ihc_mask``.
        exclude_tumor_flag_zero: Exclude records whose ``tumor_flag`` is zero.
        clinical_json: Optional age/location sidecar. When supplied, it must
            contain exactly the same case IDs as the patient JSON.

    Returns:
        Records in the same order as the authoritative patient JSON.
    """

    if ihc_policy not in IHC_POLICIES:
        raise ValueError(
            f"ihc_policy must be one of {IHC_POLICIES}, got {ihc_policy!r}"
        )

    json_path = Path(patients_json).expanduser()
    root = Path(data_root).expanduser().resolve()
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Patient JSON not found: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid patient JSON at {json_path}: {exc}") from exc

    if not isinstance(payload, list):
        raise TypeError(f"{json_path} must contain a top-level JSON list")

    clinical_by_id = (
        {} if clinical_json is None else _load_clinical_metadata(clinical_json)
    )
    records: list[PatientRecord] = []
    seen_case_ids: set[str] = set()
    h5_owner: dict[Path, str] = {}
    wsi_owner: dict[str, str] = {}

    for position, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise TypeError(f"Patient entry {position} must be a JSON object")

        case_id_value = item.get("case_id")
        if not isinstance(case_id_value, str) or not case_id_value.strip():
            raise ValueError(f"Patient entry {position} has an invalid case_id")
        case_id = case_id_value.strip()
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate case_id in patient JSON: {case_id}")
        seen_case_ids.add(case_id)

        class_name = item.get("classification")
        if class_name not in CLASS_TO_INDEX:
            raise ValueError(
                f"{case_id}: unknown classification {class_name!r}; "
                f"expected one of {CLASS_NAMES}"
            )

        tumor_flag = item.get("tumor_flag")
        if isinstance(tumor_flag, bool) or tumor_flag not in (0, 1):
            raise ValueError(f"{case_id}: tumor_flag must be integer 0 or 1")

        ihc_binary = _require_mapping(item.get("ihc_binary"), "ihc_binary", case_id)
        ihc_inferred_map = _require_mapping(
            item.get("ihc_inferred"), "ihc_inferred", case_id
        )
        if set(ihc_binary) != set(IHC_NAMES):
            raise ValueError(f"{case_id}: ihc_binary keys must be exactly {IHC_NAMES}")
        if set(ihc_inferred_map) != set(IHC_NAMES):
            raise ValueError(
                f"{case_id}: ihc_inferred keys must be exactly {IHC_NAMES}"
            )

        raw_ihc: list[float] = []
        inferred_flags: list[bool] = []
        for marker in IHC_NAMES:
            value = ihc_binary[marker]
            if isinstance(value, bool) or value not in (0, 1):
                raise ValueError(
                    f"{case_id}: ihc_binary[{marker!r}] must be integer 0 or 1"
                )
            inferred = ihc_inferred_map[marker]
            if not isinstance(inferred, bool):
                raise TypeError(f"{case_id}: ihc_inferred[{marker!r}] must be boolean")
            raw_ihc.append(float(value))
            inferred_flags.append(inferred)

        if ihc_policy == "all":
            ihc_mask = [True] * len(IHC_NAMES)
            ihc_values = raw_ihc
        else:
            ihc_mask = [not inferred for inferred in inferred_flags]
            ihc_values = [
                value if available else 0.0
                for value, available in zip(raw_ihc, ihc_mask)
            ]

        relative_paths = _require_sequence(
            item.get("relative_conch_paths"), "relative_conch_paths", case_id
        )
        wsi_ids_value = _require_sequence(item.get("wsi_ids"), "wsi_ids", case_id)
        if not relative_paths:
            raise ValueError(f"{case_id}: relative_conch_paths cannot be empty")
        if len(relative_paths) != len(wsi_ids_value):
            raise ValueError(
                f"{case_id}: relative_conch_paths and wsi_ids lengths differ"
            )
        wsi_count = item.get("wsi_count")
        if wsi_count != len(relative_paths):
            raise ValueError(
                f"{case_id}: wsi_count={wsi_count!r} does not match "
                f"{len(relative_paths)} feature paths"
            )

        h5_paths: list[Path] = []
        wsi_ids: list[str] = []
        for relative_path, wsi_id in zip(relative_paths, wsi_ids_value):
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"{case_id}: invalid relative_conch_paths entry")
            path_fragment = Path(relative_path)
            if path_fragment.is_absolute():
                raise ValueError(
                    f"{case_id}: CONCH path must be relative to data_root: "
                    f"{relative_path}"
                )
            h5_path = root / path_fragment
            if not h5_path.exists():
                raise FileNotFoundError(
                    f"{case_id}: CONCH feature file not found: {h5_path}"
                )
            canonical_h5_path = h5_path.resolve()
            prior_owner = h5_owner.get(canonical_h5_path)
            if prior_owner is not None:
                raise ValueError(
                    f"CONCH feature file is assigned to both {prior_owner} and "
                    f"{case_id}: {h5_path}"
                )
            h5_owner[canonical_h5_path] = case_id
            h5_paths.append(h5_path)

            if not isinstance(wsi_id, str) or not wsi_id:
                raise ValueError(f"{case_id}: invalid wsi_ids entry")
            prior_wsi_owner = wsi_owner.get(wsi_id)
            if prior_wsi_owner is not None:
                raise ValueError(
                    f"WSI ID {wsi_id!r} is assigned to both "
                    f"{prior_wsi_owner} and {case_id}"
                )
            wsi_owner[wsi_id] = case_id
            wsi_ids.append(wsi_id)

        if exclude_tumor_flag_zero and tumor_flag == 0:
            continue

        if clinical_json is None:
            clinical = {
                "age": 0.0,
                "age_mask": False,
                "location_index": UNKNOWN_LOCATION_INDEX,
                "location_mask": False,
            }
        else:
            try:
                clinical = clinical_by_id[case_id]
            except KeyError as exc:
                raise ValueError(f"{case_id}: missing from clinical JSON") from exc

        records.append(
            PatientRecord(
                case_id=case_id,
                class_name=class_name,
                class_index=CLASS_TO_INDEX[class_name],
                tumor_flag=int(tumor_flag),
                h5_paths=tuple(h5_paths),
                wsi_ids=tuple(wsi_ids),
                ihc=tuple(ihc_values),
                ihc_mask=tuple(ihc_mask),
                ihc_inferred=tuple(inferred_flags),
                ihc_policy=ihc_policy,
                age=clinical["age"],
                age_mask=clinical["age_mask"],
                location_index=clinical["location_index"],
                location_mask=clinical["location_mask"],
            )
        )

    if clinical_json is not None:
        extra_clinical_ids = sorted(set(clinical_by_id) - seen_case_ids)
        missing_clinical_ids = sorted(seen_case_ids - set(clinical_by_id))
        if extra_clinical_ids or missing_clinical_ids:
            raise ValueError(
                "Clinical/patient JSON case IDs differ: "
                f"missing={missing_clinical_ids}, extra={extra_clinical_ids}"
            )
    if not records:
        raise ValueError("No patient records remain after applying filters")
    return records


def select_records(
    records: Sequence[PatientRecord], case_ids: Sequence[str]
) -> list[PatientRecord]:
    """Select records in exactly the order given by ``case_ids``.

    Unknown or duplicate IDs are errors, which prevents silent split/data
    mismatches.
    """

    by_id: dict[str, PatientRecord] = {}
    for record in records:
        if record.case_id in by_id:
            raise ValueError(f"Duplicate record case_id: {record.case_id}")
        by_id[record.case_id] = record

    selected: list[PatientRecord] = []
    seen: set[str] = set()
    for case_id in case_ids:
        if case_id in seen:
            raise ValueError(f"Duplicate requested case_id: {case_id}")
        seen.add(case_id)
        try:
            selected.append(by_id[case_id])
        except KeyError as exc:
            raise KeyError(f"Unknown case_id in split: {case_id}") from exc
    return selected


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to read CONCH feature files. Install h5py in "
            "the training environment."
        ) from exc
    return h5py


def _import_tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build the training dataset. The paper "
            "environment uses tensorflow==2.14.0."
        ) from exc
    return tf


def _case_seed(seed: int, case_id: str) -> int:
    digest = hashlib.blake2b(f"{seed}\0{case_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _load_patient_bag(
    record: PatientRecord,
    feature_key: str,
    feature_dim: int,
    max_patches: int | None,
    training: bool,
    training_rng: np.random.Generator,
    seed: int,
) -> np.ndarray:
    """Open, concatenate and optionally subsample one patient's WSI bags."""

    h5py = _import_h5py()
    with ExitStack() as stack:
        datasets: list[Any] = []
        lengths: list[int] = []
        for h5_path in record.h5_paths:
            try:
                h5_file = stack.enter_context(h5py.File(h5_path, "r"))
            except OSError as exc:
                raise OSError(
                    f"{record.case_id}: could not open HDF5 file {h5_path}: {exc}"
                ) from exc
            if feature_key not in h5_file:
                raise KeyError(
                    f"{record.case_id}: HDF5 file has no {feature_key!r} dataset: "
                    f"{h5_path}"
                )
            features = h5_file[feature_key]
            if len(features.shape) != 2 or features.shape[1] != feature_dim:
                raise ValueError(
                    f"{record.case_id}: expected features shape [N, {feature_dim}] "
                    f"in {h5_path}, got {features.shape}"
                )
            datasets.append(features)
            lengths.append(int(features.shape[0]))

        total_patches = sum(lengths)
        if total_patches == 0:
            raise ValueError(f"{record.case_id}: patient patch bag is empty")

        if max_patches is None or total_patches <= max_patches:
            selected_global: np.ndarray | None = None
        else:
            rng = (
                training_rng
                if training
                else np.random.default_rng(_case_seed(seed, record.case_id))
            )
            # Sorted unique indices are accepted efficiently by h5py and retain
            # the original slide/patch order after random selection.
            selected_global = np.sort(
                rng.choice(total_patches, size=max_patches, replace=False)
            )

        arrays: list[np.ndarray] = []
        offset = 0
        for features, length in zip(datasets, lengths):
            if selected_global is None:
                array = np.asarray(features[...], dtype=np.float32)
            else:
                in_slide = (
                    selected_global[
                        (selected_global >= offset)
                        & (selected_global < offset + length)
                    ]
                    - offset
                )
                if in_slide.size == 0:
                    offset += length
                    continue
                array = np.asarray(features[in_slide], dtype=np.float32)
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"{record.case_id}: non-finite CONCH features in "
                    f"{features.file.filename}"
                )
            arrays.append(array)
            offset += length

    if not arrays:
        raise RuntimeError(f"{record.case_id}: patch sampling produced an empty bag")
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)


def _normalise_class_weights(
    class_weights: Mapping[Any, float] | Sequence[float] | None,
) -> dict[int, float] | None:
    if class_weights is None:
        return None

    resolved: dict[int, float] = {}
    if isinstance(class_weights, Mapping):
        for class_index, class_name in enumerate(CLASS_NAMES):
            candidates = (class_index, str(class_index), class_name)
            matches = [key for key in candidates if key in class_weights]
            if not matches:
                raise ValueError(
                    f"class_weights is missing class {class_index} ({class_name})"
                )
            resolved[class_index] = float(class_weights[matches[0]])
    elif isinstance(class_weights, Sequence) and not isinstance(
        class_weights, (str, bytes)
    ):
        if len(class_weights) != len(CLASS_NAMES):
            raise ValueError(f"class_weights must contain {len(CLASS_NAMES)} entries")
        resolved = {
            class_index: float(weight)
            for class_index, weight in enumerate(class_weights)
        }
    else:
        raise TypeError("class_weights must be a mapping, sequence, or None")

    for class_index, weight in resolved.items():
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"class_weights[{class_index}] must be finite and positive"
            )
    return resolved


def make_tf_dataset(
    records: Sequence[PatientRecord],
    batch_size: int,
    training: bool,
    max_patches: int | None,
    seed: int,
    feature_dim: int = 768,
    feature_key: str = "features",
    class_weights: Mapping[Any, float] | Sequence[float] | None = None,
    include_ihc: bool = True,
    include_clinical: bool = False,
) -> Any:
    """Create a patient-level ``tf.data.Dataset`` with padded patch bags.

    The unweighted element structure is ``(inputs, label)``. ``inputs`` always
    has ``patches`` and ``patch_mask``. With ``include_ihc=True`` it also has
    ``ihc`` and ``ihc_mask``; with
    ``include_clinical=True``, it also contains scalar ``age``, ``age_mask``,
    ``location`` and ``location_mask`` tensors. If
    ``class_weights`` is supplied, elements are ``(inputs, label,
    sample_weight)`` as expected by Keras.

    Training uses a persistent seeded RNG, so patient order and patch subsets
    change on each new iteration/epoch while remaining reproducible. Evaluation
    reads all patches when ``max_patches`` is ``None``; otherwise each patient
    receives a stable, seeded subset independent of iteration order.
    """

    if not records:
        raise ValueError("records cannot be empty")
    if isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(feature_dim, bool) or feature_dim <= 0:
        raise ValueError("feature_dim must be a positive integer")
    if not isinstance(feature_key, str) or not feature_key:
        raise ValueError("feature_key must be a non-empty string")
    if max_patches is not None and (isinstance(max_patches, bool) or max_patches <= 0):
        raise ValueError("max_patches must be a positive integer or None")
    if not isinstance(include_ihc, bool) or not isinstance(include_clinical, bool):
        raise TypeError("include_ihc and include_clinical must be boolean")
    if include_clinical and any(
        record.location_index < 0 or record.location_index >= len(LOCATION_NAMES)
        for record in records
    ):
        raise ValueError("records contains an invalid clinical location index")

    tf = _import_tensorflow()
    record_list = tuple(records)
    case_ids = [record.case_id for record in record_list]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("records contains duplicate case IDs")
    weights = _normalise_class_weights(class_weights)

    # Kept outside the generator factory so repeated Dataset iterations advance
    # the RNG instead of repeating the same training subset every epoch.
    training_rng = np.random.default_rng(seed)

    def generator() -> Any:
        order = np.arange(len(record_list))
        if training:
            training_rng.shuffle(order)
        for record_index in order:
            record = record_list[int(record_index)]
            patches = _load_patient_bag(
                record=record,
                feature_key=feature_key,
                feature_dim=feature_dim,
                max_patches=max_patches,
                training=training,
                training_rng=training_rng,
                seed=seed,
            )
            inputs = {
                "patches": patches,
                "patch_mask": np.ones(patches.shape[0], dtype=np.bool_),
            }
            if include_ihc:
                inputs.update(
                    {
                        "ihc": np.asarray(record.ihc, dtype=np.float32),
                        "ihc_mask": np.asarray(record.ihc_mask, dtype=np.bool_),
                    }
                )
            if include_clinical:
                inputs.update(
                    {
                        "age": np.asarray([record.age], dtype=np.float32),
                        "age_mask": np.asarray([record.age_mask], dtype=np.bool_),
                        "location": np.asarray(
                            [record.location_index], dtype=np.int32
                        ),
                        "location_mask": np.asarray(
                            [record.location_mask], dtype=np.bool_
                        ),
                    }
                )
            label = np.int32(record.class_index)
            if weights is None:
                yield inputs, label
            else:
                yield inputs, label, np.float32(weights[record.class_index])

    input_signature = {
        "patches": tf.TensorSpec(
            shape=(None, feature_dim), dtype=tf.float32, name="patches"
        ),
        "patch_mask": tf.TensorSpec(shape=(None,), dtype=tf.bool, name="patch_mask"),
    }
    if include_ihc:
        input_signature.update(
            {
                "ihc": tf.TensorSpec(
                    shape=(len(IHC_NAMES),), dtype=tf.float32, name="ihc"
                ),
                "ihc_mask": tf.TensorSpec(
                    shape=(len(IHC_NAMES),), dtype=tf.bool, name="ihc_mask"
                ),
            }
        )
    if include_clinical:
        input_signature.update(
            {
                "age": tf.TensorSpec(shape=(1,), dtype=tf.float32, name="age"),
                "age_mask": tf.TensorSpec(
                    shape=(1,), dtype=tf.bool, name="age_mask"
                ),
                "location": tf.TensorSpec(
                    shape=(1,), dtype=tf.int32, name="location"
                ),
                "location_mask": tf.TensorSpec(
                    shape=(1,), dtype=tf.bool, name="location_mask"
                ),
            }
        )
    label_signature = tf.TensorSpec(shape=(), dtype=tf.int32, name="label")
    if weights is None:
        output_signature: Any = (input_signature, label_signature)
    else:
        output_signature = (
            input_signature,
            label_signature,
            tf.TensorSpec(shape=(), dtype=tf.float32, name="sample_weight"),
        )

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=output_signature
    )
    padded_input_shapes = {
        "patches": (None, feature_dim),
        "patch_mask": (None,),
    }
    if include_ihc:
        padded_input_shapes.update(
            {
                "ihc": (len(IHC_NAMES),),
                "ihc_mask": (len(IHC_NAMES),),
            }
        )
    if include_clinical:
        padded_input_shapes.update(
            {
                "age": (1,),
                "age_mask": (1,),
                "location": (1,),
                "location_mask": (1,),
            }
        )
    input_padding_values = {
        "patches": tf.constant(0.0, dtype=tf.float32),
        "patch_mask": tf.constant(False, dtype=tf.bool),
    }
    if include_ihc:
        input_padding_values.update(
            {
                "ihc": tf.constant(0.0, dtype=tf.float32),
                "ihc_mask": tf.constant(False, dtype=tf.bool),
            }
        )
    if include_clinical:
        input_padding_values.update(
            {
                "age": tf.constant(0.0, dtype=tf.float32),
                "age_mask": tf.constant(False, dtype=tf.bool),
                "location": tf.constant(
                    UNKNOWN_LOCATION_INDEX, dtype=tf.int32
                ),
                "location_mask": tf.constant(False, dtype=tf.bool),
            }
        )
    if weights is None:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(padded_input_shapes, ()),
            padding_values=(
                input_padding_values,
                tf.constant(0, dtype=tf.int32),
            ),
            drop_remainder=False,
        )
    else:
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(padded_input_shapes, (), ()),
            padding_values=(
                input_padding_values,
                tf.constant(0, dtype=tf.int32),
                tf.constant(0.0, dtype=tf.float32),
            ),
            drop_remainder=False,
        )

    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    return dataset.prefetch(tf.data.AUTOTUNE)


def _dataset_fingerprint(records: Sequence[PatientRecord]) -> str:
    canonical_rows = sorted(
        f"{record.case_id}\t{record.class_name}" for record in records
    )
    return hashlib.sha256("\n".join(canonical_rows).encode("utf-8")).hexdigest()


def _class_counts(records: Sequence[PatientRecord]) -> dict[str, int]:
    counts = Counter(record.class_name for record in records)
    return {class_name: counts.get(class_name, 0) for class_name in CLASS_NAMES}


def create_patient_splits(
    records: Sequence[PatientRecord],
    n_splits: int = 5,
    eval_fraction: float = 0.125,
    seed: int = 42,
) -> dict[str, Any]:
    """Create deterministic patient-level train/eval/test folds.

    Outer test folds use ``StratifiedKFold``. Within each outer non-test pool,
    ``eval_fraction`` is measured against that pool and rounded to the nearest
    integer, then ``StratifiedShuffleSplit`` creates the train/eval split.
    Training partitions must contain every configured class. Eval and test
    partitions may have zero examples of a rare class: the 206-patient cohort
    has only four Choroid plexus tumor cases, so one of five outer test folds
    must necessarily omit that class.

    With five outer folds and ``eval_fraction=0.125``, the approximate full-cohort
    proportions are 70% train, 10% eval and 20% test. Records are sorted by case
    ID before splitting, making output stable if the source JSON order changes.
    """

    try:
        from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required to create patient splits. The paper "
            "environment uses scikit-learn==1.3.0."
        ) from exc

    def validate_fraction(name: str, value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite number")
        value = float(value)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1")
        return value

    if len(records) < 2:
        raise ValueError("At least two patient records are required")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    eval_fraction = validate_fraction("eval_fraction", eval_fraction)
    if eval_fraction >= 0.5:
        raise ValueError("eval_fraction must be strictly less than 0.5")

    ordered_records = sorted(records, key=lambda record: record.case_id)
    case_ids = [record.case_id for record in ordered_records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("records contains duplicate case IDs")
    labels = np.asarray(
        [record.class_index for record in ordered_records], dtype=np.int32
    )
    label_counts = Counter(labels.tolist())
    missing_classes = [
        class_name
        for class_index, class_name in enumerate(CLASS_NAMES)
        if class_index not in label_counts
    ]
    if missing_classes:
        raise ValueError(f"records is missing configured classes: {missing_classes}")
    too_small_for_training = {
        INDEX_TO_CLASS[label]: count
        for label, count in label_counts.items()
        if count < 3
    }
    if too_small_for_training:
        raise ValueError(
            "Each class needs at least three patients to preserve a class in "
            "the training partition after outer and inner splitting; too small: "
            f"{too_small_for_training}"
        )

    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    folds: list[dict[str, Any]] = []
    all_indices = np.arange(len(ordered_records), dtype=np.int32)

    def rounded_size(total: int, fraction: float) -> int:
        # Explicit half-up rounding avoids Python's banker-rounding ambiguity in
        # the persisted split contract.
        return math.floor(total * fraction + 0.5)

    def random_state(*parts: int) -> int:
        # scikit-learn's legacy RandomState accepts integers below 2**32.
        return int(sum(parts) % (2**32 - 1))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The least populated class in y has only .* members",
            category=UserWarning,
            module="sklearn.model_selection._split",
        )
        outer_folds = list(outer.split(all_indices, labels))

    for fold_index, (train_val_indices, test_indices) in enumerate(outer_folds):
        pool_size = len(train_val_indices)
        eval_size = rounded_size(pool_size, eval_fraction)
        final_train_size = pool_size - eval_size
        number_of_classes = len(CLASS_NAMES)
        if final_train_size < number_of_classes or eval_size < number_of_classes:
            raise ValueError(
                f"Fold {fold_index} train/eval sizes are too small for "
                f"{number_of_classes}-class stratification: "
                f"train={final_train_size}, eval={eval_size}"
            )

        pool_labels = labels[train_val_indices]
        pool_label_counts = Counter(pool_labels.tolist())
        insufficient_inner = {
            INDEX_TO_CLASS[label]: count
            for label, count in pool_label_counts.items()
            if count < 2
        }
        if insufficient_inner:
            raise ValueError(
                f"Fold {fold_index} needs at least two non-test patients per "
                "class for train/eval stratification; too small: "
                f"{insufficient_inner}"
            )

        inner = StratifiedShuffleSplit(
            n_splits=1,
            train_size=final_train_size,
            test_size=eval_size,
            random_state=random_state(seed, 10_000 * fold_index, 1),
        )
        train_relative, eval_relative = next(
            inner.split(train_val_indices, pool_labels)
        )
        train_indices = train_val_indices[train_relative]
        eval_indices = train_val_indices[eval_relative]
        if set(labels[train_indices].tolist()) != set(range(number_of_classes)):
            raise RuntimeError(
                f"Fold {fold_index} training partition does not contain every class"
            )

        train_records = [ordered_records[int(i)] for i in train_indices]
        eval_records = [ordered_records[int(i)] for i in eval_indices]
        test_records = [ordered_records[int(i)] for i in test_indices]

        train_ids = sorted(record.case_id for record in train_records)
        eval_ids = sorted(record.case_id for record in eval_records)
        test_ids = sorted(record.case_id for record in test_records)
        partition_sets = [
            set(train_ids),
            set(eval_ids),
            set(test_ids),
        ]
        if any(
            left & right
            for index, left in enumerate(partition_sets)
            for right in partition_sets[index + 1 :]
        ):
            raise RuntimeError(f"Patient leakage detected in fold {fold_index}")
        if set().union(*partition_sets) != set(case_ids):
            raise RuntimeError(f"Fold {fold_index} does not cover every patient")

        folds.append(
            {
                "fold": fold_index,
                "train": train_ids,
                "eval": eval_ids,
                "test": test_ids,
                "class_counts": {
                    "train": _class_counts(train_records),
                    "eval": _class_counts(eval_records),
                    "test": _class_counts(test_records),
                },
            }
        )

    test_membership = Counter(case_id for fold in folds for case_id in fold["test"])
    if set(test_membership) != set(case_ids) or any(
        count != 1 for count in test_membership.values()
    ):
        raise RuntimeError("Outer test folds do not partition patients exactly once")

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": int(seed),
        "n_splits": int(n_splits),
        "fraction_reference": SPLIT_FRACTION_REFERENCE,
        "eval_fraction": float(eval_fraction),
        "num_patients": len(ordered_records),
        "dataset_fingerprint": _dataset_fingerprint(ordered_records),
        "class_names": list(CLASS_NAMES),
        "ihc_names": list(IHC_NAMES),
        "folds": folds,
    }


def _validate_split_payload(splits: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "seed",
        "n_splits",
        "fraction_reference",
        "eval_fraction",
        "num_patients",
        "dataset_fingerprint",
        "class_names",
        "ihc_names",
        "folds",
    }
    missing = required - set(splits)
    if missing:
        raise ValueError(f"Split JSON is missing fields: {sorted(missing)}")
    unknown = set(splits) - required
    if unknown:
        raise ValueError(f"Split JSON has unknown fields: {sorted(unknown)}")
    if splits["schema_version"] != SPLIT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported split schema_version={splits['schema_version']!r}"
        )
    if splits["fraction_reference"] != SPLIT_FRACTION_REFERENCE:
        raise ValueError(
            f"Split JSON fraction_reference must be {SPLIT_FRACTION_REFERENCE!r}"
        )
    for field in ("seed", "n_splits", "num_patients"):
        value = splits[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Split JSON {field} must be an integer")
    if not 0 <= splits["seed"] < 2**32:
        raise ValueError("Split JSON seed must be in [0, 2**32)")
    if splits["n_splits"] < 2:
        raise ValueError("Split JSON n_splits must be at least 2")
    if splits["num_patients"] <= 0:
        raise ValueError("Split JSON num_patients must be positive")
    eval_fraction = splits["eval_fraction"]
    if isinstance(eval_fraction, bool) or not isinstance(eval_fraction, (int, float)):
        raise TypeError("Split JSON eval_fraction must be a finite number")
    eval_fraction = float(eval_fraction)
    if not math.isfinite(eval_fraction) or not 0.0 < eval_fraction < 0.5:
        raise ValueError("Split JSON eval_fraction must be between 0 and 0.5")
    fingerprint = splits["dataset_fingerprint"]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError(
            "Split JSON dataset_fingerprint must be a lowercase SHA-256 hex digest"
        )
    if splits["class_names"] != list(CLASS_NAMES):
        raise ValueError("Split JSON class_names do not match the code mapping")
    if splits["ihc_names"] != list(IHC_NAMES):
        raise ValueError("Split JSON ihc_names do not match the code mapping")
    folds = splits["folds"]
    if not isinstance(folds, list) or len(folds) != splits["n_splits"]:
        raise ValueError("Split JSON folds do not match n_splits")
    expected_fold_fields = {"fold", *SPLIT_PARTITIONS, "class_counts"}
    cohort_ids: set[str] | None = None
    test_membership: Counter[str] = Counter()
    cohort_class_counts: dict[str, int] | None = None
    for expected_fold, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            raise TypeError(f"Fold {expected_fold} must be a JSON object")
        if set(fold) != expected_fold_fields:
            missing_fold = expected_fold_fields - set(fold)
            unknown_fold = set(fold) - expected_fold_fields
            raise ValueError(
                f"Fold {expected_fold} fields do not match schema; "
                f"missing={sorted(missing_fold)}, unknown={sorted(unknown_fold)}"
            )
        fold_number = fold.get("fold")
        if isinstance(fold_number, bool) or not isinstance(fold_number, int):
            raise TypeError(f"Fold {expected_fold} index must be an integer")
        if fold_number != expected_fold:
            raise ValueError("Fold indexes must be contiguous and zero-based")
        partition_sets: dict[str, set[str]] = {}
        for partition in SPLIT_PARTITIONS:
            ids = fold.get(partition)
            if not isinstance(ids, list) or not all(
                isinstance(case_id, str) and bool(case_id) for case_id in ids
            ):
                raise ValueError(
                    f"Fold {expected_fold} {partition} must be a list of "
                    "non-empty case IDs"
                )
            if not ids:
                raise ValueError(f"Fold {expected_fold} {partition} cannot be empty")
            if len(ids) != len(set(ids)):
                raise ValueError(
                    f"Fold {expected_fold} {partition} contains duplicate IDs"
                )
            if ids != sorted(ids):
                raise ValueError(f"Fold {expected_fold} {partition} IDs must be sorted")
            partition_sets[partition] = set(ids)
        if any(
            left & right
            for index, left in enumerate(partition_sets.values())
            for right in list(partition_sets.values())[index + 1 :]
        ):
            raise ValueError(f"Fold {expected_fold} has patient leakage")
        fold_union = set().union(*partition_sets.values())
        if len(fold_union) != splits["num_patients"]:
            raise ValueError(
                f"Fold {expected_fold} does not contain num_patients unique IDs"
            )
        if cohort_ids is None:
            cohort_ids = fold_union
        elif fold_union != cohort_ids:
            raise ValueError(
                f"Fold {expected_fold} contains a different patient cohort"
            )
        test_membership.update(fold["test"])

        class_counts = fold["class_counts"]
        if not isinstance(class_counts, Mapping) or set(class_counts) != set(
            SPLIT_PARTITIONS
        ):
            raise ValueError(
                f"Fold {expected_fold} class_counts must contain exactly "
                f"{SPLIT_PARTITIONS}"
            )
        fold_class_totals = {class_name: 0 for class_name in CLASS_NAMES}
        for partition in SPLIT_PARTITIONS:
            counts = class_counts[partition]
            if not isinstance(counts, Mapping) or set(counts) != set(CLASS_NAMES):
                raise ValueError(
                    f"Fold {expected_fold} {partition} class_counts keys do "
                    "not match class_names"
                )
            for class_name in CLASS_NAMES:
                count = counts[class_name]
                if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                    raise ValueError(
                        f"Fold {expected_fold} {partition} class count for "
                        f"{class_name!r} must be a non-negative integer"
                    )
                if partition == "train" and count == 0:
                    raise ValueError(
                        f"Fold {expected_fold} training class count for "
                        f"{class_name!r} must be positive"
                    )
                fold_class_totals[class_name] += count
            if sum(counts.values()) != len(fold[partition]):
                raise ValueError(
                    f"Fold {expected_fold} {partition} class_counts do not "
                    "sum to the partition size"
                )
        if cohort_class_counts is None:
            cohort_class_counts = fold_class_totals
        elif fold_class_totals != cohort_class_counts:
            raise ValueError(
                f"Fold {expected_fold} class_counts imply a different cohort"
            )

    if cohort_ids is None or len(cohort_ids) != splits["num_patients"]:
        raise ValueError("Split JSON does not describe num_patients case IDs")
    if set(test_membership) != cohort_ids or any(
        count != 1 for count in test_membership.values()
    ):
        raise ValueError("Outer test folds must partition every patient exactly once")


def save_patient_splits(
    splits: Mapping[str, Any], output_path: str | os.PathLike[str]
) -> Path:
    """Validate and atomically write a stable, human-readable split JSON."""

    _validate_split_payload(splits)
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(splits, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def load_patient_splits(split_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a split JSON produced by ``save_patient_splits``."""

    path = Path(split_path).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Split JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid split JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Split JSON must contain a top-level object")
    _validate_split_payload(payload)
    return payload


def validate_splits_against_records(
    splits: Mapping[str, Any], records: Sequence[PatientRecord]
) -> None:
    """Verify that a split file belongs to exactly this patient cohort.

    Schema validation alone cannot detect a same-sized but different cohort.
    This check compares patient IDs, labels, and the stored cohort fingerprint,
    and confirms that outer test folds partition the current patients once.
    """

    _validate_split_payload(splits)
    if not records:
        raise ValueError("records cannot be empty")
    by_id = {record.case_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("records contains duplicate case IDs")
    expected_ids = set(by_id)
    if splits["num_patients"] != len(records):
        raise ValueError(
            "Split cohort size does not match current records: "
            f"{splits['num_patients']} != {len(records)}"
        )
    expected_fingerprint = _dataset_fingerprint(records)
    if splits["dataset_fingerprint"] != expected_fingerprint:
        raise ValueError(
            "Split dataset_fingerprint does not match the current patient IDs "
            "and labels; regenerate the split file"
        )

    test_membership: Counter[str] = Counter()
    for fold in splits["folds"]:
        fold_index = int(fold["fold"])
        fold_union = set().union(
            *(set(fold[partition]) for partition in SPLIT_PARTITIONS)
        )
        if fold_union != expected_ids:
            missing = sorted(expected_ids - fold_union)
            unknown = sorted(fold_union - expected_ids)
            raise ValueError(
                f"Fold {fold_index} does not match the current cohort; "
                f"missing={missing[:5]}, unknown={unknown[:5]}"
            )
        test_membership.update(fold["test"])

        stored_counts = fold["class_counts"]
        for partition in SPLIT_PARTITIONS:
            observed = _class_counts([by_id[case_id] for case_id in fold[partition]])
            if stored_counts[partition] != observed:
                raise ValueError(
                    f"Fold {fold_index} {partition} class_counts do not "
                    "match current labels"
                )

    if set(test_membership) != expected_ids or any(
        count != 1 for count in test_membership.values()
    ):
        raise ValueError(
            "Outer test folds must contain every current patient exactly once"
        )
