"""Evaluation utilities for the pathology CREDIT pipeline.

The CREDIT head has ``2 * C + 1`` outputs.  Its first ``C`` values describe
the intersection distribution :math:`p^*`, the next ``C`` values describe
probability-interval lengths, and the final value is the shared interpolation
coefficient beta.  The intervals are reconstructed as

``lower = p_star - beta * length`` and ``upper = lower + length``.

The main entry point, :func:`evaluate_credit`, deliberately returns a
JSON-serializable summary and NumPy per-sample arrays as two separate objects.
This keeps aggregate result files compact without losing the data needed for
case-level error and uncertainty analysis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any, Literal

import numpy as np

ArrayLike = Any
OutputMode = Literal["raw", "activated", "mixed"]


def _as_numpy(value: ArrayLike, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Convert NumPy, PyTorch, or TensorFlow-like values without importing them."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _validate_activated_unit_interval(
    values: np.ndarray,
    *,
    name: str,
    tolerance: float,
) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        observed_min = float(np.min(values))
        observed_max = float(np.max(values))
        raise ValueError(
            f"activated {name} must be in [0, 1]; observed range "
            f"[{observed_min:.6g}, {observed_max:.6g}]"
        )
    return np.clip(values, 0.0, 1.0)


def decode_credit_output(
    output: ArrayLike,
    *,
    num_classes: int | None = None,
    output_mode: OutputMode = "mixed",
    tolerance: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Decode a CREDIT head into probabilities and credal intervals.

    Parameters
    ----------
    output:
        Array of shape ``(N, 2*C+1)`` (or one row of that shape).
    num_classes:
        Optional explicit ``C``.  If omitted, it is inferred from the final
        dimension.
    output_mode:
        ``"raw"`` applies softmax to the first head and sigmoid to both the
        length and beta heads.  ``"activated"`` expects all three heads to be
        activated already.  ``"mixed"`` supports the original TensorFlow
        implementation in this repository: logits for the first head, but
        sigmoid-activated lengths and beta.
    Returns
    -------
    dict
        NumPy arrays named ``p_star``, ``interval_lengths``, ``beta``,
        ``lower_probs``, ``upper_probs``, ``effective_interval_lengths``, and
        ``interval_clipped``.
    """

    values = _as_numpy(output, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(
            f"CREDIT output must have shape (N, 2*C+1); got {values.shape}"
        )
    if values.shape[0] == 0:
        raise ValueError("CREDIT output must contain at least one sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("CREDIT output contains non-finite values")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")

    output_width = int(values.shape[1])
    if output_width < 5 or (output_width - 1) % 2 != 0:
        raise ValueError(
            f"CREDIT output width must equal 2*C+1 for C >= 2; got {output_width}"
        )
    inferred_classes = (output_width - 1) // 2
    if num_classes is None:
        num_classes = inferred_classes
    if num_classes < 2:
        raise ValueError(f"num_classes must be at least 2; got {num_classes}")
    if inferred_classes != num_classes:
        raise ValueError(
            f"output width {output_width} implies {inferred_classes} classes, "
            f"not num_classes={num_classes}"
        )
    if output_mode not in {"raw", "activated", "mixed"}:
        raise ValueError(
            "output_mode must be one of {'raw', 'activated', 'mixed'}; "
            f"got {output_mode!r}"
        )

    first_head = values[:, :num_classes]
    length_head = values[:, num_classes : 2 * num_classes]
    beta_head = values[:, 2 * num_classes :]

    if output_mode in {"raw", "mixed"}:
        p_star = _softmax(first_head)
    else:
        p_star = _validate_activated_unit_interval(
            first_head, name="p_star", tolerance=tolerance
        )
        row_sums = np.sum(p_star, axis=1, keepdims=True)
        if np.any(row_sums <= 0.0) or not np.allclose(
            row_sums, 1.0, atol=max(tolerance, 1e-8), rtol=1e-6
        ):
            bad_index = int(
                np.flatnonzero(
                    ~np.isclose(
                        row_sums[:, 0],
                        1.0,
                        atol=max(tolerance, 1e-8),
                        rtol=1e-6,
                    )
                )[0]
            )
            raise ValueError(
                "activated p_star rows must sum to one; "
                f"row {bad_index} sums to {row_sums[bad_index, 0]:.8g}"
            )
        # Remove harmless floating-point drift after validation.
        p_star = p_star / row_sums

    if output_mode == "raw":
        interval_lengths = _sigmoid(length_head)
        beta = _sigmoid(beta_head)
    else:
        interval_lengths = _validate_activated_unit_interval(
            length_head, name="interval_lengths", tolerance=tolerance
        )
        beta = _validate_activated_unit_interval(
            beta_head, name="beta", tolerance=tolerance
        )

    raw_lower = p_star - beta * interval_lengths
    raw_upper = raw_lower + interval_lengths
    lower_probs = np.clip(raw_lower, 0.0, 1.0)
    upper_probs = np.clip(raw_upper, 0.0, 1.0)

    # Clipping preserves p_star inside every interval, hence also preserves a
    # non-empty credal set because p_star itself lies on the probability simplex.
    interval_clipped = np.any(
        (np.abs(lower_probs - raw_lower) > tolerance)
        | (np.abs(upper_probs - raw_upper) > tolerance),
        axis=1,
    )

    return {
        "p_star": p_star,
        "interval_lengths": interval_lengths,
        "beta": beta[:, 0],
        "lower_probs": lower_probs,
        "upper_probs": upper_probs,
        "effective_interval_lengths": upper_probs - lower_probs,
        "interval_clipped": interval_clipped,
    }


def _entropy_bits(probabilities: np.ndarray) -> float:
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log2(probabilities[positive])))


def _adjust_probability_sum(
    probabilities: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Remove bisection/rounding residual while staying inside the bounds."""

    adjusted = probabilities.copy()
    residual = 1.0 - float(np.sum(adjusted))
    if abs(residual) <= tolerance:
        return adjusted

    if residual > 0.0:
        room = upper - adjusted
        indices = np.flatnonzero(room > tolerance)
        for index in indices:
            amount = min(residual, float(room[index]))
            adjusted[index] += amount
            residual -= amount
            if residual <= tolerance:
                break
    else:
        room = adjusted - lower
        indices = np.flatnonzero(room > tolerance)
        for index in indices:
            amount = min(-residual, float(room[index]))
            adjusted[index] -= amount
            residual += amount
            if residual >= -tolerance:
                break

    if abs(residual) > 10.0 * tolerance:
        raise RuntimeError(
            f"could not make bounded distribution sum to one; residual={residual:.3g}"
        )
    return adjusted


def _validate_credal_bounds(
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower_array = _as_numpy(lower, dtype=np.float64)
    upper_array = _as_numpy(upper, dtype=np.float64)
    if lower_array.ndim != 1 or upper_array.ndim != 1:
        raise ValueError("lower and upper bounds must be one-dimensional")
    if lower_array.shape != upper_array.shape or lower_array.size < 2:
        raise ValueError(
            "lower and upper bounds must have equal shape with at least two classes"
        )
    if not np.all(np.isfinite(lower_array)) or not np.all(np.isfinite(upper_array)):
        raise ValueError("credal bounds contain non-finite values")
    if np.any(lower_array < -tolerance) or np.any(upper_array > 1.0 + tolerance):
        raise ValueError("credal bounds must lie in [0, 1]")
    lower_array = np.clip(lower_array, 0.0, 1.0)
    upper_array = np.clip(upper_array, 0.0, 1.0)
    if np.any(lower_array > upper_array + tolerance):
        raise ValueError("a lower probability exceeds its upper probability")

    lower_sum = float(np.sum(lower_array))
    upper_sum = float(np.sum(upper_array))
    if lower_sum > 1.0 + tolerance or upper_sum < 1.0 - tolerance:
        raise ValueError(
            "empty credal set: bounds do not intersect the probability simplex "
            f"(sum(lower)={lower_sum:.8g}, sum(upper)={upper_sum:.8g})"
        )
    return lower_array, upper_array


def _maximum_entropy_distribution(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Water-fill the box-constrained simplex to maximize Shannon entropy."""

    low_level = float(np.min(lower) - 1.0)
    high_level = float(np.max(upper) + 1.0)
    for _ in range(100):
        level = (low_level + high_level) / 2.0
        candidate_sum = float(np.sum(np.clip(level, lower, upper)))
        if candidate_sum < 1.0:
            low_level = level
        else:
            high_level = level
    probabilities = np.clip((low_level + high_level) / 2.0, lower, upper)
    return _adjust_probability_sum(probabilities, lower, upper, tolerance=tolerance)


def _minimum_entropy_distribution(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float,
    max_vertex_candidates: int,
) -> np.ndarray:
    """Find the exact minimum-entropy vertex of a bounded simplex.

    A concave function reaches its minimum on a compact polytope at a vertex.
    A vertex of this box-constrained simplex has at most one coordinate away
    from a bound, so enumerating the free coordinate and the other coordinates'
    lower/upper choices is exact. CREDIT in this project has ten classes (5,120
    candidates per sample); the configurable guard prevents accidental
    exponential work for very high-dimensional uses.
    """

    num_classes = int(lower.size)
    candidate_count = num_classes * (2 ** (num_classes - 1))
    if candidate_count > max_vertex_candidates:
        raise RuntimeError(
            "exact minimum-entropy enumeration would require "
            f"{candidate_count:,} candidates; increase max_vertex_candidates "
            "explicitly if this is intentional"
        )

    best_probabilities: np.ndarray | None = None
    best_entropy = np.inf
    all_indices = np.arange(num_classes)

    for free_index in range(num_classes):
        fixed_indices = all_indices[all_indices != free_index]
        for choices in product((0, 1), repeat=num_classes - 1):
            probabilities = np.empty(num_classes, dtype=np.float64)
            choice_array = np.asarray(choices, dtype=bool)
            probabilities[fixed_indices] = np.where(
                choice_array, upper[fixed_indices], lower[fixed_indices]
            )
            free_probability = 1.0 - float(np.sum(probabilities[fixed_indices]))
            if (
                free_probability < lower[free_index] - tolerance
                or free_probability > upper[free_index] + tolerance
            ):
                continue
            probabilities[free_index] = np.clip(
                free_probability, lower[free_index], upper[free_index]
            )
            probabilities = _adjust_probability_sum(
                probabilities, lower, upper, tolerance=tolerance
            )
            entropy = _entropy_bits(probabilities)
            if entropy < best_entropy:
                best_entropy = entropy
                best_probabilities = probabilities.copy()

    if best_probabilities is None:
        raise RuntimeError("no feasible credal-set vertex was found")
    return best_probabilities


def credal_entropy_bounds(
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    tolerance: float = 1e-9,
    max_vertex_candidates: int = 1_000_000,
) -> dict[str, Any]:
    """Compute exact Shannon-entropy bounds for one interval distribution.

    Entropies use bits (base-2 logarithms).  Following the CREDIT evaluation
    convention, ``TU = H_upper``, ``AU = H_lower``, and ``EU = TU - AU``.
    The entropy-minimizing and entropy-maximizing distributions are returned to
    make the calculation auditable.
    """

    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and > 0")
    if max_vertex_candidates < 1:
        raise ValueError("max_vertex_candidates must be positive")
    lower_array, upper_array = _validate_credal_bounds(
        lower, upper, tolerance=tolerance
    )

    minimum_distribution = _minimum_entropy_distribution(
        lower_array,
        upper_array,
        tolerance=tolerance,
        max_vertex_candidates=max_vertex_candidates,
    )
    maximum_distribution = _maximum_entropy_distribution(
        lower_array, upper_array, tolerance=tolerance
    )
    entropy_lower = _entropy_bits(minimum_distribution)
    entropy_upper = _entropy_bits(maximum_distribution)
    # Do not expose tiny negative epistemic uncertainty due to round-off.
    epistemic = max(0.0, entropy_upper - entropy_lower)

    return {
        "entropy_lower_bits": entropy_lower,
        "entropy_upper_bits": entropy_upper,
        "AU": entropy_lower,
        "TU": entropy_upper,
        "EU": epistemic,
        "minimum_entropy_distribution": minimum_distribution,
        "maximum_entropy_distribution": maximum_distribution,
    }


def _prepare_hard_labels(
    y_true: ArrayLike, num_samples: int, num_classes: int
) -> np.ndarray:
    labels = _as_numpy(y_true)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.ndim != 1:
        raise ValueError(
            f"y_true must contain hard labels with shape (N,); got {labels.shape}"
        )
    if labels.shape[0] != num_samples:
        raise ValueError(
            f"y_true contains {labels.shape[0]} samples but output contains {num_samples}"
        )
    if not np.all(np.isfinite(labels)):
        raise ValueError("y_true contains non-finite labels")
    integer_labels = labels.astype(np.int64)
    if not np.array_equal(labels, integer_labels):
        raise ValueError("y_true must contain integer class indices")
    if np.any(integer_labels < 0) or np.any(integer_labels >= num_classes):
        raise ValueError(f"y_true labels must be in [0, {num_classes - 1}]")
    return integer_labels


def _confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def _binary_roc_auc(binary_labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Tie-aware Mann-Whitney implementation of binary ROC AUC."""

    positives = binary_labels.astype(bool)
    num_positive = int(np.sum(positives))
    num_negative = int(binary_labels.size - num_positive)
    if num_positive == 0 or num_negative == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Ranks are one-based; tied observations receive their average rank.
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end

    positive_rank_sum = float(np.sum(ranks[positives]))
    statistic = positive_rank_sum - num_positive * (num_positive + 1) / 2.0
    return statistic / (num_positive * num_negative)


def expected_calibration_error(
    y_true: ArrayLike,
    probabilities: ArrayLike,
    *,
    num_bins: int = 15,
) -> tuple[float, list[dict[str, Any]]]:
    """Return top-label ECE and JSON-serializable equal-width bin details."""

    probs = _as_numpy(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape (N, C) with N > 0 and C >= 2")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(probs < -1e-6) or np.any(probs > 1.0 + 1e-6):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("probability rows must sum to one")
    probs = np.clip(probs, 0.0, 1.0)
    probs /= np.sum(probs, axis=1, keepdims=True)
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    labels = _prepare_hard_labels(y_true, probs.shape[0], probs.shape[1])
    predicted = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    correct = predicted == labels
    bin_indices = np.minimum((confidence * num_bins).astype(np.int64), num_bins - 1)

    ece = 0.0
    bins: list[dict[str, Any]] = []
    for index in range(num_bins):
        members = bin_indices == index
        count = int(np.sum(members))
        lower = index / num_bins
        upper = (index + 1) / num_bins
        if count:
            mean_confidence = float(np.mean(confidence[members]))
            accuracy = float(np.mean(correct[members]))
            gap = abs(accuracy - mean_confidence)
            ece += (count / labels.size) * gap
        else:
            mean_confidence = None
            accuracy = None
            gap = None
        bins.append(
            {
                "bin_index": index,
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": None if gap is None else float(gap),
            }
        )
    return float(ece), bins


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.isfinite(values)
    if not np.any(finite):
        return None
    return float(np.mean(values[finite]))


def evaluate_credit(
    y_true: ArrayLike,
    output: ArrayLike | Mapping[str, ArrayLike],
    *,
    num_classes: int | None = None,
    output_mode: OutputMode = "mixed",
    class_names: Sequence[str] | None = None,
    ece_bins: int = 15,
    entropy_tolerance: float = 1e-9,
    max_vertex_candidates: int = 1_000_000,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate CREDIT classification, calibration, and uncertainty outputs.

    ``output`` may be a raw/activated ``(N, 2*C+1)`` array or the dictionary
    returned by :func:`decode_credit_output`.  The returned tuple is
    ``(summary, per_sample)``: ``summary`` contains only JSON-native values,
    while ``per_sample`` intentionally contains NumPy arrays.

    Entropy optimization is isolated per sample.  A failed sample receives NaN
    per-sample entropy values and a JSON error record, rather than invalidating
    the classification and calibration metrics for the whole fold.
    """

    if isinstance(output, Mapping):
        required = {"p_star", "lower_probs", "upper_probs"}
        missing = required.difference(output)
        if missing:
            raise ValueError(
                "decoded CREDIT output is missing keys: " + ", ".join(sorted(missing))
            )
        decoded = {key: _as_numpy(value) for key, value in output.items()}
        p_star = _as_numpy(decoded["p_star"], dtype=np.float64)
        lower_probs = _as_numpy(decoded["lower_probs"], dtype=np.float64)
        upper_probs = _as_numpy(decoded["upper_probs"], dtype=np.float64)
        if p_star.ndim != 2 or p_star.shape[1] < 2:
            raise ValueError("decoded p_star must have shape (N, C) with C >= 2")
        if p_star.shape[0] == 0:
            raise ValueError("decoded p_star must contain at least one sample")
        if lower_probs.shape != p_star.shape or upper_probs.shape != p_star.shape:
            raise ValueError(
                "decoded p_star, lower_probs, and upper_probs shapes differ"
            )
        if not np.all(np.isfinite(p_star)):
            raise ValueError("decoded p_star contains non-finite values")
        if np.any(p_star < -1e-6) or np.any(p_star > 1.0 + 1e-6):
            raise ValueError("decoded p_star must lie in [0, 1]")
        row_sums = np.sum(p_star, axis=1, keepdims=True)
        if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6):
            raise ValueError("decoded p_star rows must sum to one")
        p_star = np.clip(p_star, 0.0, 1.0)
        p_star /= np.sum(p_star, axis=1, keepdims=True)
        decoded["p_star"] = p_star
        decoded["lower_probs"] = lower_probs
        decoded["upper_probs"] = upper_probs
    else:
        decoded = decode_credit_output(
            output,
            num_classes=num_classes,
            output_mode=output_mode,
        )
        p_star = decoded["p_star"]
        lower_probs = decoded["lower_probs"]
        upper_probs = decoded["upper_probs"]

    inferred_classes = int(p_star.shape[1])
    if num_classes is not None and num_classes != inferred_classes:
        raise ValueError(
            f"decoded output has {inferred_classes} classes, not {num_classes}"
        )
    num_classes = inferred_classes
    num_samples = int(p_star.shape[0])
    labels = _prepare_hard_labels(y_true, num_samples, num_classes)

    if class_names is None:
        resolved_names = [str(index) for index in range(num_classes)]
    else:
        if len(class_names) != num_classes:
            raise ValueError(
                f"class_names has length {len(class_names)}; expected {num_classes}"
            )
        resolved_names = [str(name) for name in class_names]

    predicted = np.argmax(p_star, axis=1).astype(np.int64)
    correct = predicted == labels
    confidence = np.max(p_star, axis=1)
    confusion = _confusion_matrix(labels, predicted, num_classes)
    supports = np.sum(confusion, axis=1)
    predicted_counts = np.sum(confusion, axis=0)
    true_positives = np.diag(confusion)

    precisions = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_counts > 0,
    )
    recalls = np.divide(
        true_positives,
        supports,
        out=np.zeros(num_classes, dtype=np.float64),
        where=supports > 0,
    )
    f1_scores = np.divide(
        2.0 * precisions * recalls,
        precisions + recalls,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precisions + recalls) > 0,
    )

    class_aurocs: list[float | None] = []
    for class_index in range(num_classes):
        class_aurocs.append(
            _binary_roc_auc(
                (labels == class_index).astype(np.int8), p_star[:, class_index]
            )
        )
    all_aurocs_defined = all(value is not None for value in class_aurocs)
    if all_aurocs_defined:
        auroc_values = np.asarray(class_aurocs, dtype=np.float64)
        auroc_ovr_macro: float | None = float(np.mean(auroc_values))
        auroc_ovr_weighted: float | None = float(
            np.average(auroc_values, weights=supports)
        )
    else:
        auroc_ovr_macro = None
        auroc_ovr_weighted = None

    per_class: list[dict[str, Any]] = []
    for class_index in range(num_classes):
        per_class.append(
            {
                "class_index": class_index,
                "class_name": resolved_names[class_index],
                "precision": float(precisions[class_index]),
                "recall": float(recalls[class_index]),
                "f1": float(f1_scores[class_index]),
                "support": int(supports[class_index]),
                "auroc_ovr": class_aurocs[class_index],
            }
        )

    epsilon = 1e-12
    true_probabilities = p_star[np.arange(num_samples), labels]
    per_sample_nll = -np.log(np.clip(true_probabilities, epsilon, 1.0))
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    # Multiclass Brier score: sum of classwise squared errors per sample.
    per_sample_brier = np.sum((p_star - one_hot) ** 2, axis=1)
    ece, calibration_bins = expected_calibration_error(
        labels, p_star, num_bins=ece_bins
    )

    entropy_lower = np.full(num_samples, np.nan, dtype=np.float64)
    entropy_upper = np.full(num_samples, np.nan, dtype=np.float64)
    entropy_minimizers = np.full_like(p_star, np.nan)
    entropy_maximizers = np.full_like(p_star, np.nan)
    entropy_failures: list[dict[str, Any]] = []

    for sample_index in range(num_samples):
        try:
            bounds = credal_entropy_bounds(
                lower_probs[sample_index],
                upper_probs[sample_index],
                tolerance=entropy_tolerance,
                max_vertex_candidates=max_vertex_candidates,
            )
            entropy_lower[sample_index] = bounds["entropy_lower_bits"]
            entropy_upper[sample_index] = bounds["entropy_upper_bits"]
            entropy_minimizers[sample_index] = bounds["minimum_entropy_distribution"]
            entropy_maximizers[sample_index] = bounds["maximum_entropy_distribution"]
        except (ValueError, RuntimeError, FloatingPointError) as error:
            entropy_failures.append({"sample_index": sample_index, "error": str(error)})

    total_uncertainty = entropy_upper
    aleatoric_uncertainty = entropy_lower
    epistemic_uncertainty = np.maximum(0.0, entropy_upper - entropy_lower)
    entropy_valid = np.isfinite(entropy_lower) & np.isfinite(entropy_upper)

    present_classes = supports > 0
    balanced_accuracy = float(np.mean(recalls[present_classes]))
    summary: dict[str, Any] = {
        "num_samples": num_samples,
        "num_classes": num_classes,
        "accuracy": float(np.mean(correct)),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "weighted_f1": float(np.average(f1_scores, weights=supports)),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "auroc_ovr_macro": auroc_ovr_macro,
        "auroc_ovr_weighted": auroc_ovr_weighted,
        "auroc_ovr_defined": all_aurocs_defined,
        "nll": float(np.mean(per_sample_nll)),
        "brier": float(np.mean(per_sample_brier)),
        "brier_definition": "mean sum of classwise squared errors",
        "ece": ece,
        "ece_num_bins": int(ece_bins),
        "calibration_bins": calibration_bins,
        "uncertainty": {
            "entropy_units": "bits",
            "TU_definition": "maximum entropy over the credal set",
            "AU_definition": "minimum entropy over the credal set",
            "EU_definition": "TU - AU",
            "mean_TU": _finite_mean(total_uncertainty),
            "mean_AU": _finite_mean(aleatoric_uncertainty),
            "mean_EU": _finite_mean(epistemic_uncertainty),
            "mean_effective_interval_width": _finite_mean(
                (upper_probs - lower_probs).reshape(-1)
            ),
            "num_entropy_valid": int(np.sum(entropy_valid)),
            "num_entropy_failed": int(num_samples - np.sum(entropy_valid)),
            "entropy_failures": entropy_failures,
        },
    }

    per_sample: dict[str, np.ndarray] = {
        "y_true": labels,
        "y_pred": predicted,
        "correct": correct,
        "confidence": confidence,
        "p_star": p_star,
        "lower_probs": lower_probs,
        "upper_probs": upper_probs,
        "effective_interval_lengths": upper_probs - lower_probs,
        "nll": per_sample_nll,
        "brier": per_sample_brier,
        "entropy_lower_bits": entropy_lower,
        "entropy_upper_bits": entropy_upper,
        "TU": total_uncertainty,
        "AU": aleatoric_uncertainty,
        "EU": epistemic_uncertainty,
        "entropy_valid": entropy_valid,
        "minimum_entropy_distribution": entropy_minimizers,
        "maximum_entropy_distribution": entropy_maximizers,
    }
    for optional_key in ("interval_lengths", "beta", "interval_clipped"):
        if optional_key in decoded:
            per_sample[optional_key] = _as_numpy(decoded[optional_key])

    return summary, per_sample


__all__ = [
    "credal_entropy_bounds",
    "decode_credit_output",
    "evaluate_credit",
    "expected_calibration_error",
]
