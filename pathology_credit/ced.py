"""Credal ensemble distillation operations from the CED paper.

Teacher logits use shape ``[batch, ensemble_members, classes]``.  The teacher
credal label is produced by applying distillation-temperature-scaled softmax independently
to each member, taking class-wise minima and maxima, and computing the unique
intersection probability and beta from equations (6), (9), and (10).

The loss follows equation (16): cross-entropy between teacher and student
intersection probabilities, plus the *sum* of squared class-wise interval
length errors and squared beta error, averaged over the batch and multiplied by
``distillation_temperature ** 2``.  Unlike the legacy CIFAR implementation in this
repository, the interval term is not divided by the number of classes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import tensorflow as tf

Tensor = tf.Tensor


def _positive_distillation_temperature(
    distillation_temperature: float | Tensor, dtype: tf.dtypes.DType
) -> Tensor:
    value = tf.cast(tf.convert_to_tensor(distillation_temperature), dtype)
    tf.debugging.assert_positive(
        value, message="distillation_temperature must be positive"
    )
    return value


def _split_credit_output(student_output: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    output = tf.convert_to_tensor(student_output)
    static_width = output.shape[-1]
    if static_width is not None:
        if static_width < 3 or (static_width - 1) % 2 != 0:
            raise ValueError(
                "student_output last dimension must have the CREDIT form 2C+1"
            )
        num_classes = (static_width - 1) // 2
        return tuple(tf.split(output, [num_classes, num_classes, 1], axis=-1))

    width = tf.shape(output)[-1]
    checks = [
        tf.debugging.assert_greater_equal(
            width, 3, message="student_output must contain at least one class"
        ),
        tf.debugging.assert_equal(
            tf.math.floormod(width - 1, 2),
            0,
            message="student_output last dimension must have the form 2C+1",
        ),
    ]
    with tf.control_dependencies(checks):
        num_classes = tf.math.floordiv(width - 1, 2)
        split_sizes = tf.stack([num_classes, num_classes, 1])
        parts = tf.split(output, split_sizes, axis=-1)
    return parts[0], parts[1], parts[2]


def stack_teacher_logits(
    teacher_models: Iterable[Callable[..., Tensor]],
    inputs,
    training: bool = False,
) -> Tensor:
    """Run frozen teacher members and stack outputs as ``[B, M, C]``.

    ``teacher_models`` must be a concrete non-empty Python iterable.  Passing
    ``training=False`` is important: dropout and other training-time behavior
    are disabled while constructing the distillation targets.
    """

    teachers = tuple(teacher_models)
    if not teachers:
        raise ValueError("teacher_models must contain at least one model")
    member_logits = [
        tf.convert_to_tensor(teacher(inputs, training=training)) for teacher in teachers
    ]
    return tf.stack(member_logits, axis=1)


def teacher_probabilities(
    teacher_logits: Tensor, distillation_temperature: float | Tensor = 1.0
) -> Tensor:
    """Convert teacher logits to distillation-temperature-scaled probabilities."""

    logits = tf.convert_to_tensor(teacher_logits)
    if not logits.dtype.is_floating:
        logits = tf.cast(logits, tf.float32)
    tf.debugging.assert_rank(
        logits, 3, message="teacher_logits must have shape [batch, members, classes]"
    )
    tf.debugging.assert_positive(
        tf.shape(logits)[1], message="teacher_logits must include an ensemble member"
    )
    tf.debugging.assert_greater_equal(
        tf.shape(logits)[2],
        2,
        message="teacher_logits must include at least two classes",
    )
    temperature_tensor = _positive_distillation_temperature(
        distillation_temperature, logits.dtype
    )
    return tf.nn.softmax(logits / temperature_tensor, axis=-1)


def teacher_probability_intervals(
    teacher_logits: Tensor, distillation_temperature: float | Tensor = 1.0
) -> tuple[Tensor, Tensor]:
    """Return class-wise teacher lower and upper probabilities."""

    probabilities = teacher_probabilities(teacher_logits, distillation_temperature)
    lower = tf.reduce_min(probabilities, axis=1)
    upper = tf.reduce_max(probabilities, axis=1)
    return lower, upper


def intersection_from_intervals(
    lower: Tensor, upper: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(intersection, interval_lengths, beta)`` for valid bounds.

    ``tf.math.divide_no_nan`` sets beta to zero for a degenerate zero-length
    credal set.  In that case the choice of beta is immaterial because the
    intersection and both bounds are identical.
    """

    lower = tf.convert_to_tensor(lower)
    upper = tf.cast(tf.convert_to_tensor(upper), lower.dtype)
    tf.debugging.assert_equal(
        tf.shape(lower), tf.shape(upper), message="lower and upper shapes must match"
    )
    tf.debugging.assert_greater_equal(
        upper,
        lower,
        message="upper probabilities must not be below lower probabilities",
    )

    interval_lengths = upper - lower
    numerator = 1.0 - tf.reduce_sum(lower, axis=-1, keepdims=True)
    denominator = tf.reduce_sum(interval_lengths, axis=-1, keepdims=True)
    beta = tf.math.divide_no_nan(numerator, denominator)
    beta = tf.clip_by_value(beta, 0.0, 1.0)
    intersection = lower + beta * interval_lengths
    return intersection, interval_lengths, beta


def teacher_credal_targets(
    teacher_logits: Tensor, distillation_temperature: float | Tensor = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    """Return teacher ``(intersection_probability, interval_lengths, beta)``."""

    lower, upper = teacher_probability_intervals(
        teacher_logits, distillation_temperature
    )
    return intersection_from_intervals(lower, upper)


def credit_probabilities(student_output: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Decode CREDIT output into ``(intersection, interval_lengths, beta)``.

    ``student_output`` is expected to be the post-activation output of
    :class:`pathology_credit.models.CreditOutput`: raw logits in the first C
    positions, followed by sigmoid interval lengths and sigmoid beta.
    """

    logits, interval_lengths, beta = _split_credit_output(student_output)
    if not logits.dtype.is_floating:
        logits = tf.cast(logits, tf.float32)
        interval_lengths = tf.cast(interval_lengths, tf.float32)
        beta = tf.cast(beta, tf.float32)
    intersection = tf.nn.softmax(logits, axis=-1)
    return intersection, interval_lengths, beta


def reconstruct_intervals_from_components(
    intersection: Tensor,
    interval_lengths: Tensor,
    beta: Tensor,
    clip: bool = True,
) -> tuple[Tensor, Tensor]:
    """Reconstruct lower and upper probabilities using equations (12)-(13)."""

    intersection = tf.convert_to_tensor(intersection)
    interval_lengths = tf.cast(
        tf.convert_to_tensor(interval_lengths), intersection.dtype
    )
    beta = tf.cast(tf.convert_to_tensor(beta), intersection.dtype)
    lower = intersection - beta * interval_lengths
    upper = intersection + (1.0 - beta) * interval_lengths
    if clip:
        lower = tf.maximum(lower, tf.cast(0.0, lower.dtype))
        upper = tf.minimum(upper, tf.cast(1.0, upper.dtype))
    return lower, upper


def reconstruct_intervals(
    student_output: Tensor,
    clip: bool = True,
) -> tuple[Tensor, Tensor]:
    """Decode a CREDIT output and return class-wise ``(lower, upper)`` bounds."""

    intersection, interval_lengths, beta = credit_probabilities(student_output)
    return reconstruct_intervals_from_components(
        intersection, interval_lengths, beta, clip=clip
    )


def ced_loss(
    student_output: Tensor,
    teacher_logits: Tensor,
    distillation_temperature: float | Tensor,
) -> Tensor:
    """Compute the paper-faithful scalar CED loss (equation 16).

    Args:
        student_output: CREDIT output with shape ``[B, 2C+1]``.
        teacher_logits: Stacked ensemble logits with shape ``[B, M, C]``.
        distillation_temperature: Positive temperature shared by teacher and
            student intersection probabilities.
    """

    student_logits, student_lengths, student_beta = _split_credit_output(student_output)
    if not student_logits.dtype.is_floating:
        student_logits = tf.cast(student_logits, tf.float32)
        student_lengths = tf.cast(student_lengths, tf.float32)
        student_beta = tf.cast(student_beta, tf.float32)
    teacher_logits = tf.stop_gradient(
        tf.cast(tf.convert_to_tensor(teacher_logits), student_logits.dtype)
    )
    tf.debugging.assert_rank(
        teacher_logits,
        3,
        message="teacher_logits must have shape [batch, members, classes]",
    )
    tf.debugging.assert_equal(
        tf.shape(student_logits)[0],
        tf.shape(teacher_logits)[0],
        message="student and teacher batch sizes must match",
    )
    tf.debugging.assert_equal(
        tf.shape(student_logits)[-1],
        tf.shape(teacher_logits)[-1],
        message="student and teacher class counts must match",
    )

    temperature_tensor = _positive_distillation_temperature(
        distillation_temperature, student_logits.dtype
    )
    teacher_intersection, teacher_lengths, teacher_beta = teacher_credal_targets(
        teacher_logits, temperature_tensor
    )

    # log_softmax is the numerically stable form of the paper's
    # -sum(p_teacher * log(p_student)) term.
    student_log_probabilities = tf.nn.log_softmax(
        student_logits / temperature_tensor, axis=-1
    )
    cross_entropy = -tf.reduce_sum(
        teacher_intersection * student_log_probabilities, axis=-1
    )
    interval_squared_error = tf.reduce_sum(
        tf.square(teacher_lengths - student_lengths), axis=-1
    )
    beta_squared_error = tf.squeeze(tf.square(teacher_beta - student_beta), axis=-1)
    per_example_loss = cross_entropy + interval_squared_error + beta_squared_error
    return tf.square(temperature_tensor) * tf.reduce_mean(per_example_loss)


__all__ = [
    "ced_loss",
    "credit_probabilities",
    "intersection_from_intervals",
    "reconstruct_intervals",
    "reconstruct_intervals_from_components",
    "stack_teacher_logits",
    "teacher_credal_targets",
    "teacher_probabilities",
    "teacher_probability_intervals",
]
