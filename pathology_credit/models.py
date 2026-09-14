"""TensorFlow models for the multimodal pathology CREDIT pipeline.

The models consume padded, variable-length bags of pre-computed CONCH patch
features.  ``patch_mask`` keeps padding out of the attention normalization.  If
IHC is enabled, the five binary IHC values are masked before the MLP.  The mask
can optionally be concatenated for a missingness ablation, but the primary
configuration disables that shortcut because inferred-value patterns may be
class-correlated.

Input dictionaries use these keys:

``patches``
    Float tensor with shape ``[batch, patches, feature_dim]``.
``patch_mask``
    Boolean tensor with shape ``[batch, patches]``; ``True`` marks a real patch.
``ihc``
    Float tensor with shape ``[batch, 5]``.
``ihc_mask``
    Boolean tensor with shape ``[batch, 5]``; ``True`` means the IHC value may
    be used under the selected data policy.

The optional hierarchical experiment replaces the flat ten-class head with
the pathologist's sequence ``C10 vs C1-9`` -> ``C9 vs C1-8`` -> ``C1-C8``.
Age and normalized tumor location can enter each node through a small clinical
prior branch. The three clinical contributions are bounded by independently
learned gates and the final ten probabilities are composed exactly from the
three conditional decisions.
"""

from __future__ import annotations

import math

import tensorflow as tf

Tensor = tf.Tensor


def _kernel_regularizer(l2_value: float) -> tf.keras.regularizers.Regularizer | None:
    if (
        isinstance(l2_value, bool)
        or not isinstance(l2_value, (int, float))
        or not math.isfinite(float(l2_value))
        or l2_value < 0.0
    ):
        raise ValueError("l2 must be finite and non-negative")
    if l2_value == 0.0:
        return None
    return tf.keras.regularizers.l2(l2_value)


def _validate_model_arguments(
    feature_dim: int,
    num_classes: int,
    projection_dim: int,
    attention_dim: int,
    ihc_hidden_dim: int,
    fusion_dim: int,
    patch_dropout: float,
    fusion_dropout: float,
    l2: float,
) -> None:
    dimensions = {
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "projection_dim": projection_dim,
        "attention_dim": attention_dim,
        "ihc_hidden_dim": ihc_hidden_dim,
        "fusion_dim": fusion_dim,
    }
    for name, value in dimensions.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in {
        "patch_dropout": patch_dropout,
        "fusion_dropout": fusion_dropout,
    }.items():
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")


@tf.keras.utils.register_keras_serializable(package="pathology_credit")
class GatedAttentionMIL(tf.keras.layers.Layer):
    """Lightweight gated-attention multiple-instance pooling.

    This is the gated attention mechanism from attention MIL: a tanh branch
    and a sigmoid branch jointly score each projected patch.  Masked positions
    receive exactly zero normalized attention.  An entirely masked bag maps to
    a zero vector; the data loader should still reject such bags because they do
    not contain usable pathology evidence.
    """

    def __init__(
        self,
        projection_dim: int = 256,
        attention_dim: int = 128,
        dropout_rate: float = 0.1,
        l2: float = 1.0e-4,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if projection_dim <= 0 or attention_dim <= 0:
            raise ValueError("projection_dim and attention_dim must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")

        self.projection_dim = int(projection_dim)
        self.attention_dim = int(attention_dim)
        self.dropout_rate = float(dropout_rate)
        self.l2_value = float(l2)
        regularizer = _kernel_regularizer(self.l2_value)

        self.projection = tf.keras.layers.Dense(
            self.projection_dim,
            kernel_regularizer=regularizer,
            name="patch_projection",
        )
        self.projection_norm = tf.keras.layers.LayerNormalization(
            epsilon=1.0e-6, name="patch_projection_norm"
        )
        self.projection_activation = tf.keras.layers.Activation(
            "gelu", name="patch_projection_gelu"
        )
        self.projection_dropout = tf.keras.layers.Dropout(
            self.dropout_rate, name="patch_projection_dropout"
        )
        self.attention_tanh = tf.keras.layers.Dense(
            self.attention_dim,
            activation="tanh",
            kernel_regularizer=regularizer,
            name="attention_tanh",
        )
        self.attention_sigmoid = tf.keras.layers.Dense(
            self.attention_dim,
            activation="sigmoid",
            kernel_regularizer=regularizer,
            name="attention_sigmoid",
        )
        self.attention_score = tf.keras.layers.Dense(
            1,
            use_bias=False,
            kernel_regularizer=regularizer,
            name="attention_score",
        )

    def call(
        self,
        patches: Tensor,
        patch_mask: Tensor | None = None,
        training: bool | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        projected = self.projection(patches)
        projected = self.projection_norm(projected)
        projected = self.projection_activation(projected)
        projected = self.projection_dropout(projected, training=training)

        gated = self.attention_tanh(projected) * self.attention_sigmoid(projected)
        scores = tf.squeeze(self.attention_score(gated), axis=-1)

        if patch_mask is None:
            mask = tf.ones_like(scores, dtype=tf.bool)
        else:
            mask = tf.cast(patch_mask, tf.bool)

        # The second normalization makes masked attention exactly zero and also
        # gives a finite zero-vector result for an accidentally empty bag.
        very_negative = tf.cast(-1.0e9, scores.dtype)
        masked_scores = tf.where(mask, scores, very_negative)
        attention = tf.nn.softmax(masked_scores, axis=1)
        attention = tf.where(mask, attention, tf.zeros_like(attention))
        attention = tf.math.divide_no_nan(
            attention, tf.reduce_sum(attention, axis=1, keepdims=True)
        )

        pooled = tf.einsum("bn,bnd->bd", attention, projected)
        if return_attention:
            return pooled, attention
        return pooled

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "projection_dim": self.projection_dim,
                "attention_dim": self.attention_dim,
                "dropout_rate": self.dropout_rate,
                "l2": self.l2_value,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="pathology_credit")
class CreditOutput(tf.keras.layers.Layer):
    """The paper's ``2C+1`` CREDIT output transformation.

    The first ``C`` elements remain raw logits.  The next ``C`` elements are
    transformed to interval lengths with a sigmoid, and the final scalar is the
    sigmoid-transformed beta used to place each interval around the predicted
    intersection probability.
    """

    def __init__(self, num_classes: int, l2: float = 1.0e-4, **kwargs) -> None:
        super().__init__(**kwargs)
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.num_classes = int(num_classes)
        self.l2_value = float(l2)
        self.output_projection = tf.keras.layers.Dense(
            2 * self.num_classes + 1,
            kernel_regularizer=_kernel_regularizer(self.l2_value),
            name="raw_credit_parameters",
        )

    def call(self, inputs: Tensor) -> Tensor:
        raw_output = self.output_projection(inputs)
        logits, raw_lengths, raw_beta = tf.split(
            raw_output, [self.num_classes, self.num_classes, 1], axis=-1
        )
        return tf.concat(
            [logits, tf.nn.sigmoid(raw_lengths), tf.nn.sigmoid(raw_beta)],
            axis=-1,
        )

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "l2": self.l2_value})
        return config


@tf.keras.utils.register_keras_serializable(package="pathology_credit")
class PriorStrength(tf.keras.layers.Layer):
    """Add a bounded, learnable amount of clinical-prior evidence to logits."""

    def __init__(self, initial_strength: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0.0 < initial_strength < 1.0:
            raise ValueError("initial_strength must be strictly between 0 and 1")
        self.initial_strength = float(initial_strength)

    def build(self, input_shape: object) -> None:
        raw_initial = math.log(self.initial_strength / (1.0 - self.initial_strength))
        self.raw_strength = self.add_weight(
            name="raw_strength",
            shape=(),
            initializer=tf.keras.initializers.Constant(raw_initial),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tuple[Tensor, Tensor] | list[Tensor]) -> Tensor:
        wsi_logits, prior_logits = inputs
        strength = tf.nn.sigmoid(self.raw_strength)
        return wsi_logits + tf.cast(strength, wsi_logits.dtype) * prior_logits

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({"initial_strength": self.initial_strength})
        return config


@tf.keras.utils.register_keras_serializable(package="pathology_credit")
class HierarchicalLogits(tf.keras.layers.Layer):
    """Compose C1-C10 log-probabilities from the three conditional heads."""

    def __init__(self, epsilon: float = 1.0e-7, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, inputs: tuple[Tensor, Tensor, Tensor] | list[Tensor]) -> Tensor:
        c10_logit, c9_logit, c1_c8_logits = inputs
        p_c10 = tf.nn.sigmoid(c10_logit)
        p_c9_given_tumor = tf.nn.sigmoid(c9_logit)
        p_c1_c8_given_glial_path = tf.nn.softmax(c1_c8_logits, axis=-1)

        p_not_c10 = 1.0 - p_c10
        p_c1_c8 = (
            p_not_c10
            * (1.0 - p_c9_given_tumor)
            * p_c1_c8_given_glial_path
        )
        p_c9 = p_not_c10 * p_c9_given_tumor
        probabilities = tf.concat([p_c1_c8, p_c9, p_c10], axis=-1)
        probabilities = tf.clip_by_value(
            probabilities, self.epsilon, 1.0
        )
        probabilities = tf.math.divide_no_nan(
            probabilities,
            tf.reduce_sum(probabilities, axis=-1, keepdims=True),
        )
        return tf.math.log(probabilities)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


@tf.keras.utils.register_keras_serializable(package="pathology_credit")
class HierarchicalCreditOutput(tf.keras.layers.Layer):
    """Attach CREDIT interval parameters to hierarchical class logits."""

    def __init__(self, num_classes: int, l2: float = 1.0e-4, **kwargs) -> None:
        super().__init__(**kwargs)
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        self.num_classes = int(num_classes)
        self.l2_value = float(l2)
        self.interval_projection = tf.keras.layers.Dense(
            self.num_classes + 1,
            kernel_regularizer=_kernel_regularizer(self.l2_value),
            name="raw_interval_parameters",
        )

    def call(self, inputs: tuple[Tensor, Tensor] | list[Tensor]) -> Tensor:
        embedding, class_logits = inputs
        credit_features = tf.concat([embedding, class_logits], axis=-1)
        raw_interval = self.interval_projection(credit_features)
        raw_lengths, raw_beta = tf.split(
            raw_interval, [self.num_classes, 1], axis=-1
        )
        return tf.concat(
            [class_logits, tf.nn.sigmoid(raw_lengths), tf.nn.sigmoid(raw_beta)],
            axis=-1,
        )

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "l2": self.l2_value})
        return config


def _make_inputs(
    feature_dim: int, use_ihc: bool, use_clinical_prior: bool
) -> dict[str, Tensor]:
    inputs: dict[str, Tensor] = {
        "patches": tf.keras.Input(
            shape=(None, feature_dim), dtype=tf.float32, name="patches"
        ),
        "patch_mask": tf.keras.Input(shape=(None,), dtype=tf.bool, name="patch_mask"),
    }
    if use_ihc:
        inputs.update(
            {
                "ihc": tf.keras.Input(shape=(5,), dtype=tf.float32, name="ihc"),
                "ihc_mask": tf.keras.Input(shape=(5,), dtype=tf.bool, name="ihc_mask"),
            }
        )
    if use_clinical_prior:
        inputs.update(
            {
                "age": tf.keras.Input(shape=(1,), dtype=tf.float32, name="age"),
                "age_mask": tf.keras.Input(
                    shape=(1,), dtype=tf.bool, name="age_mask"
                ),
                "location": tf.keras.Input(
                    shape=(1,), dtype=tf.int32, name="location"
                ),
                "location_mask": tf.keras.Input(
                    shape=(1,), dtype=tf.bool, name="location_mask"
                ),
            }
        )
    return inputs


def _fusion_embedding(
    inputs: dict[str, Tensor],
    *,
    projection_dim: int,
    attention_dim: int,
    ihc_hidden_dim: int,
    fusion_dim: int,
    patch_dropout: float,
    fusion_dropout: float,
    l2: float,
    use_ihc: bool,
    ihc_mask_as_feature: bool,
) -> Tensor:
    regularizer = _kernel_regularizer(l2)
    mil_layer = GatedAttentionMIL(
        projection_dim=projection_dim,
        attention_dim=attention_dim,
        dropout_rate=patch_dropout,
        l2=l2,
        name="gated_attention_mil",
    )
    pathology_embedding, _ = mil_layer(
        inputs["patches"], inputs["patch_mask"], return_attention=True
    )

    branches = [pathology_embedding]
    if use_ihc:
        ihc_mask = tf.keras.layers.Rescaling(scale=1.0, name="ihc_mask_float")(
            inputs["ihc_mask"]
        )
        masked_ihc = tf.keras.layers.Multiply(name="masked_ihc")(
            [inputs["ihc"], ihc_mask]
        )
        if ihc_mask_as_feature:
            ihc_features = tf.keras.layers.Concatenate(name="ihc_with_mask")(
                [masked_ihc, ihc_mask]
            )
        else:
            ihc_features = masked_ihc
        ihc_features = tf.keras.layers.Dense(
            ihc_hidden_dim,
            activation="gelu",
            kernel_regularizer=regularizer,
            name="ihc_dense_1",
        )(ihc_features)
        ihc_features = tf.keras.layers.LayerNormalization(
            epsilon=1.0e-6, name="ihc_norm"
        )(ihc_features)
        ihc_features = tf.keras.layers.Dense(
            ihc_hidden_dim,
            activation="gelu",
            kernel_regularizer=regularizer,
            name="ihc_dense_2",
        )(ihc_features)
        branches.append(ihc_features)

    if len(branches) == 1:
        fused = branches[0]
    else:
        fused = tf.keras.layers.Concatenate(name="modality_fusion")(branches)

    fused = tf.keras.layers.Dense(
        fusion_dim,
        kernel_regularizer=regularizer,
        name="fusion_dense",
    )(fused)
    fused = tf.keras.layers.LayerNormalization(epsilon=1.0e-6, name="fusion_norm")(
        fused
    )
    fused = tf.keras.layers.Activation("gelu", name="fusion_gelu")(fused)
    fused = tf.keras.layers.Dropout(fusion_dropout, name="fusion_dropout")(fused)
    return fused


def _clinical_embedding(
    inputs: dict[str, Tensor],
    *,
    age_hidden_dim: int,
    location_embedding_dim: int,
    clinical_hidden_dim: int,
    num_locations: int,
    age_scale_years: float,
    l2: float,
) -> Tensor:
    """Encode age/location without using any cohort-level label statistics."""

    regularizer = _kernel_regularizer(l2)
    age_mask = tf.keras.layers.Rescaling(scale=1.0, name="age_mask_float")(
        inputs["age_mask"]
    )
    masked_age = tf.keras.layers.Multiply(name="masked_age")(
        [inputs["age"], age_mask]
    )
    scaled_age = tf.keras.layers.Rescaling(
        scale=1.0 / age_scale_years, name="age_in_25_year_units"
    )(masked_age)
    age_features = tf.keras.layers.Dense(
        age_hidden_dim,
        activation="gelu",
        kernel_regularizer=regularizer,
        name="age_dense",
    )(scaled_age)

    location_mask = tf.keras.layers.Rescaling(
        scale=1.0, name="location_mask_float"
    )(inputs["location_mask"])
    location_features = tf.keras.layers.Embedding(
        input_dim=num_locations,
        output_dim=location_embedding_dim,
        embeddings_regularizer=regularizer,
        name="location_embedding",
    )(inputs["location"])
    location_features = tf.keras.layers.Flatten(name="location_embedding_flat")(
        location_features
    )
    location_features = tf.keras.layers.Multiply(name="masked_location_embedding")(
        [location_features, location_mask]
    )

    clinical = tf.keras.layers.Concatenate(name="clinical_features")(
        [age_features, location_features, age_mask, location_mask]
    )
    clinical = tf.keras.layers.Dense(
        clinical_hidden_dim,
        activation="gelu",
        kernel_regularizer=regularizer,
        name="clinical_dense",
    )(clinical)
    return tf.keras.layers.LayerNormalization(
        epsilon=1.0e-6, name="clinical_norm"
    )(clinical)


def _hierarchical_logits(
    embedding: Tensor,
    inputs: dict[str, Tensor],
    *,
    use_clinical_prior: bool,
    age_hidden_dim: int,
    location_embedding_dim: int,
    clinical_hidden_dim: int,
    num_locations: int,
    age_scale_years: float,
    prior_strength_init: float,
    l2: float,
) -> Tensor:
    """Build the shared WSI hierarchy and optional node-specific prior heads."""

    regularizer = _kernel_regularizer(l2)
    wsi_logits = (
        tf.keras.layers.Dense(
            1, kernel_regularizer=regularizer, name="wsi_c10_logit"
        )(embedding),
        tf.keras.layers.Dense(
            1, kernel_regularizer=regularizer, name="wsi_c9_logit"
        )(embedding),
        tf.keras.layers.Dense(
            8, kernel_regularizer=regularizer, name="wsi_c1_c8_logits"
        )(embedding),
    )

    if use_clinical_prior:
        clinical = _clinical_embedding(
            inputs,
            age_hidden_dim=age_hidden_dim,
            location_embedding_dim=location_embedding_dim,
            clinical_hidden_dim=clinical_hidden_dim,
            num_locations=num_locations,
            age_scale_years=age_scale_years,
            l2=l2,
        )
        prior_logits = (
            tf.keras.layers.Dense(
                1, kernel_regularizer=regularizer, name="prior_c10_logit"
            )(clinical),
            tf.keras.layers.Dense(
                1, kernel_regularizer=regularizer, name="prior_c9_logit"
            )(clinical),
            tf.keras.layers.Dense(
                8, kernel_regularizer=regularizer, name="prior_c1_c8_logits"
            )(clinical),
        )
        node_names = ("c10_prior_strength", "c9_prior_strength", "c1_c8_prior_strength")
        combined_logits = tuple(
            PriorStrength(initial_strength=prior_strength_init, name=name)(
                [wsi_logit, prior_logit]
            )
            for name, wsi_logit, prior_logit in zip(
                node_names, wsi_logits, prior_logits
            )
        )
    else:
        combined_logits = wsi_logits

    return HierarchicalLogits(name="hierarchical_log_probs")(combined_logits)


def _flat_prior_logits(
    embedding: Tensor,
    inputs: dict[str, Tensor],
    *,
    num_classes: int,
    age_hidden_dim: int,
    location_embedding_dim: int,
    clinical_hidden_dim: int,
    num_locations: int,
    age_scale_years: float,
    prior_strength_init: float,
    l2: float,
) -> Tensor:
    """Fuse WSI and clinical evidence directly in one flat class-logit head."""

    regularizer = _kernel_regularizer(l2)
    wsi_logits = tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=regularizer,
        name="wsi_logits",
    )(embedding)
    clinical = _clinical_embedding(
        inputs,
        age_hidden_dim=age_hidden_dim,
        location_embedding_dim=location_embedding_dim,
        clinical_hidden_dim=clinical_hidden_dim,
        num_locations=num_locations,
        age_scale_years=age_scale_years,
        l2=l2,
    )
    prior_logits = tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=regularizer,
        name="prior_logits",
    )(clinical)
    return PriorStrength(
        initial_strength=prior_strength_init,
        name="flat_prior_strength",
    )([wsi_logits, prior_logits])


def _validate_hierarchical_arguments(
    *,
    num_classes: int,
    use_ihc: bool,
    use_hierarchy: bool,
    use_clinical_prior: bool,
    age_hidden_dim: int,
    location_embedding_dim: int,
    clinical_hidden_dim: int,
    num_locations: int,
    age_scale_years: float,
    prior_strength_init: float,
) -> None:
    if use_hierarchy and num_classes != 10:
        raise ValueError("The C10/C9/C1-C8 hierarchy requires exactly 10 classes")
    if use_hierarchy and use_ihc:
        raise ValueError("The hierarchical experiment does not accept IHC inputs")
    for name, value in {
        "age_hidden_dim": age_hidden_dim,
        "location_embedding_dim": location_embedding_dim,
        "clinical_hidden_dim": clinical_hidden_dim,
        "num_locations": num_locations,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(age_scale_years) or age_scale_years <= 0.0:
        raise ValueError("age_scale_years must be finite and positive")
    if not 0.0 < prior_strength_init < 1.0:
        raise ValueError("prior_strength_init must be strictly between 0 and 1")


def build_fusion_teacher(
    feature_dim: int = 768,
    num_classes: int = 10,
    projection_dim: int = 256,
    attention_dim: int = 128,
    ihc_hidden_dim: int = 32,
    fusion_dim: int = 128,
    patch_dropout: float = 0.1,
    fusion_dropout: float = 0.2,
    l2: float = 1.0e-4,
    use_ihc: bool = True,
    ihc_mask_as_feature: bool = False,
    use_hierarchy: bool = False,
    use_clinical_prior: bool = False,
    age_hidden_dim: int = 8,
    location_embedding_dim: int = 8,
    clinical_hidden_dim: int = 16,
    num_locations: int = 11,
    age_scale_years: float = 25.0,
    prior_strength_init: float = 0.1,
) -> tf.keras.Model:
    """Build one independently trainable ten-class ensemble member.

    Flat models output raw logits. Hierarchical models output normalized
    ten-class log-probabilities, which remain valid inputs to a from-logits
    cross-entropy. All deep-ensemble members use the same architecture and
    hyperparameters; initialization, shuffling, patch sampling, and dropout
    masks differ by seed.
    """

    if not all(
        isinstance(value, bool)
        for value in (use_ihc, ihc_mask_as_feature, use_hierarchy, use_clinical_prior)
    ):
        raise TypeError("model feature switches must be boolean")
    _validate_model_arguments(
        feature_dim,
        num_classes,
        projection_dim,
        attention_dim,
        ihc_hidden_dim,
        fusion_dim,
        patch_dropout,
        fusion_dropout,
        l2,
    )
    _validate_hierarchical_arguments(
        num_classes=num_classes,
        use_ihc=use_ihc,
        use_hierarchy=use_hierarchy,
        use_clinical_prior=use_clinical_prior,
        age_hidden_dim=age_hidden_dim,
        location_embedding_dim=location_embedding_dim,
        clinical_hidden_dim=clinical_hidden_dim,
        num_locations=num_locations,
        age_scale_years=age_scale_years,
        prior_strength_init=prior_strength_init,
    )
    inputs = _make_inputs(feature_dim, use_ihc, use_clinical_prior)
    embedding = _fusion_embedding(
        inputs,
        projection_dim=projection_dim,
        attention_dim=attention_dim,
        ihc_hidden_dim=ihc_hidden_dim,
        fusion_dim=fusion_dim,
        patch_dropout=patch_dropout,
        fusion_dropout=fusion_dropout,
        l2=l2,
        use_ihc=use_ihc,
        ihc_mask_as_feature=ihc_mask_as_feature,
    )
    if use_hierarchy:
        logits = _hierarchical_logits(
            embedding,
            inputs,
            use_clinical_prior=use_clinical_prior,
            age_hidden_dim=age_hidden_dim,
            location_embedding_dim=location_embedding_dim,
            clinical_hidden_dim=clinical_hidden_dim,
            num_locations=num_locations,
            age_scale_years=age_scale_years,
            prior_strength_init=prior_strength_init,
            l2=l2,
        )
    elif use_clinical_prior:
        logits = _flat_prior_logits(
            embedding,
            inputs,
            num_classes=num_classes,
            age_hidden_dim=age_hidden_dim,
            location_embedding_dim=location_embedding_dim,
            clinical_hidden_dim=clinical_hidden_dim,
            num_locations=num_locations,
            age_scale_years=age_scale_years,
            prior_strength_init=prior_strength_init,
            l2=l2,
        )
    else:
        logits = tf.keras.layers.Dense(
            num_classes,
            kernel_regularizer=_kernel_regularizer(l2),
            name="logits",
        )(embedding)
    return tf.keras.Model(inputs=inputs, outputs=logits, name="fusion_teacher")


def build_credit_student(
    feature_dim: int = 768,
    num_classes: int = 10,
    projection_dim: int = 256,
    attention_dim: int = 128,
    ihc_hidden_dim: int = 32,
    fusion_dim: int = 128,
    patch_dropout: float = 0.1,
    fusion_dropout: float = 0.2,
    l2: float = 1.0e-4,
    use_ihc: bool = True,
    ihc_mask_as_feature: bool = False,
    use_hierarchy: bool = False,
    use_clinical_prior: bool = False,
    age_hidden_dim: int = 8,
    location_embedding_dim: int = 8,
    clinical_hidden_dim: int = 16,
    num_locations: int = 11,
    age_scale_years: float = 25.0,
    prior_strength_init: float = 0.1,
) -> tf.keras.Model:
    """Build a CREDIT student with a paper-compatible ``2C+1`` output."""

    if not all(
        isinstance(value, bool)
        for value in (use_ihc, ihc_mask_as_feature, use_hierarchy, use_clinical_prior)
    ):
        raise TypeError("model feature switches must be boolean")
    _validate_model_arguments(
        feature_dim,
        num_classes,
        projection_dim,
        attention_dim,
        ihc_hidden_dim,
        fusion_dim,
        patch_dropout,
        fusion_dropout,
        l2,
    )
    _validate_hierarchical_arguments(
        num_classes=num_classes,
        use_ihc=use_ihc,
        use_hierarchy=use_hierarchy,
        use_clinical_prior=use_clinical_prior,
        age_hidden_dim=age_hidden_dim,
        location_embedding_dim=location_embedding_dim,
        clinical_hidden_dim=clinical_hidden_dim,
        num_locations=num_locations,
        age_scale_years=age_scale_years,
        prior_strength_init=prior_strength_init,
    )
    inputs = _make_inputs(feature_dim, use_ihc, use_clinical_prior)
    embedding = _fusion_embedding(
        inputs,
        projection_dim=projection_dim,
        attention_dim=attention_dim,
        ihc_hidden_dim=ihc_hidden_dim,
        fusion_dim=fusion_dim,
        patch_dropout=patch_dropout,
        fusion_dropout=fusion_dropout,
        l2=l2,
        use_ihc=use_ihc,
        ihc_mask_as_feature=ihc_mask_as_feature,
    )
    if use_hierarchy:
        class_logits = _hierarchical_logits(
            embedding,
            inputs,
            use_clinical_prior=use_clinical_prior,
            age_hidden_dim=age_hidden_dim,
            location_embedding_dim=location_embedding_dim,
            clinical_hidden_dim=clinical_hidden_dim,
            num_locations=num_locations,
            age_scale_years=age_scale_years,
            prior_strength_init=prior_strength_init,
            l2=l2,
        )
        output = HierarchicalCreditOutput(
            num_classes=num_classes, l2=l2, name="credit_head"
        )([embedding, class_logits])
    elif use_clinical_prior:
        class_logits = _flat_prior_logits(
            embedding,
            inputs,
            num_classes=num_classes,
            age_hidden_dim=age_hidden_dim,
            location_embedding_dim=location_embedding_dim,
            clinical_hidden_dim=clinical_hidden_dim,
            num_locations=num_locations,
            age_scale_years=age_scale_years,
            prior_strength_init=prior_strength_init,
            l2=l2,
        )
        output = HierarchicalCreditOutput(
            num_classes=num_classes, l2=l2, name="credit_head"
        )([embedding, class_logits])
    else:
        output = CreditOutput(
            num_classes=num_classes, l2=l2, name="credit_head"
        )(embedding)
    return tf.keras.Model(inputs=inputs, outputs=output, name="credit_student")


__all__ = [
    "CreditOutput",
    "GatedAttentionMIL",
    "HierarchicalCreditOutput",
    "HierarchicalLogits",
    "PriorStrength",
    "build_credit_student",
    "build_fusion_teacher",
]
