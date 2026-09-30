"""Dependency-free identity for the learned codec artifact."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

CODEC_SCHEMA_VERSION = "alfworld_contextual_denoising_codec_v3"
CODEC_SKILL_VIEW_VERSION = (
    "alfworld_conditioned_verified_terminal_success_skill_pairs_view_v2"
)
CODEC_TRAINING_CONTRACT_VERSION = "codec_training_contract_v3"
CODEC_ENCODER_ROUTING_VERSION = "hybrid_tail_pool_skill_and_frozen_query_encoders_v3"
CODEC_NUMERIC_POLICY_VERSION = (
    "bf16_supported_backbones_fp32_lora_latent_projector_loss_v1"
)
CODEC_TRAINING_EXAMPLE_POLICY = (
    "unique_verified_terminal_success_skill_condition_pairs_v2"
)
CODEC_SKILL_INPUT_VERSION = "skill_only_encoder_contextual_decoder_v2"
CODEC_ENCODER_POOLING = "last_token_0.75_last4_mean_0.25_v1"
CODEC_QUERY_ENCODER_POOLING = "last_token_0.75_last4_mean_0.25_v1"
OFFICIAL_LAST_TOKEN_L2_POOLING = "official_last_token_l2_normalized_v1"
QUERY_ENCODER_POOLING_RUNTIME_AUDIT_FIELD = "query_encoder_pooling_runtime_audit"
QUERY_ENCODER_POOLING_RUNTIME_AUDIT_SCHEMA = "query_encoder_pooling_runtime_audit_v1"
QUERY_ENCODER_POOLING_OVERRIDE_SCOPE = (
    "formal_generate_assignments_query_encoder_only_v1"
)
CODEC_TRAINING_OBJECTIVE = "contextual_spherical_denoising_reconstruction_v1"
CODEC_LATENT_ONLY_TRAINING_OBJECTIVE = (
    "latent_only_spherical_denoising_reconstruction_v1"
)
CODEC_TRAIN_BATCH_TAIL_POLICY_LEGACY = "legacy_partial_v1"
CODEC_TRAIN_BATCH_TAIL_POLICY_REPEAT_EPOCH_PREFIX = (
    "repeat_epoch_prefix_to_full_batch_v1"
)
CODEC_TRAIN_BATCH_TAIL_POLICIES = (
    CODEC_TRAIN_BATCH_TAIL_POLICY_LEGACY,
    CODEC_TRAIN_BATCH_TAIL_POLICY_REPEAT_EPOCH_PREFIX,
)
CODEC_ARTIFACT_POLICY_FIELDS = (
    "encoder_pooling",
    "query_encoder_pooling",
    "decoder_task_conditioned",
    "training_objective",
    "latent_noise_angle_std",
    "latent_noise_max_angle",
    "latent_noise_prob",
    "latent_noise_ramp_epochs",
    "encoder_lora_last_n_layers",
)

# These values can change learned weights or model selection. Runtime-only
# logging, worker count, and output paths are deliberately excluded.
CODEC_TRAINING_CONTRACT_FIELDS = (
    "encoder_model",
    "query_encoder_model",
    "decoder_model",
    "attention_implementation",
    "backbone_dtype",
    "float32_matmul_precision",
    "fused_optimizer",
    "checkpoint_selection_mode",
    "forced_train_oversample_factor",
    "validation_ratio",
    "split_seed",
    "training_seed",
    "early_stopping_patience",
    "min_epochs",
    "epochs",
    "loss_mode",
    "batch_size",
    "gradient_accumulation",
    "max_length",
    "num_prefix_tokens",
    "encoder_lora_rank",
    "encoder_lora_alpha",
    "latent_noise_angle_std",
    "latent_noise_max_angle",
    "latent_noise_prob",
    "latent_noise_ramp_epochs",
    "encoder_lora_dropout",
    "decoder_lora_rank",
    "decoder_lora_alpha",
    "encoder_lora_last_n_layers",
    "decoder_lora_dropout",
    "train_encoder",
    "encoder_skill_prompt",
    "encoder_query_prompt",
    "decoder_prompt",
    "encoder_lr",
    "decoder_lr",
    "projector_lr",
    "warmup_ratio",
)


def decoder_prompt_uses_task_context(prompt: object) -> bool:
    """Return whether a decoder prompt consumes explicit task metadata.

    A supported prompt is either fully contextual (both ``task_type`` and
    ``goal`` placeholders) or fully latent-only (neither placeholder). Partial
    conditioning and the old opaque ``context`` placeholder are rejected.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Codec decoder prompt must be a non-empty string")
    if "{context}" in prompt:
        raise ValueError(
            "Codec decoder prompt must not use the opaque {context} placeholder"
        )
    has_task_type = "{task_type}" in prompt
    has_goal = "{goal}" in prompt
    if has_task_type != has_goal:
        raise ValueError(
            "Codec decoder prompt must contain both {task_type} and {goal}, "
            "or contain neither for latent-only reconstruction"
        )
    return bool(has_task_type)


def codec_training_objective(*, decoder_task_conditioned: bool) -> str:
    if not isinstance(decoder_task_conditioned, bool):
        raise TypeError("decoder_task_conditioned must be a boolean")
    return (
        CODEC_TRAINING_OBJECTIVE
        if decoder_task_conditioned
        else CODEC_LATENT_ONLY_TRAINING_OBJECTIVE
    )


def normalize_encoder_pooling_profile(value: object) -> str:
    """Return one of the two explicit v3-compatible encoder poolers.

    ``official_last_token_l2`` is accepted only as a legacy short alias.
    Runtime metadata always records ``official_last_token_l2_normalized_v1``
    so an override cannot be hidden behind aliases.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Encoder pooling profile must be a non-empty string")
    normalized = value.strip()
    if normalized == "official_last_token_l2":
        normalized = OFFICIAL_LAST_TOKEN_L2_POOLING
    supported = {CODEC_QUERY_ENCODER_POOLING, OFFICIAL_LAST_TOKEN_L2_POOLING}
    if normalized not in supported:
        raise ValueError(
            f"Unsupported encoder pooling profile={value!r}; "
            f"expected one of {sorted(supported)}"
        )
    return normalized


def evaluation_query_encoder_pooling_override(
    cfg: Mapping[str, Any],
) -> str | None:
    """Read the formal-assignment-only compatibility override from config."""

    if not isinstance(cfg, Mapping):
        raise TypeError("Configuration must be a mapping")
    evaluation = cfg.get("evaluation")
    if evaluation is None:
        return None
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation configuration must be a mapping")
    raw = evaluation.get("query_encoder_pooling_override")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    return normalize_encoder_pooling_profile(raw)


def build_query_encoder_pooling_runtime_audit(
    *,
    artifact_skill_pooling: object,
    artifact_query_pooling: object,
    evaluation_override: object | None,
) -> dict[str, Any]:
    """Describe declared, internal, and formal-assignment pooling honestly."""

    declared_skill = normalize_encoder_pooling_profile(artifact_skill_pooling)
    declared_query = normalize_encoder_pooling_profile(artifact_query_pooling)
    override = (
        None
        if evaluation_override is None
        else normalize_encoder_pooling_profile(evaluation_override)
    )
    actual = override or declared_query
    intentional = override is not None and actual != declared_query
    return {
        "schema_version": QUERY_ENCODER_POOLING_RUNTIME_AUDIT_SCHEMA,
        "artifact_declared_skill_encoder_pooling": declared_skill,
        "artifact_declared_query_encoder_pooling": declared_query,
        "training_skill_encoder_pooling": declared_skill,
        "training_and_flow_internal_query_encoder_pooling": declared_query,
        "actual_runtime_query_encoder_pooling": actual,
        "configured_evaluation_override": override,
        "intentional_compatibility_override": intentional,
        "override_scope": (
            QUERY_ENCODER_POOLING_OVERRIDE_SCOPE if intentional else None
        ),
    }


def validate_query_encoder_pooling_runtime_audit(
    value: object,
    *,
    expected_artifact_skill_pooling: object = CODEC_ENCODER_POOLING,
    expected_artifact_query_pooling: object = CODEC_QUERY_ENCODER_POOLING,
    expected_evaluation_override: object | None,
) -> dict[str, Any]:
    """Require runtime audit metadata to equal the configured dual profile."""

    if not isinstance(value, Mapping):
        raise TypeError("Query encoder pooling runtime audit must be a mapping")
    expected = build_query_encoder_pooling_runtime_audit(
        artifact_skill_pooling=expected_artifact_skill_pooling,
        artifact_query_pooling=expected_artifact_query_pooling,
        evaluation_override=expected_evaluation_override,
    )
    observed = dict(value)
    if observed != expected:
        raise ValueError(
            "Query encoder pooling runtime audit mismatch: "
            f"observed={observed!r} expected={expected!r}"
        )
    return expected


def validate_codec_artifact_policy(
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the fixed semantics and typed ranges recorded by v3 artifacts."""

    if not isinstance(values, Mapping):
        raise TypeError("codec v3 artifact policy must be a mapping")
    for field, expected in (
        ("encoder_pooling", CODEC_ENCODER_POOLING),
        ("query_encoder_pooling", CODEC_QUERY_ENCODER_POOLING),
    ):
        observed = values.get(field)
        if not isinstance(observed, str) or observed != expected:
            raise ValueError(
                f"codec v3 {field} must equal {expected!r}; observed={observed!r}"
            )
    decoder_task_conditioned = values.get("decoder_task_conditioned")
    if not isinstance(decoder_task_conditioned, bool):
        raise TypeError("codec v3 decoder_task_conditioned must be a boolean")
    expected_objective = codec_training_objective(
        decoder_task_conditioned=decoder_task_conditioned
    )
    observed_objective = values.get("training_objective")
    if (
        not isinstance(observed_objective, str)
        or observed_objective != expected_objective
    ):
        raise ValueError(
            "codec v3 training_objective must equal "
            f"{expected_objective!r}; observed={observed_objective!r}"
        )

    noise: dict[str, float] = {}
    for field in (
        "latent_noise_angle_std",
        "latent_noise_max_angle",
        "latent_noise_prob",
    ):
        raw = values.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"codec v3 {field} must be a finite number")
        normalized = float(raw)
        if not math.isfinite(normalized):
            raise ValueError(f"codec v3 {field} must be finite")
        noise[field] = normalized
    if noise["latent_noise_angle_std"] < 0.0:
        raise ValueError("codec v3 latent_noise_angle_std must be non-negative")
    if noise["latent_noise_max_angle"] < noise["latent_noise_angle_std"]:
        raise ValueError(
            "codec v3 latent_noise_max_angle must be >= latent_noise_angle_std"
        )
    if not 0.0 <= noise["latent_noise_prob"] <= 1.0:
        raise ValueError("codec v3 latent_noise_prob must be in [0, 1]")

    integers: dict[str, int] = {}
    for field, minimum in (
        ("latent_noise_ramp_epochs", 0),
        ("encoder_lora_last_n_layers", 1),
    ):
        raw = values.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"codec v3 {field} must be an integer")
        if raw < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"codec v3 {field} must be {qualifier}")
        integers[field] = int(raw)

    return {
        "encoder_pooling": CODEC_ENCODER_POOLING,
        "query_encoder_pooling": CODEC_QUERY_ENCODER_POOLING,
        "decoder_task_conditioned": decoder_task_conditioned,
        "training_objective": expected_objective,
        **noise,
        **integers,
    }


# Compatibility names consumed by the frozen v3 autoencoder implementation.
CODEC_V3_ARTIFACT_POLICY_FIELDS = CODEC_ARTIFACT_POLICY_FIELDS
validate_codec_v3_artifact_policy = validate_codec_artifact_policy


def codec_training_contract(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    training_seed = normalized.get("training_seed")
    if training_seed is None or int(training_seed) < 0:
        normalized["training_seed"] = normalized.get("split_seed")
    tail_policy = normalized.get("train_batch_tail_policy")
    if tail_policy in (None, ""):
        tail_policy = CODEC_TRAIN_BATCH_TAIL_POLICY_LEGACY
    if tail_policy not in CODEC_TRAIN_BATCH_TAIL_POLICIES:
        raise ValueError(
            "Unsupported Codec train_batch_tail_policy: " f"{tail_policy!r}"
        )
    contract_values = {
        field: normalized.get(field) for field in CODEC_TRAINING_CONTRACT_FIELDS
    }
    # The opt-in tail policy is identity-bearing because it changes which rows
    # enter the learned update.
    if tail_policy != CODEC_TRAIN_BATCH_TAIL_POLICY_LEGACY:
        contract_values["train_batch_tail_policy"] = tail_policy
    return {
        "contract_version": CODEC_TRAINING_CONTRACT_VERSION,
        "codec_schema_version": CODEC_SCHEMA_VERSION,
        "skill_view_version": CODEC_SKILL_VIEW_VERSION,
        "encoder_routing_version": CODEC_ENCODER_ROUTING_VERSION,
        "numeric_policy_version": CODEC_NUMERIC_POLICY_VERSION,
        "training_example_policy": CODEC_TRAINING_EXAMPLE_POLICY,
        "skill_input_version": CODEC_SKILL_INPUT_VERSION,
        "values": contract_values,
    }


def codec_training_contract_sha256(values: Mapping[str, Any]) -> str:
    payload = json.dumps(
        codec_training_contract(values),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CODEC_SCHEMA_VERSION",
    "CODEC_SKILL_VIEW_VERSION",
    "CODEC_ENCODER_ROUTING_VERSION",
    "CODEC_NUMERIC_POLICY_VERSION",
    "CODEC_TRAINING_EXAMPLE_POLICY",
    "CODEC_SKILL_INPUT_VERSION",
    "CODEC_ENCODER_POOLING",
    "CODEC_QUERY_ENCODER_POOLING",
    "OFFICIAL_LAST_TOKEN_L2_POOLING",
    "QUERY_ENCODER_POOLING_RUNTIME_AUDIT_FIELD",
    "QUERY_ENCODER_POOLING_RUNTIME_AUDIT_SCHEMA",
    "QUERY_ENCODER_POOLING_OVERRIDE_SCOPE",
    "CODEC_TRAINING_OBJECTIVE",
    "CODEC_LATENT_ONLY_TRAINING_OBJECTIVE",
    "CODEC_TRAIN_BATCH_TAIL_POLICY_LEGACY",
    "CODEC_TRAIN_BATCH_TAIL_POLICY_REPEAT_EPOCH_PREFIX",
    "CODEC_TRAIN_BATCH_TAIL_POLICIES",
    "CODEC_ARTIFACT_POLICY_FIELDS",
    "CODEC_TRAINING_CONTRACT_FIELDS",
    "CODEC_TRAINING_CONTRACT_VERSION",
    "codec_training_contract",
    "codec_training_contract_sha256",
    "codec_training_objective",
    "decoder_prompt_uses_task_context",
    "normalize_encoder_pooling_profile",
    "evaluation_query_encoder_pooling_override",
    "build_query_encoder_pooling_runtime_audit",
    "validate_query_encoder_pooling_runtime_audit",
    "validate_codec_artifact_policy",
]
