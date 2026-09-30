from __future__ import annotations

import copy

import hashlib

import json

from pathlib import Path

from typing import Iterable, Mapping

from skillflow.summarization._alfworld_validation import (
    ChainContractError,
    validate_executable_skill,
    validate_task,
)

COLLECTION_ASSIGNMENT_SCHEMA_VERSION = "collection_assignment_v1"

COLLECTION_ASSIGNMENT_SOURCE = "iterative_collection_v1"

EMPTY_SUMMARY_SOURCE_KIND = "empty_summary_fallback"

ASSIGNMENT_METADATA_KEYS = {
    "schema_version",
    "source_kind",
    "collection_round",
    "skill_sha256",
    "generated_from_round",
}

def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

def skill_sha256(value: object) -> str:
    """Hash one strictly validated executable skill."""

    return _canonical_json_sha256(validate_executable_skill(value))

def validate_collection_assignment_rollout(
    rollout: object,
    *,
    expected_round: int | None = None,
) -> dict:
    """Validate the exact retry state that was injected into one rollout."""

    if not isinstance(rollout, dict):
        raise ChainContractError("collection rollout must be a JSON object")
    if rollout.get("assignment_source") != COLLECTION_ASSIGNMENT_SOURCE:
        raise ChainContractError(
            f"collection assignment_source must be {COLLECTION_ASSIGNMENT_SOURCE!r}"
        )
    metadata = rollout.get("assignment_metadata")
    if not isinstance(metadata, dict) or set(metadata) != ASSIGNMENT_METADATA_KEYS:
        raise ChainContractError(
            "collection assignment metadata has an unexpected shape"
        )
    if metadata.get("schema_version") != COLLECTION_ASSIGNMENT_SCHEMA_VERSION:
        raise ChainContractError("unexpected collection assignment schema")
    round_id = metadata.get("collection_round")
    if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
        raise ChainContractError("collection_round must be non-negative")
    if expected_round is not None and round_id != expected_round:
        raise ChainContractError(
            f"collection rollout round mismatch: {round_id} != {expected_round}"
        )

    raw_skill = rollout.get("skill_used")
    if raw_skill == "":
        if metadata.get("skill_sha256") is not None:
            raise ChainContractError("unskilled assignment must have null skill_sha256")
        source_kind = metadata.get("source_kind")
        generated_from_round = metadata.get("generated_from_round")
        if source_kind == "unskilled_initial":
            if generated_from_round is not None:
                raise ChainContractError(
                    "initial unskilled assignment must have null generated_from_round"
                )
            if round_id != 0:
                raise ChainContractError(
                    "unskilled_initial is valid only in collection round 0"
                )
            return copy.deepcopy(metadata)
        if source_kind != EMPTY_SUMMARY_SOURCE_KIND:
            raise ChainContractError(
                "empty collection skill must be initial or an empty-summary fallback"
            )
        if (
            isinstance(generated_from_round, bool)
            or not isinstance(generated_from_round, int)
            or generated_from_round < 0
            or generated_from_round >= round_id
        ):
            raise ChainContractError(
                "empty-summary fallback must come from an earlier round"
            )
        return copy.deepcopy(metadata)

    if metadata.get("source_kind") != "transient_summary":
        raise ChainContractError("non-empty collection skill must be transient_summary")
    try:
        parsed = json.loads(str(raw_skill))
    except Exception as exc:
        raise ChainContractError("collection skill_used must be valid JSON") from exc
    skill = validate_executable_skill(parsed)
    if metadata.get("skill_sha256") != skill_sha256(skill):
        raise ChainContractError("collection assignment skill hash mismatch")
    generated_from_round = metadata.get("generated_from_round")
    if (
        isinstance(generated_from_round, bool)
        or not isinstance(generated_from_round, int)
        or generated_from_round < 0
        or generated_from_round >= round_id
    ):
        raise ChainContractError(
            "transient skill must have been generated in an earlier round"
        )
    return copy.deepcopy(metadata)
