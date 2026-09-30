"""Compact the audited WebShop projection before placing it in a model prompt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


WEBSHOP_MODEL_PROJECTION_COMPACTION_VERSION = "webshop_model_projection_compaction_v1"

_REDUNDANT_PER_FIELD_KEYS = frozenset({"original_sha256", "shown_chars"})
_ZERO_DIAGNOSTIC_KEYS = frozenset(
    {
        "redacted_instance_identifiers",
        "neutralized_chat_control_tokens",
        "item_text_truncated_count",
    }
)


def _compact(value: object) -> Any:
    if isinstance(value, Mapping):
        source = dict(value)
        result: dict[str, Any] = {}
        for raw_key, nested in source.items():
            key = str(raw_key)
            # The trajectory-level hash already binds the exact original
            # observation/action list. Repeating random-looking SHA strings for
            # every field wastes thousands of summary tokens without adding a
            # new integrity guarantee.
            if key in _REDUNDANT_PER_FIELD_KEYS:
                continue
            if key in _ZERO_DIAGNOSTIC_KEYS and not nested:
                continue
            if key == "proposed_action" and nested == source.get("action"):
                result["proposed_action_same_as_action"] = True
                continue
            result[key] = _compact(nested)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_compact(item) for item in value]
    return value


def compact_webshop_prompt_projection(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a smaller model view while preserving steps and truncation facts.

    The detailed projection remains available to callers. The compact view
    keeps the one trajectory-level SHA-256, canonical goal hash, all steps,
    shown text, original lengths, omission counts, and every non-zero redaction
    diagnostic. It removes only redundant per-field hashes/lengths and duplicate
    proposed actions.
    """

    if not isinstance(projection, Mapping):
        raise TypeError("WebShop prompt projection must be a mapping")
    compacted = _compact(projection)
    if not isinstance(compacted, dict):
        raise AssertionError("Compacted WebShop projection must remain an object")
    trajectory = compacted.get("trajectory")
    if (
        not isinstance(trajectory, dict)
        or trajectory.get("all_steps_preserved") is not True
    ):
        raise ValueError("WebShop prompt compaction requires all preserved steps")
    if not str(trajectory.get("original_observation_action_list_sha256") or ""):
        raise ValueError("WebShop prompt compaction requires the source-list hash")
    return compacted


__all__ = [
    "WEBSHOP_MODEL_PROJECTION_COMPACTION_VERSION",
    "compact_webshop_prompt_projection",
]
