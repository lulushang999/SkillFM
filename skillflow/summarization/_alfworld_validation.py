from __future__ import annotations

import copy

import difflib

import hashlib

import json

import math

import re

from typing import Any, Optional

GENERATED_SUMMARY_VALIDATOR_VERSION = "alfworld_iterative_summary_v1"

SKILL_NEAR_DUPLICATE_SEQUENCE_THRESHOLD = 0.88

SKILL_NEAR_DUPLICATE_TOKEN_JACCARD_THRESHOLD = 0.78

_SKILL_CONCEPT_PATTERNS = {
    "alternative": r"\b(?:alternative|another|different|switch|untried|unused|new choice)\b",
    "container": r"\b(?:cabinet|drawer|container|receptacle|open|closed)\b",
    "count": r"\b(?:both|count|second|two|twice)\b",
    "exploration": r"\b(?:explore|inspect|locate|look|navigate|search|room|visible)\b",
    "goal_grounding": r"\b(?:goal|requirement|subgoal|target|unsatisfied)\b",
    "inventory": r"\b(?:inventory|carrying|held|holding|possess)\b",
    "loop": r"\b(?:loop|cycle|repeat|repeated|repeating)\b",
    "no_progress": r"\b(?:no progress|unchanged|no state change|without progress|fails? to (?:advance|change)|made no change)\b",
    "observation": r"\b(?:observation|current state|environment evidence|resulting state)\b",
    "pickup": r"\b(?:acquire|pick up|pickup|take|hold the target)\b",
    "placement": r"\b(?:place|placement|put|destination|target receptacle)\b",
    "state_change": r"\b(?:clean|cool|heat|slice|microwave|fridge|sink|knife|appliance)\b",
    "verification": r"\b(?:check|confirm|verify|ensure|observe the result)\b",
}

_SKILL_DECISION_RULE_TOKENS = frozenset(
    {
        "after",
        "before",
        "check",
        "confirm",
        "count",
        "first",
        "if",
        "instead",
        "inventory",
        "never",
        "not",
        "once",
        "otherwise",
        "retry",
        "second",
        "stop",
        "switch",
        "then",
        "unless",
        "until",
        "verify",
        "without",
    }
)

TASK_TYPES = {
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
}

EXECUTABLE_SKILL_KEYS = {
    "when_to_use",
    "high_level_guidance",
    "low_level_guidance",
}

TASK_KEYS = {"task_id", "task_type", "canonical_goal"}

class ChainContractError(ValueError):
    """Raised when a generated summary or stored chain violates the contract."""

class GeneratedSummaryFormatError(ChainContractError):
    """The model output is not a valid three-string structured skill."""

class GeneratedSummaryExactDuplicateError(ChainContractError):
    """The model output exactly repeats a skill already in the same sample."""

def _require_exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ChainContractError(f"{label} must be a JSON object")
    actual = {str(key) for key in value}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ChainContractError(f"{label} keys mismatch: missing={missing} extra={extra}")
    return value

def _require_string(
    value: object,
    label: str,
    *,
    min_chars: int = 1,
    max_chars: Optional[int] = 1600,
) -> str:
    if not isinstance(value, str):
        raise ChainContractError(f"{label} must be a string")
    if value != value.strip():
        raise ChainContractError(f"{label} must not have leading/trailing whitespace")
    if len(value) < min_chars:
        raise ChainContractError(
            f"{label} length must be at least {min_chars}, got {len(value)}"
        )
    if max_chars is not None and len(value) > max_chars:
        raise ChainContractError(
            f"{label} length must be in [{min_chars}, {max_chars}], got {len(value)}"
        )
    if re.search(r"```|</?think>|\[/?think\]", value, flags=re.I):
        raise ChainContractError(f"{label} contains forbidden chat/markdown wrappers")
    return value

def _normalize_goal(goal: object) -> str:
    return re.sub(r"\s+", " ", str(goal or "").strip().casefold()).rstrip(".?!")

def _match_goal(patterns: list[str], goal: str, task_type: str) -> re.Match[str]:
    for pattern in patterns:
        match = re.fullmatch(pattern, goal, flags=re.I)
        if match is not None:
            return match
    raise ChainContractError(
        f"Unsupported canonical goal for task_type={task_type!r}: {goal!r}"
    )

def extract_goal_slots(task_type: object, canonical_goal: object) -> dict[str, Any]:
    """Parse the six ALFWorld task families into explicit immutable goal slots.

    The parser is deliberately strict. A new goal grammar must be added here rather
    than silently asking the summarizer to invent target slots.
    """
    task_type = str(task_type or "").strip()
    if task_type not in TASK_TYPES:
        raise ChainContractError(f"Unsupported task_type={task_type!r}")
    goal = _normalize_goal(canonical_goal)
    if not goal:
        raise ChainContractError("canonical_goal is required")

    if task_type == "look_at_obj_in_light":
        match = _match_goal(
            [
                r"look at (?:the )?(?P<object>[a-z0-9_]+) under (?:the )?(?P<tool>[a-z0-9_]+)",
                r"examine (?:the )?(?P<object>[a-z0-9_]+) with (?:the )?(?P<tool>[a-z0-9_]+)",
            ],
            goal,
            task_type,
        )
        return {
            "task_type": task_type,
            "target_object": match.group("object"),
            "target_receptacle": "none",
            "required_state": "illuminated",
            "required_tool": match.group("tool"),
            "object_count": 1,
        }

    if task_type == "pick_two_obj_and_place":
        match = _match_goal(
            [
                r"find two (?P<object>[a-z0-9_]+) and put them (?:in|on) (?P<receptacle>[a-z0-9_]+)",
                r"put two (?P<object>[a-z0-9_]+) (?:in|on) (?P<receptacle>[a-z0-9_]+)",
            ],
            goal,
            task_type,
        )
        return {
            "task_type": task_type,
            "target_object": match.group("object"),
            "target_receptacle": match.group("receptacle"),
            "required_state": "none",
            "required_tool": "none",
            "object_count": 2,
        }

    state_specs = {
        "pick_clean_then_place_in_recep": ("clean", "clean", "sinkbasin"),
        "pick_heat_then_place_in_recep": ("heat", "hot", "microwave"),
        "pick_cool_then_place_in_recep": ("cool", "cool", "fridge"),
    }
    if task_type in state_specs:
        verb, required_state, required_tool = state_specs[task_type]
        adjective = {"clean": "clean", "heat": "hot", "cool": "cool"}[verb]
        match = _match_goal(
            [
                rf"{verb} (?:some|a|an|the) (?P<object>[a-z0-9_]+) and put it (?:in|on) (?P<receptacle>[a-z0-9_]+)",
                rf"put (?:some|a|an|the) {adjective} (?P<object>[a-z0-9_]+) (?:in|on) (?P<receptacle>[a-z0-9_]+)",
            ],
            goal,
            task_type,
        )
        return {
            "task_type": task_type,
            "target_object": match.group("object"),
            "target_receptacle": match.group("receptacle"),
            "required_state": required_state,
            "required_tool": required_tool,
            "object_count": 1,
        }

    match = _match_goal(
        [
            r"put (?:some|a|an|the) (?P<object>[a-z0-9_]+) (?:in|on) (?P<receptacle>[a-z0-9_]+)",
        ],
        goal,
        task_type,
    )
    return {
        "task_type": task_type,
        "target_object": match.group("object"),
        "target_receptacle": match.group("receptacle"),
        "required_state": "none",
        "required_tool": "none",
        "object_count": 1,
    }

def validate_executable_skill(
    value: object,
    *,
    expected_goal_slots: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    # Task identity conditions the executor externally. The stored skill contract
    # is intentionally limited to the exact three-field structured JSON shape.
    del expected_goal_slots
    skill = _require_exact_keys(value, EXECUTABLE_SKILL_KEYS, "executable_skill")
    return {
        "when_to_use": _require_string(
            skill["when_to_use"],
            "executable_skill.when_to_use",
            max_chars=None,
        ),
        "high_level_guidance": _require_string(
            skill["high_level_guidance"],
            "executable_skill.high_level_guidance",
            max_chars=None,
        ),
        "low_level_guidance": _require_string(
            skill["low_level_guidance"],
            "executable_skill.low_level_guidance",
            max_chars=None,
        ),
    }

def _normalized_skill_novelty_text(skill: dict[str, Any]) -> str:
    text = " ".join(
        str(skill[field])
        for field in (
            "high_level_guidance",
            "low_level_guidance",
        )
    ).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))

def _skill_concept_fingerprint(
    skill: dict[str, Any],
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(
            sorted(
                concept
                for concept, pattern in _SKILL_CONCEPT_PATTERNS.items()
                if re.search(pattern, str(skill[field]), flags=re.I)
            )
        )
        for field in (
            "when_to_use",
            "high_level_guidance",
            "low_level_guidance",
        )
    )

def skill_near_duplicate_evidence(
    candidate: object,
    prior: object,
) -> dict[str, Any] | None:
    """Return deterministic evidence when two compact skills repeat one lesson.

    This is a storage-quality gate, not an environment verifier.  It compares
    only already validated prose using literal sequence overlap, token-set
    overlap, and a small ALFWORLD concept fingerprint.  No model/API call or
    task outcome is involved.
    """

    candidate_skill = validate_executable_skill(candidate)
    prior_skill = validate_executable_skill(prior)
    candidate_text = _normalized_skill_novelty_text(candidate_skill)
    prior_text = _normalized_skill_novelty_text(prior_skill)
    sequence_ratio = difflib.SequenceMatcher(
        None,
        candidate_text,
        prior_text,
        autojunk=False,
    ).ratio()
    candidate_tokens = set(candidate_text.split())
    prior_tokens = set(prior_text.split())
    union = candidate_tokens | prior_tokens
    token_jaccard = (
        len(candidate_tokens & prior_tokens) / len(union) if union else 1.0
    )
    candidate_concepts = _skill_concept_fingerprint(candidate_skill)
    prior_concepts = _skill_concept_fingerprint(prior_skill)
    concept_count = sum(len(values) for values in candidate_concepts)
    concept_equivalent = (
        concept_count >= 3 and candidate_concepts == prior_concepts
    )
    novel_tokens = candidate_tokens - prior_tokens
    candidate_concept_set = {
        concept for field_concepts in candidate_concepts for concept in field_concepts
    }
    prior_concept_set = {
        concept for field_concepts in prior_concepts for concept in field_concepts
    }
    novel_concepts = candidate_concept_set - prior_concept_set
    novel_decision_tokens = novel_tokens & _SKILL_DECISION_RULE_TOKENS
    # Preserve cumulative revisions that really add a conditional branch or
    # intervention.  These naturally share most wording with their parent, so
    # a raw edit ratio alone would incorrectly reject the best kind of chain
    # update (for example, adding "after one retry, check inventory instead").
    material_addition = (
        len(candidate_text.split()) >= len(prior_text.split()) + 4
        and len(novel_tokens) >= 3
        and bool(novel_concepts or novel_decision_tokens)
    )
    if material_addition:
        return None
    if not (
        sequence_ratio >= SKILL_NEAR_DUPLICATE_SEQUENCE_THRESHOLD
        or token_jaccard >= SKILL_NEAR_DUPLICATE_TOKEN_JACCARD_THRESHOLD
        or concept_equivalent
    ):
        return None
    return {
        "sequence_ratio": round(float(sequence_ratio), 6),
        "token_jaccard": round(float(token_jaccard), 6),
        "concept_equivalent": bool(concept_equivalent),
        "novel_concepts": sorted(novel_concepts),
        "novel_decision_tokens": sorted(novel_decision_tokens),
        "candidate_concepts": [list(values) for values in candidate_concepts],
        "prior_concepts": [list(values) for values in prior_concepts],
        "policy": GENERATED_SUMMARY_VALIDATOR_VERSION,
    }

def validate_summary_payload(
    value: object,
    *,
    task_type: object,
    canonical_goal: object,
) -> dict[str, Any]:
    extract_goal_slots(task_type, canonical_goal)
    return validate_executable_skill(value)

def _task_from_rollout(rollout: dict[str, Any]) -> dict[str, str]:
    task_id = _require_string(str(rollout.get("task_id") or "").strip(), "task.task_id", max_chars=128)
    task_type = _require_string(str(rollout.get("task_type") or "").strip(), "task.task_type", max_chars=128)
    canonical_goal = _require_string(
        str(rollout.get("canonical_goal") or rollout.get("goal") or "").strip(),
        "task.canonical_goal",
        max_chars=1000,
    )
    extract_goal_slots(task_type, canonical_goal)
    return {"task_id": task_id, "task_type": task_type, "canonical_goal": canonical_goal}

def validate_generated_summary_for_rollout(
    value: object,
    rollout: dict[str, Any],
) -> dict[str, Any]:
    """Validate a retry skill or the round-zero success distillation."""

    from skillflow.summarization._alfworld_assignment import (
        validate_collection_assignment_rollout,
    )

    if not isinstance(rollout, dict) or not isinstance(rollout.get("success"), bool):
        raise ChainContractError("rollout.success must be a JSON boolean")
    validate_collection_assignment_rollout(rollout)
    if rollout["success"] is True and rollout.get("skill_used") != "":
        raise ChainContractError(
            "a successful skilled rollout publishes skill_used directly"
        )
    task = _task_from_rollout(rollout)
    try:
        candidate = validate_summary_payload(
            value,
            task_type=task["task_type"],
            canonical_goal=task["canonical_goal"],
        )
    except ChainContractError as exc:
        raise GeneratedSummaryFormatError(str(exc)) from exc

    raw_skill = str(rollout.get("skill_used") or "").strip()
    if rollout["success"] is False and raw_skill:
        previous = validate_executable_skill(json.loads(raw_skill))
        if candidate == previous:
            raise GeneratedSummaryExactDuplicateError(
                "failed-rollout summary exactly repeats the executed skill"
            )
        duplicate = skill_near_duplicate_evidence(candidate, previous)
        if duplicate is not None:
            raise GeneratedSummaryExactDuplicateError(
                "failed-rollout summary is not a material revision of the "
                f"executed skill: {duplicate}"
            )
    return candidate

def validate_task(value: object) -> dict[str, str]:
    task = _require_exact_keys(value, TASK_KEYS, "task")
    clean = {
        "task_id": _require_string(task["task_id"], "task.task_id", max_chars=128),
        "task_type": _require_string(task["task_type"], "task.task_type", max_chars=128),
        "canonical_goal": _require_string(task["canonical_goal"], "task.canonical_goal", max_chars=1000),
    }
    extract_goal_slots(clean["task_type"], clean["canonical_goal"])
    return clean
