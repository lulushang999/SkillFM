# ruff: noqa: E501

"""QA-only trajectory summarization with strict, sample-bound skills.

This module intentionally does not reuse the ALFWorld prompt, rollout validator,
retry policy, or capped-generation recovery contract.  ``QASkillSummarizer``
inherits the shared model construction and generation configuration from
``SkillSummarizer`` only; every operation that interprets a rollout or generated
text is QA-specific here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Any

import torch

from skillflow.summarization.base import (
    SkillSummarizer,
    _RecoverableSummaryOutputError,
    _check_no_prompt_truncation,
    _eos_token_ids,
    _final_answer_token_ids,
    _generation_cutoff_failure_reasons,
)


QA_SUMMARIZER_IMPLEMENTATION_VERSION = "qa_sample_skill_json_v2"
QA_SUMMARY_PROMPT_SCHEMA_VERSION = "qa_sample_skill_summary_prompt_v1"
QA_SUMMARY_RESULT_SCHEMA_VERSION = "qa_sample_skill_summary_result_v1"
QA_SKILL_FIELDS = (
    "when_to_use",
    "high_level_guidance",
    "low_level_guidance",
)
QA_SKILL_FIELD_MAX_CHARS = 1600
QA_LEGACY_PROMPT_CONTRACT_SHA256 = (
    "8fc977fbfd8c1302798a0ba4a82392f69cea3b86d579aac6b1d4c68e524667e9"
)
# Ordinary grammatical use is not a factual answer assertion.
_FUNCTION_WORD_ANSWERS = frozenset(
    "a an the to of in on at by for from with as and or but if than then "
    "is are was were be been being it its this that these those "
    "he she they them his her their we us you your yes no not".split()
)
QA_RETRY_CORRECTION = """Correction attempt: {ATTEMPT}/{MAX_ATTEMPTS}
The previous response was rejected for QA summary contract error: {ERROR}.
Affected fields: {FIELDS}.
Rewrite the skill as procedures, not a solution to this query. Replace concrete
names, places, dates, technical answer terms and asserted facts with roles such
as the subject, requested property, source and destination. Keep useful ordering
and verification checks. Do not supply example answers. Return only the exact
three-field JSON object."""
QA_PENDING_CORRECTION = """Pending-sample repair pass: {ROUND}.
An earlier distillation could not be accepted. Use one short procedural sentence
per field. Describe the query pattern, the evidence operation and the completion
check without naming this query's entities or stating their properties."""

QA_OUTPUT_CONTRACT = """Return exactly one JSON object and nothing else:
{
  "when_to_use": "Describe the query pattern and evidence need that should trigger this skill.",
  "high_level_guidance": "Describe the ordered reasoning or retrieval steps and the completion check.",
  "low_level_guidance": "Give concise evidence-use and answer-verification rules grounded in the trajectory while generalizing away instance-specific facts and answers."
}
The object must contain exactly those three keys. Every value must be one
non-empty string. Do not emit markdown, analysis, metadata, outcome labels,
arrays, nulls, extra keys, the gold answer, or the trajectory's final answer.
End immediately after the JSON object."""

QA_SUCCESS_PROMPT = """You distill one successful QA trajectory into one reusable skill.

Query: {QUERY}

Successful trajectory:
{TRAJECTORY}

Think privately about which actions and evidence made the trajectory succeed.
Preserve only supported strategy and ordering: do not invent intermediate steps
for a one-step trajectory, and retain necessary evidence links for a multi-step
trajectory. Generalize away instance-specific entities, passage wording, and
the exact answer without inventing facts or failure lessons.

{OUTPUT_CONTRACT}"""

QA_FAILURE_PROMPT = """You distill one failed QA trajectory into a reusable skill for a better next attempt.

Query: {QUERY}
{CURRENT_SKILL_BLOCK}
Failed trajectory:
{TRAJECTORY}

Failure status: {FAILURE_REASON}

Think privately about the first unsupported answer, missing evidence,
unproductive search, broken evidence link, or absent verification rule. Produce
a skill that materially improves the next attempt while generalizing away
instance-specific entities and answers.

{OUTPUT_CONTRACT}"""

_PROMPT_TRAJECTORY_PROJECTION = (
    "query_observation_action_terminal_observation_outcome_v1"
)
_SUMMARY_AUGMENTED_FIELDS = frozenset(
    {
        "distilled_skill",
        "qa_summary",
        "qa_summary_result",
        "summarized_skill",
        "summary_result",
    }
)


class QASummaryError(ValueError):
    """Base error for invalid QA summary inputs or outputs."""


class QASummaryContractError(QASummaryError):
    """A generated value is not a safe exact three-string QA skill."""

    def __init__(self, code: str, message: str, *, fields: Sequence[str] = ()):
        super().__init__(message)
        self.code = str(code)
        self.fields = tuple(fields)


class IncompleteQASummaryError(QASummaryError):
    """At least one source sample has no accepted skill."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QASummaryError(f"Value is not finite canonical JSON: {exc}") from exc


def _required_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise QASummaryError(f"{label} must be non-empty")
    return text


def qa_query_text(rollout: Mapping[str, Any]) -> str:
    """Resolve the query without depending on ALFWorld's canonical-goal field."""

    if not isinstance(rollout, Mapping):
        raise QASummaryError("QA rollout must be a mapping")
    return _required_text(
        rollout.get("query")
        or rollout.get("canonical_query")
        or rollout.get("question")
        or rollout.get("goal"),
        "QA rollout query",
    )


def qa_query_sha256(query: object) -> str:
    """Hash the exact stripped query shown to the summarizer."""

    return hashlib.sha256(_required_text(query, "query").encode("utf-8")).hexdigest()


def qa_source_rollout_projection(rollout: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fields produced by summarization before source hashing."""

    if not isinstance(rollout, Mapping):
        raise QASummaryError("QA rollout must be a mapping")
    return {
        str(key): value
        for key, value in rollout.items()
        if str(key) not in _SUMMARY_AUGMENTED_FIELDS
    }


def qa_rollout_sha256(rollout: Mapping[str, Any]) -> str:
    """Bind a skill to the complete pre-summary rollout, including EM evidence."""

    return hashlib.sha256(
        _canonical_json_bytes(qa_source_rollout_projection(rollout))
    ).hexdigest()


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QASummaryContractError(
                "duplicate_json_key", "QA skill JSON contains a duplicate key"
            )
        result[key] = value
    return result


def _normalized_answer_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = "".join(character if character.isalnum() else " " for character in text)
    return " ".join(text.split())


def _answer_values(rollout: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect forbidden answers for validation without rendering them."""

    values: list[object] = []
    golden = rollout.get("golden_answers", rollout.get("answers"))
    if isinstance(golden, str):
        values.append(golden)
    elif isinstance(golden, Sequence) and not isinstance(
        golden, (str, bytes, bytearray)
    ):
        values.extend(golden)

    for key in ("final_answer", "prediction"):
        if rollout.get(key) is not None:
            values.append(rollout.get(key))
    final_info = rollout.get("final_info")
    if isinstance(final_info, Mapping):
        for key in ("final_answer", "prediction"):
            if final_info.get(key) is not None:
                values.append(final_info.get(key))

    # Some collectors keep the submitted answer only inside the terminal action.
    steps = rollout.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes, bytearray)):
        for raw_step in reversed(steps):
            if not isinstance(raw_step, Mapping):
                continue
            action = str(raw_step.get("action") or "")
            match = re.fullmatch(
                r"\s*<answer>\s*(.*?)\s*</answer>\s*",
                action,
                flags=re.IGNORECASE | re.DOTALL,
            ) or re.fullmatch(
                r"\s*answer\[\s*(.*?)\s*\]\s*",
                action,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                values.append(match.group(1))
                break

    normalized = {
        _normalized_answer_text(value)
        for value in values
        if _normalized_answer_text(value)
    }
    return tuple(sorted(normalized))


def _contains_answer(text: str, normalized_answer: str) -> bool:
    normalized_text = _normalized_answer_text(text)
    if not normalized_text or not normalized_answer:
        return False
    # Word-sequence matching avoids rejecting an answer such as "no" merely
    # because it occurs inside "not". Scripts commonly written without spaces
    # need a direct substring check instead.
    if any(ord(character) > 127 for character in normalized_answer):
        return normalized_answer in normalized_text
    return f" {normalized_answer} " in f" {normalized_text} "


def _leaks_answer(text: str, answer: str) -> bool:
    if not _contains_answer(text, answer):
        return False
    if answer not in _FUNCTION_WORD_ANSWERS:
        return True
    normalized = _normalized_answer_text(text)
    if normalized == answer:
        return True
    escaped = re.escape(answer)
    if answer in {"yes", "no"} and re.search(
        rf"\b(?:answer|response|result) (?:is|should be|must be) {escaped}\b",
        normalized,
    ):
        return True
    return bool(
        re.search(rf"(?:answer\s*\[\s*{escaped}\s*\]|<answer>\s*{escaped}\s*</answer>|[\"'`]\s*{escaped}\s*[\"'`])", text, re.I)
        or re.search(rf"\b(?:answer|response|result)\s*(?::|(?:is|should be|must be)\s+)\s*{escaped}\s*(?:[.!;,]|$)", text, re.I)
        or re.search(rf"\b(?:answer|respond|return|output|submit|choose|select)(?:\s+with)?\s+{escaped}\s*(?:[.!;,]|$)", text, re.I)
    )


def _redacted_output(text: str, rollout: Mapping[str, Any]) -> str:
    """Audit final output only; mask answers without exposing hidden labels."""
    redacted = unicodedata.normalize("NFKC", text)
    redacted = re.sub(
        r"<think>.*?(?:</think>|$)|\[think\].*?(?:\[/think\]|$)",
        "[OMITTED_REASONING]", redacted, flags=re.I | re.S,
    )
    for answer in sorted(_answer_values(rollout), key=len, reverse=True):
        if answer in _FUNCTION_WORD_ANSWERS:
            continue
        pattern = r"[\W_]*".join(re.escape(word) for word in answer.split())
        if answer.isascii():
            pattern = rf"(?<![^\W_]){pattern}(?![^\W_])"
        redacted = re.sub(pattern, "[INSTANCE_ANSWER]", redacted, flags=re.I)
    return redacted[:16000]


def parse_qa_skill_json(
    value: object,
    *,
    rollout: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Parse one exact JSON object and reject gold/final-answer leakage."""

    if isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raw = str(value or "").strip()
        if not raw:
            raise QASummaryContractError("empty_output", "QA skill output is empty")
        try:
            parsed = json.loads(raw, object_pairs_hook=_duplicate_rejecting_object)
        except QASummaryContractError:
            raise
        except Exception as exc:
            raise QASummaryContractError(
                "invalid_json", "QA skill output must be exactly one JSON object"
            ) from exc

    if not isinstance(parsed, dict):
        raise QASummaryContractError(
            "not_json_object", "QA skill output must be a JSON object"
        )
    actual = {str(key) for key in parsed}
    expected = set(QA_SKILL_FIELDS)
    if actual != expected:
        raise QASummaryContractError(
            "skill_keys_mismatch",
            "QA skill JSON must contain exactly the three executable-skill keys",
        )

    skill: dict[str, str] = {}
    for field in QA_SKILL_FIELDS:
        value = parsed[field]
        if not isinstance(value, str):
            raise QASummaryContractError(
                "skill_field_not_string", "Every QA skill field must be a string"
            )
        if not value or value != value.strip():
            raise QASummaryContractError(
                "skill_field_empty_or_untrimmed",
                "Every QA skill field must be non-empty and trimmed",
            )
        if len(value) > QA_SKILL_FIELD_MAX_CHARS:
            raise QASummaryContractError(
                "skill_field_too_long", "A QA skill field exceeds its length limit"
            )
        if re.search(r"```|</?think>|\[/?think\]", value, flags=re.IGNORECASE):
            raise QASummaryContractError(
                "forbidden_wrapper", "QA skill fields contain a forbidden wrapper"
            )
        skill[field] = value

    if rollout is not None:
        answers = _answer_values(rollout)
        leaked_fields = tuple(
            field for field, text in skill.items()
            if any(_leaks_answer(text, answer) for answer in answers)
        )
        if leaked_fields:
            # Never include the rejected answer in this exception: retry prompts
            # use only the error code, keeping gold data out of model messages.
            raise QASummaryContractError(
                "answer_leak", "QA skill contains forbidden instance-answer text",
                fields=leaked_fields,
            )
    return skill


def qa_skill_sha256(skill: object) -> str:
    """Hash the strict three-field skill in canonical field order."""

    parsed = parse_qa_skill_json(skill)
    ordered = {field: parsed[field] for field in QA_SKILL_FIELDS}
    return hashlib.sha256(_canonical_json_bytes(ordered)).hexdigest()


def _trajectory_text(
    steps: object,
    *,
    final_observation: object,
    success: bool,
    max_steps: int,
) -> str:
    """Render only observation/action pairs, terminal observation and outcome."""

    if isinstance(steps, (str, bytes, bytearray)) or not isinstance(steps, Sequence):
        raise QASummaryError("QA rollout steps must be a sequence")
    rows = list(steps)
    if not rows:
        raise QASummaryError("QA rollout steps must not be empty")
    if max_steps > 0 and len(rows) > max_steps:
        raise QASummaryError(
            "QA trajectory exceeds summarizer_max_trajectory_steps; silent "
            f"truncation is forbidden: steps={len(rows)} limit={max_steps}"
        )

    lines: list[str] = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise QASummaryError(f"QA trajectory step {position} must be a mapping")
        observation = str(row.get("observation") or "").strip()
        action = str(row.get("action") or "").strip()
        if not action:
            raise QASummaryError(
                f"QA trajectory step {position} requires a non-empty action"
            )
        lines.append(f"Observation: {observation}")
        lines.append(f"Action: {action}")

    terminal = str(final_observation or "").strip()
    if not terminal:
        terminal = str(rows[-1].get("next_observation") or "").strip()
    lines.append(f"Observation: {terminal}")
    lines.append(f"Outcome: {'success' if success else 'failure'}")
    return "\n".join(lines)


def _render_current_skill_block(rollout: Mapping[str, Any]) -> str:
    raw_skill = rollout.get("skill")
    if raw_skill in (None, ""):
        raw_skill = rollout.get("skill_used")
    if raw_skill in (None, ""):
        return ""
    if isinstance(raw_skill, Mapping):
        text = json.dumps(
            dict(raw_skill),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    else:
        text = str(raw_skill).strip()
    if not text:
        return ""
    return f"\nSkill used in the failed attempt:\n{text}\n"


def render_qa_summary_prompt(
    rollout: Mapping[str, Any],
    *,
    max_steps: int = 0,
) -> str:
    """Render the single QA-only user message for one source sample."""

    if not isinstance(rollout, Mapping):
        raise QASummaryError("QA rollout must be a mapping")
    success = rollout.get("success")
    if not isinstance(success, bool):
        raise QASummaryError("QA rollout.success must be boolean")
    query = qa_query_text(rollout)
    trajectory = _trajectory_text(
        rollout.get("steps"),
        final_observation=rollout.get("final_observation"),
        success=success,
        max_steps=max_steps,
    )
    if success:
        return (
            QA_SUCCESS_PROMPT.replace("{QUERY}", query)
            .replace("{TRAJECTORY}", trajectory)
            .replace("{OUTPUT_CONTRACT}", QA_OUTPUT_CONTRACT)
        )
    failure_reason = str(
        rollout.get("failure_reason")
        or rollout.get("termination_reason")
        or "ended_without_exact_match"
    ).strip()
    return (
        QA_FAILURE_PROMPT.replace("{QUERY}", query)
        .replace("{CURRENT_SKILL_BLOCK}", _render_current_skill_block(rollout))
        .replace("{TRAJECTORY}", trajectory)
        .replace("{FAILURE_REASON}", failure_reason)
        .replace("{OUTPUT_CONTRACT}", QA_OUTPUT_CONTRACT)
    )


def qa_summary_prompt_contract() -> dict[str, Any]:
    """Return the canonical contract used for checkpoint invalidation."""

    contract: dict[str, Any] = {
        "schema_version": QA_SUMMARY_PROMPT_SCHEMA_VERSION,
        "implementation_version": QA_SUMMARIZER_IMPLEMENTATION_VERSION,
        "message_roles": ["user"],
        "success_prompt": QA_SUCCESS_PROMPT,
        "failure_prompt": QA_FAILURE_PROMPT,
        "output_contract": QA_OUTPUT_CONTRACT,
        "output_fields": list(QA_SKILL_FIELDS),
        "trajectory_projection": _PROMPT_TRAJECTORY_PROJECTION,
        "answer_leak_policy": "normalized_answer_exclusion_function_word_context_v2",
        "function_word_answers": sorted(_FUNCTION_WORD_ANSWERS),
        "retry_correction": QA_RETRY_CORRECTION,
        "pending_correction": QA_PENDING_CORRECTION,
        "compatible_result_prompt_contract_sha256": [QA_LEGACY_PROMPT_CONTRACT_SHA256],
    }
    contract["prompt_contract_sha256"] = hashlib.sha256(
        _canonical_json_bytes(contract)
    ).hexdigest()
    return contract


QA_SUMMARY_PROMPT_CONTRACT_SHA256 = str(
    qa_summary_prompt_contract()["prompt_contract_sha256"]
)


def qa_accepted_summary_prompt_hashes() -> tuple[str, ...]:
    """Legacy results retain provenance and must pass the current parser."""
    return (QA_SUMMARY_PROMPT_CONTRACT_SHA256, QA_LEGACY_PROMPT_CONTRACT_SHA256)


def qa_capped_complete_json_recovery(
    text: object,
    rollout: Mapping[str, Any],
) -> tuple[bool, str]:
    """Accept a length-capped generation only if its final JSON is complete/safe."""

    try:
        parse_qa_skill_json(text, rollout=rollout)
    except QASummaryContractError:
        return False, "cap_incomplete_or_invalid_json"
    return True, "cap_recovered_complete_json"


@dataclass(frozen=True)
class QASummaryResult:
    """One source sample's accepted skill or explicit terminal failure."""

    task_id: str
    rollout_id: str
    status: str
    attempts: int
    query_sha256: str
    source_rollout_sha256: str
    prompt_contract_sha256: str
    skill: dict[str, str] | None
    skill_sha256: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.skill is not None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSONL-safe record that never includes gold answers."""

        return {
            "schema_version": QA_SUMMARY_RESULT_SCHEMA_VERSION,
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "status": self.status,
            "attempts": self.attempts,
            "query_sha256": self.query_sha256,
            "source_rollout_sha256": self.source_rollout_sha256,
            "prompt_contract_sha256": self.prompt_contract_sha256,
            "skill": dict(self.skill) if self.skill is not None else None,
            "skill_sha256": self.skill_sha256,
            "error": self.error,
        }

    as_mapping = to_dict


def _result_identity(rollout: Mapping[str, Any]) -> tuple[str, str, str, str]:
    task_id = _required_text(
        rollout.get("task_id") or rollout.get("id"), "QA rollout task_id"
    )
    source_sha256 = qa_rollout_sha256(rollout)
    rollout_id = str(rollout.get("rollout_id") or "").strip()
    if not rollout_id:
        rollout_id = f"qa-rollout-{source_sha256[:24]}"
    query_sha = qa_query_sha256(qa_query_text(rollout))
    return task_id, rollout_id, query_sha, source_sha256


def _successful_result(
    rollout: Mapping[str, Any], skill: dict[str, str], *, attempts: int
) -> QASummaryResult:
    task_id, rollout_id, query_sha, source_sha = _result_identity(rollout)
    return QASummaryResult(
        task_id=task_id,
        rollout_id=rollout_id,
        status="ok",
        attempts=int(attempts),
        query_sha256=query_sha,
        source_rollout_sha256=source_sha,
        prompt_contract_sha256=QA_SUMMARY_PROMPT_CONTRACT_SHA256,
        skill=dict(skill),
        skill_sha256=qa_skill_sha256(skill),
        error=None,
    )


def _failed_result(
    rollout: Mapping[str, Any], *, attempts: int, error: str
) -> QASummaryResult:
    task_id, rollout_id, query_sha, source_sha = _result_identity(rollout)
    return QASummaryResult(
        task_id=task_id,
        rollout_id=rollout_id,
        status="contract_exhausted",
        attempts=int(attempts),
        query_sha256=query_sha,
        source_rollout_sha256=source_sha,
        prompt_contract_sha256=QA_SUMMARY_PROMPT_CONTRACT_SHA256,
        skill=None,
        skill_sha256=None,
        error=str(error),
    )


def require_complete_summaries(
    results: Sequence[QASummaryResult],
    *,
    expected_count: int | None = None,
) -> list[QASummaryResult]:
    """Fail closed unless every expected sample has one unique accepted skill."""

    rows = list(results)
    if expected_count is not None and len(rows) != int(expected_count):
        raise IncompleteQASummaryError(
            "QA summary result count mismatch: "
            f"expected={int(expected_count)} actual={len(rows)}"
        )
    task_ids = [result.task_id for result in rows]
    if len(set(task_ids)) != len(task_ids):
        raise IncompleteQASummaryError("QA summary results contain duplicate task ids")
    failures = [result.task_id for result in rows if not result.ok]
    if failures:
        raise IncompleteQASummaryError(
            "QA summary stage is incomplete: "
            f"failed={len(failures)} task_ids={failures[:16]}"
        )
    return rows


def _error_code(error: Exception) -> str:
    if isinstance(error, QASummaryContractError):
        return error.code
    if isinstance(error, _RecoverableSummaryOutputError):
        return str(getattr(error, "status", "recoverable_generation_error"))
    return type(error).__name__


class QASkillSummarizer(SkillSummarizer):
    """QA adapter that reuses model loading but no ALFWorld summary semantics."""

    implementation_version = QA_SUMMARIZER_IMPLEMENTATION_VERSION

    def _raw_prompt(self, rollout: dict) -> str:
        prompt = render_qa_summary_prompt(rollout, max_steps=self.max_steps)
        repair_round = int(getattr(self, "pending_retry_round", 0))
        if repair_round:
            prompt += "\n\n" + QA_PENDING_CORRECTION.replace("{ROUND}", str(repair_round))
        return prompt

    def _prompt(self, rollout: dict) -> str:
        # The QA contract has exactly one user message and no system message.
        return self._render_chat(
            [{"role": "user", "content": self._raw_prompt(rollout)}]
        )

    def _contract_retry_prompt(
        self,
        rollout: dict,
        *,
        rejected_output: str,
        validation_error: Exception,
        next_attempt: int,
    ) -> str:
        del rejected_output
        # Do not replay rejected text: it can contain a leaked gold/final answer.
        correction = (QA_RETRY_CORRECTION
            .replace("{ATTEMPT}", str(next_attempt))
            .replace("{MAX_ATTEMPTS}", str(self.contract_max_attempts))
            .replace("{ERROR}", _error_code(validation_error))
            .replace("{FIELDS}", ", ".join(getattr(validation_error, "fields", ())) or "entire object")
        )
        return self._render_chat(
            [
                {
                    "role": "user",
                    "content": f"{self._raw_prompt(rollout)}\n\n{correction}",
                }
            ]
        )

    @staticmethod
    def _parse(text: str, rollout: dict) -> dict[str, str]:
        return parse_qa_skill_json(text, rollout=rollout)

    def drain_attempt_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = list(getattr(self, "_attempt_diagnostics", ()))
        self._attempt_diagnostics = []
        return diagnostics

    def _record_attempt(self, rollout: dict, generated: object, *, attempt: int,
                        error: Exception | None) -> None:
        if not hasattr(self, "_attempt_diagnostics"):
            self._attempt_diagnostics = []
        output = generated if isinstance(generated, str) else ""
        self._attempt_diagnostics.append({
            "task_id": str(rollout.get("task_id") or rollout.get("id") or ""),
            "attempt": attempt,
            "pending_retry_round": int(getattr(self, "pending_retry_round", 0)),
            "status": "rejected" if error else "ok",
            "error": _error_code(error) if error else None,
            "fields": list(getattr(error, "fields", ())),
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "rejected_output_redacted": _redacted_output(output, rollout) if error else None,
        })

    def _generate_texts_vllm(
        self,
        prompts: list[str],
        rollouts: list[dict],
        *,
        label: str,
    ) -> list[str | _RecoverableSummaryOutputError]:
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        prompt_token_ids = [
            [int(token) for token in token_ids] for token_ids in encoded["input_ids"]
        ]
        prompt_limit = min(
            self.max_input_tokens,
            max(1, self.context_window_tokens - self.max_new_tokens),
        )
        oversized = [
            (index, len(token_ids))
            for index, token_ids in enumerate(prompt_token_ids)
            if len(token_ids) > prompt_limit
        ]
        if oversized:
            raise RuntimeError(
                f"{label} QA prompt exceeds the non-truncating token budget: "
                f"limit={prompt_limit} rows={oversized[:8]}"
            )

        outputs = self.llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_token_ids],
            self._vllm_sampling_params(),
            use_tqdm=False,
        )
        if len(outputs) != len(rollouts):
            raise RuntimeError(
                f"{label} vLLM output/rollout count mismatch: "
                f"outputs={len(outputs)} rollouts={len(rollouts)}"
            )

        results: list[str | _RecoverableSummaryOutputError] = []
        for request_output, rollout in zip(outputs, rollouts):
            candidates = list(getattr(request_output, "outputs", ()) or ())
            if len(candidates) != 1:
                raise RuntimeError(
                    f"{label} requires exactly one vLLM continuation per QA prompt"
                )
            candidate = candidates[0]
            token_ids = [int(token) for token in candidate.token_ids]
            try:
                final_ids = _final_answer_token_ids(
                    token_ids,
                    thinking_enabled=self.enable_thinking,
                    thinking_end_token_id=self.thinking_end_token_id,
                )
                text = self.tokenizer.decode(final_ids, skip_special_tokens=True)
            except _RecoverableSummaryOutputError as exc:
                results.append(exc)
                continue
            if str(getattr(candidate, "finish_reason", "")) == "length":
                recovered, status = qa_capped_complete_json_recovery(text, rollout)
                self.generation_outcomes[
                    str(rollout.get("rollout_id") or rollout.get("task_id") or "")
                ] = status
                if not recovered:
                    results.append(
                        _RecoverableSummaryOutputError(status, status=status)
                    )
                    continue
            results.append(text)
        return results

    def _generate_texts(
        self,
        prompts: list[str],
        rollouts: list[dict],
        *,
        device,
        label: str,
    ) -> list[str | _RecoverableSummaryOutputError]:
        if len(prompts) != len(rollouts):
            raise RuntimeError(
                f"{label} QA prompt/rollout count mismatch: "
                f"prompts={len(prompts)} rollouts={len(rollouts)}"
            )
        if not hasattr(self, "generation_outcomes"):
            self.generation_outcomes = {}
        if getattr(self, "backend", "transformers") == "vllm":
            return self._generate_texts_vllm(prompts, rollouts, label=label)

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=not self.fail_on_prompt_truncation,
            max_length=(
                self.max_input_tokens if not self.fail_on_prompt_truncation else None
            ),
        )
        if self.fail_on_prompt_truncation:
            _check_no_prompt_truncation(
                inputs["attention_mask"],
                configured_input_limit=self.max_input_tokens,
                generation_tokens=self.max_new_tokens,
                context_window=self.context_window_tokens,
                label=label,
            )
        inputs = inputs.to(device)
        generated = self.llm.generate(**inputs, **self._generation_kwargs())
        input_length = inputs["input_ids"].shape[1]
        generated_ids = generated[:, input_length:]
        cutoff_reasons = _generation_cutoff_failure_reasons(
            generated_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_ids=_eos_token_ids(self.llm, self.tokenizer),
            rollouts=rollouts,
        )

        results: list[str | _RecoverableSummaryOutputError] = []
        for ids, cutoff_reason, rollout in zip(generated_ids, cutoff_reasons, rollouts):
            token_ids = [int(token) for token in ids.detach().cpu().tolist()]
            try:
                final_ids = _final_answer_token_ids(
                    token_ids,
                    thinking_enabled=self.enable_thinking,
                    thinking_end_token_id=self.thinking_end_token_id,
                )
                text = self.tokenizer.decode(final_ids, skip_special_tokens=True)
            except _RecoverableSummaryOutputError as exc:
                results.append(exc)
                continue
            if cutoff_reason is not None:
                recovered, status = qa_capped_complete_json_recovery(text, rollout)
                self.generation_outcomes[
                    str(rollout.get("rollout_id") or rollout.get("task_id") or "")
                ] = status
                if not recovered:
                    results.append(
                        _RecoverableSummaryOutputError(status, status=status)
                    )
                    continue
            results.append(text)
        return results

    def _parse_with_contract_retries(
        self,
        initial_text: str | _RecoverableSummaryOutputError,
        rollout: dict,
        *,
        device,
    ) -> QASummaryResult:
        return self._parse_batch_with_contract_retries(
            [initial_text], [rollout], device=device
        )[0]

    def _parse_batch_with_contract_retries(
        self,
        initial_texts: list[str | _RecoverableSummaryOutputError],
        rollouts: list[dict],
        *,
        device,
    ) -> list[QASummaryResult]:
        if len(initial_texts) != len(rollouts):
            raise RuntimeError(
                "QA summary parse input/rollout count mismatch: "
                f"texts={len(initial_texts)} rollouts={len(rollouts)}"
            )
        if not rollouts:
            return []

        parsed: list[QASummaryResult | None] = [None] * len(rollouts)
        pending = list(enumerate(initial_texts))
        for attempt in range(1, self.contract_max_attempts + 1):
            retry_indexes: list[int] = []
            retry_prompts: list[str] = []
            retry_rollouts: list[dict] = []
            for index, generated in pending:
                rollout = rollouts[index]
                try:
                    if isinstance(generated, _RecoverableSummaryOutputError):
                        raise generated
                    skill = self._parse(str(generated or ""), rollout)
                    parsed[index] = _successful_result(rollout, skill, attempts=attempt)
                    self._record_attempt(rollout, generated, attempt=attempt, error=None)
                except (QASummaryContractError, _RecoverableSummaryOutputError) as exc:
                    code = _error_code(exc)
                    self._record_attempt(rollout, generated, attempt=attempt, error=exc)
                    if attempt >= self.contract_max_attempts:
                        parsed[index] = _failed_result(
                            rollout, attempts=attempt, error=code
                        )
                        self.generation_outcomes[
                            str(
                                rollout.get("rollout_id")
                                or rollout.get("task_id")
                                or ""
                            )
                        ] = "contract_exhausted"
                        continue
                    retry_indexes.append(index)
                    retry_rollouts.append(rollout)
                    retry_prompts.append(
                        self._contract_retry_prompt(
                            rollout,
                            rejected_output=(
                                str(generated or "")
                                if not isinstance(
                                    generated, _RecoverableSummaryOutputError
                                )
                                else ""
                            ),
                            validation_error=exc,
                            next_attempt=attempt + 1,
                        )
                    )

            if not retry_indexes:
                break
            next_texts = self._generate_texts(
                retry_prompts,
                retry_rollouts,
                device=device,
                label=f"qa summarizer contract retry {attempt + 1}",
            )
            if len(next_texts) != len(retry_indexes):
                raise RuntimeError(
                    "QA summarizer retry output count mismatch: "
                    f"outputs={len(next_texts)} rollouts={len(retry_indexes)}"
                )
            pending = list(zip(retry_indexes, next_texts))

        if any(result is None for result in parsed):
            raise AssertionError("QA summarizer left a sample without a result")
        return [result for result in parsed if result is not None]

    @torch.inference_mode()
    def summarize_batch(
        self,
        rollouts: list[dict],
        progress_callback=None,
    ) -> list[QASummaryResult]:
        if not rollouts:
            return []
        self.generation_outcomes = {}
        prompts = [self._prompt(rollout) for rollout in rollouts]
        device = (
            None
            if getattr(self, "backend", "transformers") == "vllm"
            else next(self.llm.parameters()).device
        )
        results: list[QASummaryResult] = []
        total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        progress_every = max(
            1,
            int(self.cfg.get("logging", {}).get("progress_every_batches", 5)),
        )
        for batch_index, start in enumerate(
            range(0, len(prompts), self.batch_size), start=1
        ):
            batch_prompts = prompts[start : start + self.batch_size]
            batch_rollouts = rollouts[start : start + self.batch_size]
            texts = self._generate_texts(
                batch_prompts,
                batch_rollouts,
                device=device,
                label="qa summarizer",
            )
            results.extend(
                self._parse_batch_with_contract_retries(
                    texts, batch_rollouts, device=device
                )
            )
            if progress_callback is not None and (
                batch_index == 1
                or batch_index % progress_every == 0
                or batch_index == total_batches
            ):
                progress_callback(
                    min(start + self.batch_size, len(prompts)),
                    len(prompts),
                    batch_index,
                    total_batches,
                )
        return results


__all__ = [
    "IncompleteQASummaryError",
    "QA_FAILURE_PROMPT",
    "QA_OUTPUT_CONTRACT",
    "QA_SKILL_FIELDS",
    "QA_SUCCESS_PROMPT",
    "QA_SUMMARIZER_IMPLEMENTATION_VERSION",
    "QA_SUMMARY_PROMPT_CONTRACT_SHA256",
    "QA_SUMMARY_PROMPT_SCHEMA_VERSION",
    "QA_SUMMARY_RESULT_SCHEMA_VERSION",
    "QASkillSummarizer",
    "QASummaryContractError",
    "QASummaryError",
    "QASummaryResult",
    "parse_qa_skill_json",
    "qa_accepted_summary_prompt_hashes",
    "qa_capped_complete_json_recovery",
    "qa_query_sha256",
    "qa_query_text",
    "qa_rollout_sha256",
    "qa_skill_sha256",
    "qa_source_rollout_projection",
    "qa_summary_prompt_contract",
    "render_qa_summary_prompt",
    "require_complete_summaries",
]
