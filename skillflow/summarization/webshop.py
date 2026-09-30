# ruff: noqa: E501

"""Strict one-baseline-sample-to-one-skill summarization for WebShop.

Only model construction and low-level generation helpers are inherited from the
shared summarizer. Prompt rendering, projection, parsing, capped-generation
recovery, retries, and result binding are all WebShop-specific, so no ALFWorld
trajectory semantics can enter this branch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any
import unicodedata

import torch

from skillflow.summarization.base import (
    SkillSummarizer,
    _RecoverableSummaryOutputError,
    _check_no_prompt_truncation,
    _eos_token_ids,
    _final_answer_token_ids,
    _generation_cutoff_failure_reasons,
)
from skillflow.summarization.webshop_prompts import (
    DEFAULT_MAX_ACTION_CHARS,
    DEFAULT_MAX_ADMISSIBLE_ACTIONS,
    DEFAULT_MAX_ADMISSIBLE_ACTION_CHARS,
    DEFAULT_MAX_GOAL_CHARS,
    DEFAULT_MAX_OBSERVATION_CHARS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_TRAJECTORY_STEPS,
    WEBSHOP_SKILL_FIELDS,
    WEBSHOP_SUMMARIZER_IMPLEMENTATION_VERSION,
    prompt_contract_sha256,
    render_webshop_summary_prompt,
    webshop_goal_sha256,
    webshop_goal_text,
)
from skillflow.summarization.webshop_token_budget import fit_webshop_summary_prompt


WEBSHOP_SUMMARY_RESULT_SCHEMA_VERSION = "webshop_sample_skill_summary_result_v1"
WEBSHOP_SKILL_FIELD_MAX_CHARS = 1600
_SUMMARY_AUGMENTED_FIELDS = frozenset(
    {
        "distilled_skill",
        "webshop_summary",
        "webshop_summary_result",
        "summarized_skill",
        "summary_result",
    }
)
_ASIN_RE = re.compile(r"(?i)\bB0[A-Z0-9]{8}\b")
_IDENTIFIER_LABEL_RE = re.compile(r"(?i)\b(?:asin|product[ _-]?id|item[ _-]?id)\b")
# The trajectory renderer deliberately replaces real catalog identifiers with
# this exact safe token. A recipe such as click[[PRODUCT_ID]] therefore contains
# no instance identifier. Ignore only this token for the label check; continue
# checking the unmodified output for actual ASINs and source identifiers.
_REDACTED_PRODUCT_ID_RE = re.compile(r"(?i)\[PRODUCT_ID\]")
_SESSION_IDENTIFIER_RE = re.compile(
    r"(?i)\bsession(?:\s+(?:id|number))?\s*[:=#-]?\s*\d+\b"
    r"|\bsession\s+(?:id|number)\b"
)
_CLICK_CATALOG_ID_RE = re.compile(r"(?i)\bclick\[\s*([A-Z0-9][A-Z0-9_-]{7,31})\s*\]")
_CHAT_CONTROL_TOKEN_RE = re.compile(r"(?i)<\|[^<>\r\n]{1,64}\|>|\[/?INST\]|<</?SYS>>")
_DISALLOWED_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


_SOURCE_TEXT_TOKEN_RE = re.compile(r"[^\W_]+(?:\x27[^\W_]+)?", flags=re.UNICODE)
# A seven-token exact overlap is already highly specific. The five-token rule
# catches compact product titles made from long words, but only when the
# normalized fragment is correspondingly long. Both rules also require at
# least three non-generic tokens below.
_SOURCE_TEXT_WINDOW_RULES = ((7, 0), (5, 48))
_SOURCE_TEXT_MIN_DISTINCTIVE_TOKENS = 3
_GENERIC_WEBSHOP_TOKENS = frozenset(
    """
    a an and are as at back be before button buy by candidate candidates check
    click compare constraint constraints cost description each every feature
    features find for from goal has have if in into is item items less limit
    listing listings mandatory more next of on open option options or page
    previous price prices product products purchase quantity required result
    results review reviews search select seller shopping than that the then
    these this those through to use verify when with without
    """.split()
)


class WebShopSummaryError(ValueError):
    """Base error for invalid WebShop summary inputs or outputs."""


class WebShopSummaryContractError(WebShopSummaryError):
    """A generated value is not a safe exact three-string WebShop skill."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


class IncompleteWebShopSummaryError(WebShopSummaryError):
    """At least one source sample has no unique accepted skill."""

    def __init__(self, message: str, *, results: Sequence = ()):
        super().__init__(message)
        # Keep rejected and accepted batch members available to the caller for
        # diagnostics. They are not committed as training skills by this error.
        self.results = tuple(results)


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
        raise WebShopSummaryError(f"Value is not finite canonical JSON: {exc}") from exc


def _required_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise WebShopSummaryError(f"{label} must be non-empty")
    return text


def _sample_id(rollout: Mapping[str, Any]) -> int | str:
    value = rollout.get("sample_id", 0)
    if isinstance(value, bool):
        raise WebShopSummaryError("WebShop rollout.sample_id must not be boolean")
    if isinstance(value, int):
        if value < 0:
            raise WebShopSummaryError("WebShop rollout.sample_id must be non-negative")
        return value
    text = str(value or "").strip()
    if not text:
        raise WebShopSummaryError("WebShop rollout.sample_id must be non-empty")
    return text


def webshop_source_rollout_projection(
    rollout: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only summary-produced fields before source-rollout hashing."""

    if not isinstance(rollout, Mapping):
        raise WebShopSummaryError("WebShop rollout must be a mapping")
    return {
        str(key): value
        for key, value in rollout.items()
        if str(key) not in _SUMMARY_AUGMENTED_FIELDS
    }


def webshop_rollout_sha256(rollout: Mapping[str, Any]) -> str:
    """Bind a skill to the complete pre-summary baseline rollout."""

    return hashlib.sha256(
        _canonical_json_bytes(webshop_source_rollout_projection(rollout))
    ).hexdigest()


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WebShopSummaryContractError(
                "duplicate_json_key",
                "WebShop skill JSON contains a duplicate key",
            )
        result[key] = value
    return result


def _walk_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for nested in value.values():
            result.extend(_walk_strings(nested))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for nested in value:
            result.extend(_walk_strings(nested))
        return result
    return []


def _source_catalog_identifiers(rollout: Mapping[str, Any]) -> tuple[str, ...]:
    identifiers: set[str] = set()
    for text in _walk_strings(rollout.get("steps") or []):
        identifiers.update(match.casefold() for match in _ASIN_RE.findall(text))
        identifiers.update(
            match.casefold()
            for match in _CLICK_CATALOG_ID_RE.findall(text)
            # WebShop uses ordinary UI labels such as click[Description]. A
            # catalog token must contain a digit; otherwise common procedural
            # words would be falsely treated as instance identifiers.
            if any(character.isdigit() for character in match)
        )
    # These complete identifiers must never be copied even if they do not look
    # like an ASIN. Bare session numbers are intentionally not forbidden: they
    # can also be legitimate quantities or prices. The identifier label itself
    # is rejected separately.
    for key in ("task_id", "rollout_id"):
        text = str(rollout.get(key) or "").strip()
        if len(text) >= 5:
            identifiers.add(text.casefold())
    return tuple(sorted(identifiers))


def _normalized_source_tokens(value: object) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = normalized.casefold().replace("’", "\x27")
    return tuple(_SOURCE_TEXT_TOKEN_RE.findall(normalized))


def _source_goal_observation_texts(
    rollout: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return only source text that the WebShop summary prompt can expose."""

    texts: list[str] = []
    goal = rollout.get("canonical_goal") or rollout.get("goal")
    if goal is not None:
        texts.append(str(goal))
    raw_steps = rollout.get("steps")
    if isinstance(raw_steps, Sequence) and not isinstance(
        raw_steps, (str, bytes, bytearray)
    ):
        for raw_step in raw_steps:
            if isinstance(raw_step, Mapping):
                texts.append(str(raw_step.get("observation") or ""))
    texts.append(str(rollout.get("final_observation") or ""))
    return tuple(text for text in texts if text)


def _qualifying_source_windows(
    tokens: tuple[str, ...],
) -> dict[int, set[tuple[str, ...]]]:
    result: dict[int, set[tuple[str, ...]]] = {}
    for size, min_chars in _SOURCE_TEXT_WINDOW_RULES:
        windows: set[tuple[str, ...]] = set()
        for start in range(max(0, len(tokens) - size + 1)):
            window = tokens[start : start + size]
            if min_chars and sum(map(len, window)) + size - 1 < min_chars:
                continue
            distinctive = {
                token for token in window if token not in _GENERIC_WEBSHOP_TOKENS
            }
            if len(distinctive) >= _SOURCE_TEXT_MIN_DISTINCTIVE_TOKENS:
                windows.add(window)
        if windows:
            result[size] = windows
    return result


def _contains_source_phrase_leak(
    skill: Mapping[str, str],
    rollout: Mapping[str, Any],
) -> bool:
    """Detect normalized verbatim title/copy spans without matching short UI text."""

    skill_windows: dict[int, set[tuple[str, ...]]] = {}
    for field in WEBSHOP_SKILL_FIELDS:
        for size, windows in _qualifying_source_windows(
            _normalized_source_tokens(skill[field])
        ).items():
            skill_windows.setdefault(size, set()).update(windows)
    if not skill_windows:
        return False

    for source_text in _source_goal_observation_texts(rollout):
        source_tokens = _normalized_source_tokens(source_text)
        for size, candidates in skill_windows.items():
            for start in range(max(0, len(source_tokens) - size + 1)):
                if source_tokens[start : start + size] in candidates:
                    return True
    return False


def parse_webshop_skill_json(
    value: object,
    *,
    rollout: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Parse one exact JSON object and reject source-instance leakage."""

    if isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raw = str(value or "").strip()
        if not raw:
            raise WebShopSummaryContractError(
                "empty_output", "WebShop skill output is empty"
            )
        try:
            parsed = json.loads(raw, object_pairs_hook=_duplicate_rejecting_object)
        except WebShopSummaryContractError:
            raise
        except Exception as exc:
            raise WebShopSummaryContractError(
                "invalid_json",
                "WebShop skill output must be exactly one JSON object",
            ) from exc

    if not isinstance(parsed, dict):
        raise WebShopSummaryContractError(
            "not_json_object", "WebShop skill output must be a JSON object"
        )
    if set(parsed) != set(WEBSHOP_SKILL_FIELDS):
        raise WebShopSummaryContractError(
            "skill_keys_mismatch",
            "WebShop skill JSON must contain exactly the three executable-skill keys",
        )

    skill: dict[str, str] = {}
    for field in WEBSHOP_SKILL_FIELDS:
        field_value = parsed[field]
        if not isinstance(field_value, str):
            raise WebShopSummaryContractError(
                "skill_field_not_string",
                "Every WebShop skill field must be a string",
            )
        if not field_value or field_value != field_value.strip():
            raise WebShopSummaryContractError(
                "skill_field_empty_or_untrimmed",
                "Every WebShop skill field must be non-empty and trimmed",
            )
        if len(field_value) > WEBSHOP_SKILL_FIELD_MAX_CHARS:
            raise WebShopSummaryContractError(
                "skill_field_too_long",
                "A WebShop skill field exceeds its length limit",
            )
        if re.search(
            r"```|</?think>|\[/?think\]",
            field_value,
            flags=re.IGNORECASE,
        ):
            raise WebShopSummaryContractError(
                "forbidden_wrapper",
                "WebShop skill fields contain a forbidden wrapper",
            )
        if _CHAT_CONTROL_TOKEN_RE.search(field_value) or _DISALLOWED_CONTROL_RE.search(
            field_value
        ):
            raise WebShopSummaryContractError(
                "forbidden_chat_control",
                "WebShop skill fields contain a forbidden chat control token",
            )
        skill[field] = field_value

    rendered = "\n".join(skill.values())
    identifier_labels = _REDACTED_PRODUCT_ID_RE.sub("", rendered)
    if (
        _ASIN_RE.search(rendered)
        or _IDENTIFIER_LABEL_RE.search(identifier_labels)
        or _SESSION_IDENTIFIER_RE.search(rendered)
    ):
        raise WebShopSummaryContractError(
            "instance_identifier_leak",
            "WebShop skill contains a forbidden catalog/session identifier",
        )
    if rollout is not None:
        folded = rendered.casefold()
        if any(
            identifier in folded for identifier in _source_catalog_identifiers(rollout)
        ):
            raise WebShopSummaryContractError(
                "instance_identifier_leak",
                "WebShop skill copied a forbidden source identifier",
            )
        if _contains_source_phrase_leak(skill, rollout):
            raise WebShopSummaryContractError(
                "source_catalog_text_leak",
                "WebShop skill copied a forbidden source catalog phrase",
            )
    return skill


def webshop_skill_sha256(skill: object) -> str:
    """Hash the strict three-field skill using compact sorted canonical JSON."""

    parsed = parse_webshop_skill_json(skill)
    ordered = {field: parsed[field] for field in WEBSHOP_SKILL_FIELDS}
    return hashlib.sha256(_canonical_json_bytes(ordered)).hexdigest()


def webshop_capped_complete_json_recovery(
    text: object,
    rollout: Mapping[str, Any],
) -> tuple[bool, str]:
    """Accept a capped generation only when its final JSON is complete and safe."""

    try:
        parse_webshop_skill_json(text, rollout=rollout)
    except WebShopSummaryContractError:
        return False, "cap_incomplete_or_invalid_json"
    return True, "cap_recovered_complete_json"


@dataclass(frozen=True)
class WebShopSummaryResult:
    """One source sample's accepted skill or explicit terminal failure."""

    task_id: str
    sample_id: int | str
    rollout_id: str
    session: int
    task_type: str
    status: str
    attempts: int
    canonical_goal_sha256: str
    source_rollout_sha256: str
    prompt_contract_sha256: str
    skill: dict[str, str] | None
    skill_sha256: str | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.skill is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": WEBSHOP_SUMMARY_RESULT_SCHEMA_VERSION,
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "rollout_id": self.rollout_id,
            "session": self.session,
            "task_type": self.task_type,
            "status": self.status,
            "attempts": self.attempts,
            "canonical_goal_sha256": self.canonical_goal_sha256,
            "source_rollout_sha256": self.source_rollout_sha256,
            "prompt_contract_sha256": self.prompt_contract_sha256,
            "skill": dict(self.skill) if self.skill is not None else None,
            "skill_sha256": self.skill_sha256,
            "error": self.error,
        }

    as_mapping = to_dict


def _result_identity(
    rollout: Mapping[str, Any],
) -> tuple[str, int | str, str, int, str, str, str]:
    task_id = _required_text(
        rollout.get("task_id") or rollout.get("id"),
        "WebShop rollout task_id",
    )
    source_sha = webshop_rollout_sha256(rollout)
    rollout_id = str(rollout.get("rollout_id") or "").strip()
    if not rollout_id:
        rollout_id = f"webshop-baseline-{source_sha[:24]}"
    if "session" not in rollout or isinstance(rollout.get("session"), bool):
        raise WebShopSummaryError("WebShop rollout.session must be an integer")
    try:
        session = int(rollout["session"])
    except (TypeError, ValueError) as exc:
        raise WebShopSummaryError("WebShop rollout.session must be an integer") from exc
    if session < 0:
        raise WebShopSummaryError("WebShop rollout.session must be non-negative")
    task_type = _required_text(rollout.get("task_type"), "WebShop rollout task_type")
    goal_sha = webshop_goal_sha256(webshop_goal_text(rollout))
    return (
        task_id,
        _sample_id(rollout),
        rollout_id,
        session,
        task_type,
        goal_sha,
        source_sha,
    )


def _successful_result(
    rollout: Mapping[str, Any],
    skill: dict[str, str],
    *,
    attempts: int,
    prompt_sha256: str,
) -> WebShopSummaryResult:
    (
        task_id,
        sample_id,
        rollout_id,
        session,
        task_type,
        goal_sha,
        source_sha,
    ) = _result_identity(rollout)
    return WebShopSummaryResult(
        task_id=task_id,
        sample_id=sample_id,
        rollout_id=rollout_id,
        session=session,
        task_type=task_type,
        status="ok",
        attempts=int(attempts),
        canonical_goal_sha256=goal_sha,
        source_rollout_sha256=source_sha,
        prompt_contract_sha256=prompt_sha256,
        skill=dict(skill),
        skill_sha256=webshop_skill_sha256(skill),
        error=None,
    )


def _failed_result(
    rollout: Mapping[str, Any],
    *,
    attempts: int,
    error: str,
    prompt_sha256: str,
) -> WebShopSummaryResult:
    (
        task_id,
        sample_id,
        rollout_id,
        session,
        task_type,
        goal_sha,
        source_sha,
    ) = _result_identity(rollout)
    return WebShopSummaryResult(
        task_id=task_id,
        sample_id=sample_id,
        rollout_id=rollout_id,
        session=session,
        task_type=task_type,
        status="contract_exhausted",
        attempts=int(attempts),
        canonical_goal_sha256=goal_sha,
        source_rollout_sha256=source_sha,
        prompt_contract_sha256=prompt_sha256,
        skill=None,
        skill_sha256=None,
        error=str(error),
    )


def require_complete_summaries(
    results: Sequence[WebShopSummaryResult],
    *,
    expected_count: int | None = None,
) -> list[WebShopSummaryResult]:
    """Fail closed unless each task/sample pair has one accepted skill."""

    rows = list(results)
    if expected_count is not None and len(rows) != int(expected_count):
        raise IncompleteWebShopSummaryError(
            "WebShop summary result count mismatch: "
            f"expected={int(expected_count)} actual={len(rows)}"
        )
    identities = [(row.task_id, row.sample_id) for row in rows]
    if len(set(identities)) != len(identities):
        raise IncompleteWebShopSummaryError(
            "WebShop summary results contain duplicate task/sample identities"
        )
    failures = [identity for identity, row in zip(identities, rows) if not row.ok]
    if failures:
        details = [
            {
                "task_id": row.task_id,
                "sample_id": row.sample_id,
                "attempts": row.attempts,
                "error": row.error,
            }
            for row in rows
            if not row.ok
        ]
        raise IncompleteWebShopSummaryError(
            "WebShop summary stage is incomplete: "
            f"failed={len(failures)} identities={failures[:16]} "
            f"details={json.dumps(details[:16], ensure_ascii=True, sort_keys=True)}",
            results=rows,
        )
    return rows


def _error_code(error: Exception) -> str:
    if isinstance(error, WebShopSummaryContractError):
        return error.code
    if isinstance(error, _RecoverableSummaryOutputError):
        return str(getattr(error, "status", "recoverable_generation_error"))
    return type(error).__name__


class WebShopSkillSummarizer(SkillSummarizer):
    """WebShop adapter reusing model loading but no ALFWorld summary semantics."""

    implementation_version = WEBSHOP_SUMMARIZER_IMPLEMENTATION_VERSION

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        summary_cfg = cfg.get("summary", {})
        if not isinstance(summary_cfg, Mapping):
            raise ValueError("summary config must be a mapping")
        model_cfg = cfg.get("models", {})

        def configured(name: str, default: int) -> int:
            return int(
                summary_cfg.get(
                    name,
                    model_cfg.get(f"summarizer_{name}", default),
                )
            )

        self.max_goal_chars = configured("max_goal_chars", DEFAULT_MAX_GOAL_CHARS)
        self.max_observation_chars = configured(
            "max_observation_chars", DEFAULT_MAX_OBSERVATION_CHARS
        )
        self.max_action_chars = configured("max_action_chars", DEFAULT_MAX_ACTION_CHARS)
        self.max_admissible_actions = configured(
            "max_admissible_actions", DEFAULT_MAX_ADMISSIBLE_ACTIONS
        )
        self.max_admissible_action_chars = configured(
            "max_admissible_action_chars",
            DEFAULT_MAX_ADMISSIBLE_ACTION_CHARS,
        )
        # This is a summary-contract field, not a model projection field. It is
        # checked against the real non-truncating model window and then bound
        # into the prompt contract hash separately from the six projection caps.
        self.max_prompt_tokens = int(
            summary_cfg.get("max_prompt_tokens", DEFAULT_MAX_PROMPT_TOKENS)
        )
        hard_prompt_limit = min(
            int(self.max_input_tokens),
            max(1, int(self.context_window_tokens) - int(self.max_new_tokens)),
        )
        if self.max_prompt_tokens <= 0 or self.max_prompt_tokens > hard_prompt_limit:
            raise ValueError(
                "summary.max_prompt_tokens must be positive and no greater than "
                "the summarizer non-truncating input window: "
                f"configured={self.max_prompt_tokens} limit={hard_prompt_limit}"
            )
        # Validate every configured limit and materialize the exact contract hash.
        self.active_prompt_contract_sha256 = prompt_contract_sha256(
            **self._projection_kwargs(),
            max_prompt_tokens=self.max_prompt_tokens,
        )

    def _projection_kwargs(self) -> dict[str, int]:
        return {
            "max_steps": int(getattr(self, "max_steps", DEFAULT_MAX_TRAJECTORY_STEPS)),
            "max_goal_chars": int(
                getattr(self, "max_goal_chars", DEFAULT_MAX_GOAL_CHARS)
            ),
            "max_observation_chars": int(
                getattr(
                    self,
                    "max_observation_chars",
                    DEFAULT_MAX_OBSERVATION_CHARS,
                )
            ),
            "max_action_chars": int(
                getattr(self, "max_action_chars", DEFAULT_MAX_ACTION_CHARS)
            ),
            "max_admissible_actions": int(
                getattr(
                    self,
                    "max_admissible_actions",
                    DEFAULT_MAX_ADMISSIBLE_ACTIONS,
                )
            ),
            "max_admissible_action_chars": int(
                getattr(
                    self,
                    "max_admissible_action_chars",
                    DEFAULT_MAX_ADMISSIBLE_ACTION_CHARS,
                )
            ),
        }

    def _prompt_sha256(self) -> str:
        return str(
            getattr(self, "active_prompt_contract_sha256", "")
            or prompt_contract_sha256(
                **self._projection_kwargs(),
                max_prompt_tokens=int(
                    getattr(self, "max_prompt_tokens", DEFAULT_MAX_PROMPT_TOKENS)
                ),
            )
        )

    def _raw_prompt(self, rollout: dict) -> str:
        return render_webshop_summary_prompt(rollout, **self._projection_kwargs())

    def _render_prompt_chat(self, content: str) -> str:
        return self._render_chat([{"role": "user", "content": str(content)}])

    def _count_prompt_tokens(self, prompt: str) -> int:
        encoded = self.tokenizer(
            str(prompt),
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        token_ids = encoded["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if not isinstance(token_ids, Sequence) or isinstance(
            token_ids, (str, bytes, bytearray)
        ):
            raise RuntimeError("WebShop tokenizer returned invalid input_ids")
        if (
            token_ids
            and isinstance(token_ids[0], Sequence)
            and not isinstance(token_ids[0], (str, bytes, bytearray))
        ):
            if len(token_ids) != 1:
                raise RuntimeError(
                    "WebShop tokenizer returned multiple rows for one summary prompt"
                )
            token_ids = token_ids[0]
        return len(token_ids)

    def _fit_prompt(self, rollout: dict, *, content_suffix: str = "") -> str:
        fitted = fit_webshop_summary_prompt(
            rollout,
            projection_limits=self._projection_kwargs(),
            max_prompt_tokens=int(
                getattr(self, "max_prompt_tokens", DEFAULT_MAX_PROMPT_TOKENS)
            ),
            render_chat=self._render_prompt_chat,
            count_tokens=self._count_prompt_tokens,
            content_suffix=content_suffix,
        )
        return fitted.prompt

    def _prompt(self, rollout: dict) -> str:
        return self._fit_prompt(rollout)

    def _contract_retry_prompt(
        self,
        rollout: dict,
        *,
        rejected_output: str,
        validation_error: Exception,
        next_attempt: int,
    ) -> str:
        del rejected_output
        correction = f"""Correction attempt: {next_attempt}/{self.contract_max_attempts}
The previous response was rejected for WebShop summary contract error:
{_error_code(validation_error)}.
Think privately, then return only a corrected exact three-field JSON skill. Do
not repeat catalog-instance text, identifiers, scores, or unsupported product
claims."""
        return self._fit_prompt(
            rollout,
            content_suffix=f"\n\n{correction}",
        )

    @staticmethod
    def _parse(text: str, rollout: dict) -> dict[str, str]:
        return parse_webshop_skill_json(text, rollout=rollout)

    def _attach_generation_diagnostic(
        self,
        error: _RecoverableSummaryOutputError,
        token_ids: list[int],
        *,
        finish_reason: str,
    ) -> _RecoverableSummaryOutputError:
        error.raw_output = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        error.output_token_count = len(token_ids)
        error.finish_reason = finish_reason
        return error

    def _record_rejection(
        self,
        rollout: dict,
        generated: str | _RecoverableSummaryOutputError,
        error: Exception,
        *,
        attempt: int,
    ) -> None:
        record = {
            "schema_version": "webshop_summary_rejected_attempt_v1",
            "task_id": rollout.get("task_id"),
            "sample_id": _sample_id(rollout),
            "rollout_id": rollout.get("rollout_id"),
            "source_rollout_sha256": webshop_rollout_sha256(rollout),
            "attempt": attempt,
            "error": _error_code(error),
            "raw_output": getattr(generated, "raw_output", str(generated)),
            "output_token_count": getattr(generated, "output_token_count", None),
            "finish_reason": getattr(generated, "finish_reason", None),
        }
        if not hasattr(self, "attempt_diagnostics"):
            self.attempt_diagnostics = []
        self.attempt_diagnostics.append(record)
        # The worker's append-only run log must retain the actual cause before
        # the all-or-nothing batch gate raises and the supervisor cleans up.
        print(
            "[WEBSHOP-SUMMARY-REJECT] "
            + json.dumps(record, ensure_ascii=True, sort_keys=True),
            flush=True,
        )

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
                f"{label} WebShop prompt exceeds the non-truncating token budget: "
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
                    f"{label} requires exactly one vLLM continuation per WebShop prompt"
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
                results.append(
                    self._attach_generation_diagnostic(
                        exc,
                        token_ids,
                        finish_reason=str(getattr(candidate, "finish_reason", "")),
                    )
                )
                continue
            if str(getattr(candidate, "finish_reason", "")) == "length":
                recovered, status = webshop_capped_complete_json_recovery(text, rollout)
                self.generation_outcomes[
                    str(rollout.get("rollout_id") or rollout.get("task_id") or "")
                ] = status
                if not recovered:
                    results.append(
                        self._attach_generation_diagnostic(
                            _RecoverableSummaryOutputError(status, status=status),
                            token_ids,
                            finish_reason="length",
                        )
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
                f"{label} WebShop prompt/rollout count mismatch: "
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
                results.append(
                    self._attach_generation_diagnostic(
                        exc,
                        token_ids,
                        finish_reason="length" if cutoff_reason else "stop",
                    )
                )
                continue
            if cutoff_reason is not None:
                recovered, status = webshop_capped_complete_json_recovery(text, rollout)
                self.generation_outcomes[
                    str(rollout.get("rollout_id") or rollout.get("task_id") or "")
                ] = status
                if not recovered:
                    results.append(
                        self._attach_generation_diagnostic(
                            _RecoverableSummaryOutputError(status, status=status),
                            token_ids,
                            finish_reason="length",
                        )
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
    ) -> WebShopSummaryResult:
        return self._parse_batch_with_contract_retries(
            [initial_text], [rollout], device=device
        )[0]

    def _parse_batch_with_contract_retries(
        self,
        initial_texts: list[str | _RecoverableSummaryOutputError],
        rollouts: list[dict],
        *,
        device,
    ) -> list[WebShopSummaryResult]:
        if len(initial_texts) != len(rollouts):
            raise RuntimeError(
                "WebShop summary parse input/rollout count mismatch: "
                f"texts={len(initial_texts)} rollouts={len(rollouts)}"
            )
        if not rollouts:
            return []

        prompt_sha = self._prompt_sha256()
        parsed: list[WebShopSummaryResult | None] = [None] * len(rollouts)
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
                    parsed[index] = _successful_result(
                        rollout,
                        skill,
                        attempts=attempt,
                        prompt_sha256=prompt_sha,
                    )
                except (
                    WebShopSummaryContractError,
                    _RecoverableSummaryOutputError,
                ) as exc:
                    code = _error_code(exc)
                    self._record_rejection(rollout, generated, exc, attempt=attempt)
                    if attempt >= self.contract_max_attempts:
                        parsed[index] = _failed_result(
                            rollout,
                            attempts=attempt,
                            error=code,
                            prompt_sha256=prompt_sha,
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
                label=f"webshop summarizer contract retry {attempt + 1}",
            )
            if len(next_texts) != len(retry_indexes):
                raise RuntimeError(
                    "WebShop summarizer retry output count mismatch: "
                    f"outputs={len(next_texts)} rollouts={len(retry_indexes)}"
                )
            pending = list(zip(retry_indexes, next_texts))

        if any(result is None for result in parsed):
            raise AssertionError("WebShop summarizer left a sample without a result")
        return [result for result in parsed if result is not None]

    @torch.inference_mode()
    def summarize_batch(
        self,
        rollouts: list[dict],
        progress_callback=None,
    ) -> list[WebShopSummaryResult]:
        if not rollouts:
            return []
        self.generation_outcomes = {}
        self.attempt_diagnostics = []
        prompts = [self._prompt(rollout) for rollout in rollouts]
        device = (
            None
            if getattr(self, "backend", "transformers") == "vllm"
            else next(self.llm.parameters()).device
        )
        results: list[WebShopSummaryResult] = []
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
                label="webshop summarizer",
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

    def summarize_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        progress_callback=None,
    ) -> list[WebShopSummaryResult]:
        """Summarize every row in order and fail unless all receive one skill."""

        rollouts = [dict(row) for row in rows]
        results = self.summarize_batch(rollouts, progress_callback=progress_callback)
        return require_complete_summaries(results, expected_count=len(rollouts))


__all__ = [
    "IncompleteWebShopSummaryError",
    "WEBSHOP_SKILL_FIELD_MAX_CHARS",
    "WEBSHOP_SUMMARY_RESULT_SCHEMA_VERSION",
    "WebShopSkillSummarizer",
    "WebShopSummaryContractError",
    "WebShopSummaryError",
    "WebShopSummaryResult",
    "parse_webshop_skill_json",
    "require_complete_summaries",
    "webshop_capped_complete_json_recovery",
    "webshop_rollout_sha256",
    "webshop_skill_sha256",
    "webshop_source_rollout_projection",
]
