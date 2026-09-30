"""Tokenizer-aware, all-step-preserving WebShop summary prompt fitting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any


WEBSHOP_TOKEN_BUDGET_POLICY_VERSION = (
    "webshop_tokenizer_counted_all_steps_geometric_fit_v1"
)
_TEXT_LIMIT_KEYS = (
    "max_goal_chars",
    "max_observation_chars",
    "max_action_chars",
    "max_admissible_action_chars",
)


class WebShopPromptBudgetError(ValueError):
    """A complete trajectory cannot fit the configured summary input budget."""


@dataclass(frozen=True, slots=True)
class FittedWebShopPrompt:
    prompt: str
    token_count: int
    effective_projection_limits: dict[str, int]
    compaction_rounds: int


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WebShopPromptBudgetError(f"{label} must be a positive integer")
    return value


def _limits(value: Mapping[str, Any]) -> dict[str, int]:
    expected = {"max_steps", "max_admissible_actions", *_TEXT_LIMIT_KEYS}
    if set(value) != expected:
        raise WebShopPromptBudgetError(
            "projection limits must contain exactly " + repr(sorted(expected))
        )
    result = {
        key: _positive_integer(value[key], label=key)
        for key in ("max_steps", *_TEXT_LIMIT_KEYS)
    }
    actions = value["max_admissible_actions"]
    if isinstance(actions, bool) or not isinstance(actions, int) or actions < 0:
        raise WebShopPromptBudgetError(
            "max_admissible_actions must be a non-negative integer"
        )
    result["max_admissible_actions"] = actions
    return result


def _shrink(limits: Mapping[str, int], *, ratio: float) -> dict[str, int]:
    result = dict(limits)
    for key in _TEXT_LIMIT_KEYS:
        result[key] = max(1, int(math.floor(result[key] * ratio)))
    result["max_admissible_actions"] = max(
        0, int(math.floor(result["max_admissible_actions"] * ratio))
    )
    if result == dict(limits):
        for key in (*_TEXT_LIMIT_KEYS, "max_admissible_actions"):
            minimum = 0 if key == "max_admissible_actions" else 1
            if result[key] > minimum:
                result[key] -= 1
                break
    return result


def fit_webshop_summary_prompt(
    rollout: Mapping[str, Any],
    *,
    projection_limits: Mapping[str, int],
    max_prompt_tokens: int,
    render_chat: Callable[[str], str],
    count_tokens: Callable[[str], int],
    content_suffix: str = "",
    max_rounds: int = 32,
) -> FittedWebShopPrompt:
    """Fit one complete trajectory without tokenizer-side truncation.

    Every candidate uses the original number of trajectory steps. Only the
    explicitly lossy per-field display caps and number of displayed admissible
    actions are reduced. The exact full observation/action list remains bound
    by the trajectory-level SHA-256 embedded in every candidate projection.
    """

    from skillflow.summarization.webshop_prompts import render_webshop_summary_prompt

    budget = _positive_integer(max_prompt_tokens, label="max_prompt_tokens")
    rounds = _positive_integer(max_rounds, label="max_rounds")
    limits = _limits(projection_limits)
    suffix = str(content_suffix)

    last_count = 0
    for round_index in range(rounds + 1):
        content = render_webshop_summary_prompt(rollout, **limits) + suffix
        prompt = str(render_chat(content))
        count = count_tokens(prompt)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise WebShopPromptBudgetError(
                "count_tokens must return a non-negative integer"
            )
        if count <= budget:
            return FittedWebShopPrompt(
                prompt=prompt,
                token_count=count,
                effective_projection_limits=dict(limits),
                compaction_rounds=round_index,
            )
        last_count = count
        if round_index == rounds:
            break
        # Use actual tokenizer pressure, with a geometric ceiling that ensures
        # deterministic progress even when tokenization is locally non-monotonic.
        ratio = min(0.8, max(0.1, (budget / max(1, count)) * 0.9))
        next_limits = _shrink(limits, ratio=ratio)
        if next_limits == limits:
            break
        limits = next_limits

    raise WebShopPromptBudgetError(
        "WebShop summary prompt cannot fit without dropping trajectory steps: "
        f"budget={budget} last_token_count={last_count} "
        f"minimum_effective_limits={limits}"
    )


__all__ = [
    "FittedWebShopPrompt",
    "WEBSHOP_TOKEN_BUDGET_POLICY_VERSION",
    "WebShopPromptBudgetError",
    "fit_webshop_summary_prompt",
]
