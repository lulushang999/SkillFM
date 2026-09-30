from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from skillflow.evaluation.names import normalize_benchmark_name


WEBSHOP_SYSTEM = (
    "You are an expert autonomous agent operating in the WebShop e-commerce "
    "environment. Choose exactly one available action that best advances the "
    "shopping instruction and follow the required response format exactly."
)

SEARCHQA_SYSTEM = (
    "You are an expert search-augmented question-answering agent. Use retrieved "
    "passages as evidence, issue at most one search or final-answer action per "
    "turn, and follow the required response format exactly."
)

SKILL_BLOCK = """## Relevant Skill
{skill_text}

Treat the skill as guidance, not as evidence or a fixed action sequence. Current
observations and retrieved passages take precedence over stale skill details.

"""

WEBSHOP_PROMPT = """Your shopping instruction is:
{task_description}

{skill_block}## Recent Interactions
Prior to this step, you have already taken {completed_steps} step(s). The most
recent {visible_steps} observation/action pairs are:
{history}

## Current WebShop Observation
{current_observation}

## Available Actions
[
{available_actions}
]

Choose exactly one listed click action, or replace <your query> in the listed
search action with a concise query. Put only the selected environment action
inside:
<action>...</action>
"""

SEARCHQA_PROMPT = """Answer this question:
{task_description}

{skill_block}## Search Interaction History
Prior to this turn, you have already taken {completed_steps} action(s). The most
recent {visible_steps} search/answer actions and returned observations are:
{history}

## Current Search Observation
{current_observation}

Choose exactly one action:
- If more evidence is needed, output <search>one concise query</search>.
- If the evidence is sufficient, output <answer>the short final answer</answer>.

Output exactly one complete <search> or <answer> field. Do not output both.
"""


@dataclass(frozen=True)
class ParsedBenchmarkAction:
    action: str
    extraction_mode: str
    format_valid: bool
    kind: str


def system_prompt(benchmark: object) -> str:
    name = normalize_benchmark_name(benchmark)
    return WEBSHOP_SYSTEM if name == "webshop" else SEARCHQA_SYSTEM


def _visible_history(
    history: list[dict],
    *,
    history_length: int,
    observation_chars: int,
    searchqa: bool,
) -> tuple[str, int]:
    recent = history[-history_length:] if history_length > 0 else []
    start = len(history) - len(recent)
    lines: list[str] = []
    if start:
        lines.append(
            f"[Earlier completed steps 1-{start} are outside the visible window.]"
        )
    for offset, record in enumerate(recent, start=start + 1):
        action = str(record.get("action") or "")
        if searchqa:
            observation = str(
                record.get("next_observation") or record.get("observation") or ""
            )
            label = "Returned observation"
        else:
            observation = str(record.get("observation") or "")
            label = "Observation"
        if observation_chars > 0:
            observation = observation[-observation_chars:]
        lines.append(
            f"[{label} {offset}: {observation!r}, Action {offset}: {action!r}]"
        )
    return "\n".join(lines) or "None yet.", len(recent)


def render_benchmark_prompt(
    benchmark: object,
    *,
    task_description: str,
    skill_text: str,
    current_observation: str,
    admissible_actions: Sequence[str],
    history: list[dict],
    history_length: int,
    observation_chars: int,
) -> str:
    name = normalize_benchmark_name(benchmark)
    skill_block = (
        SKILL_BLOCK.format(skill_text=str(skill_text).strip())
        if str(skill_text).strip()
        else ""
    )
    rendered_history, visible_steps = _visible_history(
        history,
        history_length=max(0, int(history_length)),
        observation_chars=max(0, int(observation_chars)),
        searchqa=name == "searchqa",
    )
    common = {
        "task_description": str(task_description).strip(),
        "skill_block": skill_block,
        "completed_steps": len(history),
        "visible_steps": visible_steps,
        "history": rendered_history,
        "current_observation": str(current_observation).strip() or "None yet.",
    }
    if name == "webshop":
        available = ",\n".join(f"'{action}'" for action in admissible_actions)
        return WEBSHOP_PROMPT.format(
            **common,
            available_actions=available or "'No actions were supplied.'",
        )
    return SEARCHQA_PROMPT.format(**common)


def _parse_action_tag(response: str) -> ParsedBenchmarkAction:
    raw = str(response or "")
    matches = list(
        re.finditer(
            r"<action>\s*(.*?)\s*</action>",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not matches:
        return ParsedBenchmarkAction(raw[-30:], "response_tail", False, "invalid")
    action = matches[0].group(1).strip()
    valid = len(matches) == 1 and bool(action)
    return ParsedBenchmarkAction(action, "action_tag", valid, "action")


def _parse_searchqa_tag(response: str) -> ParsedBenchmarkAction:
    raw = str(response or "")
    pattern = re.compile(
        r"<(search|answer)>\s*(.*?)\s*</\1>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(raw))
    opening_tags = re.findall(r"<(?:search|answer)>", raw, flags=re.IGNORECASE)
    if not matches:
        return ParsedBenchmarkAction(raw[-80:], "response_tail", False, "invalid")
    kind = matches[0].group(1).casefold()
    value = matches[0].group(2).strip()
    action = f"<{kind}>{value}</{kind}>"
    valid = len(matches) == 1 and len(opening_tags) == 1 and bool(value)
    return ParsedBenchmarkAction(action, f"{kind}_tag", valid, kind)


def parse_benchmark_action(response: str, benchmark: object) -> ParsedBenchmarkAction:
    name = normalize_benchmark_name(benchmark)
    if name == "webshop":
        return _parse_action_tag(response)
    return _parse_searchqa_tag(response)


def has_complete_benchmark_action(response: str, benchmark: object) -> bool:
    return parse_benchmark_action(response, benchmark).format_valid


def benchmark_action_is_admissible(
    action: str,
    admissible_actions: Sequence[str],
    benchmark: object,
    *,
    format_valid: bool,
) -> bool:
    name = normalize_benchmark_name(benchmark)
    if not format_valid:
        return False
    if name == "searchqa":
        return True
    candidates = [str(item) for item in admissible_actions]
    folded_action = str(action).strip().casefold()
    if action in candidates and folded_action != "search[<your query>]":
        return True
    wildcard_available = any(
        str(candidate).strip().casefold() == "search[<your query>]"
        for candidate in candidates
    )
    match = re.fullmatch(r"search\[(.*?)\]", str(action).strip(), flags=re.I | re.S)
    if not wildcard_available or match is None:
        return False
    query = match.group(1).strip()
    return bool(query) and query.casefold() != "<your query>"


__all__ = [
    "ParsedBenchmarkAction",
    "benchmark_action_is_admissible",
    "has_complete_benchmark_action",
    "normalize_benchmark_name",
    "parse_benchmark_action",
    "render_benchmark_prompt",
    "system_prompt",
]
