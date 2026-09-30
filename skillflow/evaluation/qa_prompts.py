"""ALFWorld-style executor prompt and strict action syntax for QA skill collection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


QA_AGENT_SYSTEM_PROMPT = (
    "You are an expert question-answering agent. At every step, choose exactly "
    "one action from the supplied candidate list. Search when more evidence is "
    "needed; answer as soon as the available evidence is sufficient. Follow the "
    "required response format exactly. Keep private reasoning brief and do not "
    "repeat it. For answer actions, submit only the shortest answer span, never "
    "a sentence or explanation."
)
QA_AGENT_USER_TEMPLATE = """Your query is:
{query}

## Relevant Skill
{skill_block}Treat the skill as guidance, not as factual evidence or a fixed
action sequence.
If the skill conflicts with the current observation or candidate actions, follow
the current observation and candidate actions.

## Recent Interactions
Prior to this step, you have already taken {completed_steps} step(s). Below are
the most recent {visible_steps} observations and corresponding actions you took:
{history}

## Current Observation
You are now at step {current_step} of {max_steps}. Your current observation is:
{current_observation}

## Candidate Actions
[
{candidate_actions}
]

If a candidate contains a placeholder, replace it with concise text. The
answer[...] action submits the final answer and ends the episode.
Put the selected action inside:
<action>...</action>
"""

NQ_DATASET = "nq"
HOTPOTQA_DATASET = "hotpotqa"
SUPPORTED_QA_DATASETS = frozenset({NQ_DATASET, HOTPOTQA_DATASET, "triviaqa", "popqa", "2wikimultihopqa", "musique", "bamboogle"})
ANSWER_CANDIDATE = "answer[<your answer>]"
SEARCH_CANDIDATE = "search[<your query>]"
DEFAULT_QA_MAX_STEPS = 4

_WRAPPER = re.compile(r"<action\s*>\s*(.*?)\s*</action\s*>", re.IGNORECASE | re.DOTALL)
_OPEN = re.compile(r"<action\s*>", re.IGNORECASE)
_CLOSE = re.compile(r"</action\s*>", re.IGNORECASE)
_INNER = re.compile(
    r"(?P<kind>answer|search)\s*\[(?P<value>.*)\]",
    re.IGNORECASE | re.DOTALL,
)
_INNER_START = re.compile(r"(?:answer|search)\s*\[", re.IGNORECASE)
_PLACEHOLDERS = frozenset({"<your answer>", "<your query>"})


@dataclass(frozen=True)
class ParsedQAAction:
    action: str
    kind: str
    value: str
    extraction_mode: str
    format_valid: bool


def normalize_qa_dataset(value: object) -> str:
    text = str(value or "").strip().casefold().replace("-", "").replace("_", "")
    aliases = {
        "nq": NQ_DATASET,
        "naturalquestions": NQ_DATASET,
        "hotpot": HOTPOTQA_DATASET,
        "hotpotqa": HOTPOTQA_DATASET,
        "triviaqa": "triviaqa", "popqa": "popqa",
        "2wiki": "2wikimultihopqa", "2wikimultihopqa": "2wikimultihopqa",
        "musique": "musique", "bamboogle": "bamboogle",
    }
    if text not in aliases:
        raise ValueError(
            f"Unsupported SearchQA dataset {value!r}"
        )
    return aliases[text]


def qa_horizon(dataset: object, *, max_steps: int) -> int:
    """Return the shared action budget after validating the dataset name."""

    normalize_qa_dataset(dataset)
    if isinstance(max_steps, bool):
        raise ValueError("max_steps must be a positive integer")
    horizon = int(max_steps)
    if horizon <= 0:
        raise ValueError("max_steps must be a positive integer")
    return horizon


def candidate_actions_for_step(
    dataset: object,
    *,
    current_step: int,
    max_steps: int,
) -> list[str]:
    """Offer search and answer at every valid step for either QA dataset."""

    step = int(current_step)
    if step <= 0:
        raise ValueError("current_step must be positive")
    name = normalize_qa_dataset(dataset)
    horizon = qa_horizon(name, max_steps=max_steps)
    if step > horizon:
        raise ValueError(f"current_step={step} exceeds dataset horizon={horizon}")
    return [SEARCH_CANDIDATE, ANSWER_CANDIDATE]


def _visible_history(
    history: Sequence[dict],
    *,
    history_length: int,
    observation_chars: int,
) -> tuple[str, int]:
    limit = max(0, int(history_length))
    recent = list(history[-limit:]) if limit else []
    hidden = len(history) - len(recent)
    lines = []
    if hidden:
        lines.append(
            f"[Earlier completed steps 1-{hidden} are outside the visible window.]"
        )
    for number, row in enumerate(recent, start=hidden + 1):
        observation = str(row.get("observation") or "")
        if observation_chars > 0:
            observation = observation[-int(observation_chars) :]
        lines.append(
            f"[Observation {number}: {observation!r}, "
            f"Action {number}: {str(row.get('action') or '')!r}]"
        )
    return "\n".join(lines) or "None yet.", len(recent)


def render_qa_agent_messages(
    *,
    query: object,
    skill_text: object,
    current_observation: object,
    candidate_actions: Sequence[str],
    history: Sequence[dict],
    history_length: int,
    observation_chars: int,
    max_steps: int,
) -> list[dict[str, str]]:
    query_text = str(query or "").strip()
    if not query_text:
        raise ValueError("QA executor query must be non-empty")
    candidates = [str(value).strip() for value in candidate_actions]
    if not candidates or any(not value for value in candidates):
        raise ValueError("QA executor candidate actions must be non-empty strings")
    if isinstance(max_steps, bool) or int(max_steps) <= 0:
        raise ValueError("QA executor max_steps must be a positive integer")
    rendered_history, visible_steps = _visible_history(
        history,
        history_length=history_length,
        observation_chars=max(0, int(observation_chars)),
    )
    skill = str(skill_text or "").strip()
    user = QA_AGENT_USER_TEMPLATE.format(
        query=query_text,
        skill_block=f"{skill}\n\n" if skill else "\n",
        completed_steps=len(history),
        visible_steps=visible_steps,
        history=rendered_history,
        current_step=len(history) + 1,
        current_observation=(str(current_observation or "").strip() or "None yet."),
        candidate_actions=",\n".join(f"'{value}'" for value in candidates),
        max_steps=int(max_steps),
    )
    return [
        {"role": "system", "content": QA_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_qa_inner_action(value: object) -> ParsedQAAction:
    raw = str(value or "").strip()
    match = _INNER.fullmatch(raw)
    if match is None or len(_INNER_START.findall(raw)) != 1:
        return ParsedQAAction(raw, "invalid", "", "inner_action", False)
    kind = match.group("kind").casefold()
    action_value = match.group("value").strip()
    folded_value = action_value.casefold()
    if not action_value or any(marker in folded_value for marker in _PLACEHOLDERS):
        return ParsedQAAction(raw, kind, action_value, "inner_action", False)
    return ParsedQAAction(
        f"{kind}[{action_value}]", kind, action_value, "inner_action", True
    )


def parse_qa_action(response: object) -> ParsedQAAction:
    raw = str(response or "")
    matches = list(_WRAPPER.finditer(raw))
    if (
        len(matches) != 1
        or len(_OPEN.findall(raw)) != 1
        or len(_CLOSE.findall(raw)) != 1
    ):
        return ParsedQAAction(raw[-80:], "invalid", "", "response_tail", False)
    parsed = parse_qa_inner_action(matches[0].group(1))
    return ParsedQAAction(
        parsed.action,
        parsed.kind,
        parsed.value,
        "action_tag",
        parsed.format_valid,
    )


def qa_action_is_admissible(
    parsed: ParsedQAAction,
    candidate_actions: Sequence[str],
) -> bool:
    if not parsed.format_valid:
        return False
    allowed = set()
    for candidate in candidate_actions:
        normalized = str(candidate).strip().casefold()
        if normalized == ANSWER_CANDIDATE:
            allowed.add("answer")
        elif normalized == SEARCH_CANDIDATE:
            allowed.add("search")
        else:
            raise ValueError(f"Unsupported QA candidate action: {candidate!r}")
    return parsed.kind in allowed


def has_complete_qa_action(response: object) -> bool:
    return parse_qa_action(response).format_valid


__all__ = [
    "ANSWER_CANDIDATE",
    "DEFAULT_QA_MAX_STEPS",
    "HOTPOTQA_DATASET",
    "NQ_DATASET",
    "ParsedQAAction",
    "QA_AGENT_SYSTEM_PROMPT",
    "QA_AGENT_USER_TEMPLATE",
    "SEARCH_CANDIDATE",
    "SUPPORTED_QA_DATASETS",
    "candidate_actions_for_step",
    "has_complete_qa_action",
    "normalize_qa_dataset",
    "parse_qa_action",
    "parse_qa_inner_action",
    "qa_action_is_admissible",
    "qa_horizon",
    "render_qa_agent_messages",
]
