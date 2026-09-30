"""SearchQA main-method episode runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from typing import Any

from skillflow.evaluation.alfworld_env import BatchEnvironmentInterface
from skillflow.evaluation.actor_base import ActionGeneration
from skillflow.evaluation.qa_env import (
    Retriever,
    make_qa_summary_batch_env,
)
from skillflow.evaluation.qa_prompts import (
    normalize_qa_dataset,
    parse_qa_action,
    qa_action_is_admissible,
)
from skillflow.evaluation.skill_text import (
    canonical_skill_text,
    parse_executable_skill,
)


QA_SUMMARY_ROLLOUT_SCHEMA_VERSION = "qa_summary_rollout_v1"
QA_SUMMARY_METRICS_SCHEMA_VERSION = "qa_summary_rollout_metrics_v1"
INVALID_QA_ACTION = "__qa_summary_invalid_action__"
DEFAULT_QA_SUMMARY_PROTOCOL = "imf_main_evaluation"

EnvironmentFactory = Callable[
    [Mapping[str, Any], Sequence[Mapping[str, Any]], int],
    BatchEnvironmentInterface,
]


def _required_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _task_id(task: Mapping[str, Any], position: int) -> str:
    return _required_text(
        task.get("task_id") or task.get("id"),
        f"tasks[{position}].task_id",
    )


def _query(task: Mapping[str, Any], task_id: str) -> str:
    return _required_text(
        task.get("query") or task.get("question") or task.get("goal"),
        f"task {task_id}.query",
    )


def _answers(task: Mapping[str, Any], task_id: str) -> list[str]:
    raw = task.get("golden_answers", task.get("answer"))
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"Task {task_id!r} golden_answers must be a sequence")
    answers = [str(value).strip() for value in values if str(value).strip()]
    if not answers:
        raise ValueError(f"Task {task_id!r} has no golden answers")
    return answers


def _protocol_id(cfg: Mapping[str, Any]) -> str:
    branch = cfg.get("qa_summary") or {}
    if not isinstance(branch, Mapping):
        raise ValueError("qa_summary config must be a mapping")
    return str(branch.get("protocol_id") or DEFAULT_QA_SUMMARY_PROTOCOL).strip()


def _skill_payload(value: object, *, task_id: str) -> tuple[dict[str, str], str, str]:
    """Return the structured skill, canonical display JSON and stable hash."""

    raw = value
    if not isinstance(raw, (str, Mapping)) and hasattr(raw, "skill"):
        raw = getattr(raw, "skill")
    if isinstance(raw, Mapping) and "skill" in raw:
        status = str(raw.get("status") or "ok").strip().casefold()
        if status != "ok":
            raise ValueError(
                f"Task {task_id!r} has a non-success QA summary result: {status}"
            )
        raw = raw.get("skill")
    if raw in (None, ""):
        raise ValueError(f"Task {task_id!r} has no injectable QA skill")
    try:
        parsed = parse_executable_skill(raw)
    except ValueError as exc:
        raise ValueError(f"Task {task_id!r} has an invalid QA skill: {exc}") from exc
    skill = {
        "when_to_use": str(parsed["when_to_use"]),
        "high_level_guidance": str(parsed["high_level_guidance"]),
        "low_level_guidance": str(parsed["low_level_guidance"]),
    }
    canonical = canonical_skill_text(skill)
    digest_payload = json.dumps(
        skill,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return skill, canonical, hashlib.sha256(digest_payload).hexdigest()


def _resolve_skills(
    task_ids: Sequence[str],
    skills_by_task: Mapping[str, object] | None,
) -> tuple[list[dict[str, str] | str], list[str], list[str | None], bool]:
    if not isinstance(skills_by_task, Mapping):
        raise TypeError("skills_by_task must be a task-id mapping or None")
    payloads: list[dict[str, str]] = []
    canonical_texts: list[str] = []
    digests: list[str] = []
    for task_id in task_ids:
        if task_id not in skills_by_task:
            raise ValueError(f"Missing QA skill for task {task_id!r}")
        payload, canonical, digest = _skill_payload(
            skills_by_task[task_id],
            task_id=task_id,
        )
        payloads.append(payload)
        canonical_texts.append(canonical)
        digests.append(digest)
    return payloads, canonical_texts, digests, True


def _make_environment(
    cfg: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    seed: int,
    *,
    retriever: Retriever | None,
    env_factory: EnvironmentFactory,
) -> BatchEnvironmentInterface:
    if retriever is None:
        return env_factory(cfg, tasks, seed)
    if env_factory is not make_qa_summary_batch_env:
        raise ValueError(
            "retriever can only be passed with make_qa_summary_batch_env; "
            "inject retrieval inside a custom env_factory instead"
        )
    return make_qa_summary_batch_env(cfg, tasks, seed, retriever=retriever)


def _failure_reason(info: Mapping[str, Any], *, invalid_count: int) -> str:
    reason = str(info.get("termination_reason") or "").strip()
    if reason == "answer_submitted":
        return "incorrect_answer"
    if reason:
        return reason
    if invalid_count:
        return "invalid_action"
    return "ended_without_exact_match"


def run_qa_summary_batch(
    cfg: Mapping[str, Any],
    generator: Any,
    tasks: Sequence[Mapping[str, Any]],
    skills_by_task: Mapping[str, object],
    *,
    max_steps: int | None = None,
    seed: int = 0,
    retriever: Retriever | None = None,
    env_factory: EnvironmentFactory = make_qa_summary_batch_env,
) -> list[dict[str, Any]]:
    """Run one batch with generated skills."""

    if not tasks:
        return []
    task_rows = [dict(task) for task in tasks]
    task_ids = [_task_id(task, index) for index, task in enumerate(task_rows)]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("QA summary batch contains duplicate task ids")
    queries = [_query(task, task_id) for task, task_id in zip(task_rows, task_ids)]
    datasets = [
        normalize_qa_dataset(task.get("dataset") or task.get("data_source"))
        for task in task_rows
    ]
    answers = [_answers(task, task_id) for task, task_id in zip(task_rows, task_ids)]
    skill_payloads, skill_texts, skill_hashes, skill_injected = _resolve_skills(
        task_ids,
        skills_by_task,
    )
    protocol_id = _protocol_id(cfg)
    arm = "skilled"

    env = _make_environment(
        cfg,
        task_rows,
        int(seed),
        retriever=retriever,
        env_factory=env_factory,
    )
    try:
        observations, infos = env.reset()
        batch_size = len(task_rows)
        if len(observations) != batch_size or len(infos) != batch_size:
            raise RuntimeError("QA environment reset returned a partial batch")
        horizons: list[int] = []
        for index, info in enumerate(infos):
            horizon = int((info or {}).get("horizon", 0))
            if horizon <= 0:
                raise RuntimeError(
                    "QA environment did not expose a positive horizon for "
                    f"{task_ids[index]!r}"
                )
            horizons.append(horizon)
        loop_limit = max(horizons)
        if max_steps is not None:
            requested_limit = int(max_steps)
            if requested_limit <= 0:
                raise ValueError("max_steps must be positive")
            if requested_limit < loop_limit:
                raise ValueError(
                    "max_steps cannot be lower than a dataset horizon: "
                    f"max_steps={requested_limit} required={loop_limit}"
                )
            loop_limit = requested_limit

        histories: list[list[dict[str, Any]]] = [[] for _ in task_rows]
        rewards = [0.0] * batch_size
        dones = [False] * batch_size
        last_infos = [dict(info or {}) for info in infos]

        for _ in range(loop_limit):
            active = [index for index, done in enumerate(dones) if not done]
            if not active:
                break
            items: list[dict[str, Any]] = []
            for index in active:
                candidates = [
                    str(value)
                    for value in last_infos[index].get("admissible_commands", [])
                ]
                if not candidates:
                    raise RuntimeError(
                        f"Active QA task {task_ids[index]!r} has no candidate actions"
                    )
                items.append(
                    {
                        "task_id": task_ids[index],
                        "goal": queries[index],
                        "skill": skill_payloads[index],
                        "observation": str(observations[index]),
                        "admissible": candidates,
                        "environment_admissible": candidates,
                        "history": histories[index],
                        "horizon": horizons[index],
                        "_index": index,
                    }
                )
            generated: list[ActionGeneration] = list(generator.generate_batch(items))
            if len(generated) != len(items):
                raise RuntimeError(
                    "QA generator returned the wrong number of rows: "
                    f"expected={len(items)} actual={len(generated)}"
                )

            actions = ["" for _ in task_rows]
            metadata: list[dict[str, Any]] = [{} for _ in task_rows]
            for item, decision in zip(items, generated):
                index = int(item["_index"])
                proposed_action = str(decision.action or "").strip()
                parsed_response = parse_qa_action(decision.response)
                format_valid = bool(decision.format_valid) and bool(
                    parsed_response.format_valid
                    and parsed_response.action == proposed_action
                )
                admissible = (
                    format_valid
                    and bool(decision.action_is_admissible)
                    and qa_action_is_admissible(
                        parsed_response,
                        item["environment_admissible"],
                    )
                )
                incomplete = bool(decision.generation_incomplete)
                action_executed = admissible and not incomplete
                environment_action = (
                    proposed_action if action_executed else INVALID_QA_ACTION
                )
                actions[index] = environment_action
                metadata[index] = {
                    "model_output": str(decision.response),
                    "action_extraction_mode": str(decision.extraction_mode),
                    "action_format_valid": format_valid,
                    "action_was_admissible": admissible,
                    "generation_incomplete": incomplete,
                    "action_executed": action_executed,
                    "invalid": not action_executed,
                    "proposed_action": proposed_action,
                    "environment_action": environment_action,
                    "admissible_actions": list(item["environment_admissible"]),
                }

            next_obs, next_rewards, next_dones, next_infos = env.step(actions)
            if not all(
                len(values) == batch_size
                for values in (next_obs, next_rewards, next_dones, next_infos)
            ):
                raise RuntimeError("QA environment step returned a partial batch")
            for index in active:
                histories[index].append(
                    {
                        "step": len(histories[index]) + 1,
                        "observation": str(observations[index]),
                        "action": actions[index],
                        **metadata[index],
                        "reward": float(next_rewards[index]),
                        "done": bool(next_dones[index]),
                        "next_observation": str(next_obs[index]),
                        "environment_info": dict(next_infos[index] or {}),
                    }
                )
                rewards[index] = float(next_rewards[index])
                dones[index] = bool(next_dones[index])
                last_infos[index] = dict(next_infos[index] or {})
            observations = [str(value) for value in next_obs]

        unfinished = [task_ids[index] for index, done in enumerate(dones) if not done]
        if unfinished:
            raise RuntimeError(
                "QA episodes exceeded their declared horizons without terminating: "
                + ", ".join(unfinished[:8])
            )

        rows: list[dict[str, Any]] = []
        for index, task in enumerate(task_rows):
            final_info = last_infos[index]
            exact_match = bool(
                final_info.get(
                    "exact_match",
                    final_info.get("won", rewards[index] > 0),
                )
            )
            score = float(final_info.get("task_score", rewards[index]))
            invalid_count = sum(
                not bool(step.get("action_executed")) for step in histories[index]
            )
            row: dict[str, Any] = {
                "schema_version": QA_SUMMARY_ROLLOUT_SCHEMA_VERSION,
                "rollout_id": f"{arm}:{datasets[index]}:{task_ids[index]}",
                "benchmark": "searchqa",
                "protocol_id": protocol_id,
                "dataset": datasets[index],
                "source_split": str(task.get("source_split") or "test"),
                "task_id": task_ids[index],
                "query": queries[index],
                "score": score,
                "score_percent": score * 100.0,
                "em": exact_match,
                "exact_match": exact_match,
                "success": exact_match,
                "done": bool(dones[index]),
                "horizon": horizons[index],
                "num_steps": len(histories[index]),
                "invalid": invalid_count > 0,
                "invalid_action_count": invalid_count,
                "invalid_action_rate": (
                    invalid_count / len(histories[index]) if histories[index] else 0.0
                ),
                "skill": skill_payloads[index],
                "skill_canonical_json": skill_texts[index],
                "skill_injected": skill_injected,
                "injected_skill_sha256": skill_hashes[index],
                "final_answer": str(final_info.get("final_answer") or ""),
                "final_observation": str(observations[index]),
                "final_info": final_info,
                "failure_reason": (
                    ""
                    if exact_match
                    else _failure_reason(
                        final_info,
                        invalid_count=invalid_count,
                    )
                ),
                "golden_answers": answers[index],
                "steps": histories[index],
            }
            rows.append(row)
        return rows
    finally:
        env.close()

