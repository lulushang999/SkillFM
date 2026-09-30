"""Dataset-aware environment for NQ and HotpotQA skill collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
import re
from typing import Any
from urllib.parse import urlparse

from skillflow.evaluation.alfworld_env import BatchEnvironmentInterface
from skillflow.evaluation.http_json import post_json
from skillflow.evaluation.qa_scoring import (
    format_retrieved_documents,
    searchqa_exact_match,
)
from skillflow.evaluation.retrieval_config import (
    QASummaryConfigError,
    _validate_retrieval_endpoints,
)
from skillflow.evaluation.qa_prompts import (
    DEFAULT_QA_MAX_STEPS,
    candidate_actions_for_step,
    normalize_qa_dataset,
    parse_qa_inner_action,
    qa_action_is_admissible,
    qa_horizon,
)


Retriever = Callable[[str, int], object]
DEFAULT_RETRIEVAL_OBSERVATION_CHARS = 6_000
QA_RETRIEVAL_PORT_ENV = "SKILLFLOW_QA_RETRIEVAL_PORT"
_CANONICAL_PORT_PATTERN = re.compile(r"[1-9][0-9]{0,4}")


def resolve_qa_retrieval_endpoints(
    retrieval_url: object,
    retrieval_batch_url: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Apply the per-process QA retriever port without mutating its config.

    Keeping the configured endpoints untouched preserves the collection contract
    across shards, while an explicit environment override lets each shard use a
    local retriever replica. An override fails closed unless both source and
    rewritten endpoints satisfy the same strict local-endpoint contract as the
    checked-in QA config.
    """

    url = str(retrieval_url or "")
    batch_url = str(retrieval_batch_url or "")
    source = os.environ if environ is None else environ
    raw_port = source.get(QA_RETRIEVAL_PORT_ENV)
    if raw_port is None:
        return url, batch_url
    if (
        not isinstance(raw_port, str)
        or _CANONICAL_PORT_PATTERN.fullmatch(raw_port) is None
    ):
        raise ValueError(
            f"{QA_RETRIEVAL_PORT_ENV} must be a canonical decimal port in "
            "the range 1..65535"
        )
    port = int(raw_port)
    if port > 65_535:
        raise ValueError(
            f"{QA_RETRIEVAL_PORT_ENV} must be a canonical decimal port in "
            "the range 1..65535"
        )

    try:
        validated_url, _ = _validate_retrieval_endpoints(
            {"url": url, "batch_url": batch_url}
        )
    except QASummaryConfigError as exc:
        raise ValueError(
            f"{QA_RETRIEVAL_PORT_ENV} requires valid local configured retrieval "
            f"endpoints: {exc}"
        ) from exc
    host = urlparse(validated_url).hostname
    if host is None:  # Covered by validation; retain an explicit fail-closed guard.
        raise ValueError(f"{QA_RETRIEVAL_PORT_ENV} could not resolve a local host")
    rewritten = {
        "url": f"http://{host}:{port}/retrieve",
        "batch_url": f"http://{host}:{port}/retrieve_batch",
    }
    try:
        return _validate_retrieval_endpoints(rewritten)
    except QASummaryConfigError as exc:  # Defensive validation of constructed URLs.
        raise ValueError(
            f"{QA_RETRIEVAL_PORT_ENV} produced invalid local retrieval endpoints: {exc}"
        ) from exc




def _required_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _golden_answers(task: Mapping[str, Any], task_id: str) -> list[str]:
    raw = task.get("golden_answers", task.get("answer"))
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"Task {task_id!r} golden_answers must be a sequence")
    answers = [str(value).strip() for value in values if str(value).strip()]
    if not answers:
        raise ValueError(f"Task {task_id!r} has no golden answers")
    return answers


def _information_observation(
    documents: str,
    *,
    max_chars: int,
) -> tuple[str, bool, int]:
    """Bound retrieved evidence before it reaches either model prompt."""

    content = str(documents or "").strip() or "No results found."
    full = f"<information>\n{content}\n</information>"
    original_chars = len(full)
    if original_chars <= max_chars:
        return full, False, original_chars
    marker = "\n[retrieval truncated]\n"
    prefix = "<information>\n"
    suffix = "</information>"
    budget = max_chars - len(prefix) - len(marker) - len(suffix)
    if budget <= 0:
        raise ValueError("retrieval_observation_chars is too small for audit markers")
    visible = content[:budget].rstrip()
    compact = f"{prefix}{visible}{marker}{suffix}"
    if len(compact) > max_chars:
        raise AssertionError("retrieval observation cap was exceeded")
    return compact, True, original_chars


class QASummaryBatchEnvironment(BatchEnvironmentInterface):
    """Retrieval-backed QA episodes with model-controlled stopping."""

    def __init__(
        self,
        tasks: Sequence[Mapping[str, Any]],
        *,
        retrieval_url: str = "",
        retrieval_batch_url: str = "",
        retrieval_topk: int = 3,
        retrieval_timeout_seconds: float = 600,
        retrieval_observation_chars: int = DEFAULT_RETRIEVAL_OBSERVATION_CHARS,
        max_steps: int = DEFAULT_QA_MAX_STEPS,
        retriever: Retriever | None = None,
    ) -> None:
        if not tasks:
            raise ValueError("QA summary environment requires at least one task")
        self.tasks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for position, raw_task in enumerate(tasks):
            task = dict(raw_task)
            task_id = _required_text(
                task.get("task_id") or task.get("id"),
                f"tasks[{position}].task_id",
            )
            if task_id in seen_ids:
                raise ValueError(f"Duplicate QA task id: {task_id}")
            seen_ids.add(task_id)
            dataset = normalize_qa_dataset(
                task.get("dataset") or task.get("data_source")
            )
            query = _required_text(
                task.get("query") or task.get("question") or task.get("goal"),
                f"task {task_id}.query",
            )
            self.tasks.append(
                {
                    **task,
                    "task_id": task_id,
                    "dataset": dataset,
                    "query": query,
                    "golden_answers": _golden_answers(task, task_id),
                }
            )

        self.batch_size = len(self.tasks)
        self.max_steps = int(max_steps)
        if isinstance(max_steps, bool) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.retrieval_topk = int(retrieval_topk)
        self.retrieval_timeout_seconds = float(retrieval_timeout_seconds)
        self.retrieval_observation_chars = int(retrieval_observation_chars)
        if self.retrieval_topk <= 0 or self.retrieval_timeout_seconds <= 0:
            raise ValueError("retrieval_topk and timeout must be positive")
        if self.retrieval_observation_chars < 256:
            raise ValueError("retrieval_observation_chars must be at least 256")
        self.retrieval_url = str(retrieval_url or "").strip()
        self.retrieval_batch_url = str(retrieval_batch_url or "").strip()
        self._retriever = retriever
        if self._retriever is None and (
            not self.retrieval_url or not self.retrieval_batch_url
        ):
            raise ValueError(
                "QA search actions require both retrieval_url and "
                "retrieval_batch_url, or an injected retriever"
            )

        self._turns: list[int] = []
        self._dones: list[bool] = []
        self._observations: list[str] = []
        self._infos: list[dict[str, Any]] = []
        self.reset()

    def horizon(self, index: int) -> int:
        return qa_horizon(
            self.tasks[index]["dataset"],
            max_steps=self.max_steps,
        )

    def _candidates(self, index: int) -> list[str]:
        if self._dones[index] or self._turns[index] >= self.horizon(index):
            return []
        return candidate_actions_for_step(
            self.tasks[index]["dataset"],
            current_step=self._turns[index] + 1,
            max_steps=self.max_steps,
        )

    def _base_info(self, index: int) -> dict[str, Any]:
        task = self.tasks[index]
        return {
            "task_id": task["task_id"],
            "task_description": task["query"],
            "query": task["query"],
            "dataset": task["dataset"],
            "horizon": self.horizon(index),
            "turn": self._turns[index],
            "admissible_commands": self._candidates(index),
            "won": False,
            "exact_match": False,
            "task_score": 0.0,
        }

    def reset(self) -> tuple[list[str], list[dict[str, Any]]]:
        self._turns = [0] * self.batch_size
        self._dones = [False] * self.batch_size
        self._observations = [
            "No external information retrieved yet." for _ in self.tasks
        ]
        self._infos = [self._base_info(index) for index in range(self.batch_size)]
        return list(self._observations), [dict(info) for info in self._infos]

    def _retrieve_many(self, queries: Sequence[str]) -> list[str]:
        query_rows = [str(query).strip() for query in queries]
        if not query_rows or any(not query for query in query_rows):
            raise ValueError("retrieval queries must be non-empty strings")
        if self._retriever is not None:
            return [
                format_retrieved_documents(self._retriever(query, self.retrieval_topk))
                for query in query_rows
            ]
        payload = post_json(
            self.retrieval_batch_url,
            "",
            {
                "queries": query_rows,
                "topk": self.retrieval_topk,
                "return_scores": False,
            },
            timeout=self.retrieval_timeout_seconds,
        )
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(query_rows):
            raise RuntimeError(
                "QA batch retriever returned a partial response: "
                f"expected={len(query_rows)} actual="
                f"{len(results) if isinstance(results, list) else 'invalid'}"
            )
        return [format_retrieved_documents(result) for result in results]

    def step(
        self,
        actions: list[str],
    ) -> tuple[list[str], list[float], list[bool], list[dict[str, Any]]]:
        if len(actions) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} QA actions, got {len(actions)}"
            )
        rewards = [0.0] * self.batch_size
        decisions: dict[int, tuple[int, Any, bool]] = {}
        search_indexes: list[int] = []
        search_queries: list[str] = []
        for index, raw_action in enumerate(actions):
            if self._dones[index]:
                continue
            current_step = self._turns[index] + 1
            candidates = candidate_actions_for_step(
                self.tasks[index]["dataset"],
                current_step=current_step,
                max_steps=self.max_steps,
            )
            parsed = parse_qa_inner_action(raw_action)
            admissible = qa_action_is_admissible(parsed, candidates)
            decisions[index] = (current_step, parsed, admissible)
            if (
                parsed.format_valid
                and admissible
                and parsed.kind == "search"
                and current_step < self.horizon(index)
            ):
                search_indexes.append(index)
                search_queries.append(parsed.value)
        retrieved_rows = self._retrieve_many(search_queries) if search_queries else []
        retrieved_by_index = dict(zip(search_indexes, retrieved_rows, strict=True))

        for index in range(self.batch_size):
            if self._dones[index]:
                continue
            current_step, parsed, admissible = decisions[index]
            self._turns[index] = current_step
            info = self._base_info(index)
            info.update(
                {
                    "action_kind": parsed.kind,
                    "action_format_valid": parsed.format_valid,
                    "action_was_admissible": admissible,
                }
            )

            if parsed.format_valid and admissible and parsed.kind == "answer":
                exact_match = searchqa_exact_match(
                    parsed.value,
                    self.tasks[index]["golden_answers"],
                )
                rewards[index] = float(exact_match)
                self._dones[index] = True
                self._observations[index] = "Answer submitted."
                info.update(
                    {
                        "final_answer": parsed.value,
                        "exact_match": exact_match,
                        "won": exact_match,
                        "task_score": float(exact_match),
                        "termination_reason": "answer_submitted",
                    }
                )
            elif parsed.format_valid and admissible and parsed.kind == "search":
                info["search_query"] = parsed.value
                if current_step >= self.horizon(index):
                    info["retrieval_performed"] = False
                else:
                    documents = retrieved_by_index[index]
                    observation, truncated, original_chars = _information_observation(
                        documents,
                        max_chars=self.retrieval_observation_chars,
                    )
                    self._observations[index] = observation
                    info.update(
                        {
                            "retrieval_performed": True,
                            "retrieval_truncated": truncated,
                            "retrieval_original_chars": original_chars,
                            "retrieval_visible_chars": len(observation),
                            "retrieval_observation_chars": (
                                self.retrieval_observation_chars
                            ),
                        }
                    )
            else:
                self._observations[index] = (
                    "Invalid action. Select exactly one supplied answer[...] or "
                    "search[...] candidate inside one <action>...</action> wrapper."
                )

            if not self._dones[index] and current_step >= self.horizon(index):
                self._dones[index] = True
                self._observations[
                    index
                ] = "Action limit reached without a valid final answer."
                info["termination_reason"] = "max_steps"

            info["turn"] = self._turns[index]
            info["admissible_commands"] = self._candidates(index)
            self._infos[index] = info

        return (
            list(self._observations),
            rewards,
            list(self._dones),
            [dict(info) for info in self._infos],
        )


def make_qa_summary_batch_env(
    cfg: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    seed: int,
    *,
    retriever: Retriever | None = None,
) -> QASummaryBatchEnvironment:
    """Construct the QA environment from the isolated collection config."""

    del seed
    qa_cfg = cfg.get("qa_summary") or {}
    if not isinstance(qa_cfg, Mapping):
        raise ValueError("qa_summary config must be a mapping")

    environment = qa_cfg.get("environment")
    if environment is None:
        benchmark = cfg.get("benchmark") or {}
        environment = (
            benchmark.get("environment") if isinstance(benchmark, Mapping) else {}
        )
    if environment is None:
        environment = {}
    if not isinstance(environment, Mapping):
        raise ValueError("qa_summary.environment must be a mapping")

    retrieval = cfg.get("retrieval") or {}
    if not isinstance(retrieval, Mapping):
        raise ValueError("retrieval config must be a mapping")
    datasets = cfg.get("datasets") or {}
    if not isinstance(datasets, Mapping):
        raise ValueError("datasets config must be a mapping")
    nq_cfg = datasets.get("nq") or {}
    hotpot_cfg = datasets.get("hotpotqa") or {}
    if not isinstance(nq_cfg, Mapping) or not isinstance(hotpot_cfg, Mapping):
        raise ValueError("datasets.nq and datasets.hotpotqa must be mappings")
    sampling = cfg.get("sampling") or {}
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling config must be a mapping")
    raw_max_steps = sampling.get(
        "max_steps",
        qa_cfg.get("max_steps", DEFAULT_QA_MAX_STEPS),
    )
    if isinstance(raw_max_steps, bool):
        raise ValueError("sampling.max_steps must be a positive integer")
    max_steps = int(raw_max_steps)
    if max_steps <= 0:
        raise ValueError("sampling.max_steps must be a positive integer")
    dataset_horizons = {
        int(item.get("max_steps", max_steps)) for item in (nq_cfg, hotpot_cfg)
    }
    if dataset_horizons != {max_steps}:
        raise ValueError("NQ and HotpotQA max_steps must both match sampling.max_steps")

    retrieval_url = str(
        retrieval.get("url", environment.get("retrieval_url")) or ""
    )
    retrieval_batch_url = str(
        retrieval.get("batch_url", environment.get("retrieval_batch_url")) or ""
    )
    if retriever is None:
        retrieval_url, retrieval_batch_url = resolve_qa_retrieval_endpoints(
            retrieval_url,
            retrieval_batch_url,
        )

    return QASummaryBatchEnvironment(
        tasks,
        retrieval_url=retrieval_url,
        retrieval_batch_url=retrieval_batch_url,
        retrieval_topk=int(retrieval.get("topk", environment.get("retrieval_topk", 3))),
        retrieval_timeout_seconds=float(
            retrieval.get(
                "timeout_seconds",
                environment.get("retrieval_timeout_seconds", 600),
            )
        ),
        retrieval_observation_chars=int(
            (cfg.get("env") or {}).get(
                "retrieval_observation_chars",
                DEFAULT_RETRIEVAL_OBSERVATION_CHARS,
            )
        ),
        max_steps=max_steps,
        retriever=retriever,
    )


__all__ = [
    "DEFAULT_RETRIEVAL_OBSERVATION_CHARS",
    "QA_RETRIEVAL_PORT_ENV",
    "QASummaryBatchEnvironment",
    "Retriever",
    "make_qa_summary_batch_env",
    "resolve_qa_retrieval_endpoints",
]
