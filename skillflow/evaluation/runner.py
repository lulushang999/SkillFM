"""Evaluate one generated skill per task, with a complete denominator."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import random
from typing import Any

from .skill_text import canonical_skill_text, parse_executable_skill

DOMAINS = ("alfworld", "nq", "multihop", "webshop")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                rows.append(value)
    return rows


def prepare_inputs(domain: str, tasks: list[dict], assignments: list[dict]):
    if not tasks:
        raise ValueError("The evaluation task list is empty")
    by_id = {}
    for row in assignments:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id or task_id in by_id:
            raise ValueError(f"Missing or duplicate assignment task_id: {task_id!r}")
        # A release assignment may declare NFE; never silently use another value.
        for key in ("nfe", "sampling_steps", "flow_forward_calls"):
            if key in row and row[key] != 1:
                raise ValueError(f"Assignment {task_id} must use NFE=1")
        by_id[task_id] = row
    normalized = []
    seen = set()
    for source in tasks:
        row = dict(source)
        task_id = str(row.get("task_id") or row.get("id") or "").strip()
        if not task_id or task_id in seen:
            raise ValueError(f"Missing or duplicate task_id: {task_id!r}")
        seen.add(task_id)
        row["task_id"] = task_id
        row.setdefault("source_split", "test")
        goal = row.get("goal") or row.get("query") or row.get("question") or row.get("canonical_goal")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(f"Task {task_id} requires a public goal/question")
        row["goal"] = goal.strip()
        if domain in {"nq", "multihop"}:
            from .qa_prompts import normalize_qa_dataset
            row["query"] = goal.strip()
            row["dataset"] = normalize_qa_dataset(row.get("dataset") or ("nq" if domain == "nq" else "hotpotqa"))
            answers = row.get("golden_answers", row.get("answer"))
            if isinstance(answers, str):
                answers = [answers]
            if not isinstance(answers, list) or not answers or any(not isinstance(x, str) or not x.strip() for x in answers):
                raise ValueError(f"Task {task_id} requires non-empty golden_answers")
            row["golden_answers"] = answers
        elif domain == "alfworld" and not row.get("game_file"):
            raise ValueError(f"Task {task_id} requires game_file")
        elif domain == "webshop":
            if isinstance(row.get("session"), bool) or not isinstance(row.get("session"), int):
                raise ValueError(f"Task {task_id} requires integer WebShop session")
        normalized.append(row)
    unknown = set(by_id) - seen
    if unknown:
        raise ValueError(f"Assignments contain {len(unknown)} unknown task IDs")
    return normalized, by_id


def resolve_skill(task: dict, assignments: dict, domain: str):
    task_id = task["task_id"]
    row = assignments.get(task_id)
    if row is None:
        return None, None, "missing_skill_assignment"
    if str(row.get("status", "ok")) not in {"ok", "success", "generated"}:
        return None, None, str(row.get("failure_reason") or "skill_generation_failed")
    try:
        skill = parse_executable_skill(row.get("skill"))
    except ValueError:
        return None, None, "invalid_generated_skill"
    receipt = None
    if domain == "multihop":
        from .grounded import safe_skill
        skill, receipt = safe_skill(task["query"], skill)
    return skill, receipt, None


def failure_row(task: dict, reason: str, *, attempted: bool = False, generation_failed: bool = True) -> dict:
    return {
        "task_id": task["task_id"], "dataset": task.get("dataset", ""),
        "source_split": task["source_split"], "goal": task["goal"],
        "success": False, "exact_match": False, "score": 0.0,
        "done": False, "evaluation_complete": attempted,
        "rollout_attempted": attempted, "failure_reason": reason,
        "skill_injected": False, "skill_generation_failed": generation_failed,
        "nfe": 1, "num_steps": 0, "steps": [],
    }


class _PromptRecorder:
    def __init__(self, *args, **kwargs):
        self.actual_prompts = defaultdict(list)
        self._current_prompts = []
        super().__init__(*args, **kwargs)

    def _prompt(self, *args, **kwargs):
        prompt = super()._prompt(*args, **kwargs)
        self._current_prompts.append(prompt)
        return prompt

    def generate_batch(self, items):
        self._current_prompts = []
        result = super().generate_batch(items)
        if len(self._current_prompts) != len(items):
            raise RuntimeError("Actor did not produce exactly one prompt per task")
        for item, prompt in zip(items, self._current_prompts):
            self.actual_prompts[str(item["task_id"])].append(prompt)
        return result


def make_actor(domain: str, cfg: dict):
    if domain in {"nq", "multihop"}:
        from .qa_actor import QASummaryActionGenerator as Base
    elif domain == "webshop":
        from .webshop_actor import BenchmarkActionGenerator as Base
    else:
        from .actor_base import ActionGenerator as Base
    class RecordingActor(_PromptRecorder, Base):
        pass
    return RecordingActor(cfg)


def _make_env(domain: str, cfg: dict, tasks: list[dict], seed: int):
    if domain == "alfworld":
        from .alfworld_env import ALFWorldBatchInterface
        setting = cfg.get("environment") or {}
        return ALFWorldBatchInterface(
            setting["config"], [row["game_file"] for row in tasks], seed,
            train_eval=str(setting.get("train_eval", "eval_in_distribution")),
        )
    from .webshop_env import WebShopHTTPBatchInterface
    setting = cfg.get("environment") or {}
    return WebShopHTTPBatchInterface(
        tasks, bridge_url=setting["bridge_url"], seed=seed,
        timeout_seconds=float(setting.get("timeout_seconds", 600)),
        expected_runtime_identity=setting.get("expected_runtime_identity"),
    )


def run_environment_batch(domain: str, cfg: dict, actor, tasks: list[dict], skills: dict, *, seed: int, env_factory=None):
    """ALFWorld/WebShop action loop with the original candidate-action policy."""
    env = (env_factory or _make_env)(domain, cfg, tasks, seed)
    try:
        observations, infos = env.reset()
        size = len(tasks)
        if len(observations) != size or len(infos) != size:
            raise RuntimeError("Environment reset returned a partial batch")
        if domain == "webshop":
            for task, info in zip(tasks, infos):
                actual_goal = str(info.get("task_description") or "").strip()
                if actual_goal != task["goal"]:
                    raise ValueError(f"WebShop session/goal mismatch for {task['task_id']}")
        histories = [[] for _ in tasks]
        done, rewards = [False] * size, [0.0] * size
        horizon = int(cfg["sampling"]["max_steps"])
        for _ in range(horizon):
            active = [i for i in range(size) if not done[i]]
            if not active:
                break
            items = []
            for i in active:
                candidates = [str(x) for x in infos[i].get("admissible_commands", [])]
                if not candidates:
                    raise RuntimeError(f"No available actions for {tasks[i]['task_id']}")
                items.append({
                    "task_id": tasks[i]["task_id"], "goal": tasks[i]["goal"],
                    "skill": canonical_skill_text(skills[tasks[i]["task_id"]]),
                    "observation": str(observations[i]), "admissible": candidates,
                    "environment_admissible": candidates, "history": histories[i],
                })
            decisions = actor.generate_batch(items)
            if len(decisions) != len(items):
                raise RuntimeError("Actor returned a partial batch")
            actions, metadata = [""] * size, {}
            for i, item, decision in zip(active, items, decisions):
                executed = bool(decision.format_valid and decision.action_is_admissible and not decision.generation_incomplete)
                actions[i] = str(decision.action) if executed else "__invalid_action__"
                metadata[i] = {
                    "model_output": decision.response, "proposed_action": decision.action,
                    "action_extraction_mode": decision.extraction_mode,
                    "action_format_valid": bool(decision.format_valid),
                    "action_was_admissible": bool(decision.action_is_admissible),
                    "generation_incomplete": bool(decision.generation_incomplete),
                    "action_executed": executed, "admissible_actions": item["admissible"],
                }
            next_obs, next_rewards, next_done, next_infos = env.step(actions)
            if any(len(value) != size for value in (next_obs, next_rewards, next_done, next_infos)):
                raise RuntimeError("Environment step returned a partial batch")
            for i in active:
                histories[i].append({
                    "step": len(histories[i]) + 1, "observation": str(observations[i]),
                    "action": actions[i], "next_observation": str(next_obs[i]),
                    "reward": float(next_rewards[i]), "done": bool(next_done[i]),
                    **metadata[i],
                })
                rewards[i], done[i], infos[i] = float(next_rewards[i]), bool(next_done[i]), dict(next_infos[i])
            observations = next_obs
        rows = []
        for i, task in enumerate(tasks):
            score = float(infos[i].get("task_score", rewards[i]))
            success = bool(infos[i].get("won", score == 1.0))
            invalid_count = sum(not step["action_executed"] for step in histories[i])
            rows.append({
                "task_id": task["task_id"], "dataset": task.get("dataset", domain),
                "source_split": task["source_split"], "goal": task["goal"],
                "success": success, "score": score, "done": done[i],
                "evaluation_complete": True, "rollout_attempted": True,
                "skill_injected": True, "skill": skills[task["task_id"]],
                "num_steps": len(histories[i]), "invalid_action_count": invalid_count,
                "final_observation": str(observations[i]), "final_info": infos[i],
                "failure_reason": "" if success else ("horizon_exhausted" if not done[i] else "environment_failure"),
                "steps": histories[i],
            })
        return rows
    finally:
        env.close()


def aggregate(rows: list[dict], domain: str) -> dict:
    def metrics(group):
        count = len(group)
        success = sum(bool(row.get("success")) for row in group)
        return {
            "count": count, "success": success, "success_rate": success / count,
            "mean_score": sum(float(row.get("score", 0.0)) for row in group) / count,
            "rollouts_attempted": sum(bool(row.get("rollout_attempted")) for row in group),
            "generation_failures": sum(bool(row.get("skill_generation_failed")) for row in group),
            "evaluation_aborted": sum(str(row.get("failure_reason", "")).startswith("evaluation_aborted:") for row in group),
        }
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("dataset") or domain)].append(row)
    return {"domain": domain, "method": "classic_codec_imf", "nfe": 1,
            **metrics(rows), "datasets": {key: metrics(value) for key, value in sorted(groups.items())}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", required=True, help="Evaluation tasks; never read during training")
    parser.add_argument("--assignments", required=True, help="JSONL with task_id and generated skill")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(cfg, dict) or cfg.get("domain") != args.domain:
        raise ValueError("Actor config domain does not match --domain")
    if cfg.get("nfe") != 1:
        raise ValueError("The anonymous main-method release requires NFE=1")
    tasks, assignments = prepare_inputs(args.domain, read_jsonl(args.tasks), read_jsonl(args.assignments))
    resolved = {row["task_id"]: resolve_skill(row, assignments, args.domain) for row in tasks}
    if args.validate_only:
        print(json.dumps({"tasks": len(tasks), "valid_skills": sum(value[0] is not None for value in resolved.values()), "nfe": 1}))
        return 0
    output = Path(args.output)
    summary_path = output.with_suffix(".summary.json")
    failure_path = output.with_suffix(".failures.jsonl")
    for path in (output, summary_path, failure_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    seed = int((cfg.get("run") or {}).get("seed", 42))
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rows_by_id = {}
    valid_tasks = []
    for task in tasks:
        skill, _, error = resolved[task["task_id"]]
        if error:
            rows_by_id[task["task_id"]] = failure_row(task, error)
        else:
            valid_tasks.append(task)
    batch_size = int((cfg.get("rollout") or {}).get("env_batch_size", 1))
    if batch_size <= 0:
        raise ValueError("rollout.env_batch_size must be positive")
    if args.domain == "multihop":
        from .grounded import prompt_runtime
        runtime = prompt_runtime()
    else:
        runtime = nullcontext()
    actor, error = None, None
    attempted_ids = set()
    try:
        with runtime:
            if valid_tasks:
                actor = make_actor(args.domain, cfg)
            for start in range(0, len(valid_tasks), batch_size):
                batch = valid_tasks[start:start + batch_size]
                skills = {task["task_id"]: resolved[task["task_id"]][0] for task in batch}
                attempted_ids.update(task["task_id"] for task in batch)
                if args.domain in {"nq", "multihop"}:
                    from .qa_runtime import run_qa_summary_batch
                    batch_rows = run_qa_summary_batch(cfg, actor, batch, skills, seed=seed)
                else:
                    batch_rows = run_environment_batch(args.domain, cfg, actor, batch, skills, seed=seed)
                for row in batch_rows:
                    task_id = row["task_id"]
                    row.update(nfe=1, rollout_attempted=True, evaluation_complete=True)
                    receipt = resolved[task_id][1]
                    if receipt is not None:
                        row["grounded_skill_policy"] = receipt
                    prompts = actor.actual_prompts.pop(task_id, [])
                    if len(prompts) != len(row["steps"]):
                        raise RuntimeError("Prompt/trajectory length mismatch")
                    for step, prompt in zip(row["steps"], prompts):
                        step["actor_prompt"] = prompt
                    rows_by_id[task_id] = row
                print(json.dumps({"completed": len(rows_by_id), "total": len(tasks)}), flush=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if actor is not None:
            try:
                actor.close()
            except Exception as exc:
                error = error or f"{type(exc).__name__}: {exc}"
    for task in tasks:
        if task["task_id"] not in rows_by_id:
            row = failure_row(task, "evaluation_aborted: " + str(error),
                              attempted=task["task_id"] in attempted_ids, generation_failed=False)
            row["evaluation_complete"] = False
            if actor is not None:
                row["actor_prompts_before_error"] = actor.actual_prompts.get(task["task_id"], [])
            rows_by_id[task["task_id"]] = row
    rows = [rows_by_id[task["task_id"]] for task in tasks]
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with failure_path.open("x", encoding="utf-8") as stream:
        for row in rows:
            if not row["success"]:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = aggregate(rows, args.domain)
    summary.update(status="failed" if error else "complete", error=error)
    with summary_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if error else 0
