"""Distill training trajectories into the three-field training skill format."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def load_config(path: Path, domain: str) -> dict:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Summarizer configuration must be a mapping")
    config = copy.deepcopy({k: v for k, v in payload.items() if k != "domains"})
    profile = payload.get("domains", {}).get(domain, {})
    for key, value in profile.items():
        if isinstance(value, dict):
            config.setdefault(key, {}).update(value)
        else:
            config[key] = value
    return config


def read_rollouts(path: Path, dataset: str) -> list[dict]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {number} must contain a JSON object")
        task_id = str(row.get("task_id") or row.get("id") or "").strip()
        if not task_id or task_id in seen:
            raise ValueError(f"Line {number} needs a unique, non-empty task_id")
        seen.add(task_id)
        split = str(row.get("source_split") or row.get("split") or "train").strip()
        if split != "train":
            raise ValueError(f"Line {number} is not a training trajectory")
        source_dataset = str(row.get("dataset") or dataset).lower()
        source_dataset = {"hotpotqa": "multihop", "naturalquestions": "nq",
                          "natural_questions": "nq"}.get(source_dataset, source_dataset)
        if source_dataset != dataset:
            raise ValueError(f"Line {number} dataset does not match --dataset")
        row["task_id"] = task_id
        row.setdefault("task_type", dataset)
        goal = str(row.get("canonical_goal") or row.get("query") or
                   row.get("question") or row.get("goal") or "").strip()
        if not goal:
            raise ValueError(f"Line {number} needs canonical_goal, query, or goal")
        row["canonical_goal"] = goal
        if dataset in {"nq", "multihop"}:
            row.setdefault("query", goal)
        if dataset == "alfworld":
            skill = row.setdefault("skill_used", "")
            if isinstance(skill, dict):
                row["skill_used"] = json.dumps(skill, ensure_ascii=False)
            if "assignment_metadata" not in row:
                if row["skill_used"]:
                    raise ValueError("An ALFWorld skilled rollout needs its original assignment_metadata")
                row.setdefault("assignment_source", "iterative_collection_v1")
                row["assignment_metadata"] = {
                    "schema_version": "collection_assignment_v1",
                    "source_kind": "unskilled_initial",
                    "collection_round": 0,
                    "skill_sha256": None,
                    "generated_from_round": None,
                }
        rows.append(row)
    if not rows:
        raise ValueError("Input contains no trajectories")
    return rows


def validate_rollouts(rows: list[dict], domain: str, cfg: dict) -> None:
    max_steps = int(cfg["models"]["summarizer_max_trajectory_steps"])
    if domain == "qa":
        from .qa import render_qa_summary_prompt
        for row in rows:
            render_qa_summary_prompt(row, max_steps=max_steps)
    elif domain == "webshop":
        from .webshop_prompts import render_webshop_summary_prompt
        limits = {k: v for k, v in cfg.get("summary", {}).items()
                  if k != "max_prompt_tokens"}
        for row in rows:
            render_webshop_summary_prompt(row, max_steps=max_steps, **limits)
    else:
        from ._alfworld_assignment import validate_collection_assignment_rollout
        from ._alfworld_validation import extract_goal_slots
        from .base import SkillSummarizer
        renderer = SkillSummarizer.__new__(SkillSummarizer)
        renderer.max_steps = max_steps
        for row in rows:
            validate_collection_assignment_rollout(row)
            extract_goal_slots(row["task_type"], row["canonical_goal"])
            renderer._raw_prompt(row)


def summarize(rows: list[dict], domain: str, cfg: dict) -> list[dict]:
    if domain == "qa":
        from .qa import QASkillSummarizer as Summarizer
    elif domain == "webshop":
        from .webshop import WebShopSkillSummarizer as Summarizer
    else:
        from .base import SkillSummarizer as Summarizer

    # A successful ALFWorld rollout already using a skill publishes that skill,
    # exactly as in the collection algorithm; it is not distilled a second time.
    skills = [None] * len(rows)
    pending = []
    for index, row in enumerate(rows):
        if domain == "alfworld" and row["success"] and row.get("skill_used"):
            from ._alfworld_validation import validate_executable_skill
            skills[index] = validate_executable_skill(json.loads(row["skill_used"]))
        else:
            pending.append(index)
    if pending:
        model = Summarizer(cfg)
        try:
            results = model.summarize_batch([rows[index] for index in pending])
        finally:
            model.close()
        if len(results) != len(pending):
            raise RuntimeError("Summarizer returned an incomplete result batch")
        failed = []
        for index, result in zip(pending, results):
            skill = result if domain == "alfworld" else (result.skill if result.ok else None)
            if skill is None:
                failed.append(rows[index]["task_id"])
            else:
                skills[index] = skill
        if failed:
            raise RuntimeError(
                f"Summarization failed for {len(failed)}/{len(rows)} trajectories; "
                f"no training file was written. Failed task IDs: {', '.join(failed)}"
            )
    return skills


def training_records(rows: list[dict], skills: list[dict], dataset: str) -> list[dict]:
    if len(rows) != len(skills) or any(not isinstance(skill, dict) for skill in skills):
        raise ValueError("Each trajectory must have an accepted skill")
    records = []
    for row, skill in zip(rows, skills):
        text = json.dumps(skill, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        records.append({
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "canonical_goal": row["canonical_goal"],
            "executable_skill": text,
            "skill_id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "dataset": dataset,
            "source_split": "train",
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("alfworld", "nq", "multihop", "webshop"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Training trajectory JSONL")
    parser.add_argument("--output", type=Path, help="New training skill JSONL (never overwritten)")
    parser.add_argument("--config", type=Path, default=Path("configs/summarizer.yaml"))
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs without loading model weights")
    args = parser.parse_args()
    if not args.validate_only and args.output is None:
        parser.error("--output is required unless --validate-only is used")
    if args.output and args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    domain = "qa" if args.dataset in {"nq", "multihop"} else args.dataset
    config = load_config(args.config, domain)
    rows = read_rollouts(args.input, args.dataset)
    validate_rollouts(rows, domain, config)
    if args.validate_only:
        print(f"Validated {len(rows)} {args.dataset} training trajectories; no model loaded.")
        return
    records = training_records(rows, summarize(rows, domain, config), args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} accepted training skills to {args.output}")


if __name__ == "__main__":
    main()
