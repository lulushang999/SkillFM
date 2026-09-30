from __future__ import annotations

import hashlib
import random
from typing import Any, Collection, Mapping, Sequence

from skillflow.skills.executable import canonicalize_executable_skill


SemanticGroup = str
SUCCESS_SKILL_PROVENANCE = "verified_terminal_success_v1"
BENCHMARK_CURATED_SKILL_PROVENANCE = "curated_static_binding_v1"
BENCHMARK_VERIFIED_SKILL_PROVENANCE = "environment_verified_terminal_skill_v1"
CODEC_SKILL_PROVENANCES = frozenset(
    {
        SUCCESS_SKILL_PROVENANCE,
        BENCHMARK_CURATED_SKILL_PROVENANCE,
        BENCHMARK_VERIFIED_SKILL_PROVENANCE,
    }
)


def _skill_digest(value: object) -> str:
    canonical = canonicalize_executable_skill(value)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metadata_id(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return -1
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:7], "big")


def validate_codec_skill_example(example: Mapping[str, object]) -> None:
    """Validate the shared Codec row shape and explicit source provenance."""

    provenance = str(example.get("skill_provenance") or "")
    if provenance not in CODEC_SKILL_PROVENANCES:
        raise ValueError(
            "Codec row lacks terminal-success provenance or a supported "
            f"benchmark provenance: {provenance!r}"
        )
    if not str(example.get("canonical_goal") or "").strip():
        raise ValueError("Codec row requires canonical_goal for decoder conditioning")
    skill_text = str(example.get("skill_text") or "").strip()
    digest = _skill_digest(skill_text)
    if str(example.get("skill_sha256") or "") != digest:
        raise ValueError("Codec row skill_sha256 does not match canonical skill text")
    if not str(example.get("task_type") or "").strip():
        raise ValueError("Codec row requires task_type provenance")
    if int(example.get("chain_step", -1)) != 1:
        raise ValueError("Codec row must come from the terminal success step")
    if not str(example.get("node_id") or "").strip():
        raise ValueError("Codec row requires a terminal success node_id")
    source_count = example.get("source_count")
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count <= 0
    ):
        raise ValueError("Codec row source_count must be a positive integer")


def validate_success_skill_example(example: Mapping[str, object]) -> None:
    """Validate one Codec row derived from an ALFWorld terminal success."""

    if example.get("skill_provenance") != SUCCESS_SKILL_PROVENANCE:
        raise ValueError("Codec row lacks verified terminal-success provenance")
    validate_codec_skill_example(example)


def _validate_codec_split_example(
    example: Mapping[str, object],
    *,
    allowed_provenances: Collection[str] | None,
) -> None:
    validate_codec_skill_example(example)
    allowed = (
        frozenset({SUCCESS_SKILL_PROVENANCE})
        if allowed_provenances is None
        else frozenset(str(item).strip() for item in allowed_provenances)
    )
    if not allowed:
        raise ValueError("Codec split provenance allowlist must not be empty")
    unsupported = allowed - CODEC_SKILL_PROVENANCES
    if unsupported:
        raise ValueError(
            f"Codec split allowlist has unsupported provenances: {sorted(unsupported)}"
        )
    observed = str(example.get("skill_provenance") or "")
    if observed not in allowed:
        raise ValueError(
            "Codec split row provenance is not enabled for this source mode: "
            f"{observed!r}"
        )


def build_success_skill_examples(corpus: Any) -> list[dict]:
    """Materialize unique verified terminal-success skill/condition pairs.

    The full frozen chain is still validated by load_chain_corpus before this
    function is called. Codec supervision deliberately ignores every failure
    transition, intermediate revision, and step-zero no-skill/source-anchor node.
    Task metadata conditions the decoder prompt, but never enters the trainable
    skill encoder or the frozen query-encoder target.

    The accepted frozen-chain shape is exactly ``unskilled -> success_skill``.
    Any other shape is rejected so default, failure, and intermediate nodes
    cannot enter codec training.
    """

    examples: list[dict] = []
    # A canonical skill can be valid under several task conditions. Keep one
    # row per (skill, condition) so the contextual decoder actually observes
    # every supported goal, while the semantic split below still groups all
    # copies of the same skill on one side to prevent text leakage.
    by_skill_condition: dict[tuple[str, str, str], dict] = {}

    for chain in corpus.chains:
        terminal_step = int(chain.absorbing_from_step)
        if terminal_step != 1 or len(chain.nodes) != 2:
            raise ValueError(
                "Codec chains must contain exactly an unskilled source and a "
                "step-1 terminal success skill: "
                f"chain={chain.chain_id!r} absorbing_step={terminal_step} "
                f"node_count={len(chain.nodes)}"
            )
        source_node = chain.nodes[0]
        node = chain.nodes[terminal_step]
        if (
            int(source_node.step) != 0
            or str(source_node.node_type) != "unskilled"
            or int(node.step) != 1
            or str(node.node_type) != "success_skill"
        ):
            raise ValueError(
                "Codec chains must use node types unskilled -> success_skill: "
                f"chain={chain.chain_id!r}"
            )
        skill_text = canonicalize_executable_skill(node.skill_text)

        task_type = str(chain.task_type).strip()
        canonical_goal = str(chain.canonical_goal).strip()
        skill_condition = (skill_text, task_type, canonical_goal)
        existing = by_skill_condition.get(skill_condition)
        if existing is not None:
            existing["source_count"] = int(existing["source_count"]) + 1
            existing["source_node_ids"].append(str(node.node_id))
            existing["source_chain_ids"].append(str(chain.chain_id))
            continue

        digest = _skill_digest(skill_text)
        row = {
            "skill_text": skill_text,
            "skill_provenance": SUCCESS_SKILL_PROVENANCE,
            "task_type": task_type,
            "canonical_goal": canonical_goal,
            "group_id": str(chain.task_id),
            "chain_id": str(chain.chain_id),
            "chain_label": _metadata_id(chain.chain_id),
            "chain_step": terminal_step,
            "slot_label": _metadata_id(digest),
            "example_label": _metadata_id(digest),
            "node_id": str(node.node_id),
            "skill_sha256": digest,
            "source_count": 1,
            "source_node_ids": [str(node.node_id)],
            "source_chain_ids": [str(chain.chain_id)],
        }
        examples.append(row)
        by_skill_condition[skill_condition] = row

    if not examples:
        raise ValueError("No verified terminal-success skills found for codec training")
    return examples


def codec_semantic_group(example: Mapping[str, object]) -> SemanticGroup:
    """Return the exact skill identity that must stay on one split side.

    The trainable skill encoder sees only skill text. Grouping by task metadata
    would leak an identical encoder input across train and validation.
    """
    skill_text = str(example.get("skill_text") or "").strip()
    if not skill_text:
        raise ValueError("Codec skill grouping requires non-empty skill_text")
    return _skill_digest(skill_text)


def split_examples_by_semantic_group(
    examples: Sequence[dict],
    *,
    validation_ratio: float,
    seed: int,
    allowed_provenances: Collection[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Hold out whole canonical-skill groups from a success-only corpus."""

    ratio = min(0.5, max(0.0, float(validation_ratio)))
    learned_examples = list(examples)
    if not learned_examples:
        raise ValueError("Codec dataset requires at least one supervised skill")
    for row in learned_examples:
        _validate_codec_split_example(row, allowed_provenances=allowed_provenances)
    if ratio <= 0:
        return learned_examples, []

    grouped: dict[SemanticGroup, list[dict]] = {}
    for row in learned_examples:
        semantic_group = codec_semantic_group(row)
        grouped.setdefault(semantic_group, []).append(row)
    if not grouped:
        raise ValueError("No codec skill groups available for validation split")

    strata: dict[str, list[SemanticGroup]] = {}
    for semantic_group, rows in grouped.items():
        counts: dict[str, int] = {}
        for row in rows:
            task_type = str(row.get("task_type") or "").strip()
            if not task_type:
                raise ValueError("Every learned codec skill needs task_type provenance")
            counts[task_type] = counts.get(task_type, 0) + 1
        stratum = max(sorted(counts), key=lambda value: counts[value])
        strata.setdefault(stratum, []).append(semantic_group)

    rng = random.Random(int(seed))
    validation_groups: set[SemanticGroup] = set()
    for key in sorted(strata):
        semantic_groups = sorted(strata[key])
        rng.shuffle(semantic_groups)
        n_validation = int(round(len(semantic_groups) * ratio))
        if len(semantic_groups) >= 2:
            n_validation = min(
                len(semantic_groups) - 1,
                max(1, n_validation),
            )
        else:
            n_validation = 0
        validation_groups.update(semantic_groups[:n_validation])

    train: list[dict] = []
    validation: list[dict] = []
    for semantic_group, rows in grouped.items():
        (validation if semantic_group in validation_groups else train).extend(rows)
    if not train or not validation:
        raise ValueError(
            f"Invalid skill-group split: train={len(train)} "
            f"validation={len(validation)} groups={len(grouped)} ratio={ratio}"
        )
    return train, validation


def assert_codec_split_isolation(
    train: Sequence[Mapping[str, object]],
    validation: Sequence[Mapping[str, object]],
    *,
    allowed_provenances: Collection[str] | None = None,
) -> None:
    try:
        for row in (*train, *validation):
            _validate_codec_split_example(row, allowed_provenances=allowed_provenances)
    except ValueError as exc:
        raise RuntimeError("Codec split contains invalid skill provenance") from exc
    if not validation:
        return

    train_skills = {codec_semantic_group(row) for row in train}
    validation_skills = {codec_semantic_group(row) for row in validation}
    overlap_skills = train_skills & validation_skills
    if overlap_skills:
        raise RuntimeError(
            "Codec split leakage detected: "
            f"overlapping_skill_groups={len(overlap_skills)}"
        )
