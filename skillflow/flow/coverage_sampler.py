from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


PAIR_COVERAGE_SAMPLER_SCHEMA_VERSION = (
    "deterministic_ddp_full_pair_coverage_then_weighted_replacement_v1"
)


def _positive_int(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class CoverageBatch:
    indices: np.ndarray
    valid_mask: np.ndarray


class CoverageFirstDDPSampler:
    """Deterministically shard one global pair permutation across DDP ranks."""

    def __init__(
        self,
        *,
        pair_count: int,
        local_batch_size: int,
        world_size: int,
        seed: int,
    ) -> None:
        self.pair_count = _positive_int(pair_count, label="pair_count")
        self.local_batch_size = _positive_int(
            local_batch_size, label="local_batch_size"
        )
        self.world_size = _positive_int(world_size, label="world_size")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed = int(seed)
        self.global_batch_size = self.local_batch_size * self.world_size
        self.coverage_steps = int(
            math.ceil(self.pair_count / self.global_batch_size)
        )
        self.global_permutation = np.random.default_rng(self.seed).permutation(
            self.pair_count
        ).astype(np.int64, copy=False)

    def batch(self, *, step: int, rank: int) -> CoverageBatch:
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("coverage step must be an integer")
        if not 0 <= step < self.coverage_steps:
            raise ValueError(
                f"coverage step {step} is outside [0, {self.coverage_steps})"
            )
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("rank must be an integer")
        if not 0 <= rank < self.world_size:
            raise ValueError(f"rank {rank} is outside [0, {self.world_size})")
        start = (
            step * self.global_batch_size + rank * self.local_batch_size
        )
        positions = start + np.arange(self.local_batch_size, dtype=np.int64)
        valid = positions < self.pair_count
        # Invalid tail slots are deterministic placeholders. The training loop
        # replaces only those slots with hierarchy-weighted random samples.
        indices = self.global_permutation[positions % self.pair_count]
        return CoverageBatch(
            indices=indices.astype(np.int64, copy=False),
            valid_mask=valid.astype(np.bool_, copy=False),
        )


def build_coverage_evidence(
    sampler: CoverageFirstDDPSampler,
) -> dict[str, int | bool | str]:
    counts = np.zeros(sampler.pair_count, dtype=np.int64)
    valid_slot_count = 0
    for step in range(sampler.coverage_steps):
        for rank in range(sampler.world_size):
            batch = sampler.batch(step=step, rank=rank)
            valid_indices = batch.indices[batch.valid_mask]
            counts[valid_indices] += 1
            valid_slot_count += int(batch.valid_mask.sum())
    covered = int((counts > 0).sum())
    duplicates = int((counts > 1).sum())
    complete = (
        valid_slot_count == sampler.pair_count
        and covered == sampler.pair_count
        and duplicates == 0
        and bool(np.all(counts == 1))
    )
    if not complete:
        raise AssertionError(
            "Deterministic DDP coverage schedule does not cover every pair once"
        )
    total_slots = sampler.coverage_steps * sampler.global_batch_size
    return {
        "pair_coverage_sampler_schema_version": (
            PAIR_COVERAGE_SAMPLER_SCHEMA_VERSION
        ),
        "pair_coverage_complete": True,
        "coverage_pair_count": sampler.pair_count,
        "covered_pair_count": covered,
        "coverage_duplicate_pair_count": duplicates,
        "coverage_steps": sampler.coverage_steps,
        "coverage_global_batch_size": sampler.global_batch_size,
        "coverage_local_batch_size": sampler.local_batch_size,
        "coverage_world_size": sampler.world_size,
        "coverage_sampler_seed": sampler.seed,
        "coverage_permutation_sha256": hashlib.sha256(
            sampler.global_permutation.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "coverage_padding_slot_count": total_slots - sampler.pair_count,
        "weighted_replacement_start_step": sampler.coverage_steps + 1,
    }


def validate_coverage_evidence(
    value: Mapping[str, object],
) -> dict[str, int | bool | str]:
    """Validate the persisted proof for the coverage-first training phase."""

    if (
        value.get("pair_coverage_sampler_schema_version")
        != PAIR_COVERAGE_SAMPLER_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported retained-pair coverage sampler schema")
    if value.get("pair_coverage_complete") is not True:
        raise ValueError("Retained-pair coverage is not complete")

    integer_fields = (
        "coverage_pair_count",
        "covered_pair_count",
        "coverage_duplicate_pair_count",
        "coverage_steps",
        "coverage_global_batch_size",
        "coverage_local_batch_size",
        "coverage_world_size",
        "coverage_sampler_seed",
        "coverage_padding_slot_count",
        "weighted_replacement_start_step",
    )
    normalized: dict[str, int | bool | str] = {
        "pair_coverage_sampler_schema_version": (
            PAIR_COVERAGE_SAMPLER_SCHEMA_VERSION
        ),
        "pair_coverage_complete": True,
    }
    for field in integer_fields:
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"{field} must be an integer")
        normalized[field] = int(raw)

    pair_count = int(normalized["coverage_pair_count"])
    covered = int(normalized["covered_pair_count"])
    duplicates = int(normalized["coverage_duplicate_pair_count"])
    steps = int(normalized["coverage_steps"])
    global_batch = int(normalized["coverage_global_batch_size"])
    local_batch = int(normalized["coverage_local_batch_size"])
    world = int(normalized["coverage_world_size"])
    padding = int(normalized["coverage_padding_slot_count"])
    replacement_start = int(normalized["weighted_replacement_start_step"])
    if pair_count <= 0 or covered != pair_count or duplicates != 0:
        raise ValueError(
            "Retained-pair coverage counts must prove complete exact-once coverage"
        )
    if steps <= 0 or global_batch <= 0 or local_batch <= 0 or world <= 0:
        raise ValueError("Retained-pair coverage dimensions must be positive")
    if global_batch != local_batch * world:
        raise ValueError("Coverage global batch does not equal local batch * world size")
    expected_steps = int(math.ceil(pair_count / global_batch))
    if steps != expected_steps:
        raise ValueError("Coverage step count is inconsistent with pair/global batch counts")
    if padding != steps * global_batch - pair_count or padding < 0:
        raise ValueError("Coverage padding slot count is inconsistent")
    if replacement_start != steps + 1:
        raise ValueError("Weighted replacement must start after full pair coverage")

    permutation_sha256 = str(value.get("coverage_permutation_sha256") or "")
    if len(permutation_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in permutation_sha256
    ):
        raise ValueError("coverage_permutation_sha256 must be a lowercase SHA-256")
    normalized["coverage_permutation_sha256"] = permutation_sha256
    return normalized


def require_coverage_steps(
    *,
    training_steps: int,
    sampler: CoverageFirstDDPSampler,
    candidate_steps: Sequence[int] = (),
) -> dict[str, int | bool | str]:
    steps = _positive_int(training_steps, label="training_steps")
    if steps < sampler.coverage_steps:
        raise ValueError(
            "flow.training_steps cannot cover every retained pair once: "
            f"training_steps={steps} required_coverage_steps="
            f"{sampler.coverage_steps}"
        )
    normalized_candidates: list[int] = []
    for value in candidate_steps:
        normalized_candidates.append(
            _positive_int(value, label="candidate step")
        )
    if normalized_candidates and min(normalized_candidates) < sampler.coverage_steps:
        raise ValueError(
            "The earliest Flow checkpoint candidate occurs before full pair "
            f"coverage: earliest={min(normalized_candidates)} "
            f"coverage_steps={sampler.coverage_steps}"
        )
    return build_coverage_evidence(sampler)
