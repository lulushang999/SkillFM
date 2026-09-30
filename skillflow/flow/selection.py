"""Training-only candidate selection for the main method."""
from __future__ import annotations
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence
import numpy as np
SAMPLE_COUNT = 64
SEED = 20260913
FLOW_LEARNING_RATE_SCHEDULE = "linear_warmup_cosine_decay_v1"

def flow_condition_key(*, task_type, canonical_goal, mode=None):
    return (str(task_type), str(canonical_goal))

def _flow_learning_rate_for_step(
    optimizer_step: int,
    *,
    training_steps: int,
    peak_learning_rate: float,
    minimum_learning_rate: float,
    warmup_steps: int,
    schedule: str,
) -> float:
    """Return the LR for a one-indexed optimizer update.

    V1 warms linearly from ``peak / warmup_steps`` on update 1 to the peak on
    ``warmup_steps``. Remaining updates follow a cosine to the exact minimum on
    ``training_steps``. The explicit one-indexed definition avoids framework
    scheduler initialization/off-by-one differences across PyTorch versions.
    """

    for label, value in (
        ("optimizer_step", optimizer_step),
        ("training_steps", training_steps),
        ("warmup_steps", warmup_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    if optimizer_step > training_steps:
        raise ValueError("optimizer_step must not exceed training_steps")
    if warmup_steps >= training_steps:
        raise ValueError("warmup_steps must be less than training_steps")
    if schedule != FLOW_LEARNING_RATE_SCHEDULE:
        raise ValueError(
            f"unsupported Flow learning-rate schedule {schedule!r}; expected "
            f"{FLOW_LEARNING_RATE_SCHEDULE!r}"
        )
    peak = float(peak_learning_rate)
    minimum = float(minimum_learning_rate)
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("peak_learning_rate must be finite and positive")
    if not math.isfinite(minimum) or not 0.0 < minimum < peak:
        raise ValueError(
            "minimum_learning_rate must be finite, positive, and below the peak"
        )
    if optimizer_step <= warmup_steps:
        return peak * float(optimizer_step) / float(warmup_steps)
    progress = float(optimizer_step - warmup_steps) / float(
        training_steps - warmup_steps
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (peak - minimum) * cosine

def _stable_group_key(seed: int, condition: tuple[str, ...]) -> str:
    payload = (
        f"{int(seed)}\0" + "\0".join(str(value) for value in condition)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _unit_rows(value: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) <= 0:
        raise ValueError(f"{label} must be a non-empty matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains NaN/Inf")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or np.any(~np.isfinite(norms)):
        raise ValueError(f"{label} contains zero or invalid-norm rows")
    return result / norms

def _pairwise_geodesic_angles(
    predicted: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    prediction = _unit_rows(predicted, "generated condition endpoints")
    reference = _unit_rows(target, "condition terminal endpoints")
    if prediction.shape[1] != reference.shape[1]:
        raise ValueError(
            "Endpoint latent dimension mismatch: "
            f"predicted={prediction.shape} target={reference.shape}"
        )
    cosine = (prediction @ reference.T).clip(-1.0, 1.0)
    sine = np.sqrt(np.maximum(0.0, 1.0 - np.square(cosine)))
    return np.arctan2(sine, cosine)

def _condition_groups(
    dataset: InternalValidationDataset,
) -> dict[tuple[str, ...], np.ndarray]:
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, (task_type, canonical_goal) in enumerate(
        zip(dataset.task_types, dataset.canonical_goals)
    ):
        condition = flow_condition_key(
            task_type=task_type,
            canonical_goal=canonical_goal,
            mode=dataset.condition_mode,
        )
        groups.setdefault(condition, []).append(index)
    return {
        condition: np.asarray(indices, dtype=np.int64)
        for condition, indices in sorted(groups.items())
    }

def _macro_condition_mean(
    values: Mapping[tuple[str, ...], float],
) -> tuple[float, dict[str, float]]:
    if not values:
        raise ValueError("Condition-set score requires at least one condition")
    by_type_values: dict[str, list[float]] = {}
    for condition, raw in sorted(values.items()):
        task_type = condition[0]
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("Condition-set score contains NaN/Inf")
        by_type_values.setdefault(task_type, []).append(value)
    by_type = {
        task_type: float(np.mean(condition_values))
        for task_type, condition_values in sorted(by_type_values.items())
    }
    return float(np.mean(list(by_type.values()))), by_type

def _condition_set_score_components(
    generated: np.ndarray,
    dataset: InternalValidationDataset,
) -> dict[str, Any]:
    generated_rows = _unit_rows(generated, "generated endpoints")
    if generated_rows.shape != dataset.terminal_latents.shape:
        raise ValueError(
            "Endpoint shape mismatch: "
            f"predicted={generated_rows.shape} "
            f"target={dataset.terminal_latents.shape}"
        )

    forward_angles = np.empty(dataset.size, dtype=np.float64)
    reverse_angles = np.empty(dataset.size, dtype=np.float64)
    condition_forward_mse: dict[tuple[str, ...], float] = {}
    condition_reverse_mse: dict[tuple[str, ...], float] = {}
    condition_chamfer_mse: dict[tuple[str, ...], float] = {}
    condition_forward_mean: dict[tuple[str, ...], float] = {}
    condition_reverse_mean: dict[tuple[str, ...], float] = {}
    for condition, indices in _condition_groups(dataset).items():
        pairwise = _pairwise_geodesic_angles(
            generated_rows[indices],
            dataset.terminal_latents[indices],
        )
        # Precision: each generated sample must land near some verified terminal
        # under exactly the same (task_type, canonical_goal) condition. Coverage:
        # each verified terminal must be represented by some generated sample.
        forward = np.min(pairwise, axis=1)
        reverse = np.min(pairwise, axis=0)
        forward_angles[indices] = forward
        reverse_angles[indices] = reverse
        forward_mse = float(np.mean(np.square(forward)))
        reverse_mse = float(np.mean(np.square(reverse)))
        condition_forward_mse[condition] = forward_mse
        condition_reverse_mse[condition] = reverse_mse
        condition_chamfer_mse[condition] = 0.5 * (forward_mse + reverse_mse)
        condition_forward_mean[condition] = float(np.mean(forward))
        condition_reverse_mean[condition] = float(np.mean(reverse))

    chamfer_mse, per_type_chamfer = _macro_condition_mean(
        condition_chamfer_mse
    )
    forward_mse, per_type_forward_mse = _macro_condition_mean(
        condition_forward_mse
    )
    reverse_mse, per_type_reverse_mse = _macro_condition_mean(
        condition_reverse_mse
    )
    forward_mean, per_type_forward_mean = _macro_condition_mean(
        condition_forward_mean
    )
    reverse_mean, per_type_reverse_mean = _macro_condition_mean(
        condition_reverse_mean
    )
    return {
        "forward_angles": forward_angles,
        "reverse_angles": reverse_angles,
        "chamfer_mse": chamfer_mse,
        "forward_mse": forward_mse,
        "reverse_mse": reverse_mse,
        "forward_mean": forward_mean,
        "reverse_mean": reverse_mean,
        "per_type_chamfer_mse": per_type_chamfer,
        "per_type_forward_mse": per_type_forward_mse,
        "per_type_reverse_mse": per_type_reverse_mse,
        "per_type_forward_mean": per_type_forward_mean,
        "per_type_reverse_mean": per_type_reverse_mean,
    }

FEATURE_PATTERNS = {
    "anchor": r"\b(anchor|question entity|stated entit|named subject|starting entity)",
    "bridge": r"\b(bridge|intermediate entit|next hop|next relation|follow.*relation)",
    "relation_direction": r"\b(direction|subject.*object|agent.*patient|revers|who did|role)",
    "comparison": r"\b(compar|both subjects|two subjects|same attribute|operand)",
    "intersection": r"\b(intersect|shared|common to|satisf.*both|both constraints)",
    "yes_no": r"\b(yes.?no|boolean|full predicate|evidence of no|negative answer)",
    "temporal": r"\b(chronolog|earlier|later|before|after|date|year|time constraint)",
    "count": r"\b(count|number of|distinct|deduplicat)",
    "identity": r"\b(identity|identit|disambiguat|same.name|homonym)",
    "qualifier": r"\b(qualifier|constraint|filter|restrict|requested.*precision)",
    "support": r"\b(evidence|support|verif|passage|document)",
    "answer_type": r"\b(answer type|requested type|answer form|endpoint|shortest.*answer|short answer)",
    "search": r"\b(search|retriev|query|queries)",
    "stop_condition": r"\b(stop|only after|complete chain|every.*link|all.*relation)",
}

RELATIONS = {
    "birthplace": r"\b(birthplace|place of birth|born in|born where)",
    "birth_date": r"\b(birth date|date of birth|birth year|born when)",
    "death": r"\b(death|died)",
    "nationality": r"\b(nationality|citizenship|citizen)",
    "occupation": r"\b(occupation|profession|job|career)",
    "education": r"\b(education|school|college|university|alma mater)",
    "authorship": r"\b(author|writ|novelist|book)",
    "film_direction": r"\b(director|directed|filmmaker)",
    "music": r"\b(compos|musician|song|album|singer)",
    "family": r"\b(parent|father|mother|spouse|husband|wife|sibling|daughter|son)",
    "location": r"\b(locat|city|country|region|state|headquarters|residence)",
    "membership": r"\b(member|team|organization|organisation|club|party)",
    "award": r"\b(award|prize|winner)",
}

def _features(text: str, patterns: Mapping[str, str]) -> set[str]:
    return {key for key, pattern in patterns.items() if re.search(pattern, text.casefold())}

def assess_decode(raw_text: str, status: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Measure syntactic validity and explicit rule features, not meaning/EM."""
    parsed = None
    error = None
    try:
        value = json.loads(raw_text)
        keys = {"when_to_use", "high_level_guidance", "low_level_guidance"}
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("expected exactly three skill fields")
        if any(not isinstance(v, str) or not v.strip() for v in value.values()):
            raise ValueError("skill fields must be nonempty strings")
        parsed = value
    except (TypeError, ValueError) as exc:
        error = str(exc)
    target = " ".join(str(expected.get(k, "")) for k in (
        "when_to_use", "high_level_guidance", "low_level_guidance"))
    observed = " ".join(parsed.values()) if parsed else ""
    wanted_ops, got_ops = _features(target, FEATURE_PATTERNS), _features(observed, FEATURE_PATTERNS)
    wanted_rel, got_rel = _features(target, RELATIONS), _features(observed, RELATIONS)
    wanted, got = ({"op:" + x for x in wanted_ops} | {"relation:" + x for x in wanted_rel},
                   {"op:" + x for x in got_ops} | {"relation:" + x for x in got_rel})
    # F1 penalizes indiscriminately listing all operations as well as omission.
    overlap = len(wanted & got)
    recall = overlap / len(wanted) if wanted else 0.0
    precision = overlap / len(got) if got else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    valid = parsed is not None and status == "ok"
    return {
        "schema_valid": parsed is not None,
        "eos_terminated": status == "ok",
        "schema_and_eos_valid": valid,
        "parse_error": error,
        "expected_features": sorted(wanted),
        "observed_features": sorted(got),
        "missing_features": sorted(wanted - got),
        "extra_features": sorted(got - wanted),
        "feature_recall_proxy": recall if valid else 0.0,
        "feature_precision_proxy": precision if valid else 0.0,
        "feature_f1_proxy": f1 if valid else 0.0,
        "semantic_correctness_verified": False,
    }

def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Empty decoded subset")
    n = len(rows)
    return {
        "count": n,
        "schema_valid_rate": sum(bool(r["schema_valid"]) for r in rows) / n,
        "eos_rate": sum(bool(r["eos_terminated"]) for r in rows) / n,
        "valid_rate": sum(bool(r["schema_and_eos_valid"]) for r in rows) / n,
        "feature_fidelity_proxy": sum(float(r["feature_f1_proxy"]) for r in rows) / n,
        "feature_recall_proxy": sum(float(r["feature_recall_proxy"]) for r in rows) / n,
        "semantic_accuracy": None,
    }

def rank_key(record: Mapping[str, Any]) -> tuple[float, float, float, int]:
    summary = record["decoded"]
    values = (float(summary["valid_rate"]), float(summary["feature_fidelity_proxy"]),
              float(record["geodesic_mse"]))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite candidate quality score")
    if not 0 <= values[0] <= 1 or not 0 <= values[1] <= 1 or values[2] < 0:
        raise ValueError("Candidate score outside its range")
    return (-values[0], -values[1], values[2], int(record["step"]))

def fixed_subset(task_ids: Sequence[str], count: int = SAMPLE_COUNT,
                 strata: Sequence[str] | None = None) -> list[int]:
    if len(task_ids) < count or len(set(task_ids)) != len(task_ids):
        raise ValueError("Holdout has too few or duplicate task IDs")
    def order(i):
        return hashlib.sha256(f"{SEED}:{task_ids[i]}".encode()).hexdigest()
    if strata is None:
        strata = ["all"] * len(task_ids)
    if len(strata) != len(task_ids):
        raise ValueError("Holdout stratum/task-ID lengths differ")
    groups = {label: sorted((i for i, value in enumerate(strata) if value == label), key=order)
              for label in sorted(set(strata))}
    chosen = []
    cursor = 0
    while len(chosen) < count:
        for group in groups.values():
            if cursor < len(group):
                chosen.append(group[cursor])
                if len(chosen) == count:
                    break
        cursor += 1
    return chosen

def question_text(text: str) -> str:
    return re.sub(r"^Worker:\s*", "", str(text).strip()).strip()

def operator(question: str) -> str:
    q = question_text(question).lower().lstrip("? ")
    if re.match(r"(?:are|is|was|were|do|does|did|has|have|had|can|could)\b", q):
        return "yes_no"
    if re.search(r"\bboth\b|\bin common\b|\bcommon to\b|\bshared by\b|\bshare(?:s|d)? (?:the )?same\b", q):
        return "intersection"
    if " and " in q and (re.match(r"what (?:are|were) (?!the (?:names|dates|coordinates))", q) or re.search(r"\b(?:are|were),? (?:which|what) (?:type|kind|genre|class|nationality)", q)):
        return "intersection"
    if re.search(r"\b(?:earlier|later|older|younger|larger|smaller|longer|shorter|higher|lower|taller|more|less|first)\b", q) and re.search(r"\b(?:or|and|two|between)\b", q):
        return "comparison"
    if re.search(r"\bor\b|which of (?:the|these|those|following)", q):
        return "candidate_filter"
    return "chain"
