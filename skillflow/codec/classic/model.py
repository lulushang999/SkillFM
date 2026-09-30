from __future__ import annotations

import hashlib
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from skillflow.models.import_safety import hide_optional_audio_packages
from skillflow.skills.skill_text import clean_text
from skillflow.codec.classic.training_contract import (
    CODEC_ENCODER_POOLING,
    CODEC_ENCODER_ROUTING_VERSION,
    CODEC_NUMERIC_POLICY_VERSION,
    CODEC_QUERY_ENCODER_POOLING,
    CODEC_SCHEMA_VERSION,
    CODEC_SKILL_INPUT_VERSION,
    CODEC_SKILL_VIEW_VERSION,
    CODEC_TRAINING_EXAMPLE_POLICY,
    decoder_prompt_uses_task_context,
    normalize_encoder_pooling_profile,
    validate_codec_artifact_policy,
)


# Qwen3 has a roughly 152k-token vocabulary. Materializing FP32 logits and
# cross-entropy state for every position of a 4096-token sequence can consume
# tens of GiB. Project only supervised positions and checkpoint bounded blocks.
DECODER_LOSS_TOKEN_CHUNK_SIZE = 256


BACKBONE_DTYPE_CHOICES = ("auto", "bfloat16", "float32")


def resolve_backbone_dtype(
    value: object = "auto",
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.dtype:
    """Use BF16 only on supported CUDA hardware; otherwise use FP32."""

    normalized = str(value or "auto").strip().lower()
    aliases = {"bf16": "bfloat16", "fp32": "float32", "float": "float32"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in BACKBONE_DTYPE_CHOICES:
        raise ValueError(
            f"Unsupported codec backbone dtype={value!r}; "
            f"expected one of {BACKBONE_DTYPE_CHOICES}"
        )
    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if normalized == "auto":
        if (
            resolved_device.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            return torch.bfloat16
        return torch.float32
    if normalized == "bfloat16":
        if resolved_device.type != "cuda" or not torch.cuda.is_bf16_supported():
            return torch.float32
        return torch.bfloat16
    return torch.float32


def backbone_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unsupported resolved codec dtype={dtype}")


def model_source_fingerprint(path: Union[str, Path]) -> str:
    """Fingerprint exact local HF model and tokenizer contents.

    The identity is deliberately independent of the absolute checkout path.
    Model metadata, tokenizer files, and every weight shard are hashed. This
    detects same-sized weight replacement and proves that query inference uses
    the recorded original embedding model.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        return hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    digest = hashlib.sha256()
    digest.update(b"skillflow-full-model-content-v1\0")
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",
    ):
        item = source / name
        if item.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(item.read_bytes())
    weight_files = sorted(
        [*source.glob("*.safetensors"), *source.glob("pytorch_model*.bin")],
        key=lambda item: item.name,
    )
    if not weight_files:
        raise FileNotFoundError(f"Embedding model source has no weight files: {source}")
    for item in weight_files:
        digest.update(item.name.encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _backbone_attention_kwargs(
    attention_implementation: Optional[str],
    dtype: torch.dtype,
) -> dict[str, str]:
    implementation = str(attention_implementation or "").strip()
    if not implementation:
        return {}
    # FlashAttention-2 deliberately rejects FP32. CPU and older CUDA devices
    # use PyTorch SDPA when the numeric-policy resolver selects FP32.
    if dtype == torch.float32 and implementation == "flash_attention_2":
        implementation = "sdpa"
    return {"attn_implementation": implementation}


def _disable_autocast_for(value: torch.Tensor):
    """Keep trainable numeric-sensitive branches in true FP32."""

    device_type = value.device.type
    try:
        available = torch.amp.autocast_mode.is_autocast_available(device_type)
    except AttributeError:
        available = device_type in {"cpu", "cuda"}
    if not available:
        return nullcontext()
    return torch.autocast(device_type=device_type, enabled=False)


def _require_finite(value: torch.Tensor, label: str) -> torch.Tensor:
    if not torch.is_floating_point(value):
        raise TypeError(f"{label} must be floating point, got {value.dtype}")
    finite = torch.isfinite(value)
    if value.device.type == "cuda":
        # Keep the check ordered on the CUDA stream without forcing a host
        # synchronization for every numerically sensitive intermediate.
        # A failed device assertion aborts this training process, just like the
        # former synchronous FloatingPointError path did.
        torch._assert_async(finite.all())
    elif not bool(finite.all().item()):
        invalid = int((~finite).sum().item())
        raise FloatingPointError(f"{label} contains {invalid} NaN/Inf value(s)")
    return value


def _normalize_fp32(
    value: torch.Tensor,
    label: str,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    with _disable_autocast_for(value):
        value = _require_finite(value.float(), label)
        norms = _require_finite(
            value.norm(p=2, dim=-1, keepdim=True),
            f"{label} norm",
        )
        invalid = norms <= float(eps)
        if value.device.type == "cuda":
            torch._assert_async((~invalid).all())
        elif bool(invalid.any().item()):
            rows = invalid.flatten().nonzero(as_tuple=False).flatten().tolist()
            raise FloatingPointError(
                f"{label} contains zero/near-zero vector(s) at rows {rows[:8]}"
            )
        return _require_finite(value / norms, f"normalized {label}")


def enable_memory_efficient_gradient_checkpointing(model: nn.Module) -> bool:
    """Enable the non-reentrant checkpointing required by codec LoRA.

    Non-reentrant checkpointing can compute LoRA parameter gradients even when
    the frozen backbone input does not require gradients. That lets encoder
    backpropagation remain valid while the base backbone is frozen.
    """

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return True


DEFAULT_DECODER_PROMPT = (
    "Decode the latent into exactly one compact ALFWorld skill JSON object.\n"
    "Task type: {task_type}\n"
    "Task goal: {goal}\n"
    "Use task context for required entities and state changes, and use the "
    "latent prefix for the learned strategy.\n"
    "Output JSON only with exactly three non-empty string fields: when_to_use, "
    "high_level_guidance, and low_level_guidance.\n"
    "Do not emit task metadata, outcome labels, arrays, markdown, or extra keys.\n"
    "Keep the skill concise and end immediately after the JSON object.\n"
    "JSON:\n"
)

LATENT_ONLY_DECODER_PROMPT = (
    "Reconstruct the original skill JSON from the latent tokens.\n"
)

DEFAULT_ENCODER_SKILL_PROMPT = (
    "Represent this compact ALFWorld skill as a normalized strategy vector.\n"
    "Skill JSON:\n{skill}\n"
)

DEFAULT_ENCODER_QUERY_PROMPT = (
    "Represent the following ALFWorld task condition as one query "
    "vector for selecting or transforming a skill. "
    "Focus on task family, target object, target receptacle, required "
    "state changes, and inspection/light constraints.\n"
    "Task type: {task_type}\n"
    "Task goal: {goal}\n"
)


def role_free_skill_json(
    text: object,
    *,
    expected_task_type: Optional[str] = None,
    canonical_goal: Optional[str] = None,
) -> str:
    """Validate the single shared outcome-free executable-skill contract."""
    from skillflow.skills.executable import canonicalize_executable_skill

    # Outcome fields are rejected by the schema rather than stripped silently.
    # This prevents accidental leakage from becoming an apparently valid codec.
    return canonicalize_executable_skill(
        text,
        expected_task_type=expected_task_type,
        canonical_goal=canonical_goal,
    )


def skill_encoder_text(
    skill_text: object,
    *,
    skill_prompt: str = DEFAULT_ENCODER_SKILL_PROMPT,
) -> str:
    """Render the only input accepted by the trainable skill encoder."""

    canonical = role_free_skill_json(skill_text)
    template = str(skill_prompt or DEFAULT_ENCODER_SKILL_PROMPT)
    if "{skill}" not in template:
        raise ValueError("Skill encoder prompt must contain {skill}")
    if any(token in template for token in ("{task_type}", "{goal}", "{context}")):
        raise ValueError("Skill encoder prompt must contain skill only")
    return template.format(skill=canonical)


def frozen_query_encoder_text(
    *,
    task_type: object,
    goal: object,
    query_prompt: str = DEFAULT_ENCODER_QUERY_PROMPT,
) -> str:
    """Render a task query for the independent frozen base embedding model."""

    task_type_text = clean_text(task_type)
    goal_text = clean_text(goal)
    if not task_type_text or not goal_text:
        raise ValueError("Frozen query encoder requires non-empty task_type and goal")
    template = str(query_prompt or DEFAULT_ENCODER_QUERY_PROMPT)
    if not all(token in template for token in ("{task_type}", "{goal}")):
        raise ValueError(
            "Frozen query encoder prompt must contain {task_type} and {goal}"
        )
    return template.format(task_type=task_type_text, goal=goal_text)


def get_hidden_size(config: Any) -> int:
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Cannot determine hidden size")


def get_context_window(config: Any, tokenizer: Any) -> int:
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        try:
            value = int(value)
        except Exception:
            continue
        if 0 < value < 1_000_000:
            return value
    try:
        value = int(tokenizer.model_max_length)
    except Exception as exc:
        raise ValueError("Cannot determine tokenizer/model context window") from exc
    if not 0 < value < 1_000_000:
        raise ValueError(f"Invalid tokenizer/model context window: {value}")
    return value


def last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device)
    positions = positions.unsqueeze(0).expand_as(attention_mask)
    last_index = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
    batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_index, last_index]


def official_last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply Qwen3-Embedding's official last-token pooling and L2 norm."""

    return _normalize_fp32(
        last_token_pool(hidden_states, attention_mask).float(),
        "official last-token pooled encoder output",
    )


def hybrid_tail_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the v3 artifact's 0.75 last-token + 0.25 last-four mean pool."""

    if hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("Pooling requires rank-3 hidden states and a rank-2 mask")
    if hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("Pooling hidden-state and attention-mask shapes disagree")
    active = attention_mask.ne(0)
    nonempty = active.any(dim=1)
    if hidden_states.device.type == "cuda":
        torch._assert_async(nonempty.all())
    elif not bool(nonempty.all().item()):
        raise ValueError("Pooling attention mask contains an empty row")
    reverse_rank = torch.flip(
        torch.cumsum(torch.flip(active.long(), dims=(1,)), dim=1),
        dims=(1,),
    )
    tail_mask = active & reverse_rank.le(4)
    with _disable_autocast_for(hidden_states):
        states = _require_finite(hidden_states.float(), "hybrid pooling states")
        tail_count = tail_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
        tail_mean = (states * tail_mask.unsqueeze(-1)).sum(dim=1) / tail_count
        pooled = 0.75 * last_token_pool(states, attention_mask) + 0.25 * tail_mean
        return _normalize_fp32(pooled, "hybrid tail pooled encoder output")


def pool_encoder_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pooling_profile: object,
) -> torch.Tensor:
    """Dispatch the explicit artifact or official-compatibility pooler."""

    profile = normalize_encoder_pooling_profile(pooling_profile)
    if profile == CODEC_QUERY_ENCODER_POOLING:
        return hybrid_tail_pool(hidden_states, attention_mask)
    return official_last_token_pool(hidden_states, attention_mask)


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def configure_decoder_tokenizer(
    tokenizer: Any,
    *,
    context_window: Optional[int] = None,
) -> None:
    """Configure decoder-only batching and synchronize its real context window."""

    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "left"
    if context_window is not None:
        resolved_context_window = int(context_window)
        if not 0 < resolved_context_window < 1_000_000:
            raise ValueError(
                f"Invalid decoder model context window: {resolved_context_window}"
            )
        tokenizer.model_max_length = resolved_context_window


def decoder_eos_token_ids(model: Any, tokenizer: Any) -> tuple[int, ...]:
    """Return every EOS id accepted by the tokenizer or decoder config."""

    values = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    ]
    output: set[int] = set()
    for value in values:
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        for candidate in candidates:
            try:
                output.add(int(candidate))
            except (TypeError, ValueError):
                continue
    return tuple(sorted(output))


class ConditionalDecoderGenerationError(RuntimeError):
    """A strict conditional decode did not terminate within its valid budget."""

    def __init__(
        self,
        *,
        batch_row: int,
        max_new_tokens: int,
        recognized_eos_token_ids: tuple[int, ...],
    ) -> None:
        self.batch_row = int(batch_row)
        self.max_new_tokens = int(max_new_tokens)
        self.recognized_eos_token_ids = tuple(
            int(value) for value in recognized_eos_token_ids
        )
        super().__init__(
            "Conditional decoder exhausted max_new_tokens without EOS; "
            f"batch_row={self.batch_row} max_new_tokens={self.max_new_tokens} "
            f"recognized_eos_token_ids={list(self.recognized_eos_token_ids)}"
        )


@dataclass(frozen=True)
class ConditionalDecoderResult:
    """Outcome of the one permitted conditional decode for one latent row."""

    batch_row: int
    status: str
    text: str
    max_new_tokens: int
    recognized_eos_token_ids: tuple[int, ...]


def prepend_prefix_to_left_padded_prompt(
    prefix: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    prompt_attention: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep each active ``prefix + prompt`` sequence contiguous and right-aligned."""

    if prefix.ndim != 3 or prompt_embeddings.ndim != 3:
        raise ValueError("Decoder prefix and prompt embeddings must both be rank 3")
    if prompt_attention.ndim != 2:
        raise ValueError("Decoder prompt attention mask must be rank 2")
    if prefix.size(0) != prompt_embeddings.size(0):
        raise ValueError("Decoder prefix/prompt batch size mismatch")
    if prompt_embeddings.shape[:2] != prompt_attention.shape:
        raise ValueError("Decoder prompt embeddings/attention shape mismatch")
    if prefix.size(2) != prompt_embeddings.size(2):
        raise ValueError("Decoder prefix/prompt hidden size mismatch")
    batch_size, prompt_width = prompt_attention.shape
    if int(batch_size) <= 0 or int(prompt_width) <= 0:
        raise ValueError("Conditional decoder prompt cannot be empty")

    binary = ((prompt_attention == 0) | (prompt_attention == 1)).all()
    prompt_lengths = prompt_attention.long().sum(dim=1)
    nonempty = (prompt_lengths > 0).all()
    pad_lengths = int(prompt_width) - prompt_lengths
    prompt_positions = torch.arange(
        int(prompt_width),
        device=prompt_attention.device,
    ).unsqueeze(0)
    expected_attention = prompt_positions >= pad_lengths.unsqueeze(1)
    contiguous = prompt_attention.ne(0).eq(expected_attention).all()
    if prompt_attention.device.type == "cuda":
        # Validate every row with one asynchronous device-side assertion.
        torch._assert_async(torch.stack((binary, nonempty, contiguous)).all())
    else:
        if not bool(binary.item()):
            raise ValueError("Decoder prompt attention mask must be binary")
        if not bool(nonempty.item()):
            raise ValueError("Conditional decoder prompt cannot be empty")
        if not bool(contiguous.item()):
            raise ValueError(
                "Conditional decoder prompts must use contiguous left padding"
            )

    prefix_width = int(prefix.size(1))
    combined_positions = (
        torch.arange(
            int(prompt_width) + prefix_width,
            device=prompt_attention.device,
        )
        .unsqueeze(0)
        .expand(int(batch_size), -1)
    )
    relative_to_prefix = combined_positions - pad_lengths.unsqueeze(1)
    is_prefix = relative_to_prefix.ge(0) & relative_to_prefix.lt(prefix_width)
    prompt_indices = torch.where(
        combined_positions < pad_lengths.unsqueeze(1),
        combined_positions,
        combined_positions - prefix_width,
    ).clamp_(0, int(prompt_width) - 1)
    prefix_indices = relative_to_prefix.clamp(0, max(0, prefix_width - 1))
    batch_indices = torch.arange(
        int(batch_size),
        device=prompt_attention.device,
    ).unsqueeze(1)
    gathered_prompt = prompt_embeddings[batch_indices, prompt_indices]
    if prefix_width > 0:
        gathered_prefix = prefix[batch_indices, prefix_indices]
    else:
        gathered_prefix = torch.zeros_like(gathered_prompt)
    combined_embeddings = torch.where(
        is_prefix.unsqueeze(-1),
        gathered_prefix,
        gathered_prompt,
    )
    combined_attention = torch.where(
        is_prefix,
        torch.ones((), dtype=prompt_attention.dtype, device=prompt_attention.device),
        prompt_attention[batch_indices, prompt_indices],
    )
    return combined_embeddings, combined_attention


def position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Create padding-invariant positions for a left-padded active sequence."""

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be rank 2")
    binary = ((attention_mask == 0) | (attention_mask == 1)).all()
    nonempty = (attention_mask.sum(dim=1) > 0).all()
    if attention_mask.device.type == "cuda":
        torch._assert_async(torch.stack((binary, nonempty)).all())
    else:
        if not bool(binary.item()):
            raise ValueError("attention_mask must be binary")
        if not bool(nonempty.item()):
            raise ValueError("Every attention row must contain an active token")
    positions = attention_mask.long().cumsum(dim=-1) - 1
    return positions.masked_fill(attention_mask == 0, 0)


def apply_text_prompt(prompt: str, text: str) -> str:
    body = str(text or "").strip()
    template = str(prompt or "")
    if not template:
        return body
    if "{text}" in template:
        return template.format(text=body)
    return template + body


def apply_decoder_context_prompt(prompt: str, context: object = None) -> str:
    """Render either a contextual or latent-only skill decoder prompt."""
    template = str(prompt or DEFAULT_DECODER_PROMPT)
    if not decoder_prompt_uses_task_context(template):
        return template
    if not isinstance(context, Mapping):
        raise ValueError("Decoder context must be a task metadata mapping")
    task_type = clean_text(context.get("task_type"))
    goal = clean_text(context.get("canonical_goal") or context.get("goal"))
    if not task_type or not goal:
        raise ValueError("Decoder context requires non-empty task_type and goal")
    return template.replace("{task_type}", task_type).replace("{goal}", goal)


def _example_decoder_context(example: Any) -> dict[str, str]:
    if not isinstance(example, Mapping):
        raise ValueError(
            "Contextual codec training examples must include task metadata"
        )
    task_type = clean_text(example.get("task_type"))
    goal = clean_text(example.get("canonical_goal") or example.get("goal"))
    if not task_type or not goal:
        raise ValueError(
            "Contextual codec training example requires task_type and canonical_goal"
        )
    return {"task_type": task_type, "canonical_goal": goal}


def _example_skill_text(example: Any) -> str:
    if isinstance(example, dict):
        text = str(example.get("skill_text") or example.get("text") or "").strip()
    else:
        text = str(example or "").strip()
    if not text:
        raise ValueError("Codec training example has empty skill text")
    return role_free_skill_json(text)


def tokenize_skill_batch(
    texts: list[Any],
    encoder_tokenizer: Any,
    decoder_tokenizer: Any,
    max_length: int,
    num_prefix_tokens: int = 0,
    encoder_prompt: str = DEFAULT_ENCODER_SKILL_PROMPT,
    decoder_prompt: str = DEFAULT_DECODER_PROMPT,
) -> dict[str, torch.Tensor]:
    """Tokenize skill latents with contextual or latent-only reconstruction.

    The trainable encoder still receives only canonical skill JSON. Task type
    and goal, when enabled by the prompt, are decoder-side constraints and
    never query-encoder supervision.
    """

    if not texts:
        raise ValueError("Codec training batch cannot be empty")
    skill_texts = [_example_skill_text(example) for example in texts]
    decoder_template = str(decoder_prompt or DEFAULT_DECODER_PROMPT)
    decoder_task_conditioned = decoder_prompt_uses_task_context(decoder_template)
    decoder_contexts: list[object] = (
        [_example_decoder_context(example) for example in texts]
        if decoder_task_conditioned
        else [None] * len(texts)
    )
    encoder_texts = [
        skill_encoder_text(text, skill_prompt=encoder_prompt) for text in skill_texts
    ]
    encoder_inputs = encoder_tokenizer(
        encoder_texts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    encoder_lengths = encoder_inputs["attention_mask"].sum(dim=1)
    if bool((encoder_lengths > int(max_length)).any()):
        offending = encoder_lengths[encoder_lengths > int(max_length)].tolist()
        raise ValueError(
            "Skill encoder input exceeds max_length; truncation is forbidden: "
            f"max_length={max_length} offending={offending[:8]}"
        )

    eos_token_id = decoder_tokenizer.eos_token_id
    try:
        decoder_context_window = int(decoder_tokenizer.model_max_length)
    except Exception as exc:
        raise ValueError(
            "Decoder tokenizer must expose a finite model_max_length"
        ) from exc
    if not 0 < decoder_context_window < 1_000_000:
        raise ValueError(
            f"Invalid decoder tokenizer model_max_length={decoder_context_window}"
        )
    rendered_prompts = [
        apply_decoder_context_prompt(decoder_template, context=context)
        for context in decoder_contexts
    ]
    prompt_id_rows = decoder_tokenizer(
        rendered_prompts,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    target_id_rows = decoder_tokenizer(
        skill_texts,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    if len(prompt_id_rows) != len(skill_texts) or len(target_id_rows) != len(
        skill_texts
    ):
        raise RuntimeError("Decoder tokenizer changed the Codec batch cardinality")

    sequences: list[list[int]] = []
    label_masks: list[list[int]] = []
    for prompt_row, target_row in zip(prompt_id_rows, target_id_rows):
        prompt_ids = list(prompt_row)
        target = list(target_row)
        if eos_token_id is not None:
            target = target + [eos_token_id]
        total_length = len(prompt_ids) + len(target)
        if total_length > int(max_length):
            raise ValueError(
                "Skill decoder training sequence exceeds max_length; "
                f"truncation is forbidden: prompt={len(prompt_ids)} "
                f"target={len(target)} total={total_length} max_length={max_length}"
            )
        full_length = int(num_prefix_tokens) + total_length
        if full_length > decoder_context_window:
            raise ValueError(
                "Skill decoder training sequence exceeds the model context window: "
                f"prefix={num_prefix_tokens} text={total_length} "
                f"full={full_length} context_window={decoder_context_window}"
            )
        sequences.append(prompt_ids + target)
        label_masks.append([0] * len(prompt_ids) + [1] * len(target))

    width = max(len(ids) for ids in sequences)
    decoder_ids = torch.full(
        (len(skill_texts), width),
        decoder_tokenizer.pad_token_id,
        dtype=torch.long,
    )
    decoder_mask = torch.zeros_like(decoder_ids)
    decoder_label_mask = torch.zeros_like(decoder_ids)
    for index, (ids, labels) in enumerate(zip(sequences, label_masks)):
        # Training and inference share contiguous left-padding:
        # [padding, latent prefix, active prompt/target].
        row_start = width - len(ids)
        decoder_ids[index, row_start:] = torch.tensor(ids, dtype=torch.long)
        decoder_mask[index, row_start:] = 1
        decoder_label_mask[index, row_start:] = torch.tensor(labels, dtype=torch.long)

    return {
        "encoder_ids": encoder_inputs["input_ids"],
        "encoder_mask": encoder_inputs["attention_mask"],
        "decoder_ids": decoder_ids,
        "decoder_mask": decoder_mask,
        "decoder_label_mask": decoder_label_mask,
    }


class LoRALinear(nn.Module):
    """Dependency-free LoRA wrapper which can be merged into a standard Linear."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can wrap only torch.nn.Linear")
        self.base = base
        self.rank = max(1, min(int(rank), base.in_features, base.out_features))
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(max(0.0, float(dropout)))
        # Optimizer parameters and moments remain FP32 even when the frozen
        # transformer backbone computes in BF16.
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False).to(
            device=base.weight.device,
            dtype=torch.float32,
        )
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False).to(
            device=base.weight.device,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.base.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # The frozen base stays under the caller's BF16 autocast policy.
        base_output = self.base(value.to(self.base.weight.dtype))
        with _disable_autocast_for(value):
            lora_input = self.dropout(value.float())
            residual = self.lora_b(self.lora_a(lora_input)) * self.scaling
        return base_output + residual.to(base_output.dtype)

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        base = self.base
        delta = self.lora_b.weight.float() @ self.lora_a.weight.float()
        base.weight.add_((self.scaling * delta).to(base.weight.dtype))
        return base


def inject_lora(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> int:
    """Replace descendant Linear modules with trainable low-rank adapters."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
            )
            count += 1
        else:
            count += inject_lora(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
    return count


def merge_lora(module: nn.Module) -> int:
    """Merge every LoRA wrapper so save_pretrained emits a standard HF model."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged_linear())
            count += 1
        else:
            count += merge_lora(child)
    return count


def angular_perturb_latent(
    latent: torch.Tensor,
    *,
    angle_std: float,
    max_angle: float,
    probability: float,
    scale: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a differentiable random rotation on the unit hypersphere."""

    angle_std = float(angle_std)
    max_angle = float(max_angle)
    probability = float(probability)
    scale = float(scale)
    if latent.ndim != 2 or latent.size(-1) < 2:
        raise ValueError("latent noise expects a rank-2 tensor with dim >= 2")
    if angle_std < 0.0:
        raise ValueError("latent noise angle_std must be non-negative")
    if max_angle < angle_std:
        raise ValueError("latent noise max_angle must be >= angle_std")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("latent noise probability must be in [0, 1]")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("latent noise scale must be in [0, 1]")
    with _disable_autocast_for(latent):
        clean = _normalize_fp32(latent, "clean latent")
        if angle_std == 0.0 or max_angle == 0.0 or probability == 0.0 or scale == 0.0:
            return clean, torch.zeros(clean.size(0), device=clean.device)
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=torch.float32,
            generator=generator,
        )
        tangent = noise - (noise * clean).sum(dim=-1, keepdim=True) * clean
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        # Choose the coordinate least aligned with each latent to construct an
        # exactly orthogonal fallback, including deterministic adversarial RNGs.
        basis_index = clean.abs().argmin(dim=-1, keepdim=True)
        fallback = torch.zeros_like(clean).scatter(1, basis_index, 1.0)
        fallback = fallback - ((fallback * clean).sum(dim=-1, keepdim=True) * clean)
        fallback = F.normalize(fallback, p=2, dim=-1, eps=1e-8)
        tangent = torch.where(tangent_norm > 1e-8, tangent, fallback)
        tangent = F.normalize(tangent, p=2, dim=-1, eps=1e-8)
        angles = (
            torch.randn(
                clean.size(0),
                device=clean.device,
                dtype=torch.float32,
                generator=generator,
            )
            .abs()
            .mul_(angle_std * scale)
        )
        angles = angles.clamp_max(max_angle * scale)
        active = torch.rand(
            clean.size(0),
            device=clean.device,
            generator=generator,
        ).lt(probability)
        angles = torch.where(active, angles, torch.zeros_like(angles))
        noisy = (
            angles.cos().unsqueeze(-1) * clean + angles.sin().unsqueeze(-1) * tangent
        )
        return _normalize_fp32(noisy, "angularly perturbed latent"), angles


class SkillAutoencoder(nn.Module):
    """A stable semantic encoder plus a prefix-conditioned skill decoder."""

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        num_prefix_tokens: int,
        gradient_checkpointing: bool = False,
        attention_implementation: Optional[str] = None,
        backbone_dtype: object = "auto",
        latent_noise_angle_std: float = 0.0,
        latent_noise_max_angle: float = 0.0,
        latent_noise_prob: float = 0.0,
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.last_loss_metrics: dict[str, torch.Tensor] = {}
        self.backbone_dtype = resolve_backbone_dtype(backbone_dtype)
        self.resolved_backbone_dtype_name = backbone_dtype_name(self.backbone_dtype)
        self.latent_noise_angle_std = float(latent_noise_angle_std)
        self.latent_noise_max_angle = float(latent_noise_max_angle)
        self.latent_noise_prob = float(latent_noise_prob)
        self.latent_noise_scale = 0.0
        if self.latent_noise_angle_std < 0.0:
            raise ValueError("latent_noise_angle_std must be non-negative")
        if self.latent_noise_max_angle < self.latent_noise_angle_std:
            raise ValueError("latent_noise_max_angle must be >= latent_noise_angle_std")
        if not 0.0 <= self.latent_noise_prob <= 1.0:
            raise ValueError("latent_noise_prob must be in [0, 1]")

        with hide_optional_audio_packages():
            from transformers import AutoModel, AutoModelForCausalLM

            attention_kwargs = _backbone_attention_kwargs(
                attention_implementation,
                self.backbone_dtype,
            )

            self.encoder_prime = AutoModel.from_pretrained(
                encoder_path,
                dtype=self.backbone_dtype,
                trust_remote_code=True,
                **attention_kwargs,
            )
            self.decoder = AutoModelForCausalLM.from_pretrained(
                decoder_path,
                dtype=self.backbone_dtype,
                trust_remote_code=True,
                **attention_kwargs,
            )

        encoder_dim = get_hidden_size(self.encoder_prime.config)
        self.decoder_dim = get_hidden_size(self.decoder.config)
        # The projector is trainable and keeps FP32 parameters plus Adam moments.
        # Only its output is cast at the decoder embedding boundary.
        self.vector_to_prefix = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, num_prefix_tokens * self.decoder_dim),
        ).float()

        if gradient_checkpointing:
            enable_memory_efficient_gradient_checkpointing(self.encoder_prime)
            enable_memory_efficient_gradient_checkpointing(self.decoder)
        self.decoder.config.use_cache = False

    def encode_base_batch(
        self,
        encoder_ids: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_output = self.encoder_prime(
            input_ids=encoder_ids,
            attention_mask=encoder_mask,
            position_ids=position_ids_from_attention_mask(encoder_mask),
            return_dict=True,
        )
        return pool_encoder_hidden_states(
            encoder_output.last_hidden_state,
            encoder_mask,
            pooling_profile=CODEC_ENCODER_POOLING,
        )

    def encode_batch(
        self,
        encoder_ids: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_base_batch(encoder_ids, encoder_mask)

    def set_noise_progress(self, progress: float) -> None:
        progress = float(progress)
        if not 0.0 <= progress <= 1.0:
            raise ValueError("latent noise progress must be in [0, 1]")
        self.latent_noise_scale = progress

    def prefix_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        with _disable_autocast_for(latent):
            latent = _normalize_fp32(latent, "latent-to-prefix input")
            # FSDP mixed precision materializes this projector in its compute
            # dtype during the enclosing forward even though its master
            # parameters remain FP32.  Autocast is deliberately disabled here
            # so normalization stays FP32; make the following boundary
            # explicit instead of relying on autocast to reconcile a FP32
            # activation with BF16 projector parameters.  Outside FSDP the
            # projector remains FP32, so this is a no-op.
            projector_dtype = next(
                (
                    parameter.dtype
                    for parameter in self.vector_to_prefix.parameters()
                    if torch.is_floating_point(parameter)
                ),
                latent.dtype,
            )
            projector_input = latent.to(dtype=projector_dtype)
            prefix = self.vector_to_prefix(projector_input).view(
                latent.size(0), self.num_prefix_tokens, self.decoder_dim
            )
            return _require_finite(prefix.float(), "latent prefix")

    def _decoder_lm_loss(
        self,
        prefix: torch.Tensor,
        decoder_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
        decoder_label_mask: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if reduction not in {"none", "mean"}:
            raise ValueError(f"Unsupported decoder loss reduction={reduction!r}")
        prefix = _require_finite(prefix.float(), "decoder latent prefix")
        text_embeddings = self.decoder.get_input_embeddings()(decoder_ids)
        _require_finite(text_embeddings, "decoder token embeddings")
        # This is the only FP32 -> backbone-dtype boundary for the projector.
        decoder_prefix = prefix.to(dtype=text_embeddings.dtype)
        inputs_embeds, attention_mask = prepend_prefix_to_left_padded_prompt(
            decoder_prefix,
            text_embeddings,
            decoder_mask,
        )

        text_labels = decoder_ids.masked_fill(
            (decoder_mask == 0) | (decoder_label_mask == 0),
            -100,
        )
        # Match inference exactly: left padding, then latent prefix, then active
        # decoder prompt/target. Build the same layout on-device for the whole
        # batch so label construction never synchronizes once per row.
        batch_size, prompt_width = decoder_mask.shape
        prefix_width = int(prefix.size(1))
        prompt_lengths = decoder_mask.long().sum(dim=1)
        pad_lengths = int(prompt_width) - prompt_lengths
        combined_positions = (
            torch.arange(
                int(prompt_width) + prefix_width,
                device=decoder_ids.device,
            )
            .unsqueeze(0)
            .expand(int(batch_size), -1)
        )
        relative_to_prefix = combined_positions - pad_lengths.unsqueeze(1)
        is_prefix = relative_to_prefix.ge(0) & relative_to_prefix.lt(prefix_width)
        prompt_indices = torch.where(
            combined_positions < pad_lengths.unsqueeze(1),
            combined_positions,
            combined_positions - prefix_width,
        ).clamp_(0, int(prompt_width) - 1)
        batch_indices = torch.arange(
            int(batch_size),
            device=decoder_ids.device,
        ).unsqueeze(1)
        labels = text_labels[batch_indices, prompt_indices].masked_fill(
            is_prefix,
            -100,
        )

        # Qwen3ForCausalLM is exactly decoder.model -> lm_head. Calling the
        # wrapper would allocate vocabulary-sized logits at every prompt,
        # padding, and target position. The prompt hidden states still remain
        # in the causal body; only their unnecessary lm_head projections are
        # skipped here.
        decoder_model_type = (
            str(getattr(getattr(self.decoder, "config", None), "model_type", ""))
            .strip()
            .lower()
        )
        decoder_body = getattr(self.decoder, "model", None)
        decoder_head = self.decoder.get_output_embeddings()
        if (
            decoder_model_type != "qwen3"
            or decoder_body is None
            or decoder_head is None
        ):
            raise RuntimeError(
                "Memory-bounded decoder loss requires Qwen3ForCausalLM "
                "with .model and output embeddings"
            )
        output = decoder_body(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids_from_attention_mask(attention_mask),
            use_cache=False,
            return_dict=True,
        )
        hidden_states = output.last_hidden_state
        shift_labels = labels[:, 1:]
        valid_tokens = shift_labels.ne(-100)
        valid_positions = valid_tokens.nonzero(as_tuple=False)

        # Graph-connected zero also makes a malformed/empty-label row safe in
        # DDP instead of dropping the decoder gradient bucket on one rank.
        loss_sums = hidden_states[:, 0, 0].float() * 0.0
        if valid_positions.numel() > 0:
            example_indices = valid_positions[:, 0]
            time_indices = valid_positions[:, 1]
            supervised_hidden = hidden_states[
                example_indices,
                time_indices,
            ]
            supervised_labels = shift_labels[
                example_indices,
                time_indices,
            ]

            def block_cross_entropy(
                block_hidden: torch.Tensor,
                block_labels: torch.Tensor,
            ) -> torch.Tensor:
                block_logits = decoder_head(block_hidden).float()
                _require_finite(block_logits, "decoder supervised logits")
                losses = F.cross_entropy(
                    block_logits,
                    block_labels,
                    reduction="none",
                )
                return _require_finite(losses.float(), "decoder token loss")

            chunk_size = DECODER_LOSS_TOKEN_CHUNK_SIZE
            for start in range(0, supervised_hidden.size(0), chunk_size):
                end = min(start + chunk_size, supervised_hidden.size(0))
                block_hidden = supervised_hidden[start:end]
                block_labels = supervised_labels[start:end]
                if torch.is_grad_enabled() and block_hidden.requires_grad:
                    block_losses = activation_checkpoint(
                        block_cross_entropy,
                        block_hidden,
                        block_labels,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    block_losses = block_cross_entropy(
                        block_hidden,
                        block_labels,
                    )
                loss_sums = loss_sums.index_add(
                    0,
                    example_indices[start:end],
                    block_losses,
                )

        valid_counts = valid_tokens.sum(dim=1).clamp_min(1).to(loss_sums.dtype)
        per_example = _require_finite(
            loss_sums / valid_counts,
            "per-example decoder loss",
        )
        if reduction == "none":
            return per_example
        return _require_finite(per_example.mean(), "decoder mean loss")

    def forward(
        self,
        encoder_ids: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
        decoder_label_mask: torch.Tensor,
        validation_noise_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Return the configured skill reconstruction cross-entropy."""

        clean_latent = self.encode_batch(encoder_ids, encoder_mask)
        noise_angles = clean_latent.new_zeros(clean_latent.size(0))
        decoder_latent = clean_latent
        explicit_validation_noise = validation_noise_seed is not None
        if explicit_validation_noise and self.training:
            raise ValueError("Explicit validation noise requires model.eval()")
        if self.training or explicit_validation_noise:
            noise_generator: Optional[torch.Generator] = None
            noise_scale = float(getattr(self, "latent_noise_scale", 0.0))
            if explicit_validation_noise:
                # Score the fully ramped training mixture with dropout disabled,
                # without consuming or changing the training RNG stream.
                noise_generator = torch.Generator(device=clean_latent.device)
                noise_generator.manual_seed(int(validation_noise_seed))
                noise_scale = 1.0
            decoder_latent, noise_angles = angular_perturb_latent(
                clean_latent,
                angle_std=float(getattr(self, "latent_noise_angle_std", 0.0)),
                max_angle=float(getattr(self, "latent_noise_max_angle", 0.0)),
                probability=float(getattr(self, "latent_noise_prob", 0.0)),
                scale=noise_scale,
                generator=noise_generator,
            )
        prefix = self.prefix_from_latent(decoder_latent)
        per_example_losses = self._decoder_lm_loss(
            prefix,
            decoder_ids,
            decoder_mask,
            decoder_label_mask,
            reduction="none",
        )
        reconstruction_loss = _require_finite(
            per_example_losses.mean(),
            "skill reconstruction loss",
        )
        reconstruction_support = reconstruction_loss.new_tensor(
            float(per_example_losses.numel())
        )
        self.last_loss_metrics = {
            "total_loss": reconstruction_loss.detach(),
            "reconstruction_loss": reconstruction_loss.detach(),
            "reconstruction_support_count": reconstruction_support.detach(),
            "latent_noise_angle_mean": noise_angles.mean().detach(),
            "latent_noise_applied_fraction": noise_angles.gt(0).float().mean().detach(),
        }
        return reconstruction_loss


class SkillAutoencoderCodec:
    """Frozen skill encoder/decoder artifact."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 512,
        backbone_dtype: object = "auto",
        load_decoder: bool = True,
        attention_implementation: Optional[str] = None,
    ):
        checkpoint = Path(checkpoint_dir).expanduser().resolve()
        self.checkpoint = checkpoint
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("Skill codec max_length must be positive")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.backbone_dtype = resolve_backbone_dtype(
            backbone_dtype,
            device=self.device,
        )
        self.resolved_backbone_dtype_name = backbone_dtype_name(self.backbone_dtype)

        state_path = checkpoint / "vector_to_prefix.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing skill codec state: {state_path}")
        prefix_state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(prefix_state, dict):
            raise RuntimeError("Skill codec state must be a mapping")
        self.num_prefix_tokens = int(prefix_state["num_prefix_tokens"])
        self.encoder_dim = int(prefix_state["encoder_dim"])
        self.decoder_dim = int(prefix_state["decoder_dim"])
        if min(self.num_prefix_tokens, self.encoder_dim, self.decoder_dim) <= 0:
            raise RuntimeError("Skill codec dimensions must all be positive")
        self.codec_id = str(prefix_state.get("codec_id") or "").strip()
        if not self.codec_id:
            raise RuntimeError("Skill codec artifact has an empty codec_id")
        self.schema_version = str(prefix_state.get("schema_version") or "")
        if self.schema_version != CODEC_SCHEMA_VERSION:
            raise RuntimeError(
                "Codec schema mismatch; the v3 codec artifact is incompatible; "
                "must be retrained: "
                f"checkpoint={self.schema_version!r} "
                f"expected={CODEC_SCHEMA_VERSION!r}"
            )
        self.skill_view_version = str(prefix_state.get("skill_view_version") or "")
        if self.skill_view_version != CODEC_SKILL_VIEW_VERSION:
            raise RuntimeError(
                "Codec skill-view mismatch; the success-only corpus must be "
                "retrained: "
                f"checkpoint={self.skill_view_version!r} "
                f"expected={CODEC_SKILL_VIEW_VERSION!r}"
            )
        expected_policy = {
            "encoder_routing_version": CODEC_ENCODER_ROUTING_VERSION,
            "numeric_policy_version": CODEC_NUMERIC_POLICY_VERSION,
            "training_example_policy": CODEC_TRAINING_EXAMPLE_POLICY,
            "skill_input_version": CODEC_SKILL_INPUT_VERSION,
        }
        for key, expected in expected_policy.items():
            observed = str(prefix_state.get(key) or "")
            if observed != expected:
                raise RuntimeError(
                    f"Skill codec policy mismatch for {key}: "
                    f"checkpoint={observed!r} expected={expected!r}"
                )
        required_flags = (
            "skill_only_encoder",
            "frozen_original_query_encoder",
            "role_free_skill_json",
            "training_frozen_contract",
        )
        missing_flags = [
            key for key in required_flags if prefix_state.get(key) is not True
        ]
        if missing_flags:
            raise RuntimeError(
                "Skill codec checkpoint is missing v3 contract flags: "
                + ", ".join(missing_flags)
            )
        try:
            artifact_policy = validate_codec_artifact_policy(prefix_state)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Skill codec v3 artifact policy is invalid: {exc}"
            ) from exc
        self.encoder_pooling = str(artifact_policy["encoder_pooling"])
        self.query_encoder_pooling = str(artifact_policy["query_encoder_pooling"])
        self.decoder_task_conditioned = bool(
            artifact_policy["decoder_task_conditioned"]
        )
        self.training_objective = str(artifact_policy["training_objective"])
        self.latent_noise_angle_std = float(artifact_policy["latent_noise_angle_std"])
        self.latent_noise_max_angle = float(artifact_policy["latent_noise_max_angle"])
        self.latent_noise_prob = float(artifact_policy["latent_noise_prob"])
        self.latent_noise_ramp_epochs = int(artifact_policy["latent_noise_ramp_epochs"])
        self.encoder_lora_last_n_layers = int(
            artifact_policy["encoder_lora_last_n_layers"]
        )

        self.chains_sha256 = str(prefix_state.get("chains_sha256") or "").strip()
        if len(self.chains_sha256) != 64:
            raise RuntimeError("Skill codec artifact has no valid chains_sha256")
        self.decoder_prompt = str(prefix_state.get("decoder_prompt") or "")
        self.encoder_skill_prompt = str(prefix_state.get("encoder_skill_prompt") or "")
        self.encoder_query_prompt = str(prefix_state.get("encoder_query_prompt") or "")
        if not self.decoder_prompt.strip():
            raise RuntimeError("Frozen skill decoder prompt must be non-empty")
        try:
            prompt_task_conditioned = decoder_prompt_uses_task_context(
                self.decoder_prompt
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Frozen skill decoder prompt is invalid: {exc}"
            ) from exc
        if prompt_task_conditioned != self.decoder_task_conditioned:
            raise RuntimeError(
                "Frozen skill decoder prompt disagrees with "
                "decoder_task_conditioned artifact metadata"
            )
        if "{skill}" not in self.encoder_skill_prompt or any(
            token in self.encoder_skill_prompt
            for token in ("{task_type}", "{goal}", "{context}")
        ):
            raise RuntimeError(
                "Frozen skill encoder prompt must contain only a {skill} field"
            )
        if (
            not all(
                token in self.encoder_query_prompt
                for token in ("{task_type}", "{goal}")
            )
            or "{skill}" in self.encoder_query_prompt
        ):
            raise RuntimeError(
                "Frozen query encoder prompt must contain {task_type} and "
                "{goal}, and must not contain {skill}"
            )
        self.training_backbone_dtype = str(
            prefix_state.get("training_backbone_dtype") or ""
        )
        if self.training_backbone_dtype not in {"bfloat16", "float32"}:
            raise RuntimeError(
                "Skill codec artifact has no valid training_backbone_dtype"
            )

        self.query_encoder_source = str(
            prefix_state.get("query_encoder_source") or ""
        ).strip()
        self.query_encoder_source_fingerprint = str(
            prefix_state.get("query_encoder_source_fingerprint") or ""
        ).strip()
        if not self.query_encoder_source:
            raise RuntimeError("Skill codec artifact has no query encoder source")
        try:
            int(self.query_encoder_source_fingerprint, 16)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Skill codec query encoder fingerprint is not hexadecimal"
            ) from exc
        if len(self.query_encoder_source_fingerprint) != 64:
            raise RuntimeError(
                "Skill codec query encoder fingerprint must contain 64 hex digits"
            )
        observed_query_fingerprint = model_source_fingerprint(self.query_encoder_source)
        if observed_query_fingerprint != self.query_encoder_source_fingerprint:
            raise RuntimeError(
                "Original frozen query encoder source changed after codec "
                "training: "
                f"expected={self.query_encoder_source_fingerprint} "
                f"observed={observed_query_fingerprint}"
            )

        self.loss_mode = str(prefix_state.get("loss_mode") or "").strip().lower()
        if self.loss_mode != "reconstruction":
            raise RuntimeError(
                "Codec artifacts require contextual reconstruction training, "
                f"observed={self.loss_mode!r}"
            )

        encoder_dir = checkpoint / "encoder_prime"
        if not encoder_dir.exists():
            raise FileNotFoundError(
                f"Missing frozen skill encoder directory: {encoder_dir}"
            )
        decoder_dir = checkpoint / "decoder"
        if load_decoder and not decoder_dir.exists():
            raise FileNotFoundError(
                f"Missing frozen skill decoder directory: {decoder_dir}"
            )

        with hide_optional_audio_packages():
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

            attention_kwargs = _backbone_attention_kwargs(
                attention_implementation,
                self.backbone_dtype,
            )
            encoder_tokenizer_dir = checkpoint / "encoder_tokenizer"
            decoder_tokenizer_dir = checkpoint / "decoder_tokenizer"
            if not encoder_tokenizer_dir.exists():
                raise FileNotFoundError(
                    f"Missing frozen skill encoder tokenizer: "
                    f"{encoder_tokenizer_dir}"
                )
            self.encoder_tokenizer = AutoTokenizer.from_pretrained(
                encoder_tokenizer_dir,
                trust_remote_code=True,
            )
            ensure_pad_token(self.encoder_tokenizer)
            self.encoder_tokenizer.padding_side = "left"
            self.encoder_prime = AutoModel.from_pretrained(
                encoder_dir,
                dtype=self.backbone_dtype,
                trust_remote_code=True,
                **attention_kwargs,
            ).to(self.device)
            self.encoder_context_window = get_context_window(
                self.encoder_prime.config,
                self.encoder_tokenizer,
            )

            self.decoder_tokenizer = None
            self.decoder = None
            self.decoder_context_window = 0
            if load_decoder:
                if not decoder_tokenizer_dir.exists():
                    raise FileNotFoundError(
                        f"Missing frozen skill decoder tokenizer: "
                        f"{decoder_tokenizer_dir}"
                    )
                self.decoder_tokenizer = AutoTokenizer.from_pretrained(
                    decoder_tokenizer_dir,
                    trust_remote_code=True,
                )
                configure_decoder_tokenizer(self.decoder_tokenizer)
                self.decoder = AutoModelForCausalLM.from_pretrained(
                    decoder_dir,
                    dtype=self.backbone_dtype,
                    trust_remote_code=True,
                    **attention_kwargs,
                ).to(self.device)
                self.decoder_context_window = get_context_window(
                    self.decoder.config,
                    self.decoder_tokenizer,
                )
                configure_decoder_tokenizer(
                    self.decoder_tokenizer,
                    context_window=self.decoder_context_window,
                )

        self.vector_to_prefix = None
        if load_decoder:
            self.vector_to_prefix = nn.Sequential(
                nn.LayerNorm(self.encoder_dim),
                nn.Linear(
                    self.encoder_dim,
                    self.num_prefix_tokens * self.decoder_dim,
                ),
            ).to(self.device, dtype=torch.float32)
            self.vector_to_prefix.load_state_dict(prefix_state["state_dict"])

        self.encoder_prime.eval()
        self.encoder_prime.requires_grad_(False)
        if self.decoder is not None:
            self.decoder.eval()
            self.decoder.requires_grad_(False)
        if self.vector_to_prefix is not None:
            self.vector_to_prefix.eval()
            self.vector_to_prefix.requires_grad_(False)
        self._encoder_device = self.device
        self._decoder_device = self.device if self.decoder is not None else None

    def activate_encoder(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "SkillAutoencoderCodec":
        target = torch.device(device or self.device)
        self.device = target
        self.encoder_prime.to(target)
        self._encoder_device = target
        return self

    def offload_encoder(self) -> "SkillAutoencoderCodec":
        previous = self._encoder_device
        self.encoder_prime.to("cpu")
        self._encoder_device = torch.device("cpu")
        if previous is not None and previous.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def activate_decoder(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "SkillAutoencoderCodec":
        if self.decoder is None or self.vector_to_prefix is None:
            raise RuntimeError("This skill codec was loaded without a decoder")
        target = torch.device(device or self.device)
        self.device = target
        self.decoder.to(target)
        self.vector_to_prefix.to(target, dtype=torch.float32)
        self._decoder_device = target
        return self

    def offload_decoder(self) -> "SkillAutoencoderCodec":
        if self.decoder is None or self.vector_to_prefix is None:
            return self
        previous = self._decoder_device
        self.decoder.to("cpu")
        self.vector_to_prefix.to("cpu", dtype=torch.float32)
        self._decoder_device = torch.device("cpu")
        if previous is not None and previous.type == "cuda":
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def encode(
        self,
        texts: list[Any],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Encode canonical skill JSON only; query routing is not supported."""

        if not texts:
            return np.zeros((0, self.encoder_dim), dtype=np.float32)
        if int(batch_size) <= 0:
            raise ValueError("Skill encode batch_size must be positive")
        self.activate_encoder()
        rendered = [
            skill_encoder_text(
                value,
                skill_prompt=self.encoder_skill_prompt,
            )
            for value in texts
        ]
        chunks: list[np.ndarray] = []
        for start in range(0, len(rendered), int(batch_size)):
            batch_texts = rendered[start : start + int(batch_size)]
            inputs = self.encoder_tokenizer(
                batch_texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            lengths = inputs["attention_mask"].sum(dim=1)
            if bool((lengths > self.max_length).any()):
                offending = lengths[lengths > self.max_length].tolist()
                raise ValueError(
                    "Skill encoder input exceeds max_length; truncation is "
                    f"forbidden: max_length={self.max_length} "
                    f"offending={offending[:8]}"
                )
            if bool((lengths > self.encoder_context_window).any()):
                offending = lengths[lengths > self.encoder_context_window].tolist()
                raise ValueError(
                    "Skill encoder input exceeds model context window: "
                    f"context_window={self.encoder_context_window} "
                    f"offending={offending[:8]}"
                )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self.encoder_prime(
                **inputs,
                position_ids=position_ids_from_attention_mask(inputs["attention_mask"]),
                return_dict=True,
            )
            latent = pool_encoder_hidden_states(
                output.last_hidden_state,
                inputs["attention_mask"],
                pooling_profile=self.encoder_pooling,
            )
            chunk = latent.cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(chunk).all():
                raise FloatingPointError("Runtime skill encoder produced NaN/Inf")
            chunks.append(chunk)
        return np.concatenate(chunks, axis=0)

    @torch.no_grad()
    def decode_results(
        self,
        latents: Union[np.ndarray, torch.Tensor],
        contexts: list[object],
        batch_size: int = 4,
        max_new_tokens: int = 1024,
    ) -> list[ConditionalDecoderResult]:
        if self.decoder is None or self.decoder_tokenizer is None:
            raise RuntimeError("This skill codec was loaded without a decoder")
        if self.vector_to_prefix is None:
            raise RuntimeError("This skill codec has no latent projector")
        if int(batch_size) <= 0 or int(max_new_tokens) <= 0:
            raise ValueError("Decode batch_size and max_new_tokens must be positive")
        if isinstance(latents, np.ndarray):
            latents_tensor = torch.from_numpy(latents.astype(np.float32, copy=False))
        else:
            latents_tensor = latents.detach().cpu().float()
        if latents_tensor.ndim != 2 or latents_tensor.size(1) != self.encoder_dim:
            raise ValueError(
                "Skill latent shape mismatch: "
                f"got={tuple(latents_tensor.shape)} "
                f"expected=(*, {self.encoder_dim})"
            )
        if len(contexts) != int(latents_tensor.size(0)):
            raise ValueError(
                "Decoder context/latent length mismatch: "
                f"{len(contexts)} != {int(latents_tensor.size(0))}"
            )
        if latents_tensor.size(0) == 0:
            return []

        self.activate_decoder()
        results: list[ConditionalDecoderResult] = []
        for start in range(0, latents_tensor.size(0), int(batch_size)):
            latent = latents_tensor[start : start + int(batch_size)].to(
                self.device,
                dtype=torch.float32,
            )
            with _disable_autocast_for(latent):
                latent = _normalize_fp32(latent, "runtime decoder latent")
                prefix = self.vector_to_prefix(latent).view(
                    latent.size(0),
                    self.num_prefix_tokens,
                    self.decoder_dim,
                )
                prefix = _require_finite(
                    prefix.float(),
                    "runtime decoder prefix",
                )
            prompt_rows = [
                apply_decoder_context_prompt(self.decoder_prompt, context=context)
                for context in contexts[start : start + int(latent.size(0))]
            ]
            prompt_inputs = self.decoder_tokenizer(
                prompt_rows,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1)
            if bool((prompt_lengths > self.max_length).any()):
                offending = prompt_lengths[prompt_lengths > self.max_length].tolist()
                raise ValueError(
                    "Skill decoder prompt exceeds max_length; truncation is "
                    f"forbidden: max_length={self.max_length} "
                    f"offending={offending[:8]}"
                )
            full_generation_lengths = (
                prompt_lengths + self.num_prefix_tokens + int(max_new_tokens)
            )
            if bool((full_generation_lengths > self.decoder_context_window).any()):
                offending = full_generation_lengths[
                    full_generation_lengths > self.decoder_context_window
                ].tolist()
                raise ValueError(
                    "Skill decoder prefix+prompt+generation budget exceeds "
                    f"context_window={self.decoder_context_window}; "
                    f"offending={offending[:8]}"
                )
            prompt_ids = prompt_inputs["input_ids"].to(self.device)
            prompt_attention = prompt_inputs["attention_mask"].to(self.device)
            prompt_embeddings = self.decoder.get_input_embeddings()(prompt_ids)
            _require_finite(prompt_embeddings, "runtime decoder prompt embeddings")
            decoder_prefix = prefix.to(dtype=prompt_embeddings.dtype)
            inputs_embeds, attention_mask = prepend_prefix_to_left_padded_prompt(
                decoder_prefix,
                prompt_embeddings,
                prompt_attention,
            )
            eos_token_ids = decoder_eos_token_ids(
                self.decoder,
                self.decoder_tokenizer,
            )
            if not eos_token_ids:
                raise RuntimeError(
                    "Skill decoder tokenizer/model must define eos_token_id"
                )
            generation_eos_token_ids: int | list[int] = (
                eos_token_ids[0] if len(eos_token_ids) == 1 else list(eos_token_ids)
            )
            output_ids = self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids_from_attention_mask(attention_mask),
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                num_beams=1,
                eos_token_id=generation_eos_token_ids,
                pad_token_id=self.decoder_tokenizer.pad_token_id,
            )
            decoded_rows = self.decoder_tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
            )
            if len(decoded_rows) != len(output_ids):
                raise RuntimeError(
                    "Skill decoder tokenizer returned the wrong row count: "
                    f"decoded={len(decoded_rows)} generated={len(output_ids)}"
                )
            for row_index, (ids, decoded_text) in enumerate(
                zip(output_ids, decoded_rows)
            ):
                saw_eos = any(
                    bool(ids.eq(int(eos_token_id)).any())
                    for eos_token_id in eos_token_ids
                )
                results.append(
                    ConditionalDecoderResult(
                        batch_row=start + row_index,
                        status="ok" if saw_eos else "max_tokens_without_eos",
                        text=str(decoded_text).strip() if saw_eos else "",
                        max_new_tokens=int(max_new_tokens),
                        recognized_eos_token_ids=eos_token_ids,
                    )
                )
        return results

    @torch.no_grad()
    def decode(
        self,
        latents: Union[np.ndarray, torch.Tensor],
        contexts: list[object],
        batch_size: int = 4,
        max_new_tokens: int = 1024,
    ) -> list[str]:
        """Fail if any skill decode does not terminate with an EOS token."""

        results = self.decode_results(
            latents,
            contexts=contexts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        for result in results:
            if result.status != "ok":
                raise ConditionalDecoderGenerationError(
                    batch_row=result.batch_row,
                    max_new_tokens=result.max_new_tokens,
                    recognized_eos_token_ids=result.recognized_eos_token_ids,
                )
        return [result.text for result in results]


class FrozenQueryEncoder:
    """Independent, immutable original embedding model for task queries."""

    def __init__(
        self,
        model_path: Union[str, Path],
        *,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 512,
        query_prompt: str = DEFAULT_ENCODER_QUERY_PROMPT,
        backbone_dtype: object = "auto",
        attention_implementation: Optional[str] = None,
        expected_source_fingerprint: Optional[str] = None,
        pooling_profile: object = CODEC_QUERY_ENCODER_POOLING,
    ) -> None:
        raw_source = str(model_path or "").strip()
        if not raw_source:
            raise ValueError("Frozen query encoder model_path is required")
        candidate = Path(raw_source).expanduser()
        self.source_path = str(
            candidate.resolve() if candidate.exists() else raw_source
        )
        self.source_fingerprint = model_source_fingerprint(self.source_path)
        expected = str(expected_source_fingerprint or "").strip()
        if expected and expected != self.source_fingerprint:
            raise RuntimeError(
                "Frozen query encoder source fingerprint mismatch: "
                f"expected={expected} observed={self.source_fingerprint}"
            )
        self.query_prompt = str(query_prompt or DEFAULT_ENCODER_QUERY_PROMPT)
        self.pooling_profile = normalize_encoder_pooling_profile(pooling_profile)
        if not all(token in self.query_prompt for token in ("{task_type}", "{goal}")):
            raise ValueError(
                "Frozen query encoder prompt must contain {task_type} and {goal}"
            )
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("Frozen query encoder max_length must be positive")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.backbone_dtype = resolve_backbone_dtype(
            backbone_dtype,
            device=self.device,
        )
        self.resolved_backbone_dtype_name = backbone_dtype_name(self.backbone_dtype)

        with hide_optional_audio_packages():
            from transformers import AutoModel, AutoTokenizer

            attention_kwargs = _backbone_attention_kwargs(
                attention_implementation,
                self.backbone_dtype,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.source_path,
                trust_remote_code=True,
            )
            ensure_pad_token(self.tokenizer)
            self.tokenizer.padding_side = "left"
            self.encoder = AutoModel.from_pretrained(
                self.source_path,
                dtype=self.backbone_dtype,
                trust_remote_code=True,
                **attention_kwargs,
            ).to(self.device)
        self.encoder_context_window = get_context_window(
            self.encoder.config,
            self.tokenizer,
        )
        self.encoder_dim = get_hidden_size(self.encoder.config)
        self.dim = self.encoder_dim
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self._active_device = self.device

    def metadata(self) -> dict[str, object]:
        return {
            "encoder_routing_version": CODEC_ENCODER_ROUTING_VERSION,
            "query_encoder_source": self.source_path,
            "query_encoder_source_fingerprint": self.source_fingerprint,
            "query_encoder_dim": self.encoder_dim,
            "query_encoder_prompt": self.query_prompt,
            "query_encoder_backbone_dtype": self.resolved_backbone_dtype_name,
            "actual_runtime_query_encoder_pooling": self.pooling_profile,
            "query_encoder_frozen": True,
        }

    def activate(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "FrozenQueryEncoder":
        target = torch.device(device or self.device)
        self.device = target
        self.encoder.to(target)
        self._active_device = target
        return self

    def offload(self) -> "FrozenQueryEncoder":
        previous = self._active_device
        self.encoder.to("cpu")
        self._active_device = torch.device("cpu")
        if previous is not None and previous.type == "cuda":
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def encode(
        self,
        goals: list[str],
        *,
        task_types: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        if len(goals) != len(task_types):
            raise ValueError(
                "Frozen query goal/task_type length mismatch: "
                f"{len(goals)} != {len(task_types)}"
            )
        if not goals:
            return np.zeros((0, self.encoder_dim), dtype=np.float32)
        if int(batch_size) <= 0:
            raise ValueError("Frozen query encode batch_size must be positive")
        self.activate()
        rendered = [
            frozen_query_encoder_text(
                task_type=task_type,
                goal=goal,
                query_prompt=self.query_prompt,
            )
            for task_type, goal in zip(task_types, goals)
        ]
        chunks: list[np.ndarray] = []
        for start in range(0, len(rendered), int(batch_size)):
            batch_texts = rendered[start : start + int(batch_size)]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            lengths = inputs["attention_mask"].sum(dim=1)
            if bool((lengths > self.max_length).any()):
                offending = lengths[lengths > self.max_length].tolist()
                raise ValueError(
                    "Frozen query encoder input exceeds max_length; truncation "
                    f"is forbidden: max_length={self.max_length} "
                    f"offending={offending[:8]}"
                )
            if bool((lengths > self.encoder_context_window).any()):
                offending = lengths[lengths > self.encoder_context_window].tolist()
                raise ValueError(
                    "Frozen query encoder input exceeds model context window: "
                    f"context_window={self.encoder_context_window} "
                    f"offending={offending[:8]}"
                )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self.encoder(
                **inputs,
                position_ids=position_ids_from_attention_mask(inputs["attention_mask"]),
                return_dict=True,
            )
            latent = pool_encoder_hidden_states(
                output.last_hidden_state,
                inputs["attention_mask"],
                pooling_profile=self.pooling_profile,
            )
            chunk = latent.cpu().numpy().astype(np.float32, copy=False)
            if not np.isfinite(chunk).all():
                raise FloatingPointError("Frozen query encoder produced NaN/Inf")
            chunks.append(chunk)
        return np.concatenate(chunks, axis=0)
