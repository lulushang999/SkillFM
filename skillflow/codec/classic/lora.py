"""Deterministic LoRA transformer-layer selection for codec training."""

from __future__ import annotations


LORA_SCOPE_ALL = "all_transformer_layers"


def selected_transformer_layer_indices(
    total_layers: int,
) -> tuple[int, ...]:
    """Select every transformer block for the formal codec LoRA."""

    total = int(total_layers)
    if total <= 0:
        raise ValueError("LoRA target model must expose at least one transformer layer")
    return tuple(range(total))


__all__ = [
    "LORA_SCOPE_ALL",
    "selected_transformer_layer_indices",
]
