"""Shared helpers for classic Codec training."""
from __future__ import annotations
import math
import random
from typing import Iterator, Optional
import torch
from torch.utils.data import Sampler
from .model import SkillAutoencoder, inject_lora
from .lora import selected_transformer_layer_indices
from .training_contract import (CODEC_TRAIN_BATCH_TAIL_POLICIES, CODEC_TRAIN_BATCH_TAIL_POLICY_LEGACY, CODEC_TRAIN_BATCH_TAIL_POLICY_REPEAT_EPOCH_PREFIX)

class ReconstructionBatchSampler(Sampler[list[int]]):
    """Build plain deterministic batches of independent reconstruction rows."""

    def __init__(
        self,
        examples: list[dict],
        batch_size: int,
        *,
        seed: int,
        shuffle: bool,
        pad_to_multiple: int = 1,
        tail_policy: str = CODEC_TRAIN_BATCH_TAIL_POLICY_LEGACY,
    ):
        if not examples:
            raise ValueError("Reconstruction batch plan cannot be empty")
        if int(batch_size) <= 0:
            raise ValueError("Reconstruction batch_size must be positive")
        self.example_count = len(examples)
        self.reconstruction_batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.pad_to_multiple = max(1, int(pad_to_multiple))
        if tail_policy not in CODEC_TRAIN_BATCH_TAIL_POLICIES:
            raise ValueError(
                f"Unsupported reconstruction tail policy: {tail_policy!r}"
            )
        self.tail_policy = str(tail_policy)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        indices = list(range(self.example_count))
        if self.shuffle:
            random.Random(self.seed + self.epoch * 104729).shuffle(indices)
        width = self.reconstruction_batch_size
        if (
            self.tail_policy
            == CODEC_TRAIN_BATCH_TAIL_POLICY_REPEAT_EPOCH_PREFIX
        ):
            tail_padding = (-len(indices)) % width
            if tail_padding:
                epoch_prefix = list(indices)
                indices.extend(
                    epoch_prefix[offset % len(epoch_prefix)]
                    for offset in range(tail_padding)
                )
        batches = [
            indices[start : start + width]
            for start in range(0, len(indices), width)
        ]
        padding_source = [list(batch) for batch in batches]
        padding = (-len(batches)) % self.pad_to_multiple
        for offset in range(padding):
            batches.append(list(padding_source[offset % len(padding_source)]))
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())

def _reconstruction_lr_lambda(
    *,
    total_steps: int,
    warmup_ratio: float,
):
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError("Codec scheduler requires positive total_steps")
    warmup = int(total_steps * max(0.0, float(warmup_ratio)))

    def schedule(step: int) -> float:
        step = max(0, int(step))
        if warmup > 0 and step < warmup:
            return max(1e-6, float(step + 1) / float(warmup))
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule

def _capture_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

def _restore_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"Best codec state contains unknown parameters: {missing[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))

def _decoder_transformer_layers(decoder: torch.nn.Module) -> list[torch.nn.Module]:
    candidates = [
        getattr(decoder, "layers", None),
        getattr(getattr(decoder, "model", None), "layers", None),
        getattr(getattr(decoder, "transformer", None), "h", None),
        getattr(getattr(decoder, "gpt_neox", None), "layers", None),
    ]
    for value in candidates:
        if value is not None:
            return list(value)
    return []

def _inject_transformer_block_lora(
    backbone: torch.nn.Module,
    *,
    component: str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    last_n_layers: Optional[int] = None,
) -> tuple[int, tuple[str, ...]]:
    """Inject LoRA into every Linear of a deterministic block selection.

    Scoping to transformer blocks deliberately leaves input embeddings and a
    decoder's top-level output projection untouched.  Those weights can be
    shared/tied, and replacing only one reference would silently break that
    invariant.  Every attention and MLP Linear inside each selected block is
    still adapted.
    """

    layers = _decoder_transformer_layers(backbone)
    if not layers:
        raise RuntimeError(
            f"Could not locate {component} transformer layers for LoRA injection"
        )
    all_indices = selected_transformer_layer_indices(len(layers))
    if last_n_layers is None:
        indices = all_indices
    else:
        last_n_layers = int(last_n_layers)
        if not 0 < last_n_layers <= len(all_indices):
            raise ValueError(f"Invalid {component} LoRA last_n_layers={last_n_layers}")
        indices = all_indices[-last_n_layers:]
    if not indices:
        return 0, ()

    targets = tuple(
        sorted(
            f"layers.{layer_index}.{name}"
            for layer_index in indices
            for name, child in layers[layer_index].named_modules()
            if name and isinstance(child, torch.nn.Linear)
        )
    )
    if not targets:
        raise RuntimeError(
            f"No {component} Linear modules were available for all-layer LoRA"
        )

    inserted = 0
    for layer_index in indices:
        inserted += inject_lora(
            layers[layer_index],
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
    if inserted != len(targets):
        raise RuntimeError(
            f"{component} LoRA target discovery/injection mismatch: "
            f"discovered={len(targets)} inserted={inserted}"
        )
    return len(indices), targets

def _configure_encoder_training(
    model: SkillAutoencoder,
    *,
    enabled: bool,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    last_n_layers: Optional[int] = None,
) -> int:
    """Keep the embedding backbone frozen; --train_encoder enables encoder LoRA."""
    model.encoder_prime.requires_grad_(False)
    model.encoder_lora_module_count = 0
    model.encoder_lora_target_names = ()
    if not enabled:
        # A disabled encoder LoRA leaves the backbone as a true feature extractor.
        disable_checkpointing = getattr(
            model.encoder_prime, "gradient_checkpointing_disable", None
        )
        if callable(disable_checkpointing):
            disable_checkpointing()
        return 0
    selected_layers, targets = _inject_transformer_block_lora(
        model.encoder_prime,
        component="encoder",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        last_n_layers=last_n_layers,
    )
    model.encoder_lora_module_count = len(targets)
    model.encoder_lora_target_names = targets
    return selected_layers

def _configure_decoder_training(
    model: SkillAutoencoder,
    *,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
) -> int:
    """Freeze the decoder and inject mergeable LoRA into configured blocks."""
    model.decoder.requires_grad_(False)
    model.decoder_lora_module_count = 0
    model.decoder_lora_target_names = ()
    selected_layers, targets = _inject_transformer_block_lora(
        model.decoder,
        component="decoder",
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    model.decoder_lora_module_count = len(targets)
    model.decoder_lora_target_names = targets
    return selected_layers

from torch.utils.data import DataLoader
from accelerate import Accelerator

def _assert_finite_loss_and_metrics(
    loss: torch.Tensor,
    metrics: dict[str, torch.Tensor],
    *,
    stage: str,
    epoch: int,
) -> None:
    tensor_metrics = [
        value.detach().float().reshape(-1)
        for value in (loss, *metrics.values())
        if torch.is_tensor(value)
    ]
    if loss.device.type == "cuda":
        # Preserve strict detection without synchronizing once per metric.
        torch._assert_async(torch.isfinite(torch.cat(tensor_metrics)).all())
        return
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise FloatingPointError(
            f"Codec loss is non-finite at epoch={epoch} stage={stage}: "
            f"{loss.detach().float().cpu().item()}"
        )
    non_finite = sorted(
        name
        for name, value in metrics.items()
        if torch.is_tensor(value)
        and not bool(torch.isfinite(value.detach()).all().item())
    )
    if non_finite:
        raise FloatingPointError(
            "Codec metrics contain non-finite values at "
            f"epoch={epoch} stage={stage}: {non_finite}"
        )

def _validation_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
    *,
    noise_seed: int,
) -> dict[str, float]:
    # Do not prepare/shard this loader with Accelerate: assign every complete
    # validation text batch to exactly one rank, run the unwrapped module,
    # and reduce sufficient statistics once at the end. Unequal local batch
    # counts are safe because validation does not invoke DDP forward collectives.
    raw_model = accelerator.unwrap_model(model)
    raw_model.eval()
    metric_names = sorted(
        set((getattr(raw_model, "last_loss_metrics", {}) or {}).keys())
        | {"total_loss"}
    )
    if not metric_names:
        return {"total_loss": float("inf")}
    local_sums = {
        name: torch.zeros((), device=accelerator.device, dtype=torch.float64)
        for name in metric_names
    }
    local_exposures = torch.zeros(
        (), device=accelerator.device, dtype=torch.float64
    )
    local_objective_exposures = torch.zeros(
        (), device=accelerator.device, dtype=torch.float64
    )
    local_batches = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    for batch_index, cpu_batch in enumerate(loader):
        if batch_index % int(accelerator.num_processes) != int(
            accelerator.process_index
        ):
            continue
        batch = {
            key: (
                value.to(accelerator.device, non_blocking=True)
                if torch.is_tensor(value)
                else value
            )
            for key, value in cpu_batch.items()
        }
        # Keep validation in eval mode so dropout is disabled while scoring the
        # configured reconstruction distribution. The global batch index keeps
        # any configured perturbations stable across epochs and ranks.
        loss = raw_model(
            **batch,
            validation_noise_seed=int(noise_seed) + int(batch_index),
        )
        batch_size = int(batch["encoder_ids"].shape[0])
        local_metrics = dict(getattr(raw_model, "last_loss_metrics", {}) or {})
        local_metrics["total_loss"] = loss.detach()
        _assert_finite_loss_and_metrics(
            loss,
            local_metrics,
            stage="reconstruction",
            epoch=0,
        )
        if set(local_metrics) != set(metric_names):
            raise RuntimeError(
                "Codec validation metric schema changed between batches: "
                f"expected={metric_names} observed={sorted(local_metrics)}"
            )
        objective_exposure = local_metrics.get(
            "reconstruction_support_count"
        )
        objective_exposure = (
            objective_exposure.detach().to(dtype=torch.float64)
            if objective_exposure is not None
            else torch.tensor(
                float(batch_size),
                device=accelerator.device,
                dtype=torch.float64,
            )
        )
        for name, value in local_metrics.items():
            scalar = value.detach().to(dtype=torch.float64)
            if name.endswith("_support_count"):
                local_sums[name] = local_sums[name] + scalar
            else:
                local_sums[name] = (
                    local_sums[name] + scalar * objective_exposure
                )
        local_exposures = local_exposures + batch_size
        local_objective_exposures = (
            local_objective_exposures + objective_exposure
        )
        local_batches = local_batches + 1

    reduced = accelerator.reduce(
        torch.stack(
            [
                local_exposures,
                local_objective_exposures,
                local_batches,
                *[local_sums[name] for name in metric_names],
            ]
        ),
        reduction="sum",
    )
    global_exposures = float(reduced[0].item())
    global_objective_exposures = float(reduced[1].item())
    global_batches = float(reduced[2].item())
    if (
        global_exposures <= 0
        or global_objective_exposures <= 0
        or global_batches <= 0
    ):
        return {"total_loss": float("inf")}
    output: dict[str, float] = {
        "batch_count": global_batches,
        "example_exposures": global_exposures,
        "objective_weight_exposures": global_objective_exposures,
    }
    for index, name in enumerate(metric_names, start=3):
        total = float(reduced[index].item())
        output[name] = (
            total
            if name.endswith("_support_count")
            else total / global_objective_exposures
        )
    return output
