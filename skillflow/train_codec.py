"""Train the contextual denoising Codec from the released training skills.

Run with ``torchrun --nproc_per_node=N -m skillflow.train_codec``. The query
encoder stays frozen and is never loaded into the reconstruction optimizer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import uuid
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import broadcast_object_list, set_seed
from torch.utils.data import DataLoader

from .codec.classic.dataset import (
    BENCHMARK_CURATED_SKILL_PROVENANCE,
    SUCCESS_SKILL_PROVENANCE,
    assert_codec_split_isolation,
    split_examples_by_semantic_group,
    validate_codec_skill_example,
)
from .codec.classic.early_stopping import update_early_stopping
from .codec.classic.model import (
    SkillAutoencoder,
    backbone_dtype_name,
    configure_decoder_tokenizer,
    ensure_pad_token,
    get_context_window,
    get_hidden_size,
    merge_lora,
    model_source_fingerprint,
    resolve_backbone_dtype,
    tokenize_skill_batch,
)
from .codec.classic.training_contract import (
    CODEC_SCHEMA_VERSION,
    CODEC_SKILL_VIEW_VERSION,
    CODEC_ENCODER_ROUTING_VERSION,
    CODEC_NUMERIC_POLICY_VERSION,
    CODEC_TRAINING_EXAMPLE_POLICY,
    CODEC_SKILL_INPUT_VERSION,
    CODEC_ENCODER_POOLING,
    CODEC_QUERY_ENCODER_POOLING,
    codec_training_contract_sha256,
    codec_training_objective,
    decoder_prompt_uses_task_context,
    validate_codec_artifact_policy,
)
from .codec.classic.training_helpers import (
    ReconstructionBatchSampler,
    _reconstruction_lr_lambda,
    _configure_encoder_training,
    _configure_decoder_training,
    _capture_trainable_state,
    _restore_trainable_state,
    _validation_loss,
    _assert_finite_loss_and_metrics,
)
from .io import read_jsonl, sha256
from .models.import_safety import hide_optional_audio_packages
from .skills.executable import canonicalize_executable_skill


def load_examples(path: str | Path, *, deduplicate: bool = False) -> list[dict[str, Any]]:
    """Project training JSONL without changing its skill text or condition.

    ALFWorld uses unique (canonical skill, task type, goal) triples for Codec;
    Flow still sees every source row. Other domains retain every binding.
    """
    examples: list[dict[str, Any]] = []
    by_condition: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line_number, source in enumerate(read_jsonl(path), 1):
        if source.get("source_split") != "train":
            raise ValueError(f"Expected training skills at row {line_number}")
        task_id = str(source["task_id"])
        skill = canonicalize_executable_skill(source["executable_skill"])
        task_type = str(source["task_type"]).strip()
        goal = str(source["canonical_goal"]).strip()
        identity = (skill, task_type, goal)
        if deduplicate and identity in by_condition:
            by_condition[identity]["source_count"] += 1
            continue
        digest = hashlib.sha256(skill.encode("utf-8")).hexdigest()
        row = {
            "skill_text": skill,
            "skill_sha256": digest,
            "skill_provenance": (SUCCESS_SKILL_PROVENANCE if source["dataset"] == "alfworld" else BENCHMARK_CURATED_SKILL_PROVENANCE),
            "task_type": task_type,
            "canonical_goal": goal,
            "group_id": task_id,
            "chain_id": str(source["skill_id"]),
            "node_id": str(source["skill_id"]) + ":target",
            "chain_step": 1,
            "source_count": 1,
        }
        validate_codec_skill_example(row)
        examples.append(row)
        by_condition[identity] = row
    if not examples:
        raise ValueError("Training skill file is empty")
    return examples


def prepare_data(config: dict[str, Any], world_size: int):
    """Build the exact semantic split and deterministic optimizer budget."""
    codec = dict(config["codec"])
    accumulation = int(codec["gradient_accumulation"])
    global_batch = int(codec["global_batch_size"])
    if world_size < 1 or accumulation < 1 or global_batch % (world_size * accumulation):
        raise ValueError("Codec global_batch_size must be divisible by world_size * gradient_accumulation")
    codec["batch_size"] = global_batch // (world_size * accumulation)
    examples = load_examples(config["data"]["skills"], deduplicate=bool(codec.get("deduplicate_skill_condition", False)))
    source_count = sum(row["source_count"] for row in examples)
    if source_count != int(config["data"]["expected_count"]):
        raise ValueError("Training skill count disagrees with configuration")
    allowed = {row["skill_provenance"] for row in examples}
    train, validation = split_examples_by_semantic_group(
        examples, validation_ratio=float(codec["validation_ratio"]),
        seed=int(codec["split_seed"]), allowed_provenances=allowed,
    )
    assert_codec_split_isolation(train, validation, allowed_provenances=allowed)
    selection = codec["checkpoint_selection_mode"]
    if selection == "benchmark_final_epoch_all_rows_v1":
        if validation:
            raise ValueError("Final-epoch selection requires all rows in training")
    elif selection != "internal_validation_best_v1" or not validation:
        raise ValueError("Internal-validation selection requires a semantic holdout")
    sampler = ReconstructionBatchSampler(
        train, codec["batch_size"], seed=int(codec["training_seed"]), shuffle=True,
        pad_to_multiple=world_size * accumulation, tail_policy=codec["train_batch_tail_policy"],
    )
    steps_per_epoch = len(sampler) // (world_size * accumulation)
    summary = {
        "source_rows": source_count, "codec_rows": len(examples),
        "train_rows": len(train), "validation_rows": len(validation),
        "world_size": world_size, "batch_size_per_rank": codec["batch_size"],
        "gradient_accumulation": accumulation, "global_batch_size": global_batch,
        "steps_per_epoch": steps_per_epoch,
        "maximum_optimizer_steps": steps_per_epoch * int(codec["epochs"]),
        "semantic_overlap": 0,
    }
    return codec, train, validation, sampler, summary


def artifact_metadata(codec: dict[str, Any], *, training_dtype: str,
                      query_source: str, query_fingerprint: str,
                      data_sha256: str, selected_epoch: int,
                      epochs_completed: int, best_loss: float,
                      summary: dict[str, Any]) -> dict[str, Any]:
    """Create the complete policy consumed by SkillAutoencoderCodec."""
    conditioned = decoder_prompt_uses_task_context(codec["decoder_prompt"])
    metadata = {
        "codec_id": uuid.uuid4().hex,
        "schema_version": CODEC_SCHEMA_VERSION,
        "skill_view_version": CODEC_SKILL_VIEW_VERSION,
        "chains_sha256": data_sha256,
        "training_contract_sha256": codec_training_contract_sha256(codec),
        "encoder_routing_version": CODEC_ENCODER_ROUTING_VERSION,
        "numeric_policy_version": CODEC_NUMERIC_POLICY_VERSION,
        "training_example_policy": CODEC_TRAINING_EXAMPLE_POLICY,
        "skill_input_version": CODEC_SKILL_INPUT_VERSION,
        "skill_only_encoder": True,
        "frozen_original_query_encoder": True,
        "query_encoder_frozen": True,
        "query_encoder_source": query_source,
        "query_encoder_source_fingerprint": query_fingerprint,
        "query_encoder_loaded_during_training": False,
        "query_encoder_optimized_during_training": False,
        "role_free_skill_json": True,
        "training_frozen_contract": True,
        "training_backbone_dtype": training_dtype,
        "requested_backbone_dtype": codec["backbone_dtype"],
        "accelerator_mixed_precision": "bf16" if training_dtype == "bfloat16" else "no",
        "fp32_sensitive_paths": ["encoder_lora", "decoder_lora", "latent_normalization", "latent_to_prefix_projector", "decoder_cross_entropy_logits", "loss_reductions", "gradient_norm_and_finiteness_checks"],
        "encoder_pooling": CODEC_ENCODER_POOLING,
        "query_encoder_pooling": CODEC_QUERY_ENCODER_POOLING,
        "decoder_task_conditioned": conditioned,
        "training_objective": codec_training_objective(decoder_task_conditioned=conditioned),
        "loss_mode": "reconstruction",
        "teacher_forcing": True,
        "selected_checkpoint_epoch": selected_epoch,
        "best_epoch": selected_epoch,
        "epochs_completed": epochs_completed,
        "best_validation_loss": best_loss if math.isfinite(best_loss) else None,
        "final_weights_source": "internal_validation_best" if codec["validation_ratio"] else "final_epoch",
        "checkpoint_selection_mode": codec["checkpoint_selection_mode"],
        "training_plan": summary,
    }
    for key in ("encoder_skill_prompt", "encoder_query_prompt", "decoder_prompt",
                "latent_noise_angle_std", "latent_noise_max_angle", "latent_noise_prob", "latent_noise_ramp_epochs",
                "encoder_lora_last_n_layers", "train_encoder", "encoder_lora_rank", "encoder_lora_alpha", "encoder_lora_dropout",
                "decoder_lora_rank", "decoder_lora_alpha", "decoder_lora_dropout", "attention_implementation"):
        metadata[key] = codec[key]
    validate_codec_artifact_policy(metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="New Codec artifact directory; required for training")
    parser.add_argument("--validate-only", action="store_true", help="Check data and budget without loading a model")
    parser.add_argument("--world-size", type=int, help="World size for --validate-only")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.validate_only:
        world = args.world_size or int(config["codec"]["recommended_world_size"])
        _, _, _, _, summary = prepare_data(config, world)
        print(json.dumps(summary, indent=2))
        return
    if not args.output:
        parser.error("--output is required for training")
    if args.world_size is not None:
        parser.error("Use torchrun to set the training world size; --world-size is for validation only")
    if not torch.cuda.is_available():
        raise RuntimeError("Codec training requires CUDA; --validate-only is CPU-only")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dtype = resolve_backbone_dtype(config["codec"]["backbone_dtype"], device=f"cuda:{local_rank}")
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["codec"]["gradient_accumulation"]),
        mixed_precision="bf16" if dtype == torch.bfloat16 else "no",
        dataloader_config=DataLoaderConfiguration(even_batches=False, split_batches=False, non_blocking=True),
        step_scheduler_with_optimizer=False,
    )
    codec, train_rows, validation_rows, sampler, summary = prepare_data(config, accelerator.num_processes)
    set_seed(int(codec["training_seed"]))
    torch.set_float32_matmul_precision(codec["float32_matmul_precision"])
    output = Path(args.output)
    accelerator.wait_for_everyone()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace an existing Codec run: {output}")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "training_config.yaml").write_text(yaml.safe_dump({**config, "codec": codec}, sort_keys=False))
        (output / "training_plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    query_source = str(Path(codec["query_encoder_model"]).expanduser().resolve())
    fingerprint = [model_source_fingerprint(query_source) if accelerator.is_main_process else None]
    broadcast_object_list(fingerprint, from_process=0)
    with hide_optional_audio_packages():
        from transformers import AutoTokenizer
    encoder_tokenizer = AutoTokenizer.from_pretrained(codec["encoder_model"], trust_remote_code=True)
    decoder_tokenizer = AutoTokenizer.from_pretrained(codec["decoder_model"], trust_remote_code=True)
    ensure_pad_token(encoder_tokenizer)
    encoder_tokenizer.padding_side = "left"
    configure_decoder_tokenizer(decoder_tokenizer)
    model = SkillAutoencoder(
        codec["encoder_model"], codec["decoder_model"], int(codec["num_prefix_tokens"]),
        gradient_checkpointing=True, attention_implementation=codec["attention_implementation"],
        backbone_dtype=backbone_dtype_name(dtype),
        latent_noise_angle_std=float(codec["latent_noise_angle_std"]),
        latent_noise_max_angle=float(codec["latent_noise_max_angle"]),
        latent_noise_prob=float(codec["latent_noise_prob"]),
    )
    decoder_window = get_context_window(model.decoder.config, decoder_tokenizer)
    configure_decoder_tokenizer(decoder_tokenizer, context_window=decoder_window)
    if int(codec["max_length"]) > get_context_window(model.encoder_prime.config, encoder_tokenizer):
        raise ValueError("Encoder context cannot accommodate the configured training budget")
    if int(codec["num_prefix_tokens"]) + int(codec["max_length"]) > decoder_window:
        raise ValueError("Decoder context cannot accommodate prefix plus training tokens")
    encoder_layers = _configure_encoder_training(model, enabled=bool(codec["train_encoder"]),
        lora_rank=int(codec["encoder_lora_rank"]), lora_alpha=float(codec["encoder_lora_alpha"]),
        lora_dropout=float(codec["encoder_lora_dropout"]), last_n_layers=int(codec["encoder_lora_last_n_layers"]))
    decoder_layers = _configure_decoder_training(model, lora_rank=int(codec["decoder_lora_rank"]),
        lora_alpha=float(codec["decoder_lora_alpha"]), lora_dropout=float(codec["decoder_lora_dropout"]))
    model.vector_to_prefix.requires_grad_(True)
    groups = [{"name": name, "params": [p for p in component.parameters() if p.requires_grad], "lr": float(codec[lr])}
              for name, component, lr in (("encoder", model.encoder_prime, "encoder_lr"),
              ("decoder", model.decoder, "decoder_lr"), ("projector", model.vector_to_prefix, "projector_lr"))]
    if any(not group["params"] for group in groups):
        raise RuntimeError("Every Codec component must have trainable parameters")
    optimizer = torch.optim.AdamW(groups, weight_decay=float(codec["weight_decay"]), fused=bool(codec["fused_optimizer"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_reconstruction_lr_lambda(
        total_steps=summary["maximum_optimizer_steps"], warmup_ratio=float(codec["warmup_ratio"])))
    collate = partial(tokenize_skill_batch, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer,
        max_length=int(codec["max_length"]), num_prefix_tokens=int(codec["num_prefix_tokens"]),
        encoder_prompt=codec["encoder_skill_prompt"], decoder_prompt=codec["decoder_prompt"])
    workers = int(codec["num_workers"])
    loader_options = {"num_workers": workers, "pin_memory": True, "persistent_workers": workers > 0}
    loader = DataLoader(train_rows, batch_sampler=sampler, collate_fn=collate, **loader_options)
    validation_loader = None
    if validation_rows:
        validation_loader = DataLoader(validation_rows, batch_sampler=ReconstructionBatchSampler(
            validation_rows, codec["batch_size"], seed=int(codec["split_seed"]), shuffle=False),
            collate_fn=collate, **loader_options)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    if len(loader) != summary["steps_per_epoch"] * int(codec["gradient_accumulation"]):
        raise RuntimeError("Accelerate changed the fixed optimizer budget")
    best_loss, best_epoch, stale_epochs, best_state = float("inf"), 0, 0, None
    records, validation_records = [], []
    global_step, epochs_completed = 0, 0
    model.train()
    for epoch in range(int(codec["epochs"])):
        ramp = int(codec["latent_noise_ramp_epochs"])
        noise_enabled = all(float(codec[k]) > 0 for k in ("latent_noise_angle_std", "latent_noise_max_angle", "latent_noise_prob"))
        progress = (min(1.0, (epoch + 1) / ramp) if ramp else 1.0) if noise_enabled else 0.0
        accelerator.unwrap_model(model).set_noise_progress(progress)
        sampler.set_epoch(epoch)
        window_loss = torch.zeros((), device=accelerator.device)
        microbatches = 0
        for batch in loader:
            with accelerator.accumulate(model):
                loss = model(**batch)
                metrics = accelerator.unwrap_model(model).last_loss_metrics
                _assert_finite_loss_and_metrics(loss, metrics, stage="reconstruction", epoch=epoch + 1)
                window_loss += loss.detach().float()
                microbatches += 1
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(codec["gradient_clip_norm"]))
                    if not bool(torch.isfinite(grad_norm).all()):
                        raise FloatingPointError("Non-finite Codec gradient norm")
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                mean_loss = float(accelerator.reduce(window_loss / microbatches, reduction="mean").item())
                if scheduler.last_epoch != global_step:
                    raise RuntimeError("Learning-rate scheduler and optimizer updates diverged")
                if accelerator.is_main_process:
                    records.append((global_step, mean_loss, epoch + 1))
                    if global_step == 1 or global_step % int(codec["log_every"]) == 0:
                        print(f"step={global_step}/{summary['maximum_optimizer_steps']} epoch={epoch + 1} loss={mean_loss:.6f}", flush=True)
                window_loss.zero_()
                microbatches = 0
        epochs_completed = epoch + 1
        should_stop = False
        if validation_loader is not None:
            with torch.no_grad():
                result = _validation_loss(model, validation_loader, accelerator, noise_seed=int(codec["training_seed"]))
            val_loss = result["total_loss"]
            if not math.isfinite(val_loss):
                raise FloatingPointError("Non-finite Codec validation loss")
            update = update_early_stopping(epoch=epoch + 1, validation_loss=val_loss, best_validation_loss=best_loss,
                best_epoch=best_epoch, stale_epochs=stale_epochs, track_best=not noise_enabled or progress == 1.0,
                min_epochs=int(codec["min_epochs"]), patience=int(codec["early_stopping_patience"]))
            best_loss, best_epoch, stale_epochs = update.best_validation_loss, update.best_epoch, update.stale_epochs
            should_stop = update.should_stop
            if accelerator.is_main_process:
                validation_records.append((epoch + 1, val_loss))
                if update.improved:
                    best_state = _capture_trainable_state(accelerator.unwrap_model(model))
                print(f"validation epoch={epoch + 1} loss={val_loss:.6f} best_epoch={best_epoch}", flush=True)
            model.train()
        if should_stop:
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(model)
        if validation_rows:
            if best_state is None:
                raise RuntimeError("No valid checkpoint after the noise ramp")
            _restore_trainable_state(raw, best_state)
        else:
            if epochs_completed != int(codec["epochs"]):
                raise RuntimeError("Final-epoch Codec stopped before its complete training budget")
            best_epoch = epochs_completed
        metadata = artifact_metadata(codec, training_dtype=backbone_dtype_name(dtype), query_source=query_source,
            query_fingerprint=fingerprint[0], data_sha256=sha256(config["data"]["skills"]),
            selected_epoch=best_epoch, epochs_completed=epochs_completed, best_loss=best_loss, summary=summary)
        metadata.update({"encoder_lora_all_transformer_layers": False, "encoder_lora_transformer_layers": encoder_layers,
            "encoder_lora_linear_targets": raw.encoder_lora_module_count, "decoder_lora_all_transformer_layers": True,
            "decoder_lora_transformer_layers": decoder_layers, "decoder_lora_linear_targets": raw.decoder_lora_module_count})
        metadata["encoder_lora_merged_modules"] = merge_lora(raw.encoder_prime)
        metadata["decoder_lora_merged_modules"] = merge_lora(raw.decoder)
        raw.eval().requires_grad_(False)
        raw.encoder_prime.save_pretrained(output / "encoder_prime")
        raw.decoder.save_pretrained(output / "decoder")
        encoder_tokenizer.save_pretrained(output / "encoder_tokenizer")
        decoder_tokenizer.save_pretrained(output / "decoder_tokenizer")
        torch.save({**metadata, "state_dict": raw.vector_to_prefix.state_dict(), "num_prefix_tokens": raw.num_prefix_tokens,
            "encoder_dim": get_hidden_size(raw.encoder_prime.config), "decoder_dim": raw.decoder_dim}, output / "vector_to_prefix.pt")
        (output / "codec_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for filename, header, rows in (("loss.csv", ["step", "loss", "epoch"], records),
                                     ("validation_loss.csv", ["epoch", "loss"], validation_records)):
            with (output / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
        print(f"Saved Codec to {output}", flush=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
