from __future__ import annotations

import contextlib
import json
import os
from typing import Iterable

import torch

from skillflow.summarization._alfworld_validation import (
    ChainContractError,
    GeneratedSummaryExactDuplicateError,
    GeneratedSummaryFormatError,
    validate_generated_summary_for_rollout,
)
from skillflow.summarization._distributed import local_rank, rank
from skillflow.summarization._attention import attention_kwargs
from skillflow.summarization._import_safety import hide_optional_audio_packages


# Prompt text and generation settings are hashed into durable per-item
# checkpoints. Bump the implementation label whenever the prompt contract changes.
SUMMARIZER_IMPLEMENTATION_VERSION = "vllm_iterative_skill_json_v2"


COMMON_OUTPUT_CONTRACT = """Return exactly one JSON object:
{
  "when_to_use": "Describe the goal pattern, observable prerequisites, and decision point that should trigger this skill.",
  "high_level_guidance": "Describe the ordered subgoals, prerequisites, state transitions, and completion checks.",
  "low_level_guidance": "Give concrete action-selection and verification rules grounded in trajectory evidence while generalizing away scene-specific entity identifiers."
}
"""


SUCCESS_PROMPT = """You distill one successful ALFWorld interaction trajectory into one reusable final skill.

Task type: {TASK_TYPE}
Goal: {TASK_GOAL}

Successful trajectory:
{TRAJECTORY}

Think privately about which observed actions and state transitions made the
trajectory succeed. Preserve the necessary ordering, prerequisites, progress
checks, and completion evidence. Generalize away incidental scene identifiers
without inventing actions or failure lessons that are not supported by this
successful trajectory.

{OUTPUT_CONTRACT}
"""

FAILURE_PROMPT = """You revise the next executable ALFWORLD skill after one failed interaction trajectory.

Task type: {TASK_TYPE}
Goal: {TASK_GOAL}

Skill used in the failed attempt:
{CURRENT_SKILL}

Failed trajectory:
{TRAJECTORY}

Failure status:
{FAILURE_REASON}

Think privately about the first unsupported assumption, missing prerequisite,
bad ordering decision, loop, or absent verification rule. Produce a replacement
skill for the next rollout. It must materially improve on the executed skill
and generalize away incidental scene identifiers.

{OUTPUT_CONTRACT}
"""

@contextlib.contextmanager
def _silence(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _trajectory_text(
    steps: Iterable[dict],
    *,
    final_observation: object,
    success: bool,
    max_steps: int,
) -> str:
    """Render only the observation/action sequence and terminal outcome."""

    rows = list(steps or [])
    if max_steps > 0 and len(rows) > max_steps:
        raise RuntimeError(
            "Summarizer trajectory exceeds summarizer_max_trajectory_steps; "
            f"steps={len(rows)} limit={max_steps}. Silent head/tail truncation is forbidden."
        )

    lines: list[str] = []
    for row in rows:
        lines.append(f"Observation: {str(row.get('observation') or '').strip()}")
        lines.append(f"Action: {str(row.get('action') or '').strip()}")

    terminal_observation = str(final_observation or "").strip()
    if not terminal_observation and rows:
        terminal_observation = str(rows[-1].get("next_observation") or "").strip()
    lines.append(f"Observation: {terminal_observation}")
    lines.append(f"Outcome: {'success' if success else 'failure'}")
    return "\n".join(lines)


def _context_window_tokens(model, tokenizer) -> int:
    config = getattr(model, "config", None)
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if value is not None:
            try:
                value = int(value)
                if 0 < value < 1_000_000:
                    return value
            except Exception:
                pass
    value = getattr(tokenizer, "model_max_length", None)
    try:
        value = int(value)
        if 0 < value < 1_000_000:
            return value
    except Exception:
        pass
    return 0


def _check_no_prompt_truncation(
    attention_mask: torch.Tensor,
    *,
    configured_input_limit: int,
    generation_tokens: int,
    context_window: int,
    label: str,
) -> None:
    lengths = attention_mask.sum(dim=1).detach().cpu().tolist()
    max_len = int(max(lengths, default=0))
    limits = [int(configured_input_limit)]
    if context_window > 0:
        limits.append(max(1, int(context_window) - int(generation_tokens)))
    limit = min(x for x in limits if x > 0)
    if max_len <= limit:
        return
    raise RuntimeError(
        f"{label} prompt would exceed the non-truncating token budget: "
        f"max_prompt_tokens={max_len}, configured_input_limit={configured_input_limit}, "
        f"context_window={context_window or 'unknown'}, reserved_generation_tokens={generation_tokens}. "
        "The pipeline now refuses silent tokenizer truncation; increase the model/context budget "
        "or reduce the visible trajectory window."
    )


def _eos_token_ids(model, tokenizer) -> set[int]:
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
    return output


def _chat_template_supports_variable(tokenizer, variable: str) -> bool:
    """Return whether the configured Jinja chat template references a variable."""

    template = getattr(tokenizer, "chat_template", None)
    values = template.values() if isinstance(template, dict) else [template]
    return any(str(variable) in str(value or "") for value in values)


class _RecoverableSummaryOutputError(Exception):
    """One generated sample is unusable, but the summarizer process is healthy."""

    def __init__(self, message: str, *, status: str = "recoverable_output_error"):
        super().__init__(message)
        self.status = str(status)


def _native_thinking_end_token_id(tokenizer) -> int | None:
    """Resolve Qwen's native </think> delimiter without hard-coding a model id."""

    try:
        encoded = tokenizer.encode("</think>", add_special_tokens=False)
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
            token_id = int(encoded[0])
            if token_id != getattr(tokenizer, "unk_token_id", None):
                return token_id
    except Exception:
        pass

    try:
        token_id = tokenizer.convert_tokens_to_ids("</think>")
        if token_id is not None and int(token_id) != getattr(
            tokenizer, "unk_token_id", None
        ):
            return int(token_id)
    except Exception:
        pass
    return None


def _final_answer_token_ids(
    generated_token_ids: list[int],
    *,
    thinking_enabled: bool,
    thinking_end_token_id: int | None,
) -> list[int]:
    """Drop native Qwen reasoning tokens and retain only the final answer tokens."""

    tokens = [int(token) for token in generated_token_ids]
    if not thinking_enabled:
        return tokens
    if thinking_end_token_id is None:
        raise _RecoverableSummaryOutputError(
            "Native summarizer thinking is enabled, but the tokenizer has no "
            "single </think> delimiter token."
        )
    close_positions = [
        index for index, token in enumerate(tokens) if token == thinking_end_token_id
    ]
    if not close_positions:
        raise _RecoverableSummaryOutputError(
            "Summarizer ended without closing its native <think> block; refusing "
            "to mix hidden diagnosis text into the stored skill JSON."
        )
    return tokens[close_positions[-1] + 1 :]


def _generation_cutoff_failure_reasons(
    generated_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    rollouts: list[dict],
) -> list[str | None]:
    row_count = int(generated_ids.shape[0])
    if row_count != len(rollouts):
        raise RuntimeError(
            "Summarizer generation output/rollout count mismatch: "
            f"generated_rows={row_count} rollouts={len(rollouts)}"
        )
    if int(generated_ids.shape[1]) < int(max_new_tokens):
        return [None] * row_count

    failures: list[str | None] = []
    for token_row, rollout in zip(generated_ids, rollouts):
        tokens = [int(token) for token in token_row.detach().cpu().tolist()]
        if eos_token_ids and any(token in eos_token_ids for token in tokens):
            failures.append(None)
            continue
        failures.append(
            "Summarizer generation reached max_new_tokens without EOS; only an "
            "already complete final structured-skill JSON object may be recovered. "
            f"task_id={rollout.get('task_id')!r} "
            f"rollout_id={rollout.get('rollout_id')!r} "
            f"generated_tokens={len(tokens)} max_new_tokens={max_new_tokens}"
        )
    return failures


_SUMMARY_OUTPUT_KEYS = {
    "when_to_use",
    "high_level_guidance",
    "low_level_guidance",
}


def _extract_last_summary_json(text: str) -> dict:
    """Extract the last complete structured-skill object from final-answer text."""

    raw_text = str(text or "")
    decoder = json.JSONDecoder()
    objects: list[dict] = []
    skill_candidates: list[dict] = []
    for start, character in enumerate(raw_text):
        if character != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(raw_text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        objects.append(parsed)
        if {str(key) for key in parsed} & _SUMMARY_OUTPUT_KEYS:
            skill_candidates.append(parsed)
    if skill_candidates:
        return skill_candidates[-1]
    if objects:
        return objects[-1]
    raise ValueError("no complete JSON object was found in the final answer")


def _parse_summary_text(text: str, rollout: dict) -> dict:
    """Extract and validate one final structured skill."""

    if not isinstance(rollout.get("success"), bool):
        raise ChainContractError("Summarizer rollout.success must be boolean")
    raw_text = str(text or "").strip()
    try:
        parsed = _extract_last_summary_json(raw_text)
    except Exception as exc:
        preview = raw_text[:800].replace("\n", "\\n")
        raise GeneratedSummaryFormatError(
            "Summarizer final answer must contain one structured-skill JSON object; "
            f"task_id={rollout.get('task_id')!r} error={type(exc).__name__}: {exc}; "
            f"output_preview={preview!r}"
        ) from exc

    try:
        return validate_generated_summary_for_rollout(parsed, rollout)
    except GeneratedSummaryExactDuplicateError as exc:
        raise GeneratedSummaryExactDuplicateError(
            f"Invalid summarizer contract for task_id={rollout.get('task_id')!r}: {exc}"
        ) from exc
    except GeneratedSummaryFormatError as exc:
        raise GeneratedSummaryFormatError(
            f"Invalid summarizer contract for task_id={rollout.get('task_id')!r}: {exc}"
        ) from exc


def _capped_complete_json_recovery(
    text: str,
    rollout: dict,
    *,
    cutoff_reason: str,
) -> tuple[bool, str]:
    """Accept a no-EOS cap only when the final answer is already strictly valid."""

    try:
        _parse_summary_text(text, rollout)
    except (
        GeneratedSummaryExactDuplicateError,
        GeneratedSummaryFormatError,
    ) as exc:
        return (
            False,
            f"{cutoff_reason} generation_status=cap_incomplete_or_invalid_json "
            f"strict_final_json_error={str(exc).replace(chr(10), ' ')[:800]}",
        )
    return True, "cap_recovered_complete_json"



class SkillSummarizer:
    """High-throughput summarizer with a vLLM production backend.

    The launcher shards collection trajectories across isolated GPU processes.
    Each shard owns one vLLM replica, allowing vLLM to batch variable-length
    prompts continuously without external distributed-process state leaking into
    the engine. The Transformers backend is available only as an explicitly
    configured development backend.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tokenizer = None
        self.llm = None
        self.generation_outcomes: dict[str, str] = {}
        name = cfg["models"].get("summarizer") or cfg["models"]["backbone"]
        trust = bool(cfg["models"].get("trust_remote_code", True))
        quiet = bool(cfg.get("logging", {}).get("quiet", True))
        self.backend = str(
            cfg["models"].get("summarizer_backend", "transformers")
        ).strip().lower()
        if self.backend not in {"vllm", "transformers"}:
            raise ValueError(
                "models.summarizer_backend must be exactly 'vllm' or 'transformers'"
            )
        self.enable_thinking = bool(
            cfg["models"].get("summarizer_enable_thinking", False)
        )
        self.thinking_end_token_id: int | None = None
        self.batch_size = max(1, int(cfg["models"].get("summarizer_batch_size", 2)))
        self.max_input_tokens = int(cfg["models"].get("summarizer_max_input_tokens", 4096))
        configured_max_new_tokens = int(cfg["models"].get("summarizer_max_new_tokens", 1024))
        self.max_new_tokens = max(256, configured_max_new_tokens)
        self.final_answer_reserve_tokens = int(
            cfg["models"].get("summarizer_final_answer_reserve_tokens", 512)
        )
        self.thinking_budget_tokens = int(
            cfg["models"].get(
                "summarizer_thinking_budget_tokens",
                self.max_new_tokens - self.final_answer_reserve_tokens,
            )
        )
        if self.final_answer_reserve_tokens < 256:
            raise ValueError(
                "models.summarizer_final_answer_reserve_tokens must be at least 256"
            )
        if self.enable_thinking:
            if self.thinking_budget_tokens < 256:
                raise ValueError(
                    "models.summarizer_thinking_budget_tokens must be at least 256"
                )
            if (
                self.thinking_budget_tokens + self.final_answer_reserve_tokens
                > self.max_new_tokens
            ):
                raise ValueError(
                    "summarizer thinking budget plus final-answer reserve exceeds "
                    "summarizer_max_new_tokens"
                )
        self.contract_max_attempts = int(
            cfg["models"].get("summarizer_contract_max_attempts", 3)
        )
        if not 1 <= self.contract_max_attempts <= 10:
            raise ValueError(
                "models.summarizer_contract_max_attempts must be between 1 and 10; "
                "the value counts the initial generation as one attempt"
            )
        self.max_steps = int(cfg["models"].get("summarizer_max_trajectory_steps", 24))
        rollout_max_steps = int(cfg.get("sampling", {}).get("max_steps", 0))
        if self.max_steps < rollout_max_steps:
            raise ValueError(
                "models.summarizer_max_trajectory_steps must be at least "
                f"sampling.max_steps ({rollout_max_steps}); silent trajectory "
                "truncation is forbidden."
            )
        self.fail_on_prompt_truncation = bool(
            cfg["models"].get("fail_on_prompt_truncation", True)
        )

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")

        with hide_optional_audio_packages():
            from transformers import AutoTokenizer
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            try:
                hf_logging.disable_progress_bar()
            except Exception:
                pass
            # vLLM owns background processes and loggers whose handlers must
            # outlive construction. Redirect only the synchronous Transformers
            # fallback; binding vLLM loggers to a soon-closed devnull stream can
            # cause shutdown-time logging failures.
            with _silence(quiet and self.backend != "vllm"):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    name,
                    trust_remote_code=trust,
                    use_fast=bool(
                        cfg["models"].get(
                            "summarizer_tokenizer_use_fast", True
                        )
                    ),
                )
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.enable_thinking:
                    self.thinking_end_token_id = _native_thinking_end_token_id(
                        self.tokenizer
                    )
                    if self.thinking_end_token_id is None:
                        raise RuntimeError(
                            "models.summarizer_enable_thinking=true requires a Qwen "
                            "tokenizer with a native </think> delimiter token"
                        )
                self.chat_template_thinking_budget_kw = next(
                    (
                        variable
                        for variable in ("thinking_budget", "reasoning_budget")
                        if _chat_template_supports_variable(
                            self.tokenizer, variable
                        )
                    ),
                    None,
                )
                # Batched causal generation must be left padded. We do not rely on
                # tokenizer truncation; oversized prompts are checked explicitly.
                self.tokenizer.padding_side = "left"
                self.tokenizer.truncation_side = "left"
                if self.backend == "vllm":
                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            "models.summarizer_backend='vllm' requires a CUDA GPU"
                        )
                    torch.cuda.set_device(local_rank())
                    try:
                        from vllm import LLM, SamplingParams
                    except ModuleNotFoundError as exc:
                        raise RuntimeError(
                            "vLLM is required for summarization. Install the "
                            "requirements in the GPU environment before launching."
                        ) from exc
                    self._vllm_sampling_params_cls = SamplingParams
                    max_model_len = int(
                        cfg["models"].get(
                            "summarizer_vllm_max_model_len",
                            self.max_input_tokens + self.max_new_tokens,
                        )
                    )
                    if max_model_len < self.max_input_tokens + self.max_new_tokens:
                        raise ValueError(
                            "models.summarizer_vllm_max_model_len must cover "
                            "summarizer_max_input_tokens + summarizer_max_new_tokens"
                        )
                    tensor_parallel_size = int(
                        cfg["models"].get("summarizer_vllm_tensor_parallel_size", 1)
                    )
                    if tensor_parallel_size <= 0:
                        raise ValueError(
                            "models.summarizer_vllm_tensor_parallel_size must be "
                            "a positive integer"
                        )
                    optional_vllm_kwargs = {}
                    if "summarizer_vllm_language_model_only" in cfg["models"]:
                        optional_vllm_kwargs["language_model_only"] = bool(
                            cfg["models"].get("summarizer_vllm_language_model_only")
                        )
                    if (
                        "summarizer_vllm_disable_custom_all_reduce"
                        in cfg["models"]
                    ):
                        optional_vllm_kwargs["disable_custom_all_reduce"] = bool(
                            cfg["models"].get(
                                "summarizer_vllm_disable_custom_all_reduce"
                            )
                        )
                    self.llm = LLM(
                        model=name,
                        tokenizer=name,
                        trust_remote_code=trust,
                        dtype=str(
                            cfg["models"].get(
                                "summarizer_vllm_dtype", "bfloat16"
                            )
                        ),
                        tensor_parallel_size=tensor_parallel_size,
                        max_model_len=max_model_len,
                        gpu_memory_utilization=float(
                            cfg["models"].get(
                                "summarizer_vllm_gpu_memory_utilization", 0.92
                            )
                        ),
                        max_num_seqs=int(
                            cfg["models"].get(
                                "summarizer_vllm_max_num_seqs", self.batch_size
                            )
                        ),
                        max_num_batched_tokens=int(
                            cfg["models"].get(
                                "summarizer_vllm_max_num_batched_tokens", 32768
                            )
                        ),
                        enable_chunked_prefill=bool(
                            cfg["models"].get(
                                "summarizer_vllm_enable_chunked_prefill", True
                            )
                        ),
                        enable_prefix_caching=bool(
                            cfg["models"].get(
                                "summarizer_vllm_enable_prefix_caching", True
                            )
                        ),
                        enforce_eager=bool(
                            cfg["models"].get(
                                "summarizer_vllm_enforce_eager", False
                            )
                        ),
                        attention_config={
                            "backend": str(
                                cfg["models"].get(
                                    "summarizer_vllm_attention_backend",
                                    "FLASH_ATTN",
                                )
                            ),
                            "flash_attn_version": int(
                                cfg["models"].get(
                                    "summarizer_vllm_flash_attn_version", 2
                                )
                            ),
                        },
                        disable_log_stats=quiet,
                        generation_config="vllm",
                        seed=int((cfg.get("run") or {}).get("seed", 42))
                        + rank(),
                        **optional_vllm_kwargs,
                    )
                    self.context_window_tokens = max_model_len
                else:
                    from transformers import AutoModelForCausalLM

                    kwargs = {
                        "trust_remote_code": trust,
                        "low_cpu_mem_usage": True,
                    }
                    if torch.cuda.is_available():
                        kwargs["device_map"] = {"": local_rank()}
                        kwargs["dtype"] = torch.bfloat16
                    else:
                        kwargs["dtype"] = torch.float32
                    kwargs.update(
                        attention_kwargs(
                            cfg, "summarizer_attention_implementation"
                        )
                    )
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        name, **kwargs
                    ).eval()
                    self.llm.requires_grad_(False)
                    self.context_window_tokens = _context_window_tokens(
                        self.llm, self.tokenizer
                    )
            if self.enable_thinking:
                print(
                    "[SUMMARIZER-THINKING-BUDGET] "
                    f"mode={'chat_template' if self.chat_template_thinking_budget_kw else 'max_new_tokens_cap'} "
                    f"budget_tokens={self.thinking_budget_tokens} "
                    f"final_answer_reserve_tokens={self.final_answer_reserve_tokens} "
                    f"max_new_tokens={self.max_new_tokens}",
                    flush=True,
                )

    def close(self) -> None:
        """Shut down the vLLM core before Python logging is finalized."""

        if getattr(self, "backend", "transformers") != "vllm":
            return
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        renderer = getattr(llm, "renderer", None)
        renderer_shutdown = getattr(renderer, "shutdown", None)
        if callable(renderer_shutdown):
            renderer_shutdown()
        engine = getattr(llm, "llm_engine", None)
        engine_core = getattr(engine, "engine_core", None)
        core_shutdown = getattr(engine_core, "shutdown", None)
        if not callable(core_shutdown):
            raise RuntimeError(
                "Installed vLLM no longer exposes llm_engine.engine_core.shutdown"
            )
        core_shutdown()
        self.llm = None

    def _raw_prompt(self, rollout: dict) -> str:
        success = rollout.get("success")
        if not isinstance(success, bool):
            raise ChainContractError("SkillSummarizer requires boolean rollout.success")
        task_type = str(rollout.get("task_type") or "").strip()
        canonical_goal = str(rollout.get("canonical_goal") or "").strip()
        if not task_type or not canonical_goal:
            raise RuntimeError(
                "Summarizer rollout is missing task_type or canonical_goal"
            )
        trajectory = _trajectory_text(
            rollout.get("steps", []),
            final_observation=rollout.get("final_observation"),
            success=success,
            max_steps=self.max_steps,
        )
        template = SUCCESS_PROMPT if success else FAILURE_PROMPT
        current_skill = (
            str(rollout.get("skill_used") or "").strip()
            or "No skill was injected in this attempt."
        )
        return (
            template.replace("{TASK_TYPE}", task_type)
            .replace("{TASK_GOAL}", canonical_goal)
            .replace("{CURRENT_SKILL}", current_skill)
            .replace("{TRAJECTORY}", trajectory)
            .replace(
                "{FAILURE_REASON}",
                str(rollout.get("failure_reason") or "ended_without_success"),
            )
            .replace("{OUTPUT_CONTRACT}", COMMON_OUTPUT_CONTRACT)
        )

    def _render_chat(self, messages: list[dict[str, str]]) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError(
                "The configured summarizer must provide a chat template; plain-prompt "
                "fallback is forbidden."
            )
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": self.enable_thinking,
        }
        budget_kw = getattr(self, "chat_template_thinking_budget_kw", None)
        if self.enable_thinking and budget_kw:
            max_new_tokens = max(256, int(getattr(self, "max_new_tokens", 1024)))
            final_answer_reserve_tokens = int(
                getattr(
                    self,
                    "final_answer_reserve_tokens",
                    min(512, max_new_tokens // 2),
                )
            )
            template_kwargs[budget_kw] = int(
                getattr(
                    self,
                    "thinking_budget_tokens",
                    max(256, max_new_tokens - final_answer_reserve_tokens),
                )
            )
        try:
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError as exc:
            raise RuntimeError(
                "The summarizer chat template must explicitly support "
                f"enable_thinking={self.enable_thinking!r}."
            ) from exc

    def _prompt(self, rollout: dict) -> str:
        return self._render_chat(
            [{"role": "user", "content": self._raw_prompt(rollout)}]
        )

    def _contract_retry_prompt(
        self,
        rollout: dict,
        *,
        rejected_output: str,
        validation_error: Exception,
        next_attempt: int,
    ) -> str:
        error_preview = " ".join(str(validation_error).split())[:400]
        if not error_preview:
            error_preview = type(validation_error).__name__
        correction = f"""The previous final answer could not be accepted.

Correction attempt: {next_attempt}/{self.contract_max_attempts}
Validation error: {error_preview}

Think privately and return a corrected JSON object with exactly the three
non-empty string fields when_to_use, high_level_guidance, and
low_level_guidance. Return the final JSON object only."""
        raw_prompt = self._raw_prompt(rollout)

        prompt_limit = max(
            1, int(getattr(self, "max_input_tokens", 4096))
        )
        context_window = int(getattr(self, "context_window_tokens", 0))
        max_new_tokens = int(getattr(self, "max_new_tokens", 0))
        if context_window > 0:
            prompt_limit = min(
                prompt_limit,
                max(1, context_window - max_new_tokens),
            )

        def token_ids(text: str) -> list[int]:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            ids = encoded["input_ids"]
            if ids and isinstance(ids[0], (list, tuple)):
                if len(ids) != 1:
                    raise RuntimeError(
                        "retry prompt tokenizer returned multiple rows"
                    )
                ids = ids[0]
            return [int(token_id) for token_id in ids]

        def token_count(text: str) -> int:
            return len(token_ids(text))

        def decode_tail(ids: list[int], keep: int) -> str:
            tail_ids = ids[-keep:] if keep else []
            if not tail_ids:
                return ""
            return self.tokenizer.decode(
                tail_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        def full_render(rejected_tail: str) -> str:
            return self._render_chat(
                [
                    {"role": "user", "content": raw_prompt},
                    {"role": "assistant", "content": rejected_tail},
                    {"role": "user", "content": correction},
                ]
            )

        rejected_output = str(rejected_output or "")
        rejected_ids = token_ids(rejected_output)
        full_prompt = full_render(rejected_output)
        if token_count(full_prompt) <= prompt_limit:
            return full_prompt

        base_prompt = full_render("")
        if token_count(base_prompt) <= prompt_limit:
            low = 0
            high = len(rejected_ids)
            best_prompt = base_prompt
            best_keep = 0
            while low <= high:
                keep = (low + high) // 2
                candidate = full_render(decode_tail(rejected_ids, keep))
                if token_count(candidate) <= prompt_limit:
                    best_prompt = candidate
                    best_keep = keep
                    low = keep + 1
                else:
                    high = keep - 1
            print(
                "[SUMMARIZER-RETRY-PROMPT-TRUNCATE] "
                f"task_id={rollout.get('task_id')!r} "
                f"attempt={next_attempt}/{self.contract_max_attempts} "
                "mode=rejected_output_tail "
                f"prompt_limit={prompt_limit} "
                f"rejected_tokens_dropped={len(rejected_ids) - best_keep} "
                "trajectory_tokens_dropped=0",
                flush=True,
            )
            return best_prompt

        success = rollout.get("success")
        if not isinstance(success, bool):
            raise ChainContractError(
                "SkillSummarizer requires boolean rollout.success"
            )
        task_type = str(rollout.get("task_type") or "").strip()
        canonical_goal = str(rollout.get("canonical_goal") or "").strip()
        trajectory = _trajectory_text(
            rollout.get("steps", []),
            final_observation=rollout.get("final_observation"),
            success=success,
            max_steps=self.max_steps,
        )
        trajectory_ids = token_ids(trajectory)
        failure_context = ""
        if not success:
            current_skill = (
                str(rollout.get("skill_used") or "").strip()
                or "No skill was injected in this attempt."
            )
            failure_context = (
                f"\nSkill used in the failed attempt: {current_skill}\n"
                "Failure status: "
                + str(
                    rollout.get("failure_reason")
                    or "ended_without_success"
                )
            )

        def compact_render(trajectory_tail: str) -> str:
            content = f"""Task type: {task_type}
Goal: {canonical_goal}{failure_context}

Complete trajectory (tail retained only if required by token budget):
{trajectory_tail}

{correction}

{COMMON_OUTPUT_CONTRACT}"""
            return self._render_chat([{"role": "user", "content": content}])

        compact_empty = compact_render("")
        if token_count(compact_empty) > prompt_limit:
            compact_empty = self._render_chat(
                [
                    {
                        "role": "user",
                        "content": (
                            f"Task type: {task_type}\nGoal: {canonical_goal}\n"
                            f"Correction attempt: {next_attempt}/"
                            f"{self.contract_max_attempts}. Previous JSON invalid. "
                            "Return JSON only with exactly these non-empty string "
                            "fields: when_to_use, high_level_guidance, "
                            "low_level_guidance."
                        ),
                    }
                ]
            )
            if token_count(compact_empty) > prompt_limit:
                raise RuntimeError(
                    "Configured summarizer_max_input_tokens is too small for "
                    "the mandatory retry task/goal/JSON contract"
                )
            print(
                "[SUMMARIZER-RETRY-PROMPT-TRUNCATE] "
                f"task_id={rollout.get('task_id')!r} "
                f"attempt={next_attempt}/{self.contract_max_attempts} "
                "mode=minimal_contract "
                f"prompt_limit={prompt_limit} "
                f"rejected_tokens_dropped={len(rejected_ids)} "
                f"trajectory_tokens_dropped={len(trajectory_ids)}",
                flush=True,
            )
            return compact_empty

        low = 0
        high = len(trajectory_ids)
        best_prompt = compact_empty
        best_keep = 0
        while low <= high:
            keep = (low + high) // 2
            candidate = compact_render(decode_tail(trajectory_ids, keep))
            if token_count(candidate) <= prompt_limit:
                best_prompt = candidate
                best_keep = keep
                low = keep + 1
            else:
                high = keep - 1
        print(
            "[SUMMARIZER-RETRY-PROMPT-TRUNCATE] "
            f"task_id={rollout.get('task_id')!r} "
            f"attempt={next_attempt}/{self.contract_max_attempts} "
            "mode=single_user_trajectory_tail "
            f"prompt_limit={prompt_limit} "
            f"rejected_tokens_dropped={len(rejected_ids)} "
            f"trajectory_tokens_dropped={len(trajectory_ids) - best_keep}",
            flush=True,
        )
        return best_prompt

    @staticmethod
    def _parse(text: str, rollout: dict) -> dict:
        """Extract the final structured JSON skill without repairing its fields."""
        return _parse_summary_text(text, rollout)

    def _generation_kwargs(self) -> dict:
        temperature = float(
            self.cfg.get("models", {}).get("summarizer_temperature", 0.0)
        )
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        # Do not replace the model's chat-end token with the tokenizer's single
        # EOS value. Some Qwen checkpoints expose an additional <|im_end|> in
        # generation_config; dropping it makes otherwise finished answers run
        # all the way to max_new_tokens. Generation and cutoff detection must
        # use the exact same union of declared stop-token ids.
        eos_token_ids = sorted(_eos_token_ids(self.llm, self.tokenizer))
        if eos_token_ids:
            generation_kwargs["eos_token_id"] = (
                eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids
            )
        if temperature > 0:
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": float(
                        self.cfg.get("models", {}).get("summarizer_top_p", 0.9)
                    ),
                }
            )
        return generation_kwargs

    def _vllm_sampling_params(self):
        temperature = float(
            self.cfg.get("models", {}).get("summarizer_temperature", 0.0)
        )
        eos_token_ids = sorted(_eos_token_ids(None, self.tokenizer))
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (
            im_end_id is not None
            and im_end_id != getattr(self.tokenizer, "unk_token_id", None)
        ):
            eos_token_ids = sorted({*eos_token_ids, int(im_end_id)})
        kwargs = {
            "max_tokens": self.max_new_tokens,
            "temperature": temperature,
            "top_p": (
                float(
                    self.cfg.get("models", {}).get("summarizer_top_p", 1.0)
                )
                if temperature > 0
                else 1.0
            ),
            "skip_special_tokens": True,
        }
        if eos_token_ids:
            kwargs["stop_token_ids"] = eos_token_ids
        return self._vllm_sampling_params_cls(**kwargs)

    def _generate_texts_vllm(
        self,
        prompts: list[str],
        rollouts: list[dict],
        *,
        label: str,
    ) -> list[str | _RecoverableSummaryOutputError]:
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        prompt_token_ids = [
            [int(token) for token in token_ids]
            for token_ids in encoded["input_ids"]
        ]
        prompt_lengths = [len(token_ids) for token_ids in prompt_token_ids]
        prompt_limit = min(
            self.max_input_tokens,
            max(1, self.context_window_tokens - self.max_new_tokens),
        )
        oversized = [
            (index, length)
            for index, length in enumerate(prompt_lengths)
            if length > prompt_limit
        ]
        if oversized:
            preview = oversized[:8]
            raise RuntimeError(
                f"{label} prompt would exceed the non-truncating vLLM token "
                f"budget: limit={prompt_limit} rows={preview}. Increase the "
                "configured context budget; silent truncation is forbidden."
            )
        outputs = self.llm.generate(
            [
                {"prompt_token_ids": token_ids}
                for token_ids in prompt_token_ids
            ],
            self._vllm_sampling_params(),
            use_tqdm=False,
        )
        if len(outputs) != len(rollouts):
            raise RuntimeError(
                f"{label} vLLM output/rollout count mismatch: "
                f"outputs={len(outputs)} rollouts={len(rollouts)}"
            )
        results: list[str | _RecoverableSummaryOutputError] = []
        for request_output, rollout in zip(outputs, rollouts):
            candidates = list(getattr(request_output, "outputs", ()) or ())
            if len(candidates) != 1:
                raise RuntimeError(
                    f"{label} requires exactly one vLLM continuation per prompt"
                )
            candidate = candidates[0]
            token_ids = [int(token) for token in candidate.token_ids]
            try:
                final_ids = _final_answer_token_ids(
                    token_ids,
                    thinking_enabled=self.enable_thinking,
                    thinking_end_token_id=self.thinking_end_token_id,
                )
                text = self.tokenizer.decode(
                    final_ids,
                    skip_special_tokens=True,
                )
            except _RecoverableSummaryOutputError as exc:
                results.append(exc)
                continue
            if str(getattr(candidate, "finish_reason", "")) == "length":
                cutoff_reason = (
                    "Summarizer generation reached max_new_tokens without a "
                    "stop token."
                )
                recovered, status = _capped_complete_json_recovery(
                    text,
                    rollout,
                    cutoff_reason=cutoff_reason,
                )
                failure_key = str(
                    rollout.get("rollout_id") or rollout.get("task_id") or ""
                )
                self.generation_outcomes[failure_key] = status
                if not recovered:
                    results.append(
                        _RecoverableSummaryOutputError(
                            status,
                            status="cap_incomplete_or_invalid_json",
                        )
                    )
                    continue
            results.append(text)
        return results

    def _generate_texts(
        self,
        prompts: list[str],
        rollouts: list[dict],
        *,
        device,
        label: str,
    ) -> list[str | _RecoverableSummaryOutputError]:
        if len(prompts) != len(rollouts):
            raise RuntimeError(
                f"{label} prompt/rollout count mismatch: "
                f"prompts={len(prompts)} rollouts={len(rollouts)}"
            )
        if not hasattr(self, "generation_outcomes"):
            self.generation_outcomes = {}
        if getattr(self, "backend", "transformers") == "vllm":
            return self._generate_texts_vllm(
                prompts,
                rollouts,
                label=label,
            )
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=not self.fail_on_prompt_truncation,
            max_length=(
                self.max_input_tokens if not self.fail_on_prompt_truncation else None
            ),
        )
        if self.fail_on_prompt_truncation:
            _check_no_prompt_truncation(
                inputs["attention_mask"],
                configured_input_limit=self.max_input_tokens,
                generation_tokens=self.max_new_tokens,
                context_window=self.context_window_tokens,
                label=label,
            )
        inputs = inputs.to(device)
        out = self.llm.generate(**inputs, **self._generation_kwargs())
        input_len = inputs["input_ids"].shape[1]
        generated_ids = out[:, input_len:]
        cutoff_reasons = _generation_cutoff_failure_reasons(
            generated_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_ids=_eos_token_ids(self.llm, self.tokenizer),
            rollouts=rollouts,
        )
        results: list[str | _RecoverableSummaryOutputError] = []
        for ids, cutoff_reason, rollout in zip(
            generated_ids, cutoff_reasons, rollouts
        ):
            token_ids = [int(token) for token in ids.detach().cpu().tolist()]
            try:
                final_token_ids = _final_answer_token_ids(
                    token_ids,
                    thinking_enabled=self.enable_thinking,
                    thinking_end_token_id=self.thinking_end_token_id,
                )
            except _RecoverableSummaryOutputError as exc:
                status = (
                    "cap_incomplete_or_invalid_json"
                    if cutoff_reason is not None
                    else exc.status
                )
                reason = str(exc)
                if cutoff_reason is not None:
                    reason = (
                        f"{cutoff_reason} "
                        "generation_status=cap_incomplete_or_invalid_json "
                        f"final_answer_extraction_error={reason}"
                    )
                    failure_key = str(
                        rollout.get("rollout_id")
                        or rollout.get("task_id")
                        or ""
                    )
                    self.generation_outcomes[failure_key] = status
                results.append(
                    _RecoverableSummaryOutputError(
                        f"{reason} task_id={rollout.get('task_id')!r} "
                        f"rollout_id={rollout.get('rollout_id')!r}",
                        status=status,
                    )
                )
                continue
            text = self.tokenizer.decode(
                final_token_ids,
                skip_special_tokens=True,
            )
            if cutoff_reason is not None:
                recovered, recovery_status = _capped_complete_json_recovery(
                    text,
                    rollout,
                    cutoff_reason=cutoff_reason,
                )
                failure_key = str(
                    rollout.get("rollout_id") or rollout.get("task_id") or ""
                )
                if not recovered:
                    self.generation_outcomes[failure_key] = (
                        "cap_incomplete_or_invalid_json"
                    )
                    results.append(
                        _RecoverableSummaryOutputError(
                            recovery_status,
                            status="cap_incomplete_or_invalid_json",
                        )
                    )
                    continue
                self.generation_outcomes[failure_key] = recovery_status
                print(
                    "[SUMMARIZER-CAP-RECOVERY] "
                    "status=cap_recovered_complete_json "
                    f"task_id={rollout.get('task_id')!r} "
                    f"rollout_id={rollout.get('rollout_id')!r} "
                    f"generated_tokens={len(token_ids)} "
                    f"max_new_tokens={self.max_new_tokens}",
                    flush=True,
                )
            results.append(text)
        return results

    def _parse_with_contract_retries(
        self,
        initial_text: str | _RecoverableSummaryOutputError,
        rollout: dict,
        *,
        device,
    ) -> dict | None:
        result: str | _RecoverableSummaryOutputError = initial_text
        last_error: Exception | None = None
        for attempt in range(1, self.contract_max_attempts + 1):
            try:
                if isinstance(result, _RecoverableSummaryOutputError):
                    raise result
                return self._parse(str(result or ""), rollout)
            except (
                GeneratedSummaryExactDuplicateError,
                GeneratedSummaryFormatError,
                _RecoverableSummaryOutputError,
            ) as exc:
                last_error = exc
                if attempt >= self.contract_max_attempts:
                    failure_key = str(
                        rollout.get("rollout_id")
                        or rollout.get("task_id")
                        or ""
                    )
                    self.generation_outcomes[failure_key] = (
                        "contract_exhausted_empty_skill"
                    )
                    reason = str(exc).replace("\n", " ")[:400]
                    print(
                        "[SUMMARIZER-EMPTY-SKILL] "
                        f"task_id={rollout.get('task_id')!r} "
                        f"rollout_id={rollout.get('rollout_id')!r} "
                        f"attempts={self.contract_max_attempts} "
                        f"reason={reason}",
                        flush=True,
                    )
                    return None

                next_attempt = attempt + 1
                reason = str(exc).replace("\n", " ")[:400]
                print(
                    "[SUMMARIZER-CONTRACT-RETRY] "
                    f"task_id={rollout.get('task_id')!r} "
                    f"attempt={next_attempt}/{self.contract_max_attempts} "
                    f"reason={reason}",
                    flush=True,
                )
                retry_prompt = self._contract_retry_prompt(
                    rollout,
                    rejected_output=(
                        str(result or "")
                        if not isinstance(result, _RecoverableSummaryOutputError)
                        else ""
                    ),
                    validation_error=exc,
                    next_attempt=next_attempt,
                )
                result = self._generate_texts(
                    [retry_prompt],
                    [rollout],
                    device=device,
                    label=f"summarizer contract retry {next_attempt}",
                )[0]

        raise AssertionError(f"unreachable summarizer retry state: {last_error}")

    def _parse_batch_with_contract_retries(
        self,
        initial_texts: list[str | _RecoverableSummaryOutputError],
        rollouts: list[dict],
        *,
        device,
    ) -> list[dict | None]:
        """Parse one generation wave and batch every contract-retry wave.

        ``LLM.generate`` already keeps a submitted request list continuously
        scheduled. Retrying each rejected item in a separate call defeats that
        scheduler and leaves the GPU decoding one request at a time. Preserve
        result ordering with explicit indexes while submitting all rejected
        items from the same attempt together.
        """

        if len(initial_texts) != len(rollouts):
            raise RuntimeError(
                "summarizer parse input/rollout count mismatch: "
                f"texts={len(initial_texts)} rollouts={len(rollouts)}"
            )
        if not rollouts:
            return []

        parsed: list[dict | None] = [None] * len(rollouts)
        pending = list(enumerate(initial_texts))
        for attempt in range(1, self.contract_max_attempts + 1):
            retry_indexes: list[int] = []
            retry_prompts: list[str] = []
            retry_rollouts: list[dict] = []

            for index, result in pending:
                rollout = rollouts[index]
                try:
                    if isinstance(result, _RecoverableSummaryOutputError):
                        raise result
                    parsed[index] = self._parse(str(result or ""), rollout)
                except (
                    GeneratedSummaryExactDuplicateError,
                    GeneratedSummaryFormatError,
                    _RecoverableSummaryOutputError,
                ) as exc:
                    if attempt >= self.contract_max_attempts:
                        failure_key = str(
                            rollout.get("rollout_id")
                            or rollout.get("task_id")
                            or ""
                        )
                        self.generation_outcomes[failure_key] = (
                            "contract_exhausted_empty_skill"
                        )
                        reason = str(exc).replace("\n", " ")[:400]
                        print(
                            "[SUMMARIZER-EMPTY-SKILL] "
                            f"task_id={rollout.get('task_id')!r} "
                            f"rollout_id={rollout.get('rollout_id')!r} "
                            f"attempts={self.contract_max_attempts} "
                            f"reason={reason}",
                            flush=True,
                        )
                        continue

                    next_attempt = attempt + 1
                    reason = str(exc).replace("\n", " ")[:400]
                    print(
                        "[SUMMARIZER-CONTRACT-RETRY] "
                        f"task_id={rollout.get('task_id')!r} "
                        f"attempt={next_attempt}/{self.contract_max_attempts} "
                        f"reason={reason}",
                        flush=True,
                    )
                    retry_indexes.append(index)
                    retry_rollouts.append(rollout)
                    retry_prompts.append(
                        self._contract_retry_prompt(
                            rollout,
                            rejected_output=(
                                str(result or "")
                                if not isinstance(
                                    result, _RecoverableSummaryOutputError
                                )
                                else ""
                            ),
                            validation_error=exc,
                            next_attempt=next_attempt,
                        )
                    )

            if not retry_indexes:
                break

            next_attempt = attempt + 1
            retry_texts = self._generate_texts(
                retry_prompts,
                retry_rollouts,
                device=device,
                label=f"summarizer contract retry {next_attempt}",
            )
            if len(retry_texts) != len(retry_indexes):
                raise RuntimeError(
                    "Summarizer contract retry output count mismatch: "
                    f"attempt={next_attempt} outputs={len(retry_texts)} "
                    f"rollouts={len(retry_indexes)}"
                )
            pending = list(zip(retry_indexes, retry_texts))

        return parsed

    @torch.inference_mode()
    def summarize_batch(
        self,
        rollouts: list[dict],
        progress_callback=None,
    ) -> list[dict | None]:
        if not rollouts:
            return []
        self.generation_outcomes = {}
        prompts = [self._prompt(r) for r in rollouts]
        parsed = []
        device = (
            None
            if getattr(self, "backend", "transformers") == "vllm"
            else next(self.llm.parameters()).device
        )
        total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        progress_every = max(1, int(self.cfg.get("logging", {}).get("progress_every_batches", 5)))
        for batch_index, start in enumerate(range(0, len(prompts), self.batch_size), 1):
            batch = prompts[start : start + self.batch_size]
            batch_rollouts = rollouts[start : start + self.batch_size]
            texts = self._generate_texts(
                batch,
                batch_rollouts,
                device=device,
                label="summarizer",
            )
            parsed.extend(
                self._parse_batch_with_contract_retries(
                    texts,
                    batch_rollouts,
                    device=device,
                )
            )
            if progress_callback is not None and (batch_index == 1 or batch_index % progress_every == 0 or batch_index == total_batches):
                progress_callback(
                    min(start + self.batch_size, len(prompts)),
                    len(prompts),
                    batch_index,
                    total_batches,
                )
        return parsed
