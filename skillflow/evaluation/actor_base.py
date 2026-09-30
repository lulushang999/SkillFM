from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from skillflow.evaluation.attention import attention_kwargs
from skillflow.evaluation.import_safety import hide_optional_audio_packages
from skillflow.evaluation.skill_text import render_skill_for_agent


SYSTEM = (
    "You are an expert agent operating in the ALFRED Embodied Environment. "
    "Think carefully, then choose exactly one action from the supplied candidate "
    "list and follow the required response format exactly."
)

NON_THINKING_SYSTEM = (
    "You are an expert agent operating in the ALFRED Embodied Environment. "
    "Choose exactly one action from the supplied candidate list and output it "
    "immediately in the required response format."
)

ALFWORLD_TEMPLATE = """Your task is to:
{task_description}

## Relevant Skill
{skills}

Treat the skill as guidance, not as a fixed action sequence. If the skill
conflicts with the current observation or the candidate actions, follow the
current observation and candidate actions.

## Recent Interactions
Prior to this step, you have already taken {current_step_minus_one} step(s). Below are the most recent {history_window} observations and the corresponding actions you took:
{action_history}

## Current Observation
You are now at step {current_step} and your current observation is: {current_observation}

## Candidate Actions
[
{admissible_actions}
]

{response_contract}
"""

STEP = ALFWORLD_TEMPLATE

ACTION_RESPONSE_FORMAT = """Put the selected action inside:
<action>...</action>"""


def displayed_admissible_actions(admissible: Sequence[str]) -> list[str]:
    """Return the environment candidate list without policy filtering."""

    return [str(action) for action in admissible]




def _format_admissible_actions(admissible: Sequence[str]) -> str:
    return ",\n".join(
        f"'{action}'" for action in displayed_admissible_actions(admissible)
    )


def _format_action_history(
    history: list[dict],
    history_length: int,
    observation_chars: int,
) -> tuple[str, int]:
    recent = history[-history_length:] if history_length > 0 else []
    valid_len = len(recent)
    start_idx = len(history) - valid_len
    lines = []
    if start_idx > 0:
        lines.append(
            f"[Earlier completed steps 1-{start_idx} are outside the visible window.]"
        )
    obs_chars = max(0, int(observation_chars))
    for j, rec in enumerate(recent):
        step_num = start_idx + j + 1
        obs = str(rec.get("observation", ""))
        if obs_chars > 0:
            obs = obs[-obs_chars:]
        act = rec.get("action", "")
        lines.append(f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']")
    return "\n".join(lines), valid_len


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
    attention_mask,
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


def _format_skills(skill: str) -> str:
    raw = str(skill or "").strip()
    if not raw:
        return ""
    return render_skill_for_agent(raw)


def extract_action(response: str) -> tuple[str, str, bool]:
    """Extract one tagged action without repairing it against candidates.

    A malformed response is passed to ALFWorld as its final 30 characters so
    the environment can return normal invalid-command feedback on the next step.
    """

    raw = str(response or "")
    matches = list(
        re.finditer(
            r"<action>\s*(.*?)\s*</action>",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if not matches:
        return raw[-30:], "response_tail", False
    action = matches[0].group(1).strip()
    return action, "action_tag", len(matches) == 1 and bool(action)


def _has_complete_executor_response(text: str) -> bool:
    return extract_action(text)[2]


@dataclass(frozen=True)
class _RawGenerationBatch:
    responses: list[str]
    incomplete_rows: frozenset[int]


@dataclass(frozen=True)
class ActionGeneration:
    """One model response plus the unmodified extracted environment action."""

    action: str
    response: str
    extraction_mode: str
    format_valid: bool
    action_is_admissible: bool
    generation_incomplete: bool


def _eos_token_ids(model, tokenizer, generation_config=None) -> set[int]:
    values = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(generation_config, "eos_token_id", None),
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


@contextlib.contextmanager
def _silence(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


class ActionGenerator:
    requires_native_thinking = True

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.llm = self.tokenizer = None
        name = cfg["models"]["backbone"]
        trust = bool(cfg["models"].get("trust_remote_code", True))
        gpu = int(
            os.environ.get(
                "SKILLFLOW_LOCAL_RANK",
                os.environ.get("LOCAL_RANK", "0"),
            )
        )
        quiet = bool(cfg.get("logging", {}).get("quiet_model_load", True))
        self.backend = str(
            cfg.get("models", {}).get("agent_backend", "transformers")
        ).strip().lower()
        if self.backend not in {"vllm", "transformers"}:
            raise ValueError(
                "models.agent_backend must be exactly 'vllm' or 'transformers'"
            )
        rollout_cfg = cfg.get("rollout", {}) or {}
        if self.backend == "vllm":
            self.llm_batch_size = max(
                1,
                int(
                    cfg.get("models", {}).get(
                        "agent_vllm_request_batch_size",
                        rollout_cfg.get("env_batch_size", 1),
                    )
                ),
            )
        else:
            self.llm_batch_size = max(
                1,
                int(
                    rollout_cfg.get(
                        "llm_batch_size",
                        rollout_cfg.get("env_batch_size", 1),
                    )
                ),
            )
        self.history_length = max(0, int(cfg.get("env", {}).get("history_length", cfg.get("rollout", {}).get("history_length", 8))))
        self.history_observation_chars = max(
            0,
            int(
                cfg.get("env", {}).get(
                    "history_observation_chars",
                    cfg.get("rollout", {}).get("history_observation_chars", 800),
                )
            ),
        )
        self.fail_on_prompt_truncation = bool(
            cfg.get("models", {}).get("fail_on_prompt_truncation", True)
        )
        requires_native_thinking = cfg.get("models", {}).get(
            "agent_requires_native_thinking", self.requires_native_thinking
        )
        if not isinstance(requires_native_thinking, bool):
            raise ValueError("models.agent_requires_native_thinking must be boolean")
        self.requires_native_thinking = requires_native_thinking
        self.enable_thinking = bool(
            cfg.get("models", {}).get("agent_enable_thinking", False)
        )
        if self.requires_native_thinking and not self.enable_thinking:
            raise ValueError(
                "models.agent_enable_thinking must be true for the Qwen3 executor"
            )
        primary_generation_tokens = int(cfg["sampling"].get("max_new_tokens", 128))
        if primary_generation_tokens <= 0:
            raise ValueError("sampling.max_new_tokens must be positive")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        with hide_optional_audio_packages():
            from transformers import AutoTokenizer, GenerationConfig
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            try:
                hf_logging.disable_progress_bar()
            except Exception:
                pass
            # vLLM installs process-wide log handlers during construction. Do
            # not bind those handlers to _silence's soon-to-close devnull stream.
            with _silence(quiet and self.backend != "vllm"):
                self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust)
                self.generation_config = GenerationConfig.from_pretrained(
                    name,
                    trust_remote_code=trust,
                )
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                # Left padding avoids right-padding artifacts in batched decoder-only generation.
                self.tokenizer.padding_side = "left"
                self.tokenizer.truncation_side = "left"
                self.max_input_tokens = max(
                    1024, int(cfg.get("models", {}).get("agent_max_input_tokens", 8192))
                )
                if self.backend == "vllm":
                    import torch

                    if not torch.cuda.is_available():
                        raise RuntimeError(
                            "models.agent_backend='vllm' requires a CUDA GPU"
                        )
                    torch.cuda.set_device(gpu)
                    try:
                        from vllm import LLM, SamplingParams
                    except ModuleNotFoundError as exc:
                        raise RuntimeError(
                            "vLLM is required for agent action generation. Install "
                            "the verified GPU environment before launching."
                        ) from exc
                    self._vllm_sampling_params_cls = SamplingParams
                    max_new_tokens = int(
                        cfg.get("sampling", {}).get("max_new_tokens", 128)
                    )
                    max_model_len = int(
                        cfg.get("models", {}).get(
                            "agent_vllm_max_model_len",
                            self.max_input_tokens + max_new_tokens,
                        )
                    )
                    if max_model_len < self.max_input_tokens + max_new_tokens:
                        raise ValueError(
                            "models.agent_vllm_max_model_len must cover "
                            "agent_max_input_tokens + sampling.max_new_tokens"
                        )
                    self.llm = LLM(
                        model=name,
                        tokenizer=name,
                        trust_remote_code=trust,
                        dtype=str(
                            cfg.get("models", {}).get(
                                "agent_vllm_dtype", "bfloat16"
                            )
                        ),
                        tensor_parallel_size=int(
                            cfg.get("models", {}).get(
                                "agent_vllm_tensor_parallel_size", 1
                            )
                        ),
                        max_model_len=max_model_len,
                        gpu_memory_utilization=float(
                            cfg.get("models", {}).get(
                                "agent_vllm_gpu_memory_utilization", 0.90
                            )
                        ),
                        max_num_seqs=int(
                            cfg.get("models", {}).get(
                                "agent_vllm_max_num_seqs",
                                self.llm_batch_size,
                            )
                        ),
                        max_num_batched_tokens=int(
                            cfg.get("models", {}).get(
                                "agent_vllm_max_num_batched_tokens", 32768
                            )
                        ),
                        enable_chunked_prefill=bool(
                            cfg.get("models", {}).get(
                                "agent_vllm_enable_chunked_prefill", True
                            )
                        ),
                        enable_prefix_caching=bool(
                            cfg.get("models", {}).get(
                                "agent_vllm_enable_prefix_caching", True
                            )
                        ),
                        enforce_eager=bool(
                            cfg.get("models", {}).get(
                                "agent_vllm_enforce_eager", False
                            )
                        ),
                        attention_config={
                            "backend": str(
                                cfg.get("models", {}).get(
                                    "agent_vllm_attention_backend", "FLASH_ATTN"
                                )
                            ),
                            "flash_attn_version": int(
                                cfg.get("models", {}).get(
                                    "agent_vllm_flash_attn_version", 2
                                )
                            ),
                        },
                        disable_log_stats=quiet,
                        generation_config="vllm",
                        seed=(
                            int((cfg.get("run") or {}).get("seed", 42))
                            + int(cfg.get("_rollout_round", 0)) * 1009
                            + int(
                                os.environ.get(
                                    "SKILLFLOW_RANK",
                                    os.environ.get("RANK", "0"),
                                )
                            )
                        ),
                    )
                    self.context_window_tokens = max_model_len
                else:
                    from transformers import AutoModelForCausalLM

                    kwargs = {
                        "dtype": "auto",
                        "device_map": {"": gpu},
                        "trust_remote_code": trust,
                    }
                    kwargs.update(
                        attention_kwargs(cfg, "agent_attention_implementation")
                    )
                    self.llm = AutoModelForCausalLM.from_pretrained(
                        name, **kwargs
                    ).eval()
                    self.llm.requires_grad_(False)
                    self.context_window_tokens = _context_window_tokens(
                        self.llm, self.tokenizer
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

    def _prompt(
        self,
        goal: str,
        skill: str,
        observation: str,
        admissible: list[str],
        history: list[dict],
    ) -> str:
        raw_skill = str(skill or "").strip()
        admissible_text = _format_admissible_actions(admissible)
        action_history, valid_len = _format_action_history(
            history,
            self.history_length,
            self.history_observation_chars,
        )
        step_count = len(history)
        prompt = ALFWORLD_TEMPLATE.format(
            task_description=goal,
            skills=_format_skills(raw_skill),
            history_window=valid_len,
            action_history=action_history or "None yet.",
            current_step=step_count + 1,
            current_step_minus_one=step_count,
            current_observation=observation,
            admissible_actions=admissible_text,
            response_contract=ACTION_RESPONSE_FORMAT,
        )
        chat = [
            {
                "role": "system",
                "content": SYSTEM if self.enable_thinking else NON_THINKING_SYSTEM,
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if not self.enable_thinking:
                return self.tokenizer.apply_chat_template(chat, **template_kwargs)
            try:
                return self.tokenizer.apply_chat_template(
                    chat,
                    **template_kwargs,
                    enable_thinking=True,
                )
            except TypeError as exc:
                raise RuntimeError(
                    "The Qwen3 executor chat template must explicitly support "
                    "enable_thinking=True; refusing to rely on the model's "
                    "default thinking mode."
                ) from exc
        if self.enable_thinking:
            raise RuntimeError(
                "models.agent_enable_thinking=true requires tokenizer.apply_chat_template; "
                "refusing a plain-prompt fallback that would change thinking semantics."
            )
        return prompt

    def _generate_raw_batch(
        self,
        prompts: list[str],
        *,
        label: str,
        max_new_tokens: Optional[int] = None,
    ) -> _RawGenerationBatch:
        if not prompts:
            return _RawGenerationBatch([], frozenset())
        if max_new_tokens is None:
            max_new_tokens = int(self.cfg["sampling"].get("max_new_tokens", 128))
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if getattr(self, "backend", "transformers") == "vllm":
            return self._generate_raw_batch_vllm(
                prompts,
                label=label,
                max_new_tokens=max_new_tokens,
            )
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=not self.fail_on_prompt_truncation,
            max_length=self.max_input_tokens if not self.fail_on_prompt_truncation else None,
        )
        if self.fail_on_prompt_truncation:
            _check_no_prompt_truncation(
                inputs["attention_mask"],
                configured_input_limit=self.max_input_tokens,
                generation_tokens=max_new_tokens,
                context_window=self.context_window_tokens,
                label=label,
            )
        inputs = inputs.to(self.llm.device)
        temperature = float(self.cfg["sampling"].get("temperature", 0.0))
        generation_kwargs = {
            "do_sample": temperature > 0,
            # Sampling changes token choice, never candidate count.  The final
            # verifier contract permits exactly one action continuation.
            "num_return_sequences": 1,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        eos_token_ids = _eos_token_ids(self.llm, self.tokenizer)
        if eos_token_ids:
            ordered_eos_ids = sorted(eos_token_ids)
            generation_kwargs["eos_token_id"] = (
                ordered_eos_ids[0] if len(ordered_eos_ids) == 1 else ordered_eos_ids
            )
        if temperature > 0:
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": float(self.cfg["sampling"].get("top_p", 1.0)),
                    "top_k": int(self.cfg["sampling"].get("top_k", 0)),
                    "min_p": float(self.cfg["sampling"].get("min_p", 0.0)),
                }
            )
        out = self.llm.generate(
            **inputs,
            **generation_kwargs,
        )
        input_len = inputs.input_ids.shape[1]
        generated = out[:, input_len:]
        responses = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated
        ]
        incomplete_rows: set[int] = set()
        if generated.shape[1] >= max_new_tokens:
            for row_index, (ids, response) in enumerate(zip(generated, responses)):
                saw_eos = bool(eos_token_ids) and any(
                    bool((ids == eos_id).any().item()) for eos_id in eos_token_ids
                )
                # For an action executor, a closed action field is already a
                # complete structured answer.  A missing EOS after that field
                # must not kill every distributed worker.  Rows with neither
                # EOS nor a closed action are marked incomplete; their response
                # tail remains an invalid environment action instead of
                # triggering recovery or aborting the rollout.
                if not saw_eos and not _has_complete_executor_response(response):
                    incomplete_rows.add(row_index)
        return _RawGenerationBatch(responses, frozenset(incomplete_rows))

    def _vllm_sampling_params(self, max_new_tokens: int):
        temperature = float(self.cfg["sampling"].get("temperature", 0.0))
        kwargs = {
            "n": 1,
            "max_tokens": int(max_new_tokens),
            "temperature": temperature,
            "skip_special_tokens": True,
        }
        if temperature > 0:
            kwargs.update(
                {
                    "top_p": float(self.cfg["sampling"].get("top_p", 1.0)),
                    "top_k": int(self.cfg["sampling"].get("top_k", 0)),
                    "min_p": float(self.cfg["sampling"].get("min_p", 0.0)),
                }
            )
        eos_token_ids = sorted(
            _eos_token_ids(
                None,
                self.tokenizer,
                getattr(self, "generation_config", None),
            )
        )
        if eos_token_ids:
            kwargs["stop_token_ids"] = eos_token_ids
        return self._vllm_sampling_params_cls(**kwargs)

    def _generate_raw_batch_vllm(
        self,
        prompts: list[str],
        *,
        label: str,
        max_new_tokens: int,
    ) -> _RawGenerationBatch:
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
            max(1, self.context_window_tokens - max_new_tokens),
        )
        oversized = [
            (index, length)
            for index, length in enumerate(prompt_lengths)
            if length > prompt_limit
        ]
        if oversized:
            raise RuntimeError(
                f"{label} prompt would exceed the non-truncating vLLM token "
                f"budget: limit={prompt_limit} rows={oversized[:8]}. Increase "
                "the configured context budget or reduce the visible trajectory "
                "window; silent truncation is forbidden."
            )
        outputs = self.llm.generate(
            [
                {"prompt_token_ids": token_ids}
                for token_ids in prompt_token_ids
            ],
            self._vllm_sampling_params(max_new_tokens),
            use_tqdm=False,
        )
        if len(outputs) != len(prompts):
            raise RuntimeError(
                "Agent vLLM output/prompt count mismatch: "
                f"outputs={len(outputs)} prompts={len(prompts)}"
            )
        eos_token_ids = _eos_token_ids(
            None,
            self.tokenizer,
            getattr(self, "generation_config", None),
        )
        responses: list[str] = []
        incomplete_rows: set[int] = set()
        for row_index, request_output in enumerate(outputs):
            candidates = list(getattr(request_output, "outputs", ()) or ())
            if len(candidates) != 1:
                raise RuntimeError(
                    "Agent requires exactly one vLLM continuation per prompt"
                )
            candidate = candidates[0]
            token_ids = [int(token) for token in candidate.token_ids]
            response = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )
            responses.append(response)
            stop_reason = getattr(candidate, "stop_reason", None)
            saw_eos = (
                stop_reason in eos_token_ids
                or any(token in eos_token_ids for token in token_ids)
            )
            reached_limit = (
                str(getattr(candidate, "finish_reason", "")).lower()
                == "length"
                or len(token_ids) >= max_new_tokens
            )
            if (
                reached_limit
                and not saw_eos
                and not _has_complete_executor_response(response)
            ):
                incomplete_rows.add(row_index)
        return _RawGenerationBatch(
            responses,
            frozenset(incomplete_rows),
        )

    def generate_batch(self, items: list[dict]) -> list[ActionGeneration]:
        if not items:
            return []
        prompts = [
            self._prompt(
                item["goal"],
                item["skill"],
                item["observation"],
                item["admissible"],
                item["history"],
            )
            for item in items
        ]
        results: list[ActionGeneration] = []
        for start in range(0, len(prompts), self.llm_batch_size):
            batch_prompts = prompts[start : start + self.llm_batch_size]
            batch_items = items[start : start + self.llm_batch_size]
            generated = self._generate_raw_batch(batch_prompts, label="agent")
            responses = generated.responses
            if len(responses) != len(batch_items):
                raise RuntimeError(
                    "Agent generation batch-size mismatch: "
                    f"expected={len(batch_items)} actual={len(responses)}"
                )
            for row_index, (response, item) in enumerate(
                zip(responses, batch_items)
            ):
                action, extraction_mode, format_valid = extract_action(response)
                environment_admissible = list(
                    item.get("environment_admissible", item["admissible"])
                )
                results.append(
                    ActionGeneration(
                        action=action,
                        response=response,
                        extraction_mode=extraction_mode,
                        format_valid=format_valid,
                        action_is_admissible=action in environment_admissible,
                        generation_incomplete=row_index in generated.incomplete_rows,
                    )
                )
        return results

    def generate(
        self,
        goal: str,
        skill: str,
        observation: str,
        admissible: list[str],
        history: list[dict],
    ) -> ActionGeneration:
        return self.generate_batch(
            [{"goal": goal, "skill": skill, "observation": observation, "admissible": admissible, "history": history}]
        )[0]

