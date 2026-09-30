from __future__ import annotations

from skillflow.evaluation.benchmark_prompts import (
    benchmark_action_is_admissible,
    has_complete_benchmark_action,
    parse_benchmark_action,
    render_benchmark_prompt,
    system_prompt,
)
from skillflow.evaluation.names import normalize_benchmark_name
from skillflow.evaluation.actor_base import (
    ActionGeneration,
    ActionGenerator,
    _RawGenerationBatch,
    _format_skills,
)

from .webshop_memory import render_benchmark_prompt


class BenchmarkActionGenerator(ActionGenerator):
    """Reuse the existing model backend with an opt-in non-ALFWorld prompt.

    The formal :class:`ActionGenerator` and its ALFWorld prompt remain untouched;
    this subclass is used only by ``evaluate_benchmark``.
    """

    requires_native_thinking = True

    def __init__(self, cfg: dict):
        benchmark_cfg = cfg.get("benchmark") or {}
        self.benchmark_name = normalize_benchmark_name(benchmark_cfg.get("name"))
        super().__init__(cfg)

    def _prompt(
        self,
        goal: str,
        skill: str,
        observation: str,
        admissible: list[str],
        history: list[dict],
    ) -> str:
        raw_skill = str(skill or "").strip()
        prompt = render_benchmark_prompt(
            self.benchmark_name,
            task_description=goal,
            skill_text=_format_skills(raw_skill) if raw_skill else "",
            current_observation=observation,
            admissible_actions=admissible,
            history=history,
            history_length=self.history_length,
            observation_chars=self.history_observation_chars,
        )
        chat = [
            {"role": "system", "content": system_prompt(self.benchmark_name)},
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
                    "The benchmark executor chat template must explicitly support "
                    "enable_thinking=True."
                ) from exc
        if self.enable_thinking:
            raise RuntimeError(
                "models.agent_enable_thinking=true requires tokenizer.apply_chat_template"
            )
        return prompt

    def _vllm_sampling_params(self, max_new_tokens):
        params = super()._vllm_sampling_params(max_new_tokens)
        expected = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                    "n": 1, "max_tokens": 4096, "presence_penalty": 0.0,
                    "repetition_penalty": 1.0}
        if any(getattr(params, key, None) != value for key, value in expected.items()):
            raise RuntimeError("WebShop official actor sampling changed")
        return params

    def _generate_raw_batch(
        self,
        prompts: list[str],
        *,
        label: str,
        max_new_tokens: int | None = None,
    ) -> _RawGenerationBatch:
        generated = super()._generate_raw_batch(
            prompts,
            label=label,
            max_new_tokens=max_new_tokens,
        )
        incomplete = frozenset(
            index
            for index in generated.incomplete_rows
            if not has_complete_benchmark_action(
                generated.responses[index], self.benchmark_name
            )
        )
        return _RawGenerationBatch(generated.responses, incomplete)

    def generate_batch(self, items: list[dict]) -> list[ActionGeneration]:
        if not items:
            return []
        prompts = [
            self._prompt(
                item["goal"],
                item["skill"],
                item["observation"],
                item.get("admissible", []),
                item["history"],
            )
            for item in items
        ]
        results: list[ActionGeneration] = []
        for start in range(0, len(prompts), self.llm_batch_size):
            batch_prompts = prompts[start : start + self.llm_batch_size]
            batch_items = items[start : start + self.llm_batch_size]
            generated = self._generate_raw_batch(batch_prompts, label="benchmark_agent")
            if len(generated.responses) != len(batch_items):
                raise RuntimeError(
                    "Benchmark agent generation batch-size mismatch: "
                    f"expected={len(batch_items)} actual={len(generated.responses)}"
                )
            for row_index, (response, item) in enumerate(
                zip(generated.responses, batch_items)
            ):
                parsed = parse_benchmark_action(response, self.benchmark_name)
                environment_admissible = list(
                    item.get("environment_admissible", item.get("admissible", []))
                )
                results.append(
                    ActionGeneration(
                        action=parsed.action,
                        response=response,
                        extraction_mode=parsed.extraction_mode,
                        format_valid=parsed.format_valid,
                        action_is_admissible=benchmark_action_is_admissible(
                            parsed.action,
                            environment_admissible,
                            self.benchmark_name,
                            format_valid=parsed.format_valid,
                        ),
                        generation_incomplete=row_index in generated.incomplete_rows,
                    )
                )
        return results


__all__ = ["BenchmarkActionGenerator"]
