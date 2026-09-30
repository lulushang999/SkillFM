"""Qwen action generator for the isolated QA summary-collection branch."""

from __future__ import annotations

from collections.abc import Mapping

from skillflow.evaluation.actor_base import ActionGeneration, ActionGenerator
from skillflow.evaluation.qa_prompts import (
    parse_qa_action,
    qa_action_is_admissible,
    render_qa_agent_messages,
)
from skillflow.evaluation.skill_text import render_skill_for_agent


class QASummaryActionGenerator(ActionGenerator):
    """Reuse model loading/generation without reusing the ALFWorld prompt."""

    requires_native_thinking = True

    def _prompt(
        self,
        goal: str,
        skill: object,
        observation: str,
        admissible: list[str],
        history: list[dict],
        max_steps: int,
    ) -> str:
        if isinstance(skill, Mapping):
            rendered_skill = render_skill_for_agent(skill)
        else:
            raw_skill = str(skill or "").strip()
            rendered_skill = render_skill_for_agent(raw_skill) if raw_skill else ""
        messages = render_qa_agent_messages(
            query=goal,
            skill_text=rendered_skill,
            current_observation=observation,
            candidate_actions=admissible,
            history=history,
            history_length=self.history_length,
            observation_chars=self.history_observation_chars,
            max_steps=max_steps,
        )
        if not hasattr(self.tokenizer, "apply_chat_template"):
            if self.enable_thinking:
                raise RuntimeError(
                    "models.agent_enable_thinking=true requires "
                    "tokenizer.apply_chat_template"
                )
            return messages[-1]["content"]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if not self.enable_thinking:
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                **template_kwargs,
                enable_thinking=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "The QA executor chat template must explicitly support "
                "enable_thinking=True."
            ) from exc

    def generate_batch(self, items: list[dict]) -> list[ActionGeneration]:
        if not items:
            return []
        prompts = [
            self._prompt(
                str(item["goal"]),
                item.get("skill") or "",
                str(item.get("observation") or ""),
                [str(value) for value in item.get("admissible", [])],
                list(item.get("history") or []),
                int(item["horizon"]),
            )
            for item in items
        ]
        results: list[ActionGeneration] = []
        for start in range(0, len(prompts), self.llm_batch_size):
            batch_prompts = prompts[start : start + self.llm_batch_size]
            batch_items = items[start : start + self.llm_batch_size]
            generated = self._generate_raw_batch(
                batch_prompts,
                label="qa_summary_agent",
            )
            if len(generated.responses) != len(batch_items):
                raise RuntimeError(
                    "QA agent generation batch-size mismatch: "
                    f"expected={len(batch_items)} actual={len(generated.responses)}"
                )
            for row_index, (response, item) in enumerate(
                zip(generated.responses, batch_items)
            ):
                parsed = parse_qa_action(response)
                candidates = [
                    str(value)
                    for value in item.get(
                        "environment_admissible",
                        item.get("admissible", []),
                    )
                ]
                results.append(
                    ActionGeneration(
                        action=parsed.action,
                        response=response,
                        extraction_mode=parsed.extraction_mode,
                        format_valid=parsed.format_valid,
                        action_is_admissible=qa_action_is_admissible(
                            parsed,
                            candidates,
                        ),
                        generation_incomplete=(row_index in generated.incomplete_rows),
                    )
                )
        return results


__all__ = ["QASummaryActionGenerator"]
