from __future__ import annotations

import json
import re
from typing import Any

def validate_executable_skill(value: object) -> dict[str, str]:
    keys = {"when_to_use", "high_level_guidance", "low_level_guidance"}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError("skill must contain exactly the three executable fields")
    if any(not isinstance(value[key], str) or not value[key].strip() for key in keys):
        raise ValueError("every skill field must be a non-empty string")
    return {key: value[key].strip() for key in ("when_to_use", "high_level_guidance", "low_level_guidance")}


def clean_text(value: Any) -> str:
    """Normalize whitespace in task/context text without inventing content."""
    if not isinstance(value, str):
        return ""
    value = value.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", value).strip()


def parse_executable_skill(value: object) -> dict[str, Any]:
    """Parse the single role-free executable-skill JSON contract."""
    if isinstance(value, dict):
        parsed = value
    else:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("Executable skill is empty")
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise ValueError(f"Executable skill is not exact JSON: {exc}") from exc
    return validate_executable_skill(parsed)


def canonical_skill_text(value: object) -> str:
    return json.dumps(
        parse_executable_skill(value),
        ensure_ascii=False,
        indent=2,
    )


def is_valid_executable_skill(value: object) -> bool:
    try:
        parse_executable_skill(value)
    except ValueError:
        return False
    return True


def render_skill_for_agent(value: object) -> str:
    """Render the three compact guidance fields for the action model."""
    skill = parse_executable_skill(value)
    return "\n".join(
        [
            f"When to use: {skill['when_to_use']}",
            f"High-level guidance: {skill['high_level_guidance']}",
            f"Low-level guidance: {skill['low_level_guidance']}",
        ]
    )


__all__ = [
    "canonical_skill_text",
    "clean_text",
    "is_valid_executable_skill",
    "parse_executable_skill",
    "render_skill_for_agent",
]
