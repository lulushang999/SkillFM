"""The shared three-field executable skill schema."""
from __future__ import annotations
import json
from skillflow.summarization._alfworld_validation import ChainContractError, validate_executable_skill


def canonicalize_executable_skill(value, **_):
    obj = value if isinstance(value, dict) else json.loads(value)
    if isinstance(obj, dict) and isinstance(obj.get('next_executable_skill'), dict):
        obj = obj['next_executable_skill']
    return json.dumps(validate_executable_skill(obj), ensure_ascii=False, indent=2)


def parse_executable_skill(value):
    try:
        return json.loads(canonicalize_executable_skill(value))
    except (ChainContractError, ValueError, TypeError):
        return None
