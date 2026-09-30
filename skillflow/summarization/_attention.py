from __future__ import annotations


def resolve_attention_implementation(cfg: dict, specific_key: str) -> str | None:
    models = cfg.get("models", {}) or {}
    value = models.get(specific_key)
    text = str(value or "").strip()
    if not text or text.lower() in {"auto", "default", "none", "false", "off"}:
        return None
    return text


def attention_kwargs(cfg: dict, specific_key: str) -> dict:
    implementation = resolve_attention_implementation(cfg, specific_key)
    return {"attn_implementation": implementation} if implementation else {}
