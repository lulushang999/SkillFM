"""Canonical dataset-name normalization shared across component layers."""

from __future__ import annotations


_BENCHMARK_ALIASES = {
    "webshop": "webshop",
    "web_shop": "webshop",
    "search": "searchqa",
    "search_qa": "searchqa",
    "searchqa": "searchqa",
}


def normalize_benchmark_name(value: object) -> str:
    """Return the canonical name of one supported modular dataset."""

    text = str(value or "").strip().casefold().replace("-", "_")
    try:
        return _BENCHMARK_ALIASES[text]
    except KeyError as exc:
        supported = ", ".join(sorted(set(_BENCHMARK_ALIASES.values())))
        raise ValueError(
            f"Unsupported modular benchmark {value!r}; expected one of: {supported}"
        ) from exc


__all__ = ["normalize_benchmark_name"]
