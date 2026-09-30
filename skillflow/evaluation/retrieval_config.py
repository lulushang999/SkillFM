from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from urllib.parse import ParseResult, urlparse
_LOCAL_RETRIEVAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
class QASummaryConfigError(ValueError):
    pass

def _required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise QASummaryConfigError(f"{label} is required")
    return text

def _local_retrieval_endpoint(
    value: object,
    *,
    label: str,
    expected_path: str,
) -> tuple[str, ParseResult, tuple[str, str, int]]:
    text = _required_text(value, label=label)
    parsed = urlparse(text)
    try:
        port = parsed.port
    except ValueError as exc:
        raise QASummaryConfigError(f"{label} has an invalid port") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOCAL_RETRIEVAL_HOSTS
        or port is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise QASummaryConfigError(
            f"{label} must be a local HTTP endpoint with an explicit port"
        )
    if parsed.path != expected_path or parsed.params or parsed.query or parsed.fragment:
        raise QASummaryConfigError(
            f"{label} path must be exactly {expected_path} with no parameters, "
            "query, or fragment"
        )
    return text, parsed, (parsed.scheme, parsed.hostname, port)


def _validate_retrieval_endpoints(retrieval: Mapping[str, Any]) -> tuple[str, str]:
    url, _, origin = _local_retrieval_endpoint(
        retrieval.get("url"),
        label="retrieval.url",
        expected_path="/retrieve",
    )
    batch_url, _, batch_origin = _local_retrieval_endpoint(
        retrieval.get("batch_url"),
        label="retrieval.batch_url",
        expected_path="/retrieve_batch",
    )
    if batch_origin != origin:
        raise QASummaryConfigError(
            "retrieval.url and retrieval.batch_url must use the same local origin"
        )
    return url, batch_url

