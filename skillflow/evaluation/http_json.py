from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


def post_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    max_response_bytes: int = 64 * 1024 * 1024,
) -> dict[str, Any]:
    url = (
        str(base_url)
        if endpoint == ""
        else urljoin(str(base_url).rstrip("/") + "/", endpoint.lstrip("/"))
    )
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(timeout)) as response:
            raw = response.read(max_response_bytes + 1)
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP service rejected {url}: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach local benchmark service {url}: {exc}") from exc
    if len(raw) > max_response_bytes:
        raise RuntimeError(f"Benchmark service response exceeded {max_response_bytes} bytes: {url}")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Benchmark service returned invalid JSON from {url}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Benchmark service response must be one object: {url}")
    return decoded


__all__ = ["post_json"]
