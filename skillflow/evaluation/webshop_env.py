from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from skillflow.evaluation.alfworld_env import BatchEnvironmentInterface
from skillflow.evaluation.http_json import post_json


class WebShopHTTPBatchInterface(BatchEnvironmentInterface):
    """Thin client for the isolated Python-3.10 WebShop bridge."""

    def __init__(
        self,
        tasks: Sequence[dict[str, Any]],
        *,
        bridge_url: str,
        seed: int = 0,
        timeout_seconds: float = 600,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        expected_runtime_identity: Mapping[str, Any] | None = None,
    ):
        if not tasks:
            raise ValueError("WebShop requires at least one task")
        self.tasks = [dict(task) for task in tasks]
        self.batch_size = len(self.tasks)
        self.bridge_url = str(bridge_url).strip()
        if not self.bridge_url and transport is None:
            raise ValueError("WebShop bridge_url is required")
        self.seed = int(seed)
        self.timeout_seconds = float(timeout_seconds)
        self._transport = transport
        self._closed = False
        self.expected_runtime_identity = dict(expected_runtime_identity or {})
        self.runtime_identity: dict[str, Any] | None = None

    def _validate_runtime_identity(
        self,
        infos: Sequence[dict[str, Any]],
    ) -> None:
        identities: list[dict[str, Any]] = []
        for index, info in enumerate(infos):
            raw_identity = info.get("runtime_identity")
            if not isinstance(raw_identity, Mapping):
                if self.expected_runtime_identity:
                    raise RuntimeError(
                        "WebShop bridge response is missing runtime_identity "
                        f"for batch row {index}"
                    )
                continue
            identity = dict(raw_identity)
            missing = [
                key for key in self.expected_runtime_identity if key not in identity
            ]
            mismatches = {
                key: {"expected": expected, "actual": identity.get(key)}
                for key, expected in self.expected_runtime_identity.items()
                if key in identity and identity[key] != expected
            }
            if missing or mismatches:
                raise RuntimeError(
                    "WebShop runtime identity mismatch before rollout: "
                    f"missing={missing} mismatches={mismatches}"
                )
            identities.append(identity)

        if identities and len(identities) != len(infos):
            raise RuntimeError(
                "WebShop bridge returned runtime_identity for only part of the batch"
            )
        if identities and any(identity != identities[0] for identity in identities[1:]):
            raise RuntimeError(
                "WebShop bridge returned inconsistent runtime_identity values"
            )
        if identities:
            identity = identities[0]
            if self.runtime_identity is not None and identity != self.runtime_identity:
                raise RuntimeError(
                    "WebShop runtime_identity changed within one client session"
                )
            self.runtime_identity = identity

    def _call(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            response = self._transport(endpoint, payload)
        else:
            response = post_json(
                self.bridge_url,
                endpoint,
                payload,
                timeout=self.timeout_seconds,
            )
        if not isinstance(response, dict):
            raise RuntimeError("WebShop bridge response must be a mapping")
        return response

    def _decode_batch(
        self,
        response: dict[str, Any],
        *,
        include_step_fields: bool,
    ):
        observations = response.get("observations")
        infos = response.get("infos")
        if not isinstance(observations, list) or not isinstance(infos, list):
            raise RuntimeError("WebShop bridge response lacks observations/infos lists")
        if len(observations) != self.batch_size or len(infos) != self.batch_size:
            raise RuntimeError(
                "WebShop bridge batch-size mismatch: "
                f"expected={self.batch_size} observations={len(observations)} infos={len(infos)}"
            )
        obs = [str(item) for item in observations]
        normalized_infos = [dict(item or {}) for item in infos]
        self._validate_runtime_identity(normalized_infos)
        if not include_step_fields:
            return obs, normalized_infos
        rewards = response.get("rewards")
        dones = response.get("dones")
        if not isinstance(rewards, list) or not isinstance(dones, list):
            raise RuntimeError("WebShop step response lacks rewards/dones lists")
        if len(rewards) != self.batch_size or len(dones) != self.batch_size:
            raise RuntimeError("WebShop bridge returned a partial step batch")
        return (
            obs,
            [float(item) for item in rewards],
            [bool(item) for item in dones],
            normalized_infos,
        )

    def reset(self) -> tuple[list[str], list[dict[str, Any]]]:
        response = self._call(
            "/v1/reset",
            {
                "sessions": [int(task["session"]) for task in self.tasks],
                "seed": self.seed,
            },
        )
        return self._decode_batch(response, include_step_fields=False)

    def step(
        self,
        actions: list[str],
    ) -> tuple[list[str], list[float], list[bool], list[dict[str, Any]]]:
        if len(actions) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} WebShop actions, got {len(actions)}")
        response = self._call("/v1/step", {"actions": list(actions)})
        return self._decode_batch(response, include_step_fields=True)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._call("/v1/close", {})
        finally:
            self._closed = True


__all__ = ["WebShopHTTPBatchInterface"]
