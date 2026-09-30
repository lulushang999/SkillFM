#!/usr/bin/env python3
"""Serve the official WebShop text environment from an isolated Python 3.10 env.

This process intentionally uses only a localhost JSON API. The main SkillFlow
environment therefore does not need WebShop's pinned Gym, Java, Flask, requests,
or pyserini dependency versions.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

def _configure_isolated_java() -> None:
    prefix = Path(sys.prefix).resolve()
    if (prefix / "bin" / "java").is_file():
        os.environ["JAVA_HOME"] = str(prefix)
        os.environ["PATH"] = f"{prefix / 'bin'}:{os.environ.get('PATH', '')}"


_configure_isolated_java()



BRIDGE_SCHEMA_VERSION = "skillflow_webshop_bridge_v1"


class WebShopRuntime:
    def __init__(
        self,
        *,
        webshop_root: Path,
        product_set: str,
        file_path: Path,
        attr_path: Path,
        num_products: int | None,
        human_goals: bool = False,
        show_attrs: bool = False,
        source_revision: str = "8e66726ed866a4e0a7f053586a41022798192e6c",
        index_path: Path | None = None,
        index_manifest: dict[str, Any] | None = None,
    ):
        sys.path.insert(0, str(webshop_root))
        try:
            module = importlib.import_module(
                "web_agent_site.envs.web_agent_text_env"
            )
            WebAgentTextEnv = module.WebAgentTextEnv
        except Exception as exc:
            raise RuntimeError(
                f"Cannot import official WebShop from {webshop_root}: {exc}"
            ) from exc
        if index_path is not None:
            resolved_index = index_path.expanduser().resolve()

            def load_bound_index(num_products: int | None = None):
                from pyserini.search.lucene import LuceneSearcher

                return LuceneSearcher(str(resolved_index))

            # Bind SimServer to the manifest-validated index without modifying
            # the pinned upstream checkout.
            module.init_search_engine = load_bound_index
        self._environment_class = WebAgentTextEnv
        if product_set not in {"small", "all"}:
            raise ValueError(f"Unsupported WebShop product set: {product_set!r}")
        self.product_set = product_set
        self.file_path = file_path
        self.attr_path = attr_path
        self.num_products = num_products
        self.human_goals = bool(human_goals)
        self.show_attrs = bool(show_attrs)
        self.source_revision = str(source_revision)
        self.index_path = index_path.expanduser().resolve() if index_path else None
        self.index_manifest = dict(index_manifest or {})
        self.envs: list[Any] = []
        self.sessions: list[int] = []
        self.shared_server: Any | None = None
        self.server_seed: int | None = None
        self.task_descriptions: list[str] = []
        self.dones: list[bool] = []
        self.last_observations: list[str] = []
        self.last_infos: list[dict[str, Any]] = []
        self.last_rewards: list[float] = []
        self._lock = threading.Lock()

    def runtime_identity(self) -> dict[str, Any]:
        """Return the immutable data/protocol identity plus the pinned server seed."""

        return {
            "product_set": self.product_set,
            "product_filename": self.file_path.name,
            "product_file_size_bytes": self.file_path.stat().st_size,
            "attr_filename": self.attr_path.name,
            "attr_file_size_bytes": self.attr_path.stat().st_size,
            "num_products": self.num_products,
            "human_goals": self.human_goals,
            "show_attrs": self.show_attrs,
            "source_revision": self.source_revision,
            "server_seed": self.server_seed,
            "product_sha256": (self.index_manifest.get("product_file") or {}).get(
                "sha256"
            ),
            "attribute_sha256": (
                self.index_manifest.get("attribute_file") or {}
            ).get("sha256"),
            "human_instruction_sha256": (
                self.index_manifest.get("human_instruction_file") or {}
            ).get("sha256"),
            "index_schema_version": self.index_manifest.get("schema_version"),
            "index_manifest_sha256": self.index_manifest.get("manifest_sha256"),
            "index_document_count": self.index_manifest.get("document_count"),
        }

    @staticmethod
    def _admissible(available: dict[str, Any]) -> list[str]:
        actions: list[str] = []
        if bool(available.get("has_search_bar")):
            actions.append("search[<your query>]")
        actions.extend(
            f"click[{value}]" for value in (available.get("clickables") or [])
        )
        return actions

    def _close_unlocked(self) -> None:
        for env in self.envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        self.envs = []
        self.sessions = []
        self.task_descriptions = []
        self.dones = []
        self.last_observations = []
        self.last_infos = []
        self.last_rewards = []

    def reset(self, sessions: list[int], seed: int) -> dict[str, Any]:
        if not sessions:
            raise ValueError("sessions must be a non-empty list")
        if len(set(sessions)) != len(sessions) or any(item < 0 for item in sessions):
            raise ValueError("sessions must contain unique non-negative integers")
        with self._lock:
            requested_seed = int(seed)
            if self.server_seed is not None and requested_seed != self.server_seed:
                raise ValueError(
                    "WebShop bridge seed cannot change after SimServer initialization: "
                    f"expected={self.server_seed} actual={requested_seed}"
                )
            self._close_unlocked()
            self.sessions = [int(session) for session in sessions]
            for offset in range(len(sessions)):
                if self.shared_server is None:
                    import random
                    import numpy as np
                    import torch
                    random.seed(requested_seed)
                    np.random.seed(requested_seed)
                    torch.manual_seed(requested_seed)
                kwargs: dict[str, Any] = {
                    "observation_mode": "text",
                    "file_path": str(self.file_path),
                    "attr_path": str(self.attr_path),
                    "seed": requested_seed + offset,
                    "num_products": self.num_products,
                    "human_goals": self.human_goals,
                    "show_attrs": self.show_attrs,
                }
                if self.shared_server is not None:
                    kwargs["server"] = self.shared_server
                env = self._environment_class(**kwargs)
                if self.shared_server is None:
                    self.shared_server = env.server
                    self.server_seed = requested_seed
                self.envs.append(env)

            observations: list[str] = []
            infos: list[dict[str, Any]] = []
            self.task_descriptions = []
            for index, (env, session) in enumerate(zip(self.envs, sessions)):
                observation, _ = env.reset(session=int(session))
                goal = str(env.instruction_text).strip()
                available = dict(env.get_available_actions() or {})
                info = {
                    "schema_version": BRIDGE_SCHEMA_VERSION,
                    "runtime_identity": self.runtime_identity(),
                    "session": int(session),
                    "task_description": goal,
                    "available_actions": available,
                    "admissible_commands": self._admissible(available),
                    "won": False,
                    "task_score": 0.0,
                    "user_slot": index,
                }
                observations.append(str(observation))
                infos.append(info)
                self.task_descriptions.append(goal)
            self.dones = [False] * len(self.envs)
            self.last_observations = list(observations)
            self.last_infos = [dict(info) for info in infos]
            self.last_rewards = [0.0] * len(self.envs)
            return {"observations": observations, "infos": infos}

    def step(self, actions: list[str]) -> dict[str, Any]:
        with self._lock:
            if not self.envs:
                raise RuntimeError("Reset the WebShop bridge before stepping")
            if len(actions) != len(self.envs):
                raise ValueError(
                    f"Expected {len(self.envs)} actions, received {len(actions)}"
                )
            observations: list[str] = []
            rewards: list[float] = []
            dones: list[bool] = []
            infos: list[dict[str, Any]] = []
            for index, (env, action) in enumerate(zip(self.envs, actions)):
                if self.dones[index]:
                    observations.append(self.last_observations[index])
                    rewards.append(self.last_rewards[index])
                    dones.append(True)
                    infos.append(dict(self.last_infos[index]))
                    continue
                observation, reward, done, _ = env.step(str(action))
                score = float(reward)
                won = bool(done and score == 1.0)
                available = {} if done else dict(env.get_available_actions() or {})
                info = {
                    "schema_version": BRIDGE_SCHEMA_VERSION,
                    "runtime_identity": self.runtime_identity(),
                    # WebAgentTextEnv automatically resets after terminal steps,
                    # so env.session may already identify a different task here.
                    "session": self.sessions[index],
                    "task_description": self.task_descriptions[index],
                    "available_actions": available,
                    "admissible_commands": self._admissible(available),
                    "won": won,
                    "task_score": score,
                    "user_slot": index,
                }
                observations.append(str(observation))
                rewards.append(score)
                dones.append(bool(done))
                infos.append(info)
                self.dones[index] = bool(done)
                self.last_observations[index] = str(observation)
                self.last_rewards[index] = score
                self.last_infos[index] = dict(info)
            return {
                "observations": observations,
                "rewards": rewards,
                "dones": dones,
                "infos": infos,
            }

    def close(self) -> dict[str, Any]:
        with self._lock:
            self._close_unlocked()
        return {"closed": True}


class BridgeHandler(BaseHTTPRequestHandler):
    runtime: WebShopRuntime
    protocol_version = "HTTP/1.1"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/health":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._send(
            HTTPStatus.OK,
            {
                "ok": True,
                "schema_version": BRIDGE_SCHEMA_VERSION,
                "active_environments": len(self.runtime.envs),
                "runtime_identity": self.runtime.runtime_identity(),
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 8 * 1024 * 1024:
                raise ValueError("invalid request body size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request JSON must be one object")
            if self.path == "/v1/reset":
                sessions = payload.get("sessions")
                if not isinstance(sessions, list):
                    raise ValueError("sessions must be a list")
                response = self.runtime.reset(
                    [int(item) for item in sessions],
                    int(payload.get("seed", 0)),
                )
            elif self.path == "/v1/step":
                actions = payload.get("actions")
                if not isinstance(actions, list):
                    raise ValueError("actions must be a list")
                response = self.runtime.step([str(item) for item in actions])
            elif self.path == "/v1/close":
                response = self.runtime.close()
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send(HTTPStatus.OK, response)
        except Exception as exc:
            self._send(
                HTTPStatus.BAD_REQUEST,
                {"error": str(exc), "schema_version": BRIDGE_SCHEMA_VERSION},
            )

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("[webshop-bridge] " + (fmt % args) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webshop-root", required=True)
    parser.add_argument("--index-path", required=True, help="Prepared Lucene index")
    parser.add_argument("--file-path", help="Defaults to data/items_shuffle.json under WebShop")
    parser.add_argument("--attr-path", help="Defaults to data/items_ins_v2.json under WebShop")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8040)
    args = parser.parse_args(argv)
    root = Path(args.webshop_root).expanduser().resolve()
    products = Path(args.file_path) if args.file_path else root / "data/items_shuffle.json"
    attributes = Path(args.attr_path) if args.attr_path else root / "data/items_ins_v2.json"
    index = Path(args.index_path).expanduser().resolve()
    if not products.is_file() or not attributes.is_file() or not index.is_dir():
        raise FileNotFoundError("WebShop product/attribute files and Lucene index are required")
    runtime = WebShopRuntime(webshop_root=root, product_set="all", file_path=products,
                             attr_path=attributes, num_products=None, index_path=index)
    BridgeHandler.runtime = runtime
    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    try:
        server.serve_forever()
    finally:
        runtime.close()
        server.server_close()


if __name__ == "__main__":
    main()
