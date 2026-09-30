from __future__ import annotations

import contextlib
import os
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence

import yaml


class BatchEnvironmentInterface(ABC):
    @abstractmethod
    def reset(self) -> tuple[list[str], list[dict[str, Any]]]: ...

    @abstractmethod
    def step(self, actions: list[str]) -> tuple[list[str], list[float], list[bool], list[dict[str, Any]]]: ...

    def close(self) -> None:
        pass


@contextlib.contextmanager
def _maybe_silence(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _expand_envvars(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_envvars(x) for x in value]
    if isinstance(value, dict):
        return {k: _expand_envvars(v) for k, v in value.items()}
    return value


def _resolve_config_path(configured_path: str) -> Path:
    if not configured_path or configured_path == "auto":
        raise ValueError("split.alfworld_config must be an explicit existing path")
    path = Path(configured_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"ALFWorld config does not exist: {path}")
    return path


def _resolve_environment_class(environment_module, env_type: str):
    resolver = getattr(environment_module, "get_environment", None)
    if callable(resolver):
        return resolver(env_type)
    module_names = {
        "AlfredTWEnv": "alfred_tw_env",
        "AlfredThorEnv": "alfred_thor_env",
        "AlfredHybrid": "alfred_hybrid",
    }
    module_name = module_names.get(env_type)
    if module_name is None:
        raise ValueError(f"Unsupported ALFWorld environment type: {env_type}")
    module = import_module(f"alfworld.agents.environment.{module_name}")
    try:
        return getattr(module, env_type)
    except AttributeError as exc:
        raise RuntimeError(f"Installed ALFWorld does not provide {env_type} in {module.__name__}") from exc


def _as_list(value, batch_size: int) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        if hasattr(value, "tolist"):
            out = value.tolist()
            if isinstance(out, list):
                return out
    except Exception:
        pass
    return [value for _ in range(batch_size)]


def _infos_to_list(infos: Any, batch_size: int) -> list[dict[str, Any]]:
    if isinstance(infos, (list, tuple)):
        rows = []
        for item in list(infos)[:batch_size]:
            rows.append(dict(item or {}) if isinstance(item, dict) else {"value": item})
        while len(rows) < batch_size:
            rows.append({})
        return rows
    rows = [dict() for _ in range(batch_size)]
    for key, value in (infos or {}).items():
        vals = _as_list(value, batch_size)
        if len(vals) != batch_size:
            vals = (vals + [vals[-1] if vals else None] * batch_size)[:batch_size]
        for i in range(batch_size):
            rows[i][key] = vals[i]
    return rows


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return _scalar_text(value[0] if value else "")
    return str(value).strip()


def _normalize_game_file(value: Any) -> str:
    text = _scalar_text(value)
    if not text:
        return ""
    return str(Path(text).expanduser().resolve(strict=False))


def _extract_game_file(info: dict[str, Any]) -> str:
    for key in ("extra.gamefile", "gamefile", "game_file", "extra_gamefile"):
        value = _scalar_text((info or {}).get(key))
        if value:
            return value
    extra = (info or {}).get("extra")
    if isinstance(extra, dict):
        for key in ("gamefile", "game_file"):
            value = _scalar_text(extra.get(key))
            if value:
                return value
    return ""


def _construct_env_with_pinned_games(
    env_class,
    config: dict[str, Any],
    game_paths: Sequence[Path],
    train_eval: str,
    env_type: str,
):
    """Construct ALFWorld without its redundant full-dataset discovery pass.

    ``AlfredTWEnv.__init__`` calls ``collect_game_files`` before callers get a
    chance to replace ``game_files``.  During manifest preparation that means
    walking and parsing an entire split once per small goal-resolution batch.
    TextWorld only needs the already resolved game files, so override discovery
    for this one instance while preserving ALFWorld's normal constructor and
    ``init_env`` implementation.

    The optimization is deliberately limited to ``AlfredTWEnv``.  Embodied or
    future environment types may require additional constructor side effects
    and therefore retain the conservative upstream behavior.
    """

    pinned = [str(path) for path in game_paths]
    collect = getattr(env_class, "collect_game_files", None)
    if env_type != "AlfredTWEnv" or not callable(collect):
        return env_class(config, train_eval=train_eval)

    collect_called = False

    class _PinnedAlfredTWEnv(env_class):
        def collect_game_files(self, *args, **kwargs):  # type: ignore[override]
            nonlocal collect_called
            collect_called = True
            self.game_files = list(pinned)
            self.num_games = len(pinned)

    base = _PinnedAlfredTWEnv(config, train_eval=train_eval)
    if not collect_called:
        raise RuntimeError(
            "Installed AlfredTWEnv no longer invokes collect_game_files during "
            "construction; refusing an unverified manifest fast path."
        )
    return base


def _make_base_env(config_path: str, game_files: Sequence[str], seed: int, train_eval: str):
    if not game_files:
        raise ValueError("ALFWorld rollout requires at least one game_file")
    game_paths = [Path(g) for g in game_files]
    missing = [str(g) for g in game_paths if not g.exists()]
    if missing:
        raise FileNotFoundError(f"ALFWorld game_file does not exist: {missing[0]}")
    try:
        import alfworld.agents.environment as environment
    except ImportError as e:
        raise RuntimeError("Install ALFWorld with `pip install alfworld && alfworld-download`") from e

    config_file = _resolve_config_path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        config = _expand_envvars(yaml.safe_load(f))
    config.setdefault("general", {})["random_seed"] = seed
    config.setdefault("env", {})["domain_randomization"] = False
    config.setdefault("logging", {})["quiet"] = True
    dataset = config.setdefault("dataset", {})
    # Non-TextWorld implementations may still inspect a dataset root during
    # construction.  The AlfredTWEnv fast path below skips discovery and pins
    # these exact files before its constructor would scan the common ancestor.
    common_root = Path(os.path.commonpath([str(g.parent) for g in game_paths]))
    for name in ("data_path", "eval_id_data_path", "eval_ood_data_path"):
        dataset[name] = str(common_root)
    env_type = config["env"]["type"]
    env_class = _resolve_environment_class(environment, env_type)
    quiet = bool(config.get("logging", {}).get("quiet", True))
    with _maybe_silence(quiet):
        base = _construct_env_with_pinned_games(
            env_class,
            config,
            game_paths,
            train_eval,
            env_type,
        )
    assigned = False
    for name in ("game_files", "gamefiles"):
        if hasattr(base, name):
            setattr(base, name, [str(g) for g in game_paths])
            assigned = True
    if not assigned:
        raise RuntimeError(
            f"ALFWorld {type(base).__name__} exposes neither game_files nor gamefiles; cannot pin tasks"
        )
    if hasattr(base, "num_games"):
        base.num_games = len(game_paths)
    return base, quiet


class ALFWorldBatchInterface(BatchEnvironmentInterface):
    """Batched ALFWorld adapter. One init_env call serves multiple pinned game files."""

    def __init__(
        self,
        config_path: str,
        game_files: Sequence[str],
        seed: int = 0,
        train_eval: str = "eval_in_distribution",
    ):
        self.batch_size = len(game_files)
        self.game_files = [str(Path(g).expanduser().resolve(strict=False)) for g in game_files]
        self._user_to_env = list(range(self.batch_size))
        self._env_to_user = list(range(self.batch_size))
        base, quiet = _make_base_env(config_path, game_files, seed, train_eval)
        with _maybe_silence(quiet):
            self.env = base.init_env(batch_size=self.batch_size)
        self.last_infos: list[dict[str, Any]] = []

    def _build_slot_mapping(self, info_list: list[dict[str, Any]]) -> None:
        if len(info_list) != self.batch_size:
            raise RuntimeError(f"ALFWorld reset returned {len(info_list)} infos for batch_size={self.batch_size}")
        actual = [_normalize_game_file(_extract_game_file(info)) for info in info_list]
        if not all(actual):
            raise RuntimeError(
                "ALFWorld reset did not expose per-slot game_file metadata; "
                "cannot verify manifest/env alignment."
            )
        by_game: dict[str, deque[int]] = defaultdict(deque)
        for env_idx, game_file in enumerate(actual):
            by_game[game_file].append(env_idx)

        user_to_env: list[int] = []
        missing = []
        for user_idx, expected in enumerate(self.game_files):
            env_indices = by_game.get(_normalize_game_file(expected))
            if not env_indices:
                missing.append((user_idx, expected))
                continue
            user_to_env.append(env_indices.popleft())
        if missing:
            observed = sorted(set(actual))
            detail = "; ".join(f"user_slot={idx} expected={path}" for idx, path in missing[:5])
            raise RuntimeError(
                "ALFWorld reset game_file mismatch between manifest and environment. "
                f"Missing expected slots: {detail}. Observed env games: {observed[:8]}"
            )

        env_to_user = [-1 for _ in range(self.batch_size)]
        for user_idx, env_idx in enumerate(user_to_env):
            env_to_user[env_idx] = user_idx
        if any(idx < 0 for idx in env_to_user):
            raise RuntimeError(f"ALFWorld reset returned unmapped env slots: {env_to_user}")
        self._user_to_env = user_to_env
        self._env_to_user = env_to_user

    def _to_user_order(self, values: list[Any]) -> list[Any]:
        return [values[self._user_to_env[user_idx]] for user_idx in range(self.batch_size)]

    def _annotate_infos(self, info_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        annotated = []
        for user_idx, info in enumerate(info_list):
            env_idx = self._user_to_env[user_idx]
            row = dict(info or {})
            row["requested_game_file"] = self.game_files[user_idx]
            row["env_game_file"] = _extract_game_file(info)
            row["env_slot"] = env_idx
            row["user_slot"] = user_idx
            annotated.append(row)
        return annotated

    def reset(self) -> tuple[list[str], list[dict[str, Any]]]:
        obs, infos = self.env.reset()
        obs_list = [str(x) for x in _as_list(obs, self.batch_size)]
        info_list = _infos_to_list(infos, self.batch_size)
        self._build_slot_mapping(info_list)
        obs_user = self._to_user_order(obs_list)
        info_user = self._annotate_infos(self._to_user_order(info_list))
        self.last_infos = info_user
        return obs_user, info_user

    def step(self, actions: list[str]) -> tuple[list[str], list[float], list[bool], list[dict[str, Any]]]:
        if len(actions) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} actions, got {len(actions)}")
        env_actions = ["" for _ in range(self.batch_size)]
        for user_idx, action in enumerate(actions):
            env_actions[self._user_to_env[user_idx]] = action
        obs, scores, dones, infos = self.env.step(env_actions)
        obs_list = [str(x) for x in _as_list(obs, self.batch_size)]
        scores_list = [float(x) for x in _as_list(scores, self.batch_size)]
        dones_list = [bool(x) for x in _as_list(dones, self.batch_size)]
        info_list = _infos_to_list(infos, self.batch_size)
        info_user = self._annotate_infos(self._to_user_order(info_list))
        self.last_infos = info_user
        return self._to_user_order(obs_list), self._to_user_order(scores_list), self._to_user_order(dones_list), info_user

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close:
            close()


def make_batch_env(cfg: dict, tasks: Sequence[dict], seed: int) -> BatchEnvironmentInterface:
    return ALFWorldBatchInterface(
        cfg["split"]["alfworld_config"],
        [str(t.get("game_file", "")) for t in tasks],
        seed,
    )
