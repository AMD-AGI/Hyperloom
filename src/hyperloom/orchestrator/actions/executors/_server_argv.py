# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The writes to the server argument string, and the argv they yield.

:func:`add_server_arg_unless_pinned` is how a composer adds to the string: one
merge that honours whatever the caller already pinned, so a per-model injection
cannot silently overwrite an operator's choice. :func:`seal_server_argv` ends
the composition -- it normalises the string, applies the shell guard, tokenises
it into the argv the framework's parser will receive, and writes the canonical
form back. Consumers downstream of materialisation read a :class:`ServerArgv`
rather than the raw env value.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hyperloom.inference_optimizer.framework_registry import server_args_env_name

from ._grid_server_args import merge_server_args
from ._grid_server_args import tokenize_server_args_preserving_json
from ._grid_server_args import validate_server_args_shell_safe


@dataclass(frozen=True)
class ServerArgv:
    """The final server argument list for one benchmark configuration.

    Attributes:
        framework: Framework the argv belongs to, lower-cased.
        env_name: Benchmark env the string is transported in.
        text: The sealed argument string, exactly as the YAML now carries it.
        argv: ``text`` split into argv tokens.
        tokenized: False when the string could not be split without moving a
            token boundary; ``argv`` is then empty.
    """

    framework: str
    env_name: str
    text: str
    argv: tuple[str, ...]
    tokenized: bool

    @property
    def digest(self) -> str:
        """str: Identity of this argv -- the framework and the tokens, nothing
        per-run -- for keying a one-shot repair."""
        body = "\x1f".join((self.framework, *self.argv))
        return hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()


def _sealed(framework: str | None, env_name: str, text: str) -> ServerArgv:
    """Build a :class:`ServerArgv` from an already-guarded argument string."""
    name = (framework or "").strip().lower()
    split = tokenize_server_args_preserving_json(text)
    if split is None:
        return ServerArgv(framework=name, env_name=env_name, text=text, argv=(), tokenized=False)
    normalized, tokens = split
    return ServerArgv(framework=name, env_name=env_name, text=normalized, argv=tuple(tokens), tokenized=True)


def add_server_arg_unless_pinned(
    envs: MutableMapping[str, Any],
    framework: str | None,
    arg: str,
    *,
    pinned_by: Sequence[str],
) -> bool:
    """Merge ``arg`` into the composed argument string unless it is already pinned.

    Every injection reaching the argument env goes through here, so a model- or
    host-specific flag is added the one way: merged onto the left of what is
    there (last wins), and skipped outright when the caller already chose.

    Args:
        envs: The benchmark env mapping being materialised.
        framework: The framework the config serves.
        arg: The whole flag and its value, e.g. ``--block-size 128``.
        pinned_by: Spellings whose presence in the current string means the
            caller already pinned this flag, e.g. ``("block-size",
            "block_size")``.

    Returns:
        bool: Whether ``arg`` was added.
    """
    env_name = server_args_env_name(framework)
    existing = str(envs.get(env_name, "")).strip()
    if any(spelling in existing for spelling in pinned_by):
        return False
    envs[env_name] = merge_server_args(existing, arg)
    return True


def seal_server_argv(envs: MutableMapping[str, Any], framework: str | None) -> ServerArgv:
    """Write the final server argument string into ``envs`` and return its argv.

    Must be the last statement that touches the argument env.

    Args:
        envs: The benchmark env mapping being materialised.
        framework: The framework the config serves.

    Returns:
        ServerArgv: The sealed argv.

    Raises:
        ValueError: When the composed string carries shell control syntax.
    """
    env_name = server_args_env_name(framework)
    text = validate_server_args_shell_safe(str(envs.get(env_name) or ""))
    sealed = _sealed(framework, env_name, text)
    if sealed.text:
        envs[env_name] = sealed.text
    else:
        envs.pop(env_name, None)
    return sealed


def _load_config(config_path: str | Path) -> dict[str, Any]:
    """Read a materialised benchmark YAML, empty when it is not a mapping."""
    with Path(config_path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg if isinstance(cfg, dict) else {}


def _benchmark_envs(config_path: str | Path) -> tuple[str | None, dict[str, Any]]:
    """Read a config's framework and benchmark envs; empty when absent."""
    bench = _load_config(config_path).get("benchmark")
    if not isinstance(bench, dict):
        return None, {}
    envs = bench.get("envs")
    return bench.get("framework"), envs if isinstance(envs, dict) else {}


def config_server_argv(config_path: str | Path) -> ServerArgv:
    """Return the sealed argv a materialised config will launch with.

    Args:
        config_path: Path to the materialised YAML.

    Returns:
        ServerArgv: The argv carried by the config's benchmark envs.
    """
    framework, envs = _benchmark_envs(config_path)
    env_name = server_args_env_name(framework)
    return _sealed(framework, env_name, str(envs.get(env_name) or "").strip())


def config_launch_env(config_path: str | Path, base: Mapping[str, str]) -> dict[str, str]:
    """Return the environment the server will be launched into.

    The rendered YAML's benchmark envs are exported around the server, so they
    -- not the orchestrator's own environment -- decide which interpreter and
    which installed packages it resolves.

    Args:
        config_path: Path to the materialised YAML.
        base: The environment the benchmark subprocess is handed.

    Returns:
        dict[str, str]: ``base`` overlaid with the config's benchmark envs.
    """
    _framework, envs = _benchmark_envs(config_path)
    merged = {str(key): str(value) for key, value in base.items()}
    merged.update({str(key): str(value) for key, value in envs.items() if value is not None})
    return merged


def reseal_config_argv(config_path: str | Path, text: str) -> ServerArgv:
    """Replace a materialised config's server argv, through the same seal.

    Args:
        config_path: Path to the materialised YAML, rewritten in place.
        text: The replacement argument string.

    Returns:
        ServerArgv: The re-sealed argv.

    Raises:
        ValueError: When the replacement carries shell control syntax.
    """
    path = Path(config_path)
    cfg = _load_config(path)
    bench = cfg.setdefault("benchmark", {})
    envs = bench.setdefault("envs", {})
    envs[server_args_env_name(bench.get("framework"))] = text.strip()
    sealed = seal_server_argv(envs, bench.get("framework"))
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return sealed


__all__ = [
    "ServerArgv",
    "add_server_arg_unless_pinned",
    "config_launch_env",
    "config_server_argv",
    "reseal_config_argv",
    "seal_server_argv",
]
