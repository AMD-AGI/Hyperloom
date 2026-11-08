# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The writes to the server argument string, and the argv they yield."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import BenchmarkPoint, read_benchmark_point
from hyperloom.inference_optimizer.framework_registry import expert_parallel_flag, server_args_env_name

from hyperloom.inference_optimizer.grid_server_args import merge_server_args
from hyperloom.inference_optimizer.grid_server_args import validate_server_args_shell_safe
from inference_suite.argv import tokenize_server_args_preserving_json

from ._multi_node_env import is_multi_node

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerArgv:
    """The final server argument list for one benchmark configuration.

    Attributes:
        framework: Framework the argv belongs to, lower-cased.
        text: The sealed argument string, exactly as the YAML now carries it.
        argv: ``text`` split into argv tokens.
        tokenized: False when the string could not be split without moving a
            token boundary; ``argv`` is then empty.
    """

    framework: str
    text: str
    argv: tuple[str, ...]
    tokenized: bool


def _sealed(framework: str | None, text: str) -> ServerArgv:
    """Build a :class:`ServerArgv` from an already-guarded argument string."""
    name = (framework or "").strip().lower()
    split = tokenize_server_args_preserving_json(text)
    if split is None:
        return ServerArgv(framework=name, text=text, argv=(), tokenized=False)
    normalized, tokens = split
    return ServerArgv(framework=name, text=normalized, argv=tuple(tokens), tokenized=True)


def add_server_arg_unless_pinned(
    envs: MutableMapping[str, Any],
    framework: str | None,
    arg: str,
    *,
    pinned_by: Sequence[str],
) -> None:
    """Merge ``arg`` into the composed argument string unless it is already pinned.

    Args:
        envs: The benchmark env mapping being materialised.
        framework: The framework the config serves.
        arg: The whole flag and its value, e.g. ``--block-size 128``.
        pinned_by: Flag aliases whose whole-token presence (``--flag v`` or
            ``--flag=v``) in the current string means the caller already pinned
            this flag, e.g. ``("--block-size", "--block_size")``.
    """
    env_name = server_args_env_name(framework)
    existing = str(envs.get(env_name, "")).strip()
    split = tokenize_server_args_preserving_json(existing)
    if _pinned_values(split[1] if split else [], pinned_by):
        return
    envs[env_name] = merge_server_args(existing, arg)


def _pinned_values(tokens: Sequence[str], aliases: Sequence[str]) -> list[str | None]:
    """Return the value of every whole-token occurrence of ``aliases``; ``None`` where one has none."""
    values: list[str | None] = []
    for index, token in enumerate(tokens):
        name, assigned, value = token.partition("=")
        if name not in aliases:
            continue
        if assigned:
            values.append(value)
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
            values.append(tokens[index + 1])
        else:
            values.append(None)
    return values


def _expert_parallel_arg(text: str, framework: str | None, envs: Mapping[str, Any]) -> str:
    """Return the flag ``text`` still owes the server to run at ``envs["EP"]``; empty when it owes none.

    The flag shards experts across the ``envs["TP"]`` ranks this config launches with, so an EP that does not divide
    them gets none. A multi-node server is launched by the multi-node launcher, which passes EP itself.

    Raises:
        ValueError: When ``text`` pins an expert-parallel size other than ``envs["EP"]``.
    """
    size = int(envs.get("EP") or 1)
    row = expert_parallel_flag(framework)
    if row is None or size <= 1 or is_multi_node():
        return ""
    flag, aliases, sized = row
    split = tokenize_server_args_preserving_json(text)
    pinned = _pinned_values(split[1] if split else [], aliases)
    if sized and any(value is None or not value.isdigit() or int(value) != size for value in pinned):
        raise ValueError(
            f"EP={size} but the server args already pin an expert-parallel size of {pinned} "
            f"({' / '.join(aliases)}); drop the pin or pass --ep to match it"
        )
    if pinned:
        return ""
    tp = int(envs.get("TP") or size)
    if tp % size:
        log.warning(
            "EP=%d does not divide the launched TP=%d (after any clamp to the visible GPUs); "
            "launching without expert parallelism.",
            size,
            tp,
        )
        return ""
    return f"{flag} {size}" if sized else flag


def seal_server_argv(
    envs: MutableMapping[str, Any],
    framework: str | None,
) -> ServerArgv:
    """Write the final server argument string into ``envs`` and return its argv.

    The run's ``EP`` env is honoured here, below every composer, so no
    replacement of the argument string can drop the expert-parallel flag.

    Args:
        envs: The benchmark env mapping being materialised.
        framework: The framework the config serves.

    Returns:
        ServerArgv: The sealed argv.

    Raises:
        ValueError: When the composed string carries shell control syntax, or
            pins an expert-parallel size that contradicts ``envs["EP"]``.
    """
    env_name = server_args_env_name(framework)
    text = validate_server_args_shell_safe(str(envs.get(env_name) or ""))
    text = merge_server_args(text, _expert_parallel_arg(text, framework, envs))
    sealed = _sealed(framework, text)
    if sealed.text or env_name in envs:
        envs[env_name] = sealed.text
    return sealed


class ConfigUnreadable(Exception):
    """A materialised benchmark YAML that cannot be read as a config."""


def config_point(config_path: str | Path) -> BenchmarkPoint:
    """Read the point a materialised benchmark YAML holds.

    Raises:
        ConfigUnreadable: When the file cannot be opened, is not UTF-8, is not
            YAML a safe loader can construct, or holds no ``benchmark`` mapping.
    """
    try:
        return read_benchmark_point(Path(config_path))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigUnreadable(f"{config_path}: {exc}") from exc


def config_server_argv(config_path: str | Path) -> ServerArgv:
    """Return the sealed argv a materialised config will launch with.

    Args:
        config_path: Path to the materialised YAML.

    Returns:
        ServerArgv: The argv carried by the config's benchmark envs.

    Raises:
        ConfigUnreadable: When the config cannot be read.
    """
    point = config_point(config_path)
    return _sealed(point.framework, str(point.envs.get(server_args_env_name(point.framework)) or "").strip())


def config_launch_env(config_path: str | Path, base: Mapping[str, str]) -> dict[str, str]:
    """Return the environment the server will be launched into.

    Args:
        config_path: Path to the materialised YAML.
        base: The environment the benchmark subprocess is handed.

    Returns:
        dict[str, str]: ``base`` overlaid with the config's benchmark envs.

    Raises:
        ConfigUnreadable: When the config cannot be read.
    """
    merged = {str(key): str(value) for key, value in base.items()}
    merged.update({str(key): str(value) for key, value in config_point(config_path).envs.items() if value is not None})
    return merged


__all__ = [
    "ConfigUnreadable",
    "ServerArgv",
    "add_server_arg_unless_pinned",
    "config_launch_env",
    "config_point",
    "config_server_argv",
    "seal_server_argv",
]
