# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a launch request only restates the baseline launch it would be appended to.

Explore and the prelude warm replay both append a request to the baseline's
server argv and envs. When every setting and env the request names already
holds that value in the baseline, the launch is the baseline's own, and
measuring it spends a whole benchmark round to reproduce the baseline number.
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.launch_log_evidence import launch_flag_setting_name
from hyperloom.inference_optimizer.grid_server_args import server_args_env_name

from ._server_argv import ConfigUnreadable, config_server_argv

log = logging.getLogger(__name__)


def config_envs(config_path: Path) -> dict[str, str]:
    """Read ``benchmark.envs`` out of a built config file as a ``str -> str`` dict.

    Drops the framework's server-args key (``EXTRA_SGLANG_ARGS`` and friends),
    which holds the launch argv rather than an env value, and a null value,
    which the launch leaves unset. Returns an empty dict when the file carries
    no ``benchmark.envs`` mapping.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    bench = cfg.get("benchmark") if isinstance(cfg, Mapping) else None
    envs = bench.get("envs") if isinstance(bench, Mapping) else None
    if not isinstance(envs, Mapping):
        return {}
    args_env = server_args_env_name(str(bench.get("framework") or ""))
    return {str(k): str(v) for k, v in envs.items() if str(k) != args_env and v is not None}


def named_settings(argv_text: str, framework: str) -> dict[str, list[str]]:
    """Map each setting ``argv_text`` names to its value tokens, last occurrence winning like the engine."""
    try:
        tokens = shlex.split(argv_text)
    except ValueError:
        return {}
    settings: dict[str, list[str]] = {}
    current = ""
    for token in tokens:
        if token.startswith("-"):
            name, separator, attached = token.partition("=")
            current = launch_flag_setting_name(name, framework)
            settings[current] = [attached] if separator else []
        elif current:
            settings[current].append(token)
    return settings


def _opposite_setting(name: str) -> str:
    """The setting a boolean flag's negated spelling sets: ``no_x`` for ``x`` and back."""
    return name[3:] if name.startswith("no_") else f"no_{name}"


def _names_only_flags(argv_text: str) -> bool:
    """Whether ``argv_text`` parses and opens on a flag, so ``named_settings`` accounts for every token."""
    try:
        tokens = shlex.split(argv_text)
    except ValueError:
        return False
    return not tokens or tokens[0].startswith("-")


def restates_the_baseline(
    argv_text: str,
    envs: Mapping[str, Any],
    base_argv: str,
    base_envs: Mapping[str, str],
    framework: str,
) -> bool:
    """Whether appending ``argv_text`` and ``envs`` to the baseline launch would change nothing.

    True only when every setting the request names is already in ``base_argv``
    with the same value tokens, and every env already carries that value in
    ``base_envs``. A request this cannot read in full -- an unparsable argv, a
    positional token, a flag whose negated spelling the baseline also names --
    is never a restatement, and neither is an empty one.
    """
    text = str(argv_text or "").strip()
    requested_envs = {str(k): str(v) for k, v in (envs or {}).items()}
    if (not text and not requested_envs) or not _names_only_flags(text):
        return False
    base_settings = named_settings(base_argv, framework)
    for name, values in named_settings(text, framework).items():
        if base_settings.get(name) != values or _opposite_setting(name) in base_settings:
            return False
    return all(base_envs.get(name) == value for name, value in requested_envs.items())


def config_launch_restated_by(config_path: str, argv_text: str, envs: Mapping[str, Any], framework: str) -> bool:
    """Whether appending ``argv_text`` and ``envs`` to the config at ``config_path`` would change nothing.

    False when the config is absent or unreadable: the launch to compare
    against is then unknown.
    """
    if not str(config_path or "").strip():
        return False
    try:
        base_argv = config_server_argv(config_path).text
        base_envs = config_envs(Path(config_path))
    except (ConfigUnreadable, OSError, ValueError, yaml.YAMLError) as exc:
        log.warning("baseline config %s unreadable for the restatement check (%s)", config_path, exc)
        return False
    return restates_the_baseline(argv_text, envs, base_argv, base_envs, framework)
