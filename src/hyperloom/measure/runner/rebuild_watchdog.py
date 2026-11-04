# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A longer scheduler watchdog for an sglang server that rebuilds aiter modules at its first prefill.

A tuned table that names kernel instances the image's prebuilt module lacks makes that module stale
(:func:`~hyperloom.measure.framework.table_modules.stale_table_modules`), and so does a build variable the module's config
entry reads that the server sets otherwise than the image has (:func:`~hyperloom.measure.jit.modules.env_touched_entries`);
aiter rebuilds such a module when the first request reaches the kernel. sglang's scheduler watchdog (300 s unless the
launch sets ``--watchdog-timeout``) can fire while every rank is still in the build, and the test then ends as
``jit_build_timeout``. For such a test the runner raises the watchdog of every sglang point to cover building the stale
modules one after another, each for as long as a module of its kind may take
(:func:`~hyperloom.measure.jit.modules.module_build_limit_sec`: a CK-tile module can take up to 1347 s when several
tests build at once, any other up to 273 s): it adds the flag to a launch that sets none, and raises a value the launch
sets (sglang AgentX launches carry a default one, a recipe or a candidate may set their own) when the estimate is
higher. It never lowers one: the watchdog only bounds a hang, so a longer one changes no measurement, and a hang still
ends once every module has had its time.

The flag is added after the surface and the cache key were taken: how long the watchdog waits changes neither.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.inference_optimizer.framework_registry import FRAMEWORKS
from hyperloom.measure.farm import Farm, FarmError, locate_package, server_python
from hyperloom.measure.framework.table_modules import stale_table_modules
from hyperloom.measure.jit.key import AITER_PACKAGE
from hyperloom.measure.jit.modules import (
    MODULE_BUILD_LIMIT_SEC,
    config_path,
    env_touched_entries,
    module_build_limit_sec,
    read_config,
)
from hyperloom.measure.jit.prewarmed import unlike_prewarmed
from hyperloom.measure.runner.own_jit import point_configs, server_environments
from hyperloom.measure.socket import TestCall

__all__ = ["MODULE_BUILD_SEC", "WATCHDOG_MARGIN_SEC", "raise_rebuild_watchdog", "rebuild_watchdog_sec"]

#: The seconds one stale module other than a CK-tile one is given to build: over twice the longest aiter module build
#: (273 s), and a blockscale GEMM rebuild may not finish within sglang's 300 s.
#: A CK-tile module gets :data:`~hyperloom.measure.jit.modules.CKTILE_MODULE_BUILD_LIMIT_SEC`.
MODULE_BUILD_SEC = MODULE_BUILD_LIMIT_SEC
#: Added to the builds: sglang's own default watchdog, for the step the build interrupted.
WATCHDOG_MARGIN_SEC = 300
_SGLANG = "sglang"
_FLAG = "--watchdog-timeout"
_FLAG_RE = re.compile(r"(?:^|\s)--watchdog-timeout(?:=|\s+|\Z)(?P<value>\S*)")
_UTF8 = "utf-8"


def rebuild_watchdog_sec(modules: Collection[str]) -> int:
    """The watchdog that covers rebuilding the stale ``modules`` one after another, each for as long as a module of its
    kind may take, plus the margin."""
    return sum(module_build_limit_sec(name) for name in modules) + WATCHDOG_MARGIN_SEC


def raise_rebuild_watchdog(
    call: TestCall,
    farm: Farm | None,
    environ: Mapping[str, str],
    cwd: Path,
    built: Collection[str] = (),
    gfx_arch: str | None = None,
) -> str | None:
    """Give every sglang point of ``call`` a ``--watchdog-timeout`` that covers rebuilding the modules its tuned tables or
    build variables make stale, and ``built`` (the modules the test's patch touches that its server builds, the
    compiled-code cache's), keeping a longer one its launch sets, rewriting the point files in place. A note of what was
    raised, else ``None``.

    ``environ`` is what the servers inherit, before the sheet's overrides; ``gfx_arch`` the card's architecture, when known.
    """
    configs = point_configs(call)
    if not configs:
        return None
    stale = _stale_modules(configs, server_environments(call, configs, environ), environ, farm, cwd, gfx_arch)
    modules = stale | set(built)
    if not modules:
        return None
    seconds = rebuild_watchdog_sec(modules)
    raised = 0
    for point, config in zip(call.points, configs):
        if _with_watchdog(config, seconds):
            Path(point).write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
            raised += 1
    if not raised:
        return None
    return (
        f"sglang's watchdog raised to {seconds}s: the patch, the tuned tables or the build variables make "
        f"{len(modules)} aiter module(s) stale ({', '.join(sorted(modules))}), which the server rebuilds at its first "
        "prefill"
    )


def _stale_modules(
    configs: list[dict],
    servers: list[dict[str, str]],
    environ: Mapping[str, str],
    farm: Farm | None,
    cwd: Path,
    gfx_arch: str | None,
) -> frozenset[str]:
    """The module config entries the servers' tuned tables, or the build variables they set otherwise than ``environ``
    (or than the prewarm built a module of the image under), make stale, as the aiter they import resolves them (the
    farm's, when it rebuilds aiter); none when the image has no aiter or its module config cannot be read."""
    try:
        image = locate_package(AITER_PACKAGE, server_python(configs[0], default="python3"), cwd)
    except FarmError:
        return frozenset()
    package = farm.site / AITER_PACKAGE if farm is not None and AITER_PACKAGE in farm.packages else image
    try:
        config = read_config(config_path(package))
    except (OSError, ValueError):
        return frozenset()
    return (
        stale_table_modules(config, image, package, servers)
        | env_touched_entries(config, servers, environ, gfx_arch)
        | unlike_prewarmed(image / "jit", servers, gfx_arch)
    )


def _with_watchdog(config: dict[str, Any], seconds: int) -> bool:
    """Make the watchdog of an sglang ``config`` at least ``seconds``: add the flag to server arguments that set none, or
    raise the value they set when it is lower; whether it changed them. A value that is not a number is kept."""
    bench = config.get("benchmark") or {}
    if str(bench.get("framework") or "").lower() != _SGLANG:
        return False
    if is_magpie_agentx(bench):
        if not isinstance(bench["agentx"], dict):
            return False
        launch = bench["agentx"].setdefault("launch_overrides", {})
        args = [str(token) for token in launch.get("append_args") or []]
        raised = _raised_argv(args, seconds)
        if raised is None:
            return False
        launch["append_args"] = raised
        return True
    envs = bench.setdefault("envs", {})
    name = FRAMEWORKS[_SGLANG].extra_args_env
    current = str(envs.get(name) or "")
    raised_line = _raised_line(current, seconds)
    if raised_line is None:
        return False
    envs[name] = raised_line
    return True


def _raised_argv(args: list[str], seconds: int) -> list[str] | None:
    """``args`` with the flag's last value raised to ``seconds``, or the flag added; ``None`` when they keep theirs."""
    for index in range(len(args) - 1, -1, -1):
        token = args[index]
        if token == _FLAG:
            value = _raised(args[index + 1] if index + 1 < len(args) else "", seconds)
            return None if value is None else [*args[: index + 1], value, *args[index + 2 :]]
        if token.startswith(f"{_FLAG}="):
            value = _raised(token.partition("=")[2], seconds)
            return None if value is None else [*args[:index], f"{_FLAG}={value}", *args[index + 1 :]]
    return [*args, _FLAG, str(seconds)]


def _raised_line(line: str, seconds: int) -> str | None:
    """The server-argument string ``line`` with the flag's last value raised to ``seconds``, or the flag added;
    ``None`` when it keeps its own."""
    found = list(_FLAG_RE.finditer(line))
    if not found:
        return f"{line} {_FLAG} {seconds}".strip()
    last = found[-1]
    value = _raised(last.group("value"), seconds)
    if value is None:
        return None
    return f"{line[: last.start('value')]}{value}{line[last.end('value') :]}"


def _raised(value: str, seconds: int) -> str | None:
    """``seconds`` when the present ``value`` is a number below it; ``None`` keeps ``value`` (a longer watchdog, or one
    that is not a number and is left for sglang to refuse)."""
    try:
        present = float(value)
    except ValueError:
        return None
    return str(seconds) if present < seconds else None
