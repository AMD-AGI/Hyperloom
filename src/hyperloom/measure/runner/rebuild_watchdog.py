# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A longer scheduler watchdog for an sglang server that rebuilds aiter modules at its first prefill.

A tuned table that names kernel instances the image's prebuilt module lacks makes that module stale
(:func:`~hyperloom.measure.framework.table_modules.stale_table_modules`), and aiter rebuilds it when the first request
reaches the kernel. sglang's scheduler watchdog (300 s unless the launch sets ``--watchdog-timeout``) can fire while every
rank is still in the build, and the test then ends as ``jit_build_timeout``. For such a test the runner raises the
watchdog of every sglang point whose launch does not set it, to the estimated build time of the stale modules.

The flag is added after the surface and the cache key were taken: how long the watchdog waits changes neither.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from hyperloom.common.benchmark_config import is_magpie_agentx
from hyperloom.inference_optimizer.framework_registry import FRAMEWORKS
from hyperloom.measure.farm import Farm, FarmError, locate_package, server_python
from hyperloom.measure.framework.table_modules import stale_table_modules
from hyperloom.measure.jit.key import AITER_PACKAGE
from hyperloom.measure.jit.modules import config_path, read_config
from hyperloom.measure.runner.own_jit import point_configs, server_environments
from hyperloom.measure.socket import TestCall

__all__ = ["MODULE_BUILD_SEC", "WATCHDOG_MARGIN_SEC", "raise_rebuild_watchdog", "rebuild_watchdog_sec"]

#: The seconds one stale module is given to build: over twice the longest aiter build measured before a server was ready
#: (273 s, in 1711 boots), and a blockscale GEMM rebuild did not finish within sglang's 300 s.
MODULE_BUILD_SEC = 600
#: Added to the builds: sglang's own default watchdog, for the step the build interrupted.
WATCHDOG_MARGIN_SEC = 300
_SGLANG = "sglang"
_FLAG = "--watchdog-timeout"
_FLAG_RE = re.compile(r"(?:^|\s)--watchdog-timeout(?:[=\s]|$)")
_UTF8 = "utf-8"


def rebuild_watchdog_sec(modules: int) -> int:
    """The watchdog that covers rebuilding ``modules`` stale modules one after another, plus the margin."""
    return modules * MODULE_BUILD_SEC + WATCHDOG_MARGIN_SEC


def raise_rebuild_watchdog(call: TestCall, farm: Farm | None, environ: Mapping[str, str], cwd: Path) -> str | None:
    """Give every sglang point of ``call`` whose launch sets no ``--watchdog-timeout`` one that covers rebuilding the
    modules its tuned tables make stale, rewriting the point files in place. A note of what was raised, else ``None``.

    ``environ`` is what the servers inherit, before the sheet's overrides.
    """
    configs = point_configs(call)
    if not configs:
        return None
    modules = _stale_modules(configs, server_environments(call, configs, environ), farm, cwd)
    if not modules:
        return None
    seconds = rebuild_watchdog_sec(len(modules))
    raised = 0
    for point, config in zip(call.points, configs):
        if _with_watchdog(config, seconds):
            Path(point).write_text(yaml.safe_dump(config, sort_keys=False), encoding=_UTF8)
            raised += 1
    if not raised:
        return None
    return (
        f"sglang's watchdog raised to {seconds}s: the tuned tables make {len(modules)} aiter module(s) stale "
        f"({', '.join(sorted(modules))}), which the server rebuilds at its first prefill"
    )


def _stale_modules(configs: list[dict], servers: list[dict[str, str]], farm: Farm | None, cwd: Path) -> frozenset[str]:
    """The module config entries the servers' tuned tables make stale, as the aiter they import resolves them (the farm's,
    when it rebuilds aiter); none when the image has no aiter or its module config cannot be read."""
    try:
        image = locate_package(AITER_PACKAGE, server_python(configs[0], default="python3"), cwd)
    except FarmError:
        return frozenset()
    package = farm.site / AITER_PACKAGE if farm is not None and AITER_PACKAGE in farm.packages else image
    try:
        config = read_config(config_path(package))
    except (OSError, ValueError):
        return frozenset()
    return stale_table_modules(config, image, package, servers)


def _with_watchdog(config: dict[str, Any], seconds: int) -> bool:
    """Add ``--watchdog-timeout seconds`` to the server arguments of an sglang ``config`` that sets none; whether it did."""
    bench = config.get("benchmark") or {}
    if str(bench.get("framework") or "").lower() != _SGLANG:
        return False
    if is_magpie_agentx(bench):
        if not isinstance(bench["agentx"], dict):
            return False
        launch = bench["agentx"].setdefault("launch_overrides", {})
        args = [str(token) for token in launch.get("append_args") or []]
        if any(token == _FLAG or token.startswith(f"{_FLAG}=") for token in args):
            return False
        launch["append_args"] = [*args, _FLAG, str(seconds)]
        return True
    envs = bench.setdefault("envs", {})
    name = FRAMEWORKS[_SGLANG].extra_args_env
    current = str(envs.get(name) or "")
    if _FLAG_RE.search(current):
        return False
    envs[name] = f"{current} {_FLAG} {seconds}".strip()
    return True
