# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Where a session's benchmark runs keep their JIT caches, so each run starts on the kernels the last one compiled."""

from __future__ import annotations

from pathlib import Path

import yaml

from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.session.session_paths import jit_cache_dir


def benchmark_framework(config_path: Path) -> str:
    """Read the framework a rendered Magpie config benchmarks, or "" when it names none."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return str((cfg.get("benchmark") or {}).get("framework") or "")


def apply_session_jit_cache(env: dict[str, str], *, framework: str, session_dir: Path) -> None:
    """Point ``framework``'s JIT cache env names at session-scoped directories, in place.

    Shape-specialised kernels compile on first use, and a serving benchmark meets new shapes for most of a run: one
    AgentX round compiles hundreds of variants, tens of them after the server reports ready, each stalling the whole
    engine. Left in the image the cache dies with the container, so the first round after a restart pays that cost
    and reads as a regression against rounds that did not. One directory per session makes successive rounds
    comparable; an operator who set the name keeps their own location.
    """
    names = framework_registry.jit_cache_envs(framework)
    if not names:
        return
    root = jit_cache_dir(session_dir)
    for name in names:
        if env.get(name):
            continue
        path = root / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)


__all__ = ["apply_session_jit_cache", "benchmark_framework"]
