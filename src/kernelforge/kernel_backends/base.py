# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-backend system-prompt assembly for the iteration loop."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from kernelforge.config import Config
from kernelforge.kernel_backends.constants import (
    KERNEL_BACKEND_PROMPT_MODULES,
    resolve_language_dirs,
)

log = logging.getLogger(__name__)


# Whole-repo task families that can carry an AITER-framework operator.
_AITER_TASK_TYPES = {"image_kernel", "repository"}


def _is_aiter_operator(task_type: str, source_paths: list[str] | None) -> bool:
    """True when the task optimizes an AITER-framework operator."""
    if (task_type or "").strip().lower() not in _AITER_TASK_TYPES:
        return False
    return any("aiter" in Path(str(p)).parts for p in (source_paths or []))


def build_single_kernel_backend_prompt(
    config: Config,
    kernel_backend_name: str,
    *,
    task_type: str = "",
    source_paths: list[str] | None = None,
) -> str:
    """Build ONE kernel backend's system prompt for the autonomous forge-loop (no network).

    One kernel backend runs per kernel, so the prompt carries that backend's role
    and development discipline plus a knowledge block it can Read on demand,
    assembled in layers from the curated ``local_knowledge/`` tree (see
    ``build_forge_knowledge``). Under ``Config.defer_knowledge_maps``
    (experimental, off) every level is a one-line pointer instead of an inlined
    map. Returns the prompt text, or "" for an unknown kernel_backend.
    """
    backend = (kernel_backend_name or "").strip()
    module_path = KERNEL_BACKEND_PROMPT_MODULES.get(backend)
    if module_path is None:
        return ""

    from kernelforge.knowledge import build_forge_knowledge

    root = Path(config.local_knowledge_dir)
    language = resolve_language_dirs(backend, root)
    include_aiter = backend == "aiter" or _is_aiter_operator(task_type, source_paths)
    # Experimental ablation-only knob (off by default): see Config.include_mori_kb.
    include_mori = bool(getattr(config, "include_mori_kb", False))

    knowledge = build_forge_knowledge(
        root,
        language=language,
        include_aiter=include_aiter,
        include_mori=include_mori,
        defer_all=bool(getattr(config, "defer_knowledge_maps", False)),
    )

    build_prompt = importlib.import_module(module_path).build_system_prompt
    return build_prompt(config.gpu_target, knowledge)
