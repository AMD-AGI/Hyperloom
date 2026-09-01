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


# Whole-repo task families that can carry an AITER-framework operator. Snippet
# ("<src>2<dst>") tasks never do — their sources are copied to the workspace root
# without the aiter repo tree.
_AITER_TASK_TYPES = {"image_kernel", "repository"}


def _is_aiter_operator(task_type: str, source_paths: list[str] | None) -> bool:
    """True when the task optimizes an AITER-framework operator.

    An AITER op only comes from a whole-repo task (``image_kernel`` /
    ``repository``) whose sources live under the aiter repo, so their resolved
    path carries an ``aiter`` component (e.g. ``.../aiter/ops/triton/...`` or
    ``.../aiter/csrc/pa/...``). Matching a path *component* (not a substring)
    avoids false positives from task/workspace names like ``aiter_pa_decode``.
    """
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

    One kernel backend runs per kernel, so the prompt carries that kernel backend's role and
    development discipline plus a knowledge block it can Read on demand.

    Knowledge comes from the curated ``local_knowledge/`` tree, assembled in
    layers for the task.
      * ``hardware/`` + ``common_methodology/`` — always.
      * ``framework/aiter/`` — when the target is an AITER operator (whole-repo
        task with an ``aiter`` path component) or the kernel backend is the aiter kernel_backend.
      * ``framework/mori/`` — experimental, off by default; see
        ``Config.include_mori_kb``.
      * ``languages/<lang>/`` — the kernel's implementation language(s), resolved
        from the kernel backend by ``resolve_language_dirs`` (aiter / hipblaslt
        have no language folder, so they get no language layer; triton and gluon
        each carry the other, being one toolchain at two levels).

    Returns the prompt text, or "" for an unknown kernel_backend.
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

    knowledge = build_forge_knowledge(root, language=language, include_aiter=include_aiter, include_mori=include_mori)

    build_prompt = importlib.import_module(module_path).build_system_prompt
    return build_prompt(config.gpu_target, knowledge)
