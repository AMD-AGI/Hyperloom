# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The explore grid a ``custom`` adapter declares for itself in ``levers.json``.

EXPLORE owns the declaration: the CLI loads it once at launch so a malformed file fails there, and the explore
executor seeds the variants the ledger has not measured yet ahead of the model's own proposals.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .._proposal_identity import is_executable, normalize_proposal


log = logging.getLogger(__name__)

#: The adapter's declared lever file, resolved inside ``$HYPERLOOM_BYPASS_SCRIPTS_DIR``.
#: The scripts dir is the adapter's contract directory and is already persisted and
#: re-exported on ``--resume``, so the declaration travels with the session.
DECLARED_LEVERS_FILENAME = "levers.json"


class DeclaredLeversError(ValueError):
    """A ``levers.json`` exists but cannot be used as a lever declaration."""


def declared_levers_path() -> Path | None:
    """Return the adapter's ``levers.json`` path, or ``None`` when unavailable."""
    scripts_dir = (os.environ.get("HYPERLOOM_BYPASS_SCRIPTS_DIR") or "").strip()
    if not scripts_dir:
        return None
    candidate = Path(scripts_dir) / DECLARED_LEVERS_FILENAME
    return candidate if candidate.is_file() else None


def load_declared_levers() -> list[dict[str, Any]]:
    """Load the lever grid a ``custom`` adapter declares for itself.

    A ``custom`` workload has no framework source for the model to read, so
    ``_default_grid_for_framework`` returns ``[]`` and EXPLORE has nothing to sweep.
    The adapter declares its own knob space instead.

    Each entry is judged by ``is_executable(normalize_proposal(...))`` -- the same
    rule, and the same owner, the executor applies when it builds the grid. A
    declaration that normalizes to nothing a restart could apply is rejected here
    rather than silently recorded as a baseline no-op later.

    Raises:
        DeclaredLeversError: When the file exists but is unreadable, is not valid
            JSON, is not a list of variant objects, or declares a variant no server
            restart could apply. Malformed input is loud by design: every silent
            path here surfaces later as ``empty_grid``, which reads as "the model
            proposed nothing".
    """
    path = declared_levers_path()
    if path is None:
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DeclaredLeversError(f"{path} is unreadable: {exc}") from exc
    try:
        declared = json.loads(raw)
    except ValueError as exc:
        raise DeclaredLeversError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(declared, list):
        raise DeclaredLeversError(f"{path} must be a JSON list of variant objects, got {type(declared).__name__}")
    variants: list[dict[str, Any]] = []
    for index, entry in enumerate(declared):
        if not isinstance(entry, dict):
            raise DeclaredLeversError(f"{path}[{index}] must be a variant object, got {type(entry).__name__}")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise DeclaredLeversError(f"{path}[{index}] has no 'name'")
        if not is_executable(normalize_proposal(entry)):
            # Includes the shapes that *look* set but normalize away: an
            # ``extra_envs`` given as a string or a list becomes ``{}``.
            raise DeclaredLeversError(
                f"{path}[{index}] ({name!r}) declares nothing a restart could apply: "
                "set at least one of extra_args, extra_envs (an object), remove_args, "
                "unset_envs, or args_mode='replace'"
            )
        variant = dict(entry)
        # ``default_grid`` is the seed-grid provenance label; anything else is
        # attributed to the orchestration agent in the session breakdown.
        variant.setdefault("provenance", "default_grid")
        variants.append(variant)
    return variants


def declared_lever_payload(shared_state: Any, framework: str) -> list[dict[str, Any]]:
    """Return the ``custom`` adapter's declared levers the ledger has not benchmarked.

    Skips declarations whose name already appears in the explore ledger's ``tested``
    rows -- the bucket ``apply_explore_search_update`` actually writes. Re-seeding a
    name the ledger has measured re-spends the round's wall-clock budget ahead of the
    model's own proposals.
    """
    if framework != "custom":
        return []
    search = getattr(shared_state, "explore_search", None)
    tested = (search or {}).get("tested") if isinstance(search, dict) else None
    benched = {str(row.get("name") or "") for row in (tested or {}).values() if isinstance(row, dict)}
    fresh = [v for v in load_declared_levers() if str(v.get("name") or "") not in benched]
    if fresh:
        log.info(
            "explore: seeding %d declared lever variant(s) from %s",
            len(fresh),
            DECLARED_LEVERS_FILENAME,
        )
    return fresh
