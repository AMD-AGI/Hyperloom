# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``recipe_steps``: the flat, ordered replay of one enablement.

The array is the intra-round execution order of the final apply round, which by
construction replays everything accumulated before it: every applied setup
command, then the base artifacts a linked build produced, then the accumulated
patches. Position carries the whole ordering contract -- there is no nesting, no
grouping key and no round index in an element.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

from .credentials import classify_credential_class

SETUP_KIND = "setup"
BUILD_KIND = "build"
PATCH_KIND = "patch"

#: Keys a ``build`` element always declares. Shape is fixed by the contract;
#: only population varies, so a key is never omitted -- only its value may be
#: ``None``, which states "no durable value exists" rather than "this producer
#: does not implement the field".
_BUILD_CONTRACT_KEYS: tuple[str, ...] = (
    "component",
    "ref",
    "gpu_arch",
    "max_jobs",
    "build_task_id",
    "build_driver",
    "build_inputs",
)


def command_digest(cmd: str) -> str:
    """Return the sha256 of a verbatim setup command."""
    return hashlib.sha256(str(cmd).encode("utf-8")).hexdigest()


def _is_attempt_row(entry: Any) -> bool:
    """True for a build-attempt row; routing sentinels carry no ``ok`` key."""
    return isinstance(entry, dict) and entry.get("ok") is not None


def select_linked_build(enablement: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return the ``(sentinel, attempt_row)`` pair of the final round's build.

    A build is linked to the final round when its routing sentinel's
    ``probe_task_id`` equals ``last_specialist_task_id`` -- an equality between
    two fields product code already writes, so "this build produced the
    environment the final round validated" is decidable from data rather than
    inferred from recency. The attempt row is then joined by
    ``Path(attempt_root).name == task_id``; being an equality it is
    order-independent, so concurrent completions interleaving in the manifest
    cannot bind a step to another build's row.

    Returns:
        ``(None, None)`` when no sentinel is linked; ``(sentinel, None)`` when
        the linked build has no matching attempt row.
    """
    manifest = enablement.get("build_manifest")
    final_task = str(enablement.get("last_specialist_task_id") or "").strip()
    if not isinstance(manifest, list) or not final_task:
        return None, None
    for entry in manifest:
        if not isinstance(entry, dict) or _is_attempt_row(entry):
            continue
        if str(entry.get("probe_task_id") or "").strip() != final_task:
            continue
        task_id = str(entry.get("task_id") or "").strip()
        for row in manifest:
            if _is_attempt_row(row) and task_id and Path(str(row.get("attempt_root") or "")).name == task_id:
                return entry, row
        return entry, None
    return None, None


def _build_step(
    enablement: Mapping[str, Any],
    attempt_summary: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project the one build linked to the final round, if any.

    The element sits between the setup steps and the patch steps because what it
    stands for in a replay is the built base-artifact state the patches were
    validated on top of.
    """
    sentinel, row = select_linked_build(enablement)
    if sentinel is None:
        return []
    step: dict[str, Any] = {"kind": BUILD_KIND}
    step.update({key: None for key in _BUILD_CONTRACT_KEYS})
    step["build_task_id"] = str(sentinel.get("task_id") or "").strip() or None
    if row is None:
        return [step]
    summary = attempt_summary(row)
    raw_action = row.get("action")
    action: dict[str, Any] = raw_action if isinstance(raw_action, dict) else {}
    step["ref"] = summary.get("ref") or None
    step["gpu_arch"] = summary.get("gpu_arch") or None
    # The summary defaults these to ""/0; a fabricated zero is indistinguishable
    # from a real one and would be read as a collected fact.
    if "component" in action:
        step["component"] = summary.get("component")
    if "max_jobs" in action:
        step["max_jobs"] = summary.get("max_jobs")
    inputs = row.get("build_inputs")
    if isinstance(inputs, dict) and inputs:
        step["build_inputs"] = dict(inputs)
    step["build_driver"] = str(row.get("build_driver") or "").strip() or None
    return [step]


def _digest_occurrences(ledger: list[Mapping[str, Any]]) -> dict[int, int]:
    """Number each ledger row within its ``cmd_digest`` group, in ``seq`` order.

    Counted over every attempt, applied or not, so the ordinal names which
    execution a row belongs to rather than which successful one.
    """
    seen: dict[str, int] = {}
    occurrences: dict[int, int] = {}
    for row in sorted(ledger, key=lambda r: int(r.get("seq") or 0)):
        digest = str(row.get("cmd_digest") or "")
        seen[digest] = seen.get(digest, 0) + 1
        occurrences[id(row)] = seen[digest]
    return occurrences


def _setup_steps(enablement: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project one step per *applied execution*, not per deduped string."""
    commands = [str(c) for c in (enablement.get("setup_commands") or []) if str(c)]
    ledger = [row for row in (enablement.get("setup_executions") or []) if isinstance(row, Mapping)]
    if not ledger:
        # State written before the ledger existed still projects the R1a step
        # set: projecting zero steps from an absent ledger would silently narrow
        # a frozen field's coverage, with no code to say so.
        return [_setup_step(cmd, occurrence=None) for cmd in commands]
    by_digest = {command_digest(cmd): cmd for cmd in commands}
    occurrences = _digest_occurrences(ledger)
    steps: list[dict[str, Any]] = []
    for row in sorted(ledger, key=lambda r: int(r.get("seq") or 0)):
        # ``setup.cmd`` means "run this": a skipped command never ran and a
        # failed one did not satisfy the applied contract, so neither becomes a
        # step. Both stay in the ledger, which is audit data.
        if str(row.get("outcome") or "") != "applied":
            continue
        cmd = by_digest.get(str(row.get("cmd_digest") or ""))
        if cmd is not None:
            steps.append(_setup_step(cmd, occurrence=occurrences.get(id(row))))
    return steps


def _setup_step(cmd: str, *, occurrence: int | None) -> dict[str, Any]:
    return {
        "kind": SETUP_KIND,
        "cmd": cmd,
        "occurrence": occurrence,
        "credential_class": classify_credential_class(cmd),
    }


def _patch_steps(enablement: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project the accumulated patches in ``kept_patches`` order."""
    framework_root = str(enablement.get("framework_root") or "")
    patch_roots = enablement.get("patch_roots")
    patch_roots = patch_roots if isinstance(patch_roots, Mapping) else {}
    roots_by_path = {
        str(record.get("path") or ""): str(record.get("id") or "")
        for record in (enablement.get("roots") or [])
        if isinstance(record, Mapping)
    }
    steps: list[dict[str, Any]] = []
    for raw in enablement.get("kept_patches") or []:
        path = str(raw)
        if not path:
            continue
        root = str(patch_roots.get(path) or "") or framework_root
        steps.append(
            {
                "kind": PATCH_KIND,
                "path": path,
                "root": framework_root,
                "root_id": roots_by_path.get(root) or None,
            }
        )
    return steps


def build_recipe_steps(
    enablement: Mapping[str, Any],
    *,
    attempt_summary: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project one enablement's durable state onto the ordered replay array.

    Args:
        enablement: The enablement state, keyed by ``EnablementRound`` field name.
        attempt_summary: The shipped build-attempt projection, injected so the
            build step reuses it rather than re-reading the manifest in parallel.

    Returns:
        Every applied setup step, then at most one build step, then the patch
        steps. Empty when the enablement contributed nothing.
    """
    return _setup_steps(enablement) + _build_step(enablement, attempt_summary) + _patch_steps(enablement)
