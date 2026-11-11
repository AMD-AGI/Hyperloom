# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether the served process really imported the sources a patch wrote."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, NamedTuple

from .kernel_decisions import _honest_flag

_TRITON_KERNEL_DECORATORS = frozenset({"jit", "autotune", "heuristics"})
_KERNEL_COMPILE_CUE = re.compile(r"compil|triton", re.IGNORECASE)


def _is_triton_kernel_decorator(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return (
        isinstance(target, ast.Attribute)
        and target.attr in _TRITON_KERNEL_DECORATORS
        and isinstance(target.value, ast.Name)
        and target.value.id == "triton"
    )


def _triton_kernel_names(source_file: str, repo_root: str | Path | None) -> list[str]:
    """Names of the ``@triton.*`` kernels the patched source defines; empty when it cannot be read."""
    path = Path(source_file)
    if not path.is_absolute() and repo_root:
        path = Path(repo_root) / path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError, ValueError):
        return []
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(_is_triton_kernel_decorator(decorator) for decorator in node.decorator_list)
    ]


def _kernel_compiled_in_log(names: list[str], text: str) -> bool:
    patterns = [re.compile(rf"\b{re.escape(name)}\b") for name in names]
    return any(
        _KERNEL_COMPILE_CUE.search(line) and any(pattern.search(line) for pattern in patterns)
        for line in text.splitlines()
    )


def _confirm_source_imported(
    source_file: str,
    workspace: str | Path | None,
    *,
    repo_root: str | Path | None = None,
) -> bool | None:
    """Best-effort confirm the patched source was actually imported/compiled.

    Greps the re-baseline server log for evidence the patched module's basename
    was imported/loaded/compiled, so a measured E2E delta is attributed to code
    the workload really ran. A plain import logs nothing, so a Triton kernel the
    patched file defines showing up in a compile line also counts as evidence.
    Returns a tri-state:

    * ``True``  — the module basename appears in import/load/compile context,
      or one of its Triton kernels appears in a compile line.
    * ``False`` — the server log is readable and neither the basename nor any
      of its Triton kernels appears (evidence the file was not exercised).
    * ``None``  — unknown (no source_file, no readable log) — never penalized.
    """
    if not source_file or not workspace:
        return None
    ws = Path(workspace)
    logs = [p for p in (ws / "server.log", ws.parent / "server.log") if p.exists()]
    if not logs:
        try:
            logs = sorted(ws.rglob("server.log"))[:1]
        except OSError:
            logs = []
    if not logs:
        return None
    stem = Path(source_file).stem
    if not stem:
        return None
    try:
        text = logs[0].read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if stem not in text:
        return _kernel_compiled_in_log(_triton_kernel_names(source_file, repo_root), text)
    # Confirmed only when the basename co-occurs with an import/compile cue.
    for line in text.splitlines():
        if stem in line and re.search(r"import|load|compil|build|\.py", line, re.IGNORECASE):
            return True
    # Present but not in an obvious import context.
    return None


def _confirm_sources_imported(
    source_files: list[str],
    workspace: str | Path | None,
    *,
    repo_root: str | Path | None = None,
) -> tuple[bool | None, dict[str, bool | None]]:
    """Confirm every file a patch wrote was exercised by the served process.

    A patch can span several files -- a new module, the dispatcher that routes
    to it, the original source it replaces -- so each is graded on its own with
    :func:`_confirm_source_imported` and the verdicts are combined:

    * ``True``  — every file shows import evidence.
    * ``False`` — no file appears in the log at all, which is the unambiguous
      "the served process never ran any of this" case.
    * ``None``  — anything mixed. A module can be imported lazily or folded
      into another, so partial evidence is recorded for audit rather than held
      against the patch.
    """
    ordered = list(dict.fromkeys(path for path in source_files if str(path or "").strip()))
    if not ordered:
        return None, {}
    per_file = {path: _confirm_source_imported(path, workspace, repo_root=repo_root) for path in ordered}
    verdicts = list(per_file.values())
    if all(verdict is True for verdict in verdicts):
        return True, per_file
    if all(verdict is False for verdict in verdicts):
        return False, per_file
    return None, per_file


class _SourceImport(NamedTuple):
    confirmed: bool | None
    evidence: dict[str, bool | None]
    decision: str
    downgraded: bool


def _confirm_source_import(
    payload: dict[str, Any],
    *,
    workspace: Any,
    decision: str,
) -> _SourceImport:
    if not _honest_flag("HL_CONFIRM_SOURCE_IMPORTED"):
        return _SourceImport(confirmed=None, evidence={}, decision=decision, downgraded=False)
    # Grade the whole write set; the single target is the fallback for a
    # patch whose bundle declared none.
    written = [str(path) for path in (payload.get("patch_write_paths") or []) if str(path or "").strip()]
    if not written:
        written = [str(payload.get("target_file") or payload.get("source_file") or "")]
    repo_root = payload.get("repo") or payload.get("kernel_repo")
    confirmed, evidence = _confirm_sources_imported(written, workspace, repo_root=repo_root)
    downgraded = decision == "KEEP" and confirmed is False and _honest_flag("HL_CONFIRM_SOURCE_IMPORTED_STRICT")
    return _SourceImport(
        confirmed=confirmed,
        evidence=evidence,
        decision="NEEDS_REVIEW" if downgraded else decision,
        downgraded=downgraded,
    )
