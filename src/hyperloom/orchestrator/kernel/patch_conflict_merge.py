# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Land a Controller patch whose context an already-kept patch has moved.

Controller lanes run in parallel from one pinned base commit, so two lanes that
touch the same file each ship a diff written against that same base.
Integration applies them one at a time and commits every KEEP, which leaves the
second lane's diff stale by the time its turn comes: ``git apply`` refuses it
and a measured speedup is dropped.

Observed in the Kimi-K3 session of 2026-09-13, where ``flydsl_moe_stage2``
(1.1727x micro) was lost because ``flydsl_moe_stage1`` landed first
(``error: patch failed: aiter/ops/flydsl/moe_kernels.py:14``). The two patches
touched disjoint functions and defined disjoint module-level symbols; they
collided only because each inserted its own sweep helpers at the same anchor.

A stale diff is resolved here in three escalating steps: ``git apply -3``,
which settles pure line drift; keeping both sides of every conflict region
whose merge base is empty, which is what two independent insertions at one
anchor look like; and an LLM for the regions where the lanes genuinely edited
the same lines. Anything the last two steps reconstruct must still contain
every line either side added, or it is discarded.

Nothing here decides a KEEP. The E2E gate downstream still measures and still
reverts, so a bad merge costs what a dropped patch already costs -- the patch
does not land -- and can never produce an unmeasured KEEP.
"""

from __future__ import annotations

import ast
import asyncio
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hyperloom.orchestrator.actions.executors._patch_snapshot import (
    _commit_strip_level,
    _patch_touched_paths,
)
from hyperloom.orchestrator.actions.executors.integrate_patch import (
    _git_apply,
    _git_restore_to_head,
)
from hyperloom.orchestrator.specialists.patch_safety import patch_file_targets

STRATEGY_STRICT = "strict"
STRATEGY_THREE_WAY = "three_way"
STRATEGY_UNION = "union_disjoint"
STRATEGY_LLM = "llm"

_LLM_TIMEOUT_S = 300.0
_LLM_MAX_TOKENS = 32000

_MARKER_RE = re.compile(r"^(<<<<<<< |\|\|\|\|\|\|\||=======$|>>>>>>> )", re.MULTILINE)
_CONFLICT_RE = re.compile(
    r"^<<<<<<< [^\n]*\n(?P<ours>.*?)^\|\|\|\|\|\|\|[^\n]*\n(?P<base>.*?)^=======\n(?P<theirs>.*?)^>>>>>>> [^\n]*\n",
    re.DOTALL | re.MULTILINE,
)

_RESOLVER_SYSTEM = """You resolve a git merge conflict between two independently measured GPU kernel optimizations.

Both sides were benchmarked and both must survive. This is a union of two optimizations, never a choice between them.

Rules:
1. Reply with the complete resolved file and nothing else: no prose, no code fences.
2. Remove every conflict marker.
3. Keep every behavioral change from both sides. Where the sides insert independent definitions at the same anchor, keep both.
4. Never leave two module-level definitions of one name. If both sides define a helper identically, keep one copy; if they define it differently, reconcile them into a single definition that satisfies both call sites.
5. Change nothing outside the conflicted regions."""


@dataclass(frozen=True)
class PatchMergeOutcome:
    """What the resolution managed, and how much of the patch it had to rebuild."""

    applied: bool
    strategy: str = ""
    error: str = ""
    steps: tuple[str, ...] = ()
    #: Set when the patch could not be reconciled with the worktree, as opposed
    #: to git failing to write a patch that had already passed ``--check``.
    conflicted: bool = False

    @property
    def reconstructed(self) -> bool:
        """Whether the worktree holds a merge rather than the patch verbatim."""
        return self.applied and self.strategy not in ("", STRATEGY_STRICT)

    def note(self) -> str:
        """The steps taken, as a suffix for the integration result's reason."""
        return " (" + "; ".join(self.steps) + ")" if self.steps else ""


class ConflictResolver(Protocol):
    """Resolves one conflicted file. Returns the merged text, or ``None``."""

    async def __call__(
        self,
        *,
        relative_path: str,
        conflicted_text: str,
        ours_label: str,
        theirs_label: str,
        intent: str,
    ) -> str | None: ...


def _run_git(repo: Path, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _unmerged_paths(repo: Path) -> list[str]:
    completed = _run_git(repo, "diff", "--name-only", "--diff-filter=U")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _strip_level(repo: Path, patch_path: Path) -> int:
    """The ``-p`` level for a patch, chosen by which paths exist.

    ``--check`` probing cannot pick a level for the patch this module exists to
    handle, because that patch fails the check at every level.
    """
    pairs = patch_file_targets(patch_path.read_text(encoding="utf-8", errors="replace"))
    return _commit_strip_level(repo, pairs) if pairs else 1


def _added_lines(patch_text: str) -> set[str]:
    """The distinctive lines a diff adds, whitespace-normalized.

    Blank and near-blank additions carry no identity, so they cannot witness
    that a side survived a merge.
    """
    added: set[str] = set()
    for line in patch_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = " ".join(line[1:].split())
        if len(body) >= 4:
            added.add(body)
    return added


def _missing_additions(repo: Path, patch_path: Path) -> list[str]:
    """Lines ``patch_path`` adds that the worktree does not carry.

    Set-based on purpose: two lanes that inserted byte-identical boilerplate are
    correctly merged by keeping one copy, and that copy witnesses both sides.
    """
    wanted = _added_lines(patch_path.read_text(encoding="utf-8", errors="replace"))
    if not wanted:
        return []
    present: set[str] = set()
    for relative in _patch_touched_paths(repo, [patch_path]):
        try:
            text = (repo / relative).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        present |= {" ".join(line.split()) for line in text.splitlines() if line.strip()}
    return sorted(wanted - present)


def _module_level_names(text: str) -> list[str]:
    """Names a Python source binds at module level, duplicates included."""
    names: list[str] = []
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def _tally(names: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return counts


def _head_text(repo: Path, relative: str) -> str:
    completed = _run_git(repo, "show", f"HEAD:{relative}")
    return completed.stdout if completed.returncode == 0 else ""


def _rewrite_with_base_markers(repo: Path, relative: str) -> str:
    """Re-render one conflicted file with the merge base between the markers.

    ``git apply -3`` writes two-way markers, which cannot distinguish "both
    sides inserted here" from "both sides rewrote what was here". The index
    still holds all three stages, so ask git to render them.
    """
    _run_git(repo, "checkout", "--merge", "--conflict=diff3", "--", relative)
    return (repo / relative).read_text(encoding="utf-8", errors="replace")


def _union_disjoint(text: str) -> tuple[str, int, int]:
    """Keep both sides of every conflict region whose merge base is empty.

    Returns the rewritten text, the count of regions unioned and the count left
    for the resolver. A non-empty base means both lanes changed the same lines,
    which is not a mechanical call.
    """
    unioned = 0
    overlapping = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal unioned, overlapping
        if match.group("base").strip():
            overlapping += 1
            return match.group(0)
        unioned += 1
        ours, theirs = match.group("ours"), match.group("theirs")
        separator = "" if ours.endswith("\n\n") or not theirs.strip() else "\n"
        return f"{ours}{separator}{theirs}"

    return _CONFLICT_RE.sub(replace, text), unioned, overlapping


def _reject_reason(repo: Path, relative: str, resolved: str) -> str:
    """Why this resolution is not a merge, or ``""`` if it is one."""
    if _MARKER_RE.search(resolved):
        return f"{relative}: conflict markers survived"
    if not resolved.strip():
        return f"{relative}: resolution is empty"
    if not relative.endswith(".py"):
        return ""
    try:
        merged_names = _module_level_names(resolved)
    except SyntaxError as error:
        return f"{relative}: merged source does not parse ({error.msg}, line {error.lineno})"
    try:
        before = _tally(_module_level_names(_head_text(repo, relative)))
    except SyntaxError:
        before = {}
    shadowed = sorted(name for name, count in _tally(merged_names).items() if count > 1 and count > before.get(name, 0))
    return f"{relative}: merge redefines module-level {', '.join(shadowed)}" if shadowed else ""


def llm_resolution_available() -> bool:
    """Whether a conflict resolver could actually issue a call right now."""
    from hyperloom.common.llm_config import anthropic_transport_ready

    return bool(anthropic_transport_ready())


def _strip_code_fence(text: str) -> str | None:
    """Unwrap a fenced reply, for a model that ignores the no-fences rule."""
    body = (text or "").strip()
    if not body:
        return None
    if body.startswith("```"):
        lines = body.splitlines()
        end = len(lines) - 1
        while end > 0 and not lines[end].startswith("```"):
            end -= 1
        body = "\n".join(lines[1:end])
        if not body.strip():
            return None
    return body if body.endswith("\n") else body + "\n"


def _anthropic_resolver() -> ConflictResolver:
    async def resolve(
        *,
        relative_path: str,
        conflicted_text: str,
        ours_label: str,
        theirs_label: str,
        intent: str,
    ) -> str | None:
        from hyperloom.common.llm_config import aanthropic_completion, resolve_forge_llm_model

        prompt = (
            f"File: {relative_path}\n"
            f"`ours` is the optimization already landed: {ours_label}\n"
            f"`theirs` is the optimization being landed now: {theirs_label}\n"
            f"{intent}\n\n"
            "Conflicted source:\n\n```\n" + conflicted_text + "\n```"
        )
        result = await aanthropic_completion(
            component="forge",
            operation="patch_conflict_merge",
            model=resolve_forge_llm_model("claude"),
            system=_RESOLVER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_LLM_MAX_TOKENS,
            timeout_s=_LLM_TIMEOUT_S,
        )
        return _strip_code_fence(result.text or "")

    return resolve


async def _resolve_conflicts(
    repo: Path,
    conflicted: Sequence[str],
    *,
    resolver: ConflictResolver | None,
    ours_label: str,
    theirs_label: str,
    intent: str,
    steps: list[str],
) -> tuple[str, str]:
    """Write a resolution for every conflicted file. Returns (strategy, error)."""
    import httpx

    from hyperloom.common.llm_config import LLMConfigError

    strategy = STRATEGY_UNION
    for relative in conflicted:
        marked = _rewrite_with_base_markers(repo, relative)
        resolved, unioned, overlapping = _union_disjoint(marked)
        if overlapping or _MARKER_RE.search(resolved):
            if resolver is None:
                return strategy, f"{relative}: {overlapping} overlapping region(s), no resolver available"
            try:
                proposed = await resolver(
                    relative_path=relative,
                    conflicted_text=marked,
                    ours_label=ours_label,
                    theirs_label=theirs_label,
                    intent=intent,
                )
            except (LLMConfigError, httpx.HTTPError, asyncio.TimeoutError) as error:
                return strategy, f"{relative}: resolver call failed ({error!r})"
            if not proposed:
                return strategy, f"{relative}: resolver returned nothing"
            resolved, strategy = proposed, STRATEGY_LLM
        elif unioned:
            steps.append(f"{relative}: kept both sides of {unioned} disjoint insertion(s)")

        rejection = _reject_reason(repo, relative, resolved)
        if rejection:
            return strategy, rejection
        (repo / relative).write_text(resolved, encoding="utf-8")
        _run_git(repo, "add", "--", relative)
    return strategy, ""


async def apply_patch_resolving_conflicts(
    repo: Path,
    patch_path: Path,
    *,
    operator_id: str = "",
    landed_operator_ids: Sequence[str] = (),
    landed_patches: Sequence[Path] = (),
    intent: str = "",
    resolver: ConflictResolver | None = None,
) -> PatchMergeOutcome:
    """Apply ``patch_path`` into ``repo``, rebuilding it if its context has moved.

    Args:
        repo: The integration checkout, at the commit carrying every KEEP so far.
        patch_path: The publication's diff, written against the pinned base.
        operator_id: The incoming operator, named for the resolver.
        landed_operator_ids: Operators already committed, named for the resolver.
        landed_patches: Their diffs. Every line they added must survive a merge.
        intent: What the incoming patch optimizes, for the resolver.
        resolver: Overrides the Anthropic resolver; tests inject here.

    Returns:
        A :class:`PatchMergeOutcome`. Whenever ``applied`` is false the worktree
        has been restored to HEAD, which carries every KEEP landed so far.
    """
    touched = _patch_touched_paths(repo, [patch_path])
    applies, strict_error = _git_apply(repo, patch_path, three_way=False, check_only=True)
    if applies:
        applied, error = _git_apply(repo, patch_path, three_way=False, check_only=False)
        if applied:
            return PatchMergeOutcome(applied=True, strategy=STRATEGY_STRICT)
        _git_restore_to_head(repo, touched or None)
        return PatchMergeOutcome(applied=False, error=error or "git apply failed")

    strict_error = strict_error or "git apply check failed"
    steps = [f"verbatim apply refused: {_first_line(strict_error)}"]

    def give_up(step: str) -> PatchMergeOutcome:
        _git_restore_to_head(repo, touched or None)
        return PatchMergeOutcome(applied=False, error=strict_error, steps=(*steps, step), conflicted=True)

    three_way = _run_git(repo, "apply", "-3", f"-p{_strip_level(repo, patch_path)}", str(patch_path))
    conflicted = _unmerged_paths(repo)
    if not conflicted:
        if three_way.returncode != 0:
            return give_up(f"three-way apply failed: {_first_line(three_way.stderr)}")
        steps.append("three-way merge absorbed the drift")
        return PatchMergeOutcome(applied=True, strategy=STRATEGY_THREE_WAY, steps=tuple(steps))

    steps.append(f"three-way left conflicts in {', '.join(conflicted)}")
    if resolver is None and llm_resolution_available():
        resolver = _anthropic_resolver()
    strategy, error = await _resolve_conflicts(
        repo,
        conflicted,
        resolver=resolver,
        ours_label=", ".join(landed_operator_ids) or "previously kept patches",
        theirs_label=operator_id or patch_path.stem,
        intent=intent,
        steps=steps,
    )
    if error:
        return give_up(error)

    for source in (patch_path, *landed_patches):
        missing = _missing_additions(repo, Path(source))
        if missing:
            return give_up(f"merge dropped {len(missing)} added line(s), first: {missing[0][:80]}")

    if strategy == STRATEGY_LLM:
        steps.append("resolver merged the overlapping region(s)")
    return PatchMergeOutcome(applied=True, strategy=strategy, steps=tuple(steps))


__all__ = [
    "STRATEGY_LLM",
    "STRATEGY_STRICT",
    "STRATEGY_THREE_WAY",
    "STRATEGY_UNION",
    "ConflictResolver",
    "PatchMergeOutcome",
    "apply_patch_resolving_conflicts",
    "llm_resolution_available",
]
