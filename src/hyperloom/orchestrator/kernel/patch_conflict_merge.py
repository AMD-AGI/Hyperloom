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
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hyperloom.common.unified_diff import parse_unified_diff
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

_LLM_TIMEOUT_S = 120.0
_LLM_CONNECT_TIMEOUT_S = 15.0
_LLM_MAX_TOKENS = 8000
#: Lines of unconflicted source shown either side of a region, for context.
_CONTEXT_LINES = 40

_MARKER_RE = re.compile(r"^(<<<<<<< |\|\|\|\|\|\|\||=======$|>>>>>>> )", re.MULTILINE)
_CONFLICT_RE = re.compile(
    r"^<<<<<<< [^\n]*\n(?P<ours>.*?)^\|\|\|\|\|\|\|[^\n]*\n(?P<base>.*?)^=======\n(?P<theirs>.*?)^>>>>>>> [^\n]*\n",
    re.DOTALL | re.MULTILINE,
)

_RESOLVER_SYSTEM = """You resolve one conflicted region of a git merge between two independently measured GPU kernel optimizations.

Both sides were benchmarked and both must survive. This is a union of two optimizations, never a choice between them.

The region is given with three sections: `ours`, the merge `base` both sides started from, and `theirs`. The lines shown either side of it are context; they are not yours to change.

Rules:
1. Reply with the lines that replace the region, and nothing else: no prose, no code fences, no conflict markers.
2. Keep every behavioral change from both sides. Where the sides insert independent definitions at the same anchor, keep both; where they rewrote one line, compose their effects instead of choosing a winner.
3. Reproduce each side's added lines verbatim wherever composing them allows it. A resolution that paraphrases a benchmarked line away is discarded unread.
4. Never leave two definitions of one name. If both sides define a helper identically, keep one copy; if they define it differently, reconcile them into a single definition that satisfies both call sites.
5. Indent as the region does. The reply is spliced in exactly where the markers are."""


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
    """Resolves one conflicted region."""

    async def __call__(
        self,
        *,
        relative_path: str,
        region: str,
        context_before: str,
        context_after: str,
        ours_label: str,
        theirs_label: str,
        intent: str,
    ) -> str | None:
        """Return the lines that replace ``region``, or ``None`` to decline."""


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


def _staged_additions(repo: Path) -> list[str]:
    """Paths staged by the attempt that HEAD does not carry."""
    completed = _run_git(repo, "diff", "--cached", "--name-only", "--diff-filter=A", "HEAD")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _abandon_attempt(repo: Path, touched: Sequence[str]) -> None:
    """Undo everything a refused attempt wrote, index included.

    ``git apply -3`` implies ``--index``, so a refused attempt leaves its work
    staged. The next lane's KEEP stages its own paths and then commits the
    whole index, which would publish this lane's leftovers under that lane's
    name. ``touched`` alone does not cover them: it is read before the patch
    runs, and a file the patch creates does not exist yet to be named.
    """
    created = _staged_additions(repo)
    for relative in created:
        (repo / relative).unlink(missing_ok=True)
    paths = list(dict.fromkeys([*touched, *created, *_unmerged_paths(repo)]))
    if not paths:
        _git_restore_to_head(repo)
        return
    # Unstage first: `checkout --force HEAD --` cannot clear an index entry for
    # a path HEAD has never carried, and refuses the whole pathspec over it.
    _run_git(repo, "reset", "-q", "--", *paths)
    survivors = [path for path in paths if path not in created]
    if survivors:
        _git_restore_to_head(repo, survivors)


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
    bodies = (" ".join(line.split()) for change in parse_unified_diff(patch_text) for line in change.added)
    return {body for body in bodies if len(body) >= 4}


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


def _resolver_backend() -> str:
    """The agent backend a resolver would call, or ``""`` when none can.

    Forge's rewrite lane picks its backend with
    :func:`preferred_agent_backend`, and the last rung of this ladder is a
    forge call like any other: an OpenAI-only box has to reach Codex here for
    the same reason it reaches Codex there, or every overlapping-region lane on
    that box falls back to the dropped patch this module exists to stop.
    """
    from hyperloom.common.llm_config import (
        AGENT_BACKEND_CODEX,
        anthropic_transport_ready,
        openai_agent_credentialed,
        preferred_agent_backend,
    )

    backend = preferred_agent_backend()
    if backend == AGENT_BACKEND_CODEX:
        return backend if openai_agent_credentialed() else ""
    return backend if anthropic_transport_ready() else ""


def llm_resolution_available() -> bool:
    """Whether a conflict resolver could actually issue a call right now."""
    return bool(_resolver_backend())


def _blank_trimmed(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _strip_code_fence(text: str) -> str | None:
    """Unwrap a fenced reply, for a model that ignores the no-fences rule.

    Only blank lines are trimmed. The reply is spliced in where the markers
    were, so the indentation of its first line of source is part of the answer.
    """
    lines = _blank_trimmed((text or "").splitlines())
    if lines and lines[0].lstrip().startswith("```"):
        closing = next(
            (index for index in range(len(lines) - 1, 0, -1) if lines[index].lstrip().startswith("```")),
            len(lines),
        )
        lines = _blank_trimmed(lines[1:closing])
    return "\n".join(lines) + "\n" if lines else None


def _resolver_prompt(
    *,
    relative_path: str,
    region: str,
    context_before: str,
    context_after: str,
    ours_label: str,
    theirs_label: str,
    intent: str,
) -> str:
    """The one region, its two labels and its surrounding source."""
    return (
        f"File: {relative_path}\n"
        f"`ours` is the optimization already landed: {ours_label}\n"
        f"`theirs` is the optimization being landed now: {theirs_label}\n"
        f"{intent}\n\n"
        f"Lines before the region:\n```\n{context_before}\n```\n\n"
        f"Conflicted region:\n```\n{region}```\n\n"
        f"Lines after the region:\n```\n{context_after}\n```"
    )


def _conflict_resolver() -> ConflictResolver | None:
    """A resolver on whichever backend this deployment runs, or ``None``."""
    from hyperloom.common.llm_config import AGENT_BACKEND_CLAUDE, AGENT_BACKEND_CODEX

    backend = _resolver_backend()
    if backend == AGENT_BACKEND_CODEX:
        return _codex_resolver()
    if backend == AGENT_BACKEND_CLAUDE:
        return _anthropic_resolver()
    return None


def _anthropic_resolver() -> ConflictResolver:
    async def resolve(
        *,
        relative_path: str,
        region: str,
        context_before: str,
        context_after: str,
        ours_label: str,
        theirs_label: str,
        intent: str,
    ) -> str | None:
        from hyperloom.common.llm_config import (
            aanthropic_completion,
            build_http_timeout,
            resolve_forge_llm_model,
        )

        prompt = _resolver_prompt(
            relative_path=relative_path,
            region=region,
            context_before=context_before,
            context_after=context_after,
            ours_label=ours_label,
            theirs_label=theirs_label,
            intent=intent,
        )
        result = await aanthropic_completion(
            component="forge",
            operation="patch_conflict_merge",
            model=resolve_forge_llm_model("claude"),
            system=_RESOLVER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_LLM_MAX_TOKENS,
            # Both transports: the HTTP client reads ``timeout`` and would
            # otherwise fall back to httpx's five-second default, which no
            # generation of this size finishes inside.
            timeout=build_http_timeout(connect=_LLM_CONNECT_TIMEOUT_S, read=_LLM_TIMEOUT_S),
            timeout_s=_LLM_TIMEOUT_S,
        )
        return _strip_code_fence(result.text or "")

    return resolve


def _codex_resolver() -> ConflictResolver:
    async def resolve(
        *,
        relative_path: str,
        region: str,
        context_before: str,
        context_after: str,
        ours_label: str,
        theirs_label: str,
        intent: str,
    ) -> str | None:
        from hyperloom.common.llm_config import (
            achat_completion,
            apply_reasoning_effort,
            build_http_timeout,
            get_async_openai_client,
            resolve_forge_llm_model,
        )
        from hyperloom.orchestrator.roles.base import build_chat_messages

        prompt = _resolver_prompt(
            relative_path=relative_path,
            region=region,
            context_before=context_before,
            context_after=context_after,
            ours_label=ours_label,
            theirs_label=theirs_label,
            intent=intent,
        )
        params: dict[str, object] = {
            "model": resolve_forge_llm_model("codex"),
            "messages": build_chat_messages(_RESOLVER_SYSTEM, prompt),
            "max_completion_tokens": _LLM_MAX_TOKENS,
        }
        apply_reasoning_effort(params)
        client = get_async_openai_client(
            timeout=build_http_timeout(connect=_LLM_CONNECT_TIMEOUT_S, read=_LLM_TIMEOUT_S),
        )
        result = await achat_completion(client, component="forge", operation="patch_conflict_merge", **params)
        return _strip_code_fence(result.text or "")

    return resolve


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _restored_first_indent(region: str, reply: str) -> str:
    """Give the reply's first line the indent the region's first line had.

    A model reproduces the rest of a region at the file's own indentation but
    starts line one at the column the ``<<<<<<<`` it replaces began at. Only
    that line is touched, and only to lengthen an indent the region's own first
    line already starts with, so a correctly indented reply is returned as-is.
    """
    body = [line for line in region.splitlines() if line.strip() and not _MARKER_RE.match(line)]
    wanted = _indent_of(body[0]) if body else ""
    if not wanted or not reply.strip():
        return reply
    present = _indent_of(reply.splitlines()[0])
    return wanted[len(present) :] + reply if wanted.startswith(present) else reply


async def _resolve_regions(
    text: str,
    *,
    relative: str,
    resolver: ConflictResolver,
    ours_label: str,
    theirs_label: str,
    intent: str,
) -> tuple[str, str]:
    """Splice a resolution into every conflict region. Returns (text, error).

    One call per region, carrying the surrounding source as context. Sending
    the whole file instead would ask the model to reproduce thousands of
    unconflicted lines to decide a handful, which no generation budget covers
    and which puts every one of those lines at risk of being rewritten.
    """
    pieces: list[str] = []
    cursor = 0
    for match in _CONFLICT_RE.finditer(text):
        proposed = await resolver(
            relative_path=relative,
            region=match.group(0),
            context_before="\n".join(text[: match.start()].splitlines()[-_CONTEXT_LINES:]),
            context_after="\n".join(text[match.end() :].splitlines()[:_CONTEXT_LINES]),
            ours_label=ours_label,
            theirs_label=theirs_label,
            intent=intent,
        )
        if not proposed:
            return "", f"{relative}: resolver returned nothing"
        spliced = _restored_first_indent(match.group(0), proposed)
        pieces.append(text[cursor : match.start()])
        pieces.append(spliced if spliced.endswith("\n") else spliced + "\n")
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), ""


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
    strategy = STRATEGY_UNION
    for relative in conflicted:
        marked = _rewrite_with_base_markers(repo, relative)
        resolved, unioned, overlapping = _union_disjoint(marked)
        if unioned:
            steps.append(f"{relative}: kept both sides of {unioned} disjoint insertion(s)")
        unresolved = bool(overlapping) or bool(_MARKER_RE.search(resolved))
        # A union that leaves one name defined twice -- both lanes shipped their
        # own copy of a helper -- is not a merge either, and it is exactly what
        # the resolver's fourth rule exists to reconcile. Hand it the marked
        # file rather than ending the ladder on a rejection.
        rejection = "" if unresolved else _reject_reason(repo, relative, resolved)
        if unresolved or rejection:
            if resolver is None:
                return strategy, rejection or f"{relative}: {overlapping} overlapping region(s), no resolver available"
            if rejection:
                steps.append(f"{relative}: union rejected ({rejection}), asking the resolver")
            try:
                resolved, error = await _resolve_regions(
                    marked if rejection else resolved,
                    relative=relative,
                    resolver=resolver,
                    ours_label=ours_label,
                    theirs_label=theirs_label,
                    intent=intent,
                )
            # The resolver is an injected callable reaching a network and an
            # optional SDK; its failures share no base class, and an escape
            # would abort integration with a marked-up tree still on disk.
            except Exception as failure:  # noqa: BLE001 - a broken resolver costs the lane, nothing else
                return strategy, f"{relative}: resolver call failed ({failure!r})"
            if error:
                return strategy, error
            strategy = STRATEGY_LLM
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
        resolver: Overrides this deployment's resolver; tests inject here.

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
        _abandon_attempt(repo, touched)
        return PatchMergeOutcome(applied=False, error=error or "git apply failed")

    strict_error = strict_error or "git apply check failed"
    steps = [f"verbatim apply refused: {_first_line(strict_error)}"]

    def give_up(step: str) -> PatchMergeOutcome:
        _abandon_attempt(repo, touched)
        return PatchMergeOutcome(applied=False, error=strict_error, steps=(*steps, step), conflicted=True)

    three_way = _run_git(repo, "apply", "-3", f"-p{_strip_level(repo, patch_path)}", str(patch_path))
    conflicted = _unmerged_paths(repo)
    if not conflicted:
        if three_way.returncode != 0:
            return give_up(f"three-way apply failed: {_first_line(three_way.stderr)}")
        # Not put through the added-line check below: git merged the hunks
        # itself, and it cannot drop a side without leaving the path unmerged.
        # That check exists for the two rungs that reconstruct a file instead.
        steps.append("three-way merge absorbed the drift")
        return PatchMergeOutcome(applied=True, strategy=STRATEGY_THREE_WAY, steps=tuple(steps))

    steps.append(f"three-way left conflicts in {', '.join(conflicted)}")
    if resolver is None:
        resolver = _conflict_resolver()
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
