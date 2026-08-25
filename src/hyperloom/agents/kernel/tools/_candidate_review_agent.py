###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Agent review of the deterministically produced kernel-candidate table.

The deterministic tiers fail in a way no "fill in the blanks" pass can catch:
they do not come up empty, they come up *confidently wrong*. A launcher frame
proves who launched a kernel, not who defines it; a keyword grep on ``dispatch``
lands on whichever vendor header mentions the word. Both produce a real,
existing, root-resident path that passes every mechanical check.

Reviewing that needs the context the tiers do not have -- what the model is,
how it is being served, what the trace actually recorded -- and enough of the
framework tree to confirm a file really defines the kernel it is credited with.
Rather than pre-loading any of that into a prompt, this hands the agent the
*paths* and lets it read what it needs.

Three properties keep the added freedom bounded:

* **Proposals only.** The agent may revise where a kernel lives, whether it is
  worth dispatching, and the operand dims to tune it against. It may not touch
  the trace's own measurements -- GPU share, duration, launch count -- nor the
  keys the row is identified by. Those carry the impact ranking, the closing
  gain figure and the attempt ledger. :data:`IMMUTABLE_FIELDS` is enforced here,
  not requested in prose.

  Operand dims are the exception, and deliberately so: a graph replay has no
  CPU-side parent op, so the profiler records no arguments for exactly the
  kernels that dominate a captured model, and the field arrives empty. Refusing
  the review's answer there does not preserve a measurement -- it hands the
  choice to a tuning backend that cannot see the serving configuration. Review
  dims therefore carry their own provenance so a later reader can still tell a
  recovered shape from a computed one.
* **Nothing is taken on faith.** A revised path must exist under a known
  framework root. This is not a correctness check; it stops an invented path
  from being written.
* **Nothing is destroyed.** Every revision records ``previous_source_file`` and
  ``previous_method``, and the pre-review table is kept beside the reviewed one,
  so a bad review is auditable and reversible.

The session may run shell commands (demangling a mangled vendor symbol is
exactly the job), so the code under optimization is fingerprinted before and
after: the candidate source files themselves, and the directories holding them
so a file the session *created* is caught too. A review that modified either is
discarded rather than applied -- the benchmark that follows would otherwise
measure an unrecorded edit and credit it to the kernel rewrite. Writes through
the ``Write`` tool do not get that far; they are refused outside the run
directory (see :data:`_GATED_TOOLS`).
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from hyperloom.common import kernel_source_contract as _KSC
except ImportError:  # pragma: no cover - standalone invocation
    _KSC = None  # type: ignore[assignment]

try:
    from hyperloom.common.kernel_shape_contract import (
        REVIEW_BACKFILL_PROVENANCE,
        REVIEW_DERIVED_PROVENANCE,
        REVIEW_SHAPE_PROVENANCE,
    )
except ImportError:  # pragma: no cover - standalone invocation
    REVIEW_BACKFILL_PROVENANCE = "review_backfill"
    REVIEW_DERIVED_PROVENANCE = "review_derived"
    REVIEW_SHAPE_PROVENANCE = frozenset({REVIEW_BACKFILL_PROVENANCE, REVIEW_DERIVED_PROVENANCE})

#: Written by the agent; its presence is what marks the session successful.
REVISIONS_FILENAME = "kernel_candidates_revisions.json"

#: The pre-review table, kept so a bad review can be told from a bad parse.
RAW_CANDIDATES_FILENAME = "kernel_candidates.raw.json"

#: Rejected outright. Two different reasons, both fatal to accept:
#:
#: ``gpu_pct`` / ``duration_us`` / ``call_count`` are event durations read
#: straight off the trace -- there is no parsing ambiguity to correct. They also
#: feed the dispatch floor and the final gain accounting, so a plausible edit
#: here lets the review talk a kernel past the gate it is not supposed to open,
#: and makes the closing report unfalsifiable.
#:
#: ``kernel_id`` / ``name`` / ``device_kernel_name`` are join keys, not facts.
#: ``kernel_id`` is half the attempt-ledger identity; ``name`` is what every
#: shape and metric lookup keys on. Revising either detaches the row from its
#: own history and from the CSVs the review is meant to consult.
IMMUTABLE_FIELDS: frozenset[str] = frozenset(
    {
        "call_count",
        "device_kernel_name",
        "duration_us",
        "gpu_pct",
        "kernel_id",
        "name",
    }
)

#: Recomputed from the operand dims, so a revision naming one is dropped with a
#: note rather than silently overwritten downstream. The session supplies
#: ``shapes`` and ``input_dtypes``; everything the harness builder needs is
#: derived from those by :func:`_rederive_after_review`, which keeps the three
#: representations from drifting apart.
DERIVED_SHAPE_FIELDS: frozenset[str] = frozenset(
    {
        "input_shapes",
        "invocation_cases",
        "raw_arg_spec",
    }
)

_ACTION_KEEP = "keep"
_ACTION_REWRITE = "rewrite"
_ACTION_UNRESOLVE = "unresolve"
_ACTION_DROP = "drop"
_ACTIONS = frozenset({_ACTION_KEEP, _ACTION_REWRITE, _ACTION_UNRESOLVE, _ACTION_DROP})

#: Pre-approved outright: reading and searching cannot alter the tree, so no
#: per-call decision is worth making. ``Edit`` is withheld deliberately -- the
#: agent proposes, it does not patch, and the framework tree here is the code
#: under optimization.
ALLOWED_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob")

#: Permitted but decided per call, not pre-approved. Listing a tool in
#: ``allowed_tools`` tells the SDK to skip the permission callback for it, so a
#: tool that needs a boundary must be left out of that list and admitted by
#: :func:`_write_boundary_guard` instead.
#:
#: ``Write`` is the answer channel -- the session reports by writing the
#: revisions file -- and is confined to the run directory. ``Bash`` is how a
#: mangled vendor symbol gets demangled, which is exactly the job, and it is
#: **not** confined: a shell command can write anywhere the process can, and no
#: inspection of the command string changes that. The two backends therefore
#: reach different guarantees, which is worth stating rather than implying:
#: Codex passes ``writable_roots=(run_dir,)`` and its sandbox refuses a path
#: outside it, covering shell commands too. On the Claude backend a write
#: through ``Write`` is *prevented* and a write through ``Bash`` is *detected*,
#: by the fingerprint taken around the session.
_GATED_TOOLS: tuple[str, ...] = ("Bash", "Write")

#: Tool inputs that name the file a write targets, in the order the SDK fills
#: them. Kept as a tuple rather than a single key because the write tools do not
#: agree on the spelling, and a guard that reads the wrong key admits everything.
_WRITE_PATH_KEYS: tuple[str, ...] = ("file_path", "path", "filePath", "notebook_path")

_DENIED_TOOLS: tuple[str, ...] = (
    "Edit",
    "NotebookEdit",
    "Task",
    "TaskOutput",
    "TaskStop",
    "WebFetch",
    "WebSearch",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "SlashCommand",
)

_MAX_TURNS = 120
_DEFAULT_TIMEOUT_SEC = 900.0
_DEFAULT_ATTEMPTS = 2

_SYSTEM_PROMPT = (
    "You audit an automated mapping from GPU kernel symbols to the source that "
    "defines them, and decide which kernels are worth handing to a kernel "
    "optimizer. Investigate with the tools available: read the candidate table, "
    "grep the framework tree, demangle symbols, consult the model config and "
    "serving arguments. Verify before you revise -- a file that merely calls a "
    "kernel is not the file that defines it. Never modify anything outside the "
    "output directory you are given; the framework tree is the code under "
    "optimization and is checked for tampering. Report findings only by writing "
    "the revisions file you are asked for."
)


@dataclass
class ReviewOutcome:
    """What one review session produced.

    Attributes:
        status: ``completed``, ``skipped`` or a failure label recorded in the
            audit and surfaced as a trace-health warning.
        revisions: The parsed revision records (empty unless ``completed``).
        notes: One human-readable line per applied or rejected revision.
        detail: Failure detail, or ``""`` on success.
        revisions_path: Where the agent wrote its answer, when it did.
    """

    status: str = "skipped"
    revisions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    detail: str = ""
    revisions_path: Path | None = None

    @property
    def ok(self) -> bool:
        """Whether the session produced a usable revision set."""
        return self.status == "completed"


def _safe_exception_label(exc: BaseException) -> str:
    """Return a stable exception label without leaking message contents."""
    label = type(exc).__name__
    for attribute in ("status_code", "code", "errno"):
        value = getattr(exc, attribute, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return f"{label} ({attribute}={value})"
    return label


# ---------------------------------------------------------------------------
# Framework-tree tamper check
# ---------------------------------------------------------------------------


#: ``/repo/moe.py(247): _grouped_gemm`` -> ``/repo/moe.py``. Mirrors
#: :func:`kernel_source_contract.strip_line_suffix` for standalone invocation,
#: where that module is not importable.
_LINE_SUFFIX_RE = re.compile(r"^(?P<path>.+?)\(\d+\)\s*:\s*\S.*$")


def _bare_source_path(raw: str) -> str:
    """Return a stat-able path from a possibly line-annotated candidate source.

    A resolved source frequently arrives as ``path(line): function`` -- the
    contract module measured 29 of 36 entries carrying that suffix on one real
    session. ``os.stat`` cannot open such a string, and the caller below swallows
    the resulting ``OSError``, so leaving the suffix on turns the tamper check
    into a silent no-op for exactly the rows that carry the most evidence.

    Args:
        raw (str): A candidate ``source_file`` value.

    Returns:
        str: The bare path, or ``""`` when nothing path-shaped is left.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    if _KSC is not None:
        return str(_KSC.strip_line_suffix(text) or "").strip()
    match = _LINE_SUFFIX_RE.match(text)
    return match.group("path").strip() if match else text


def source_fingerprint(paths: Sequence[str]) -> dict[str, list[Any]]:
    """Record size and mtime for each readable candidate source.

    Scoped to the files the candidates actually name rather than the whole
    framework tree: those are the ones a source-resolution session has reason
    to open, and hashing gigabytes of installed packages to guard a few dozen
    files would cost more than the session it protects.
    :func:`directory_fingerprint` covers what a file-only print cannot see.

    Args:
        paths (Sequence[str]): Candidate source paths, absolute, with or
            without a ``(line): function`` suffix.

    Returns:
        dict[str, list[Any]]: ``{bare_path: [size, mtime_ns]}`` for readable
            paths. A path that could not be stat'd is absent; use
            :func:`unfingerprinted_sources` to tell that apart from coverage.
    """
    out: dict[str, list[Any]] = {}
    for raw in paths:
        path = _bare_source_path(raw)
        if not path or path in out:
            continue
        try:
            stat = os.stat(path)
        except OSError:
            continue
        out[path] = [stat.st_size, stat.st_mtime_ns]
    return out


def directory_fingerprint(paths: Sequence[str]) -> dict[str, list[Any]]:
    """Record the entry set of every directory holding a candidate source.

    A file-only print can only ever report a change to a file the table already
    named, so a file the session *creates* leaves it untouched -- a patched copy
    beside the kernel, a ``sitecustomize`` module, or any name that shadows a
    real one on ``sys.path``. Those have to land in one of these directories to
    be imported in the traced source's place, so the directory listing is the
    cheap place to notice them: names only, one ``scandir`` per directory.

    Args:
        paths (Sequence[str]): Candidate source paths, as above.

    Returns:
        dict[str, list[Any]]: ``{directory: [entry_count, digest]}`` for
            readable directories.
    """
    out: dict[str, list[Any]] = {}
    for raw in paths:
        path = _bare_source_path(raw)
        if not path:
            continue
        directory = os.path.dirname(path)
        if not directory or directory in out:
            continue
        try:
            names = sorted(entry.name for entry in os.scandir(directory))
        except OSError:
            continue
        digest = hashlib.sha256("\0".join(names).encode("utf-8", "surrogateescape")).hexdigest()
        out[directory] = [len(names), digest]
    return out


def unfingerprinted_sources(
    paths: Sequence[str],
    fingerprint: dict[str, list[Any]],
) -> list[str]:
    """Return the named sources no signature could be taken for.

    An empty fingerprint and an unmodified tree produce the same empty drift
    list, which is what let a broken tamper check read as a passing one. Naming
    the uncovered paths lets the caller report coverage instead of inferring it.

    Args:
        paths (Sequence[str]): The candidate source paths that were offered.
        fingerprint (dict[str, list[Any]]): The result of
            :func:`source_fingerprint` over the same paths.

    Returns:
        list[str]: Bare paths that were named but carry no signature.
    """
    missing: list[str] = []
    for raw in paths:
        path = _bare_source_path(raw)
        if path and path not in fingerprint and path not in missing:
            missing.append(path)
    return sorted(missing)


def fingerprint_drift(before: dict[str, list[Any]], after: dict[str, list[Any]]) -> list[str]:
    """Return the paths whose recorded signature changed between two prints.

    Both directions count. A key only in ``before`` was deleted; a key only in
    ``after`` came into existence during the session -- which is how a directory
    print reports an added file, and how a source that was absent when the first
    print was taken reports being written. Iterating ``before`` alone reports
    neither.
    """
    drifted = [path for path in set(before) | set(after) if before.get(path) != after.get(path)]
    return sorted(drifted)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_review_prompt(
    *,
    run_dir: Path,
    raw_candidates_path: Path,
    revisions_path: Path,
    reference_paths: dict[str, str],
    framework_roots: Sequence[str],
    context_block: str = "",
) -> str:
    """Render the review request as a set of paths to investigate.

    Deliberately carries no file contents. Pre-loading the framework tree would
    bound the review by whatever was guessed to be relevant, whereas the agent
    can follow the evidence -- and only ships what it actually opened.

    The closing "write nothing outside the run directory" line is enforced for
    the tool that writes -- ``Write`` is refused outside ``run_dir`` on both
    backends -- and is an instruction for the one that can write anyway: a
    ``Bash`` command on the Claude backend is bounded by detection, not by
    refusal. See :data:`_GATED_TOOLS` for the split.
    """
    task = (
        "Audit the kernel-candidate table produced by the deterministic "
        "analysis stage and correct it where the evidence disagrees."
    )
    lines = [
        task,
        "",
        f"Candidate table to audit: {raw_candidates_path}",
        "",
        "Reference material (read what you need):",
    ]
    for label, path in reference_paths.items():
        if path:
            lines.append(f"  {label}: {path}")
    if framework_roots:
        lines.append("  framework source roots:")
        lines.extend(f"    {root}" for root in framework_roots)
    if context_block:
        lines += ["", context_block]
    lines += [
        "",
        "For every entry in hot_kernels decide one of:",
        "  keep       the current source_file plausibly defines this kernel",
        "  rewrite    the current path is wrong and you verified a better one",
        "  unresolve  this kernel has no single defining source, or the path is",
        "             wrong and you could not determine the right one",
        "  drop       this entry is not a kernel worth optimizing at all",
        "",
        "Rules:",
        "  - Verify a rewrite by opening the file and confirming it defines the",
        "    kernel. A file that only calls or dispatches to it does not count.",
        "  - Prefer unresolve over a guess. A wrong path costs an entire",
        "    optimization attempt; an empty one just falls through.",
        "  - You may revise source_file, reusable_native_kernel, skip_reason,",
        "    benchmark_files, shapes and input_dtypes. You may not revise what",
        "    the trace measured (gpu_pct, duration_us, call_count) or the",
        "    identity it is keyed by (kernel_id, name, device_kernel_name);",
        "    those are ignored if present. input_shapes, invocation_cases and",
        "    raw_arg_spec are recomputed from shapes, so do not send them.",
        "  - benchmark_files comes from a curated table keyed by coarse name",
        "    markers, so it often names a harness for the wrong member of a",
        "    kernel family. Replace it with harnesses you located and can open,",
        "    or with an empty list when this kernel has none. Paths that do not",
        "    exist are dropped.",
        "  - Entries you do not mention are left exactly as they are.",
        "  - Do not copy reusable_native_kernel back from the table you were",
        "    given. Every unresolved row carries false there, and returning that",
        "    value alongside a corrected path refuses the kernel you just found.",
        "    Send the field only to veto a kernel the rules would otherwise",
        "    accept, and always with a skip_reason saying why; a false with no",
        "    skip_reason is ignored. Routability is recomputed from the path you",
        "    give, so a rewrite needs nothing else from you.",
        "",
        "On shapes -- read this before proposing any:",
        "  A graph replay has no CPU-side parent op, so the profiler records no",
        "  arguments for a graph-launched kernel and shapes arrives empty. That",
        "  is not harmless: with no shapes the tuning backend picks its own, and",
        "  it cannot see the serving configuration. A prefill kernel serving an",
        "  8192-token input has been tuned at sequence length 512 this way, which",
        "  measured a large speedup that vanished end to end.",
        "  So an empty shapes is worth filling. Two ways, in this order:",
        f"    {REVIEW_BACKFILL_PROVENANCE}  another row of analysis.md already records",
        "                      the dims. A composite operator that kept its module",
        "                      attribution lists the device kernels it launches in",
        "                      its Kernel Name cell, and carries the arguments this",
        "                      row is missing. Confirm the row really launches this",
        "                      kernel before taking its dims -- a neighbouring",
        "                      instantiation of the same kernel family is a",
        "                      different problem size, not this one.",
        f"    {REVIEW_DERIVED_PROVENANCE}    you computed the dims from the model config,",
        "                      the serving arguments and the kernel signature.",
        "  analysis.md is TraceLens' only supported output. Do not go looking for",
        "  its intermediate files; they are internal and may not be there.",
        "  Set shape_provenance to whichever applies; any other value is read as",
        f"  {REVIEW_DERIVED_PROVENANCE}. Give one shape string per operand, dims first and",
        '  dtype second, e.g. "(8192,6144) bf16".',
        "  State where the dims came from in reason -- which operator's row, or",
        "  which config fields and what arithmetic -- so a reader can check it in",
        "  seconds. An unstated derivation is not reviewable, and a shape nothing",
        "  can check is what the tuning backend already produces on its own.",
        "",
        f"Write your answer to {revisions_path} as JSON:",
        '  {"revisions": [{"kernel_id": "k001", "action": "rewrite",',
        '                  "source_file": "/abs/path.py",',
        '                  "reusable_native_kernel": true,',
        '                  "skip_reason": "",',
        '                  "shapes": ["(8192,6144) bf16", "(6144,1536) fp4"],',
        '                  "input_dtypes": ["bf16", "fp4"],',
        f'                  "shape_provenance": "{REVIEW_DERIVED_PROVENANCE}",',
        '                  "reason": "one sentence citing what you checked"}]}',
        "",
        "Use action keep with shapes when the recorded path is already right and",
        "only the dims are missing.",
        "",
        f"Write nothing outside {run_dir}. Do not modify framework source.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session drivers
# ---------------------------------------------------------------------------


def _resolve_backend() -> str:
    """Return ``codex`` or ``claude`` from the configured credentials."""
    from hyperloom.common import llm_config  # noqa: PLC0415

    if llm_config.is_anthropic_only():
        return "claude"
    if llm_config.has_openai_side():
        return "codex"
    return "claude"


def _within(path: str, root: Path) -> bool:
    """Whether ``path`` resolves inside ``root``.

    Resolves both sides before comparing, so ``..`` segments and a symlink
    pointing out of the run directory are rejected rather than spelled around.
    """
    try:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = root / target
        resolved = target.resolve()
        base = root.resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return resolved == base or base in resolved.parents


def _write_boundary_guard(
    run_dir: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> Callable[[str, dict[str, Any], Any], Any]:
    """Build the ``can_use_tool`` callback that confines writes to ``run_dir``.

    This is the enforcement the Claude backend previously only described. The
    prompt's closing "write nothing outside the run directory" line is now
    backed for the tool that writes: a ``Write`` naming a path outside the run
    directory is refused, and the refusal reaches the session as a tool error it
    can react to rather than as a silent success.

    ``Bash`` is admitted. A shell command can write anywhere the process can,
    and reading the command string to guess whether it will is the kind of guard
    that fails quietly on the first construction nobody predicted -- so it is
    not attempted here, and :func:`directory_fingerprint` carries that case
    instead. Routing ``Bash`` through the same callback still buys one thing:
    every gated call has a single place that decided it, and that place logs.

    Args:
        run_dir (Path): The only directory the session may write to.
        log (Callable[[str], None] | None): Optional diagnostics callback.

    Returns:
        Callable: An async ``can_use_tool`` callback.
    """
    import claude_agent_sdk as sdk  # type: ignore[import-not-found]  # noqa: PLC0415

    async def _can_use_tool(tool_name: str, tool_input: dict[str, Any], context: Any) -> Any:
        if tool_name != "Write":
            return sdk.PermissionResultAllow()
        target = next(
            (
                str(tool_input.get(key) or "").strip()
                for key in _WRITE_PATH_KEYS
                if str(tool_input.get(key) or "").strip()
            ),
            "",
        )
        # A write whose target cannot be read off the input is refused rather
        # than allowed: an unrecognised spelling must not become a bypass.
        if not target or not _within(target, run_dir):
            message = (
                f"refused: {tool_name} may only write under {run_dir}; "
                f"{target or '(no path in tool input)'} is outside it"
            )
            if log is not None:
                log(f"[review-agent] WARNING: {message}")
            return sdk.PermissionResultDeny(message=message)
        return sdk.PermissionResultAllow()

    return _can_use_tool


async def _run_claude_session(
    prompt: str,
    *,
    run_dir: Path,
    model: str,
    timeout_sec: float,
    log: Callable[[str], None] | None,
) -> str:
    """Drive one tool-enabled Claude Agent SDK session; return any SDK error."""
    import claude_agent_sdk as sdk  # type: ignore[import-not-found]  # noqa: PLC0415

    from hyperloom.common.llm_config import claude_sdk_env_options  # noqa: PLC0415

    kwargs: dict[str, Any] = dict(claude_sdk_env_options(model=model))
    kwargs.update(
        {
            "model": model,
            "system_prompt": _SYSTEM_PROMPT,
            "max_turns": _MAX_TURNS,
            # The gated tools stay out of ``allowed_tools`` on purpose: a tool
            # listed there is pre-approved and the callback is never consulted.
            "allowed_tools": list(ALLOWED_TOOLS),
            "disallowed_tools": list(_DENIED_TOOLS),
            "cwd": str(run_dir),
            "can_use_tool": _write_boundary_guard(run_dir, log=log),
        }
    )
    # Degrade one option at a time, and give up the write boundary last: an SDK
    # that does not know ``cwd`` must not cost the guard as collateral. Dropping
    # ``can_use_tool`` has to widen ``allowed_tools`` in the same step, because a
    # gated tool with no decider left would stall on a prompt nobody answers --
    # that is the pre-existing behaviour, where detection carries the whole
    # guarantee.
    without_guard = {
        **{key: value for key, value in kwargs.items() if key != "can_use_tool"},
        "allowed_tools": list(ALLOWED_TOOLS) + list(_GATED_TOOLS),
    }
    variants = (
        kwargs,
        {key: value for key, value in kwargs.items() if key != "cwd"},
        without_guard,
        {key: value for key, value in without_guard.items() if key != "cwd"},
    )
    options = None
    for variant in variants:
        try:
            options = sdk.ClaudeAgentOptions(**variant)
            break
        except TypeError:
            continue
    if options is None:  # pragma: no cover - no known SDK rejects the last variant
        raise TypeError("claude_agent_sdk.ClaudeAgentOptions rejected every option set")

    # The in-process SDK query has no client-side read timeout, so a stalled
    # gateway stream blocks until the overall bound -- fifteen minutes of a run
    # that is already over. Bound each wait for the next message instead, using
    # the same two tiers the TraceLens runner established: the SDK is silent by
    # design between the block that launches a tool and the result that ends it,
    # so a single bound tight enough to catch a dead stream would kill a working
    # tool call.
    # Absolute, not relative: this module is loaded as a top-level module by the
    # analysis tool, not as a package member.
    from tracelens_skill_runner import (  # noqa: PLC0415
        _resolve_stream_idle_timeout_sec,
        _resolve_tool_idle_timeout_sec,
        _tool_call_transition,
    )

    idle_timeout = _resolve_stream_idle_timeout_sec()
    tool_idle_timeout = _resolve_tool_idle_timeout_sec(idle_timeout)
    sdk_error = ""

    async def _drive() -> None:
        nonlocal sdk_error
        tool_in_flight = False
        stream = sdk.query(prompt=prompt, options=options)
        stream_iter = stream.__aiter__() if hasattr(stream, "__aiter__") else stream
        while True:
            wait_for = tool_idle_timeout if tool_in_flight else idle_timeout
            try:
                if wait_for > 0:
                    message = await asyncio.wait_for(stream_iter.__anext__(), timeout=wait_for)
                else:
                    message = await stream_iter.__anext__()
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                # Name the phase: silence during a tool call means the tool
                # overran its bound, not that the gateway died.
                phase = "while a tool call was in flight" if tool_in_flight else "with no tool call in flight"
                sdk_error = f"stream idle timeout: no SDK message for {wait_for:.0f}s {phase}"
                if log is not None:
                    log(f"[review-agent] WARNING: {sdk_error}")
                # Tear the generator down so its transport does not leak.
                aclose = getattr(stream_iter, "aclose", None)
                if aclose is not None:
                    try:
                        await asyncio.wait_for(aclose(), timeout=10.0)
                    except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                        pass
                break
            transition = _tool_call_transition(message)
            if transition == "start":
                tool_in_flight = True
            elif transition == "end":
                tool_in_flight = False
            if log is not None:
                for text in _message_text(message):
                    if text.strip():
                        log(f"[review-agent] {text.strip()[:400]}")

    try:
        await asyncio.wait_for(_drive(), timeout=max(60.0, timeout_sec))
    except Exception as exc:  # noqa: BLE001 - artifact presence decides success
        return _safe_exception_label(exc)
    return sdk_error


def _message_text(message: Any) -> list[str]:
    """Best-effort text extraction that never breaks the session loop."""
    try:
        from hyperloom.common.claude_oneshot import message_text  # noqa: PLC0415

        return list(message_text(message))
    except Exception:  # noqa: BLE001 - logging aid only
        return []


async def _run_codex_session(
    prompt: str,
    *,
    run_dir: Path,
    model: str,
    timeout_sec: float,
) -> str:
    """Drive one Codex Agent SDK turn scoped to ``run_dir``; return any error."""
    from hyperloom.common.codex_session import (  # noqa: PLC0415
        CodexSessionError,
        run_codex_turn,
    )

    try:
        result = await run_codex_turn(
            prompt=prompt,
            developer_instructions=_SYSTEM_PROMPT,
            cwd=run_dir,
            model=model,
            timeout_sec=max(60.0, timeout_sec),
            writable_roots=(run_dir,),
        )
    except CodexSessionError as exc:
        return _safe_exception_label(exc)
    return str(getattr(result, "error", "") or "")


def _resolve_model(backend: str) -> str:
    """Resolve the session model from the configured environment."""
    explicit = str(os.environ.get("HYPERLOOM_LLM_SOURCE_MODEL") or "").strip()
    if explicit:
        return explicit
    if backend == "codex":
        return str(os.environ.get("CODEX_MODEL") or "").strip() or "gpt-5-codex"
    return str(os.environ.get("CLAUDE_MODEL") or "").strip() or "claude-opus-5"


# ---------------------------------------------------------------------------
# Revision loading and application
# ---------------------------------------------------------------------------


def load_revisions(revisions_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Read the revision file the agent wrote.

    Returns:
        tuple[list[dict[str, Any]], str]: ``(revisions, error)``; ``error`` is
            empty when the file parsed into a revision list.
    """
    try:
        payload = json.loads(Path(revisions_path).read_text(encoding="utf-8"))
    except OSError:
        return [], "revisions file was not written"
    except (TypeError, ValueError):
        return [], "revisions file is not valid JSON"
    if not isinstance(payload, dict):
        return [], "revisions file is not a JSON object"
    revisions = payload.get("revisions")
    if not isinstance(revisions, list):
        return [], "revisions file has no 'revisions' list"
    return [r for r in revisions if isinstance(r, dict)], ""


def _verified_harnesses(proposed: Any) -> list[str] | None:
    """Keep the proposed harness paths that exist, or ``None`` when unset.

    The curated harness table is keyed by coarse name markers, so it offers a
    plausible file for a whole family of kernels rather than the one that
    exercises this kernel. The session can look, which is worth more than the
    marker match -- but only files it can name and that are really there
    survive, since a non-empty list reads downstream as a runnable harness.
    """
    if not isinstance(proposed, list):
        return None
    return [
        str(entry) for entry in proposed if isinstance(entry, str) and entry.strip() and os.path.isfile(entry.strip())
    ]


def _acceptable_path(picked: str, roots: Sequence[str]) -> str:
    """Return the canonical form of ``picked``, or ``""`` when unverifiable."""
    if _KSC is None:
        return ""
    bare = _KSC.strip_line_suffix(picked)
    return _KSC.canonical_source_path(bare, tuple(roots)) or ""


def _proposed_strings(value: Any) -> list[str] | None:
    """Keep the non-empty strings in a proposed list, or ``None`` when unset."""
    if not isinstance(value, list):
        return None
    return [entry.strip() for entry in value if isinstance(entry, str) and entry.strip()]


def _proposed_shape_provenance(value: Any) -> str:
    """Narrow a claimed shape provenance to one the review is allowed to assert.

    A session that located a recorded shape and one that computed a shape from
    the model config are both useful, but only the first is a measurement. The
    claim is therefore restricted to the two review values, and anything else --
    including a session naming ``torch_trace`` -- degrades to the derived label
    rather than being taken at its word. Laundering a derivation as a
    measurement would strip the one signal that tells a later reader whether a
    disappointing end-to-end result is worth blaming on the shape.
    """
    claimed = str(value or "").strip().lower()
    if claimed in REVIEW_SHAPE_PROVENANCE:
        return claimed
    return REVIEW_DERIVED_PROVENANCE


def _record_shape_proposal(
    entry: dict[str, Any],
    revision: dict[str, Any],
    *,
    kernel_id: str,
    notes: list[str],
) -> None:
    """Stage proposed operand dims for the deterministic re-derivation pass.

    Held under ``review_*`` keys rather than written straight onto the row, so
    the stamping pass stays the only thing that decides what the harness builder
    finally sees -- the same split the routability hint already uses.
    """
    shapes = _proposed_strings(revision.get("shapes"))
    if shapes is None:
        return
    if not shapes:
        notes.append(f"{kernel_id}: empty shapes proposal ignored")
        return
    entry["review_shapes"] = shapes
    entry["review_shape_provenance"] = _proposed_shape_provenance(revision.get("shape_provenance"))
    dtypes = _proposed_strings(revision.get("input_dtypes"))
    if dtypes:
        entry["review_input_dtypes"] = dtypes
    notes.append(f"{kernel_id}: shapes -> {len(shapes)} operand(s) ({entry['review_shape_provenance']})")


def _record_judgement_proposals(
    entry: dict[str, Any],
    revision: dict[str, Any],
    *,
    kernel_id: str,
    notes: list[str],
) -> None:
    """Stage the routability hint and the verified harness list.

    A veto is only taken with a stated reason. The table handed to the session
    already carries ``reusable_native_kernel``, and an unresolved row carries
    ``false``; a session correcting that row's path has been observed returning
    the field unchanged while its own prose argued the file is editable. Nothing
    distinguishes that echo from an intended refusal except the reason the
    prompt asks for alongside it, and refusing on the echo threw away every
    kernel the review had just located.

    The asymmetry is deliberate and matches the one below it: a permissive hint
    is ignored because ``classify_patchability`` still has to agree, so it costs
    nothing to drop. A restrictive one has no second gate behind it.
    """
    proposed_skip = revision.get("skip_reason")
    skip_text = proposed_skip.strip() if isinstance(proposed_skip, str) else ""
    if isinstance(proposed_skip, str):
        entry["review_skip_reason"] = skip_text
    proposed_reusable = revision.get("reusable_native_kernel")
    if isinstance(proposed_reusable, bool):
        if proposed_reusable or skip_text:
            entry["review_reusable_hint"] = proposed_reusable
        else:
            notes.append(f"{kernel_id}: veto ignored, no skip_reason given (a refusal has to say why)")
    harnesses = _verified_harnesses(revision.get("benchmark_files"))
    if harnesses is not None:
        entry["review_benchmark_files"] = harnesses
        notes.append(f"{kernel_id}: benchmark_files -> {len(harnesses)} verified path(s)")


def apply_revisions(
    candidates: list[dict[str, Any]],
    revisions: Sequence[dict[str, Any]],
    *,
    framework_roots: Sequence[str],
    protected_ids: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    """Apply the agent's proposals to ``candidates`` in place.

    Only the judgement fields move. Derived state (``source_type``,
    ``kernel_repo``, backends, category, routability) is deliberately left for
    the caller to recompute through the deterministic stamping pass, so
    :func:`classify_patchability` stays the single gate rather than gaining a
    second, model-written one.

    Args:
        candidates: The finalized candidate rows, mutated in place.
        revisions: Revision records parsed from the agent's answer.
        framework_roots: Roots a revised path must resolve under.
        protected_ids: Candidates resolved by an authoritative tier. The active
            finder demangles the device symbol and pins the source in the
            installed tree; reading the same tree cannot beat knowing which
            symbol the binary actually exports, so those are left alone.

    Returns:
        list[str]: One note per applied or rejected revision.
    """
    by_id = {str(c.get("kernel_id") or ""): c for c in candidates if isinstance(c, dict)}
    notes: list[str] = []
    for revision in revisions:
        kernel_id = str(revision.get("kernel_id") or "").strip()
        entry = by_id.get(kernel_id)
        if entry is None:
            notes.append(f"{kernel_id or '(no id)'}: unknown kernel_id, ignored")
            continue
        action = str(revision.get("action") or "").strip().lower()
        if kernel_id in protected_ids and action != _ACTION_KEEP:
            notes.append(f"{kernel_id}: {action} refused, resolved by an authoritative tier")
            continue
        if action not in _ACTIONS:
            notes.append(f"{kernel_id}: unknown action {action!r}, ignored")
            continue
        touched = sorted(IMMUTABLE_FIELDS.intersection(revision) - {"kernel_id"})
        if touched:
            notes.append(f"{kernel_id}: ignored measured field(s) {', '.join(touched)}")
        derived = sorted(DERIVED_SHAPE_FIELDS.intersection(revision))
        if derived:
            notes.append(f"{kernel_id}: ignored derived field(s) {', '.join(derived)}; propose shapes instead")
        reason = str(revision.get("reason") or "").strip()

        # Operand dims are worth having whether or not the path moved, and the
        # rows that most need them are the ones the deterministic tiers already
        # located: under graph capture a replay records no arguments, so a
        # correctly resolved kernel can still arrive with no shape at all.
        if action == _ACTION_KEEP:
            _record_shape_proposal(entry, revision, kernel_id=kernel_id, notes=notes)
            _record_judgement_proposals(entry, revision, kernel_id=kernel_id, notes=notes)
            continue

        previous_file = str(entry.get("source_file") or "")
        previous_method = str(entry.get("source_resolution_method") or "")

        if action in (_ACTION_UNRESOLVE, _ACTION_DROP):
            entry["previous_source_file"] = previous_file
            entry["previous_method"] = previous_method
            entry["source_file"] = ""
            entry.pop("source_line", None)
            entry.pop("source_function", None)
            entry["source_resolution_method"] = "llm_review"
            entry["review_action"] = action
            entry["review_reason"] = reason or "no defining source"
            notes.append(f"{kernel_id}: {action} (was {previous_file or '(none)'})")
            continue

        picked = str(revision.get("source_file") or "").strip()
        if not picked:
            notes.append(f"{kernel_id}: rewrite without a path, ignored")
            continue
        canonical = _acceptable_path(picked, framework_roots)
        if not canonical:
            notes.append(f"{kernel_id}: rejected unverifiable path {picked!r}")
            continue
        previous_bare = _KSC.strip_line_suffix(previous_file) if _KSC else previous_file
        # A rewrite that lands on the path already recorded is not a correction,
        # but the rest of the same revision still is: falling through keeps a
        # confirmed location from costing the shapes proposed alongside it.
        if canonical != previous_bare:
            entry["previous_source_file"] = previous_file
            entry["previous_method"] = previous_method
            entry["source_file"] = canonical
            entry.pop("source_line", None)
            entry.pop("source_function", None)
            entry["source_resolution_method"] = "llm_review"
            entry["review_action"] = action
            entry["review_reason"] = reason or "no reason given"
            notes.append(f"{kernel_id}: {previous_file or '(none)'} -> {canonical}")

        _record_shape_proposal(entry, revision, kernel_id=kernel_id, notes=notes)
        _record_judgement_proposals(entry, revision, kernel_id=kernel_id, notes=notes)
    return notes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_candidate_review_async(
    *,
    run_dir: Path,
    raw_candidates_path: Path,
    reference_paths: dict[str, str],
    framework_roots: Sequence[str],
    context_block: str = "",
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    attempts: int = _DEFAULT_ATTEMPTS,
    log: Callable[[str], None] | None = None,
    session_runner: Callable[..., Any] | None = None,
) -> ReviewOutcome:
    """Run the review session and return its parsed revisions.

    Retries a failed session once by default: the pass is mandatory on the agent
    route, and a gateway hiccup should not be the reason a run's candidate table
    goes unaudited. A definitive failure is reported rather than raised -- the
    deterministic table is still usable, and killing a multi-hour optimization
    over an advisory pass would trade a small loss for a total one.

    This is the real implementation; :func:`run_candidate_review` wraps it for
    the synchronous CLI path. Both backends are async SDKs, so a caller that
    already owns an event loop should await this rather than reach for the
    wrapper.

    Args:
        run_dir: Session directory; the only place the agent may write.
        raw_candidates_path: The pre-review candidate table to audit.
        reference_paths: Labelled artifact paths offered to the agent.
        framework_roots: Roots a revised path must resolve under.
        context_block: Rendered model/serving context, or ``""``.
        timeout_sec: Wall-clock bound per attempt.
        attempts: Total attempts, including the first.
        log: Optional diagnostics callback.
        session_runner: Injection point for the session call (tests); may be
            sync or return an awaitable.

    Returns:
        ReviewOutcome: The session result; never raises.
    """

    def _say(message: str) -> None:
        if log is not None:
            log(f"candidate_review: {message}")

    revisions_path = Path(run_dir) / REVISIONS_FILENAME
    prompt = build_review_prompt(
        run_dir=Path(run_dir),
        raw_candidates_path=Path(raw_candidates_path),
        revisions_path=revisions_path,
        reference_paths=reference_paths,
        framework_roots=framework_roots,
        context_block=context_block,
    )

    try:
        backend = _resolve_backend()
        model = _resolve_model(backend)
    except Exception as exc:  # noqa: BLE001 - configuration is reported, not raised
        detail = _safe_exception_label(exc)
        _say(f"configuration failed: {detail}")
        return ReviewOutcome(status="configuration_error", detail=detail)

    last_detail = ""
    for attempt in range(1, max(1, int(attempts)) + 1):
        revisions_path.unlink(missing_ok=True)
        _say(f"attempt {attempt}/{attempts} via {backend} ({model})")
        try:
            if session_runner is not None:
                produced = session_runner(
                    prompt=prompt,
                    run_dir=Path(run_dir),
                    model=model,
                    timeout_sec=timeout_sec,
                )
                error = await produced if inspect.isawaitable(produced) else produced
            elif backend == "codex":
                error = await _run_codex_session(
                    prompt,
                    run_dir=Path(run_dir),
                    model=model,
                    timeout_sec=timeout_sec,
                )
            else:
                error = await _run_claude_session(
                    prompt,
                    run_dir=Path(run_dir),
                    model=model,
                    timeout_sec=timeout_sec,
                    log=log,
                )
        except Exception as exc:  # noqa: BLE001 - advisory pass, never fatal
            error = _safe_exception_label(exc)

        revisions, parse_error = load_revisions(revisions_path)
        if not parse_error:
            # The SDK can report an error after the answer landed; the artifact
            # is what decides, exactly as the TraceLens skill runner does.
            return ReviewOutcome(
                status="completed",
                revisions=revisions,
                revisions_path=revisions_path,
            )
        last_detail = error or parse_error
        _say(f"attempt {attempt} unusable: {last_detail}")

    return ReviewOutcome(status="failed", detail=last_detail or "no revisions produced")


def run_candidate_review(
    *,
    run_dir: Path,
    raw_candidates_path: Path,
    reference_paths: dict[str, str],
    framework_roots: Sequence[str],
    context_block: str = "",
    timeout_sec: float = _DEFAULT_TIMEOUT_SEC,
    attempts: int = _DEFAULT_ATTEMPTS,
    log: Callable[[str], None] | None = None,
    session_runner: Callable[..., Any] | None = None,
) -> ReviewOutcome:
    """Synchronous entry point for :func:`run_candidate_review_async`.

    Owns the event loop, so it is only usable from a thread that has none. The
    guard below turns what would otherwise surface as an ``asyncio.run`` failure
    inside an advisory stage into a message naming the coroutine to await.

    Args:
        See :func:`run_candidate_review_async`.

    Returns:
        ReviewOutcome: The session result; never raises for session failures.

    Raises:
        RuntimeError: When called from a thread that already runs a loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run_candidate_review() owns an event loop; await run_candidate_review_async() from async callers"
        )
    return asyncio.run(
        run_candidate_review_async(
            run_dir=run_dir,
            raw_candidates_path=raw_candidates_path,
            reference_paths=reference_paths,
            framework_roots=framework_roots,
            context_block=context_block,
            timeout_sec=timeout_sec,
            attempts=attempts,
            log=log,
            session_runner=session_runner,
        )
    )


__all__ = [
    "ALLOWED_TOOLS",
    "DERIVED_SHAPE_FIELDS",
    "IMMUTABLE_FIELDS",
    "RAW_CANDIDATES_FILENAME",
    "REVIEW_BACKFILL_PROVENANCE",
    "REVIEW_DERIVED_PROVENANCE",
    "REVIEW_SHAPE_PROVENANCE",
    "REVISIONS_FILENAME",
    "ReviewOutcome",
    "apply_revisions",
    "build_review_prompt",
    "directory_fingerprint",
    "fingerprint_drift",
    "load_revisions",
    "run_candidate_review",
    "run_candidate_review_async",
    "source_fingerprint",
    "unfingerprinted_sources",
]
