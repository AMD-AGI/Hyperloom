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

* **Proposals only.** The agent may revise where a kernel lives and whether it
  is worth dispatching. It may not touch what the trace measured -- GPU share,
  durations, shapes, argument specs. Those are evidence, and everything
  downstream (impact ranking, harness generation, the final report) is computed
  from them. :data:`IMMUTABLE_FIELDS` is enforced here, not requested in prose.
* **Nothing is taken on faith.** A revised path must exist under a known
  framework root. This is not a correctness check; it stops an invented path
  from being written.
* **Nothing is destroyed.** Every revision records ``previous_source_file`` and
  ``previous_method``, and the pre-review table is kept beside the reviewed one,
  so a bad review is auditable and reversible.

The session may run shell commands (demangling a mangled vendor symbol is
exactly the job), so the framework tree is fingerprinted before and after. A
review that modified the code under optimization is discarded rather than
applied: the benchmark that follows would otherwise measure an unrecorded edit.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from hyperloom.common import kernel_source_contract as _KSC
except ImportError:  # pragma: no cover - standalone invocation
    _KSC = None  # type: ignore[assignment]

#: Written by the agent; its presence is what marks the session successful.
REVISIONS_FILENAME = "kernel_candidates_revisions.json"

#: The pre-review table, kept so a bad review can be told from a bad parse.
RAW_CANDIDATES_FILENAME = "kernel_candidates.raw.json"

#: Measured by the trace. A revision naming any of these is rejected: the
#: impact ranking, the tuning harness and the final report are all computed
#: from them, so a plausible-looking edit here is indistinguishable from data.
IMMUTABLE_FIELDS: frozenset[str] = frozenset(
    {
        "call_count",
        "device_kernel_name",
        "duration_us",
        "gpu_pct",
        "input_dtypes",
        "input_shapes",
        "invocation_cases",
        "kernel_id",
        "name",
        "raw_arg_spec",
        "shape_provenance",
        "shapes",
    }
)

_ACTION_KEEP = "keep"
_ACTION_REWRITE = "rewrite"
_ACTION_UNRESOLVE = "unresolve"
_ACTION_DROP = "drop"
_ACTIONS = frozenset({_ACTION_KEEP, _ACTION_REWRITE, _ACTION_UNRESOLVE, _ACTION_DROP})

#: Read and search freely; run shell commands; write only the revision file.
#: ``Edit`` is withheld deliberately -- the agent proposes, it does not patch,
#: and the framework tree here is the code under optimization.
ALLOWED_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob", "Bash", "Write")

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


def source_fingerprint(paths: Sequence[str]) -> dict[str, list[Any]]:
    """Record size and mtime for each readable path.

    Scoped to the files the candidates actually name rather than the whole
    framework tree: those are the ones a source-resolution session has reason
    to open, and hashing gigabytes of installed packages to guard a few dozen
    files would cost more than the session it protects.

    Args:
        paths (Sequence[str]): Candidate source paths, absolute.

    Returns:
        dict[str, list[Any]]: ``{path: [size, mtime_ns]}`` for readable paths.
    """
    out: dict[str, list[Any]] = {}
    for raw in paths:
        path = str(raw or "").strip()
        if not path or path in out:
            continue
        try:
            stat = os.stat(path)
        except OSError:
            continue
        out[path] = [stat.st_size, stat.st_mtime_ns]
    return out


def fingerprint_drift(before: dict[str, list[Any]], after: dict[str, list[Any]]) -> list[str]:
    """Return the paths whose size or mtime changed between two fingerprints."""
    drifted: list[str] = []
    for path, signature in before.items():
        if after.get(path) != signature:
            drifted.append(path)
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
    """
    lines = [
        "Audit the kernel-candidate table produced by the deterministic "
        "analysis stage and correct it where the evidence disagrees.",
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
        "  - You may revise source_file, reusable_native_kernel, skip_reason and",
        "    recommended_backends. You may not revise anything the trace",
        "    measured (gpu_pct, duration_us, call_count, shapes, raw_arg_spec);",
        "    such fields are ignored if present.",
        "  - Entries you do not mention are left exactly as they are.",
        "",
        f"Write your answer to {revisions_path} as JSON:",
        '  {"revisions": [{"kernel_id": "k001", "action": "rewrite",',
        '                  "source_file": "/abs/path.py",',
        '                  "reusable_native_kernel": true,',
        '                  "skip_reason": "",',
        '                  "reason": "one sentence citing what you checked"}]}',
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
            "allowed_tools": list(ALLOWED_TOOLS),
            "disallowed_tools": list(_DENIED_TOOLS),
            "cwd": str(run_dir),
        }
    )
    try:
        options = sdk.ClaudeAgentOptions(**kwargs)
    except TypeError:
        kwargs.pop("cwd", None)
        options = sdk.ClaudeAgentOptions(**kwargs)

    async def _drive() -> None:
        async for message in sdk.query(prompt=prompt, options=options):
            if log is not None:
                for text in _message_text(message):
                    if text.strip():
                        log(f"[review-agent] {text.strip()[:400]}")

    try:
        await asyncio.wait_for(_drive(), timeout=max(60.0, timeout_sec))
    except Exception as exc:  # noqa: BLE001 - artifact presence decides success
        return _safe_exception_label(exc)
    return ""


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


def _acceptable_path(picked: str, roots: Sequence[str]) -> str:
    """Return the canonical form of ``picked``, or ``""`` when unverifiable."""
    if _KSC is None:
        return ""
    bare = _KSC.strip_line_suffix(picked)
    return _KSC.canonical_source_path(bare, tuple(roots)) or ""


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
    by_id = {
        str(c.get("kernel_id") or ""): c for c in candidates if isinstance(c, dict)
    }
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
        reason = str(revision.get("reason") or "").strip()
        if action == _ACTION_KEEP:
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
        if canonical == previous_bare:
            continue
        entry["previous_source_file"] = previous_file
        entry["previous_method"] = previous_method
        entry["source_file"] = canonical
        entry.pop("source_line", None)
        entry.pop("source_function", None)
        entry["source_resolution_method"] = "llm_review"
        entry["review_action"] = action
        entry["review_reason"] = reason or "no reason given"
        notes.append(f"{kernel_id}: {previous_file or '(none)'} -> {canonical}")

        proposed_skip = revision.get("skip_reason")
        if isinstance(proposed_skip, str):
            entry["review_skip_reason"] = proposed_skip.strip()
        proposed_reusable = revision.get("reusable_native_kernel")
        if isinstance(proposed_reusable, bool):
            entry["review_reusable_hint"] = proposed_reusable
    return notes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
    session_runner: Callable[..., str] | None = None,
) -> ReviewOutcome:
    """Run the review session and return its parsed revisions.

    Retries a failed session once by default: the pass is mandatory on the agent
    route, and a gateway hiccup should not be the reason a run's candidate table
    goes unaudited. A definitive failure is reported rather than raised -- the
    deterministic table is still usable, and killing a multi-hour optimization
    over an advisory pass would trade a small loss for a total one.

    Args:
        run_dir: Session directory; the only place the agent may write.
        raw_candidates_path: The pre-review candidate table to audit.
        reference_paths: Labelled artifact paths offered to the agent.
        framework_roots: Roots a revised path must resolve under.
        context_block: Rendered model/serving context, or ``""``.
        timeout_sec: Wall-clock bound per attempt.
        attempts: Total attempts, including the first.
        log: Optional diagnostics callback.
        session_runner: Injection point for the session call (tests).

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
                error = session_runner(
                    prompt=prompt,
                    run_dir=Path(run_dir),
                    model=model,
                    timeout_sec=timeout_sec,
                )
            elif backend == "codex":
                error = asyncio.run(
                    _run_codex_session(
                        prompt,
                        run_dir=Path(run_dir),
                        model=model,
                        timeout_sec=timeout_sec,
                    )
                )
            else:
                error = asyncio.run(
                    _run_claude_session(
                        prompt,
                        run_dir=Path(run_dir),
                        model=model,
                        timeout_sec=timeout_sec,
                        log=log,
                    )
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


__all__ = [
    "ALLOWED_TOOLS",
    "IMMUTABLE_FIELDS",
    "RAW_CANDIDATES_FILENAME",
    "REVISIONS_FILENAME",
    "ReviewOutcome",
    "apply_revisions",
    "build_review_prompt",
    "fingerprint_drift",
    "load_revisions",
    "run_candidate_review",
    "source_fingerprint",
]
