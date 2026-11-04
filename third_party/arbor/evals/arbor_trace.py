"""arbor_trace.py — trajectory recorder for Arbor sessions.

Arbor's orchestrator is Claude Code itself, so there is no Python loop to
instrument. Everything the agent does arrives here through two channels:

  * **Hooks** (``hook`` subcommand) — Claude Code invokes this file on every
    tool call, prompt, subagent finish, and session boundary. Passive: the
    agent's prompt is untouched, so recording cannot change what is measured.
  * **Marks** (``mark`` subcommand) — four explicit calls in the slash-command
    markdown carry the intent that is not recoverable from the filesystem:
    which branch was picked, what the hypothesis was, how a gate resolved.

Design contract, inherited from Hyperloom's ``orchestrator/trace``:

  * **Best-effort I/O.** Every entry point swallows its own exceptions and
    exits 0. A recorder fault must never abort the run — a hook that raises
    can kill a five-hour GPU session, which is a far worse outcome than a
    missing trace line.
  * **Redacted before disk.** Tool inputs carry ``docker exec -e KEY=...``
    and the process environment carries a live gateway subscription key, so
    :func:`redact` runs on everything on the way out.
  * **Bounded.** A TraceLens dump or a profiling log is tens of megabytes.
    Large fields are head/tail clipped with the dropped byte count recorded.
  * **Stdlib only.** Matches bench.py's constraint: Python 3.10+, no
    third-party dependencies, because this runs on hosts where nothing is
    installed.

Written artifacts live under ``<session_dir>/reports/trace/``, mirroring
Hyperloom's layout so both harnesses join on one schema.
"""

from __future__ import annotations

# Import budget is load-bearing, not style. `hook` runs twice per tool call, in a
# fresh interpreter each time, so every module imported here is latency the user
# feels on every action the agent takes. `hashlib`, `subprocess`, and `typing`
# each cost ~11 ms to import and none is needed on the hook path, so they are
# imported inside the functions that actually use them. `Any` survives in
# annotations only because `from __future__ import annotations` leaves them
# unevaluated strings.
import fcntl
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Pointer written by ``mark session-start``. Hooks have no way to learn the
#: Arbor session directory on their own — the slash command creates it at run
#: time from a UTC timestamp — so the pointer is the handoff between them.
#:
#: Scoped by hostname because ``$HOME`` is NFS and shared across every compute
#: node: two campaigns running in parallel (say an Opus run and a Sonnet run)
#: would otherwise write the same file, and the loser would read the winner's
#: session_dir and file its trajectory into the other campaign's directory.
#: The host stamp inside the file catches the read, but only after the write
#: has already clobbered the other job's pointer.
POINTER_PATH = Path.home() / ".arbor" / f"current_session.{os.uname().nodename}"

#: Used when no pointer exists and ``session_root`` is not host-writable.
#: Losing the trajectory is worse than filing it under the Claude session id.
FALLBACK_ROOT = Path.home() / ".arbor" / "trajectories"

#: Per-field clip budget. Large enough to keep a full patch or a server log
#: tail intact, small enough that a profiling session's stdout cannot fill a
#: disk. Clipped fields keep head and tail plus a byte count of what went.
MAX_FIELD_BYTES = 32_768

#: Rotate ``events.jsonl`` past this size. A `/loop` session runs for days.
MAX_EVENTS_BYTES = 256 * 1024 * 1024

#: Mark kinds accepted by ``mark``. Closed so a typo surfaces as a rejected
#: mark rather than a silently unjoinable row.
VALID_MARK_KINDS = frozenset(
    {"session-start", "branch", "gate", "escalation", "note"}
)

#: Gate outcomes, matching the vocabulary in hyperloom-optimize.md.
VALID_DECISIONS = frozenset({"KEEP", "REVERT"})


# ---------------------------------------------------------------------------
# Redaction — ported verbatim from Hyperloom
# ---------------------------------------------------------------------------

# Source: hyperloom/orchestrator/trace/conversation_trace.py:69-86. Kept
# byte-identical so a value masked in one harness is masked in the other;
# divergence here would mean a credential that Hyperloom scrubs survives an
# Arbor run of the same session.
def _keep_numeric(match: "re.Match[str]") -> str:
    """Redact a labelled value unless it is purely a number."""
    label, value = match.group(1), match.group(0)[match.end(1) - match.start(0):]
    if re.fullmatch(r"[\d_,.]+", value):
        return match.group(0)
    return f"{label}[REDACTED]"


_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization: Bearer <token>
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-_.=]{8,}"), r"\1[REDACTED]"),
    # Provider key prefixes: ak-, sk-, pk-, sk-lf-, pk-lf- ...
    (re.compile(r"\b((?:ak|sk|pk)-(?:lf-)?)[A-Za-z0-9\-_]{6,}"), r"\1[REDACTED]"),
    # GitHub tokens: ghp_, gho_, ghs_, ghr_, github_pat_
    (re.compile(r"\b(gh[pousr]_|github_pat_)[A-Za-z0-9_]{10,}"), r"\1[REDACTED]"),
    # KEY=value / TOKEN=value / SECRET=value / PASSWORD=value (env shape)
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*"
            r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|AUTH|CREDENTIAL)"
            r"[A-Z0-9_]*\s*[=:]\s*)"
            r"[^\s,;'\"]+"
        ),
        # Callable, not a literal: a bare number is never a credential, but
        # "TOKEN" matches the field names token *counters* use. Observed in a
        # real trajectory as `subagent_tokens: [REDACTED]`, which quietly
        # destroyed the per-subagent usage the cost metrics want. Keep numeric
        # values, redact everything else.
        _keep_numeric,
    ),
)

# Arbor addition: the AMD gateway ships its key in a header rather than in any
# shape above, and that header is in this process's own environment — so it
# reaches disk through `env`, `docker exec -e`, and settings dumps unless it is
# named explicitly.
_REDACT_PATTERNS_ARBOR: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?i)(Ocp-Apim-Subscription-Key\s*[=:]\s*)[^\s,;'\"]+"),
        r"\1[REDACTED]",
    ),
)

_ALL_REDACT_PATTERNS = _REDACT_PATTERNS + _REDACT_PATTERNS_ARBOR


def redact(text: str) -> str:
    """Strip recognizable secret *values* from ``text`` before it hits disk.

    Conservative and idempotent: the label or prefix survives so a redacted
    line still reads sensibly, and only the secret material is replaced.
    """
    if not text:
        return text
    out = text
    for pattern, repl in _ALL_REDACT_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def redact_obj(obj: Any) -> Any:
    """Recursively redact every string inside a JSON-shaped value."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Bounding
# ---------------------------------------------------------------------------


def clip(text: str, limit: int = MAX_FIELD_BYTES) -> tuple[str, int]:
    """Head/tail clip ``text`` to ``limit`` bytes.

    Returns the clipped text and the number of bytes dropped (0 when it fit).
    Both ends are kept because a failing command's cause is usually at the top
    and its verdict at the bottom.
    """
    raw = text.encode("utf-8", "replace")
    if len(raw) <= limit:
        return text, 0
    half = limit // 2
    head = raw[:half].decode("utf-8", "ignore")
    tail = raw[-half:].decode("utf-8", "ignore")
    dropped = len(raw) - len(head.encode()) - len(tail.encode())
    return f"{head}\n\n... [{dropped} bytes clipped] ...\n\n{tail}", dropped


def clip_obj(obj: Any, limit: int = MAX_FIELD_BYTES) -> Any:
    """Clip every string inside a JSON-shaped value, recording drops."""
    if isinstance(obj, str):
        clipped, dropped = clip(obj, limit)
        return clipped if not dropped else clipped
    if isinstance(obj, dict):
        return {k: clip_obj(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clip_obj(v, limit) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Time / identity helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    """UTC, ISO-8601, microseconds + ``+00:00`` — matches Hyperloom's now_iso."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def digest(obj: Any) -> str:
    """Stable short hash of a JSON-shaped value, for dedup and change identity."""
    import hashlib  # off the hook path; see the import budget note at the top

    try:
        blob = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        blob = repr(obj)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]


def arbor_repo_sha(repo: Path | None = None) -> str:
    """HEAD sha of the Arbor checkout, or ``"unknown"``.

    Recorded per session because the prompt text in the ``.md`` commands *is*
    the thing under evaluation — a trajectory is only comparable to another
    when you know which revision of the instructions produced it.
    """
    import subprocess  # off the hook path; see the import budget note at the top

    repo = repo or Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _host_writable(path: Path, *, probe: bool = True) -> bool:
    """Whether the host process can create ``path``.

    Load-bearing: config.json points ``session_root`` at a *container* path
    (/workspace/...) while hooks run on the host, whereas the campaign configs
    in configs/mi355x/ point it at a host path. bench.py:_read_output copes
    with the same ambiguity via a docker-cp fallback; this is that pattern.

    ``probe=False`` accepts an existing directory without the create/unlink
    round trip. Hooks take that path: the question is settled once, when the
    session opens, and re-answering it on every tool call would spend three
    syscalls apiece to learn what has not changed.
    """
    try:
        if not probe and path.is_dir():
            return True
        path.mkdir(parents=True, exist_ok=True)
        marker = path / ".arbor_write_probe"
        marker.write_text("", encoding="utf-8")
        marker.unlink()
        return True
    except Exception:
        return False


#: A Claude session id is a UUID. Anything else became a directory name under
#: ~/.arbor/trajectories/ unchallenged — a stray test left one literally called
#: ``s``. Validate before turning caller-supplied text into a path.
_SESSION_ID_RE = re.compile(r"^[0-9a-fA-F-]{8,64}$")


def _safe_session_id(session_id: str | None) -> str:
    """Return a session id usable as a path component, or ``unknown``."""
    if session_id and _SESSION_ID_RE.match(session_id):
        return session_id
    return "unknown"


def read_pointer() -> dict[str, Any]:
    """Read the current-session pointer, or ``{}`` when absent/unreadable."""
    try:
        return json.loads(POINTER_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_trace_dir(claude_session_id: str = "unknown") -> Path:
    """Locate the directory this process should append to.

    Order: explicit env override, then the pointer written by
    ``mark session-start``, then a fallback under ``~/.arbor/trajectories``.
    The fallback is never an error — a trajectory filed under the Claude
    session id is recoverable, a dropped one is not.
    """
    override = os.environ.get("ARBOR_SESSION_DIR")
    if override:
        candidate = Path(override) / "reports" / "trace"
        if _host_writable(candidate, probe=False):
            return candidate

    ptr = read_pointer()
    # The pointer already records where `mark session-start` landed, including
    # the case where it had to fall back. Trusting it keeps the hook path at one
    # stat() instead of re-deciding the container-vs-host question every call.
    #
    # But only on the host that wrote it. A finished Slurm job leaves a pointer
    # naming a released compute node's /tmp; on the login node that path does
    # not exist, and the old code would happily *create* it and file unrelated
    # sessions there. A pointer from another machine is stale by construction.
    # A missing host counts as untrusted, not as trusted. Pointers written
    # before this field existed are exactly the stale ones left behind by
    # finished jobs, so treating absence as "probably fine" would leave the bug
    # in place for the only case that actually occurs.
    if ptr and ptr.get("host") != os.uname().nodename:
        ptr = {}
    trace_dir = ptr.get("trace_dir")
    if trace_dir and Path(trace_dir).is_dir():
        return Path(trace_dir)

    session_dir = ptr.get("session_dir")
    if session_dir:
        candidate = Path(session_dir) / "reports" / "trace"
        if _host_writable(candidate, probe=False):
            return candidate

    fallback = FALLBACK_ROOT / _safe_session_id(claude_session_id) / "reports" / "trace"
    # Same reasoning as _host_writable(probe=False): after the first event of a
    # session the directory exists, and paying a mkdir on every subsequent hook
    # buys nothing. append_jsonl creates parents anyway if this is skipped.
    if not fallback.is_dir():
        try:
            fallback.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    return fallback


# ---------------------------------------------------------------------------
# Append-only I/O
# ---------------------------------------------------------------------------


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON row under an exclusive lock. Never raises.

    The lock is not decorative: the escalation cycle in hyperloom-optimize.md
    dispatches three subagents in a single message, so their hooks fire
    concurrently against this file, and rows longer than PIPE_BUF would
    interleave without it.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > MAX_EVENTS_BYTES:
            path.rename(path.with_suffix(path.suffix + f".{int(datetime.now().timestamp())}"))
        line = json.dumps(row, default=str, ensure_ascii=False) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        # Best-effort by contract. A trace line is never worth a failed run.
        pass


def update_json(path: Path, mutate) -> None:
    """Read-modify-write a JSON file under an exclusive lock. Never raises."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read().strip()
                data = json.loads(raw) if raw else {}
                data = mutate(data)
                fh.seek(0)
                fh.truncate()
                fh.write(json.dumps(data, indent=2, default=str, ensure_ascii=False))
                fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Hook handling
# ---------------------------------------------------------------------------

#: Claude Code hook payload fields, confirmed by probing the installed CLI
#: (2.1.235) rather than assumed. PreToolUse and PostToolUse both carry
#: ``tool_use_id``, and PostToolUse carries ``duration_ms`` — which is why the
#: transcript join is exact and timings need no reconstruction.
_HOOK_EVENT_MAP = {
    "SessionStart": "session_start",
    "UserPromptSubmit": "user_prompt",
    "PreToolUse": "tool_call",
    "PostToolUse": "tool_result",
    "SubagentStop": "subagent_stop",
    "Stop": "turn_end",
    "SessionEnd": "session_end",
    "PreCompact": "compact",
}


def handle_hook(payload: dict[str, Any]) -> None:
    """Translate one hook payload into an ``events.jsonl`` row."""
    hook_name = payload.get("hook_event_name", "unknown")
    session_id = payload.get("session_id", "unknown")
    trace_dir = resolve_trace_dir(session_id)

    row: dict[str, Any] = {
        "ts": now_iso(),
        "event": _HOOK_EVENT_MAP.get(hook_name, hook_name.lower()),
        "hook_event_name": hook_name,
        "claude_session_id": session_id,
        "prompt_id": payload.get("prompt_id"),
        "cwd": payload.get("cwd"),
        "permission_mode": payload.get("permission_mode"),
        "effort": payload.get("effort"),
    }

    if hook_name in ("PreToolUse", "PostToolUse"):
        row["tool_use_id"] = payload.get("tool_use_id")
        row["tool_name"] = payload.get("tool_name")
        # No digest here: collect.py computes it from the transcript's copy of
        # the same input, so hashing on the hook path would buy nothing and cost
        # a hashlib import on every tool call.
        row["tool_input"] = clip_obj(redact_obj(payload.get("tool_input")))
    if hook_name == "PostToolUse":
        row["tool_response"] = clip_obj(redact_obj(payload.get("tool_response")))
        row["duration_ms"] = payload.get("duration_ms")
    if hook_name == "UserPromptSubmit":
        row["prompt"] = clip(redact(str(payload.get("prompt") or "")))[0]
    if hook_name == "SessionStart":
        row["source"] = payload.get("source")

    append_jsonl(trace_dir / "events.jsonl", row)

    # Every session that touches this Arbor run gets recorded, because a
    # /loop run spans restarts and each restart opens a new transcript file.
    # Without the full list, collect.py can only see the last leg.
    if hook_name in ("SessionStart", "PostToolUse"):
        _register_session(trace_dir, session_id, payload.get("transcript_path"))


def _register_session(trace_dir: Path, session_id: str, transcript_path: str | None) -> None:
    """Add this Claude session + transcript to meta.json's stitch list."""
    if not session_id or session_id == "unknown":
        return

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        sessions = data.setdefault("claude_sessions", [])
        for entry in sessions:
            if entry.get("session_id") == session_id:
                if transcript_path and not entry.get("transcript_path"):
                    entry["transcript_path"] = transcript_path
                entry["last_seen"] = now_iso()
                return data
        sessions.append({
            "session_id": session_id,
            "transcript_path": transcript_path,
            "first_seen": now_iso(),
            "last_seen": now_iso(),
        })
        return data

    update_json(trace_dir / "meta.json", mutate)


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------


def handle_mark(args) -> None:
    """Record one semantic anchor from a slash command."""
    if args.kind == "session-start":
        _mark_session_start(args)
        return

    trace_dir = resolve_trace_dir(os.environ.get("CLAUDE_SESSION_ID", "unknown"))
    row: dict[str, Any] = {
        "ts": now_iso(),
        "event": "mark",
        "kind": args.kind,
        "iteration": args.iter,
    }
    if args.kind == "branch":
        row.update({
            "branch_id": args.branch_id,
            "tier": args.tier,
            "hypothesis": redact(args.hypothesis or ""),
        })
    elif args.kind == "gate":
        decision = (args.decision or "").upper()
        row.update({
            "decision": decision if decision in VALID_DECISIONS else f"INVALID:{decision}",
            "tput": args.tput,
            "delta_pct": args.delta_pct,
            "correctness": args.correctness,
            "change": redact(args.change or ""),
        })
    elif args.kind == "escalation":
        row["null_streak"] = args.null_streak
    elif args.kind == "note":
        row["text"] = redact(args.text or "")

    append_jsonl(trace_dir / "events.jsonl", row)
    print(f"[arbor-trace] mark {args.kind} recorded -> {trace_dir}")


def _migrate_fallback_events(trace_dir: Path) -> int:
    """Fold events written before the pointer existed into the session.

    Hooks start firing at ``SessionStart``, which is necessarily before the
    agent can run ``mark session-start``. Those early rows land under
    ``~/.arbor/trajectories/<session_id>/`` and used to stay there: on a real
    run that stranded 29 events, costing 15 orchestrator steps their timing and
    permission mode. Moving them at open time closes the window.
    """
    moved = 0
    try:
        if not FALLBACK_ROOT.is_dir():
            return 0
        for src in FALLBACK_ROOT.glob("*/reports/trace/events.jsonl"):
            try:
                rows = [line for line in src.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
            except Exception:
                continue
            if not rows:
                continue
            dest = trace_dir / "events.jsonl"
            with open(dest, "a", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    for line in rows:
                        fh.write(line + "\n")
                    fh.flush()
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            moved += len(rows)
            # Rename rather than delete: if the migration was wrong, the
            # evidence is still on disk.
            src.rename(src.with_suffix(".jsonl.migrated"))
    except Exception:
        return moved
    return moved


def _mark_session_start(args) -> None:
    """Open a session: write the pointer, then seed meta.json.

    The pointer is what lets passive hooks find a session directory whose name
    is only decided at run time, so this mark is the one that must not be
    skipped — the other three degrade fidelity, this one decides whether the
    trajectory lands next to the session artifacts or in the fallback.
    """
    session_dir = Path(args.session_dir).resolve()
    trace_dir = session_dir / "reports" / "trace"
    writable = _host_writable(trace_dir)
    if not writable:
        trace_dir = FALLBACK_ROOT / session_dir.name / "reports" / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)

    try:
        POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        POINTER_PATH.write_text(
            json.dumps({
                "session_dir": str(session_dir if writable else trace_dir.parent.parent),
                "trace_dir": str(trace_dir),
                "started": now_iso(),
                "config_path": args.config,
                # Read back by resolve_trace_dir; a pointer written elsewhere
                # must not steer hooks on this machine.
                "host": os.uname().nodename,
            }, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    config: dict[str, Any] = {}
    if args.config:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except Exception:
            config = {"_error": f"could not read {args.config}"}

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data.setdefault("arbor_session_dir", str(session_dir))
        data.setdefault("started", now_iso())
        data["arbor_git_sha"] = arbor_repo_sha()
        data["config"] = redact_obj(config)
        data["host"] = os.uname().nodename
        data["claude_version"] = os.environ.get("CLAUDE_CODE_VERSION", "")
        data["model"] = os.environ.get("ANTHROPIC_MODEL", "")
        data["host_writable_session_root"] = writable
        data.setdefault("claude_sessions", [])
        # Recorded because it is a property of the MODEL, not of the run or the
        # collector: some models return no reasoning text on any capture route
        # while still billing the tokens. Probed 2026-08-24 against the gateway
        # directly — opus-5 withholds, sonnet-4.6 and haiku-4.5 do not.
        # See docs/archive/evals-README.md, "Reasoning text".
        data["thinking_capture"] = "withheld-by-model"
        return data

    update_json(trace_dir / "meta.json", mutate)

    migrated = _migrate_fallback_events(trace_dir)
    if migrated:
        print(f"[arbor-trace] migrated {migrated} pre-pointer event(s) from the fallback")

    print(f"[arbor-trace] session opened -> {trace_dir}")
    if not writable:
        print(
            f"[arbor-trace] NOTE: {session_dir} is not host-writable "
            f"(likely a container path); tracing to {trace_dir} instead."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_hook_fast() -> int:
    """The hook path, with no argparse.

    Called before ``main`` builds a parser, because argparse costs ~2 ms to
    import and the hook runs twice per tool call for the entire session.
    """
    try:
        raw = sys.stdin.read()
        handle_hook(json.loads(raw) if raw.strip() else {})
    except Exception:
        pass  # contract: a hook never fails the run
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="arbor_trace", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("hook", help="Read a Claude Code hook payload on stdin and record it")

    m = sub.add_parser("mark", help="Record a semantic anchor from a slash command")
    m.add_argument("kind", choices=sorted(VALID_MARK_KINDS))
    m.add_argument("--session-dir", help="Arbor session directory (session-start only)")
    m.add_argument("--config", help="Path to config.json (session-start only)")
    m.add_argument("--iter", type=int, default=None)
    m.add_argument("--branch-id")
    m.add_argument("--tier", choices=["config", "framework", "kernel", "feature"])
    m.add_argument("--hypothesis")
    m.add_argument("--decision", help="KEEP or REVERT")
    m.add_argument("--tput", type=float)
    m.add_argument("--delta-pct", type=float)
    m.add_argument("--correctness", choices=["pass", "fail"])
    m.add_argument("--change", help="Short description of the change gated")
    m.add_argument("--null-streak", type=int)
    m.add_argument("--text")

    c = sub.add_parser(
        "collect",
        help="Assemble + render + score a session (collect, render, report in one call)",
    )
    c.add_argument("--session-dir", required=True)
    c.add_argument("--transcript", action="append", default=None,
                   help="Explicit transcript JSONL; repeatable. Enables collecting a "
                        "session recorded before tracing was installed.")
    c.add_argument("--compact", action="store_true",
                   help="Clip large tool outputs in SESSION_LOG.md")

    sub.add_parser("where", help="Print the resolved trace directory and exit")

    args = parser.parse_args(argv)

    if args.cmd == "hook":
        return _run_hook_fast()

    if args.cmd == "mark":
        try:
            handle_mark(args)
        except Exception as exc:  # marks are user-invoked, so say something
            print(f"[arbor-trace] mark failed (continuing): {exc}", file=sys.stderr)
        return 0

    if args.cmd == "collect":
        # Imported here, not at module scope: `hook` runs on every tool call and
        # must not pay for the analysis modules it never touches.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from collect import collect as _collect
        from render import render as _render
        from report import render_report as _report

        session_dir = Path(args.session_dir).resolve()
        summary = _collect(session_dir, args.transcript)
        log = _render(session_dir, compact=args.compact)
        rep = _report(session_dir)
        print(
            f"[arbor-trace] {summary['steps']} steps, "
            f"{summary['tool_calls']} tool calls, "
            f"{summary['gates']} gates, "
            f"iterations {summary['iterations'] or '—'}"
        )
        print(f"[arbor-trace] log    {log}")
        print(f"[arbor-trace] report {rep}")
        return 0

    if args.cmd == "where":
        print(resolve_trace_dir())
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    # Hook dispatch bypasses main() entirely so the hot path never constructs an
    # argument parser it does not need.
    if len(sys.argv) == 2 and sys.argv[1] == "hook":
        sys.exit(_run_hook_fast())
    sys.exit(main())
