# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_unix
from hyperloom.common.timeutil import iso_z, now_iso

from ._common import (
    _benchmark_report_candidates,
    _benchmark_report_metrics,
    _find_benchmark_report,
    _latest_benchmark_report,
    _load_json_safe,
    _rel,
    _resolve_under_session,
    _to_float,
    _to_int,
)


log = logging.getLogger(__name__)


# Invocation-record env filter (allowlist + secret-pattern denylist).
# Only surface workload-influencing knobs; everything else (secrets, host
# fingerprints, shell aliases) is dropped from the breakdown JSON.
_ENV_ALLOWLIST_EXACT: frozenset[str] = frozenset(
    {
        "TP",
        "FRAMEWORK",
        "GPU_TYPE",
        "PRECISION",
        "CONC",
        "ISL",
        "OSL",
        "MAX_MODEL_LEN",
        "USER_DATA_PATH",
        "MODEL_PATH",
        "MODEL_NAME",
    }
)


_ENV_ALLOWLIST_PREFIXES: tuple[str, ...] = (
    "HYPERLOOM_",
    "VLLM_",
    "SGLANG_",
    "RAY_",
    "HSA_",
    "ROCM_",
    "TORCH_",
    "HF_",
)


# Strip credential-shaped keys even under allowlisted prefixes.
_ENV_DENY_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|AUTH|CREDENTIAL|COOKIE|API_KEY)",
    re.IGNORECASE,
)


# Non-secret keys kept verbatim even when the deny pattern matches a substring.
# ``MAGPIE_EVAL_TOKENIZED_REQUESTS`` records the accuracy eval's prompt wire
# format (``false`` => string prompts, forced on PD so the sglang_router does
# not 422; absent => lm_eval's default token-id prompts). Recording it lets a
# reader tell a PD run's accuracy from an aggregated one's -- but "TOKENIZED"
# contains "TOKEN", so the credential denylist would otherwise drop it.
_ENV_ALLOWLIST_FORCE: frozenset[str] = frozenset(
    {
        "MAGPIE_EVAL_TOKENIZED_REQUESTS",
    }
)


def _filter_envs(envs: dict[str, Any] | None) -> dict[str, str]:
    """Apply the allowlist + secret denylist; returns a fresh ``dict[str, str]`` with stringified values.

    Args:
        envs (dict[str, Any] | None): Raw environment mapping to filter, or
            ``None``.

    Returns:
        dict[str, str]: The allowlisted, secret-stripped subset with every
        value coerced to ``str``. Empty when ``envs`` is not a dict.
    """
    if not isinstance(envs, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in envs.items():
        if not isinstance(k, str):
            continue
        forced = k in _ENV_ALLOWLIST_FORCE
        keep = forced or (k in _ENV_ALLOWLIST_EXACT) or any(k.startswith(p) for p in _ENV_ALLOWLIST_PREFIXES)
        if not keep:
            continue
        # Force-listed keys are known non-secret; skip the substring denylist
        # (it would otherwise strip e.g. TOKENIZED for containing "TOKEN").
        if not forced and _ENV_DENY_PATTERN.search(k):
            continue
        out[k] = "" if v is None else str(v)
    return out


# framework-args extraction patterns.
# Pass-0 ("log_non_default_args"): vllm/sglang's ``non-default args: {...}``
# echo of the resolved parsed argv dict — the most authoritative source.
_FRAMEWORK_ARGS_NON_DEFAULT_RE = re.compile(
    r"non[-_]default args:\s*(\{.+\})\s*$",
    re.IGNORECASE,
)


# Pass-1 ("log_args_line"): parsed-launch-arg echo under a stable header.
_FRAMEWORK_ARGS_HEADER_RE = re.compile(
    r"^[^|]*?Server arguments?:\s*(.+)$",
    re.IGNORECASE,
)


_FRAMEWORK_ARGS_NAMESPACE_RE = re.compile(
    r"^[^|]*?Args:\s*Namespace\((.+)\)\s*$",
    re.IGNORECASE,
)


_FRAMEWORK_ARGS_LAUNCH_RE = re.compile(
    r"^[^|]*?Launch(?:ing)? server with:\s*(.+)$",
    re.IGNORECASE,
)


# Pass-2 ("log_python_cmd"): a literal python/vllm/sglang command in
# server.log, accepted after stripping the vllm/sglang log prefix.
_LOG_PREFIX_RE = re.compile(
    r"^\s*(?:\([^)]*\)\s+)?(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE)\s+"
    r"\d[\d:\-\s]*\[[^\]]+\]\s*",
    re.IGNORECASE,
)


_LOG_TIMESTAMP_RE = re.compile(
    r"^\s*\[?\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[^\]]*\]?\s*",
)


_PYTHON_CMD_PREFIXES: tuple[str, ...] = (
    "python",
    "python3",
    "vllm",
    "sglang.launch_server",
    "inference-optimizer",
    "ray",
)


# Server log size cap so the per-line scan stays bounded (covers the startup banner).
_SERVER_LOG_MAX_BYTES = 256 * 1024

# Tail kept for enablement trigger evidence: tracebacks put the real error last.
_ENABLEMENT_LOG_EXCERPT_CHARS = 2000


def _eg(state: dict, name: str, default: Any = None) -> Any:
    """Read an enablement round field from a v4 nested or v3 flat state dict."""
    nested = state.get("enablement")
    if isinstance(nested, dict):
        return nested.get(name, default)
    return state.get(f"enablement_{name}", default)


def _strip_log_prefix(line: str) -> str:
    """Strip a leading ``[ts] LEVEL [src.py:NN]`` style prefix from a log line.

    Args:
        line (str): A single raw log line.

    Returns:
        str: The line with any recognized timestamp / level prefix removed
        and surrounding whitespace stripped.
    """
    s = line
    s = _LOG_PREFIX_RE.sub("", s)
    s = _LOG_TIMESTAMP_RE.sub("", s)
    return s.strip()


def _starts_with_python_prefix(text: str) -> bool:
    """Report whether ``text`` begins with a known launch-command prefix.

    A prefix in :data:`_PYTHON_CMD_PREFIXES` counts only when it is a whole
    token — i.e. immediately followed by end-of-string, whitespace, ``-`` or
    ``.`` — so ``pythonic`` does not match ``python``.

    Args:
        text (str): Candidate command line (already stripped of any log
            prefix by the caller).

    Returns:
        bool: ``True`` when ``text`` starts with a recognized command token.
    """
    head = text.lstrip()
    for prefix in _PYTHON_CMD_PREFIXES:
        if head.startswith(prefix):
            tail = head[len(prefix) :]
            if not tail or tail[0] in (" ", "\t", "-", "."):
                return True
    return False


def _load_yaml_dict_safe(config_yaml: Path) -> dict | None:
    """Parse ``config_yaml`` to its top-level dict, or ``None`` on miss/failure. Never raises.

    Shared by both yaml passes so the file is read at most once.

    Args:
        config_yaml (Path): The YAML file to parse.

    Returns:
        dict | None: The decoded top-level mapping, or ``None`` when the file
        fails to read/parse, or the document is not a dict.
    """
    import yaml

    try:
        text = config_yaml.read_text(encoding="utf-8", errors="replace")
        data = yaml.safe_load(text)
    except (OSError, yaml.YAMLError) as exc:
        log.debug("failed to parse %s for framework_args: %r", config_yaml, exc)
        return None
    return data if isinstance(data, dict) else None


def _yaml_cmd_from_dict(data: dict) -> str:
    """Find a ``cmd`` / ``command`` / ``launch`` field (top-level or under ``benchmark``); ``""`` on miss.

    Args:
        data (dict): The parsed YAML config mapping.

    Returns:
        str: The first non-empty command string found, or ``""`` when none of
        the recognized fields hold one.
    """
    for key in ("cmd", "command", "launch"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    bench = data.get("benchmark")
    if isinstance(bench, dict):
        for key in ("cmd", "command"):
            val = bench.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def _yaml_benchmark_synthesis(data: dict) -> str:
    """Synthesize a readable arg string from a magpie ``benchmark.*`` dict (not a literal cmdline).

    Returns ``""`` unless both ``benchmark.framework`` and
    ``benchmark.model`` are non-empty.

    Args:
        data (dict): The parsed YAML config mapping.

    Returns:
        str: A space-joined ``key=value`` summary (framework / model plus any
        present precision / tp / gpu / envs), or ``""`` when framework or model
        is missing.
    """
    bench = data.get("benchmark")
    if not isinstance(bench, dict):
        return ""
    fw = bench.get("framework")
    model = bench.get("model")
    fw_ok = isinstance(fw, str) and fw.strip()
    model_ok = isinstance(model, str) and model.strip()
    if not fw_ok or not model_ok:
        return ""
    parts: list[str] = [f"framework={fw.strip()}", f"model={model.strip()}"]
    prec = bench.get("precision")
    if prec is not None and str(prec).strip():
        parts.append(f"precision={str(prec).strip()}")
    tp = bench.get("tp")
    if tp is not None and str(tp).strip():
        parts.append(f"tp={str(tp).strip()}")
    gpu_sel = bench.get("gpu_selection")
    if isinstance(gpu_sel, dict):
        for key in ("gpu_type", "gpu", "type"):
            v = gpu_sel.get(key)
            if v is not None and str(v).strip():
                parts.append(f"gpu={str(v).strip()}")
                break
    elif isinstance(gpu_sel, str) and gpu_sel.strip():
        parts.append(f"gpu={gpu_sel.strip()}")
    envs = bench.get("envs")
    if isinstance(envs, dict) and envs:
        env_pairs = " ".join(f"{k}={v}" for k, v in envs.items() if isinstance(k, str))
        if env_pairs:
            parts.append(f"envs=[{env_pairs}]")
    return " ".join(parts)


def _extract_framework_args(
    server_log: Path | None,
    config_yaml: Path | None = None,
) -> tuple[str, str]:
    """Best-effort extract the launch command for a benchmark variant; never raises.

    Returns ``(args_string, source)`` where ``source`` is one of, in
    priority order: ``log_non_default_args`` / ``log_args_line`` /
    ``log_python_cmd`` / ``yaml_cmd`` / ``yaml_benchmark`` (synthesized,
    not a literal cmdline) / ``unknown`` (empty args).

    Args:
        server_log (Path | None): The ``server.log`` to scan, or ``None``.
        config_yaml (Path | None): The variant config YAML used as a fallback
            source, or ``None``. Defaults to ``None``.

    Returns:
        tuple[str, str]: ``(args_string, source)`` — the extracted launch args
        and a provenance label, with ``("", "unknown")`` when no source yields
        anything.
    """
    chunk: str = ""
    if server_log is not None:
        try:
            if server_log.exists():
                with server_log.open("r", encoding="utf-8", errors="replace") as fh:
                    chunk = fh.read(_SERVER_LOG_MAX_BYTES)
        except OSError:
            chunk = ""

    lines = chunk.splitlines() if chunk else []

    # Pass 0: ``non-default args: {...}`` echo, parsed via ast.literal_eval;
    # a failed eval is treated as a miss.
    for line in lines:
        m = _FRAMEWORK_ARGS_NON_DEFAULT_RE.search(line)
        if not m:
            continue
        dict_text = m.group(1)
        try:
            parsed = ast.literal_eval(dict_text)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(parsed, dict):
            continue
        # Sorted-by-key repr() values: stable across runs, paths stay quoted.
        formatted = " ".join(f"{k}={parsed[k]!r}" for k in sorted(parsed.keys(), key=str))
        return formatted, "log_non_default_args"

    # Pass 1: stable launch-summary headers (survive preceding log noise).
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        m = _FRAMEWORK_ARGS_HEADER_RE.match(stripped)
        if m:
            return m.group(1).strip(), "log_args_line"
        m = _FRAMEWORK_ARGS_LAUNCH_RE.match(stripped)
        if m:
            return m.group(1).strip(), "log_args_line"
        m = _FRAMEWORK_ARGS_NAMESPACE_RE.match(stripped)
        if m:
            return f"Namespace({m.group(1).strip()})", "log_args_line"

    # Pass 2: a literal python/vllm/sglang command, after stripping the log prefix.
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _starts_with_python_prefix(stripped):
            return stripped, "log_python_cmd"
        cleaned = _strip_log_prefix(stripped)
        if cleaned and _starts_with_python_prefix(cleaned):
            return cleaned, "log_python_cmd"

    # Pass 3 + 4: yaml fallback — literal ``cmd:`` (Pass 3) beats the magpie ``benchmark.*`` synthesis (Pass 4).
    if config_yaml is not None:
        try:
            yaml_exists = config_yaml.exists()
        except OSError:
            yaml_exists = False
        if yaml_exists:
            data = _load_yaml_dict_safe(config_yaml)
            if isinstance(data, dict):
                cmd = _yaml_cmd_from_dict(data)
                if cmd:
                    return cmd, "yaml_cmd"
                synth = _yaml_benchmark_synthesis(data)
                if synth:
                    return synth, "yaml_benchmark"

    return "", "unknown"


def _read_invocation_envs(config_path: Path | None) -> dict[str, str]:
    """Read ``benchmark.envs`` (or top-level ``envs:``) from a variant config; allowlisted subset, never raises.

    Args:
        config_path (Path | None): The variant config YAML to read, or
            ``None``.

    Returns:
        dict[str, str]: The allowlisted, secret-stripped env subset. Empty when
        the path is missing or parsing fails.
    """
    if config_path is None:
        return {}
    try:
        if not config_path.exists():
            return {}
    except OSError:
        return {}
    import yaml

    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
        data = yaml.safe_load(text)
    except (OSError, yaml.YAMLError) as exc:
        log.debug("failed to parse invocation config %s: %r", config_path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    bench = data.get("benchmark") if isinstance(data.get("benchmark"), dict) else {}
    raw_envs = bench.get("envs") if isinstance(bench, dict) else None
    if raw_envs is None:
        raw_envs = data.get("envs")
    return _filter_envs(raw_envs if isinstance(raw_envs, dict) else None)


def _detect_image_for_session(manifest: dict[str, Any]) -> str | None:
    """Resolve the container image for ``collect_session``.

    Prefers the manifest field (the spawn-time image), then falls back to the
    env / mount-point chain the manifest helper uses. Kept separate from
    :func:`manifest._detect_image` to avoid an import cycle.

    Resolution order: manifest ``image`` field → ``HYPERLOOM_IMAGE`` /
    ``CONTAINER_IMAGE`` / ``IMAGE`` env vars → known image marker files →
    a ``unknown@<short-cgroup-id>`` derived from ``/proc/1/cgroup``.

    Args:
        manifest (dict[str, Any]): The parsed ``manifest.json`` dict.

    Returns:
        str | None: The resolved container image reference, or ``None`` when
        no source yields a value.
    """
    manifest_image = manifest.get("image") if isinstance(manifest, dict) else None
    if isinstance(manifest_image, str) and manifest_image.strip():
        return manifest_image.strip()
    for var in ("HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE"):
        val = (os.environ.get(var) or "").strip()
        if val:
            return val
    for marker in ("/etc/podinfo/image", "/etc/hyperloom-image"):
        try:
            p = Path(marker)
            if p.exists():
                txt = p.read_text(encoding="utf-8", errors="replace").strip()
                if txt:
                    return txt
        except OSError:
            continue
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists():
            for line in cgroup.read_text(encoding="utf-8", errors="replace").splitlines():
                if "docker" not in line and "containerd" not in line:
                    continue
                m = re.search(r"([0-9a-f]{12,64})", line)
                if m:
                    return f"unknown@{m.group(1)[:12]}"
    except OSError as exc:
        # /proc/1/cgroup may be unreadable; fall through to None.
        log.debug("cgroup-based image detection failed: %r", exc)
    return None


def _leg_start_ts(state: dict[str, Any], start_ts: str) -> str:
    """When the session's current run leg began.

    ``start_ts`` alone does not answer this. A resume re-anchors it only after
    a crash or a stop with a reason; a resume after a clean stop deliberately
    keeps it, so that ``--max-hours`` still counts from the original start.
    ``state.resumed_ts`` is stamped by every resume, so the later of the two is
    the boundary on both paths.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        start_ts (str): The session's resolved start (see
            :func:`collect_session`).

    Returns:
        str: The later of the two timestamps, or whichever one is parseable.
    """
    resumed_ts = str(state.get("resumed_ts") or "")
    dated = [(to_unix(ts), ts) for ts in (start_ts, resumed_ts)]
    parseable = [(at, ts) for at, ts in dated if at is not None]
    if not parseable:
        return start_ts
    return max(parseable)[1]


def _close_phase_stop_reason(state: dict[str, Any], *, leg_start_ts: str) -> tuple[str, str]:
    """Recover terminal reason/time from the current leg's CLOSE transition (next-best when ``state.stop_reason`` wasn't mirrored).

    A resume clears ``state.stop_reason`` and ``stop_ts`` but cannot clear the
    previous leg's CLOSE row, and that row is not evidence about the leg
    running now: honouring it reports a live session as having stopped, for
    the reason it stopped last time. A row from before the leg boundary is
    skipped whole -- reason and timestamp -- because the timestamp is stamped
    as the session's end even when the reason itself is not adopted, and the
    scan carries on so a history written out of order can still be answered
    from a row that does belong to this leg.

    A row is only disqualified on comparable evidence. When either timestamp
    is missing or unparseable the row stands, since the whole point of the
    fallback is a session whose reason never reached the state file.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        leg_start_ts (str): Start of the current leg (see
            :func:`_leg_start_ts`); ``""`` when the session recorded none.

    Returns:
        tuple[str, str]: ``(reason, ts)`` from the most recent CLOSE
        transition of the current leg, or ``("", "")`` when there is none.
    """
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        return "", ""
    leg_start = to_unix(leg_start_ts)
    for row in reversed(history):
        if not isinstance(row, dict):
            continue
        if str(row.get("to_phase") or "").strip().upper() != "CLOSE":
            continue
        reason = str(row.get("reason") or row.get("stop_reason") or row.get("exit_reason") or "").strip()
        ts = str(row.get("ts") or row.get("entered_ts") or "").strip()
        closed_at = to_unix(ts)
        if leg_start is not None and closed_at is not None and closed_at < leg_start:
            continue
        return reason, ts
    return "", ""


def _first_recorded_end(*candidates: Any) -> str:
    """The first candidate that reads as a timestamp, canonicalised to ``...Z``.

    A value that does not parse is no more an end time than a missing one:
    passed through it lands in ``ended_at_utc`` verbatim and collapses the
    measured duration to zero, where the next candidate (or the export clock)
    still answers.

    Args:
        *candidates (Any): Recorded end timestamps, best evidence first.

    Returns:
        str: The first parseable candidate, or ``""`` when none is.
    """
    for value in candidates:
        if to_unix(value) is not None:
            return iso_z(value)
    return ""


def _session_has_ended(stop_reason: Any) -> bool:
    """Whether a stop reason marks the session as no longer running.

    Args:
        stop_reason (Any): Raw ``stop_reason`` from a state or session section.

    Returns:
        bool: ``True`` once a non-blank stop reason has been recorded.
    """
    return bool(str(stop_reason or "").strip())


def _measured_duration_seconds(start_ts: Any, ended_at_utc: Any, stop_reason: Any) -> int | None:
    """Seconds the session ran, or ``None`` when no window can be established.

    A finished session is measured to its recorded end; only one still running
    may be measured up to now, since extrapolating a finished session grows its
    duration on every re-export and reads as a plausible number rather than as
    missing evidence.

    Args:
        start_ts (Any): Start of the window (see :func:`collect_session` for
            which start that is across a resume).
        ended_at_utc (Any): Recorded end of the window, if any.
        stop_reason (Any): Terminal reason; a non-blank one means the session
            is no longer running.

    Returns:
        int | None: Whole seconds between start and end, or ``None``.
    """
    start = to_unix(start_ts)
    if start is None:
        return None
    end = to_unix(ended_at_utc)
    if end is None and not _session_has_ended(stop_reason):
        end = datetime.now(timezone.utc).timestamp()
    if end is None or end <= start:
        return None
    return int(round(end - start))


def session_elapsed_minutes(session_section: dict[str, Any]) -> float:
    """Wall-clock minutes of the leg described by a resolved ``session`` section.

    Derived from the section's own timestamps rather than stored, so a section
    assembled from the live recorder's snapshot reports the same elapsed time
    as one built by :func:`collect_session`. ``session_meta`` measures the same
    window from the same fields; the two agree because both producers of the
    section carry those timestamps, not because either reads the other.

    Args:
        session_section (dict[str, Any]): A ``session`` section.

    Returns:
        float: Minutes elapsed, or ``0.0`` when no window can be established.
    """
    duration_s = _measured_duration_seconds(
        session_section.get("start_ts") or session_section.get("created_at_utc"),
        session_section.get("ended_at_utc"),
        session_section.get("stop_reason"),
    )
    return round(duration_s / 60.0, 2) if duration_s is not None else 0.0


def _should_use_close_stop_reason(stop_reason: str, close_stop_reason: str) -> bool:
    """Decide whether the CLOSE-phase stop reason should override the session's.

    Args:
        stop_reason: The session-level stop reason.
        close_stop_reason: The CLOSE-phase stop reason.

    Returns:
        ``True`` when the close reason is more specific — i.e. it is set and the
        session reason is empty, or the session merely timed out while the close
        reason did not.
    """
    if not close_stop_reason:
        return False
    if not stop_reason:
        return True
    return stop_reason == "time_exhausted" and close_stop_reason != "time_exhausted"


# Session metadata
def _collect_recovery(state: dict[str, Any]) -> dict[str, Any]:
    """Project SharedState's crash / interruption / resume signals.

    Folds crash / degraded-mode / pending-revalidation signals into the
    ``session.recovery`` block so a resumed run is not read as a clean monotonic
    one. Pure / best-effort: unparseable fields are skipped, never raised.

    Args:
        state (dict[str, Any]): Parsed ``state.json`` (SharedState-shaped).

    Returns:
        dict[str, Any]: The ``recovery`` block (see schema ``Recovery``).
    """
    crash_count = _to_int(state.get("crash_count")) or 0
    crash_ts_iso: list[str] = []
    raw_ts = state.get("crash_timestamps")
    if isinstance(raw_ts, list):
        for t in raw_ts:
            try:
                crash_ts_iso.append(datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat())
            except (TypeError, ValueError, OSError, OverflowError):
                continue

    last_exc: dict[str, Any] | None = None
    lte = state.get("last_tick_exception")
    if isinstance(lte, dict) and lte:
        # Drop the large traceback; keep the compact postmortem header.
        last_exc = {
            "tick": lte.get("tick"),
            "ts": lte.get("ts"),
            "stage": lte.get("stage"),
            "agent": lte.get("agent"),
            "type": lte.get("type"),
            "message": (str(lte.get("message") or "")[:500] or None),
        }

    resume_pending = bool(state.get("resume_pending_revalidation"))
    degraded = bool(state.get("degraded_mode"))
    recovered = bool(crash_count > 0 or crash_ts_iso or resume_pending or last_exc)
    return {
        "recovered": recovered,
        "crash_count": crash_count,
        "crash_timestamps": crash_ts_iso,
        "degraded_mode": degraded,
        "resume_pending_revalidation": resume_pending,
        "last_tick_exception": last_exc,
    }


def collect_session(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the session-identification + lifecycle section.

    Merges identifiers and timing from ``state`` and ``manifest`` (state
    taking precedence on overlapping fields), resolves the container image,
    and stamps ``ended_at_utc`` from the recorded stop timestamp only once a
    ``stop_reason`` is present -- one the state file carries, or one recovered
    from a CLOSE transition belonging to the current leg (see
    :func:`_close_phase_stop_reason`). When no image can be detected a warning
    is appended.

    ``elapsed_minutes`` runs from ``state.start_ts``, the same anchor
    ``--max-hours`` is counted against, to the recorded end (or to now while
    the run is still going), so the two stay comparable. A resume re-anchors
    ``start_ts`` only when the previous leg crashed or stopped for a recorded
    reason; after a clean stop it keeps the original start, and the elapsed
    time then spans the gap between the legs -- as the budget does. The
    manifest's ``created_at_utc`` names the first launch either way, and is
    the fallback start only for a session that never recorded one.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json`` (SharedState-shaped).
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: The session section (ids, timestamps, stop reason,
        elapsed minutes, host, image, code revision, pid, tick count, etc.).
    """
    start_ts = str(state.get("start_ts") or manifest.get("created_at_utc") or "")
    stop_reason = str(state.get("stop_reason") or "").strip()
    close_stop_reason, close_ts = _close_phase_stop_reason(state, leg_start_ts=_leg_start_ts(state, start_ts))
    if _should_use_close_stop_reason(stop_reason, close_stop_reason):
        stop_reason = close_stop_reason
    ended_at_utc = ""
    if _session_has_ended(stop_reason):
        # ``stop_ts`` is stamped once, when the reason is written, so a re-export
        # of a finished session keeps reporting the same end. The CLOSE
        # transition and the export clock are only next-best guesses.
        ended_at_utc = _first_recorded_end(state.get("stop_ts"), close_ts) or now_iso(timespec="seconds")
    image = _detect_image_for_session(manifest)
    if image is None:
        warnings.append("image: not configured (set HYPERLOOM_IMAGE env var)")
    section = {
        "session_id": str(state.get("session_id") or manifest.get("session_id") or ""),
        "claw_session_id": manifest.get("claw_session_id") or state.get("claw_session_id"),
        "sandbox_user_id": manifest.get("sandbox_user_id") or state.get("sandbox_user_id"),
        "created_at_utc": manifest.get("created_at_utc") or start_ts,
        "start_ts": start_ts,
        "ended_at_utc": ended_at_utc,
        "stop_reason": stop_reason,
        "max_minutes": int(state.get("max_minutes") or manifest.get("max_minutes") or 0),
        "elapsed_minutes": 0.0,
        "host": str(manifest.get("host") or ""),
        "image": image,
        "code_revision": str(manifest.get("code_revision") or ""),
        "pid": int(manifest.get("pid") or 0),
        "session_dir": str(session_dir),
        # USER_DATA_PATH root (the operator-chosen workspace base). Manifest is
        # snapshotted at session start; env is the in-process fallback.
        "user_data_path": str(
            manifest.get("user_data_path") or state.get("user_data_path") or os.environ.get("USER_DATA_PATH") or ""
        ),
        "tick_count": int(state.get("tick") or 0),
        # Crash / interruption / resume history.
        "recovery": _collect_recovery(state),
    }
    section["elapsed_minutes"] = session_elapsed_minutes(section)
    return section


def _session_duration_seconds(
    session_section: dict[str, Any],
    manifest: dict[str, Any],
) -> int:
    """How long the session ran, in whole seconds.

    Measures the same window as ``session.elapsed_minutes`` (see
    :func:`collect_session`) so the machine field and the human-readable one
    cannot disagree, then falls back to ``elapsed_minutes`` for callers that
    supply it and no usable timestamps.

    Args:
        session_section (dict[str, Any]): The resolved ``session`` dict.
        manifest (dict[str, Any]): Parsed ``manifest.json``.

    Returns:
        int: The duration, or ``0`` when it cannot be established.
    """
    duration_s = _measured_duration_seconds(
        session_section.get("start_ts") or session_section.get("created_at_utc") or manifest.get("created_at_utc"),
        session_section.get("ended_at_utc"),
        session_section.get("stop_reason"),
    )
    if duration_s is not None:
        return duration_s
    elapsed_min = session_section.get("elapsed_minutes")
    if isinstance(elapsed_min, (int, float)) and elapsed_min > 0:
        return int(round(elapsed_min * 60))
    return 0


# session_meta enrichment
def collect_session_meta(
    manifest: dict[str, Any],
    session_section: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the ``session_meta`` enrichment block.

    Emitted straight from the manifest + resolved ``session`` section; the CI
    step only gap-fills fields the sandbox could not know (e.g. ``category``).

    The duration is measured from the session's own timestamps rather than
    read from a sibling key. Two producers fill the ``session`` section -- the
    live recorder's snapshot and this module's collector -- and only the
    collector writes ``elapsed_minutes``, so a run recorded live reported a
    session that lasted zero seconds.

    Args:
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        session_section (dict[str, Any]): The already-built ``session`` dict.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: ``{code_revision, image, image_id,
        session_duration_seconds}``.
    """
    image = session_section.get("image")
    image_str = image if isinstance(image, str) and image.strip() else ""
    duration_s = _session_duration_seconds(session_section, manifest)
    return {
        "code_revision": str(manifest.get("code_revision") or ""),
        "image": image_str or None,
        "image_id": image_str.split("/")[-1] if image_str else "",
        "session_duration_seconds": duration_s,
    }


# Workload
def collect_workload(
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the workload-description section.

    Merges framework name / model / GPU / parallelism fields from ``state`` and
    ``manifest`` (state preferred) plus the workload knobs (``conc`` / ``isl``
    / ``osl`` / ``max_model_len`` / ``precision``) nested under
    ``manifest.workload``, and the optimization objective.

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        warnings (list[str]): Shared warnings list (unused here but kept for a
            uniform collector signature).

    Returns:
        dict[str, Any]: The workload section with coerced numeric knobs and a
        defaulted ``objective`` mapping.
    """
    wl = manifest.get("workload") or {}
    return {
        "framework_name": str(state.get("framework") or manifest.get("framework") or ""),
        "framework_version": str(manifest.get("framework_version") or ""),
        "model_name": str(state.get("model_name") or manifest.get("model_name") or ""),
        "model_path": str(state.get("model_path") or manifest.get("model_path") or ""),
        "model_class": str(state.get("model_class") or ""),
        "gpu_type": str(state.get("gpu_type") or manifest.get("gpu_type") or ""),
        "tp": _to_int(manifest.get("tp")),
        "conc": _to_int(wl.get("conc")),
        "isl": _to_int(wl.get("isl")),
        "osl": _to_int(wl.get("osl")),
        "max_model_len": _to_int(wl.get("max_model_len")),
        "precision": str(wl.get("precision") or ""),
        "objective": dict(manifest.get("objective") or {"kind": "time_only", "value": None}),
    }


# Model basics
def collect_model_info(
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the ``model_info`` section (state.model_info passthrough).

    The summary is computed once at launch (``cli_bootstrap`` →
    ``summarize_model_config``) and persisted on ``state.model_info``, so the
    breakdown just mirrors it verbatim. Returns ``{}`` when the field is absent
    (sessions whose state predates it) or empty (non-transformers models such
    as diffusion checkpoints, where the config.json could not be parsed); the
    frontend treats an empty object as "model info unavailable".

    Args:
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (unused here but kept for a
            uniform collector signature).

    Returns:
        dict[str, Any]: The model_info object, or ``{}`` when unavailable.
    """
    info = state.get("model_info")
    return dict(info) if isinstance(info, dict) else {}


# Baseline
def collect_baseline(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the baseline-measurement section.

    Resolves the baseline workspace (re-rooting container-style paths under
    ``session_dir``), reads ttft / e2el from its ``benchmark_report.json``,
    and — when state didn't resolve — falls back to a disk walk of
    ``runs/baseline/``. Reconstructs an attempts history from disk when
    ``state.baseline_attempts`` is empty, and extracts the launch invocation
    (framework args + envs) from the matching server.log / config yaml. Each
    fallback or extraction miss is recorded in ``warnings``.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: The baseline section including throughput, accuracy,
        ttft / e2el (with a ``ttft_e2el_source`` provenance label), the
        attempts history, failure streak, and the launch ``invocation``.
    """
    last_b = state.get("last_baseline") or {}
    workspace_str = last_b.get("workspace") or ""
    # Re-root container-style paths under the on-disk session_dir.
    workspace = _resolve_under_session(session_dir, workspace_str)
    if workspace_str and workspace is None:
        warnings.append(
            f"baseline workspace {workspace_str!r} does not resolve under {session_dir}; "
            "ttft_mean_ms / e2el_mean_ms will be null."
        )
    report_path = _find_benchmark_report(workspace) if workspace else None
    report = _load_json_safe(report_path, warnings) if report_path else None

    _, ttft, _tpot, e2el = _benchmark_report_metrics(report if isinstance(report, dict) else None)

    # When the state workspace doesn't resolve, fall back to the most recent
    # ``runs/baseline/<hash>/.../benchmark_report.json`` on disk.
    ttft_source: str | None = "state_workspace" if ttft is not None else None
    if ttft is None:
        baseline_root = session_dir / "runs" / "baseline"
        candidates = sorted(
            (
                p
                for task_dir in baseline_root.iterdir()
                if task_dir.is_dir()
                for p in _benchmark_report_candidates(task_dir)
            )
            if baseline_root.exists()
            else [],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            disk_report_path = candidates[0]
            disk_report = _load_json_safe(disk_report_path, warnings) or {}
            _, ttft_disk, _tpot, e2el_disk = _benchmark_report_metrics(disk_report)
            if ttft_disk is not None:
                ttft = ttft_disk
                if e2el is None and e2el_disk is not None:
                    e2el = e2el_disk
                if report_path is None:
                    report_path = disk_report_path
                ttft_source = "runs_baseline_disk"
                warnings.append(
                    "baseline.ttft_mean_ms reconstructed from runs/baseline/ disk walk; "
                    "state.last_baseline.workspace did not resolve."
                )
    if ttft_source is None:
        ttft_source = "unavailable"

    attempts = state.get("baseline_attempts") or []
    history: list[dict[str, Any]] = []
    for a in attempts:
        if not isinstance(a, dict):
            continue
        history.append(
            {
                "ts": a.get("ts"),
                "task_id": a.get("task_id"),
                "status": a.get("status"),
                "decision": a.get("decision"),
                "key_metric": _to_float(a.get("key_metric")),
                "workspace": a.get("workspace"),
                "error_class": a.get("error_class"),
                "error_excerpt": a.get("error_excerpt"),
                "stderr_tail": a.get("stderr_tail"),
                "stderr_log_path": a.get("stderr_log_path"),
                "extras": a.get("extras") or {},
            }
        )

    # Disk-walking fallback when state.baseline_attempts is empty; each
    # reconstructed entry is marked ``status="reconstructed"``.
    if not history:
        reconstructed = _reconstruct_baseline_attempts(session_dir, warnings)
        if reconstructed:
            history = reconstructed
            warnings.append(
                "baseline.attempts_history reconstructed from runs/baseline/ "
                "(state.baseline_attempts was empty); detail fields are partial."
            )

    config_path_raw = state.get("baseline_config_path") or None
    config_resolved = _resolve_under_session(session_dir, config_path_raw) if config_path_raw else None
    server_log_path: Path | None = None
    # Prefer the already-located report so the invocation matches ttft/e2el;
    # else fall back to the resolved workspace.
    if report_path is not None:
        candidate_log = report_path.parent / "server.log"
        if candidate_log.exists():
            server_log_path = candidate_log
    if server_log_path is None and workspace:
        bench_dirs = sorted(
            workspace.glob("benchmark_*/server.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if bench_dirs:
            server_log_path = bench_dirs[0]

    # When disk-walked, the matching config yaml usually sits near the report;
    # prefer it when state.baseline_config_path didn't resolve.
    if config_resolved is None and report_path is not None:
        for candidate in (
            report_path.parent / "baseline_config.with_envs.yaml",
            report_path.parent.parent / "baseline_config.with_envs.yaml",
            report_path.parent.parent.parent / "baseline_config.with_envs.yaml",
        ):
            if candidate.exists():
                config_resolved = candidate
                break

    args_str, args_source = _extract_framework_args(
        server_log_path,
        config_yaml=config_resolved,
    )
    invocation = {
        "framework_args": args_str,
        "framework_args_source": args_source,
        "extra_envs": _read_invocation_envs(config_resolved),
        "config_path": _rel(config_resolved, session_dir)
        if config_resolved
        else (config_path_raw if config_path_raw else None),
        "server_log_path": _rel(server_log_path, session_dir) if server_log_path else None,
    }
    if args_source == "unknown":
        warnings.append(
            "framework_args extraction failed for "
            f"{(_rel(server_log_path, session_dir) if server_log_path else 'no server.log')}; "
            "tried server.log + yaml"
        )

    from ... import framework_registry

    return {
        "throughput_tok_s_per_gpu": _to_float(state.get("baseline_tput")) or 0.0,
        # Framework-dependent unit (serving = tok/s, scriptable xDiT = img/s).
        "throughput_unit": framework_registry.throughput_unit(state.get("framework")),
        "accuracy": _to_float(state.get("baseline_accuracy")) or 0.0,
        "ttft_mean_ms": ttft,
        "e2el_mean_ms": e2el,
        "ttft_e2el_source": ttft_source,
        "config_path": config_path_raw,
        "benchmark_report_path": _rel(report_path, session_dir) if report_path else None,
        "attempts_history": history,
        "failure_streak": int(state.get("baseline_failure_streak") or 0),
        # ALL baseline failures regardless of error_class.
        "total_failures": int(state.get("baseline_total_failures") or 0),
        "invocation": invocation,
        # Baseline-arm roofline ceiling backup; frontend fallback. {} when absent.
        "roofline_ceiling": state.get("baseline_roofline_ceiling") or {},
    }


def _reconstruct_baseline_attempts(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Walk ``<sd>/runs/baseline/<hash>/**/benchmark_report.json`` and
    synthesize :class:`BaselineAttemptSummary` rows for each.

    Used when ``state.baseline_attempts`` is empty but the on-disk
    runs/baseline/ tree shows that baseline ran (one or many times).
    Reads only what we can be certain of; everything else stays empty.

    Args:
        session_dir (Path): Absolute session root.
        warnings (list[str]): Shared warnings list (mutated in place when a
            report fails to parse).

    Returns:
        list[dict[str, Any]]: One reconstructed attempt row per discovered
        report (each marked ``status="reconstructed"``), ordered by mtime.
        Empty when no ``runs/baseline/`` tree exists.
    """
    root = session_dir / "runs" / "baseline"
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    reports = sorted(
        (p for task_dir in root.iterdir() if task_dir.is_dir() for p in _benchmark_report_candidates(task_dir)),
        key=lambda p: p.stat().st_mtime,
    )
    for report_path in reports:
        bench_dir = report_path.parent
        # task dir = <sd>/runs/baseline/<HASH>; reports may be direct or under *_round/.
        try:
            task_dir = root / report_path.relative_to(root).parts[0]
        except (ValueError, IndexError):
            task_dir = bench_dir.parent
        ts_iso = ""
        # Prefer a parseable ``benchmark_<UTC>`` suffix, else mtime.
        m = re.search(r"benchmark_(\d{8})[T_](\d{6})", bench_dir.name)
        if m:
            try:
                dt = datetime.strptime(
                    f"{m.group(1)}T{m.group(2)}",
                    "%Y%m%dT%H%M%S",
                ).replace(tzinfo=timezone.utc)
                ts_iso = dt.isoformat(timespec="seconds")
            except ValueError:
                ts_iso = ""
        if not ts_iso:
            try:
                ts_iso = datetime.fromtimestamp(
                    report_path.stat().st_mtime,
                    tz=timezone.utc,
                ).isoformat(timespec="seconds")
            except OSError:
                ts_iso = ""
        report = _load_json_safe(report_path, warnings)
        out_tput, _ttft, _tpot, _e2el = _benchmark_report_metrics(report if isinstance(report, dict) else None)
        out.append(
            {
                "ts": ts_iso,
                "task_id": task_dir.name,
                "status": "reconstructed",
                "decision": "",
                "key_metric": out_tput,
                "workspace": _rel(task_dir, session_dir) or str(task_dir),
                "error_class": None,
                "error_excerpt": None,
                "stderr_tail": None,
                "stderr_log_path": None,
                # Reconstruction reads the on-disk reports; audit extras are state-only.
                "extras": {},
            }
        )
    return out


# Final (validated)
def collect_final(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the final (validated) configuration section.

    Reads ``current_best`` plus the validated cumulative-gain bookkeeping,
    builds the ordered ``action_path`` from ``optimization_stack``, and — when
    ``current_best`` lacks ttft / e2el — reconstructs them from disk (latest
    ``validate_stack`` report first, then ``current_best``'s own workspace,
    then the top stack entry's report),
    recording the provenance in ``ttft_e2el_source`` and a ``warnings`` note.
    Also assembles the replayable launch ``invocation``.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: The final section (throughput, validated cumulative
        gain, stack-length bookkeeping, action path, ttft / e2el, invocation,
        and closing-phase markers).
    """
    cb = state.get("current_best") or {}
    stack = state.get("optimization_stack") or []
    stack_len = len(stack) if isinstance(stack, list) else 0
    val_stack_len = int(state.get("cumulative_gain_validated_stack_len") or 0)

    action_path: list[str] = []
    for entry in stack if isinstance(stack, list) else []:
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action") or "")
        variant = str(entry.get("variant_name") or "")
        action_path.append(f"{action}:{variant}" if variant else action)

    ttft = _to_float(cb.get("ttft_mean_ms"))
    e2el = _to_float(cb.get("e2el_mean_ms"))
    ttft_e2el_source = "current_best" if (ttft is not None or e2el is not None) else "unavailable"

    # Disk-walk reconstruction when a latency metric is unset: validate_stack
    # first (authoritative), then current_best, then stack top. Gate on EITHER
    # metric so xDiT diffusion (ttft meaningless, e2el meaningful) is covered.
    reconstructed_report: Path | None = None
    if ttft is None or e2el is None:
        reconstructed_report = _find_latest_validate_stack_report(session_dir)
        if reconstructed_report is not None:
            ttft_e2el_source = "validate_stack_disk"
        else:
            reconstructed_report = _find_current_best_report(session_dir, state)
            if reconstructed_report is not None:
                ttft_e2el_source = "current_best_disk"
            else:
                reconstructed_report = _find_stack_top_report(session_dir, state)
                if reconstructed_report is not None:
                    ttft_e2el_source = "stack_top_disk"
    if reconstructed_report is not None:
        report = _load_json_safe(reconstructed_report, warnings)
        if isinstance(report, dict):
            _, ttft_disk, _tpot, e2el_disk = _benchmark_report_metrics(report)
            if ttft is None and ttft_disk is not None:
                ttft = ttft_disk
            if e2el is None and e2el_disk is not None:
                e2el = e2el_disk
        warnings.append(
            f"final.ttft_mean_ms reconstructed from {ttft_e2el_source}; "
            "current_best did not record it (Coordinator gap)"
        )

    invocation = _build_final_invocation(
        session_dir,
        state,
        reconstructed_report,
        warnings,
    )
    from hyperloom.orchestrator.source_snapshot import source_layer_reproducible

    source_layers = [
        {
            "id": str(e.get("variant_name") or e.get("name") or ""),
            "snapshot_dir": str(e.get("source_snapshot") or ""),
            "framework_root": str(e.get("framework_root") or ""),
            "base_sha": str(e.get("base_sha") or ""),
            "reproducible": source_layer_reproducible(e),
        }
        for e in (stack if isinstance(stack, list) else [])
        if isinstance(e, dict) and e.get("scope") == "source_patch"
    ]
    if source_layers:
        invocation["source_layers"] = source_layers

    from ... import framework_registry

    return {
        "throughput_tok_s_per_gpu": _to_float(cb.get("tput")),
        # True throughput unit (tok/s vs img/s for scriptable xDiT).
        "throughput_unit": framework_registry.throughput_unit(state.get("framework")),
        # Which field holds the primary result (e2el_mean_ms vs throughput).
        "primary_metric": framework_registry.primary_metric_name(state.get("framework")),
        "cumulative_gain_pct_validated": _to_float(state.get("cumulative_gain_validated")) or 0.0,
        "revalidation_pending": bool(state.get("resume_pending_revalidation") or False),
        # A GEAK e2e candidate whose self-reported win is not yet confirmed by a
        # main-flow rebench; surfaced as an audit-only note and EXCLUDED from the
        # headline gain. Empty on native/validated sessions.
        "geak_pending": (dict(state.get("geak_pending") or {}) if isinstance(state.get("geak_pending"), dict) else {}),
        "validated_at_stack_len": val_stack_len,
        "validated_ts": str(state.get("cumulative_gain_validated_ts") or ""),
        "stack_changed_after_validation": stack_len > val_stack_len > 0,
        "extra_server_args": str(cb.get("extra_server_args") or ""),
        "extra_envs": dict(cb.get("extra_envs") or {}),
        "action_path": action_path,
        "ttft_mean_ms": ttft,
        "e2el_mean_ms": e2el,
        "ttft_e2el_source": ttft_e2el_source,
        "invocation": invocation,
        "closing_phase_entered": bool(state.get("closing_started_unix") or 0),
        "closing_started_unix": float(state.get("closing_started_unix") or 0.0),
        "closing_report_task_id": str(state.get("closing_report_task_id") or ""),
    }


def _find_latest_validate_stack_report(session_dir: Path) -> Path | None:
    """Most-recent validate_stack benchmark_report.json (authoritative for the final stack's clock).

    Args:
        session_dir (Path): Absolute session root.

    Returns:
        Path | None: The newest ``benchmark_report.json`` under
        ``runs/validate_stack/``, or ``None`` when none exist.
    """
    root = session_dir / "runs" / "validate_stack"
    if not root.exists():
        return None
    return _latest_benchmark_report(
        p for task_dir in root.iterdir() if task_dir.is_dir() for p in _benchmark_report_candidates(task_dir)
    )


def _find_current_best_report(
    session_dir: Path,
    state: dict[str, Any],
) -> Path | None:
    """Best-effort benchmark report for ``state.current_best`` (via workspace or action/variant/tput match).

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        Path | None: The matched report, or ``None`` when ``current_best`` is
        missing or no report resolves.
    """
    cb = state.get("current_best") or {}
    if not isinstance(cb, dict):
        return None
    workspace = _resolve_under_session(session_dir, cb.get("workspace"))
    if workspace is not None:
        report = _find_benchmark_report(workspace)
        if report is not None:
            return report
    return _find_matching_action_report(session_dir, cb)


def _find_stack_top_report(
    session_dir: Path,
    state: dict[str, Any],
) -> Path | None:
    """Last optimization_stack entry's benchmark_report.json (next-best when no validate_stack run exists).

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        Path | None: The top stack entry's report (via workspace or
        action/variant/tput match), or ``None`` when the stack is empty or
        nothing resolves.
    """
    cb = state.get("current_best") or {}
    stack = state.get("optimization_stack") or []
    if not stack and isinstance(cb, dict):
        stack = cb.get("optimization_stack") or []
    if not isinstance(stack, list) or not stack:
        return None
    last = stack[-1] if isinstance(stack[-1], dict) else None
    if last is None:
        return None
    workspace_str = last.get("workspace") or ""
    if not workspace_str:
        return _find_matching_action_report(session_dir, last)
    workspace = _resolve_under_session(session_dir, workspace_str)
    if workspace is None:
        return _find_matching_action_report(session_dir, last)
    report = _find_benchmark_report(workspace)
    if report is not None:
        return report
    return _find_matching_action_report(session_dir, last)


def _find_matching_action_report(
    session_dir: Path,
    entry: dict[str, Any],
) -> Path | None:
    """Match a report under ``runs/<action>/`` by variant name and tput (conservative; tput beats variant, latency-less reports skipped).

    Args:
        session_dir (Path): Absolute session root.
        entry (dict[str, Any]): A stack / current_best entry carrying
            ``action``, ``variant_name``, and ``tput``.

    Returns:
        Path | None: The best-scoring matching report, or ``None`` when the
        action is missing or no candidate matches.
    """
    action = str(entry.get("action") or "").strip()
    if not action:
        return None
    root = session_dir / "runs" / action
    if not root.exists():
        return None

    variant = str(entry.get("variant_name") or entry.get("name") or "").strip().lower()
    target_tput = _to_float(entry.get("tput"))
    scored: list[tuple[int, float, Path]] = []
    for report_path in root.rglob("benchmark_report.json"):
        rel = report_path.relative_to(root).as_posix().lower()
        variant_match = bool(variant and variant in rel)
        report = _load_json_safe(report_path, [])
        out_tput, ttft, _tpot, e2el = _benchmark_report_metrics(report if isinstance(report, dict) else None)
        if ttft is None and e2el is None:
            continue
        tput_match = False
        if target_tput is not None and out_tput is not None:
            tolerance = max(abs(target_tput) * 0.005, 1e-6)
            tput_match = abs(out_tput - target_tput) <= tolerance
        if not (tput_match or variant_match):
            continue
        score = (2 if tput_match else 0) + (1 if variant_match else 0)
        scored.append((score, report_path.stat().st_mtime, report_path))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def _build_final_invocation(
    session_dir: Path,
    state: dict[str, Any],
    benchmark_report: Path | None,
    warnings: list[str],
) -> dict[str, Any]:
    """Best-effort :class:`BenchmarkInvocation` for the final stack run (config = the launched ``*.with_envs.yaml`` sibling).

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        benchmark_report (Path | None): The resolved final benchmark report
            whose siblings supply the config / server log, or ``None``.
        warnings (list[str]): Shared warnings list (mutated in place when args
            extraction fails).

    Returns:
        dict[str, Any]: The invocation dict (framework args + source, extra
        envs, and relative config / server-log paths).
    """
    config_path: Path | None = None
    server_log_path: Path | None = None
    if benchmark_report is not None:
        bench_dir = benchmark_report.parent
        task_dir = bench_dir.parent
        for candidate in (
            bench_dir / "baseline_config.with_envs.yaml",
            task_dir / "baseline_config.with_envs.yaml",
            task_dir.parent / "baseline_config.with_envs.yaml",
        ):
            if candidate.exists():
                config_path = candidate
                break
        log_candidate = bench_dir / "server.log"
        if log_candidate.exists():
            server_log_path = log_candidate
    args_str, args_source = _extract_framework_args(
        server_log_path,
        config_yaml=config_path,
    )
    if args_source == "unknown":
        warnings.append(
            "framework_args extraction failed for "
            f"{(_rel(server_log_path, session_dir) if server_log_path else 'no server.log')}; "
            "tried server.log + yaml"
        )
    return {
        "framework_args": args_str,
        "framework_args_source": args_source,
        "extra_envs": _read_invocation_envs(config_path),
        "config_path": _rel(config_path, session_dir) if config_path else None,
        "server_log_path": _rel(server_log_path, session_dir) if server_log_path else None,
    }


def _as_int(value: Any, *, default: int = 0) -> int:
    """Coerce a state counter to int, falling back to ``default``."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _stack_action_summary(action: dict[str, Any]) -> dict[str, Any]:
    """Project a stack-action dict onto EnablementStackActionSummary."""
    return {
        "kind": str(action.get("kind") or ""),
        "framework": str(action.get("framework") or ""),
        "capability": str(action.get("capability") or ""),
        "acquisition_method": str(action.get("acquisition_method") or ""),
        "repo_url": str(action.get("repo_url") or ""),
        "ref": str(action.get("ref") or ""),
        "index_url": str(action.get("index_url") or ""),
        "reason": str(action.get("reason") or ""),
    }


def _runtime_summary(runtime: dict[str, Any], *, promoted: bool) -> dict[str, Any]:
    """Project a FrameworkRuntime-shaped dict onto EnablementAttemptRuntime."""
    versions = runtime.get("installed_versions")
    return {
        "venv_root": str(runtime.get("venv_root") or ""),
        "bin_path": str(runtime.get("bin_path") or ""),
        "python_path": str(runtime.get("python_path") or ""),
        "installed_versions": {str(k): str(v) for k, v in versions.items()} if isinstance(versions, dict) else {},
        "promoted": bool(promoted),
    }


def _build_attempt_summary(manifest_entry: dict[str, Any]) -> dict[str, Any]:
    """Project a BuildResult.to_state() entry onto TargetedBuildAttemptSummary."""
    action = manifest_entry.get("action") or {}
    installed = manifest_entry.get("installed_versions") or {}
    probes = manifest_entry.get("build_probes") or []
    return {
        "component": str(action.get("component") or manifest_entry.get("component") or ""),
        "ref": str(
            installed.get("aiter_ref")
            or installed.get("vllm_ref")
            or installed.get("sgl_kernel_ref")
            or action.get("ref")
            or ""
        ),
        "gpu_arch": str(installed.get("arch") or action.get("gpu_arch") or ""),
        "max_jobs": int(action.get("max_jobs") or 0),
        "ok": bool(manifest_entry.get("ok")),
        "failure_class": str(manifest_entry.get("failure_class") or "ok"),
        "failure_summary": str(manifest_entry.get("failure_summary") or ""),
        "installed_versions": {str(k): str(v) for k, v in installed.items()} if isinstance(installed, dict) else {},
        "build_probes": [str(p) for p in probes[:8]],
        "build_log_path": str(manifest_entry.get("build_log_path") or ""),
        "attempt_root": str(manifest_entry.get("attempt_root") or ""),
    }


def collect_enablement(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the enablement observability section.

    Covers the whole subsystem, not just the artifacts it happens to leave
    behind: the admitted lane, the round lifecycle (dispatch / attempts / stall /
    outcome), the boot- or eval-origin trigger, the patches and stack actions it
    landed, the attempt runtimes it provisioned, and the targeted builds it ran.

    A boot-origin round repaired by a plain source patch provisions no runtime
    and builds nothing, so gating emission on those artifacts alone made the most
    common kind of enablement invisible. Emission is therefore keyed on the lane
    having done something, or on it having been explicitly turned off — with
    ``all`` the default, "armed but never needed" is the uninteresting case and
    stays hidden, while "opted out" explains why nothing tried to repair a run
    that failed to establish a baseline.
    """
    active_runtime_raw = _eg(state, "active_runtime")
    attempt_runtimes_raw = _eg(state, "attempt_runtimes")
    failure_kind = str(_eg(state, "failure_kind", "") or "")
    build_manifest_raw = _eg(state, "build_manifest")
    last_build_failure_raw = _eg(state, "last_build_failure")
    kept_patches_raw = _eg(state, "kept_patches")
    kept_stack_action_raw = _eg(state, "kept_stack_action")

    origin = str(_eg(state, "origin", "") or "")
    # eval_kind is NOT cleared on success, so it can identify an eval-origin
    # enablement even after the run succeeds and origin is reset to "".
    eval_kind = str(_eg(state, "baseline_eval_kind", "") or "")
    # Sessions predating the flag load with the SharedState default, so that is
    # also the right value to report for them.
    mode = str(state.get("enablement_mode") or "all").strip().lower() or "all"
    attempts = _as_int(_eg(state, "attempts"))
    dispatched = bool(_eg(state, "inflight_task_id"))
    have_active = isinstance(active_runtime_raw, dict) and bool(active_runtime_raw)
    have_attempts = isinstance(attempt_runtimes_raw, list) and bool(attempt_runtimes_raw)
    have_build_manifest = isinstance(build_manifest_raw, list) and bool(build_manifest_raw)
    have_last_failure = isinstance(last_build_failure_raw, dict) and bool(last_build_failure_raw)
    have_kept_patches = isinstance(kept_patches_raw, list) and bool(kept_patches_raw)
    # Detect eval-origin by active origin OR persisted kind from a completed run.
    have_eval = origin == "eval" or bool(eval_kind)
    engaged = bool(attempts > 0 or dispatched or have_kept_patches or have_eval)
    if not (engaged or mode == "off" or have_active or have_attempts or have_build_manifest or have_last_failure):
        return {}

    out: dict[str, Any] = {
        "mode": mode,
        "engaged": engaged,
        "origin": "eval" if have_eval else "boot",
        "attempts": attempts,
        "dispatched": dispatched,
        "succeeded": bool(_eg(state, "succeeded")),
        "pending": bool(_eg(state, "pending")),
        "validation_pending": bool(_eg(state, "validation_pending")),
        "stall_streak": _as_int(_eg(state, "stall_streak")),
    }
    inflight_tid = str(_eg(state, "inflight_task_id", "") or "")
    if inflight_tid:
        out["inflight_task_id"] = inflight_tid
    last_spec_tid = str(_eg(state, "last_specialist_task_id", "") or "")
    if last_spec_tid:
        out["last_specialist_task_id"] = last_spec_tid
    reval_gen = _as_int(_eg(state, "revalidation_generation"))
    if reval_gen:
        out["revalidation_generation"] = reval_gen
    reval_tid = str(_eg(state, "revalidation_task_id", "") or "")
    if reval_tid:
        out["revalidation_task_id"] = reval_tid
    # The boot-origin trigger evidence: without it a launch-failure round shows
    # no reason for having run at all.
    launch_log = str(_eg(state, "launch_log", "") or "")
    if launch_log:
        out["launch_log_excerpt"] = launch_log[-_ENABLEMENT_LOG_EXCERPT_CHARS:]
    if have_kept_patches:
        out["kept_patches"] = [_rel(Path(str(p)), session_dir) or str(p) for p in kept_patches_raw]
    kept_artifacts_raw = _eg(state, "kept_artifacts")
    if isinstance(kept_artifacts_raw, list) and kept_artifacts_raw:
        out["kept_artifacts"] = [
            {
                "target": str(a.get("target") or ""),
                "rel_target": str(a.get("rel_target") or ""),
                "kind": str(a.get("kind") or ""),
            }
            for a in kept_artifacts_raw
            if isinstance(a, dict) and a.get("target")
        ]
    framework_root = str(_eg(state, "framework_root", "") or "")
    if framework_root:
        out["framework_root"] = framework_root
    if isinstance(kept_stack_action_raw, dict) and kept_stack_action_raw:
        out["kept_stack_action"] = _stack_action_summary(kept_stack_action_raw)
    candidate_refs = _eg(state, "candidate_refs")
    if isinstance(candidate_refs, list) and candidate_refs:
        out["candidate_refs"] = [str(r) for r in candidate_refs]
    setup_commands = _eg(state, "setup_commands")
    if isinstance(setup_commands, list) and setup_commands:
        out["setup_commands"] = [str(c) for c in setup_commands]
    localization = _eg(state, "localization_manifest")
    if isinstance(localization, list) and localization:
        out["localization_manifest"] = [str(p) for p in localization]
    build_novelty = _eg(state, "build_novelty")
    if isinstance(build_novelty, list) and build_novelty:
        out["build_novelty"] = [str(k) for k in build_novelty]
    human_review = _eg(state, "human_review_logged")
    if isinstance(human_review, list) and human_review:
        out["human_review_count"] = len(human_review)
    accepted_cfg = str(_eg(state, "accepted_config_path", "") or "")
    if accepted_cfg:
        out["accepted_config_path"] = _rel(Path(accepted_cfg), session_dir) or accepted_cfg
    setting_script_path = session_dir / "reports" / "enablement" / "enablement_setting.sh"
    if setting_script_path.is_file():
        out["setting_script"] = str(
            _rel(setting_script_path, session_dir) or "reports/enablement/enablement_setting.sh"
        )
    accepted_config = _eg(state, "accepted_config")
    if isinstance(accepted_config, dict) and accepted_config:
        out["accepted_config"] = {
            "extra_server_args": str(accepted_config.get("extra_server_args") or ""),
            "extra_envs": {str(k): str(v) for k, v in (accepted_config.get("extra_envs") or {}).items()},
        }
    if have_eval:
        out["trigger_kind"] = eval_kind
        out["observed_accuracy"] = float(_eg(state, "observed_accuracy", 0.0) or 0.0)
        out["accuracy_floor"] = float(_eg(state, "accuracy_floor", 0.0) or 0.0)
        out["observed_task"] = str(_eg(state, "observed_task", "") or "")
        out["observed_metric"] = str(_eg(state, "observed_metric", "") or "")
        out["eval_contract_fingerprint"] = str(_eg(state, "eval_contract_fingerprint", "") or "")
        probe_cfg = str(_eg(state, "probe_config_path", "") or "")
        if probe_cfg:
            out["probe_config_path"] = _rel(Path(probe_cfg), session_dir) or probe_cfg
        evidence = str(_eg(state, "baseline_eval_evidence", "") or "")
        if evidence:
            out["trigger_evidence_excerpt"] = evidence[-_ENABLEMENT_LOG_EXCERPT_CHARS:]
    active_root = str(active_runtime_raw.get("venv_root") or "") if have_active else ""
    if have_active:
        out["active_runtime"] = _runtime_summary(active_runtime_raw, promoted=True)
    if have_attempts:
        out["attempt_runtimes"] = [
            _runtime_summary(r, promoted=(str(r.get("venv_root") or "") == active_root))
            for r in attempt_runtimes_raw
            if isinstance(r, dict)
        ]
    if failure_kind:
        out["failure_kind"] = failure_kind
    # targeted-build attempt history
    if have_build_manifest:
        build_attempts = [
            _build_attempt_summary(e)
            for e in build_manifest_raw
            if isinstance(e, dict) and e.get("ok") is not None  # skip routing sentinels
        ]
        if build_attempts:
            out["build_attempts"] = build_attempts
            out["build_attempt_count"] = len(build_attempts)
    if have_last_failure:
        out["last_build_failure"] = {
            "failure_class": str(last_build_failure_raw.get("failure_class") or ""),
            "failure_summary": str(last_build_failure_raw.get("failure_summary") or ""),
        }
    return out
