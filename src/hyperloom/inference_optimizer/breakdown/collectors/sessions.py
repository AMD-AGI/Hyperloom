# Copyright Advanced Micro Devices, Inc. All rights reserved.

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


from ._common import (
    _benchmark_report_candidates,
    _benchmark_report_metrics,
    _find_benchmark_report,
    _iso_z,
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


# Defense-in-depth: strip credential-shaped keys even under allowlisted
# prefixes (case-insensitive substring match catches ``HF_API_TOKEN`` etc).
_ENV_DENY_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|AUTH|CREDENTIAL|COOKIE|API_KEY)",
    re.IGNORECASE,
)


def _filter_envs(envs: dict[str, Any] | None) -> dict[str, str]:
    """Apply the allowlist + secret denylist; returns a fresh ``dict[str, str]`` (values stringified).

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
        keep = (k in _ENV_ALLOWLIST_EXACT) or any(k.startswith(p) for p in _ENV_ALLOWLIST_PREFIXES)
        if not keep:
            continue
        if _ENV_DENY_PATTERN.search(k):
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


def _strip_log_prefix(line: str) -> str:
    """Strip a leading ``[ts] LEVEL [src.py:NN]`` style prefix (incl. the process tag) from a log line.

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
        dict | None: The decoded top-level mapping, or ``None`` when PyYAML is
        unavailable, the file fails to read/parse, or the document is not a
        dict.
    """
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        log.debug("PyYAML unavailable; skipping yaml framework_args fallback")
        return None
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
    # a failed eval is treated as a miss (anti-hallucination invariant).
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
        the path is missing, PyYAML is unavailable, or parsing fails.
    """
    if config_path is None:
        return {}
    try:
        if not config_path.exists():
            return {}
    except OSError:
        return {}
    try:
        import yaml
    except ImportError:
        log.debug("PyYAML unavailable; skipping invocation env extraction")
        return {}
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

    Prefers the manifest field (written once at session start, captures
    the spawn-time image even if the env later changes). Falls back to
    the same env / mount-point chain the manifest helper uses, so V1
    manifests (no ``image`` field) still surface a value when one of
    the envs is set. Mirrors :func:`manifest._detect_image` but kept as
    a separate function to avoid an import cycle.

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
        # /proc/1/cgroup may be unreadable in restricted sandboxes,
        # non-Linux hosts, or stripped-down containers. Best-effort
        # source — fall through to None so consumers see an honest
        # "image not detected" rather than a fabricated value.
        log.debug("cgroup-based image detection failed: %r", exc)
    return None


def _utc_now_iso() -> str:
    """Return the current UTC time as a second-precision ISO-8601 string.

    Returns:
        str: e.g. ``"2026-06-02T18:29:00+00:00"``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _close_phase_stop_reason(state: dict[str, Any]) -> tuple[str, str]:
    """Recover terminal reason/time from the CLOSE phase transition (next-best when ``state.stop_reason`` wasn't mirrored).

    Args:
        state (dict[str, Any]): Parsed ``state.json``.

    Returns:
        tuple[str, str]: ``(reason, ts)`` from the most recent CLOSE
        transition, or ``("", "")`` when no such transition exists.
    """
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        return "", ""
    for row in reversed(history):
        if not isinstance(row, dict):
            continue
        if str(row.get("to_phase") or "").strip().upper() != "CLOSE":
            continue
        reason = str(row.get("reason") or row.get("stop_reason") or row.get("exit_reason") or "").strip()
        ts = str(row.get("ts") or row.get("entered_ts") or "").strip()
        return reason, ts
    return "", ""


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

    SharedState already tracks whether a run crashed, was continued by the
    steward, entered degraded mode, or has an accepted stack awaiting
    post-resume revalidation — but none of it reaches the breakdown, so a
    resumed run reads as if it proceeded monotonically. This folds those
    signals into the ``session.recovery`` block so a reader can see the run
    was interrupted and continued (the context behind gaps like an empty
    ``geak_result`` lost to a kill before the tick-boundary save). Pure /
    best-effort: unparseable fields are skipped, never raised.

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
                crash_ts_iso.append(
                    datetime.fromtimestamp(float(t), tz=timezone.utc).isoformat()
                )
            except (TypeError, ValueError, OSError, OverflowError):
                continue

    last_exc: dict[str, Any] | None = None
    lte = state.get("last_tick_exception")
    if isinstance(lte, dict) and lte:
        # Drop the (large) traceback; keep the compact postmortem header.
        last_exc = {
            "tick": lte.get("tick"),
            "ts": lte.get("ts"),
            "stage": lte.get("stage"),
            "agent": lte.get("agent"),
            "type": lte.get("type"),
            "message": (str(lte.get("message") or "")[:500] or None),
        }

    infra = state.get("steward_infra_failures_by_round")
    infra_by_round: dict[str, int] = {}
    infra_total = 0
    if isinstance(infra, dict):
        for k, v in infra.items():
            iv = _to_int(v)
            if iv is None:
                continue
            infra_by_round[str(k)] = iv
            infra_total += iv

    steward_continuation = bool(state.get("steward_continuation_used"))
    resume_pending = bool(state.get("resume_pending_revalidation"))
    degraded = bool(state.get("degraded_mode"))
    recovered = bool(
        crash_count > 0
        or crash_ts_iso
        or steward_continuation
        or resume_pending
        or last_exc
    )
    return {
        "recovered": recovered,
        "crash_count": crash_count,
        "crash_timestamps": crash_ts_iso,
        "degraded_mode": degraded,
        "steward_continuation_used": steward_continuation,
        "resume_pending_revalidation": resume_pending,
        "steward_infra_failures_total": infra_total,
        "steward_infra_failures_by_round": infra_by_round,
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
    taking precedence on overlapping fields), computes ``elapsed_minutes``
    from the start timestamp, resolves the container image, and stamps
    ``ended_at_utc`` only once a ``stop_reason`` is present. When no image can
    be detected a warning is appended.

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
    close_stop_reason, close_ts = _close_phase_stop_reason(state)
    if _should_use_close_stop_reason(stop_reason, close_stop_reason):
        stop_reason = close_stop_reason
    ended_at_utc = ""
    if stop_reason:
        ended_at_utc = _iso_z(close_ts) if close_ts else _utc_now_iso()
    elapsed_min: float | None = None
    if start_ts:
        try:
            start = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
            elapsed_min = (datetime.now(timezone.utc) - start).total_seconds() / 60.0
        except (ValueError, TypeError):
            pass
    image = _detect_image_for_session(manifest)
    if image is None:
        warnings.append("image: not configured (set HYPERLOOM_IMAGE env var)")
    return {
        "session_id": str(state.get("session_id") or manifest.get("session_id") or ""),
        "claw_session_id": manifest.get("claw_session_id") or state.get("claw_session_id"),
        "sandbox_user_id": manifest.get("sandbox_user_id") or state.get("sandbox_user_id"),
        "created_at_utc": manifest.get("created_at_utc") or start_ts,
        "ended_at_utc": ended_at_utc,
        "stop_reason": stop_reason,
        "max_minutes": int(state.get("max_minutes") or manifest.get("max_minutes") or 0),
        "elapsed_minutes": round(elapsed_min, 2) if elapsed_min is not None else 0.0,
        "host": str(manifest.get("host") or ""),
        "image": image,
        "code_revision": str(manifest.get("code_revision") or ""),
        "pid": int(manifest.get("pid") or 0),
        "session_dir": str(session_dir),
        # USER_DATA_PATH root (the operator-chosen workspace base; session_dir
        # nests under it in per_model_ts layout). Snapshotted in the manifest at
        # session start; env is the in-process fallback for older manifests.
        "user_data_path": str(
            manifest.get("user_data_path")
            or state.get("user_data_path")
            or os.environ.get("USER_DATA_PATH")
            or ""
        ),
        "tick_count": int(state.get("tick") or 0),
        # Crash / interruption / resume history so a resumed run is not read as
        # a clean monotonic one (context behind e.g. an empty geak_result).
        "recovery": _collect_recovery(state),
    }


# session_meta enrichment
def collect_session_meta(
    manifest: dict[str, Any],
    session_section: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the ``session_meta`` enrichment block.

    Historically this block was injected post-export by ``ci/optimize_submit.py``
    (``_backfill_ci_metrics_file``), so any session that never went through that
    CI path landed in pulse without a ``session_meta``. The exporter now always
    emits it straight from the manifest + resolved ``session`` section, so the
    block no longer depends on CI; the CI step degrades to a gap-filler for the
    fields the sandbox could not know (e.g. ``category``).

    Args:
        manifest (dict[str, Any]): Parsed ``manifest.json``.
        session_section (dict[str, Any]): The already-built ``session`` dict.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: ``{code_revision, image, image_id,
        session_duration_seconds}``. Mirrors the field contract the CI backfill
        used so downstream readers (pulse ``sbd_store`` / ``normalize``) resolve
        the same values whether they came from the exporter or CI.
    """
    image = session_section.get("image")
    image_str = image if isinstance(image, str) and image.strip() else ""
    elapsed_min = session_section.get("elapsed_minutes")
    duration_s = (
        int(round(elapsed_min * 60))
        if isinstance(elapsed_min, (int, float)) and elapsed_min > 0
        else 0
    )
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
    # Re-root container-style paths under the on-disk session_dir (see ``_resolve_under_session``).
    workspace = _resolve_under_session(session_dir, workspace_str)
    if workspace_str and workspace is None:
        warnings.append(
            f"baseline workspace {workspace_str!r} does not resolve under {session_dir}; "
            "ttft_mean_ms / e2el_mean_ms will be null."
        )
    report_path = _find_benchmark_report(workspace) if workspace else None
    report = _load_json_safe(report_path, warnings) if report_path else None

    _, ttft, _tpot, e2el = _benchmark_report_metrics(report if isinstance(report, dict) else None)

    # Symmetric to A2: when the state workspace doesn't resolve, fall back to
    # the most recent ``runs/baseline/<hash>/.../benchmark_report.json`` on disk.
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
        # Throughput unit is framework-dependent: serving frameworks report
        # tok/s, scriptable xDiT diffusion reports img/s. The numeric field name
        # is kept for backwards compatibility; this records the true unit so the
        # value is not silently read as tokens/s.
        "throughput_unit": framework_registry.throughput_unit(state.get("framework")),
        "accuracy": _to_float(state.get("baseline_accuracy")) or 0.0,
        "ttft_mean_ms": ttft,
        "e2el_mean_ms": e2el,
        "ttft_e2el_source": ttft_source,
        "config_path": config_path_raw,
        "benchmark_report_path": _rel(report_path, session_dir) if report_path else None,
        "attempts_history": history,
        "failure_streak": int(state.get("baseline_failure_streak") or 0),
        # Combined backstop: ALL baseline failures regardless of error_class;
        # surfaces the fast-fail trigger that per-class streaks alone can hide.
        "total_failures": int(state.get("baseline_total_failures") or 0),
        "invocation": invocation,
        # Standalone baseline-arm roofline ceiling backup (state.json#baseline_roofline_ceiling);
        # frontend ceiling fallback when the roofline step failed. {} when absent.
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
    ``validate_stack`` report first, then the top stack entry's report),
    recording the provenance in ``ttft_e2el_source`` and a ``warnings`` note.
    Also assembles the replayable launch ``invocation``.

    Args:
        session_dir (Path): Absolute session root.
        state (dict[str, Any]): Parsed ``state.json``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        dict[str, Any]: The final section (throughput, validated/per-round
        cumulative gain, stack-length bookkeeping, action path, ttft / e2el,
        invocation, and closing-phase markers).
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

    # Disk-walk reconstruction when ``current_best.ttft_mean_ms`` is unset:
    # validate_stack first (authoritative), then current_best, then stack top.
    reconstructed_report: Path | None = None
    # Reconstruct from disk when EITHER latency metric is missing. For scriptable
    # xDiT diffusion ttft is a meaningless 0.0/None and e2el is the meaningful
    # signal, so gating only on ttft dropped e2el from the final section.
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

    from ... import framework_registry

    return {
        "throughput_tok_s_per_gpu": _to_float(cb.get("tput")),
        # See collect_baseline: records the true throughput unit (tok/s vs img/s
        # for scriptable xDiT) alongside the compat-named numeric field.
        "throughput_unit": framework_registry.throughput_unit(state.get("framework")),
        # Which field holds the primary result: e2el_mean_ms (scriptable/xDiT)
        # vs throughput_tok_s_per_gpu (serving). Lets consumers pick per model.
        "primary_metric": framework_registry.primary_metric_name(state.get("framework")),
        "cumulative_gain_pct_validated": _to_float(state.get("cumulative_gain_validated")) or 0.0,
        "cumulative_gain_pct_per_round_sum": _to_float(state.get("cumulative_gain")) or 0.0,
        # Provenance/basis of the recorded gain so the renderer can tell a
        # same-harness-validated number from a cross-harness PROVISIONAL one
        # (e.g. a geak e2e win pending its same-harness rebench). Empty on
        # native/legacy sessions (renders as validated, unchanged).
        "cumulative_gain_provenance": str(state.get("cumulative_gain_provenance") or ""),
        "revalidation_pending": bool(state.get("resume_pending_revalidation") or False),
        # A GEAK(GEAK) e2e candidate whose self-reported win has NOT yet
        # been confirmed by a main-flow rebench. Present => the renderer surfaces
        # it as an audit-only note and EXCLUDES it from the headline gain (the
        # candidate is intentionally absent from current_best / action_path until
        # a measured rebench validates it). Empty on native/validated sessions.
        "geak_pending": (
            dict(state.get("geak_pending") or {})
            if isinstance(state.get("geak_pending"), dict)
            else {}
        ),
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

