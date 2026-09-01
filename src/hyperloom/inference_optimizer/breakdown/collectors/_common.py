# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` /
``state`` / ``manifest`` returning its schema section (see :mod:`.schema`).
Collectors never mutate state, fabricate values, or raise — failures are
recorded in ``warnings`` and the section returns a best-effort partial.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from hyperloom.common.coerce import to_float
from hyperloom.common.jsonio import read_json, read_jsonl

from ...session.paths import is_path_within


_FRAMEWORK_PHASES = frozenset({"FRAMEWORK_AGENT", "EXPLORE"})
_AUTHORING_TASK_KINDS = frozenset(
    {
        "explore_apply_retry",
        "framework_authoring",
        "framework_local_explore",
    }
)


# Shared helpers
def _mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a dict, otherwise an empty mapping."""
    return value if isinstance(value, dict) else {}


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    """Keep only dictionary rows from a list-shaped value."""
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _first(*values: Any) -> Any:
    """Return the first value that is neither ``None`` nor an empty string."""
    return next((value for value in values if value is not None and value != ""), None)


def _optional_bool(value: Any) -> bool | None:
    """Coerce conventional boolean spellings without accepting arbitrary numbers."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "passed", "succeeded"}:
            return True
        if normalized in {"0", "false", "no", "off", "failed"}:
            return False
    return None


def _string_list(value: Any) -> list[str]:
    """Normalize a list-like value to non-empty strings."""
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item) for item in value if item not in (None, "")]


def _load_json_safe(
    path: Path | None,
    warnings: list[str],
    *,
    require_dict: bool = False,
) -> Any | None:
    """Parse a JSON file, recording any failure instead of raising.

    Args:
        path (Path | None): File to read, or ``None``.
        warnings (list[str]): Shared warnings list; a parse/read failure is
            appended here (mutated in place).
        require_dict (bool): When True, a top-level non-object is treated as
            a parse failure (recorded in ``warnings``, returns ``None``).
            Defaults to False so existing callers that accept any JSON value
            keep their behaviour.

    Returns:
        Any | None: The decoded JSON value, or ``None`` if ``path`` is
        ``None``, the file does not exist, or decoding failed.
    """
    if path is None:
        return None
    if not path.exists():
        return None
    return read_json(
        path,
        default=None,
        require_dict=require_dict,
        on_error=lambda exc: warnings.append(f"failed to parse {path}: {exc!r}"),
    )


def _load_jsonl_safe(path: Path | None, warnings: list[str]) -> list[dict[str, Any]]:
    """Parse a JSON-Lines file into a list of dict rows, never raising.

    Blank lines are skipped. Malformed lines and read failures are recorded
    in ``warnings`` and otherwise ignored; only dict-valued rows are kept.

    Args:
        path (Path | None): The ``.jsonl`` file to read, or ``None``.
        warnings (list[str]): Shared warnings list (mutated in place).

    Returns:
        list[dict[str, Any]]: One dict per well-formed object line. Empty when
        ``path`` is ``None`` / missing or no line parsed to a dict.
    """
    if path is None or not path.exists():
        return []

    def _warn(exc: BaseException) -> None:
        prefix = "failed to read" if isinstance(exc, OSError) else "malformed jsonl line in"
        warnings.append(f"{prefix} {path}: {exc!r}")

    return read_jsonl(path, require_dict=True, skip_malformed=True, on_error=_warn)


def _to_float(value: Any) -> float | None:
    """Coerce an arbitrary value to ``float`` without raising.

    Booleans are rejected (returned as ``None``) so ``True``/``False`` never
    silently become ``1.0``/``0.0``. Strings are stripped, the sentinel
    ``"SKIPPED"`` and empty strings map to ``None``, and thousands separators
    (``,``) are removed before parsing.

    Args:
        value (Any): The value to convert.

    Returns:
        float | None: The parsed float, or ``None`` when the value is missing,
        a bool, or not numeric.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() == "SKIPPED":
            return None
        return to_float(text.replace(",", ""))
    return to_float(value)


def _to_int(value: Any) -> int | None:
    """Coerce a value to ``int`` via :func:`_to_float`, never raising.

    Args:
        value (Any): The value to convert.

    Returns:
        int | None: The truncated integer, or ``None`` when ``value`` is not
        numeric (same rules as :func:`_to_float`).
    """
    number = _to_float(value)
    return int(number) if number is not None else None


def _rel(path: Path | None, session_dir: Path) -> str | None:
    """Express ``path`` relative to ``session_dir`` as a POSIX string.

    Args:
        path (Path | None): The path to relativize, or ``None``.
        session_dir (Path): The session root the result is relative to.

    Returns:
        str | None: The POSIX-style relative path, or ``None`` when ``path``
        is ``None``. Falls back to ``str(path)`` when ``path`` is not under
        ``session_dir``.
    """
    if path is None:
        return None
    try:
        return path.resolve().relative_to(session_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _benchmark_report_metrics(
    report: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Extract (output_throughput, ttft, tpot, e2el) from a benchmark_report.json across schema generations.

    Priority: V2 nested (``throughput.*`` / ``latency.<m>.mean_ms``), flat
    top-level, then ``result.<flat>``.

    Args:
        report (dict[str, Any] | None): A parsed ``benchmark_report.json``, or
            ``None``.

    Returns:
        tuple[float | None, float | None, float | None, float | None]:
        ``(output_throughput, ttft, tpot, e2el)`` with each element ``None``
        when not present or not numeric.
    """
    if not isinstance(report, dict):
        return (None, None, None, None)
    tput_section = report.get("throughput") if isinstance(report.get("throughput"), dict) else None
    lat_section = report.get("latency") if isinstance(report.get("latency"), dict) else None
    result_section = report.get("result") if isinstance(report.get("result"), dict) else None

    def _from_lat(metric: str) -> Any:
        """Read ``latency.<metric>.mean_ms`` from the V2 latency section.

        Args:
            metric (str): Latency metric name (e.g. ``"ttft"``, ``"tpot"``).

        Returns:
            Any: The metric's ``mean_ms`` value, or ``None`` when absent.
        """
        if isinstance(lat_section, dict):
            sub = lat_section.get(metric)
            if isinstance(sub, dict):
                return sub.get("mean_ms")
        return None

    out_tput = _to_float(
        (tput_section or {}).get("output_throughput")
        or (tput_section or {}).get("output_throughput_tok_s")
        or report.get("output_throughput_tok_s")
        or report.get("output_throughput")
        or (result_section or {}).get("output_throughput_tok_s")
    )
    ttft = _to_float(_from_lat("ttft") or report.get("mean_ttft_ms") or (result_section or {}).get("mean_ttft_ms"))
    tpot = _to_float(_from_lat("tpot") or report.get("mean_tpot_ms") or (result_section or {}).get("mean_tpot_ms"))
    e2el = _to_float(_from_lat("e2el") or report.get("mean_e2el_ms") or (result_section or {}).get("mean_e2el_ms"))
    return (out_tput, ttft, tpot, e2el)


def _benchmark_report_candidates(root: Path) -> list[Path]:
    """Return benchmark reports under a task/workspace root (handles the several on-disk layouts).

    Args:
        root (Path): The task or workspace directory to search.

    Returns:
        list[Path]: Candidate ``benchmark_report.json`` paths (direct and
        glob-matched). Empty when ``root`` does not exist.
    """
    if not root.exists():
        return []

    candidates: list[Path] = []
    direct = root / "benchmark_report.json"
    if direct.exists():
        candidates.append(direct)

    patterns = (
        "benchmark_*/benchmark_report.json",
        "measure_round/benchmark_*/benchmark_report.json",
        "warmup_round/benchmark_*/benchmark_report.json",
    )
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    return candidates


def _latest_benchmark_report(candidates: Iterable[Path]) -> Path | None:
    """Return the most recently modified existing report among candidates.

    Args:
        candidates: Candidate report paths.

    Returns:
        The newest existing path by mtime, or ``None`` when none exist.
    """
    reports = [p for p in candidates if p.exists()]
    if not reports:
        return None
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0]


def _find_benchmark_report(workspace: Path | None) -> Path | None:
    """Locate the most recent (by mtime) ``benchmark_report.json`` under a task workspace, else ``None``.

    Args:
        workspace (Path | None): The task workspace to search, or ``None``.

    Returns:
        Path | None: The newest matching report, or ``None`` when ``workspace``
        is ``None`` / missing or no report exists.
    """
    if workspace is None or not workspace.exists():
        return None
    return _latest_benchmark_report(_benchmark_report_candidates(workspace))


def _resolve_under_session(
    session_dir: Path,
    raw: str | None,
    anchors: tuple[str, ...] = ("runs", "kernel-agent", "kernel-agent-workspace"),
) -> Path | None:
    """Best-effort resolve a possibly-container-rooted path under ``session_dir``; never raises.

    Tries the raw path as-is, then re-roots each ``anchors`` suffix at
    ``session_dir`` (container paths like ``/workspace/runs/...`` map to the
    wekafs ``<session_dir>/runs/...`` view).     Returns the first existing
    candidate that lies inside ``session_dir``, else ``None``.

    A raw path resolving outside the session is treated as a foreign view
    and re-rooted, so a container path that also exists on this host cannot
    resolve to another session's artifacts.

    Args:
        session_dir (Path): The on-disk session root to re-root under.
        raw (str | None): The (possibly container-rooted) path to resolve, or
            ``None``.
        anchors (tuple[str, ...]): Path-segment names whose suffix is re-rooted
            at ``session_dir``. Defaults to ``("runs", "kernel-agent",
            "kernel-agent-workspace")``.

    Returns:
        Path | None: An existing path inside ``session_dir``, or ``None`` when
        ``raw`` is empty / unusable or nothing resolves inside the session.
    """
    if not raw:
        return None
    try:
        p = Path(str(raw))
    except (TypeError, ValueError):
        return None
    if p.exists() and is_path_within(p, session_dir):
        return p
    for anchor in anchors:
        try:
            idx = p.parts.index(anchor)
        except ValueError:
            continue
        candidate = session_dir.joinpath(*p.parts[idx:])
        if candidate.exists() and is_path_within(candidate, session_dir):
            return candidate
    return None


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Walk a nested dict by successive keys without raising.

    Args:
        d (Any): The (possibly nested) mapping to traverse.
        *keys (str): Keys to follow in order.
        default (Any): Value returned when traversal hits a missing key, a
            non-dict node, or a ``None`` leaf. Defaults to ``None``.

    Returns:
        Any: The resolved value, or ``default`` if any step fails or the
        final value is ``None``.
    """
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _parse_iso_unix(ts: Any) -> float | None:
    """Best-effort ISO-8601 -> unix seconds. ``None`` on any failure.

    Args:
        ts (Any): An ISO-8601 string or already-numeric timestamp.

    Returns:
        float | None: The timestamp in Unix seconds, or ``None`` when ``ts`` is
        empty or unparseable.
    """
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def phase_at(
    ts_unix: float,
    phase_boundaries: list[tuple[float, str]],
    *,
    fallback: str = "",
) -> str:
    """Return the phase active at ``ts_unix``.

    Args:
        ts_unix: The timestamp to classify.
        phase_boundaries: ``(unix_ts, phase_name)`` transition points sorted
            ascending; the last one at or before ``ts_unix`` wins.
        fallback: Phase name returned when every boundary is later.

    Returns:
        The phase name active at ``ts_unix``, or ``fallback``.
    """
    current = fallback
    for boundary, phase in phase_boundaries:
        if boundary <= ts_unix:
            current = phase
        else:
            break
    return current


def _load_optimization_journal(
    session_dir: Path | None,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Read ``reports/optimization_journal.json`` entries (the canonical action ledger); ``[]`` on legacy sessions.

    Args:
        session_dir (Path | None): Absolute session root, or ``None``.
        warnings (list[str]): Shared warnings list (mutated in place on parse
            failure).

    Returns:
        list[dict[str, Any]]: The journal ``entries`` list, or ``[]`` when the
        file is missing / malformed or ``session_dir`` is ``None``.
    """
    if session_dir is None:
        return []
    data = _load_json_safe(
        session_dir / "reports" / "optimization_journal.json",
        warnings,
    )
    if not isinstance(data, dict):
        return []
    entries = data.get("entries")
    return entries if isinstance(entries, list) else []


def _scan_profile_reports(session_dir: Path) -> list[tuple[Path, Path]]:
    """List ``(task_dir, benchmark_report.json)`` pairs under runs/profile/.

    Args:
        session_dir (Path): Absolute session root.

    Returns:
        list[tuple[Path, Path]]: One ``(task_dir, report_path)`` pair per
        profile task that has a benchmark report. Empty when no
        ``runs/profile/`` tree exists.
    """
    out: list[tuple[Path, Path]] = []
    root = session_dir / "runs" / "profile"
    if not root.exists():
        return out
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        report = _find_benchmark_report(task_dir)
        if report is not None:
            out.append((task_dir, report))
    return out
