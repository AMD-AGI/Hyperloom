# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from hyperloom.common.coerce import to_float
from hyperloom.common.jsonio import read_json, read_jsonl

from ...session.paths import is_path_within


_FRAMEWORK_PHASES = frozenset({"FRAMEWORK_AGENT", "EXPLORE"})
_KERNEL_PHASES = frozenset({"KERNEL_AGENT"})
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
    """Parse a JSON file, recording any failure instead of raising."""
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
    """Parse a JSON-Lines file into a list of dict rows, never raising."""
    if path is None or not path.exists():
        return []

    def _warn(exc: BaseException) -> None:
        prefix = "failed to read" if isinstance(exc, OSError) else "malformed jsonl line in"
        warnings.append(f"{prefix} {path}: {exc!r}")

    return read_jsonl(path, require_dict=True, skip_malformed=True, on_error=_warn)


def _to_float(value: Any) -> float | None:
    """Coerce an arbitrary value to ``float`` without raising."""
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() == "SKIPPED":
            return None
        return to_float(text.replace(",", ""))
    return to_float(value)


def _to_int(value: Any) -> int | None:
    """Coerce a value to ``int`` via :func:`_to_float`, never raising."""
    number = _to_float(value)
    return int(number) if number is not None else None


def _rel(path: Path | None, session_dir: Path) -> str | None:
    """Express ``path`` relative to ``session_dir`` as a POSIX string."""
    if path is None:
        return None
    try:
        return path.resolve().relative_to(session_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _benchmark_report_metrics(
    report: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Extract (output_throughput, ttft, tpot, e2el) from a benchmark_report.json across schema generations."""
    if not isinstance(report, dict):
        return (None, None, None, None)
    tput_section = report.get("throughput") if isinstance(report.get("throughput"), dict) else None
    lat_section = report.get("latency") if isinstance(report.get("latency"), dict) else None
    result_section = report.get("result") if isinstance(report.get("result"), dict) else None

    def _from_lat(metric: str) -> Any:
        """Read ``latency.<metric>.mean_ms`` from the V2 latency section."""
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
    """Return benchmark reports under a task/workspace root (handles the several on-disk layouts)."""
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
    """Return the most recently modified existing report among candidates."""
    reports = [p for p in candidates if p.exists()]
    if not reports:
        return None
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0]


def _find_benchmark_report(workspace: Path | None) -> Path | None:
    """Locate the most recent (by mtime) ``benchmark_report.json`` under a task workspace, else ``None``."""
    if workspace is None or not workspace.exists():
        return None
    return _latest_benchmark_report(_benchmark_report_candidates(workspace))


def _resolve_under_session(
    session_dir: Path,
    raw: str | None,
    anchors: tuple[str, ...] = ("runs", "kernel-agent", "kernel-agent-workspace"),
) -> Path | None:
    """Best-effort resolve a possibly-container-rooted path under ``session_dir``; never raises."""
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
    """Walk a nested dict by successive keys without raising."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _operation_task_id(operation: Any) -> str:
    """Return the orchestrator task id a recorded operation was run under."""
    return str(
        _first(
            _safe_get(operation, "extensions", "task_id"),
            _safe_get(operation, "outputs", "task_id"),
            _safe_get(operation, "metadata", "extras", "task_id"),
        )
        or ""
    )


def _parse_iso_unix(ts: Any) -> float | None:
    """Best-effort ISO-8601 -> unix seconds. ``None`` on any failure."""
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
    """Return the phase active at ``ts_unix``."""
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
    """Read ``reports/optimization_journal.json`` entries (the canonical action ledger); ``[]`` on legacy sessions."""
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
    """List ``(task_dir, benchmark_report.json)`` pairs under runs/profile/."""
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
