"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function: it reads only from
``session_dir``, ``state`` (a :class:`SharedState`-shaped dict) and
``manifest`` (a manifest.json dict), and returns the matching schema
section. Collectors NEVER mutate state, NEVER fabricate values, and
NEVER raise — failures are caught, recorded in ``warnings``, and the
section returns a best-effort partial.

The schema each collector matches is defined in :mod:`.schema`.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Invocation-record env filter (allowlist + secret-pattern denylist)
# ---------------------------------------------------------------------------
# We only surface env vars that are known to influence the launched
# benchmark workload (knobs an operator would reasonably want to replay).
# Anything outside the allowlist gets dropped — keeps secrets, host
# fingerprints, and shell aliases out of the breakdown JSON entirely.
_ENV_ALLOWLIST_EXACT: frozenset[str] = frozenset({
    "TP", "FRAMEWORK", "GPU_TYPE", "PRECISION", "CONC", "ISL", "OSL",
    "MAX_MODEL_LEN", "USER_DATA_PATH", "MODEL_PATH", "MODEL_NAME",
})
_ENV_ALLOWLIST_PREFIXES: tuple[str, ...] = (
    "HYPERLOOM_", "VLLM_", "SGLANG_", "RAY_", "HSA_", "ROCM_", "TORCH_", "HF_",
)
# Defense-in-depth: even if a future operator adds a credential-shaped
# key under one of the allowlisted prefixes (e.g. ``HF_API_TOKEN``) we
# still strip it before emit. Match is case-insensitive; substring match
# is intentional (catches ``MY_API_KEY``, ``X_PASSWORD_FILE`` etc.).
_ENV_DENY_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|AUTH|CREDENTIAL|COOKIE|API_KEY)",
    re.IGNORECASE,
)


def _filter_envs(envs: dict[str, Any] | None) -> dict[str, str]:
    """Apply the allowlist + secret denylist to a raw env dict.

    Always returns a fresh ``dict[str, str]`` (values stringified, None
    becomes ``""``). Non-string keys are dropped silently.
    """
    if not isinstance(envs, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in envs.items():
        if not isinstance(k, str):
            continue
        keep = (k in _ENV_ALLOWLIST_EXACT) or any(
            k.startswith(p) for p in _ENV_ALLOWLIST_PREFIXES
        )
        if not keep:
            continue
        if _ENV_DENY_PATTERN.search(k):
            continue
        out[k] = "" if v is None else str(v)
    return out


# --- framework-args extraction patterns -----------------------------------
# Pass-0 ("log_non_default_args"): vllm (and recent sglang) print a single
# line like ``non-default args: {'model': '/path', 'tensor_parallel_size':
# 8, ...}`` right after argv parsing — it captures the *resolved* parsed
# arg dict (post-CLI, post-env-override) and is therefore the most
# authoritative cmdline-equivalent the framework itself emits. We match
# loosely (``re.search``) so any leading ``(APIServer pid=...) INFO ...
# [utils.py:233]`` log prefix is allowed; the dict literal must end the
# line so a greedy ``\{.+\}`` is safe.
_FRAMEWORK_ARGS_NON_DEFAULT_RE = re.compile(
    r"non[-_]default args:\s*(\{.+\})\s*$",
    re.IGNORECASE,
)
# Pass-1 ("log_args_line"): the runner echoes the parsed launch arguments
# under one of these stable headers — most reliable signal because it
# survives any number of preceding INFO log lines.
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
# Pass-2 ("log_python_cmd"): a literal ``python ... vllm/sglang ...``
# command somewhere in server.log. We accept it after stripping a
# leading ``(APIServer pid=...) INFO 05-12 ... [utils.py:299]``-style
# log prefix that vllm/sglang emit.
_LOG_PREFIX_RE = re.compile(
    r"^\s*(?:\([^)]*\)\s+)?(?:INFO|WARN|WARNING|ERROR|DEBUG|TRACE)\s+"
    r"\d[\d:\-\s]*\[[^\]]+\]\s*",
    re.IGNORECASE,
)
_LOG_TIMESTAMP_RE = re.compile(
    r"^\s*\[?\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[^\]]*\]?\s*",
)
_PYTHON_CMD_PREFIXES: tuple[str, ...] = (
    "python", "python3", "vllm", "sglang.launch_server",
    "inference-optimizer", "ray",
)
# Server log size cap — pathological logs (multi-MB) get truncated so
# the per-line scan stays bounded. 256 KB easily covers any startup
# banner that echoes the launch line.
_SERVER_LOG_MAX_BYTES = 256 * 1024


def _strip_log_prefix(line: str) -> str:
    """Strip a leading ``[ts] LEVEL [src.py:NN]`` style prefix from a log line.

    The vllm/sglang server echoes the launch command on its own line
    after ~50 INFO log lines, each shaped like
    ``(APIServer pid=1757439) INFO 05-12 14:21:14 [utils.py:299] ...``.
    To match the literal command we strip both the parenthetical
    process tag and the level/timestamp/source-frame prefix, leaving
    just the trailing payload.
    """
    s = line
    s = _LOG_PREFIX_RE.sub("", s)
    s = _LOG_TIMESTAMP_RE.sub("", s)
    return s.strip()


def _starts_with_python_prefix(text: str) -> bool:
    head = text.lstrip()
    for prefix in _PYTHON_CMD_PREFIXES:
        if head.startswith(prefix):
            tail = head[len(prefix):]
            if not tail or tail[0] in (" ", "\t", "-", "."):
                return True
    return False


def _load_yaml_dict_safe(config_yaml: Path) -> dict | None:
    """Parse ``config_yaml`` and return the top-level dict, or ``None`` on
    any miss / parse failure / non-dict root. Never raises.

    Shared by both yaml-based passes (Pass 3 ``yaml_cmd`` and Pass 4
    ``yaml_benchmark``) so the file is read + parsed at most once per
    ``_extract_framework_args`` invocation.
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
    """Look for a ``cmd`` / ``command`` / ``launch`` field in a parsed yaml.

    Reads from these locations (first-non-empty wins):
      * top-level ``cmd`` / ``command`` / ``launch``
      * nested ``benchmark.cmd`` / ``benchmark.command``
    Returns ``""`` on any miss.
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
    """Synthesize a readable arg string from a magpie ``benchmark.*`` dict.

    Magpie-style materialized configs don't carry a literal ``cmd:`` field
    — the launcher assembles the cmdline at runtime from structured
    ``benchmark.{framework, model, precision, tp, gpu_selection, envs}``.
    When neither server.log nor a yaml ``cmd:`` field is available, we
    stringify those structured fields so the operator at least sees which
    framework + model + precision + tp + envs were configured. The
    ``yaml_benchmark`` source label flags downstream consumers that this
    is a synthesized representation, not a literal cmdline.

    Returns ``""`` unless both ``benchmark.framework`` AND ``benchmark.model``
    are non-empty (those two are the minimum needed for the synthesis to
    convey anything useful).
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
        env_pairs = " ".join(
            f"{k}={v}" for k, v in envs.items() if isinstance(k, str)
        )
        if env_pairs:
            parts.append(f"envs=[{env_pairs}]")
    return " ".join(parts)


def _extract_framework_args(
    server_log: Path | None,
    config_yaml: Path | None = None,
) -> tuple[str, str]:
    """Best-effort extract the launch command for a benchmark variant.

    Returns ``(args_string, source)`` where ``source`` documents lineage:
      * ``"log_non_default_args"`` — found a ``non-default args: {...}``
        line (vllm / recent sglang echo of the parsed argv dict).
        Highest priority because it's emitted *by the framework itself*
        after argv parsing, so it captures the actually-used values.
      * ``"log_args_line"`` — found a line like ``Server arguments: ...``
        or ``Args: Namespace(...)`` in server.log
      * ``"log_python_cmd"`` — found a line starting with ``python``,
        ``python3``, ``vllm``, ``sglang.launch_server`` etc. in server.log
      * ``"yaml_cmd"`` — fell back to ``cmd:`` / ``command:`` / ``launch:``
        in the materialized config yaml
      * ``"yaml_benchmark"`` — synthesized from magpie ``benchmark.*``
        structured fields (framework / model / precision / tp / envs).
        NOT a literal cmdline — flagged via the source label so consumers
        can treat it differently from log-extracted values.
      * ``"unknown"`` — none of the above; ``args_string`` is empty
    Never raises. Caller decides whether to surface a warning when source
    is ``unknown``.
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

    # Pass 0: vllm / sglang ``non-default args: {...}`` echo. The dict
    # is the framework's own post-parse view of argv, so it beats any of
    # the literal-cmdline passes when present. Parse via ast.literal_eval
    # — if the eval fails (malformed dict, embedded objects), we treat
    # the line as a miss and continue to Pass 1 rather than regex-hacking
    # a half-parse (anti-hallucination invariant).
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
        # Sorted-by-key, repr() values — stable across runs and keeps
        # path strings quoted so an operator can copy-paste verbatim.
        formatted = " ".join(
            f"{k}={parsed[k]!r}" for k in sorted(parsed.keys(), key=str)
        )
        return formatted, "log_non_default_args"

    # Pass 1: stable launch-summary headers. These are emitted by the
    # framework itself once it's parsed argv, so they survive any
    # preceding log noise and are the most authoritative when present.
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

    # Pass 2: a literal python/vllm/sglang command somewhere in the log.
    # Strip the noisy ``(...) INFO 05-12 14:21:14 [utils.py:299]`` prefix
    # before testing — the command itself starts on the same line.
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _starts_with_python_prefix(stripped):
            return stripped, "log_python_cmd"
        cleaned = _strip_log_prefix(stripped)
        if cleaned and _starts_with_python_prefix(cleaned):
            return cleaned, "log_python_cmd"

    # Pass 3 + Pass 4: yaml fallback. Load the config yaml at most once,
    # then prefer a literal ``cmd:`` field (Pass 3) over the magpie
    # ``benchmark.*`` synthesis (Pass 4) — a literal cmdline is always
    # more authoritative than a structured-config reconstruction.
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
    """Read ``benchmark.envs`` from a ``baseline_config.with_envs.yaml``
    (or any sibling variant config) and return the allowlisted subset.

    Falls back to the top-level ``envs:`` block for older config layouts
    that didn't yet nest under ``benchmark:``. Returns ``{}`` on any
    read / parse failure or if PyYAML is somehow missing — never raises.
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _load_json_safe(path: Path | None, warnings: list[str]) -> Any | None:
    if path is None:
        return None
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to parse {path}: {exc!r}")
        return None


def _load_jsonl_safe(path: Path | None, warnings: list[str]) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"malformed jsonl line in {path}: {exc!r}")
                continue
            if isinstance(entry, dict):
                out.append(entry)
    except OSError as exc:
        warnings.append(f"failed to read {path}: {exc!r}")
    return out


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() == "SKIPPED":
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rel(path: Path | None, session_dir: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(session_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _benchmark_report_metrics(
    report: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Extract (output_throughput, ttft_mean_ms, tpot_mean_ms, e2el_mean_ms) from
    a benchmark_report.json regardless of schema generation.

    Supported shapes (in priority order):

    * V2 (current): top-level ``throughput.output_throughput`` +
      ``latency.<metric>.mean_ms`` (e.g. ``latency.ttft.mean_ms``).
    * Pre-V2 flat: ``output_throughput_tok_s`` / ``mean_ttft_ms`` at the
      top level.
    * Legacy nested-under-result: ``result.<flat>``.
    """
    if not isinstance(report, dict):
        return (None, None, None, None)
    tput_section = report.get("throughput") if isinstance(report.get("throughput"), dict) else None
    lat_section = report.get("latency") if isinstance(report.get("latency"), dict) else None
    result_section = report.get("result") if isinstance(report.get("result"), dict) else None

    def _from_lat(metric: str) -> Any:
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
    ttft = _to_float(
        _from_lat("ttft")
        or report.get("mean_ttft_ms")
        or (result_section or {}).get("mean_ttft_ms")
    )
    tpot = _to_float(
        _from_lat("tpot")
        or report.get("mean_tpot_ms")
        or (result_section or {}).get("mean_tpot_ms")
    )
    e2el = _to_float(
        _from_lat("e2el")
        or report.get("mean_e2el_ms")
        or (result_section or {}).get("mean_e2el_ms")
    )
    return (out_tput, ttft, tpot, e2el)


def _benchmark_report_candidates(root: Path) -> list[Path]:
    """Return benchmark reports under a task/workspace root.

    Hyperloom has used several shapes over time:

    * ``<task>/benchmark_*/benchmark_report.json``
    * ``<task>/{warmup_round,measure_round}/benchmark_*/benchmark_report.json``
    * ``<task>/.../variant_*/benchmark_*/benchmark_report.json`` for explore
    * ``<benchmark_*>/benchmark_report.json`` when state.workspace points
      directly at the benchmark directory.
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
    reports = [p for p in candidates if p.exists()]
    if not reports:
        return None
    reports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return reports[0]


def _find_benchmark_report(workspace: Path | None) -> Path | None:
    """Locate a ``benchmark_report.json`` under a task workspace.

    Returns the most recent one (by mtime) if multiple are present (e.g.
    retries or warmup+measure rounds within the same task), else ``None``.
    """
    if workspace is None or not workspace.exists():
        return None
    return _latest_benchmark_report(_benchmark_report_candidates(workspace))


def _resolve_under_session(
    session_dir: Path,
    raw: str | None,
    anchors: tuple[str, ...] = ("runs", "kernel-agent", "kernel-agent-workspace"),
) -> Path | None:
    """Best-effort resolve a possibly-container-rooted path under ``session_dir``.

    State-recorded paths frequently look like ``/workspace/runs/baseline/<sid>/``
    because the orchestrator wrote them from inside a container, but the
    breakdown is generated against a wekafs view where the same artefacts
    live under ``<session_dir>/runs/baseline/<sid>/``.

    Resolution order:

    1. The raw path as-is (covers the development / test case where the
       state-recorded path is already real).
    2. For each anchor in ``anchors``, find the first occurrence of the
       anchor in the raw path's parts and re-root that suffix at
       ``session_dir``. The default anchors cover the three on-disk
       conventions hyperloom uses (``runs/...``, ``kernel-agent/...``,
       ``kernel-agent-workspace/...``).

    Returns the first existing :class:`Path`, or ``None`` if nothing
    resolves. Never raises — callers that care about the failure should
    inspect the return value and append to ``warnings`` themselves.
    """
    if not raw:
        return None
    try:
        p = Path(str(raw))
    except (TypeError, ValueError):
        return None
    if p.exists():
        return p
    for anchor in anchors:
        try:
            idx = p.parts.index(anchor)
        except ValueError:
            continue
        candidate = session_dir.joinpath(*p.parts[idx:])
        if candidate.exists():
            return candidate
    return None


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        if k not in cur:
            return default
        cur = cur[k]
    return cur if cur is not None else default


def _close_phase_stop_reason(state: dict[str, Any]) -> tuple[str, str]:
    """Recover terminal reason/time from the CLOSE phase transition.

    Older sessions can complete the phase state machine without mirroring the
    CLOSE transition's reason back to top-level ``state.stop_reason``. The
    phase history row is still Coordinator-authored and vocab-validated, so it
    is the next-best lifecycle source for breakdown metadata.
    """
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        return "", ""
    for row in reversed(history):
        if not isinstance(row, dict):
            continue
        if str(row.get("to_phase") or "").strip().upper() != "CLOSE":
            continue
        reason = str(
            row.get("reason")
            or row.get("stop_reason")
            or row.get("exit_reason")
            or ""
        ).strip()
        ts = str(row.get("ts") or row.get("entered_ts") or "").strip()
        return reason, ts
    return "", ""


# ---------------------------------------------------------------------------
# §1 Session metadata
# ---------------------------------------------------------------------------
def collect_session(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Top-level session identification + lifecycle."""
    start_ts = str(state.get("start_ts") or manifest.get("created_at_utc") or "")
    stop_reason = str(state.get("stop_reason") or "").strip()
    close_stop_reason, close_ts = _close_phase_stop_reason(state)
    if not stop_reason:
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
        warnings.append(
            "image: not configured (set HYPERLOOM_IMAGE env var)"
        )
    return {
        "session_id":       str(state.get("session_id") or manifest.get("session_id") or ""),
        "claw_session_id":  manifest.get("claw_session_id") or state.get("claw_session_id"),
        "sandbox_user_id":  manifest.get("sandbox_user_id") or state.get("sandbox_user_id"),
        "created_at_utc":   manifest.get("created_at_utc") or start_ts,
        "ended_at_utc":     ended_at_utc,
        "stop_reason":      stop_reason,
        "max_minutes":      int(state.get("max_minutes") or manifest.get("max_minutes") or 0),
        "elapsed_minutes":  round(elapsed_min, 2) if elapsed_min is not None else 0.0,
        "host":             str(manifest.get("host") or ""),
        "image":            image,
        "code_revision":    str(manifest.get("code_revision") or ""),
        "pid":              int(manifest.get("pid") or 0),
        "session_dir":      str(session_dir),
        "tick_count":       int(state.get("tick") or 0),
    }


# ---------------------------------------------------------------------------
# §2 Workload
# ---------------------------------------------------------------------------
def collect_workload(
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    wl = manifest.get("workload") or {}
    return {
        "framework":         str(state.get("framework") or manifest.get("framework") or ""),
        "framework_version": str(manifest.get("framework_version") or ""),
        "model_name":        str(state.get("model_name") or manifest.get("model_name") or ""),
        "model_path":        str(state.get("model_path") or manifest.get("model_path") or ""),
        "model_class":       str(state.get("model_class") or ""),
        "gpu_type":          str(state.get("gpu_type") or manifest.get("gpu_type") or ""),
        "tp":                _to_int(manifest.get("tp")),
        "conc":              _to_int(wl.get("conc")),
        "isl":               _to_int(wl.get("isl")),
        "osl":               _to_int(wl.get("osl")),
        "max_model_len":     _to_int(wl.get("max_model_len")),
        "precision":         str(wl.get("precision") or ""),
        "objective":         dict(manifest.get("objective") or {"kind": "time_only", "value": None}),
    }


# ---------------------------------------------------------------------------
# §3 Baseline
# ---------------------------------------------------------------------------
def collect_baseline(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    last_b = state.get("last_baseline") or {}
    workspace_str = last_b.get("workspace") or ""
    # Re-root container-style paths (e.g. ``/workspace/runs/baseline/<sid>/``)
    # under the actual on-disk session_dir so we can still read
    # ``benchmark_report.json`` from a wekafs view. See ``_resolve_under_session``.
    workspace = _resolve_under_session(session_dir, workspace_str)
    if workspace_str and workspace is None:
        warnings.append(
            f"baseline workspace {workspace_str!r} does not resolve under {session_dir}; "
            "ttft_mean_ms / e2el_mean_ms will be null."
        )
    report_path = _find_benchmark_report(workspace) if workspace else None
    report = _load_json_safe(report_path, warnings) if report_path else None

    _, ttft, _tpot, e2el = _benchmark_report_metrics(report if isinstance(report, dict) else None)

    # Symmetric to A2 (final.ttft validate_stack disk walk): when
    # state.last_baseline.workspace doesn't resolve to a readable
    # benchmark_report.json, but ``runs/baseline/<hash>/benchmark_*/
    # benchmark_report.json`` exists on disk, use the most recent. Real
    # production sessions (e.g. zgong's V2) hit this gap — wekafs view
    # has the report file but the state-recorded workspace doesn't
    # match the wekafs root.
    ttft_source: str | None = "state_workspace" if ttft is not None else None
    if ttft is None:
        baseline_root = session_dir / "runs" / "baseline"
        candidates = sorted(
            (
                p
                for task_dir in baseline_root.iterdir()
                if task_dir.is_dir()
                for p in _benchmark_report_candidates(task_dir)
            ) if baseline_root.exists() else [],
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
        history.append({
            "ts":            a.get("ts"),
            "task_id":       a.get("task_id"),
            "status":        a.get("status"),
            "decision":      a.get("decision"),
            "key_metric":    _to_float(a.get("key_metric")),
            "workspace":     a.get("workspace"),
            "error_class":   a.get("error_class"),
        })

    # Disk-walking fallback: state.baseline_attempts is empty in many
    # production sessions even when multiple baseline runs left
    # ``runs/baseline/<hash>/`` dirs behind. Reconstruct so dashboards
    # don't show "0 baseline attempts" for sessions that obviously had
    # several. Each entry is marked ``status="reconstructed"`` so it's
    # visibly distinct from a state-recorded one.
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
    # Prefer the report we already located (state-resolved or disk-walked) so
    # the invocation surfaces the same benchmark variant as ttft/e2el. Fall
    # back to the resolved workspace on the off chance state still pointed
    # at the right place but the report itself was unreadable.
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

    # If we disk-walked into a benchmark dir, the matching config yaml
    # usually sits as a sibling (or one level up under <task>/). Prefer
    # that over state.baseline_config_path when the latter doesn't
    # resolve, so the yaml fallback in _extract_framework_args has a
    # real file to read.
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
        server_log_path, config_yaml=config_resolved,
    )
    invocation = {
        "framework_args":        args_str,
        "framework_args_source": args_source,
        "extra_envs":            _read_invocation_envs(config_resolved),
        "config_path":           _rel(config_resolved, session_dir) if config_resolved else (
            config_path_raw if config_path_raw else None
        ),
        "server_log_path":       _rel(server_log_path, session_dir) if server_log_path else None,
    }
    if args_source == "unknown":
        warnings.append(
            "framework_args extraction failed for "
            f"{(_rel(server_log_path, session_dir) if server_log_path else 'no server.log')}; "
            "tried server.log + yaml"
        )

    return {
        "throughput_tok_s_per_gpu": _to_float(state.get("baseline_tput")) or 0.0,
        "accuracy":                 _to_float(state.get("baseline_accuracy")) or 0.0,
        "ttft_mean_ms":             ttft,
        "e2el_mean_ms":             e2el,
        "ttft_e2el_source":         ttft_source,
        "config_path":              config_path_raw,
        "benchmark_report_path":    _rel(report_path, session_dir) if report_path else None,
        "attempts_history":         history,
        "failure_streak":           int(state.get("baseline_failure_streak") or 0),
        "invocation":               invocation,
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
    """
    root = session_dir / "runs" / "baseline"
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    reports = sorted(
        (
            p
            for task_dir in root.iterdir()
            if task_dir.is_dir()
            for p in _benchmark_report_candidates(task_dir)
        ),
        key=lambda p: p.stat().st_mtime,
    )
    for report_path in reports:
        bench_dir = report_path.parent
        # task dir = <sd>/runs/baseline/<HASH>; benchmark reports may sit
        # directly below it or under warmup_round/measure_round.
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
                    report_path.stat().st_mtime, tz=timezone.utc,
                ).isoformat(timespec="seconds")
            except OSError:
                ts_iso = ""
        report = _load_json_safe(report_path, warnings)
        out_tput, _ttft, _tpot, _e2el = _benchmark_report_metrics(
            report if isinstance(report, dict) else None
        )
        out.append({
            "ts":           ts_iso,
            "task_id":      task_dir.name,
            "status":       "reconstructed",
            "decision":     "",
            "key_metric":   out_tput,
            "workspace":    _rel(task_dir, session_dir) or str(task_dir),
            "error_class":  None,
        })
    return out


# ---------------------------------------------------------------------------
# §4 Final (validated)
# ---------------------------------------------------------------------------
def collect_final(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
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
    ttft_e2el_source = "current_best" if ttft is not None else "unavailable"

    # Disk-walk reconstruction. Coordinator usually leaves
    # ``current_best.ttft_mean_ms`` unset — the value lives only in the
    # actual benchmark_report.json on disk. Walk validate_stack first
    # (most authoritative: it's the run that produced the validated
    # cumulative gain), fall back to the latest stack entry's workspace.
    reconstructed_report: Path | None = None
    if ttft is None:
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
        session_dir, state, reconstructed_report, warnings,
    )

    return {
        "throughput_tok_s_per_gpu":          _to_float(cb.get("tput")),
        "cumulative_gain_pct_validated":     _to_float(state.get("cumulative_gain_validated")) or 0.0,
        "cumulative_gain_pct_per_round_sum": _to_float(state.get("cumulative_gain")) or 0.0,
        "validated_at_stack_len":            val_stack_len,
        "validated_ts":                      str(state.get("cumulative_gain_validated_ts") or ""),
        "stack_changed_after_validation":    stack_len > val_stack_len > 0,
        "extra_server_args":                 str(cb.get("extra_server_args") or ""),
        "extra_envs":                        dict(cb.get("extra_envs") or {}),
        "action_path":                       action_path,
        "ttft_mean_ms":                      ttft,
        "e2el_mean_ms":                      e2el,
        "ttft_e2el_source":                  ttft_e2el_source,
        "invocation":                        invocation,
        "closing_phase_entered":             bool(state.get("closing_started_unix") or 0),
        "closing_started_unix":              float(state.get("closing_started_unix") or 0.0),
        "closing_report_task_id":            str(state.get("closing_report_task_id") or ""),
    }


def _find_latest_validate_stack_report(session_dir: Path) -> Path | None:
    """Most-recent validate_stack benchmark_report.json under session_dir.

    validate_stack is the action that re-runs the entire optimization
    stack to produce a confirmed cumulative gain number, so its
    benchmark report is the authoritative source for "what does the
    final stack actually clock at".
    """
    root = session_dir / "runs" / "validate_stack"
    if not root.exists():
        return None
    return _latest_benchmark_report(
        p
        for task_dir in root.iterdir()
        if task_dir.is_dir()
        for p in _benchmark_report_candidates(task_dir)
    )


def _find_current_best_report(
    session_dir: Path, state: dict[str, Any],
) -> Path | None:
    """Best-effort benchmark report for ``state.current_best``.

    Some production states keep final latency on disk only while
    ``current_best.workspace`` points directly at the benchmark directory.
    Others omit workspace but still carry enough action/variant/tput
    identity to match the report under ``runs/<action>/``.
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
    session_dir: Path, state: dict[str, Any],
) -> Path | None:
    """Last optimization_stack entry's benchmark_report.json (best-effort).

    Used when no validate_stack run was recorded — the topmost stack
    entry's KEEP run is the next-best evidence of "the final
    configuration's measured throughput / latency".
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
    session_dir: Path, entry: dict[str, Any],
) -> Path | None:
    """Match a report under ``runs/<action>/`` by variant name and tput.

    This is intentionally conservative: it is used only when state lacks a
    usable workspace. A tput match wins, variant path match is the next best
    signal, and reports without latency are ignored because they cannot
    repair ``final.ttft/e2el``.
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
        out_tput, ttft, _tpot, e2el = _benchmark_report_metrics(
            report if isinstance(report, dict) else None
        )
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
    """Best-effort :class:`BenchmarkInvocation` for the final stack run.

    The "config" we surface here is the ``baseline_config.with_envs.yaml``
    sibling of whichever benchmark_report we used for ttft / e2el. That
    is, the file the workload was actually launched with — so an
    operator can replay the exact same command + envs.
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
        server_log_path, config_yaml=config_path,
    )
    if args_source == "unknown":
        warnings.append(
            "framework_args extraction failed for "
            f"{(_rel(server_log_path, session_dir) if server_log_path else 'no server.log')}; "
            "tried server.log + yaml"
        )
    return {
        "framework_args":        args_str,
        "framework_args_source": args_source,
        "extra_envs":            _read_invocation_envs(config_path),
        "config_path":           _rel(config_path, session_dir) if config_path else None,
        "server_log_path":       _rel(server_log_path, session_dir) if server_log_path else None,
    }


# ---------------------------------------------------------------------------
# §5 Phase timeline
# ---------------------------------------------------------------------------
# Action labels whose ``<action>_attempts`` lists feed the phase timeline
# and capability tallies. Carries BOTH the post-merge ``explore`` action
# and the legacy ``backends`` / ``params`` / ``validate_stack`` names so
# that breakdown can reprocess archived (pre-merge) sessions, whose
# state.json still writes the legacy ``*_attempts`` lists, alongside
# current sessions that write ``explore_attempts``. Missing lists are
# skipped harmlessly, so a session only ever populates its own vocabulary.
_AUDIT_ACTIONS = (
    "baseline", "profile", "explore",
    "backends", "params", "validate_stack",
    "sweep", "roofline",
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


def _iso_z(ts: Any) -> str:
    """Normalise any ISO-8601 timestamp to a canonical ``...Z`` UTC form.

    The two timeline sources disagree on suffix: the journal emits
    ``2026-06-02T06:08:30Z`` while ``state.phase_history`` rows carry
    ``2026-06-02T06:08:08+00:00``. Mixing them leaves segment boundary
    rows in the ``+00:00`` shape, which strict front-end date parsers
    render blank, and makes the lexicographic window comparison fragile
    at equal-second boundaries. Collapsing both to second-precision
    ``Z`` keeps display and ``[entered_ts, exit_ts)`` matching consistent.
    Returns the input unchanged when it is empty or unparseable.
    """
    if ts is None:
        return ""
    s = str(ts).strip()
    if not s:
        return ""
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return s
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_optimization_journal(
    session_dir: Path | None, warnings: list[str],
) -> list[dict[str, Any]]:
    """Read ``reports/optimization_journal.json`` entries.

    The journal is the orchestrator's own append-only, atomically-flushed
    decision log (one row per KEEP / REVERT / no_promote, carrying
    ``phase`` + ``ts`` + ``kind`` + ``change`` + ``outcome`` + gain). It
    is the canonical action ledger and — unlike the per-action
    ``*_attempts`` audit lists — already records the target_analysis /
    roofline / specialist / explore-winner decisions that the audit lists
    omit. Returns ``[]`` when absent (legacy sessions); callers then fall
    back to the audit-list scrape alone.
    """
    if session_dir is None:
        return []
    data = _load_json_safe(
        session_dir / "reports" / "optimization_journal.json", warnings,
    )
    if not isinstance(data, dict):
        return []
    entries = data.get("entries")
    return entries if isinstance(entries, list) else []


def _journal_entry_to_event(e: dict[str, Any]) -> dict[str, Any]:
    """Map one optimization_journal entry to a phase_timeline event.

    Keeps the journal's ``phase`` so :func:`collect_phase_segments` can
    bucket the event by its declared phase (exact) instead of guessing
    from the ``ts`` window (which mis-files boundary actions such as the
    PRELUDE baseline).
    """
    metric = e.get("throughput_after")
    metric_kind = "output_throughput" if metric is not None else None
    if metric is None and e.get("gain_pct") is not None:
        metric = e.get("gain_pct")
        metric_kind = "gain_pct"
    change = str(e.get("change") or "")
    kind = str(e.get("kind") or "")
    # ``kind == "other"`` is the journal's catch-all bucket (target_analysis /
    # roofline / specialist / explore quant candidates / sweep ...). The real
    # operation name lives in ``change`` -- surface that as the action so the
    # timeline shows the actual step instead of an opaque "other".
    if kind and kind.lower() != "other":
        action = kind
    else:
        action = change or kind or "other"
    return {
        "ts":              _iso_z(e.get("ts")),
        "action":          action,
        "task_id":         str(e.get("task_id") or ""),
        "kernel_id":       None,
        "status":          "",
        "decision":        str(e.get("outcome") or ""),
        "key_metric":      _to_float(metric),
        "key_metric_kind": metric_kind,
        "workspace":       None,
        "error_class":     e.get("error_class"),
        "phase":           str(e.get("phase") or ""),
        "change":          change,
        "extras":          {k: v for k, v in (
            ("variant_name", e.get("variant_name")),
            ("reason", e.get("reason")),
        ) if v},
    }


def collect_phase_timeline(
    session_dir: Path | None,
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Flat, chronological action timeline.

    Merges two complementary sources so no action family is dropped:

    * ``reports/optimization_journal.json`` — the canonical decision log
      (target_analysis / baseline / roofline / specialist / explore
      winners / sweep). Carries ``phase`` for exact segment attribution.
    * the per-action ``*_attempts`` audit lists + ``kernel_opt`` /
      ``kernel_integrate`` histories — add per-attempt rows (incl.
      failures) and the kernel lanes the journal records only as a single
      KEEP.

    Events are de-duplicated by ``(action, ts-to-second, change)`` with
    the journal copy winning, then sorted by ``ts``. Passing
    ``session_dir=None`` degrades gracefully to the audit-list scrape
    (used by unit fixtures that have no on-disk journal).
    """
    events: list[dict[str, Any]] = []

    # ── Source 1: canonical journal (preferred; carries phase) ──
    for e in _load_optimization_journal(session_dir, warnings):
        if isinstance(e, dict):
            events.append(_journal_entry_to_event(e))

    # ── Source 2: per-action audit lists (complementary / legacy) ──
    for action in _AUDIT_ACTIONS:
        attempts = state.get(f"{action}_attempts") or []
        if not isinstance(attempts, list):
            continue
        for entry in attempts:
            if not isinstance(entry, dict):
                continue
            events.append({
                "ts":             entry.get("ts") or "",
                "action":         action,
                "task_id":        str(entry.get("task_id") or ""),
                "kernel_id":      None,
                "status":         str(entry.get("status") or ""),
                "decision":       str(entry.get("decision") or ""),
                "key_metric":     _to_float(entry.get("key_metric")),
                "key_metric_kind": entry.get("key_metric_kind"),
                "workspace":      entry.get("workspace"),
                "error_class":    entry.get("error_class"),
                "phase":          "",
                "change":         action,
                "extras":         dict(entry.get("extras") or {}),
            })

    # Kernel opt attempts (per-kernel history -> flatten to per-attempt rows)
    kernel_opt = state.get("kernel_opt_attempts") or {}
    if isinstance(kernel_opt, dict):
        for kid, ent in kernel_opt.items():
            if not isinstance(ent, dict):
                continue
            for h in ent.get("history") or []:
                if not isinstance(h, dict):
                    continue
                events.append({
                    "ts":          h.get("ts") or ent.get("last_ts") or "",
                    "action":      "kernel_opt",
                    "task_id":     "",
                    "kernel_id":   str(kid),
                    "status":      "",
                    "decision":    str(h.get("decision") or ""),
                    "key_metric":  None,
                    "key_metric_kind": None,
                    "workspace":   None,
                    "error_class": None,
                    "phase":       "",
                    "change":      f"kernel_opt:{kid}",
                    "extras":      {},
                })

    # Integrate attempts (decision history per patch key)
    integ = state.get("kernel_integrate_attempts") or {}
    if isinstance(integ, dict):
        for key, ent in integ.items():
            if not isinstance(ent, dict):
                continue
            for a in ent.get("attempts") or []:
                if not isinstance(a, dict):
                    continue
                kid = str(ent.get("kernel_id") or "")
                events.append({
                    "ts":          a.get("ts") or "",
                    "action":      "integrate",
                    "task_id":     "",
                    "kernel_id":   kid,
                    "status":      str(a.get("status") or ""),
                    "decision":    str(a.get("decision") or ""),
                    "key_metric":  _to_float(a.get("gain_pct")),
                    "key_metric_kind": "gain_pct",
                    "workspace":   a.get("workspace"),
                    "error_class": None,
                    "phase":       "",
                    "change":      f"integrate:{kid}",
                    "extras":      {"patch_path": ent.get("patch_path"),
                                    "report_path": a.get("report_path")},
                })

    # Canonicalise every ts to ``...Z`` so journal (Z) and legacy audit /
    # phase_history (+00:00) rows dedup, sort and render consistently.
    for ev in events:
        ev["ts"] = _iso_z(ev.get("ts"))

    # De-dup: journal rows are appended first and win on collision.
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for ev in events:
        key = (
            str(ev.get("action") or ""),
            (str(ev.get("ts") or ""))[:19],
            str(ev.get("change") or ev.get("task_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ev)

    deduped.sort(key=lambda e: e.get("ts") or "")
    return deduped


# ---------------------------------------------------------------------------
# §6 Capability summary
# ---------------------------------------------------------------------------
def _capability_for_action(
    state: dict[str, Any], action: str,
) -> dict[str, Any]:
    """Per-action capability tally.

    Primary source is the rich ``<action>_attempts`` audit list added in
    HL V2 (each entry has ``decision`` in
    ``{promoted, salvaged, rejected, failed, ...}``). On older V1 sessions
    or partial state.json snapshots those lists are missing or empty even
    when the action actually ran successfully, so we also walk
    ``optimization_stack`` as a fallback: every stack entry whose
    ``action`` matches counts as a KEEP (the stack only collects
    successfully promoted entries by construction). Without this fallback
    sessions that had a clear ``explore:vllm_kv_fp8`` final.action_path
    would still report ``explore: not_attempted`` in capability_summary.
    """
    attempts_list = state.get(f"{action}_attempts") or []
    n_attempts = len(attempts_list) if isinstance(attempts_list, list) else 0
    n_keeps = sum(
        1 for a in attempts_list
        if isinstance(a, dict) and a.get("decision") in ("promoted", "salvaged")
    ) if isinstance(attempts_list, list) else 0

    # Fallback / augmentation from optimization_stack — only adopted
    # entries land here. Counts an entry once per action; the kernel
    # integrate path uses ``action="integrate"`` so it does not collide
    # with explore/sweep here.
    stack = state.get("optimization_stack") or []
    if isinstance(stack, list):
        stack_keeps = sum(
            1 for e in stack
            if isinstance(e, dict) and str(e.get("action") or "") == action
        )
    else:
        stack_keeps = 0
    if stack_keeps > n_keeps:
        n_keeps = stack_keeps
        # Stack-derived keeps imply at least as many attempts.
        if n_attempts < stack_keeps:
            n_attempts = stack_keeps

    status = (
        "kept"      if n_keeps > 0 else
        "tried"     if n_attempts > 0 else
        "not_attempted"
    )
    return {
        "status":   status,
        "attempts": n_attempts,
        "keeps":    n_keeps,
    }


def collect_capability_summary(
    state: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    oob_invocations: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    # Final integrate (e2e) outcome per kernel. A kernel-opt KEEP that was
    # REVERTED at integrate is NOT a real adoption -- end-to-end throughput
    # regressed and the patch never entered the final stack -- so it must
    # not inflate the geak/oob "kept" tally. Reading the integrate decision
    # + e2e gain here keeps capability_summary in lock-step with
    # kernel_lifecycle.final_decision (which already reads integrate).
    integ = state.get("kernel_integrate_attempts") or {}
    integ_by_kid: dict[str, dict[str, Any]] = {}
    if isinstance(integ, dict):
        for ent in integ.values():
            if not isinstance(ent, dict):
                continue
            kid = str(ent.get("kernel_id") or "")
            if not kid:
                continue
            integ_by_kid[kid] = {
                "decision": str(ent.get("last_decision") or "").upper(),
                "e2e_gain_pct": _to_float(ent.get("best_gain_pct")),
            }

    # GEAK / OOB driven by actual invocations on disk (most reliable),
    # reconciled against the integrate (e2e) verdict.
    def _from_invocations(invs: list[dict[str, Any]]) -> dict[str, Any]:
        attempts = len(invs)
        adopted = 0
        reverted = 0
        best_e2e: float | None = None
        for v in invs:
            if v.get("decision") != "KEEP":
                continue
            outcome = integ_by_kid.get(str(v.get("kernel_id") or ""))
            if outcome is None:
                # micro-KEEP with no integrate record -> stands as kept.
                adopted += 1
                continue
            g = outcome["e2e_gain_pct"]
            if g is not None and (best_e2e is None or g > best_e2e):
                best_e2e = g
            if outcome["decision"] in ("REVERT", "REJECT"):
                reverted += 1
            else:
                adopted += 1
        status = (
            "kept"          if adopted > 0 else
            "reverted"      if reverted > 0 else
            "attempted"     if attempts > 0 else
            "not_attempted"
        )
        row: dict[str, Any] = {
            "status": status, "attempts": attempts, "keeps": adopted,
        }
        if reverted:
            row["reverts"] = reverted
        if best_e2e is not None:
            row["e2e_gain_pct"] = best_e2e
        return row

    geak_cap = _from_invocations(geak_invocations)
    oob_cap = _from_invocations(oob_invocations)

    # Legacy capability rows. Archived (pre-merge) sessions carry their
    # search activity under ``backends_search`` / ``params_search`` and
    # the ``validate_stack`` action; surface those rows so breakdown can
    # reprocess them. On a current (post-merge) session these rows are
    # ``not_attempted`` while ``explore`` (below) carries the activity.
    backends = _capability_for_action(state, "backends")
    backends_search = state.get("backends_search") or {}
    if isinstance(backends_search, dict):
        backends["tested"] = len(backends_search.get("tested") or {})
        if backends_search.get("accepted"):
            backends["best_gain_pct"] = max(
                (_to_float(v.get("gain_pct")) or 0.0
                 for v in backends_search["accepted"]
                 if isinstance(v, dict)),
                default=None,
            )

    params = _capability_for_action(state, "params")
    params_search = state.get("params_search") or {}
    if isinstance(params_search, dict):
        params["tested"] = len(params_search.get("tested") or {})
        if params_search.get("accepted"):
            params["best_gain_pct"] = max(
                (_to_float(v.get("gain_pct")) or 0.0
                 for v in params_search["accepted"]
                 if isinstance(v, dict)),
                default=None,
            )

    validate = _capability_for_action(state, "validate_stack")
    validate["last_validated_gain_pct"] = _to_float(
        state.get("cumulative_gain_validated")
    )

    sweep_cap = _capability_for_action(state, "sweep")
    last_sweep = state.get("last_sweep") or {}
    if isinstance(last_sweep, dict):
        sweep_cap["grid_size"] = _to_int(last_sweep.get("grid_size"))
        bo = last_sweep.get("best_overall")
        if isinstance(bo, dict):
            sweep_cap["best_throughput"] = _to_float(
                bo.get("output_throughput") or bo.get("tput")
            )
        if sweep_cap.get("attempts", 0) > 0:
            sweep_cap["status"] = "completed"

    # merged explore action capability row. Carries
    # the unified explore_search ledger activity, including the
    # validated cumulative gain previously surfaced by validate_stack.
    explore = _capability_for_action(state, "explore")
    explore["last_validated_gain_pct"] = _to_float(
        state.get("cumulative_gain_validated")
    )
    explore_search = state.get("explore_search") or {}
    if isinstance(explore_search, dict):
        explore["tested"] = len(explore_search.get("tested") or {})
        accepted_entries = [
            v for v in (explore_search.get("accepted") or [])
            if isinstance(v, dict)
        ]
        if accepted_entries:
            explore["best_gain_pct"] = max(
                (_to_float(v.get("gain_pct")) or 0.0 for v in accepted_entries),
                default=None,
            )
        keep_unstable_count = sum(
            1 for entry in (explore_search.get("rejected") or [])
            if isinstance(entry, dict)
            and entry.get("reason") == "stack_unstable"
        )
        if keep_unstable_count:
            explore["keep_unstable_count"] = keep_unstable_count
        explore["winners_history"] = len(
            explore_search.get("winners_history") or []
        )

    # specialist sub-agent capability row. Counts
    # are derived from ``specialist_rounds`` so they always agree with
    # ``specialist_runs`` (single source).
    specialist_row = _specialist_capability_row(state)
    return {
        "geak":           geak_cap,
        "oob":            oob_cap,
        # primary (post-merge) row; backends / params / validate_stack
        # are kept as compatibility rows for archived sessions.
        "explore":        explore,
        "backends":       backends,
        "params":         params,
        "sweep":          sweep_cap,
        "validate_stack": validate,
        # sub-agent visibility row.
        "specialist":     specialist_row,
    }


def _specialist_capability_row(state: dict[str, Any]) -> dict[str, Any]:
    """Derive ``capability_summary.specialist`` from
    ``specialist_rounds``.

    Single source of truth per Inv-12.2: the row never diverges from
    the ``specialist_runs`` section because both read the same
    SharedState ledger.

    The headline counts (``status`` / ``attempts`` / ``keeps`` /
    ``tested``) aggregate across every domain. ``by_specialist``
    breaks them out per SpecialistDomain.key so the dashboard can
    render six (or seven, including session_steward) per-specialist
    cards without re-aggregating client-side.
    """
    rounds = state.get("specialist_rounds") or []
    if not isinstance(rounds, list) or not rounds:
        return {
            "status":    "not_attempted",
            "attempts":  0,
            "keeps":     0,
            "tested":    0,
            "by_specialist": _empty_by_specialist_capability(),
        }

    attempts = 0
    proposals_total = 0
    proposals_kept = 0
    # Per-domain counters. Seeded with the catalogue so consumers
    # iterate every known specialist without presence checks; unknown
    # domain strings still in state survive (forward compat with new
    # SpecialistDomain entries).
    by_specialist_raw: dict[str, dict[str, int]] = {
        d: {"attempts": 0, "keeps": 0, "tested": 0, "rejected": 0}
        for d in _SPECIALIST_DOMAIN_KEYS
    }

    for r in rounds:
        if not isinstance(r, dict):
            continue
        attempts += 1
        proposals_total += int(r.get("proposals_total") or 0)
        proposals_kept += int(r.get("proposals_kept") or 0)

        # Per-domain tallies. Trust ``domain_breakdown`` when the
        # Coordinator wrote it (most accurate, especially for parallel
        # multi-domain rounds); otherwise fall back to ``domains[]``
        # split evenly across the round's totals.
        round_breakdown = r.get("domain_breakdown")
        if isinstance(round_breakdown, dict) and round_breakdown:
            for dom, payload in round_breakdown.items():
                if not isinstance(payload, dict):
                    continue
                bucket = by_specialist_raw.setdefault(str(dom), {
                    "attempts": 0, "keeps": 0, "tested": 0, "rejected": 0,
                })
                bucket["attempts"] += int(payload.get("dispatched") or 0)
                bucket["keeps"] += int(payload.get("proposals_kept") or 0)
                bucket["tested"] += int(payload.get("proposals_total") or 0)
                bucket["rejected"] += int(payload.get("proposals_rejected") or 0)
        else:
            # Round predates ``domain_breakdown``; impute equal share
            # across the round's knowledge-domain tags (or the legacy
            # ``domains[]`` list) so the per-domain numbers add up to the
            # parent row even on legacy rounds.
            domains = r.get("tags") or r.get("domains") or []
            if isinstance(domains, list) and domains:
                share_total = int(r.get("proposals_total") or 0) // len(domains)
                share_kept = int(r.get("proposals_kept") or 0) // len(domains)
                for dom in domains:
                    bucket = by_specialist_raw.setdefault(str(dom), {
                        "attempts": 0, "keeps": 0, "tested": 0, "rejected": 0,
                    })
                    bucket["attempts"] += 1
                    bucket["tested"] += share_total
                    bucket["keeps"] += share_kept

    if attempts == 0:
        status = "not_attempted"
    elif proposals_kept > 0:
        status = "kept"
    elif proposals_total > 0:
        status = "tried"
    else:
        status = "attempted"

    # Materialize the by_specialist dict with status derived per-domain.
    by_specialist: dict[str, dict[str, Any]] = {}
    for dom, raw in by_specialist_raw.items():
        if raw["attempts"] == 0:
            dom_status = "not_attempted"
        elif raw["keeps"] > 0:
            dom_status = "kept"
        elif raw["tested"] > 0:
            dom_status = "tried"
        else:
            dom_status = "attempted"
        by_specialist[dom] = {
            "status":   dom_status,
            "attempts": raw["attempts"],
            "keeps":    raw["keeps"],
            "tested":   raw["tested"],
        }

    return {
        "status":         status,
        "attempts":       attempts,
        "keeps":          proposals_kept,
        "tested":         proposals_total,
        "by_specialist":  by_specialist,
    }


def _empty_by_specialist_capability() -> dict[str, dict[str, Any]]:
    """Seed every catalogue domain with a not_attempted CapabilityEntry.

    Stable-shape contract: the dashboard never has to ``KeyError``-
    guard when iterating ``capability_summary.specialist.by_specialist``
    on a session that didn't dispatch every domain.
    """
    return {
        d: {"status": "not_attempted", "attempts": 0, "keeps": 0, "tested": 0}
        for d in _SPECIALIST_DOMAIN_KEYS
    }


# ---------------------------------------------------------------------------
# §7 / §8 GEAK / OOB invocations
# ---------------------------------------------------------------------------
def _kernel_agent_run_dirs(session_dir: Path) -> list[Path]:
    """All ``<sd>/kernel-agent/runs/<sid>/`` dirs (and legacy fallbacks).

    Post-migration, ``Coordinator.kernel_request_handlers`` spawns
    ``kernel_optimization.py --workspace-path $SD`` and the tool creates
    ``<sd>/kernel-agent/runs/<session_id>/...``. Pre-migration sessions
    landed under ``<sd>/kernel-agent-workspace/kernel-agent/runs/...``
    (Coordinator used to pass ``--workspace-path $SD/kernel-agent-workspace``)
    or, even older, ``<sd>/kernel-agent-workspace/<kid>/kernel-agent/runs/...``.
    We scan all three so historical sessions still render.
    """
    candidates: list[Path] = []
    # Canonical (current): <sd>/kernel-agent/runs/<sid>/
    new_root = session_dir / "kernel-agent" / "runs"
    if new_root.is_dir():
        for sub in new_root.glob("*"):
            if sub.is_dir() and sub not in candidates:
                candidates.append(sub)
    # Legacy double-nested: <sd>/kernel-agent-workspace/kernel-agent/runs/<sid>/
    legacy_root = session_dir / "kernel-agent-workspace"
    if legacy_root.is_dir():
        for sub in (legacy_root / "kernel-agent" / "runs").glob("*"):
            if sub.is_dir() and sub not in candidates:
                candidates.append(sub)
        # Even older per-kernel form: <sd>/kernel-agent-workspace/<kid>/kernel-agent/runs/<sid>/
        for kid_dir in legacy_root.glob("*/kernel-agent/runs/*"):
            if kid_dir.is_dir() and kid_dir not in candidates:
                candidates.append(kid_dir)
    return candidates


def _parse_invocation_attempt(
    attempt: dict[str, Any],
    run_dir: Path,
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Parse one ``optimization_attempts.jsonl`` row into an Invocation.

    Per-attempt fields are read from ``attempt`` directly. Kernel-level
    artifacts (verification / result) are referenced by path only — the
    KEEP/PARTIAL/REVERT decision is NOT stamped here because
    ``results/<kid>.json`` and ``verification/<kid>.json`` describe the
    kernel's BEST attempt, not every attempt. Stamping happens in
    :func:`_stamp_kernel_level_decisions` after all attempts are parsed.
    """
    kid = str(attempt.get("kernel_id") or "")
    backend = str(attempt.get("backend") or "").lower()
    attempt_id = str(
        attempt.get("attempt_id")
        or attempt.get("id")
        or attempt.get("run_id")
        or "",
    )

    # Resolve auxiliary paths
    prompt_path: Path | None = None
    for p in (run_dir / "prompts").glob(f"{attempt_id}*") if attempt_id else []:
        prompt_path = p
        break

    optimized_files: list[str] = []
    if attempt_id:
        for p in sorted((run_dir / "optimized").glob(f"{attempt_id}*")):
            optimized_files.append(_rel(p, session_dir) or str(p))

    verification_path = run_dir / "verification" / f"{kid}.json" if kid else None
    result_path = run_dir / "results" / f"{kid}.json" if kid else None

    # Per-attempt decision: derived ONLY from this attempt's own fields.
    decision = str(attempt.get("decision") or "").upper()
    if not decision:
        status = str(attempt.get("status") or "").lower()
        if status in ("failed", "error", "crashed"):
            decision = "FAILED"
        # otherwise leave empty; kernel-level decision is stamped later

    # Per-attempt speedup (preferred) — falls back to None if not recorded.
    micro_speedup = _to_float(
        attempt.get("speedup")
        or attempt.get("micro_speedup")
    )

    return {
        "kernel_id":       kid,
        "attempt_id":      attempt_id,
        "run_id":          str(attempt.get("run_id") or run_dir.name),
        "ts":              str(attempt.get("ts") or attempt.get("started_at") or ""),
        "backend":         backend,
        "model":           attempt.get("model"),
        "kernel_metadata": _shape_kernel_metadata({}, attempt),
        "prompt_path":     _rel(prompt_path, session_dir) if prompt_path else None,
        "optimized_files": optimized_files,
        "result_path":     _rel(result_path, session_dir) if result_path and result_path.exists() else None,
        "verification_path": _rel(verification_path, session_dir) if verification_path and verification_path.exists() else None,
        "decision":        decision,
        "micro_speedup":   micro_speedup,
        # compile_passed / correctness_passed are kernel-level (in verification.json);
        # stamped later if this attempt is the BEST one for the kernel.
        "compile_passed":  None,
        "correctness_passed": None,
        "best_artifact_path": None,
        "error":           attempt.get("error"),
        "cli_log_path":    None,
    }


def _stamp_kernel_level_decisions(
    invocations: list[dict[str, Any]],
    run_dirs: list[Path],
    session_dir: Path,
    warnings: list[str],
) -> None:
    """Stamp the kernel-level KEEP/PARTIAL/REVERT decision onto the
    single BEST attempt for each kernel.

    For each ``(run_dir, kernel_id)`` group, find ``results/<kid>.json``
    and ``verification/<kid>.json`` (kernel-level, one file per kernel,
    written at the END of the kernel-agent run). Pick the attempt with
    the highest ``micro_speedup`` (ties: latest ts), and stamp the
    kernel-level decision + verification fields onto only that attempt.
    All other attempts for the same kernel keep their per-attempt
    ``decision`` (empty string for non-failed in-progress steps).
    """
    # Group attempts by (run_id, kernel_id) — same kid in different
    # run_dirs is treated as separate sessions.
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for inv in invocations:
        key = (inv.get("run_id") or "", inv.get("kernel_id") or "")
        if not key[1]:
            continue
        groups.setdefault(key, []).append(inv)

    # Map run_id -> Path for kernel-level json lookup.
    run_by_id: dict[str, Path] = {rd.name: rd for rd in run_dirs}

    for (run_id, kid), atts in groups.items():
        run_dir = run_by_id.get(run_id)
        if run_dir is None:
            continue
        result_path = run_dir / "results" / f"{kid}.json"
        verification_path = run_dir / "verification" / f"{kid}.json"
        result = _load_json_safe(result_path if result_path.exists() else None, warnings) or {}
        verification = _load_json_safe(verification_path if verification_path.exists() else None, warnings) or {}

        decision = ""
        proposal = result.get("proposal") if isinstance(result, dict) else None
        if isinstance(proposal, dict):
            decision = str(proposal.get("decision") or "").upper()

        if not decision and not verification:
            continue  # nothing to stamp

        # Pick the BEST attempt: highest micro_speedup, ties broken by latest ts.
        def _attempt_key(a: dict[str, Any]) -> tuple[float, str]:
            spd = a.get("micro_speedup")
            return (
                float(spd) if isinstance(spd, (int, float)) else float("-inf"),
                str(a.get("ts") or ""),
            )

        # Attribute the kernel-level KEEP to the backend the agent actually
        # adopted: ``verification.best_attempt_id`` (exact) then
        # ``best_backend``. The per-attempt ``micro_speedup`` is usually
        # absent in ``optimization_attempts.jsonl`` (the speedup lives only
        # in verification.json), so the bare micro/ts heuristic would tie-
        # break onto whichever attempt ran last -- frequently a FAILED lane
        # -- and mislabel it as the winner. Falls back to the heuristic only
        # when verification carries no backend hint.
        best = None
        if isinstance(verification, dict):
            want_id = str(verification.get("best_attempt_id") or "")
            want_backend = str(verification.get("best_backend") or "").lower()
            if want_id:
                best = next(
                    (a for a in atts if str(a.get("attempt_id") or "") == want_id),
                    None,
                )
            if best is None and want_backend:
                cands = [
                    a for a in atts
                    if str(a.get("backend") or "").lower() == want_backend
                ]
                if cands:
                    best = max(cands, key=_attempt_key)
        if best is None:
            best = max(atts, key=_attempt_key)

        if decision:
            best["decision"] = decision
        if isinstance(verification, dict):
            if best.get("micro_speedup") is None and verification.get("micro_speedup") is not None:
                best["micro_speedup"] = _to_float(verification.get("micro_speedup"))
            best["compile_passed"] = verification.get("compile_passed")
            best["correctness_passed"] = verification.get("correctness_passed")
            best["best_artifact_path"] = (
                verification.get("best_artifact_path")
                or (result.get("best_artifact_path") if isinstance(result, dict) else None)
            )
        if isinstance(result, dict) and result.get("cli_log_path"):
            best["cli_log_path"] = result["cli_log_path"]
        # Refresh kernel metadata from the (richer) result file
        best["kernel_metadata"] = _shape_kernel_metadata(result, {
            "name": best.get("kernel_metadata", {}).get("name"),
            "source_file": best.get("kernel_metadata", {}).get("source_file"),
        })


def _shape_kernel_metadata(
    result: dict[str, Any],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    """Best-effort kernel metadata for the UI.

    Prefers fields explicit in ``result['kernel_metadata']`` (newer
    kernel_optimization.py writes that); otherwise falls back to the
    attempt's own metadata.
    """
    meta = result.get("kernel_metadata") if isinstance(result, dict) else None
    if isinstance(meta, dict) and meta:
        return {
            "name":        meta.get("name") or "",
            "source_file": meta.get("source_file") or result.get("source_file") or "",
            "shapes":      list(meta.get("shapes") or []),
            "gpu_pct":     _to_float(meta.get("gpu_pct")),
            "arithmetic_intensity": _to_float(meta.get("arithmetic_intensity")),
        }
    return {
        "name":        str(attempt.get("name") or ""),
        "source_file": str(attempt.get("source_file") or ""),
        "shapes":      [],
        "gpu_pct":     None,
        "arithmetic_intensity": None,
    }


def _infer_run_dir_kernel_id(run_dir: Path) -> str:
    """Recover the kernel id for a run dir whose ``optimization_attempts``
    rows omit ``kernel_id``.

    The kernel agent writes exactly one ``results/<kid>.json`` +
    ``verification/<kid>.json`` per optimised kernel. When a run dir holds a
    single such kid we can confidently attribute every id-less attempt in
    that run to it; ambiguous (multi-kid) run dirs are left untouched so we
    never mis-file an attempt.
    """
    kids: set[str] = set()
    for sub in ("results", "verification"):
        d = run_dir / sub
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            if p.stem:
                kids.add(p.stem)
    return next(iter(kids)) if len(kids) == 1 else ""


def collect_kernel_invocations(
    session_dir: Path,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(geak_invocations, oob_invocations)``.

    Reads ``optimization_attempts.jsonl`` from every kernel-agent run dir,
    splits attempts by ``backend`` field, and stamps the kernel-level
    KEEP/PARTIAL/REVERT decision onto only the BEST attempt for each
    kernel (via :func:`_stamp_kernel_level_decisions`).
    """
    all_invocations: list[dict[str, Any]] = []
    run_dirs = _kernel_agent_run_dirs(session_dir)
    for run_dir in run_dirs:
        attempts = _load_jsonl_safe(run_dir / "optimization_attempts.jsonl", warnings)
        parsed = [
            _parse_invocation_attempt(att, run_dir, session_dir, warnings)
            for att in attempts
        ]
        # Backfill kernel_id for attempts whose jsonl row omitted it (some
        # kernel-agent versions never stamp kernel_id into
        # optimization_attempts.jsonl). Without a kernel_id the invocation
        # can neither be attributed to its detected kernel (-> UI shows
        # "not_attempted") nor get its KEEP/REVERT decision stamped, even
        # though capability_summary still counts it in the global totals --
        # which is exactly the "attempts>0 but per-kernel not_attempted"
        # mismatch operators see.
        if any(not (inv.get("kernel_id") or "") for inv in parsed):
            inferred = _infer_run_dir_kernel_id(run_dir)
            if inferred:
                for inv in parsed:
                    if inv.get("kernel_id"):
                        continue
                    inv["kernel_id"] = inferred
                    rp = run_dir / "results" / f"{inferred}.json"
                    vp = run_dir / "verification" / f"{inferred}.json"
                    if inv.get("result_path") is None and rp.exists():
                        inv["result_path"] = _rel(rp, session_dir) or str(rp)
                    if inv.get("verification_path") is None and vp.exists():
                        inv["verification_path"] = _rel(vp, session_dir) or str(vp)
        for inv in parsed:
            backend = inv.get("backend") or ""
            if not backend:
                inv["backend"] = "unknown"
            all_invocations.append(inv)

    # Stamp kernel-level KEEP/PARTIAL/REVERT onto the BEST attempt per kernel
    # (kernel-agent writes one verification/<kid>.json + one results/<kid>.json
    # per kernel, not per attempt).
    _stamp_kernel_level_decisions(all_invocations, run_dirs, session_dir, warnings)

    geak: list[dict[str, Any]] = []
    oob: list[dict[str, Any]] = []
    for inv in all_invocations:
        backend = inv.get("backend") or ""
        if backend == "geak":
            geak.append(inv)
        elif backend in ("claude", "codex"):
            oob.append(inv)
        else:
            oob.append(inv)
    geak.sort(key=lambda e: (e.get("kernel_id") or "", e.get("ts") or ""))
    oob.sort(key=lambda e: (e.get("kernel_id") or "", e.get("ts") or ""))
    return geak, oob


# ---------------------------------------------------------------------------
# §9 Kernel lifecycle
# ---------------------------------------------------------------------------
def _scan_profile_reports(session_dir: Path) -> list[tuple[Path, Path]]:
    """Return list of ``(task_dir, benchmark_report.json)`` under runs/profile/."""
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


def _read_kernel_candidates(
    session_dir: Path, state: dict[str, Any], warnings: list[str],
) -> list[dict[str, Any]]:
    """Return the ``hot_kernels`` array from ``kernel_candidates.json``.

    The MAE-era dashboard required GPU duration + call count for every
    detected kernel. That data is NOT in benchmark_report.json; it lives
    in kernel-agent's ``kernel_candidates.json`` which is the input the
    kernel agent uses to decide what to optimize.

    Resolves the file via:
      1. ``state.last_trace_analyze.candidates_path`` — orchestrator-recorded path
      2. ``session_dir / kernel-agent / runs / <session_id> / kernel_candidates.json``
         (new layout after the all-artefacts-under-USER_DATA_PATH migration)
      3. ``session_dir / kernel-agent / **/kernel_candidates.json`` glob fallback (new)
      4. ``session_dir / kernel-agent-workspace / kernel-agent / runs / hyperloom /
         kernel_candidates.json`` (legacy double-nested layout from pre-migration
         sessions, kept for breakdown replay of historical runs)
      5. ``session_dir / kernel-agent-workspace / **/kernel_candidates.json`` glob fallback

    Legacy ``state.last_select_kernels`` field (pre-M4) was removed
    in this branch; historical state.json that still carries it is
    silently ignored — the on-disk glob fallbacks above keep
    breakdown replay working on old sessions.
    """
    sk = state.get("last_trace_analyze") or {}
    raw_path = sk.get("candidates_path") if isinstance(sk, dict) else None
    candidate_paths: list[Path] = []
    if raw_path:
        # The recorded path is usually a container path (/workspace/... or
        # /workspace/hyperloom/...) that doesn't exist on wekafs; rewrite the
        # prefix to the session_dir's actual on-disk root before falling back
        # to glob. The "kernel-agent" / "kernel-agent-workspace" anchors below
        # let us re-root either the new (single-nested) or legacy
        # (double-nested) layout.
        p = Path(str(raw_path))
        candidate_paths.append(p)
        for anchor in ("kernel-agent-workspace", "kernel-agent"):
            try:
                idx = p.parts.index(anchor)
                candidate_paths.append(session_dir.joinpath(*p.parts[idx:]))
                break
            except ValueError:
                continue
    # New layout (post-migration): tools write under <sd>/kernel-agent/runs/<session_id>/.
    candidate_paths.extend(
        sorted((session_dir / "kernel-agent").rglob("kernel_candidates.json"))
    )
    # Legacy double-nested layout: pre-migration sessions wrote under
    # <sd>/kernel-agent-workspace/kernel-agent/runs/hyperloom/. Keep the
    # globs around so historical sessions still rehydrate.
    candidate_paths.append(
        session_dir / "kernel-agent-workspace" / "kernel-agent"
        / "runs" / "hyperloom" / "kernel_candidates.json"
    )
    candidate_paths.extend(
        sorted((session_dir / "kernel-agent-workspace").rglob("kernel_candidates.json"))
    )
    for path in candidate_paths:
        if not path or not path.exists():
            continue
        data = _load_json_safe(path, warnings)
        if isinstance(data, dict):
            hk = data.get("hot_kernels")
            # Only accept a NON-empty list. An on-disk ``hot_kernels: []``
            # (the kernel-agent sometimes writes an empty candidates file for
            # a later trace round) must not short-circuit the
            # ``state.hot_kernels_top15`` fallback below -- otherwise detected
            # kernels vanish even though the orchestrator recorded them, and
            # any optimised kernel (k0xx) loses its row + geak/oob verdict.
            if isinstance(hk, list) and hk:
                return hk
    # Final fallback: state.last_trace_analyze.hot_kernels_top15.
    # This is what the orchestrator actually copied out of
    # kernel_candidates.json — usually the same shape, just truncated to
    # 15. Used when the on-disk file is missing (e.g. test fixtures or
    # sessions where the kernel-agent workspace got rotated away).
    inline = sk.get("hot_kernels_top15") if isinstance(sk, dict) else None
    if isinstance(inline, list):
        return inline
    return []


def _index_invocations_by_kernel(
    invs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fold per-attempt invocations into per-kernel summary for one lane.

    The returned dict maps kernel_id -> ``{attempts, best_speedup,
    decision, last_status}`` so the detected-kernel collector can stamp
    every detected kernel with both lanes' per-kernel verdicts.
    """
    out: dict[str, dict[str, Any]] = {}
    for inv in invs:
        kid = str(inv.get("kernel_id") or "")
        if not kid:
            continue
        ent = out.setdefault(kid, {
            "attempts": 0, "best_speedup": None,
            "decision": "", "last_status": "",
        })
        ent["attempts"] += 1
        spd = inv.get("micro_speedup")
        if isinstance(spd, (int, float)):
            cur = ent["best_speedup"]
            if cur is None or float(spd) > cur:
                ent["best_speedup"] = float(spd)
        # KEEP/PARTIAL outrank everything else; FAILED never overrides KEEP.
        dec = str(inv.get("decision") or "")
        if dec in ("KEEP", "PARTIAL") or not ent["decision"]:
            ent["decision"] = dec or ent["decision"]
        ent["last_status"] = str(inv.get("status") or ent["last_status"])
    return out


def _collect_detected_kernels(
    session_dir: Path,
    state: dict[str, Any],
    geak: list[dict[str, Any]],
    oob: list[dict[str, Any]],
    warnings: list[str],
    *,
    cap: int | None = None,
) -> list[dict[str, Any]]:
    """Build the canonical per-kernel lifecycle row used by the report.

    Each entry merges:
      * static profile fields (gpu_pct, duration_us, call_count,
        bandwidth/compute util, kernel_category, bottleneck, name,
        source_file, arithmetic_intensity, reusable_native_kernel) from
        ``kernel_candidates.json`` (preferred) or ``benchmark_report.kernel_summary``,
      * ``selected_for_optimization`` — whether the orchestrator routed
        this kernel into the optimization pipeline,
      * ``geak`` / ``oob`` — per-lane ``{attempts, best_speedup,
        decision}`` summaries reduced from the invocation lists,
      * ``adopted_by`` — which lane's patch ended up in the final stack
        (looked up via ``kernel_integrate_attempts`` KEEP entries),
      * ``final_decision`` — ``kept`` / ``reverted`` / ``rejected`` /
        ``not_optimized``.

    The merge is keyed by ``kernel_id``; rows from
    ``kernel_candidates.json`` take precedence on shape conflicts since
    that file is what the kernel agent actually consumed.
    """
    by_kid: dict[str, dict[str, Any]] = {}

    # 1) candidates.json (preferred — has call_count / duration_us)
    for k in _read_kernel_candidates(session_dir, state, warnings):
        if not isinstance(k, dict):
            continue
        kid = str(k.get("kernel_id") or k.get("name") or "")
        if not kid:
            continue
        by_kid[kid] = {
            "kernel_id":               kid,
            "name":                    str(k.get("name") or ""),
            "gpu_pct":                 _to_float(k.get("gpu_pct")),
            "duration_us":             _to_float(k.get("duration_us")),
            "call_count":              int(k.get("call_count") or 0) or None,
            "bandwidth_util_pct":      _to_float(k.get("bandwidth_utilization_pct")),
            "compute_util_pct":        _to_float(k.get("compute_utilization_pct")),
            "kernel_category":         str(k.get("kernel_category") or ""),
            "bottleneck":              str(k.get("bottleneck") or ""),
            "arithmetic_intensity":    _to_float(k.get("arithmetic_intensity")),
            "reusable_native_kernel":  bool(k.get("reusable_native_kernel")),
            "source_file":             k.get("source_file") or "",
            "recommended_actions":     list(k.get("recommended_actions") or []),
            "recommended_backends":    list(k.get("recommended_backends") or []),
            "optimization_notes":      str(k.get("optimization_notes") or ""),
        }

    # 2) benchmark_report.kernel_summary fallback — pulls in the long
    #    tail of trace kernels that didn't make the top-N candidates
    #    list. We dedupe by name against the candidates entries (so
    #    k001..k0NN absorb their fallback rows instead of producing
    #    twin entries), and any genuinely new fallback entry gets a
    #    short ``rNNN`` alias as ``kernel_id`` so the table column
    #    stays narrow.
    name_to_kid = {e["name"]: kid for kid, e in by_kid.items() if e.get("name")}
    residual_counter = 0
    for task_dir, report_path in _scan_profile_reports(session_dir):
        report = _load_json_safe(report_path, warnings)
        if not isinstance(report, dict):
            continue
        kernel_summary = report.get("kernel_summary") or []
        bottlenecks = report.get("top_bottlenecks") or []
        bottleneck_by_kid = {
            b.get("kernel_id"): b
            for b in (bottlenecks if isinstance(bottlenecks, list) else [])
            if isinstance(b, dict)
        }
        for k in kernel_summary if isinstance(kernel_summary, list) else []:
            if not isinstance(k, dict):
                continue
            name_str = str(k.get("name") or "")
            if not name_str:
                continue
            existing_kid = name_to_kid.get(name_str)
            if existing_kid is not None:
                # Merge missing fields into the candidates-side entry
                # rather than appending a duplicate row.
                entry = by_kid[existing_kid]
                if entry.get("gpu_pct") is None:
                    entry["gpu_pct"] = _to_float(k.get("gpu_pct"))
                if entry.get("duration_us") is None:
                    t_ms = _to_float(k.get("time_ms"))
                    entry["duration_us"] = (t_ms * 1000.0) if t_ms is not None else None
                if not entry.get("bottleneck"):
                    bn = bottleneck_by_kid.get(k.get("kernel_id")) or {}
                    entry["bottleneck"] = str(bn.get("bottleneck") or k.get("bottleneck") or "")
                if entry.get("arithmetic_intensity") is None:
                    entry["arithmetic_intensity"] = _to_float(k.get("arithmetic_intensity"))
                continue

            input_kid = str(k.get("kernel_id") or "")
            # Keep the input kernel_id when it's already a short alias
            # (orchestrator-assigned, e.g. ``k002``) instead of clobbering
            # it with a residual alias. Mangled C++ symbols (input_kid ==
            # name, len ≫ 16) get a generated alias to keep the table
            # narrow.
            is_short_alias = (
                input_kid
                and input_kid != name_str
                and len(input_kid) <= 8
                and input_kid not in by_kid
            )
            if is_short_alias:
                alias = input_kid
            else:
                residual_counter += 1
                alias = f"r{residual_counter:03d}"
            bn = bottleneck_by_kid.get(k.get("kernel_id")) or {}
            t_ms = _to_float(k.get("time_ms"))
            by_kid[alias] = {
                "kernel_id":              alias,
                "name":                   name_str,
                "gpu_pct":                _to_float(k.get("gpu_pct")),
                "duration_us":            (t_ms * 1000.0) if t_ms is not None else None,
                "call_count":             None,
                "bandwidth_util_pct":     None,
                "compute_util_pct":       None,
                "kernel_category":        "",
                "bottleneck":             str(bn.get("bottleneck") or k.get("bottleneck") or ""),
                "arithmetic_intensity":   _to_float(k.get("arithmetic_intensity")),
                "reusable_native_kernel": bool(k.get("reusable_native_kernel")),
                "source_file":            k.get("source_file") or "",
                "recommended_actions":    [],
                "recommended_backends":   [],
                "optimization_notes":     "",
                "detected_from_task":     task_dir.name,
                "benchmark_report_path":  _rel(report_path, session_dir) or str(report_path),
            }
            name_to_kid[name_str] = alias

    # 3) lifecycle stamps (selected / geak / oob / adopted_by / final_decision)
    selected_ids = {
        str(e.get("kernel_id") or "")
        for e in ((state.get("last_trace_analyze") or {}).get("hot_kernels_top15") or [])
        if isinstance(e, dict)
    }
    geak_idx = _index_invocations_by_kernel(geak)
    oob_idx = _index_invocations_by_kernel(oob)

    integ = state.get("kernel_integrate_attempts") or {}
    adopted_kids: set[str] = set()
    reverted_kids: set[str] = set()
    integ_gain_by_kid: dict[str, float | None] = {}
    if isinstance(integ, dict):
        for ent in integ.values():
            if not isinstance(ent, dict):
                continue
            kid = str(ent.get("kernel_id") or "")
            if not kid:
                continue
            integ_gain_by_kid[kid] = _to_float(ent.get("best_gain_pct"))
            dec = ent.get("last_decision")
            if dec == "KEEP":
                adopted_kids.add(kid)
            elif dec in ("REVERT", "REJECT"):
                reverted_kids.add(kid)

    rejected_kids = {
        str(k or "") for k in (state.get("rejected_kernel_ids") or [])
    } - adopted_kids

    for kid, entry in by_kid.items():
        entry["selected_for_optimization"] = kid in selected_ids
        entry["geak"] = geak_idx.get(kid)  # None if lane never touched this kid
        entry["oob"] = oob_idx.get(kid)
        # e2e (integrate) gain so the table shows WHY a micro-KEPT kernel was
        # reverted (regressed end-to-end) rather than just a bare verdict.
        if kid in integ_gain_by_kid:
            entry["integrate_gain_pct"] = integ_gain_by_kid[kid]
        if kid in adopted_kids:
            # Disambiguate which lane's patch was kept.
            g_kept = bool(entry["geak"] and entry["geak"]["decision"] in ("KEEP", "PARTIAL"))
            o_kept = bool(entry["oob"] and entry["oob"]["decision"] in ("KEEP", "PARTIAL"))
            if g_kept and not o_kept:
                entry["adopted_by"] = "geak"
            elif o_kept and not g_kept:
                entry["adopted_by"] = "oob"
            elif g_kept and o_kept:
                # Prefer the lane with higher micro-speedup.
                g_spd = (entry["geak"] or {}).get("best_speedup") or 0.0
                o_spd = (entry["oob"] or {}).get("best_speedup") or 0.0
                entry["adopted_by"] = "geak" if g_spd >= o_spd else "oob"
            else:
                # Integrate KEEP but neither lane recorded KEEP — likely a
                # manually staged patch; record as 'kernel_agent' rather
                # than guessing.
                entry["adopted_by"] = "kernel_agent"
            entry["final_decision"] = "kept"
        elif kid in reverted_kids:
            entry["adopted_by"] = None
            entry["final_decision"] = "reverted"
        elif kid in rejected_kids:
            entry["adopted_by"] = None
            entry["final_decision"] = "rejected"
        elif entry["geak"] or entry["oob"]:
            entry["adopted_by"] = None
            entry["final_decision"] = "attempted"
        else:
            entry["adopted_by"] = None
            entry["final_decision"] = "not_optimized"

    # Sort by GPU share descending so the table is actionable top-down.
    out = sorted(
        by_kid.values(),
        key=lambda e: (-(e.get("gpu_pct") or 0.0), e.get("kernel_id") or ""),
    )
    if cap is not None and len(out) > cap:
        return out[:cap]
    return out


def _collect_recommended_kernels(state: dict[str, Any]) -> list[dict[str, Any]]:
    sk = state.get("last_trace_analyze") or {}
    if not isinstance(sk, dict):
        return []
    out: list[dict[str, Any]] = []
    for entry in sk.get("hot_kernels_top15") or []:
        if not isinstance(entry, dict):
            continue
        out.append({
            "kernel_id":            str(entry.get("kernel_id") or ""),
            "name":                 str(entry.get("name") or ""),
            "gpu_pct":              _to_float(entry.get("gpu_pct")),
            "recommended_backends": list(entry.get("recommended_backends") or []),
            "recommended_actions":  list(entry.get("recommended_actions") or []),
            "bottleneck":           str(entry.get("bottleneck") or ""),
            "reusable_native_kernel": bool(entry.get("reusable_native_kernel")),
        })
    return out


def _collect_optimized_kernels(
    geak: list[dict[str, Any]],
    oob: list[dict[str, Any]],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fold per-attempt invocations into per-kernel summaries."""
    by_kid: dict[str, dict[str, Any]] = {}
    for invs in (geak, oob):
        for inv in invs:
            kid = inv.get("kernel_id") or ""
            if not kid:
                continue
            entry = by_kid.setdefault(kid, {
                "kernel_id": kid,
                "backend": inv.get("backend") or "",
                "total_attempts": 0,
                "successful_attempts": 0,
                "best_micro_speedup": None,
                "last_decision": "",
                "best_artifact_path": None,
                "attempts_summary": [],
            })
            entry["total_attempts"] += 1
            spd = inv.get("micro_speedup")
            cur_best = entry["best_micro_speedup"]
            if isinstance(spd, (int, float)):
                if cur_best is None or spd > cur_best:
                    entry["best_micro_speedup"] = float(spd)
                    entry["best_artifact_path"] = inv.get("best_artifact_path") or entry["best_artifact_path"]
            if inv.get("decision") in ("KEEP", "PARTIAL"):
                entry["successful_attempts"] += 1
            entry["last_decision"] = inv.get("decision") or entry["last_decision"]
            entry["attempts_summary"].append({
                "attempt_id":   inv.get("attempt_id"),
                "backend":      inv.get("backend"),
                "decision":     inv.get("decision"),
                "micro_speedup": spd,
                "ts":           inv.get("ts"),
            })
    # Cross-reference with state.kernel_opt_attempts (the orchestrator's own
    # record of decisions — covers cases where the on-disk verification was
    # rotated away but the orchestrator decision is still in state).
    ko_attempts = state.get("kernel_opt_attempts") or {}
    if isinstance(ko_attempts, dict):
        for kid, ent in ko_attempts.items():
            if not isinstance(ent, dict):
                continue
            entry = by_kid.setdefault(kid, {
                "kernel_id": kid,
                "backend": "",
                "total_attempts": 0,
                "successful_attempts": 0,
                "best_micro_speedup": None,
                "last_decision": "",
                "best_artifact_path": None,
                "attempts_summary": [],
            })
            entry["total_attempts"] = max(entry["total_attempts"], int(ent.get("attempts", 0)))
            entry["last_decision"] = entry["last_decision"] or str(ent.get("last_decision") or "")
    return sorted(by_kid.values(), key=lambda e: e.get("kernel_id") or "")


def _collect_adopted_kernels(state: dict[str, Any]) -> list[dict[str, Any]]:
    """KEEP-promoted kernel entries (from state.kernel_integrate_attempts + optimization_stack)."""
    out: list[dict[str, Any]] = []
    integ = state.get("kernel_integrate_attempts") or {}
    if isinstance(integ, dict):
        for key, ent in integ.items():
            if not isinstance(ent, dict):
                continue
            if ent.get("last_decision") != "KEEP":
                continue
            out.append({
                "kernel_id":         str(ent.get("kernel_id") or ""),
                "patch_path":        str(ent.get("patch_path") or ""),
                "target_file":       str(ent.get("target_file") or ""),
                "extra_server_args": str(ent.get("extra_server_args") or ""),
                "e2e_gain_pct":      _to_float(ent.get("best_gain_pct")),
                "validated":         True,
                "last_status":       str(ent.get("last_status") or ""),
                "adopted_at":        str(ent.get("updated_at") or ""),
                "attempt_count":     int(ent.get("attempt_count") or 0),
            })
    return out


def _collect_rejected_kernels(state: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rejected = state.get("rejected_kernel_patches") or []
    if isinstance(rejected, list):
        for r in rejected:
            if not isinstance(r, dict):
                continue
            out.append({
                "kernel_id":      str(r.get("kernel_id") or ""),
                "reason":         str(r.get("reason") or ""),
                "patch_path":     r.get("patch_path"),
                "target_file":    r.get("target_file"),
                "attempt_count":  int(r.get("attempt_count") or 0),
                "best_gain_pct":  _to_float(r.get("best_gain_pct")),
                "ts":             str(r.get("ts") or ""),
            })
    # also surface rejected_kernel_ids that didn't make it into rejected_kernel_patches
    seen_ids = {entry["kernel_id"] for entry in out if entry.get("kernel_id")}
    for kid in state.get("rejected_kernel_ids") or []:
        kid_s = str(kid or "")
        if not kid_s or kid_s in seen_ids:
            continue
        out.append({
            "kernel_id":  kid_s,
            "reason":     "retired",
            "patch_path": None,
            "target_file": None,
            "attempt_count": 0,
            "best_gain_pct": None,
            "ts": "",
        })
    return out


def collect_kernel_lifecycle(
    session_dir: Path,
    state: dict[str, Any],
    geak: list[dict[str, Any]],
    oob: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "detected":    _collect_detected_kernels(session_dir, state, geak, oob, warnings),
        "recommended": _collect_recommended_kernels(state),
        "optimized":   _collect_optimized_kernels(geak, oob, state),
        "adopted":     _collect_adopted_kernels(state),
        "rejected":    _collect_rejected_kernels(state),
    }


# ---------------------------------------------------------------------------
# §10 Explore search ledger
# ---------------------------------------------------------------------------
def _shape_ledger(
    ledger: dict[str, Any] | None,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    if not isinstance(ledger, dict):
        return {
            "schema_version": 0, "tested_count": 0,
            "accepted": [], "rejected": [], "top_by_gain": [],
        }

    def _shape_entry(e: Any) -> dict[str, Any]:
        if not isinstance(e, dict):
            return {}
        return {
            "name":              str(e.get("name") or ""),
            "fingerprint":       str(e.get("fingerprint") or ""),
            "extra_server_args": str(e.get("extra_server_args") or ""),
            "extra_envs":        dict(e.get("extra_envs") or {}),
            "output_throughput": _to_float(e.get("output_throughput") or e.get("tput")),
            "gain_pct":          _to_float(e.get("gain_pct")),
            "ts":                str(e.get("ts") or ""),
        }

    accepted = [_shape_entry(e) for e in ledger.get("accepted") or []]
    rejected = [_shape_entry(e) for e in ledger.get("rejected") or []]
    tested = list((ledger.get("tested") or {}).values()) if isinstance(ledger.get("tested"), dict) else []
    tested_shaped = [_shape_entry(e) for e in tested]
    top_by_gain = sorted(
        (e for e in tested_shaped if e.get("gain_pct") is not None),
        key=lambda e: e.get("gain_pct") or 0.0,
        reverse=True,
    )[:top_n]
    return {
        "schema_version":  int(ledger.get("schema_version") or 0),
        "tested_count":    len(tested),
        "accepted":        accepted,
        "rejected":        rejected,
        "top_by_gain":     top_by_gain,
    }


def _patch_winners_history(
    rows: list[Any], baseline_tput: float | None,
) -> list[dict[str, Any]]:
    """Fix two recurring data-quality issues in ``backend_winners_history``:

    * ``base_tput`` is occasionally written as ``0.0`` (Coordinator
      didn't stamp the round's baseline). When that happens we fall
      back to the session's ``baseline_tput``.
    * Per-winner ``gain_pct`` is frequently null. We compute it from
      ``(winner.tput - base_tput) / base_tput * 100`` when both are
      known so the dashboard table can show real gains.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        row = dict(r)
        try:
            bt = float(row.get("base_tput") or 0.0)
        except (TypeError, ValueError):
            bt = 0.0
        if bt <= 0 and baseline_tput and baseline_tput > 0:
            row["base_tput"] = float(baseline_tput)
            row["base_tput_source"] = "session_baseline"
            bt = float(baseline_tput)
        if bt > 0:
            winners = list(row.get("winners") or [])
            new_winners: list[dict[str, Any]] = []
            for w in winners:
                if not isinstance(w, dict):
                    continue
                w2 = dict(w)
                if w2.get("gain_pct") in (None, "") and w2.get("tput") is not None:
                    try:
                        w2["gain_pct"] = (float(w2["tput"]) - bt) / bt * 100.0
                    except (TypeError, ValueError, ZeroDivisionError):
                        pass
                new_winners.append(w2)
            row["winners"] = new_winners
            best = row.get("best")
            if isinstance(best, dict) and best.get("gain_pct") in (None, "") and best.get("tput") is not None:
                try:
                    best = dict(best)
                    best["gain_pct"] = (float(best["tput"]) - bt) / bt * 100.0
                    row["best"] = best
                except (TypeError, ValueError, ZeroDivisionError):
                    pass
        out.append(row)
    return out


def collect_explore_search(
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    # Post-merge sessions carry the unified ``explore_search`` ledger;
    # archived (pre-merge) sessions carry the legacy ``params_search`` /
    # ``backends_search`` ledgers. Emit all three so breakdown reprocesses
    # both vintages — each session only populates its own ledgers, the
    # others ``_shape_ledger`` into empty shells.
    explore_ledger = _shape_ledger(state.get("explore_search"))
    explore_ledger["winner_history"] = list(state.get("params_winner_history") or [])
    explore_ledger["no_promote_streak"] = int(
        state.get("params_no_promote_streak") or 0
    )

    params_ledger = _shape_ledger(state.get("params_search"))
    params_ledger["winner_history"] = list(state.get("params_winner_history") or [])
    params_ledger["no_promote_streak"] = int(
        state.get("params_no_promote_streak") or 0
    )

    backends_ledger = _shape_ledger(state.get("backends_search"))

    baseline_tput = _to_float(state.get("baseline_tput"))
    return {
        "explore":                 explore_ledger,
        "params":                  params_ledger,
        "backends":                backends_ledger,
        "synergy_attempted":       list(state.get("synergy_attempted") or []),
        "discovered_flags":        dict(state.get("discovered_flags") or {}),
        "backend_winners_history": _patch_winners_history(
            state.get("backend_winners_history") or [], baseline_tput,
        ),
    }


# Backwards-compatible function name for older in-repo callers/tests. The
# returned shape is the merged explore ledger plus archived aliases.
collect_param_search = collect_explore_search


# ---------------------------------------------------------------------------
# §11 Sweep
# ---------------------------------------------------------------------------
_VARIANT_NAME_RE = re.compile(r"variant_(\d+)_conc(\d+)_isl(\d+)_osl(\d+)", re.IGNORECASE)


def _scan_sweep_variants(session_dir: Path) -> list[Path]:
    """Return list of variant directories under runs/sweep/<task>/variant_*/."""
    root = session_dir / "runs" / "sweep"
    if not root.exists():
        return []
    out: list[Path] = []
    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        for v in sorted(task_dir.glob("variant_*")):
            if v.is_dir():
                out.append(v)
    return out


def _shape_sweep_point(
    variant_dir: Path,
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    name = variant_dir.name
    m = _VARIANT_NAME_RE.search(name)
    conc: int | None = None
    isl_: int | None = None
    osl_: int | None = None
    if m:
        try:
            conc = int(m.group(2))
            isl_ = int(m.group(3))
            osl_ = int(m.group(4))
        except ValueError:
            pass
    report = _find_benchmark_report(variant_dir)
    report_data = _load_json_safe(report, warnings) if report else None
    status = "ok"
    out_tput = ttft = tpot = e2el = None
    if isinstance(report_data, dict):
        out_tput, ttft, tpot, e2el = _benchmark_report_metrics(report_data)
        if report_data.get("success") is False:
            status = "failed"
    elif report is None:
        status = "skipped"
    return {
        "variant_name":            name,
        "conc":                    conc,
        "isl":                     isl_,
        "osl":                     osl_,
        "output_throughput_tok_s": out_tput,
        "ttft_mean_ms":            ttft,
        "tpot_mean_ms":            tpot,
        "e2el_mean_ms":            e2el,
        "status":                  status,
        "benchmark_report_path":   _rel(report, session_dir) if report else None,
    }


def collect_sweep(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    ls = state.get("last_sweep") or {}
    if not isinstance(ls, dict):
        ls = {}

    variants_on_disk = [
        _shape_sweep_point(v, session_dir, warnings)
        for v in _scan_sweep_variants(session_dir)
    ]
    return {
        "grid_size":          _to_int(ls.get("grid_size")) or len(variants_on_disk),
        "best_overall":       dict(ls.get("best_overall") or {}),
        "best_for_each_conc": list(ls.get("best_for_each_conc") or []),
        "pareto_front":       list(ls.get("pareto_front") or []),
        "all_variants":       variants_on_disk,
        "config_path":        ls.get("config_path"),
    }


# ---------------------------------------------------------------------------
# §12 Critic / Robustness
# ---------------------------------------------------------------------------
def collect_critic_robustness(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    critic_iters: list[dict[str, Any]] = []
    critic_root = session_dir / "critic-workdir"
    if critic_root.exists():
        for iter_dir in sorted(critic_root.iterdir(), key=lambda p: p.name):
            if not iter_dir.is_dir():
                continue
            try:
                iter_n = int(iter_dir.name)
            except ValueError:
                iter_n = -1
            review = _load_json_safe(iter_dir / "review.json", warnings) or {}
            emit = _load_json_safe(iter_dir / "emit.json", warnings) or {}
            critic_iters.append({
                "iter":               iter_n,
                "ts":                 str(emit.get("ts") or review.get("ts") or ""),
                "topic":              str(emit.get("topic") or review.get("topic") or ""),
                "verdict":            str(review.get("verdict") or emit.get("verdict") or ""),
                "summary":            str(
                    review.get("summary") or emit.get("summary") or ""
                )[:500],
                "request_path":       _rel(iter_dir / "request.json", session_dir),
                "judge_bundle_path":  _rel(iter_dir / "judge_bundle.json", session_dir),
                "emit_path":          _rel(iter_dir / "emit.json", session_dir),
                "review_path":        _rel(iter_dir / "review.json", session_dir),
            })

    robustness_signals: list[dict[str, Any]] = []
    rob_root = session_dir / "robustness-workdir"
    if rob_root.exists():
        for iter_dir in sorted(rob_root.iterdir(), key=lambda p: p.name):
            if not iter_dir.is_dir():
                continue
            signal_data = _load_json_safe(iter_dir / "signal.json", warnings) or {}
            action_data = _load_json_safe(iter_dir / "action.json", warnings) or {}
            robustness_signals.append({
                "ts":      str(signal_data.get("ts") or action_data.get("ts") or ""),
                "signal":  str(signal_data.get("signal") or signal_data.get("kind") or ""),
                "action":  str(action_data.get("action") or action_data.get("kind") or ""),
                "workdir": _rel(iter_dir, session_dir) or str(iter_dir),
            })

    # kb_writes_summary mirrors the critic-agent's
    # ``commit-review`` output count, grouped by verdict. Source is
    # the same ``critic-workdir/<iter>/review.json`` we already
    # parsed above so we don't re-read the disk.
    kb_writes_summary = _critic_kb_writes_summary(critic_iters)

    return {
        "critic_iterations":  critic_iters,
        "robustness_signals": robustness_signals,
        "kb_writes_summary":  kb_writes_summary,
    }


def _critic_kb_writes_summary(
    critic_iters: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the ``critic_robustness.kb_writes_summary`` sub-block
.

    Counts each iteration's verdict; downstream dashboards group on
    ``by_verdict`` to render the KEEP / REVERT / NEEDS_INFO mix.
    """
    by_verdict: dict[str, int] = {}
    total = 0
    for entry in critic_iters:
        verdict = str((entry or {}).get("verdict") or "").strip().upper()
        if not verdict:
            continue
        total += 1
        by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
    return {
        "total":      total,
        "by_verdict": by_verdict,
    }


# ---------------------------------------------------------------------------
# §13 Telemetry
# ---------------------------------------------------------------------------
def _scan_all_benchmark_reports(session_dir: Path) -> Iterable[Path]:
    runs = session_dir / "runs"
    if not runs.exists():
        return ()
    return sorted(runs.rglob("benchmark_*/benchmark_report.json"))


def _scan_torch_traces(session_dir: Path) -> list[Path]:
    runs = session_dir / "runs"
    if not runs.exists():
        return []
    return sorted(p for p in runs.rglob("torch_trace*") if p.is_dir())


def _scan_system_profiles(session_dir: Path) -> list[Path]:
    runs = session_dir / "runs"
    if not runs.exists():
        return []
    return sorted(p for p in runs.rglob("system_profile*") if p.is_dir())


def _scan_server_logs(session_dir: Path) -> list[Path]:
    runs = session_dir / "runs"
    if not runs.exists():
        return []
    return sorted(runs.rglob("server*.log"))


def _aggregate_gpu_monitor(
    reports: list[Path],
    warnings: list[str],
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for r in reports:
        d = _load_json_safe(r, warnings)
        if not isinstance(d, dict):
            continue
        gm = d.get("gpu_monitor")
        if isinstance(gm, list):
            for s in gm:
                if isinstance(s, dict):
                    samples.append(s)
        elif isinstance(gm, dict):
            samples.append(gm)
    if not samples:
        return {}

    def _avg(key: str) -> float:
        vals = [_to_float(s.get(key)) for s in samples]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def _max(key: str) -> float:
        vals = [_to_float(s.get(key)) for s in samples]
        vals = [v for v in vals if v is not None]
        return round(max(vals), 2) if vals else 0.0

    return {
        "samples":       len(samples),
        "avg_power_w":   _avg("power_w") or _avg("power"),
        "max_power_w":   _max("power_w") or _max("power"),
        "avg_temp_c":    _avg("temperature_c") or _avg("temperature"),
        "max_temp_c":    _max("temperature_c") or _max("temperature"),
        "avg_clock_mhz": _avg("clock_mhz") or _avg("sclk_mhz"),
    }


def _collect_lane_timeline(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """v0.8 M6 (KB_design §3.12 §4.5) — per-lane capacity / occupancy
    summary derived from ``storage/coordinator.db``.

    Returns one row per known lane with:

    * ``lane``                — lane name
    * ``capacity``            — current capacity (from ``lane_capacity``,
                                falling back to defaults)
    * ``live_holders``        — distinct holders currently below their
                                expiration ts at collection time
    * ``lease_expired_count`` — total ``lease_expired`` events emitted
                                this session (from the events table)

    The per-second / per-tick ``holders_timeline`` slice the design also
    describes is deferred — it requires a dedicated sampler, which
    Robustness will land in a follow-up. The aggregate numbers above
    are enough for the breakdown's ``benchmark_lane.peak ≤ 1``
    invariant check.
    """
    db_path = session_dir / "storage" / "coordinator.db"
    if not db_path.exists():
        return []
    import sqlite3 as _sqlite3
    try:
        conn = _sqlite3.connect(str(db_path), timeout=2.0)
        conn.row_factory = _sqlite3.Row
    except _sqlite3.Error as exc:
        warnings.append(f"lane_timeline: open {db_path} failed: {exc!r}")
        return []
    try:
        try:
            cur = conn.execute(
                "SELECT lane, capacity FROM lane_capacity ORDER BY lane",
            )
            capacities = {r["lane"]: int(r["capacity"]) for r in cur.fetchall()}
        except _sqlite3.OperationalError:
            # Older DB without lane_capacity — fall back to defaults so
            # resume on an old session still produces a stable shape.
            from ..storage.schema import DEFAULT_LANE_CAPACITIES as _DEFAULT
            capacities = dict(_DEFAULT)
        try:
            cur = conn.execute(
                "SELECT lane, COUNT(*) AS n FROM leases "
                "WHERE expires_at > datetime('now') GROUP BY lane",
            )
            holders = {r["lane"]: int(r["n"]) for r in cur.fetchall()}
        except _sqlite3.OperationalError as exc:
            warnings.append(f"lane_timeline: leases query failed: {exc!r}")
            holders = {}
        try:
            cur = conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE topic = 'lease_expired'",
            )
            row = cur.fetchone()
            expired_total = int(row["n"]) if row else 0
        except _sqlite3.OperationalError:
            expired_total = 0
        # Per-lane expired count breakdown (lease_expired events carry
        # the lane in their JSON payload).
        per_lane_expired: dict[str, int] = {}
        try:
            cur = conn.execute(
                "SELECT payload FROM events WHERE topic = 'lease_expired'",
            )
            for r in cur.fetchall():
                try:
                    p = json.loads(r["payload"] or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                lane = str(p.get("lane") or "")
                if lane:
                    per_lane_expired[lane] = per_lane_expired.get(lane, 0) + 1
        except _sqlite3.OperationalError:
            pass
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    rows: list[dict[str, Any]] = []
    for lane in sorted(set(capacities) | set(holders)):
        rows.append({
            "lane":                lane,
            "capacity":            int(capacities.get(lane, 1)),
            "live_holders":        int(holders.get(lane, 0)),
            "lease_expired_count": int(per_lane_expired.get(lane, 0)),
        })
    # Append totals row for breakdown consumers that aggregate across.
    if rows:
        rows.append({
            "lane":                "__total__",
            "capacity":            sum(r["capacity"] for r in rows),
            "live_holders":        sum(r["live_holders"] for r in rows),
            "lease_expired_count": int(expired_total),
        })
    return rows


def collect_telemetry(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    baseline_report: Path | None = None
    last_b = state.get("last_baseline") or {}
    if isinstance(last_b, dict) and last_b.get("workspace"):
        workspace = _resolve_under_session(session_dir, last_b.get("workspace"))
        if workspace is not None:
            baseline_report = _find_benchmark_report(workspace)

    profile_reports = [
        report for _task, report in _scan_profile_reports(session_dir)
    ]
    all_reports = list(_scan_all_benchmark_reports(session_dir))

    return {
        "baseline_report_path": _rel(baseline_report, session_dir) if baseline_report else None,
        "profile_report_paths": [_rel(p, session_dir) or str(p) for p in profile_reports],
        "torch_trace_paths":    [_rel(p, session_dir) or str(p) for p in _scan_torch_traces(session_dir)],
        "system_profile_paths": [_rel(p, session_dir) or str(p) for p in _scan_system_profiles(session_dir)],
        "server_log_paths":     [_rel(p, session_dir) or str(p) for p in _scan_server_logs(session_dir)],
        "gpu_monitor_aggregate": _aggregate_gpu_monitor(all_reports, warnings),
        # per-lane occupancy
        # / capacity summary derived from the leases DB. Sits in the
        # telemetry section so cross-cluster dashboards can chart
        # lane usage alongside GPU power / temperature.
        "lane_timeline": _collect_lane_timeline(session_dir, warnings),
    }


# ---------------------------------------------------------------------------
# §14 Attribution
# ---------------------------------------------------------------------------
# Catalogue of the 7 SpecialistDomain.key strings
# (specialist_domains.SPECIALIST_DOMAIN_KEYS). Kept inline rather than
# imported so the breakdown package stays free of orchestrator deps —
# the breakdown module is meant to be readable by tools running
# offline against a session_dir tarball.
_SPECIALIST_DOMAIN_KEYS: tuple[str, ...] = (
    "serving_specialist",
    "kernel_switch_specialist",
    "comm_specialist",
    "compiler_specialist",
    "system_specialist",
    "pr_intel_specialist",
    "session_steward_specialist",
    "research_scout_specialist",
)


def _normalize_specialist_key(provenance: str) -> str:
    """Map a raw ``provenance`` string to a stable specialist key.

    The orchestrator stamps four shapes of provenance on explore
    winners:

    * ``"specialist:<domain>"`` — the canonical case for v0.8+ runs.
      Strip the prefix so the consumer iterates one bare key per
      catalogue entry instead of having to special-case the prefix.
    * ``"default_grid"`` — cold-start grid run when no specialist has
      produced a proposal_set yet.
    * ``"llm_direct"`` — orchestration LLM authored the variant
      directly.
    * ``"legacy:<action>"`` — resume of a pre-v0.8 session whose
      promoted actions predate the specialist split. Folded under
      ``"legacy_<action>"`` so it doesn't pollute the specialist
      catalogue while still being inspectable.

    Empty / unknown values become ``"unknown"`` so the by_domain dict
    is never keyed by ``""``.
    """
    s = (provenance or "").strip()
    if not s:
        return "unknown"
    if s.startswith("specialist:"):
        # Trust the orchestrator's domain string verbatim — adding a new
        # SpecialistDomain in specialist_domains.py shouldn't require a
        # parallel update here.
        return s[len("specialist:"):] or "unknown"
    if s.startswith("legacy:"):
        return f"legacy_{s[len('legacy:'):]}"
    return s


def _action_family(action: str) -> str:
    """Map an action label to a family for source_breakdown bucketing."""
    s = (action or "").lower()
    if s.startswith("kernel_opt") or s == "integrate":
        return "kernel"
    # Legacy stack-entry action labels from archived sessions.
    if s == "backends":
        return "backends"
    if s == "params":
        return "params"
    if s == "validate_stack":
        return "validate"
    if s == "sweep":
        return "sweep"
    # merged explore action. Bucketed into
    # its own ``explore`` family so the attribution table can show a
    # single row that subsumes the legacy backends + params buckets.
    if s == "explore":
        return "explore"
    # FRAMEWORK_PR (PRELUDE → FRAMEWORK_PR → EXPLORE).
    # Surfaces upstream PR-driven KEEPs as their own headline row in
    # ``source_breakdown`` so the dashboard's per-source totals reconcile
    # against ``validated_total_pct``. Without this, framework_pr KEEPs
    # fell through to ``"other"`` and silently disappeared from the
    # leaderboard's per-source columns (manifest of: Params=0% +
    # Backends=1.74% + Kernel=0% summing to ≪ Validated=22.85%).
    if s == "framework_pr":
        return "framework_pr"
    # GEMM_TUNING (KERNEL phase entry lever).
    # FP8 GEMM tuning runs as the deterministic operator-level
    # KERNEL-entry step before source-level kernel_opt; ``coordinator``
    # promotes a successful tune (best_speedup > 1.0) into
    # ``optimization_stack`` with ``action="gemm_tuning"``. We bucket
    # these KEEPs separately from generic ``kernel`` so the dashboard
    # can attribute speedup to the deterministic tuner vs source-level
    # GEAK/OOB rewrites — same pattern as framework_pr.
    if s == "gemm_tuning":
        return "gemm_tuning"
    return "other"


def _promote_legacy_gain_entries(
    state_entries: list[Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Lift a pre-v0.7 ``list[float | None]`` gain ledger into the V1 schema.

    State written by older Coordinator versions stored per-entry
    ``cum_gain_after`` floats only. Cross-reference the parallel
    ``state.optimization_stack`` to recover action / variant_name / ts /
    extra_server_args, and compute ``delta_pct`` as the diff against the
    prior entry's ``cum_gain_after``. Entries the legacy ledger left as
    ``None`` (seeded / resumed sessions) become objects with
    ``cum_gain_after = delta_pct = None`` so index-alignment with
    ``optimization_stack`` is preserved.
    """
    stack = state.get("optimization_stack") or []
    out: list[dict[str, Any]] = []
    prev_cum = 0.0
    for i, val in enumerate(state_entries):
        cum_after: float | None
        if isinstance(val, (int, float)):
            cum_after = float(val)
        else:
            cum_after = None
        delta = (cum_after - prev_cum) if cum_after is not None else None
        se = stack[i] if i < len(stack) and isinstance(stack[i], dict) else {}
        out.append({
            "ts": str(se.get("ts") or ""),
            "action": str(se.get("action") or ""),
            "variant_name": se.get("variant_name") or se.get("kernel_id"),
            "stack_len_before": i,
            "stack_len_after": i + 1,
            "cum_gain_before": prev_cum,
            "cum_gain_after": cum_after,
            "delta_pct": delta,
            "extra_server_args": str(
                se.get("extra_server_args")
                or se.get("candidate_extra_server_args")
                or ""
            ),
        })
        if cum_after is not None:
            prev_cum = cum_after
    return out


# ---------------------------------------------------------------------------
# §13b Roofline (single-path + watermark refresh model)
# ---------------------------------------------------------------------------
def collect_roofline(
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Surface the per-session roofline comparison for breakdown
    consumers.

    Reads :attr:`SharedState.roofline_snapshots` (append-only history
    written by :meth:`SharedState.record_trace_analyze`) and shapes a
    single entry suitable for the ``Roofline`` renderer in
    :mod:`breakdown.reporters._renderers.roofline`.

    The shape matches what the renderer expects:

    .. code-block:: python

        [{
            "source_path": str,            # state.json (authoritative source)
            "mode":        "single_snapshot" | "before_after",
            "baseline":    {snapshot_id, ts, compute_pct, idle_pct,
                            comm_pct, top_bottleneck, top_kernel: {...},
                            analysis_md_path, trace_input},
            "latest":      {... same keys ...},
            "delta":       {compute_pct, idle_pct, comm_pct,
                            top_kernel_efficiency_pct},   # only when before_after
        }]

    Returns an empty list when ``state.roofline_snapshots`` is absent /
    empty (no roofline action ever completed). Best-effort: parsing
    errors are recorded in ``warnings`` and the section degrades to an
    empty list rather than blocking the whole breakdown export.
    """
    snapshots = state.get("roofline_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        return []
    try:
        # Lazy import to keep the breakdown package free of orchestrator
        # imports at module load time (avoids the circular import path
        # ``orchestrator → breakdown → orchestrator``).
        from ..orchestrator.roofline_snapshot import (
            build_roofline_comparison_from_history,
        )
        cmp = build_roofline_comparison_from_history(snapshots)
    except Exception as exc:  # noqa: BLE001 — defensive
        warnings.append(
            f"collect_roofline: failed to build comparison from "
            f"roofline_snapshots ({len(snapshots)} entries): "
            f"{type(exc).__name__}: {exc}"
        )
        return []
    if not cmp:
        return []
    entry: dict[str, Any] = {
        "source_path": "state.json#roofline_snapshots",
        "mode": cmp.get("mode") or "single_snapshot",
        "baseline": cmp.get("baseline") or {},
        "latest": cmp.get("latest") or {},
    }
    delta = cmp.get("delta")
    if isinstance(delta, dict) and delta:
        entry["delta"] = delta
    return [entry]


def collect_attribution(
    state: dict[str, Any],
    geak_invocations: list[dict[str, Any]],
    oob_invocations: list[dict[str, Any]],
    adopted_kernels: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    # Prefer authoritative per-stack ledger if Coordinator wrote it
    # (new state field `gain_per_stack_entry`). Otherwise reconstruct a
    # best-effort approximation from optimization_stack alone.
    state_entries = state.get("gain_per_stack_entry")
    state_provided = isinstance(state_entries, list) and len(state_entries) > 0
    promoted_from_legacy = False
    if state_provided and any(not isinstance(e, dict) for e in state_entries):
        # Older state: bare numeric ledger. Promote into V1 schema
        # so source_breakdown bucketing + dashboards see rich entries.
        entries = _promote_legacy_gain_entries(state_entries, state)
        promoted_from_legacy = True
    elif state_provided:
        entries = list(state_entries)
    else:
        entries = _reconstruct_gain_ledger(state, warnings)

    # Classify the attribution lineage so consumers (dashboard, LLM
    # narrator) can render an honest provenance label rather than
    # silently presenting a reconstructed split as validated.
    stack = state.get("optimization_stack") or []
    stack_len = len(stack) if isinstance(stack, list) else 0
    method: str
    if state_provided:
        if promoted_from_legacy:
            # Numbers lifted post-hoc to dicts — fields are honest but
            # the ledger itself isn't a per-event capture by Coordinator.
            method = "reconstructed"
        else:
            all_deltas_set = all(
                isinstance(e, dict) and e.get("delta_pct") is not None
                for e in state_entries
            )
            method = "validated" if all_deltas_set else "reconstructed"
    elif stack_len == 1:
        # Single-entry stack: ``final.action_path`` unambiguously
        # identifies the one source of gain.
        method = "single_source"
    elif stack_len > 1:
        method = "reconstructed"
    else:
        method = "missing"

    # Bucket entries by family for source_breakdown. Honor validated
    # total as the ground-truth denominator.
    validated_total = _to_float(state.get("cumulative_gain_validated")) or 0.0
    family_totals: dict[str, float] = {
        "kernel": 0.0, "sweep": 0.0, "other": 0.0,
        # Legacy buckets — populated when reprocessing archived
        # (pre-merge) sessions whose optimization_stack entries carry
        # the ``backends`` / ``params`` / ``validate_stack`` action.
        "backends": 0.0, "params": 0.0, "validate": 0.0,
        # explore family — the unified EXPLORE action that subsumes
        # the former backends + params buckets on current sessions.
        "explore": 0.0,
        # FRAMEWORK_PR family. Stays separate from the
        # legacy ``other`` bucket so the dashboard can surface a
        # dedicated ``framework_pr_pct_of_total`` row that reconciles
        # against ``validated_total_pct``.
        "framework_pr": 0.0,
        # GEMM_TUNING family. The FP8 A8W8 block-scale GEMM tuner runs
        # at KERNEL entry and contributes its own row so the dashboard
        # can show "deterministic tuner vs source-level kernel rewrite"
        # separately. Separate from ``kernel`` family so a tuner KEEP
        # doesn't get blended with GEAK/OOB attribution.
        "gemm_tuning": 0.0,
    }
    for e in entries:
        if not isinstance(e, dict):
            continue
        delta = _to_float(e.get("delta_pct"))
        if delta is None:
            continue
        fam = _action_family(str(e.get("action") or ""))
        family_totals[fam] = family_totals.get(fam, 0.0) + max(delta, 0.0)

    # Split "kernel" between GEAK / OOB based on adopted KEEP entries' backend
    geak_kept_kids = {inv.get("kernel_id") for inv in geak_invocations if inv.get("decision") == "KEEP"}
    oob_kept_kids  = {inv.get("kernel_id") for inv in oob_invocations  if inv.get("decision") == "KEEP"}
    kernel_total = family_totals.get("kernel", 0.0)
    geak_total = 0.0
    oob_total = 0.0
    for k in adopted_kernels:
        kid = k.get("kernel_id")
        gain = _to_float(k.get("e2e_gain_pct")) or 0.0
        if kid in geak_kept_kids:
            geak_total += gain
        elif kid in oob_kept_kids:
            oob_total += gain
    if geak_total + oob_total == 0.0 and kernel_total > 0.0:
        # All-OOB by default when no KEEP'd adopt entry is on disk
        oob_total = kernel_total

    notes: list[str] = []
    if not state_provided:
        notes.append(
            "gain_per_stack_entry not written by Coordinator; "
            "attribution reconstructed best-effort from optimization_stack."
        )
    elif promoted_from_legacy:
        notes.append(
            "gain_per_stack_entry was a pre-v0.7 numeric ledger; "
            "promoted to V1 StackGainEntry shape using parallel data from "
            "optimization_stack (delta_pct computed as diff vs prior entry's "
            "cum_gain_after)."
        )

    # per-phase gain
    # breakdown. Cross-references the optimization_stack with the
    # phase_history timestamps so each KEEP'd entry is bucketed into
    # the phase that was active at its acceptance time. EXPLORE
    # entries further break down by specialist domain (when
    # winners_history carries a ``provenance`` field).
    phase_breakdown = _collect_phase_breakdown(state, entries, warnings)

    return {
        "gain_per_stack_entry": entries,
        "method":               method,
        "source_breakdown": {
            "geak_pct_of_total":     round(geak_total, 2),
            "oob_pct_of_total":      round(oob_total, 2),
            # primary row.
            "explore_pct_of_total":  round(family_totals.get("explore", 0.0), 2),
            # FRAMEWORK_PR phase row. Tracks gain
            # attributable to upstream PR adoption (between PRELUDE
            # and EXPLORE). Always emitted (0.0 when the phase is
            # disabled or contributed nothing) so the dashboard can
            # iterate the catalogue without presence checks.
            "framework_pr_pct_of_total": round(family_totals.get("framework_pr", 0.0), 2),
            # GEMM_TUNING row (KERNEL-entry deterministic FP8 GEMM
            # tuner). Always emitted (0.0 when the workload is not
            # FP8 / the tuner skipped / no KEEP); the dashboard can
            # iterate without presence checks.
            "gemm_tuning_pct_of_total": round(family_totals.get("gemm_tuning", 0.0), 2),
            # Legacy bucket rows — preserved so archived-session reports
            # reconcile (0.0 on current sessions, where activity is in
            # ``explore_pct_of_total``).
            "backends_pct_of_total": round(family_totals.get("backends", 0.0), 2),
            "params_pct_of_total":   round(family_totals.get("params", 0.0), 2),
            "sweep_pct_of_total":    round(family_totals.get("sweep", 0.0), 2),
            "validated_total_pct":   round(validated_total, 2),
        },
        "phase_breakdown": phase_breakdown,
        "notes": notes,
    }


def _collect_phase_breakdown(
    state: dict[str, Any],
    entries: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    """KB_design §3.12 §4.6 + §3.13 M7 §6 — per-phase gain attribution.

    Walks ``optimization_stack`` / ``gain_per_stack_entry`` and assigns
    each KEEP entry to the phase that owned its acceptance timestamp.
    Returns a shape compatible with the §3.13 M7 §6 example::

        {
            "prelude": 0.0,
            "explore": {
                "total_gain_pct": 18.4,
                "by_domain": {
                    "serving_specialist": 9.7,
                    "default_grid":         2.5,
                    ...
                },
            },
            "kernel": {
                "total_gain_pct": 7.1,
                "by_kernel_id": {
                    "fmoe_fp8_blockscale_g1u1": 4.3,
                    ...
                },
            },
            "sweep": 0.0,
            "close":  0.0,
        }

    When phase_history is missing (legacy resume), every entry lands
    under ``unattributed`` so the dashboard can call attention to
    the gap.
    """
    # Build a phase timeline → bucket lookup once. Each row of
    # ``phase_history`` has ``to_phase`` + ``ts_unix``; entries are
    # ordered. For a given entry timestamp we pick the latest row
    # whose ``ts_unix`` is ≤ entry ts.
    history = state.get("phase_history") or []
    if not isinstance(history, list):
        history = []
    timeline: list[tuple[float, str]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        try:
            ts = float(row.get("ts_unix") or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        phase = str(row.get("to_phase") or "").strip().upper()
        if phase:
            timeline.append((ts, phase))
    timeline.sort(key=lambda r: r[0])

    def _phase_for(ts_unix: float) -> str:
        if not timeline:
            return ""
        current = ""
        for ts, ph in timeline:
            if ts <= ts_unix:
                current = ph
            else:
                break
        return current

    # winners_history (explore) gives us per-entry provenance; map
    # fingerprint → provenance for the explore bucket.
    explore_search = state.get("explore_search") or {}
    provenance_by_fp: dict[str, str] = {}
    if isinstance(explore_search, dict):
        for w in explore_search.get("winners_history") or []:
            if not isinstance(w, dict):
                continue
            fp = str(w.get("fingerprint") or "")
            prov = str(w.get("provenance") or "").strip()
            if fp and prov:
                provenance_by_fp[fp] = prov

    phase_buckets: dict[str, dict[str, Any]] = {
        "prelude": {"total_gain_pct": 0.0},
        # FRAMEWORK_PR is the dedicated upstream-PR bake-in
        # phase between PRELUDE and EXPLORE. KEEPs here come from
        # framework_pr action entries (one ts/gain per adopted PR).
        "framework_pr": {"total_gain_pct": 0.0, "by_pr": {}},
        "explore": {"total_gain_pct": 0.0, "by_domain": {}},
        "kernel":  {"total_gain_pct": 0.0, "by_kernel_id": {}},
        # GEMM_TUNING is the deterministic FP8 GEMM tuner that runs
        # at KERNEL entry. Logically inside the KERNEL phase but
        # bucketed separately so the dashboard can split deterministic
        # tuner gain from source-level GEAK/OOB rewrites.
        # ``by_tuned_file`` keys on ``state.optimization_stack[].tuned_file``
        # (absolute path to the produced CSV) so each adopted tune is
        # individually attributable.
        "gemm_tuning": {"total_gain_pct": 0.0, "by_tuned_file": {}},
        "sweep":   {"total_gain_pct": 0.0},
        "close":   {"total_gain_pct": 0.0},
        "unattributed": {"total_gain_pct": 0.0},
    }

    for e in entries:
        if not isinstance(e, dict):
            continue
        delta = _to_float(e.get("delta_pct"))
        if delta is None or delta <= 0:
            continue
        ts = e.get("ts_unix")
        if ts is None:
            ts_str = str(e.get("ts") or "")
            if ts_str:
                try:
                    # Best-effort parse — strip trailing 'Z' / TZ offsets.
                    from datetime import datetime as _dt
                    ts = _dt.fromisoformat(
                        ts_str.replace("Z", "+00:00")
                    ).timestamp()
                except (TypeError, ValueError):
                    ts = 0.0
        try:
            ts_f = float(ts or 0.0)
        except (TypeError, ValueError):
            ts_f = 0.0
        phase = _phase_for(ts_f).lower()
        action = str(e.get("action") or "").lower()
        fam = _action_family(action)
        # ``gemm_tuning`` runs *inside* the KERNEL phase but is bucketed
        # separately so the dashboard can split deterministic-tuner
        # gain from source-level GEAK/OOB rewrite gain. Override the
        # phase mapping by action family whenever the entry is a
        # gemm_tuning entry, regardless of phase_history's coarser
        # KERNEL label.
        if fam == "gemm_tuning":
            phase = "gemm_tuning"
        # When phase_history isn't usable, fall back to the action
        # family so we still bucket something usefully.
        elif phase not in phase_buckets:
            if fam in ("explore", "backends", "params"):
                phase = "explore"
            elif fam == "kernel":
                phase = "kernel"
            elif fam == "sweep":
                phase = "sweep"
            elif fam == "framework_pr":
                phase = "framework_pr"
            else:
                phase = "unattributed"
        bucket = phase_buckets[phase]
        bucket["total_gain_pct"] = round(
            float(bucket["total_gain_pct"]) + float(delta), 2,
        )
        if phase == "explore":
            by_domain = bucket.setdefault("by_domain", {})
            fp = str(e.get("fingerprint") or e.get("variant_fingerprint") or "")
            raw_prov = (
                provenance_by_fp.get(fp)
                or str(e.get("provenance") or "")
                or "default_grid"
            )
            # Normalize to a bare specialist key (strip ``specialist:``
            # prefix; fold ``legacy:*`` into ``legacy_*``). Without this
            # the dashboard had to manually splice the prefix every time
            # it iterated by_domain — error-prone and easy to miss when
            # the orchestrator adds a new SpecialistDomain.
            domain = _normalize_specialist_key(raw_prov)
            by_domain[domain] = round(
                float(by_domain.get(domain, 0.0)) + float(delta), 2,
            )
        elif phase == "kernel":
            by_kid = bucket.setdefault("by_kernel_id", {})
            kid = str(e.get("kernel_id") or e.get("action_kernel_id") or "?")
            by_kid[kid] = round(
                float(by_kid.get(kid, 0.0)) + float(delta), 2,
            )
        elif phase == "framework_pr":
            # variant_name on framework_pr entries is the PR ref
            # (``PR:<repo>#<num>`` / ``PR:<num>``); fall back to the
            # entry's ``ref`` field when present, else ``?`` so the
            # bucket key is always a string.
            by_pr = bucket.setdefault("by_pr", {})
            pr_key = (
                str(e.get("variant_name") or "").strip()
                or str(e.get("ref") or "").strip()
                or "?"
            )
            by_pr[pr_key] = round(
                float(by_pr.get(pr_key, 0.0)) + float(delta), 2,
            )
        elif phase == "gemm_tuning":
            # Coordinator stamps the tuned CSV path on each
            # ``optimization_stack[]`` entry; use that as the bucket
            # key so the dashboard can show "this tune of this CSV"
            # individually. Fall back to ``variant_name`` (currently
            # always ``"a8w8_blockscale_tuned_gemm"``) and finally
            # ``"?"`` so the key is always a string.
            by_tuned = bucket.setdefault("by_tuned_file", {})
            tuned_key = (
                str(e.get("tuned_file") or "").strip()
                or str(e.get("variant_name") or "").strip()
                or "?"
            )
            by_tuned[tuned_key] = round(
                float(by_tuned.get(tuned_key, 0.0)) + float(delta), 2,
            )

    # Drop the empty-by-default unattributed bucket when nothing
    # landed there — keeps the JSON clean.
    if phase_buckets["unattributed"]["total_gain_pct"] == 0.0:
        phase_buckets.pop("unattributed", None)

    if not timeline:
        warnings.append(
            "attribution.phase_breakdown: phase_history empty; gains "
            "bucketed via action family fallback"
        )

    return phase_buckets


def _reconstruct_gain_ledger(
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Approximate per-stack contribution when Coordinator hasn't recorded it.

    optimization_stack is an ordered list. We treat each entry's
    ``gain_pct`` field as that step's delta. This is best-effort only;
    consumers should check ``attribution.notes`` for the caveat.
    """
    stack = state.get("optimization_stack") or []
    if not isinstance(stack, list):
        return []
    cum_before = 0.0
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(stack):
        if not isinstance(entry, dict):
            continue
        delta = _to_float(entry.get("gain_pct"))
        cum_after = cum_before + (delta or 0.0)
        out.append({
            "ts":                str(entry.get("ts") or ""),
            "stack_len_before":  i,
            "stack_len_after":   i + 1,
            "action":            str(entry.get("action") or ""),
            "variant_name":      str(entry.get("variant_name") or ""),
            "cum_gain_before":   round(cum_before, 4),
            "cum_gain_after":    round(cum_after, 4),
            "delta_pct":         delta,
            "extra_server_args": str(entry.get("extra_server_args") or ""),
        })
        cum_before = cum_after
    return out


# ---------------------------------------------------------------------------
# source_files map
# ---------------------------------------------------------------------------
_KERNEL_ROOFLINE_REL_PATH = "reports/kernel_roofline.json"


# ---------------------------------------------------------------------------
# Roofline — optimization-progress curve (Dashboard 对接清单 §2)
# ---------------------------------------------------------------------------
# Default ratio of vendor-peak HBM bandwidth that's actually achievable.
# Vendor specs are theoretical maxima; sustained throughput is bounded
# by memory-controller scheduling, cache-line granularity, thermal /
# power throttling, and (for multi-GPU) XGMI arbitration. 70% is the
# conservative target Hyperloom optimizes against — see Dashboard-
# Roofline 对接清单 §2 for rationale.
DEFAULT_ROOFLINE_TARGET_RATIO = 0.70


def collect_roofline_progress(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Build the ``roofline_progress`` section feeding the
    optimization-progress chart (Dashboard-Roofline 对接清单 §2).

    Originally exported as the top-level ``roofline`` field, but that
    name collided with the markdown-report renderer's existing
    ``roofline`` list contract (per-final.json comparison snapshots
    produced by :func:`collect_roofline`, populated from
    ``state.roofline_snapshots``). After the merge of #368 + #370 the
    two collectors silently shadowed each other in the same file
    (Python kept the second definition); the dashboard chart payload
    won and the markdown report's Roofline section started rendering
    empty. Renaming this one to ``roofline_progress`` resolves the
    clash — both surfaces are now stable, addressable independently,
    and free to evolve.

    Pulls everything from in-memory ``state`` + ``manifest``; never
    re-runs benchmarks. The dashboard reads sbd alone — no
    ``state.json`` walk on the consumer side.

    Output shape (RooflineProgress TypedDict):

    * ``trajectory[]`` — 1 baseline point + N KEEP points, sorted by
      ts. Always at least one point (baseline) when ``baseline_tput``
      is known; empty when the session never finished baseline.
    * ``ceiling_tok_per_sec`` / ``target_tok_per_sec`` —
      ``state.roofline_snapshots[-1]`` (latest snapshot, not [0]; the
      ceiling can drift across re-runs as the snapshot pipeline
      refines its peak estimate). ``None`` when no snapshot exists;
      ``ceiling_available`` is the explicit boolean flag.
    * ``current_best_tput`` — last trajectory point's tput, validated
      against ``state.current_best.tput`` when present.
    * ``current_best_pct_of_*`` — convenience ratios for the dashboard
      callout. ``None`` when the ceiling is missing.
    * ``snapshots[]`` — full passthrough of ``state.roofline_snapshots``
      for tooltips / drill-downs.

    Never raises; structured failures land in ``warnings`` and the
    section becomes a partial dict.
    """
    # ── Trajectory: baseline + each KEEP ────────────────────────────────
    baseline_tput = _to_float(state.get("baseline_tput")) or 0.0
    trajectory: list[dict[str, Any]] = []
    if baseline_tput > 0:
        trajectory.append({
            "ts":         str(manifest.get("created_at_utc") or ""),
            "tput":       baseline_tput,
            "label":      "baseline",
            "action":     "baseline",
            "gain_pct":   0.0,
            "flags":      "",
            "extra_envs": {},
        })

    stack = state.get("optimization_stack") or []
    if isinstance(stack, list):
        # The stack is already in promotion order, but sort by ts as a
        # belt-and-braces guard against legacy paths that prepended.
        ordered = sorted(
            (e for e in stack if isinstance(e, dict)),
            key=lambda e: str(e.get("ts") or ""),
        )
        for entry in ordered:
            tput = _to_float(entry.get("tput"))
            if tput is None or tput <= 0:
                continue
            gain_pct = (
                ((tput - baseline_tput) / baseline_tput * 100.0)
                if baseline_tput > 0 else 0.0
            )
            trajectory.append({
                "ts":         str(entry.get("ts") or ""),
                "tput":       tput,
                "label":      str(entry.get("variant_name") or entry.get("action") or ""),
                "action":     str(entry.get("action") or ""),
                "gain_pct":   round(gain_pct, 4),
                "flags":      str(entry.get("candidate_extra_server_args") or ""),
                "extra_envs": dict(entry.get("extra_envs") or {}),
            })

    # ── Reference lines: ceiling + target ───────────────────────────────
    snapshots_raw = state.get("roofline_snapshots") or []
    snapshots: list[dict[str, Any]] = []
    if isinstance(snapshots_raw, list):
        for snap in snapshots_raw:
            if isinstance(snap, dict):
                snapshots.append(_normalize_roofline_snapshot(snap))

    # Use the LATEST snapshot for headline numbers — the ceiling
    # estimate refines as the watermark roofline pipeline reruns.
    latest_snap = snapshots[-1] if snapshots else None
    ceiling_tok = (
        _to_float(latest_snap.get("theoretical_peak_tok_per_sec"))
        if latest_snap else None
    )
    ceiling_available = ceiling_tok is not None and ceiling_tok > 0
    target_tok = (
        round(ceiling_tok * DEFAULT_ROOFLINE_TARGET_RATIO, 4)
        if ceiling_available else None
    )

    # ── Headline numbers ────────────────────────────────────────────────
    current_best_tput = (
        trajectory[-1]["tput"] if trajectory else 0.0
    )
    cumulative_gain_pct = _to_float(state.get("cumulative_gain")) or 0.0
    pct_of_ceiling = (
        round(current_best_tput / ceiling_tok * 100.0, 4)
        if ceiling_available and current_best_tput > 0 else None
    )
    pct_of_target = (
        round(current_best_tput / target_tok * 100.0, 4)
        if (target_tok and target_tok > 0 and current_best_tput > 0) else None
    )

    out: dict[str, Any] = {
        "ceiling_tok_per_sec":          ceiling_tok,
        "target_tok_per_sec":           target_tok,
        "ceiling_ratio_target":         DEFAULT_ROOFLINE_TARGET_RATIO,
        "ceiling_available":            ceiling_available,
        "trajectory":                   trajectory,
        "baseline_tput":                baseline_tput,
        "current_best_tput":            current_best_tput,
        "cumulative_gain_pct":          round(cumulative_gain_pct, 4),
        "current_best_pct_of_ceiling":  pct_of_ceiling,
        "current_best_pct_of_target":   pct_of_target,
        "roofline_failure_streak":      _to_int(state.get("roofline_failure_streak")) or 0,
        "snapshots":                    snapshots,
    }
    if latest_snap:
        out["snapshot_top_bottleneck"] = str(latest_snap.get("top_bottleneck") or "")
        within = _to_float(latest_snap.get("within_roofline_pct"))
        if within is not None:
            out["snapshot_within_roofline_pct"] = within
        gap = _to_float(latest_snap.get("gap_to_roofline_pct"))
        if gap is not None:
            out["snapshot_gap_to_roofline_pct"] = gap

    # Sanity check: trajectory tail should match state.current_best.tput
    # when both are known. A divergence means the stack wasn't fully
    # promoted (resume mid-promotion) — surface it instead of hiding.
    cb_tput = _to_float((state.get("current_best") or {}).get("tput"))
    if (
        cb_tput is not None and cb_tput > 0
        and current_best_tput > 0
        and abs(cb_tput - current_best_tput) / max(cb_tput, 1.0) > 0.001
    ):
        warnings.append(
            f"roofline.current_best_tput ({current_best_tput:.2f}) does not "
            f"match state.current_best.tput ({cb_tput:.2f}); the trajectory "
            f"may be missing a promotion event."
        )
    return out


def _normalize_roofline_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """Coerce one ``state.roofline_snapshots[]`` entry to the schema."""
    top_kernel_raw = snap.get("top_kernel") or {}
    top_kernel: dict[str, Any] = {}
    if isinstance(top_kernel_raw, dict):
        top_kernel = {
            "name":           str(top_kernel_raw.get("name") or ""),
            "bound_type":     str(top_kernel_raw.get("bound_type") or ""),
            "efficiency_pct": _to_float(top_kernel_raw.get("efficiency_pct")) or 0.0,
            "gpu_pct":        _to_float(top_kernel_raw.get("gpu_pct")) or 0.0,
        }
    return {
        "snapshot_id":                  _to_int(snap.get("snapshot_id")) or 0,
        "ts":                           str(snap.get("ts") or ""),
        "achieved_tok_per_sec":         _to_float(snap.get("achieved_tok_per_sec")) or 0.0,
        "theoretical_peak_tok_per_sec": _to_float(snap.get("theoretical_peak_tok_per_sec")) or 0.0,
        "within_roofline_pct":          _to_float(snap.get("within_roofline_pct")) or 0.0,
        "gap_to_roofline_pct":          _to_float(snap.get("gap_to_roofline_pct")) or 0.0,
        "compute_pct":                  _to_float(snap.get("compute_pct")) or 0.0,
        "idle_pct":                     _to_float(snap.get("idle_pct")) or 0.0,
        "comm_pct":                     _to_float(snap.get("comm_pct")) or 0.0,
        "top_bottleneck":               str(snap.get("top_bottleneck") or ""),
        "top_kernel":                   top_kernel,
        "analysis_md_path":             str(snap.get("analysis_md_path") or ""),
        "kernel_roofline_path":         str(snap.get("kernel_roofline_path") or ""),
        "trace_input":                  str(snap.get("trace_input") or ""),
    }


def collect_kernel_roofline(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Mirror ``<session_dir>/reports/kernel_roofline.json`` into the
    breakdown's ``kernel_roofline`` section.

    The kernel-agent watermark roofline pipeline writes this file once
    per successful trace analysis (minute-cadence on a live session,
    once at end-of-session post-mortem). The dashboard renders one row
    per kernel for the "Kernel Roofline 表格" panel
    (Dashboard-Roofline 对接清单 §1).

    File missing → empty dict + warning. Malformed JSON → empty dict +
    warning (``_load_json_safe`` already records the parse error).
    Non-list ``kernels`` field is replaced with ``[]`` so the consumer
    can iterate unconditionally.

    Each kernel entry's fields are coerced through the standard helpers
    so an upstream type change (e.g. ``call_count`` arriving as a JSON
    float) doesn't break the dashboard's downstream parsers.
    """
    path = session_dir / _KERNEL_ROOFLINE_REL_PATH
    if not path.exists():
        # Quiet on absence — most sessions never run the roofline
        # pipeline and a warning would clutter every breakdown.
        return {}
    blob = _load_json_safe(path, warnings)
    if not isinstance(blob, dict):
        warnings.append(
            f"kernel_roofline: {_KERNEL_ROOFLINE_REL_PATH} is not a JSON object"
        )
        return {}

    raw_kernels = blob.get("kernels")
    if raw_kernels is None:
        kernels: list[dict[str, Any]] = []
    elif not isinstance(raw_kernels, list):
        warnings.append(
            "kernel_roofline.kernels is not a list; dropping entries"
        )
        kernels = []
    else:
        kernels = [_normalize_kernel_roofline_entry(k) for k in raw_kernels
                   if isinstance(k, dict)]

    out: dict[str, Any] = {
        "schema_version":         _to_int(blob.get("schema_version")) or 1,
        "source":                 str(blob.get("source") or ""),
        "analysis_md_path":       str(blob.get("analysis_md_path") or ""),
        "kernel_candidates_path": str(blob.get("kernel_candidates_path") or ""),
        "trace_input":            str(blob.get("trace_input") or ""),
        "trace_input_type":       str(blob.get("trace_input_type") or ""),
        "kernels":                kernels,
    }
    return out


def _normalize_kernel_roofline_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce one kernel entry to the schema shape with stable types."""
    return {
        "kernel_id":             str(raw.get("kernel_id") or ""),
        "name":                  str(raw.get("name") or ""),
        "source_file":           str(raw.get("source_file") or ""),
        "kernel_category":       str(raw.get("kernel_category") or ""),
        "bound_type":            str(raw.get("bound_type") or ""),
        "arithmetic_intensity":  _to_float(raw.get("arithmetic_intensity")) or 0.0,
        "flops_per_byte":        _to_float(raw.get("flops_per_byte")) or 0.0,
        "efficiency_percent":    _to_float(raw.get("efficiency_percent")) or 0.0,
        "gpu_pct":               _to_float(raw.get("gpu_pct")) or 0.0,
        "call_count":            _to_int(raw.get("call_count")) or 0,
        "duration_us":           _to_float(raw.get("duration_us")) or 0.0,
        "reusable_native_kernel": bool(raw.get("reusable_native_kernel")),
    }


# ---------------------------------------------------------------------------
# Kernel Optimization Summary (Breakdown 面板对接文档 §A1; PR #399)
# ---------------------------------------------------------------------------
_KERNEL_OPT_SUMMARY_REL_PATH = "reports/kernel_optimization_summary.json"


def collect_kernel_optimization_summary(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Mirror ``<session_dir>/reports/kernel_optimization_summary.json``
    into the breakdown's ``kernel_optimization_summary`` section
    (Breakdown 面板对接文档 §A1).

    The file is produced deterministically by the ``report`` action
    (``orchestrator.kernel_attempt_summary.build_kernel_optimization_summary``),
    which runs as CLOSE step 1 — strictly *before* the
    ``session_breakdown`` action (step 2) writes this breakdown — so the
    report is already on disk for a normally-closed session. Mirroring
    it here lets the dashboard read sbd alone instead of walking the
    ``reports/`` + kernel-agent tree itself.

    Behaviour (matches :func:`collect_kernel_roofline`):

    * File missing → ``{}``, **no warning**. Offline / pre-#399 sessions
      and any session where the ``report`` action never ran simply won't
      have it; the dashboard hides Block 1 on an empty dict.
    * Malformed / non-object JSON → ``{}`` + warning (never raises).
    * Otherwise the report is mirrored **verbatim** (so producer-side
      field additions ride through without a breakdown schema change),
      apart from light shape guards on the containers the dashboard
      iterates (``by_kernel`` list; ``totals`` /
      ``*_breakdown`` / ``field_glossary`` objects), plus an added
      ``report_path`` rel-link to the source file.
    """
    path = session_dir / _KERNEL_OPT_SUMMARY_REL_PATH
    if not path.exists():
        # Quiet on absence — keeps breakdowns of legacy / non-report
        # sessions warning-free (mirrors collect_kernel_roofline).
        return {}
    blob = _load_json_safe(path, warnings)
    if not isinstance(blob, dict):
        warnings.append(
            f"kernel_optimization_summary: {_KERNEL_OPT_SUMMARY_REL_PATH} "
            "is not a JSON object"
        )
        return {}

    out = dict(blob)

    raw_by_kernel = out.get("by_kernel")
    if raw_by_kernel is None:
        out["by_kernel"] = []
    elif not isinstance(raw_by_kernel, list):
        warnings.append(
            "kernel_optimization_summary.by_kernel is not a list; dropping entries"
        )
        out["by_kernel"] = []
    else:
        # Drop any non-dict rows so the dashboard can iterate safely;
        # otherwise pass each row through verbatim (nested verification /
        # backend_ladder shapes documented in §A1.4).
        out["by_kernel"] = [r for r in raw_by_kernel if isinstance(r, dict)]

    for key in (
        "totals", "rejection_breakdown", "unattempted_reason_breakdown",
        "failure_reason_breakdown", "field_glossary",
    ):
        val = out.get(key)
        if val is not None and not isinstance(val, dict):
            warnings.append(
                f"kernel_optimization_summary.{key} is not an object; dropping"
            )
            out[key] = {}

    takeaways = out.get("top_takeaways")
    if takeaways is not None and not isinstance(takeaways, list):
        warnings.append(
            "kernel_optimization_summary.top_takeaways is not a list; dropping"
        )
        out["top_takeaways"] = []

    out["report_path"] = _rel(path, session_dir) or _KERNEL_OPT_SUMMARY_REL_PATH
    return out


# ---------------------------------------------------------------------------
# Conc Sweep Summary (Breakdown 面板对接文档 §A2; PR #399)
# ---------------------------------------------------------------------------
_CONC_SWEEP_SUMMARY_REL_PATH = "reports/conc_sweep_summary.json"


def collect_conc_sweep_summary(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Mirror ``<session_dir>/reports/conc_sweep_summary.json`` into the
    breakdown's ``conc_sweep_summary`` section (Breakdown 面板对接文档 §A2).

    Produced by the ``conc_sweep`` action during SWEEP (well before
    CLOSE), so it is already on disk when sbd exports. Surfacing it here
    extends the single-CONC headline gain into the full baseline-vs-
    current_best CONC ladder for the dashboard's dual-curve chart.

    Behaviour:

    * File missing → ``{}``, **no warning** (conc_sweep often disabled /
      skipped; the dashboard hides Block 2 on an empty dict).
    * Malformed / non-object JSON → ``{}`` + warning (never raises).
    * Otherwise mirrored **verbatim** + an added ``report_path``. The
      only shape guard is on ``comparison`` (the array the dual-curve
      chart iterates directly).

    IMPORTANT — do **not** synthesize the optional blocks: when the
    producer writes ``status="skipped"`` it intentionally omits
    ``baseline`` / ``optimized`` / ``comparison`` / ``summary`` (§A2.4),
    and ``roofline_ceiling`` (§A2.9) is absent on older products.
    Consumers must branch on ``status`` before reading those keys; we
    pass through exactly what the producer wrote.
    """
    path = session_dir / _CONC_SWEEP_SUMMARY_REL_PATH
    if not path.exists():
        return {}
    blob = _load_json_safe(path, warnings)
    if not isinstance(blob, dict):
        warnings.append(
            f"conc_sweep_summary: {_CONC_SWEEP_SUMMARY_REL_PATH} "
            "is not a JSON object"
        )
        return {}

    out = dict(blob)

    comparison = out.get("comparison")
    if comparison is not None and not isinstance(comparison, list):
        warnings.append(
            "conc_sweep_summary.comparison is not a list; dropping entries"
        )
        out["comparison"] = []

    out["report_path"] = _rel(path, session_dir) or _CONC_SWEEP_SUMMARY_REL_PATH
    return out


# ---------------------------------------------------------------------------
# Optimization stack — raw KEEP ledger passthrough
# ---------------------------------------------------------------------------
# ``state.optimization_stack[]`` is the authoritative ordered list of
# every promotion the Coordinator has accepted in this session. Other
# breakdown sections summarise it for specific consumers
# (``final.action_path`` is a label-only string list,
# ``attribution.gain_per_stack_entry`` is the gain ledger,
# ``roofline_progress.trajectory`` is the chart-friendly view) but
# none of them carry the full per-entry metadata downstream tooling
# may need — e.g. ``tuned_file`` / ``final_report_path`` on a
# ``gemm_tuning`` entry, or ``workspace`` for arbitrary KEEPs.
#
# This passthrough surfaces the raw stack at sbd top level so
# consumers that need the full evidence set can read sbd alone (no
# state.json walk on the consumer side, matching the dashboard
# read-once contract).
#
# Field shape mirrors the in-state-dict shape; entries are coerced
# defensively (string/None/dict) so downstream tooling never has to
# guard against type drift.
def collect_optimization_stack(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Mirror ``state.optimization_stack[]`` to the breakdown.

    Returns ``[]`` when the field is absent or empty (pre-baseline /
    fresh session). Never raises.
    """
    stack = state.get("optimization_stack") or []
    if not isinstance(stack, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in stack:
        if not isinstance(entry, dict):
            continue
        out.append(_normalize_optimization_stack_entry(entry))
    return out


def _normalize_optimization_stack_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce one stack entry to the schema shape with stable types.

    Unknown / future fields pass through verbatim under their raw
    keys so the schema can lag the state writer without losing data
    (forward compatibility — see Inv-10.1 fact-layer compat).
    """
    # Known fields — coerced types
    out: dict[str, Any] = {
        "action":                       str(raw.get("action") or ""),
        "variant_name":                 str(raw.get("variant_name") or ""),
        "candidate_extra_server_args":  str(raw.get("candidate_extra_server_args") or ""),
        "extra_envs":                   dict(raw.get("extra_envs") or {}),
        "tput":                         _to_float(raw.get("tput")),
        "ts":                           str(raw.get("ts") or ""),
        "workspace":                    raw.get("workspace"),
    }
    # gemm_tuning-specific evidence (optional, only populated by the
    # Coordinator's ``_promote_gemm_tuning_keep`` path).
    if "tuned_file" in raw:
        out["tuned_file"] = str(raw.get("tuned_file") or "")
    if "final_report_path" in raw:
        out["final_report_path"] = str(raw.get("final_report_path") or "")
    if "source" in raw:
        out["source"] = str(raw.get("source") or "")
    if "gain_pct" in raw:
        out["gain_pct"] = _to_float(raw.get("gain_pct"))
    if "kernel_id" in raw:
        out["kernel_id"] = str(raw.get("kernel_id") or "")
    if "fingerprint" in raw:
        out["fingerprint"] = str(raw.get("fingerprint") or "")
    if "provenance" in raw:
        out["provenance"] = str(raw.get("provenance") or "")
    if "task_id" in raw:
        out["task_id"] = str(raw.get("task_id") or "")
    return out


def collect_source_files(
    session_dir: Path,
    baseline_path: str | None,
    profile_reports: list[str],
    sweep_reports: list[str],
) -> dict[str, Any]:
    kernel_attempts = [
        _rel(run_dir / "optimization_attempts.jsonl", session_dir) or str(run_dir / "optimization_attempts.jsonl")
        for run_dir in _kernel_agent_run_dirs(session_dir)
        if (run_dir / "optimization_attempts.jsonl").exists()
    ]
    critic = session_dir / "critic-workdir"
    rob = session_dir / "robustness-workdir"
    out: dict[str, Any] = {
        "manifest":           "manifest.json",
        "state":              "state.json",
        "baseline_report":    baseline_path,
        "critic_workdir":     "critic-workdir" if critic.exists() else None,
        "robustness_workdir": "robustness-workdir" if rob.exists() else None,
    }
    # Skip emitting list-valued categories that are empty so the
    # source_files renderer doesn't surface ``count=0, first_values=—``
    # rows for kernels / sweep / profile work that simply didn't run.
    # SourceFiles schema fields are all NotRequired-by-convention.
    for key, lst in (
        ("profile_reports", profile_reports),
        ("sweep_reports",   sweep_reports),
        ("kernel_attempts", kernel_attempts),
    ):
        if lst:
            out[key] = lst
    return out


# ---------------------------------------------------------------------------
# §16 Phase segments — phase state machine
# ---------------------------------------------------------------------------
def collect_phase_segments(
    state: dict[str, Any],
    phase_timeline: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Group action events by phase using ``phase_history`` boundaries.

    Returns a list of segments shaped like::

        {
            "phase":            "EXPLORE",
            "entered_ts":       "...iso...",
            "entered_unix":     float | None,
            "exit_ts":          "...iso..." | "",
            "exit_unix":        float | None,
            "exit_reason":      "plateau_explore",
            "evidence":         {...},
            "events":           [<sub-events folded into this phase>...],
            "actions":          [<phase_timeline events in this phase>...],
            "elapsed_seconds":  float | None,
        }

    Only ``phase_history`` rows that carry a non-empty ``to_phase`` are
    real phase transitions and become segment boundaries. Rows without it
    (e.g. ``{"event": "framework_pr_phase_done"}``) are *sub-events* — we
    fold them into the segment whose window contains them instead of
    letting them spawn a phantom empty-named segment that also splits the
    surrounding phase in two.

    A segment's exit comes from the *next transition* (not the next raw
    row), so ``exit_unix`` / ``exit_reason`` / ``elapsed_seconds`` line up
    with the real phase boundary. Both numeric (`*_unix`) and ISO
    (`*_ts`) representations are always emitted, derived from each other
    when only one is present.

    Action attribution prefers each event's own ``phase`` (journal-sourced
    events carry it) so boundary actions such as the PRELUDE baseline land
    in the right phase; events without a ``phase`` fall back to the
    ``[entered_ts, exit_ts)`` window.

    Empty when ``phase_history`` is missing — readers fall back to the
    flat ``phase_timeline`` (v1 shape).
    """
    history = state.get("phase_history") or []
    if not isinstance(history, list) or not history:
        return []

    rows = [r for r in history if isinstance(r, dict)]
    transitions = [r for r in rows if str(r.get("to_phase") or "")]
    sub_events = [r for r in rows if not str(r.get("to_phase") or "")]

    def _unix(row: dict[str, Any]) -> float | None:
        u = row.get("ts_unix")
        if isinstance(u, (int, float)):
            return float(u)
        return _parse_iso_unix(row.get("ts"))

    segments: list[dict[str, Any]] = []
    proxy_seen = False
    for idx, row in enumerate(transitions):
        entered_ts = _iso_z(row.get("ts"))
        entered_unix = _unix(row)
        exit_ts = ""
        exit_unix: float | None = None
        exit_reason = ""
        if idx + 1 < len(transitions):
            nxt = transitions[idx + 1]
            exit_ts = _iso_z(nxt.get("ts"))
            exit_reason = str(nxt.get("reason") or "")
            exit_unix = _unix(nxt)
        elapsed: float | None = None
        if entered_unix is not None and exit_unix is not None:
            elapsed = max(0.0, float(exit_unix) - float(entered_unix))
        evidence_dict = dict(row.get("evidence") or {})
        segments.append({
            "phase":           str(row.get("to_phase") or ""),
            "from_phase":      str(row.get("from_phase") or ""),
            "entered_ts":      entered_ts,
            "entered_unix":    entered_unix,
            "exit_ts":         exit_ts,
            "exit_unix":       exit_unix,
            "exit_reason":     exit_reason,
            "evidence":        evidence_dict,
            "events":          [],
            "actions":         [],
            "elapsed_seconds": elapsed,
        })

    def _owner_by_window(ts: str) -> dict[str, Any] | None:
        """Return the segment whose ``[entered_ts, exit_ts)`` ISO window
        holds ``ts``. ISO-8601 sorts lexicographically, so string compare
        matches chronological order (same convention as the v1 collector).
        """
        if not ts:
            return segments[-1] if segments else None
        for s in segments:
            lo_ts, hi_ts = s["entered_ts"], s["exit_ts"]
            if lo_ts and ts < lo_ts:
                continue
            if hi_ts and ts >= hi_ts:
                continue
            return s
        return segments[-1] if segments else None

    # Fold non-transition rows (sub-events) into their containing segment.
    for ev in sub_events:
        ev_evidence = dict(ev.get("evidence") or {})
        if ev_evidence.get("r09_provisional") or (
            str(ev_evidence.get("evidence") or "") == "m2_proxy"
        ):
            proxy_seen = True
        ev_ts = _iso_z(ev.get("ts"))
        s = _owner_by_window(ev_ts)
        if s is not None:
            s["events"].append({
                "event":  str(ev.get("event") or ev.get("reason") or ""),
                "reason": str(ev.get("reason") or ""),
                "ts":     ev_ts,
                "evidence": ev_evidence,
            })

    # Attribute timeline actions: prefer the event's declared ``phase``
    # (journal-sourced), else the ts window.
    phase_to_segs: dict[str, list[dict[str, Any]]] = {}
    for s in segments:
        phase_to_segs.setdefault(s["phase"], []).append(s)
    for ev in phase_timeline or []:
        if not isinstance(ev, dict):
            continue
        ts = str(ev.get("ts") or "")
        if not ts:
            continue
        target = None
        ev_phase = str(ev.get("phase") or "")
        if ev_phase and ev_phase in phase_to_segs:
            cands = phase_to_segs[ev_phase]
            if len(cands) == 1:
                target = cands[0]
            else:
                for s in cands:
                    lo_ts, hi_ts = s["entered_ts"], s["exit_ts"]
                    if lo_ts and ts < lo_ts:
                        continue
                    if hi_ts and ts >= hi_ts:
                        continue
                    target = s
                    break
                target = target or cands[0]
        if target is None:
            target = _owner_by_window(ts)
        if target is not None:
            target["actions"].append(ev)

    if proxy_seen:
        # KB_gaps/Gap-15 / KB_design §3.14 R-09 — surface a single
        # session-level marker so dashboards can flag legacy-proxy
        # exits without scraping per-segment evidence.
        warnings.append(
            "plateau_proxy_provisional: legacy params_no_promote_streak "
            "proxy fired (R-09); set INFERENCE_OPTIMIZER_DISABLE_PLATEAU_PROXY=1 "
            "once the fleet is fully v0.8 to fail closed"
        )
    return segments


# ---------------------------------------------------------------------------
# §15 KB Provenance — Cortex KB integration audit
# ---------------------------------------------------------------------------
def collect_kb_provenance(
    session_dir: Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    """Collect the Cortex KB integration audit for ``session_breakdown.json``.

    Three sources merged into one section:

    1. SharedState (``state.json``) — ``cortex_session_id`` (hyperloom-
       local id, NOT a KB sid) + ``warm_start_*`` snapshots.
    2. NDJSON queues (``runtime/cortex/.kb_*.ndjson``) — counts of
       drained / dead-letter rows. The flusher daemon writes one
       ``drain_bookmark`` per round; we just sum the deltas.
    3. Synchronous audit log (``runtime/cortex/.kb_audit.jsonl``) — per
       Cortex CLI call status. Useful for diagnosing T0 sync failures
       from the breakdown JSON alone.

    Returns a stable shape for live RecipeKB / PR Monitor observability.
    The old T2/T3 graph edge placeholders were removed with the
    KnowledgePlane Cortex graph surface.
    """
    from ..session_paths import (
        cortex_audit_jsonl as _audit_path,
        cortex_dead_letter_ndjson as _dl_path,
        cortex_flushed_ndjson as _flushed_path,
        cortex_flusher_pid as _flusher_pid_path,
        cortex_flusher_status_json as _flusher_status_path,
        cortex_pending_ndjson as _pending_path,
        pr_monitor_status_json as _pr_status_path,
    )

    # Surface the PR Monitor reachability snapshot written at cli boot.
    # We use ``warnings`` (top-level breakdown.warnings) rather than a
    # dedicated section so the operator can grep for ``pr_monitor``
    # regardless of the schema version they expect.
    pr_status_path = _pr_status_path(session_dir)
    if pr_status_path.exists():
        try:
            with pr_status_path.open("r", encoding="utf-8") as f:
                pr_status = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(
                f"pr_monitor:status_marker_unreadable:{exc!r}"[:240]
            )
        else:
            if not pr_status.get("enabled"):
                warnings.append("pr_monitor:disabled")
            elif not pr_status.get("reachable"):
                # operator dashboard uses this exact string to light up
                # the "PR Monitor cross-cluster ingress" alert.
                url = str(pr_status.get("url") or "")
                warnings.append(
                    f"pr_monitor:unreachable:{url}"[:240] if url
                    else "pr_monitor:unreachable"
                )

    def _count_lines(p: Path) -> int:
        try:
            if not p.exists():
                return 0
            with p.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except OSError as exc:
            warnings.append(f"kb_provenance: failed to count {p}: {exc!r}")
            return 0

    def _read_last_n_audit(p: Path, n: int = 50) -> list[dict[str, Any]]:
        try:
            if not p.exists():
                return []
            with p.open("r", encoding="utf-8") as f:
                rows = [
                    json.loads(line) for line in f
                    if line.strip()
                ]
            return rows[-n:]
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"kb_provenance: failed to read audit {p}: {exc!r}")
            return []

    pending_path = _pending_path(session_dir)
    flushed_path = _flushed_path(session_dir)
    dl_path = _dl_path(session_dir)
    audit_path = _audit_path(session_dir)

    audit_tail = _read_last_n_audit(audit_path, n=50)
    # Status counts aggregated across the audit tail.
    status_counts: dict[str, int] = {}
    for row in audit_tail:
        st = str(row.get("status") or "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    cortex_sid = (state.get("cortex_session_id") or "").strip()
    warm = state.get("warm_start_recipe") or {}
    pitfalls = state.get("warm_start_pitfalls") or []
    lessons = state.get("warm_start_lessons") or []
    # warm-recipe replay outcome (one-shot reproduce of the KB
    # best_config at PRELUDE). Empty dict before the replay completes /
    # when ``--no-warm-replay`` was set; otherwise carries:
    #   {status, expected_gain_pct, actual_gain_pct,
    #    warm_recipe_tier, warm_recipe_conf, replay_task_id, reason}
    warm_replay_outcome = state.get("warm_replay_outcome") or {}

    out: dict[str, Any] = {
        "cortex_session_id":      cortex_sid,
        "warm_start_ts":          state.get("warm_start_ts") or "",
        "warm_start_recipe_seen": bool(warm and warm.get("raw")),
        "warm_start_recipe_tier": str(warm.get("tier") or "") if isinstance(warm, dict) else "",
        "warm_start_pitfall_count": len(pitfalls) if isinstance(pitfalls, list) else 0,
        "warm_start_lesson_count": len(lessons) if isinstance(lessons, list) else 0,
        # operator-visible replay summary. The outcome dict is
        # passed through verbatim so dashboards can render status
        # transitions over time.
        "warm_replay": dict(warm_replay_outcome) if isinstance(warm_replay_outcome, dict) else {},
        "warm_replay_attempted":   bool(state.get("warm_replay_attempted")),
        "warm_history_injected":   bool(state.get("warm_history_injected")),
        "stack_fingerprint":      manifest.get("stack_fingerprint") or {},
        "queue": {
            "pending_lines":     _count_lines(pending_path),
            "flushed_bookmarks": _count_lines(flushed_path),
            "dead_letter_lines": _count_lines(dl_path),
        },
        "audit_tail_count":     len(audit_tail),
        "audit_status_counts":  status_counts,
        "flusher_status": _collect_flusher_status(
            session_dir,
            status_path=_flusher_status_path(session_dir),
            pid_path=_flusher_pid_path(session_dir),
            warnings=warnings,
        ),
        "kb_degraded_reason": (manifest.get("kb_degraded_reason") or "") or None,
        "pr_degraded_reason": (manifest.get("pr_degraded_reason") or "") or None,
    }
    fs = out["flusher_status"]
    # Only emit a warning when a boot marker was written (i.e. cli ran
    # the spawn helper this session); a missing marker is treated as
    # "legacy / pre-Dead-E session" rather than a misconfiguration.
    if fs.get("reason") != "no_marker":
        if not fs.get("enabled", True):
            warnings.append("kb_flusher:disabled")
        elif not fs.get("alive", False):
            warnings.append("kb_flusher:not_alive")
    return out


def _collect_flusher_status(
    session_dir: Path,
    *,
    status_path: Path,
    pid_path: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Merge ``.kb_flusher_status.json`` (boot marker) with a live
    ``kill -0 $pid`` probe so the breakdown reader sees one stable
    shape.
    """
    base: dict[str, Any] = {
        "enabled":       False,
        "spawned":       False,
        "alive":         False,
        "pid":           None,
        "cortex_kb_url": None,
        "interval_sec":  0.0,
        "batch_size":    0,
        "reason":        "no_marker",
        "ts":            "",
        "pid_path":      str(pid_path),
    }
    if status_path.exists():
        try:
            with status_path.open("r", encoding="utf-8") as f:
                marker = json.load(f)
            if isinstance(marker, dict):
                for k in (
                    "enabled", "spawned", "pid", "cortex_kb_url",
                    "interval_sec", "batch_size", "reason", "ts", "pid_path",
                ):
                    if k in marker:
                        base[k] = marker[k]
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(
                f"kb_flusher:status_marker_unreadable:{exc!r}"[:240]
            )

    pid_alive = False
    pid_from_file: int | None = None
    if pid_path.exists():
        try:
            raw = pid_path.read_text(encoding="utf-8").strip().splitlines()
            pid_from_file = int(raw[0]) if raw else None
        except (OSError, ValueError):
            pid_from_file = None
        if pid_from_file:
            try:
                os.kill(pid_from_file, 0)
                pid_alive = True
            except (OSError, ProcessLookupError):
                pid_alive = False
    if pid_from_file and not base.get("pid"):
        base["pid"] = pid_from_file
    base["alive"] = pid_alive
    return base


# ---------------------------------------------------------------------------
# specialist_runs section
# ---------------------------------------------------------------------------
def _coerce_round_id(value: Any) -> int | str:
    """Normalise a ``specialist_rounds[*].round_id`` for the breakdown.

    ``record_specialist_round`` stores ``round_id`` as a string — it may
    be a numeric round counter ("3"), an explore-round label
    ("explore-001"), or a task-id hash when a single specialist task is
    the round anchor. Return an int when the value is purely numeric so
    downstream numeric consumers keep working, otherwise preserve the
    string. Empty / None collapses to ``0``. Never raises.
    """
    if value is None or value == "":
        return 0
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lstrip("-").isdigit():
        try:
            return int(text)
        except ValueError:
            return text
    return text


def collect_specialist_runs(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
    *,
    include_transcripts: bool = False,
) -> list[dict[str, Any]]:
    """Build the ``specialist_runs`` breakdown section.

    Two data sources merge here:

    1. ``state.json.specialist_rounds[]`` — per-round summary the
       Coordinator writes after every EXPLORE specialist dispatch.
    2. ``<session_dir>/runs/specialist/<task_id>/specialist_done.json``
       — the runner's per-task transcript / proposal_set artifact.

    The function is best-effort: a missing ``runs/specialist/``
    directory simply means no transcripts get attached. We never
    crash the export — every recoverable issue lands in ``warnings``
    via the caller's ``_safe_collect`` wrapper.

    Args:
        session_dir: absolute session root.
        state: parsed ``state.json`` as returned by
            :func:`_load_state`.
        warnings: shared warnings list (mutated in place).
        include_transcripts: when True, the transcript file bytes are
            inlined under each transcript ref's ``body`` field. The
            CLI flag ``--breakdown-include-transcripts`` controls
            this; default is False (path-only, smaller payload).
    """
    rounds = state.get("specialist_rounds") or []
    if not isinstance(rounds, list) or not rounds:
        return []

    # Pre-index the runs/specialist/ directory so the round-merge
    # below is O(1) per task lookup.
    runs_root = session_dir / "runs" / "specialist"
    by_task: dict[str, Path] = {}
    if runs_root.exists():
        try:
            for child in runs_root.iterdir():
                if not child.is_dir():
                    continue
                done_path = child / "specialist_done.json"
                if done_path.exists():
                    by_task[child.name] = done_path
        except OSError as exc:
            warnings.append(
                f"specialist_runs: failed to scan {runs_root}: {exc!r}"
            )

    out: list[dict[str, Any]] = []
    for raw in rounds:
        if not isinstance(raw, dict):
            continue
        # ``record_specialist_round`` writes singular ``domain`` /
        # ``task_id`` (one specialist task anchors a round) and a bare
        # ``confidence``; tolerate both the plural / aggregated shapes
        # (multi-task rounds) and the singular shapes so neither form is
        # silently dropped.
        domains = list(raw.get("domains") or [])
        if not domains and raw.get("domain"):
            domains = [str(raw.get("domain"))]
        entry: dict[str, Any] = {
            "round_id":          _coerce_round_id(raw.get("round_id")),
            "dispatched_at":     str(raw.get("dispatched_at") or ""),
            "completed_at":      str(raw.get("completed_at") or ""),
            "domains":           domains,
            "tags":              list(raw.get("tags") or []),
            "parallelism":       int(raw.get("parallelism") or 0),
            "proposals_total":   int(raw.get("proposals_total") or 0),
            "proposals_kept":    int(raw.get("proposals_kept") or 0),
            "proposals_rejected": int(raw.get("proposals_rejected") or 0),
            "proposals_skipped": int(raw.get("proposals_skipped") or 0),
            "kb_edge_ids":       list(raw.get("kb_edge_ids") or []),
            "confidence_avg":    _to_float(
                raw.get("confidence_avg")
                if raw.get("confidence_avg") is not None
                else raw.get("confidence")
            ),
            "domain_breakdown":  _normalize_specialist_domain_breakdown(
                raw.get("domain_breakdown"),
            ),
            "notes":             list(raw.get("notes") or []),
        }
        # Attach transcript refs from runs/specialist/. Tolerate the
        # singular ``task_id`` anchor when no ``task_ids`` list exists.
        task_ids = list(raw.get("task_ids") or [])
        if not task_ids and raw.get("task_id"):
            task_ids = [str(raw.get("task_id"))]
        transcripts: list[dict[str, Any]] = []
        for tid in task_ids:
            tid_str = str(tid)
            done_path = by_task.get(tid_str)
            if done_path is None:
                continue
            ref: dict[str, Any] = {
                "task_id": tid_str,
                "domain": _domain_for_task(raw, tid_str),
                "path": _rel(done_path, session_dir) or str(done_path),
            }
            if include_transcripts:
                try:
                    ref["body"] = done_path.read_text(
                        encoding="utf-8", errors="replace",
                    )
                except OSError as exc:
                    warnings.append(
                        f"specialist_runs: cannot read transcript "
                        f"{done_path}: {exc!r}"
                    )
            transcripts.append(ref)
        entry["transcripts"] = transcripts
        out.append(entry)
    return out


def _normalize_specialist_domain_breakdown(
    raw: Any,
) -> dict[str, dict[str, int]]:
    if not isinstance(raw, dict):
        return {}
    norm: dict[str, dict[str, int]] = {}
    for domain, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        norm[str(domain)] = {
            "dispatched":         int(payload.get("dispatched") or 0),
            "proposals_total":    int(payload.get("proposals_total") or 0),
            "proposals_kept":     int(payload.get("proposals_kept") or 0),
            "proposals_rejected": int(payload.get("proposals_rejected") or 0),
        }
    return norm


def _domain_for_task(round_entry: dict[str, Any], task_id: str) -> str:
    """Best-effort lookup of the domain associated with ``task_id``
    inside a single ``specialist_rounds`` entry. Returns "" when the
    round doesn't carry the mapping (older rounds packed in M5)."""
    mapping = round_entry.get("task_domains")
    if isinstance(mapping, dict):
        v = mapping.get(task_id)
        if isinstance(v, str):
            return v
    # Fallback: if the round has exactly one knowledge-domain tag (or
    # one legacy domain) we can attribute the task to it without
    # ambiguity.
    domains = round_entry.get("tags") or round_entry.get("domains") or []
    if isinstance(domains, list) and len(domains) == 1:
        return str(domains[0])
    if round_entry.get("domain"):
        return str(round_entry.get("domain"))
    return ""


__all__ = [
    "collect_attribution",
    "collect_baseline",
    "collect_capability_summary",
    "collect_critic_robustness",
    "collect_final",
    "collect_explore_search",
    "collect_kb_provenance",
    "collect_kernel_invocations",
    "collect_kernel_lifecycle",
    "collect_param_search",
    "collect_phase_segments",
    "collect_phase_timeline",
    "collect_session",
    "collect_source_files",
    "collect_specialist_runs",
    "collect_sweep",
    "collect_telemetry",
    "collect_workload",
]
