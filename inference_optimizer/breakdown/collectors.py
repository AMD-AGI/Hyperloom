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
import glob
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
    except OSError:
        pass
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


def _find_benchmark_report(workspace: Path | None) -> Path | None:
    """Locate the ``benchmark_*/benchmark_report.json`` under a task workspace.

    Returns the most recent one (by mtime) if multiple are present (e.g.
    retries within the same task), else ``None``.
    """
    if workspace is None or not workspace.exists():
        return None
    candidates: list[Path] = []
    for p in workspace.glob("benchmark_*/benchmark_report.json"):
        candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


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
    stop_reason = str(state.get("stop_reason") or "")
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
        "ended_at_utc":     _utc_now_iso() if stop_reason else "",
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
        candidates = sorted(
            (session_dir / "runs" / "baseline").glob(
                "*/benchmark_*/benchmark_report.json"
            ),
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
    """Walk ``<sd>/runs/baseline/<hash>/benchmark_*/benchmark_report.json``
    and synthesize :class:`BaselineAttemptSummary` rows for each.

    Used when ``state.baseline_attempts`` is empty but the on-disk
    runs/baseline/ tree shows that baseline ran (one or many times).
    Reads only what we can be certain of; everything else stays empty.
    """
    root = session_dir / "runs" / "baseline"
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    reports = sorted(
        root.glob("*/benchmark_*/benchmark_report.json"),
        key=lambda p: p.stat().st_mtime,
    )
    for report_path in reports:
        bench_dir = report_path.parent
        # task dir = <sd>/runs/baseline/<HASH> ; bench dir = <task>/benchmark_<ts>
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
        "extra_sglang_args":                 str(cb.get("extra_sglang_args") or ""),
        "extra_envs":                        dict(cb.get("extra_envs") or {}),
        "action_path":                       action_path,
        "ttft_mean_ms":                      ttft,
        "e2el_mean_ms":                      e2el,
        "ttft_e2el_source":                  ttft_e2el_source,
        "invocation":                        invocation,
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
    candidates = sorted(
        root.glob("*/benchmark_*/benchmark_report.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _find_stack_top_report(
    session_dir: Path, state: dict[str, Any],
) -> Path | None:
    """Last optimization_stack entry's benchmark_report.json (best-effort).

    Used when no validate_stack run was recorded — the topmost stack
    entry's KEEP run is the next-best evidence of "the final
    configuration's measured throughput / latency".
    """
    stack = state.get("optimization_stack") or []
    if not isinstance(stack, list) or not stack:
        return None
    last = stack[-1] if isinstance(stack[-1], dict) else None
    if last is None:
        return None
    workspace_str = last.get("workspace") or ""
    if not workspace_str:
        return None
    workspace = _resolve_under_session(session_dir, workspace_str)
    if workspace is None:
        return None
    return _find_benchmark_report(workspace)


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
_AUDIT_ACTIONS = (
    "baseline", "profile", "backends", "params", "sweep", "validate_stack",
)


def collect_phase_timeline(
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
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
                    "workspace":   None,
                    "error_class": None,
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
                events.append({
                    "ts":          a.get("ts") or "",
                    "action":      "integrate",
                    "task_id":     "",
                    "kernel_id":   str(ent.get("kernel_id") or ""),
                    "status":      str(a.get("status") or ""),
                    "decision":    str(a.get("decision") or ""),
                    "key_metric":  _to_float(a.get("gain_pct")),
                    "key_metric_kind": "gain_pct",
                    "workspace":   a.get("workspace"),
                    "error_class": None,
                    "extras":      {"patch_path": ent.get("patch_path"),
                                    "report_path": a.get("report_path")},
                })

    events.sort(key=lambda e: e.get("ts") or "")
    return events


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
    sessions that had a clear ``backends:vllm_kv_fp8`` final.action_path
    would still report ``backends: not_attempted`` in capability_summary.
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
    # with backends/params/sweep here.
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
    # GEAK / OOB driven by actual invocations on disk (most reliable)
    def _from_invocations(invs: list[dict[str, Any]]) -> dict[str, Any]:
        attempts = len(invs)
        keeps = sum(1 for v in invs if v.get("decision") == "KEEP")
        status = (
            "kept"          if keeps > 0 else
            "attempted"     if attempts > 0 else
            "not_attempted"
        )
        return {"status": status, "attempts": attempts, "keeps": keeps}

    geak_cap = _from_invocations(geak_invocations)
    oob_cap = _from_invocations(oob_invocations)

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

    validate = _capability_for_action(state, "validate_stack")
    validate["last_validated_gain_pct"] = _to_float(
        state.get("cumulative_gain_validated")
    )

    return {
        "geak":           geak_cap,
        "oob":            oob_cap,
        "backends":       backends,
        "params":         params,
        "sweep":          sweep_cap,
        "validate_stack": validate,
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
        for att in attempts:
            inv = _parse_invocation_attempt(att, run_dir, session_dir, warnings)
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
      1. ``state.last_select_kernels.candidates_path`` — orchestrator-recorded path
      2. ``session_dir / kernel-agent / runs / <session_id> / kernel_candidates.json``
         (new layout after the all-artefacts-under-USER_DATA_PATH migration)
      3. ``session_dir / kernel-agent / **/kernel_candidates.json`` glob fallback (new)
      4. ``session_dir / kernel-agent-workspace / kernel-agent / runs / hyperloom /
         kernel_candidates.json`` (legacy double-nested layout from pre-migration
         sessions, kept for breakdown replay of historical runs)
      5. ``session_dir / kernel-agent-workspace / **/kernel_candidates.json`` glob fallback
    """
    sk = state.get("last_select_kernels") or {}
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
            if isinstance(hk, list):
                return hk
    # Final fallback: state.last_select_kernels.hot_kernels_top15.
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
        for e in ((state.get("last_select_kernels") or {}).get("hot_kernels_top15") or [])
        if isinstance(e, dict)
    }
    geak_idx = _index_invocations_by_kernel(geak)
    oob_idx = _index_invocations_by_kernel(oob)

    integ = state.get("kernel_integrate_attempts") or {}
    adopted_kids: set[str] = set()
    reverted_kids: set[str] = set()
    if isinstance(integ, dict):
        for ent in integ.values():
            if not isinstance(ent, dict):
                continue
            kid = str(ent.get("kernel_id") or "")
            if not kid:
                continue
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
    sk = state.get("last_select_kernels") or {}
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
                "extra_sglang_args": str(ent.get("extra_sglang_args") or ""),
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
# §10 Param / backends search
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
            "extra_sglang_args": str(e.get("extra_sglang_args") or ""),
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


def collect_param_search(
    state: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    params_ledger = _shape_ledger(state.get("params_search"))
    params_ledger["winner_history"] = list(state.get("params_winner_history") or [])
    params_ledger["no_promote_streak"] = int(state.get("params_no_promote_streak") or 0)

    backends_ledger = _shape_ledger(state.get("backends_search"))

    baseline_tput = _to_float(state.get("baseline_tput"))
    return {
        "params":                  params_ledger,
        "backends":                backends_ledger,
        "synergy_attempted":       list(state.get("synergy_attempted") or []),
        "discovered_flags":        dict(state.get("discovered_flags") or {}),
        "backend_winners_history": _patch_winners_history(
            state.get("backend_winners_history") or [], baseline_tput,
        ),
    }


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

    return {
        "critic_iterations":  critic_iters,
        "robustness_signals": robustness_signals,
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
    }


# ---------------------------------------------------------------------------
# §14 Attribution
# ---------------------------------------------------------------------------
def _action_family(action: str) -> str:
    """Map an action label to a family for source_breakdown bucketing."""
    s = (action or "").lower()
    if s.startswith("kernel_opt") or s == "integrate":
        return "kernel"
    if s == "backends":
        return "backends"
    if s == "params":
        return "params"
    if s == "sweep":
        return "sweep"
    if s == "validate_stack":
        return "validate"
    return "other"


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
    if state_provided:
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
        "kernel": 0.0, "backends": 0.0, "params": 0.0,
        "sweep": 0.0, "validate": 0.0, "other": 0.0,
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

    return {
        "gain_per_stack_entry": entries,
        "method":               method,
        "source_breakdown": {
            "geak_pct_of_total":     round(geak_total, 2),
            "oob_pct_of_total":      round(oob_total, 2),
            "backends_pct_of_total": round(family_totals.get("backends", 0.0), 2),
            "params_pct_of_total":   round(family_totals.get("params", 0.0), 2),
            "sweep_pct_of_total":    round(family_totals.get("sweep", 0.0), 2),
            "validated_total_pct":   round(validated_total, 2),
        },
        "notes": notes,
    }


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
            "extra_sglang_args": str(entry.get("extra_sglang_args") or ""),
        })
        cum_before = cum_after
    return out


# ---------------------------------------------------------------------------
# source_files map
# ---------------------------------------------------------------------------
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


__all__ = [
    "collect_attribution",
    "collect_baseline",
    "collect_capability_summary",
    "collect_critic_robustness",
    "collect_final",
    "collect_kernel_invocations",
    "collect_kernel_lifecycle",
    "collect_param_search",
    "collect_phase_timeline",
    "collect_session",
    "collect_source_files",
    "collect_sweep",
    "collect_telemetry",
    "collect_workload",
]
