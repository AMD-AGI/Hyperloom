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
from datetime import datetime, timedelta, timezone
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
# sglang ≥0.4 emits a single ``server_args=ServerArgs(model_path=...,
# attention_backend='aiter', mem_fraction_static=0.68, ...)`` line during
# startup. The dataclass repr is balanced (``ServerArgs(...)``) but contains
# nested parens (e.g. ``cuda_graph_bs=[...]``, tuples) — to keep the regex
# robust we anchor on ``server_args=ServerArgs(`` and lazily capture up to
# the trailing ``)`` at end-of-line. Captured group is the dataclass body
# (no surrounding ``ServerArgs(...)``); collectors prepend ``ServerArgs(``
# back when rendering so the source is unambiguous.
_FRAMEWORK_ARGS_SERVERARGS_RE = re.compile(
    r"server_args=ServerArgs\((.+)\)\s*$",
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
        # sglang ≥0.4: ``[ts] server_args=ServerArgs(<body>)`` — the only
        # post-parse arg echo for this framework. We re-wrap the captured
        # body as ``ServerArgs(<body>)`` so consumers can tell at a glance
        # which dataclass these kwargs belong to.
        m = _FRAMEWORK_ARGS_SERVERARGS_RE.search(stripped)
        if m:
            return f"ServerArgs({m.group(1).strip()})", "log_args_line"

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

    NOTE: kept for backwards compatibility with internal callers. The
    richer multi-source probe used by :func:`collect_session` lives in
    :func:`_extract_image_info`; this helper now delegates to that
    function with ``state={}`` and ``session_dir=None`` so the two
    code paths can't drift.
    """
    info = _extract_image_info({}, manifest, None)
    if info.get("image"):
        return info["image"]
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
# §1 Session metadata — helper: image / id / digest extraction
# ---------------------------------------------------------------------------
# Multi-source probe for the container image fingerprint. Real sessions
# almost never expose a populated ``image`` field today (manifest writes
# ``None`` because the spawning runtime hasn't been wired to surface
# rocm/sglang:<tag>), and the historical env-var fallback only fires
# inside the running pod — never at sbd-export time on a separate host.
# We widen the probe to also look at the materialized baseline yaml
# (``runs/baseline/*/baseline_config.with_envs.yaml`` carries the same
# image when set) and the magpie benchmark config (``docker_image``
# field). When every source comes back empty we record None — consumers
# fall back to the ``data_provenance`` source list to see exactly which
# candidate paths were probed.
_IMAGE_ENV_VARS: tuple[str, ...] = (
    "HYPERLOOM_IMAGE", "CONTAINER_IMAGE", "IMAGE",
)


def _coerce_image_str(value: Any) -> str | None:
    if isinstance(value, str):
        s = value.strip()
        if s and s.lower() not in {"null", "none"}:
            return s
    return None


def _scan_yaml_for_image(path: Path) -> dict[str, str | None]:
    """Grep-style scan of a YAML file for image / image_id / image_digest.

    Avoids a PyYAML dependency (the rest of breakdown is yaml-free) and
    is robust to nesting since we only need top-level key:value matches
    of the form ``<key>: <value>``. Returns up to three keys; missing
    fields are None.
    """
    out: dict[str, str | None] = {"image": None, "image_id": None, "image_digest": None}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    # Match exactly the field names we care about. ``docker_image`` is
    # the magpie benchmark config alias for the container image.
    patterns = {
        "image":        re.compile(r"^\s*(?:docker_image|image)\s*:\s*(?P<v>.+)\s*$", re.MULTILINE),
        "image_id":     re.compile(r"^\s*image_id\s*:\s*(?P<v>.+)\s*$", re.MULTILINE),
        "image_digest": re.compile(r"^\s*image_digest\s*:\s*(?P<v>.+)\s*$", re.MULTILINE),
    }
    for key, regex in patterns.items():
        m = regex.search(text)
        if not m:
            continue
        raw = m.group("v").strip().strip('"').strip("'")
        coerced = _coerce_image_str(raw)
        if coerced and out[key] is None:
            out[key] = coerced
    return out


def _extract_image_info(
    state: dict[str, Any],
    manifest: dict[str, Any],
    session_dir: Path | None,
) -> dict[str, str | None]:
    """Multi-source resolver for image / image_id / image_digest.

    Priority (highest to lowest):

    1. ``state.json`` top-level (``image`` / ``container_image`` / ``image_id`` /
       ``image_digest``)
    2. ``manifest.json`` top-level (same field names; manifest also
       supports ``metadata.image`` for nested-shape sessions)
    3. ``runs/baseline/*/baseline_config.with_envs.yaml`` (magpie
       baseline config; carries ``docker_image`` and/or ``image``)
    4. ``runs/baseline/*/benchmark_*/config.yaml`` (magpie nested
       benchmark config, same field shape)
    5. Env vars ``HYPERLOOM_IMAGE`` / ``CONTAINER_IMAGE`` / ``IMAGE``
       (final fallback — only fires inside the running pod)

    Each field can be sourced independently — e.g. image comes from
    state but the digest only from manifest. None is returned for any
    field we never observed.
    """
    image: str | None = None
    image_id: str | None = None
    image_digest: str | None = None

    def _take_from(d: dict[str, Any], *, image_keys: Iterable[str], id_keys: Iterable[str], digest_keys: Iterable[str]) -> None:
        nonlocal image, image_id, image_digest
        if not isinstance(d, dict):
            return
        if image is None:
            for k in image_keys:
                v = _coerce_image_str(d.get(k))
                if v:
                    image = v
                    break
        if image_id is None:
            for k in id_keys:
                v = _coerce_image_str(d.get(k))
                if v:
                    image_id = v
                    break
        if image_digest is None:
            for k in digest_keys:
                v = _coerce_image_str(d.get(k))
                if v:
                    image_digest = v
                    break

    # Source 1: state.json top-level.
    _take_from(
        state if isinstance(state, dict) else {},
        image_keys=("image", "container_image"),
        id_keys=("image_id",),
        digest_keys=("image_digest",),
    )
    # Source 2: manifest.json top-level + manifest.metadata.image (nested
    # shape used by some older runners).
    _take_from(
        manifest if isinstance(manifest, dict) else {},
        image_keys=("image", "container_image"),
        id_keys=("image_id",),
        digest_keys=("image_digest",),
    )
    if image is None and isinstance(manifest, dict):
        meta = manifest.get("metadata")
        if isinstance(meta, dict):
            _take_from(
                meta,
                image_keys=("image", "container_image"),
                id_keys=("image_id",),
                digest_keys=("image_digest",),
            )

    # Sources 3 + 4: materialized baseline yamls. We only need to read
    # files when we're still missing one of the three fields — the
    # ``runs/baseline`` tree can be large and we should not pay the
    # disk cost gratuitously.
    needs_more = image is None or image_id is None or image_digest is None
    if needs_more and session_dir is not None and session_dir.exists():
        for pattern in (
            "runs/baseline/*/baseline_config.with_envs.yaml",
            "runs/baseline/*/benchmark_*/config.yaml",
        ):
            if not (image is None or image_id is None or image_digest is None):
                break
            try:
                for path in session_dir.glob(pattern):
                    found = _scan_yaml_for_image(path)
                    if image is None and found["image"]:
                        image = found["image"]
                    if image_id is None and found["image_id"]:
                        image_id = found["image_id"]
                    if image_digest is None and found["image_digest"]:
                        image_digest = found["image_digest"]
                    if image and image_id and image_digest:
                        break
            except OSError:
                continue

    # Source 5: env vars (last resort, only meaningful in-pod).
    if image is None:
        for var in _IMAGE_ENV_VARS:
            val = _coerce_image_str(os.environ.get(var))
            if val:
                image = val
                break

    return {"image": image, "image_id": image_id, "image_digest": image_digest}


# ---------------------------------------------------------------------------
# §1 Session metadata — helper: session lifecycle timestamps
# ---------------------------------------------------------------------------
def _parse_iso(ts: str | None) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _unix_to_iso(unix_ts: float | None) -> str | None:
    if unix_ts is None:
        return None
    try:
        f = float(unix_ts)
    except (TypeError, ValueError):
        return None
    if not f or f != f:  # zero or NaN
        return None
    try:
        return datetime.fromtimestamp(f, tz=timezone.utc).isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return None


def _extract_session_timing(
    state: dict[str, Any],
    manifest: dict[str, Any],
    phase_timeline: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Resolve the real session start / end timestamps + duration.

    started_at_utc preference:
      1. ``state.started_at`` (explicit field)
      2. ``state.start_ts`` (the conventional field today; ISO string)
      3. ``state.startup_ts``
      4. ``manifest.created_at_utc`` (proxy — captures spawn time, may
         predate the orchestrator's actual ``run`` entry by a few seconds)
      5. ``phase_timeline[0].ts`` (derived; only used when no real
         lifecycle timestamp exists in state)

    ended_at_utc preference:
      1. ``state.stopped_at`` (explicit shutdown timestamp)
      2. ``state.last_tick_ts``
      3. ``state.closing_started_unix`` (epoch float; recorded the
         moment the orchestrator transitioned into closing)
      4. ``max(phase_timeline[i].ended_ts_utc or ts)`` (derived; only
         used when no real shutdown timestamp exists)

    duration_seconds is the arithmetic difference when both timestamps
    are resolvable; it is rounded to 1s and is None when either is
    missing. The function does NOT raise — every source is best-effort
    and any unparseable input is silently skipped.
    """
    # Started.
    started_dt: datetime | None = None
    started_source: str | None = None
    for src, val in (
        ("state.started_at", state.get("started_at") if isinstance(state, dict) else None),
        ("state.start_ts",   state.get("start_ts")   if isinstance(state, dict) else None),
        ("state.startup_ts", state.get("startup_ts") if isinstance(state, dict) else None),
    ):
        dt = _parse_iso(val if isinstance(val, str) else None)
        if dt is not None:
            started_dt, started_source = dt, src
            break
    if started_dt is None and isinstance(manifest, dict):
        dt = _parse_iso(manifest.get("created_at_utc") if isinstance(manifest.get("created_at_utc"), str) else None)
        if dt is not None:
            started_dt, started_source = dt, "manifest.created_at_utc"
    if started_dt is None and phase_timeline:
        for evt in phase_timeline:
            if not isinstance(evt, dict):
                continue
            dt = _parse_iso(evt.get("ts") if isinstance(evt.get("ts"), str) else None)
            if dt is not None:
                started_dt, started_source = dt, "phase_timeline[0].ts (derived)"
                break

    # Ended.
    ended_dt: datetime | None = None
    ended_source: str | None = None
    for src, val in (
        ("state.stopped_at",   state.get("stopped_at")   if isinstance(state, dict) else None),
        ("state.last_tick_ts", state.get("last_tick_ts") if isinstance(state, dict) else None),
    ):
        dt = _parse_iso(val if isinstance(val, str) else None)
        if dt is not None:
            ended_dt, ended_source = dt, src
            break
    if ended_dt is None:
        unix = state.get("closing_started_unix") if isinstance(state, dict) else None
        try:
            unix_f = float(unix) if unix is not None else 0.0
        except (TypeError, ValueError):
            unix_f = 0.0
        if unix_f:
            try:
                ended_dt = datetime.fromtimestamp(unix_f, tz=timezone.utc)
                ended_source = "state.closing_started_unix"
            except (OSError, OverflowError, ValueError):
                ended_dt = None
    if ended_dt is None and phase_timeline:
        latest: datetime | None = None
        for evt in phase_timeline:
            if not isinstance(evt, dict):
                continue
            for key in ("ended_ts_utc", "ts"):
                dt = _parse_iso(evt.get(key) if isinstance(evt.get(key), str) else None)
                if dt is not None and (latest is None or dt > latest):
                    latest = dt
        if latest is not None:
            ended_dt, ended_source = latest, "phase_timeline.max(ended_ts_utc|ts) (derived)"

    # Use microsecond precision when the underlying source carries
    # sub-second detail (closing_started_unix typically does). This
    # keeps session_ended_at_utc >= the closing event's ts (which is
    # also serialized at microsecond precision), so the closing-event
    # duration back-fill in :func:`enrich_session_and_timeline` won't
    # be defeated by a 0.x-second truncation.
    started_iso = (
        started_dt.isoformat(timespec="microseconds")
        if started_dt and started_dt.microsecond
        else (started_dt.isoformat(timespec="seconds") if started_dt else None)
    )
    ended_iso = (
        ended_dt.isoformat(timespec="microseconds")
        if ended_dt and ended_dt.microsecond
        else (ended_dt.isoformat(timespec="seconds") if ended_dt else None)
    )
    duration: float | None = None
    if started_dt is not None and ended_dt is not None:
        diff = (ended_dt - started_dt).total_seconds()
        # Never emit a negative duration — would indicate clock skew or
        # a misparsed timestamp; surface as None so consumers don't
        # render nonsense.
        if diff >= 0:
            duration = round(diff, 1)
    return {
        "session_started_at_utc":   started_iso,
        "session_ended_at_utc":     ended_iso,
        "session_duration_seconds": duration,
        "_started_source":          started_source,
        "_ended_source":            ended_source,
    }


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
    elapsed_min_from_now: float | None = None
    if start_ts:
        try:
            start = datetime.fromisoformat(start_ts.replace("Z", "+00:00"))
            elapsed_min_from_now = (datetime.now(timezone.utc) - start).total_seconds() / 60.0
        except (ValueError, TypeError):
            pass
    image_info = _extract_image_info(state, manifest, session_dir)
    image = image_info["image"]
    if image is None:
        # Keep the legacy "image: not configured" prefix for backwards
        # compat with consumers (and tests) that grep on it. Append a
        # richer enumeration of the candidate sources we tried so an
        # operator can see at a glance that we DID probe state.json /
        # manifest.json / baseline yamls / env before giving up.
        warnings.append(
            "image: not configured — no image metadata in state.json / "
            "manifest.json / runs/baseline yamls / env "
            "(HYPERLOOM_IMAGE|CONTAINER_IMAGE|IMAGE)"
        )

    # Real session timing. We do NOT have phase_timeline at this point
    # in the exporter flow (it's collected after session_meta) — so we
    # rely on state.json + manifest.created_at_utc + closing_started_unix
    # here, and let the exporter back-fill the derived ``ended_ts_utc``
    # variant if it later wants to (today it doesn't; the state-based
    # ``closing_started_unix`` covers every real session we've observed
    # in /home/chenluo/sbd/v2).
    timing = _extract_session_timing(state, manifest, phase_timeline=None)
    session_started_at_utc = timing["session_started_at_utc"]
    session_ended_at_utc = timing["session_ended_at_utc"]
    session_duration_seconds = timing["session_duration_seconds"]

    # Prefer the derived elapsed_minutes when we have a real start+end
    # pair; otherwise fall back to ``now - start_ts`` (legacy behaviour).
    elapsed_min: float
    if session_duration_seconds is not None:
        elapsed_min = round(session_duration_seconds / 60.0, 2)
    elif elapsed_min_from_now is not None:
        elapsed_min = round(elapsed_min_from_now, 2)
    else:
        elapsed_min = 0.0

    return {
        "session_id":       str(state.get("session_id") or manifest.get("session_id") or ""),
        "claw_session_id":  manifest.get("claw_session_id") or state.get("claw_session_id"),
        "sandbox_user_id":  manifest.get("sandbox_user_id") or state.get("sandbox_user_id"),
        "created_at_utc":   manifest.get("created_at_utc") or start_ts,
        # Legacy ``ended_at_utc`` (= dump time when stop_reason is set)
        # is preserved verbatim for backwards compatibility. Consumers
        # that want the real session end should read
        # ``session_ended_at_utc`` instead.
        "ended_at_utc":     _utc_now_iso() if stop_reason else "",
        "stop_reason":      stop_reason,
        "max_minutes":      int(state.get("max_minutes") or manifest.get("max_minutes") or 0),
        "elapsed_minutes":  elapsed_min,
        "host":             str(manifest.get("host") or ""),
        "image":            image,
        "image_id":         image_info["image_id"],
        "image_digest":     image_info["image_digest"],
        "session_started_at_utc":   session_started_at_utc,
        "session_ended_at_utc":     session_ended_at_utc,
        "session_duration_seconds": session_duration_seconds,
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
def _build_invocation_for_workspace(
    session_dir: Path,
    workspace_str: str | None,
    warnings: list[str],
) -> dict[str, Any]:
    """Best-effort BenchmarkInvocation for an arbitrary task workspace."""
    workspace = _resolve_under_session(session_dir, workspace_str) if workspace_str else None
    if workspace is None:
        return {
            "framework_args": "",
            "framework_args_source": "unknown",
            "extra_envs": {},
            "config_path": None,
            "server_log_path": None,
        }
    report_path = _find_benchmark_report(workspace)
    server_log_path: Path | None = None
    if report_path is not None:
        candidate_log = report_path.parent / "server.log"
        if candidate_log.exists():
            server_log_path = candidate_log
    if server_log_path is None:
        bench_dirs = sorted(
            workspace.glob("benchmark_*/server.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if bench_dirs:
            server_log_path = bench_dirs[0]
    config_resolved: Path | None = None
    for candidate in (
        workspace / "baseline_config.with_envs.yaml",
        workspace / "params_base.with_envs.yaml",
        workspace.parent / "baseline_config.with_envs.yaml",
    ):
        if candidate.exists():
            config_resolved = candidate
            break
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
    return {
        "framework_args":        args_str,
        "framework_args_source": args_source,
        "extra_envs":            _read_invocation_envs(config_resolved),
        "config_path":           _rel(config_resolved, session_dir) if config_resolved else None,
        "server_log_path":       _rel(server_log_path, session_dir) if server_log_path else None,
    }


def _load_workload_dims(session_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_json_safe(session_dir / "manifest.json", []) or {}
    wl = manifest.get("workload") or {}
    return {
        "conc":      _to_int(wl.get("conc")),
        "isl":       _to_int(wl.get("isl")),
        "osl":       _to_int(wl.get("osl")),
        "tp":        _to_int(manifest.get("tp")),
        "precision": str(wl.get("precision") or ""),
    }


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
            "invocation":    _build_invocation_for_workspace(
                session_dir, a.get("workspace"), warnings,
            ),
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
        # Case A: workspace is the task dir → walk down into benchmark_*/.
        bench_dirs = sorted(
            workspace.glob("benchmark_*/server.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if bench_dirs:
            server_log_path = bench_dirs[0]
        else:
            # Case B: state.last_baseline.workspace points at the leaf
            # ``benchmark_*`` dir itself (recent orchestrator versions do
            # this). ``workspace/server.log`` is then the direct sibling
            # of the benchmark_report.json.
            direct_log = workspace / "server.log"
            if direct_log.exists():
                server_log_path = direct_log

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

    # Session-wide disk walk fallback. When state.last_baseline.workspace
    # didn't resolve (or pointed at a clean-up'd location) AND we have no
    # report_path, we can still find baseline_config.with_envs.yaml and
    # the most recent server.log directly under runs/baseline/. This
    # covers two real-world cases observed in v2/:
    #
    #   * baseline succeeded but state.last_baseline carries a stale
    #     ``/workspace/...`` path whose anchors don't match runs/.
    #   * baseline_report.json's metrics dict is empty (ttft=None), so
    #     the disk-walk further up didn't get to update report_path.
    #
    # We deliberately do NOT touch ttft/e2el here — that path already
    # ran its own walk against the same tree. We only rescue the
    # invocation extraction so framework_args_source ends up at
    # ``yaml_benchmark`` (or ``log_args_line``) instead of ``unknown``.
    if server_log_path is None or config_resolved is None:
        baseline_root = session_dir / "runs" / "baseline"
        if baseline_root.exists():
            if server_log_path is None:
                disk_logs = sorted(
                    baseline_root.glob("*/benchmark_*/server.log"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if disk_logs:
                    server_log_path = disk_logs[0]
            if config_resolved is None:
                disk_yamls = sorted(
                    baseline_root.glob("*/baseline_config.with_envs.yaml"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if disk_yamls:
                    config_resolved = disk_yamls[0]

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
        "workload_dims":            _load_workload_dims(session_dir, state),
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


def _phase_event_duration(
    session_dir: Path | None,
    entry: dict[str, Any],
    warnings: list[str],
) -> float | None:
    """Best-effort wall-clock duration for an audit-attempt event.

    Source priority:

    1. ``extras.duration_seconds`` (already recorded by the audit).
    2. ``benchmark_report.json`` under the entry's workspace.
    3. None — keep callers honest about unknown durations.
    """
    extras = entry.get("extras") if isinstance(entry.get("extras"), dict) else {}
    cand = _to_float(extras.get("duration_seconds")) if extras else None
    if cand is not None:
        return cand
    if session_dir is None:
        return None
    workspace_str = entry.get("workspace")
    if not workspace_str:
        return None
    workspace = _resolve_under_session(session_dir, str(workspace_str))
    if workspace is None:
        return None
    # The latest benchmark_*/benchmark_report.json under workspace —
    # _find_benchmark_report only matches direct ``benchmark_*`` children,
    # so we also rglob one level deeper to catch ``variant_*/benchmark_*``
    # layouts (params/backends round dirs).
    report = _find_benchmark_report(workspace)
    if report is None:
        candidates = sorted(
            workspace.rglob("benchmark_report.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        report = candidates[0] if candidates else None
    if report is None:
        return None
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    cand = data.get("duration_seconds")
    if cand is None and isinstance(data.get("result"), dict):
        cand = data["result"].get("duration_seconds")
    if cand is None:
        cand = data.get("execution_time")
    return _to_float(cand)


def _add_seconds_iso(ts: str, seconds: float | None) -> str | None:
    """Compute ``ts + seconds`` in iso8601 UTC; return None if either is missing."""
    if not ts or seconds is None:
        return None
    try:
        # ``fromisoformat`` accepts the ``+00:00`` suffix that hyperloom
        # writes; if the ts is malformed we fail closed (None).
        base = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    try:
        ended = base + timedelta(seconds=float(seconds))
    except (ValueError, OverflowError):
        return None
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=timezone.utc)
    return ended.astimezone(timezone.utc).isoformat(timespec="seconds")


def collect_phase_timeline(
    state: dict[str, Any],
    warnings: list[str],
    session_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Chronological per-action event list.

    ``session_dir`` is optional for v1 callers; v1.1 adds it so each
    event can be enriched with ``duration_seconds`` (read from the
    workspace's ``benchmark_report.json``) and the derived
    ``ended_ts_utc``. Pass ``None`` to skip enrichment.
    """
    events: list[dict[str, Any]] = []
    for action in _AUDIT_ACTIONS:
        attempts = state.get(f"{action}_attempts") or []
        if not isinstance(attempts, list):
            continue
        for entry in attempts:
            if not isinstance(entry, dict):
                continue
            duration = _phase_event_duration(session_dir, entry, warnings)
            ts = entry.get("ts") or ""
            events.append({
                "ts":             ts,
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
                "duration_seconds": duration,
                "ended_ts_utc":   _add_seconds_iso(ts, duration),
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
                    "duration_seconds": None,
                    "ended_ts_utc": None,
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
                    "duration_seconds": None,
                    "ended_ts_utc": None,
                })

    # TraceLens analysis runs surface as their own timeline events so
    # the kernel-profiling phase shows up alongside baseline / params /
    # backends. The ts comes from the kernel-agent run dir mtime when
    # there's no recorded session_state.json timestamp; we accept any
    # is-better-than-nothing source here because TraceLens itself
    # doesn't write an audit attempt.
    if session_dir is not None:
        for run_dir in _kernel_agent_run_dirs(session_dir):
            status_root = run_dir / "status" / "tracelens_analysis"
            if not status_root.exists():
                continue
            for status_path in sorted(status_root.glob("*.json")):
                try:
                    payload = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = {}
                ts = ""
                if isinstance(payload, dict):
                    ts = str(
                        payload.get("ended_at")
                        or payload.get("ts")
                        or payload.get("started_at")
                        or ""
                    )
                if not ts:
                    try:
                        ts = datetime.fromtimestamp(
                            status_path.stat().st_mtime, tz=timezone.utc,
                        ).isoformat(timespec="seconds")
                    except OSError:
                        ts = ""
                # P2-3: read started_at / ended_at / duration_seconds
                # from the status JSON when the kernel-agent writer
                # populates them; historical sessions won't have these
                # three keys, so all three stay None and the timeline
                # event remains zero-duration (which is honest, not a
                # bug).
                started_at = ""
                ended_at = ""
                duration_seconds: float | None = None
                if isinstance(payload, dict):
                    started_at = str(payload.get("started_at") or "")
                    ended_at = str(payload.get("ended_at") or "")
                    duration_seconds = _to_float(payload.get("duration_seconds"))
                evt_ts = started_at or ts
                events.append({
                    "ts":          evt_ts,
                    "action":      "tracelens_analysis",
                    "task_id":     run_dir.name,
                    "kernel_id":   None,
                    "status":      str((payload or {}).get("status") or "") if isinstance(payload, dict) else "",
                    "decision":    "",
                    "key_metric":  None,
                    "key_metric_kind": None,
                    "workspace":   _rel(run_dir, session_dir),
                    "error_class": None,
                    "extras":      {"status_json": _rel(status_path, session_dir)},
                    "duration_seconds": duration_seconds,
                    "ended_ts_utc": ended_at or _add_seconds_iso(evt_ts, duration_seconds),
                })

    # P2-2: closing phase events. Three input shapes — pick whichever
    # the orchestrator wrote (newer sessions may emit a
    # ``closing_attempts`` list; historical sessions only set
    # ``closing_started_unix`` + ``closing_report_task_id``).
    closing_attempts = state.get("closing_attempts")
    closing_events_added = 0
    if isinstance(closing_attempts, list):
        for entry in closing_attempts:
            if not isinstance(entry, dict):
                continue
            ts = str(entry.get("ts") or "")
            duration = _phase_event_duration(session_dir, entry, warnings)
            events.append({
                "ts":             ts,
                "action":         "closing",
                "task_id":        str(entry.get("task_id") or ""),
                "kernel_id":      None,
                "status":         str(entry.get("status") or ""),
                "decision":       str(entry.get("decision") or ""),
                "key_metric":     _to_float(entry.get("key_metric")),
                "key_metric_kind": entry.get("key_metric_kind"),
                "workspace":      entry.get("workspace"),
                "error_class":    entry.get("error_class"),
                "extras":         dict(entry.get("extras") or {}),
                "duration_seconds": duration,
                "ended_ts_utc":   _add_seconds_iso(ts, duration),
            })
            closing_events_added += 1
    # Synthesize one closing event from final-state breadcrumbs when no
    # attempts list exists but the orchestrator did enter the closing
    # phase (``closing_started_unix`` set OR ``closing_report_task_id``
    # set OR ``final.closing_phase_entered`` true via collect_final
    # contract — we read state directly so we don't depend on the
    # invocation order of collectors).
    final_state = state.get("final") if isinstance(state.get("final"), dict) else {}
    closing_started_unix = _to_float(
        state.get("closing_started_unix") or final_state.get("closing_started_unix")
    )
    closing_task_id = str(
        state.get("closing_report_task_id")
        or final_state.get("closing_report_task_id")
        or ""
    )
    closing_phase_entered_flag = bool(
        state.get("closing_phase")
        or final_state.get("closing_phase_entered")
        or closing_started_unix
        or closing_task_id
    )
    final_entered_at_utc = str(final_state.get("entered_at_utc") or "")
    if closing_events_added == 0 and closing_phase_entered_flag:
        # Resolve closing start ts. Priority:
        # 1) ``final.entered_at_utc`` (explicit string)
        # 2) ``closing_started_unix`` (epoch float)
        ts = final_entered_at_utc
        if not ts and closing_started_unix:
            try:
                # Use microsecond precision so the synthesized duration
                # against the latest audit-attempt ts isn't off by up
                # to one second (the secs-only iso truncation can
                # land slightly before the real closing start when the
                # unix ts has a sub-second tail, which would make
                # ``latest_attempt - start`` artificially positive).
                ts = datetime.fromtimestamp(
                    closing_started_unix, tz=timezone.utc,
                ).isoformat(timespec="microseconds")
            except (OSError, OverflowError, ValueError):
                ts = ""
        # Best-effort duration: latest *audit-attempt* ts minus closing
        # start. We only count audit attempts that occurred at or after
        # the closing start; nothing before counts as part of closing.
        duration: float | None = None
        if ts:
            try:
                start_dt = datetime.fromisoformat(ts)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                latest_dt = None
                for action in _AUDIT_ACTIONS:
                    for entry in state.get(f"{action}_attempts") or []:
                        if not isinstance(entry, dict):
                            continue
                        e_ts = str(entry.get("ts") or "")
                        if not e_ts:
                            continue
                        try:
                            dt = datetime.fromisoformat(e_ts)
                        except (ValueError, TypeError):
                            continue
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if dt >= start_dt and (latest_dt is None or dt > latest_dt):
                            latest_dt = dt
                if latest_dt is not None:
                    duration = max(0.0, (latest_dt - start_dt).total_seconds())
            except (ValueError, TypeError):
                duration = None
        events.append({
            "ts":               ts,
            "action":           "closing",
            "task_id":          closing_task_id,
            "kernel_id":        None,
            "status":           "",
            "decision":         "entered" if closing_phase_entered_flag else "",
            "key_metric":       None,
            "key_metric_kind":  None,
            "workspace":        None,
            "error_class":      None,
            "extras": {
                "synthesized": True,
                "closing_started_unix": closing_started_unix,
                "closing_phase_flag": bool(state.get("closing_phase")),
            },
            "duration_seconds": duration,
            "ended_ts_utc":     _add_seconds_iso(ts, duration),
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
        "status":          str(attempt.get("status") or ""),
        # compile_passed / correctness_passed are kernel-level (in verification.json);
        # stamped later if this attempt is the BEST one for the kernel.
        "compile_passed":  None,
        "correctness_passed": None,
        "best_artifact_path": None,
        "proposal_reasons": [],
        "verification_summary": {},
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
        proposal = result.get("proposal") if isinstance(result, dict) else None
        if isinstance(proposal, dict):
            reasons = proposal.get("reasons")
            if isinstance(reasons, list):
                best["proposal_reasons"] = [str(r) for r in reasons if r]
        if isinstance(verification, dict):
            best["verification_summary"] = {
                "micro_speedup": _to_float(verification.get("micro_speedup")),
                "compile_passed": verification.get("compile_passed"),
                "correctness_passed": verification.get("correctness_passed"),
                "best_artifact_path": verification.get("best_artifact_path"),
            }
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
      1. ``state.last_trace_analyze.candidates_path`` — orchestrator-recorded path
      2. ``session_dir / kernel-agent / runs / <session_id> / kernel_candidates.json``
         (new layout after the all-artefacts-under-USER_DATA_PATH migration)
      3. ``session_dir / kernel-agent / **/kernel_candidates.json`` glob fallback (new)
      4. ``session_dir / kernel-agent-workspace / kernel-agent / runs / hyperloom /
         kernel_candidates.json`` (legacy double-nested layout from pre-migration
         sessions, kept for breakdown replay of historical runs)
      5. ``session_dir / kernel-agent-workspace / **/kernel_candidates.json`` glob fallback
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
            if isinstance(hk, list):
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
        # P2-4: pass through the structured shape info so the
        # downstream roofline merge can match by (name, input_dims)
        # instead of by name alone. ``input_shapes`` is the rich form
        # (list of {call_num, shape}), ``shapes`` is the flat list —
        # we keep both since the merge helper normalizes each into
        # the same canonical tuple.
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
            "input_shapes":            k.get("input_shapes") or k.get("shapes") or None,
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


_INPUT_DIMS_INT_RE = re.compile(r"\d+")


def _extract_parenthesised_ints(text: str) -> tuple[int, ...] | None:
    """Return the integers from the first parenthesised group in ``text``.

    Used to parse one tensor's shape entry like ``"(15360,6144) bf16"``
    — we keep only digits inside the outermost ``()`` so a trailing
    dtype token (``bf16``, ``c10::Half``) doesn't pollute the result.
    Falls back to ``None`` when no parenthesised group is present.
    """
    start = text.find("(")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return None
    inner = text[start + 1:end]
    nums = _INPUT_DIMS_INT_RE.findall(inner)
    if not nums:
        return None
    return tuple(int(x) for x in nums)


def _normalize_input_dims(value: Any) -> tuple[tuple[int, ...], ...] | None:
    """Reduce a free-form input-dims field to a canonical tuple of tuples.

    Two equivalent encodings hit this collector:

    * TraceLens ``Input Dims`` strings — e.g.
      ``"((15360, 6144), (6144, 43008))"``. The exact bracketing /
      spacing varies between TraceLens versions; the only stable
      signal is the sequence of integers grouped by the inner parens.
    * kernel_candidates.json ``shapes`` lists — e.g.
      ``["(15360,21504) bf16", "(21504,6144) bf16"]``. Each entry is
      one tensor's shape with a trailing dtype token we explicitly
      strip (otherwise ``bf16``'s digit ``16`` would be captured as a
      bogus dim).

    We normalize both to ``((15360, 21504), (21504, 6144))`` so that
    detected kernels and TraceLens rows for the same operation can be
    matched by ``(name, input_dims)``.

    Returns ``None`` when the input has no recognisable shape data
    (caller should fall back to name-only matching).
    """
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        groups: list[tuple[int, ...]] = []
        for item in value:
            if isinstance(item, (tuple, list)):
                ints = [int(x) for x in item if isinstance(x, (int, float))]
                if ints:
                    groups.append(tuple(ints))
                continue
            if isinstance(item, dict):
                shape_str = str(item.get("shape") or "")
                ints = _extract_parenthesised_ints(shape_str)
                if ints is None:
                    # No parens at all → take all the digits we find.
                    nums = _INPUT_DIMS_INT_RE.findall(shape_str)
                    if nums:
                        ints = tuple(int(x) for x in nums)
                if ints:
                    groups.append(tuple(ints))
                continue
            if isinstance(item, str):
                ints = _extract_parenthesised_ints(item)
                if ints is None:
                    nums = _INPUT_DIMS_INT_RE.findall(item)
                    if nums:
                        ints = tuple(int(x) for x in nums)
                if ints:
                    groups.append(tuple(ints))
                continue
        return tuple(groups) if groups else None
    if isinstance(value, str):
        # Walk the parenthesis structure to recover per-tensor groups.
        # Fall back to a single flat tuple of all ints when the string
        # has no nested grouping (e.g. ``"(15360, 6144)"``).
        groups: list[tuple[int, ...]] = []
        depth = 0
        buf: list[str] = []
        captured_any = False
        for ch in value:
            if ch == "(":
                depth += 1
                if depth >= 2:
                    buf.append(ch)
                continue
            if ch == ")":
                if depth >= 2:
                    buf.append(ch)
                depth -= 1
                if depth == 1 and buf:
                    ints = _INPUT_DIMS_INT_RE.findall("".join(buf))
                    if ints:
                        groups.append(tuple(int(x) for x in ints))
                        captured_any = True
                    buf = []
                continue
            if depth >= 2:
                buf.append(ch)
        if captured_any:
            return tuple(groups)
        # Single-level fallback: ``"(15360, 6144)"`` → ((15360, 6144),)
        ints = _extract_parenthesised_ints(value)
        if ints is None:
            nums = _INPUT_DIMS_INT_RE.findall(value)
            if nums:
                ints = tuple(int(x) for x in nums)
        if ints:
            return (tuple(ints),)
        return None
    return None


def _merge_roofline_into_detected(
    session_dir: Path,
    detected: list[dict[str, Any]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Annotate detected kernels with TraceLens roofline fields when available.

    Match strategy (P2-4) — keyed by ``(name, input_dims_tuple)`` so
    multi-shape ops (e.g. several ``aten::mm`` rows with different
    contractions) each pick up their own roofline. When either side
    is missing ``input_dims`` we fall back to name-only matching, and
    finally to the legacy prefix-match (detected name is a strict
    prefix of a TraceLens op name).
    """
    if not detected:
        return detected
    tl_rows: list[dict[str, Any]] = []
    for run_dir in _kernel_agent_run_dirs(session_dir):
        tracelens_dir = run_dir / "tracelens"
        if not tracelens_dir.exists():
            continue
        cat_rows = _read_tracelens_category_data(tracelens_dir, warnings)
        if cat_rows:
            tl_rows.extend(cat_rows)
            continue
        prio_rows, _summary = _read_tracelens_priority_data(tracelens_dir, warnings)
        tl_rows.extend(prio_rows)
    if not tl_rows:
        return detected
    by_key: dict[tuple[str, tuple[tuple[int, ...], ...]], dict[str, Any]] = {}
    by_name_only: dict[str, dict[str, Any]] = {}
    for row in tl_rows:
        name = str(row.get("name") or "")
        if not name:
            continue
        dims = _normalize_input_dims(row.get("input_dims"))
        if dims is not None:
            by_key.setdefault((name, dims), row)
        # First row per name wins for the name-only fallback. Category
        # data is sorted desc by impact so this is the highest-impact
        # shape — matches the previous v1.1 behaviour.
        by_name_only.setdefault(name, row)
    out: list[dict[str, Any]] = []
    for entry in detected:
        merged = dict(entry)
        # P2-4: derive normalized input_dims from whichever shape field
        # the detected row carries. Surface it under ``extras.input_dims``
        # so consumers can render per-shape rows. We try multiple
        # candidate fields because kernel_candidates uses ``input_shapes``
        # / ``shapes`` while benchmark_report fallback rows often have
        # nothing structured.
        dims_raw = (
            entry.get("input_dims")
            or entry.get("Input Dims")
            or entry.get("input_shapes")
            or entry.get("shapes")
        )
        dims = _normalize_input_dims(dims_raw)
        if dims is not None:
            extras = dict(merged.get("extras") or {})
            extras.setdefault("input_dims", [list(g) for g in dims])
            merged["extras"] = extras
        name = str(entry.get("name") or "")
        target: dict[str, Any] | None = None
        if name and dims is not None:
            target = by_key.get((name, dims))
        if target is None and name:
            target = by_name_only.get(name)
        if target is None and name:
            # Legacy prefix match — kept so existing behaviour and
            # tests covering it (test_kernel_lifecycle_detected_inherits_roofline)
            # continue to work.
            for tl_name, tl_row in by_name_only.items():
                if tl_name.startswith(name):
                    target = tl_row
                    break
        if target is not None:
            for key in (
                "efficiency_percent",
                "bound_type",
                "tflops_achieved",
                "flops_per_byte",
                "library",
            ):
                if merged.get(key) in (None, "") and target.get(key) is not None:
                    merged[key] = target.get(key)
            if merged.get("arithmetic_intensity") is None and target.get("flops_per_byte") is not None:
                merged["arithmetic_intensity"] = target.get("flops_per_byte")
        out.append(merged)
    return out


def collect_kernel_lifecycle(
    session_dir: Path,
    state: dict[str, Any],
    geak: list[dict[str, Any]],
    oob: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    detected = _collect_detected_kernels(session_dir, state, geak, oob, warnings)
    detected = _merge_roofline_into_detected(session_dir, detected, warnings)
    return {
        "detected":    detected,
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


def _promote_legacy_gain_entries(
    state_entries: list[Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Lift a pre-v0.7 ``list[float | None]`` gain ledger into the V1 schema.

    State written by older Coordinator versions stored per-entry
    ``cum_gain_after`` floats only. Cross-reference the parallel
    ``state.optimization_stack`` to recover action / variant_name / ts /
    extra_sglang_args, and compute ``delta_pct`` as the diff against the
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
            "extra_sglang_args": str(
                se.get("extra_sglang_args")
                or se.get("candidate_extra_sglang_args")
                or ""
            ),
        })
        if cum_after is not None:
            prev_cum = cum_after
    return out


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
        # Pre-v0.7 state: bare numeric ledger. Promote into V1 schema
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
    elif promoted_from_legacy:
        notes.append(
            "gain_per_stack_entry was a pre-v0.7 numeric ledger; "
            "promoted to V1 StackGainEntry shape using parallel data from "
            "optimization_stack (delta_pct computed as diff vs prior entry's "
            "cum_gain_after)."
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


# ---------------------------------------------------------------------------
# v1.1 — decision journal + kernel profiling
# ---------------------------------------------------------------------------
_DECISION_JOURNAL_STANDARD_VARIANT_CAP = 30


def _search_entry_fp(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    fp = entry.get("fingerprint")
    if fp:
        return str(fp)
    args = str(entry.get("extra_sglang_args") or "")
    envs = dict(entry.get("extra_envs") or {})
    return _params_entry_fp({"extra_sglang_args": args, "extra_envs": envs})


def _find_audit_attempt(
    state: dict[str, Any],
    action: str,
    *,
    round_id: str | None,
    ts: str | None,
) -> dict[str, Any] | None:
    attempts = state.get(f"{action}_attempts") or []
    if not isinstance(attempts, list):
        return None
    for entry in reversed(attempts):
        if not isinstance(entry, dict):
            continue
        extras = entry.get("extras") or {}
        if round_id and extras.get("round_id") == round_id:
            return entry
    if ts:
        for entry in reversed(attempts):
            if isinstance(entry, dict) and entry.get("ts") == ts:
                return entry
    return attempts[-1] if attempts else None


def _variant_benchmark_report_path(
    session_dir: Path,
    workspace_str: str | None,
    variant_name: str | None,
    warnings: list[str],
) -> str | None:
    """Locate ``benchmark_report.json`` for a variant.

    The ``workspace_str`` we receive can take any of three on-disk shapes:

    1. ``runs/{params,backends}/<round-task>`` — the round directory, which
       contains one ``variant_*<name>*`` subdirectory per variant.
    2. ``runs/{params,backends}/<round-task>/variant_*<name>*/benchmark_*``
       — the variant's own benchmark dir (the report sits beside it).
    3. ``runs/{params,backends}/<round-task>/combo`` — combo round root,
       containing one ``variant_*combo*`` subdirectory.

    We try the cheapest direct match first, then progressively widen the
    search. The variant_name pattern is tolerant of slight slug renames
    (``variant_NN_<name>`` is the canonical layout, but ``combo/`` adds a
    ``combo_`` prefix that wouldn't fit a literal ``f"*{name}*"`` glob).
    """
    workspace = _resolve_under_session(session_dir, workspace_str) if workspace_str else None
    if workspace is None:
        return None
    # Case 2: workspace already IS a benchmark dir.
    direct = workspace / "benchmark_report.json"
    if direct.exists():
        return _rel(direct, session_dir)
    # Case 1: workspace is the round dir; glob into variant subdirs.
    if variant_name:
        # Combo runs nest one level deeper: ``combo/variant_*combo_<name>*/...``.
        # Try non-combo first (cheaper), then combo, then any variant dir.
        for pattern in (
            f"variant_*{variant_name}*/benchmark_*/benchmark_report.json",
            f"*{variant_name}*/benchmark_*/benchmark_report.json",
            f"combo/variant_*{variant_name}*/benchmark_*/benchmark_report.json",
            f"combo/*{variant_name}*/benchmark_*/benchmark_report.json",
        ):
            try:
                matches = sorted(
                    workspace.glob(pattern),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except OSError:
                continue
            if matches:
                return _rel(matches[0], session_dir)
    # Last resort: a single benchmark_*/benchmark_report.json directly under
    # workspace (covers task-root workspaces with no variant subfolder).
    report = _find_benchmark_report(workspace)
    return _rel(report, session_dir) if report else None


def _duration_from_report(report_rel: str | None, session_dir: Path) -> float | None:
    """Read a wall-clock duration from a benchmark_report.json (relative path).

    Tries (in priority order):

    * ``duration_seconds`` (V2 schema; the workload's serve duration).
    * ``result.duration_seconds`` (pre-V2 nested form).
    * ``execution_time`` (older benchmark_runner schema; total runner
      wall-clock including server bring-up). Recorded as a last resort
      since it overshoots the pure workload duration by ~minutes, but
      it's still more useful than None for timeline visualization.

    Returns None when none of the fields are present or the file can't
    be read.
    """
    if not report_rel:
        return None
    p = session_dir / report_rel
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    cand = data.get("duration_seconds")
    if cand is None and isinstance(data.get("result"), dict):
        cand = data["result"].get("duration_seconds")
    if cand is None:
        cand = data.get("execution_time")
    return _to_float(cand)


def _shape_variant_decision(
    *,
    session_dir: Path,
    name: str,
    fingerprint: str,
    extra_sglang_args: str,
    extra_envs: dict[str, Any] | None,
    status: str,
    output_throughput: float | None,
    gain_pct_vs_base: float | None,
    outcome: str,
    reject_reason: str | None,
    workspace_str: str | None,
    warnings: list[str],
    decision_note: str | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    envs = _filter_envs(extra_envs or {})
    report_rel = _variant_benchmark_report_path(
        session_dir, workspace_str, name, warnings,
    )
    # Fall back to scraping the report when a duration wasn't supplied
    # by the caller (state-recorded ``result.duration_seconds`` is not
    # always present, but the report file usually has it).
    if duration_seconds is None:
        duration_seconds = _duration_from_report(report_rel, session_dir)
    return {
        "name":                    str(name or ""),
        "fingerprint":             str(fingerprint or ""),
        "extra_sglang_args":       str(extra_sglang_args or ""),
        "extra_envs":              envs,
        "status":                  str(status or ""),
        "output_throughput":       output_throughput,
        "gain_pct_vs_base":        gain_pct_vs_base,
        "gain_pct_vs_current_best": None,
        "outcome":                 outcome,
        "reject_reason":           reject_reason,
        "benchmark_report_path":   report_rel,
        "invocation":              _build_invocation_for_workspace(
            session_dir, workspace_str, warnings,
        ),
        "duration_seconds":        duration_seconds,
        "decision_note":           decision_note,
    }


def _variants_from_search_last_round(
    session_dir: Path,
    phase: str,
    search: dict[str, Any],
    *,
    workspace_str: str | None,
    selected_new: set[str],
    round_winners: set[str],
    warnings: list[str],
    winner_meta_by_fp: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    lr = search.get("last_round") or {}
    if not isinstance(lr, dict) or not lr:
        return []
    tested = search.get("tested") or {}
    if not isinstance(tested, dict):
        tested = {}
    rejected_by_fp: dict[str, dict[str, Any]] = {}
    for row in search.get("rejected") or []:
        if isinstance(row, dict):
            fp = _search_entry_fp(row)
            if fp:
                rejected_by_fp[fp] = row
    fps = lr.get("tested_fp")
    if not isinstance(fps, list) or not fps:
        fps = list(tested.keys())
    winner_meta_by_fp = winner_meta_by_fp or {}
    variants: list[dict[str, Any]] = []
    seen_fps: set[str] = set()
    for fp in fps:
        fp = str(fp)
        if not fp or fp in seen_fps:
            continue
        seen_fps.add(fp)
        entry = tested.get(fp) if isinstance(tested.get(fp), dict) else {}
        name = str(entry.get("name") or rejected_by_fp.get(fp, {}).get("name") or "")
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        gain = _to_float(entry.get("gain_pct"))
        if gain is None:
            gain = _to_float(rejected_by_fp.get(fp, {}).get("gain_pct"))
        # winners_history sometimes records gain when the search ledger
        # left it null (the winners mirror is published after the gain
        # was computed for the round-decision audit).
        if gain is None and fp in winner_meta_by_fp:
            gain = _to_float(winner_meta_by_fp[fp].get("gain_pct"))
        tput = _to_float(
            entry.get("output_throughput")
            or result.get("output_throughput")
            or rejected_by_fp.get(fp, {}).get("tput")
        )
        if tput is None and fp in winner_meta_by_fp:
            tput = _to_float(winner_meta_by_fp[fp].get("tput"))
        ws = str(result.get("workspace") or workspace_str or "")
        if name in selected_new:
            outcome = "promoted"
        elif name in round_winners:
            outcome = "round_winner"
        elif fp in rejected_by_fp:
            outcome = "rejected"
        else:
            outcome = "tested"
        reject_reason = None
        if outcome == "rejected":
            reject_reason = str(rejected_by_fp.get(fp, {}).get("reason") or "")
        # decision_note: prefer winner_meta (matches the round-winners
        # mirror), then the variant's own ``result.note`` (set by the
        # grid runner when picking a non-default value), then any
        # ``note`` recorded against the rejection ledger entry.
        note = (
            winner_meta_by_fp.get(fp, {}).get("note")
            or result.get("note")
            or entry.get("note")
            or rejected_by_fp.get(fp, {}).get("note")
        )
        duration = _to_float(result.get("duration_seconds"))
        variants.append(_shape_variant_decision(
            session_dir=session_dir,
            name=name,
            fingerprint=fp,
            extra_sglang_args=str(entry.get("extra_sglang_args") or ""),
            extra_envs=dict(entry.get("extra_envs") or {}),
            status=str(result.get("status") or "succeeded"),
            output_throughput=tput,
            gain_pct_vs_base=gain,
            outcome=outcome,
            reject_reason=reject_reason or None,
            workspace_str=ws or workspace_str,
            warnings=warnings,
            decision_note=str(note) if note else None,
            duration_seconds=duration,
        ))
    for fp, row in rejected_by_fp.items():
        if fp in seen_fps:
            continue
        note = row.get("note")
        variants.append(_shape_variant_decision(
            session_dir=session_dir,
            name=str(row.get("name") or ""),
            fingerprint=fp,
            extra_sglang_args=str(row.get("extra_sglang_args") or ""),
            extra_envs=dict(row.get("extra_envs") or {}),
            status="succeeded",
            output_throughput=_to_float(row.get("tput")),
            gain_pct_vs_base=_to_float(row.get("gain_pct")),
            outcome="rejected",
            reject_reason=str(row.get("reason") or "") or None,
            workspace_str=workspace_str,
            warnings=warnings,
            decision_note=str(note) if note else None,
        ))
    return variants


def _variants_from_disk_walk(
    session_dir: Path,
    workspace_str: str | None,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Reconstruct variant rows by globbing ``<workspace>/variant_*/benchmark_*/benchmark_report.json``.

    Used when the search ledger has no record for the round (e.g. an
    early discarded round whose ``last_round`` was overwritten by the
    next round). Yields ``status="reconstructed"`` rows so consumers can
    distinguish them from state-driven entries.
    """
    workspace = _resolve_under_session(session_dir, workspace_str) if workspace_str else None
    if workspace is None or not workspace.exists():
        return []
    out: list[dict[str, Any]] = []
    # Direct ``variant_*/`` children, plus ``combo/variant_*/`` for combo
    # rounds. Skip anything that doesn't actually have a benchmark report.
    candidates: list[Path] = []
    for sub in sorted(workspace.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith("variant_"):
            candidates.append(sub)
        elif sub.name == "combo":
            for child in sorted(sub.iterdir()):
                if child.is_dir() and child.name.startswith("variant_"):
                    candidates.append(child)
    for variant_dir in candidates:
        report_path = _find_benchmark_report(variant_dir)
        if not report_path:
            continue
        report = _load_json_safe(report_path, warnings) or {}
        out_tput, _ttft, _tpot, _e2el = _benchmark_report_metrics(report if isinstance(report, dict) else None)
        # Strip ``variant_NN_`` prefix so ``name`` matches the search
        # ledger's name field (used downstream for de-dup / matching).
        name = re.sub(r"^variant_\d+_", "", variant_dir.name)
        # Prefer the explicit duration_seconds field; fall back to the
        # benchmark runner's execution_time so disk-walked rows still
        # carry a duration.
        duration = _to_float((report or {}).get("duration_seconds"))
        if duration is None:
            duration = _to_float((report or {}).get("execution_time"))
        out.append(_shape_variant_decision(
            session_dir=session_dir,
            name=name,
            fingerprint="",
            extra_sglang_args="",
            extra_envs={},
            status="reconstructed",
            output_throughput=out_tput,
            gain_pct_vs_base=None,
            outcome="tested",
            reject_reason=None,
            workspace_str=str(variant_dir),
            warnings=warnings,
            duration_seconds=duration,
        ))
    return out


# Mirror of the constants the Coordinator uses for one-shot/cross-round
# promotion (see orchestrator/coordinator.py, near PROMOTE_THRESHOLD_PCT).
# Used by the collector's promotion_rule inference for pre-Phase-2 state.json
# files whose extras don't carry the audit fields directly. Kept in sync
# by visual inspection — coordinator.py is the source of truth.
_PROMOTE_THRESHOLD_PCT_INFER = 0.2
_CROSS_ROUND_LOOKBACK_INFER = 3
_CROSS_ROUND_MIN_APPEARANCES_INFER = 2
_CROSS_ROUND_MIN_AVG_GAIN_PCT_INFER = 0.1


def _infer_promotion_rule(
    attempt: dict[str, Any],
    phase_attempts: list[dict[str, Any]] | None,
    attempt_index: int | None,
) -> tuple[str | None, str | None]:
    """Heuristically reconstruct ``(promotion_rule, promotion_rule_detail)``
    for a {phase}_attempts entry whose ``extras`` dict predates the
    Coordinator Phase-2 audit wiring (no ``promotion_rule`` field).

    Mirrors the rules in :class:`Coordinator` (see ``PROMOTE_THRESHOLD_PCT``
    branch). We're deliberately conservative: when the signal is ambiguous
    (e.g. ``decision=='discarded'`` with a gain that meets the one-shot
    bar — could be ``accuracy_blocked`` or a stale ledger — we return
    ``(None, None)`` rather than guessing.

    Returns:
      * ``("single_shot", detail)``         — promoted, gain_vs_cb ≥ 0.2%
      * ``("cross_round_consistent", det)`` — promoted, gain_vs_cb < 0.2%
        AND a real cross-round signal exists in ``phase_attempts``
      * ``("below_threshold", detail)``     — discarded, gain_vs_cb < 0.2%
      * ``(None, None)``                    — can't tell
    """
    if not isinstance(attempt, dict):
        return None, None
    extras = attempt.get("extras") or {}
    decision = str(attempt.get("decision") or "")
    gain_vs_cb = _to_float(extras.get("gain_vs_cb"))
    bv_name = extras.get("best_variant_name")
    thresh = _PROMOTE_THRESHOLD_PCT_INFER

    if decision == "promoted":
        if gain_vs_cb is not None and gain_vs_cb >= thresh:
            return (
                "single_shot",
                f"inferred: gain_vs_cb={gain_vs_cb:.2f}% >= "
                f"single_shot_threshold={thresh}%",
            )
        # Promoted but sub-threshold ⇒ likely cross-round. Sanity-check
        # by looking back over the prior rounds within the same phase
        # for the same variant_name. Without a confirmed back-window we
        # leave the rule unset (better honest-null than a wrong label).
        if (
            phase_attempts
            and attempt_index is not None
            and isinstance(bv_name, str)
            and bv_name
        ):
            window = phase_attempts[
                max(0, attempt_index - _CROSS_ROUND_LOOKBACK_INFER + 1):
                attempt_index + 1
            ]
            appearances = 0
            gains: list[float] = []
            for prev in window:
                if not isinstance(prev, dict):
                    continue
                prev_extras = prev.get("extras") or {}
                if prev_extras.get("best_variant_name") == bv_name:
                    appearances += 1
                    pg = _to_float(prev_extras.get("gain_vs_cb"))
                    if pg is not None:
                        gains.append(pg)
            if (
                appearances >= _CROSS_ROUND_MIN_APPEARANCES_INFER
                and gains
                and (sum(gains) / len(gains)) >= _CROSS_ROUND_MIN_AVG_GAIN_PCT_INFER
            ):
                avg = sum(gains) / len(gains)
                return (
                    "cross_round_consistent",
                    f"inferred: variant={bv_name} appeared "
                    f">={_CROSS_ROUND_MIN_APPEARANCES_INFER} of last "
                    f"{_CROSS_ROUND_LOOKBACK_INFER} rounds with "
                    f"avg_gain={avg:.2f}% "
                    f"(min_avg={_CROSS_ROUND_MIN_AVG_GAIN_PCT_INFER}%)",
                )
        return None, None

    if decision == "discarded":
        if gain_vs_cb is not None and gain_vs_cb < thresh:
            return (
                "below_threshold",
                f"inferred: gain_vs_cb={gain_vs_cb:.2f}% < "
                f"single_shot_threshold={thresh}% "
                f"and no cross_round_consistent winner detected",
            )
        # discarded but met one-shot bar ⇒ likely accuracy_blocked, but
        # we can't confirm without the accuracy result. Stay honest.
        return None, None

    return None, None


def _round_decision_from_attempt(
    attempt: dict[str, Any] | None,
    *,
    phase_attempts: list[dict[str, Any]] | None = None,
    attempt_index: int | None = None,
) -> dict[str, Any]:
    if not isinstance(attempt, dict):
        return {}
    extras = attempt.get("extras") or {}
    # Pass-through: the Coordinator (post-Phase-2) already wrote these
    # five fields into extras. If they're there, use them verbatim.
    promotion_rule = extras.get("promotion_rule")
    promotion_rule_detail = extras.get("promotion_rule_detail")
    keep_threshold_pct = _to_float(extras.get("keep_threshold_pct"))
    accuracy_gate_passed = extras.get("accuracy_gate_passed")
    variants_tested_count = _to_int(extras.get("variants_tested_count"))
    # Inference fallback for pre-Phase-2 state.json files: if the audit
    # didn't record ``promotion_rule`` (typical of v2/ historical
    # sessions), reconstruct it from ``decision`` + ``gain_vs_cb`` +
    # cross-round attempt history. We never overwrite an explicit value.
    if promotion_rule is None:
        inferred_rule, inferred_detail = _infer_promotion_rule(
            attempt, phase_attempts, attempt_index,
        )
        if inferred_rule is not None:
            promotion_rule = inferred_rule
            if not promotion_rule_detail:
                promotion_rule_detail = inferred_detail
            if keep_threshold_pct is None:
                # Mirror the coordinator default so the JSON consumer
                # sees the bar the inference was made against.
                keep_threshold_pct = _PROMOTE_THRESHOLD_PCT_INFER
    return {
        "outcome": str(attempt.get("decision") or ""),
        "best_variant_name": extras.get("best_variant_name"),
        "gain_vs_cb_pct": _to_float(extras.get("gain_vs_cb")),
        "best_gain_pct_vs_base": _to_float(extras.get("best_gain_pct_vs_base")),
        "promotion_rule": promotion_rule,
        "promotion_rule_detail": promotion_rule_detail,
        "keep_threshold_pct": keep_threshold_pct,
        "accuracy_gate_passed": accuracy_gate_passed,
        "variants_tested_count": variants_tested_count,
    }


def _cap_variants(
    variants: list[dict[str, Any]],
    *,
    detail_level: str,
) -> list[dict[str, Any]]:
    if detail_level == "verbose" or len(variants) <= _DECISION_JOURNAL_STANDARD_VARIANT_CAP:
        return variants
    promoted = [v for v in variants if v.get("outcome") in ("promoted", "round_winner")]
    rejected = [v for v in variants if v.get("outcome") == "rejected"]
    tested = [v for v in variants if v.get("outcome") == "tested"]
    tested.sort(
        key=lambda v: abs(_to_float(v.get("gain_pct_vs_base")) or 0.0),
        reverse=True,
    )
    cap = max(0, _DECISION_JOURNAL_STANDARD_VARIANT_CAP - len(promoted) - len(rejected))
    kept_fps = {v.get("fingerprint") for v in promoted + rejected}
    out = list(promoted)
    for v in rejected:
        if v.get("fingerprint") not in kept_fps:
            out.append(v)
    for v in tested[:cap]:
        out.append(v)
    return out


def collect_decision_journal(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
    *,
    detail_level: str = "standard",
) -> list[dict[str, Any]]:
    """Build the per-round decision journal.

    v1.1 traversal order (vs v1):

    * Walk **every** ``{phase}_attempts`` entry (params + backends), not
      just ones present in ``backend_winners_history``. Earlier rounds
      whose audit was recorded but whose winners weren't mirrored to the
      history list (e.g. a discarded first round when the second round
      replaced its ``last_round``) still surface here.
    * Index ``backend_winners_history`` by ``round_id`` so we can backfill
      per-variant ``decision_note`` / ``gain_pct`` from the published
      winners mirror onto the search-ledger view.
    * Dedupe by ``round_id``: an attempt with ``round_id == "<phase>-001"``
      and a ``last_round`` whose ``round_id`` resolves to the same value
      (or to ``"<phase>-last"`` when the last_round mirrors the latest
      attempt) collapse into a single journal row.
    """
    journal: list[dict[str, Any]] = []

    # Build a winner-meta index keyed by ``(phase, round_id, fp)`` so the
    # last-round walker can backfill ``decision_note`` and ``gain_pct``.
    winners_by_round: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    history_rows_by_round: dict[tuple[str, str], dict[str, Any]] = {}
    for row in state.get("backend_winners_history") or []:
        if not isinstance(row, dict):
            continue
        ph = str(row.get("action") or "params")
        rid = str(row.get("round_id") or "")
        if not rid:
            continue
        history_rows_by_round[(ph, rid)] = row
        winners_idx: dict[str, dict[str, Any]] = {}
        for w in row.get("winners") or []:
            if not isinstance(w, dict):
                continue
            fp = _search_entry_fp(w)
            if fp:
                winners_idx[fp] = w
        winners_by_round[(ph, rid)] = winners_idx

    seen_round_ids: set[tuple[str, str]] = set()

    # baseline tput is the canonical reference for ``gain_pct_vs_base``
    # backfill. We use it ONLY when no per-variant gain was recorded in
    # state (the search ledger sometimes omits it).
    baseline_tput_ref = _to_float(state.get("baseline_tput"))

    def _emit(
        *,
        ts: str,
        phase: str,
        round_id: str | None,
        task_id: str,
        workspace: str | None,
        baseline_ref_tput: float | None,
        variants: list[dict[str, Any]],
        attempt: dict[str, Any] | None,
        phase_attempts: list[dict[str, Any]] | None = None,
        attempt_index: int | None = None,
    ) -> None:
        journal.append({
            "ts":                 ts,
            "phase":              phase,
            "round_id":           round_id,
            "task_id":            task_id,
            "workspace":          workspace,
            "baseline_ref_tput":  baseline_ref_tput,
            "current_best_tput":  _to_float((state.get("current_best") or {}).get("tput")),
            "keep_threshold_pct": None,
            "variants":           _cap_variants(variants, detail_level=detail_level),
            "round_decision":     _round_decision_from_attempt(
                attempt,
                phase_attempts=phase_attempts,
                attempt_index=attempt_index,
            ),
        })

    # Pass 1: every ``{phase}_attempts`` entry becomes a journal row.
    # The earliest round (which legitimately has ``round_id is None``
    # because it predates the audit's round-id assignment) gets a
    # synthetic ``"<phase>-000"`` so consumers can address it stably.
    for phase in ("params", "backends"):
        attempts = state.get(f"{phase}_attempts") or []
        if not isinstance(attempts, list):
            continue
        # Counter for synthesizing round_ids only when the audit didn't
        # record one. We start at 0 so the first synthesized id is
        # ``<phase>-000``, distinct from any real ``<phase>-001+``.
        synth_idx = 0
        # Filter to dict entries so the cross-round inference indexes
        # line up with what the inference walks (it slices the same
        # list with attempt_index). Original behaviour preserved.
        dict_attempts = [a for a in attempts if isinstance(a, dict)]
        for attempt_idx, attempt in enumerate(dict_attempts):
            extras = attempt.get("extras") or {}
            raw_rid = extras.get("round_id")
            if raw_rid:
                round_id = str(raw_rid)
            else:
                round_id = f"{phase}-{synth_idx:03d}"
                synth_idx += 1
            key = (phase, round_id)
            if key in seen_round_ids:
                continue
            seen_round_ids.add(key)
            ts = str(attempt.get("ts") or "")
            workspace = attempt.get("workspace")
            task_id = str(attempt.get("task_id") or "")
            history_row = history_rows_by_round.get(key) or {}
            winners_idx = winners_by_round.get(key) or {}
            search = state.get(f"{phase}_search") or {}
            lr = search.get("last_round") if isinstance(search, dict) else {}
            # ``last_round`` is the ledger's per-round scratchpad — the
            # search code overwrites it once per round, so we can only
            # use it for the LATEST attempt of each phase (the most
            # recent round_id in attempts wins). Older rounds fall back
            # to the winners_history mirror, which IS preserved across
            # rounds.
            is_latest_attempt = attempt is dict_attempts[-1]
            lr_round_id = str((lr or {}).get("round_id") or "") if isinstance(lr, dict) else ""
            use_last_round = (
                isinstance(lr, dict) and bool(lr.get("tested_fp")) and (
                    lr_round_id == round_id
                    or (is_latest_attempt and not lr_round_id)
                )
            )
            if use_last_round:
                variants = _variants_from_search_last_round(
                    session_dir, phase, search,
                    workspace_str=workspace,
                    selected_new=set((lr or {}).get("selected_new") or []),
                    round_winners=set(
                        (lr or {}).get("round_winners")
                        or {str(w.get("name") or "") for w in (history_row.get("winners") or []) if isinstance(w, dict)}
                    ),
                    warnings=warnings,
                    winner_meta_by_fp=winners_idx,
                )
            else:
                # No last_round mirror for this round — fall back to the
                # winners_history payload (round_winner-only view, but
                # preserves note + gain_pct for older rounds).
                variants = []
                for w in history_row.get("winners") or []:
                    if not isinstance(w, dict):
                        continue
                    name = str(w.get("name") or "")
                    fp = _search_entry_fp(w)
                    note = w.get("note")
                    variants.append(_shape_variant_decision(
                        session_dir=session_dir,
                        name=name,
                        fingerprint=fp,
                        extra_sglang_args=str(w.get("extra_sglang_args") or ""),
                        extra_envs=dict(w.get("extra_envs") or {}),
                        status="succeeded",
                        output_throughput=_to_float(w.get("tput")),
                        gain_pct_vs_base=_to_float(w.get("gain_pct")),
                        outcome="round_winner",
                        reject_reason=None,
                        workspace_str=workspace,
                        warnings=warnings,
                        decision_note=str(note) if note else None,
                    ))
                # Disk-walk fallback: when neither the search ledger
                # nor the winners mirror covers this round, reconstruct
                # variants from the benchmark_report.json files left on
                # disk under the round workspace. This is the only way
                # to surface a round 0 (round_id=None in extras) whose
                # last_round was overwritten by the next round.
                if not variants:
                    variants = _variants_from_disk_walk(
                        session_dir, workspace, warnings,
                    )
            base_ref = _to_float(history_row.get("base_tput"))
            if base_ref is None and isinstance(lr, dict):
                base_ref = _to_float(lr.get("base_tput"))
            # winners_history.base_tput is sometimes recorded as 0.0 in
            # older sessions even though baseline_tput is well-known.
            # Treat <=0 as "not recorded" so gain backfill can use the
            # real baseline reference instead of dividing by zero.
            if base_ref is None or base_ref <= 0.0:
                base_ref = baseline_tput_ref
            # Backfill ``gain_pct_vs_base`` for variants whose state-side
            # gain is null but whose throughput is known. Round-decision
            # consumers expect a numeric gain whenever both throughputs
            # are available.
            if base_ref and base_ref > 0.0:
                for v in variants:
                    if v.get("gain_pct_vs_base") is None and v.get("output_throughput") is not None:
                        v["gain_pct_vs_base"] = (
                            (v["output_throughput"] - base_ref) / base_ref * 100.0
                        )
            _emit(
                ts=ts,
                phase=phase,
                round_id=round_id,
                task_id=task_id,
                workspace=workspace,
                baseline_ref_tput=base_ref,
                variants=variants,
                attempt=attempt,
                phase_attempts=dict_attempts,
                attempt_index=attempt_idx,
            )

    # Pass 2: a winners_history row whose round_id was NOT covered by
    # any audit attempt (rare but observed in older sessions where
    # state.json was rolled back). Skip when we already emitted that
    # round in Pass 1.
    for (phase, round_id), row in history_rows_by_round.items():
        key = (phase, round_id)
        if key in seen_round_ids:
            continue
        seen_round_ids.add(key)
        attempt = _find_audit_attempt(state, phase, round_id=round_id, ts=row.get("ts"))
        workspace = attempt.get("workspace") if attempt else None
        # Pull the same dict-attempts list this phase used in Pass 1 so
        # the promotion_rule inference can do its lookback walk even on
        # rows that only surface via the history fallback.
        ph_attempts_raw = state.get(f"{phase}_attempts") or []
        ph_attempts_dict = (
            [a for a in ph_attempts_raw if isinstance(a, dict)]
            if isinstance(ph_attempts_raw, list) else []
        )
        try:
            attempt_index = (
                ph_attempts_dict.index(attempt)
                if isinstance(attempt, dict) and attempt in ph_attempts_dict
                else None
            )
        except ValueError:
            attempt_index = None
        winners_idx = winners_by_round.get(key) or {}
        variants: list[dict[str, Any]] = []
        for w in row.get("winners") or []:
            if not isinstance(w, dict):
                continue
            note = w.get("note")
            variants.append(_shape_variant_decision(
                session_dir=session_dir,
                name=str(w.get("name") or ""),
                fingerprint=_search_entry_fp(w),
                extra_sglang_args=str(w.get("extra_sglang_args") or ""),
                extra_envs=dict(w.get("extra_envs") or {}),
                status="succeeded",
                output_throughput=_to_float(w.get("tput")),
                gain_pct_vs_base=_to_float(w.get("gain_pct")),
                outcome="round_winner",
                reject_reason=None,
                workspace_str=workspace,
                warnings=warnings,
                decision_note=str(note) if note else None,
            ))
        _emit(
            ts=str(row.get("ts") or ""),
            phase=phase,
            round_id=round_id,
            task_id=str((attempt or {}).get("task_id") or ""),
            workspace=workspace,
            baseline_ref_tput=_to_float(row.get("base_tput")),
            variants=variants,
            attempt=attempt,
            phase_attempts=ph_attempts_dict,
            attempt_index=attempt_index,
        )

    journal.sort(key=lambda e: e.get("ts") or "")
    return journal


def _parse_kernel_summary_csv(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    import csv
    rows: list[dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                rows.append({
                    "kernel_id":  row.get("kernel_id") or row.get("name") or "",
                    "name":       row.get("name") or row.get("kernel_name") or "",
                    "gpu_pct":    _to_float(row.get("gpu_pct") or row.get("gpu_time_pct")),
                    "duration_us": _to_float(row.get("duration_us") or row.get("duration")),
                    "bottleneck": row.get("bottleneck") or "",
                })
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"kernel_summary.csv parse failed ({path}): {exc!r}")
    return rows[:25]


def _parse_tracelens_status(path: Path, warnings: list[str]) -> dict[str, Any]:
    data = _load_json_safe(path, warnings)
    if not isinstance(data, dict):
        return {}
    # Prefer an explicit ``summary`` / ``analysis_summary`` field. Do NOT
    # fall back to ``status`` here — a single token like ``"ok"`` is not
    # informative as a summary, and would crowd out the much richer
    # ``analysis.md`` text the kernel-profiling collector pulls in as a
    # secondary fallback. ``status`` is preserved alongside in the
    # timeline event extras for callers that actually care.
    summary = data.get("summary") or data.get("analysis_summary") or ""
    return {
        "tool": "tracelens_analysis",
        "analysis_summary": str(summary) if summary else None,
        "top_kernels": list(data.get("top_kernels") or data.get("kernels") or [])[:25],
    }


# TraceLens roofline category files we know how to parse. Order matters:
# the resulting top_kernels list is sorted by ``percent_of_total`` so any
# ordering quirk between categories doesn't affect the output, but we still
# walk gemm first because it dominates compute on every workload we've
# touched.
_TRACELENS_CATEGORY_FILES: tuple[str, ...] = (
    "gemm_metrics.json",
    "rmsnorm_metrics.json",
    "elementwise_metrics.json",
    "kernel_fusion_metrics.json",
    "reduce_metrics.json",
    "other_metrics.json",
)


def _shape_tracelens_op(op: dict[str, Any], category: str) -> dict[str, Any]:
    """Shape one entry from a TraceLens ``category_data/<cat>_metrics.json``
    ``operations[]`` list into the v1.1 ``top_kernels`` row.

    The roofline fields live under ``efficiency.{...}`` inside the
    category JSON; we hoist them flat onto the ``top_kernels`` row so
    consumers don't need to know the source schema. Missing fields stay
    None — we never fabricate roofline numbers.
    """
    eff = op.get("efficiency") if isinstance(op.get("efficiency"), dict) else {}
    name = str(op.get("name") or "")
    time_ms = _to_float(op.get("time_ms"))
    return {
        "kernel_id":          name,  # TraceLens has no separate id
        "name":               name,
        "category":           category,
        "operation_count":    int(op.get("count") or 0) or None,
        "duration_us":        (time_ms * 1000.0) if time_ms is not None else None,
        "time_ms":            time_ms,
        "percent_of_total":   _to_float(op.get("percent_of_total")),
        "percent_of_category": _to_float(op.get("percent_of_category")),
        "efficiency_percent": _to_float(eff.get("efficiency_percent")),
        "bound_type":         eff.get("bound_type") or None,
        "tflops_achieved":    _to_float(eff.get("tflops_achieved")),
        "flops_per_byte":     _to_float(eff.get("flops_per_byte")),
        "arithmetic_intensity": _to_float(eff.get("flops_per_byte")),
        "library":            op.get("library") or None,
        "input_dims":         op.get("Input Dims") or op.get("input_dims") or None,
        # gpu_pct mirrors percent_of_total for category-data ops (the
        # category JSON's "percent_of_total" IS the kernel's GPU-time
        # share for that op grouping); we keep both keys filled so old
        # consumers reading ``gpu_pct`` still see a value.
        "gpu_pct":            _to_float(op.get("percent_of_total")),
        "bottleneck":         eff.get("bound_type") or "",
    }


def _read_tracelens_category_data(
    tracelens_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Walk ``<tracelens>/category_data/*_metrics.json`` into top_kernels rows.

    Returns rows from all parseable category files, sorted by
    ``percent_of_total`` desc. Returns ``[]`` if the directory is missing
    or every file fails to parse — callers should treat empty as "no
    roofline data available" and try the next fallback (priority_data).
    """
    cat_dir = tracelens_dir / "category_data"
    if not cat_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    # Walk the canonical category list first, then anything else that
    # ends in ``_metrics.json`` (forward-compat — TraceLens may add new
    # categories without us needing a code change).
    seen_files: set[str] = set()
    for fname in _TRACELENS_CATEGORY_FILES:
        path = cat_dir / fname
        if not path.exists():
            continue
        seen_files.add(fname)
        data = _load_json_safe(path, warnings)
        if not isinstance(data, dict):
            continue
        category = str(data.get("category") or fname.replace("_metrics.json", ""))
        for op in data.get("operations") or []:
            if isinstance(op, dict):
                rows.append(_shape_tracelens_op(op, category))
    for path in sorted(cat_dir.glob("*_metrics.json")):
        if path.name in seen_files:
            continue
        data = _load_json_safe(path, warnings)
        if not isinstance(data, dict):
            continue
        category = str(data.get("category") or path.stem.replace("_metrics", ""))
        for op in data.get("operations") or []:
            if isinstance(op, dict):
                rows.append(_shape_tracelens_op(op, category))
    rows.sort(
        key=lambda r: (r.get("percent_of_total") or 0.0),
        reverse=True,
    )
    return rows


def _read_tracelens_priority_data(
    tracelens_dir: Path,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], str | None]:
    """Parse ``<tracelens>/priority_data.json`` into (top_kernels, summary).

    Used as a secondary fallback when ``category_data/`` isn't present
    or yielded no rows. ``priority_data.json`` only carries the
    impact-ranked roofline samples (no per-shape breakdown) but it's
    still richer than nothing.
    """
    path = tracelens_dir / "priority_data.json"
    if not path.exists():
        return ([], None)
    data = _load_json_safe(path, warnings)
    if not isinstance(data, dict):
        return ([], None)
    rows: list[dict[str, Any]] = []
    for src in (data.get("findings") or []):
        if not isinstance(src, dict):
            continue
        for member in src.get("members") or []:
            if isinstance(member, dict):
                rows.append({
                    "kernel_id":          str(member.get("operation") or ""),
                    "name":               str(member.get("operation") or ""),
                    "category":           str(member.get("category") or ""),
                    "operation_count":    None,
                    "duration_us":        (_to_float(member.get("time_ms")) or 0.0) * 1000.0
                        if member.get("time_ms") is not None else None,
                    "time_ms":            _to_float(member.get("time_ms")),
                    "percent_of_total":   None,
                    "percent_of_category": None,
                    "efficiency_percent": _to_float(member.get("efficiency_pct")),
                    "bound_type":         member.get("bound_type") or None,
                    "tflops_achieved":    None,
                    "flops_per_byte":     None,
                    "arithmetic_intensity": None,
                    "library":            member.get("library") or None,
                    "input_dims":         None,
                    "impact_score":       _to_float(member.get("impact_score")),
                    "gpu_pct":            None,
                    "bottleneck":         member.get("bound_type") or "",
                })
    if not rows:
        # ``all_estimates`` is the fallback list of impact estimates that
        # priority_data emits even when ``findings[].members`` is empty.
        for op in data.get("all_estimates") or []:
            if isinstance(op, dict):
                rows.append({
                    "kernel_id":          str(op.get("operation") or ""),
                    "name":               str(op.get("operation") or ""),
                    "category":           str(op.get("category") or ""),
                    "operation_count":    None,
                    "duration_us":        (_to_float(op.get("time_ms")) or 0.0) * 1000.0
                        if op.get("time_ms") is not None else None,
                    "time_ms":            _to_float(op.get("time_ms")),
                    "percent_of_total":   None,
                    "percent_of_category": None,
                    "efficiency_percent": _to_float(op.get("efficiency_pct")),
                    "bound_type":         op.get("bound_type") or None,
                    "tflops_achieved":    None,
                    "flops_per_byte":     None,
                    "arithmetic_intensity": None,
                    "library":            op.get("library") or None,
                    "input_dims":         None,
                    "impact_score":       _to_float(op.get("impact_score")),
                    "gpu_pct":            None,
                    "bottleneck":         op.get("bound_type") or "",
                })
    # Synthesize a summary line from the priorities list when it's
    # non-empty (analysis.md is the preferred source — we only land
    # here when analysis.md was unreadable).
    summary: str | None = None
    priorities = data.get("priorities")
    if isinstance(priorities, list) and priorities:
        head = priorities[0]
        if isinstance(head, dict):
            summary = (
                f"Top priority: {head.get('display_name') or head.get('category') or '?'} "
                f"impact_score={head.get('impact_score')!r}"
            )
    return (rows, summary)


def _read_tracelens_analysis_md(
    tracelens_dir: Path,
    warnings: list[str],
) -> str | None:
    """Read the first paragraph (≤600 chars) of ``analysis.md``.

    The file is human-authored markdown summarizing the run's findings.
    We surface the lead so dashboards can render a concise headline
    without pulling the whole document.
    """
    path = tracelens_dir / "analysis.md"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        warnings.append(f"failed to read {path}: {exc!r}")
        return None
    # First non-empty paragraph, capped at 600 chars. We stop at the
    # first blank-line break so multi-paragraph analyses don't bloat
    # the JSON.
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip() and lines:
            break
        if raw.strip().startswith("#"):
            # Skip leading headings — they're rarely informative on
            # their own.
            if not lines:
                continue
        if raw.strip():
            lines.append(raw.strip())
        elif not lines:
            continue
    summary = " ".join(lines).strip()
    if not summary:
        return None
    return summary[:600] + ("…" if len(summary) > 600 else "")


def collect_kernel_profiling(
    session_dir: Path,
    state: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    profile_root = session_dir / "runs" / "profile"
    attempt_by_task: dict[str, dict[str, Any]] = {}
    for entry in state.get("profile_attempts") or []:
        if isinstance(entry, dict) and entry.get("task_id"):
            attempt_by_task[str(entry["task_id"])] = entry

    if profile_root.exists():
        for task_dir in sorted(p for p in profile_root.iterdir() if p.is_dir()):
            task_id = task_dir.name
            attempt = attempt_by_task.get(task_id) or {}
            extras = attempt.get("extras") or {}
            report = _find_benchmark_report(task_dir)
            report_data = _load_json_safe(report, warnings) if report else None
            top_kernels: list[dict[str, Any]] = []
            if isinstance(report_data, dict):
                for k in report_data.get("kernel_summary") or []:
                    if isinstance(k, dict):
                        top_kernels.append({
                            "kernel_id": k.get("kernel_id") or k.get("name") or "",
                            "name":      k.get("name") or "",
                            "gpu_pct":   _to_float(k.get("gpu_pct")),
                            "duration_us": _to_float(k.get("time_ms")),
                            "bottleneck": k.get("bottleneck") or "",
                        })
            trace_paths: list[str] = []
            for sub in ("torch_trace", "capture_traces"):
                troot = task_dir / sub
                if not troot.exists():
                    for bench in task_dir.glob("benchmark_*"):
                        candidate = bench / sub
                        if candidate.exists():
                            troot = candidate
                            break
                if troot.exists():
                    for tp in sorted(troot.rglob("*.trace.json.gz"))[:20]:
                        rel = _rel(tp, session_dir)
                        if rel:
                            trace_paths.append(rel)
            kernel_summary_csv: Path | None = None
            for candidate in task_dir.rglob("kernel_summary.csv"):
                kernel_summary_csv = candidate
                break
            if kernel_summary_csv and not top_kernels:
                top_kernels = _parse_kernel_summary_csv(kernel_summary_csv, warnings)
            candidates_json = task_dir / "kernel_candidates.json"
            if not candidates_json.exists():
                for candidate in task_dir.rglob("kernel_candidates.json"):
                    candidates_json = candidate
                    break
            launch = _build_invocation_for_workspace(
                session_dir, str(task_dir), warnings,
            )
            if extras.get("profile_args"):
                launch["framework_args"] = str(extras["profile_args"])
                launch["framework_args_source"] = "profile_attempt_extras"
            runs.append({
                "run_id":              task_id,
                "ts":                  str(attempt.get("ts") or ""),
                "task_id":             task_id,
                "framework":           str(state.get("framework") or ""),
                "profile_config_path": extras.get("config_path"),
                "launch":              launch,
                "artifacts": {
                    "benchmark_report_path": _rel(report, session_dir) if report else None,
                    "trace_paths":           trace_paths,
                    "kernel_summary_csv":    _rel(kernel_summary_csv, session_dir)
                        if kernel_summary_csv else None,
                    "kernel_candidates_json": _rel(candidates_json, session_dir)
                        if candidates_json.exists() else None,
                    "tracelens_status_json": None,
                    "tracelens_log":         None,
                },
                "outputs": {
                    "tool": "magpie_torch_profiler",
                    "top_kernels": top_kernels[:25],
                    # Surface tracelens_analysis.summary if the profile
                    # report happens to embed one (newer benchmark
                    # runners synthesize a 1–2 sentence headline).
                    "analysis_summary": (
                        (
                            str(report_data.get("tracelens_analysis", {}).get("summary"))
                            or str(report_data.get("tracelens_analysis", {}).get("analysis_summary"))
                        )
                        if isinstance(report_data, dict)
                        and isinstance(report_data.get("tracelens_analysis"), dict)
                        else None
                    ) or None,
                },
            })

    for run_dir in _kernel_agent_run_dirs(session_dir):
        status_root = run_dir / "status" / "tracelens_analysis"
        if not status_root.exists():
            continue
        # The TraceLens artifact tree (priority_data.json, category_data/,
        # analysis.md) is shared across every status file in this run dir.
        # Compute its fallback rows / summary once so multi-status runs
        # don't redo the disk walk per status file.
        tracelens_dir = run_dir / "tracelens"
        cat_rows = (
            _read_tracelens_category_data(tracelens_dir, warnings)
            if tracelens_dir.exists() else []
        )
        priority_rows: list[dict[str, Any]] = []
        priority_summary: str | None = None
        if tracelens_dir.exists():
            priority_rows, priority_summary = _read_tracelens_priority_data(
                tracelens_dir, warnings,
            )
        analysis_md_summary = (
            _read_tracelens_analysis_md(tracelens_dir, warnings)
            if tracelens_dir.exists() else None
        )
        for status_path in sorted(status_root.glob("*.json")):
            parsed = _parse_tracelens_status(status_path, warnings)
            # Merge the fallbacks: status.json wins when it has data,
            # category_data fills in next, priority_data last.
            top_kernels = list(parsed.get("top_kernels") or [])
            if not top_kernels and cat_rows:
                top_kernels = list(cat_rows)
            if not top_kernels and priority_rows:
                top_kernels = list(priority_rows)
            summary = (
                parsed.get("analysis_summary")
                or analysis_md_summary
                or priority_summary
            )
            log_path = run_dir / "logs" / "tracelens_analysis" / f"{status_path.stem}.log"
            # P2-3: surface started_at/ended_at/duration_seconds when
            # the kernel-agent writer recorded them (terminal states
            # only). Historical sessions have only ``started_at`` /
            # ``updated_at`` — leave the new fields None.
            raw_status = _load_json_safe(status_path, warnings) or {}
            ts_started = (
                str(raw_status.get("started_at") or "")
                if isinstance(raw_status, dict) else ""
            )
            ts_ended = (
                str(raw_status.get("ended_at") or "")
                if isinstance(raw_status, dict) else ""
            )
            duration = (
                _to_float(raw_status.get("duration_seconds"))
                if isinstance(raw_status, dict) else None
            )
            runs.append({
                "run_id":              status_path.stem,
                "ts":                  ts_started,
                "task_id":             run_dir.name,
                "framework":           str(state.get("framework") or ""),
                "profile_config_path": None,
                "launch":              {},
                "artifacts": {
                    "benchmark_report_path": None,
                    "trace_paths":           [],
                    "kernel_summary_csv":    None,
                    "kernel_candidates_json": None,
                    "tracelens_status_json": _rel(status_path, session_dir),
                    "tracelens_log":         _rel(log_path, session_dir)
                        if log_path.exists() else None,
                },
                "outputs": {
                    "tool": "tracelens_analysis",
                    "top_kernels": top_kernels[:25],
                    "analysis_summary": summary,
                },
                "duration_seconds":    duration,
                "ended_ts_utc":        ts_ended or _add_seconds_iso(ts_started, duration),
            })

    runs.sort(key=lambda r: r.get("ts") or r.get("run_id") or "")
    return runs


def _kernel_name_index(
    state: dict[str, Any],
    session_dir: Path | None,
    warnings: list[str],
) -> dict[str, str]:
    """Build a ``kid -> human readable kernel name`` lookup.

    Sources (priority order — first non-empty wins):

    1. ``state.last_select_kernels.hot_kernels_top15[].{kernel_id,name}``
    2. ``state.kernel_integrate_attempts[*].{kernel_id, target_file}``
       (target_file basename is the fallback name when the integrate
       entry doesn't carry one explicitly).
    3. ``kernel-agent/runs/<sid>/kernel_candidates.json`` files
       (``kernel_id`` / ``name`` rows produced by the kernel agent).
    """
    by_kid: dict[str, str] = {}
    sk = state.get("last_select_kernels") or {}
    if isinstance(sk, dict):
        for entry in sk.get("hot_kernels_top15") or []:
            if not isinstance(entry, dict):
                continue
            kid = str(entry.get("kernel_id") or "")
            name = str(entry.get("name") or "")
            if kid and name and kid not in by_kid:
                by_kid[kid] = name
    integ = state.get("kernel_integrate_attempts") or {}
    if isinstance(integ, dict):
        for ent in integ.values():
            if not isinstance(ent, dict):
                continue
            kid = str(ent.get("kernel_id") or "")
            if not kid or kid in by_kid:
                continue
            name = str(ent.get("kernel_name") or ent.get("target_file") or "")
            if name:
                # target_file is usually a path — take its basename so the
                # column stays narrow.
                by_kid[kid] = name.rsplit("/", 1)[-1]
    if session_dir is not None:
        for run_dir in _kernel_agent_run_dirs(session_dir):
            for cand_path in run_dir.glob("kernel_candidates.json"):
                data = _load_json_safe(cand_path, warnings)
                if not isinstance(data, list):
                    if isinstance(data, dict):
                        data = data.get("kernels") or data.get("candidates") or []
                    else:
                        continue
                for k in data:
                    if not isinstance(k, dict):
                        continue
                    kid = str(k.get("kernel_id") or "")
                    name = str(k.get("name") or "")
                    if kid and name and kid not in by_kid:
                        by_kid[kid] = name
    return by_kid


# Backends we recognise in path-based inference. ``oob`` is the umbrella
# kernel-agent harness that hosts ``claude`` / ``codex`` / ``cursor``
# task workspaces; ``geak`` is the standalone GEAK runner. Paths under
# ``/kernel-agent/oob/`` are mapped to ``oob`` here because that's the
# only information the on-disk path carries — the specific tool can be
# recovered from ``kernel-agent/runs/<sid>/results/<kid>.json``.
_BACKEND_TOKENS_IN_ORDER: tuple[str, ...] = (
    "geak", "oob", "claude", "codex", "cursor",
)


def _infer_backend_from_paths(*candidates: Any) -> str | None:
    """Scan one or more string candidates (paths / URIs) for a known
    backend token. Returns the first match in order of ``candidates``;
    within a single candidate, GEAK / OOB win because those segments
    appear in canonical kernel-agent paths like
    ``/kernel-agent/geak/<sid>/...`` and ``/kernel-agent/oob/<sid>/...``.
    Returns None if no token matches or candidates are empty/non-strings.
    """
    for cand in candidates:
        if not cand:
            continue
        text = str(cand)
        if "/kernel-agent/" not in text and not any(
            f"/{tok}/" in text or f"/{tok}-" in text for tok in _BACKEND_TOKENS_IN_ORDER
        ):
            # cheap reject when nothing backend-shaped is in the string
            continue
        # Look for ``/kernel-agent/<backend>/`` first — the most reliable form.
        for tok in ("geak", "oob"):
            if f"/kernel-agent/{tok}/" in text:
                return tok
        # Then fall back to bare ``/<backend>/`` (e.g. tooling layouts).
        for tok in _BACKEND_TOKENS_IN_ORDER:
            if f"/{tok}/" in text or f"/{tok}-" in text:
                return tok
    return None


def _load_kernel_agent_kernel_index(
    session_dir: Path | None,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    """Build a per-kernel index from ``kernel-agent/runs/<sid>/results/<kid>.json``.

    Each entry surfaces just the fields the KDP collector needs to fill
    ``step.backend`` and ``step.speedup`` when the per-attempt records
    in ``state.json`` don't carry them directly:

    * ``best_backend``      — ``verification.best_backend``
    * ``selected_backends`` — list (orchestrator's chosen backend set)
    * ``attempt_backends``  — list of every attempt's backend (ordered)
    * ``micro_speedup``     — kernel-level micro_speedup from
      verification (also looks at top-level ``verification/<kid>.json``)
    * ``best_artifact_path`` — useful for path-based inference fallback

    Multiple run dirs for one kid are merged (later runs override
    earlier ones for fields they actually populate). Returns ``{}`` when
    ``session_dir`` is None or no kernel-agent runs exist.
    """
    if session_dir is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for run_dir in _kernel_agent_run_dirs(session_dir):
        results_dir = run_dir / "results"
        verify_dir = run_dir / "verification"
        if not results_dir.is_dir():
            continue
        for result_path in sorted(results_dir.glob("*.json")):
            kid = result_path.stem
            data = _load_json_safe(result_path, warnings)
            if not isinstance(data, dict):
                continue
            verification = data.get("verification")
            if not isinstance(verification, dict):
                # also try the standalone per-kid verification file
                vp = verify_dir / f"{kid}.json"
                v_data = _load_json_safe(vp if vp.exists() else None, warnings)
                verification = v_data if isinstance(v_data, dict) else {}
            attempts = data.get("attempts") if isinstance(data.get("attempts"), list) else []
            attempt_backends = [
                str(a.get("backend") or "").lower()
                for a in attempts
                if isinstance(a, dict) and a.get("backend")
            ]
            selected = data.get("selected_backends")
            selected_list = [str(b).lower() for b in selected] if isinstance(selected, list) else []
            entry = {
                "best_backend":      (str(verification.get("best_backend") or "").lower() or None),
                "selected_backends": selected_list,
                "attempt_backends":  attempt_backends,
                "micro_speedup":     _to_float(verification.get("micro_speedup")),
                "best_artifact_path": (
                    verification.get("best_artifact_path")
                    or data.get("best_artifact_path")
                ),
            }
            prev = out.get(kid)
            if prev is None:
                out[kid] = entry
            else:
                # Merge — prefer non-empty / non-None values from the new entry.
                for k, v in entry.items():
                    if v:
                        prev[k] = v
    return out


def _extras_kernel_speedup(extras: dict[str, Any] | None) -> float | None:
    """Pull a kernel-level speedup out of an ``extras`` dict.

    Looks at the keys the orchestrator / kernel-agent has historically
    used: ``kernel_speedup``, ``speedup_x``, ``geak_speedup``,
    ``kernel_speedup_x``, ``micro_speedup``, and nested
    ``benchmark.speedup``. Returns the first non-None float, or None.
    """
    if not isinstance(extras, dict) or not extras:
        return None
    for key in (
        "kernel_speedup", "speedup_x", "geak_speedup",
        "kernel_speedup_x", "micro_speedup", "speedup",
    ):
        v = _to_float(extras.get(key))
        if v is not None:
            return v
    bench = extras.get("benchmark")
    if isinstance(bench, dict):
        v = _to_float(bench.get("speedup") or bench.get("micro_speedup"))
        if v is not None:
            return v
    return None


def collect_kernel_decision_path(
    state: dict[str, Any],
    warnings: list[str],
    session_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Per-kernel causal chain across select → kernel_opt → integrate → validate.

    Groups every step by the kernel id the orchestrator assigned
    (``k001`` / ``k002`` / …) and orders steps within a group by ``ts``.
    Each step carries the same ``duration_seconds`` enrichment used by
    :func:`collect_phase_timeline`, plus the outcome / decision_note /
    gain_pct / speedup pulled from whichever record best describes that
    step:

    * ``select`` — one event per kid from
      ``last_select_kernels.hot_kernels_top15`` (the orchestrator's
      single-shot select snapshot; there's no per-kid select_attempt
      list in the current shared_state). ``ts`` is the snapshot ts.
    * ``kernel_opt`` — flattened ``state.kernel_opt_attempts[kid].history[]``
      rows, plus the entry's terminal ``last_decision`` when the
      history doesn't already include it. ``backend`` is recovered
      from history extras when present; otherwise None.
    * ``integrate`` — flattened
      ``state.kernel_integrate_attempts[*].attempts[]`` rows, keyed by
      ``ent.kernel_id``.
    * ``validate`` — a kernel-owned validate_stack event (rare); we
      surface it when ``validate_stack_attempts[].extras.kernel_id``
      matches a kid we've already tracked.

    Returns ``[]`` (without emitting a warning) when none of the
    upstream sources have any rows for this session.
    """
    name_by_kid = _kernel_name_index(state, session_dir, warnings)
    # Per-kid index built from ``kernel-agent/runs/<sid>/results/<kid>.json``
    # — best source of truth for backend / kernel_speedup when the
    # per-attempt history in ``state.json`` doesn't carry them.
    ka_index = _load_kernel_agent_kernel_index(session_dir, warnings)
    steps_by_kid: dict[str, list[dict[str, Any]]] = {}

    def _push(kid: str, step: dict[str, Any]) -> None:
        if not kid:
            return
        bucket = steps_by_kid.setdefault(kid, [])
        bucket.append(step)

    # 1) select_kernels — snapshot from ``last_select_kernels``. We do
    #    NOT emit a step per missing entry; only kids that the
    #    orchestrator actually surfaced as a hot kernel get a "select"
    #    step.
    sk = state.get("last_select_kernels") or {}
    sk_ts = str(sk.get("ts") or "") if isinstance(sk, dict) else ""
    if isinstance(sk, dict):
        hot = sk.get("hot_kernels_top15") or []
        if isinstance(hot, list):
            for entry in hot:
                if not isinstance(entry, dict):
                    continue
                kid = str(entry.get("kernel_id") or "")
                if not kid:
                    continue
                _push(kid, {
                    "kid":              kid,
                    "kernel_name":      name_by_kid.get(kid) or str(entry.get("name") or ""),
                    "step":             "select",
                    "backend":          None,
                    "ts":               sk_ts,
                    "duration_seconds": None,
                    "ended_ts_utc":     None,
                    "task_id":          "",
                    "workspace":        None,
                    "outcome":          "selected",
                    "decision_note":    str(entry.get("bottleneck") or ""),
                    "gain_pct":         None,
                    "speedup":          None,
                    "extras": {
                        "gpu_pct":               _to_float(entry.get("gpu_pct")),
                        "bottleneck":            str(entry.get("bottleneck") or ""),
                        "recommended_backends":  list(entry.get("recommended_backends") or []),
                        "recommended_actions":   list(entry.get("recommended_actions") or []),
                        "reusable_native_kernel": bool(entry.get("reusable_native_kernel")),
                    },
                })

    # 2) select_kernels_attempts — when an audit-list form exists,
    #    treat each entry as one "select" step (one per task_id). The
    #    current orchestrator doesn't write this list, but other
    #    pipelines might; we keep the path forward-compatible.
    sk_attempts = state.get("select_kernels_attempts")
    if isinstance(sk_attempts, list):
        for entry in sk_attempts:
            if not isinstance(entry, dict):
                continue
            extras = entry.get("extras") if isinstance(entry.get("extras"), dict) else {}
            kids = []
            single = entry.get("kernel_id") or extras.get("kernel_id")
            if single:
                kids = [str(single)]
            else:
                hot = extras.get("hot_kernels") or extras.get("hot_kernels_top15") or []
                if isinstance(hot, list):
                    kids = [str(h.get("kernel_id") or "")
                            for h in hot if isinstance(h, dict) and h.get("kernel_id")]
            duration = _phase_event_duration(session_dir, entry, warnings)
            ts = str(entry.get("ts") or "")
            for kid in kids:
                if not kid:
                    continue
                _push(kid, {
                    "kid":              kid,
                    "kernel_name":      name_by_kid.get(kid, ""),
                    "step":             "select",
                    "backend":          None,
                    "ts":               ts,
                    "duration_seconds": duration,
                    "ended_ts_utc":     _add_seconds_iso(ts, duration),
                    "task_id":          str(entry.get("task_id") or ""),
                    "workspace":        entry.get("workspace"),
                    "outcome":          str(entry.get("decision") or "selected"),
                    "decision_note":    str(entry.get("decision_note") or ""),
                    "gain_pct":         None,
                    "speedup":          None,
                    "extras":           dict(extras),
                })

    # 3) kernel_opt_attempts → one step per history entry.
    kernel_opt = state.get("kernel_opt_attempts") or {}
    if isinstance(kernel_opt, dict):
        for kid, ent in kernel_opt.items():
            if not isinstance(ent, dict):
                continue
            kid_s = str(kid)
            kid_idx = ka_index.get(kid_s) or {}
            # Best backend for this kid — used when neither the per-history
            # entry nor ent itself names one. We prefer the verification's
            # ``best_backend`` (what was actually selected), then any
            # ``selected_backends`` list, then attempt_backends, then a
            # path-based inference from ``last_artifact_path``.
            kid_best_backend = (
                kid_idx.get("best_backend")
                or (kid_idx.get("selected_backends") or [None])[0]
                or (kid_idx.get("attempt_backends") or [None])[0]
                or _infer_backend_from_paths(
                    kid_idx.get("best_artifact_path"),
                    ent.get("last_artifact_path"),
                )
            )
            kid_micro_speedup = (
                kid_idx.get("micro_speedup")
                if kid_idx.get("micro_speedup") is not None
                else _to_float(ent.get("last_micro_speedup"))
            )
            history = ent.get("history") or []
            history_list = history if isinstance(history, list) else []
            for h in history_list:
                if not isinstance(h, dict):
                    continue
                ts = str(h.get("ts") or "")
                extras = h.get("extras") if isinstance(h.get("extras"), dict) else {}
                duration = _to_float(extras.get("duration_seconds")) if extras else None
                # Backend resolution: explicit per-attempt first, then ent,
                # then kernel-agent results.json, then path-based.
                backend = (
                    (extras.get("backend") if extras else None)
                    or h.get("backend")
                    or ent.get("backend")
                    or kid_best_backend
                    or _infer_backend_from_paths(
                        h.get("workspace") or (extras.get("workspace") if extras else None),
                        h.get("artifact_path") or (extras.get("artifact_path") if extras else None),
                    )
                )
                # Speedup: per-attempt fields first (orchestrator writes
                # ``micro`` in history rows), then extras, then carry the
                # kernel-level micro_speedup onto the terminal history row.
                step_speedup = _to_float(
                    h.get("micro_speedup")
                    or h.get("speedup")
                    or h.get("micro")
                )
                if step_speedup is None:
                    step_speedup = _extras_kernel_speedup(extras)
                _push(kid_s, {
                    "kid":              kid_s,
                    "kernel_name":      name_by_kid.get(kid_s, ""),
                    "step":             "kernel_opt",
                    "backend":          str(backend).lower() if backend else None,
                    "ts":               ts,
                    "duration_seconds": duration,
                    "ended_ts_utc":     _add_seconds_iso(ts, duration),
                    "task_id":          str(h.get("task_id") or extras.get("task_id") or ""),
                    "workspace":        h.get("workspace") or extras.get("workspace"),
                    "outcome":          str(h.get("decision") or ""),
                    "decision_note":    str(h.get("note") or extras.get("note") or ""),
                    "gain_pct":         _to_float(h.get("gain_pct") or extras.get("gain_pct")),
                    "speedup":          step_speedup,
                    "extras":           dict(extras) if extras else {},
                })
            # The terminal history row didn't always carry a kernel-level
            # speedup — patch it in from ``ent.last_micro_speedup`` /
            # ``ka_index.micro_speedup`` so the final attempt reflects the
            # benchmark-measured kernel_speedup. We only patch when the
            # row's existing speedup is None (don't overwrite real data).
            if history_list and kid_micro_speedup is not None:
                # Find the latest kernel_opt step we just emitted for this
                # kid (history is in chronological order, and _push appends).
                terminal_step = None
                for s in reversed(steps_by_kid.get(kid_s, [])):
                    if s.get("step") == "kernel_opt":
                        terminal_step = s
                        break
                if terminal_step is not None and terminal_step.get("speedup") is None:
                    terminal_step["speedup"] = kid_micro_speedup
            # If the per-attempt history is empty but we still have a
            # terminal ``last_decision``, surface a single synthetic
            # step so the chain isn't completely silent.
            if not history_list and (ent.get("last_decision") or ent.get("rejected_reason")):
                ts = str(ent.get("last_ts") or "")
                fallback_backend = (
                    ent.get("backend")
                    or kid_best_backend
                    or _infer_backend_from_paths(ent.get("last_artifact_path"))
                )
                _push(kid_s, {
                    "kid":              kid_s,
                    "kernel_name":      name_by_kid.get(kid_s, ""),
                    "step":             "kernel_opt",
                    "backend":          str(fallback_backend).lower() if fallback_backend else None,
                    "ts":               ts,
                    "duration_seconds": None,
                    "ended_ts_utc":     None,
                    "task_id":          "",
                    "workspace":        None,
                    "outcome":          str(ent.get("last_decision") or ent.get("rejected_reason") or ""),
                    "decision_note":    str(ent.get("rejected_reason") or ""),
                    "gain_pct":         None,
                    "speedup":          kid_micro_speedup,
                    "extras":           {"attempts": int(ent.get("attempts") or 0)},
                })

    # 4) kernel_integrate_attempts → one step per attempt entry.
    integ = state.get("kernel_integrate_attempts") or {}
    if isinstance(integ, dict):
        for ent in integ.values():
            if not isinstance(ent, dict):
                continue
            kid = str(ent.get("kernel_id") or "")
            if not kid:
                continue
            kid_idx = ka_index.get(kid) or {}
            # The integrate run consumes the kernel-agent's chosen patch;
            # ``ent.patch_path`` points into ``/kernel-agent/<backend>/``
            # so it's the authoritative source for the backend label.
            integ_backend_default = (
                ent.get("backend")
                or _infer_backend_from_paths(
                    ent.get("patch_path"),
                    ent.get("target_file"),
                    kid_idx.get("best_artifact_path"),
                )
                or kid_idx.get("best_backend")
            )
            # Carry the kernel-level micro_speedup onto integrate steps
            # so the kernel_speedup signal isn't lost just because the
            # integrate row itself only carries an e2e ``gain_pct``.
            integ_speedup_default = (
                kid_idx.get("micro_speedup")
                if kid_idx.get("micro_speedup") is not None
                else None
            )
            attempts = ent.get("attempts") or []
            if not isinstance(attempts, list):
                continue
            for a in attempts:
                if not isinstance(a, dict):
                    continue
                ts = str(a.get("ts") or "")
                a_extras = a.get("extras") if isinstance(a.get("extras"), dict) else {}
                duration = _to_float(a_extras.get("duration_seconds")) if a_extras else None
                if duration is None and a.get("workspace"):
                    duration = _phase_event_duration(session_dir, a, warnings)
                attempt_backend = (
                    a.get("backend")
                    or (a_extras.get("backend") if a_extras else None)
                    or integ_backend_default
                )
                # Per-attempt speedup if recorded; otherwise the
                # kernel-level micro_speedup carried over from
                # results/<kid>.json. We do NOT promote ``gain_pct``
                # (e2e) into ``speedup`` — those are different units.
                attempt_speedup = _extras_kernel_speedup(a_extras)
                if attempt_speedup is None:
                    attempt_speedup = _to_float(
                        a.get("kernel_speedup")
                        or a.get("micro_speedup")
                        or a.get("speedup")
                    )
                if attempt_speedup is None:
                    attempt_speedup = integ_speedup_default
                _push(kid, {
                    "kid":              kid,
                    "kernel_name":      name_by_kid.get(kid, ""),
                    "step":             "integrate",
                    "backend":          str(attempt_backend).lower() if attempt_backend else None,
                    "ts":               ts,
                    "duration_seconds": duration,
                    "ended_ts_utc":     _add_seconds_iso(ts, duration),
                    "task_id":          str(a.get("task_id") or ""),
                    "workspace":        a.get("workspace"),
                    "outcome":          str(a.get("decision") or a.get("status") or ""),
                    "decision_note":    str(a.get("note") or ""),
                    "gain_pct":         _to_float(a.get("gain_pct")),
                    "speedup":          attempt_speedup,
                    "extras": {
                        "patch_path":  ent.get("patch_path"),
                        "target_file": ent.get("target_file"),
                        "report_path": a.get("report_path"),
                        "new_tput":    _to_float(a.get("new_tput")),
                    },
                })

    # 5) validate_stack_attempts — surface ones tagged with a kernel_id
    #    (rare but explicit). We do NOT pull every validate_stack into
    #    every kid's chain because validate_stack is action-level, not
    #    kernel-level.
    for entry in state.get("validate_stack_attempts") or []:
        if not isinstance(entry, dict):
            continue
        extras = entry.get("extras") if isinstance(entry.get("extras"), dict) else {}
        kid = str(entry.get("kernel_id") or extras.get("kernel_id") or "")
        if not kid:
            continue
        ts = str(entry.get("ts") or "")
        duration = _phase_event_duration(session_dir, entry, warnings)
        _push(kid, {
            "kid":              kid,
            "kernel_name":      name_by_kid.get(kid, ""),
            "step":             "validate",
            "backend":          None,
            "ts":               ts,
            "duration_seconds": duration,
            "ended_ts_utc":     _add_seconds_iso(ts, duration),
            "task_id":          str(entry.get("task_id") or ""),
            "workspace":        entry.get("workspace"),
            "outcome":          str(entry.get("decision") or entry.get("status") or ""),
            "decision_note":    str(extras.get("note") or ""),
            "gain_pct":         _to_float(entry.get("key_metric")),
            "speedup":          None,
            "extras":           dict(extras),
        })

    # Order steps within each group by ts (lexicographic ISO8601 sort
    # is chronological for these strings). Empty-ts steps land first
    # so they don't shadow the dated history.
    out: list[dict[str, Any]] = []
    step_order = {"select": 0, "kernel_opt": 1, "integrate": 2, "validate": 3}
    for kid in sorted(steps_by_kid.keys()):
        bucket = steps_by_kid[kid]
        bucket.sort(key=lambda s: (s.get("ts") or "", step_order.get(s.get("step") or "", 9)))
        # backends_attempted = ordered set of distinct backends across
        # kernel_opt steps in this chain.
        backends_seen: list[str] = []
        for s in bucket:
            b = s.get("backend")
            if b and b not in backends_seen:
                backends_seen.append(b)
        durations = [s.get("duration_seconds") for s in bucket
                     if isinstance(s.get("duration_seconds"), (int, float))]
        total_dur = sum(durations) if durations else None
        final_outcome = ""
        for s in reversed(bucket):
            if s.get("outcome"):
                final_outcome = str(s["outcome"])
                break
        out.append({
            "kid":         kid,
            "kernel_name": name_by_kid.get(kid, "") or (bucket[0].get("kernel_name") or "" if bucket else ""),
            "steps":       bucket,
            "summary": {
                "total_steps":           len(bucket),
                "backends_attempted":    backends_seen,
                "final_outcome":         final_outcome,
                "total_duration_seconds": total_dur,
            },
        })
    return out


# ---------------------------------------------------------------------------
# Post-processing: cross-section reconciliation for session timing
# ---------------------------------------------------------------------------
def enrich_session_and_timeline(
    session_meta: dict[str, Any],
    phase_timeline: list[dict[str, Any]] | None,
    state: dict[str, Any],
) -> None:
    """Cross-fill session timing + closing-event duration in place.

    Called from the exporter once both ``session_meta`` and
    ``phase_timeline`` are collected (the latter wasn't available when
    ``collect_session`` ran, since collectors execute sequentially).
    Two reconciliations happen here:

    1. If ``session_meta.session_started_at_utc`` or
       ``session_ended_at_utc`` is None we re-run
       :func:`_extract_session_timing` with the now-available
       phase_timeline as the derived-fallback source. Any newly
       resolved field is written into the session dict; values
       already set are preserved (state-sourced timestamps beat
       derived ones).
    2. For any closing event whose ``duration_seconds`` is None we
       compute it as ``session_ended_at_utc - closing.ts`` (when both
       are resolvable). This gives consumers a real wall-clock cost
       for the closing phase instead of an unknowable None.
    """
    if not isinstance(session_meta, dict):
        return

    # 1) Top-up session timing via phase_timeline-derived fallbacks.
    if session_meta.get("session_started_at_utc") in (None, "") \
            or session_meta.get("session_ended_at_utc") in (None, ""):
        derived = _extract_session_timing(
            state if isinstance(state, dict) else {},
            {},
            phase_timeline if isinstance(phase_timeline, list) else None,
        )
        if session_meta.get("session_started_at_utc") in (None, ""):
            session_meta["session_started_at_utc"] = derived.get("session_started_at_utc")
        if session_meta.get("session_ended_at_utc") in (None, ""):
            session_meta["session_ended_at_utc"] = derived.get("session_ended_at_utc")
        # Recompute duration if we now have both endpoints and didn't before.
        if session_meta.get("session_duration_seconds") is None:
            s_dt = _parse_iso(session_meta.get("session_started_at_utc"))
            e_dt = _parse_iso(session_meta.get("session_ended_at_utc"))
            if s_dt is not None and e_dt is not None:
                diff = (e_dt - s_dt).total_seconds()
                if diff >= 0:
                    session_meta["session_duration_seconds"] = round(diff, 1)
        # Mirror duration into elapsed_minutes when we now have one and
        # the legacy field was 0.0 / None (the wall-clock-from-now
        # fallback is less accurate than a real measured duration).
        if session_meta.get("session_duration_seconds") is not None:
            session_meta["elapsed_minutes"] = round(
                session_meta["session_duration_seconds"] / 60.0, 2,
            )

    # 2) Fill closing-event durations against session_ended_at_utc.
    if not isinstance(phase_timeline, list):
        return
    sess_end_dt = _parse_iso(session_meta.get("session_ended_at_utc"))
    if sess_end_dt is None:
        return
    for evt in phase_timeline:
        if not isinstance(evt, dict):
            continue
        if evt.get("action") != "closing":
            continue
        if evt.get("duration_seconds") is not None:
            continue
        evt_ts_dt = _parse_iso(evt.get("ts"))
        if evt_ts_dt is None:
            continue
        diff = (sess_end_dt - evt_ts_dt).total_seconds()
        if diff < -1.0:
            # session_ended_at_utc landed materially before this closing
            # event — the timeline isn't internally consistent (clock
            # skew or ts metadata inversion). Leave duration None and
            # don't synthesize an end ts that contradicts the timeline.
            continue
        # Sub-second skew (closing.ts has microsecond precision while
        # session_ended_at_utc was truncated to seconds, or vice versa)
        # is treated as a flat-zero duration rather than skipped — the
        # closing-phase essentially completed at the same instant the
        # session ended, which is the correct semantic.
        if diff < 0:
            diff = 0.0
        evt["duration_seconds"] = round(diff, 1)
        if evt.get("ended_ts_utc") in (None, ""):
            evt["ended_ts_utc"] = _add_seconds_iso(evt.get("ts") or "", evt["duration_seconds"])


# ---------------------------------------------------------------------------
# §15 roofline collector
# ---------------------------------------------------------------------------
# Discover roofline ``final.json`` files in the well-known locations
# (orchestrator output, standalone roofline tool, kernel-agent
# roofline). Each file may have one of two shapes:
#
#   1. top-level ``mode`` / ``baseline`` / ``latest`` / ``delta``
#      (the standalone roofline tool's wire shape)
#   2. wrapped under ``roofline_comparison`` (the orchestrator's
#      ``reports/final.json`` shape — see test fixture)
#
# Each discovered file becomes one entry whose ``source_path`` is the
# session-relative file path. Files are returned in mtime order
# (oldest first) so the list itself conveys a timeline. Invalid JSON
# is a warning, not a crash; an unrecognised top-level shape is a
# warning (so an operator notices we found a final.json that doesn't
# match either expected schema) but is silently dropped from the
# output.
_ROOFLINE_FINAL_PATTERNS: tuple[str, ...] = (
    "reports/final.json",
    "reports/**/final.json",
    "runs/roofline/**/final.json",
    "kernel-agent/runs/*/roofline/**/final.json",
)


def _roofline_extract_payload(blob: Any) -> dict[str, Any] | None:
    """Lift the roofline payload from either of the two shapes we accept.

    Returns the dict that carries ``mode`` / ``baseline`` / ``latest``
    / ``delta`` keys, or None if the blob doesn't match either layout.
    """
    if not isinstance(blob, dict):
        return None
    # Shape 1: top-level. Accept it if at least one of the marker
    # keys is present (``mode`` alone is enough — minimum shape used
    # by the standalone tool).
    if any(k in blob for k in ("mode", "baseline", "latest", "delta", "roofline")):
        # If the blob has ``roofline_comparison`` *as well* we still
        # prefer the wrapped sub-dict (orchestrator shape carries both
        # an outer ``mode``/``ts`` and a nested comparison; the inner
        # dict is the truthful one).
        if isinstance(blob.get("roofline_comparison"), dict):
            return blob["roofline_comparison"]
        if isinstance(blob.get("roofline"), dict):
            return blob["roofline"]
        # Direct top-level — must carry at least ``mode`` or ``baseline``
        # to qualify; an isolated ``delta`` doesn't.
        if "mode" in blob or "baseline" in blob or "latest" in blob:
            return blob
    # Shape 2: wrapped under ``roofline_comparison``.
    rc = blob.get("roofline_comparison")
    if isinstance(rc, dict):
        return rc
    return None


def _roofline_normalize_snapshot(snap: Any) -> dict[str, Any] | None:
    """Coerce a snapshot to the schema shape with explicit None fields.

    Returns None when ``snap`` is missing entirely (preserves the
    "snapshot absent" signal — the schema's ``RooflineEntry.latest``
    is ``Optional`` and the renderer skips None blocks).
    """
    if snap is None:
        return None
    if not isinstance(snap, dict):
        return None
    tk = snap.get("top_kernel")
    norm_tk: dict[str, Any] | None
    if isinstance(tk, dict) and tk:
        norm_tk = {
            "name":           tk.get("name"),
            "gpu_pct":        _to_float(tk.get("gpu_pct")),
            "efficiency_pct": _to_float(tk.get("efficiency_pct")),
            "bound_type":     tk.get("bound_type"),
        }
    else:
        norm_tk = None
    return {
        "snapshot_id":    snap.get("snapshot_id"),
        "ts":             snap.get("ts"),
        "compute_pct":    _to_float(snap.get("compute_pct")),
        "idle_pct":       _to_float(snap.get("idle_pct")),
        "comm_pct":       _to_float(snap.get("comm_pct")),
        "top_bottleneck": snap.get("top_bottleneck"),
        "top_kernel":     norm_tk,
    }


def _roofline_normalize_delta(delta: Any) -> dict[str, Any] | None:
    """Pass deltas through verbatim when dict, None otherwise."""
    if delta is None:
        return None
    if not isinstance(delta, dict) or not delta:
        return None
    return dict(delta)


def collect_roofline(
    session_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Discover roofline final.json files; emit one entry per file.

    Files are de-duplicated by resolved path (so the broader globs
    don't surface the same orchestrator report twice) and sorted by
    mtime ascending — the oldest snapshot first, matching the
    "timeline" semantics tests assert.
    """
    out: list[dict[str, Any]] = []
    if not session_dir.exists():
        return out

    discovered: dict[Path, float] = {}
    for pattern in _ROOFLINE_FINAL_PATTERNS:
        try:
            for hit in session_dir.glob(pattern):
                try:
                    if not hit.is_file():
                        continue
                    mtime = hit.stat().st_mtime
                except OSError:
                    continue
                discovered.setdefault(hit, mtime)
        except OSError:
            continue

    for path in sorted(discovered.keys(), key=lambda p: discovered[p]):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            warnings.append(f"collect_roofline: failed to read {path}: {exc!r}")
            continue
        try:
            blob = json.loads(text)
        except json.JSONDecodeError as exc:
            warnings.append(f"collect_roofline: invalid JSON at {path}: {exc!r}")
            continue
        payload = _roofline_extract_payload(blob)
        if payload is None:
            warnings.append(
                f"collect_roofline: {path} has no recognisable roofline shape "
                "(missing mode / baseline / latest / roofline_comparison)"
            )
            continue
        try:
            rel = str(path.relative_to(session_dir))
        except ValueError:
            rel = str(path)
        mode_raw = payload.get("mode")
        out.append({
            "source_path": rel,
            "mode":        mode_raw if isinstance(mode_raw, str) else None,
            "baseline":    _roofline_normalize_snapshot(payload.get("baseline")),
            "latest":      _roofline_normalize_snapshot(payload.get("latest")),
            "delta":       _roofline_normalize_delta(payload.get("delta")),
        })
    return out


# ---------------------------------------------------------------------------
# data_provenance — per-section source artifact probes
# ---------------------------------------------------------------------------
# Glob expansion under ``session_dir`` is capped so a pathological tree
# (e.g. tens of thousands of variant directories) can't slow the export
# to a crawl. We don't need an exact count for provenance — just enough
# to distinguish "found" from "missing".
_PROVENANCE_GLOB_MAX_HITS = 50

_PROVENANCE_IMAGE_ENV_VARS: tuple[str, ...] = (
    "HYPERLOOM_IMAGE",
    "CONTAINER_IMAGE",
    "IMAGE",
)


def _probe_file(
    session_dir: Path,
    relative_glob: str,
    role: str,
    *,
    required: bool,
) -> dict[str, Any]:
    """Stat-only existence probe for ``session_dir / relative_glob``."""
    probe: dict[str, Any] = {
        "path":     relative_glob,
        "role":     role,
        "required": bool(required),
        "found":    False,
        "found_count": 0,
        "representative_path": None,
    }
    if not session_dir.exists():
        probe["note"] = "session_dir does not exist"
        return probe

    rep: str | None = None
    count = 0
    try:
        if any(ch in relative_glob for ch in "*?["):
            iterator = session_dir.glob(relative_glob)
        else:
            candidate = session_dir / relative_glob
            iterator = iter([candidate]) if candidate.exists() else iter([])
        for hit in iterator:
            try:
                if not hit.exists():
                    continue
            except OSError as exc:
                probe["note"] = f"{type(exc).__name__}: {exc}"
                continue
            count += 1
            if rep is None:
                try:
                    rep = str(hit.relative_to(session_dir))
                except ValueError:
                    rep = str(hit)
            if count >= _PROVENANCE_GLOB_MAX_HITS:
                break
    except (OSError, ValueError) as exc:
        probe["note"] = f"glob failed: {type(exc).__name__}: {exc}"
        return probe

    if count:
        probe["found"] = True
        probe["found_count"] = count
        probe["representative_path"] = rep
    return probe


def _probe_env(
    name: str,
    role: str,
    *,
    required: bool,
) -> dict[str, Any]:
    raw = os.environ.get(name)
    found = bool(raw)
    probe: dict[str, Any] = {
        "path":     f"env:{name}",
        "role":     role,
        "required": bool(required),
        "found":    found,
        "found_count": 1 if found else 0,
        "representative_path": raw if found else None,
    }
    return probe


def _make_or_probe(
    candidates: list[dict[str, Any]],
    *,
    path: str,
    role: str,
) -> dict[str, Any]:
    found_any = any(c.get("found") for c in candidates)
    return {
        "path":     path,
        "role":     role,
        "required": True,
        "found":    found_any,
        "found_count": sum(c.get("found_count") or 0 for c in candidates),
        "representative_path": next(
            (c.get("representative_path") for c in candidates if c.get("found")),
            None,
        ),
    }


def _provenance_populated(value: Any, *, section: str | None = None) -> bool:
    """Return True iff a built section carries non-trivial data."""
    if section == "attribution" and isinstance(value, dict):
        gpse = value.get("gain_per_stack_entry") or []
        return bool(gpse) and value.get("method") != "missing"
    if section == "param_search" and isinstance(value, dict):
        for sub in ("params", "backends"):
            ledger = value.get(sub) or {}
            if (ledger.get("tested_count") or 0) > 0:
                return True
            if ledger.get("accepted") or ledger.get("rejected"):
                return True
        return False
    if section == "baseline" and isinstance(value, dict):
        return bool(
            (value.get("throughput_tok_s_per_gpu") or 0) > 0
            or value.get("attempts_history")
            or value.get("benchmark_report_path")
        )
    if section == "final" and isinstance(value, dict):
        return bool(
            value.get("throughput_tok_s_per_gpu") is not None
            or value.get("cumulative_gain_pct_validated") is not None
            or value.get("action_path")
        )
    if section == "critic_robustness" and isinstance(value, dict):
        return bool(value.get("critic_iterations") or value.get("robustness_signals"))
    if section == "kernel_lifecycle" and isinstance(value, dict):
        return any(
            bool(value.get(k))
            for k in ("detected", "recommended", "optimized", "adopted", "rejected")
        )
    if section == "capability_summary" and isinstance(value, dict):
        for cap in value.values():
            if not isinstance(cap, dict):
                continue
            if (cap.get("attempts") or 0) > 0 or (cap.get("keeps") or 0) > 0:
                return True
            if (cap.get("tested") or 0) > 0:
                return True
        return False

    if value is None:
        return False
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (list, dict)):
                if _provenance_populated(item):
                    return True
            elif item not in (None, "", 0, 0.0, False):
                return True
        return False
    if isinstance(value, dict):
        for v in value.values():
            if isinstance(v, (list, dict)):
                if _provenance_populated(v):
                    return True
            elif v not in (None, "", 0, 0.0, False, []):
                return True
        return False
    return value not in ("", 0, 0.0, False)


def _provenance_status(
    sources: list[dict[str, Any]],
    *,
    populated: bool,
) -> tuple[str, list[str]]:
    missing: list[str] = []
    for p in sources:
        if not p.get("required"):
            continue
        if not p.get("found"):
            missing.append(str(p.get("role") or p.get("path") or "(unknown)"))
    if not missing:
        return "complete", []
    return ("partial" if populated else "empty"), missing


def collect_data_provenance(
    session_dir: Path,
    breakdown: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    """Probe the on-disk artifacts that feed every breakdown section."""
    sd = session_dir
    out: list[dict[str, Any]] = []

    def _emit(section: str,
              sources: list[dict[str, Any]],
              populated_value: Any,
              *,
              notes: list[str] | None = None) -> None:
        populated = _provenance_populated(populated_value, section=section)
        status, missing = _provenance_status(sources, populated=populated)
        out.append({
            "section":          section,
            "status":           status,
            "populated":        populated,
            "sources":          sources,
            "missing_required": missing,
            "notes":            list(notes or []),
        })

    # ---- session ----
    # Manifest + state are the two canonical inputs. Image / timing
    # source enumeration is best-effort metadata so consumers see
    # exactly which candidates were probed when ``session.image`` /
    # ``session_started_at_utc`` end up None.
    session_sources = [
        _probe_file(sd, "manifest.json", "session manifest", required=True),
        _probe_file(sd, "state.json",    "session state",    required=True),
    ]
    # Image OR-probe (all optional — sessions in /home/chenluo/sbd/v2
    # legitimately have image=null everywhere).
    img_state    = _probe_file(sd, "state.json",
                               "container image (state.json: image/container_image)", required=False)
    img_manifest = _probe_file(sd, "manifest.json",
                               "container image (manifest.json: image/container_image)", required=False)
    img_baseline = _probe_file(sd, "runs/baseline/*/baseline_config.with_envs.yaml",
                               "container image (baseline_config.with_envs.yaml: docker_image/image)",
                               required=False)
    img_bench    = _probe_file(sd, "runs/baseline/*/benchmark_*/config.yaml",
                               "container image (benchmark config.yaml: docker_image)", required=False)
    session_sources.extend([img_state, img_manifest, img_baseline, img_bench])
    for env_name in _PROVENANCE_IMAGE_ENV_VARS:
        session_sources.append(
            _probe_env(env_name, f"container image env ({env_name})", required=False)
        )

    # Per-session notes about what (if anything) populated the image
    # and timing fields. Derived strictly from the already-built
    # ``session`` payload so this collector doesn't redo any work.
    notes: list[str] = []
    sess_block = breakdown.get("session") if isinstance(breakdown.get("session"), dict) else {}
    if sess_block.get("image") in (None, ""):
        notes.append(
            "image: no image metadata found in any candidate source "
            "(state.json / manifest.json / baseline yamls / env)"
        )
    started_missing = sess_block.get("session_started_at_utc") in (None, "")
    ended_missing   = sess_block.get("session_ended_at_utc")   in (None, "")
    if started_missing and ended_missing:
        notes.append(
            "session timing: no startup/shutdown timestamps in state.json "
            "(state.start_ts / state.closing_started_unix both absent); "
            "consider checking *_attempts ts events for a derived bound"
        )
    elif started_missing:
        notes.append(
            "session timing: started_at missing in state.json; ended_at present"
        )
    elif ended_missing:
        notes.append(
            "session timing: ended_at missing in state.json "
            "(state.stopped_at / last_tick_ts / closing_started_unix all absent)"
        )

    _emit("session", session_sources, breakdown.get("session"), notes=notes or None)

    # ---- workload ----
    _emit("workload",
          [_probe_file(sd, "manifest.json", "session manifest", required=True),
           _probe_file(sd, "state.json",    "session state (optional fallback)", required=False)],
          breakdown.get("workload"))

    # ---- baseline ----
    bp_flat   = _probe_file(sd, "runs/baseline/*/benchmark_report.json",
                            "baseline benchmark_report (flat)", required=False)
    bp_nested = _probe_file(sd, "runs/baseline/*/*/benchmark_report.json",
                            "baseline benchmark_report (nested)", required=False)
    baseline_or_probe = _make_or_probe(
        [bp_flat, bp_nested],
        path="(runs/baseline/*[/**]/benchmark_report.json)",
        role="baseline benchmark_report",
    )
    _emit("baseline",
          [bp_flat, bp_nested, baseline_or_probe,
           _probe_file(sd, "runs/baseline/*/baseline_config.with_envs.yaml",
                       "Magpie baseline yaml", required=False),
           _probe_file(sd, "runs/baseline/*/server.log",
                       "baseline server.log (ServerArgs)", required=False),
           _probe_file(sd, "state.json", "session state (baseline_attempts)", required=False)],
          breakdown.get("baseline"))

    # ---- final ----
    _emit("final",
          [_probe_file(sd, "state.json",
                       "session state (final / current_best / validated)", required=True),
           _probe_file(sd, "runs/*/*/benchmark_report.json",
                       "any benchmark_report (ttft/e2el fallback)", required=False)],
          breakdown.get("final"))

    # ---- decision_journal ----
    _emit("decision_journal",
          [_probe_file(sd, "state.json",
                       "session state (params/backends_attempts)", required=True),
           _probe_file(sd, "runs/params/*/variant_*/benchmark_*/benchmark_report.json",
                       "params variant benchmark_report", required=False),
           _probe_file(sd, "runs/backends/*/variant_*/benchmark_*/benchmark_report.json",
                       "backends variant benchmark_report", required=False)],
          breakdown.get("decision_journal"))

    # ---- sweep ----
    sweep_sources = [
        _probe_file(sd, "runs/sweep/*/variant_*/benchmark_*/benchmark_report.json",
                    "sweep variant benchmark_report", required=False),
        _probe_file(sd, "runs/sweep/*/*/benchmark_report.json",
                    "sweep variant benchmark_report (flat)", required=False),
        _probe_file(sd, "state.json", "session state (last_sweep)", required=False),
    ]
    _emit("sweep", sweep_sources, breakdown.get("sweep"),
          notes=(["sweep was not exercised this session"]
                 if not _provenance_populated(breakdown.get("sweep"), section="sweep") else None))

    # ---- phase_timeline ----
    _emit("phase_timeline",
          [_probe_file(sd, "state.json",
                       "session state (*_attempts)", required=True),
           _probe_file(sd, "runs/*/*/benchmark_report.json",
                       "any benchmark_report (duration)", required=False)],
          breakdown.get("phase_timeline"))

    # ---- kernel_profiling ----
    pp_flat   = _probe_file(sd, "runs/profile/*/benchmark_report.json",
                            "profile benchmark_report (flat)", required=False)
    pp_nested = _probe_file(sd, "runs/profile/*/*/benchmark_report.json",
                            "profile benchmark_report (nested)", required=False)
    profile_or_probe = _make_or_probe(
        [pp_flat, pp_nested],
        path="(runs/profile/*[/**]/benchmark_report.json)",
        role="profile benchmark_report",
    )
    tl_canonical = _probe_file(sd, "kernel-agent/runs/*/status/tracelens_analysis/*.json",
                               "TraceLens status JSON (canonical)", required=False)
    tl_legacy = _probe_file(sd, "kernel-agent-workspace/**/status/tracelens_analysis/*.json",
                            "TraceLens status JSON (legacy workspace)", required=False)
    tl_or_probe = _make_or_probe(
        [tl_canonical, tl_legacy],
        path="(**/status/tracelens_analysis/*.json)",
        role="TraceLens status JSON",
    )
    _emit("kernel_profiling",
          [pp_flat, pp_nested, profile_or_probe,
           tl_canonical, tl_legacy, tl_or_probe,
           _probe_file(sd, "kernel-agent/runs/*/tracelens/priority_data.json",
                       "TraceLens priority_data", required=False),
           _probe_file(sd, "kernel-agent/runs/*/tracelens/category_data/*_metrics.json",
                       "TraceLens category metrics", required=False),
           _probe_file(sd, "kernel-agent/runs/*/tracelens/analysis.md",
                       "TraceLens analysis.md", required=False),
           _probe_file(sd, "runs/profile/*/torch_trace/*.trace.json.gz",
                       "torch trace files", required=False),
           _probe_file(sd, "runs/profile/*/*/torch_trace/*.trace.json.gz",
                       "torch trace files (nested)", required=False),
           _probe_file(sd, "runs/profile/*/kernel_summary.csv",
                       "magpie kernel_summary.csv", required=False),
           _probe_file(sd, "runs/profile/*/*/kernel_summary.csv",
                       "magpie kernel_summary.csv (nested)", required=False)],
          breakdown.get("kernel_profiling"))

    # ---- kernel_decision_path ----
    _emit("kernel_decision_path",
          [_probe_file(sd, "state.json",
                       "session state (select_kernels/kernel_opt/kernel_integrate_attempts)",
                       required=True),
           _probe_file(sd, "kernel-agent/runs/*/results/*.json",
                       "kernel-agent per-attempt results", required=False)],
          breakdown.get("kernel_decision_path"))

    # ---- kernel_lifecycle ----
    _emit("kernel_lifecycle",
          [_probe_file(sd, "state.json",
                       "session state (recommended/optimized/adopted/rejected_kernels)",
                       required=True),
           _probe_file(sd, "kernel-agent/runs/*/tracelens/category_data/*_metrics.json",
                       "TraceLens category metrics (roofline)", required=False)],
          breakdown.get("kernel_lifecycle"))

    # ---- geak / oob invocations ----
    def _geak_oob_sources() -> list[dict[str, Any]]:
        return [
            _probe_file(sd, "kernel-agent/runs/*/optimization_attempts.jsonl",
                        "kernel-agent optimization_attempts.jsonl", required=False),
            _probe_file(sd, "kernel-agent-workspace/**/optimization_attempts.jsonl",
                        "legacy kernel-agent-workspace attempts", required=False),
        ]
    _emit("geak_invocations", _geak_oob_sources(), breakdown.get("geak_invocations"))
    _emit("oob_invocations",  _geak_oob_sources(), breakdown.get("oob_invocations"))

    # ---- critic / robustness ----
    _emit("critic_robustness",
          [_probe_file(sd, "critic-workdir",     "critic-agent workdir",     required=False),
           _probe_file(sd, "robustness-workdir", "robustness-agent workdir", required=False)],
          breakdown.get("critic_robustness"))

    # ---- roofline ----
    rp_orchestrator = _probe_file(sd, "reports/final.json",
                                  "orchestrator final report", required=False)
    rp_orchestrator_nested = _probe_file(sd, "reports/**/final.json",
                                         "orchestrator final report (nested)",
                                         required=False)
    rp_standalone = _probe_file(sd, "runs/roofline/**/final.json",
                                "standalone roofline tool output", required=False)
    rp_kernel_agent = _probe_file(sd, "kernel-agent/runs/*/roofline/**/final.json",
                                  "kernel-agent roofline output", required=False)
    roofline_or_probe = _make_or_probe(
        [rp_orchestrator, rp_orchestrator_nested, rp_standalone, rp_kernel_agent],
        path="(any of the roofline final.json locations above)",
        role="roofline final.json (any location)",
    )
    _emit("roofline",
          [rp_orchestrator, rp_orchestrator_nested, rp_standalone,
           rp_kernel_agent, roofline_or_probe],
          breakdown.get("roofline"))

    # ---- attribution ----
    _emit("attribution",
          [_probe_file(sd, "state.json",
                       "session state (gain_per_stack_entry / optimization_stack)",
                       required=True)],
          breakdown.get("attribution"))

    # ---- param_search ----
    _emit("param_search",
          [_probe_file(sd, "state.json",
                       "session state (params_search / backends_search)",
                       required=True)],
          breakdown.get("param_search"))

    return out


__all__ = [
    "collect_attribution",
    "collect_baseline",
    "collect_capability_summary",
    "collect_critic_robustness",
    "collect_data_provenance",
    "collect_decision_journal",
    "collect_final",
    "collect_kernel_decision_path",
    "collect_kernel_invocations",
    "collect_kernel_lifecycle",
    "collect_kernel_profiling",
    "collect_param_search",
    "collect_phase_timeline",
    "collect_roofline",
    "collect_session",
    "collect_source_files",
    "collect_sweep",
    "collect_telemetry",
    "collect_workload",
]
