#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""
Import session breakdown JSON files into the `perf_runs_dev` PostgreSQL table.

Two connection modes are supported:

MODE A: ``--mode ssh-kubectl`` (default; matches the new infra)
    Pipes SQL through `ssh amd@<hop1> -- kubectl exec ... psql`. Requires no
    local Python DB driver. Best for one-shot historical backfills and works
    from any laptop.

MODE B: ``--mode local``
    Connects with psycopg2 to a local TCP endpoint (usually a port-forward).
    Faster for many rows, but you must set up the tunnel first, e.g.::

        ssh -L 5432:127.0.0.1:5432 amd@10.245.143.31 \\
            "kubectl port-forward -n primus-safe \\
                \\$(kubectl get pod -n primus-safe -l postgres-operator.crunchydata.com/role=master -o name) \\
                5432:5432"

Examples
--------
    # Single file -- write to perf_runs_dev via SSH+kubectl
    python scripts/import_session_breakdown.py /path/to/session_breakdown.json

    # Batch import everything we just pulled with fetch_remote_sessions.py
    python scripts/import_session_breakdown.py --dir ./remote_sessions

    # Dry-run: parse + print row, do not write to DB
    python scripts/import_session_breakdown.py --dry-run file.json

    # Use a different table or DB
    python scripts/import_session_breakdown.py --table perf_runs_dev file.json

    # Local psycopg2 mode (requires tunnel above)
    python scripts/import_session_breakdown.py --mode local file.json

Environment variables (used as defaults)
---------------------------------------
- PERF_RUNS_DB_URL       Full postgresql:// URL. If set, overrides individual flags (local mode).
- PERF_RUNS_DB_HOST      Default: 127.0.0.1
- PERF_RUNS_DB_PORT      Default: 5432
- PERF_RUNS_DB_USER      Default: postgres
- PERF_RUNS_DB_PASSWORD  Default: (empty; not needed when connecting from inside the pod)
- PERF_RUNS_DB_NAME      Default: primus-safe-db
- PERF_RUNS_TABLE        Default: perf_runs_dev
- PERF_RUNS_SSH_HOP1     Default: amd@10.245.143.31
- PERF_RUNS_K8S_NS       Default: primus-safe

Notes
-----
- Idempotency is via `unique_key` -- the script will (on local mode) create a
  UNIQUE index if it doesn't exist, so `ON CONFLICT (unique_key) DO UPDATE`
  works.
- `unique_key = BASE64(model_name + "+" + image_short)` where image_short is
  the last two path components of the image without the version tag.
  This matches the contract of the POST /perf-leaderboard/api/v1/perf-runs API
  so dev rows can be promoted to prod without rekeying.
- In ssh-kubectl mode we use PostgreSQL dollar-quoted strings to embed JSON
  safely without any escaping.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

IMPORTER_NAME = "import_session_breakdown.py"
IMPORTER_VERSION = "1.1.0"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_IMAGES: Dict[str, str] = {
    # sglang profilerfix: rocprofiler captures HipGraphLaunch kernels (issue #352)
    # Pre-profilerfix image (restore when reverting): "harbor.core42.primus-safe.amd.com/proxy/lmsysorg/sglang:v0.5.11-rocm720-mi30x"
    "sglang": "harbor.core42.primus-safe.amd.com/proxy/primussafe/sglang:v0.5.11-rocm720-mi30x-profilerfix",
    "vllm":   "harbor.core42.primus-safe.amd.com/proxy/vllm/vllm-openai-rocm:v0.19.0",
}

DEFAULT_DB = {
    "host":     os.environ.get("PERF_RUNS_DB_HOST", "127.0.0.1"),
    "port":     int(os.environ.get("PERF_RUNS_DB_PORT", "5432")),
    "user":     os.environ.get("PERF_RUNS_DB_USER", "postgres"),
    "password": os.environ.get("PERF_RUNS_DB_PASSWORD", ""),
    "dbname":   os.environ.get("PERF_RUNS_DB_NAME", "primus-safe-db"),
}

DEFAULT_TABLE = os.environ.get("PERF_RUNS_TABLE", "perf_runs_dev")
DEFAULT_SSH_HOP1 = os.environ.get("PERF_RUNS_SSH_HOP1", "amd@10.245.143.31")
DEFAULT_K8S_NAMESPACE = os.environ.get("PERF_RUNS_K8S_NS", "primus-safe")


def _import_psycopg2():
    """Lazily import psycopg2 so ssh-kubectl mode works without the driver.

    Deferring the import keeps the script usable in the default ssh-kubectl
    mode (which shells out to ``psql``) on machines that have no local
    PostgreSQL driver installed.

    Returns:
        tuple: ``(psycopg2_module, psycopg2.extras.Json)`` when the driver is
            importable.

    Raises:
        SystemExit: If psycopg2 is not installed; an install hint is written
            to stderr and the process exits with status 1.
    """
    try:
        import psycopg2 as _pg
        from psycopg2.extras import Json as _Json
        return _pg, _Json
    except ImportError:
        sys.stderr.write(
            "[FATAL] --mode local requires psycopg2. Install with:\n"
            "    pip install psycopg2-binary\n"
            "Or use --mode ssh-kubectl (default).\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Walk a chain of nested dict keys, tolerating missing/None levels.

    Args:
        d (Any): The starting object, typically a dict.
        *keys (str): Successive keys to descend through.
        default (Any): Value returned when any level is missing, is not a
            dict, or resolves to None.

    Returns:
        Any: The value found at the nested key path, otherwise ``default``.
    """
    cur = d
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default


def truncate(s: Optional[str], n: int) -> Optional[str]:
    """Truncate a string to at most ``n`` characters.

    Args:
        s (Optional[str]): The string to truncate, or None.
        n (int): Maximum allowed length.

    Returns:
        Optional[str]: None when ``s`` is None, otherwise ``s`` clipped to the
            first ``n`` characters.
    """
    if s is None:
        return None
    return s if len(s) <= n else s[:n]


def derive_image(framework_name: str, session_image: Optional[str]) -> str:
    """Resolve the container image for a run, preferring the session value.

    Args:
        framework_name (str): Framework name (e.g. ``"sglang"`` / ``"vllm"``)
            used to pick a default when no session image is present.
        session_image (Optional[str]): Image recorded on the session, if any.

    Returns:
        str: ``session_image`` when set, else the framework's default image, or
            ``"unknown"`` if the framework has no default.
    """
    if session_image:
        return session_image
    fw = (framework_name or "").lower()
    return DEFAULT_IMAGES.get(fw, "unknown")


def derive_image_short(image: str) -> str:
    """Strip version tag/digest, return the last 2 path components (or 1 if only one)."""
    img = image.split("@")[0]            # strip digest
    img = img.split(":")[0]              # strip tag
    parts = [p for p in img.split("/") if p]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else "unknown"


def derive_unique_key(model_name: str, image: str) -> str:
    """BASE64("<model_name>+<image_short>"); matches the perf-runs API unique_key
    contract so dev rows promote to prod without rekeying."""
    raw = f"{model_name}+{derive_image_short(image)}"
    return base64.b64encode(raw.encode("utf-8")).decode("ascii")


_STOP_REASON_FAILED_TOKENS = ("timeout", "killed", "error", "failed", "abort", "oom", "crash")


def derive_status(session: Dict) -> str:
    """Map session.stop_reason to {success, running, failed}: empty=>running,
    failure token=>failed, else success (report emitted = clean termination)."""
    stop = (session.get("stop_reason") or "").strip().lower()
    if not stop:
        return "running"
    if any(tok in stop for tok in _STOP_REASON_FAILED_TOKENS):
        return "failed"
    return "success"


def detect_category(data: Dict) -> str:
    """MoE if a kernel category/name or model name has MoE markers; VLM for
    vision markers; else Dense."""
    kernels = safe_get(data, "kernel_lifecycle", "detected", default=[]) or []
    if isinstance(kernels, list):
        for k in kernels:
            if not isinstance(k, dict):
                continue
            cat = (k.get("kernel_category") or "").lower()
            name = (k.get("name") or "").lower()
            if "moe" in cat or "moe" in name:
                return "MoE"

    model_name = (safe_get(data, "workload", "model_name") or "").lower()
    if any(tag in model_name for tag in ("moe", "mixtral", "a3b", "a14b", "a22b")):
        return "MoE"
    if any(tag in model_name for tag in ("-vl-", "vision", "llava", "internvl", "qwen-vl", "qwen2-vl", "qwen2.5-vl", "qwen3-vl")):
        return "VLM"

    return "Dense"


def reshape_snapshot(snap: Optional[Dict]) -> Optional[Dict]:
    """Flatten a roofline snapshot into a compact display-friendly dict.

    Args:
        snap (Optional[Dict]): A raw roofline snapshot, or None.

    Returns:
        Optional[Dict]: A flattened dict with compute/idle/comm percentages,
            top-kernel name/efficiency, and snapshot id/timestamp; None when
            ``snap`` is not a dict.
    """
    if not isinstance(snap, dict):
        return None
    return {
        "compute": snap.get("compute_pct"),
        "idle": snap.get("idle_pct"),
        "comm": snap.get("comm_pct"),
        "top_bottleneck": snap.get("top_bottleneck"),
        "top_kernel_name": safe_get(snap, "top_kernel", "name"),
        "top_kernel_efficiency": safe_get(snap, "top_kernel", "efficiency_pct"),
        "snapshot_id": snap.get("snapshot_id"),
        "ts": snap.get("ts"),
    }


def extract_roofline(data: Dict) -> Optional[Dict]:
    """Pass through the first roofline entry verbatim for full fidelity (every
    nested field). Previously reshaped fields remain under entry.baseline.* /
    entry.latest.* — nothing removed."""
    rl_list = safe_get(data, "roofline", default=[]) or []
    if isinstance(rl_list, list) and rl_list and isinstance(rl_list[0], dict):
        return copy.deepcopy(rl_list[0])
    return None


def compute_gain_pct(baseline_tput: Optional[float], opt_tput: Optional[float]) -> Optional[float]:
    """Compute percentage throughput improvement from baseline to optimized.

    Args:
        baseline_tput (Optional[float]): Baseline throughput (tok/s/GPU).
        opt_tput (Optional[float]): Optimized throughput (tok/s/GPU).

    Returns:
        Optional[float]: ``(opt - baseline) / baseline * 100`` rounded to two
            decimals, or None when either input is missing or the baseline is
            non-positive.
    """
    if baseline_tput is None or opt_tput is None:
        return None
    try:
        if baseline_tput <= 0:
            return None
        return round((opt_tput - baseline_tput) / baseline_tput * 100.0, 2)
    except (TypeError, ZeroDivisionError):
        return None


def compute_kernel_gain(attribution_src: Dict) -> Optional[float]:
    """Sum the GEAK + OOB contributions to total cumulative gain.

    Args:
        attribution_src (Dict): The ``attribution.source_breakdown`` sub-dict.

    Returns:
        Optional[float]: Combined ``geak_pct_of_total + oob_pct_of_total``
            rounded to two decimals, or None when neither value is present.
    """
    geak = attribution_src.get("geak_pct_of_total")
    oob = attribution_src.get("oob_pct_of_total")
    if geak is None and oob is None:
        return None
    total = (geak or 0) + (oob or 0)
    return round(total, 2)


def compute_param_gain(attribution_src: Dict) -> Optional[float]:
    """Param-attribution gain = params choice + sweep variant search (sweep is a
    parameter-space search, so it belongs in the 'param' category)."""
    params = attribution_src.get("params_pct_of_total")
    sweep = attribution_src.get("sweep_pct_of_total")
    if params is None and sweep is None:
        return None
    total = (params or 0) + (sweep or 0)
    return round(total, 2)


def compute_backend_gain(attribution_src: Dict) -> Optional[float]:
    """Return the backend variant-choice contribution (e.g. attention=aiter).

    Args:
        attribution_src (Dict): The ``attribution.source_breakdown`` sub-dict.

    Returns:
        Optional[float]: ``backends_pct_of_total`` rounded to two decimals, or
            None when the value is absent.
    """
    backends = attribution_src.get("backends_pct_of_total")
    if backends is None:
        return None
    return round(float(backends), 2)


def compute_geak_gain(attribution_src: Dict) -> Optional[float]:
    """Return the GEAK-source gain percentage.

    Args:
        attribution_src (Dict): The ``attribution.source_breakdown`` sub-dict.

    Returns:
        Optional[float]: ``geak_pct_of_total`` rounded to two decimals, or None
            when the value is missing or non-numeric.
    """
    v = attribution_src.get("geak_pct_of_total")
    if not isinstance(v, (int, float)):
        return None
    return round(float(v), 2)


def compute_oob_gain(attribution_src: Dict) -> Optional[float]:
    """Return the OOB-source gain percentage.

    Args:
        attribution_src (Dict): The ``attribution.source_breakdown`` sub-dict.

    Returns:
        Optional[float]: ``oob_pct_of_total`` rounded to two decimals, or None
            when the value is missing or non-numeric.
    """
    v = attribution_src.get("oob_pct_of_total")
    if not isinstance(v, (int, float)):
        return None
    return round(float(v), 2)


def compute_framework_gain(attribution: Dict) -> Optional[float]:
    """Framework-source gain = SUM(delta_pct) over gain_per_stack_entry[] where
    action='framework_pr'. None when absent so normalisation collapses it to 0.00."""
    entries = attribution.get("gain_per_stack_entry")
    if not isinstance(entries, list):
        return None
    total = 0.0
    found = False
    for e in entries:
        if isinstance(e, dict) and e.get("action") == "framework_pr":
            delta = e.get("delta_pct")
            if isinstance(delta, (int, float)):
                total += float(delta)
                found = True
    if not found:
        return None
    return round(total, 2)


# ---------------------------------------------------------------------------
# Workload-dim defaults and framework_args parsing
#
# Most JSONs leave workload.tp/isl/... null (launcher arg string is the real
# source of truth); we parse it and fall back to platform defaults. Duplicates
# scripts/fix_null_fields.py:derive_fields().
# ---------------------------------------------------------------------------

DEFAULT_DURATION_SECONDS = 10800  # 3 hours
DEFAULT_ISL = 1024
DEFAULT_OSL = 1024
DEFAULT_CONC = 64
DEFAULT_TP = 1
DEFAULT_PREC = "fp8"

_RE_TP_IN_ARGS = re.compile(
    r"(?:"
    r"\bTP\s*=\s*"                                      # env style: TP=8
    r"|\btensor[_-]parallel[_-]size\s*=\s*"             # vllm kwargs
    r"|\btp_size\s*=\s*"                                # sglang ServerArgs
    r"|--tp\s+|--tp=\s*"                                # cmdline
    r"|--tensor[_-]parallel[_-]size[=\s]+"
    r")(\d+)",
    re.IGNORECASE,
)
_RE_CONC_IN_ARGS = re.compile(r"\bCONC\s*=\s*(\d+)", re.IGNORECASE)
_RE_ISL_IN_ARGS = re.compile(r"\bISL\s*=\s*(\d+)", re.IGNORECASE)
_RE_OSL_IN_ARGS = re.compile(r"\bOSL\s*=\s*(\d+)", re.IGNORECASE)
_RE_PREC_IN_ARGS = re.compile(r"\bprecision\s*=\s*([A-Za-z0-9]+)", re.IGNORECASE)
_RE_MODEL_SIZE_TOKEN = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)[Bb](?![A-Za-z])")


def _collect_args_text(data: Dict) -> str:
    """Concatenate every place where launcher args might live.

    Args:
        data (Dict): Full session breakdown JSON.

    Returns:
        str: A ``" | "``-joined string of all ``framework_args`` values found
            under baseline/final/workload invocation blocks.
    """
    parts: list[str] = []
    for path in (
        ("baseline", "invocation", "framework_args"),
        ("final",    "invocation", "framework_args"),
        ("workload", "invocation", "framework_args"),
    ):
        v = safe_get(data, *path)
        if isinstance(v, str) and v:
            parts.append(v)
    return " | ".join(parts)


def _parse_int_from_args(rx: re.Pattern, text: str) -> Optional[int]:
    """Search ``text`` with a regex and return its first group as an int.

    Args:
        rx (re.Pattern): Compiled regex whose first capture group is numeric.
        text (str): Text to search (typically the joined launcher args).

    Returns:
        Optional[int]: The parsed integer, or None when there is no match or
            the captured group is not an integer.
    """
    m = rx.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except (ValueError, TypeError):
        return None


def _infer_prec_from_name(name: str) -> str:
    """Infer precision from quantization hints in a model name.

    Args:
        name (str): Model name to inspect.

    Returns:
        str: ``"int4"``, ``"fp4"``, or ``"bf16"`` when the name carries a
            matching marker, otherwise the platform default precision.
    """
    n = (name or "").lower()
    if any(t in n for t in ("awq", "gptq", "int4", "i4")):
        return "int4"
    if "fp4" in n:
        return "fp4"
    if "bf16" in n:
        return "bf16"
    return DEFAULT_PREC


def _infer_tp_from_name(name: str) -> int:
    """Infer a tensor-parallel size from the parameter size in a model name.

    Size-tier fallback: ``<=70B -> 1``, ``70-180B -> 4``, ``>180B -> 8``. Also
    recognises spelled-out very-large families (``kimi-k2`` / ``deepseek-r1`` /
    ``deepseek-v3`` / ``1T``) and maps them to 8.

    Args:
        name (str): Model name to inspect for a ``<N>B`` size token.

    Returns:
        int: The inferred tensor-parallel size, or the platform default when no
            size signal is found.
    """
    if not name:
        return DEFAULT_TP
    sizes: list[float] = []
    for m in _RE_MODEL_SIZE_TOKEN.finditer(name):
        try:
            sizes.append(float(m.group(1)))
        except ValueError:
            pass
    if not sizes:
        n_lower = name.lower()
        if any(t in n_lower for t in ("kimi-k2", "1t-", "-1t", "deepseek-r1", "deepseek-v3")):
            return 8
        return DEFAULT_TP
    biggest = max(sizes)
    if biggest <= 70:
        return 1
    if biggest <= 180:
        return 4
    return 8


def resolve_workload_dims(data: Dict) -> Dict[str, Any]:
    """Resolve finalised workload dimensions for a session.

    Determines values for ``prec``/``tp``/``isl``/``osl``/``conc``/
    ``duration_seconds`` using this precedence per field:

      1. ``workload.<field>`` if set
      2. parsed from ``framework_args`` (TP=, CONC=, ISL=, OSL=, precision=)
      3. model-name inference (prec/tp only)
      4. platform default

    Args:
        data (Dict): Full session breakdown JSON.

    Returns:
        Dict[str, Any]: Mapping with keys ``prec``, ``tp``, ``isl``, ``osl``,
            ``conc``, and ``duration_seconds``.
    """
    workload = data.get("workload") or {}
    session = data.get("session") or {}
    args = _collect_args_text(data)
    model_name = (workload.get("model_name") or "").strip()

    # prec
    raw_prec = workload.get("precision")
    if isinstance(raw_prec, str) and raw_prec.strip():
        prec = raw_prec.lower()
    else:
        parsed_prec = _RE_PREC_IN_ARGS.search(args)
        if parsed_prec:
            prec = parsed_prec.group(1).lower()
        else:
            prec = _infer_prec_from_name(model_name)

    # tp
    raw_tp = workload.get("tp")
    if isinstance(raw_tp, (int, float)) and raw_tp:
        tp = int(raw_tp)
    else:
        parsed_tp = _parse_int_from_args(_RE_TP_IN_ARGS, args)
        tp = parsed_tp if parsed_tp is not None else _infer_tp_from_name(model_name)

    # isl / osl / conc
    def _pick(field_name: str, rx: re.Pattern, default: int) -> int:
        """Pick an int workload dim from workload field, args, or default.

        Args:
            field_name (str): Key to read from the ``workload`` dict.
            rx (re.Pattern): Regex used to parse the value from launcher args.
            default (int): Platform default used as the final fallback.

        Returns:
            int: The resolved integer value.
        """
        raw = workload.get(field_name)
        if isinstance(raw, (int, float)) and raw:
            return int(raw)
        parsed = _parse_int_from_args(rx, args)
        return parsed if parsed is not None else default

    isl  = _pick("isl",  _RE_ISL_IN_ARGS,  DEFAULT_ISL)
    osl  = _pick("osl",  _RE_OSL_IN_ARGS,  DEFAULT_OSL)
    conc = _pick("conc", _RE_CONC_IN_ARGS, DEFAULT_CONC)

    # duration_seconds
    sds = session.get("session_duration_seconds")
    if isinstance(sds, (int, float)) and sds > 0:
        duration_seconds = int(sds)
    else:
        duration_seconds = DEFAULT_DURATION_SECONDS

    return {
        "prec":             prec,
        "tp":               tp,
        "isl":              isl,
        "osl":              osl,
        "conc":             conc,
        "duration_seconds": duration_seconds,
    }


def extract_row(data: Dict) -> Dict[str, Any]:
    """Convert a session breakdown JSON into a perf_runs DB row dict.

    Cleans the model name, derives image/category/precision/dims, computes the
    gain (preferring ``final.cumulative_gain_pct_validated`` then a raw
    throughput delta), normalises missing attribution gains to 0.00, clips
    negative gains to 0, and attaches the enriched ``raw_data`` payload.

    Args:
        data (Dict): Full (V2-shaped) session breakdown JSON.

    Returns:
        Dict[str, Any]: A row dict whose keys map to the perf_runs table
            columns, including the nested ``raw_data`` JSON.
    """
    workload = data.get("workload") or {}
    baseline = data.get("baseline") or {}
    final = data.get("final") or {}
    session = data.get("session") or {}
    attribution = data.get("attribution") or {}
    attribution_src = attribution.get("source_breakdown") or {}

    # model_name sometimes arrives as a filesystem path; clean prefixes so the
    # leaderboard shows a HF-style name and unique_key stays comparable.
    model_name = (_clean_model_name(workload.get("model_name"))
                  or (workload.get("model_name") or "").strip())
    framework_name = (workload.get("framework") or "").strip()
    framework_ver = (workload.get("framework_version") or "").strip()
    framework_label = framework_name if not framework_ver else f"{framework_name} {framework_ver}"

    image = derive_image(framework_name, session.get("image"))

    baseline_tput = baseline.get("throughput_tok_s_per_gpu")
    opt_tput = final.get("throughput_tok_s_per_gpu")

    # Resolve normally-null columns (prec/tp/isl/osl/conc/duration) by parsing
    # launcher args + platform defaults; replaces scripts/fix_null_fields.py.
    dims = resolve_workload_dims(data)

    code_rev = session.get("code_revision") or ""

    # gain: prefer final.cumulative_gain_pct_validated (trusted end-to-end metric
    # for ranking); fall back to raw throughput delta % for legacy sessions.
    validated_gain = final.get("cumulative_gain_pct_validated")
    throughput_delta_pct = compute_gain_pct(baseline_tput, opt_tput)
    if isinstance(validated_gain, (int, float)):
        gain_value = round(float(validated_gain), 2)
        gain_source = "validated"
    elif throughput_delta_pct is not None:
        gain_value = throughput_delta_pct
        gain_source = "throughput_delta"
    else:
        gain_value = None
        gain_source = "missing"

    row: Dict[str, Any] = {
        "model_name":                truncate(model_name, 255),
        "framework":                 truncate(framework_label, 64),
        "image":                     truncate(image, 512),
        "category":                  truncate(detect_category(data), 32),
        "prec":                      truncate(dims["prec"], 32),
        "gain":                      gain_value,
        "roofline":                  extract_roofline(data),
        "duration_seconds":          dims["duration_seconds"],
        "kernel_gain":               compute_kernel_gain(attribution_src),
        "param_gain":                compute_param_gain(attribution_src),
        "backend_gain":              compute_backend_gain(attribution_src),
        "geak_gain":                 compute_geak_gain(attribution_src),
        "oob_gain":                  compute_oob_gain(attribution_src),
        "framework_gain":            compute_framework_gain(attribution),
        "baseline_tok_per_s_per_gpu": baseline_tput,
        "opt_tok_per_s_per_gpu":     opt_tput,
        "tp":                        dims["tp"],
        "isl":                       dims["isl"],
        "osl":                       dims["osl"],
        "conc":                      dims["conc"],
        # post_perf_runs.py's build_body rejects incomplete/baseline-failed
        # sessions, so anything POSTed is effectively a successful run.
        "status":                    "success",
        "version":                   truncate(code_rev, 64),
        "unique_key":                truncate(derive_unique_key(model_name, image), 128),
        "claw_session_id":           truncate(session.get("claw_session_id"), 128),
    }

    # Derivation provenance, persisted by enrich_raw_data.
    row["_meta"] = {
        "gain_source":           gain_source,
        "validated_gain_pct":    float(validated_gain) if isinstance(validated_gain, (int, float)) else None,
        "throughput_delta_pct":  throughput_delta_pct,
    }

    # Force missing attribution gains to 0.00 (never NULL) and round to 2dp for
    # the NUMERIC(8,2) target column.
    for key in ("kernel_gain", "param_gain", "backend_gain",
                "geak_gain", "oob_gain", "framework_gain"):
        v = row[key]
        if v is None:
            row[key] = 0.00
        elif isinstance(v, (int, float)):
            row[key] = round(float(v), 2)

    # Clip negative gains to 0 (only when present and < 0). Process-history paths
    # in raw_data keep negatives (meaningful for REVERT/NEEDS_REVIEW);
    # enrich_raw_data() mirrors this clip only on display-facing nested paths.
    for key in ("gain", "kernel_gain", "param_gain", "backend_gain",
                "geak_gain", "oob_gain", "framework_gain"):
        v = row.get(key)
        if isinstance(v, (int, float)) and v < 0:
            row[key] = 0.0

    # raw_data = original JSON + `_enrichment`; pop `_meta` before the SQL stage.
    meta = row.pop("_meta", {}) or {}
    row["raw_data"] = enrich_raw_data(data, row, meta=meta)

    return row


def format_duration_pretty(seconds: Optional[int]) -> str:
    """Format a duration in seconds as a compact ``<h>h<mm>m<ss>s`` string.

    Args:
        seconds (Optional[int]): Duration in seconds, or None.

    Returns:
        str: ``"n/a"`` when ``seconds`` is None, otherwise the formatted
            duration (e.g. ``"3h00m00s"``).
    """
    if seconds is None:
        return "n/a"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def enrich_raw_data(original: Dict, row: Dict[str, Any], meta: Optional[Dict] = None) -> Dict:
    """Build the JSON persisted into perf_runs.raw_data: deep-copy the original,
    backfill session.image when missing, and add a top-level `_enrichment` block
    of inferred/computed values (`meta` carries derivation provenance)."""
    enriched = copy.deepcopy(original)

    session = enriched.get("session")
    if not isinstance(session, dict):
        session = {}
        enriched["session"] = session
    session_image_was_present = bool(session.get("image"))
    if not session_image_was_present and row.get("image"):
        session["image"] = row["image"]

    session_duration_was_present = bool(session.get("session_duration_seconds"))
    if not session_duration_was_present and row.get("duration_seconds"):
        session["session_duration_seconds"] = row["duration_seconds"]

    workload = enriched.get("workload")
    if not isinstance(workload, dict):
        workload = {}
        enriched["workload"] = workload
    workload_fallbacks_applied: Dict[str, Any] = {}
    for wl_key, row_key in (
        ("tp",        "tp"),
        ("isl",       "isl"),
        ("osl",       "osl"),
        ("conc",      "conc"),
        ("precision", "prec"),
    ):
        if not workload.get(wl_key):
            workload[wl_key] = row[row_key]
            workload_fallbacks_applied[wl_key] = row[row_key]

    # Mirror the column-level gain clip into display-facing raw_data paths so UIs
    # see the same non-negative number as the column (only when present and < 0).
    # Process-history paths are intentionally left untouched.
    raw_clip_applied: Dict[str, Any] = {}

    def _clip_neg(container: Dict, key: str, full_path: str) -> None:
        """Clip a negative numeric field to 0 in-place and record the change.

        Args:
            container (Dict): Dict holding the value to clip.
            key (str): Key whose numeric value should be clipped.
            full_path (str): Dotted path used as the key in the
                ``raw_clip_applied`` provenance map.
        """
        v = container.get(key)
        if isinstance(v, (int, float)) and v < 0:
            container[key] = 0
            raw_clip_applied[full_path] = float(v)

    final_block = enriched.get("final")
    if isinstance(final_block, dict):
        _clip_neg(final_block, "cumulative_gain_pct_validated",     "final.cumulative_gain_pct_validated")
        _clip_neg(final_block, "cumulative_gain_pct_per_round_sum", "final.cumulative_gain_pct_per_round_sum")
        _clip_neg(final_block, "e2e_gain_pct",                       "final.e2e_gain_pct")

    attr_block = enriched.get("attribution")
    if isinstance(attr_block, dict):
        src_block = attr_block.get("source_breakdown")
        if isinstance(src_block, dict):
            _clip_neg(src_block, "oob_pct_of_total",  "attribution.source_breakdown.oob_pct_of_total")
            _clip_neg(src_block, "geak_pct_of_total", "attribution.source_breakdown.geak_pct_of_total")

    meta = meta or {}
    enrichment = {
        "imported_at_utc":            datetime.now(timezone.utc).isoformat(),
        "importer":                   IMPORTER_NAME,
        "importer_version":           IMPORTER_VERSION,
        "unique_key":                 row["unique_key"],
        "category":                   row["category"],
        "image_used":                 row["image"],
        "image_fallback_applied":     not session_image_was_present,
        "duration_fallback_applied":  not session_duration_was_present,
        "workload_fallbacks_applied": workload_fallbacks_applied,
        "gain_pct":                   row["gain"],
        "gain_source":                meta.get("gain_source"),
        "validated_gain_pct":         meta.get("validated_gain_pct"),
        "throughput_delta_pct":       meta.get("throughput_delta_pct"),
        "kernel_gain_pct":            row["kernel_gain"],
        "param_gain_pct":             row["param_gain"],
        "backend_gain_pct":           row["backend_gain"],
        "geak_gain_pct":              row["geak_gain"],
        "oob_gain_pct":               row["oob_gain"],
        "framework_gain_pct":         row["framework_gain"],
        "raw_data_neg_gain_clipped":  raw_clip_applied,
        "duration_seconds":           row["duration_seconds"],
        "duration_pretty":            format_duration_pretty(row["duration_seconds"]),
        "baseline_tput":              row["baseline_tok_per_s_per_gpu"],
        "opt_tput":                   row["opt_tok_per_s_per_gpu"],
        "status":                     row["status"],
        "stop_reason":                session.get("stop_reason"),
        "version":                    row["version"],
        "claw_session_id":            row.get("claw_session_id"),
    }
    enriched["_enrichment"] = enrichment
    return enriched


# ---------------------------------------------------------------------------
# Table name validation
# ---------------------------------------------------------------------------

_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_table(name: str) -> str:
    """Validate that a table name is a plain SQL identifier.

    Args:
        name (str): Proposed table name.

    Returns:
        str: The same ``name`` when it is a valid identifier.

    Raises:
        ValueError: If ``name`` is not a plain ``[A-Za-z_][A-Za-z0-9_]*``
            identifier (defence in depth against SQL injection).
    """
    if not _SAFE_IDENT.match(name):
        raise ValueError(f"Refusing unsafe table name: {name!r}")
    return name


# ---------------------------------------------------------------------------
# SQL builders
# ---------------------------------------------------------------------------

def build_upsert_sql(table: str) -> str:
    """Build the parameterised INSERT ... ON CONFLICT upsert for local mode.

    Args:
        table (str): Destination table name (validated via :func:`safe_table`).

    Returns:
        str: A psycopg2-style SQL statement using ``%(name)s`` placeholders that
            upserts on ``unique_key`` and returns the row id plus an inserted
            flag.
    """
    table = safe_table(table)
    return f"""
INSERT INTO {table} (
    model_name, framework, image, category, prec, gain, roofline,
    duration_seconds, kernel_gain, param_gain, backend_gain,
    geak_gain, oob_gain, framework_gain,
    baseline_tok_per_s_per_gpu, opt_tok_per_s_per_gpu,
    tp, isl, osl, conc, status, raw_data, version, unique_key,
    claw_session_id
) VALUES (
    %(model_name)s, %(framework)s, %(image)s, %(category)s, %(prec)s, %(gain)s, %(roofline)s,
    %(duration_seconds)s, %(kernel_gain)s, %(param_gain)s, %(backend_gain)s,
    %(geak_gain)s, %(oob_gain)s, %(framework_gain)s,
    %(baseline_tok_per_s_per_gpu)s, %(opt_tok_per_s_per_gpu)s,
    %(tp)s, %(isl)s, %(osl)s, %(conc)s, %(status)s, %(raw_data)s, %(version)s, %(unique_key)s,
    %(claw_session_id)s
)
ON CONFLICT (unique_key) DO UPDATE SET
    model_name                 = EXCLUDED.model_name,
    framework                  = EXCLUDED.framework,
    image                      = EXCLUDED.image,
    category                   = EXCLUDED.category,
    prec                       = EXCLUDED.prec,
    gain                       = EXCLUDED.gain,
    roofline                   = EXCLUDED.roofline,
    duration_seconds           = EXCLUDED.duration_seconds,
    kernel_gain                = EXCLUDED.kernel_gain,
    param_gain                 = EXCLUDED.param_gain,
    backend_gain               = EXCLUDED.backend_gain,
    geak_gain                  = EXCLUDED.geak_gain,
    oob_gain                   = EXCLUDED.oob_gain,
    framework_gain             = EXCLUDED.framework_gain,
    baseline_tok_per_s_per_gpu = EXCLUDED.baseline_tok_per_s_per_gpu,
    opt_tok_per_s_per_gpu      = EXCLUDED.opt_tok_per_s_per_gpu,
    tp                         = EXCLUDED.tp,
    isl                        = EXCLUDED.isl,
    osl                        = EXCLUDED.osl,
    conc                       = EXCLUDED.conc,
    status                     = EXCLUDED.status,
    raw_data                   = EXCLUDED.raw_data,
    version                    = EXCLUDED.version,
    claw_session_id            = EXCLUDED.claw_session_id,
    updated_at                 = now()
RETURNING id, (xmax = 0) AS inserted;
"""


def build_create_unique_index_sql(table: str) -> str:
    """Build the ``CREATE UNIQUE INDEX IF NOT EXISTS`` SQL for ``unique_key``.

    Args:
        table (str): Destination table name (validated via :func:`safe_table`).

    Returns:
        str: SQL that ensures a unique index on ``<table>(unique_key)`` so the
            upsert's ``ON CONFLICT (unique_key)`` clause works.
    """
    table = safe_table(table)
    return (
        f"CREATE UNIQUE INDEX IF NOT EXISTS {table}_unique_key_idx "
        f"ON {table}(unique_key);"
    )


# ---------------------------------------------------------------------------
# MODE A: local psycopg2
# ---------------------------------------------------------------------------

def connect_local(args: argparse.Namespace):
    """Open a psycopg2 connection for local mode.

    Prefers ``$PERF_RUNS_DB_URL`` / ``--db-url`` when set; otherwise builds the
    connection from the individual host/port/user/dbname (and optional
    password) flags.

    Args:
        args (argparse.Namespace): Parsed CLI args carrying the DB connection
            settings.

    Returns:
        psycopg2.extensions.connection: An open database connection.
    """
    pg, _ = _import_psycopg2()
    url = os.environ.get("PERF_RUNS_DB_URL") or args.db_url
    if url:
        return pg.connect(url)
    kwargs = dict(host=args.host, port=args.port,
                  user=args.user, dbname=args.dbname)
    if args.password:
        kwargs["password"] = args.password
    return pg.connect(**kwargs)


def upsert_row_local(conn, row: Dict[str, Any], sql: str) -> Tuple[int, bool]:
    """Execute the upsert for a single row over a local psycopg2 connection.

    Wraps JSON columns (``roofline``/``raw_data``) in ``psycopg2.extras.Json``
    and commits the transaction.

    Args:
        conn: An open psycopg2 connection.
        row (Dict[str, Any]): The row dict produced by :func:`extract_row`.
        sql (str): The upsert SQL from :func:`build_upsert_sql`.

    Returns:
        Tuple[int, bool]: The row id and a flag that is True when the row was
            inserted (False when it updated an existing row).
    """
    _, Json = _import_psycopg2()
    payload = dict(row)
    if payload.get("roofline") is not None:
        payload["roofline"] = Json(payload["roofline"])
    payload["raw_data"] = Json(payload["raw_data"])

    with conn.cursor() as cur:
        cur.execute(sql, payload)
        new_id, inserted = cur.fetchone()
    conn.commit()
    return new_id, bool(inserted)


# ---------------------------------------------------------------------------
# MODE B: ssh + kubectl exec + psql (no local DB driver needed)
# ---------------------------------------------------------------------------

def _dollar_tag(content: str, base: str = "perf") -> str:
    """Find a PostgreSQL dollar-quote tag that does not appear in ``content``.

    Args:
        content (str): The string the tag will wrap; the chosen tag is
            guaranteed not to occur within it.
        base (str): Base label embedded in the tag for readability.

    Returns:
        str: A ``$<base>_<nonce>$`` tag safe to use as a dollar-quote delimiter.
    """
    for _ in range(8):
        tag = f"${base}_{secrets.token_hex(4)}$"
        if tag not in content:
            return tag
    # Pathological case: nest a random nonce inside
    return f"${base}_{secrets.token_hex(16)}$"


def _sql_str_literal(s: Optional[str]) -> str:
    """Render a value as a safe dollar-quoted SQL string literal.

    Args:
        s (Optional[str]): The value to render, or None.

    Returns:
        str: ``"NULL"`` when ``s`` is None, otherwise the value wrapped in a
            unique dollar-quote tag (no escaping required).
    """
    if s is None:
        return "NULL"
    s = str(s)
    tag = _dollar_tag(s, base="s")
    return f"{tag}{s}{tag}"


def _sql_number_literal(v: Any) -> str:
    """Render a numeric/bool value as a SQL literal.

    Args:
        v (Any): A number, bool, or None.

    Returns:
        str: ``"NULL"`` for None, ``"TRUE"``/``"FALSE"`` for booleans, or the
            numeric value rendered as a float/int literal.
    """
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return repr(float(v)) if isinstance(v, float) else str(int(v))


def _sql_jsonb_literal(obj: Any) -> str:
    """Render a Python object as a dollar-quoted ``::jsonb`` SQL literal.

    Args:
        obj (Any): Any JSON-serialisable object, or None.

    Returns:
        str: ``"NULL"`` when ``obj`` is None, otherwise the JSON-encoded value
            wrapped in a unique dollar-quote tag and cast to ``jsonb``.
    """
    if obj is None:
        return "NULL"
    js = json.dumps(obj, ensure_ascii=False)
    tag = _dollar_tag(js, base="j")
    return f"{tag}{js}{tag}::jsonb"


def build_upsert_statement_inline(row: Dict[str, Any], table: str) -> str:
    """Build one SQL statement with all values inlined (dollar-quoted)."""
    table = safe_table(table)
    parts = (
        f"INSERT INTO {table} (\n"
        f"  model_name, framework, image, category, prec, gain, roofline,\n"
        f"  duration_seconds, kernel_gain, param_gain, backend_gain,\n"
        f"  geak_gain, oob_gain, framework_gain,\n"
        f"  baseline_tok_per_s_per_gpu, opt_tok_per_s_per_gpu,\n"
        f"  tp, isl, osl, conc, status, raw_data, version, unique_key,\n"
        f"  claw_session_id\n"
        f") VALUES (\n"
        f"  {_sql_str_literal(row['model_name'])},\n"
        f"  {_sql_str_literal(row['framework'])},\n"
        f"  {_sql_str_literal(row['image'])},\n"
        f"  {_sql_str_literal(row['category'])},\n"
        f"  {_sql_str_literal(row['prec'])},\n"
        f"  {_sql_number_literal(row['gain'])},\n"
        f"  {_sql_jsonb_literal(row['roofline'])},\n"
        f"  {_sql_number_literal(row['duration_seconds'])},\n"
        f"  {_sql_number_literal(row['kernel_gain'])},\n"
        f"  {_sql_number_literal(row['param_gain'])},\n"
        f"  {_sql_number_literal(row['backend_gain'])},\n"
        f"  {_sql_number_literal(row['geak_gain'])},\n"
        f"  {_sql_number_literal(row['oob_gain'])},\n"
        f"  {_sql_number_literal(row['framework_gain'])},\n"
        f"  {_sql_number_literal(row['baseline_tok_per_s_per_gpu'])},\n"
        f"  {_sql_number_literal(row['opt_tok_per_s_per_gpu'])},\n"
        f"  {_sql_number_literal(row['tp'])},\n"
        f"  {_sql_number_literal(row['isl'])},\n"
        f"  {_sql_number_literal(row['osl'])},\n"
        f"  {_sql_number_literal(row['conc'])},\n"
        f"  {_sql_str_literal(row['status'])},\n"
        f"  {_sql_jsonb_literal(row['raw_data'])},\n"
        f"  {_sql_str_literal(row['version'])},\n"
        f"  {_sql_str_literal(row['unique_key'])},\n"
        f"  {_sql_str_literal(row.get('claw_session_id'))}\n"
        f")\n"
        f"ON CONFLICT (unique_key) DO UPDATE SET\n"
        f"  model_name                 = EXCLUDED.model_name,\n"
        f"  framework                  = EXCLUDED.framework,\n"
        f"  image                      = EXCLUDED.image,\n"
        f"  category                   = EXCLUDED.category,\n"
        f"  prec                       = EXCLUDED.prec,\n"
        f"  gain                       = EXCLUDED.gain,\n"
        f"  roofline                   = EXCLUDED.roofline,\n"
        f"  duration_seconds           = EXCLUDED.duration_seconds,\n"
        f"  kernel_gain                = EXCLUDED.kernel_gain,\n"
        f"  param_gain                 = EXCLUDED.param_gain,\n"
        f"  backend_gain               = EXCLUDED.backend_gain,\n"
        f"  geak_gain                  = EXCLUDED.geak_gain,\n"
        f"  oob_gain                   = EXCLUDED.oob_gain,\n"
        f"  framework_gain             = EXCLUDED.framework_gain,\n"
        f"  baseline_tok_per_s_per_gpu = EXCLUDED.baseline_tok_per_s_per_gpu,\n"
        f"  opt_tok_per_s_per_gpu      = EXCLUDED.opt_tok_per_s_per_gpu,\n"
        f"  tp                         = EXCLUDED.tp,\n"
        f"  isl                        = EXCLUDED.isl,\n"
        f"  osl                        = EXCLUDED.osl,\n"
        f"  conc                       = EXCLUDED.conc,\n"
        f"  status                     = EXCLUDED.status,\n"
        f"  raw_data                   = EXCLUDED.raw_data,\n"
        f"  version                    = EXCLUDED.version,\n"
        f"  claw_session_id            = EXCLUDED.claw_session_id,\n"
        f"  updated_at                 = now();\n"
    )
    return parts


def build_ssh_kubectl_psql(hop1: str, namespace: str, user: str, dbname: str) -> List[str]:
    """Return argv to invoke psql on the master postgres pod via SSH+kubectl (stdin piped)."""
    remote_cmd = (
        f"kubectl exec -i -n {namespace} "
        f"$(kubectl get pod -n {namespace} "
        f"-l postgres-operator.crunchydata.com/role=master -o name) "
        f"-- psql -v ON_ERROR_STOP=1 -X -q "
        f"-U {user} {dbname}"
    )
    return [
        "ssh",
        "-o", "StrictHostKeyChecking=accept-new",
        hop1,
        remote_cmd,
    ]


def execute_remote_sql(sql: str, *, hop1: str, namespace: str, user: str, dbname: str) -> Tuple[str, str]:
    """Pipe SQL through SSH+kubectl ``psql`` and capture its output.

    Args:
        sql (str): SQL text to send on stdin.
        hop1 (str): SSH jump host.
        namespace (str): Kubernetes namespace of the postgres pod.
        user (str): PostgreSQL role to connect as.
        dbname (str): Database name to connect to.

    Returns:
        Tuple[str, str]: The captured ``(stdout, stderr)`` decoded as UTF-8.

    Raises:
        RuntimeError: If the remote ``psql`` process exits non-zero.
    """
    argv = build_ssh_kubectl_psql(hop1, namespace, user, dbname)
    proc = subprocess.run(argv, input=sql.encode("utf-8"), capture_output=True)
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"psql via ssh+kubectl failed (exit={proc.returncode}):\n"
            f"--- stderr ---\n{err}\n--- stdout ---\n{out}\n"
        )
    return out, err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def iter_json_files(paths: Iterable[str], scan_dir: Optional[str]) -> List[Path]:
    """Expand input paths and an optional scan dir into a unique file list.

    Directories are searched recursively for ``*.json``; missing paths emit a
    warning. The result is de-duplicated while preserving order.

    Args:
        paths (Iterable[str]): Files and/or directories passed on the CLI.
        scan_dir (Optional[str]): Extra directory to scan recursively for
            ``*.json``, or None.

    Returns:
        List[Path]: Ordered, de-duplicated list of JSON file paths.
    """
    files: List[Path] = []
    for p in paths or []:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.rglob("*.json")))
        elif pp.exists():
            files.append(pp)
        else:
            sys.stderr.write(f"[WARN] path not found: {p}\n")
    if scan_dir:
        files.extend(sorted(Path(scan_dir).rglob("*.json")))
    # dedupe while preserving order
    seen = set()
    unique = []
    for f in files:
        rf = f.resolve()
        if rf not in seen:
            seen.add(rf)
            unique.append(f)
    return unique


def looks_like_session_breakdown(data: Any) -> bool:
    """Heuristically test whether ``data`` is a canonical V2 session breakdown.

    Args:
        data (Any): Parsed JSON object.

    Returns:
        bool: True when ``data`` is a dict carrying workload/baseline/session
            keys (or a ``session_breakdown`` schema_version marker).
    """
    if not isinstance(data, dict):
        return False
    return ("workload" in data and "baseline" in data and "session" in data) or \
           "schema_version" in data and "session_breakdown" in str(data.get("schema_version", ""))


def looks_like_v1_flat_schema(data: Any) -> bool:
    """Recognise the legacy 'V1 flat' layout: top-level dict with model,
    framework, baseline_tput and best_tput (not nested under workload/baseline)."""
    if not isinstance(data, dict):
        return False
    return (
        "model" in data
        and "framework" in data
        and "baseline_tput" in data
        and "best_tput" in data
        and "workload" not in data
        and "baseline" not in data
    )


# ---------------------------------------------------------------------------
# Universal migrator -- handles every schema variant we've seen in the wild
# ---------------------------------------------------------------------------

def _deep_get(d: Any, *path, default=None):
    """Descend nested dicts by a sequence of keys, tolerating gaps.

    Args:
        d (Any): Starting object, typically a dict.
        *path: Successive keys to descend through.
        default: Value returned when any level is missing/not a dict/None.

    Returns:
        Any: The value at the nested path, otherwise ``default``.
    """
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _first_truthy(*vals):
    """Return the first value that is neither None nor an empty string.

    Args:
        *vals: Candidate values in priority order.

    Returns:
        Any: The first non-None, non-empty value, or None if none qualify.
    """
    for v in vals:
        if v is not None and v != "":
            return v
    return None


def _is_pos_number(v: Any) -> bool:
    """Test whether a value is a strictly positive (non-bool) number.

    Args:
        v (Any): Value to test.

    Returns:
        bool: True when ``v`` is an int/float (not a bool) greater than 0.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _clean_model_name(raw: Any) -> Optional[str]:
    """Strip storage path prefixes to recover a HuggingFace-style model name.

    V1.5 sometimes stores ``/wekafs/models/<org-repo>`` as ``model``; this
    removes known storage prefixes and surrounding slashes.

    Args:
        raw (Any): Candidate model name (possibly a storage path).

    Returns:
        Optional[str]: The cleaned model name, or None when ``raw`` is not a
            usable non-empty string.
    """
    if not isinstance(raw, str) or not raw:
        return None
    s = raw.strip()
    for prefix in ("/wekafs/models/", "/workspace/", "/data/models/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.strip("/") or None


def looks_like_universal_schema(data: Any) -> bool:
    """Loosely test whether ``data`` carries a model name and framework.

    Accepts any dict that has *both* an identifiable model name and framework,
    even if workload/baseline/session are missing or under non-standard keys.
    The authoritative test remains :func:`migrate_universal_to_v2`.

    Args:
        data (Any): Parsed JSON object.

    Returns:
        bool: True when both a model name and framework can be located.
    """
    if not isinstance(data, dict):
        return False
    model = _first_truthy(
        data.get("model_name"),
        data.get("model"),
        _deep_get(data, "workload", "model_name"),
        _deep_get(data, "workload", "model"),
    )
    framework = _first_truthy(
        data.get("framework"),
        _deep_get(data, "workload", "framework"),
    )
    return bool(model and framework)


def migrate_universal_to_v2(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Best-effort migrator from any non-canonical session_breakdown layout to a
    V2-compatible dict. Returns None if essential numerics can't be recovered.

    Sources checked (in priority order):
      baseline per-gpu throughput:
        baseline.throughput_tok_s_per_gpu              (V2)
        baseline.throughput_output_tps_per_gpu         (V1.5 bielik)
        baseline_tok_per_s_per_gpu                     (V1.5 minimax)
        baseline_throughput                            (V1.5 various)
        final.baseline_throughput                      (V1.5 qwen3-30b-a3b-2507)
        baseline_tput / tp                             (V1 flat, totals)
        baseline.output_throughput / tp                (alternative totals)
        baseline.benchmark.output_throughput / tp

      opt per-gpu throughput:
        final.throughput_tok_s_per_gpu
        final.throughput_output_tps_per_gpu
        final.optimized_throughput                     (V1.5 bielik)
        best.throughput_tok_s_per_gpu
        optimized_throughput / best_throughput
        opt_tok_per_s_per_gpu
        best_tput / tp                                 (V1 flat totals)
        (else fallback to baseline -- represents 'no optimisation found')

      gain pct:
        final.cumulative_gain_pct_validated, final.gain_pct,
        cumulative_gain_*_validated, cumulative_gain_pct, gain_pct,
        winner_gain_pct, best.gain_pct
        (else recompute (opt-base)/base*100)

    Args:
        raw (Dict[str, Any]): A session breakdown dict in any known layout.

    Returns:
        Optional[Dict[str, Any]]: A V2-shaped dict (workload/baseline/final/
            session, plus ``_universal_source`` provenance), or None when the
            essential model name or baseline throughput cannot be recovered.
    """
    if not isinstance(raw, dict):
        return None

    model_name = _clean_model_name(_first_truthy(
        raw.get("model_name"),
        raw.get("model"),
        _deep_get(raw, "workload", "model_name"),
        _deep_get(raw, "workload", "model"),
    ))
    if not model_name:
        return None

    framework_raw = _first_truthy(
        raw.get("framework"),
        _deep_get(raw, "workload", "framework"),
    )
    if not isinstance(framework_raw, str) or not framework_raw:
        return None
    framework = framework_raw.lower()

    tp_raw = _first_truthy(
        raw.get("tp"),
        _deep_get(raw, "workload", "tp"),
        raw.get("num_gpus"),
        raw.get("gpu_count"),
    )
    try:
        tp = int(tp_raw) if tp_raw else 1
    except (TypeError, ValueError):
        tp = 1
    if tp <= 0:
        tp = 1

    # baseline per-gpu
    baseline_per_gpu: Optional[float] = None
    for v in (
        _deep_get(raw, "baseline", "throughput_tok_s_per_gpu"),
        _deep_get(raw, "baseline", "throughput_output_tps_per_gpu"),
        raw.get("baseline_tok_per_s_per_gpu"),
        raw.get("baseline_throughput"),
        _deep_get(raw, "final", "baseline_throughput"),
    ):
        if _is_pos_number(v):
            baseline_per_gpu = float(v)
            break
    if baseline_per_gpu is None:
        for v_total in (
            raw.get("baseline_tput"),
            _deep_get(raw, "baseline", "output_throughput"),
            _deep_get(raw, "baseline", "benchmark", "output_throughput"),
        ):
            if _is_pos_number(v_total):
                baseline_per_gpu = float(v_total) / tp
                break
    if baseline_per_gpu is None or baseline_per_gpu <= 0:
        return None

    # opt per-gpu (fall back to baseline -> 0% gain)
    opt_per_gpu: Optional[float] = None
    for v in (
        _deep_get(raw, "final", "throughput_tok_s_per_gpu"),
        _deep_get(raw, "final", "throughput_output_tps_per_gpu"),
        _deep_get(raw, "final", "optimized_throughput"),
        _deep_get(raw, "best", "throughput_tok_s_per_gpu"),
        raw.get("optimized_throughput"),
        raw.get("best_throughput"),
        raw.get("opt_tok_per_s_per_gpu"),
    ):
        if _is_pos_number(v):
            opt_per_gpu = float(v)
            break
    if opt_per_gpu is None:
        for v_total in (
            raw.get("best_tput"),
            _deep_get(raw, "final", "output_throughput"),
            _deep_get(raw, "final", "benchmark", "output_throughput"),
        ):
            if _is_pos_number(v_total):
                opt_per_gpu = float(v_total) / tp
                break
    if opt_per_gpu is None:
        opt_per_gpu = baseline_per_gpu

    # gain pct
    gain: Optional[float] = None
    for path in (
        ("final", "cumulative_gain_pct_validated"),
        ("final", "gain_pct"),
        ("cumulative_gain_validated",),
        ("cumulative_gain_pct_validated",),
        ("cumulative_gain_pct",),
        ("cumulative_gain",),
        ("gain_pct",),
        ("winner_gain_pct",),
        ("best", "gain_pct"),
    ):
        v = _deep_get(raw, *path)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            gain = float(v)
            break
    if gain is None:
        gain = (opt_per_gpu - baseline_per_gpu) / baseline_per_gpu * 100.0

    # workload-dim defaults
    isl = _first_truthy(raw.get("isl"), _deep_get(raw, "workload", "isl"))
    osl = _first_truthy(raw.get("osl"), _deep_get(raw, "workload", "osl"))
    conc = _first_truthy(raw.get("conc"), raw.get("concurrency"),
                         _deep_get(raw, "workload", "conc"),
                         _deep_get(raw, "workload", "concurrency"))
    precision_raw = _first_truthy(raw.get("precision"),
                                  _deep_get(raw, "workload", "precision"))
    precision = precision_raw.lower() if isinstance(precision_raw, str) else None

    # duration
    duration_s: Optional[int] = None
    for v in (
        _deep_get(raw, "session", "session_duration_seconds"),
        raw.get("session_duration_seconds"),
    ):
        if _is_pos_number(v):
            duration_s = int(v); break
    if duration_s is None:
        for minutes_field in ("wall_clock_minutes", "elapsed_minutes",
                              "elapsed_minutes_before_sandbox_restart",
                              "budget_used_minutes"):
            v = raw.get(minutes_field)
            if _is_pos_number(v):
                duration_s = int(v * 60); break
    if duration_s is None:
        for hours_field in ("budget_used_hours", "wall_clock_hours"):
            v = raw.get(hours_field)
            if _is_pos_number(v):
                duration_s = int(v * 3600); break

    # session id (use as claw_session_id fallback)
    session_id = _first_truthy(
        raw.get("claw_session_id"),
        _deep_get(raw, "session", "claw_session_id"),
        raw.get("session_id"),
    )

    image = _first_truthy(
        raw.get("image"),
        _deep_get(raw, "session", "image"),
    )

    return {
        "workload": {
            "model_name":        model_name,
            "framework":         framework,
            "framework_version": _first_truthy(raw.get("framework_version"),
                                               _deep_get(raw, "workload", "framework_version")) or "",
            "tp":                tp,
            "isl":               isl,
            "osl":               osl,
            "conc":              conc,
            "precision":         precision,
            "invocation": {"framework_args": ""},
        },
        "baseline": {"throughput_tok_s_per_gpu": baseline_per_gpu},
        "final": {
            "throughput_tok_s_per_gpu":      opt_per_gpu,
            "cumulative_gain_pct_validated": gain,
        },
        "session": {
            "session_duration_seconds": duration_s,
            "claw_session_id":          session_id,
            "code_revision":            _first_truthy(
                raw.get("hyperloom_source_commit"),
                raw.get("hyperloom_commit"),
                _deep_get(raw, "session", "code_revision"),
            ) or "",
            "image":                    image,
            "stop_reason":              "",
        },
        "_universal_source": raw,  # provenance
    }


def migrate_v1_to_v2(v1: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a V1 flat session_breakdown to the V2 nested form so it can be fed
    through the normal extract_row pipeline.

    Key conversions:
      * V1 baseline_tput / best_tput are *total* output_throughput across all
        GPUs (verified against airoboros-70b-3.3 note: 'output_throughput=80.22
        tok/s (20.05 tok/s/GPU)' with tp=4). We divide by tp so the V2 row's
        baseline.throughput_tok_s_per_gpu means per-GPU (matching V2 semantics).
      * Image is not present in V1 -- leave session.image unset and let
        derive_image() fall back to DEFAULT_IMAGES[framework].
      * gain_pct goes into final.cumulative_gain_pct_validated since V1 agents
        only ever reported the post-validate cumulative gain.

    Args:
        v1 (Dict[str, Any]): A V1 flat session breakdown dict.

    Returns:
        Dict[str, Any]: The equivalent V2 nested dict, with the original V1
            payload preserved under ``_v1_source``.
    """
    tp_raw = v1.get("tp")
    try:
        tp = int(tp_raw) if tp_raw else 1
    except (TypeError, ValueError):
        tp = 1
    if tp <= 0:
        tp = 1

    def to_per_gpu(total: Any) -> Optional[float]:
        """Convert a cluster-total throughput to a per-GPU value.

        Args:
            total (Any): Total throughput across all GPUs.

        Returns:
            Optional[float]: ``total / tp`` when ``total`` is a positive
                number, otherwise None.
        """
        if isinstance(total, (int, float)) and total > 0:
            return float(total) / tp
        return None

    baseline_per_gpu = to_per_gpu(v1.get("baseline_tput"))
    best_per_gpu = to_per_gpu(v1.get("best_tput"))

    wall_min = v1.get("wall_clock_minutes")
    duration_s: Optional[int] = None
    if isinstance(wall_min, (int, float)) and wall_min > 0:
        duration_s = int(wall_min * 60)

    framework_full = v1.get("framework") or ""
    fw_ver = v1.get("framework_version") or v1.get("framework_ver") or ""
    framework_lower = framework_full.lower() if isinstance(framework_full, str) else ""

    return {
        "workload": {
            "model_name":        v1.get("model"),
            "framework":         framework_lower,
            "framework_version": fw_ver,
            "tp":                tp,
            "ep":                v1.get("ep"),
            "isl":               v1.get("isl"),
            "osl":               v1.get("osl"),
            "conc":              v1.get("conc"),
            "precision":         (v1.get("precision") or "").lower() if isinstance(v1.get("precision"), str) else None,
            "gpu_type":          v1.get("gpu_type"),
            "invocation": {
                "framework_args": "",  # V1 didn't preserve the launcher arg string
            },
        },
        "baseline": {
            "throughput_tok_s_per_gpu": baseline_per_gpu,
            "benchmark": {
                "output_throughput": v1.get("baseline_tput"),  # total, kept for provenance
            },
        },
        "final": {
            "throughput_tok_s_per_gpu":    best_per_gpu,
            "cumulative_gain_pct_validated": v1.get("gain_pct"),
            "benchmark": {
                "output_throughput": v1.get("best_tput"),
            },
        },
        "session": {
            "session_duration_seconds": duration_s,
            "claw_session_id":          v1.get("claw_session_id") or v1.get("session_id"),
            "code_revision":            v1.get("hyperloom_source_commit") or v1.get("hyperloom_commit") or "",
            "image":                    v1.get("image"),  # usually None -> derive_image() falls back
            "stop_reason":              "",
        },
        # Stash the original V1 payload so raw_data preserves provenance.
        "_v1_source": v1,
    }


def parse_file(path: Path) -> Optional[Dict[str, Any]]:
    """Read and validate a JSON file, returning its extracted row.

    Accepts both bare session breakdowns and the ``{"source": ..., "data":
    <breakdown>}`` wrapper. Files that fail to parse or don't look like a
    session breakdown are skipped (with a printed notice).

    Args:
        path (Path): Path to the JSON file.

    Returns:
        Optional[Dict[str, Any]]: The extracted row dict, or None when the file
            is unparseable or not a session breakdown.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] {path}: cannot parse JSON ({e})")
        return None

    if not looks_like_session_breakdown(data):
        # Also accept the v2 wrapper {"source": ..., "data": <breakdown>}.
        if isinstance(data, dict) and isinstance(data.get("data"), dict) and looks_like_session_breakdown(data["data"]):
            data = data["data"]
        else:
            print(f"[SKIP] {path}: does not look like a session breakdown")
            return None

    return extract_row(data)


def format_row_summary(row: Dict[str, Any]) -> str:
    """Render a human-readable multi-line summary of a parsed row.

    Args:
        row (Dict[str, Any]): The row dict produced by :func:`extract_row`.

    Returns:
        str: A formatted summary used in dry-run and import logging.
    """
    return (
        f"  model={row['model_name']!r} framework={row['framework']!r} "
        f"prec={row['prec']!r} category={row['category']!r}\n"
        f"  baseline={row['baseline_tok_per_s_per_gpu']} -> opt={row['opt_tok_per_s_per_gpu']}  "
        f"gain={row['gain']}%  kernel_gain={row['kernel_gain']}%  "
        f"param_gain={row['param_gain']}%  backend_gain={row['backend_gain']}%\n"
        f"  geak_gain={row['geak_gain']}%  oob_gain={row['oob_gain']}%  "
        f"framework_gain={row['framework_gain']}%\n"
        f"  tp={row['tp']} isl={row['isl']} osl={row['osl']} conc={row['conc']}  "
        f"duration={format_duration_pretty(row['duration_seconds'])}\n"
        f"  status={row['status']!r}  claw_session_id={row.get('claw_session_id')!r}\n"
        f"  unique_key={row['unique_key']!r}\n"
        f"  version={row['version']!r}  image={row['image']!r}"
    )


def main():
    """CLI entry point: parse args and import the requested JSON files.

    Builds the argument parser, validates the destination table, collects the
    input files, and dispatches to dry-run, local, or ssh-kubectl mode.
    """
    parser = argparse.ArgumentParser(
        description="Import session breakdown JSON files into perf_runs_dev.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("paths", nargs="*", help="JSON files or directories to import")
    parser.add_argument("--dir", help="Additionally scan a directory recursively for *.json")
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not write to the DB")
    parser.add_argument("--table", default=DEFAULT_TABLE,
                        help=f"Destination table (default: {DEFAULT_TABLE})")

    # Mode selection
    parser.add_argument("--mode", choices=("ssh-kubectl", "local"),
                        default="ssh-kubectl",
                        help="Connection mode (default: ssh-kubectl)")

    # ssh-kubectl mode
    parser.add_argument("--hop1", default=DEFAULT_SSH_HOP1,
                        help=f"SSH jump host (default: {DEFAULT_SSH_HOP1})")
    parser.add_argument("--namespace", default=DEFAULT_K8S_NAMESPACE,
                        help=f"K8s namespace (default: {DEFAULT_K8S_NAMESPACE})")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Number of rows per SSH+psql invocation in ssh-kubectl mode")

    # local mode
    parser.add_argument("--db-url", default=os.environ.get("PERF_RUNS_DB_URL"),
                        help="Full postgresql:// URL (overrides individual flags, local mode)")
    parser.add_argument("--host", default=DEFAULT_DB["host"])
    parser.add_argument("--port", type=int, default=DEFAULT_DB["port"])
    parser.add_argument("--user", default=DEFAULT_DB["user"])
    parser.add_argument("--password", default=DEFAULT_DB["password"])
    parser.add_argument("--dbname", default=DEFAULT_DB["dbname"])

    parser.add_argument("--skip-index-check", action="store_true",
                        help="Do not attempt to create the unique index on unique_key")

    args = parser.parse_args()

    # Validate table name early so we don't waste a connection.
    try:
        safe_table(args.table)
    except ValueError as e:
        parser.error(str(e))

    files = iter_json_files(args.paths, args.dir)
    if not files:
        parser.error("no input files (pass paths or --dir)")

    if args.dry_run:
        print(f"== DRY RUN: {len(files)} file(s), target table={args.table} ==")
        ok = 0
        for f in files:
            row = parse_file(f)
            if row is None:
                continue
            print(f"[DRY-RUN] {f}")
            print(format_row_summary(row))
            print()
            ok += 1
        print(f"== Done: {ok}/{len(files)} parsed ==")
        return

    if args.mode == "local":
        _run_local_mode(args, files)
    else:
        _run_ssh_kubectl_mode(args, files)


def _run_local_mode(args: argparse.Namespace, files: List[Path]) -> None:
    """Import files using a direct psycopg2 connection (local mode).

    Connects, optionally ensures the unique index, then parses and upserts each
    file, printing per-file status and a final tally.

    Args:
        args (argparse.Namespace): Parsed CLI args.
        files (List[Path]): JSON files to import.
    """
    print(f"== local mode: connecting to {args.host}:{args.port}/{args.dbname} ==")
    try:
        conn = connect_local(args)
    except Exception as e:
        sys.stderr.write(f"[FATAL] DB connection failed: {e}\n")
        sys.stderr.write(
            "Hint: did you start the SSH tunnel?\n"
            "  ssh -L 5432:127.0.0.1:5432 amd@10.245.143.31 \\\n"
            "      'kubectl port-forward -n primus-safe \\\n"
            "         $(kubectl get pod -n primus-safe "
            "-l postgres-operator.crunchydata.com/role=master -o name) \\\n"
            "         5432:5432'\n"
        )
        sys.exit(2)

    upsert_sql = build_upsert_sql(args.table)

    try:
        if not args.skip_index_check:
            try:
                with conn.cursor() as cur:
                    cur.execute(build_create_unique_index_sql(args.table))
                conn.commit()
            except Exception as e:
                sys.stderr.write(
                    f"[WARN] could not ensure unique index on {args.table}(unique_key): {e}\n"
                    f"       If duplicate unique_keys already exist, clean them up first.\n"
                )

        ok = 0
        for f in files:
            row = parse_file(f)
            if row is None:
                print()
                continue
            try:
                new_id, inserted = upsert_row_local(conn, row, upsert_sql)
                action = "INSERT" if inserted else "UPDATE"
                print(f"[{action} id={new_id}] {f}")
                print(format_row_summary(row))
                ok += 1
            except Exception as e:
                conn.rollback()
                sys.stderr.write(f"[ERROR] {f}: {e}\n")
            print()

        print(f"== Done: {ok}/{len(files)} succeeded ==")
    finally:
        conn.close()


def _run_ssh_kubectl_mode(args: argparse.Namespace, files: List[Path]) -> None:
    """Import files by piping SQL through SSH+kubectl ``psql``.

    Optionally ensures the unique index, parses all files, then upserts them in
    batches (falling back to one-by-one retries on a failed batch) to amortise
    SSH/kubectl startup cost.

    Args:
        args (argparse.Namespace): Parsed CLI args.
        files (List[Path]): JSON files to import.
    """
    print(
        f"== ssh-kubectl mode: hop1={args.hop1} ns={args.namespace} "
        f"db={args.dbname} table={args.table} ==")

    # Optionally ensure the unique index (one round trip).
    if not args.skip_index_check:
        try:
            execute_remote_sql(
                build_create_unique_index_sql(args.table) + "\n",
                hop1=args.hop1, namespace=args.namespace,
                user=args.user, dbname=args.dbname,
            )
            print("[ok] unique index ensured")
        except Exception as e:
            sys.stderr.write(
                f"[WARN] could not ensure unique index on {args.table}(unique_key): {e}\n"
            )

    # Parse all files first; collect valid (path, row) pairs.
    parsed: List[Tuple[Path, Dict[str, Any]]] = []
    for f in files:
        row = parse_file(f)
        if row is not None:
            parsed.append((f, row))

    if not parsed:
        print("== Done: 0 files to import ==")
        return

    # Batch upserts to amortize SSH/kubectl startup cost.
    batch_size = max(1, int(args.batch_size))
    ok = 0
    for i in range(0, len(parsed), batch_size):
        chunk = parsed[i:i + batch_size]
        sql_parts = ["BEGIN;\n"]
        for _, row in chunk:
            sql_parts.append(build_upsert_statement_inline(row, args.table))
        sql_parts.append("COMMIT;\n")
        sql = "\n".join(sql_parts)

        try:
            execute_remote_sql(
                sql, hop1=args.hop1, namespace=args.namespace,
                user=args.user, dbname=args.dbname,
            )
            for f, row in chunk:
                print(f"[UPSERT] {f}")
                print(format_row_summary(row))
                print()
            ok += len(chunk)
        except Exception as e:
            sys.stderr.write(
                f"[ERROR] batch {i // batch_size} ({len(chunk)} files) failed: {e}\n"
            )
            # Retry one-by-one to identify the bad file.
            for f, row in chunk:
                try:
                    single_sql = "BEGIN;\n" + build_upsert_statement_inline(row, args.table) + "COMMIT;\n"
                    execute_remote_sql(
                        single_sql, hop1=args.hop1, namespace=args.namespace,
                        user=args.user, dbname=args.dbname,
                    )
                    print(f"[UPSERT/retry] {f}")
                    print(format_row_summary(row))
                    print()
                    ok += 1
                except Exception as ee:
                    sys.stderr.write(f"[ERROR] {f}: {ee}\n")

    print(f"== Done: {ok}/{len(files)} succeeded ==")


if __name__ == "__main__":
    main()
