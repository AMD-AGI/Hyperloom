# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pure, self-contained helpers used by the Coordinator.

No dependency on Coordinator state; must not import ``coordinator`` (one-way
dependency).
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Leading-underscore module "constants" below are read only from *other*
# modules (e.g. ``phases/kernel.py``, ``loop/writeback.py`` import them
# directly) rather than from within this file. Static analysis that only
# tracks in-module reads (e.g. CodeQL's unused-global-variable check) can't
# see that cross-module usage on its own, so list them here to mark them as
# intentionally exported.
__all__ = [
    "_GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT",
    "_MIN_KERNEL_ENGAGED_GAIN_PCT",
]


def format_exc_brief(exc: BaseException, limit: int | None = None) -> str:
    """Render an exception as ``"TypeName: message"``, optionally truncated.

    Args:
        exc: The exception (or any ``BaseException``) to format.
        limit: When set, truncate the message to this many characters.

    Returns:
        ``f"{type(exc).__name__}: {str(exc)[:limit]}"`` (no truncation when
        ``limit`` is ``None``).
    """
    msg = str(exc)
    if limit is not None:
        msg = msg[:limit]
    return f"{type(exc).__name__}: {msg}"


def _infer_model_class_from_config(model_path: str) -> str:
    """Infer a deterministic model_class from local model metadata.

    Args:
        model_path: Local model directory path; its ``config.json`` is read
            when present.

    Returns:
        A model-class label: ``moe_mla_nsa``, ``moe_mla``, ``moe_swa`` or
        ``dense``.
    """
    import json

    raw_path = (model_path or "").strip()
    payload: dict[str, Any] = {}
    if raw_path:
        # ``model_path`` may be an HF repo id; resolve to the local weights dir so
        # the config-based classification works (the raw string still feeds the
        # keyword fallback below). Lazy import: stdlib-only leaf, no import cycle.
        from hyperloom.inference_optimizer.model_config_utils import (
            resolve_local_model_dir,
        )

        _resolved = resolve_local_model_dir(raw_path)
        cfg = (_resolved / "config.json") if _resolved is not None else Path(raw_path) / "config.json"
        try:
            if cfg.is_file():
                data = json.loads(cfg.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    payload = data
        except Exception:  # noqa: BLE001 - best effort only.
            log.debug("model_class inference: failed to read %s", cfg, exc_info=True)

    text_parts: list[str] = [raw_path.lower()]
    arch = payload.get("architectures")
    if isinstance(arch, list):
        text_parts.extend(str(x).lower() for x in arch if x)
    elif arch:
        text_parts.append(str(arch).lower())
    for key in ("model_type", "attention_type", "attn_type"):
        if payload.get(key):
            text_parts.append(str(payload[key]).lower())
    text = " ".join(text_parts)

    def _positive_int(*keys: str) -> bool:
        """Whether any of the given payload keys holds a positive integer.

        Booleans are explicitly ignored (they are not treated as ints).

        Args:
            *keys: Payload keys to check.

        Returns:
            ``True`` if at least one key parses to an integer > 0.
        """
        for key in keys:
            val = payload.get(key)
            if isinstance(val, bool):
                continue
            try:
                if val is not None and int(val) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    is_moe = _positive_int(
        "num_experts",
        "n_routed_experts",
        "num_local_experts",
        "moe_num_experts",
    ) or any(
        k in text
        for k in (
            "moe",
            "mixtral",
            "deepseek-v2",
            "deepseek-v3",
            "deepseek-r1",
            "kimi",
            "glm-5",
            "glm5",
        )
    )
    is_mla = any(
        k in text
        for k in (
            "mla",
            "multi-head latent",
            "deepseek",
            "kimi",
            "glm-5",
            "glm5",
        )
    )
    is_nsa = any(
        k in text
        for k in (
            "nsa",
            "native sparse attention",
            "glm-5",
            "glm5",
        )
    )
    if is_moe and is_mla and is_nsa:
        return "moe_mla_nsa"
    if is_moe and is_mla:
        return "moe_mla"
    if is_moe:
        return "moe_swa"
    return "dense"


# task.params fields fingerprinted by the self-loop guard to detect a
# proposal repeating the same params after the same failure mode.
_BASELINE_FINGERPRINT_KEYS: tuple[str, ...] = (
    "benchmark_script",
    "result_dir",
    "extra_server_args",
    "extra_envs",
    "model_path",
    "gpu_type",
    "config_path",
    "disable_run_eval",
)

# Flags whose argparse consumes multiple bare tokens before the next ``--``.
_MULTI_VALUE_SGLANG_FLAGS: frozenset[str] = frozenset(
    {
        "--cuda-graph-bs",
        "--cuda-graph-max-bs",
    }
)

_DEFAULT_ROOFLINE_WATERMARK_RATIO: float = 1.10  # 10% step over last roofline
_ROOFLINE_WATERMARK_RATIO_ENV: str = "HYPERLOOM_ROOFLINE_WATERMARK_RATIO"


def effective_closing_grace_sec(
    max_minutes: float | None,
    closing_grace_sec: float | None,
) -> float:
    """Resolve the closing-phase grace window after the wall-clock deadline.

    Explicit ``closing_grace_sec`` (including ``0`` to disable) wins;
    otherwise default to ``min(120, max_minutes * 60 * 0.02)``.

    Args:
        max_minutes: The wall-clock budget in minutes (used for the default).
        closing_grace_sec: Explicit grace window in seconds; when not
            ``None`` it is used verbatim.

    Returns:
        The closing-phase grace window in seconds.
    """
    if closing_grace_sec is not None:
        return float(closing_grace_sec)
    return min(120.0, (max_minutes or 0.0) * 60.0 * 0.02)


def _parse_iso_unix(ts: str) -> float:
    """Parse an ISO 8601 UTC timestamp into unix seconds; ``0.0`` on failure.

    Naive timestamps are treated as UTC. Never raises.

    Args:
        ts: ISO 8601 timestamp string (``Z`` suffix accepted).

    Returns:
        The timestamp in unix seconds, or ``0.0`` when empty/unparseable.
    """
    s = (ts or "").strip()
    if not s:
        return 0.0
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_baseline_workload_extra(yaml_path: str) -> dict[str, Any]:
    """Extract KB workload-tag fields from a baseline-materialized Magpie YAML.

    Reads workload-shape fields outside ``_collect_workload_tags`` from
    ``benchmark.envs`` extra-args blobs and top-level ``benchmark`` fields.
    Defensive — parse errors return ``{}``.

    Args:
        yaml_path: Path to the baseline-materialized Magpie YAML.

    Returns:
        The extracted workload-tag fields, or ``{}`` on parse error.
    """
    import yaml as _yaml

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
    except (OSError, _yaml.YAMLError):
        return {}
    out: dict[str, Any] = {}
    bm = cfg.get("benchmark") if isinstance(cfg, dict) else None
    if not isinstance(bm, dict):
        return out
    for src, dst in (
        ("workload_mode", "workload_mode"),
        ("quant_scheme", "quant_scheme"),
    ):
        v = bm.get(src)
        if v not in (None, "", 0):
            out[dst] = v
    envs = bm.get("envs") if isinstance(bm.get("envs"), dict) else {}
    extra_args_str = ""
    for env_key in ("EXTRA_SGLANG_ARGS", "EXTRA_VLLM_ARGS"):
        v = envs.get(env_key)
        if isinstance(v, str) and v.strip():
            extra_args_str = v.strip()
            break
    tokens = extra_args_str.split() if extra_args_str else []
    for i, tok in enumerate(tokens):
        if tok in ("--max-running-requests",) and i + 1 < len(tokens):
            try:
                out["max_running_requests"] = int(tokens[i + 1])
            except ValueError:
                # Non-integer CLI value; leave the field unset.
                pass
        elif tok in ("--max-num-seqs",) and i + 1 < len(tokens):
            try:
                out["max_num_seqs"] = int(tokens[i + 1])
            except ValueError:
                # Non-integer CLI value; leave the field unset.
                pass
        elif tok == "--enable-chunked-prefill":
            out["chunked_prefill_enabled"] = True
        elif tok == "--disable-chunked-prefill":
            out["chunked_prefill_enabled"] = False
        elif tok == "--enable-torch-compile":
            out["enable_torch_compile"] = True
    # Torch compile may also be a separate env var.
    if "enable_torch_compile" not in out:
        tc_env = envs.get("ENABLE_TORCH_COMPILE")
        if isinstance(tc_env, str):
            out["enable_torch_compile"] = tc_env.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
    return out


def _baseline_params_fingerprint(params: dict[str, Any] | None) -> dict[str, Any]:
    """Project ``params`` to the keys that determine baseline behavior.

    Missing keys recorded as ``None``; ``extra_envs`` normalized to a sorted
    list of stringified ``[key, value]`` pairs so ordering doesn't affect
    equality.

    Args:
        params: Task params to project (``None`` treated as empty).

    Returns:
        A fingerprint dict over the baseline-determining keys.
    """
    params = params or {}
    out: dict[str, Any] = {}
    for key in _BASELINE_FINGERPRINT_KEYS:
        if key == "extra_envs":
            envs = params.get(key) or {}
            if isinstance(envs, dict):
                out[key] = sorted([str(k), str(v)] for k, v in envs.items())
            else:
                out[key] = None
            continue
        value = params.get(key)
        out[key] = None if value is None else str(value)
    return out


def _resolve_roofline_watermark_ratio() -> float:
    """Resolve the roofline watermark ratio from ``$HYPERLOOM_ROOFLINE_WATERMARK_RATIO`` (fallback 1.10).

    Returns:
        The watermark ratio (> 1.0), falling back to ``1.10`` when unset,
        unparseable, or not greater than 1.0.
    """
    raw = (os.environ.get(_ROOFLINE_WATERMARK_RATIO_ENV) or "").strip()
    if not raw:
        return _DEFAULT_ROOFLINE_WATERMARK_RATIO
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_ROOFLINE_WATERMARK_RATIO
    if val <= 1.0:
        return _DEFAULT_ROOFLINE_WATERMARK_RATIO
    return val


def _merge_cumulative_extra_sglang_args(
    base_args: str,
    candidate_args: str,
    full_args: str,
) -> str:
    """Build cumulative launch args for a KEEP without double-stacking.

    Prefer the full stack and dedupe, since joining ``base + candidate``
    when both are full stacks duplicates flags.

    Args:
        base_args: The baseline extra-args string.
        candidate_args: The candidate extra-args string for the KEEP.
        full_args: The full cumulative stack, preferred when present.

    Returns:
        The deduped cumulative launch-args string.
    """
    base = str(base_args or "").strip()
    candidate = str(candidate_args or "").strip()
    full = str(full_args or "").strip()
    if full and full != candidate:
        merged = full
    elif candidate and base:
        if candidate.startswith(base) or base in candidate.split():
            merged = candidate
        else:
            merged = f"{base} {candidate}".strip()
    else:
        merged = candidate or full or base
    return _dedupe_extra_server_args(merged)


_SPACE_VALUE_FLAGS = (
    "--json-model-override-args",
    "--override-generation-config",
    "--tool-call-parser",
)


def _dedupe_extra_server_args(args_str: str) -> str:
    """Collapse repeated ``--flag value`` pairs into a unique launch string.

    Keep each flag once with its last value (first-seen order preserved),
    since argparse ``action="store"`` only honors the last value. Flags in
    ``_MULTI_VALUE_SGLANG_FLAGS`` keep their multi-value runs. Args with
    JSON/space-valued flags are left untouched because downstream launch
    scripts expand them unquoted.

    Args:
        args_str: The extra server-args string to dedupe.

    Returns:
        The deduped args string, or the input unchanged when it contains a
        JSON/space-valued flag; ``""`` for empty input.
    """
    if not args_str:
        return ""
    if any(flag in args_str for flag in _SPACE_VALUE_FLAGS):
        return args_str
    tokens = args_str.split()
    pair_by_flag: dict[str, list[str]] = {}
    order: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            if "=" in t:
                flag, _, value = t.partition("=")
                values = [value] if value else []
                i += 1
            else:
                flag = t
                i += 1
                values = []
                if flag in _MULTI_VALUE_SGLANG_FLAGS:
                    while i < len(tokens) and not tokens[i].startswith("--"):
                        values.append(tokens[i])
                        i += 1
                elif i < len(tokens) and not tokens[i].startswith("--"):
                    values = [tokens[i]]
                    i += 1
            pair = [flag, *values] if values else [flag]
            if flag not in pair_by_flag:
                order.append(flag)
            pair_by_flag[flag] = pair
        else:
            # Stray positional token; preserve as-is.
            key = f"__positional_{len(order)}__"
            order.append(key)
            pair_by_flag[key] = [t]
            i += 1
    out: list[str] = []
    for k in order:
        out.extend(pair_by_flag[k])
    return " ".join(out)


# Advisory fields carried on a Critic ``review_verdict`` payload beyond the
# bare verdict/reasoning. The list-valued keys are normalised to lists with
# empty entries dropped; the string keys are kept only when non-blank.
_VERDICT_ADVISORY_LIST_KEYS: tuple[str, ...] = (
    "required_evidence",
    "risks",
    "notes",
    "kb_evidence",
    "packet_evidence",
)
_VERDICT_ADVISORY_TEXT_KEYS: tuple[str, ...] = (
    "advice_text",
    "alternative_action",
)


def serialize_verdict_advisory(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the advisory field set from a ``review_verdict`` payload.

    Produces the canonical advisory subset (``required_evidence`` / ``risks`` /
    ``advice_text`` / ``alternative_action`` / ``notes`` / ``kb_evidence`` /
    ``packet_evidence``). This is the single definition of that field set,
    shared by verdict rebroadcast payload assembly and compact inbox rendering
    so the two never drift apart.

    Empty values are omitted; list-valued fields are coerced to lists with
    ``None``/empty entries dropped.

    Args:
        payload: A ``review_verdict`` intent/message payload.

    Returns:
        A dict holding only the present, non-empty advisory fields.
    """
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Any] = {}
    for key in _VERDICT_ADVISORY_LIST_KEYS:
        raw = payload.get(key)
        if isinstance(raw, (list, tuple)):
            items = [item for item in raw if item not in (None, "")]
        elif raw in (None, ""):
            items = []
        else:
            items = [raw]
        if items:
            out[key] = list(items)
    for key in _VERDICT_ADVISORY_TEXT_KEYS:
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            out[key] = raw
    return out


# Minimum over-baseline gain a same-harness revalidation must show for the
# optimization to count as "engaged" (i.e. the tuned config actually took
# effect). Only used to detect a collapse back to ~baseline (an un-optimized
# relaunch), never to gate a specific optimization — kept general across all
# winner kinds. Overridable via env for tuning.
try:
    _MIN_KERNEL_ENGAGED_GAIN_PCT: float = float(
        os.environ.get("INFERENCE_OPTIMIZER_MIN_ENGAGED_GAIN_PCT", "").strip() or "2.0"
    )
except (TypeError, ValueError):
    _MIN_KERNEL_ENGAGED_GAIN_PCT = 2.0

# |measurement_divergence_pct| above this (GEAK vs orchestrator on the SAME
# config) is logged as a measurement-mismatch warning at geak promote.
# Reporting only — never gates scheduling. Overridable via env.
try:
    _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT: float = float(
        os.environ.get("INFERENCE_OPTIMIZER_MEASUREMENT_DIVERGENCE_WARN_PCT", "").strip() or "3.0"
    )
except (TypeError, ValueError):
    _GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT = 3.0


def _split_env_and_flags(env_str: str) -> tuple[dict[str, str], str]:
    """Split a bench-style config string into (env dict, flags string).

    ``accepted_config.env`` (and any ``KEY=VAL KEY=VAL`` / ``--flag val`` blob)
    is parsed so that every ``KEY=VAL`` token becomes a real environment
    variable and every ``--flag`` (or ``--flag=val``) token is folded back into
    a server-args string. Single source of truth for this parse so the promote
    path, the resume-materialize path, and the revalidation path all agree.
    General: no key/optimization is special-cased.

    Args:
        env_str: The raw config blob (may mix ``KEY=VAL`` and ``--flag`` tokens).

    Returns:
        ``(envs, flags)`` where ``envs`` is a ``dict[str, str]`` of real env
        vars and ``flags`` is a space-joined server-args string ("" when none).
    """
    envs: dict[str, str] = {}
    flag_tokens: list[str] = []
    try:
        tokens = shlex.split(str(env_str or ""))
    except ValueError:
        tokens = str(env_str or "").split()
    for tok in tokens:
        if tok.startswith("-"):
            flag_tokens.append(tok)
        elif "=" in tok:
            k, v = tok.split("=", 1)
            if k:
                envs[k] = v
    return envs, " ".join(flag_tokens).strip()


def _geak_revalidation_decision(
    *,
    measured: Any,
    baseline: Any,
    got_hash: str,
    expected_hash: str,
    min_engaged_gain_pct: float,
) -> str:
    """Decide a geak same-harness (2b) rebench outcome.

    Returns ``"validated"`` only when BOTH hold:
      * config identity — the ran variant's fingerprint matches the expected
        (skipped when no expected hash was pinned); catches an executor-side
        drop/alter of the optimized config; and
      * engagement — the measured throughput cleared baseline by at least
        ``min_engaged_gain_pct`` (i.e. the optimization actually took effect and
        did not collapse back to an un-optimized relaunch).
    Otherwise returns ``"fallback"`` so the caller replays via the GEAK harness
    (2a). General: the engagement floor only detects a baseline collapse, it is
    not tied to any specific optimization's expected magnitude.

    Args:
        measured: Rebench output throughput (tok/s).
        baseline: Orchestrator raw baseline throughput (same harness as measured).
        got_hash: Fingerprint of the variant that actually ran.
        expected_hash: Pinned expected fingerprint ("" => identity check skipped).
        min_engaged_gain_pct: Minimum over-baseline gain to count as engaged.

    Returns:
        ``"validated"`` or ``"fallback"``.
    """
    measured_ok = isinstance(measured, (int, float)) and measured > 0
    baseline_ok = isinstance(baseline, (int, float)) and baseline > 0
    if not (measured_ok and baseline_ok):
        return "fallback"
    cfg_ok = (not expected_hash) or (str(got_hash or "") == str(expected_hash))
    engaged = float(measured) >= float(baseline) * (1.0 + float(min_engaged_gain_pct) / 100.0)
    return "validated" if (cfg_ok and engaged) else "fallback"


def _geak_sweep_measured_tput(res: dict[str, Any]) -> float | None:
    """Extract a single measured throughput from a ``sweep_via_geak`` result.

    Used by the GEAK-harness (2a) rebench to source the MEASURED headline
    throughput (rather than GEAK's self-reported speedup). Prefers
    ``best_for_each_conc`` (already the per-conc best), falling back to the first
    succeeded ``sweep_grid`` entry. Returns ``None`` when no positive throughput
    is present.
    """
    if not isinstance(res, dict):
        return None
    best = res.get("best_for_each_conc")
    if isinstance(best, dict):
        for entry in best.values():
            if isinstance(entry, dict):
                t = entry.get("output_throughput")
                if isinstance(t, (int, float)) and t > 0:
                    return float(t)
    grid = res.get("sweep_grid")
    if isinstance(grid, list):
        for entry in grid:
            if isinstance(entry, dict) and entry.get("status") == "succeeded":
                t = entry.get("output_throughput")
                if isinstance(t, (int, float)) and t > 0:
                    return float(t)
    return None


def _parse_server_arg_value(server_args: str, flag: str) -> str | None:
    """Extract a CLI flag's value from a server-args string.

    Handles both ``--flag value`` and ``--flag=value`` forms. Hyperloom keeps
    serving-fidelity knobs (``--max-model-len``, ``--gpu-memory-utilization``)
    as raw flags inside the baseline server-args string rather than as
    structured fields, so the geak handoff must recover them from there.

    Args:
        server_args: The full server-args string (e.g. baseline EXTRA_VLLM_ARGS).
        flag: The flag to look up, INCLUDING leading dashes (e.g. ``--max-model-len``).

    Returns:
        The flag's value as a string, or ``None`` when the flag is absent or
        present without a value.
    """
    if not server_args or not flag:
        return None
    try:
        toks = shlex.split(server_args)
    except ValueError:
        toks = server_args.split()
    prefix = flag + "="
    for i, tok in enumerate(toks):
        if tok == flag:
            return toks[i + 1] if i + 1 < len(toks) else None
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


def _resolve_serving_fidelity(
    *,
    baseline_server_args: str,
    state_max_model_len: int = 0,
) -> dict[str, Any]:
    """Resolve serving-fidelity knobs to forward in the geak handoff.

    Returns a dict carrying ONLY the resolved keys (``max_model_len`` int and/or
    ``mem_fraction`` float). Unresolved knobs are OMITTED so the GEAK vllm
    adapter applies its own production-faithful defaults (no 0 sentinel to
    disambiguate). Source precedence — robust to both the dedicated CLI arg and
    the common case where fidelity knobs ride inside the baseline server-args
    string (e.g. ``--max-model-len 2248 --gpu-memory-utilization 0.9``):

      * ``max_model_len``: ``state.max_model_len`` > ``--max-model-len`` in the
        baseline server-args > ``MAX_MODEL_LEN`` env.
      * ``mem_fraction``: ``--gpu-memory-utilization`` in the baseline
        server-args > ``GPU_MEMORY_UTILIZATION`` env. (There is no structured
        ``state.mem_fraction``; Hyperloom keeps it as a raw flag.)

    Args:
        baseline_server_args: The baseline arm's runtime server-args string.
        state_max_model_len: The dedicated ``state.max_model_len`` (0 when unset).

    Returns:
        A dict with the resolved subset of ``{"max_model_len", "mem_fraction"}``.
    """
    out: dict[str, Any] = {}

    mml = int(state_max_model_len or 0)
    if mml <= 0:
        v = _parse_server_arg_value(baseline_server_args, "--max-model-len")
        try:
            mml = int(v) if v else 0
        except (TypeError, ValueError):
            mml = 0
    if mml <= 0:
        try:
            mml = int(os.environ.get("MAX_MODEL_LEN", "0") or 0)
        except (TypeError, ValueError):
            mml = 0
    if mml > 0:
        out["max_model_len"] = mml

    v = _parse_server_arg_value(baseline_server_args, "--gpu-memory-utilization")
    try:
        mem = float(v) if v else 0.0
    except (TypeError, ValueError):
        mem = 0.0
    if mem <= 0:
        try:
            mem = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0") or 0.0)
        except (TypeError, ValueError):
            mem = 0.0
    if mem > 0:
        out["mem_fraction"] = mem

    return out


#: Launch flags that are RUN-/TOPOLOGY-specific (host, device set, model path,
#: parallelism, ports, seeds) — the consuming harness sets these itself per
#: launch, so they are stripped from the forwarded ``server_launch_flags``.
#: Everything NOT listed here (engine knobs: mem-fraction, radix cache,
#: chunked-prefill, cuda-graph, attention backend, quant, kv-cache dtype, …) is
#: kept, so the sync is COMPLETE by construction (allow-nothing blacklist rather
#: than a hand-picked whitelist that silently drops un-enumerated knobs).
_RUN_SPECIFIC_LAUNCH_FLAGS: frozenset[str] = frozenset(
    {
        "--model-path",
        "--tokenizer-path",
        "--served-model-name",
        "--host",
        "--port",
        "--nccl-port",
        "--dist-init-addr",
        "--base-gpu-id",
        "--gpu-id-step",
        "--node-rank",
        "--nnodes",
        "--tensor-parallel-size",
        "--tp-size",
        "--tp",
        "--data-parallel-size",
        "--dp-size",
        "--pipeline-parallel-size",
        "--pp-size",
        "--random-seed",
        "--download-dir",
        "--pid",
    }
)

#: Profiling-only launch flags: present on a roofline/profile server launch but
#: NOT part of a clean throughput baseline. Stripped so a scraped argv never
#: carries profiler instrumentation into the reproduced baseline.
_PROFILING_LAUNCH_FLAGS: frozenset[str] = frozenset(
    {
        "--enable-profile-cuda-graph",
        "--enable-shape-discovery-for-cuda-graph-profile",
        "--enable-profile",
        "--enable-torch-compile-debug-mode",
        "--debug-cuda-graph",
    }
)

#: Per-backend token that marks the START of the launch argv on a captured
#: command line (``server.log`` header / ``set -x`` stderr echo). A new backend
#: is one map entry; an unknown backend disables the scrape (no guess).
_LAUNCH_ARGV_MARKERS: dict[str, str] = {
    "sglang": "launch_server",
    "vllm": "vllm",
}


def _split_launch_flags(argv_tail: str) -> str:
    """Drop run/topology-specific + profiling-only flags from a launch argv tail.

    Keeps every ENGINE knob (the whole point: no whitelist) and removes only the
    per-run flags in :data:`_RUN_SPECIFIC_LAUNCH_FLAGS` and the profiler-only
    flags in :data:`_PROFILING_LAUNCH_FLAGS`, handling both ``--flag value`` and
    ``--flag=value`` forms plus valueless store-true flags.
    """
    try:
        toks = shlex.split(argv_tail)
    except ValueError:
        toks = argv_tail.split()
    kept: list[str] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        base = tok.split("=", 1)[0]
        if base in _RUN_SPECIFIC_LAUNCH_FLAGS or base in _PROFILING_LAUNCH_FLAGS:
            # ``--flag=value`` is one token; ``--flag value`` consumes the next
            # token too (unless the next token is itself a flag => valueless).
            if "=" not in tok and i + 1 < len(toks) and not toks[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        kept.append(tok)
        i += 1
    return " ".join(kept)


def _launch_argv_from_log(path: str, marker: str) -> str:
    """Extract + normalize the engine launch argv from one benchmark log."""
    import re as _re

    pat = _re.compile(
        r"(?:-m\s+\S*" + _re.escape(marker) + r"\S*|" + _re.escape(marker) + r")\b(.*)$"
    )
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if marker not in line or "--model-path" not in line:
                    continue
                m = pat.search(line)
                tail = (m.group(1) if m else "").strip()
                if not tail:
                    idx = line.find("--")
                    tail = line[idx:].strip() if idx >= 0 else ""
                flags = _split_launch_flags(tail)
                if flags:
                    return flags
    except OSError:
        return ""
    return ""


def _scrape_resolved_launch_flags(
    session_dir: Any, backend: str, target_tput: float = 0.0
) -> str:
    """Recover the orchestrator's FULL resolved server-launch flags from logs.

    The recipe YAML only carries the recipe-level ``EXTRA_*_ARGS`` delta; the
    harness launch script (e.g. InferenceX ``sglang_mi300x.sh``) bakes in the
    rest (``--mem-fraction-static``, ``--disable-radix-cache``,
    ``--chunked-prefill-size`` …). The ONLY complete, authoritative record of
    what the engine actually ran with is the launched argv, echoed into each
    benchmark's ``server.log`` / ``benchmark_stderr.log``.

    Selection is by THROUGHPUT, not recency: we find the benchmark whose measured
    ``output_throughput`` equals ``target_tput`` (``current_best``'s number) and
    scrape ITS sibling server log — i.e. replay the exact launch that produced
    the throughput we are asking GEAK to reproduce. This is deterministic
    and never mistakes a profiling/roofline or losing-candidate launch for the
    baseline. Falls back to the most recent clean launch when no throughput
    match exists (or ``target_tput<=0``).

    Args:
        session_dir: The run's session directory (root of ``runs/``).
        backend: Serving backend ("sglang" | "vllm" | …).
        target_tput: ``current_best`` throughput to match a benchmark by.

    Returns:
        The resolved engine-knob flag string (run-specific + profiling stripped),
        or ``""`` when no argv is found (consumer keeps its adapter defaults).
    """
    marker = _LAUNCH_ARGV_MARKERS.get(str(backend or "").strip().lower())
    if not marker:
        return ""
    try:
        import glob as _glob

        runs_root = Path(session_dir) / "runs"
        # 1) Throughput-matched selection: find the benchmark dir whose
        #    inferencex_result.json output_throughput == target_tput, scrape its
        #    sibling server log. Deterministic reproduction of THE best launch.
        if target_tput and target_tput > 0:
            best_path, best_err = "", 1e9
            for rp in _glob.glob(
                str(runs_root / "**" / "inferencex_result.json"), recursive=True
            ):
                if "geak" in rp or "_baseline_source_overlay" in rp:
                    continue
                try:
                    tp = float(
                        (json.loads(Path(rp).read_text(encoding="utf-8")) or {}).get(
                            "output_throughput"
                        )
                        or 0.0
                    )
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    continue
                if tp <= 0:
                    continue
                err = abs(tp - target_tput) / target_tput
                if err < best_err:
                    best_err, best_path = err, rp
            if best_path and best_err <= 0.005:  # within 0.5% => same measurement
                bench_dir = Path(best_path).parent
                for name in ("server.log", "benchmark_stderr.log"):
                    flags = _launch_argv_from_log(str(bench_dir / name), marker)
                    if flags:
                        return flags
        # 2) Fallback: most recent clean (non-profiling) launch across the run.
        candidates: list[tuple[float, str]] = []
        for name in ("server.log", "benchmark_stderr.log"):
            for p in _glob.glob(str(runs_root / "**" / name), recursive=True):
                if "geak" in p or "_baseline_source_overlay" in p:
                    continue
                try:
                    candidates.append((os.path.getmtime(p), p))
                except OSError:
                    continue
        for _, path in sorted(candidates, reverse=True):
            flags = _launch_argv_from_log(path, marker)
            if flags:
                return flags
    except Exception:  # noqa: BLE001 — best-effort; absence => adapter default
        return ""
    return ""
