# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure, self-contained helpers used by the Coordinator."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.env_safety import filter_untrusted_env_mapping, is_allowed_variant_env_key
from hyperloom.common.visible_devices import (
    HIP_LEVEL_VARS,
    effective_mask_tokens,
    VISIBLE_DEVICE_VARS,
    is_rocr_level,
    mask_tokens,
    parse_device_list,
)

from ..specialists.patch_safety import (
    ADVISE_VERDICT,
    advisory_only_reason_codes,
    advisory_rules_govern,
)

log = logging.getLogger(__name__)

# Constants below are read from other modules; listed here to mark them as intentionally exported.
__all__ = [
    "TIME_BUDGET_EXEMPT_ACTIONS",
    "_GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT",
    "_MIN_KERNEL_ENGAGED_GAIN_PCT",
    "action_fits_time_budget",
    "baseline_benchmark_script",
    "coerce_needs_gpu",
    "expected_action_cost_minutes",
    "measured_baseline_runtime_sec",
]


def coerce_needs_gpu(value: Any) -> bool:
    """Coerce a ``needs_gpu`` specialist parameter value to a Python bool."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def format_exc_brief(exc: BaseException, limit: int | None = None) -> str:
    """Render an exception as ``\"TypeName: message\"``, optionally truncated."""
    msg = str(exc)
    if limit is not None:
        msg = msg[:limit]
    return f"{type(exc).__name__}: {msg}"


def _infer_model_class_from_config(model_path: str) -> str:
    """Infer a deterministic model_class from local model metadata."""
    import json

    raw_path = (model_path or "").strip()
    payload: dict[str, Any] = {}
    if raw_path:
        # ``model_path`` may be an HF repo id; resolve to the local weights dir so the config-based classification
        # works (the raw string still feeds the keyword fallback below).
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

    # A multimodal checkpoint keeps the language model one level down, so the
    # expert counts and the LM architecture live there rather than at the top.
    # Read both, outer first: a VL wrapper would otherwise classify as dense.
    payloads: list[dict[str, Any]] = [payload]
    for nested_key in ("text_config", "llm_config", "language_config"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            payloads.append(nested)

    text_parts: list[str] = [raw_path.lower()]
    for scope in payloads:
        arch = scope.get("architectures")
        if isinstance(arch, list):
            text_parts.extend(str(x).lower() for x in arch if x)
        elif arch:
            text_parts.append(str(arch).lower())
        for key in ("model_type", "attention_type", "attn_type"):
            if scope.get(key):
                text_parts.append(str(scope[key]).lower())
    text = " ".join(text_parts)

    def _positive_int(*keys: str) -> bool:
        """Whether any of the given keys holds a positive integer, in the top-level config or a nested LM config."""
        for scope in payloads:
            for key in keys:
                val = scope.get(key)
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


# task.params fields fingerprinted by the self-loop guard.
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

# Consecutive roofline failures tolerated before the watermark stops re-arming.
_MAX_ROOFLINE_FAILURE_RETRIES: int = 3


# Actions that must stay startable no matter how little budget is left: they are how a session ends cleanly, so a time
# gate that refused them would strand the run with nothing to show.
TIME_BUDGET_EXEMPT_ACTIONS: frozenset[str] = frozenset(
    {
        "report",
        "session_breakdown",
    }
)

# The lanes that serialize GPU work: an action requiring one of them spends its time running a benchmark round, so
# what this session measured says more about it than a catalogue estimate does.
_GPU_BENCH_LANES: frozenset[str] = frozenset(
    {
        "benchmark_lane",
        "profile_lane",
    }
)


def baseline_benchmark_script(state: Any) -> str | None:
    """Read the script belonging to the accepted baseline anchor."""
    accepted = getattr(state, "baseline_benchmark_script", None)
    if accepted is not None:
        return accepted or None
    last_baseline = getattr(state, "last_baseline", {}) or {}
    if last_baseline.get("decision") != "promoted":
        return None
    fingerprint = (last_baseline.get("extras") or {}).get("fingerprint") or {}
    return str(fingerprint.get("benchmark_script") or "").strip() or None


def measured_baseline_runtime_sec(shared_state: Any | None) -> float:
    """Read this session's own measured baseline round, in seconds."""
    try:
        return max(0.0, float(getattr(shared_state, "baseline_runtime_sec", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _action_benches_on_gpu(meta: Any | None) -> bool:
    """Whether an action's cost is dominated by a benchmark round on the GPU."""
    lanes = getattr(meta, "requires_lanes", ()) or ()
    try:
        return any(str(lane) in _GPU_BENCH_LANES for lane in lanes)
    except TypeError:
        return False


def expected_action_cost_minutes(
    meta: Any | None,
    *,
    measured_baseline_sec: float = 0.0,
) -> float:
    """Read an action's expected cost, preferring what this session measured."""
    try:
        catalogue_min = float(getattr(meta, "typical_runtime_min", 0.0) or 0.0)
    except (TypeError, ValueError):
        catalogue_min = 0.0
    if measured_baseline_sec <= 0.0 or not _action_benches_on_gpu(meta):
        return catalogue_min
    return max(catalogue_min, measured_baseline_sec / 60.0)


def action_fits_time_budget(
    *,
    usable_sec: float | None,
    expected_cost_minutes: float,
) -> bool:
    """Decide whether an action's expected cost still fits the remaining budget."""
    if usable_sec is None:
        return True
    if expected_cost_minutes <= 0.0:
        return True
    return usable_sec >= expected_cost_minutes * 60.0


def _parse_iso_unix(ts: str) -> float:
    """Parse an ISO 8601 UTC timestamp into unix seconds; ``0.0`` on failure."""
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
    """Extract KB workload-tag fields from a baseline-materialized Magpie YAML."""
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
    """Project ``params`` to the keys that determine baseline behavior."""
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


def approved_proposal_idempotency_key(action_name: str, params: dict[str, Any] | None) -> str:
    """Content-addressed idempotency key for an approved proposal."""
    params = params or {}
    payload: Any = _baseline_params_fingerprint(params) if action_name == "baseline" else params
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode(),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return f"approved:{action_name}:{digest}"


def _resolve_roofline_watermark_ratio() -> float:
    """Resolve the roofline watermark ratio."""
    return _DEFAULT_ROOFLINE_WATERMARK_RATIO


def _merge_cumulative_extra_server_args(
    base_args: str,
    candidate_args: str,
    full_args: str,
) -> str:
    """Build cumulative launch args for a KEEP without double-stacking."""
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


def _dedupe_extra_server_args(args_str: str) -> str:
    """Collapse repeated ``--flag value`` pairs into a unique launch string."""
    if not args_str:
        return ""
    # Imported here, not at module scope: ``actions.executors`` re-enters this module through ``session_breakdown``,
    # so a top-level import makes any importer that reaches ``coordinator_helpers`` first (e.g. phases.kernel) fail on
    # a partially initialised module.
    from ..actions.executors._grid_server_args import (  # noqa: PLC0415
        tokenize_server_args_preserving_json,
    )

    parsed = tokenize_server_args_preserving_json(args_str)
    if parsed is None:
        return args_str
    normalized, tokens = parsed
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
    rendered = " ".join(out)
    return rendered if rendered != normalized else normalized


# Advisory fields carried on a Critic ``review_verdict`` payload beyond the bare verdict/reasoning.
_VERDICT_ADVISORY_LIST_KEYS: tuple[str, ...] = (
    "required_evidence",
    "risks",
    "notes",
    "kb_evidence",
    "packet_evidence",
)
# The verdict that ends a proposal's life; its counterpart ``ADVISE_VERDICT`` lets the proposal through.
_REJECT_VERDICT: str = "reject"

_VERDICT_ADVISORY_TEXT_KEYS: tuple[str, ...] = (
    "advice_text",
    "alternative_action",
)


def serialize_verdict_advisory(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the advisory field set from a ``review_verdict`` payload."""
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


# The fields a Critic states its grounds in: ``reasoning`` on a single verdict, ``rationale`` on one ``verdict_map``
# entry — the per-variant shape PolicyGate documents and every fixture uses.
_VERDICT_PROSE_KEYS: tuple[str, ...] = ("reasoning", "rationale")

# What a citation looks like: the code opens the verdict's grounds and a colon introduces the finding, the shape the
# field verdict used -- ``"specialist_quantitative_claim_violation: the proposal payload carries the forbidden
# predicted_gain_pct field."`` Nothing may precede the code but whitespace or a backtick, and only the opening line of
# each prose field is read.
_CITATION_OPENER: str = r"[ \t]*`?"


def _opening_prose_lines(entry: dict[str, Any]) -> list[str]:
    """Return the opening line of each prose field ``entry`` states grounds in."""
    openings: list[str] = []
    for key in _VERDICT_PROSE_KEYS:
        for line in str(entry.get(key) or "").splitlines():
            if line.strip():
                openings.append(line)
                break
    return openings


def cited_advisory_reason_code(entry: dict[str, Any]) -> str:
    """Return the advisory-only rule ``entry`` cites, from the field or its prose."""
    advisory = advisory_only_reason_codes()
    explicit = str(entry.get("failure_reason_code") or "").strip()
    if explicit:
        return explicit if explicit in advisory else ""
    # At most one code can open one line, so the sort only fixes the order the candidates are tried in.
    for opening in _opening_prose_lines(entry):
        for code in sorted(advisory):
            if re.match(rf"{_CITATION_OPENER}{re.escape(code)}`?[ \t]*:", opening):
                return code
    return ""


# Priority a batch of per-variant verdicts collapses by: one approved variant carries the proposal, otherwise one
# reject sinks it, and advice outranks a request for more review. :func:`collapse_verdict_map` applies this to the
# proceedable subset first so a genuine reject cannot sink siblings that may still run.
_VERDICT_COLLAPSE_ORDER: tuple[str, ...] = ("approve", _REJECT_VERDICT, ADVISE_VERDICT, "needs_review")
_PROCEEDABLE_VERDICTS: frozenset[str] = frozenset({"approve", ADVISE_VERDICT})


def collapse_verdicts(verdicts: Iterable[str]) -> str:
    """Collapse per-variant verdicts into the one the proposal is decided on."""
    present = set(verdicts)
    for candidate in _VERDICT_COLLAPSE_ORDER:
        if candidate in present:
            return candidate
    return "needs_review"


def proceedable_variant_names(held_by_name: Mapping[str, str]) -> set[str]:
    """Return variant names whose held verdict lets them reach a benchmark."""
    return {name for name, verdict in held_by_name.items() if verdict in _PROCEEDABLE_VERDICTS and str(name).strip()}


def collapse_verdict_map(held_by_name: Mapping[str, str]) -> tuple[str, set[str] | None]:
    """Collapse a held ``verdict_map`` and name the variants that may run."""
    proceedable = proceedable_variant_names(held_by_name)
    if proceedable:
        return collapse_verdicts(held_by_name[name] for name in proceedable), proceedable
    return collapse_verdicts(held_by_name.values()), None


def _states_findings(value: Any) -> bool:
    """Return whether a findings field states anything at all."""
    if isinstance(value, (list, tuple)):
        return any(bool(item) for item in value)
    return bool(value)


def verdict_rests_on_one_ground(entry: dict[str, Any]) -> bool:
    """Return whether ``entry`` refuses for a single reason."""
    if _states_findings(entry.get("required_evidence")):
        return False
    risks = entry.get("risks")
    if not isinstance(risks, (list, tuple)):
        return not _states_findings(risks)
    return len([risk for risk in risks if risk]) <= 1


# The findings a review lists outside its prose: the evidence it still wants and the risks it names.
_VERDICT_FINDING_KEYS: tuple[str, ...] = ("required_evidence", "risks")


def _batch_states_findings(payload: dict[str, Any]) -> bool:
    """Return whether a batch review states a finding of its own."""
    if not isinstance(payload, dict):
        return False
    return any(_states_findings(payload.get(key)) for key in _VERDICT_FINDING_KEYS)


def _inheritable_reason_code(payload: dict[str, Any]) -> str:
    """Return the payload's declared code, when it cannot soften a variant's reject."""
    code = str(payload.get("failure_reason_code") or "").strip()
    return "" if code in advisory_only_reason_codes() else code


def verdict_map_entry_grounds(entry: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Return the grounds one ``verdict_map`` entry rests on."""
    if not isinstance(entry, dict):
        return {}
    grounds = dict(entry)
    if not isinstance(payload, dict):
        return grounds
    if not grounds.get("failure_reason_code"):
        code = _inheritable_reason_code(payload)
        if code:
            grounds["failure_reason_code"] = code
    return grounds


def _stated_verdict(entry: dict[str, Any]) -> str:
    """Return the verdict ``entry`` states, whatever a hold makes of it."""
    return str(entry.get("verdict") or "").strip()


def verdict_held_to_its_rule(entry: dict[str, Any], *, action_name: str) -> tuple[str, str]:
    """Return the verdict a ``review_verdict`` entry carries, and why it moved."""
    if not isinstance(entry, dict):
        return "", ""
    verdict = _stated_verdict(entry)
    if verdict != _REJECT_VERDICT:
        return verdict, ""
    if not advisory_rules_govern(action_name):
        return verdict, ""
    if not verdict_rests_on_one_ground(entry):
        return verdict, ""
    reason_code = cited_advisory_reason_code(entry)
    if reason_code:
        return ADVISE_VERDICT, reason_code
    return verdict, ""


def verdict_map_entry_held_to_its_rule(
    entry: dict[str, Any],
    payload: dict[str, Any],
    *,
    action_name: str,
) -> tuple[str, str]:
    """Return the verdict one ``verdict_map`` entry carries, and why it moved."""
    grounds = verdict_map_entry_grounds(entry, payload)
    if _batch_states_findings(payload):
        return _stated_verdict(grounds), ""
    return verdict_held_to_its_rule(grounds, action_name=action_name)


# Minimum over-baseline gain a same-harness revalidation must show to count as "engaged"; detects a collapse back to
# ~baseline.
_MIN_KERNEL_ENGAGED_GAIN_PCT: float = 2.0

# |measurement_divergence_pct| above this (GEAK vs orchestrator, same config) is logged as a measurement-mismatch
# warning at geak promote.
_GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT: float = 3.0


def _split_env_and_flags(env_str: str) -> tuple[dict[str, str], str]:
    """Split a bench-style config string into (env dict, flags string)."""
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


def _accepted_config_as_variant(cfg: Any) -> tuple[str, dict[str, str]]:
    """Normalize a GEAK ``accepted_config`` into the ``(args, envs)`` a variant runs."""
    cfg = cfg if isinstance(cfg, dict) else {}
    if cfg.get("env_unparsed"):
        log.warning("GEAK accepted_config.env_unparsed reports discarded source text; using only validated env_map")
    flags = str(cfg.get("flags") or "").strip()
    if "env_map" in cfg:
        envs = cfg["env_map"]
        if not isinstance(envs, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
            or "\0" in value
            for key, value in envs.items()
        ):
            raise ValueError("GEAK accepted_config.env_map must map strings to strings")
    else:
        envs, extra_flags = _split_env_and_flags(str(cfg.get("env") or ""))
        if extra_flags:
            flags = (flags + " " + extra_flags).strip()
    envs, _dropped = filter_untrusted_env_mapping(envs, allow_predicate=is_allowed_variant_env_key)
    return flags, envs


def _accepted_config_controls(cfg: Any) -> dict[str, Any]:
    """Normalize explicit GEAK launch controls; unmarked flags remain a delta.

    ``args_mode=replace`` attests that ``flags`` is complete. Neither a result
    schema version nor an empty environment mapping carries that meaning.
    Environment removals precede current assignments; an assignment re-enables
    the name even when inherited controls still list it in ``unset_envs``.
    """
    from ..actions.executors._proposal_identity import controls_of, normalize_proposal

    return controls_of(normalize_proposal(cfg if isinstance(cfg, dict) else {}))


def _geak_revalidation_decision(
    *,
    measured: Any,
    baseline: Any,
    got_hash: str,
    expected_hash: str,
    min_engaged_gain_pct: float,
    current_best: Any = None,
) -> str:
    """Decide a geak same-harness (2b) rebench outcome."""
    measured_ok = isinstance(measured, (int, float)) and measured > 0
    baseline_ok = isinstance(baseline, (int, float)) and baseline > 0
    if not (measured_ok and baseline_ok):
        return "fallback"
    cfg_ok = (not expected_hash) or (str(got_hash or "") == str(expected_hash))
    engaged = float(measured) >= float(baseline) * (1.0 + float(min_engaged_gain_pct) / 100.0)
    if not (cfg_ok and engaged):
        return "fallback"
    if isinstance(current_best, (int, float)) and current_best > 0 and float(measured) <= float(current_best):
        return "no_promote"
    return "validated"


def _geak_result_has_material(
    result: Any,
    *,
    prev_best_flags: str = "",
    prev_best_envs: Any = None,
    prev_best_controls: Any = None,
) -> bool:
    """Decide whether a GEAK result carries a material optimization product."""
    from hyperloom.orchestrator.actions.executors._canonical_fingerprint import (
        canonical_fingerprint,
    )

    def _has_nonempty(entries: Any) -> bool:
        # A list whose items are all empty/blank (e.g. ``[""]``) is not material.
        if not isinstance(entries, (list, tuple, set)):
            return bool(entries)
        return any(str(e).strip() for e in entries)

    if not isinstance(result, dict) or not result:
        return True
    try:
        accepted_flags, parsed_envs = _accepted_config_as_variant(result.get("accepted_config"))
    except ValueError as exc:
        log.warning("GEAK result has invalid accepted_config: %s", exc)
        return False
    if _has_nonempty(result.get("accepted_kernels")):
        return True
    if _has_nonempty(result.get("accepted_heads")):
        return True
    if str(result.get("final_overlay") or "").strip():
        return True
    if str(result.get("final_patch") or "").strip():
        return True
    controls = _accepted_config_controls(result.get("accepted_config"))
    # A missing / all-empty accepted_config carries no config optimization; a
    # bare fingerprint mismatch against a non-empty current_best is NOT material
    # (promoting it would wipe the existing config to empty).
    if not accepted_flags and not parsed_envs and not controls:
        return False
    # Both sides go through the same guard: a resume can hand current_best the raw accepted_config, and an untrusted
    # key on one side only reads as a diff.
    prev_envs, _dropped = filter_untrusted_env_mapping(
        dict(prev_best_envs or {}),
        allow_predicate=is_allowed_variant_env_key,
    )
    got_fp = canonical_fingerprint(accepted_flags, parsed_envs, **controls)
    prev_fp = canonical_fingerprint(
        str(prev_best_flags or ""), prev_envs, **(_accepted_config_controls(prev_best_controls) if controls else {})
    )
    return got_fp != prev_fp


def _normalize_geak_overlay_dir(overlay: str) -> str:
    """Normalize a GEAK ``final_overlay`` path to the loadable overlay dir."""
    if not overlay:
        return overlay
    try:
        p = Path(overlay)
        child = p / "overlay"
        if p.is_dir() and child.is_dir():
            return str(child)
    except (OSError, ValueError):
        return overlay
    return overlay


# A GEAK candidate slot tag (``cand_c0_triton``, ``c1_triton``), as opposed to the name of the kernel the slot
# produced.
_GEAK_CAND_TAG_RE = re.compile(r"^(cand[_-])?c\d+([_-]|$)", re.IGNORECASE)


def geak_is_cand_tag(name: Any) -> bool:
    """Return True when ``name`` is a GEAK slot tag, not a kernel symbol."""
    text = str(name or "").strip()
    return bool(text) and bool(_GEAK_CAND_TAG_RE.match(text))


def _geak_spec_name(spec: Any) -> str:
    """Return the display name of one GEAK acceptance entry."""
    if isinstance(spec, str):
        return spec.strip()
    if not isinstance(spec, dict):
        return ""
    return str(spec.get("short_name") or spec.get("kernel_id") or spec.get("cand_tag") or "").strip()


def geak_spec_name(spec: Any) -> str:
    """Public alias of :func:`_geak_spec_name` for out-of-module readers."""
    return _geak_spec_name(spec)


def geak_spec_kind(spec: Any) -> str | None:
    """Return the acceptance ``kind``, or ``None`` when the source omits it."""
    if not isinstance(spec, dict):
        return None
    raw = spec.get("kind")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def geak_spec_is_env(spec: Any) -> bool:
    """Return True only when the acceptance is *known* to be an env selection."""
    return geak_spec_kind(spec) == "env"


def _geak_accepted_kernel_specs(result: Any) -> list[dict[str, Any]]:
    """Return the authored kernels a GEAK result accepted, both lanes, deduped."""
    if not isinstance(result, dict):
        return []
    out: list[dict[str, Any]] = []
    index: dict[tuple[str, str], int] = {}
    lanes = (result.get("accepted_kernels") or []) + (result.get("accepted_heads") or [])
    for k in lanes:
        if not isinstance(k, dict):
            continue
        if geak_spec_is_env(k):
            continue
        try:
            delta = float(k.get("e2e_delta_pct") or 0.0)
        except (TypeError, ValueError):
            continue
        if delta <= 0.0:
            continue
        name = str(k.get("short_name") or k.get("kernel_id") or k.get("cand_tag") or "").strip()
        if not name:
            continue
        twin = (str(k.get("op_kind") or ""), f"{delta:.4f}")
        pos = index.get(twin)
        if pos is None:
            index[twin] = len(out)
            out.append(k)
            continue
        existing_name = _geak_spec_name(out[pos])
        if _GEAK_CAND_TAG_RE.match(existing_name) and not _GEAK_CAND_TAG_RE.match(name):
            out[pos] = k
            continue
        if _GEAK_CAND_TAG_RE.match(name) and not _GEAK_CAND_TAG_RE.match(existing_name):
            continue
        if name == existing_name:
            continue
        out.append(k)
    return out


def _geak_has_accepted_kernel(result: Any) -> bool:
    """Report whether a GEAK result carries an accepted kernel that gained."""
    return bool(_geak_accepted_kernel_specs(result))


def _geak_overlay_is_loadable(overlay: str) -> bool:
    """Report whether an overlay dir can actually install an authored kernel."""
    from hyperloom.common.overlay import overlay_is_loadable

    return overlay_is_loadable(overlay)


def _geak_overlay_digest(overlay: str) -> str:
    """Digest the overlay's bind manifest, or ``\"\"`` when it has none."""
    if not overlay:
        return ""
    root = Path(overlay)
    try:
        raw = (root / "_overlay_manifest.json").read_bytes()
    except (OSError, ValueError):
        return ""
    hasher = hashlib.sha256()
    hasher.update(raw)
    try:
        spec = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        spec = None
    if isinstance(spec, dict):
        bodies: set[str] = set()
        for mod in spec.get("modules") or []:
            if isinstance(mod, dict) and str(mod.get("file") or "").strip():
                bodies.add(str(mod["file"]).strip())
        for rebind in spec.get("rebinds") or []:
            if isinstance(rebind, dict) and str(rebind.get("impl_module") or "").strip():
                bodies.add(f"{str(rebind['impl_module']).strip()}.py")
        for rel in sorted(bodies):
            hasher.update(rel.encode("utf-8", "replace"))
            try:
                hasher.update(hashlib.sha256((root / rel).read_bytes()).digest())
            except (OSError, ValueError):
                continue
    return hasher.hexdigest()[:16]


def _geak_sweep_measured_tput(res: dict[str, Any]) -> float | None:
    """The measured throughput a ``sweep_via_geak`` replay produced, or None."""
    if not isinstance(res, dict):
        return None
    best = res.get("promotion_measurement")
    if not isinstance(best, dict):
        return None
    tput = best.get("output_throughput")
    return float(tput) if isinstance(tput, (int, float)) and tput > 0 else None


#: Visible-device env masks, in the repo's ROCm precedence order.
#: The pin-resolution chain, imported rather than re-declared: the same tuple
#: and the same parser had five copies in this repo (``bus/gpu_pool``,
#: ``policy/gate``, ``actions/executors/_ray_serving``, ``common/env_safety``,
#: and this module) and their empty-mask semantics had already drifted apart.
#: ``hyperloom.common.visible_devices`` is now the single definition and is
#: dependency-free, so this pure-helper layer can use it without dragging in
#: the SQLite connection ``gpu_pool`` owns.
#:
#: Note this resolver uses the FULL chain, not the three vars the
#: capacity-counting layers read: it answers "where is this run pinned", and a
#: run pinned with ``HSA_VISIBLE_DEVICES`` or ``GPU_DEVICE_ORDINAL`` is really
#: pinned. Those layers keep their narrower :data:`COUNTING_VISIBLE_DEVICE_VARS`
#: because widening them would change GPU accounting repo-wide.
_VISIBLE_DEVICE_VARS: tuple[str, ...] = VISIBLE_DEVICE_VARS

_mask_tokens = mask_tokens
_parse_device_list = parse_device_list


def _is_autofilled_rocr(*, value: str, recipe_envs: Mapping[str, Any]) -> bool:
    """Is this recipe's ROCR mask the materializer's autofill rather than a pin?

    ``materialize_config_with_envs`` unconditionally writes
    ``ROCR_VISIBLE_DEVICES=0..tp-1`` into ``benchmark.envs`` whenever the mask
    is absent or narrower than TP (``_workload_envs.py``). Every materialized
    recipe therefore carries the key, so a recipe ROCR value that is
    byte-identical to that default carries no information about where the run
    is actually pinned — treating it as a pin is what made this resolver
    override a real ``HIP_VISIBLE_DEVICES`` and re-pin GEAK to cards ``0..tp-1``.

    A hand-authored ``ROCR_VISIBLE_DEVICES: "0,1"`` at ``TP=2`` is
    indistinguishable from the autofill and is also treated as "not a pin";
    that is harmless, because the unpinned path emits the same ``gpu_ids`` and
    merely omits ``gpu_pin``.

    When the recipe carries no usable ``TP`` — a hand-written or pre-clamp YAML
    — there is no width to compare against, so the test falls back to the SHAPE
    the materializer always produces: a mask that is exactly ``0..n-1`` for its
    own length. Returning ``False`` there instead would let the synthetic mask
    pose as a pin for precisely the recipes that never recorded a TP, which is
    the hole this function exists to close.

    Args:
        value: The recipe's ROCR mask, already stripped.
        recipe_envs: The recipe's ``benchmark.envs`` (read for its resolved TP).

    Returns:
        ``True`` when the value equals the ``0..tp-1`` the materializer would
        have synthesized — or, absent a recipe TP, the ``0..n-1`` shape of one.
    """
    tokens = _mask_tokens(value)
    if not tokens:
        return False
    try:
        tp = int(str(recipe_envs.get("TP") or 0))
    except (TypeError, ValueError):
        tp = 0
    if tp <= 0:
        tp = len(tokens)
    return tokens == [str(i) for i in range(tp)]


def _mask_value(raw: Any) -> str:
    """Normalize a raw mask (string or YAML sequence) to its string form.

    A YAML ``ROCR_VISIBLE_DEVICES: [4, 5]`` reaches us as a list, and
    ``str([4, 5])`` would produce ``"[4, 5]"`` — a value no consumer can export.

    Args:
        raw: The value as read from the env mapping or the recipe.

    Returns:
        The comma-joined, stripped mask; ``""`` for an empty or blank one.
    """
    if isinstance(raw, (list, tuple)):
        return ",".join(str(p).strip() for p in raw if str(p).strip())
    return str(raw if raw is not None else "").strip()


def _resolve_inner_hip_mask(
    *,
    var: str,
    env: Mapping[str, str],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    """The HIP-level mask nested inside a winning ROCr-level pin, if any.

    ``ROCR_VISIBLE_DEVICES=4,5,6,7`` with ``HIP_VISIBLE_DEVICES=2,3`` does not
    mean "cards 2 and 3": HIP indexes INTO what ROCr exposed, so the run is on
    absolute cards 6 and 7. Dropping the inner mask and advertising
    ``0..tp-1`` would move the servers to cards 4 and 5 — a quieter version of
    the same #1312 bug, so the inner mask travels with the pin.

    Args:
        var: The winning mask variable.
        env: Process environment mapping.
        recipe: The baseline recipe's ``benchmark.envs``.

    Returns:
        ``{"var", "value", "ids", "count", "source"}`` for the innermost
        HIP-level mask, or ``{}`` when the winner is not ROCr-level or no
        HIP-level mask is set.
    """
    if not is_rocr_level(var):
        return {}
    for hip_var in HIP_LEVEL_VARS:
        for source, table in (("process_env", env), ("baseline_recipe", recipe)):
            raw = table.get(hip_var)
            if raw is None:
                continue
            value = _mask_value(raw)
            if not value:
                # Set but empty is terminal here too, for the same reason it is
                # in :func:`_resolve_gpu_pin`: ``HIP="" + CUDA=4,5`` exposes
                # zero devices, so the CUDA mask must not be picked up as the
                # inner one. Reported as a zero-device inner mask, which drives
                # the whole pin to ``count == 0``.
                return {"var": hip_var, "value": "", "ids": [], "count": 0, "source": source}
            return {
                "var": hip_var,
                "value": value,
                "ids": _parse_device_list(value),
                "count": len(effective_mask_tokens(value)),
                "source": source,
            }
    return {}


def _resolve_gpu_pin(
    *,
    recipe_envs: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the run's ACTUAL GPU pin for the geak handoff.

    GEAK launches full servers out-of-process and re-writes a visible-devices
    mask for each one. Without the pin it can only guess, and the guess
    (``0..tp-1``) silently lands on physical GPU 0 — see issue #1312, where a
    run pinned elsewhere collided with a foreign tenant on card 0. Forwarding
    the pin lets the consumer compose masks instead of clobbering them.

    Precedence is VARIABLE-major: ``ROCR_VISIBLE_DEVICES`` before ``HIP``
    before ``CUDA`` — the repo-wide order — and within each variable the
    process env before the baseline recipe. Source-major ordering was wrong in
    both directions: a leftover recipe ``CUDA_VISIBLE_DEVICES`` would outrank a
    real process ROCR pin, and the recipe's autofilled ROCR (see
    :func:`_is_autofilled_rocr`) would outrank everything.

    A mask that is SET BUT EMPTY ends the walk where it stands. It is not a
    weaker pin that a real mask further down can beat: it hides every device,
    and a HIP-level mask indexes into what ROCr left visible rather than
    restoring it. Treating it as a fallback is what let ``ROCR="" +
    HIP=4,5`` report two cards for a run that ROCm refuses to start at all.

    Args:
        recipe_envs: The baseline recipe's ``benchmark.envs`` mapping (may be
            ``None`` when no recipe is materialized yet).
        environ: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        ``{"var", "value", "ids", "count", "source"}`` for the winning mask.
        ``ids`` are the ABSOLUTE NUMERIC device ids and ``count`` is how many
        devices the mask exposes; both derive from
        :func:`effective_mask_tokens`, so ``count >= len(ids)`` always, and
        they differ only when the mask is (partly) non-numeric — a UUID mask
        gives ``ids == []`` with a non-zero ``count``. ``source`` is
        ``"process_env"`` or ``"baseline_recipe"``.
        A mask that is SET BUT EMPTY yields ``count == 0`` (zero devices
        visible) rather than ``{}``, and wins outright over anything below it
        in the chain; when the winner is a ROCr-level mask and a HIP-level mask
        is also in force, the latter travels under ``"inner"`` because it
        selects a subset *within* the ROCr-visible set — and an EMPTY inner
        mask drives the pin's own ``count`` to 0, since it leaves nothing
        usable however many cards the ROCr mask exposes.
        ``{}`` only when no mask is set anywhere — meaning "whole machine
        visible", not "pinned to 0".
    """
    env = os.environ if environ is None else environ
    recipe = dict(recipe_envs or {})
    for var in _VISIBLE_DEVICE_VARS:
        for source, table in (("process_env", env), ("baseline_recipe", recipe)):
            raw = table.get(var)
            if raw is None:
                continue
            value = _mask_value(raw)
            if not value:
                # Present but empty: TERMINAL, not a fallback. An empty mask
                # hides every device, and nothing further down the chain can
                # re-expose one — a HIP-level mask can only index INTO what
                # ROCr left visible. Measured on ROCm 7.2 / MI350X, reading the
                # ROCr agent count out of ``rocminfo`` rather than
                # ``torch.cuda.device_count()`` (which reports a lazy ``1``
                # here and only raises on first use):
                #
                #   ROCR=""                -> 0 agents
                #   ROCR="" + HIP=0        -> 0 agents
                #   ROCR="" + CUDA=4,5     -> 0 agents
                #   ROCR="" + HSA=4,5      -> 0 agents
                #   HIP=""  + CUDA=4,5     -> 0 usable devices
                #
                # ``ROCR="" + HIP=4,5`` does not even reach a device count: HIP
                # aborts with "HIP_VISIBLE_DEVICES contains more devices than
                # ROCR_VISIBLE_DEVICES". Letting the HIP mask win here put two
                # cards that cannot exist into the handoff, and the consumer
                # died on that abort at server start.
                return {"var": var, "value": "", "ids": [], "count": 0, "source": source}
            if (
                source == "baseline_recipe"
                and is_rocr_level(var)
                and _is_autofilled_rocr(value=value, recipe_envs=recipe)
            ):
                continue
            pin: dict[str, Any] = {
                "var": var,
                "value": value,
                "ids": _parse_device_list(value),
                "count": len(effective_mask_tokens(value)),
                "source": source,
            }
            inner = _resolve_inner_hip_mask(var=var, env=env, recipe=recipe)
            if inner:
                pin["inner"] = inner
                if int(inner.get("count") or 0) <= 0:
                    # An empty HIP mask nested in a ROCr pin still leaves the
                    # run with nothing usable: ``ROCR=4,5 + HIP=""`` keeps two
                    # ROCr agents but exposes zero devices to HIP (measured, as
                    # above). The pin keeps the ROCr mask as its ``value`` for
                    # diagnostics, but its device count is the effective one, so
                    # the handoff reports the coordinate space as ``"none"``
                    # instead of advertising the two cards ROCr still shows.
                    pin["count"] = 0
            return pin
    return {}


def _resolve_handoff_gpu_ids(*, gpu_pin: Mapping[str, Any] | None, tp: int) -> str:
    """Resolve the handoff's ``gpu_ids`` in the coordinate system GEAK applies it in.

    ``gpu_ids`` is a HIP-level device list: the consumer exports it as
    ``HIP_VISIBLE_DEVICES``/``CUDA_VISIBLE_DEVICES`` for the servers it
    launches, and HIP indexes into the ROCr-visible set. So:

      * pinned with a ROCr-level mask, from the process env or from the
        baseline recipe — either way it is in force for the servers GEAK
        launches and renumbers their devices, so the ids must be LOGICAL
        positions inside it
        (``ROCR=6`` → ``"0"``), capped at ``tp`` (``ROCR=4,5,6,7`` with
        ``tp=2`` → ``"0,1"``) and at the mask width when ``tp`` overshoots it.
        Counted from :func:`effective_mask_tokens`, so a UUID mask resolves to
        the right number of logical slots and a repeated ordinal does not
        invent one. A HIP-level mask nested inside the ROCr slice is already in
        logical coordinates and is forwarded instead (``ROCR=4,5,6,7`` +
        ``HIP=2,3`` is cards 6 and 7, so ``"2,3"``);
      * any other pin — ROCr still shows every card, so the mask's own tokens
        pass through uncapped (``HIP=4,5`` → ``"4,5"``). They come from
        :func:`effective_mask_tokens`, the same list ``gpu_pin["count"]`` is
        derived from, so whitespace is normalized without the id list and the
        advertised device count ever disagreeing. A NON-NUMERIC mask (a UUID
        list) is forwarded token for token rather than collapsed to
        ``0..tp-1``, which would silently move the servers onto cards
        ``0..tp-1`` — the #1312 failure this resolver exists to prevent;
      * not pinned — ``0..tp-1``, unchanged.

    The absolute pin travels separately in ``handoff["gpu_pin"]``, and
    :func:`_resolve_handoff_gpu_ids_space` says which of the two coordinate
    systems the result is in. Because EVERY ROCr-level pin now yields logical
    ids, both consumer styles agree: exporting the result as
    ``HIP_VISIBLE_DEVICES`` is correct, and so is re-applying
    ``gpu_pin["value"]`` as ``ROCR_VISIBLE_DEVICES`` and then these ids as the
    inner HIP mask. No case is left in which the consumer has to switch which
    field it reads.

    Args:
        gpu_pin: The :func:`_resolve_gpu_pin` result (``{}``/``None`` = unpinned).
        tp: Tensor-parallel size; ``<= 1`` is treated as 1.

    Returns:
        A comma-separated device list, never empty.
    """
    width = max(int(tp or 1), 1)
    pin = gpu_pin or {}
    ids = list(pin.get("ids") or [])
    # Logical remapping applies to any ROCr-level pin, from either source: a
    # process-env mask reaches the servers through GEAK, and a recipe mask
    # reaches them directly as ``handoff["launch_recipe"]``. Either way the
    # servers see a renumbered set, so absolute ids would index out of it.
    if _pin_renumbers_devices(pin):
        # Token count, not len(ids): a UUID mask parses to zero numeric ids but
        # still exposes that many cards to the child. Defaulted rather than
        # ``or``-chained, so an explicit ``count == 0`` (an empty mask, or an
        # empty HIP mask nested in this pin) stays zero instead of falling back
        # to the ids of a mask that exposes nothing.
        visible = int(pin.get("count", len(ids)))
        if visible > 0:
            # A HIP-level mask nested inside the ROCr pin is ALREADY expressed
            # in the child's logical coordinates, so it is forwarded as-is
            # rather than overwritten with ``0..n-1``. Out-of-range entries are
            # dropped: they name devices the ROCr mask never exposed.
            # Effective, not literal: ``-1`` names no device and a repeated
            # ordinal is not a second one, and either would otherwise travel
            # into ``gpu_ids`` and inflate the ``tp`` derived from it.
            inner = effective_mask_tokens((pin.get("inner") or {}).get("value"))
            kept = [tok for tok in inner if not tok.isdigit() or int(tok) < visible]
            if kept:
                return ",".join(kept[:width])
            return ",".join(str(i) for i in range(min(visible, width)))
    # Forward the EFFECTIVE tokens, not a re-serialization of the parsed ints:
    # a UUID mask has no ints to re-serialize and would otherwise collapse to
    # ``0..tp-1`` (the #1312 failure), and ``pin["count"]`` is derived from this
    # same list, so the id list and the advertised device count cannot disagree.
    tokens = effective_mask_tokens(pin.get("value"))
    if tokens:
        return ",".join(tokens)
    return ",".join(str(i) for i in range(width))


def _pin_renumbers_devices(pin: Mapping[str, Any] | None) -> bool:
    """Will this pin's ROCr slice be in force for the servers GEAK launches?

    Only then are the handoff's ``gpu_ids`` logical -- and the question is about
    the SERVERS, not about the GEAK process. An earlier version asked whether
    GEAK itself inherits the mask (``source == "process_env"``), which is true
    of the process env and false of the recipe. That was the wrong level: GEAK
    starts its servers from ``handoff["launch_recipe"]``, and a recipe-sourced
    ``ROCR_VISIBLE_DEVICES`` is applied to exactly those servers. The
    renumbering still happens, one level down, so calling those ids absolute
    made a mask index out of its own slice -- ``ROCR=4,5,6,7`` re-exported as
    ``HIP=4,5,6,7`` indexes 4..7 into a four-element set and the server dies on
    an invalid ordinal. Both sources renumber; only the LEVEL of the mask
    decides.

    Args:
        pin: The :func:`_resolve_gpu_pin` result.

    Returns:
        ``True`` for any ROCr-level pin, whatever its source.
    """
    return is_rocr_level(str((pin or {}).get("var") or ""))


def _resolve_handoff_gpu_ids_space(*, gpu_pin: Mapping[str, Any] | None) -> str:
    """Which coordinate system the handoff's ``gpu_ids`` are expressed in.

    ``gpu_ids`` alone is ambiguous: ``"0,1"`` is either "the first two cards of
    the in-force ROCr mask" or "absolute cards 0 and 1", and a consumer that
    guesses wrong re-pins the servers onto physical GPU 0 — issue #1312. This
    field makes the distinction explicit so a consumer that composes masks
    itself (rather than exporting ``gpu_ids`` into HIP) can tell which it was
    handed. Consumers that ignore it keep the old, correct behaviour of
    exporting ``gpu_ids`` as ``HIP_VISIBLE_DEVICES``, which is a HIP-level
    variable in both spaces.

    ``"none"`` is the third case and the reason this is a tri-state rather
    than a boolean: the mask is SET BUT EMPTY, so the run has no visible
    devices and NO id list can be truthful. ``gpu_ids`` still carries
    ``0..tp-1`` because the consumer reads a falsy ``gpu_ids`` as "unset" and
    falls back to exactly those ids anyway (``interface/run_e2e.py``) — an
    empty string would buy nothing and lose the ability to say why. The ids are
    placeholders in that case and a consumer must not launch on them.

    Args:
        gpu_pin: The :func:`_resolve_gpu_pin` result (``{}``/``None`` = unpinned).

    Returns:
        ``"none"`` when the pin exposes zero devices, ``"logical"`` when the
        ids index into a ROCr mask that is in force for the launched servers,
        ``"absolute"`` otherwise (including unpinned).
    """
    pin = gpu_pin or {}
    if pin and int(pin.get("count") or 0) <= 0:
        return "none"
    return "logical" if _pin_renumbers_devices(pin) else "absolute"


def _coerce_tp(*args: Any, default: int = 1) -> int:
    """First positional that parses as a positive int, else ``default``.

    Every candidate is guarded, so no caller has to wrap ``int()`` in a
    ``try`` whose handler then calls ``int()`` again on a value that can raise
    the same exception it is handling.

    Args:
        *args: Candidate TP values in precedence order (``None``/blank skipped).
        default: Returned when nothing parses; floored at 1.

    Returns:
        A TP of at least 1.
    """
    for cand in args:
        text = str(cand if cand is not None else "").strip()
        if not text:
            continue
        try:
            val = int(text)
        except (TypeError, ValueError):
            continue
        if val > 0:
            return val
    return max(int(default), 1)


def _resolve_handoff_tp(*, gpu_ids: str, tp: int) -> int:
    """Clamp ``tp`` to the number of devices the handoff actually advertises.

    ``gpu_ids`` is capped at the pin's mask width, so a run whose ``$TP``
    overshoots its pin (``ROCR=6`` with ``TP=2``, or a stale ``TP=8`` against a
    materializer-clamped 4-card recipe) would otherwise ship ``tp`` and
    ``gpu_ids`` that disagree — and GEAK would launch ``--tp N`` against fewer
    visible cards and fail to load weights. Deriving both from the same resolved
    mask makes that state unrepresentable.

    Args:
        gpu_ids: The resolved handoff ``gpu_ids`` string.
        tp: The TP resolved from the recipe/process env.

    Returns:
        ``min(tp, len(gpu_ids))``, never below 1.
    """
    advertised = len(_mask_tokens(gpu_ids))
    if advertised <= 0:
        return max(int(tp or 1), 1)
    return max(min(int(tp or 1), advertised), 1)


def _parse_server_arg_value(server_args: str, flag: str) -> str | None:
    """Extract a CLI flag's value from a server-args string."""
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
            return tok[len(prefix) :]
    return None


def _resolve_serving_fidelity(
    *,
    baseline_server_args: str,
    state_max_model_len: int = 0,
) -> dict[str, Any]:
    """Resolve serving-fidelity knobs to forward in the geak handoff."""
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
