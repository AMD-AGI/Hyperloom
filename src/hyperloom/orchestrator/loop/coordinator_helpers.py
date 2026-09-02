# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure, self-contained helpers used by the Coordinator.

No dependency on Coordinator state; must not import ``coordinator`` (one-way
dependency).
"""

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

# Constants below are read from other modules; listed here to mark them as
# intentionally exported.
__all__ = [
    "TIME_BUDGET_EXEMPT_ACTIONS",
    "_GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT",
    "_MIN_KERNEL_ENGAGED_GAIN_PCT",
    "action_fits_time_budget",
    "coerce_needs_gpu",
    "expected_action_cost_minutes",
    "measured_baseline_runtime_sec",
]


def coerce_needs_gpu(value: Any) -> bool:
    """Coerce a ``needs_gpu`` specialist parameter value to a Python bool.

    Specialist params arrive as JSON-decoded values which may be a bare bool
    or a string (``"true"``, ``"1"``, ``"yes"``, ``"on"``).  Handles both so
    callers don't repeat this conversion.

    Args:
        value: The raw ``needs_gpu`` parameter value (bool, str, or anything
            else that ``bool()`` can handle).

    Returns:
        ``True`` when ``value`` is a truthy string token or a truthy non-string
        value; ``False`` otherwise.
    """
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


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
# A roofline leg costs the better part of an hour, so retrying without bound
# would spend a session re-measuring a broken collector; giving up after the
# first failure is what left four sessions with no GPU evidence at all.
_MAX_ROOFLINE_FAILURE_RETRIES: int = 3


# Actions that must stay startable no matter how little budget is left: they
# are how a session ends cleanly, so a time gate that refused them would
# strand the run with nothing to show. ``recover`` is not among them — it
# takes the server-lifecycle lane, prices at five catalogue minutes, and
# holds a twenty-minute lease; treating it as a closing action is what let a
# spent session keep working past the wall clock.
TIME_BUDGET_EXEMPT_ACTIONS: frozenset[str] = frozenset(
    {
        "report",
        "session_breakdown",
    }
)

# The lanes that serialize GPU work: an action requiring one of them spends its
# time running a benchmark round, so what this session measured says more about
# it than a catalogue estimate does.
_GPU_BENCH_LANES: frozenset[str] = frozenset(
    {
        "benchmark_lane",
        "profile_lane",
    }
)


def measured_baseline_runtime_sec(shared_state: Any | None) -> float:
    """Read this session's own measured baseline round, in seconds.

    Args:
        shared_state (Any | None): The session ``SharedState``, or ``None`` when
            the caller has no session context.

    Returns:
        float: The measured baseline runtime; ``0.0`` when the session has not
            landed a baseline yet, which every caller reads as "no measurement".
    """
    try:
        return max(0.0, float(getattr(shared_state, "baseline_runtime_sec", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _action_benches_on_gpu(meta: Any | None) -> bool:
    """Whether an action's cost is dominated by a benchmark round on the GPU.

    Read off the lanes the action must hold rather than off a list of names, so
    an action added to the catalogue is classified by what it does. The
    benchmark and profile lanes are exactly the two that serialize GPU work; an
    action holding neither (``report``, ``target_analysis``, ``specialist``)
    costs what its own bookkeeping costs and has nothing to do with model size.

    Args:
        meta (Any | None): The action's catalogue metadata.

    Returns:
        bool: ``True`` when the action runs at least one benchmark round.
    """
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
    """Read an action's expected cost, preferring what this session measured.

    Every budget guard goes through here so the field is named once. Reading it
    inline with a ``getattr`` default turned the catalogue's move off YAML —
    which renamed the field — into a gate that admitted everything without a
    word, because "no estimate on record" and "the field moved" look the same
    from a default.

    The catalogue's estimates are calibrated on small models (``baseline`` 5
    min, ``roofline`` 10 min) while the two sessions that motivated the
    wall-clock work measured 51 and 125 minutes of baseline and an 81-minute
    roofline. A guard anchored on the catalogue alone therefore admits arms the
    session cannot pay for — it would not have stopped either field run. So one
    measured baseline round is taken as a *floor* on any action that runs a
    benchmark round of its own: it is this model on this GPU under this
    workload, which is what those actions spend their time doing. It is a floor
    rather than a replacement because an action that benches several variants
    costs more than one round, never less, and the catalogue is the only thing
    that knows how many.

    Args:
        meta (Any | None): The action's catalogue metadata, or ``None`` for an
            action the catalogue does not carry.
        measured_baseline_sec (float): This session's measured baseline runtime
            in seconds, from :func:`measured_baseline_runtime_sec`; ``0.0``
            before a baseline lands, which leaves the catalogue in charge.

    Returns:
        float: The expected cost in minutes; ``0.0`` when nothing is on record.
    """
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
    """Decide whether an action's expected cost still fits the remaining budget.

    The anchor is the action's *expected* cost (its typical runtime), not its
    pessimistic tail. Judging fit on the pessimistic tail would abandon usable
    budget — with 90 minutes left we would refuse an action that finishes in 60
    minutes half the time — and the session already has a wall-clock reaper for
    the overruns, so the optimistic anchor is the one that keeps the tail of a
    session productive. This mirrors how the grid admits variants.

    Args:
        usable_sec: Budget left after the closing reserve, from
            ``SharedState.session_budget_usable_sec``; ``None`` means unbounded.
        expected_cost_minutes: The action's expected cost in minutes; values at
            or below zero mean "no estimate on record".

    Returns:
        ``True`` when the action may start: the budget is unbounded, no estimate
        is on record, or the expected cost fits what is left.
    """
    if usable_sec is None:
        return True
    if expected_cost_minutes <= 0.0:
        return True
    return usable_sec >= expected_cost_minutes * 60.0


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


def approved_proposal_idempotency_key(action_name: str, params: dict[str, Any] | None) -> str:
    """Content-addressed idempotency key for an approved proposal.

    ``baseline`` keys on :func:`_baseline_params_fingerprint` so params outside
    the eight behavior-determining fields cannot mint a distinct key for what is
    the same run; every other action hashes the full params. Two proposals that
    would launch the same work therefore collide and only one is queued.

    Args:
        action_name: The proposed action kind.
        params: Materialized task params (``None`` treated as empty).

    Returns:
        The ``approved:<action>:<digest>`` key.
    """
    params = params or {}
    payload: Any = _baseline_params_fingerprint(params) if action_name == "baseline" else params
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode(),
        usedforsecurity=False,
    ).hexdigest()[:16]
    return f"approved:{action_name}:{digest}"


def _resolve_roofline_watermark_ratio() -> float:
    """Resolve the roofline watermark ratio.

    Returns:
        The fixed watermark ratio (> 1.0).
    """
    return _DEFAULT_ROOFLINE_WATERMARK_RATIO


def _merge_cumulative_extra_server_args(
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


def _dedupe_extra_server_args(args_str: str) -> str:
    """Collapse repeated ``--flag value`` pairs into a unique launch string.

    Keep each flag once with its last value (first-seen order preserved),
    since argparse ``action="store"`` only honors the last value. Flags in
    ``_MULTI_VALUE_SGLANG_FLAGS`` keep their multi-value runs. Valid JSON blobs
    are treated as opaque tokens, so their inner quotes survive while unrelated
    duplicated flags are still collapsed. Actual whitespace-bearing argv
    values fail closed because downstream launch scripts expand them unquoted.

    Args:
        args_str: The extra server-args string to dedupe.

    Returns:
        The deduped args string, or the input unchanged when it cannot be safely
        tokenized; ``""`` for empty input.
    """
    if not args_str:
        return ""
    # Imported here, not at module scope: ``actions.executors`` re-enters this
    # module through ``session_breakdown``, so a top-level import makes any
    # importer that reaches ``coordinator_helpers`` first (e.g. phases.kernel)
    # fail on a partially initialised module.
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
# The verdict that ends a proposal's life; its counterpart ``ADVISE_VERDICT``
# lets the proposal through. See :func:`verdict_held_to_its_rule`.
_REJECT_VERDICT: str = "reject"

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


# The fields a Critic states its grounds in: ``reasoning`` on a single verdict,
# ``rationale`` on one ``verdict_map`` entry — the per-variant shape PolicyGate
# documents and every fixture uses. ``notes`` is remediation text and
# ``risks[*].summary`` describes the risk, so a rule named in either is being
# discussed rather than invoked; both stay out.
#
# Every key an entry fills is read. They are one speaker's grounds for one
# verdict and nothing ranks them, so trying them in order would let whichever
# happens to come first decide whether the citation in the other is seen.
_VERDICT_PROSE_KEYS: tuple[str, ...] = ("reasoning", "rationale")

# What a citation looks like: the code opens the verdict's grounds and a colon
# introduces the finding, the shape the field verdict used --
# ``"specialist_quantitative_claim_violation: the proposal payload carries the
# forbidden predicted_gain_pct field."`` Nothing may precede the code but
# whitespace or a backtick, and only the opening line of each prose field is
# read.
#
# The two mistakes cost different amounts. Missing a citation costs the round
# its proposals, which the next round can re-propose; reading one that was not
# made dispatches a proposal the Critic meant to block. So the scan stays
# deliberately narrow instead of learning every citation format a model might
# use -- a list marker, a quote marker or a fence is how one *enumerates* the
# rules it checked, and "- <code>: clean." must never read as grounds. The
# reliable path is the explicit ``failure_reason_code`` in the Critic's output
# schema; this is the fallback.
_CITATION_OPENER: str = r"[ \t]*`?"


def _opening_prose_lines(entry: dict[str, Any]) -> list[str]:
    """Return the opening line of each prose field ``entry`` states grounds in.

    Args:
        entry: A ``review_verdict`` payload or one ``verdict_map`` entry; the
            two spell the field differently (:data:`_VERDICT_PROSE_KEYS`), and
            an entry filling both states grounds in both.

    Returns:
        The first non-blank line of each field present, in
        :data:`_VERDICT_PROSE_KEYS` order; empty when the entry states no
        grounds in prose.
    """
    openings: list[str] = []
    for key in _VERDICT_PROSE_KEYS:
        for line in str(entry.get(key) or "").splitlines():
            if line.strip():
                openings.append(line)
                break
    return openings


def cited_advisory_reason_code(entry: dict[str, Any]) -> str:
    """Return the advisory-only rule ``entry`` cites, from the field or its prose.

    ``failure_reason_code`` is the reliable path: the Critic's output schema
    asks for the code of the rule its verdict rests on. Prose is read only as a
    fallback, for a verdict that names its rule in its grounds text instead —
    the shape observed in the field.

    A mention is not a citation: a Critic that clears one rule and refuses on
    another names both, and reading the cleared one as the grounds would
    materialise a proposal it meant to block. Only an unambiguous citation
    counts (see :data:`_CITATION_OPENER`).

    Args:
        entry: A ``review_verdict`` payload or one ``verdict_map`` entry.

    Returns:
        The cited advisory-only reason code, or ``""`` when the entry cites
        none. A code outside the advisory set yields ``""`` too: only rules
        that declared ``advise`` can move a verdict.
    """
    advisory = advisory_only_reason_codes()
    explicit = str(entry.get("failure_reason_code") or "").strip()
    if explicit:
        return explicit if explicit in advisory else ""
    # At most one code can open one line, so the sort only fixes the order the
    # candidates are tried in.
    for opening in _opening_prose_lines(entry):
        for code in sorted(advisory):
            if re.match(rf"{_CITATION_OPENER}{re.escape(code)}`?[ \t]*:", opening):
                return code
    return ""


# Priority a batch of per-variant verdicts collapses by: one approved variant
# carries the proposal, otherwise one reject sinks it, and advice outranks a
# request for more review. :func:`collapse_verdict_map` applies this to the
# proceedable subset first so a genuine reject cannot sink siblings that may
# still run.
_VERDICT_COLLAPSE_ORDER: tuple[str, ...] = ("approve", _REJECT_VERDICT, ADVISE_VERDICT, "needs_review")
_PROCEEDABLE_VERDICTS: frozenset[str] = frozenset({"approve", ADVISE_VERDICT})


def collapse_verdicts(verdicts: Iterable[str]) -> str:
    """Collapse per-variant verdicts into the one the proposal is decided on.

    Args:
        verdicts: The per-variant verdicts of one ``verdict_map``.

    Returns:
        The highest-priority verdict present, or ``needs_review`` when none of
        the known verdicts appears.
    """
    present = set(verdicts)
    for candidate in _VERDICT_COLLAPSE_ORDER:
        if candidate in present:
            return candidate
    return "needs_review"


def proceedable_variant_names(held_by_name: Mapping[str, str]) -> set[str]:
    """Return variant names whose held verdict lets them reach a benchmark.

    ``approve`` and ``advise`` both mean dispatch may proceed; ``reject`` and
    ``needs_review`` do not. Blank names cannot match a grid slot and are
    dropped.

    Args:
        held_by_name: Per-variant verdicts after any hold-to-rule.

    Returns:
        The non-blank names whose verdict is proceedable.
    """
    return {name for name, verdict in held_by_name.items() if verdict in _PROCEEDABLE_VERDICTS and str(name).strip()}


def collapse_verdict_map(held_by_name: Mapping[str, str]) -> tuple[str, set[str] | None]:
    """Collapse a held ``verdict_map`` and name the variants that may run.

    A genuine reject on one variant must not sink siblings the Critic approved
    or advised through. When any variant is proceedable, the summary is the
    collapse of *those* verdicts and the set is the materialize filter.
    Otherwise the summary is the collapse of the whole map and the filter is
    ``None`` (nothing to dispatch).

    Args:
        held_by_name: Per-variant verdicts after any hold-to-rule.

    Returns:
        ``(summary_verdict, approved_variant_names)``. The set is ``None``
        when no variant is proceedable.
    """
    proceedable = proceedable_variant_names(held_by_name)
    if proceedable:
        return collapse_verdicts(held_by_name[name] for name in proceedable), proceedable
    return collapse_verdicts(held_by_name.values()), None


def _states_findings(value: Any) -> bool:
    """Return whether a findings field states anything at all.

    Args:
        value: A ``risks`` or ``required_evidence`` value: the list the schema
            documents, or whatever shape a verdict put there instead.

    Returns:
        True when a list holds at least one non-empty item, or when a value of
        any other shape is non-empty.
    """
    if isinstance(value, (list, tuple)):
        return any(bool(item) for item in value)
    return bool(value)


def verdict_rests_on_one_ground(entry: dict[str, Any]) -> bool:
    """Return whether ``entry`` refuses for a single reason.

    A verdict can cite an advisory rule *and* refuse on its own merits in the
    same breath — "the proposal claims a 12% gain and has no rollback plan".
    Holding that verdict to the advisory rule would let the second half of the
    sentence disappear, so the hold is confined to a reject that names one
    ground and asks for nothing further: at most one risk entry, and no
    outstanding evidence request.

    The allowance rests on the citation and the risk being one statement by one
    author, which is what makes the risk the cited rule's. Findings a *batch*
    states are neither, so they are read where they are stated rather than
    counted here (:func:`verdict_map_entry_held_to_its_rule`).

    ``risks`` is a list in the schema. A verdict that states it as one sentence
    instead has stated grounds whose number cannot be read off — "the patch
    does not apply and there is no rollback plan" is two — so an unlisted
    value counts as more than one rather than as the single ground the hold is
    confined to.

    Risk *severity* deliberately plays no part. ``references/risk_rules.md``
    reserves ``blocker`` for evidence and correctness failures and lists no
    format item, yet the Critic verdict this hold was built for graded its own
    format complaint ``blocker`` — so severity separates nothing here, and
    reading it would only retire the hold on the case that motivated it.

    Args:
        entry: A ``review_verdict`` payload or one ``verdict_map`` entry.

    Returns:
        True when the verdict names at most one ground and requests no further
        evidence.
    """
    if _states_findings(entry.get("required_evidence")):
        return False
    risks = entry.get("risks")
    if not isinstance(risks, (list, tuple)):
        return not _states_findings(risks)
    return len([risk for risk in risks if risk]) <= 1


# The findings a review lists outside its prose: the evidence it still wants
# and the risks it names. A ``verdict_map`` entry is
# ``{verdict, rationale?, failure_reason_code?}`` -- the shape PolicyGate
# documents -- so these have nowhere to live but the payload, where a batch
# review states them once for every variant it looked at.
_VERDICT_FINDING_KEYS: tuple[str, ...] = ("required_evidence", "risks")


def _batch_states_findings(payload: dict[str, Any]) -> bool:
    """Return whether a batch review states a finding of its own.

    Args:
        payload: The ``review_verdict`` payload a ``verdict_map`` arrived in.

    Returns:
        True when the payload names any risk or asks for any evidence.
    """
    if not isinstance(payload, dict):
        return False
    return any(_states_findings(payload.get(key)) for key in _VERDICT_FINDING_KEYS)


def _inheritable_reason_code(payload: dict[str, Any]) -> str:
    """Return the payload's declared code, when it cannot soften a variant's reject.

    Any code outside the advisory set is inherited, including one no rule
    defines. Restricting this to codes a handed rule declares would inherit
    nothing at all -- every rule in ``review_constraints`` declares ``advise``
    (:func:`advisory_only_reason_codes`), so the two sets do not intersect --
    and would let a batch that declared a hard code have its variants
    downgraded on the advisory rules their rationales cite. An unrecognised
    string is not the Critic's permission to dispatch; it withholds the
    downgrade, which is what the batch's own findings do.

    Args:
        payload: The ``review_verdict`` payload.

    Returns:
        The declared ``failure_reason_code``, or ``""`` when it names an
        advisory-only rule.
    """
    code = str(payload.get("failure_reason_code") or "").strip()
    return "" if code in advisory_only_reason_codes() else code


def verdict_map_entry_grounds(entry: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Return the grounds one ``verdict_map`` entry rests on.

    An entry is ``{verdict, rationale?, failure_reason_code?}`` -- the shape
    PolicyGate documents -- so what it states of its own is nearly all it has.
    Prose is not inherited, and neither is a ``failure_reason_code`` naming an
    advisory rule: the batch's citation is a claim about the batch's verdict,
    and a declared code outranks the entry's own rationale rather than
    competing with it, so lending one would stop the entry's grounds being read
    at all. A code that can only withhold the downgrade is inherited
    (:func:`_inheritable_reason_code`), because the two mistakes cost different
    amounts (see :data:`_CITATION_OPENER`).

    The findings a batch states are not inherited either. They are read where
    they are stated, as a hold on every entry in the set
    (:func:`verdict_map_entry_held_to_its_rule`).

    Args:
        entry: One ``verdict_map`` entry.
        payload: The ``review_verdict`` payload that entry arrived in.

    Returns:
        The entry's own keys, plus a declared reason code that can only hold its
        reject; ``{}`` when ``entry`` is not a dict.
    """
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
    """Return the verdict ``entry`` states, whatever a hold makes of it.

    Args:
        entry: A ``review_verdict`` payload or one ``verdict_map`` entry.

    Returns:
        The stated verdict, stripped; ``""`` when the entry states none.
    """
    return str(entry.get("verdict") or "").strip()


def verdict_held_to_its_rule(entry: dict[str, Any], *, action_name: str) -> tuple[str, str]:
    """Return the verdict a ``review_verdict`` entry carries, and why it moved.

    Several review rules declare ``advise`` as their failure verdict precisely
    because rejecting on them costs the round every proposal in the set. That
    declaration is prose in the Critic prompt, so a model that rejects anyway
    silently gets its way. This holds the verdict to what the cited rule asked
    for, which makes the declaration enforceable rather than advisory.

    The hold is narrow by construction: it reaches only the proposal kinds those
    rules are about (:func:`advisory_rules_govern`), and only a reject whose
    *only* stated ground is a rule that asked for advice (see
    :func:`verdict_rests_on_one_ground`). Scoping it by proposal kind is what
    keeps ``advise`` — which means "dispatch may proceed" — from executing an
    ``integrate_patch`` the Critic refused, since the propose-time
    ``PolicyGate`` patch gate does not run a second time on the verdict.

    Args:
        entry: A ``review_verdict`` payload or one ``verdict_map`` entry, with
            a ``verdict`` and the rule it cites — in ``failure_reason_code`` or
            in its own prose (see :func:`cited_advisory_reason_code`).
        action_name: The reviewed proposal's action name.

    Returns:
        A ``(verdict, reason_code)`` pair: the verdict to act on, and the cited
        reason code when it forced a downgrade, else an empty string.
    """
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
    """Return the verdict one ``verdict_map`` entry carries, and why it moved.

    :func:`verdict_rests_on_one_ground` allows a verdict one stated risk beside
    its citation, because on the single path the two are one statement by one
    author -- "the proposal claims a 12% gain and has no rollback plan" names
    the rule and the risk in the same breath. A batch states its risks once for
    the whole set while the citation belongs to the entry, and nothing connects
    them: counting them together would identify the batch's one ground with the
    entry's rule, which is attribution in the direction that dispatches. A set
    refused because no variant in it supplies a rollback plan would run on the
    strength of a formatting rule its rationales happen to cite.

    So a finding the batch states holds every reject in the set, whatever its
    count, and only an entry's own grounds can support a downgrade. That is the
    rule the batch path already had -- what the batch states adds to the grounds
    a variant is held on, never supplies the grounds it is softened on -- with
    arithmetic that claimed more than it could tell taken out of it.

    Known limitation: the downgrade now needs a payload that states no findings
    at all. That is the batch shape the runtime teaches --
    ``{target_proposal_msg_id, verdict_map: {name: {verdict, rationale?,
    failure_reason_code?}}}``, the only batch payload spelled out anywhere in
    ``src`` (PolicyGate's repair hint) and all
    ``references/intent_envelope.md`` asks for. It is not the reject shape
    ``references/verdict_schema.md`` documents: that states one risk *and* one
    required-evidence item, as both reject exemplars in
    ``critic/tests/expected_outputs.json`` do, so a batch written in the
    single-verdict style keeps every reject it wrote -- including one resting on
    nothing but an advisory rule. Recovering that needs attribution the Critic
    is asked for, i.e. a batch shape in the output schema; guessing it from
    prose is what this reading gives up.

    Args:
        entry: One ``verdict_map`` entry.
        payload: The ``review_verdict`` payload that entry arrived in.
        action_name: The reviewed proposal's action name.

    Returns:
        A ``(verdict, reason_code)`` pair, as :func:`verdict_held_to_its_rule`
        returns it.
    """
    grounds = verdict_map_entry_grounds(entry, payload)
    if _batch_states_findings(payload):
        return _stated_verdict(grounds), ""
    return verdict_held_to_its_rule(grounds, action_name=action_name)


# Minimum over-baseline gain a same-harness revalidation must show to count as
# "engaged"; detects a collapse back to ~baseline.
_MIN_KERNEL_ENGAGED_GAIN_PCT: float = 2.0

# |measurement_divergence_pct| above this (GEAK vs orchestrator, same config) is
# logged as a measurement-mismatch warning at geak promote.
_GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT: float = 3.0


def _split_env_and_flags(env_str: str) -> tuple[dict[str, str], str]:
    """Split a bench-style config string into (env dict, flags string).

    ``accepted_config.env`` (and any ``KEY=VAL KEY=VAL`` / ``--flag val`` blob)
    is parsed so that every ``KEY=VAL`` token becomes a real environment
    variable and every ``--flag`` (or ``--flag=val``) token is folded back into
    a server-args string. Single source of truth for this parse. No
    key/optimization is special-cased.

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


def _accepted_config_as_variant(cfg: Any) -> tuple[str, dict[str, str]]:
    """Normalize a GEAK ``accepted_config`` into the ``(args, envs)`` a variant runs.

    ``accepted_config.env`` is a benchmark-harness snapshot, so it carries the
    shell/loader keys ``GridVariant`` drops before it fingerprints. Anything that
    fingerprints or dispatches that config has to see the same mapping the
    executor will, or the identity it derives describes a config nothing runs.

    Args:
        cfg: The ``accepted_config`` blob (``flags`` / ``env``); non-dict is empty.

    Returns:
        ``(flags, envs)`` with harness-only flags folded into ``flags`` and
        untrusted env names dropped.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    flags = str(cfg.get("flags") or "").strip()
    envs, extra_flags = _split_env_and_flags(str(cfg.get("env") or ""))
    if extra_flags:
        flags = (flags + " " + extra_flags).strip()
    envs, _dropped = filter_untrusted_env_mapping(envs, allow_predicate=is_allowed_variant_env_key)
    return flags, envs


def _geak_revalidation_decision(
    *,
    measured: Any,
    baseline: Any,
    got_hash: str,
    expected_hash: str,
    min_engaged_gain_pct: float,
    current_best: Any = None,
) -> str:
    """Decide a geak same-harness (2b) rebench outcome.

    Returns ``"validated"`` only when ALL hold:
      * config identity — the ran variant's fingerprint matches the expected
        (skipped when no expected hash was pinned); catches an executor-side
        drop/alter of the optimized config; and
      * engagement — the measured throughput cleared baseline by at least
        ``min_engaged_gain_pct`` (i.e. the optimization actually took effect and
        did not collapse back to an un-optimized relaunch); and
      * improvement — the measured throughput beats the current best (aligns the
        GEAK KEEP gate with forge / integrate_patch, which promote only above
        current_best).
    Returns ``"no_promote"`` when the run is well-measured and engaged over
    baseline but does not beat ``current_best`` — a real measurement, not an
    inconclusive one, so the caller must NOT replay via the GEAK harness (2a).
    Otherwise returns ``"fallback"`` so the caller replays via the GEAK harness
    (2a) for a genuinely inconclusive rebench (bad measurement / config drift /
    baseline collapse).

    Args:
        measured: Rebench output throughput (tok/s).
        baseline: Orchestrator raw baseline throughput (same harness as measured).
        got_hash: Fingerprint of the variant that actually ran.
        expected_hash: Pinned expected fingerprint ("" => identity check skipped).
        min_engaged_gain_pct: Minimum over-baseline gain to count as engaged.
        current_best: Current best throughput (tok/s); ``None``/<=0 disables the
            improvement gate.

    Returns:
        ``"validated"``, ``"no_promote"``, or ``"fallback"``.
    """
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
) -> bool:
    """Decide whether a GEAK result carries a material optimization product.

    FOR THE 2b REVALIDATION CALL SITE ONLY (writeback ``geak_fallback`` path).
    Guards the 2b promote path against pure passthrough noise: when GEAK ships
    no kernel/head/overlay/patch AND echoes the pre-KERNEL current_best config
    back unchanged, a rebench that beats current_best is measurement variance,
    not a kernel gain.

    Material means ANY of:
      * ``accepted_kernels`` has a non-empty entry (kernel rewrites);
      * ``accepted_heads`` has a non-empty entry (attention-head optimizations);
      * ``final_overlay`` non-empty (authored-kernel overlay dir);
      * ``final_patch`` non-empty (source patch);
      * ``accepted_config`` is present with a non-empty flags/env that differs
        from the pre-KERNEL current_best config (a kernel enabled via a config
        switch, e.g. an ASM-GEMM env flag). A missing or all-empty
        ``accepted_config`` is NEVER material: an empty config that merely
        differs from a non-empty current_best would otherwise promote and wipe
        the existing config.

    An empty/absent result cannot be judged here and returns ``True``; the sole
    2b call site disambiguates it (a pre-existing ``geak_e2e`` stack entry means
    a resume revalidation of an already-material win, otherwise no material).

    Args:
        result: The normalized GEAK ``geak_result`` blob.
        prev_best_flags: Pre-KERNEL current_best ``extra_server_args``.
        prev_best_envs: Pre-KERNEL current_best ``extra_envs`` mapping.

    Returns:
        ``True`` when a material product exists (or cannot be judged); else
        ``False``.
    """
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
    if _has_nonempty(result.get("accepted_kernels")):
        return True
    if _has_nonempty(result.get("accepted_heads")):
        return True
    if str(result.get("final_overlay") or "").strip():
        return True
    if str(result.get("final_patch") or "").strip():
        return True
    accepted_flags, parsed_envs = _accepted_config_as_variant(result.get("accepted_config"))
    # A missing / all-empty accepted_config carries no config optimization; a
    # bare fingerprint mismatch against a non-empty current_best is NOT material
    # (promoting it would wipe the existing config to empty).
    if not accepted_flags and not parsed_envs:
        return False
    # Both sides go through the same guard: a resume can hand current_best the
    # raw accepted_config, and an untrusted key on one side only reads as a diff.
    prev_envs, _dropped = filter_untrusted_env_mapping(
        dict(prev_best_envs or {}),
        allow_predicate=is_allowed_variant_env_key,
    )
    got_fp = canonical_fingerprint(accepted_flags, parsed_envs)
    prev_fp = canonical_fingerprint(str(prev_best_flags or ""), prev_envs)
    return got_fp != prev_fp


def _normalize_geak_overlay_dir(overlay: str) -> str:
    """Normalize a GEAK ``final_overlay`` path to the loadable overlay dir.

    GEAK sometimes hands back the parent ``.../final`` while the importable
    authored-kernel root is ``.../final/overlay``. When the given path is a
    directory containing an ``overlay`` subdirectory, return that subdirectory so
    ``run_grid`` prepends the real overlay onto PYTHONPATH; otherwise return the
    input unchanged (``run_grid`` still applies its own safety checks).
    """
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


# A GEAK candidate slot tag (``cand_c0_triton``, ``c1_triton``), as opposed to
# the name of the kernel the slot produced.
_GEAK_CAND_TAG_RE = re.compile(r"^(cand[_-])?c\d+([_-]|$)", re.IGNORECASE)


def geak_is_cand_tag(name: Any) -> bool:
    """Return True when ``name`` is a GEAK slot tag, not a kernel symbol.

    ``cand_c0_triton`` names the slot a candidate was dispatched into;
    ``dsa_sparse_attn_prefill_main_kernel`` names what the slot produced. Both
    spell the same acceptance, so every reader that has to pick one must pick
    the same one. The symbol is the stable id — it survives a re-run into a
    different slot — so the symbol wins and the tag becomes an alias.

    Args:
        name (Any): A candidate identity, in any of the written shapes.

    Returns:
        bool: True when the text matches the slot-tag form.
    """
    text = str(name or "").strip()
    return bool(text) and bool(_GEAK_CAND_TAG_RE.match(text))


def _geak_spec_name(spec: Any) -> str:
    """Return the display name of one GEAK acceptance entry.

    Accepts both shapes an acceptance is written in: the dict GEAK emits, and
    the bare string the revalidation path carries. One resolver keeps the
    ledger and the attribution row naming a kernel the same way.
    """
    if isinstance(spec, str):
        return spec.strip()
    if not isinstance(spec, dict):
        return ""
    return str(spec.get("short_name") or spec.get("kernel_id") or spec.get("cand_tag") or "").strip()


def geak_spec_name(spec: Any) -> str:
    """Public alias of :func:`_geak_spec_name` for out-of-module readers."""
    return _geak_spec_name(spec)


def geak_spec_kind(spec: Any) -> str | None:
    """Return the acceptance ``kind``, or ``None`` when the source omits it.

    ``None`` is a real state, not a default. The ``kernel_journey.json`` rows
    carry no ``kind`` field at all — measured over ``/shared_nfs/hyperloom-claw``,
    0 of 36 accepted journey rows have one — so a reader that treats a missing
    ``kind`` as "not env" and one that treats it as "env" would disagree on the
    same run. Callers get the unknown and must say what they do with it.
    """
    if not isinstance(spec, dict):
        return None
    raw = spec.get("kind")
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


def geak_spec_is_env(spec: Any) -> bool:
    """Return True only when the acceptance is *known* to be an env selection.

    An env acceptance picks an existing library or environment variable; no
    kernel was authored, so it belongs in the config half of GEAK's gain. An
    unknown ``kind`` is not env: it is admitted and tagged, never guessed.
    """
    return geak_spec_kind(spec) == "env"


def _geak_accepted_kernel_specs(result: Any) -> list[dict[str, Any]]:
    """Return the authored kernels a GEAK result accepted, both lanes, deduped.

    GEAK routes an acceptance to ``accepted_kernels`` or to ``accepted_heads``
    purely by which queue proposed it (``kernelQueue`` vs ``headQueue`` in
    ``run_e2e.py``); both entries have the same shape and both carry the same
    parity-checked same-config ``e2e_delta_pct``. GEAK's own evidence helper
    ``_wf_best_accepted_delta_pct`` reads the two lanes together, so a reader
    that takes only one of them silently drops most of the campaign — measured
    over ``/shared_nfs/hyperloom-claw``, 8 of the 11 sessions with an
    acceptance carry it in ``accepted_heads`` alone.

    Two filters apply:

    * ``e2e_delta_pct`` must be positive. This is the same admission test the
      journey backfill uses.
    * ``kind == "env"`` is excluded. Those acceptances select an existing
      library or environment variable (``ck_gemm_a8w8_blockscale_bpreshuffle``,
      ``moe_grouped_gemm_ck2stage``); no kernel was authored, so they belong in
      the config half of GEAK's gain, not in the per-kernel adoption ledger.

    Alias twins — one acceptance written under both the candidate tag and the
    kernel symbol — are collapsed on ``(op_kind, e2e_delta_pct)``, and the
    surviving row is the one named after the kernel. A candidate tag
    (``cand_c0_triton``) says only which slot proposed it; the symbol
    (``dsa_sparse_attn_prefill_main_kernel``) is what a report can name.
    """
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
    """Report whether a GEAK result carries an accepted kernel that gained.

    The delta behind this is GEAK's own parity-checked A/B of the kernel
    against the identical config, and it is independent of the run-level
    ``status``, which is a verdict on the promoted throughput basis alone. A
    ``no_gain`` run can therefore still hold a real kernel.
    """
    return bool(_geak_accepted_kernel_specs(result))


def _geak_overlay_is_loadable(overlay: str) -> bool:
    """Report whether an overlay dir can actually install an authored kernel.

    ``run_grid`` prepends the overlay onto ``PYTHONPATH``; the kernel is only
    installed when the interpreter then imports the overlay's
    ``sitecustomize.py``. A path that does not exist, or a directory holding no
    ``sitecustomize.py``, is inert: the server launches as plain baseline and
    ``run_grid`` logs a warning nobody reads. Callers use this to refuse to
    dispatch a revalidation whose only material is an overlay that cannot load.

    An importable overlay is not automatically a *kernel* overlay. GEAK also
    emits a config-only overlay -- ``{"modules": [], "rebinds": [], "note":
    "config-only result: no kernel overlay accepted ..."}`` -- which imports
    cleanly and installs nothing. Treating that as loadable would label a pure
    config win as a kernel win, which is the exact mis-crediting this gate
    exists to stop. So when a manifest is present it must name at least one
    module or rebind. An overlay with no manifest keeps the old behaviour:
    absence of evidence is not evidence of an empty overlay.
    """
    if not overlay:
        return False
    try:
        if not (Path(overlay) / "sitecustomize.py").is_file():
            return False
        manifest = Path(overlay) / "_overlay_manifest.json"
        if not manifest.is_file():
            return True
        spec = json.loads(manifest.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(spec, dict):
        return False
    return bool(spec.get("modules") or spec.get("rebinds") or spec.get("captures"))


def _geak_overlay_digest(overlay: str) -> str:
    """Digest the overlay's bind manifest, or ``""`` when it has none.

    ``_overlay_manifest.json`` is written by GEAK and records exactly which
    modules/rebinds/captures the overlay installs. Hashing it gives the
    revalidation a check on the overlay's *content*, which
    ``canonical_fingerprint`` deliberately excludes (it fingerprints
    ``(args, envs)`` only, so an overlay silently dropped between dispatch and
    launch still matches). Not every overlay carries a manifest, so an empty
    return means "no content evidence available", never "mismatch".

    The manifest names the *target* of each bind, not the kernel body, so it
    alone does not identify what would run: measured over
    ``/shared_nfs/hyperloom-claw``, three unrelated sessions share one manifest
    digest because all three patch
    ``sglang.kernels.ops.attention.decode_attention``. The bodies each entry
    points at are therefore folded in too, so the digest tracks the kernel and
    not just its address. A referenced body that cannot be read contributes its
    path alone -- the digest stays stable and comparable rather than collapsing
    to ``""``.
    """
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
            value = _mask_value(table.get(hip_var))
            if not value:
                continue
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
        visible) rather than ``{}``; when the winner is a ROCr-level mask and a
        HIP-level mask is also in force, the latter travels under ``"inner"``
        because it selects a subset *within* the ROCr-visible set.
        ``{}`` only when no mask is set anywhere — meaning "whole machine
        visible", not "pinned to 0".
    """
    env = os.environ if environ is None else environ
    recipe = dict(recipe_envs or {})
    blank: dict[str, Any] = {}
    for var in _VISIBLE_DEVICE_VARS:
        for source, table in (("process_env", env), ("baseline_recipe", recipe)):
            raw = table.get(var)
            if raw is None:
                continue
            value = _mask_value(raw)
            if not value:
                # Present but empty: a real "zero devices visible" state, not an
                # absent mask. Remember the first one and keep looking — a real
                # pin further down the chain still outranks it — but if nothing
                # else is set, report it as a zero-device pin rather than as
                # "unpinned", which reads as "whole machine".
                if not blank:
                    blank = {"var": var, "value": "", "ids": [], "count": 0, "source": source}
                continue
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
            return pin
    return blank


def _resolve_handoff_gpu_ids(*, gpu_pin: Mapping[str, Any] | None, tp: int) -> str:
    """Resolve the handoff's ``gpu_ids`` in the coordinate system GEAK applies it in.

    ``gpu_ids`` is a HIP-level device list: the consumer exports it as
    ``HIP_VISIBLE_DEVICES``/``CUDA_VISIBLE_DEVICES`` for the servers it
    launches, and HIP indexes into the ROCr-visible set. So:

      * pinned with a ROCr-level mask the child INHERITS — that mask renumbers
        the child's devices, so the ids must be LOGICAL positions inside it
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
    systems the result is in. A consumer that exports the result as
    ``HIP_VISIBLE_DEVICES`` without touching ROCr is correct in both; a
    consumer that re-applies ``gpu_pin["value"]`` as ``ROCR_VISIBLE_DEVICES``
    has just renumbered the devices itself and must use ``0..count-1``, NOT
    these ids, for the inner HIP mask.

    Args:
        gpu_pin: The :func:`_resolve_gpu_pin` result (``{}``/``None`` = unpinned).
        tp: Tensor-parallel size; ``<= 1`` is treated as 1.

    Returns:
        A comma-separated device list, never empty.
    """
    width = max(int(tp or 1), 1)
    pin = gpu_pin or {}
    ids = list(pin.get("ids") or [])
    # Logical remapping applies only to a mask the GEAK child actually
    # INHERITS. The phase launches it with ``dict(os.environ)``, so a
    # process-env ROCR mask is inherited and its ids are logical; a mask that
    # only exists in the recipe is not, ROCr shows every card, and the absolute
    # ids are the correct HIP indices.
    if _pin_is_inherited_rocr(pin):
        # Token count, not len(ids): a UUID mask parses to zero numeric ids but
        # still exposes that many cards to the child.
        visible = int(pin.get("count") or len(ids) or 0)
        if visible > 0:
            # A HIP-level mask nested inside the ROCr pin is ALREADY expressed
            # in the child's logical coordinates, so it is forwarded as-is
            # rather than overwritten with ``0..n-1``. Out-of-range entries are
            # dropped: they name devices the ROCr mask never exposed.
            inner = _mask_tokens((pin.get("inner") or {}).get("value"))
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


def _pin_is_inherited_rocr(pin: Mapping[str, Any] | None) -> bool:
    """Will the GEAK child inherit this pin as a ROCr-level device slice?

    Only then are the handoff's ``gpu_ids`` logical. The phase launches GEAK
    with ``dict(os.environ)``, so a process-env ROCr mask is inherited and
    renumbers the child's devices; a mask that only exists in the recipe is
    not, ROCr shows every card, and absolute ids are the correct HIP indices.

    Args:
        pin: The :func:`_resolve_gpu_pin` result.

    Returns:
        ``True`` for a process-env ROCr-level pin.
    """
    pin = pin or {}
    return is_rocr_level(str(pin.get("var") or "")) and str(pin.get("source") or "") == "process_env"


def _resolve_handoff_gpu_ids_space(*, gpu_pin: Mapping[str, Any] | None) -> str:
    """Which coordinate system the handoff's ``gpu_ids`` are expressed in.

    ``gpu_ids`` alone is ambiguous: ``"0,1"`` is either "the first two cards of
    the inherited ROCr mask" or "absolute cards 0 and 1", and a consumer that
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
        ids index into an inherited ROCr mask, ``"absolute"`` otherwise
        (including unpinned).
    """
    pin = gpu_pin or {}
    if pin and int(pin.get("count") or 0) <= 0:
        return "none"
    return "logical" if _pin_is_inherited_rocr(pin) else "absolute"


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
            return tok[len(prefix) :]
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
#: parallelism, ports, seeds); stripped from the forwarded
#: ``server_launch_flags`` since the consuming harness sets them per launch.
#: Everything not listed (engine knobs) is kept.
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
#: command line; an unknown backend disables the scrape.
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

    pat = _re.compile(r"(?:-m\s+\S*" + _re.escape(marker) + r"\S*|" + _re.escape(marker) + r")\b(.*)$")
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


def _scrape_resolved_launch_flags(session_dir: Any, backend: str, target_tput: float = 0.0) -> str:
    """Recover the orchestrator's FULL resolved server-launch flags from logs.

    The complete record of what the engine ran with is the launched argv,
    echoed into each benchmark's ``server.log`` / ``benchmark_stderr.log``.
    Selection is by throughput, not recency: find the benchmark whose measured
    ``output_throughput`` equals ``target_tput`` and scrape its sibling server
    log. Falls back to the most recent clean launch when no throughput match
    exists (or ``target_tput<=0``).

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
        # Throughput-matched selection: find the benchmark whose
        # inferencex_result.json output_throughput == target_tput, scrape its
        # sibling server log.
        if target_tput and target_tput > 0:
            best_path, best_err = "", 1e9
            for rp in _glob.glob(str(runs_root / "**" / "inferencex_result.json"), recursive=True):
                if "geak" in rp or "_baseline_source_overlay" in rp:
                    continue
                try:
                    tp = float((json.loads(Path(rp).read_text(encoding="utf-8")) or {}).get("output_throughput") or 0.0)
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
        # Fallback: most recent clean (non-profiling) launch across the run.
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
