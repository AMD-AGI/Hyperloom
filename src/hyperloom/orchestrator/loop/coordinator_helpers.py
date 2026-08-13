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
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..actions.executors._grid_server_args import (
    tokenize_server_args_preserving_json,
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
    "_GEAK_MEASUREMENT_DIVERGENCE_WARN_PCT",
    "_MIN_KERNEL_ENGAGED_GAIN_PCT",
    "coerce_needs_gpu",
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


# The one verdict field a Critic states its grounds in. ``notes`` is
# remediation text and ``risks[*].summary`` describes the risk, so a rule named
# in either is being discussed rather than invoked; both stay out.
_VERDICT_PROSE_KEY: str = "reasoning"

# What a citation looks like: the code opens the verdict's grounds and a colon
# introduces the finding, the shape the field verdict used --
# ``"specialist_quantitative_claim_violation: the proposal payload carries the
# forbidden predicted_gain_pct field."`` Nothing may precede the code but
# whitespace or a backtick, and only the opening line of the prose is read.
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


def _opening_line(text: str) -> str:
    """Return the first non-blank line of ``text``, or ``""`` when it has none."""
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def cited_advisory_reason_code(entry: dict[str, Any]) -> str:
    """Return the advisory-only rule ``entry`` cites, from the field or its prose.

    ``failure_reason_code`` is the reliable path: the Critic's output schema
    asks for the code of the rule its verdict rests on. Prose is read only as a
    fallback, for a verdict that names its rule in ``reasoning`` instead — the
    shape observed in the field.

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
    opening = _opening_line(str(entry.get(_VERDICT_PROSE_KEY) or ""))
    if not opening:
        return ""
    # At most one code can open one line, so the sort only fixes the order the
    # candidates are tried in.
    for code in sorted(advisory):
        if re.match(rf"{_CITATION_OPENER}{re.escape(code)}`?[ \t]*:", opening):
            return code
    return ""


# Priority a batch of per-variant verdicts collapses by: one approved variant
# carries the proposal, otherwise one reject sinks it, and advice outranks a
# request for more review.
_VERDICT_COLLAPSE_ORDER: tuple[str, ...] = ("approve", _REJECT_VERDICT, ADVISE_VERDICT, "needs_review")


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


def verdict_rests_on_one_ground(entry: dict[str, Any]) -> bool:
    """Return whether ``entry`` refuses for a single reason.

    A verdict can cite an advisory rule *and* refuse on its own merits in the
    same breath — "the proposal claims a 12% gain and has no rollback plan".
    Holding that verdict to the advisory rule would let the second half of the
    sentence disappear, so the hold is confined to a reject that names one
    ground and asks for nothing further: at most one risk entry, and no
    outstanding evidence request.

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
    if [item for item in (entry.get("required_evidence") or []) if item]:
        return False
    return len([risk for risk in (entry.get("risks") or []) if risk]) <= 1


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
    verdict = str(entry.get("verdict") or "").strip()
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
    accepted_cfg = result.get("accepted_config") or {}
    accepted_flags = str(accepted_cfg.get("flags") or "").strip()
    parsed_envs, extra_flags = _split_env_and_flags(str(accepted_cfg.get("env") or ""))
    if extra_flags:
        accepted_flags = (accepted_flags + " " + extra_flags).strip()
    # A missing / all-empty accepted_config carries no config optimization; a
    # bare fingerprint mismatch against a non-empty current_best is NOT material
    # (promoting it would wipe the existing config to empty).
    if not accepted_flags and not parsed_envs:
        return False
    got_fp = canonical_fingerprint(accepted_flags, parsed_envs)
    prev_fp = canonical_fingerprint(
        str(prev_best_flags or ""),
        dict(prev_best_envs or {}),
    )
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
