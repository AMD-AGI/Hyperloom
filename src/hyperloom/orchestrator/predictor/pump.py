# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Ask the predictor once per decision point and file its answer as proposals.

The pump creates no tasks. It writes into ``SharedState.specialist_rounds`` —
the same ledger a finished specialist writes its ``proposal_set`` into — so the
answer surfaces in ``=== Untested proposals (current cycle) ===`` and
orchestration composes the ``explore`` grid itself. Everything downstream is
Hyperloom's existing machinery: PolicyGate, the explore executor's serial
KEEP/REVERT, and for a source change the ordinary specialist → Critic →
``integrate_patch`` chain.

Decision points
---------------
``decision_point_key`` is ``c{macro_cycle}-s{stack_depth}-r{roofline_count}``,
and ``predictor_asked_keys`` records the ones already answered. The predictor's
answer is a function of its request, so re-asking at an unchanged decision point
buys the same proposals for the price of another request; that key is what makes
this safe to call on every tick.

Each component moves for a reason worth a fresh answer. A KEEP deepens the
stack, and the stack is what the answer is conditioned on -- the same AITER
backend switch measured -1.17% on a bare baseline and +2.68% stacked on fp8 KV
cache in this fleet. A ``cycle_reloop`` re-enters against a different stack. A
landed roofline is new evidence, and it is also the only thing that gives the
predictor a second look inside a cycle whose first answer landed no KEEP.

All three are *pulled* here, on the pump's own tick. Nothing outside this
package has to know the predictor exists.

Choosing what to surface
------------------------
The service samples N times and returns every distinct proposal, but in
*sampling* order: its ``chosen`` is merely the first sample that parsed. So the
head of the list carries no quality signal and the consumer has to rank.

``meta["candidates"]`` carries every raw sample, which makes the model's own
self-consistency measurable: ranking by how many samples voted for a proposal
is a real signal that costs nothing. Family de-duplication then stops one knob
sweep from taking every slot -- at N=8 three of eight proposals differed only in
``--block-size``, and spending three of four slots on that measures almost
nothing.

The first :data:`MAX_PROPOSALS` survivors are marked as the batch, which is the
grid size orchestration is told to target: one decision point's answer is one
grid. The next few are queued too, up to :data:`MAX_QUEUED`, as ordinary rows --
they cost nothing on the queue and they are what the batch refills from when
the exclusion filter thins it.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from hyperloom.common.prompt_safety import flatten_for_prompt
from hyperloom.orchestrator.actions.executors._canonical_fingerprint import (
    canonical_fingerprint,
)
from hyperloom.orchestrator.actions.executors._proposal_identity import (
    effective_fingerprint,
)
from hyperloom.orchestrator.phases.machine_state import (
    PHASE_FRAMEWORK_AGENT,
    phase_budget_remaining_seconds,
    phase_cap_seconds,
    phase_cumulative_seconds,
)
from hyperloom.orchestrator.predictor.attempted import (
    MAX_PROPOSALS,
    MAX_QUEUED,
    PROVENANCE,
    QUEUE_DOMAIN,
    QUEUE_PRIORITY,
    queued_unbenched,
)
from hyperloom.orchestrator.predictor import config as predictor_config
from hyperloom.orchestrator.predictor.client import Action, Prediction, predict
from hyperloom.orchestrator.predictor.payload import build_request

log = logging.getLogger(__name__)

#: Re-exported so importers that predate the split keep working: ``render.py``
#: reads ``QUEUE_DOMAIN`` off this module to mark first-pass rows.
__all__ = [
    "MAX_PROPOSALS",
    "MAX_QUEUED",
    "PROVENANCE",
    "QUEUE_DOMAIN",
    "QUEUE_PRIORITY",
    "already_asked",
    "decision_point_key",
    "find_mandate",
    "note_asked",
    "pump",
]

#: Cap on a mandate handed to a specialist. The prompt builder allows more; this
#: is model-authored text entering another model's prompt, so it stays short.
MAX_MANDATE_CHARS = 4000

#: Fallback cost of one explore variant, used until the session has measured its
#: own. A 120B MoE spends about as long restarting the server as benchmarking it,
#: so this is deliberately a bench-plus-restart figure rather than a bench one.
DEFAULT_MIN_VARIANT_SEC = 600.0


def decision_point_key(state: Any) -> str:
    """The identity of the decision the predictor is being asked about.

    Args:
        state (Any): The ``SharedState``.

    Returns:
        str: ``c{macro_cycle}-s{stack_depth}-r{roofline_snapshot_count}``.
    """
    cycle = int(getattr(state, "macro_cycle", 0) or 0)
    depth = len(getattr(state, "optimization_stack", None) or [])
    snapshots = getattr(state, "roofline_snapshots", None)
    generation = len(snapshots) if isinstance(snapshots, list) else 0
    return f"c{cycle}-s{depth}-r{generation}"


def already_asked(state: Any, key: str) -> bool:
    """Whether this decision point already has an answer on the queue."""
    keys = getattr(state, "predictor_asked_keys", None)
    return isinstance(keys, list) and key in keys


def note_asked(state: Any, key: str) -> None:
    """Record that this decision point has been answered; tail-trimmed."""
    from hyperloom.orchestrator.state.shared_state import _PREDICTOR_ASKED_KEYS_CAP

    keys = getattr(state, "predictor_asked_keys", None)
    if not isinstance(keys, list):
        keys = []
        state.predictor_asked_keys = keys
    if key in keys:
        return
    keys.append(key)
    if len(keys) > _PREDICTOR_ASKED_KEYS_CAP:
        del keys[:-_PREDICTOR_ASKED_KEYS_CAP]


def _declined(reason: str) -> None:
    log.debug("predictor_pump: standing down (%s)", reason)


def _min_variant_sec(state: Any) -> float:
    """What one explore variant costs, measured off this session when it can.

    ``explore_elapsed_accum_s`` over the number of benched variants is the
    session's own answer, which beats any constant: the same grid costs ~9
    minutes a variant on one model and ~13 on another, and the figure decides
    whether a request is worth sending.

    Args:
        state (Any): The ``SharedState``.

    Returns:
        float: Seconds, falling back to :data:`DEFAULT_MIN_VARIANT_SEC`.
    """
    search = getattr(state, "explore_search", None)
    tested = search.get("tested") if isinstance(search, dict) else None
    benched = len(tested) if isinstance(tested, dict) else 0
    try:
        accum = float(getattr(state, "explore_elapsed_accum_s", 0.0) or 0.0)
    except (TypeError, ValueError):
        accum = 0.0
    if benched > 0 and accum > 0:
        return accum / benched
    return DEFAULT_MIN_VARIANT_SEC


def _optimize_headroom_sec(state: Any) -> float | None:
    """Seconds OPTIMIZE can still spend, or ``None`` when nothing bounds it.

    Both terms matter and the phase itself consults both. The per-entry budget
    is what ends a cycle's OPTIMIZE (``optimize_phase_budget_exhausted``); the
    cumulative cap across every entry is what makes a re-entry after
    ``cycle_reloop`` fall straight back out (``optimize_budget_cap``) -- 59 and
    55 seconds in the run that motivated this gate, each of which still spent a
    predictor request on a decision point that had no benchmark slot to fill.

    ``None`` from either term means unbounded, so it contributes no limit
    rather than a zero: a falsy-or-infinity idiom would read a genuine ``0.0``
    remaining as "no limit" and is exactly backwards.

    Args:
        state (Any): The ``SharedState``, with ``phase`` already checked.

    Returns:
        float | None: The tighter of the two limits, or ``None`` when neither
            applies. May be negative once the cap is overshot.
    """
    limits: list[float] = []
    remaining = phase_budget_remaining_seconds(state)
    if remaining is not None:
        limits.append(float(remaining))
    cap = phase_cap_seconds(state)
    if cap is not None:
        limits.append(float(cap) - float(phase_cumulative_seconds(state)))
    return min(limits) if limits else None


def _gate(phase: Any, conf: predictor_config.PredictorConfig) -> bool:
    """Whether to spend a request at this decision point."""
    state = phase.shared_state

    if not conf.enabled:
        _declined("disabled")
        return False

    # The tick calls the FRAMEWORK pump unconditionally and lets it return
    # early, so the phase check belongs here rather than at the call site.
    if str(getattr(state, "phase", "") or "").strip().upper() != PHASE_FRAMEWORK_AGENT:
        _declined("not in FRAMEWORK_AGENT")
        return False

    framework = getattr(state, "framework", "")
    if not conf.supports(framework):
        # Flag catalogues exist for sglang and vllm only, so the consumer cannot
        # validate an answer for anything else. Declining beats sending a
        # request whose reply could not be trusted.
        _declined(f"framework {framework!r} has no flag catalogue")
        return False

    # An answer is only worth its request if OPTIMIZE can still bench one
    # variant off it. Checked before ``already_asked`` so a decision point
    # declined for want of budget is not recorded as answered: the same key
    # cannot recur (the cycle number is in it), but a future key must not
    # inherit a "done" mark from this one.
    headroom = _optimize_headroom_sec(state)
    if headroom is not None:
        needed = _min_variant_sec(state)
        if headroom < needed:
            _declined(f"OPTIMIZE headroom {headroom:.0f}s below one variant ({needed:.0f}s)")
            return False

    key = decision_point_key(state)
    if already_asked(state, key):
        _declined(f"decision point {key} already answered")
        return False

    return True


#: Env prefixes that belong to the other serving stack. Repair already drops
#: illegal *flags*; it does not drop envs, so a vLLM round used to inherit
#: ``SGLANG_*`` from an SGLang-heavy sampler.
_FOREIGN_ENV_PREFIXES: dict[str, tuple[str, ...]] = {
    "vllm": ("SGLANG_",),
    "sglang": ("VLLM_",),
}


def _envs_for_framework(envs: dict[str, Any] | None, framework: str) -> dict[str, str]:
    """Keep same-framework and shared envs; drop the other stack's prefixes."""
    raw = dict(envs or {})
    prefixes = _FOREIGN_ENV_PREFIXES.get(str(framework or "").strip().lower(), ())
    if not prefixes:
        return {str(k): str(v) for k, v in raw.items()}
    kept: dict[str, str] = {}
    dropped: list[str] = []
    for key, value in raw.items():
        name = str(key)
        if any(name.upper().startswith(prefix) for prefix in prefixes):
            dropped.append(name)
            continue
        kept[name] = str(value)
    if dropped:
        log.info(
            "predictor_pump: dropped cross-framework envs %s (framework=%s)",
            dropped,
            framework,
        )
    return kept


def _stack_base(state: Any | None) -> tuple[str, dict[str, str]]:
    """Current champion extras the next explore variant will launch on top of."""
    if state is None:
        return "", {}
    best = getattr(state, "current_best", None) or {}
    if not isinstance(best, dict):
        return "", {}
    args = str(best.get("effective_extra_server_args") or best.get("extra_server_args") or "").strip()
    raw = best.get("extra_envs") or {}
    envs = {str(k): str(v) for k, v in dict(raw).items()} if isinstance(raw, dict) else {}
    return args, envs


def _merge_launch(
    base_args: str, extra_args: str, base_envs: dict[str, str], extra_envs: dict[str, str]
) -> tuple[str, dict[str, str]]:
    """Stack ∪ proposal, last-wins on flags via ``canonical_fingerprint`` pairing."""
    merged_args = f"{base_args} {extra_args}".strip()
    merged_envs = dict(base_envs)
    merged_envs.update(extra_envs)
    return merged_args, merged_envs


def _tested_maps(state: Any | None) -> tuple[set[str], set[str]]:
    """Delta fingerprints (tested keys) and launched-recipe fingerprints."""
    if state is None:
        return set(), set()
    search = getattr(state, "explore_search", None) or {}
    tested = search.get("tested") if isinstance(search, dict) else None
    if not isinstance(tested, dict):
        return set(), set()
    delta_keys = {str(key) for key in tested}
    launch_fps: set[str] = set()
    for row in tested.values():
        if not isinstance(row, dict):
            continue
        evidence = row.get("launch_evidence")
        flags = ""
        if isinstance(evidence, dict):
            flags = str(evidence.get("requested_server_flags") or "")
        if not flags.strip():
            flags = str(row.get("extra_server_args") or row.get("extra_args") or "")
        envs = row.get("extra_envs") or {}
        if not isinstance(envs, dict):
            envs = {}
        launch_fps.add(canonical_fingerprint(flags, envs))
    return delta_keys, launch_fps


def _skip_reason(
    extra_args: str,
    extra_envs: dict[str, str],
    *,
    seen_delta: set[str],
    base_args: str,
    base_envs: dict[str, str],
    tested_delta: set[str],
    tested_launch: set[str],
    queued_delta: set[str],
) -> str | None:
    """Why this proposal should not be queued, or ``None`` to keep it.

    ``already_queued_unbenched`` is the belt to the service-side exclusion
    filter's braces. The service is the side that can act on this usefully --
    it can spend its samples elsewhere -- but the request takes tens of seconds
    to answer, a variant can finish benching inside that window, and the service
    may be an older build that ignores the exclusion set entirely. None of those
    should put a second copy of a queued row on the queue.
    """
    delta_fp = effective_fingerprint(extra_args, extra_envs)
    merged_args, merged_envs = _merge_launch(base_args, extra_args, base_envs, extra_envs)
    launch_fp = canonical_fingerprint(merged_args, merged_envs)
    base_fp = canonical_fingerprint(base_args, base_envs)
    if delta_fp in seen_delta:
        return "batch_dup"
    if launch_fp == base_fp:
        return "already_on_stack"
    if delta_fp in tested_delta:
        return "already_tested_delta"
    if launch_fp in tested_launch:
        return "already_tested_launch"
    if delta_fp in queued_delta:
        return "already_queued_unbenched"
    return None


def _sample_key(server_args: Any, envs: Any, source_change: Any) -> tuple:
    """The identity the service de-duplicated its samples on.

    Mirrors ``_distinct_actions`` in the service so a vote counted here lands on
    the same proposal the service collapsed its repeats into. Reimplementing the
    key is the price of the two sides not sharing code; a mismatch would show up
    as every proposal having exactly one vote.

    Args:
        server_args (Any): The sample's launch flags.
        envs (Any): The sample's environment variables.
        source_change (Any): The sample's prose source change.

    Returns:
        tuple: A hashable identity for one proposal.
    """
    args = dict(server_args or {}) if isinstance(server_args, dict) else {}
    env = dict(envs or {}) if isinstance(envs, dict) else {}
    return (
        tuple(sorted((str(k), str(v)) for k, v in args.items())),
        tuple(sorted((str(k), str(v)) for k, v in env.items())),
        str(source_change or ""),
    )


def _vote_counts(answer: Prediction) -> dict[tuple, int]:
    """How many raw samples voted for each distinct proposal.

    Counted from ``meta["candidates"]``, which the service sends precisely so a
    consumer can see the spread its single ``action`` hides. An empty result
    (a service that sends no candidates) leaves every proposal unranked, which
    degrades to the sampling order rather than to a wrong order.

    Args:
        answer (Prediction): The predictor's answer.

    Returns:
        dict[tuple, int]: Sample count per :func:`_sample_key`.
    """
    candidates = answer.meta.get("candidates")
    if not isinstance(candidates, list):
        return {}
    counts: dict[tuple, int] = {}
    for row in candidates:
        if not isinstance(row, dict):
            continue
        key = _sample_key(row.get("server_args"), row.get("envs"), row.get("source_change"))
        counts[key] = counts.get(key, 0) + 1
    return counts


#: Flags whose value is a structured config rather than a scalar. Discarding
#: values is the point of the family key everywhere else, but for these the
#: value is what says which knobs move: two ``--compilation-config`` proposals
#: share the flag name while opening entirely different optimizations, and
#: collapsing them dropped ``enable_sp`` + ``fuse_allreduce_rms`` from a real
#: round -- levers orchestration then listed as never benched.
_STRUCTURED_VALUE_FLAGS = frozenset({"--compilation-config"})


def _structured_keys(flag: str, value: str) -> set[str]:
    """Sub-keys a structured flag's value moves, named ``flag:path``.

    One level of nesting is enough for the shapes that exist: vLLM's
    ``pass_config`` is the only nested member anyone proposes, and a deeper walk
    would split families on a leaf the proposer never chose.
    """
    try:
        parsed = json.loads(value.strip().strip("'\""))
    except (TypeError, ValueError):
        return set()
    if not isinstance(parsed, dict):
        return set()
    out: set[str] = set()
    for key, member in parsed.items():
        out.add(f"{flag}:{key}")
        if isinstance(member, dict):
            out.update(f"{flag}:{key}.{sub}" for sub in member)
    return out


def _next_value(tokens: list[str], index: int) -> tuple[str, int]:
    """The value at ``index``, re-joining a JSON object split on whitespace.

    ``str.split`` is the tokenizer everywhere else here and it is fine for
    scalars, but ``{"a": 1}`` arrives as two tokens. Re-joining on brace balance
    recovers it without a shell-quoting model this does not otherwise need.
    """
    if index >= len(tokens) or tokens[index].startswith("-"):
        return "", index
    chunk = tokens[index]
    index += 1
    if chunk.lstrip("'\"").startswith("{"):
        while chunk.count("{") > chunk.count("}") and index < len(tokens):
            chunk = f"{chunk} {tokens[index]}"
            index += 1
    return chunk, index


def _family_key(extra_args: str, extra_envs: dict[str, str]) -> frozenset[str]:
    """Which knobs a proposal moves, ignoring the values it moves them to.

    Two proposals in the same family answer the same question with different
    numbers. With four slots to spend, measuring three points of one sweep says
    much less than measuring three different levers, and a surviving family
    member that KEEPs brings its neighbours back at the next decision point
    anyway.

    Two shapes need more than the flag name to land in the right family.
    A ``-cc.pass_config.fuse_rope_kvcache=True`` short form carries its value
    inside the token, so keeping the token whole would give one knob a family
    per value -- the opposite of the intent. A structured value
    (:data:`_STRUCTURED_VALUE_FLAGS`) hides the knobs it moves inside JSON, so
    the flag name alone merges proposals that share nothing but the flag.

    Args:
        extra_args (str): The proposal's launch flags.
        extra_envs (dict[str, str]): The proposal's environment variables.

    Returns:
        frozenset[str]: Flag names, ``flag:path`` sub-keys for structured
            values, plus ``env:``-prefixed variable names.
    """
    tokens = extra_args.split()
    names: set[str] = set()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("-"):
            continue
        name = token.split("=", 1)[0]
        names.add(name)
        if name in _STRUCTURED_VALUE_FLAGS:
            value, index = _next_value(tokens, index)
            names.update(_structured_keys(name, value))
    names.update(f"env:{name}" for name in extra_envs)
    return frozenset(names)


def _flags_text(action: Action) -> str:
    """Render an action's launch flags as a CLI fragment."""
    return " ".join(flag if value is True else f"{flag} {value}" for flag, value in action.server_args.items()).strip()


def _proposal_rows(
    answer: Prediction,
    *,
    key: str,
    framework: str = "",
    state: Any | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank, de-duplicate and truncate the answer's configuration proposals.

    Historical ``explore_search.tested`` is an eligibility gate here: a second
    request that re-proposes a delta already measured -- or a new delta whose
    launched recipe matches one already measured -- would otherwise put a known
    answer back on the queue. ``already_tested_launch`` is the case that caught
    round-2 ``--quantization fp8`` on an ``fp8_e4m3`` champion after round 1 had
    already launched that combination.

    Cross-framework envs (``SGLANG_*`` on vLLM, ``VLLM_*`` on SGLang) are
    dropped here. The service's own repair strips illegal flags; it does not
    strip envs, and a foreign env would ride into the launch.

    Args:
        answer (Prediction): The predictor's answer.
        key (str): Decision-point key, used in the variant names.
        framework (str): Session framework; used to drop the other stack's envs.
        state (Any | None): SharedState, for stack extras and ``explore_search``.

    Returns:
        tuple: ``(rows, dropped)`` -- the proposals to queue, and the ones
            ranking or family de-duplication set aside. ``dropped`` rows carry a
            ``dropped_reason`` and are recorded for offline analysis only.
    """
    base_args, base_envs = _stack_base(state)
    tested_delta, tested_launch = _tested_maps(state)
    queued_delta = set(queued_unbenched(state))
    votes = _vote_counts(answer)
    samples = answer.meta.get("samples")
    samples = int(samples) if isinstance(samples, int) else None

    seen_delta: set[str] = set()
    eligible: list[dict[str, Any]] = []
    for index, action in enumerate(answer.config_actions):
        extra_args = _flags_text(action)
        extra_envs = _envs_for_framework(action.envs, framework)
        reason = _skip_reason(
            extra_args,
            extra_envs,
            seen_delta=seen_delta,
            base_args=base_args,
            base_envs=base_envs,
            tested_delta=tested_delta,
            tested_launch=tested_launch,
            queued_delta=queued_delta,
        )
        delta_fp = effective_fingerprint(extra_args, extra_envs)
        if reason is not None:
            log.info(
                "predictor_pump: skip reason=%s fp=%s args=%r envs=%r",
                reason,
                delta_fp,
                extra_args,
                extra_envs,
            )
            continue
        seen_delta.add(delta_fp)
        # Votes are counted on the answer as the service sent it, before the
        # cross-framework env strip above: a stripped proposal no longer matches
        # any sample and would silently score zero.
        row: dict[str, Any] = {
            "name": f"primatune-{key}-{index}",
            "extra_args": extra_args,
            "extra_envs": extra_envs,
            "provenance": PROVENANCE,
            "reason": "first-pass tuning prediction",
            "votes": votes.get(_sample_key(action.server_args, action.envs, action.source_change), 0),
        }
        if samples is not None:
            row["samples"] = samples
        eligible.append(row)

    # Highest consensus first; Python's stable sort leaves the service's
    # sampling order as the tie-break, which is the only other ordering on offer.
    eligible.sort(key=lambda r: -int(r.get("votes") or 0))

    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen_families: set[frozenset[str]] = set()
    for row in eligible:
        family = _family_key(str(row["extra_args"]), dict(row["extra_envs"]))
        if family and family in seen_families:
            dropped.append({**row, "dropped_reason": "same_flag_family"})
            continue
        if len(rows) >= MAX_QUEUED:
            dropped.append({**row, "dropped_reason": "over_surface_cap"})
            continue
        seen_families.add(family)
        # The batch is what orchestration is asked to dispatch as one grid;
        # anything past it is an ordinary queue row it may reach for when the
        # batch comes up short.
        row["batch"] = len(rows) < MAX_PROPOSALS
        rows.append(row)
    if dropped:
        log.info(
            "predictor_pump: set aside %d of %d eligible proposals %r",
            len(dropped),
            len(eligible),
            [(r["dropped_reason"], r["extra_args"], r["extra_envs"]) for r in dropped],
        )
    return rows, dropped


def _patch_mandate(answer: Prediction, *, key: str) -> dict[str, str] | None:
    """The source-change mandate to offer, or ``None`` when the answer has none.

    The mandate id is derived from the decision point, so re-recording the same
    round cannot mint a second id for the same work.

    Args:
        answer (Prediction): The predictor's answer.
        key (str): Decision-point key.

    Returns:
        dict[str, str] | None: ``{mandate_id, mandate}``, or ``None``.
    """
    if not answer.has_source_change:
        return None
    # The queue line and, later, the specialist prompt interpolate this without
    # sanitising it. Flattening is what stops a newline from forging a section
    # header; it also defangs code fences and angle brackets on the way through.
    mandate = flatten_for_prompt(answer.source_change)[:MAX_MANDATE_CHARS]
    if not mandate.strip():
        return None
    return {"mandate_id": f"primatune-patch-{key}", "mandate": mandate}


def find_mandate(state: Any, mandate_id: str) -> str:
    """Resolve a queued mandate id back to its verbatim prose.

    Read by the specialist dispatch path so orchestration only has to carry an
    opaque id: it cannot reword the mandate, and an id it never passes simply
    yields an ordinary LLM-authored specialist with no predictor attribution.

    Args:
        state (Any): The ``SharedState``.
        mandate_id (str): The id offered on the queue row.

    Returns:
        str: The mandate, or ``""`` when the id is unknown.
    """
    wanted = str(mandate_id or "").strip()
    if not wanted:
        return ""
    for entry in reversed(getattr(state, "specialist_rounds", None) or []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("mandate_id") or "") != wanted:
            continue
        return str(entry.get("mandate") or "")
    return ""


def _record_round(
    phase: Any,
    *,
    key: str,
    rows: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    patch: dict[str, str] | None,
    predict_meta: dict[str, Any],
) -> None:
    """File the answer as one round on the untested-proposal queue.

    Args:
        phase (Any): The collaborator exposing ``shared_state``.
        key (str): Decision-point key, used as the idempotent ``round_id``.
        rows (list[dict[str, Any]]): Configuration proposals to surface.
        dropped (list[dict[str, Any]]): Proposals set aside, recorded only.
        patch (dict[str, str] | None): The source-change mandate, when present.
        predict_meta (dict[str, Any]): Request cost and shape, recorded only.
    """
    state = phase.shared_state
    entry: dict[str, Any] = {
        "round_id": key,
        "cycle": int(getattr(state, "macro_cycle", 0) or 0),
        "domain": QUEUE_DOMAIN,
        "priority": QUEUE_PRIORITY,
        "task_id": key,
        "proposal_set": rows,
        # Recorded, never rendered: the renderer reads cycle / domain /
        # gap_canonical_id / task_id / proposal_set and ignores the rest. These
        # two are here so a finished session can be analysed off state.json
        # alone -- what the model proposed before ranking, and what the request
        # cost.
        "dropped": dropped,
        "predict_meta": predict_meta,
    }
    if patch is not None:
        entry.update(patch)
    state.record_specialist_round(entry)


def _log_shadow(answer: Prediction, *, key: str, session_id: str) -> None:
    """Record what would have been queued, at zero benchmark cost.

    Every proposal is logged, not just the ones that would have been surfaced:
    shadow mode exists to measure what the predictor nominates, and with
    sampling on, the spread is the measurement. A run that only recorded the
    head would have shown the flag that mattered on none of its lines while the
    model was proposing it in one sample out of four.
    """
    log.info(
        "predictor_shadow: session=%s key=%s parsed=%s actions=%d samples=%s prompt_chars=%s dropped=%r",
        session_id,
        key,
        answer.parsed,
        len(answer.actions),
        answer.meta.get("samples"),
        answer.meta.get("prompt_chars"),
        answer.meta.get("dropped_flags"),
    )
    for index, action in enumerate(answer.actions):
        log.info(
            "predictor_shadow:   [%d/%d] server_args=%r envs=%r source_change=%r",
            index,
            len(answer.actions),
            action.server_args,
            action.envs,
            action.source_change[:200],
        )


async def pump(phase: Any, *, caller: str) -> None:
    """Consult the predictor once for this decision point and queue the answer.

    Safe to call on every tick and from the phase-entry hook: the gate and
    ``predictor_asked_keys`` make repeat calls at an unchanged decision point
    free. Never raises into the tick loop.

    Args:
        phase (Any): The ``FrameworkPhase`` collaborator, or anything else
            exposing ``shared_state``.
        caller (str): Label for the log ("entry" / "tick" / "run").
    """
    try:
        conf = predictor_config.load()
        if not _gate(phase, conf):
            return

        state = phase.shared_state
        key = decision_point_key(state)
        session_id = str(getattr(state, "session_id", "") or "")

        request = build_request(state, session_id=session_id, phase_label=conf.phase_label)
        started = time.monotonic()
        answer = predict(request, endpoint=conf.endpoint, timeout_sec=conf.timeout_sec)
        latency_ms = int((time.monotonic() - started) * 1000)

        if not conf.enqueues:
            _log_shadow(answer, key=key, session_id=session_id)
            return

        # Marked answered whatever came back. A predictor that declines, or one
        # whose every proposal was already measured, has answered this decision
        # point; re-POSTing the same request on the next tick would only spend
        # the request again.
        note_asked(state, key)

        if not answer.parsed or answer.is_empty:
            log.info(
                "predictor_pump: no action from %s (key=%s latency=%dms error=%r)",
                caller,
                key,
                latency_ms,
                answer.error,
            )
            return

        rows, dropped = _proposal_rows(
            answer,
            key=key,
            framework=str(getattr(state, "framework", "") or ""),
            state=state,
        )
        patch = _patch_mandate(answer, key=key)
        if not rows and patch is None:
            log.info(
                "predictor_pump: nothing left to queue from %s (key=%s, %d proposals all skipped)",
                caller,
                key,
                len(answer.config_actions),
            )
            return

        _record_round(
            phase,
            key=key,
            rows=rows,
            dropped=dropped,
            patch=patch,
            predict_meta={
                "latency_ms": latency_ms,
                "prompt_chars": answer.meta.get("prompt_chars"),
                "samples": answer.meta.get("samples"),
                "actions_returned": len(answer.actions),
            },
        )
        log.info(
            "predictor_pump: queued %d proposal(s) from %s (key=%s latency=%dms actions=%d dropped=%d patch=%s) %r",
            len(rows),
            caller,
            key,
            latency_ms,
            len(answer.actions),
            len(dropped),
            bool(patch),
            [(r["extra_args"], r["extra_envs"], r.get("votes")) for r in rows],
        )
    except Exception:  # noqa: BLE001 — advisory work must never fail a session
        log.exception("predictor_pump (%s) failed", caller)
