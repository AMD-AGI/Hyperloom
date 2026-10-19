# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The recipe-shaped view of a session's settled :class:`SharedState`.

Every projection here is a pure read of the state the optimization loop has
already settled, shaped the way the Recipe KB stores it: the launch
configuration that won, the stack that produced it, the kernels that landed,
and the workload dimensions a warm start filters on.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Collection, Iterable, Mapping, NamedTuple, Sequence

from hyperloom.common.coerce import to_float
from hyperloom.inference_optimizer.session.optimization_journal import OUTCOME_KEEP, OUTCOME_REVERT

from .workload_tags import workload_shape_tags

#: SharedState attributes naming the workload; ``model_family`` is deliberately
#: absent, being no part of the seven-dimension Recipe identity.
_MODEL_IDENTITY_ATTRS: tuple[str, ...] = ("framework", "model_class", "model_name")

#: The serving frameworks whose version the boot-time fingerprint records.
_VERSIONED_FRAMEWORKS: frozenset[str] = frozenset({"sglang", "vllm"})

#: Stack-fingerprint keys that become recipe tags under their own names.
_RUNTIME_VERSION_KEYS: tuple[tuple[str, str], ...] = (
    ("rocm", "rocm_version"),
    ("aiter", "aiter_version"),
    ("image_digest", "image_digest"),
)

#: The probe's own word for a version it could not read.
_UNIDENTIFIED_VERSION: str = "unknown"

#: Baseline workload knobs carried onto the recipe, grouped by the type that
#: makes a value meaningful. A ``False`` survives: it is an explicit disable.
_POSITIVE_INT_KNOBS: tuple[str, ...] = ("max_running_requests", "max_num_seqs")
_BOOLEAN_KNOBS: tuple[str, ...] = ("chunked_prefill_enabled", "enable_torch_compile")
_TEXT_KNOBS: tuple[str, ...] = ("quant_scheme", "workload_mode")

#: ``best_config`` fields copied from ``current_best`` as they stand.
_BEST_CONFIG_PASSTHROUGH: tuple[str, ...] = ("extra_envs", "name", "tput", "accuracy")

#: How many recent action failures the recipe's ``what_failed`` carries.
_RECENT_FAILURE_LIMIT: int = 10


class AttemptProvenance(NamedTuple):
    """Where a session's proven optimizations came from, and what it reverted.

    Keepers are indexed twice because a stack layer may identify itself either
    by name or by the research gap it closed.
    """

    kept_by_name: dict[str, str]
    kept_by_gap: dict[str, str]
    reverted: list[dict[str, Any]]

    @classmethod
    def unrecorded(cls) -> "AttemptProvenance":
        """The empty provenance a session that does not sediment its attempts reports."""
        return cls(kept_by_name={}, kept_by_gap={}, reverted=[])


class _IntegrateIndex(NamedTuple):
    """Integrate-layer entries under both keys a micro KEEP can name them by."""

    by_kernel: dict[str, dict[str, Any]]
    by_task_group: dict[str, dict[str, Any]]


class _E2eVerdict(NamedTuple):
    """What integrating a kernel measured end to end."""

    gain_pct: float
    throughput: float
    decision: str
    integrated: bool


def _known_version(raw: Any) -> str:
    """A fingerprint version, empty when the probe never identified one."""
    version = str(raw or "").strip()
    return "" if version == _UNIDENTIFIED_VERSION else version


def _model_identity_tags(state: Any) -> dict[str, Any]:
    """The workload's model-identity tags, omitting the ones this session never learned."""
    tags: dict[str, Any] = {}
    for attr in _MODEL_IDENTITY_ATTRS:
        text = str(getattr(state, attr, "") or "").strip()
        if text:
            tags[attr] = text
    precision = getattr(state, "precision", None)
    if precision not in (None, "", 0):
        tags["precision"] = precision
    return tags


def _runtime_version_tags(fingerprint_meta: Mapping[str, Any], framework: str) -> dict[str, str]:
    """Runtime versions read off the stack fingerprint the CLI wrote at boot.

    ``framework_version`` names whichever serving framework is active, so a
    recipe written under one is never matched against the other.
    """
    tags: dict[str, str] = {}
    framework_key = framework.lower()
    if framework_key in _VERSIONED_FRAMEWORKS:
        active = _known_version(fingerprint_meta.get(framework_key))
        if active:
            tags["framework_version"] = active
    for source_key, tag in _RUNTIME_VERSION_KEYS:
        version = _known_version(fingerprint_meta.get(source_key))
        if version:
            tags[tag] = version
    return tags


def _baseline_workload_tags(workload_extra: Mapping[str, Any]) -> dict[str, Any]:
    """Per-baseline workload knobs from the materialized YAML, each kept only when its type says it was set."""
    tags: dict[str, Any] = {}
    for key in _POSITIVE_INT_KNOBS:
        value = workload_extra.get(key)
        if isinstance(value, int) and value > 0:
            tags[key] = value
    for key in _BOOLEAN_KNOBS:
        value = workload_extra.get(key)
        if isinstance(value, bool):
            tags[key] = value
    for key in _TEXT_KNOBS:
        value = workload_extra.get(key)
        if isinstance(value, str) and value.strip():
            tags[key] = value.strip()
    return tags


def workload_kb_tags(state: Any) -> dict[str, Any]:
    """The workload-shape tags describing the session this recipe came from.

    Shared by the recipe attrs and by every lesson or pitfall write, so the
    warm-start reader filters on the dimensions the row was written with.
    """
    tags = _model_identity_tags(state)
    tags.update(workload_shape_tags(state))
    fingerprint_meta = state.stack_fingerprint_meta or {}
    if isinstance(fingerprint_meta, dict):
        tags.update(_runtime_version_tags(fingerprint_meta, str(tags.get("framework") or "")))
    workload_extra = state.baseline_workload_extra or {}
    if isinstance(workload_extra, dict):
        tags.update(_baseline_workload_tags(workload_extra))
    return tags


def _index_integrate_attempts(attempts: Mapping[str, Any]) -> _IntegrateIndex:
    """Index the integrate ledger by kernel and by task group; the last write wins."""
    by_kernel: dict[str, dict[str, Any]] = {}
    by_task_group: dict[str, dict[str, Any]] = {}
    for entry in attempts.values():
        if not isinstance(entry, dict):
            continue
        kernel_id = str(entry.get("kernel_id") or "")
        if kernel_id:
            by_kernel[kernel_id] = entry
        task_group_key = str(entry.get("task_group_key") or "")
        if task_group_key:
            by_task_group[task_group_key] = entry
    return _IntegrateIndex(by_kernel=by_kernel, by_task_group=by_task_group)


def _last_rebenched_throughput(integration: Mapping[str, Any]) -> float:
    """Throughput from the most recent integrate attempt that re-benched end to end."""
    for attempt in reversed(list(integration.get("attempts") or [])):
        if isinstance(attempt, dict) and attempt.get("new_tput") is not None:
            return to_float(attempt.get("new_tput"), 0.0)
    return 0.0


def _e2e_verdict(integration: Any) -> _E2eVerdict:
    """The integrate layer's verdict on a kernel, or the never-integrated default.

    Carrying the verdict lets a warm start skip a kernel that won on the micro
    bench and lost end to end.
    """
    if not isinstance(integration, dict):
        return _E2eVerdict(gain_pct=0.0, throughput=0.0, decision="", integrated=False)
    return _E2eVerdict(
        gain_pct=to_float(integration.get("best_gain_pct"), 0.0),
        throughput=_last_rebenched_throughput(integration),
        decision=str(integration.get("last_decision") or "").upper(),
        integrated=True,
    )


def _kernel_optimization_row(entry: Mapping[str, Any], ledger_id: str, index: _IntegrateIndex) -> dict[str, Any]:
    """One KernelOptimization row: a micro KEEP joined to its end-to-end verdict."""
    kernel_id = str(entry.get("current_kernel_id") or entry.get("kernel_id") or ledger_id)
    task_group_key = str(entry.get("task_group_key") or "")
    integration = index.by_task_group.get(task_group_key) if task_group_key else index.by_kernel.get(kernel_id)
    e2e = _e2e_verdict(integration)
    return {
        "kernel_id": kernel_id,
        "source_file": str(entry.get("last_source_file") or ""),
        "artifact_path": str(entry.get("last_artifact_path") or ""),
        "micro_speedup": to_float(entry.get("last_micro_speedup"), 0.0),
        "decision": OUTCOME_KEEP,
        "e2e_gain_pct": e2e.gain_pct,
        "e2e_tput": e2e.throughput,
        "e2e_decision": e2e.decision,
        "integrated": e2e.integrated,
        "ts": str(entry.get("last_ts") or ""),
    }


def kept_kernel_optimizations(state: Any) -> list[dict[str, Any]]:
    """The KEEP'd kernel optimizations, each joined to its end-to-end integrate verdict.

    A KEEP that never reached integration surfaces with ``integrated`` false
    rather than being dropped, because not having been tried end to end is not
    the same as having lost there.
    """
    optimizations = state.kernel_opt_task_attempts or {}
    if not isinstance(optimizations, dict):
        return []
    integrations = state.kernel_integrate_attempts or {}
    index = _index_integrate_attempts(integrations if isinstance(integrations, dict) else {})
    rows: list[dict[str, Any]] = []
    for ledger_id, entry in optimizations.items():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("last_decision", "")).upper() != OUTCOME_KEEP:
            continue
        rows.append(_kernel_optimization_row(entry, ledger_id, index))
    return rows


def _reverted_attempt_row(name: str, attempt: Mapping[str, Any], source: str) -> dict[str, Any]:
    """A ``what_failed`` row for an attempt this session measured and reverted."""
    row: dict[str, Any] = {
        "name": name,
        "reason": "reverted",
        "gain_pct": attempt.get("gain_pct"),
    }
    if source:
        row["source"] = source
    return row


def _gap_attempt_provenance(gap: Mapping[str, Any]) -> AttemptProvenance:
    """One research gap's attempts, split into what stuck and what was reverted."""
    source = str(gap.get("provenance") or "").strip()
    canonical_id = str(gap.get("canonical_id") or "").strip()
    kept_by_name: dict[str, str] = {}
    kept_by_gap: dict[str, str] = {}
    reverted: list[dict[str, Any]] = []
    for attempt in gap.get("attempts") or []:
        if not isinstance(attempt, dict):
            continue
        variant = str(attempt.get("variant_name") or "").strip()
        kernel = str(attempt.get("kernel_id") or "").strip()
        outcome = str(attempt.get("outcome") or "").strip().upper()
        if outcome == OUTCOME_KEEP and source:
            for name in (variant, kernel):
                if name:
                    kept_by_name.setdefault(name, source)
            if canonical_id:
                kept_by_gap.setdefault(canonical_id, source)
        elif outcome == OUTCOME_REVERT and (variant or kernel):
            reverted.append(_reverted_attempt_row(variant or kernel, attempt, source))
    return AttemptProvenance(kept_by_name=kept_by_name, kept_by_gap=kept_by_gap, reverted=reverted)


def attempt_provenance(state: Any) -> AttemptProvenance:
    """Trace each proven optimization back to the research hint it came from.

    The first gap to claim a name keeps it, so a name two gaps both attempted
    reads as belonging to the one that reached it first.
    """
    kept_by_name: dict[str, str] = {}
    kept_by_gap: dict[str, str] = {}
    reverted: list[dict[str, Any]] = []
    for gap in state.gaps or []:
        if not isinstance(gap, dict):
            continue
        from_gap = _gap_attempt_provenance(gap)
        for name, source in from_gap.kept_by_name.items():
            kept_by_name.setdefault(name, source)
        for canonical_id, source in from_gap.kept_by_gap.items():
            kept_by_gap.setdefault(canonical_id, source)
        reverted.extend(from_gap.reverted)
    return AttemptProvenance(kept_by_name=kept_by_name, kept_by_gap=kept_by_gap, reverted=reverted)


def _last_stack_launch_args(optimization_stack: Sequence[Any]) -> str:
    """Launch args from the last validated stack layer, empty when it carries none."""
    if not optimization_stack:
        return ""
    last = optimization_stack[-1]
    if not isinstance(last, dict):
        return ""
    return str(last.get("candidate_extra_server_args") or last.get("extra_server_args") or "").strip()


def _best_config(current_best: Mapping[str, Any], optimization_stack: Sequence[Any]) -> dict[str, Any]:
    """The replayable launch configuration the recipe advertises.

    The last validated stack layer overrides ``current_best``, whose
    ``extra_server_args`` may carry a string an interrupted write corrupted.
    """
    config: dict[str, Any] = {}
    current_args = current_best.get("extra_server_args")
    if current_args:
        config["extra_server_args"] = str(current_args)
    for key in _BEST_CONFIG_PASSTHROUGH:
        if key in current_best:
            config[key] = current_best[key]
    stack_args = _last_stack_launch_args(optimization_stack)
    if stack_args:
        config["extra_server_args"] = stack_args
    return config


def _stack_entry_source(entry: Mapping[str, Any], name: str, provenance: AttemptProvenance) -> str:
    """The research hint a stack layer came from, empty when nothing claims it.

    The layer's gap id is naming-independent and wins; the variant or kernel
    name answers for layers recorded before gap ids were stamped.
    """
    gap_id = str(entry.get("gap_canonical_id") or "").strip()
    if gap_id:
        by_gap = provenance.kept_by_gap.get(gap_id)
        if by_gap:
            return by_gap
    kernel_id = str(entry.get("kernel_id") or "")
    return provenance.kept_by_name.get(name) or provenance.kept_by_name.get(kernel_id) or ""


def _what_worked(
    optimization_stack: Sequence[Any],
    gain_per_entry: Sequence[Any],
    provenance: AttemptProvenance,
) -> list[dict[str, Any]]:
    """One row per stack layer: the change it made, what it gained and where it came from."""
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(optimization_stack):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("variant_name") or entry.get("name") or entry.get("kernel_id") or "")
        row: dict[str, Any] = {
            "name": name,
            "extra_server_args": str(entry.get("extra_server_args") or ""),
            "extra_envs": dict(entry.get("extra_envs") or {}),
            "gain_pct": gain_per_entry[index] if index < len(gain_per_entry) else None,
        }
        source = _stack_entry_source(entry, name, provenance)
        if source:
            row["source"] = source
        rows.append(row)
    return rows


def _what_failed(recent_failures: Sequence[Any], reverted: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The session's recent action failures, followed by every attempt it reverted."""
    rows: list[dict[str, Any]] = []
    for failure in recent_failures[-_RECENT_FAILURE_LIMIT:]:
        if isinstance(failure, dict):
            rows.append(
                {
                    "name": str(failure.get("name") or failure.get("action") or ""),
                    "reason": str(failure.get("reason") or failure.get("error_class") or ""),
                }
            )
    rows.extend(reverted)
    return rows


def _stack_action_names(optimization_stack: Iterable[Any]) -> list[str]:
    """The named actions the optimization stack is built from, in the order they landed."""
    names: list[str] = []
    for entry in optimization_stack:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("variant_name") or entry.get("name") or entry.get("action") or "").strip()
        if name:
            names.append(name)
    return names


def _session_row(
    state: Any,
    session_name: str,
    optimization_stack: Collection[Any],
    final_throughput: float,
) -> dict[str, Any]:
    """This session's row on the recipe: where it started, where it reached and by which actions.

    Self-describing in the arbor shape, so a reader needs neither the recipe's
    other sessions nor this one's journal to price what it contributed.
    """
    validated_stack_len = int(state.cumulative_gain_validated_stack_len or 0)
    return {
        "session_id": str(state.recipe_kb_session_id or session_name),
        "gain_pct": float(state.cumulative_gain_validated or 0.0),
        "stack_len": validated_stack_len or len(optimization_stack),
        "throughput_before": float(state.baseline_tput or 0.0),
        "throughput_after": final_throughput,
        "date": datetime.now(timezone.utc).isoformat(),
        "actions_taken": _stack_action_names(optimization_stack),
    }


def recipe_attrs_from_state(state: Any, session_name: str) -> dict[str, Any]:
    """Materialise the recipe-shaped view of one session's settled state.

    ``framework_version`` is left unset here because it is manifest-derived;
    the T0 backfill writes it.
    """
    optimization_stack = state.optimization_stack or []
    provenance = attempt_provenance(state) if state.recipe_sediment_enabled else AttemptProvenance.unrecorded()
    final_throughput = float(state.current_best.get("tput", 0.0))
    return {
        "best_config": _best_config(state.current_best, optimization_stack),
        "best_throughput": final_throughput,
        "what_worked": _what_worked(optimization_stack, state.gain_per_stack_entry or [], provenance),
        "what_failed": _what_failed(state.last_action_failures or [], provenance.reverted),
        "kernel_optimizations": kept_kernel_optimizations(state),
        "last_profiled": str(state.cumulative_gain_validated_ts or ""),
        "workload": workload_kb_tags(state),
        "sessions": [_session_row(state, session_name, optimization_stack, final_throughput)],
    }
