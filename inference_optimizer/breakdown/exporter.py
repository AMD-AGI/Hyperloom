# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Top-level builder for ``session_breakdown.json``.

Shared by the dump CLI, the coordinator action, and ``cli.py``'s finally
safety net, all calling :func:`build` (pure) or :func:`write_breakdown_json`
(atomic write).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from . import collectors
from .schema import SCHEMA_VERSION, SCHEMA_VERSION_V3

log = logging.getLogger(__name__)

EXPORTER_VERSION = "session-breakdown-1.0.0"
BREAKDOWN_FILENAME = "session_breakdown.json"


def _phase_event_key(ev: dict[str, Any]) -> tuple[str, str, str]:
    """Dedupe key matching :func:`collectors.collect_phase_timeline`.

    ``(action, ts-to-second, change|task_id)`` with the timestamp canonicalised
    to ``...Z`` so a recorder fragment row and the collector's audit-list row
    for the same attempt collapse to one event.
    """
    return (
        str(ev.get("action") or ""),
        collectors._iso_z(ev.get("ts"))[:19],
        str(ev.get("change") or ev.get("task_id") or ""),
    )


def _merge_phase_timeline(
    fragment: Any, collector_value: Any,
) -> list[dict[str, Any]]:
    """Union the recorder ``phase_timeline`` fragment with the collector result.

    The collector merges three sources (optimization_journal + audit lists +
    kernel_opt/integrate lanes); the recorder fragment only carries audit-action
    attempts. A plain fragment-wins replacement (``_pick``) would therefore drop
    the journal KEEP/REVERT and kernel lanes, so we keep the collector result as
    the base and only append fragment rows whose dedupe key is missing (the
    crash-survivable audit rows the on-disk state may have lost). Result stays
    sorted by ``ts`` like the collector's own output.
    """
    base: list[dict[str, Any]] = (
        list(collector_value) if isinstance(collector_value, list) else []
    )
    if not isinstance(fragment, list) or not fragment:
        return base
    seen = {_phase_event_key(ev) for ev in base if isinstance(ev, dict)}
    for ev in fragment:
        if not isinstance(ev, dict):
            continue
        # Normalise to the collector's audit-row shape so keys line up.
        norm = dict(ev)
        norm.setdefault("kernel_id", None)
        norm.setdefault("phase", "")
        norm.setdefault("change", str(ev.get("action") or ""))
        key = _phase_event_key(norm)
        if key in seen:
            continue
        seen.add(key)
        base.append(norm)
    base.sort(key=lambda e: e.get("ts") or "")
    return base


def _load_state(session_dir: Path, warnings: list[str]) -> dict[str, Any]:
    """Read ``state.json`` as a plain dict; empty dict + warning when missing."""
    state_path = session_dir / "state.json"
    if not state_path.exists():
        warnings.append(f"state.json missing at {state_path}")
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to parse state.json: {exc!r}")
        return {}


def _load_manifest(session_dir: Path, warnings: list[str]) -> dict[str, Any]:
    """Read ``manifest.json`` as a plain dict.

    Args:
        session_dir (Path): The hyperloom session directory.
        warnings (list[str]): Accumulator appended to when the file is missing
            or unparseable.

    Returns:
        dict[str, Any]: The parsed ``manifest.json`` contents, or an empty
            dict on any failure.
    """
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.exists():
        warnings.append(f"manifest.json missing at {manifest_path}")
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"failed to parse manifest.json: {exc!r}")
        return {}


_ROOFLINE_NUMERIC_FIELDS = (
    "arithmetic_intensity",
    "flops_per_byte",
    "efficiency_percent",
)


def _attach_kernel_roofline(
    kernel_journey: dict[str, Any],
    kernel_roofline: dict[str, Any],
) -> None:
    """Merge per-kernel roofline metrics into the ``kernel_journey`` view.

    For every journey entry with a matching ``kernel_roofline`` kernel (by
    ``kernel_id``, falling back to ``name``), attach the full roofline entry
    under ``roofline`` and backfill the discovery numeric fields that discovery
    surfaced empty (roofline enrichment happens after discovery records). Pure
    best-effort: a missing/empty roofline table or kernel just leaves the
    journey untouched.
    """
    if not isinstance(kernel_journey, dict) or not isinstance(
        kernel_roofline, dict,
    ):
        return
    kernels = kernel_journey.get("kernels")
    roofline_kernels = kernel_roofline.get("kernels")
    if not isinstance(kernels, list) or not isinstance(roofline_kernels, list):
        return

    by_kid: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for rk in roofline_kernels:
        if not isinstance(rk, dict):
            continue
        kid = str(rk.get("kernel_id") or "")
        name = str(rk.get("name") or "")
        if kid:
            by_kid.setdefault(kid, rk)
        if name:
            by_name.setdefault(name, rk)

    for entry in kernels:
        if not isinstance(entry, dict):
            continue
        rk = by_kid.get(str(entry.get("kernel_id") or "")) or by_name.get(
            str(entry.get("name") or ""),
        )
        if not rk:
            continue
        entry["roofline"] = dict(rk)
        # Promote bound_type onto the entry header when discovery left it blank.
        if not str(entry.get("bound_type") or "") and rk.get("bound_type"):
            entry["bound_type"] = rk.get("bound_type")
        disc = entry.get("discovery")
        if not isinstance(disc, dict):
            continue
        if not str(disc.get("bound_type") or "") and rk.get("bound_type"):
            disc["bound_type"] = rk.get("bound_type")
        for field in _ROOFLINE_NUMERIC_FIELDS:
            if disc.get(field) in (None, 0, 0.0) and rk.get(field) not in (
                None, 0, 0.0,
            ):
                disc[field] = rk.get(field)


def build(
    session_dir: Path | str,
    *,
    include_transcripts: bool | None = None,
) -> dict[str, Any]:
    """Build a complete :class:`SessionBreakdown` for ``session_dir`` (pure; reads disk, no mutation).

    Args:
        session_dir: hyperloom session directory (needs ``manifest.json``
            or ``state.json`` for usable output).
        include_transcripts: inline specialist transcripts. ``None``
            consults ``INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS=1``;
            defaults to False since transcripts are large.

    Returns:
        A dict matching :class:`schema.SessionBreakdown`.
    """
    sd = Path(session_dir).resolve()
    warnings: list[str] = []
    if include_transcripts is None:
        include_transcripts = (
            os.environ.get(
                "INFERENCE_OPTIMIZER_BREAKDOWN_INCLUDE_TRANSCRIPTS", "",
            ).strip().lower() in ("1", "true", "yes")
        )

    state = _load_state(sd, warnings)
    manifest = _load_manifest(sd, warnings)

    # Author-time recorder fragments (write-side spool). When present they are
    # the source of truth for their section (they capture facts the collectors
    # can miss: pre-dispatch/infra failures, pruned robustness signals, ...);
    # when absent the legacy collectors are used as fallback, so historical
    # sessions and recorder-disabled runs behave exactly as before.
    assembled = _load_assembled(sd, warnings)
    # Version stamp follows the aggregation path: a recorder-aggregated
    # breakdown ("new way", any fragments present) is v3.0; the legacy
    # collector-only fallback keeps the previous v2 version unchanged.
    schema_version = SCHEMA_VERSION_V3 if assembled else SCHEMA_VERSION

    def _pick(section: str, collector_value: Any) -> Any:
        """Fragment value if recorded and non-empty, else the collector value."""
        frag = assembled.get(section)
        if isinstance(frag, list) and frag:
            return frag
        if isinstance(frag, dict) and frag:
            return frag
        return collector_value

    from datetime import datetime, timezone
    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Section collectors (each catches its own errors via warnings).
    session_meta      = _pick("session", _safe_collect("session",
                                       lambda: collectors.collect_session(sd, state, manifest, warnings),
                                       warnings))
    workload          = _pick("workload", _safe_collect("workload",
                                       lambda: collectors.collect_workload(state, manifest, warnings),
                                       warnings))
    baseline          = _pick("baseline", _safe_collect("baseline",
                                       lambda: collectors.collect_baseline(sd, state, warnings),
                                       warnings))
    final             = _pick("final", _safe_collect("final",
                                       lambda: collectors.collect_final(sd, state, warnings),
                                       warnings))
    # Merge (not _pick replace): the recorder fragment only carries audit-action
    # attempts, while the collector also folds in optimization_journal KEEP/REVERT
    # and the kernel_opt/integrate lanes. Fragment-wins would drop those (and
    # cascade into action_timeline / phase_segments / token_usage which all
    # derive from this), so union + dedupe instead.
    phase_timeline    = _merge_phase_timeline(
        assembled.get("phase_timeline"),
        _safe_collect("phase_timeline",
                      lambda: collectors.collect_phase_timeline(sd, state, warnings),
                      warnings),
    )
    # Derived from the resolved (fragment-or-collector) phase_timeline.
    phase_segments    = _safe_collect("phase_segments",
                                       lambda: collectors.collect_phase_segments(
                                           state, phase_timeline, warnings,
                                       ),
                                       warnings,
                                       default=[])
    geak_c, oob_c = _safe_collect(
        "invocations",
        lambda: collectors.collect_kernel_invocations(sd, warnings),
        warnings,
        default=([], []),
    )
    geak_invocations = _pick("geak_invocations", geak_c)
    oob_invocations = _pick("oob_invocations", oob_c)
    capability_summary = _safe_collect("capability_summary",
                                        lambda: collectors.collect_capability_summary(
                                            state, geak_invocations, oob_invocations, warnings,
                                        ),
                                        warnings)
    kernel_lifecycle   = _safe_collect("kernel_lifecycle",
                                        lambda: collectors.collect_kernel_lifecycle(
                                            sd, state, geak_invocations, oob_invocations, warnings,
                                        ),
                                        warnings)
    explore_search     = _pick("explore_search", _safe_collect("explore_search",
                                        lambda: collectors.collect_explore_search(state, warnings),
                                        warnings))
    sweep              = _pick("sweep", _safe_collect("sweep",
                                        lambda: collectors.collect_sweep(sd, state, warnings),
                                        warnings))
    critic_robustness  = _pick("critic_robustness", _safe_collect("critic_robustness",
                                        lambda: collectors.collect_critic_robustness(sd, warnings),
                                        warnings))
    telemetry          = _pick("telemetry", _safe_collect("telemetry",
                                        lambda: collectors.collect_telemetry(sd, state, warnings),
                                        warnings))
    attribution        = _safe_collect("attribution",
                                        lambda: collectors.collect_attribution(
                                            state, geak_invocations, oob_invocations,
                                            kernel_lifecycle.get("adopted") or [],
                                            warnings,
                                        ),
                                        warnings)
    kb_provenance      = _pick("kb_provenance", _safe_collect("kb_provenance",
                                        lambda: collectors.collect_kb_provenance(
                                            session_dir, state, manifest, warnings,
                                        ),
                                        warnings))
    # specialist sub-agent dispatch records (single source: state + on-disk transcripts).
    specialist_runs    = _pick("specialist_runs", _safe_collect("specialist_runs",
                                        lambda: collectors.collect_specialist_runs(
                                            sd, state, warnings,
                                            include_transcripts=include_transcripts,
                                        ),
                                        warnings,
                                        default=[]))
    # Raw ``state.optimization_stack[]`` passthrough (full per-entry evidence; never raises).
    optimization_stack = _pick("optimization_stack", _safe_collect("optimization_stack",
                                        lambda: collectors.collect_optimization_stack(state),
                                        warnings,
                                        default=[]))
    # Hot-kernel roofline table (Dashboard §1) from ``<sd>/reports/kernel_roofline.json``.
    kernel_roofline    = _pick("kernel_roofline", _safe_collect("kernel_roofline",
                                        lambda: collectors.collect_kernel_roofline(
                                            sd, warnings,
                                        ),
                                        warnings,
                                        default={}))
    # Kernel-agent attempt outcome summary (Breakdown panel integration spec §A1);
    # mirrors ``reports/kernel_optimization_summary.json``, empty → dashboard hides Block 1.
    kernel_optimization_summary = _pick("kernel_optimization_summary", _safe_collect(
        "kernel_optimization_summary",
        lambda: collectors.collect_kernel_optimization_summary(sd, warnings),
        warnings,
        default={}))
    # Post-optimization concurrency sweep (Breakdown panel integration spec §A2);
    # mirrors ``reports/conc_sweep_summary.json``, empty → dashboard hides Block 2.
    conc_sweep_summary = _pick("conc_sweep_summary", _safe_collect(
        "conc_sweep_summary",
        lambda: collectors.collect_conc_sweep_summary(sd, warnings),
        warnings,
        default={}))
    # Per-snapshot roofline comparison list (markdown ``## Roofline`` source) from ``state.roofline_snapshots``.
    roofline           = _pick("roofline", _safe_collect("roofline",
                                        lambda: collectors.collect_roofline(
                                            state, warnings,
                                        ),
                                        warnings,
                                        default=[]))
    # Optimization-progress curve (Dashboard §2): stack ledger + ceiling/target
    # lines from state.json. Renamed from ``roofline`` to avoid clashing with the
    # list-shaped section above.
    roofline_progress  = _pick("roofline_progress", _safe_collect("roofline_progress",
                                        lambda: collectors.collect_roofline_progress(
                                            sd, state, manifest, warnings,
                                        ),
                                        warnings,
                                        default={}))
    # Full-trace: unified token + decision timeline. Joins the per-call
    # token ledger (reports/trace/llm_calls.jsonl + ext/*.jsonl) with the
    # KEEP/REVERT journal + dynamic_action dispatch history. Empty (zeroed
    # rollup) on sessions that predate the trace subsystem. Also writes
    # reports/trace/decision_trace.jsonl as a side effect.
    decision_trace     = _safe_collect("decision_trace",
                                        lambda: collectors.collect_decision_trace(
                                            sd, state, warnings,
                                        ),
                                        warnings,
                                        default={})
    # Promoted, discoverable token-spend rollup. Pure/derived from
    # decision_trace's already-computed token_rollup (no second ledger read)
    # plus an action_timeline correlation on task_id. Surfaces the full
    # session total + by component/phase + decision attribution at top level
    # so callers don't have to dig into decision_trace.token_rollup.
    token_usage        = _safe_collect("token_usage",
                                        lambda: collectors.collect_token_usage(
                                            decision_trace, phase_timeline, warnings,
                                        ),
                                        warnings,
                                        default={})
    # Live-Langfuse push receipt (opt-in second sink): enabled? / redacted
    # config / counts. Prefers the post-flush ``langfuse_receipt.json``;
    # falls back to a live emitter read. The local trace jsonl is always
    # written regardless of this section.
    langfuse           = _safe_collect("langfuse",
                                        lambda: collectors.collect_langfuse(
                                            sd, manifest, warnings,
                                        ),
                                        warnings,
                                        default={})
    # Kernel-major lifecycle view (discovery -> dispatch -> backend attempts ->
    # e2e), composed by the recorder assembler from its four item substreams.
    # Pure recorder section (no collector fallback): empty {} on sessions that
    # predate the substreams, so v1/v2 readers that don't know it just ignore
    # it and historical breakdowns stay byte-for-byte identical.
    # Authoritative external-tool versions, folded into a {tool: meta} map by
    # the recorder assembler. Pure recorder section (no collector fallback).
    versions           = _pick("versions", {})
    kernel_journey     = _pick("kernel_journey", {})
    # Attach a copy of the per-kernel roofline metrics onto each journey entry
    # and backfill discovery numeric fields that discovery couldn't surface
    # (arithmetic_intensity / bound_type / efficiency) since roofline is
    # enriched after discovery. Best-effort; never raises.
    _attach_kernel_roofline(kernel_journey, kernel_roofline)

    source_files = collectors.collect_source_files(
        sd,
        baseline.get("benchmark_report_path"),
        telemetry.get("profile_report_paths") or [],
        [p.get("benchmark_report_path") for p in (sweep.get("all_variants") or [])
         if p.get("benchmark_report_path")],
    )

    return {
        "schema_version":      schema_version,
        "exported_at_utc":     exported_at,
        "exporter_version":    EXPORTER_VERSION,

        "session":             session_meta,
        "workload":            workload,
        "baseline":            baseline,
        "final":               final,
        "phase_timeline":      phase_timeline,
        # Additive: v1 readers use flat ``phase_timeline``, v2 prefer ``phase_segments``.
        "phase_segments":      phase_segments,
        # v1-reader alias mirroring the flat per-action timeline.
        "action_timeline":     phase_timeline,
        "capability_summary":  capability_summary,
        "geak_invocations":    geak_invocations,
        "oob_invocations":     oob_invocations,
        "kernel_lifecycle":    kernel_lifecycle,
        "param_search":        explore_search,
        # v2-native name for the merged ledger; mirrors ``param_search``.
        "explore_search":      explore_search,
        "sweep":               sweep,
        "critic_robustness":   critic_robustness,
        "telemetry":           telemetry,
        "attribution":         attribution,
        # Cortex KB integration audit; optional, so no schema_version bump.
        "kb_provenance":       kb_provenance,
        "specialist_runs":     specialist_runs,
        # Raw KEEP ledger passthrough mirroring ``state.optimization_stack[]``.
        "optimization_stack":  optimization_stack,
        # Hot-kernel roofline table (spec §1); empty → dashboard hides it.
        "kernel_roofline":     kernel_roofline,
        # Kernel-agent attempt outcome summary (spec §A1); empty → hides Block 1.
        "kernel_optimization_summary": kernel_optimization_summary,
        # Post-optimization concurrency sweep (spec §A2); empty → hides Block 2.
        "conc_sweep_summary":  conc_sweep_summary,
        # Per-snapshot roofline comparison list (markdown source).
        "roofline":            roofline,
        # Optimization-progress curve (spec §2); ``ceiling_available`` False
        # when the watermark roofline pipeline never ran.
        "roofline_progress":   roofline_progress,
        # Full-trace token + decision timeline (FULL_TRACE_DESIGN §6).
        # ``decision_trace`` is the per-decision join (phase/tick/decision
        # + token rollup); ``token_rollup`` is the by_phase / by_component
        # / session_total summary. New optional section — v1 readers ignore
        # it. Empty on pre-trace sessions.
        "decision_trace":      decision_trace,
        # Promoted token-spend summary (full total + by component/phase +
        # decision attribution + action_timeline correlation). Derived from
        # decision_trace.token_rollup; additive, v1 readers ignore it.
        "token_usage":         token_usage,
        # Live-Langfuse push receipt; ``enabled`` False (with a
        # ``disabled_reason``) on the default path. Local jsonl ledger is
        # always written regardless.
        "langfuse":            langfuse,
        # Kernel-major lifecycle view (discovery -> dispatch -> backend
        # attempts -> e2e), composed from the recorder substreams. Additive,
        # optional; empty {} on sessions that predate the substreams.
        "kernel_journey":      kernel_journey,
        # Authoritative external-tool versions, one object per tool
        # (geak / tracelens / claude / codex / ...), keyed by tool name. Each
        # carries ``{tool, root_dir, commit, version}``. Empty {} on sessions
        # that predate the recorder.
        "versions":            versions,

        "warnings":            warnings,
        "source_files":        source_files,
    }


def _load_assembled(
    session_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    """Assemble recorder fragments into ``{section: value}`` (empty on opt-out
    or when no fragments exist). Never raises — a recorder bug must not poison
    the export; it just falls back to collectors."""
    disabled = os.environ.get(
        "INFERENCE_OPTIMIZER_BREAKDOWN_DISABLE_RECORDER", "",
    ).strip().lower() in ("1", "true", "yes")
    if disabled:
        return {}
    try:
        from .recorder import assemble_parts, has_parts

        if not has_parts(session_dir):
            return {}
        out = assemble_parts(session_dir, warnings=warnings)
        return out if isinstance(out, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.exception("recorder: assemble_parts failed")
        warnings.append(f"recorder: assemble_parts failed: {type(exc).__name__}: {exc}")
        return {}


def _safe_collect(
    name: str,
    fn: callable,
    warnings: list[str],
    *,
    default: Any = None,
):
    """Run a collector with broad exception catching; failure → warning + ``default`` (a bug in one collector must not poison the export)."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        log.exception("collector %s failed", name)
        warnings.append(f"collector:{name} failed: {type(exc).__name__}: {exc}")
        if default is not None:
            return default
        return {}


def write_breakdown_json(
    session_dir: Path | str,
    *,
    output_path: Path | str | None = None,
    include_transcripts: bool | None = None,
) -> Path:
    """Build + atomically write ``session_breakdown.json``; returns the absolute path.

    ``output_path`` defaults to ``<session_dir>/session_breakdown.json``;
    ``include_transcripts`` is as in :func:`build`.
    """
    sd = Path(session_dir).resolve()
    target = Path(output_path).resolve() if output_path else sd / BREAKDOWN_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    breakdown = build(sd, include_transcripts=include_transcripts)
    payload = json.dumps(breakdown, indent=2, sort_keys=True, default=_json_default)

    fd, tmp = tempfile.mkstemp(
        prefix=f".{BREAKDOWN_FILENAME}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    log.info("session_breakdown: wrote %s (%d bytes)", target, len(payload))
    return target


def patch_breakdown_langfuse(session_dir: Path | str) -> bool:
    """Refresh only the ``langfuse`` section of an already-written breakdown.

    ``session_breakdown.json`` is written *before* the session-end
    ``flush_session`` (the flush depends on ``decision_trace.jsonl``, which the
    breakdown produces). So the breakdown's first ``langfuse`` section carries
    the pre-flush, in-process counts (``counts_final=False``). Call this right
    after ``flush_session`` to splice in the post-flush
    ``langfuse_receipt.json`` (final counts) without rebuilding the whole file.

    Best-effort and self-skipping: returns False (no-op) when no breakdown or
    no receipt exists yet, when live push was disabled, or on any error. Never
    raises -- it must not mask the session's stop_reason at shutdown.
    """
    from ..orchestrator.trace.langfuse_emitter import read_receipt

    sd = Path(session_dir).resolve()
    target = sd / BREAKDOWN_FILENAME
    try:
        receipt = read_receipt(sd)
        if receipt is None or not target.exists():
            return False
        receipt["receipt_source"] = "receipt_file"
        breakdown = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(breakdown, dict):
            return False
        if breakdown.get("langfuse") == receipt:
            return False  # already current
        breakdown["langfuse"] = receipt
        payload = json.dumps(breakdown, indent=2, sort_keys=True, default=_json_default)
        fd, tmp = tempfile.mkstemp(
            prefix=f".{BREAKDOWN_FILENAME}.", suffix=".tmp", dir=str(target.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, target)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        log.info("session_breakdown: refreshed langfuse section in %s", target)
        return True
    except Exception:  # noqa: BLE001
        log.debug("session_breakdown: langfuse patch failed (non-fatal)", exc_info=True)
        return False


def _json_default(obj: Any) -> Any:
    """Stringify objects json.dumps can't handle natively (Path, set, ...).

    Args:
        obj (Any): The object ``json.dumps`` could not serialize.

    Returns:
        Any: ``str(obj)`` for :class:`~pathlib.Path`, a sorted list for
            ``set``.

    Raises:
        TypeError: If ``obj`` is of an unsupported type.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_minimal_final_report(
    session_dir: Path | str,
    *,
    output_path: Path | str | None = None,
) -> Path:
    """Issue-I: cli.finally safety-net for ``reports/final.md`` when the CLOSE sequencer never reached step 1.

    Stays minimal (one SharedState read) so it never raises or blocks
    shutdown. Idempotent: never overwrites an existing ``reports/final.md``.
    """
    from ..orchestrator.shared_state import SharedState
    from ..session_paths import reports_dir

    sd = Path(session_dir).resolve()
    target = (
        Path(output_path).resolve()
        if output_path
        else reports_dir(sd) / "final.md"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target

    state = SharedState.load_or_init(sd)
    breakdown_link = sd / BREAKDOWN_FILENAME

    def _fmt_attempt(d: dict[str, Any] | None, label: str) -> str:
        """Format one ``last_*`` attempt record as a markdown bullet.

        Args:
            d (dict[str, Any] | None): The attempt record (or ``None``).
            label (str): The bullet label (e.g. ``"last_sweep"``).

        Returns:
            str: A markdown bullet line; ``"(none)"`` when the record is
                empty.
        """
        if not isinstance(d, dict) or not d:
            return f"- **{label}**: (none)"
        ts = d.get("ts") or "-"
        body = json.dumps(
            {k: v for k, v in d.items() if k != "ts"},
            sort_keys=True,
            default=str,
        )[:600]
        return f"- **{label}** (`{ts}`): `{body}`"

    current_best = state.current_best or {}
    cb_action = current_best.get("action") or "-"
    cb_tput = current_best.get("tput")
    cb_tput_s = (
        f"{cb_tput:.2f}" if isinstance(cb_tput, (int, float)) else "-"
    )
    last_sweep = state.last_sweep or {}
    if last_sweep:
        sw_grid = last_sweep.get("grid_size", 0)
        sw_best = last_sweep.get("best_overall") or {}
        sw_tput = sw_best.get("output_throughput")
        sw_line = (
            f"grid_size={sw_grid} "
            f"best_tput="
            f"{(f'{sw_tput:.2f}' if isinstance(sw_tput, (int, float)) else '-')}"
        )
    else:
        sw_line = "(none)"

    lines = [
        "# Inference Optimizer — emergency final report",
        "",
        "> **Auto-generated safety-net.** The CLOSE phase 5-step "
        + "sequencer did not run to completion (process exited before "
        + "phase transition, or ``report`` executor failed). For the "
        + "full audit trail open `session_breakdown.json` next to this "
        + "file.",
        "",
        f"- session_id     : `{state.session_id or '-'}`",
        f"- model_path     : `{state.model_path or '-'}`",
        f"- framework      : `{state.framework or '-'}`",
        f"- gpu_type       : `{state.gpu_type or '-'}`",
        f"- phase (last)   : `{state.phase or '-'}`",
        f"- stop_reason    : `{state.stop_reason or '-'}`",
        f"- baseline_tput  : `{state.baseline_tput:.2f}`",
        f"- current_best   : `{cb_action}` @ `{cb_tput_s}` tok/s",
        f"- cumul_gain     : `{state.cumulative_gain:.2f}%` "
        f"(validated `{state.cumulative_gain_validated:.2f}%`)",
        f"- stack_entries  : `{len(state.optimization_stack or [])}`",
        f"- sweep summary  : {sw_line}",
        "",
        "## Last action attempts",
        "",
        _fmt_attempt(getattr(state, "last_baseline", None), "last_baseline"),
        _fmt_attempt(getattr(state, "last_profile", None), "last_profile"),
        _fmt_attempt(getattr(state, "last_explore", None), "last_explore"),
        _fmt_attempt(state.last_sweep, "last_sweep"),
        "",
        "## Structured detail",
        "",
        f"See `{breakdown_link.name}` (sibling of session root) for the "
        f"complete `phase_history` / `critic_robustness` / "
        f"`kb_provenance` blocks.",
        "",
    ]

    fd, tmp = tempfile.mkstemp(
        prefix=".final.md.", suffix=".tmp", dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text("\n".join(lines), encoding="utf-8")
        os.replace(tmp_path, target)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    log.info("emergency final report: wrote %s", target)
    return target


__all__ = [
    "BREAKDOWN_FILENAME",
    "EXPORTER_VERSION",
    "build",
    "write_breakdown_json",
    "write_minimal_final_report",
]
