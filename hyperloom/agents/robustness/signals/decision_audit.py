"""Reverse-audit decision-quality signals (G1-G7).

These rules complement Critic — which only sees *proposals* before they
run — by inspecting the *persisted result* of decisions that already
landed on disk. Many decisions in the inference_optimizer pipeline go
through programmatic paths (the ``integrate`` executor, ``grid_runner``,
``report_back/ci_metrics.py``) that bypass Critic entirely; without a
reverse-audit layer the bad outcomes (empty-patch KEEP, microbench
KEPT but E2E bypassed, schema-broken ci_metrics) sit in the artefact
tree until a human reviewer catches them.

All signals are stateless. The detector reads
:attr:`SourceData.local_decision_audit` populated by
:func:`local_probe._sample_decision_audit`; when the probe is disabled
or no decision artefacts exist yet, every rule short-circuits to ``[]``.

Severity matrix:

* HIGH + escalate/prune — G1 empty patch KEEP, G3 dispatch bypassed,
  G4 negative-delta kernel kept, G5 ci_metrics baseline=0 without
  ``status=baseline_failed`` marker, G7 OOB expected-only proposal.
* MEDIUM + alert — G2 sub-threshold KEEP (noise floor), G6 ci_metrics
  schema drift.

These signals deliberately do not auto-``delegate(report)`` — they are
*audit*, not *recovery*. The ladder's escalate hints route the
remediation through Orchestration (or the operator) so the right team
owns the fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from .symptom import Symptom, SymptomSeverity



# Canonical ci_metrics field set the report_back pipeline must produce.
# Drift detection only fires when the file exists *and* schema_version
# is set to a known value but a required field is missing — this keeps
# the rule quiet for in-flight schema migration.
_CI_METRICS_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "model",
    "framework",
    "gpu",
    "tp",
    "baseline_tok_per_gpu",
    "optimized_tok_per_gpu",
    "gain_pct",
})

# Legacy field names ci_metrics drift detection rejects (they exist in
# old artefacts but the schema migration replaced them).
_CI_METRICS_LEGACY_FIELDS: frozenset[str] = frozenset({
    "baseline_throughput",
    "baseline_tput",
    "baseline_tput_per_gpu",
    "optimized_throughput",
})


@dataclass
class DecisionAuditConfig:
    """Tunables for :func:`evaluate_decision_audit_signals`.

    ``min_keep_gain_pct`` mirrors the upstream KEEP-threshold convention
    (Coordinator's executors use 1.0% as the noise floor). Any KEEP'd
    integrate with ``gain_pct < min_keep_gain_pct`` trips G2.
    ``dispatch_bypass_pre_post_epsilon_pct`` is the absolute gain delta
    below which we suspect the patched kernel was never actually called
    (G3): a KEEP'd attempt where ``|gain_pct| < epsilon`` is suspicious.
    """

    min_keep_gain_pct: float = 1.0
    dispatch_bypass_pre_post_epsilon_pct: float = 0.5
    oob_no_harness_markers: tuple[str, ...] = field(
        default_factory=lambda: ("expected speedup", "expected ~", "expected:")
    )


def evaluate_decision_audit_signals(
    ctx: ReactorContext,
    data: SourceData,
    *,
    config: DecisionAuditConfig | None = None,
) -> list[Symptom]:
    cfg = config or DecisionAuditConfig()
    audit = data.local_decision_audit
    if not isinstance(audit, dict) or not audit:
        return []
    out: list[Symptom] = []

    integrate_entries = audit.get("recent_integrate") or []
    if isinstance(integrate_entries, list):
        out.extend(_integrate_symptoms(integrate_entries, cfg))

    ci_metrics = audit.get("ci_metrics") or {}
    ci_metrics_path = audit.get("ci_metrics_path") or ""
    if isinstance(ci_metrics, dict):
        out.extend(_ci_metrics_symptoms(ci_metrics, ci_metrics_path))

    oob_attempts = audit.get("oob_attempts") or []
    if isinstance(oob_attempts, list):
        out.extend(_oob_symptoms(oob_attempts, cfg))

    return out


# ---------------------------------------------------------------------------
# G1 / G2 / G3 — integrate result.json audit
# ---------------------------------------------------------------------------

def _integrate_symptoms(
    entries: list[dict[str, Any]],
    cfg: DecisionAuditConfig,
) -> list[Symptom]:
    out: list[Symptom] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        decision = str(entry.get("decision") or "")
        if decision not in ("KEEP", "PARTIAL"):
            continue
        out.extend(_g1_empty_patch_kept(entry))
        out.extend(_g2_decision_threshold_violated(entry, cfg))
        out.extend(_g3_kernel_dispatch_bypassed(entry, cfg))
    return out


def _g1_empty_patch_kept(entry: dict[str, Any]) -> list[Symptom]:
    """G1: integrate KEEP/PARTIAL with patch_size_bytes == 0.

    An empty patch cannot have produced any speedup; if the integrate
    decision still landed KEEP, the executor's gain measurement is
    almost certainly noise. Surface for immediate revert + KB rule.
    """
    patch_size = entry.get("patch_size_bytes")
    if not isinstance(patch_size, int) or patch_size > 0:
        return []
    kernel_id = entry.get("kernel_id") or "unknown"
    return [
        Symptom(
            name="empty_patch_kept",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"integrate decision={entry.get('decision')!r} on "
                f"kernel_id={kernel_id!r} but patch_size_bytes=0 — "
                f"no code changed, so any measured gain is noise"
            ),
            evidence={
                "kernel_id": kernel_id,
                "decision": entry.get("decision"),
                "gain_pct": entry.get("gain_pct"),
                "patch_path": entry.get("patch_path"),
                "patch_size_bytes": patch_size,
                "result_path": entry.get("result_path"),
            },
            subject={
                "kernel_id": str(kernel_id),
                "patch_path": str(entry.get("patch_path") or ""),
            },
            source="local",
            suggestion=(
                "revert this integrate decision; record a KB entry that "
                "empty patches must not be accepted regardless of "
                "noise-floor gain measurement"
            ),
        )
    ]


def _g2_decision_threshold_violated(
    entry: dict[str, Any], cfg: DecisionAuditConfig,
) -> list[Symptom]:
    """G2: KEEP decision with gain_pct below ``min_keep_gain_pct``.

    Noise-floor KEEPs (Dolphin-34B +0.07% etc.) pollute the optimization
    stack. Medium severity because this is also a *configuration* fix:
    upstream should raise its keep threshold. Robustness only flags.
    """
    if entry.get("decision") != "KEEP":
        return []
    gain_pct = entry.get("gain_pct")
    if not isinstance(gain_pct, (int, float)):
        return []
    if gain_pct >= cfg.min_keep_gain_pct:
        return []
    kernel_id = entry.get("kernel_id") or "unknown"
    return [
        Symptom(
            name="decision_threshold_violated",
            severity=SymptomSeverity.MEDIUM,
            summary=(
                f"integrate KEEP on kernel_id={kernel_id!r} with "
                f"gain_pct={gain_pct:.2f}% < min_keep_gain_pct="
                f"{cfg.min_keep_gain_pct:.1f}% — likely noise-floor"
            ),
            evidence={
                "kernel_id": kernel_id,
                "gain_pct": gain_pct,
                "min_keep_gain_pct": cfg.min_keep_gain_pct,
                "result_path": entry.get("result_path"),
            },
            subject={"kernel_id": str(kernel_id)},
            source="local",
            suggestion=(
                "raise the executor's keep threshold to >= 1% and "
                "require multi-seed confidence for sub-threshold KEEPs"
            ),
        )
    ]


def _g3_kernel_dispatch_bypassed(
    entry: dict[str, Any], cfg: DecisionAuditConfig,
) -> list[Symptom]:
    """G3: KEEP'd patch likely never executed at dispatch time.

    Two ways the executor's E2E benchmark can KEEP a no-op:

    a) ``dispatched_count == 0`` — the integrate executor recorded that
       no dispatch saw the patched kernel.
    b) ``|gain_pct|`` is within ``dispatch_bypass_pre_post_epsilon_pct``
       and ``dispatched_count`` is absent (i.e. the executor never
       emitted dispatch evidence). Suspicious — the patch may have been
       bypassed by aiter fast-path or sgl_kernel pre-compile.

    Both fire HIGH because a KEEP without proof of execution is a
    false-positive in the optimization_stack.
    """
    if entry.get("decision") != "KEEP":
        return []
    dispatched = entry.get("dispatched_count")
    gain_pct = entry.get("gain_pct")
    kernel_id = entry.get("kernel_id") or "unknown"
    if isinstance(dispatched, int) and dispatched == 0:
        reason = "dispatched_count=0"
        evidence_extra = {"dispatched_count": 0}
    elif (
        dispatched is None
        and isinstance(gain_pct, (int, float))
        and abs(gain_pct) < cfg.dispatch_bypass_pre_post_epsilon_pct
    ):
        reason = (
            "dispatch evidence missing and "
            f"|gain_pct|={abs(gain_pct):.2f}% < "
            f"{cfg.dispatch_bypass_pre_post_epsilon_pct:.2f}%"
        )
        evidence_extra = {"dispatched_count": None}
    else:
        return []
    return [
        Symptom(
            name="kernel_dispatch_bypassed",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"integrate KEEP on kernel_id={kernel_id!r} but the "
                f"patched kernel likely never executed: {reason}"
            ),
            evidence={
                "kernel_id": kernel_id,
                "gain_pct": gain_pct,
                "dispatch_bypass_pre_post_epsilon_pct": (
                    cfg.dispatch_bypass_pre_post_epsilon_pct
                ),
                "result_path": entry.get("result_path"),
                **evidence_extra,
            },
            subject={"kernel_id": str(kernel_id)},
            source="local",
            suggestion=(
                "require integrate to attach ROCprof / TraceLens "
                "dispatch_count > 0 evidence before allowing KEEP; "
                "revert the bypassed KEEP"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# G4 / G5 / G6 — ci_metrics audit (if-present only)
# ---------------------------------------------------------------------------

def _ci_metrics_symptoms(
    ci_metrics: dict[str, Any], ci_metrics_path: str,
) -> list[Symptom]:
    if not ci_metrics:
        return []
    out: list[Symptom] = []
    out.extend(_g4_negative_delta_kernel_kept(ci_metrics, ci_metrics_path))
    out.extend(_g5_baseline_zero_without_status(ci_metrics, ci_metrics_path))
    out.extend(_g6_schema_drift(ci_metrics, ci_metrics_path))
    return out


def _g4_negative_delta_kernel_kept(
    ci_metrics: dict[str, Any], ci_metrics_path: str,
) -> list[Symptom]:
    """G4: kernels_optimized > 0 AND optimized_kernel_delta_pct <= 0.

    Qwen2.5-32B-AWQ shipped ``kernels_optimized=6`` while the delta was
    ``-0.169%``. The metric advertises optimization but the actual
    contribution is negative — equivalent to "I optimized 6 things and
    made the model 0.17% slower". HIGH because downstream aggregators
    (dashboards, PR submitters) treat the field as a win count.
    """
    kernels_opt = ci_metrics.get("kernels_optimized")
    delta_pct = ci_metrics.get("optimized_kernel_delta_pct")
    if not isinstance(kernels_opt, (int, float)) or kernels_opt <= 0:
        return []
    if not isinstance(delta_pct, (int, float)) or delta_pct > 0:
        return []
    return [
        Symptom(
            name="kernel_negative_delta_kept",
            severity=SymptomSeverity.HIGH,
            summary=(
                f"ci_metrics reports kernels_optimized={int(kernels_opt)} "
                f"but optimized_kernel_delta_pct={delta_pct:.3f}% (<=0) "
                f"— kernel changes are net-negative"
            ),
            evidence={
                "kernels_optimized": kernels_opt,
                "optimized_kernel_delta_pct": delta_pct,
                "ci_metrics_path": ci_metrics_path,
            },
            subject={},
            source="local",
            suggestion=(
                "roll back kernel changes; rename the field to "
                "``kernels_kept`` (delta_pct >= +0.5%) and track "
                "``kernels_attempted_but_reverted`` separately"
            ),
        )
    ]


def _g5_baseline_zero_without_status(
    ci_metrics: dict[str, Any], ci_metrics_path: str,
) -> list[Symptom]:
    """G5: baseline_tput == 0 AND no ``status="baseline_failed"`` marker.

    Half-written ci_metrics during baseline failure looks identical to
    "no optimization space" to downstream aggregators. The fix in
    report_back is to write ``{status: "baseline_failed"}`` instead of
    zeros; Robustness flags the bad row when ANY baseline-throughput-
    style field is 0.0 without the marker.
    """
    if str(ci_metrics.get("status") or "") == "baseline_failed":
        return []
    baseline_candidates = [
        ci_metrics.get(k)
        for k in (
            "baseline_throughput",
            "baseline_tput",
            "baseline_tput_per_gpu",
            "baseline_tok_per_gpu",
        )
    ]
    baseline_values = [
        v for v in baseline_candidates
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not baseline_values:
        return []
    if any(v > 0 for v in baseline_values):
        return []
    return [
        Symptom(
            name="ci_metrics_baseline_zero",
            severity=SymptomSeverity.HIGH,
            summary=(
                "ci_metrics carries baseline throughput=0 with no "
                "``status=baseline_failed`` marker — half-written "
                "ci_metrics will be mistaken for ``no optimization "
                "space`` by downstream aggregators"
            ),
            evidence={
                "ci_metrics_path": ci_metrics_path,
                "baseline_values": baseline_values,
                "status": ci_metrics.get("status"),
            },
            subject={},
            source="local",
            suggestion=(
                "delete the partial ci_metrics file; require "
                "report_back to write {status: 'baseline_failed'} on "
                "baseline failure instead of zeros"
            ),
        )
    ]


def _g6_schema_drift(
    ci_metrics: dict[str, Any], ci_metrics_path: str,
) -> list[Symptom]:
    """G6: ci_metrics is missing required schema fields OR uses legacy
    field names. MEDIUM because the fix is in ``report_back/ci_metrics.py``;
    Robustness only surfaces the drift so the aggregator dashboard
    catches it before the data is graphed.
    """
    keys = set(ci_metrics.keys())
    missing = _CI_METRICS_REQUIRED_FIELDS - keys
    legacy = keys & _CI_METRICS_LEGACY_FIELDS
    if not missing and not legacy:
        return []
    return [
        Symptom(
            name="ci_metrics_schema_drift",
            severity=SymptomSeverity.MEDIUM,
            summary=(
                f"ci_metrics schema drift: missing={sorted(missing) or '(none)'}, "
                f"legacy_fields={sorted(legacy) or '(none)'}"
            ),
            evidence={
                "ci_metrics_path": ci_metrics_path,
                "missing": sorted(missing),
                "legacy_fields": sorted(legacy),
                "required": sorted(_CI_METRICS_REQUIRED_FIELDS),
            },
            subject={},
            source="local",
            suggestion=(
                "add pydantic schema validation to "
                "``report_back/ci_metrics.py``; refuse to write rows "
                "outside the canonical field set"
            ),
        )
    ]


# ---------------------------------------------------------------------------
# G7 — OOB optimization_attempts.jsonl audit
# ---------------------------------------------------------------------------

def _oob_symptoms(
    entries: list[dict[str, Any]],
    cfg: DecisionAuditConfig,
) -> list[Symptom]:
    """G7: OOB attempt advertises "expected speedup ~1.X" but never
    measured. Surface HIGH so downstream pipeline can refuse to count
    these towards ``kernels_optimized``.
    """
    out: list[Symptom] = []
    by_kernel: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        microbench = entry.get("microbench_speedup")
        if isinstance(microbench, (int, float)) and microbench > 0:
            continue
        report = str(entry.get("report_text") or "").lower()
        if not any(m.lower() in report for m in cfg.oob_no_harness_markers):
            continue
        kernel_id = entry.get("kernel_id") or "unknown"
        # Collapse multiple offending entries per kernel_id so we don't
        # spam the inbox with one symptom per OOB attempt.
        prior = by_kernel.get(str(kernel_id))
        if prior is None or (entry.get("ts") or "") > (prior.get("ts") or ""):
            by_kernel[str(kernel_id)] = entry
    for kernel_id, entry in by_kernel.items():
        out.append(
            Symptom(
                name="oob_no_harness",
                severity=SymptomSeverity.HIGH,
                summary=(
                    f"OOB attempt on kernel_id={kernel_id!r} advertised "
                    f"expected speedup but no microbench measurement "
                    f"was recorded"
                ),
                evidence={
                    "kernel_id": kernel_id,
                    "backend": entry.get("backend"),
                    "microbench_speedup": entry.get("microbench_speedup"),
                    "ts": entry.get("ts"),
                    "source_file": entry.get("source_file"),
                    "report_text_head": (entry.get("report_text") or "")[:160],
                },
                subject={"kernel_id": kernel_id},
                source="local",
                suggestion=(
                    "require OOB tasks to attach a microbench result "
                    "(microbench_speedup > 0); reject expected-only "
                    "reports and mark the attempt NO_HARNESS"
                ),
            )
        )
    return out


__all__ = [
    "DecisionAuditConfig",
    "evaluate_decision_audit_signals",
]
