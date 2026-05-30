"""Run all signal rules and de-duplicate Symptoms.

The classifier is intentionally dumb: it iterates a fixed list of
evaluators and appends whatever they emit. De-duplication uses
``Symptom.dedup_key()`` so the same rule firing on the same subject
twice (e.g. via inbox + coordinator events) folds into a single record;
the highest severity wins on ties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..role.prompt_inputs import ReactorContext
from ..sources.base import SourceData
from ..state_store import DetectorStateStore
from .aiter_jit import AiterJitConfig, AiterJitDetector
from .budget import BudgetConfig, evaluate_budget_signals
from .cluster_fault import ClusterFaultConfig, evaluate_cluster_fault_signals
from .crash import CrashConfig, evaluate_crash_signals
from .critic_health import (
    CriticHealthConfig,
    evaluate_critic_health_signals,
)
from .decision_audit import (
    DecisionAuditConfig,
    evaluate_decision_audit_signals,
)
from .event import EventConfig, evaluate_event_signals
from .external_deps import (
    ExternalDepsConfig,
    TraceLensCliFiredOnce,
    evaluate_external_deps_signals,
)
from .gpu_leak import GpuLeakConfig, GpuLeakDetector
from .health import HealthConfig, evaluate_health_signals
from .kernel_pipeline import (
    KernelPipelineConfig,
    RayPendingDetector,
    evaluate_kernel_pipeline_signals,
)
from .local_health import LocalHealthConfig, evaluate_local_health_signals
from .preflight import (
    AmdahlCeilingConfig,
    AmdahlCeilingDetector,
    ColdStartConfig,
    ModelGpuFitConfig,
    ModelGpuFitDetector,
    evaluate_cold_start_signals,
)
from .state_integrity import (
    StateIntegrityConfig,
    evaluate_state_integrity_signals,
)
from .progress import ProgressConfig, ProgressDetector
from .repeated_payload import (
    RepeatedPayloadConfig,
    evaluate_repeated_payload_signals,
)
from .stall import StallConfig, evaluate_stall_signals
from .symptom import Symptom


SignalEvaluator = Callable[[ReactorContext, SourceData], list[Symptom]]


@dataclass
class Classifier:
    """Compose the configured signal evaluators.

    Most evaluators are pure functions; the ``gpu_leak`` rule is
    stateful (it counts consecutive hits to suppress false positives
    from cold-start VRAM ramps) and therefore lives behind a
    :class:`GpuLeakDetector` instance constructed in
    :meth:`__post_init__`.
    """

    stall_config: StallConfig = field(default_factory=StallConfig)
    crash_config: CrashConfig = field(default_factory=CrashConfig)
    event_config: EventConfig = field(default_factory=EventConfig)
    health_config: HealthConfig = field(default_factory=HealthConfig)
    local_health_config: LocalHealthConfig = field(default_factory=LocalHealthConfig)
    cluster_fault_config: ClusterFaultConfig = field(
        default_factory=ClusterFaultConfig
    )
    gpu_leak_config: GpuLeakConfig = field(default_factory=GpuLeakConfig)
    budget_config: BudgetConfig = field(default_factory=BudgetConfig)
    aiter_jit_config: AiterJitConfig = field(default_factory=AiterJitConfig)
    progress_config: ProgressConfig = field(default_factory=ProgressConfig)
    repeated_payload_config: RepeatedPayloadConfig = field(
        default_factory=RepeatedPayloadConfig
    )
    decision_audit_config: DecisionAuditConfig = field(
        default_factory=DecisionAuditConfig
    )
    model_gpu_fit_config: ModelGpuFitConfig = field(
        default_factory=ModelGpuFitConfig
    )
    amdahl_ceiling_config: AmdahlCeilingConfig = field(
        default_factory=AmdahlCeilingConfig
    )
    cold_start_config: ColdStartConfig = field(default_factory=ColdStartConfig)
    critic_health_config: CriticHealthConfig = field(
        default_factory=CriticHealthConfig
    )
    kernel_pipeline_config: KernelPipelineConfig = field(
        default_factory=KernelPipelineConfig
    )
    state_integrity_config: StateIntegrityConfig = field(
        default_factory=StateIntegrityConfig
    )
    external_deps_config: ExternalDepsConfig = field(
        default_factory=ExternalDepsConfig
    )
    extra_evaluators: list[SignalEvaluator] = field(default_factory=list)
    # Cross-tick persistence for stateful sub-detectors. When the
    # caller wires this in (factory always does), each detector below
    # receives a slot view it uses to survive subprocess restarts.
    # ``None`` (e.g. ad-hoc tests) keeps everything in-memory only.
    state_store: "DetectorStateStore | None" = None
    _gpu_leak_detector: GpuLeakDetector = field(init=False, repr=False)
    _aiter_jit_detector: AiterJitDetector = field(init=False, repr=False)
    _progress_detector: ProgressDetector = field(init=False, repr=False)
    _model_gpu_fit_detector: ModelGpuFitDetector = field(init=False, repr=False)
    _amdahl_ceiling_detector: AmdahlCeilingDetector = field(
        init=False, repr=False
    )
    _ray_pending_detector: RayPendingDetector = field(init=False, repr=False)
    _tracelens_cli_latch: TraceLensCliFiredOnce = field(init=False, repr=False)

    def __post_init__(self) -> None:
        store = self.state_store
        self._gpu_leak_detector = GpuLeakDetector(
            self.gpu_leak_config,
            state_view=store.view("gpu_leak") if store else None,
        )
        self._aiter_jit_detector = AiterJitDetector(
            self.aiter_jit_config,
            state_view=store.view("aiter_jit") if store else None,
        )
        self._progress_detector = ProgressDetector(
            self.progress_config,
            state_view=store.view("progress") if store else None,
        )
        self._model_gpu_fit_detector = ModelGpuFitDetector(
            self.model_gpu_fit_config,
            state_view=store.view("preflight_model_gpu_fit") if store else None,
        )
        self._amdahl_ceiling_detector = AmdahlCeilingDetector(
            self.amdahl_ceiling_config,
            state_view=store.view("preflight_amdahl") if store else None,
        )
        self._ray_pending_detector = RayPendingDetector(
            self.kernel_pipeline_config,
            state_view=store.view("ray_pending") if store else None,
        )
        self._tracelens_cli_latch = TraceLensCliFiredOnce(
            state_view=store.view("tracelens_cli_latch") if store else None,
        )

    def classify(self, data: SourceData, ctx: ReactorContext) -> list[Symptom]:
        symptoms: list[Symptom] = []
        symptoms.extend(
            evaluate_stall_signals(ctx, data, config=self.stall_config)
        )
        symptoms.extend(
            evaluate_crash_signals(ctx, data, config=self.crash_config)
        )
        symptoms.extend(
            evaluate_event_signals(ctx, data, config=self.event_config)
        )
        symptoms.extend(
            evaluate_health_signals(ctx, data, config=self.health_config)
        )
        symptoms.extend(
            evaluate_local_health_signals(ctx, data, config=self.local_health_config)
        )
        symptoms.extend(self._gpu_leak_detector.evaluate(ctx, data))
        symptoms.extend(
            evaluate_cluster_fault_signals(
                ctx, data, config=self.cluster_fault_config
            )
        )
        # Budget signals are pure-context (the SharedState time-budget
        # block is rendered by Coordinator and parsed into ctx); they
        # do not need a SourceData fetch so we evaluate them last.
        symptoms.extend(
            evaluate_budget_signals(ctx, config=self.budget_config)
        )
        # Stateful detectors (aiter cache regression, gain plateau).
        symptoms.extend(self._aiter_jit_detector.evaluate(ctx, data))
        symptoms.extend(self._progress_detector.evaluate(ctx, data))
        # B1 same-fingerprint retry detector — stateless; reads
        # ``coordinator_events`` / inbox to walk the streak.
        symptoms.extend(
            evaluate_repeated_payload_signals(
                ctx, data, config=self.repeated_payload_config,
            )
        )
        # G decision-audit signals — stateless; reads the persisted
        # decision artefacts ``LocalProbe._sample_decision_audit``
        # collected into :attr:`SourceData.local_decision_audit`.
        symptoms.extend(
            evaluate_decision_audit_signals(
                ctx, data, config=self.decision_audit_config,
            )
        )
        # C preflight signals — model-GPU fit (boot-time, once per
        # manifest fingerprint), Amdahl kernel-ceiling (post-profile),
        # cold-start vs budget (every tick when cold).
        symptoms.extend(self._model_gpu_fit_detector.evaluate(ctx, data))
        symptoms.extend(self._amdahl_ceiling_detector.evaluate(ctx, data))
        symptoms.extend(
            evaluate_cold_start_signals(
                ctx, data, config=self.cold_start_config,
            )
        )
        # E critic health (KB outage / unavailable streak / prune stuck /
        # runtime stuck).
        symptoms.extend(
            evaluate_critic_health_signals(
                ctx, data, config=self.critic_health_config,
            )
        )
        # F kernel pipeline + external backend health. The stateless
        # rules (F2/F3/F4/F5) live in the module helper; the stateful
        # F1 ray-pending detector tracks consecutive-tick streaks.
        symptoms.extend(self._ray_pending_detector.evaluate(ctx, data))
        symptoms.extend(
            evaluate_kernel_pipeline_signals(
                ctx, data, config=self.kernel_pipeline_config,
            )
        )
        # I state integrity (state.json / WAL / leases / agent files /
        # Coordinator PID). Stateless — each tick re-evaluates the
        # current filesystem snapshot.
        symptoms.extend(
            evaluate_state_integrity_signals(
                ctx, data, config=self.state_integrity_config,
            )
        )
        # J external dependencies (gateway / WekaFS / TraceLens CLI).
        # The TraceLens CLI latch is owned here so the detector fires
        # at most once per session.
        symptoms.extend(
            evaluate_external_deps_signals(
                ctx, data,
                config=self.external_deps_config,
                tracelens_latch=self._tracelens_cli_latch,
            )
        )
        for fn in self.extra_evaluators:
            symptoms.extend(fn(ctx, data))
        return _dedup(symptoms)


def _dedup(symptoms: list[Symptom]) -> list[Symptom]:
    by_key: dict[tuple[str, ...], Symptom] = {}
    for sym in symptoms:
        key = sym.dedup_key()
        existing = by_key.get(key)
        if existing is None or sym.severity.rank > existing.severity.rank:
            by_key[key] = sym
    # Stable order: HIGH first, then MEDIUM, then LOW; tie-break by name.
    return sorted(
        by_key.values(),
        key=lambda s: (-s.severity.rank, s.name, sorted(s.subject.items())),
    )


__all__ = ["Classifier", "SignalEvaluator"]
