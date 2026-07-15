# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Run all signal rules and de-duplicate Symptoms.

A single :data:`_SIGNAL_REGISTRY` declares every rule in evaluation order: its
config slot, whether it is a pure evaluator or a stateful detector, the
state-store slot for detectors, and any cross-entry kwargs. :meth:`Classifier.classify`
walks the registry in order, then folds duplicates via ``Symptom.dedup_key()``
(highest severity wins on ties). The registry is the single source both the
classifier and :func:`factory.build_reactor_components` read, so the
Config->SignalConfig wiring cannot drift from the run order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

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

# Slot name for the configless TraceLens CLI latch (sibling of external_deps).
_TRACELENS_STATE_KEY: str = "tracelens_cli_latch"


@dataclass(frozen=True)
class SignalSpec:
    """One row of the signal registry — a rule in the classifier pipeline.

    Exactly one of ``evaluator`` / ``detector_cls`` is set:

    * ``evaluator`` — a pure module-level ``evaluate_*`` function called as
      ``evaluator(ctx, [data,] config=cfg, **extra)``.
    * ``detector_cls`` — a stateful detector constructed once with its config
      and a per-slot state view, then driven via ``.evaluate(ctx, data)``.
    """

    name: str
    config_attr: str | None
    config_factory: Callable[[], Any] | None
    evaluator: SignalEvaluator | None = None
    detector_cls: type | None = None
    state_view_key: str | None = None
    # Whether the stateless evaluator takes ``SourceData`` positionally.
    # ``evaluate_budget_signals(ctx, *, config)`` does NOT — it is pure-context.
    needs_source_data: bool = True
    # Produce extra kwargs for the evaluator from the live classifier, used for
    # cross-entry injection (e.g. external_deps borrows the TraceLens latch).
    extra_kwargs_factory: "Callable[[Classifier], dict[str, Any]] | None" = None


def _external_deps_extra_kwargs(classifier: "Classifier") -> dict[str, Any]:
    """Inject the shared TraceLens CLI latch into ``evaluate_external_deps_signals``.

    The latch is a configless one-shot helper owned by the classifier (so J3
    fires at most once per session); it is a sibling of the external_deps row
    rather than its own signal.

    Args:
        classifier (Classifier): The live classifier holding the built latch.

    Returns:
        dict[str, Any]: The ``tracelens_latch`` kwarg for the evaluator.
    """
    return {"tracelens_latch": classifier._tracelens_latch}  # noqa: SLF001 — same-module owner


# The single ordered registry. ORDER IS PART OF THE CONTRACT: ``classify``
# appends in this order and ``_dedup`` keeps the first-inserted symptom on an
# equal-severity tie. ``kernel_pipeline`` intentionally appears via two rows
# (stateful RayPendingDetector + stateless evaluator) sharing one config slot.
_SIGNAL_REGISTRY: tuple[SignalSpec, ...] = (
    SignalSpec("stall", "stall", StallConfig, evaluator=evaluate_stall_signals),
    SignalSpec("crash", "crash", CrashConfig, evaluator=evaluate_crash_signals),
    SignalSpec("event", "event", EventConfig, evaluator=evaluate_event_signals),
    SignalSpec("health", "health", HealthConfig, evaluator=evaluate_health_signals),
    SignalSpec(
        "local_health",
        "local_health",
        LocalHealthConfig,
        evaluator=evaluate_local_health_signals,
    ),
    SignalSpec(
        "gpu_leak",
        "gpu_leak",
        GpuLeakConfig,
        detector_cls=GpuLeakDetector,
        state_view_key="gpu_leak",
    ),
    SignalSpec(
        "cluster_fault",
        "cluster_fault",
        ClusterFaultConfig,
        evaluator=evaluate_cluster_fault_signals,
    ),
    SignalSpec(
        "budget",
        "budget",
        BudgetConfig,
        evaluator=evaluate_budget_signals,
        needs_source_data=False,
    ),
    SignalSpec(
        "aiter_jit",
        "aiter_jit",
        AiterJitConfig,
        detector_cls=AiterJitDetector,
        state_view_key="aiter_jit",
    ),
    SignalSpec(
        "progress",
        "progress",
        ProgressConfig,
        detector_cls=ProgressDetector,
        state_view_key="progress",
    ),
    SignalSpec(
        "repeated_payload",
        "repeated_payload",
        RepeatedPayloadConfig,
        evaluator=evaluate_repeated_payload_signals,
    ),
    SignalSpec(
        "decision_audit",
        "decision_audit",
        DecisionAuditConfig,
        evaluator=evaluate_decision_audit_signals,
    ),
    SignalSpec(
        "model_gpu_fit",
        "model_gpu_fit",
        ModelGpuFitConfig,
        detector_cls=ModelGpuFitDetector,
        state_view_key="preflight_model_gpu_fit",
    ),
    SignalSpec(
        "amdahl_ceiling",
        "amdahl_ceiling",
        AmdahlCeilingConfig,
        detector_cls=AmdahlCeilingDetector,
        state_view_key="preflight_amdahl",
    ),
    SignalSpec(
        "cold_start",
        "cold_start",
        ColdStartConfig,
        evaluator=evaluate_cold_start_signals,
    ),
    SignalSpec(
        "critic_health",
        "critic_health",
        CriticHealthConfig,
        evaluator=evaluate_critic_health_signals,
    ),
    # F1 ray-pending is stateful; F2/F4/F5 live in the module helper — both
    # driven off one KernelPipelineConfig slot.
    SignalSpec(
        "ray_pending",
        "kernel_pipeline",
        KernelPipelineConfig,
        detector_cls=RayPendingDetector,
        state_view_key="ray_pending",
    ),
    SignalSpec(
        "kernel_pipeline",
        "kernel_pipeline",
        KernelPipelineConfig,
        evaluator=evaluate_kernel_pipeline_signals,
    ),
    SignalSpec(
        "state_integrity",
        "state_integrity",
        StateIntegrityConfig,
        evaluator=evaluate_state_integrity_signals,
    ),
    # TraceLens CLI latch is owned by the classifier and injected here so J3
    # fires at most once per session.
    SignalSpec(
        "external_deps",
        "external_deps",
        ExternalDepsConfig,
        evaluator=evaluate_external_deps_signals,
        extra_kwargs_factory=_external_deps_extra_kwargs,
    ),
)


def signal_registry_config_attrs() -> tuple[str, ...]:
    """Return the distinct config slot names in registry order.

    Used by :func:`factory.build_reactor_components` (and its tests) to prove
    the Config->SignalConfig map covers every registry entry.

    Returns:
        tuple[str, ...]: Unique ``config_attr`` values, first-seen order.
    """
    seen: list[str] = []
    for spec in _SIGNAL_REGISTRY:
        if spec.config_attr and spec.config_attr not in seen:
            seen.append(spec.config_attr)
    return tuple(seen)


@dataclass
class Classifier:
    """Compose the configured signal evaluators via :data:`_SIGNAL_REGISTRY`.

    Rather than 19 typed config fields, the classifier takes a single
    ``configs`` map keyed by ``SignalSpec.config_attr``; any slot the caller
    omits falls back to the registry's default ``config_factory``. Stateful
    rules are built once in :meth:`__post_init__` with their persistence views.
    """

    # Config slot overrides keyed by ``SignalSpec.config_attr``; omitted slots
    # use the registry default. Built by ``factory.build_reactor_components``.
    configs: Mapping[str, Any] = field(default_factory=dict)
    extra_evaluators: list[SignalEvaluator] = field(default_factory=list)
    # Cross-tick persistence for stateful sub-detectors so they survive
    # subprocess restarts; ``None`` keeps everything in-memory only.
    state_store: "DetectorStateStore | None" = None
    _configs: dict[str, Any] = field(init=False, repr=False)
    _detectors: dict[str, Any] = field(init=False, repr=False)
    _tracelens_latch: TraceLensCliFiredOnce = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the config map and construct the stateful sub-detectors.

        Each config slot resolves to the caller override or the registry
        default. Each detector receives a per-name state view from
        ``state_store`` (or ``None`` for in-memory operation).
        """
        store = self.state_store
        self._configs = {}
        for spec in _SIGNAL_REGISTRY:
            attr = spec.config_attr
            if attr is None or attr in self._configs or spec.config_factory is None:
                continue
            override = self.configs.get(attr) if self.configs else None
            self._configs[attr] = override if override is not None else spec.config_factory()

        self._detectors = {}
        for spec in _SIGNAL_REGISTRY:
            if spec.detector_cls is None:
                continue
            cfg = self._configs.get(spec.config_attr) if spec.config_attr else None
            view = store.view(spec.state_view_key) if (store is not None and spec.state_view_key) else None
            self._detectors[spec.name] = spec.detector_cls(cfg, state_view=view)

        # Configless one-shot latch injected into external_deps (sibling row).
        self._tracelens_latch = TraceLensCliFiredOnce(
            state_view=store.view(_TRACELENS_STATE_KEY) if store is not None else None,
        )

    @property
    def signal_configs(self) -> dict[str, Any]:
        """Resolved config slot map (override-or-default per registry entry).

        Returns:
            dict[str, Any]: A copy of the resolved ``config_attr -> config``
                map; covers every distinct registry config slot.
        """
        return dict(self._configs)

    def classify(self, data: SourceData, ctx: ReactorContext) -> list[Symptom]:
        """Run every registered signal in order and return de-duplicated symptoms.

        Walks :data:`_SIGNAL_REGISTRY` (stateful detectors + pure evaluators),
        appends any ``extra_evaluators``, then folds duplicates via
        :func:`_dedup`.

        Args:
            data (SourceData): Collected source data for this tick.
            ctx (ReactorContext): Reactor context for this tick.

        Returns:
            list[Symptom]: De-duplicated symptoms ordered by severity (HIGH
                first) then name and subject.
        """
        symptoms: list[Symptom] = []
        for spec in _SIGNAL_REGISTRY:
            symptoms.extend(self._run_spec(spec, ctx, data))
        for fn in self.extra_evaluators:
            symptoms.extend(fn(ctx, data))
        return _dedup(symptoms)

    def _run_spec(
        self,
        spec: SignalSpec,
        ctx: ReactorContext,
        data: SourceData,
    ) -> list[Symptom]:
        """Invoke one registry entry and return its symptoms.

        Args:
            spec (SignalSpec): The registry row to run.
            ctx (ReactorContext): Reactor context for the current tick.
            data (SourceData): Collected source data for the current tick.

        Returns:
            list[Symptom]: Symptoms emitted by the detector / evaluator.
        """
        if spec.detector_cls is not None:
            return self._detectors[spec.name].evaluate(ctx, data)
        evaluator = spec.evaluator
        assert evaluator is not None  # registry invariant: evaluator xor detector
        kwargs: dict[str, Any] = {}
        cfg = self._configs.get(spec.config_attr) if spec.config_attr else None
        if cfg is not None:
            kwargs["config"] = cfg
        if spec.extra_kwargs_factory is not None:
            kwargs.update(spec.extra_kwargs_factory(self))
        if spec.needs_source_data:
            return evaluator(ctx, data, **kwargs)
        return evaluator(ctx, **kwargs)


def _dedup(symptoms: list[Symptom]) -> list[Symptom]:
    """Collapse symptoms sharing a dedup key, keeping the highest severity.

    Args:
        symptoms (list[Symptom]): Raw symptoms from all evaluators.

    Returns:
        list[Symptom]: De-duplicated symptoms sorted by descending severity,
            then name, then subject.
    """
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


__all__ = [
    "Classifier",
    "SignalEvaluator",
    "SignalSpec",
    "signal_registry_config_attrs",
]
