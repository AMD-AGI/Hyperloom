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
from .cluster_fault import ClusterFaultConfig, evaluate_cluster_fault_signals
from .crash import CrashConfig, evaluate_crash_signals
from .event import EventConfig, evaluate_event_signals
from .gpu_leak import GpuLeakConfig, GpuLeakDetector
from .health import HealthConfig, evaluate_health_signals
from .local_health import LocalHealthConfig, evaluate_local_health_signals
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
    extra_evaluators: list[SignalEvaluator] = field(default_factory=list)
    _gpu_leak_detector: GpuLeakDetector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._gpu_leak_detector = GpuLeakDetector(self.gpu_leak_config)

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
