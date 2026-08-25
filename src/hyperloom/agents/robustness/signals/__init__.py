# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Signal rules.

Each rule consumes :class:`ReactorContext` plus :class:`SourceData` and
yields zero or more :class:`Symptom` records; the classifier composes
the rules and de-duplicates by ``(name, subject_key)``.
"""

from .aiter_jit import (
    AiterJitConfig,
    AiterJitDetector,
)
from .budget import BudgetConfig, evaluate_budget_signals
from .classifier import Classifier, SignalSpec
from .conversation_progress import (
    ConversationProgressConfig,
    evaluate_conversation_progress_signals,
)
from .crash import evaluate_crash_signals
from .critic_health import (
    CriticHealthConfig,
    evaluate_critic_health_signals,
)
from .decision_audit import (
    DecisionAuditConfig,
    evaluate_decision_audit_signals,
)
from .event import evaluate_event_signals
from .event_view import EventRow, build_event_view, family_of
from .external_deps import (
    ExternalDepsConfig,
    TraceLensCliFiredOnce,
    evaluate_external_deps_signals,
)
from .gpu_leak import (
    GpuLeakConfig,
    GpuLeakDetector,
)
from .kernel_pipeline import (
    KernelPipelineConfig,
    RayPendingDetector,
    evaluate_kernel_pipeline_signals,
)
from .local_health import evaluate_local_health_signals
from .phase_budget import PhaseBudgetConfig, evaluate_phase_budget_signals
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
from .progress import (
    ProgressConfig,
    ProgressDetector,
)
from .repeated_payload import (
    RepeatedPayloadConfig,
    evaluate_repeated_payload_signals,
)
from .stall import evaluate_stall_signals
from .symptom import Symptom, SymptomSeverity

__all__ = [
    "AiterJitConfig",
    "AiterJitDetector",
    "AmdahlCeilingConfig",
    "AmdahlCeilingDetector",
    "BudgetConfig",
    "Classifier",
    "ColdStartConfig",
    "ConversationProgressConfig",
    "CriticHealthConfig",
    "DecisionAuditConfig",
    "ExternalDepsConfig",
    "GpuLeakConfig",
    "GpuLeakDetector",
    "KernelPipelineConfig",
    "ModelGpuFitConfig",
    "ModelGpuFitDetector",
    "PhaseBudgetConfig",
    "ProgressConfig",
    "ProgressDetector",
    "RayPendingDetector",
    "RepeatedPayloadConfig",
    "SignalSpec",
    "StateIntegrityConfig",
    "Symptom",
    "SymptomSeverity",
    "TraceLensCliFiredOnce",
    "EventRow",
    "build_event_view",
    "evaluate_budget_signals",
    "family_of",
    "evaluate_cold_start_signals",
    "evaluate_conversation_progress_signals",
    "evaluate_crash_signals",
    "evaluate_critic_health_signals",
    "evaluate_decision_audit_signals",
    "evaluate_event_signals",
    "evaluate_external_deps_signals",
    "evaluate_kernel_pipeline_signals",
    "evaluate_local_health_signals",
    "evaluate_phase_budget_signals",
    "evaluate_repeated_payload_signals",
    "evaluate_stall_signals",
    "evaluate_state_integrity_signals",
]
