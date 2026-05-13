"""Signal rules.

Each rule consumes :class:`ReactorContext` plus :class:`SourceData` and
yields zero or more :class:`Symptom` records. The classifier composes
the rules and de-duplicates by ``(name, subject_key)``.

M1 implements four rules:

* :func:`evaluate_stall_signals` — agent reactor went silent
* :func:`evaluate_crash_signals` — repeated session crashes
* :func:`evaluate_event_signals` — policy_denied / delegated_result
  patterns from the inbox and Coordinator events
* :func:`evaluate_health_signals` — pod-level phase failures from the
  robustness-server snapshot

The full GPU / disk / log rule set lands in M2.
"""

from .classifier import Classifier
from .cluster_fault import evaluate_cluster_fault_signals
from .crash import evaluate_crash_signals
from .event import evaluate_event_signals
from .health import evaluate_health_signals
from .local_health import evaluate_local_health_signals
from .stall import evaluate_stall_signals
from .symptom import Symptom, SymptomSeverity

__all__ = [
    "Classifier",
    "Symptom",
    "SymptomSeverity",
    "evaluate_cluster_fault_signals",
    "evaluate_crash_signals",
    "evaluate_event_signals",
    "evaluate_health_signals",
    "evaluate_local_health_signals",
    "evaluate_stall_signals",
]
