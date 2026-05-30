"""Unit tests for ``signals/cluster_fault.py`` (M2)."""

from __future__ import annotations


from hyperloom.agents.robustness.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from hyperloom.agents.robustness.signals import SymptomSeverity
from hyperloom.agents.robustness.signals.cluster_fault import (
    ClusterFaultConfig,
    evaluate_cluster_fault_signals,
)
from hyperloom.agents.robustness.sources.base import SourceData


def _ctx() -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(session_id="sess-1"),
        inbox=[],
        now_unix=1_700_000_000.0,
    )


def _fault(
    *,
    phase: str = "Isolating",
    name: str = "g53-gpu_ecc",
    node: str = "g53",
    monitor_id: str = "gpu_ecc",
    affected_workloads: int = 1,
    affected_gpus: int = 1,
    auto_repair: bool = True,
) -> dict:
    return {
        "name": name,
        "monitor_id": monitor_id,
        "node_name": node,
        "phase": phase,
        "auto_repair": auto_repair,
        "affected_workload_count": affected_workloads,
        "affected_gpu_count": affected_gpus,
        "action": "isolate",
        "created_at": "2026-05-09T12:00:00Z",
    }


def test_no_faults_yields_no_symptoms():
    out = evaluate_cluster_fault_signals(_ctx(), SourceData(cluster_faults=[]))
    assert out == []


def test_succeeded_phase_is_silent():
    """Auto-repair completed -> no agent action expected."""

    data = SourceData(cluster_faults=[_fault(phase="Succeeded")])
    out = evaluate_cluster_fault_signals(_ctx(), data)
    assert out == []


def test_isolating_low_blast_radius_is_medium():
    data = SourceData(
        cluster_faults=[
            _fault(phase="Isolating", affected_workloads=1, affected_gpus=2)
        ]
    )
    out = evaluate_cluster_fault_signals(_ctx(), data)
    assert len(out) == 1
    assert out[0].name == "cluster_fault"
    assert out[0].severity is SymptomSeverity.MEDIUM
    assert out[0].source == "server"
    assert out[0].subject == {"node": "g53", "fault": "g53-gpu_ecc"}
    assert out[0].evidence["phase"] == "Isolating"
    assert out[0].evidence["affected_workload_count"] == 1


def test_failed_phase_is_high_regardless_of_blast_radius():
    data = SourceData(
        cluster_faults=[
            _fault(phase="Failed", affected_workloads=0, affected_gpus=0)
        ]
    )
    out = evaluate_cluster_fault_signals(_ctx(), data)
    assert len(out) == 1
    assert out[0].severity is SymptomSeverity.HIGH
    assert "auto-repair failed" in out[0].suggestion.lower()


def test_isolating_promotes_to_high_on_workload_threshold():
    cfg = ClusterFaultConfig(high_workload_threshold=4)
    data = SourceData(
        cluster_faults=[
            _fault(phase="Isolating", affected_workloads=4, affected_gpus=1)
        ]
    )
    out = evaluate_cluster_fault_signals(_ctx(), data, config=cfg)
    assert out[0].severity is SymptomSeverity.HIGH


def test_isolating_promotes_to_high_on_gpu_threshold():
    cfg = ClusterFaultConfig(high_gpu_threshold=8)
    data = SourceData(
        cluster_faults=[
            _fault(phase="Isolating", affected_workloads=1, affected_gpus=8)
        ]
    )
    out = evaluate_cluster_fault_signals(_ctx(), data, config=cfg)
    assert out[0].severity is SymptomSeverity.HIGH


def test_unknown_phase_is_ignored():
    """Future phases the upstream might add should not throw."""

    data = SourceData(cluster_faults=[_fault(phase="DraftAdded")])
    out = evaluate_cluster_fault_signals(_ctx(), data)
    assert out == []


def test_non_dict_entries_are_skipped():
    data = SourceData(cluster_faults=["not a dict", None, _fault()])  # type: ignore[list-item]
    out = evaluate_cluster_fault_signals(_ctx(), data)
    assert len(out) == 1
    assert out[0].evidence["fault_name"] == "g53-gpu_ecc"


def test_string_counts_are_coerced():
    """robust-api emits ints, but be defensive against str-typed envs."""

    fault = _fault(phase="Isolating", affected_gpus=1)
    fault["affected_workload_count"] = "5"  # promote to high via string
    data = SourceData(cluster_faults=[fault])
    out = evaluate_cluster_fault_signals(
        _ctx(), data, config=ClusterFaultConfig(high_workload_threshold=4)
    )
    assert len(out) == 1
    assert out[0].severity is SymptomSeverity.HIGH


def test_multiple_faults_each_yield_one_symptom():
    data = SourceData(
        cluster_faults=[
            _fault(name="g53-gpu_ecc", node="g53", phase="Isolating"),
            _fault(name="g54-net_drop", node="g54", phase="Failed"),
        ]
    )
    out = evaluate_cluster_fault_signals(_ctx(), data)
    assert len(out) == 2
    keys = {s.subject["fault"] for s in out}
    assert keys == {"g53-gpu_ecc", "g54-net_drop"}


def test_classifier_includes_cluster_fault_rule():
    """Make sure the rule is wired into the central classifier."""

    from robustness_agent.signals import Classifier

    clf = Classifier()
    data = SourceData(
        cluster_faults=[
            _fault(phase="Failed", affected_workloads=0, affected_gpus=0)
        ]
    )
    out = clf.classify(data, _ctx())
    names = [s.name for s in out]
    assert "cluster_fault" in names
