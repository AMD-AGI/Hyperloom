"""Unit tests for J1 / J2 / J3 external-dependency signals."""

from __future__ import annotations

from robustness_agent.role.prompt_inputs import (
    ReactorContext,
    SharedStateSnapshot,
)
from robustness_agent.signals import SymptomSeverity
from robustness_agent.signals.external_deps import (
    ExternalDepsConfig,
    TraceLensCliFiredOnce,
    evaluate_external_deps_signals,
)
from robustness_agent.sources.base import SourceData


def _ctx() -> ReactorContext:
    return ReactorContext(
        tick_index=1,
        shared_state=SharedStateSnapshot(),
        inbox=[],
        now_unix=1.0,
    )


# ---------------------------------------------------------------------------
# J1 — gateway_auth_outage
# ---------------------------------------------------------------------------

def test_j1_gateway_401_fires_high():
    data = SourceData(local_external_deps={
        "gateway": {
            "url": "https://gateway/v1/models",
            "reachable": True,
            "status": "unauthorized",
            "status_code": 401,
        },
    })
    out = evaluate_external_deps_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "gateway_auth_outage")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["status_code"] == 401


def test_j1_gateway_403_also_fires():
    """Some gateways return 403 instead of 401 for revoked keys."""
    data = SourceData(local_external_deps={
        "gateway": {
            "url": "https://gateway/v1/models",
            "reachable": True,
            "status": "http_error",
            "status_code": 403,
        },
    })
    out = evaluate_external_deps_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "gateway_auth_outage")
    assert sym.evidence["status_code"] == 403


def test_j1_silent_when_gateway_ok():
    data = SourceData(local_external_deps={
        "gateway": {
            "url": "https://gateway/v1/models",
            "reachable": True,
            "status": "ok",
            "status_code": 200,
        },
    })
    out = evaluate_external_deps_signals(_ctx(), data)
    assert all(s.name != "gateway_auth_outage" for s in out)


def test_j1_silent_when_gateway_connect_error():
    """500/connect errors are NOT gateway_auth_outage."""
    data = SourceData(local_external_deps={
        "gateway": {
            "url": "https://gateway/v1/models",
            "reachable": False,
            "status": "error",
            "status_code": None,
            "error": "connect: ConnectError",
        },
    })
    out = evaluate_external_deps_signals(_ctx(), data)
    assert all(s.name != "gateway_auth_outage" for s in out)


# ---------------------------------------------------------------------------
# J2 — wekafs_degraded
# ---------------------------------------------------------------------------

def test_j2_wekafs_unreachable_fires_high():
    data = SourceData(local_external_deps={
        "mounts": [
            {"env_name": "TRACELENS_ROOT",
             "path": "/wekafs/hyperloom/TraceLens",
             "ok": False, "error": "not_found", "latency_ms": 0.1},
        ],
    })
    out = evaluate_external_deps_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "wekafs_degraded")
    assert sym.severity is SymptomSeverity.HIGH
    assert sym.evidence["env_name"] == "TRACELENS_ROOT"


def test_j2_wekafs_latency_medium_at_warn():
    data = SourceData(local_external_deps={
        "mounts": [
            {"env_name": "TRACELENS_ROOT", "path": "/x",
             "ok": True, "error": None, "latency_ms": 7000.0},
        ],
    })
    out = evaluate_external_deps_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "wekafs_degraded")
    assert sym.severity is SymptomSeverity.MEDIUM


def test_j2_wekafs_latency_high_at_critical():
    data = SourceData(local_external_deps={
        "mounts": [
            {"env_name": "TRACELENS_ROOT", "path": "/x",
             "ok": True, "error": None, "latency_ms": 20000.0},
        ],
    })
    out = evaluate_external_deps_signals(_ctx(), data)
    sym = next(s for s in out if s.name == "wekafs_degraded")
    assert sym.severity is SymptomSeverity.HIGH


def test_j2_silent_when_mount_healthy():
    data = SourceData(local_external_deps={
        "mounts": [
            {"env_name": "TRACELENS_ROOT", "path": "/x",
             "ok": True, "error": None, "latency_ms": 12.0},
        ],
    })
    out = evaluate_external_deps_signals(_ctx(), data)
    assert all(s.name != "wekafs_degraded" for s in out)


def test_j2_fires_per_mount():
    data = SourceData(local_external_deps={
        "mounts": [
            {"env_name": "TRACELENS_ROOT", "path": "/a",
             "ok": False, "error": "not_found", "latency_ms": 1.0},
            {"env_name": "INFERENCEX_PATH", "path": "/b",
             "ok": True, "error": None, "latency_ms": 20000.0},
            {"env_name": "OOB_SRC", "path": "/c",
             "ok": True, "error": None, "latency_ms": 50.0},  # healthy
        ],
    })
    out = evaluate_external_deps_signals(_ctx(), data)
    syms = [s for s in out if s.name == "wekafs_degraded"]
    # /a unreachable + /b slow, /c silent.
    assert len(syms) == 2
    paths = {s.evidence["path"] for s in syms}
    assert paths == {"/a", "/b"}


def test_j2_custom_thresholds_apply():
    cfg = ExternalDepsConfig(
        mount_latency_warn_ms=100.0,
        mount_latency_critical_ms=200.0,
    )
    data = SourceData(local_external_deps={
        "mounts": [
            {"env_name": "TRACELENS_ROOT", "path": "/x",
             "ok": True, "error": None, "latency_ms": 150.0},
        ],
    })
    out = evaluate_external_deps_signals(_ctx(), data, config=cfg)
    sym = next(s for s in out if s.name == "wekafs_degraded")
    assert sym.severity is SymptomSeverity.MEDIUM


# ---------------------------------------------------------------------------
# J3 — tracelens_cli_missing (one-shot latch)
# ---------------------------------------------------------------------------

def test_j3_fires_once_when_neither_cli_present():
    latch = TraceLensCliFiredOnce()
    data = SourceData(local_external_deps={
        "tracelens_cli": {
            "cli_names": ["a", "b"],
            "found": {"a": False, "b": False},
            "any_present": False,
        },
    })
    first = evaluate_external_deps_signals(_ctx(), data, tracelens_latch=latch)
    second = evaluate_external_deps_signals(_ctx(), data, tracelens_latch=latch)
    assert len(first) == 1
    assert first[0].name == "tracelens_cli_missing"
    assert first[0].severity is SymptomSeverity.HIGH
    # One-shot — subsequent ticks stay silent.
    assert second == []


def test_j3_silent_when_any_cli_present():
    latch = TraceLensCliFiredOnce()
    data = SourceData(local_external_deps={
        "tracelens_cli": {
            "cli_names": ["a", "b"],
            "found": {"a": True, "b": False},
            "any_present": True,
        },
    })
    out = evaluate_external_deps_signals(_ctx(), data, tracelens_latch=latch)
    assert out == []
    assert latch.value is False


def test_j3_silent_without_latch_passed():
    """When the caller doesn't supply a latch, J3 is skipped entirely."""
    data = SourceData(local_external_deps={
        "tracelens_cli": {"any_present": False, "found": {"a": False}},
    })
    out = evaluate_external_deps_signals(_ctx(), data, tracelens_latch=None)
    assert out == []


# ---------------------------------------------------------------------------
# Empty / disabled cases
# ---------------------------------------------------------------------------

def test_silent_when_no_data():
    out = evaluate_external_deps_signals(_ctx(), SourceData())
    assert out == []
