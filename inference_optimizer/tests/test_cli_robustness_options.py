"""Unit tests for ``inference_optimizer.cli`` robustness backend wiring.

Covers two helpers:

* ``_build_robustness_options`` — translates ``argparse.Namespace`` into
  the ``request.options`` overrides that :class:`RobustnessAgentBackend`
  forwards verbatim to ``python -m robustness_agent.runtime.cli tick``.

* ``_resolve_robustness_choice`` — picks ``"mock"`` vs ``"agent"`` from
  the operator flag with multi-node auto-downgrade (the agent backend's
  ``LocalProbeSource`` family targets sandbox-local resources that all
  live in separate pods on ``--nodes >= 2``, so the cleanest path is to
  fall back to the heartbeat-only mock). Repro: sandbox
  primus-claw-20260522063032-mcctl turn=0 emitted ``ray_head_dead`` HIGH
  + ``prune_branch(kernel_opt)`` + ``escalate_strategy_change`` from a
  ``ray status`` probe failing because the Ray head lives in a separate
  RayJob pod, unreachable from the sandbox.
"""

from __future__ import annotations

import argparse

from inference_optimizer.cli import (
    _build_robustness_options,
    _resolve_robustness_choice,
)


def _ns(**kwargs) -> argparse.Namespace:
    """Build a Namespace with explicit defaults so getattr never blows up."""
    defaults: dict = {
        "nodes": 1,
        "robustness_server_url": None,
        "robustness_llm_rca": None,
        "robustness_backend": None,  # CLI default; resolves to DEFAULT_*
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_single_node_default_emits_no_overrides():
    """Operator passed no flags → empty options dict so runtime CLI
    keeps its defaults (auto-probe-inference-server stays True so a
    SIGSTOP on local sglang still triggers ``local_server_unreachable``)."""
    opts = _build_robustness_options(_ns(nodes=1))
    assert opts == {}


def test_single_node_passes_server_url_and_llm_rca():
    """Existing operator-supplied flags still propagate verbatim."""
    opts = _build_robustness_options(_ns(
        nodes=1,
        robustness_server_url="http://robustness.svc:8080",
        robustness_llm_rca=True,
    ))
    assert opts == {
        "robustness_server_url": "http://robustness.svc:8080",
        "llm_rca_enabled": True,
    }


def test_multi_node_disables_inference_probe():
    """``nodes >= 2`` → auto-probe-inference-server gets False so the
    runtime CLI does NOT append ``http://127.0.0.1:8888/health`` to
    ``probe_targets``."""
    opts = _build_robustness_options(_ns(nodes=2))
    assert opts == {
        "auto_probe_inference_server": False,
        "progress_no_levers_min_minutes": 60.0,
    }


def test_multi_node_bumps_no_levers_floor_to_60_minutes():
    """``nodes >= 2`` → progress_no_levers_min_minutes=60.0 layers a
    wall-clock buffer on top of the explore_started gate so multi-
    node + large-model setups (sglang cold start 10-15 min +
    baseline + profile + turnaround = 35-50 min before first
    explore family runs) do not trip the symptom prematurely.
    Single-node MUST keep the 45.0 default; we assert the key is
    absent there so runtime CLI falls back to its config default."""
    multi = _build_robustness_options(_ns(nodes=2))
    assert multi["progress_no_levers_min_minutes"] == 60.0
    single = _build_robustness_options(_ns(nodes=1))
    assert "progress_no_levers_min_minutes" not in single


def test_multi_node_preserves_operator_flags():
    """Multi-node auto-disable must coexist with explicit operator
    flags; both keys land in ``options`` independently."""
    opts = _build_robustness_options(_ns(
        nodes=4,
        robustness_server_url="http://robustness.svc:8080",
    ))
    assert opts == {
        "robustness_server_url": "http://robustness.svc:8080",
        "auto_probe_inference_server": False,
        "progress_no_levers_min_minutes": 60.0,
    }


def test_missing_nodes_attr_treated_as_single_node():
    """Legacy entry points that build a Namespace without ``nodes`` at
    all must not crash and must default to single-node semantics
    (no auto-disable, no no_levers floor bump)."""
    ns = argparse.Namespace(
        robustness_server_url=None,
        robustness_llm_rca=None,
    )
    opts = _build_robustness_options(ns)
    assert "auto_probe_inference_server" not in opts
    assert "progress_no_levers_min_minutes" not in opts


def test_nodes_zero_or_none_treated_as_single_node():
    """``nodes=0`` and ``nodes=None`` are both nonsensical inputs in
    practice but must safely degrade to single-node semantics
    rather than triggering the auto-disable path or the
    no_levers floor bump."""
    for opts in (
        _build_robustness_options(_ns(nodes=0)),
        _build_robustness_options(_ns(nodes=None)),
    ):
        assert "auto_probe_inference_server" not in opts
        assert "progress_no_levers_min_minutes" not in opts


# ---------------------------------------------------------------------------
# _resolve_robustness_choice — multi-node auto-downgrade to mock
# ---------------------------------------------------------------------------

def test_resolve_choice_single_node_default_keeps_agent():
    """Default path on single-node must stay ``"agent"`` so the
    real LocalProbe coverage is preserved on hosts where the
    inference server / ray actually live in the sandbox container."""
    ns = _ns(nodes=1, robustness_backend=None)
    assert _resolve_robustness_choice(ns) == "agent"


def test_resolve_choice_single_node_explicit_mock_kept():
    """Explicit ``--robustness-mock`` on single-node passes through."""
    ns = _ns(nodes=1, robustness_backend="mock")
    assert _resolve_robustness_choice(ns) == "mock"


def test_resolve_choice_multi_node_default_downgrades_to_mock(capsys):
    """``args.nodes >= 2`` with the default agent choice → mock,
    silently (the implicit default does not warrant a WARNING because
    the operator did not actively ask for the agent backend)."""
    ns = _ns(nodes=2, robustness_backend=None)
    chosen = _resolve_robustness_choice(ns)
    assert chosen == "mock"
    captured = capsys.readouterr()
    assert "WARN" not in captured.err
    assert "WARN" not in captured.out


def test_resolve_choice_multi_node_explicit_agent_downgrades_with_warning(capsys):
    """``args.nodes >= 2`` with ``--robustness-agent`` explicitly → mock
    with a WARNING on stderr that points operators at the multi-node
    SKILL section so they can read the rationale (LocalProbe family
    targets sandbox-local resources only; multi-node has every
    target in a separate pod)."""
    ns = _ns(nodes=2, robustness_backend="agent")
    chosen = _resolve_robustness_choice(ns)
    assert chosen == "mock"
    captured = capsys.readouterr()
    assert "WARN" in captured.err
    assert "auto-downgrad" in captured.err.lower()
    assert "multi_node/SKILL.md" in captured.err


def test_resolve_choice_multi_node_explicit_mock_no_warning(capsys):
    """Operators who anticipate the auto-downgrade and pass
    ``--robustness-mock`` explicitly must NOT see the WARNING (they
    already know)."""
    ns = _ns(nodes=4, robustness_backend="mock")
    chosen = _resolve_robustness_choice(ns)
    assert chosen == "mock"
    captured = capsys.readouterr()
    assert "WARN" not in captured.err


def test_resolve_choice_missing_nodes_attr_treated_as_single_node():
    """Legacy entry points that omit ``nodes`` keep the agent default
    rather than auto-downgrading (single-node semantics)."""
    ns = argparse.Namespace(robustness_backend=None)
    assert _resolve_robustness_choice(ns) == "agent"


def test_resolve_choice_nodes_zero_or_none_treated_as_single_node():
    """``nodes=0`` / ``nodes=None`` must NOT trip the multi-node
    downgrade — they degrade to single-node semantics."""
    for ns in (
        _ns(nodes=0, robustness_backend="agent"),
        _ns(nodes=None, robustness_backend="agent"),
    ):
        assert _resolve_robustness_choice(ns) == "agent"
