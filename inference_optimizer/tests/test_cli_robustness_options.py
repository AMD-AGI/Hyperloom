"""Unit tests for ``inference_optimizer.cli._build_robustness_options``.

The helper translates ``argparse.Namespace`` into the ``request.options``
overrides that :class:`RobustnessAgentBackend` forwards verbatim to
``python -m robustness_agent.runtime.cli tick``. Two important
behaviours covered here:

* Single-node default — only operator-supplied overrides land in
  ``options``; nothing is auto-injected so the runtime CLI keeps its
  factory defaults (matters for hosts running sglang/vLLM/Magpie
  locally on 127.0.0.1:8888 that the auto-probe was designed for).

* Multi-node auto-disable — when ``args.nodes >= 2`` the inference
  server runs in the head pod (separate Kubernetes pod), so the
  hardcoded ``http://127.0.0.1:8888/health`` probe in the runtime
  config can never succeed inside the sandbox container and floods
  the bus with false-positive ``local_server_unreachable`` symptoms
  each tick. Repro: sandbox primus-claw-20260522020448-z6rg6 emitted
  ``local server probe http://127.0.0.1:8888/health status=error``
  every tick alongside the (also false-positive) ``gain_plateau``.
  We expect ``_build_robustness_options`` to pre-set
  ``auto_probe_inference_server=False`` so the runtime config skips
  appending the local URL to ``probe_targets``.
"""

from __future__ import annotations

import argparse

from inference_optimizer.cli import _build_robustness_options


def _ns(**kwargs) -> argparse.Namespace:
    """Build a Namespace with explicit defaults so getattr never blows up."""
    defaults: dict = {
        "nodes": 1,
        "robustness_server_url": None,
        "robustness_llm_rca": None,
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
    ``probe_targets``. ``auto_probe_auth_proxy`` is intentionally
    NOT touched (auth-proxy is local even on multi-node)."""
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
