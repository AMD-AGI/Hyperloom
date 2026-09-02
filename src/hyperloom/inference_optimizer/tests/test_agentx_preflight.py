# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX preflight: AIPERF_BIN resolution + capability (weka-trace) check.

Contract:
- ``resolve_aiperf_bin`` prefers ``AIPERF_BIN`` env, else PATH lookup, else None.
- ``check_aiperf_capability`` raises ``AgentXPreflightError`` when the binary is
  missing OR lacks the AgentX (weka-trace) capability. It verifies *capability*,
  not mere existence. The probe is injectable so the check is testable offline.
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.agentx.preflight import (
    AgentXPreflightError,
    check_aiperf_capability,
    resolve_aiperf_bin,
)


def test_resolve_prefers_env():
    assert resolve_aiperf_bin({"AIPERF_BIN": "/venv/bin/aiperf"}) == "/venv/bin/aiperf"


def test_resolve_none_when_absent(monkeypatch):
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.agentx.preflight.shutil.which",
        lambda _n, path=None: None,
    )
    assert resolve_aiperf_bin({}) is None


def test_resolve_path_lookup_returns_which(monkeypatch):
    seen = {}

    def _which(name, path=None):
        seen["name"] = name
        seen["path"] = path
        return "/opt/venv/bin/aiperf"

    monkeypatch.setattr("hyperloom.inference_optimizer.agentx.preflight.shutil.which", _which)
    # No AIPERF_BIN override -> falls back to which(), honoring the passed env PATH.
    assert resolve_aiperf_bin({"PATH": "/opt/venv/bin"}) == "/opt/venv/bin/aiperf"
    assert seen == {"name": "aiperf", "path": "/opt/venv/bin"}


def test_missing_bin_raises():
    with pytest.raises(AgentXPreflightError) as ei:
        check_aiperf_capability(None)
    assert "AIPERF_BIN" in str(ei.value)


def test_capability_absent_raises():
    # probe returns help text WITHOUT weka-trace -> not AgentX-capable
    def _probe(_bin):
        return "usage: aiperf profile [options]\n  --public-dataset ...\n"

    with pytest.raises(AgentXPreflightError) as ei:
        check_aiperf_capability("/venv/bin/aiperf", probe=_probe)
    assert "weka-trace" in str(ei.value) or "capab" in str(ei.value).lower()


_CAPABLE_HELP = (
    "usage: aiperf profile\n"
    "  --custom-dataset-type weka-trace ...\n"
    "  --scenario TEXT  Lock all benchmark invariants for a named scenario\n"
    "  --benchmark-duration FLOAT\n"
    "  --api-host TEXT\n"
    "  --api-port INTEGER\n"
)


def test_capability_present_ok():
    def _probe(_bin):
        return _CAPABLE_HELP

    # must not raise
    check_aiperf_capability("/venv/bin/aiperf", probe=_probe)


def test_capability_rejects_build_without_progress_api():
    help_text = "weka-trace --scenario --benchmark-duration"
    with pytest.raises(AgentXPreflightError, match="phase progress"):
        check_aiperf_capability(
            "/venv/bin/aiperf",
            require_progress_api=True,
            probe=lambda _bin: help_text,
            loader_probe=lambda _bin: _NEW,
        )


def test_capability_rejects_pre_scenario_build():
    """weka-trace alone is stale: those builds predate the 062126 corpus.

    Their scenario allowlist rejects the corpus the client now requests and
    they have no ``--benchmark-duration``, so accepting them would defer the
    failure to an hour into a run instead of surfacing it at startup.
    """

    def _probe(_bin):
        return "usage: aiperf profile\n  --custom-dataset-type weka-trace ...\n"

    with pytest.raises(AgentXPreflightError) as ei:
        check_aiperf_capability("/venv/bin/aiperf", probe=_probe)
    assert "--scenario" in str(ei.value)


def test_probe_failure_raises_not_crash():
    def _probe(_bin):
        raise OSError("cannot exec")

    with pytest.raises(AgentXPreflightError):
        check_aiperf_capability("/venv/bin/aiperf", probe=_probe)


# --- loader-allowlist assertion ------------------------------------------------
#
# Flag presence cannot separate the pinned build from the previous one: aiperf
# 0.8.0 carries weka-trace, --scenario and --benchmark-duration, and defines a
# scenario by the same name, but locks different invariants and predates the
# current corpus. The allowlist is the discriminator.

_NEW = [
    "semianalysis_cc_traces_weka_with_subagents",
    "semianalysis_cc_traces_weka_with_subagents_256k",
    "semianalysis_cc_traces_weka_062126",
    "semianalysis_cc_traces_weka_062126_256k",
    "weka_trace",
]
_OLD = [  # the pre-062126 allowlist: same flags, older corpora
    "semianalysis_cc_traces_weka_with_subagents",
    "semianalysis_cc_traces_weka_with_subagents_256k",
    "weka_trace",
]


def _check(loaders, env=None):
    check_aiperf_capability(
        "/venv/bin/aiperf",
        loader_probe=lambda _b: loaders,
        probe=lambda _b: _CAPABLE_HELP,
        env=env or {},
    )


def test_pinned_allowlist_passes():
    _check(_NEW)


def test_stale_build_is_rejected():
    """The exact case a flag probe waves through."""
    with pytest.raises(AgentXPreflightError) as ei:
        _check(_OLD)
    msg = str(ei.value)
    assert "stale build" in msg
    assert "semianalysis_cc_traces_weka_062126" in msg


def test_stale_build_is_rejected_even_when_the_pinned_corpus_is_admitted():
    """The silent path a run-scoped check alone leaves open.

    Upstream's own H100/H200 recipes pin an older corpus via
    WEKA_LOADER_OVERRIDE. A stale aiperf DOES admit that corpus, so asking only
    "is this run's corpus allowed" waves the stale build through and it replays
    under the wrong invariants. Build currency has to be asserted separately.
    """
    with pytest.raises(AgentXPreflightError) as ei:
        _check(_OLD, env={"WEKA_LOADER_OVERRIDE": "semianalysis_cc_traces_weka_with_subagents"})
    assert "stale build" in str(ei.value)


def test_current_build_accepts_an_older_corpus_pin():
    """On a current build an older corpus is a legitimate operator choice."""
    _check(_NEW, env={"WEKA_LOADER_OVERRIDE": "semianalysis_cc_traces_weka_with_subagents"})


def test_unknown_corpus_pin_is_rejected_before_the_server_boots():
    with pytest.raises(AgentXPreflightError) as ei:
        _check(_NEW, env={"WEKA_LOADER_OVERRIDE": "semianalysis_cc_traces_weka_nonexistent"})
    assert "not in the" in str(ei.value)


def test_agentx_dataset_outranks_weka_loader_override():
    with pytest.raises(AgentXPreflightError):
        _check(
            _NEW,
            env={
                "AGENTX_DATASET": "semianalysis_cc_traces_weka_nonexistent",
                "WEKA_LOADER_OVERRIDE": "semianalysis_cc_traces_weka_062126",
            },
        )


def test_unreadable_allowlist_falls_back_and_says_so(capsys):
    """Refusing outright would break setups that work today over what may be an
    unusual install layout -- but the weaker check must not pass silently."""
    check_aiperf_capability(
        "/venv/bin/aiperf",
        loader_probe=lambda _b: None,
        probe=lambda _b: _CAPABLE_HELP,
        env={},
    )
    assert "could not read" in capsys.readouterr().err


def test_unreadable_allowlist_still_rejects_a_flagless_build():
    with pytest.raises(AgentXPreflightError):
        check_aiperf_capability(
            "/venv/bin/aiperf",
            loader_probe=lambda _b: None,
            probe=lambda _b: "nothing useful here",
            env={},
        )


def test_loader_probe_survives_a_hung_interpreter(monkeypatch):
    """A timeout must degrade to the flag probe, not escape the check.

    ``subprocess.run(timeout=...)`` raises ``TimeoutExpired``, which descends
    from ``SubprocessError`` rather than ``OSError`` -- so catching only the
    latter let it propagate out of ``check_aiperf_capability`` and become a hard
    preflight failure, on exactly the input the timeout exists to handle.
    """
    import subprocess

    from hyperloom.inference_optimizer.agentx import preflight as pf

    def _hang(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="python", timeout=60)

    monkeypatch.setattr(pf.subprocess, "run", _hang)
    assert pf._default_loader_probe("/venv/bin/aiperf") is None


def test_hung_interpreter_reaches_the_flag_fallback(monkeypatch, capsys):
    """End to end: a hung probe must land on the documented fallback path."""
    import subprocess

    from hyperloom.inference_optimizer.agentx import preflight as pf

    monkeypatch.setattr(
        pf.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(subprocess.TimeoutExpired("python", 60)),
    )
    check_aiperf_capability(
        "/venv/bin/aiperf",
        probe=lambda _b: _CAPABLE_HELP,
        env={},
    )
    assert "could not read" in capsys.readouterr().err
