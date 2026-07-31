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


def test_capability_present_ok():
    def _probe(_bin):
        return "usage: aiperf profile\n  --custom-dataset-type weka-trace ...\n"

    # must not raise
    check_aiperf_capability("/venv/bin/aiperf", probe=_probe)


def test_probe_failure_raises_not_crash():
    def _probe(_bin):
        raise OSError("cannot exec")

    with pytest.raises(AgentXPreflightError):
        check_aiperf_capability("/venv/bin/aiperf", probe=_probe)
