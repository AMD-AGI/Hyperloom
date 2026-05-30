"""Unit tests for ``_probe_ray_head`` + ``_parse_ray_pending_count``.

Covers the regression where the legacy regex ``(\\d+)\\s+pending``
captured the trailing hex digits of Ray node IDs (e.g. node ID ending
``...d3da81`` followed by the ``Pending:`` section header → bogus
``pending_tasks=81``).
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import patch

import pytest

from hyperloom.agents.robustness.sources import local_probe


# Real ``ray status`` output captured during the 2026-05-20 incident:
# node ID ends in ``...d3da81``, immediately followed by ``Pending:``.
RAY_STATUS_IDLE = """\
======== Autoscaler status: 2026-05-20 03:14:48.677060 ========
Node status
---------------------------------------------------------------
Active:
 1 node_dcd71ad0316b238eb2ab9323d50f23bb8201ef3c78447c14dad3da81
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/64.0 CPU
 0.0/8.0 GPU
 0B/829.27GiB memory
 0B/186.26GiB object_store_memory

Demands:
 (no resource demands)
"""

RAY_STATUS_SINGLE_DEMAND = """\
======== Autoscaler status: 2026-05-20 04:00:00 ========
Node status
---------------------------------------------------------------
Active:
 1 node_dcd71ad0316b238eb2ab9323d50f23bb8201ef3c78447c14dad3da81

Resources
---------------------------------------------------------------
Usage:
 64.0/64.0 CPU
 8.0/8.0 GPU

Demands:
 {'CPU': 1.0}: 5+ pending tasks/actors
"""

RAY_STATUS_MULTI_DEMAND = """\
======== Autoscaler status: 2026-05-20 04:01:00 ========
Node status
---------------------------------------------------------------
Active:
 1 node_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa11
 1 node_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb22

Demands:
 {'CPU': 1.0}: 4+ pending tasks
 {'GPU': 1.0}: 12+ pending tasks/actors
 {'CPU': 0.5, 'memory': 1000000000}: 2 pending actors
"""


# ---------------------------------------------------------------------------
# _parse_ray_pending_count
# ---------------------------------------------------------------------------

def test_parse_idle_status_with_digit_terminated_node_id():
    """Regression: legacy regex captured ``81`` from ``...d3da81\\nPending``."""
    assert local_probe._parse_ray_pending_count(RAY_STATUS_IDLE) == 0


def test_parse_single_demand():
    assert local_probe._parse_ray_pending_count(RAY_STATUS_SINGLE_DEMAND) == 5


def test_parse_multiple_demands_sum():
    assert local_probe._parse_ray_pending_count(RAY_STATUS_MULTI_DEMAND) == 18


def test_parse_empty_string():
    assert local_probe._parse_ray_pending_count("") == 0


def test_parse_ignores_node_id_substrings():
    """``\\d+`` inside node-hash tokens must not match without ``pending task|actor`` suffix."""
    text = """\
Active:
 1 node_0000000000000000000000000000000000000000000000000000000000000099
 2 node_0000000000000000000000000000000000000000000000000000000000000123
"""
    assert local_probe._parse_ray_pending_count(text) == 0


def test_parse_ignores_pending_nodes_header():
    text = "Pending:\n (no pending nodes)\n"
    assert local_probe._parse_ray_pending_count(text) == 0


def test_parse_handles_plus_suffix_on_count():
    text = " {'GPU': 8.0}: 99+ pending tasks/actors"
    assert local_probe._parse_ray_pending_count(text) == 99


def test_parse_actor_suffix():
    text = " {'CPU': 0.1}: 7 pending actors"
    assert local_probe._parse_ray_pending_count(text) == 7


# ---------------------------------------------------------------------------
# _probe_ray_head
# ---------------------------------------------------------------------------

def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["ray", "status"], returncode=returncode, stdout=stdout, stderr="",
    )


def test_probe_returns_empty_when_ray_not_on_path():
    with patch.object(local_probe.shutil, "which", return_value=None):
        assert local_probe._probe_ray_head(1.0) == {}


def test_probe_idle_returns_zero_pending():
    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run",
                      return_value=_completed(RAY_STATUS_IDLE)):
        out = local_probe._probe_ray_head(1.0)
    assert out["healthy"] is True
    assert out["pending_tasks"] == 0
    assert out["returncode"] == 0
    assert "Autoscaler status" in out["stdout_head"]


def test_probe_demand_block_counted():
    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run",
                      return_value=_completed(RAY_STATUS_MULTI_DEMAND)):
        out = local_probe._probe_ray_head(1.0)
    assert out["pending_tasks"] == 18


def test_probe_unhealthy_on_nonzero_exit():
    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run",
                      return_value=_completed("ConnectionError: ...", returncode=1)):
        out = local_probe._probe_ray_head(1.0)
    assert out["healthy"] is False
    assert "exit=1" in out["reason"]
    assert out["returncode"] == 1


def test_probe_unhealthy_on_timeout():
    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="ray status", timeout=1.0)

    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run", side_effect=_raise):
        out = local_probe._probe_ray_head(1.0)
    assert out["healthy"] is False
    assert "timed out" in out["reason"]
    assert out["returncode"] is None


def test_probe_unhealthy_on_oserror():
    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("ray binary missing mid-call")

    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run", side_effect=_raise):
        out = local_probe._probe_ray_head(1.0)
    assert out["healthy"] is False
    assert "FileNotFoundError" in out["reason"]


def test_probe_clamps_negative_timeout():
    captured: dict[str, Any] = {}

    def _capture(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        captured.update(kwargs)
        return _completed(RAY_STATUS_IDLE)

    with patch.object(local_probe.shutil, "which", return_value="/usr/bin/ray"), \
         patch.object(local_probe.subprocess, "run", side_effect=_capture):
        local_probe._probe_ray_head(0.0)
    assert captured["timeout"] >= 0.5


# ---------------------------------------------------------------------------
# Property: regex must not match arbitrary 64-char hex strings.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", ["00", "11", "42", "99", "ab", "ff", "9a"])
def test_regex_never_matches_node_hash_followed_by_pending_header(suffix: str):
    text = f" 1 node_{'0' * 62}{suffix}\nPending:\n (no pending nodes)\n"
    assert local_probe._parse_ray_pending_count(text) == 0
