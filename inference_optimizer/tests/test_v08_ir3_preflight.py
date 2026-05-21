"""KB_design_continue §3.3 / IR-3 — preflight soft-degrade tests.

7 cases (per the implementation plan):
1. KB ok + PR ok → both reachable, reasons ``None``.
2. KB 5xx + no flag → ``cortex_enabled=False``, ``kb_degraded_reason="ir3_auto"``;
   **cli does not abort** (soft degrade).
3. KB 5xx + ``--degraded-kb`` → script invoked with ``SKIP_KB_PROBE=1``,
   ``kb_degraded_reason="explicit_flag"``.
4. KB 401 + non-empty ``KB_SERVICE_TOKEN`` → kb reachable.
5. KB 401 + empty token → soft degrade ``ir3_auto`` +
   marker ``kb_failure_reason="missing_token"``.
6. PR timeout + ``--degraded-pr`` → ``pr_degraded_reason="explicit_flag"``,
   KB still ok.
7. Both flags → ``preflight_kb.sh`` is **not** invoked at all
   (``subprocess.run.assert_not_called()``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer import cli as cli_module


def _ns(**overrides) -> argparse.Namespace:
    defaults: dict = {
        "degraded_kb": False,
        "degraded_pr": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_marker(
    marker_path: Path, *,
    kb_reachable: bool, pr_reachable: bool,
    kb_skipped: bool = False, pr_skipped: bool = False,
    kb_failure_reason: str | None = None,
    pr_failure_reason: str | None = None,
) -> None:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "kb_reachable": kb_reachable,
        "pr_reachable": pr_reachable,
        "kb_skipped":   kb_skipped,
        "pr_skipped":   pr_skipped,
    }
    if kb_failure_reason is not None:
        payload["kb_failure_reason"] = kb_failure_reason
    if pr_failure_reason is not None:
        payload["pr_failure_reason"] = pr_failure_reason
    marker_path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_run_writes_marker(marker_path: Path, **marker_kwargs):
    """Return a ``subprocess.run`` stub that writes ``marker_path``
    with the given content + returns rc=0 / rc=1 as appropriate."""
    def _runner(cmd, env=None, check=False, timeout=None):
        _write_marker(marker_path, **marker_kwargs)
        # Compute rc: 1 if any not-skipped branch is unreachable, else 0.
        kb_skipped = marker_kwargs.get("kb_skipped", False)
        pr_skipped = marker_kwargs.get("pr_skipped", False)
        kb_ok = marker_kwargs.get("kb_reachable", False)
        pr_ok = marker_kwargs.get("pr_reachable", False)
        rc = 0
        if not kb_skipped and not kb_ok:
            rc = 1
        if not pr_skipped and not pr_ok:
            rc = 1
        return subprocess.CompletedProcess(cmd, rc)
    return _runner


@pytest.fixture
def marker_path(tmp_path, monkeypatch) -> Path:
    user_data = tmp_path / "user_data"
    monkeypatch.setenv("USER_DATA_PATH", str(user_data))
    return user_data / "runtime" / "cortex" / ".kb_preflight.json"


# ---------------------------------------------------------------------------
# 1. KB ok + PR ok → both reachable, reasons None.
# ---------------------------------------------------------------------------
def test_ir3_kb_ok_pr_ok(marker_path):
    args = _ns()
    with patch.object(
        cli_module.subprocess, "run",
        side_effect=_fake_run_writes_marker(marker_path, kb_reachable=True, pr_reachable=True),
    ):
        cli_module._run_ir3_preflight(args)
    assert args.cortex_enabled is True
    assert args.pr_monitor_enabled is True
    assert args.kb_degraded_reason is None
    assert args.pr_degraded_reason is None


# ---------------------------------------------------------------------------
# 2. KB 5xx + no flag → soft degrade ir3_auto; cli does not abort.
# ---------------------------------------------------------------------------
def test_ir3_kb_5xx_auto_degrade(marker_path):
    args = _ns()
    with patch.object(
        cli_module.subprocess, "run",
        side_effect=_fake_run_writes_marker(
            marker_path, kb_reachable=False, pr_reachable=True,
            kb_failure_reason="500",
        ),
    ):
        cli_module._run_ir3_preflight(args)
    assert args.cortex_enabled is False
    assert args.pr_monitor_enabled is True
    assert args.kb_degraded_reason == "ir3_auto"
    assert args.pr_degraded_reason is None


# ---------------------------------------------------------------------------
# 3. KB 5xx + --degraded-kb → script gets SKIP_KB_PROBE=1, reason=explicit.
# ---------------------------------------------------------------------------
def test_ir3_kb_explicit_flag(marker_path):
    args = _ns(degraded_kb=True)
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        # Write a marker as if the script ran with kb skipped.
        _write_marker(marker_path, kb_reachable=False, pr_reachable=True, kb_skipped=True)
        return subprocess.CompletedProcess(cmd, 0)

    with patch.object(cli_module.subprocess, "run", side_effect=_runner):
        cli_module._run_ir3_preflight(args)
    assert seen_env.get("SKIP_KB_PROBE") == "1"
    assert "SKIP_PR_PROBE" not in seen_env
    assert args.cortex_enabled is False
    assert args.kb_degraded_reason == "explicit_flag"
    assert args.pr_monitor_enabled is True
    assert args.pr_degraded_reason is None


# ---------------------------------------------------------------------------
# 4. KB 401 + non-empty token → kb reachable (auth path).
# ---------------------------------------------------------------------------
def test_ir3_kb_401_with_token(marker_path, monkeypatch):
    monkeypatch.setenv("KB_SERVICE_TOKEN", "tok-abc")
    args = _ns()
    # The actual probe semantics live in preflight_kb.sh; here we
    # simulate "401 with token → kb_reachable=true" by writing the
    # marker as reachable.
    with patch.object(
        cli_module.subprocess, "run",
        side_effect=_fake_run_writes_marker(marker_path, kb_reachable=True, pr_reachable=True),
    ):
        cli_module._run_ir3_preflight(args)
    assert args.cortex_enabled is True
    assert args.kb_degraded_reason is None


# ---------------------------------------------------------------------------
# 5. KB 401 + empty token → soft degrade ir3_auto, marker missing_token.
# ---------------------------------------------------------------------------
def test_ir3_kb_401_missing_token(marker_path, monkeypatch):
    monkeypatch.delenv("KB_SERVICE_TOKEN", raising=False)
    args = _ns()
    with patch.object(
        cli_module.subprocess, "run",
        side_effect=_fake_run_writes_marker(
            marker_path, kb_reachable=False, pr_reachable=True,
            kb_failure_reason="missing_token",
        ),
    ):
        cli_module._run_ir3_preflight(args)
    assert args.cortex_enabled is False
    assert args.kb_degraded_reason == "ir3_auto"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["kb_failure_reason"] == "missing_token"


# ---------------------------------------------------------------------------
# 6. PR timeout + --degraded-pr → reason=explicit_flag, KB stays ok.
# ---------------------------------------------------------------------------
def test_ir3_pr_explicit_flag_kb_ok(marker_path):
    args = _ns(degraded_pr=True)
    seen_env: dict = {}

    def _runner(cmd, env=None, check=False, timeout=None):
        seen_env.update(env or {})
        _write_marker(marker_path, kb_reachable=True, pr_reachable=False, pr_skipped=True)
        return subprocess.CompletedProcess(cmd, 0)

    with patch.object(cli_module.subprocess, "run", side_effect=_runner):
        cli_module._run_ir3_preflight(args)
    assert seen_env.get("SKIP_PR_PROBE") == "1"
    assert "SKIP_KB_PROBE" not in seen_env
    assert args.cortex_enabled is True
    assert args.kb_degraded_reason is None
    assert args.pr_monitor_enabled is False
    assert args.pr_degraded_reason == "explicit_flag"


# ---------------------------------------------------------------------------
# 7. Both flags → preflight_kb.sh NOT invoked.
# ---------------------------------------------------------------------------
def test_ir3_both_flags_short_circuit(marker_path):
    args = _ns(degraded_kb=True, degraded_pr=True)
    with patch.object(cli_module.subprocess, "run") as run_mock:
        cli_module._run_ir3_preflight(args)
        run_mock.assert_not_called()
    assert args.cortex_enabled is False
    assert args.pr_monitor_enabled is False
    assert args.kb_degraded_reason == "explicit_flag"
    assert args.pr_degraded_reason == "explicit_flag"


# ---------------------------------------------------------------------------
# Bonus: CLI flag plumbing
# ---------------------------------------------------------------------------
def test_cli_parser_exposes_degraded_flags():
    parser = cli_module._build_parser()
    args = parser.parse_args(["optimize", "--model", "/x", "--degraded-kb"])
    assert args.degraded_kb is True
    assert args.degraded_pr is False
    args = parser.parse_args(["optimize", "--model", "/x", "--degraded-pr"])
    assert args.degraded_pr is True
    assert args.degraded_kb is False
    args = parser.parse_args(["optimize", "--model", "/x"])
    assert args.degraded_kb is False
    assert args.degraded_pr is False
