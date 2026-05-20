"""v0.8 KB_design §3.2 §5.1 + KB_gaps/Gap-12 — Cortex T0 anchor tests.

Covers the v0.8 contract for the T0 ritual after the cli /
Coordinator dual-entry refactor (KB_gaps/Gap-12):

* :func:`orchestrator.cortex_t0.run_t0_anchor` is the single source
  of truth for the four T0 steps (session_begin / propose_point
  workload_node / find_recipe / traps) and the SharedState writes
  that go with them.
* cli is the **canonical** entry point — fail-fast on Cortex
  failure (``sys.exit(2)``), stdout banner the operator expects.
* :meth:`Coordinator._ensure_cortex_t0_anchored` is a **defensive
  fallback** for SDK / integration-test callers that construct a
  :class:`Coordinator` without going through the cli plumbing —
  fail-soft on Cortex failure, INFO-log banner instead of stdout.
* The fallback no-ops when ``cortex_kb`` is None, disabled, or a
  prior anchor already wrote ``shared_state.cortex_session_id``.

The cli + Cortex CLI binary are mocked with the same fake shell
script pattern as :file:`test_v08_m1_cortex_kb.py`.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from inference_optimizer.cortex_kb_client import (
    CortexBinaryNotFound,
    CortexKBClient,
    CortexKBError,
)
from inference_optimizer.orchestrator.cortex_t0 import (
    T0Result,
    run_t0_anchor,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.paths import make_session_dir
from inference_optimizer.session_paths import (
    cortex_pitfalls_json,
    cortex_sid_file,
    cortex_warm_json,
)


# ===========================================================================
# fixtures
# ===========================================================================
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _write_fake_cortex_bin(
    bin_dir: Path,
    *,
    stdout_lines: list[str],
    exit_code: int = 0,
) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / "cortex-kb"
    body = "\n".join(f"echo {json.dumps(line)}" for line in stdout_lines)
    target.write_text(
        "#!/bin/sh\n"
        f"{body}\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    target.chmod(
        target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
    )
    return target


def _state_for_session(session_dir: Path) -> SharedState:
    s = SharedState()
    s.session_id = "sid-test"
    s.model_name = "Qwen-Qwen3-8B"
    s.gpu_type = "mi300x"
    s.framework = "sglang"
    s.save(session_dir)
    return SharedState.load_or_init(session_dir)


# ===========================================================================
# 1. run_t0_anchor — short-circuit branches
# ===========================================================================
def test_t0_disabled_client_emits_banner_and_skips_state_writes(session_dir):
    client = CortexKBClient(session_dir=session_dir, enabled=False)
    state = _state_for_session(session_dir)
    lines: list[str] = []
    result = run_t0_anchor(
        client, state,
        workload="Qwen-Qwen3-8B", hw="mi300x",
        on_status=lines.append,
    )
    assert isinstance(result, T0Result)
    assert result.status == "skipped_disabled"
    assert any("DISABLED" in line for line in lines)
    # No state writes happened.
    assert state.cortex_session_id == ""
    assert state.warm_start_recipe == {}
    assert state.warm_start_pitfalls == []


def test_t0_already_anchored_session_id_does_not_re_begin(
    session_dir, tmp_path, monkeypatch,
):
    """When ``cortex_session_id`` is already non-empty (e.g. cli already
    ran T0), the helper MUST NOT call ``session begin`` again. It still
    refreshes find_recipe / traps so a long-running session that
    survives Cortex outages picks up newer KB rows."""
    bin_dir = tmp_path / "bin"
    # Distinct outputs so we can assert which calls fired.
    _write_fake_cortex_bin(bin_dir, stdout_lines=[
        "session_id: SHOULD-NOT-FIRE",  # if session_begin runs, we see this
    ])
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    client = CortexKBClient(session_dir=session_dir)
    state = _state_for_session(session_dir)
    state.cortex_session_id = "prior-sid"
    # Patch session_begin so a regression that ever calls it is loud.
    sentinel: list[str] = []

    def _boom(**kwargs):
        sentinel.append("session_begin")
        return "SHOULD-NOT-FIRE"

    with patch.object(client, "session_begin", side_effect=_boom):
        result = run_t0_anchor(
            client, state,
            workload="w", hw="mi300x",
            on_status=lambda _l: None,
        )
    assert sentinel == []
    assert result.session_id == "prior-sid"
    assert state.cortex_session_id == "prior-sid"


def test_t0_fail_soft_returns_failed_session_begin_on_cortex_error(session_dir):
    """fail_fast=False (Coordinator fallback) absorbs the error and
    returns a failed result; warm_start fields stay empty."""
    client = CortexKBClient(session_dir=session_dir)
    state = _state_for_session(session_dir)
    with patch.object(
        client, "session_begin",
        side_effect=CortexKBError("synthetic outage"),
    ):
        result = run_t0_anchor(
            client, state,
            workload="w", hw="mi300x",
            fail_fast=False,
            on_status=lambda _l: None,
        )
    assert result.status == "failed_session_begin"
    assert "synthetic outage" in result.error
    assert state.cortex_session_id == ""
    assert state.warm_start_recipe == {}


def test_t0_fail_fast_propagates_cortex_error(session_dir):
    """fail_fast=True (cli path) re-raises so cli can sys.exit(2)."""
    client = CortexKBClient(session_dir=session_dir)
    state = _state_for_session(session_dir)
    with patch.object(
        client, "session_begin",
        side_effect=CortexKBError("synthetic outage"),
    ):
        with pytest.raises(CortexKBError):
            run_t0_anchor(
                client, state,
                workload="w", hw="mi300x",
                fail_fast=True,
                on_status=lambda _l: None,
            )


def test_t0_fail_fast_propagates_binary_not_found(session_dir, monkeypatch):
    """fail_fast=True bubbles CortexBinaryNotFound so cli surfaces the
    install hint and exits."""
    monkeypatch.setenv("PATH", "/nonexistent_xyz_t0")
    monkeypatch.setenv("CORTEX_KB_BIN", "definitely-not-here-t0")
    client = CortexKBClient(session_dir=session_dir)
    state = _state_for_session(session_dir)
    with pytest.raises(CortexBinaryNotFound):
        run_t0_anchor(
            client, state,
            workload="w", hw="mi300x",
            fail_fast=True,
            on_status=lambda _l: None,
        )


def test_t0_fail_soft_absorbs_binary_not_found(session_dir, monkeypatch):
    """fail_fast=False (Coordinator fallback) absorbs the missing-
    binary case so an SDK without cortex-kb installed still boots."""
    monkeypatch.setenv("PATH", "/nonexistent_xyz_t0")
    monkeypatch.setenv("CORTEX_KB_BIN", "definitely-not-here-t0-soft")
    client = CortexKBClient(session_dir=session_dir)
    state = _state_for_session(session_dir)
    result = run_t0_anchor(
        client, state,
        workload="w", hw="mi300x",
        fail_fast=False,
        on_status=lambda _l: None,
    )
    assert result.status == "failed_session_begin"
    assert result.error == "cortex_binary_not_found"


# ===========================================================================
# 2. run_t0_anchor — happy path with the fake cortex-kb binary
# ===========================================================================
def test_t0_happy_path_writes_sid_warm_and_pitfalls(
    session_dir, tmp_path, monkeypatch,
):
    bin_dir = tmp_path / "bin"
    # The fake binary emits the same prefix for every cli verb (session
    # begin / find-recipe / traps / propose-point); the client only
    # consumes ``session_id: ...`` from session begin, the others are
    # stored verbatim as the warm/traps raw text.
    _write_fake_cortex_bin(bin_dir, stdout_lines=[
        "session_id: t0-test-sid",
        "recipe: warm_start_row_1",
        "trap: pitfall_row_1",
    ])
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    client = CortexKBClient(session_dir=session_dir)
    state = _state_for_session(session_dir)
    banner: list[str] = []
    result = run_t0_anchor(
        client, state,
        workload="Qwen-Qwen3-8B", hw="mi300x",
        image_digest="sha256:demo",
        stack_fingerprint={"rocm": "7.2.0"},
        extra_attrs={"framework": "sglang"},
        on_status=banner.append,
        session_dir=session_dir,
        save_state=True,
    )
    assert result.status == "ok"
    assert result.session_id == "t0-test-sid"
    assert state.cortex_session_id == "t0-test-sid"
    # The fake binary's stdout is *constant* across all cli verbs, so
    # we only pin the structural shape — the raw text is whatever the
    # fake binary emitted (a few echoed lines).
    assert state.warm_start_recipe["workload"] == "Qwen-Qwen3-8B"
    assert state.warm_start_recipe["hw"] == "mi300x"
    assert state.warm_start_recipe["raw"]
    assert state.warm_start_pitfalls and state.warm_start_pitfalls[0]["raw"]
    # .kb_sid + .kb_warm.json + .kb_pitfalls.json all written.
    assert cortex_sid_file(session_dir).read_text(encoding="utf-8").strip() == "t0-test-sid"
    assert cortex_warm_json(session_dir).exists()
    assert cortex_pitfalls_json(session_dir).exists()
    # Banner mentions the canonical id we just minted.
    assert any("session_id=t0-test-sid" in line for line in banner)


def test_t0_skipped_already_when_sid_present_via_anchor(
    session_dir, tmp_path, monkeypatch,
):
    """Two consecutive ``run_t0_anchor`` calls: the second sees a
    sid and reports ``skipped_already`` — no re-begin."""
    bin_dir = tmp_path / "bin"
    _write_fake_cortex_bin(bin_dir, stdout_lines=[
        "session_id: t0-test-sid",
    ])
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    client = CortexKBClient(session_dir=session_dir)
    state = _state_for_session(session_dir)
    run_t0_anchor(
        client, state,
        workload="w", hw="mi300x",
        on_status=lambda _l: None,
    )
    sid_after_first = state.cortex_session_id
    assert sid_after_first == "t0-test-sid"
    # Patch session_begin so a second call would explode if it ran.
    boom_calls: list[str] = []

    def _boom(**kwargs):
        boom_calls.append("hit")
        return "WRONG"

    with patch.object(client, "session_begin", side_effect=_boom):
        result = run_t0_anchor(
            client, state,
            workload="w", hw="mi300x",
            on_status=lambda _l: None,
        )
    assert result.status == "skipped_already"
    assert boom_calls == []
    assert state.cortex_session_id == sid_after_first


# ===========================================================================
# 3. Coordinator._ensure_cortex_t0_anchored — defensive SDK fallback
# ===========================================================================
class _StubBareCoordinator:
    """The bits of Coordinator the fallback method touches.

    We construct it with ``Coordinator.__new__`` to skip the rest of
    the constructor (which requires a real backends dict + sqlite
    db). The method is a self-contained helper so this is enough.
    """


def _bare_coord(session_dir: Path, *, client, state):
    from inference_optimizer.orchestrator.coordinator import Coordinator
    c = Coordinator.__new__(Coordinator)
    c.session_dir = session_dir
    c.shared_state = state
    c.cortex_kb = client
    return c


def test_coordinator_fallback_noop_when_cortex_disabled(session_dir):
    state = _state_for_session(session_dir)
    client = CortexKBClient(session_dir=session_dir, enabled=False)
    coord = _bare_coord(session_dir, client=client, state=state)
    coord._ensure_cortex_t0_anchored()
    assert state.cortex_session_id == ""
    assert state.warm_start_recipe == {}


def test_coordinator_fallback_noop_when_client_is_none(session_dir):
    state = _state_for_session(session_dir)
    coord = _bare_coord(session_dir, client=None, state=state)
    coord._ensure_cortex_t0_anchored()
    assert state.warm_start_recipe == {}


def test_coordinator_fallback_noop_when_session_id_already_set(
    session_dir, tmp_path, monkeypatch,
):
    """cli path: ``cortex_session_id`` already non-empty from the cli
    T0 → fallback must NOT re-fire the ritual."""
    bin_dir = tmp_path / "bin"
    _write_fake_cortex_bin(bin_dir, stdout_lines=[
        "session_id: should-not-fire",
    ])
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    state = _state_for_session(session_dir)
    state.cortex_session_id = "from-cli"
    client = CortexKBClient(session_dir=session_dir)
    coord = _bare_coord(session_dir, client=client, state=state)
    boom_calls: list[str] = []
    with patch.object(
        client, "session_begin",
        side_effect=lambda **kw: boom_calls.append("hit") or "WRONG",
    ):
        coord._ensure_cortex_t0_anchored()
    assert boom_calls == []
    assert state.cortex_session_id == "from-cli"


def test_coordinator_fallback_runs_t0_when_sid_missing(
    session_dir, tmp_path, monkeypatch,
):
    """SDK / integration-test path: ``Coordinator(...)`` constructed
    without cli plumbing. cortex_session_id is empty → fallback
    fires and writes warm_start to SharedState."""
    bin_dir = tmp_path / "bin"
    _write_fake_cortex_bin(bin_dir, stdout_lines=[
        "session_id: sdk-test-sid",
        "recipe: sdk_warm",
    ])
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    state = _state_for_session(session_dir)
    assert state.cortex_session_id == ""  # SDK path — never anchored
    client = CortexKBClient(session_dir=session_dir)
    coord = _bare_coord(session_dir, client=client, state=state)
    coord._ensure_cortex_t0_anchored()
    assert state.cortex_session_id == "sdk-test-sid"
    assert state.warm_start_recipe  # non-empty
    # Reloaded state.json carries the writes — proving save_state=True
    # ran inside the helper.
    reloaded = SharedState.load_or_init(session_dir)
    assert reloaded.cortex_session_id == "sdk-test-sid"
    assert reloaded.warm_start_recipe


def test_coordinator_fallback_absorbs_cortex_error(session_dir):
    """Cortex outage during fallback → reactor boots cleanly with
    empty warm_start, no raise."""
    state = _state_for_session(session_dir)
    client = CortexKBClient(session_dir=session_dir)
    coord = _bare_coord(session_dir, client=client, state=state)
    with patch.object(
        client, "session_begin",
        side_effect=CortexKBError("outage"),
    ):
        coord._ensure_cortex_t0_anchored()  # must not raise
    assert state.cortex_session_id == ""
    assert state.warm_start_recipe == {}


def test_coordinator_fallback_uses_state_workload_hw(
    session_dir, tmp_path, monkeypatch,
):
    """Workload / hw flow from SharedState, not from a manifest dict
    (SDK callers don't have a manifest)."""
    bin_dir = tmp_path / "bin"
    _write_fake_cortex_bin(bin_dir, stdout_lines=[
        "session_id: sdk-workload-sid",
    ])
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")
    state = _state_for_session(session_dir)
    state.model_name = "Llama-3.1-70B"
    state.gpu_type = "mi325x"
    client = CortexKBClient(session_dir=session_dir)
    coord = _bare_coord(session_dir, client=client, state=state)
    coord._ensure_cortex_t0_anchored()
    # warm_start_recipe records the workload/hw we derived from state.
    assert state.warm_start_recipe["workload"] == "Llama-3.1-70B"
    assert state.warm_start_recipe["hw"] == "mi325x"


# ===========================================================================
# 4. Re-export sanity — run_t0_anchor is importable from cli too
# ===========================================================================
def test_cli_imports_run_t0_anchor():
    """cli must import the helper so the refactored
    ``_bootstrap_cortex_kb`` keeps working without re-implementing
    the ritual locally."""
    from inference_optimizer import cli  # noqa: F401
    assert hasattr(cli, "run_t0_anchor")
    # Same callable as the canonical module's export.
    from inference_optimizer.orchestrator.cortex_t0 import (
        run_t0_anchor as canonical,
    )
    assert cli.run_t0_anchor is canonical
