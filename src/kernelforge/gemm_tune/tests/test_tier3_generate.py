# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for asking an agent to author a generated tuner.

Every failure here has to be survivable, because the tier is optional: a
missing provider, an unusable one, a session that dies, and a session that ends
without writing the file all have to come back as a ``GeneratedTuner`` the
caller can read, never as an exception that ends the tuning run. These tests
pin each of those paths, and the calling convention ``_run`` has to tolerate.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from kernelforge.agent_backends import registry as registry_mod
from kernelforge.gemm_tune.tier3.generate import (
    GeneratedTuner,
    _run,
    _user_prompt,
    generate_tuner,
)
from kernelforge.gemm_tune.tier3.mandate import TunerMandate


def _mandate() -> TunerMandate:
    return TunerMandate(
        table="bf16_tuned_gemm.csv",
        key_schema=["M", "N", "K"],
        demand_shapes=[{"M": 16, "N": 1536, "K": 7168}],
        why_existing_tiers_failed="no tuner ships for this table",
    )


class _Backend:
    """A backend whose ``run`` is synchronous and records the spec it got."""

    name = "fake-provider"

    def __init__(self, *, writes: str | None = None, raises: bool = False) -> None:
        self._writes = writes
        self._raises = raises
        self.spec = None

    def run(self, spec):
        self.spec = spec
        if self._raises:
            raise RuntimeError("session died")
        if self._writes is not None:
            with open(spec.target_files[0], "w", encoding="utf-8") as fh:
                fh.write(self._writes)
        return type("_Result", (), {"session_id": "sess-1", "end_reason": "done"})()


def _install_provider(monkeypatch: pytest.MonkeyPatch, backend, *, setup_raises: bool = False) -> None:
    """Point the registry at ``backend`` (or make provider setup fail)."""
    if setup_raises:

        def _boom(*_a, **_k):
            raise RuntimeError("no CLI installed")

        monkeypatch.setattr(registry_mod, "select_default_agent_provider", _boom)
        return

    monkeypatch.setattr(
        registry_mod,
        "select_default_agent_provider",
        lambda _model: type("_Chosen", (), {"name": "fake-provider"})(),
    )
    monkeypatch.setattr(registry_mod, "resolve_agent_runtime", lambda *_a, **_k: object())
    monkeypatch.setattr(registry_mod, "create_registered_backend", lambda _rt: backend)


class TestTheOutcomeIsAlwaysReadable:
    def test_an_unwritten_script_serialises_as_null_not_a_path(self):
        assert GeneratedTuner(False, None, "nope").to_dict() == {
            "ok": False,
            "script": None,
            "reason": "nope",
            "provider": "",
            "session_id": "",
        }

    def test_a_written_script_serialises_as_its_path(self, tmp_path):
        got = GeneratedTuner(True, tmp_path / "tuner.py", "", "p", "s").to_dict()
        assert got["ok"] is True
        assert got["script"] == str(tmp_path / "tuner.py")
        assert (got["provider"], got["session_id"]) == ("p", "s")


class TestTheBriefTheAgentGets:
    def test_the_mandate_and_the_deliverable_both_travel(self, tmp_path):
        script = tmp_path / "tuner.py"
        prompt = _user_prompt(_mandate(), script, "")

        assert "bf16_tuned_gemm.csv" in prompt  # the mandate itself
        assert str(script) in prompt
        assert "produce both output files named above" in prompt

    def test_a_retry_states_what_was_rejected(self, tmp_path):
        prompt = _user_prompt(_mandate(), tmp_path / "tuner.py", "header was wrong")

        assert "## The previous attempt was rejected" in prompt
        assert "header was wrong" in prompt

    def test_a_first_attempt_carries_no_rejection_section(self, tmp_path):
        assert "previous attempt" not in _user_prompt(_mandate(), tmp_path / "tuner.py", "")


class TestEveryFailureComesBackAsData:
    def test_no_agent_provider_in_this_install_is_skipped_not_raised(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """The standalone tuning wheel ships without the LLM stack on purpose."""
        monkeypatch.setitem(sys.modules, "kernelforge.agent_backends.registry", None)

        got = generate_tuner(_mandate(), tmp_path / "work")

        assert got.ok is False
        assert "no agent provider available in this install" in got.reason
        assert got.script_path is None

    def test_an_unusable_provider_is_reported_not_raised(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _install_provider(monkeypatch, None, setup_raises=True)

        got = generate_tuner(_mandate(), tmp_path / "work")

        assert got.ok is False
        assert "agent provider unusable" in got.reason

    def test_a_session_that_dies_is_reported_not_raised(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _install_provider(monkeypatch, _Backend(raises=True))

        got = generate_tuner(_mandate(), tmp_path / "work")

        assert got.ok is False
        assert "authoring session failed" in got.reason

    def test_a_session_that_writes_nothing_names_why_it_ended(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        _install_provider(monkeypatch, _Backend(writes=None))

        got = generate_tuner(_mandate(), tmp_path / "work")

        assert got.ok is False
        assert "without writing tuner.py" in got.reason
        assert (got.provider, got.session_id) == ("fake-provider", "sess-1")


class TestASuccessfulAuthoringSession:
    def test_the_written_script_is_returned_with_its_session(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        backend = _Backend(writes="print('hi')\n")
        _install_provider(monkeypatch, backend)

        got = generate_tuner(_mandate(), tmp_path / "work", model="some-model", timeout_s=42)

        assert got.ok is True
        assert got.script_path is not None and got.script_path.is_file()
        assert (got.provider, got.session_id, got.reason) == ("fake-provider", "sess-1", "")

    def test_the_agent_is_confined_to_the_work_dir_it_was_given(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        backend = _Backend(writes="x = 1\n")
        _install_provider(monkeypatch, backend)
        work = tmp_path / "work"

        generate_tuner(_mandate(), work, timeout_s=42)

        assert work.is_dir()
        assert backend.spec.cwd == str(work)
        assert backend.spec.writable is True
        assert backend.spec.timeout_sec == 42
        assert backend.spec.target_files == [str(work / "tuner.py")]


class TestWhicheverCallingConventionTheBackendOffers:
    def test_a_synchronous_backend_is_called_directly(self):
        calls = []
        backend = type("_B", (), {"run": lambda _self, spec: calls.append(spec) or "sync-result"})()

        assert _run(backend, "spec") == "sync-result"
        assert calls == ["spec"]

    def test_an_async_backend_gets_its_own_loop(self):
        class _AsyncBackend:
            async def run(self, spec):
                return f"async:{spec}"

        assert _run(_AsyncBackend(), "spec") == "async:spec"

    def test_an_async_backend_inside_a_running_loop_says_to_use_a_thread(self):
        """Silently nesting loops would deadlock, so this refuses with guidance."""

        class _AsyncBackend:
            async def run(self, spec):  # pragma: no cover - never reached
                return spec

        async def _from_inside_a_loop():
            with pytest.raises(RuntimeError, match="worker thread"):
                _run(_AsyncBackend(), "spec")

        asyncio.run(_from_inside_a_loop())
