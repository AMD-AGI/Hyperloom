# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``HYPERLOOM_KERNEL_MAX_TURNS`` env override for the kernel ClaudeBackend (#436).

The kernel reactor trips "Reached maximum number of turns" before GEAK is
dispatched when the per-tick turn cap is too small. PR #436 makes that cap
env-overridable in ``_build_backends`` (default 40) on the ``--kernel-claude``
path. These tests pin the three behaviours the review asked to confirm:
default, override, and the empty-string ``or "40"`` guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.cli_backends import _build_backends

ENV = "HYPERLOOM_KERNEL_MAX_TURNS"


def _kernel_backend(tmp_path: Path):
    """Build the claude-path kernel backend (kernel_codex=False)."""
    backends = _build_backends(
        claude_model="claude-opus-4-8",
        codex_model="gpt-5-codex",
        kernel_codex=False,
        critic_choice="mock",
        session_dir=tmp_path,
        robustness_choice="mock",
    )
    return backends["kernel"]


class TestKernelMaxTurnsEnv:
    def test_default_is_40_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv(ENV, raising=False)
        assert _kernel_backend(tmp_path).max_turns_default == 40

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV, "12")
        assert _kernel_backend(tmp_path).max_turns_default == 12

    def test_empty_string_falls_back_to_40(self, monkeypatch, tmp_path):
        # The `or "40"` guard: an exported-but-empty env must not raise.
        monkeypatch.setenv(ENV, "")
        assert _kernel_backend(tmp_path).max_turns_default == 40


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    yield
