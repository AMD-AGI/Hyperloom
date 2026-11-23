# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch banner names the model the orchestration backend actually runs."""

from __future__ import annotations

from types import SimpleNamespace

from hyperloom.inference_optimizer.cli.backends import _orchestration_banner


class DeepSeekBackend:
    name = "deepseek"
    model = "deepseek-v4-flash"


def test_a_harness_backend_reports_its_own_model() -> None:
    """A live dsh run printed DeepSeekBackend(gpt-5.3-codex), the Codex model it never ran."""
    assert _orchestration_banner(DeepSeekBackend()) == "DeepSeekBackend(deepseek-v4-flash)"


def test_claude_and_codex_keep_their_labels() -> None:
    claude = SimpleNamespace(name="claude", model="claude-opus-5")
    codex = SimpleNamespace(name="codex", model="gpt-5.3-codex")
    assert _orchestration_banner(claude) == "Claude(claude-opus-5)"
    assert _orchestration_banner(codex) == "Codex(gpt-5.3-codex)"


def test_a_missing_backend_reads_as_disabled() -> None:
    assert _orchestration_banner(None) == "DISABLED"
