# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The integrate re-baseline's ``--gpu-memory-utilization`` cap (``HL_INTEGRATE_VRAM_GUARD``)."""

from __future__ import annotations

from hyperloom.orchestrator.kernel import _integrate_rebaseline as integrate_rebaseline


def test_vram_guard_off_is_identity(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "0")
    assert integrate_rebaseline._vram_guarded_server_args("--foo bar") == "--foo bar"
    assert integrate_rebaseline._vram_guarded_server_args("") == ""


def test_vram_guard_appends_when_on(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    monkeypatch.delenv("HL_INTEGRATE_VRAM_UTIL_CAP", raising=False)
    out = integrate_rebaseline._vram_guarded_server_args("--trust-remote-code")
    assert "--trust-remote-code" in out
    assert "--gpu-memory-utilization 0.9" in out


def test_vram_guard_noop_if_already_set(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    existing = "--gpu-memory-utilization 0.7"
    assert integrate_rebaseline._vram_guarded_server_args(existing) == existing


def test_vram_guard_defaults_on_with_its_cap(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.delenv("HL_INTEGRATE_VRAM_GUARD", raising=False)
    monkeypatch.setenv("HL_INTEGRATE_VRAM_UTIL_CAP", "0.85")
    assert integrate_rebaseline._vram_guarded_server_args("") == "--gpu-memory-utilization 0.85"


def test_vram_guard_cap_clamped(monkeypatch) -> None:
    monkeypatch.setenv("FRAMEWORK", "vllm")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_UTIL_CAP", "5.0")
    out = integrate_rebaseline._vram_guarded_server_args("")
    assert out == "--gpu-memory-utilization 0.99"


def test_vram_guard_sglang_is_noop(monkeypatch) -> None:
    # sglang rejects --gpu-memory-utilization; the guard must be a no-op for it.
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("HL_INTEGRATE_VRAM_GUARD", "1")
    assert integrate_rebaseline._vram_guarded_server_args("--trust-remote-code") == "--trust-remote-code"
    assert "gpu-memory-utilization" not in integrate_rebaseline._vram_guarded_server_args("")
