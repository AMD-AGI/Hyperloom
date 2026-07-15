# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Stage-1 parity tests for the benchmark backend seam.

These lock in that the default (Magpie) backend produces the exact command
line the executors used to hardcode, and that backend resolution defaults to
Magpie for unset/blank/unknown values.
"""

from __future__ import annotations

from pathlib import Path

from hyperloom.orchestrator.actions.executors import benchmark_backend as bb


def test_default_backend_name_is_magpie(monkeypatch):
    monkeypatch.delenv(bb.BENCHMARK_BACKEND_ENV, raising=False)
    assert bb.resolve_backend_name() == "magpie"
    assert bb.resolve_backend().name == "magpie"


def test_blank_backend_resolves_to_magpie(monkeypatch):
    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "   ")
    assert bb.resolve_backend_name() == "magpie"


def test_unknown_backend_falls_back_to_magpie(monkeypatch):
    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "does-not-exist")
    assert bb.resolve_backend_name() == "magpie"
    assert bb.resolve_backend().name == "magpie"


def test_magpie_command_is_byte_identical(monkeypatch):
    monkeypatch.delenv(bb.BENCHMARK_BACKEND_ENV, raising=False)
    cmd = bb.build_benchmark_command(
        python_exe="/opt/venv/bin/python",
        config_path=Path("/work/config.yaml"),
        output_dir=Path("/work/out"),
    )
    assert cmd == [
        "/opt/venv/bin/python",
        "-m",
        "Magpie",
        "-v",
        "benchmark",
        "--benchmark-config",
        "/work/config.yaml",
        "--output-dir",
        "/work/out",
        "--run-mode",
        "local",
    ]


def test_case_insensitive_backend_name(monkeypatch):
    monkeypatch.setenv(bb.BENCHMARK_BACKEND_ENV, "MAGPIE")
    assert bb.resolve_backend_name() == "magpie"
    assert bb.resolve_backend().name == "magpie"