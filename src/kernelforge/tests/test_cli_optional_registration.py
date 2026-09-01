# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Failure semantics for the eagerly registered `gemm-tune` command group.

Upstream KernelForge registered this group defensively, because the tuner was
a separate distribution back then and a root install could intentionally omit
it. Vendored into Hyperloom it is a subpackage of the same
wheel, so there is no such thing as a deliberate absence: if the import fails,
the installation is broken and the run must say so rather than hand back a CLI
that is quietly missing a subcommand and then dies mid-tuning on "No such
command 'gemm-tune'". These tests pin that decision down, along with the part
of upstream's reasoning that survives it -- an error raised *inside* the
subpackage is never a "missing command".
"""

from __future__ import annotations

import builtins

import pytest

import kernelforge.cli as cli


def _break_import(monkeypatch, exc: BaseException) -> None:
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "kernelforge.gemm_tune.cli":
            raise exc
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


def test_missing_gemm_tune_subpackage_is_fatal(monkeypatch):
    """A wheel without its own subpackage is broken, not configured that way."""
    _break_import(monkeypatch, ModuleNotFoundError("no kernelforge.gemm_tune", name="kernelforge.gemm_tune"))
    with pytest.raises(ModuleNotFoundError, match="kernelforge.gemm_tune"):
        cli._register_gemm_tune()


def test_gemm_tune_internal_import_error_is_not_hidden(monkeypatch):
    """A missing transitive dependency keeps its own name and traceback."""
    _break_import(monkeypatch, ModuleNotFoundError("no transitive_dependency", name="transitive_dependency"))
    with pytest.raises(ModuleNotFoundError, match="transitive_dependency"):
        cli._register_gemm_tune()


def test_gemm_tune_syntax_error_is_not_hidden(monkeypatch):
    _break_import(monkeypatch, SyntaxError("broken optional command"))
    with pytest.raises(SyntaxError, match="broken optional command"):
        cli._register_gemm_tune()


def test_gemm_tune_is_actually_registered():
    """The guard above is only meaningful if the happy path really registers."""
    assert "gemm-tune" in main_commands(), "gemm-tune must be on the CLI after import"


def main_commands() -> dict:
    return getattr(cli.main, "commands", {})
