# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Handlers that continue on an expected failure catch exactly that failure and let anything else propagate."""

from __future__ import annotations

import importlib.abc
import importlib.metadata
import importlib.util
import sys
import types
from pathlib import Path

import httpx
import pytest

import hyperloom.inference_optimizer as io_pkg
from hyperloom.inference_optimizer import cli as io_cli
from hyperloom.inference_optimizer.recipe_snapshot_constants import (
    DEFAULT_FRAMEWORK_VERSION_SLUG,
    detect_framework_version,
)
from hyperloom.inference_optimizer.tools import backfill_langfuse

_SCRIPTS = Path(io_pkg.__file__).parent / "multi_node" / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pod_patcher():
    return _load_script("apply_tracelens_patch_multinode")


@pytest.fixture
def shape_gate():
    return _load_script("sglang_shape_gate")


class _ExplodingFinder(importlib.abc.MetaPathFinder):
    """Makes importing one top-level module raise a RuntimeError instead of an ImportError."""

    def __init__(self, name: str) -> None:
        self._name = name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._name:
            raise RuntimeError(f"{fullname} is broken")
        return


@pytest.fixture
def break_import(monkeypatch):
    """Return a function that makes ``import <name>`` raise RuntimeError."""

    def _break(name: str) -> None:
        monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(sys, "meta_path", [_ExplodingFinder(name), *sys.meta_path])

    return _break


@pytest.fixture
def missing_import(monkeypatch):
    """Return a function that makes ``import <name>`` raise ImportError."""

    def _missing(name: str) -> None:
        monkeypatch.setitem(sys.modules, name, None)

    return _missing


# ---- cli._probe_llm_catalog ----


@pytest.fixture
def catalog_probe(monkeypatch):
    monkeypatch.setattr(io_cli, "_CATALOG_RETRY_DELAYS_SEC", ())

    def _probe():
        return io_cli._probe_llm_catalog(base_url="http://gateway", api_key="k")

    return _probe


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.InvalidURL("bad url")],
)
def test_catalog_probe_gives_up_on_a_transport_failure(monkeypatch, catalog_probe, failure):
    def fake_get(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(httpx, "get", fake_get)

    assert catalog_probe() is None


def test_catalog_probe_lets_an_unexpected_error_propagate(monkeypatch, catalog_probe):
    def fake_get(*_args, **_kwargs):
        raise RuntimeError("client bug")

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(RuntimeError, match="client bug"):
        catalog_probe()


# ---- pod-side apply_tracelens_patch_multinode ----


def test_installed_version_falls_back_to_the_module_attribute(pod_patcher, monkeypatch, missing_import):
    missing_import("sglang.version")

    def not_installed(_name):
        raise importlib.metadata.PackageNotFoundError("sglang")

    monkeypatch.setattr(importlib.metadata, "version", not_installed)

    assert pod_patcher._installed_sglang_version(types.SimpleNamespace(__version__=" 0.5.9 ")) == "0.5.9"


def test_installed_version_lets_a_broken_version_module_propagate(pod_patcher, monkeypatch):
    broken = types.ModuleType("sglang.version")

    def explode(name):
        raise RuntimeError("sglang.version is broken")

    broken.__getattr__ = explode
    monkeypatch.setitem(sys.modules, "sglang.version", broken)

    with pytest.raises(RuntimeError, match="sglang.version is broken"):
        pod_patcher._installed_sglang_version(types.SimpleNamespace(__version__="0.5.9"))


def test_installed_version_lets_a_metadata_error_propagate(pod_patcher, monkeypatch, missing_import):
    missing_import("sglang.version")

    def explode(_name):
        raise RuntimeError("metadata is broken")

    monkeypatch.setattr(importlib.metadata, "version", explode)

    with pytest.raises(RuntimeError, match="metadata is broken"):
        pod_patcher._installed_sglang_version(types.SimpleNamespace(__version__="0.5.9"))


def test_anchor_path_falls_back_to_the_package_path(pod_patcher, tmp_path, missing_import):
    missing_import("sglang.srt.managers.scheduler_profiler_mixin")
    module = types.SimpleNamespace(__file__=None, __path__=[str(tmp_path)])

    assert pod_patcher._sglang_anchor_path(module) == tmp_path


def test_anchor_path_lets_a_broken_profiler_module_propagate(pod_patcher, monkeypatch):
    for name in ("sglang", "sglang.srt", "sglang.srt.managers"):
        parent = types.ModuleType(name)
        parent.__path__ = []
        monkeypatch.setitem(sys.modules, name, parent)
    leaf = "sglang.srt.managers.scheduler_profiler_mixin"

    class _BrokenModule(types.ModuleType):
        @property
        def __file__(self):
            raise RuntimeError("profiler mixin is broken")

    monkeypatch.setitem(sys.modules, leaf, _BrokenModule(leaf))
    module = types.SimpleNamespace(__file__=None, __path__=[])

    with pytest.raises(RuntimeError, match="profiler mixin is broken"):
        pod_patcher._sglang_anchor_path(module)


def _apply(pod_patcher, tmp_path):
    return pod_patcher._apply_on_pod(
        tracelens_root=str(tmp_path),
        tracelens_internal_root=str(tmp_path),
        sglang_version_pin=None,
    )


def test_apply_reports_sglang_that_cannot_be_imported(pod_patcher, tmp_path, missing_import):
    missing_import("sglang")

    result = _apply(pod_patcher, tmp_path)

    assert result["status"] == "failed"
    assert result["error"] == "sglang not importable: import of sglang halted; None in sys.modules"


def test_apply_records_an_unexpected_import_error_as_unexpected(pod_patcher, tmp_path, break_import):
    break_import("sglang")

    result = _apply(pod_patcher, tmp_path)

    assert result["status"] == "failed"
    assert result["error"] == "unexpected: RuntimeError: sglang is broken"


# ---- pod-side sglang_shape_gate ----


@pytest.mark.parametrize(("pin", "mode"), [("0.5.18", "sitecustomize"), ("0.5.17", "patched")])
def test_shape_mode_uses_the_pinned_version_without_sglang(shape_gate, monkeypatch, missing_import, pin, mode):
    missing_import("sglang")
    monkeypatch.delenv("HYPERLOOM_SGLANG_SHAPE_MODE", raising=False)
    monkeypatch.setenv("HYPERLOOM_SGLANG_VERSION_PIN", pin)

    assert shape_gate.sglang_shape_mode() == mode


def test_shape_mode_lets_an_unexpected_import_error_propagate(shape_gate, monkeypatch, break_import):
    break_import("sglang")
    monkeypatch.delenv("HYPERLOOM_SGLANG_SHAPE_MODE", raising=False)

    with pytest.raises(RuntimeError, match="sglang is broken"):
        shape_gate.sglang_shape_mode()


# ---- recipe_snapshot_constants.detect_framework_version ----


def test_framework_version_lets_an_unexpected_import_error_propagate(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_RESOLVED_FRAMEWORK", raising=False)
    monkeypatch.delenv("HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_ExplodingFinder("vllm"), *sys.meta_path])

    with pytest.raises(RuntimeError, match="vllm is broken"):
        detect_framework_version("vllm")


def test_framework_version_defaults_when_the_framework_is_not_installed(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_RESOLVED_FRAMEWORK", raising=False)
    monkeypatch.delenv("HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON", raising=False)
    monkeypatch.setitem(sys.modules, "vllm", None)

    assert detect_framework_version("vllm") == DEFAULT_FRAMEWORK_VERSION_SLUG


# ---- tools.backfill_langfuse.ingest ----


def test_ingest_reports_a_missing_sdk_with_exit_code_3(missing_import, capsys):
    missing_import("langfuse")

    assert backfill_langfuse.ingest({}) == 3
    assert "ERROR: langfuse SDK not importable." in capsys.readouterr().err


def test_ingest_lets_an_unexpected_import_error_propagate(break_import):
    break_import("langfuse")

    with pytest.raises(RuntimeError, match="langfuse is broken"):
        backfill_langfuse.ingest({})
