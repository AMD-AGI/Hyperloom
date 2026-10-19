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
    """Makes importing one top-level module raise ``error``."""

    def __init__(self, name: str, error: type[Exception]) -> None:
        self._name = name
        self._error = error

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._name:
            raise self._error(f"{fullname} is broken")
        return


@pytest.fixture
def break_import(monkeypatch):
    """Return a function that makes ``import <name>`` raise ``error`` (RuntimeError by default)."""

    def _break(name: str, error: type[Exception] = RuntimeError) -> None:
        monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(sys, "meta_path", [_ExplodingFinder(name, error), *sys.meta_path])

    return _break


@pytest.fixture
def missing_import(monkeypatch):
    """Return a function that makes ``import <name>`` raise ImportError."""

    def _missing(name: str) -> None:
        monkeypatch.setitem(sys.modules, name, None)

    return _missing


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("HYPERLOOM_SGLANG_SHAPE_MODE", "HYPERLOOM_RESOLVED_FRAMEWORK", "HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON"):
        monkeypatch.delenv(name, raising=False)


_IMPORT_SITES = {
    "shape_mode": ("sglang", lambda: _load_script("sglang_shape_gate").sglang_shape_mode()),
    "framework_version": ("vllm", lambda: detect_framework_version("vllm")),
    "backfill_ingest": ("langfuse", lambda: backfill_langfuse.ingest({})),
}


@pytest.mark.parametrize(("module", "call"), _IMPORT_SITES.values(), ids=_IMPORT_SITES)
def test_unexpected_import_error_propagates(break_import, module, call):
    break_import(module)

    with pytest.raises(RuntimeError, match=f"{module} is broken"):
        call()


# ---- cli._probe_llm_catalog ----


@pytest.fixture
def catalog_probe(monkeypatch):
    monkeypatch.setattr(io_cli, "_CATALOG_RETRY_DELAYS_SEC", ())

    def _probe(failure):
        def fake_get(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(httpx, "get", fake_get)
        return io_cli._probe_llm_catalog(base_url="http://gateway", api_key="k")

    return _probe


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.InvalidURL("bad url")],
)
def test_catalog_probe_gives_up_on_a_transport_failure(catalog_probe, failure):
    assert catalog_probe(failure) is None


def test_catalog_probe_lets_an_unexpected_error_propagate(catalog_probe):
    with pytest.raises(RuntimeError, match="client bug"):
        catalog_probe(RuntimeError("client bug"))


# ---- pod-side apply_tracelens_patch_multinode ----


def test_installed_version_falls_back_to_the_module_attribute(pod_patcher, monkeypatch, missing_import):
    missing_import("sglang.version")

    def not_installed(_name):
        raise importlib.metadata.PackageNotFoundError("sglang")

    monkeypatch.setattr(importlib.metadata, "version", not_installed)

    assert pod_patcher._installed_sglang_version(types.SimpleNamespace(__version__=" 0.5.9 ")) == "0.5.9"


@pytest.mark.parametrize("broken", ["version_module", "metadata"])
def test_installed_version_lets_an_unexpected_error_propagate(pod_patcher, monkeypatch, missing_import, broken):
    def explode(*_args):
        raise RuntimeError(f"{broken} is broken")

    if broken == "metadata":
        missing_import("sglang.version")
        monkeypatch.setattr(importlib.metadata, "version", explode)
    else:
        module = types.ModuleType("sglang.version")
        module.__getattr__ = explode
        monkeypatch.setitem(sys.modules, "sglang.version", module)

    with pytest.raises(RuntimeError, match=f"{broken} is broken"):
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

    with pytest.raises(RuntimeError, match="profiler mixin is broken"):
        pod_patcher._sglang_anchor_path(types.SimpleNamespace(__file__=None, __path__=[]))


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ImportError, "sglang not importable: sglang is broken"),
        (OSError, "sglang not importable: sglang is broken"),
        (RuntimeError, "unexpected: RuntimeError: sglang is broken"),
    ],
)
def test_apply_reports_a_failing_sglang_import(pod_patcher, tmp_path, break_import, error, message):
    break_import("sglang", error)

    result = pod_patcher._apply_on_pod(
        tracelens_root=str(tmp_path),
        tracelens_internal_root=str(tmp_path),
        sglang_version_pin=None,
    )

    assert result["status"] == "failed"
    assert result["error"] == message


# ---- pod-side sglang_shape_gate ----


@pytest.mark.parametrize(
    ("error", "pin", "mode"),
    [
        (ImportError, "0.5.18", "sitecustomize"),
        (ImportError, "0.5.17", "patched"),
        (OSError, "0.5.18", "sitecustomize"),
    ],
)
def test_shape_mode_uses_the_pinned_version_without_sglang(shape_gate, monkeypatch, break_import, error, pin, mode):
    break_import("sglang", error)
    monkeypatch.setenv("HYPERLOOM_SGLANG_VERSION_PIN", pin)

    assert shape_gate.sglang_shape_mode() == mode


# ---- recipe_snapshot_constants.detect_framework_version ----


@pytest.mark.parametrize("error", [ImportError, OSError])
def test_framework_version_defaults_when_the_framework_cannot_load(break_import, error):
    break_import("vllm", error)

    assert detect_framework_version("vllm") == DEFAULT_FRAMEWORK_VERSION_SLUG


# ---- tools.backfill_langfuse.ingest ----


def test_ingest_reports_a_missing_sdk_with_exit_code_3(missing_import):
    missing_import("langfuse")

    assert backfill_langfuse.ingest({}) == 3
