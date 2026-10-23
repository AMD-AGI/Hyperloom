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
_PROFILER_MIXIN = "sglang.srt.managers.scheduler_profiler_mixin"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ExplodingFinder(importlib.abc.MetaPathFinder):
    def __init__(self, name: str, error: type[Exception]) -> None:
        self._name = name
        self._error = error

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self._name:
            raise self._error(f"{fullname} is broken")


@pytest.fixture
def break_import(monkeypatch):
    """Return a function that makes ``import <name>`` raise ``error`` (RuntimeError by default)."""

    def _break(name: str, error: type[Exception] = RuntimeError) -> None:
        parts = name.split(".")
        for depth in range(1, len(parts)):
            parent = types.ModuleType(".".join(parts[:depth]))
            parent.__path__ = []
            monkeypatch.setitem(sys.modules, parent.__name__, parent)
        monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(sys, "meta_path", [_ExplodingFinder(name, error), *sys.meta_path])

    return _break


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("HYPERLOOM_RESOLVED_FRAMEWORK", "HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    # The pod scripts import their shipped siblings by name.
    monkeypatch.syspath_prepend(str(_SCRIPTS))


def _patcher(function: str, *args):
    return getattr(_load_script("apply_tracelens_patch_multinode"), function)(*args)


_NO_FILE_SGLANG = types.SimpleNamespace(__file__=None, __path__=[], __version__="0.5.9")
_IMPORT_SITES = {
    "shape_mode": ("sglang", lambda: _load_script("sglang_shape_gate").sglang_shape_mode()),
    "framework_version": ("vllm", lambda: detect_framework_version("vllm")),
    "backfill_ingest": ("langfuse", lambda: backfill_langfuse.ingest({})),
    "installed_version": ("sglang.version", lambda: _patcher("_installed_sglang_version", _NO_FILE_SGLANG)),
    "anchor_path": (_PROFILER_MIXIN, lambda: _patcher("_sglang_anchor_path", _NO_FILE_SGLANG)),
}


@pytest.mark.parametrize(("module", "call"), _IMPORT_SITES.values(), ids=_IMPORT_SITES)
def test_unexpected_import_error_propagates(break_import, module, call):
    break_import(module)

    with pytest.raises(RuntimeError, match=f"{module} is broken"):
        call()


def test_installed_version_lets_a_metadata_error_propagate(monkeypatch, break_import):
    break_import("sglang.version", ImportError)

    def explode(_name):
        raise RuntimeError("metadata is broken")

    monkeypatch.setattr(importlib.metadata, "version", explode)

    with pytest.raises(RuntimeError, match="metadata is broken"):
        _patcher("_installed_sglang_version", _NO_FILE_SGLANG)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ImportError, "sglang not importable: sglang is broken"),
        (OSError, "sglang not importable: sglang is broken"),
        (RuntimeError, "unexpected: RuntimeError: sglang is broken"),
    ],
)
def test_apply_reports_a_failing_sglang_import(tmp_path, break_import, error, message):
    break_import("sglang", error)

    result = _load_script("apply_tracelens_patch_multinode")._apply_on_pod(
        tracelens_root=str(tmp_path), sglang_version_pin=None
    )

    assert (result["status"], result["error"]) == ("failed", message)


def _anchor_path(tmp):
    return _patcher("_sglang_anchor_path", types.SimpleNamespace(__file__=None, __path__=[str(tmp)]))


_EXPECTED_IMPORT_FAILURES = [
    ("sglang", ImportError, lambda _tmp: _load_script("sglang_shape_gate").sglang_shape_mode(), "sitecustomize"),
    ("sglang", OSError, lambda _tmp: _load_script("sglang_shape_gate").sglang_shape_mode(), "sitecustomize"),
    ("vllm", ImportError, lambda _tmp: detect_framework_version("vllm"), DEFAULT_FRAMEWORK_VERSION_SLUG),
    ("vllm", OSError, lambda _tmp: detect_framework_version("vllm"), DEFAULT_FRAMEWORK_VERSION_SLUG),
    ("langfuse", ImportError, lambda _tmp: backfill_langfuse.ingest({}), 3),
    (
        "sglang.version",
        ImportError,
        lambda _tmp: _patcher("_installed_sglang_version", types.SimpleNamespace(__version__=" 0.5.9 ")),
        "0.5.9",
    ),
    (_PROFILER_MIXIN, ImportError, _anchor_path, None),
]


@pytest.mark.parametrize(("module", "error", "call", "expected"), _EXPECTED_IMPORT_FAILURES)
def test_an_expected_import_failure_falls_back(tmp_path, monkeypatch, break_import, module, error, call, expected):
    break_import(module, error)
    monkeypatch.setenv("HYPERLOOM_SGLANG_VERSION_PIN", "0.5.18")
    monkeypatch.setattr(importlib.metadata, "version", _raising_not_installed)

    assert call(tmp_path) == (tmp_path if expected is None else expected)


def _raising_not_installed(name):
    raise importlib.metadata.PackageNotFoundError(name)


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
