# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The start-up shim that counts what a benchmark's processes actually did.

It runs inside the operator's model environment, before and independently of torch, so
it imports only the standard library and must survive a process where torch is never
imported at all. Loaded by file path here, the way ``sitecustomize`` loads it, rather
than as a package module -- it ships as a bare asset on ``PYTHONPATH``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from hyperloom.orchestrator.measurement.aoti_evidence import EVIDENCE_DIR_ENV, evidence_dir, read_evidence

_SHIM = Path(__file__).resolve().parents[1] / "assets" / "host_probe" / "hl_aoti_evidence.py"


@pytest.fixture
def shim() -> ModuleType:
    """A fresh copy of the shim; its counters are module-global by design."""
    spec = importlib.util.spec_from_file_location("hl_aoti_evidence_under_test", _SHIM)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    sys.meta_path[:] = [hook for hook in sys.meta_path if type(hook).__module__ != module.__name__]


def test_it_imports_without_torch(shim: ModuleType) -> None:
    """The benchmark child loads it before torch, and may never import torch at all."""
    assert shim.counts() == {"compiled": 0, "loaded": 0, "invoked": 0}


def test_unarmed_it_does_nothing(shim: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(EVIDENCE_DIR_ENV, raising=False)
    assert shim.install_from_env() is False


def test_arming_is_idempotent(shim: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(tmp_path))
    assert shim.install_from_env() is True
    assert shim.install_from_env() is True


def test_an_armed_shim_writes_counters_the_aggregator_reads(
    shim: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The two halves are a contract: one writes per-process files, the other sums them."""
    # The pairing the orchestrator sets up: the shim writes into evidence_dir(workspace),
    # which is exactly where read_evidence looks.
    out = evidence_dir(tmp_path)
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(out))
    shim.install_from_env()
    shim._INSTRUMENTED = True
    shim._write({"compiled": 1, "loaded": 1, "invoked": 7})

    written = list(out.glob("*.json"))
    assert len(written) == 1, "one file per process, named for the pid"
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["schema"] == shim.SCHEMA
    assert payload["instrumented"] is True

    evidence = read_evidence(tmp_path)
    assert (evidence.observed, evidence.invoked, evidence.exported) == (True, 7, True)


def test_a_process_that_wrapped_nothing_is_not_counted_as_zero(
    shim: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It instrumented nothing, so its zeros are evidence of nothing -- not of no export."""
    out = evidence_dir(tmp_path)
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(out))
    shim.install_from_env()
    shim._write(shim.counts())

    payload = json.loads(next(iter(out.glob("*.json"))).read_text(encoding="utf-8"))
    assert payload["instrumented"] is False
    assert read_evidence(tmp_path).observed is False


def test_an_unwritable_directory_does_not_take_the_benchmark_down(
    shim: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shim rides inside the operator's process; losing evidence beats losing the run."""
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(tmp_path / "file-not-a-dir"))
    (tmp_path / "file-not-a-dir").write_text("", encoding="utf-8")
    shim.install_from_env()
    shim._write(shim.counts())  # must not raise
