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
from types import ModuleType, SimpleNamespace

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
    assert shim.counts() == {
        "compiled": 0,
        "loaded": 0,
        "invoked": 0,
        "boundary": 0,
        "timed_invoked": 0,
        "timed_boundary": 0,
        "timed_eager_boundary": 0,
    }


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


def test_every_call_is_counted_not_just_the_first(shim: ModuleType, tmp_path: Path, monkeypatch) -> None:
    """The wrapper used to restore the original after one call, which left ``invoked``
    saturated at 1 -- indistinguishable from a module that calls the package once to
    check it loads and then serves every inference from the eager model."""
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(evidence_dir(tmp_path)))
    shim.install_from_env()

    class _Package:
        def __call__(self, *args):
            return args

    assert shim._wrap(_Package, "__call__", "invoked", timed_key="timed_invoked") is True
    package = _Package()
    for _ in range(5):
        package()

    assert shim.counts()["invoked"] == 5


def test_only_the_timed_window_counts_towards_the_timed_figures(shim: ModuleType, tmp_path: Path, monkeypatch) -> None:
    """A warm-up that exports cannot stand in for a timed loop that does not."""
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(evidence_dir(tmp_path)))
    shim.install_from_env()
    shim._bump("invoked", timed_key="timed_invoked")  # warm-up
    shim.timed_window(True)
    shim._bump("invoked", timed_key="timed_invoked")
    shim.note_boundary_call()
    shim.timed_window(False)
    shim.note_boundary_call()  # the accuracy leg, after the clock stops

    counts = shim.counts()
    assert (counts["invoked"], counts["timed_invoked"]) == (2, 1)
    assert (counts["boundary"], counts["timed_boundary"]) == (2, 1)


def test_closing_the_window_publishes(shim: ModuleType, tmp_path: Path, monkeypatch) -> None:
    """A timed loop that overran its deadline is still judged on what it ran, and the
    process that ran it may be killed before ``atexit`` gets a turn."""
    out = evidence_dir(tmp_path)
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(out))
    shim.install_from_env()
    shim._INSTRUMENTED = True
    shim.timed_window(True)
    for _ in range(3):
        # The harness marks before it calls, so the package call falls in the interval
        # the mark opened; see ``harness.py``.
        shim.note_boundary_call()
        shim._bump("invoked", timed_key="timed_invoked", boundary_work=True)
    shim.timed_window(False)

    evidence = read_evidence(tmp_path)
    assert (evidence.timed_invoked, evidence.timed_boundary, evidence.exported) == (3, 3, True)
    assert evidence.timed_eager_boundary == 0


def test_a_module_exporting_only_some_calls_is_refused(shim: ModuleType, tmp_path: Path, monkeypatch) -> None:
    """Totals alone pass a partial fallback. A module calling the package k times per
    inference can serve (k-1)/k of the timed calls eagerly and still report
    ``timed_invoked >= timed_boundary`` -- a flow-matching policy whose denoising step
    runs ten times per forward is that shape. Each interval is judged on its own."""
    out = evidence_dir(tmp_path)
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(out))
    shim.install_from_env()
    shim._INSTRUMENTED = True
    shim.timed_window(True)
    for index in range(100):
        shim.note_boundary_call()
        if index < 10:  # ten exported calls of ten package calls each, ninety eager
            for _ in range(10):
                shim._bump("invoked", timed_key="timed_invoked", boundary_work=True)
    shim.timed_window(False)

    evidence = read_evidence(tmp_path)
    assert (evidence.timed_invoked, evidence.timed_boundary) == (100, 100)
    assert evidence.timed_invoked >= evidence.timed_boundary, "the totals rule alone would pass this"
    assert evidence.timed_eager_boundary == 90
    assert not evidence.exported


@pytest.mark.parametrize(
    ("timed_invoked", "timed_boundary", "exported"),
    [
        (200, 200, True),
        (400, 200, True),  # an exported sub-block called more than once per inference
        (1, 200, False),  # loaded the package, then served every inference from eager
        (0, 200, False),
    ],
)
def test_the_timed_window_decides_when_it_reported(
    tmp_path: Path, timed_invoked: int, timed_boundary: int, exported: bool
) -> None:
    out = evidence_dir(tmp_path)
    out.mkdir(parents=True)
    (out / "1.json").write_text(
        json.dumps(
            {
                "schema": "hyperloom.aoti_evidence/2",
                "instrumented": True,
                "invoked": timed_invoked,
                "timed_invoked": timed_invoked,
                "timed_boundary": timed_boundary,
            }
        ),
        encoding="utf-8",
    )
    assert read_evidence(tmp_path).exported is exported


def test_the_hot_path_publishes_once_not_per_call(shim: ModuleType, tmp_path: Path, monkeypatch) -> None:
    """Why the wrapper can stay installed at all. Publishing every call cost 4.4x on the
    inference path (34.8 -> 154.5 us, measured); counting alone is 0.4 us, and the counts
    reach disk when the window closes and again at exit."""
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(evidence_dir(tmp_path)))
    shim.install_from_env()
    writes: list[dict] = []
    monkeypatch.setattr(shim, "_write", writes.append)

    for _ in range(50):
        shim._bump("invoked", timed_key="timed_invoked")

    assert len(writes) == 1
    assert shim.counts()["invoked"] == 50


_HARNESS = Path(__file__).resolve().parents[1] / "assets" / "script_workload" / "harness.py"


def _harness_with(monkeypatch: pytest.MonkeyPatch, recorder) -> ModuleType:
    """The harness asset with everything but its timed loop stubbed out.

    Loaded by path, as ``benchmark.sh`` runs it. Its timed loop is the only thing the
    counters can be read against, and it is torch-only, so the surface around it stands
    in: this pins which calls the loop makes and in what order, not what torch does.
    """
    spec = importlib.util.spec_from_file_location("hl_harness_under_test", _HARNESS)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    monkeypatch.setattr(harness, "torch", SimpleNamespace(manual_seed=lambda _seed: None))
    monkeypatch.setattr(harness, "_aoti_evidence", lambda: recorder)
    monkeypatch.setattr(harness, "_synchronize", lambda: None)
    monkeypatch.setattr(harness, "_send_mark", lambda _text: None)
    monkeypatch.setattr(harness, "_chunk_passes", lambda _calls, passes: passes)
    monkeypatch.setattr(harness, "_pass_inputs", lambda _c, _h, _s, _p: [None])
    monkeypatch.setattr(harness, "_invoke", lambda _entry, _call, _inputs=None: None)
    monkeypatch.setattr(harness, "_snapshot", lambda output: output)
    monkeypatch.setattr(harness, "_capture", lambda output: output)
    return harness


def test_the_harness_counts_its_boundary_calls_inside_an_open_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without this the shim's timed figures stay at zero and the gate falls back to the
    weaker ``invoked > 0``, which is the hole the window exists to close. The window has to
    open before the first call and close after the last, or the figures describe the wrong
    region."""
    events: list[str] = []
    recorder = SimpleNamespace(
        timed_window=lambda open_: events.append("open" if open_ else "close"),
        note_boundary_call=lambda: events.append("call"),
    )
    harness = _harness_with(monkeypatch, recorder)

    harness._timed_passes(object(), [{"args": (), "kwargs": {}}], [[None]], {"timed_passes": 3})

    assert events == ["open", "call", "call", "call", "close"]


def test_an_unarmed_run_leaves_the_timed_loop_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every scriptable workload runs this loop; only one that declared an export is armed."""
    harness = _harness_with(monkeypatch, None)

    latencies, _ = harness._timed_passes(object(), [{"args": (), "kwargs": {}}], [[None]], {"timed_passes": 2})

    assert len(latencies) == 2
