# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``_xdit_patcher.ensure_xdit_profiler_patched``.

Pins the patcher contract: two backward-compatible, idempotent, concurrency-safe,
fail-soft patches on xDiT's ``base_model.py``:

* Patch 1 inserts ``repeat=1`` into ``torch.profiler.schedule`` so the diffusion
  profiler retains its active window (upstream default ``repeat=0`` discards it
  -> empty trace, Op count == 0).
* Patch 2 wraps ``self.pipe.scheduler.step`` in a ``record_function`` marker so
  each denoise step is annotated (``denoise_step_<i>``), giving TraceLens
  deterministic per-step roofline split boundaries.

Fixtures synthesize a fake xfuser tree in ``tmp_path`` discovered via
``$XDIT_PATH``. The fixture mirrors the real ``profile()`` shape (multi-line
``with profile(...)`` + the ``model_inference`` marker) so both patch anchors are
exercised.
"""

from __future__ import annotations

import ast
import importlib.util
import threading
from pathlib import Path

import pytest

from hyperloom.orchestrator.actions.executors import _xdit_patcher
from hyperloom.orchestrator.actions.executors._xdit_patcher import (
    _apply_annotation_patch,
    _apply_patch_atomic,
    _apply_repeat_patch,
    _discover_xfuser_base_models,
    _is_patched,
    ensure_xdit_profiler_patched,
)

# Verbatim upstream shape (incl. the exact indentation the patcher matches on):
# the multi-line ``with profile(...)`` block and the per-image
# ``record_function("model_inference")`` marker, matching runner_models/base_model.py.
_UPSTREAM_FIXTURE = """\
class xFuserModel:
    def profile(self, input_args):
        schedule = torch.profiler.schedule(
            wait=self.config.profile_wait,
            warmup=self.config.profile_warmup,
            active=self.config.profile_active,
        )
        num_repetitions = self.config.profile_wait + self.config.profile_warmup + self.config.profile_active
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule,
            record_shapes=True,
            with_stack=False,
        ) as profile_object:
            for iteration in range(num_repetitions):
                log(f"Profiling iteration {iteration + 1}/{num_repetitions}")
                with record_function("model_inference"):
                    output, timing = self._run_timed_pipe(input_args)
                profile_object.step()
"""

_REL = ("xfuser", "model_executor", "models", "runner_models", "base_model.py")
_SENTINEL = "# hyperloom: retain active window"
_ANNOT_SENTINEL = "# hyperloom: per-denoise-step annotation"


@pytest.fixture(autouse=True)
def _isolate_xdit_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear ``$XDIT_PATH`` so a synthetic ``tmp_path`` test never discovers a
    real on-pod xDiT checkout (tests that exercise discovery re-set it)."""
    monkeypatch.delenv("XDIT_PATH", raising=False)


@pytest.fixture
def fake_xdit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a minimal ``$XDIT_PATH/xfuser/.../base_model.py`` tree."""
    target = tmp_path.joinpath(*_REL)
    target.parent.mkdir(parents=True)
    target.write_text(_UPSTREAM_FIXTURE, encoding="utf-8")
    monkeypatch.setenv("XDIT_PATH", str(tmp_path))
    return target


def test_first_call_inserts_repeat_one(fake_xdit: Path) -> None:
    rc = ensure_xdit_profiler_patched()
    assert rc is True
    text = fake_xdit.read_text()
    assert "repeat=1," in text
    assert _SENTINEL in text


def test_second_call_is_a_noop(fake_xdit: Path) -> None:
    """Idempotency: re-applying must not double-patch or change bytes."""
    ensure_xdit_profiler_patched()
    after_first = fake_xdit.read_text()
    rc = ensure_xdit_profiler_patched()
    assert rc is True
    assert fake_xdit.read_text() == after_first


def test_repeat_appears_exactly_once(fake_xdit: Path) -> None:
    for _ in range(5):
        ensure_xdit_profiler_patched()
    assert fake_xdit.read_text().count("repeat=1") == 1


def test_patched_file_is_valid_python(fake_xdit: Path) -> None:
    ensure_xdit_profiler_patched()
    ast.parse(fake_xdit.read_text())


# ---------------------------------------------------------------------------
# Patch 2: per-denoise-step annotation
# ---------------------------------------------------------------------------
def test_first_call_inserts_perstep_annotation(fake_xdit: Path) -> None:
    """The annotation patch wraps scheduler.step and inserts the per-image reset."""
    rc = ensure_xdit_profiler_patched()
    assert rc is True
    text = fake_xdit.read_text()
    assert _ANNOT_SENTINEL in text
    # scheduler.step is wrapped and each step emits a denoise_step_<i> marker.
    assert 'record_function(' in text
    assert 'denoise_step_' in text
    assert "_hl_target.step = _hl_wrapped_step" in text
    # The per-image reset call is inserted before the model_inference marker.
    assert "_hl_reset_denoise_counter()" in text
    idx_reset = text.index("_hl_reset_denoise_counter()  #")
    idx_marker = text.index('with record_function("model_inference"):')
    assert idx_reset < idx_marker
    # Still valid Python after both patches.
    ast.parse(text)


def _run_patched_profile(patched_src: str, scheduler, steps_per_iter: int):
    """Exec a patched ``base_model.py`` and drive ``profile()`` with stub
    torch/profiler primitives, returning the ordered list of ``record_function``
    marker names. Raises whatever the wrapped step raises (e.g. RecursionError)."""
    import contextlib

    markers: list[str] = []

    @contextlib.contextmanager
    def record_function(name):  # noqa: ANN001
        markers.append(name)
        yield

    class _Prof:
        def step(self):  # noqa: D401
            pass

    @contextlib.contextmanager
    def profile(*_a, **_k):  # noqa: ANN002, ANN003
        yield _Prof()

    class _Activities:
        CPU = 1
        CUDA = 2

    class _Torch:
        class profiler:
            @staticmethod
            def schedule(**_k):  # noqa: ANN003
                return None

    ns: dict = {
        "torch": _Torch,
        "profile": profile,
        "ProfilerActivity": _Activities,
        "record_function": record_function,
        "log": lambda *_a, **_k: None,
    }
    exec(compile(patched_src, "<patched_base_model>", "exec"), ns)  # noqa: S102
    model_cls = ns["xFuserModel"]
    model = model_cls.__new__(model_cls)

    class _Cfg:
        profile_wait = 0
        profile_warmup = 0
        profile_active = 2  # -> 2 profiled iterations

    class _Pipe:
        pass

    model.config = _Cfg()
    model.pipe = _Pipe()
    model.pipe.scheduler = scheduler

    def _run_timed_pipe(_args):  # noqa: ANN001
        for _ in range(steps_per_iter):
            model.pipe.scheduler.step()
        return ("out", 0.0)

    model._run_timed_pipe = _run_timed_pipe
    model.profile({})
    return markers


def test_annotation_no_recursion_with_xfuser_wrapper(fake_xdit: Path) -> None:
    """Regression: xfuser's scheduler wrapper delegates ``step`` to ``self.module``
    and its ``__setattr__`` redirects writes to ``.module``. Patching/capturing the
    wrapper's ``step`` (instead of the module's) caused infinite recursion on models
    whose ``pipe.scheduler`` is the xfuser wrapper (e.g. SD3.5-large-turbo). The
    patch must target ``.module`` so the wrapped call terminates."""
    ensure_xdit_profiler_patched()
    patched_src = fake_xdit.read_text()

    class FakeInnerScheduler:
        def __init__(self) -> None:
            self.calls = 0

        def step(self, *_a, **_k):  # noqa: ANN002, ANN003
            self.calls += 1
            return "ok"

    class FakeXFuserSchedulerWrapper:
        """Mirrors xFuserSchedulerBaseWrapper: delegates step to self.module and
        redirects attribute writes onto the module."""

        def __init__(self, module) -> None:  # noqa: ANN001
            object.__setattr__(self, "module", module)

        def __setattr__(self, name, value):  # noqa: ANN001
            if name == "module":
                object.__setattr__(self, name, value)
            elif getattr(self, "module", None) is not None and hasattr(self.module, name):
                setattr(self.module, name, value)
            else:
                object.__setattr__(self, name, value)

        def step(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return self.module.step(*args, **kwargs)

    inner = FakeInnerScheduler()
    wrapper = FakeXFuserSchedulerWrapper(inner)

    # Must not raise RecursionError; 2 iterations x 3 steps = 6 real step calls.
    markers = _run_patched_profile(patched_src, wrapper, steps_per_iter=3)

    assert inner.calls == 6
    # Per-iteration reset => denoise index restarts at 0 each profiled iteration.
    assert markers.count("denoise_step_0") == 2
    assert markers.count("denoise_step_1") == 2
    assert markers.count("denoise_step_2") == 2
    assert "denoise_step_3" not in markers


def test_annotation_no_recursion_with_plain_scheduler(fake_xdit: Path) -> None:
    """A plain diffusers scheduler (no ``.module``) is patched in place and still
    produces per-step markers without recursion."""
    ensure_xdit_profiler_patched()
    patched_src = fake_xdit.read_text()

    class PlainScheduler:
        def __init__(self) -> None:
            self.calls = 0

        def step(self, *_a, **_k):  # noqa: ANN002, ANN003
            self.calls += 1
            return "ok"

    sched = PlainScheduler()
    markers = _run_patched_profile(patched_src, sched, steps_per_iter=2)

    assert sched.calls == 4
    assert markers.count("denoise_step_0") == 2
    assert markers.count("denoise_step_1") == 2


def test_both_patches_present_after_apply(fake_xdit: Path) -> None:
    """A fully patched file carries both sentinels and reports patched."""
    ensure_xdit_profiler_patched()
    assert _is_patched(fake_xdit) is True
    text = fake_xdit.read_text()
    assert _SENTINEL in text and _ANNOT_SENTINEL in text


def test_is_patched_requires_both_sentinels(fake_xdit: Path) -> None:
    """A file with only Patch 1 (repeat) is NOT considered patched (upgrade path)."""
    repeat_only, applied = _apply_repeat_patch(fake_xdit.read_text(), fake_xdit)
    assert applied is True
    fake_xdit.write_text(repeat_only, encoding="utf-8")
    assert _SENTINEL in repeat_only
    assert _ANNOT_SENTINEL not in repeat_only
    assert _is_patched(fake_xdit) is False


def test_upgrade_from_repeat_only_adds_annotation(fake_xdit: Path) -> None:
    """An older-Hyperloom file (repeat=1 only) is upgraded to add Patch 2 without
    duplicating repeat=1."""
    repeat_only, _ = _apply_repeat_patch(fake_xdit.read_text(), fake_xdit)
    fake_xdit.write_text(repeat_only, encoding="utf-8")

    rc = ensure_xdit_profiler_patched()
    assert rc is True
    text = fake_xdit.read_text()
    assert text.count("repeat=1") == 1  # not re-applied
    assert _ANNOT_SENTINEL in text
    assert _is_patched(fake_xdit) is True
    ast.parse(text)


def test_annotation_appears_exactly_once(fake_xdit: Path) -> None:
    """Idempotency across many calls: single wrapper install, single reset call."""
    for _ in range(5):
        ensure_xdit_profiler_patched()
    text = fake_xdit.read_text()
    assert text.count("_hl_target.step = _hl_wrapped_step") == 1
    assert text.count("_hl_reset_denoise_counter()  #") == 1


def test_missing_annotation_anchor_still_applies_repeat(tmp_path: Path, monkeypatch) -> None:
    """A file with the schedule block but no ``model_inference`` marker still gets
    Patch 1 (repeat); Patch 2 fails soft and the file stays valid."""
    fixture_no_marker = (
        "class xFuserModel:\n"
        "    def profile(self, input_args):\n"
        "        schedule = torch.profiler.schedule(\n"
        "            wait=self.config.profile_wait,\n"
        "            warmup=self.config.profile_warmup,\n"
        "            active=self.config.profile_active,\n"
        "        )\n"
        "        with profile(schedule=schedule) as profile_object:\n"
        "            for iteration in range(3):\n"
        "                self._run_timed_pipe(input_args)\n"
        "                profile_object.step()\n"
    )
    target = tmp_path.joinpath(*_REL)
    target.parent.mkdir(parents=True)
    target.write_text(fixture_no_marker, encoding="utf-8")
    monkeypatch.setenv("XDIT_PATH", str(tmp_path))
    monkeypatch.setattr(
        _xdit_patcher, "_discover_xfuser_base_models", lambda: [target]
    )
    rc = ensure_xdit_profiler_patched()
    assert rc is True
    text = target.read_text()
    assert _SENTINEL in text  # Patch 1 landed
    assert _ANNOT_SENTINEL not in text  # Patch 2 failed soft (no marker anchor)
    assert _is_patched(target) is False
    ast.parse(text)


def test_apply_annotation_patch_is_noop_when_already_present(fake_xdit: Path) -> None:
    """``_apply_annotation_patch`` reports no change on an already-annotated text."""
    text = fake_xdit.read_text()
    text, _ = _apply_repeat_patch(text, fake_xdit)
    text, applied1 = _apply_annotation_patch(text, fake_xdit)
    assert applied1 is True
    text2, applied2 = _apply_annotation_patch(text, fake_xdit)
    assert applied2 is False
    assert text2 == text


def test_missing_legacy_block_is_fail_soft(tmp_path: Path, monkeypatch) -> None:
    """A file without the schedule block leaves it untouched, returns False."""
    target = tmp_path.joinpath(*_REL)
    target.parent.mkdir(parents=True)
    target.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setenv("XDIT_PATH", str(tmp_path))
    # Restrict discovery to the fake file so a real installed xfuser in the
    # environment cannot make the patcher report success for another root.
    monkeypatch.setattr(
        _xdit_patcher, "_discover_xfuser_base_models", lambda: [target]
    )
    assert ensure_xdit_profiler_patched() is False
    assert target.read_text() == "x = 1\n"


def test_no_xdit_tree_is_fail_soft(monkeypatch) -> None:
    """No discoverable base_model.py (empty $XDIT_PATH, no xfuser) -> False."""
    monkeypatch.delenv("XDIT_PATH", raising=False)
    monkeypatch.setattr(
        _xdit_patcher, "_discover_xfuser_base_models", lambda: []
    )
    assert ensure_xdit_profiler_patched() is False


# ---------------------------------------------------------------------------
# _discover_xfuser_base_models: discovery-path edge cases (fail-soft)
# ---------------------------------------------------------------------------
def test_discovery_skips_when_base_model_missing(tmp_path: Path, monkeypatch) -> None:
    """``$XDIT_PATH`` set but the ``base_model.py`` file is absent → skipped."""
    monkeypatch.setenv("XDIT_PATH", str(tmp_path))
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert _discover_xfuser_base_models() == []


def test_discovery_survives_find_spec_error(monkeypatch) -> None:
    """A raising ``importlib.util.find_spec`` must not propagate → returns []."""
    monkeypatch.delenv("XDIT_PATH", raising=False)

    def _boom(name: str):
        raise ValueError("namespace package without a real spec")

    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert _discover_xfuser_base_models() == []


def test_discovery_uses_importable_xfuser_spec(tmp_path: Path, monkeypatch) -> None:
    """When ``$XDIT_PATH`` is unset, discovery falls back to the importable
    ``xfuser`` package location (``find_spec.submodule_search_locations``)."""
    pkg_root = tmp_path / "site-packages"
    target = pkg_root.joinpath(*_REL)
    target.parent.mkdir(parents=True)
    target.write_text(_UPSTREAM_FIXTURE, encoding="utf-8")
    monkeypatch.delenv("XDIT_PATH", raising=False)

    class _Spec:
        submodule_search_locations = [str(pkg_root / "xfuser")]

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: _Spec())
    found = _discover_xfuser_base_models()
    assert target.resolve() in found


# ---------------------------------------------------------------------------
# _apply_patch_atomic: IO / no-op edge cases (fail-soft, never corrupts)
# ---------------------------------------------------------------------------
def test_apply_returns_false_on_read_error(tmp_path: Path) -> None:
    """An unreadable path (here a directory) fails soft, returns False."""
    unreadable = tmp_path / "a_directory_not_a_file"
    unreadable.mkdir()
    assert _apply_patch_atomic(unreadable) is False


def test_apply_returns_false_when_patch_is_a_noop(fake_xdit: Path, monkeypatch) -> None:
    """If both patch replacements yield identical bytes, return False."""
    # Neuter Patch 1 (repeat) and Patch 2 (annotation) so neither changes text.
    monkeypatch.setattr(
        _xdit_patcher, "_PATCHED_BLOCK", _xdit_patcher._LEGACY_BLOCK
    )
    monkeypatch.setattr(
        _xdit_patcher, "_ANNOT_INSTALL_BLOCK", _xdit_patcher._ANNOT_INSTALL_ANCHOR
    )
    monkeypatch.setattr(
        _xdit_patcher, "_ANNOT_LOOP_PATCHED", _xdit_patcher._ANNOT_LOOP_ANCHOR
    )
    assert _apply_patch_atomic(fake_xdit) is False


def test_apply_returns_false_when_write_fails(fake_xdit: Path, monkeypatch) -> None:
    """A failed atomic write is reported (False) without touching the file."""
    original = fake_xdit.read_text()
    monkeypatch.setattr(
        _xdit_patcher, "atomic_write_text", lambda *a, **k: False
    )
    assert _apply_patch_atomic(fake_xdit) is False
    assert fake_xdit.read_text() == original


def test_concurrent_calls_are_safe(fake_xdit: Path) -> None:
    """flock + atomic rename: concurrent patchers must not corrupt the file."""
    results: list[bool] = []

    def _worker() -> None:
        results.append(ensure_xdit_profiler_patched())

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(results)
    assert fake_xdit.read_text().count("repeat=1") == 1
