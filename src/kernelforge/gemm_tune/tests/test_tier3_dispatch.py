# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for turning a proposed candidate into something the referee can time.

The two measurement decisions in this module were both learned by getting them
wrong on real hardware, and neither is visible from a passing tuner:

* error is measured against the magnitude of the reference *as a whole*, not
  element by element -- the element-wise reading scores the unmodified
  ``torch.matmul`` at 1.375 against its own fp32 reference, so a gate on it
  rejects the default path;
* correctness is re-checked on fresh inputs several times, because four winners
  picked on this box were wrong intermittently and a single check passes such a
  kernel roughly at random.

Both are pinned below. The rest is the honesty of the dispatch itself: a
candidate this module cannot build has to come back as ``None`` -- recorded by
the referee as "not dispatchable" -- rather than as an approximation of what it
might have meant.

torch and aiter are injected as fakes rather than imported: the point is the
dispatch logic, and requiring a GPU would mean none of it is covered anywhere.
"""

from __future__ import annotations

import sys
import types

import pytest

from kernelforge.gemm_tune.tier3.dispatch import (
    CORRECTNESS_TRIALS,
    GRAPH_INNER,
    MAX_RELATIVE_ERROR,
    _Bf16DenseAdapter,
    parse_config,
    relative_error,
)


# ── fakes ────────────────────────────────────────────────────────────────────
class _Scalar:
    """A 0-d result supporting the division and float() that the module does."""

    def __init__(self, v: float) -> None:
        self.v = float(v)

    def __truediv__(self, other):
        return _Scalar(self.v / float(other))

    def __float__(self) -> float:
        return self.v


class _T:
    """A tensor stand-in that does the real arithmetic on a flat list."""

    def __init__(self, vals) -> None:
        self.vals = [float(v) for v in vals]

    def float(self):
        return _T(self.vals)

    def t(self):
        return _T(self.vals)

    def unsqueeze(self, _dim):
        return _T(self.vals)

    def abs(self):
        return _T([abs(v) for v in self.vals])

    def max(self):
        return _Scalar(max(self.vals))

    def mean(self):
        return _Scalar(sum(self.vals) / len(self.vals))

    def __sub__(self, other):
        return _T([a - b for a, b in zip(self.vals, other.vals, strict=False)])


class _FakeCuda:
    def __init__(self, *, capture_raises: bool = False) -> None:
        self.capture_raises = capture_raises
        self.synchronised = 0
        self.captured = 0

    class _Stream:
        def wait_stream(self, _other):
            return None

    def Stream(self):  # noqa: N802 - mirrors torch.cuda.Stream
        return self._Stream()

    def current_stream(self):
        return self._Stream()

    def stream(self, _s):
        class _Ctx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *_a):
                return False

        return _Ctx()

    def synchronize(self):
        self.synchronised += 1

    def CUDAGraph(self):  # noqa: N802 - mirrors torch.cuda.CUDAGraph
        if self.capture_raises:
            raise RuntimeError("capture unsupported here")
        outer = self

        class _Graph:
            def replay(self_inner):
                return "replayed"

        outer._graph = _Graph()
        return outer._graph

    def graph(self, _g):
        outer = self

        class _Ctx:
            def __enter__(self_inner):
                outer.captured += 1
                return None

            def __exit__(self_inner, *_a):
                return False

        return _Ctx()


class _FakeTorch:
    bfloat16 = "bfloat16"

    def __init__(self, *, capture_raises: bool = False, matmul_result=None) -> None:
        self.cuda = _FakeCuda(capture_raises=capture_raises)
        self.seeds: list[int] = []
        self._matmul_result = matmul_result if matmul_result is not None else _T([1.0, 2.0, 3.0])

    def manual_seed(self, seed):
        self.seeds.append(seed)

    def randn(self, *shape, **_kw):
        return _T([1.0] * max(1, len(shape)))

    def empty(self, *_shape, **_kw):
        return _T([0.0])

    def matmul(self, _a, _b):
        return self._matmul_result


def _install_aiter(monkeypatch: pytest.MonkeyPatch, *, asm_result=None, raises: bool = False):
    """Wire a fake ``aiter`` package, including the two submodules imported."""
    calls: dict[str, int] = {"findallsols": 0, "workspace_init": 0}

    aiter = types.ModuleType("aiter")

    def _findallsols(*_a, **_k):
        calls["findallsols"] += 1

    def _hipb_mm(*_a, **_k):
        return _T([1.0, 2.0, 3.0])

    def _gemm_asm(*_a, **_k):
        if raises:
            raise RuntimeError("asm kernel exploded")
        return asm_result if asm_result is not None else _T([1.0, 2.0, 3.0])

    aiter.hipb_findallsols = _findallsols
    aiter.hipb_mm = _hipb_mm
    aiter.gemm_a16w16_asm = _gemm_asm

    ops = types.ModuleType("aiter.ops")
    opus_mod = types.ModuleType("aiter.ops.opus")

    class _Opus:
        @staticmethod
        def opus_gemm_workspace_init():
            calls["workspace_init"] += 1

        @staticmethod
        def opus_gemm_a16w16_tune(*_a, **_k):
            return _T([1.0, 2.0, 3.0])

    opus_mod.gemm_op_a16w16 = _Opus()

    flydsl = types.ModuleType("aiter.ops.flydsl")
    kernels = types.ModuleType("aiter.ops.flydsl.gemm_kernels")
    kernels.flydsl_hgemm = lambda *_a, **_k: _T([1.0, 2.0, 3.0])

    flydsl.gemm_kernels = kernels
    ops.opus = opus_mod
    ops.flydsl = flydsl
    aiter.ops = ops

    for name, mod in (
        ("aiter", aiter),
        ("aiter.ops", ops),
        ("aiter.ops.opus", opus_mod),
        ("aiter.ops.flydsl", flydsl),
        ("aiter.ops.flydsl.gemm_kernels", kernels),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch):
    """An adapter whose torch is a fake, so nothing here needs a device."""
    torch = _FakeTorch()
    a = _Bf16DenseAdapter()
    monkeypatch.setattr(a, "_torch", lambda: torch)
    a.fake_torch = torch  # type: ignore[attr-defined]
    return a


# ── the config a candidate carries ───────────────────────────────────────────
class TestReadingACandidatesConfig:
    def test_ints_and_bools_come_back_typed_not_as_strings(self):
        assert parse_config("solidx=17;async_copy=True;b_to_lds=False") == {
            "solidx": 17,
            "async_copy": True,
            "b_to_lds": False,
        }

    def test_anything_not_an_int_or_bool_stays_a_string(self):
        assert parse_config("kernelName=some_asm_kernel") == {"kernelName": "some_asm_kernel"}

    def test_surrounding_whitespace_is_not_part_of_the_key_or_value(self):
        assert parse_config(" tile_m = 128 ; tile_n = 64 ") == {"tile_m": 128, "tile_n": 64}

    def test_a_field_with_no_equals_is_skipped_rather_than_guessed(self):
        assert parse_config("solidx=3;garbage;tile_m=16") == {"solidx": 3, "tile_m": 16}

    def test_an_empty_config_is_an_empty_dict(self):
        assert parse_config("") == {}


# ── the measurement the module docstring exists for ──────────────────────────
class TestHowErrorIsMeasured:
    def test_error_is_scaled_by_the_whole_reference_not_element_by_element(self):
        """Element-wise scaling lets near-zero reference elements dominate."""
        ref = _T([1.0, 2.0, 3.0])
        got = _T([1.0, 2.0, 3.5])

        # max|got-ref| / mean|ref| == 0.5 / 2.0
        assert relative_error(got, ref) == pytest.approx(0.25)

    def test_a_near_zero_reference_element_does_not_dominate(self):
        ref = _T([1e-9, 4.0, 8.0])
        got = _T([1e-3, 4.0, 8.0])

        # The deviation is tiny against mean|ref| (=4), even though it is a
        # millionfold error on that one element.
        assert relative_error(got, ref) < MAX_RELATIVE_ERROR


# ── operands and graph capture ───────────────────────────────────────────────
class TestOperandsAreBuiltOncePerShape:
    def test_the_same_shape_reuses_its_operands_so_timing_is_not_allocation(self, adapter):
        first = adapter._ops((2, 3, 4))
        second = adapter._ops((2, 3, 4))

        assert first is second
        assert adapter.fake_torch.seeds == [0], "operands must be reproducible"

    def test_a_different_shape_gets_its_own_operands(self, adapter):
        assert adapter._ops((2, 3, 4)) is not adapter._ops((8, 3, 4))


class TestGraphCapture:
    def test_many_invocations_are_captured_so_dispatch_cost_is_amortised(self, adapter):
        calls = []
        wrapped = adapter.as_graph(lambda: calls.append(1))

        assert wrapped() == "replayed"
        assert adapter.fake_torch.cuda.captured == 1
        # 5 warm-up calls outside the capture, GRAPH_INNER inside it.
        assert len(calls) == 5 + GRAPH_INNER

    def test_a_kernel_that_cannot_be_captured_is_still_timed_raw(self, monkeypatch: pytest.MonkeyPatch):
        torch = _FakeTorch(capture_raises=True)
        a = _Bf16DenseAdapter()
        monkeypatch.setattr(a, "_torch", lambda: torch)

        def fn():
            return "raw"

        assert a.as_graph(fn) is fn

    def test_the_baseline_is_the_unmodified_matmul_under_the_same_capture(self, adapter):
        assert adapter.make_baseline("2x3x4")() == "replayed"

    def test_sync_is_the_devices_own_barrier(self, adapter):
        assert adapter.sync() == adapter.fake_torch.cuda.synchronize


# ── dispatching each backend ─────────────────────────────────────────────────
class TestWhatCanAndCannotBeDispatched:
    def test_the_torch_backend_needs_no_config(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        assert adapter._build((2, 3, 4), {"backend": "torch"}) is not None

    def test_hipblaslt_without_a_solution_index_is_not_dispatchable(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        assert adapter._build((2, 3, 4), {"backend": "hipblaslt", "config": ""}) is None

    def test_hipblaslt_creates_the_handle_before_the_first_call(self, adapter, monkeypatch):
        """hipb_mm on a handle nobody created aborts from C++, uncatchably."""
        calls = _install_aiter(monkeypatch)

        call = adapter._build((2, 3, 4), {"backend": "hipblaslt", "config": "solidx=7"})

        assert call is not None
        assert calls["findallsols"] == 1
        adapter._build((2, 3, 4), {"backend": "hipblaslt", "config": "solidx=9"})
        assert calls["findallsols"] == 1, "the handle is created once, not per candidate"

    def test_the_asm_backend_without_a_kernel_name_is_not_dispatchable(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        assert adapter._build((2, 3, 4), {"backend": "aiter_asm", "config": "splitK=4"}) is None

    def test_the_asm_backend_with_a_kernel_name_dispatches(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        call = adapter._build((2, 3, 4), {"backend": "aiter_asm", "config": "kernelName=k1;splitK=4"})
        assert call is not None and call() is not None

    def test_opus_without_a_kernel_id_is_not_dispatchable(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        assert adapter._build((2, 3, 4), {"backend": "aiter_opus", "config": "splitK=1"}) is None

    def test_opus_initialises_its_workspace_before_dispatch(self, adapter, monkeypatch):
        calls = _install_aiter(monkeypatch)

        call = adapter._build((2, 3, 4), {"backend": "aiter_opus", "config": "kernelId=3"})

        assert call is not None and call() is not None
        assert calls["workspace_init"] == 1

    def test_the_flydsl_backend_dispatches_from_its_tile_config(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        call = adapter._build(
            (2, 3, 4),
            {"backend": "aiter_flydsl", "config": "tile_m=128;tile_n=64;tile_k=32;stages=4"},
        )
        assert call is not None and call() is not None

    def test_an_unknown_backend_is_absent_rather_than_approximated(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        assert adapter._build((2, 3, 4), {"backend": "who_knows"}) is None

    def test_a_backend_that_raises_while_building_is_not_dispatchable(self, adapter, monkeypatch):
        _install_aiter(monkeypatch, raises=True)

        call = adapter._build((2, 3, 4), {"backend": "aiter_asm", "config": "kernelName=k1"})

        # Building succeeded; the raise happens on the call, and the referee
        # sees it through the correctness check rather than as a crash.
        assert call is not None
        with pytest.raises(RuntimeError):
            call()

    def test_a_backend_that_raises_while_building_is_absent_not_fatal(self, adapter, monkeypatch):
        """Creating the hipblaslt handle is the step that can fail during build."""
        calls = _install_aiter(monkeypatch)
        del calls

        def _boom(*_a, **_k):
            raise RuntimeError("no handle for you")

        monkeypatch.setattr(sys.modules["aiter"], "hipb_findallsols", _boom)

        assert adapter._build((2, 3, 4), {"backend": "hipblaslt", "config": "solidx=7"}) is None

    def test_torch_is_imported_lazily_so_the_module_loads_off_gpu(self):
        """The module must stay importable without torch; ``_torch`` is the seam."""
        torch = pytest.importorskip("torch")
        assert _Bf16DenseAdapter._torch() is torch

    def test_dispatch_records_which_candidate_is_in_play_for_the_checker(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        dispatch = adapter.make_dispatch("2x3x4")

        assert dispatch({"backend": "torch"})() == "replayed"
        assert adapter._in_play["2x3x4"] == {"backend": "torch"}

    def test_an_undispatchable_candidate_yields_no_callable(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        assert adapter.make_dispatch("2x3x4")({"backend": "who_knows"}) is None


# ── correctness ──────────────────────────────────────────────────────────────
class TestCorrectness:
    def test_with_no_candidate_in_play_there_is_nothing_to_reject(self, adapter):
        assert adapter.make_correctness("2x3x4")(lambda: None) is True

    def test_a_matching_result_passes(self, adapter, monkeypatch):
        _install_aiter(monkeypatch, asm_result=_T([1.0, 2.0, 3.0]))
        check = adapter.make_correctness("2x3x4")
        adapter._in_play["2x3x4"] = {"backend": "aiter_asm", "config": "kernelName=k1"}

        assert check(lambda: None) is True

    def test_a_result_beyond_the_error_limit_is_rejected(self, adapter, monkeypatch):
        _install_aiter(monkeypatch, asm_result=_T([100.0, 2.0, 3.0]))
        check = adapter.make_correctness("2x3x4")
        adapter._in_play["2x3x4"] = {"backend": "aiter_asm", "config": "kernelName=k1"}

        assert check(lambda: None) is False

    def test_a_candidate_that_cannot_be_rebuilt_is_rejected(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        assert adapter._is_correct((2, 3, 4), {"backend": "who_knows"}) is False

    def test_a_kernel_returning_nothing_is_rejected(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        monkeypatch.setattr(adapter, "_build", lambda *_a, **_k: lambda: None)

        assert adapter._is_correct((2, 3, 4), {"backend": "aiter_asm"}) is False

    def test_a_kernel_that_raises_is_wrong_not_an_error(self, adapter, monkeypatch):
        _install_aiter(monkeypatch, raises=True)

        assert adapter._is_correct((2, 3, 4), {"backend": "aiter_asm", "config": "kernelName=k1"}) is False

    def test_every_trial_uses_fresh_inputs(self, adapter, monkeypatch):
        """One check passes an intermittently-wrong kernel roughly at random."""
        _install_aiter(monkeypatch)
        built = []
        real_build = adapter._build
        monkeypatch.setattr(adapter, "_build", lambda k, c: built.append(adapter._operands.get(k)) or real_build(k, c))

        adapter._is_correct((2, 3, 4), {"backend": "torch"})

        assert len(built) == CORRECTNESS_TRIALS
        assert len({id(ops) for ops in built}) == CORRECTNESS_TRIALS, "inputs were reused between trials"

    def test_the_timing_operands_survive_the_check_that_borrowed_the_slot(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)
        saved = adapter._ops((2, 3, 4))

        adapter._is_correct((2, 3, 4), {"backend": "torch"})

        assert adapter._ops((2, 3, 4)) is saved

    def test_a_shape_with_no_timing_operands_leaves_none_behind(self, adapter, monkeypatch):
        _install_aiter(monkeypatch)

        adapter._is_correct((2, 3, 4), {"backend": "torch"})

        assert (2, 3, 4) not in adapter._operands
