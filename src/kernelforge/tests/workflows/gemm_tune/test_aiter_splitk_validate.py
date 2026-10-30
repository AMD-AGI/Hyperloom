# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Pure-logic tests for per-shape production split-K trial (no GPU).

The registered per-op probe is the only GPU-touching call; substituting it in
``_TRIAL_OPS`` exercises the scan / control-gate / memoization logic anywhere.
"""

from __future__ import annotations

import sys
import types

import pytest

import kernelforge.workflows.gemm_tune.aiter_splitk_validate as mod
from kernelforge.workflows.gemm_tune.aiter_splitk_validate import (
    make_support_fn,
    max_supported_splitk,
)
from kernelforge.tests.workflows.gemm_tune._fake_aiter import fake_aiter

OP = "a8w8_blockscale"


def _probe(monkeypatch, fn, op: str = OP) -> None:
    """Register ``fn`` as ``op``'s production probe for one test."""
    monkeypatch.setitem(mod._TRIAL_OPS, op, fn)


def test_none_when_control_fails(monkeypatch):
    # splitK=0 control raises (no GPU) -> None so caller keeps its static cap.
    def boom(m, n, k, sk, device="cuda"):
        raise RuntimeError("no gpu")

    _probe(monkeypatch, boom)
    assert max_supported_splitk((64, 5120, 5120)) is None


@pytest.mark.parametrize(
    "failure",
    [ImportError("no module named aiter"), OSError("libamdhip64.so.6: cannot open shared object file")],
    ids=lambda failure: type(failure).__name__,
)
def test_none_when_control_cannot_load_aiter(monkeypatch, failure):
    def unloadable(m, n, k, sk, device="cuda"):
        raise failure

    _probe(monkeypatch, unloadable)
    assert max_supported_splitk((64, 5120, 5120)) is None


def test_control_propagates_unexpected_errors(monkeypatch):
    def buggy(m, n, k, sk, device="cuda"):
        raise TypeError("probe signature drifted")

    _probe(monkeypatch, buggy)
    with pytest.raises(TypeError, match="^probe signature drifted$"):
        max_supported_splitk((64, 5120, 5120))


def test_none_when_control_unsupported(monkeypatch):
    # Control returns False (not an exception) -> still None.
    _probe(monkeypatch, lambda m, n, k, sk, device="cuda": False)
    assert max_supported_splitk((64, 5120, 5120)) is None


def test_scans_to_first_gap(monkeypatch):
    # Supports 0,1,2 then fails at 3 -> max is 2.
    _probe(monkeypatch, lambda m, n, k, sk, device="cuda": sk <= 2)
    assert max_supported_splitk((64, 5120, 5120), ceiling=6) == 2


def test_higher_max_when_supported(monkeypatch):
    # A shape that supports up to 3 keeps the extra gain cap=2 would drop.
    _probe(monkeypatch, lambda m, n, k, sk, device="cuda": sk <= 3)
    assert max_supported_splitk((16, 5120, 5120), ceiling=6) == 3


def test_exception_mid_scan_treated_as_unsupported(monkeypatch):
    # Control (sk=0) passes; a hard error at sk=2 stops the scan at 1.
    def flaky(m, n, k, sk, device="cuda"):
        if sk >= 2:
            raise RuntimeError("dispatch blew up")
        return True

    _probe(monkeypatch, flaky)
    assert max_supported_splitk((64, 5120, 5120), ceiling=6) == 1


def test_make_support_fn_memoizes_per_shape(monkeypatch):
    calls = []

    def fake(m, n, k, sk, device="cuda"):
        calls.append((m, n, k, sk))
        return sk <= 2

    _probe(monkeypatch, fake)
    fn = make_support_fn(op=OP)
    a = fn(64, 5120, 5120)
    b = fn(64, 5120, 5120)  # cached -> no new probe calls
    assert a == b == 2
    assert {(m, n, k) for (m, n, k, _sk) in calls} == {(64, 5120, 5120)}
    # control + scan 1,2,3 == 4 trials, once (not doubled by the second fn call)
    assert len(calls) == 4


class TestOpAwareness:
    """A limit measured on one op's kernel must never be handed to another.

    Answering with the wrong kernel would either license a splitK the real
    kernel rejects (engine-init crash) or tighten one it accepts (silent loss),
    so an op without a registered probe has to answer "unknown".
    """

    def test_only_verified_ops_are_registered(self):
        assert set(mod._TRIAL_OPS) == {"a8w8_blockscale", "a8w8_blockscale_bpreshuffle"}

    def test_unregistered_op_never_trials(self, monkeypatch):
        calls = []

        def fake(m, n, k, sk, device="cuda"):
            calls.append(sk)
            return True

        _probe(monkeypatch, fake)
        assert max_supported_splitk((64, 5120, 17408), op="a4w4_blockscale") is None
        assert not calls, "an unregistered op must not dispatch another op's kernel"

    def test_unregistered_op_support_fn_answers_unknown(self, monkeypatch):
        _probe(monkeypatch, lambda m, n, k, sk, device="cuda": sk <= 3)
        fn = make_support_fn(op="a4w4_blockscale")
        assert fn(64, 5120, 5120) is None


class TestResolveDevice:
    """`_resolve_device` pins the in-process trial to the tuner's assigned card instead of always using device 0 (review: multi-tenant wrong-GPU)."""

    def test_empty_gpu_ids_defaults_to_cuda(self, monkeypatch):
        for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            monkeypatch.delenv(k, raising=False)
        assert mod._resolve_device("") == "cuda"

    def test_first_id_used_when_no_visible_env(self, monkeypatch):
        for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
            monkeypatch.delenv(k, raising=False)
        assert mod._resolve_device("2,3") == "cuda:2"

    def test_visible_env_maps_physical_to_local_index(self, monkeypatch):
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6")
        # physical id 6 is the 3rd visible device -> local torch index 2
        assert mod._resolve_device("6") == "cuda:2"

    def test_assigned_card_not_visible_falls_back(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
        assert mod._resolve_device("7") == "cuda"

    def test_make_support_fn_accepts_gpu_ids(self, monkeypatch):
        # gpu_ids is threaded through without touching the GPU (control fails -> None) and does not raise.
        _probe(monkeypatch, lambda m, n, k, sk, device="cuda": False)
        fn = make_support_fn(gpu_ids="1", op=OP)
        assert fn(64, 5120, 5120) is None


class TestBpreshuffleTrial:
    """bpreshuffle trials its asm kernel, scanning to the splitK=8 aiter's tuner generates asm candidates for."""

    def test_trials_the_asm_kernel_up_to_eight(self, monkeypatch):
        calls = fake_aiter(monkeypatch, k=5120)
        fn = make_support_fn(op="a8w8_blockscale_bpreshuffle")
        assert fn(64, 5120, 5120) == 8
        assert calls == [("asm", sk) for sk in range(9)]

    def test_stops_at_the_first_splitk_the_launcher_rejects(self, monkeypatch):
        # K=1280 is 10 tiles: splitK 6 partitions as 2 tiles -> 5 splits, so the launcher rejects it.
        fake_aiter(monkeypatch, k=1280)
        fn = make_support_fn(op="a8w8_blockscale_bpreshuffle")
        assert fn(64, 5120, 1280) == 5

    def test_blockscale_still_trials_ck_up_to_six(self, monkeypatch):
        calls = fake_aiter(monkeypatch, k=5120)
        assert make_support_fn(op="a8w8_blockscale")(64, 5120, 5120) == 6
        assert calls == [("ck", sk) for sk in range(7)]


def _blockwise_reference(x, w, xs, ws):
    """fp32 blockscale GEMM written block by block, independent of the module's vectorized reference."""
    torch = pytest.importorskip("torch")

    m, k = x.shape
    out = torch.zeros(m, w.shape[0], dtype=torch.float32)
    for kb in range(0, k, 128):
        w_blk = w[:, kb : kb + 128].float() * ws[:, kb // 128].repeat_interleave(128)[: w.shape[0], None]
        out += (x[:, kb : kb + 128].float() * xs[:, kb // 128 : kb // 128 + 1]) @ w_blk.t()
    return out


def numeric_aiter(monkeypatch, scale: float = 1.0) -> list[dict]:
    """Install a CPU ``aiter`` whose asm op computes the blockscale GEMM times ``scale``, reading x_scale column-major;
    skips the test where torch is not installed, as in CI."""
    torch = pytest.importorskip("torch")

    monkeypatch.setattr(mod, "_resolve_device", lambda gpu_ids="": "cpu")

    calls: list[dict] = []
    shuffled = []

    def shuffle_weight(w, layout):
        shuffled.append(layout)
        return w.clone()

    def asm(x, w, out, xs, ws, bias=None, splitK=None, kernelName=None):
        m = x.shape[0]
        xs_row_major = xs.reshape(-1).view(-1, m).t()
        calls.append({"splitK": splitK, "kernelName": kernelName, "layout": shuffled[-1]})
        out.copy_((_blockwise_reference(x, w, xs_row_major, ws) * scale).to(out.dtype))
        return out

    aiter = types.ModuleType("aiter")
    aiter.dtypes = types.SimpleNamespace(
        fp16=torch.float16, fp8=torch.float8_e4m3fn, fp32=torch.float32, bf16=torch.bfloat16
    )
    aiter.gemm_a8w8_blockscale_bpreshuffle_asm = asm
    ops, shuffle = types.ModuleType("aiter.ops"), types.ModuleType("aiter.ops.shuffle")
    shuffle.shuffle_weight = shuffle_weight
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.ops", ops)
    monkeypatch.setitem(sys.modules, "aiter.ops.shuffle", shuffle)
    return calls


class TestAsmRelNormErr:
    """The relative-norm error of one asm split-K row, measured through the production asm op."""

    SHAPE = (4, 256, 384)

    def test_an_exact_kernel_measures_only_bf16_rounding(self, monkeypatch):
        numeric_aiter(monkeypatch)
        err = mod.make_rel_norm_fn(op="a8w8_blockscale_bpreshuffle")(*self.SHAPE, 6, "asm_knl")
        assert 0 <= err < 0.004

    def test_a_kernel_off_by_three_percent_measures_three_percent(self, monkeypatch):
        numeric_aiter(monkeypatch, scale=1.03)
        err = mod.make_rel_norm_fn(op="a8w8_blockscale_bpreshuffle")(*self.SHAPE, 6, "asm_knl")
        assert err == pytest.approx(0.03, abs=0.004)

    def test_dispatches_the_rows_kernel_and_splitk_on_preshuffled_operands(self, monkeypatch):
        calls = numeric_aiter(monkeypatch)
        mod.make_rel_norm_fn(op="a8w8_blockscale_bpreshuffle")(*self.SHAPE, 6, "asm_knl")
        assert calls == [{"splitK": 6, "kernelName": "asm_knl", "layout": (16, 16)}]

    def test_memoized_per_row(self, monkeypatch):
        calls = numeric_aiter(monkeypatch)
        fn = mod.make_rel_norm_fn(op="a8w8_blockscale_bpreshuffle")
        fn(*self.SHAPE, 6, "asm_knl")
        fn(*self.SHAPE, 6, "asm_knl")
        fn(*self.SHAPE, 4, "asm_knl")
        assert [c["splitK"] for c in calls] == [6, 4]

    def test_a_failed_dispatch_measures_nothing(self, monkeypatch):
        def rejected(*args, **kwargs):
            raise RuntimeError("Kdim alignment check failed for splitK!")

        monkeypatch.setitem(mod._REL_NORM_OPS, "a8w8_blockscale_bpreshuffle", rejected)
        assert mod.make_rel_norm_fn(op="a8w8_blockscale_bpreshuffle")(64, 5120, 17408, 6, "k") is None

    def test_a_defect_in_the_check_surfaces(self, monkeypatch):
        def buggy(*args, **kwargs):
            raise TypeError("signature drifted")

        monkeypatch.setitem(mod._REL_NORM_OPS, "a8w8_blockscale_bpreshuffle", buggy)
        with pytest.raises(TypeError, match="^signature drifted$"):
            mod.make_rel_norm_fn(op="a8w8_blockscale_bpreshuffle")(64, 5120, 17408, 6, "k")

    def test_an_op_without_a_check_measures_nothing(self, monkeypatch):
        calls = numeric_aiter(monkeypatch)
        assert mod.make_rel_norm_fn(op="a8w8_blockscale")(*self.SHAPE, 6, "k") is None
        assert not calls
