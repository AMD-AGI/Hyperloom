# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Pure-logic tests for per-shape production split-K trial (no GPU).

``_supports`` is the only GPU-touching call; monkeypatching it exercises the
scan / control-gate / memoization logic on any host.
"""

from __future__ import annotations

import kernelforge.gemm_tune.aiter_splitk_validate as mod
from kernelforge.gemm_tune.aiter_splitk_validate import make_support_fn, max_supported_splitk


def test_none_when_control_fails(monkeypatch):
    # splitK=0 control raises (no GPU) -> None so caller keeps its static cap.
    def boom(m, n, k, sk, device="cuda"):
        raise RuntimeError("no gpu")

    monkeypatch.setattr(mod, "_supports", boom)
    assert max_supported_splitk(64, 5120, 5120) is None


def test_none_when_control_unsupported(monkeypatch):
    # Control returns False (not an exception) -> still None.
    monkeypatch.setattr(mod, "_supports", lambda m, n, k, sk, device="cuda": False)
    assert max_supported_splitk(64, 5120, 5120) is None


def test_scans_to_first_gap(monkeypatch):
    # Supports 0,1,2 then fails at 3 -> max is 2.
    monkeypatch.setattr(mod, "_supports", lambda m, n, k, sk, device="cuda": sk <= 2)
    assert max_supported_splitk(64, 5120, 5120, ceiling=6) == 2


def test_higher_max_when_supported(monkeypatch):
    # A shape that supports up to 3 keeps the extra gain cap=2 would drop.
    monkeypatch.setattr(mod, "_supports", lambda m, n, k, sk, device="cuda": sk <= 3)
    assert max_supported_splitk(16, 5120, 5120, ceiling=6) == 3


def test_exception_mid_scan_treated_as_unsupported(monkeypatch):
    # Control (sk=0) passes; a hard error at sk=2 stops the scan at 1.
    def flaky(m, n, k, sk, device="cuda"):
        if sk >= 2:
            raise RuntimeError("dispatch blew up")
        return True

    monkeypatch.setattr(mod, "_supports", flaky)
    assert max_supported_splitk(64, 5120, 5120, ceiling=6) == 1


def test_make_support_fn_memoizes_per_shape(monkeypatch):
    calls = []

    def fake(m, n, k, sk, device="cuda"):
        calls.append((m, n, k, sk))
        return sk <= 2

    monkeypatch.setattr(mod, "_supports", fake)
    fn = make_support_fn()
    a = fn(64, 5120, 5120)
    b = fn(64, 5120, 5120)  # cached -> no new _supports calls
    assert a == b == 2
    assert {(m, n, k) for (m, n, k, _sk) in calls} == {(64, 5120, 5120)}
    # control + scan 1,2,3 == 4 trials, once (not doubled by the second fn call)
    assert len(calls) == 4


class TestResolveDevice:
    """`_resolve_device` pins the in-process trial to the tuner's assigned card
    instead of always using device 0 (review: multi-tenant wrong-GPU)."""

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
        # gpu_ids is threaded through without touching the GPU (control fails
        # -> None) and does not raise.
        monkeypatch.setattr(mod, "_supports", lambda m, n, k, sk, device="cuda": False)
        fn = make_support_fn(gpu_ids="1")
        assert fn(64, 5120, 5120) is None
