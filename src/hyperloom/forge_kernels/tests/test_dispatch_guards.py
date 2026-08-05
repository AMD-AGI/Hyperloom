# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Guard tests for the in-serving-process kernel-pack dispatcher.

This code runs inside vLLM / SGLang, so the property that matters is not "does
it go fast" but "does it ever do something the caller did not ask for". Every
test below drives one guard and asserts the dispatcher declines by returning
``None``, which the patched call site reads as "run the original code".
"""

from __future__ import annotations

import json

import pytest

from hyperloom.forge_kernels import _packs
from hyperloom.forge_kernels._dispatch import rowwise_softmax
from hyperloom.forge_kernels._packs import cache_max
from hyperloom.forge_kernels._packs import dtype_tag
from hyperloom.forge_kernels._packs import enabled_pack_names
from hyperloom.forge_kernels._packs import load_pack


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for key in (
        _packs.ENV_ENABLED,
        _packs.ENV_ROOT,
        _packs.ENV_VERIFY,
        _packs.ENV_CACHE_MAX,
    ):
        monkeypatch.delenv(key, raising=False)
    _packs.reset_for_tests()
    yield
    _packs.reset_for_tests()


def _install(root, name="p", *, ok=True, verified=((1024, "f32"),), builder="build"):
    pack_dir = root / name
    pack_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "kernel.py").write_text("def build(m, n, dt):\n    return None\n")
    (pack_dir / "pack.json").write_text(json.dumps({"name": name, "op": "rowwise_softmax", "builder": builder}))
    (pack_dir / "preflight.json").write_text(
        json.dumps({"ok": ok, "verified": [{"N": n, "dtype": d} for n, d in verified]})
    )
    return pack_dir


# ------------------------------------------------------------------- env


def test_off_by_default():
    # No $HYPERLOOM_FORGE_KERNEL_PACKS => the whole feature is inert, and the
    # dispatcher must not even try to import torch.
    assert enabled_pack_names() == ()
    assert rowwise_softmax(object()) is None


def test_enabled_names_are_parsed_in_order(monkeypatch):
    monkeypatch.setenv(_packs.ENV_ENABLED, " b , a ,, ")
    assert enabled_pack_names() == ("b", "a")


def test_cache_max_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv(_packs.ENV_CACHE_MAX, "not-a-number")
    assert cache_max() == 64
    monkeypatch.setenv(_packs.ENV_CACHE_MAX, "-3")
    assert cache_max() == 64
    monkeypatch.setenv(_packs.ENV_CACHE_MAX, "8")
    assert cache_max() == 8


def test_dtype_tags():
    torch = pytest.importorskip("torch")
    assert dtype_tag(torch.float32) == "f32"
    assert dtype_tag(torch.bfloat16) == "bf16"
    assert dtype_tag(torch.float16) == "f16"
    assert dtype_tag(torch.int8) is None


# ------------------------------------------------------------- pack loading


def test_missing_pack_is_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))
    monkeypatch.setenv(_packs.ENV_ENABLED, "absent")
    assert load_pack("absent") is None


def test_pack_without_preflight_is_not_loaded(tmp_path, monkeypatch):
    pack_dir = _install(tmp_path)
    (pack_dir / "preflight.json").unlink()
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))
    # An ungated pack is exactly the case this design exists to prevent.
    assert load_pack("p") is None


def test_failed_preflight_is_not_loaded(tmp_path, monkeypatch):
    _install(tmp_path, ok=False, verified=())
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))
    assert load_pack("p") is None


def test_preflight_with_no_verified_shapes_is_not_loaded(tmp_path, monkeypatch):
    _install(tmp_path, ok=True, verified=())
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))
    assert load_pack("p") is None


def test_loaded_pack_exposes_its_verified_shapes(tmp_path, monkeypatch):
    _install(tmp_path, verified=((1024, "f32"), (2048, "bf16")))
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))

    pack = load_pack("p")

    assert pack is not None
    assert pack.supports(1024, "f32")
    assert pack.supports(2048, "bf16")
    assert not pack.supports(1024, "bf16")
    assert not pack.supports(128, "f32")


# ---------------------------------------------------------------- build cache


def test_build_failure_is_blacklisted_not_retried(tmp_path, monkeypatch):
    _install(tmp_path, builder="explode")
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))
    pack = load_pack("p")
    calls: list[tuple] = []

    class _Mod:
        @staticmethod
        def explode(m, n, dt):
            calls.append((m, n, dt))
            raise RuntimeError("flydsl said no")

    pack.module = _Mod

    assert pack.build(8, 1024, "f32") is None
    assert pack.build(8, 1024, "f32") is None
    # One attempt total: a broken shape must not cost a JIT attempt per call.
    assert len(calls) == 1


def test_cache_ceiling_declines_instead_of_thrashing(tmp_path, monkeypatch):
    _install(tmp_path)
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))
    monkeypatch.setenv(_packs.ENV_CACHE_MAX, "2")
    pack = load_pack("p")

    class _Mod:
        @staticmethod
        def build(m, n, dt):
            return f"launcher-{m}"

    pack.module = _Mod

    assert pack.build(1, 1024, "f32") == "launcher-1"
    assert pack.build(2, 1024, "f32") == "launcher-2"
    assert pack.build(3, 1024, "f32") is None
    assert pack.build(1, 1024, "f32") == "launcher-1"


def test_build_if_cached_never_builds(tmp_path, monkeypatch):
    # The graph-capture path: building mid-capture is illegal, so a cold shape
    # has to fall back rather than JIT.
    _install(tmp_path)
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))
    pack = load_pack("p")

    class _Mod:
        @staticmethod
        def build(m, n, dt):
            raise AssertionError("must not build during capture")

    pack.module = _Mod

    assert pack.build_if_cached(8, 1024, "f32") is None


# ------------------------------------------------------------- tensor guards


def test_declines_non_tensor_and_wrong_rank(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    _install(tmp_path)
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))
    monkeypatch.setenv(_packs.ENV_ENABLED, "p")

    assert rowwise_softmax("not a tensor") is None
    assert rowwise_softmax(torch.zeros(4, 4, 4)) is None
    # CPU tensor: no device to dispatch to.
    assert rowwise_softmax(torch.zeros(4, 1024)) is None


def test_declines_unverified_shape(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    _install(tmp_path, verified=((1024, "f32"),))
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))
    monkeypatch.setenv(_packs.ENV_ENABLED, "p")

    # 128 is a shape a real MoE router asks for and preflight rejected.
    assert rowwise_softmax(torch.zeros(8, 128, device="cuda")) is None


def test_declines_noncontiguous_input(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    _install(tmp_path, verified=((1024, "f32"),))
    monkeypatch.setenv(_packs.ENV_ROOT, str(tmp_path))
    monkeypatch.setenv(_packs.ENV_ENABLED, "p")

    strided = torch.zeros(8, 2048, device="cuda")[:, ::2]
    assert not strided.is_contiguous()
    assert rowwise_softmax(strided) is None
