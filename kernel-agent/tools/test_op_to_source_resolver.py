# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit coverage for the op_to_source resolver routing path.

Exercises the dictionary-first routing that decides how every kernel
candidate reaches GEAK: :func:`resolve_op_source` / :class:`OpResolver`
(``single`` / ``dispatch`` / ``composite``), :meth:`OpResolution.leaf_resolutions`,
:func:`_expand_op_fanout`, the :func:`classify_patchability` op_to_source
short-circuit, and the legacy launcher/grep/pybind fallback in
:func:`_finalize_candidates`.

Fully hermetic: every case injects a synthetic mapping (so no real ``aiter``
package or live container is needed), mirroring the nested
``vllm`` / ``sglang`` -> ``{device_kernel_name: {kernel_source_path, ...}}``
shape of the committed ``data/op_to_source.json``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_TLA_PATH = Path(__file__).resolve().parent / "tracelens_analysis.py"

# An editable native source must end in .cu/.cuh/.hip and resolve to an
# absolute path; a non-editable one is e.g. an empty/unshipped csrc path.
_EDITABLE_CU = "/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/foo.cu"
_EDITABLE_CU_2 = "/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/bar.cu"


@pytest.fixture(scope="module")
def tla() -> types.ModuleType:
    """Load tracelens_analysis.py without running its CLI bootstrap."""
    sys.path.insert(0, str(_TLA_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "_tracelens_analysis_resolver_under_test",
        _TLA_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _kernel(path: str, *, patchable: bool = True, kind: str = "aiter_hip") -> dict[str, Any]:
    """One device-kernel record as stored under a container."""
    return {
        "kernel_source_path": path,
        "kernel_source_line": "1",
        "kernel_kind": kind,
        "patchable": patchable,
    }


def _entry(kind: str, *, vllm: dict | None = None, sglang: dict | None = None) -> dict[str, Any]:
    """One op_to_source mapping entry with the given containers."""
    return {
        "kind": kind,
        "python_launcher_path": ["launcher.py(1): apply"],
        "patchable": True,
        "vllm": vllm or {},
        "sglang": sglang or {},
    }


# --------------------------------------------------------------------------- #
# single
# --------------------------------------------------------------------------- #
def test_single_multiple_editable_sources_fans_out_one_leaf_per_cu(tla) -> None:
    """A routable ``single`` with two editable .cu yields two leaves, one per file."""
    mapping = {
        "aiter::foo": _entry(
            "single",
            sglang={
                "kSym1": _kernel(_EDITABLE_CU),
                "kSym2": _kernel(_EDITABLE_CU_2),
            },
        )
    }
    res = tla.resolve_op_source("aiter::foo", mapping=mapping)
    assert res is not None
    assert res.kind == "single"
    assert res.is_routable
    assert res.sources == [_EDITABLE_CU, _EDITABLE_CU_2]

    leaves = res.leaf_resolutions()
    assert len(leaves) == 2
    assert [leaf.primary_source for leaf in leaves] == [_EDITABLE_CU, _EDITABLE_CU_2]
    assert all(leaf.is_routable for leaf in leaves)


def test_single_no_editable_source_is_non_rewritable(tla) -> None:
    """A ``single`` whose only kernel has no editable source is non_rewritable."""
    mapping = {"op::x": _entry("single", vllm={"kSym": _kernel("", patchable=True)})}
    res = tla.resolve_op_source("op::x", mapping=mapping)
    assert res is not None
    assert res.status == "non_rewritable"
    assert res.patchable is False
    assert res.sources == []
    assert res.reason


def test_phase_suffix_is_stripped_before_lookup(tla) -> None:
    """A steady-state phase tag like '(prefill)' is stripped before the dict lookup."""
    mapping = {"aiter::foo": _entry("single", sglang={"kSym": _kernel(_EDITABLE_CU)})}
    res = tla.resolve_op_source("aiter::foo (prefill)", mapping=mapping)
    assert res is not None and res.is_routable
    assert res.op_name == "aiter::foo"


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def test_dispatch_matches_editable_route(tla) -> None:
    """A trace device_kernel_name matching an editable+patchable route resolves to it."""
    mapping = {
        "vllm::attn": _entry(
            "dispatch",
            vllm={
                "void paged_attention_ll4mi_QKV_mfma16_kernel": _kernel(_EDITABLE_CU),
                "kernel_unified_attentiond.kd": _kernel("/tmp/triton_gen.py", kind="triton"),
            },
        )
    }
    res = tla.resolve_op_source(
        "vllm::attn", device_kernel_name="paged_attention_ll4mi_QKV_mfma16_kernel", mapping=mapping
    )
    assert res is not None
    assert res.kind == "dispatch"
    assert res.status == "resolved"
    assert res.is_routable
    assert res.sources == [_EDITABLE_CU]
    assert res.matched_route == "void paged_attention_ll4mi_QKV_mfma16_kernel"


def test_dispatch_matches_non_editable_route(tla) -> None:
    """A route that matches but has no editable source is non_rewritable, not routable."""
    mapping = {
        "vllm::attn": _entry(
            "dispatch",
            vllm={"some_kernel.kd": _kernel("", patchable=True)},
        )
    }
    res = tla.resolve_op_source(
        "vllm::attn", device_kernel_name="some_kernel.kd", mapping=mapping
    )
    assert res is not None
    assert res.status == "non_rewritable"
    assert res.patchable is False
    assert res.sources == []
    assert res.reason
    assert res.matched_route == "some_kernel.kd"


def test_dispatch_without_device_kernel_name_is_unresolved(tla) -> None:
    """No device_kernel_name -> unresolved (the seam that triggers legacy fallback)."""
    mapping = {
        "vllm::attn": _entry("dispatch", vllm={"k.kd": _kernel(_EDITABLE_CU)})
    }
    res = tla.resolve_op_source("vllm::attn", device_kernel_name=None, mapping=mapping)
    assert res is not None
    assert res.status == "unresolved"
    assert res.patchable is None
    assert res.sources == []


# --------------------------------------------------------------------------- #
# composite
# --------------------------------------------------------------------------- #
def test_composite_mixed_keeps_only_editable_leaves(tla) -> None:
    """A composite with one editable + one non-editable kernel keeps only the editable leaf."""
    mapping = {
        "moe::fused": _entry(
            "composite",
            vllm={
                "editable_kernel": _kernel(_EDITABLE_CU),
                "unshipped_kernel": _kernel("", patchable=True),
            },
        )
    }
    res = tla.resolve_op_source("moe::fused", mapping=mapping)
    assert res is not None
    assert res.kind == "composite"
    assert res.status == "resolved"
    leaves = res.leaf_resolutions()
    assert len(leaves) == 1
    assert leaves[0].primary_source == _EDITABLE_CU


def test_composite_all_non_editable_is_non_rewritable(tla) -> None:
    """A composite with no editable kernels is non_rewritable with empty fanout."""
    mapping = {
        "moe::fused": _entry(
            "composite",
            vllm={
                "k1": _kernel("", patchable=True),
                "k2": _kernel("", patchable=True),
            },
        )
    }
    res = tla.resolve_op_source("moe::fused", mapping=mapping)
    assert res is not None
    assert res.status == "non_rewritable"
    assert res.patchable is False
    assert res.fanout == []
    assert res.leaf_resolutions() == []


# --------------------------------------------------------------------------- #
# _select_sources
# --------------------------------------------------------------------------- #
def test_select_sources_framework_hint_picks_matching_container(tla) -> None:
    """With both containers editable and neither on disk, the framework hint decides."""
    entry = _entry(
        "single",
        vllm={"kv": _kernel(_EDITABLE_CU)},
        sglang={"ks": _kernel(_EDITABLE_CU_2)},
    )
    resolver = tla.OpResolver({})
    assert resolver._select_sources(entry, "vllm") == [_EDITABLE_CU]
    assert resolver._select_sources(entry, "sglang") == [_EDITABLE_CU_2]


def test_select_sources_on_disk_tie_break_beats_framework_hint(tla, tmp_path: Path) -> None:
    """When only one container's source exists on disk, it wins over the framework hint."""
    on_disk = tmp_path / "real_kernel.cu"
    on_disk.write_text("// real device source\n")
    entry = _entry(
        "single",
        vllm={"kv": _kernel(str(on_disk))},
        sglang={"ks": _kernel(_EDITABLE_CU_2)},
    )
    resolver = tla.OpResolver({})
    # sglang hint, but only the vllm source is present on disk -> vllm wins.
    assert resolver._select_sources(entry, "sglang") == [str(on_disk)]


# --------------------------------------------------------------------------- #
# _expand_op_fanout
# --------------------------------------------------------------------------- #
def test_expand_op_fanout_splits_duration_across_leaves(tla, monkeypatch) -> None:
    """A 2-source routable op splits its duration evenly across two fanned-out candidates."""
    mapping = {
        "aiter::foo": _entry(
            "single",
            sglang={"k1": _kernel(_EDITABLE_CU), "k2": _kernel(_EDITABLE_CU_2)},
        )
    }
    monkeypatch.setattr(tla, "load_mapping", lambda: mapping)
    monkeypatch.delenv("HYPERLOOM_FRAMEWORK", raising=False)

    out = tla._expand_op_fanout([{"name": "aiter::foo", "duration_us": 100.0}])
    assert len(out) == 2
    assert {c["op_fanout_index"] for c in out} == {0, 1}
    assert all(c["op_fanout_total"] == 2 for c in out)
    assert all(c["duration_us"] == 50.0 for c in out)
    assert [c["_op_resolution"].primary_source for c in out] == [_EDITABLE_CU, _EDITABLE_CU_2]


def test_expand_op_fanout_single_leaf_does_not_split(tla, monkeypatch) -> None:
    """A single-source op keeps its original duration and attaches its leaf resolution."""
    mapping = {"aiter::foo": _entry("single", sglang={"k1": _kernel(_EDITABLE_CU)})}
    monkeypatch.setattr(tla, "load_mapping", lambda: mapping)
    monkeypatch.delenv("HYPERLOOM_FRAMEWORK", raising=False)

    out = tla._expand_op_fanout([{"name": "aiter::foo", "duration_us": 100.0}])
    assert len(out) == 1
    assert out[0]["duration_us"] == 100.0
    assert "op_fanout_index" not in out[0]
    assert out[0]["_op_resolution"].primary_source == _EDITABLE_CU


def test_expand_op_fanout_dict_miss_passes_through(tla, monkeypatch) -> None:
    """A dictionary miss passes the item through unchanged with a None resolution."""
    monkeypatch.setattr(tla, "load_mapping", lambda: {})
    monkeypatch.delenv("HYPERLOOM_FRAMEWORK", raising=False)

    out = tla._expand_op_fanout([{"name": "unknown::op", "duration_us": 100.0}])
    assert len(out) == 1
    assert out[0]["duration_us"] == 100.0
    assert out[0]["_op_resolution"] is None


# --------------------------------------------------------------------------- #
# classify_patchability op_to_source short-circuit
# --------------------------------------------------------------------------- #
def test_classify_shortcircuit_patchable_true_is_routable(tla) -> None:
    """An op_to_source verdict of patchable=True with a source file routes (True, '')."""
    candidate = {
        "name": "aiter::foo",
        "source_file": _EDITABLE_CU,
        "source_resolution_method": "op_to_source",
        "op_to_source_patchable": True,
        "op_to_source_status": "resolved",
    }
    reusable, reason = tla.classify_patchability(candidate)
    assert reusable is True
    assert reason == ""


def test_classify_shortcircuit_patchable_false_reports_reason(tla) -> None:
    """An op_to_source verdict of patchable=False is blocked with the dictionary's reason."""
    candidate = {
        "name": "op::x",
        "source_file": "launcher.py",
        "source_resolution_method": "op_to_source",
        "op_to_source_patchable": False,
        "op_to_source_reason": "no editable native/triton source",
    }
    reusable, reason = tla.classify_patchability(candidate)
    assert reusable is False
    assert reason == "op_to_source: no editable native/triton source"


def test_classify_shortcircuit_none_falls_through_to_legacy(tla) -> None:
    """A None op_to_source verdict falls through to the legacy heuristics (aten:: native block)."""
    candidate = {
        "name": "aten::mm",
        "source_file": "/some/path/aten_mm.cpp",
        "source_resolution_method": "op_to_source",
        "op_to_source_patchable": None,
    }
    reusable, reason = tla.classify_patchability(candidate)
    assert reusable is False
    # Proves the legacy path ran, not the op_to_source short-circuit branch.
    assert not reason.startswith("op_to_source:")
    assert "PyTorch native op" in reason


# --------------------------------------------------------------------------- #
# _finalize_candidates legacy fallback wiring
# --------------------------------------------------------------------------- #
def test_finalize_dict_miss_fires_legacy_grep_pybind_fallback(tla, monkeypatch) -> None:
    """A dictionary miss with no source_file runs the legacy grep/pybind chain."""
    monkeypatch.setattr(tla, "load_mapping", lambda: {})
    monkeypatch.delenv("HYPERLOOM_FRAMEWORK", raising=False)

    grep_calls: list[str] = []

    def fake_grep(name: str) -> str:
        grep_calls.append(name)
        return "/sgl-workspace/aiter/csrc/kernels/located.cu"

    pybind_calls: list[str] = []

    def fake_pybind(source_file: str, name: str, repo: str) -> str:
        pybind_calls.append(source_file)
        return source_file

    monkeypatch.setattr(tla, "locate_source_via_grep", fake_grep)
    monkeypatch.setattr(tla, "upgrade_pybind_shim_source", fake_pybind)

    out = tla._finalize_candidates(
        [{"name": "unknown::op", "duration_us": 100.0, "source_file": ""}],
        total_dur=100.0,
    )[0]
    assert grep_calls == ["unknown::op"]
    assert pybind_calls  # pybind upgrade attempted on the grep-located path
    assert out["source_file"] == "/sgl-workspace/aiter/csrc/kernels/located.cu"


def test_finalize_in_dict_non_rewritable_does_not_grep(tla, monkeypatch) -> None:
    """An in-dict non_rewritable verdict keeps the launcher as context and never greps."""
    mapping = {"op::x": _entry("single", vllm={"k": _kernel("", patchable=True)})}
    monkeypatch.setattr(tla, "load_mapping", lambda: mapping)
    monkeypatch.delenv("HYPERLOOM_FRAMEWORK", raising=False)

    def boom_grep(name: str) -> str:
        raise AssertionError("legacy grep must not run for an in-dict non_rewritable verdict")

    monkeypatch.setattr(tla, "locate_source_via_grep", boom_grep)

    out = tla._finalize_candidates(
        [{"name": "op::x", "duration_us": 100.0, "source_file": "launcher.py"}],
        total_dur=100.0,
    )[0]
    assert out["op_to_source_status"] == "non_rewritable"
    assert out["reusable_native_kernel"] is False
    assert out["skip_reason"].startswith("op_to_source:")
    assert out["source_file"] == "launcher.py"
