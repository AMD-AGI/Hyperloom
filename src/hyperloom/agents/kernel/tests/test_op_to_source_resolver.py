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

_TLA_PATH = Path(__file__).resolve().parent.parent / "tools" / "tracelens_analysis.py"

# An editable native source must end in .cu/.cuh/.hip/.h and resolve to an
# absolute path; a non-editable one is e.g. an empty/unshipped csrc path.
_EDITABLE_CU = "/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/foo.cu"
_EDITABLE_CU_2 = "/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/bar.cu"
_EDITABLE_H = "/sgl-workspace/aiter/csrc/kernels/rope/rope_common.h"


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


def test_single_patchable_header_source_is_routable(tla) -> None:
    """Curated patchable native headers are editable sources, not non_rewritable."""
    mapping = {
        "aiter::rope": _entry(
            "single",
            sglang={"kn_entry": _kernel(_EDITABLE_H)},
        )
    }
    res = tla.resolve_op_source("aiter::rope", mapping=mapping)
    assert res is not None
    assert res.status == "resolved"
    assert res.is_routable
    assert res.sources == [_EDITABLE_H]


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


def test_composite_device_name_narrows_to_matching_kernel(tla) -> None:
    """A composite given the trace device kernel name narrows to that ONE editable
    source (the hot kernel) instead of fanning out to every co-firing kernel.

    Regression: the Triton fused-MoE label aggregates the MoE GEMM plus co-firing
    quant/align helpers; without device-name narrowing the resolver fanned out to
    all three sources (quant.cu, the triton .py, int8.py) and GEAK chased the wrong
    file. With the trace's device symbol it must resolve to just the GEMM source.
    """
    mapping = {
        "moe::fused": _entry(
            "composite",
            sglang={
                "quant_kernel": _kernel(_EDITABLE_CU),          # co-firing helper
                "fused_moe_kernel": _kernel(_EDITABLE_CU_2),    # the hot GEMM
                "int8_kernel": _kernel(_EDITABLE_CU),           # another helper
            },
        )
    }
    res = tla.resolve_op_source(
        "moe::fused", framework="sglang",
        device_kernel_name="_fused_moe_kernel_sequence", mapping=mapping,
    )
    assert res is not None
    assert res.kind == "composite"
    assert res.status == "resolved"
    assert res.matched_route == "fused_moe_kernel"   # substring-matched the hot symbol
    leaves = res.leaf_resolutions()
    assert len(leaves) == 1
    assert leaves[0].primary_source == _EDITABLE_CU_2  # narrowed to the GEMM, not fan-out


def test_composite_without_device_name_still_fans_out(tla) -> None:
    """No trace device name -> unchanged behavior: fan out one leaf per editable source."""
    mapping = {
        "moe::fused": _entry(
            "composite",
            sglang={
                "fused_moe_kernel": _kernel(_EDITABLE_CU),
                "other_kernel": _kernel(_EDITABLE_CU_2),
            },
        )
    }
    res = tla.resolve_op_source("moe::fused", framework="sglang", mapping=mapping)
    assert res is not None
    assert res.kind == "composite"
    leaves = res.leaf_resolutions()
    assert {lf.primary_source for lf in leaves} == {_EDITABLE_CU, _EDITABLE_CU_2}


def test_composite_device_name_no_match_falls_back_to_fanout(tla) -> None:
    """A device name that matches none of the composite kernels falls back to fan-out."""
    mapping = {
        "moe::fused": _entry(
            "composite",
            sglang={
                "fused_moe_kernel": _kernel(_EDITABLE_CU),
                "other_kernel": _kernel(_EDITABLE_CU_2),
            },
        )
    }
    res = tla.resolve_op_source(
        "moe::fused", framework="sglang",
        device_kernel_name="totally_unrelated_symbol", mapping=mapping,
    )
    assert res is not None
    leaves = res.leaf_resolutions()
    assert {lf.primary_source for lf in leaves} == {_EDITABLE_CU, _EDITABLE_CU_2}


# --------------------------------------------------------------------------- #
# _select_source_meta
# --------------------------------------------------------------------------- #
def test_select_sources_framework_hint_picks_matching_container(tla) -> None:
    """With both containers editable and neither on disk, the framework hint decides."""
    entry = _entry(
        "single",
        vllm={"kv": _kernel(_EDITABLE_CU)},
        sglang={"ks": _kernel(_EDITABLE_CU_2)},
    )
    resolver = tla.OpResolver({})
    assert [m[0] for m in resolver._select_source_meta(entry, "vllm")] == [_EDITABLE_CU]
    assert [m[0] for m in resolver._select_source_meta(entry, "sglang")] == [_EDITABLE_CU_2]


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
    assert [m[0] for m in resolver._select_source_meta(entry, "sglang")] == [str(on_disk)]


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


def test_expand_op_fanout_explicit_framework_selects_matching_source(tla, monkeypatch) -> None:
    """An explicit ``framework`` routes to that container's source (#1 regression).

    The same op carries both ``vllm`` and ``sglang`` editable sources at distinct
    ``.cu`` paths and neither is on disk, so only the explicit framework decides.
    ``HYPERLOOM_FRAMEWORK`` is cleared to prove the value comes from the arg, not env.
    """
    mapping = {
        "aiter::foo": _entry(
            "single",
            vllm={"kv": _kernel(_EDITABLE_CU)},
            sglang={"ks": _kernel(_EDITABLE_CU_2)},
        )
    }
    monkeypatch.setattr(tla, "load_mapping", lambda: mapping)
    monkeypatch.setenv("HYPERLOOM_FRAMEWORK", "sglang")

    out_vllm = tla._expand_op_fanout(
        [{"name": "aiter::foo", "duration_us": 100.0}], framework="vllm"
    )
    assert len(out_vllm) == 1
    assert out_vllm[0]["_op_resolution"].primary_source == _EDITABLE_CU

    out_sgl = tla._expand_op_fanout(
        [{"name": "aiter::foo", "duration_us": 100.0}], framework="sglang"
    )
    assert out_sgl[0]["_op_resolution"].primary_source == _EDITABLE_CU_2


def test_finalize_candidates_threads_framework_to_resolver(tla, monkeypatch) -> None:
    """``_finalize_candidates(framework=...)`` selects the matching container's source (#1)."""
    mapping = {
        "aiter::foo": _entry(
            "single",
            vllm={"kv": _kernel(_EDITABLE_CU)},
            sglang={"ks": _kernel(_EDITABLE_CU_2)},
        )
    }
    monkeypatch.setattr(tla, "load_mapping", lambda: mapping)
    monkeypatch.delenv("HYPERLOOM_FRAMEWORK", raising=False)

    out = tla._finalize_candidates(
        [{"name": "aiter::foo", "duration_us": 100.0, "source_file": ""}],
        total_dur=100.0,
        framework="vllm",
    )[0]
    assert out["source_file"] == _EDITABLE_CU
    assert out["op_to_source_status"] == "resolved"


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


# --------------------------------------------------------------------------- #
# load_op_dominant_kernel_map (data-driven composite disambiguation)
# --------------------------------------------------------------------------- #
def test_dominant_kernel_map_picks_max_duration(tla, tmp_path) -> None:
    """The dominant device kernel = max aggregated duration, parsed from the CSV."""
    import csv as _csv

    csv_dir = tmp_path
    rows = [
        {
            "name": "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel",
            "kernel_details_summary": (
                "[{'name': '_per_token_group_quant_8bit', 'stream': 8, 'count': 2001, "
                "'total_duration_us': np.float64(9266.1), 'mean_duration_us': np.float64(4.63)}, "
                "{'name': 'fused_moe_kernel', 'stream': 8, 'count': 2001, "
                "'total_duration_us': np.float64(401466.7), 'mean_duration_us': np.float64(200.63)}]"
            ),
        },
        # second row (different shape) for the same op — durations must aggregate
        {
            "name": "sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel",
            "kernel_details_summary": (
                "[{'name': 'fused_moe_kernel', 'stream': 8, 'count': 138, "
                "'total_duration_us': np.float64(287774.7), 'mean_duration_us': np.float64(2085.3)}]"
            ),
        },
    ]
    with (csv_dir / "unified_perf_summary.csv").open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["name", "kernel_details_summary"])
        w.writeheader()
        w.writerows(rows)
    m = tla.load_op_dominant_kernel_map(csv_dir)
    assert m["sglang_profiler::fused_moe_triton_kernels_invoke_fused_moe_kernel"] == "fused_moe_kernel"


def test_dominant_kernel_map_missing_csv_is_empty(tla, tmp_path) -> None:
    assert tla.load_op_dominant_kernel_map(tmp_path) == {}
