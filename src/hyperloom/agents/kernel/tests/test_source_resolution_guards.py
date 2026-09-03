###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Guards that keep an unresolved-launcher sentinel from becoming a source_file.

TraceLens writes ``launcher_path = "Not found"`` for every Synthetic Op (a
device kernel with no cpu_op parent, e.g. a hand-written Triton kernel launched
straight from Python). That string used to survive parsing as a truthy
``source_file``, which skipped the grep fallback and made classify_patchability
reject the hottest kernels with "source not under a reusable framework root:
Not found". These tests pin the three guards that close that path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import tracelens_analysis as tl  # noqa: E402
import tracelens_skill_runner as tsr  # noqa: E402
from _llm_source_context import build_context_block  # noqa: E402
from _task_group_contract import operator_identity_key  # noqa: E402


_HEADERS = [
    "operation",
    "args",
    "kernel path",
    "time (ms)",
    "%e2e",
    "count",
    "flops/byte",
    "efficiency",
    "bound",
]

_HEADERS_WITH_KERNEL = _HEADERS + ["kernel name"]


def _row(kernel_path: str) -> dict | None:
    """Build one candidate from a data-table row carrying ``kernel_path``."""
    cells = [
        "hipModuleLaunchKernel->_mxfp8_grouped_gemm_kernel (Synthetic Op)",
        "",
        kernel_path,
        "15.671",
        "7.04",
        "114",
        "—",
        "—",
        "—",
    ]
    return tsr._row_to_candidate(
        _HEADERS,
        cells,
        category="other",
        rank=1,
        title="MXFP8 grouped GEMM",
        library="",
        impact={},
    )


def _fallback_row(operation: str, kernel_name: str) -> dict | None:
    """Build one candidate with Operation and the Kernel-Name cell set independently.

    Models the graph-collapsed fallback ``analysis.md`` TraceLens emits: the
    Operation column is a placeholder (``—``/empty) while Kernel Name carries the
    raw device symbol.
    """
    cells = [
        operation,
        "",
        "Not found",
        "15.671",
        "7.04",
        "114",
        "—",
        "—",
        "—",
        kernel_name,
    ]
    return tsr._row_to_candidate(
        _HEADERS_WITH_KERNEL,
        cells,
        category="other",
        rank=1,
        title="Graph-collapsed kernels",
        library="",
        impact={},
    )


# --- Guard 1: launcher placeholder normalisation ---------------------------


def test_not_found_sentinel_normalized_to_empty():
    cand = _row("Not found")
    assert cand is not None
    assert cand["source_file"] == ""
    assert cand["tracelens_launcher_path"] == ""


def test_not_found_sentinel_variants_normalized():
    for raw in ("Not found", "NOT FOUND", "not found", "not_found", "notfound", "  Not Found  "):
        cand = _row(raw)
        assert cand is not None, raw
        assert cand["source_file"] == "", raw


def test_legacy_dash_placeholders_still_normalized():
    for raw in ("-", "—", "–", "n/a", "unknown"):
        cand = _row(raw)
        assert cand is not None, raw
        assert cand["source_file"] == "", raw


def test_real_launcher_path_survives():
    real = "/sgl-workspace/sglang/python/sglang/kernels/ops/moe/mxfp8_moe_amd_gfx95.py(124): _grouped_gemm_mxfp8"
    cand = _row(real)
    assert cand is not None
    assert cand["tracelens_launcher_path"] == real
    assert cand["source_file"] == real.split("(", 1)[0]
    assert cand["source_line"] == 124
    assert cand["source_function"] == "_grouped_gemm_mxfp8"


def test_materialized_runtime_args_are_sanitized_before_prompting(tmp_path):
    """Production YAML selectors reach context without their adjacent secret."""
    config = tmp_path / "baseline.yaml"
    config.write_text(
        "benchmark:\n  envs:\n    EXTRA_SGLANG_ARGS: '--api-key sk-secret --moe-runner-backend triton'\n",
        encoding="utf-8",
    )
    raw = tl._runtime_server_args_from_config(str(config))
    block = build_context_block(
        server_args=raw,
        framework="sglang",
        precision="fp8",
        env={},
    )

    assert "sk-secret" not in block
    assert "api-key" not in block
    assert "triton" in block
    assert '"precision": "fp8"' in block


def test_not_found_is_in_shared_placeholder_set():
    assert "not found" in tsr._LAUNCHER_PATH_PLACEHOLDERS


# --- Guard 1b: graph-collapsed fallback substitutes the device symbol -------
#
# TraceLens' deterministic fallback (graph-collapsed / no-per-op-shapes trace)
# emits rows whose Operation is a literal "—" and whose Kernel Name is the raw
# device symbol. The reader must keep those rows, substituting the device symbol
# as the candidate identity, rather than dropping them.


def test_em_dash_operation_with_kernel_name_survives():
    cand = _fallback_row("—", "_mxfp8_grouped_gemm_kernel")
    assert cand is not None
    assert cand["name"] == "_mxfp8_grouped_gemm_kernel"
    assert cand["device_kernel_name"] == "_mxfp8_grouped_gemm_kernel"


def test_em_dash_operation_without_kernel_name_still_dropped():
    assert _fallback_row("—", "") is None
    # A placeholder kernel-name cell parses to no device symbol -> still dropped.
    assert _fallback_row("—", "—") is None


def test_empty_operation_with_kernel_name_survives():
    cand = _fallback_row("", "_norm_fwd_kernel")
    assert cand is not None
    assert cand["name"] == "_norm_fwd_kernel"
    assert cand["device_kernel_name"] == "_norm_fwd_kernel"


def test_distinct_device_symbols_do_not_collapse_into_one_identity():
    """The critical silent-drift guard.

    A half-applied relaxation that leaves a literal "—" in ``name`` collapses
    every fallback kernel into a single dedup bucket, because the operator
    identity keys on the operation. Two rows naming different device symbols must
    therefore produce two distinct ``operator_identity_key`` values.
    """
    first = _fallback_row("—", "_mxfp8_grouped_gemm_kernel")
    second = _fallback_row("—", "_flash_attn_fwd_kernel")
    assert first is not None and second is not None
    assert first["name"] != "—" and second["name"] != "—"

    # Key both against the same source/function so only the operation varies --
    # exactly how the task-group builder keys them (aggregate_by_source_function).
    shared = {"source_kind": "native", "source_path": "/repo/kernels.cpp", "function_name": "launch"}
    key_first = operator_identity_key(operation=first["name"], **shared)
    key_second = operator_identity_key(operation=second["name"], **shared)
    assert key_first != key_second


# --- Guard 2: bare runtime-API names are not kernels ------------------------


def test_bare_runtime_api_names_detected():
    for name in (
        "hipGraphLaunch",
        "hipModuleLaunchKernel",
        "hipLaunchKernel",
        "cudaLaunchKernel",
        "cudaGraphLaunch",
    ):
        assert tl.is_runtime_api_name(name), name


def test_runtime_api_wrapping_a_kernel_is_not_blocked():
    """The wrapper prefix is stripped, so the real kernel must still resolve."""
    for name in (
        "hipModuleLaunchKernel->_mxfp8_grouped_gemm_kernel (Synthetic Op)",
        "hipGraphLaunch->_mxfp8_linear_kernel (Synthetic Op)",
    ):
        assert not tl.is_runtime_api_name(name), name


def test_grep_refuses_bare_runtime_api():
    for name in ("hipGraphLaunch", "hipModuleLaunchKernel", "hipLaunchKernel"):
        assert tl.locate_source_via_grep(name) == "", name


# --- Guard 3: a source_file must be path-shaped -----------------------------


_SENTINELS = (
    "Not found",
    "N/A",
    "unknown",
    "TBD",
    "<unresolved>",
    "AITER (vendor)",
    "Triton (vendor)",
)


def test_non_path_source_file_is_zeroed():
    """Any value lacking a source extension is rejected, not just 'Not found'.

    Covers the vendor labels TraceLens also emits in this field.
    """
    for sentinel in _SENTINELS:
        item = {"name": "k", "source_file": sentinel, "kernel_repo": "", "source_type": "python"}
        assert tl.reject_non_path_source(item) is True, sentinel
        assert item["source_file"] == "", sentinel
        assert item["source_file_rejected"] == sentinel
        assert item["source_resolution_method"] == "rejected_non_path_sentinel"


def test_rejection_marker_reaches_the_production_path():
    """The audit marker must survive the real _finalize_candidates run.

    The guard used to be duplicated: _finalize_candidates zeroed source_file
    first, which made the copy inside _stamp_candidate_metadata unreachable, so
    source_resolution_method was never stamped in production even though a unit
    test asserted it.
    """
    for sentinel in _SENTINELS:
        item = {"name": "k", "source_file": sentinel, "duration_us": 1.0}
        got = tl._finalize_candidates([item])[0]
        assert got["source_file"] == "", sentinel
        assert got["source_file_rejected"] == sentinel
        assert got["source_resolution_method"] == "rejected_non_path_sentinel", sentinel


def test_successful_grep_replaces_a_prior_rejection_method(monkeypatch):
    """The final method must describe the source that actually won."""
    monkeypatch.setattr(
        tl,
        "locate_source_via_grep",
        lambda _name: "/repo/pkg/kernel.py",
    )
    item = {"name": "k", "source_file": "Not found", "duration_us": 1.0}

    got = tl._finalize_candidates([item])[0]

    assert got["source_file"] == "/repo/pkg/kernel.py"
    assert got["source_resolution_method"] == "name_grep"
    assert got["source_file_rejected"] == "Not found"


def test_path_shape_gate_accepts_every_admitted_extension():
    """Anything grep may admit must also read as a path, or it gets zeroed."""
    for ext in tl.SOURCE_EXTENSIONS:
        assert tl.looks_like_source_path(f"/repo/pkg/kernel{ext}"), ext
    for ext in tl.PATH_SHAPED_EXTENSIONS:
        assert tl.looks_like_source_path(f"/repo/pkg/kernel{ext}"), ext


def test_path_shape_gate_accepts_a_bare_source_filename():
    """A basename from a container trace is still a path, not a sentinel."""
    assert tl.looks_like_source_path("kernel.py")


def test_grep_admission_stays_within_what_source_type_can_classify():
    """The two lists serve different questions and must not be merged.

    Admitting a suffix that source_type_for() cannot classify yields
    source_type="unknown", which classify_patchability rejects -- and the file
    still competes in _rank_paths, where kind_score outweighs ext_score. A
    /csrc/ file with such a suffix therefore outranks the sibling .py and flips
    a routable candidate to non-routable, the exact symptom this module exists
    to fix.
    """
    for ext in tl.SOURCE_EXTENSIONS:
        stype = tl.source_type_for("some_kernel", f"/repo/pkg/some_kernel{ext}")
        assert stype != "unknown", f"{ext} is admitted by grep but unclassifiable"


def test_csrc_sibling_cannot_outrank_the_routable_python_source():
    """End-to-end guard on the ranking interaction described above."""
    csrc = Path("/sgl-workspace/aiter/csrc/kernels/foo_kernel.cc")
    jinja = Path("/sgl-workspace/aiter/csrc/cpp_itfs/foo_kernel.cpp.jinja")
    py = Path("/sgl-workspace/aiter/ops/foo_kernel.py")
    for intruder in (csrc, jinja):
        admitted = [p for p in (intruder, py) if p.suffix in tl.SOURCE_EXTENSIONS]
        assert admitted == [py], f"{intruder.suffix} should not pass grep admission"
        winner = str(tl._rank_paths(admitted, "foo_kernel")[0])
        item = {
            "name": "foo_kernel",
            "source_file": winner,
            "source_type": tl.source_type_for("foo_kernel", winner),
        }
        reusable, reason = tl.classify_patchability(item)
        assert reusable is True, f"{intruder.suffix} made it non-routable: {reason}"


def test_windows_separators_are_not_mistaken_for_placeholders():
    """A backslash path is still a path; is_torch_dispatch_shim_source
    normalizes the same way."""
    assert tl.looks_like_source_path(r"C:\repo\pkg\kernels\moe.py")
    item = {"name": "k", "source_file": r"C:\repo\pkg\kernels\moe.py"}
    assert tl.reject_non_path_source(item) is False


def test_aiter_cross_device_reduce_maps_to_all_reduce():
    """aiter's custom all-reduce must not degrade to a rank-0-only reference.

    The kernels are named cross_device_reduce_{1stage,2stage,half_butterfly}
    and implement all-reduce semantics. Without an explicit tag they fall
    through to the bare "reduce" entry, whose reference is
    torch.distributed.reduce -- correct on rank 0 only, which makes the parity
    gate meaningless everywhere else.
    """
    item = {"name": "_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEv", "is_multigpu": True}
    tl._enrich_kernel_contract(item, {"TP_SIZE": 8})
    contract = item["kernel_contract"]
    assert contract["collective_op"] == "all_reduce"
    assert contract["reference"] == "torch.distributed.all_reduce"
    assert contract["world_size"] == 8


def test_plain_reduce_kernel_still_maps_to_reduce():
    """The new tag must not swallow genuine dist.reduce kernels."""
    item = {"name": "some_reduce_kernel", "is_multigpu": True}
    tl._enrich_kernel_contract(item, {"TP_SIZE": 8})
    assert item["kernel_contract"]["collective_op"] == "reduce"


def test_real_path_is_not_flagged_as_rejected():
    """A genuine path must pass the guard untouched."""
    item = {"name": "k", "source_file": "/repo/pkg/kernels/moe.py"}
    assert tl.reject_non_path_source(item) is False
    assert item["source_file"] == "/repo/pkg/kernels/moe.py"
    assert "source_file_rejected" not in item
    assert "source_resolution_method" not in item


def test_path_shaped_but_absent_file_is_kept_and_flagged(tmp_path):
    """Keyed on shape, not presence: the analysis host need not own the
    serving container's filesystem."""
    missing = str(tmp_path / "does_not_exist.py")
    item = {"name": "k", "source_file": missing, "kernel_repo": "", "source_type": "python"}
    tl._stamp_candidate_metadata(item, None)
    assert item["source_file"] == missing
    assert item["source_file_missing_on_disk"] is True
    assert "source_file_rejected" not in item


def test_package_relative_path_survives():
    """TraceLens emits package-relative launchers such as sgl_kernel/moe.py."""
    item = {"name": "k", "source_file": "sgl_kernel/moe.py", "kernel_repo": "", "source_type": "python"}
    tl._stamp_candidate_metadata(item, None)
    assert item["source_file"] == "sgl_kernel/moe.py"
    assert "source_file_rejected" not in item


def test_existing_source_file_is_preserved_without_flags(tmp_path):
    real = tmp_path / "kernel.py"
    real.write_text("import triton\n", encoding="utf-8")
    item = {"name": "k", "source_file": str(real), "kernel_repo": "", "source_type": "python"}
    tl._stamp_candidate_metadata(item, None)
    assert item["source_file"] == str(real)
    assert "source_file_rejected" not in item
    assert "source_file_missing_on_disk" not in item


def test_empty_source_file_does_not_get_a_rejected_marker():
    item = {"name": "k", "source_file": "", "kernel_repo": "", "source_type": "unknown"}
    tl._stamp_candidate_metadata(item, None)
    assert item["source_file"] == ""
    assert "source_file_rejected" not in item


# --- Guard 4: trace-relative launcher paths are absolutized -----------------
#
# torch profiler records a frame path relative to the sys.path entry the module
# came from ("aiter/dist/x.py"). Patchability keys on an absolute framework
# root, so a relative path would be rejected for the wrong reason.


def test_relative_path_resolves_via_the_installed_package(monkeypatch):
    """The real case: the pinned checkout roots do not exist on this host.

    torch profiler records "vllm/models/x.py" relative to the sys.path entry the
    module came from. A pinned list cannot cover that -- the same package sits
    under /sgl-workspace in the serving image and under dist-packages on a wheel
    install -- so resolution has to locate the package at runtime.
    """
    # Pinned roots deliberately absent, exactly as on a wheel-install host.
    monkeypatch.setattr(tl, "_PACKAGE_INNER_ROOTS", ("/nonexistent/aiter/aiter",))
    tl._package_parent_dir.cache_clear()

    spec = importlib.util.find_spec("json")
    assert spec and spec.origin
    stdlib_dir = Path(spec.origin).parent

    got = tl.absolutize_launcher_path("json/decoder.py")
    assert got == str(stdlib_dir / "decoder.py")
    assert os.path.isfile(got)


def test_pinned_checkout_root_still_works_as_fallback(tmp_path, monkeypatch):
    """An editable checkout that this interpreter cannot import must still resolve."""
    pkg = tmp_path / "aiter" / "aiter" / "dist"
    pkg.mkdir(parents=True)
    (pkg / "custom_all_reduce.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(tl, "_PACKAGE_INNER_ROOTS", (str(tmp_path / "aiter" / "aiter"),))
    tl._package_parent_dir.cache_clear()

    got = tl.absolutize_launcher_path("aiter/dist/custom_all_reduce.py")
    assert got == str(pkg / "custom_all_reduce.py")


def test_non_identifier_head_is_not_probed_as_a_package(monkeypatch):
    monkeypatch.setattr(tl, "_PACKAGE_INNER_ROOTS", ())
    tl._package_parent_dir.cache_clear()
    assert tl.absolutize_launcher_path("not-an-identifier/mod.py") == "not-an-identifier/mod.py"


def test_absolute_launcher_path_is_returned_unchanged():
    assert tl.absolutize_launcher_path("/sgl-workspace/x.py") == "/sgl-workspace/x.py"


def test_unresolvable_relative_path_is_left_alone(monkeypatch, tmp_path):
    """Never fabricate: an unjoinable path stays as-is rather than becoming
    a plausible-looking path that does not exist."""
    monkeypatch.setattr(tl, "_PACKAGE_INNER_ROOTS", (str(tmp_path / "nope" / "nope"),))
    assert tl.absolutize_launcher_path("pkg/mod.py") == "pkg/mod.py"


def test_empty_path_is_safe():
    assert tl.absolutize_launcher_path("") == ""


# --------------------------------------------------------------------------
# End-to-end wiring: sentinel -> zeroed source_file -> trace-derived launcher.
# The guards above pin each stage in isolation; these pin the seam between
# them, which is where a regression would actually cost a candidate.
# --------------------------------------------------------------------------

_WIRING_KERNEL = "_mxfp8_grouped_gemm_kernel"
_WIRING_FRAME = "/repo/pkg/kernels/moe.py(124): _grouped_gemm"


def _wiring_trace(tmp_path, frame: str = _WIRING_FRAME):
    """Minimal trace where a Python frame launches ``_WIRING_KERNEL``."""
    events = [
        {"cat": "python_function", "name": "/repo/serve/runner.py(10): forward", "tid": 7, "ts": 100.0, "dur": 200.0},
        {"cat": "python_function", "name": frame, "tid": 7, "ts": 101.0, "dur": 198.0},
        {
            "cat": "cuda_runtime",
            "name": "hipModuleLaunchKernel",
            "tid": 7,
            "ts": 150.0,
            "dur": 5.0,
            "args": {"correlation": 1},
        },
        {"cat": "kernel", "name": _WIRING_KERNEL, "ts": 200.0, "dur": 10.0, "args": {"correlation": 1}},
    ]
    path = tmp_path / "wiring.json"
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    return path


def _wiring_candidate():
    """Candidate as TraceLens emits it for a Synthetic Op: sentinel source."""
    return {
        "name": f"hipModuleLaunchKernel->{_WIRING_KERNEL} (Synthetic Op)",
        "device_kernel_name": _WIRING_KERNEL,
        "source_file": "Not found",
        "duration_us": 1000.0,
        "gpu_pct": 10.0,
    }


def test_sentinel_candidate_gets_trace_derived_source(monkeypatch, tmp_path):
    """The whole point of the change: "Not found" must not end the search."""
    monkeypatch.setattr(
        tl,
        "locate_source_via_grep",
        lambda _name: "/repo/pkg/kernels/moe.py",
    )
    got = tl._finalize_candidates([_wiring_candidate()], trace_files=[_wiring_trace(tmp_path)])[0]
    assert got["source_file"] == "/repo/pkg/kernels/moe.py"
    assert got["source_line"] == 124
    assert got["source_function"] == "_grouped_gemm"
    assert got["source_resolution_method"] == "trace_python_stack"
    # The rejected sentinel is kept for audit rather than silently dropped.
    assert got["source_file_rejected"] == "Not found"


def test_trace_launcher_caller_does_not_override_grep_definition(
    monkeypatch,
    tmp_path,
):
    """A launcher call site must not replace the kernel definition found by grep."""
    launcher = "/repo/model/launcher.py"
    definition = "/repo/kernels/grouped_gemm_kernel.py"
    monkeypatch.setattr(tl, "locate_source_via_grep", lambda _name: definition)

    got = tl._finalize_candidates(
        [_wiring_candidate()],
        trace_files=[_wiring_trace(tmp_path, f"{launcher}(42): launch")],
    )[0]

    assert got["source_file"] == definition
    assert got["source_resolution_method"] == "name_grep"
    assert got["trace_launcher_file"] == launcher
    assert got.get("source_line") is None
    assert got.get("source_function") is None
    assert "trace launcher differs from grep source" in got["source_resolution_reason"]


def test_grep_prefers_defining_module_over_reexporting_init(monkeypatch, tmp_path):
    """A package ``__init__`` that re-exports a kernel is not its source.

    The re-exporter scores well on path shape alone, so plain ranking put it
    ahead of the module holding the ``@triton.jit`` body and handed a backend a
    file with no kernel in it.
    """
    pkg = tmp_path / "pkg" / "kernels" / "linear"
    (pkg / "mxfp8").mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        'from .mxfp8.rocm_native import _mxfp8_linear_kernel\n\n__all__ = ["_mxfp8_linear_kernel"]\n',
        encoding="utf-8",
    )
    definition = pkg / "mxfp8" / "rocm_native.py"
    definition.write_text(
        "import triton\n\n\n@triton.jit\ndef _mxfp8_linear_kernel(a, b):\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(tl, "kernel_search_roots", lambda: (str(tmp_path),))

    assert tl.locate_source_via_grep("_mxfp8_linear_kernel") == str(definition)


def test_grep_resolves_a_name_an_fstring_assembled_at_runtime(monkeypatch, tmp_path):
    """Only the head of an f-string-built kernel name is literal in source.

    aiter's MoE GEMMs launch as ``mfma_moe1_silu_mul_afp8_wfp8_bf16_...`` but
    are written ``f"mfma_moe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}"``, so both
    the whole-name pass and the trailing sub-window pass search text that is
    never written down, and the hottest kernels on the trace resolved to
    nothing.
    """
    root = tmp_path / "aiter" / "ops" / "flydsl" / "kernels"
    root.mkdir(parents=True)
    composer = root / "mixed_moe_gemm_2stage.py"
    composer.write_text(
        'name = f"mfma_moe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(tl, "kernel_search_roots", lambda: (str(tmp_path),))

    launched = "mfma_moe1_silu_mul_afp8_wfp8_bf16_t32x128x256_pm1_swiglu_v32"
    assert tl.locate_source_via_grep(launched) == str(composer)


def test_grep_prefers_the_file_that_spells_out_more_of_a_composed_name(
    monkeypatch,
    tmp_path,
):
    """Sibling f-strings share a short prefix; characters break the tie.

    ``mfma_moe2_a{a_dtype}...`` and ``mfma_moe2_{in_dtype}...`` both answer to
    the keyword ``mfma_moe2``, and only the first one built the launched name.
    """
    root = tmp_path / "aiter" / "ops" / "flydsl" / "kernels"
    root.mkdir(parents=True)
    mixed = root / "mixed_moe_gemm_2stage.py"
    mixed.write_text(
        'n = f"mfma_moe2_a{a_dtype}_w{b_dtype}_{out_s}_cshuffle"\n',
        encoding="utf-8",
    )
    plain = root / "moe_gemm_2stage.py"
    plain.write_text(
        'n = f"mfma_moe2_{in_dtype}_{out_s}_{epilog_tag}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(tl, "kernel_search_roots", lambda: (str(tmp_path),))

    launched = "mfma_moe2_afp8_wfp8_bf16_cshuffle_t32x128x256_vscale_v1_pm1"
    assert tl.locate_source_via_grep(launched) == str(mixed)


def test_grep_falls_back_to_ranking_when_no_file_defines_the_symbol(
    monkeypatch,
    tmp_path,
):
    """Preferring definitions must not drop mention-only hits on the floor."""
    root = tmp_path / "pkg" / "kernels"
    root.mkdir(parents=True)
    mention = root / "dispatch.py"
    mention.write_text("launch(_mxfp8_linear_kernel, grid)\n", encoding="utf-8")
    monkeypatch.setattr(tl, "kernel_search_roots", lambda: (str(tmp_path),))

    assert tl.locate_source_via_grep("_mxfp8_linear_kernel") == str(mention)


def test_unconfirmed_trace_launcher_leaves_the_source_empty(
    monkeypatch,
    tmp_path,
):
    """An unconfirmed launcher is evidence, never an attribution.

    Finalization stops here rather than guessing from the symbol; the
    whole-table review that follows can weigh the blank against the launcher
    frame and the rest of the table.
    """
    launcher = "/repo/model/launcher.py"
    monkeypatch.setattr(tl, "locate_source_via_grep", lambda _name: "")

    got = tl._finalize_candidates(
        [_wiring_candidate()],
        trace_files=[_wiring_trace(tmp_path, f"{launcher}(42): launch")],
    )[0]

    assert got["source_file"] == ""
    assert got["reusable_native_kernel"] is False
    assert got["trace_launcher_file"] == launcher
    assert got.get("source_resolution_method") != "trace_python_stack"
    assert "trace launcher unconfirmed by name grep" in got["source_resolution_reason"]


def test_wiring_leaves_a_real_source_untouched(tmp_path):
    """A candidate that already resolved must not be rewritten by the trace."""
    item = _wiring_candidate()
    item["source_file"] = "/repo/pkg/kernels/authoritative.py"
    got = tl._finalize_candidates([item], trace_files=[_wiring_trace(tmp_path)])[0]
    assert got["source_file"] == "/repo/pkg/kernels/authoritative.py"
    assert got.get("source_resolution_method") != "trace_python_stack"


def test_wiring_without_trace_files_falls_back_quietly(tmp_path):
    """No trace supplied is the normal bypass path, not a failure to report."""
    got = tl._finalize_candidates([_wiring_candidate()], trace_files=None)[0]
    assert got.get("source_resolution_method") != "trace_python_stack"
    assert "trace_resolver_error" not in str(got.get("source_resolution_reason", ""))


def test_finalization_never_calls_a_model(monkeypatch, tmp_path):
    """Source resolution is wholly deterministic; review is a later stage.

    Nothing under ``_finalize_candidates`` may reach a provider, so the
    deterministic route gets its no-LLM guarantee from the code rather than
    from a flag it has to remember to pass.
    """
    monkeypatch.setattr(tl, "locate_source_via_grep", lambda _name: "")
    artifact = tmp_path / "kernel_source_resolution.json"
    item = {
        "name": "zz_no_source_kernel",
        "source_file": "",
        "duration_us": 100.0,
        "gpu_pct": 10.0,
    }
    got = tl._finalize_candidates([item], source_resolution_out=artifact)[0]
    assert got["source_file"] == ""
    assert got["skip_reason"] == "source file not resolved"
    assert artifact.is_file()
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    assert "llm_audit" not in doc


def test_unreadable_trace_records_a_reason(monkeypatch, tmp_path):
    """P2-1: a resolver failure must be distinguishable from "nothing to do"."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("corrupt trace")

    monkeypatch.setattr(tl, "_resolve_trace_launchers", tl._resolve_trace_launchers)
    monkeypatch.setattr("_trace_launcher_resolver.resolve_launchers_from_trace", _boom)
    got = tl._finalize_candidates([_wiring_candidate()], trace_files=[_wiring_trace(tmp_path)])[0]
    reason = str(got.get("source_resolution_reason", ""))
    assert reason.startswith("trace_resolver_error: RuntimeError")
    assert got.get("source_resolution_method") != "trace_python_stack"
