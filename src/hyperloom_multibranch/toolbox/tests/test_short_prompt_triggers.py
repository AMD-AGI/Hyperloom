# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which recipe changes run the short-prompt check (board #126 c2813): an attention or prefill patch by the files it
changes, read from the patch itself; the launch variables of attention, the KV layout and aiter's attention backend, and
not aiter's all-reduce, GEMM tables or MoE sorting."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from hyperloom.common.outside_facts import SHORT_PROMPT_TOUCH
from hyperloom.contracts.measurement import Purpose
from hyperloom.contracts.recipe import ComponentChange, ComponentPatches, LaunchConfig, PatchRef, Recipe, RecipeDelta
from hyperloom.measure.compile import LocalBlobStore
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.tests._recipes import git_patch, put_patch
from hyperloom_multibranch.toolbox.acceptance import Acceptance, AcceptanceSetup
from hyperloom_multibranch.toolbox.short_prompts import ShortPromptSetup, blob_patch_files, decide, no_patch_files
from hyperloom_multibranch.toolbox.tests.helpers import IMAGE, FakeRepo, req

SETUP = ShortPromptSetup("short-prompts-test", "a" * 64)
SGLANG = ("sglang", "https://example.invalid/sglang", "4" * 40)
VLLM = ("vllm", "https://example.invalid/vllm", "5" * 40)
AITER = ("aiter", "https://example.invalid/aiter", "6" * 40)
ATTENTION_FILE = "python/sglang/srt/layers/attention/aiter_backend.py"
MOE_FILE = "python/sglang/srt/layers/moe/fused_moe_triton/layer.py"
PARENT = Recipe(base_image_digest=IMAGE, launch=LaunchConfig("--max-running-requests 128", {}))


def _with_envs(envs: dict[str, str]) -> Recipe:
    return Recipe(base_image_digest=IMAGE, launch=LaunchConfig("--max-running-requests 128", envs))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SGLANG_USE_AITER", "1"),
        ("VLLM_ROCM_USE_AITER", "1"),
        ("SGLANG_USE_AITER_ATTN", "1"),
        ("AITER_ENABLE_PREFILL_VSKIP", "1"),
        ("SGLANG_AITER_MLA_PERSIST", "1"),
        ("SGLANG_AITER_KV_CACHE_LAYOUT", "vectorized_5d"),
    ],
)
def test_an_attention_kv_layout_or_aiter_backend_variable_runs_the_check(name: str, value: str) -> None:
    plan = decide(req(), PARENT, _with_envs({name: value}), SETUP, no_patch_files)

    assert plan is not None and plan.trigger == (f"env {name}={value}",)


@pytest.mark.parametrize(
    "name",
    [
        "AITER_CUSTOM_AR_MAX_SIZE",
        "SGLANG_USE_AITER_AR",
        "AITER_USE_CK_MOE_SORTING",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
        "AITER_CONFIG_GEMM_BF16",
        "AITER_CONFIG_FMOE",
    ],
)
def test_aiters_all_reduce_gemm_table_and_moe_sorting_variables_do_not(name: str) -> None:
    assert decide(req(), PARENT, _with_envs({name: "/configs/tuned.csv"}), SETUP, no_patch_files) is None


def _sglang(
    blobs: LocalBlobStore, tmp_path: Path, path: str, component: tuple[str, str, str] = SGLANG
) -> tuple[Recipe, Recipe, PatchRef]:
    """A parent with ``component`` (sglang) and no patch, and a candidate with one patch, stored in ``blobs``, that edits
    ``path``."""
    data = git_patch(tmp_path / "patch", {path: "x = 1\n"}, {path: "x = 2\n"})
    patch = PatchRef(f"patch/{component[0]}/{blobs.put(data)}.patch", blobs.put(data))
    parent = Recipe(base_image_digest=IMAGE, components=(ComponentPatches(*component),))
    return parent, Recipe(base_image_digest=IMAGE, components=(ComponentPatches(*component, (patch,)),)), patch


def test_an_sglang_patch_runs_the_check_by_an_attention_file_it_changes(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    parent, candidate, patch = _sglang(blobs, tmp_path, ATTENTION_FILE)

    plan = decide(req(), parent, candidate, SETUP, blob_patch_files(blobs))

    assert plan is not None and plan.trigger == (f"patch-file sglang {ATTENTION_FILE}",)
    assert not re.search(SHORT_PROMPT_TOUCH, f"patch sglang {patch.path}"), "the hashed path alone says nothing"


def test_an_sglang_patch_that_changes_no_attention_file_does_not(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    parent, candidate, _ = _sglang(blobs, tmp_path, MOE_FILE)

    assert decide(req(), parent, candidate, SETUP, blob_patch_files(blobs)) is None


#: Files whose patch changes what attention, the KV cache or prefill computes: a model's attention forward, the
#: batch and graph the attention runs in, the KV cache's blocks, aiter's attention kernels.
_TOUCHING_FILES = [
    (SGLANG, "python/sglang/srt/models/deepseek_v2.py"),
    (SGLANG, "python/sglang/srt/models/llama.py"),
    (SGLANG, "python/sglang/srt/model_executor/forward_batch_info.py"),
    (SGLANG, "python/sglang/srt/model_executor/model_runner.py"),
    (SGLANG, "python/sglang/srt/managers/schedule_batch.py"),
    (SGLANG, "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"),
    (SGLANG, "python/sglang/srt/layers/rotary_embedding/base.py"),
    (VLLM, "csrc/cache_kernels.cu"),
    (VLLM, "vllm/v1/core/block_pool.py"),
    (AITER, "aiter/ops/triton/attention/lean_atten_paged.py"),
    (AITER, "aiter/ops/triton/attention/pa_decode.py"),
    (AITER, "csrc/py_itfs_cu/asm_pa.cu"),
]
#: Files whose names hold an attention word inside another word (point 1), and aiter's GEMM and MoE (the optional point).
_OTHER_FILES = [
    (SGLANG, "python/sglang/srt/function_call/deepseekv3_detector.py"),
    (VLLM, "vllm/tool_parsers/deepseekv3_tool_parser.py"),
    (SGLANG, "python/sglang/srt/configs/deepseekvl2.py"),
    (SGLANG, "python/sglang/srt/speculative/dflash_info.py"),
    (SGLANG, "python/sglang/kernels/aot/csrc/cpu/qkv_proj.cpp"),
    (SGLANG, "python/sglang/kernels/ops/moe/moe_route_radix.py"),
    (AITER, "aiter/ops/gemm_op_a8w8.py"),
    (AITER, "aiter/fused_moe.py"),
]


@pytest.mark.parametrize(("component", "path"), _TOUCHING_FILES, ids=[path for _, path in _TOUCHING_FILES])
def test_a_patch_to_a_model_batch_graph_kv_block_or_aiter_attention_file_runs_the_check(
    tmp_path: Path, component: tuple[str, str, str], path: str
) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    parent, candidate, _ = _sglang(blobs, tmp_path, path, component)

    plan = decide(req(), parent, candidate, SETUP, blob_patch_files(blobs))

    assert plan is not None and plan.trigger == (f"patch-file {component[0]} {path}",)


@pytest.mark.parametrize(("component", "path"), _OTHER_FILES, ids=[path for _, path in _OTHER_FILES])
def test_a_patch_to_a_tool_parser_vision_config_draft_model_gemm_or_moe_file_does_not(
    tmp_path: Path, component: tuple[str, str, str], path: str
) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    parent, candidate, _ = _sglang(blobs, tmp_path, path, component)

    assert decide(req(), parent, candidate, SETUP, blob_patch_files(blobs)) is None


def test_removing_an_attention_patch_runs_it_as_much_as_adding_one(tmp_path: Path) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    without, with_patch, _ = _sglang(blobs, tmp_path, ATTENTION_FILE)

    plan = decide(req(), with_patch, without, SETUP, blob_patch_files(blobs))

    assert plan is not None and plan.trigger == (f"patch-file sglang {ATTENTION_FILE}",)


@pytest.mark.parametrize(
    ("stored", "why"),
    [(None, "(BlobError: "), (b"not a patch\n", "(PatchApplyError: ")],
    ids=["missing", "not-a-patch"],
)
def test_a_patch_whose_files_cannot_be_read_runs_the_check_and_says_why(
    tmp_path: Path, stored: bytes | None, why: str
) -> None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    digest = blobs.put(stored) if stored is not None else "7" * 64
    patch = PatchRef(f"patch/sglang/{digest}.patch", digest)
    candidate = Recipe(base_image_digest=IMAGE, components=(ComponentPatches(*SGLANG, (patch,)),))
    parent = Recipe(base_image_digest=IMAGE, components=(ComponentPatches(*SGLANG),))

    plan = decide(req(), parent, candidate, SETUP, blob_patch_files(blobs))

    assert plan is not None and len(plan.trigger) == 1
    assert plan.trigger[0].startswith(f"unexpressed patch {patch.path}: its files cannot be read")
    assert why in plan.trigger[0]


@pytest.mark.parametrize(
    ("purpose", "fields", "trigger"),
    [
        (Purpose.FINAL, {}, "purpose final"),
        (Purpose.BASELINE, {}, "purpose baseline"),
        (Purpose.INTEGRATION, {"consistency_check": True}, "request consistency_check"),
    ],
    ids=["final", "baseline", "lead-adoption"],
)
def test_final_baseline_and_lead_adoption_still_run_it_with_no_change(purpose, fields, trigger) -> None:
    plan = decide(req(purpose=purpose, **fields), PARENT, PARENT, SETUP, no_patch_files)

    assert plan is not None and plan.trigger == (trigger,)


class _Planner:
    """A planner that compiles from ``blobs`` (or names none)."""

    def __init__(self, blobs: LocalBlobStore | None) -> None:
        self.blobs = blobs

    def plan(self, request, parent) -> AcceptedMeasurement:  # type: ignore[no-untyped-def]
        return AcceptedMeasurement(None, ())  # type: ignore[arg-type]


def _accepted_trigger(tmp_path: Path, path: str, *, named: bool) -> tuple[str, ...] | None:
    blobs = LocalBlobStore(tmp_path / "blobs")
    patch = put_patch(blobs, "sglang/edit.patch", git_patch(tmp_path / "patch", {path: "a\n"}, {path: "b\n"}))
    repo = FakeRepo(Recipe(base_image_digest=IMAGE, components=(ComponentPatches(*SGLANG),)))
    acceptance = Acceptance(
        AcceptanceSetup(repo, _Planner(blobs if named else None), IMAGE, short_prompts=SETUP),  # type: ignore[arg-type]
        is_accepted=lambda _id: True,
    )
    change = RecipeDelta(components=(ComponentChange("sglang", add_patches=(patch,)),))
    plan = asyncio.run(acceptance.prepare(req(change=change))).environment.short_prompts
    return None if plan is None else plan.trigger


def test_acceptance_reads_a_patchs_files_from_its_planners_blob_store(tmp_path: Path) -> None:
    assert _accepted_trigger(tmp_path / "a", ATTENTION_FILE, named=True) == (f"patch-file sglang {ATTENTION_FILE}",)
    assert _accepted_trigger(tmp_path / "b", MOE_FILE, named=True) is None


def test_acceptance_whose_planner_names_no_blob_store_counts_every_patch_as_touching(tmp_path: Path) -> None:
    trigger = _accepted_trigger(tmp_path, MOE_FILE, named=False)

    assert trigger is not None and "cannot be read" in trigger[0] and "no blob store" in trigger[0]
