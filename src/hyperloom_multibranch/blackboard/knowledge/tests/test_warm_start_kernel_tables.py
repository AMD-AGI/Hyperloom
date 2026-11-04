# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A warm-start replay stacks the champion's kernel table over the head's in the head's slot, as the merge stacks two
tables, instead of replacing the head's table.

It reads tables as the toolbox mints them and stacks them with the repository's merge."""

from __future__ import annotations

from dataclasses import replace

from hyperloom.contracts.recipe import ComponentPatches, KernelKind, LaunchConfig, Recipe, apply
from hyperloom_multibranch.blackboard.knowledge.warm_start import PreparedReplay
from hyperloom_multibranch.repo.merge import wider_table
from hyperloom_multibranch.toolbox.kernel_artifacts import mint_kernel_artifact
from hyperloom_multibranch.toolbox.tests.kernel_rows import ALPHA_NEW, ENV, SHARED, csv_text

IMAGE = "sha256:" + "a" * 64
VLLM = ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40)
ROOT = Recipe(IMAGE, LaunchConfig("--max-num-seqs 128"), (VLLM,))
CHAMPION = Recipe(IMAGE, LaunchConfig("--max-num-seqs 256", {"VLLM_USE_AITER": "1"}), (VLLM,))


def test_a_replay_stacks_the_champions_table_over_the_heads_instead_of_replacing_it() -> None:
    narrow = mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, csv_text(SHARED))
    wide = mint_kernel_artifact(KernelKind.GEMM, "a8w8", ENV, csv_text(SHARED, ALPHA_NEW))
    store = {narrow.sha256: csv_text(SHARED), wide.sha256: csv_text(SHARED, ALPHA_NEW)}
    head = replace(ROOT, kernels=(narrow,))
    champion = replace(CHAMPION, kernels=(wide,))
    found = PreparedReplay("inference:champion", champion, champion.launch)

    on_head = found.on(head, "c3", lambda wanted, current: wider_table(wanted, current, lambda a: store[a.sha256]))

    assert on_head is not None
    assert on_head.delta.add_kernels == (wide,)
    assert apply(head, on_head.delta).kernels == (wide,)
