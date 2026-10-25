# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, Recipe
from hyperloom_multibranch.specialist.following import integration_holdings
from hyperloom_multibranch.tests.branch_rig import branch_rig
from hyperloom_multibranch.tests.following_rig import diverged

BRANCH = "s1"
ROOT = Recipe(
    "rocm/vllm@sha256:" + "c" * 64,
    LaunchConfig(server_args="--max-num-seqs 128"),
    components=(ComponentPatches("vllm", "https://github.com/vllm-project/vllm", "1" * 40),),
)


async def test_the_holdings_name_the_integration_launch_its_merged_intents_and_what_the_head_lacks(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    session = await diverged(rig.repo, tmp_path / "notebook", rig.root, ROOT, BRANCH)

    text = await integration_holdings(rig.repo, await rig.notebook.snapshot(), session.own)

    assert (
        f"The integration head is {session.onto}: server arguments --max-num-seqs 128 --enable-chunked-prefill" in text
    )
    merged = f"- {session.merged[:12]} of s0 (+3.00%), scheduler / scheduler change: add args --enable-chunked-prefill"
    assert f"{merged} [not on your head]" in text
    assert f"Your head {session.own} lacks of it: --enable-chunked-prefill." in text
    assert f"- {session.own[:12]} of s1 (+2.00%), attention / attention change" in text
    assert "the integrator has not tried it yet" in text


async def test_a_head_on_the_integration_head_lacks_nothing_of_it(tmp_path):
    rig = await branch_rig(tmp_path, ROOT, BRANCH)
    session = await diverged(rig.repo, tmp_path / "notebook", rig.root, ROOT, BRANCH)

    text = await integration_holdings(rig.repo, await rig.notebook.snapshot(), session.onto)

    assert f"Your head {session.onto} holds all of it." in text and "[not on your head]" not in text
    assert "Commits of your head" not in text
