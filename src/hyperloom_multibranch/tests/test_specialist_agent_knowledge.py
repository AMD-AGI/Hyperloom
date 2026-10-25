# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

from hyperloom.contracts.recipe import ComponentPatches, LaunchConfig, PatchRef, Recipe
from hyperloom_multibranch.contracts.notebook import INTEGRATION_BRANCH, Role, Writer
from hyperloom_multibranch.contracts.notebook.records import LineageEvent, MergeAttempt
from hyperloom_multibranch.contracts.notebook.vocab import Conclusion, LineageKind
from hyperloom_multibranch.notebook import LocalNotebook
from hyperloom_multibranch.notebook.tests.factories import TTL, delta, intent, judgement, request, result
from hyperloom_multibranch.specialist.agent_knowledge import IntegrationNotes

PATCH = PatchRef("patch/sglang/" + "a" * 64 + ".patch", "a" * 64)
HEAD = Recipe(
    "rocm/sglang@sha256:" + "c" * 64,
    LaunchConfig("--kv-cache-dtype fp8_e4m3", {"SGLANG_USE_AITER": "1"}),
    components=(ComponentPatches("sglang", "https://github.com/sgl-project/sglang", "1" * 40, (PATCH,)),),
)


class Repo:
    """The integration branch at ``heads[-1]``, every commit with :data:`HEAD`'s recipe."""

    def __init__(self) -> None:
        self.heads = ["c0"]

    async def head(self, branch: str) -> str:
        assert branch == INTEGRATION_BRANCH
        return self.heads[-1]

    async def read_recipe(self, commit: str) -> Recipe:
        return HEAD if commit != "c0" else Recipe(HEAD.base_image_digest)

    async def checkout(self, commit: str, destination: Path) -> None:
        if commit != "c0":
            (destination / PATCH.path).parent.mkdir(parents=True)
            (destination / PATCH.path).write_text("--- a\n+++ b\n")


async def _merged(root: Path) -> LocalNotebook:
    """b02 commits c1 with its intent, the integrator merges it as i1; b03's c2 is never merged."""

    def open_as(role: Role, branch: str = "") -> LocalNotebook:
        return LocalNotebook(root, Writer(role, branch or role.value, branch))

    scheduler, toolbox, gate = open_as(Role.SCHEDULER), open_as(Role.TOOLBOX), open_as(Role.GATE)
    integrator = open_as(Role.INTEGRATOR)
    await scheduler.record_lineage(LineageEvent(LineageKind.ROOT, INTEGRATION_BRANCH, commit_id="c0"))
    changes = {
        "b02": ("r1", "c1", "attention prologue", "fused qk norm and rope", "--max-num-seqs 256"),
        "b03": ("r2", "c2", "moe", "x", "--max-num-seqs 512"),
    }
    for branch, (request_id, commit, target, mechanism, args) in changes.items():
        await scheduler.record_lineage(LineageEvent(LineageKind.OPEN, branch, from_commit="c0"))
        of, change = intent(branch, "c0", target, mechanism), delta(args)
        claim = await open_as(Role.SPECIALIST_RUNTIME, branch).acquire(of, change.fingerprint(), ttl_seconds=TTL)
        await toolbox.write_request(request(request_id, of, change))
        await toolbox.attach_request(claim.key, request_id, ttl_seconds=TTL)
        await toolbox.write_result(result(request_id))
        await gate.write_judgement(request_id, judgement(3.3, (request_id,)), Conclusion.COMMITTED, commit)
    merge = intent(INTEGRATION_BRANCH, "c0", "merge", "c1")
    await toolbox.write_request(request("m1", merge))
    await integrator.record_merge_attempt(MergeAttempt(("c1",), "c0", request_id="m1"))
    await toolbox.write_result(result("m1"))
    await integrator.write_judgement("m1", judgement(3.2, ("m1",)), Conclusion.MERGED, "i1")
    return open_as(Role.SPECIALIST_RUNTIME, "b01")


async def test_the_notes_are_rewritten_when_the_integration_head_moves_with_its_launch_and_what_merged(
    tmp_path: Path,
) -> None:
    repo, notebook = Repo(), await _merged(tmp_path / "notebook")
    path = tmp_path / "agent" / "mb-integration.md"
    notes = IntegrationNotes(repo, notebook, path)

    assert await notes.refresh()
    assert "c0" in path.read_text() and "fp8_e4m3" not in path.read_text()
    assert not await notes.refresh()
    repo.heads.append("i1")
    assert await notes.refresh()

    text = path.read_text()
    assert "i1" in text
    assert "--kv-cache-dtype fp8_e4m3" in text and "SGLANG_USE_AITER=1" in text
    patch = path.parent / "mb-integration" / PATCH.path
    assert str(patch) in text and patch.read_text() == "--- a\n+++ b\n"
    assert "b02" in text and "attention prologue" in text and "fused qk norm and rope" in text and "+3.30%" in text
    assert "b03" not in text
