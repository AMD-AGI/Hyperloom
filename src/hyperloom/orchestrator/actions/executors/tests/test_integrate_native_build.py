# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A candidate in a session that builds its framework server from source is built before it is measured: the bench runs
that build, a candidate that does not build is reverted unmeasured, and a build the machine failed is no verdict."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hyperloom.orchestrator.actions.executors.candidate_builds import TreeBuild, TreeBuildStatus
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.state.shared_state import SharedState

PATCH = """\
diff --git a/ggml/a.c b/ggml/a.c
--- a/ggml/a.c
+++ b/ggml/a.c
@@ -1 +1 @@
-int a;
+int a = 2;
"""


@dataclass
class FakeBuilds:
    checkout: Path
    answer: TreeBuild
    asked: list[list[str]] = field(default_factory=list)
    accepted: list[TreeBuild] = field(default_factory=list)

    def build_tree(self, paths=()) -> TreeBuild:
        self.asked.append(list(paths))
        return self.answer

    def accept(self, build: TreeBuild) -> None:
        self.accepted.append(build)


@dataclass
class Run:
    result: dict[str, Any]
    benched: list[dict[str, Any]]
    builds: FakeBuilds


async def _integrate(
    tmp_path: Path, answer: TreeBuild, *, verdict: str = "kept", apply_only: bool = False, root: Path | None = None
) -> Run:
    checkout = tmp_path / "llama.cpp"
    (checkout / "ggml").mkdir(parents=True)
    (checkout / "ggml" / "a.c").write_text("int a = 2;\n")
    patch = tmp_path / "001.patch"
    patch.write_text(PATCH)
    builds = FakeBuilds(checkout, answer)
    executor = IntegratePatchExecutor(session_dir=tmp_path / "session", native=builds)
    benched: list[dict[str, Any]] = []

    async def _resolve(attempt, params, extra):
        attempt.specialist_task_id = "t-spec"
        attempt.shared_state = SharedState(baseline_tput=40.0)

    async def _apply(attempt, params, extra):
        attempt.framework_root = root or checkout
        attempt.output_root = tmp_path / "run"
        attempt.applied.append(patch)
        return {"status": "applied_no_bench"} if apply_only else None

    async def _gate(attempt, params):
        benched.append(attempt.benchmark_params(params))
        return {"status": verdict}

    async def _nothing(*_args, **_kwargs):
        return None

    executor._stage_resolve = _resolve
    executor._stage_provision_attempt_runtime = _nothing
    executor._stage_localize_source = _nothing
    executor._stage_apply = _apply
    executor._stage_gate = _gate
    result = await executor(SimpleNamespace(task=SimpleNamespace(task_id="t-int", params={}), extra={}))
    return Run(result, benched, builds)


BUILT = TreeBuild(TreeBuildStatus.BUILT, "k" * 64, False, "", Path("/farms/k/llamacpp/bin"))


@pytest.mark.asyncio
async def test_a_kept_candidate_is_measured_on_its_build_which_every_later_round_then_runs(tmp_path: Path) -> None:
    run = await _integrate(tmp_path, BUILT)

    assert run.builds.asked == [["ggml/a.c"]]
    assert run.benched[0]["runtime_override"]["path_prefix"] == "/farms/k/llamacpp/bin"
    assert run.result["native_build"] == {"status": "built", "key": "k" * 64, "reused": False, "detail": ""}
    assert run.builds.accepted == [BUILT]


@pytest.mark.asyncio
async def test_a_reverted_candidate_leaves_the_accepted_build_as_it_is(tmp_path: Path) -> None:
    run = await _integrate(tmp_path, BUILT, verdict="reverted")

    assert run.result["native_build"]["key"] == "k" * 64
    assert run.builds.accepted == []


@pytest.mark.asyncio
@pytest.mark.parametrize("apply_only", [False, True], ids=["bench", "apply_only"])
async def test_a_candidate_that_does_not_compile_is_reverted_unmeasured(tmp_path: Path, apply_only: bool) -> None:
    failed = TreeBuild(TreeBuildStatus.COMPILE_ERROR, "e" * 64, False, "compile_error: ggml/a.c:1: error: stop")

    run = await _integrate(tmp_path, failed, apply_only=apply_only)

    assert (run.result["status"], run.result["error_class"]) == ("reverted", "native_compile_error")
    assert "ggml/a.c:1: error: stop" in run.result["reason"]
    assert run.result["native_build"]["status"] == "compile_error"
    assert (run.benched, run.builds.accepted) == ([], [])
    assert run.result["patches_reverted"] == [str(tmp_path / "001.patch")]


@pytest.mark.asyncio
async def test_a_build_the_machine_failed_is_no_verdict_on_the_candidate(tmp_path: Path) -> None:
    run = await _integrate(tmp_path, TreeBuild(TreeBuildStatus.INFRA, "i" * 64, False, "the build runner exited 1"))

    assert (run.result["status"], run.result["error_class"], run.result["failure_attribution"]) == (
        "failed",
        "native_build_infra",
        "environment",
    )
    assert (run.benched, run.builds.accepted) == ([], [])


@pytest.mark.asyncio
async def test_an_apply_only_round_is_built_and_its_build_runs_from_then_on(tmp_path: Path) -> None:
    run = await _integrate(tmp_path, BUILT, apply_only=True)

    assert run.result["status"] == "applied_no_bench"
    assert run.result["native_build"]["key"] == "k" * 64
    assert (run.benched, run.builds.accepted) == ([], [BUILT])


@pytest.mark.asyncio
async def test_a_candidate_in_another_tree_than_the_session_builds_is_not_measured(tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()

    run = await _integrate(tmp_path, BUILT, root=other)

    assert (run.result["status"], run.result["error_class"]) == ("failed", "native_build_infra")
    assert (run.builds.asked, run.benched) == ([], [])
