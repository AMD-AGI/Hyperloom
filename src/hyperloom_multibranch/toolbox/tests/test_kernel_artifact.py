# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``put_kernel_artifact``: a tuned table a specialist puts reaches the test as a file in its sandbox (CPU only).

Everything goes through the entry points: the MCP tool a specialist calls with its own token, ``submit_test`` on the
same service, the toolbox that runs the test in a local sandbox (the runner there is the file-protocol stand-in, which
places the sheet's attachments with the shipped placement and reports what the variable holds), and the experiment
repository the artifact is committed to, merged in and checked out from.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from hyperloom.contracts.measurement import TestRequest, Validity
from hyperloom.contracts.recipe import ConfigDelta, KernelArtifact, KernelKind, RecipeDelta, RecipeError, apply
from hyperloom.measure.compile import LocalBlobStore, kernel_load_env
from hyperloom.measure.points import AcceptedMeasurement, default_canonical
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.measure.tests.tunableop_table import NODE_CARDS, TUNABLEOP_ENV, TUNABLEOP_TABLE, tunableop_reads
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config
from hyperloom_multibranch.contracts.toolbox import REJECTION_CODES, RequestRejected
from hyperloom_multibranch.toolbox import kernel_artifacts
from hyperloom_multibranch.toolbox.kernel_artifacts import mint_kernel_artifact
from hyperloom_multibranch.toolbox.planner import HostedPlanner
from hyperloom_multibranch.toolbox.tests.integration_inputs import IMAGE, baseline_recipe, set_workload_environment
from hyperloom_multibranch.toolbox.tests.integration_rig import Specialist, Stack, Variant, build_stack
from hyperloom_multibranch.toolbox.tests.test_integration_cpu import _fake_runner_command

TARGET = "gemm_a8w8"
ENV = "AITER_CONFIG_GEMM_A8W8"
TABLE = "M,N,K,libtype,solidx\n4096,4096,8192,hipblaslt,17\n"
OTHER_TABLE = "M,N,K,libtype,solidx\n4096,4096,8192,hipblaslt,23\n"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class ReportingPlanner:
    """The real planner; the points it returns also tell the stand-in runner to report what ``ENV`` holds."""

    def __init__(self, inner: HostedPlanner) -> None:
        self._inner = inner

    def plan(self, request: TestRequest, parent: object) -> AcceptedMeasurement:
        accepted = self._inner.plan(request, parent)  # type: ignore[arg-type]
        names = [kernel_load_env(kernel) for kernel in request.delta.add_kernels] or [ENV]
        lines = ["fake_throughput: 1000.0", f"fake_attached_env: [{', '.join(names)}]"]
        if request.accuracy:
            lines.append("fake_accuracy: 0.8")
        for point in accepted.points:
            with point.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        return accepted


def _variant(root: Path) -> Variant:
    blobs = LocalBlobStore(root / "session" / "hosted" / "blobs")
    spec = spec_context(
        root,
        image_ref="local",
        image_digest=IMAGE,
        benchmark_backend="bypass",
        blob_root=str(blobs.root),
        blobs=blobs,
    )
    canonical = default_canonical(
        root / "canonical", primary_concurrency=4, config_path=default_baseline_config(), model_path=str(root / "model")
    )
    return Variant(
        (0,),
        spec,
        canonical,
        IMAGE,
        wrap_planner=ReportingPlanner,
        python=_fake_runner_command(root),
        poll_interval_sec=0.1,
        timeout_sec=120,
    )


@pytest.fixture
async def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Stack]:
    set_workload_environment(monkeypatch, tmp_path)
    async with build_stack(tmp_path, _variant(tmp_path), baseline_recipe()) as built:
        yield built


async def put(specialist: Specialist, **overrides: str) -> dict[str, Any]:
    arguments = {"kind": "gemm", "target": TARGET, "env": ENV, "content": TABLE, **overrides}
    return dict(json.loads(await specialist.call("put_kernel_artifact", **arguments)))


def stored(stack: Stack, text: str) -> bool:
    return (stack.blobs.root / sha(text)).exists()


async def commit_with(stack: Stack, branch: str, artifact: dict[str, Any]) -> str:
    await stack.repo.create_branch(branch, stack.baseline_commit)
    recipe = apply(
        await stack.repo.read_recipe(stack.baseline_commit),
        RecipeDelta(add_kernels=(KernelArtifact.from_document(artifact),)),
    )
    return await stack.repo.commit(branch, stack.baseline_commit, recipe, "a tuned table")


def test_the_refusals_of_the_tool_are_in_the_closed_vocabulary() -> None:
    assert {kernel_artifacts.CODE_INVALID, kernel_artifacts.CODE_VARIABLE_TAKEN} <= set(REJECTION_CODES)


async def test_a_put_table_is_loaded_by_the_test_from_a_file_in_its_sandbox(stack: Stack) -> None:
    s1 = Specialist(stack, "s1")
    artifact = await put(s1)

    assert artifact == {
        "kind": "gemm",
        "target": TARGET,
        "ref": f"kernel/gemm/{TARGET}/{sha(TABLE)}.csv",
        "sha256": sha(TABLE),
        "metadata": {"load": {"via": "env_file", "env": ENV}},
    }
    assert stack.blobs.get(sha(TABLE)) == TABLE.encode(), "the bytes are in the session's blob store"

    submitted = await s1.call(
        "submit_test",
        kind="test",
        purpose="explore",
        parent_commit=stack.baseline_commit,
        delta=RecipeDelta(add_kernels=(KernelArtifact.from_document(artifact),)).to_document(),
    )
    result = await asyncio.wait_for(stack.toolbox.outcome(json.loads(submitted)["request_id"]), 120)

    assert result.validity is Validity.SUCCEEDED, result.failure_reason
    held = json.loads((stack.root / "session" / result.output_dir / "attached_env.json").read_text())[ENV]
    placed = Path(held["path"])
    assert held["sha256"] == sha(TABLE), "the file the variable points at is the table that was put"
    assert (stack.root / "session" / "sandboxes") in placed.parents and placed.parent.name == "attach", placed
    assert stack.blobs.root not in placed.parents, "a copy in the sandbox, not the session's store"


async def test_a_tunableop_table_is_placed_where_each_device_of_the_test_reads_it(stack: Stack) -> None:
    s1 = Specialist(stack, "s1")
    artifact = await put(s1, target="qwen3-8b", env=TUNABLEOP_ENV, content=TUNABLEOP_TABLE.decode())
    offline = ConfigDelta(envs={"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "0"})
    delta = RecipeDelta(config=offline, add_kernels=(KernelArtifact.from_document(artifact),))

    submitted = await s1.call(
        "submit_test", kind="test", purpose="explore", parent_commit=stack.baseline_commit, delta=delta.to_document()
    )
    result = await asyncio.wait_for(stack.toolbox.outcome(json.loads(submitted)["request_id"]), 120)

    assert result.validity is Validity.SUCCEEDED, result.failure_reason
    held = json.loads((stack.root / "session" / result.output_dir / "attached_env.json").read_text())[TUNABLEOP_ENV]
    table = hashlib.sha256(TUNABLEOP_TABLE).hexdigest()
    assert held["sha256"] == table, "the variable names the table that was put"
    for ordinal in range(NODE_CARDS):
        read = tunableop_reads(held["path"], ordinal).name
        assert held["directory"].get(read) == table, f"device {ordinal} reads {read}, which the sandbox lacks"


async def test_branches_that_put_the_same_table_get_one_artifact_and_merge_without_a_conflict(
    stack: Stack, tmp_path: Path
) -> None:
    first = await put(Specialist(stack, "s1"))
    second = await put(Specialist(stack, "s2"))
    assert first == second
    heads = {"s1": await commit_with(stack, "s1", first), "s2": await commit_with(stack, "s2", second)}

    merge = await stack.repo.merge(heads["s2"], heads["s1"])

    assert merge.clean and merge.delta.add_kernels == (), "the table is on the head already; nothing to carry"
    await stack.repo.checkout(heads["s1"], tmp_path / "tree")
    assert (tmp_path / "tree" / first["ref"]).read_bytes() == TABLE.encode(), "checkout reads the same blob store"


async def test_branches_that_tune_one_target_differently_conflict_on_its_slot(stack: Stack) -> None:
    first = await put(Specialist(stack, "s1"))
    second = await put(Specialist(stack, "s2"), content=OTHER_TABLE)
    assert first["ref"] != second["ref"] and first["target"] == second["target"]
    heads = {"s1": await commit_with(stack, "s1", first), "s2": await commit_with(stack, "s2", second)}

    with pytest.raises(RecipeError, match=f"both sides replaced the gemm kernel for {TARGET}"):
        await stack.repo.merge(heads["s2"], heads["s1"])


async def test_a_variable_loads_one_table_so_a_second_target_for_it_is_refused(stack: Stack) -> None:
    s1, s2 = Specialist(stack, "s1"), Specialist(stack, "s2")
    await put(s1)

    message = await s2.refused(
        "put_kernel_artifact", kind="gemm", target="gemm_a8w8_small", env=ENV, content=OTHER_TABLE
    )

    assert "kernel_variable_taken" in message and TARGET in message, message
    assert not stored(stack, OTHER_TABLE), "a refused table is not stored"
    replaced = await put(s2, content=OTHER_TABLE)
    assert replaced["target"] == TARGET and stored(stack, OTHER_TABLE), "the bound target takes a new table"
    assert (await put(s1, target="gemm_a4w4", env="AITER_CONFIG_GEMM_A4W4"))["metadata"]["load"]["env"].endswith("A4W4")


async def test_the_refusal_says_the_variable_is_bound_to_a_target_any_branch_may_put_under(stack: Stack) -> None:
    first = await put(Specialist(stack, "b02"))
    other = Specialist(stack, "b07")

    message = await other.refused(
        "put_kernel_artifact", kind="gemm", target="gemm_a8w8_small", env=ENV, content=OTHER_TABLE
    )

    assert f"{ENV} is bound to target '{TARGET}' (gemm table) in this session, not to a branch" in message, message
    assert f"with target='{TARGET}'" in message and "changes no other branch's artifact" in message, message
    assert "Only a different target name" in message and "already loads" not in message, message
    retry = await put(other, target=TARGET, content=OTHER_TABLE)
    assert retry["target"] == TARGET and retry["ref"] != first["ref"], "the second branch's table has its own reference"
    assert stored(stack, TABLE) and stored(stack, OTHER_TABLE), "the first branch's table is untouched"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"env": "not a name"}, "kernel_artifact_invalid"),
        ({"target": "../x"}, "kernel_artifact_invalid"),
        ({"target": "a/b"}, "kernel_artifact_invalid"),
        ({"kind": "fusion"}, "kind"),
        ({"content": ""}, "content"),
    ],
    ids=["env-name", "target-escapes", "target-has-a-slash", "kind-without-a-load", "empty-table"],
)
async def test_a_table_the_measurement_cannot_load_is_refused_and_not_stored(
    stack: Stack, overrides: dict[str, str], message: str
) -> None:
    before = sorted(stack.blobs.root.iterdir())

    reply = await Specialist(stack, "s1").refused(
        "put_kernel_artifact", **{"kind": "gemm", "target": TARGET, "env": ENV, "content": TABLE, **overrides}
    )

    assert message in reply, reply
    assert sorted(stack.blobs.root.iterdir()) == before


def test_a_table_over_the_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernel_artifacts, "MAX_ATTACHMENT_BYTES", 8)

    with pytest.raises(RequestRejected, match="the cap is 8") as refused:
        mint_kernel_artifact(KernelKind.GEMM, TARGET, ENV, TABLE.encode())

    assert refused.value.code == kernel_artifacts.CODE_INVALID
