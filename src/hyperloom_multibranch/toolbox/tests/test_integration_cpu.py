# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One multi-branch round across the modules, on CPU (design 6.1 and 7.3, the toolbox side of MB1).

Real: the toolbox (``build_local_toolbox``: queue, acceptance, attempts, the local-sandbox executor, signed bundles
verified against each sandbox's key), the planner (``HostedPlanner``: environment compile with a real patch, point
generation, surface check), the notebook (``LocalNotebook``), the experiment repository (``GitExperimentRepo``), the
rule gate (``RuleGate`` and the InferenceX comparison) and the MCP service with one token per specialist.

Faked, and nothing else: the specialists' decisions (a script instead of a model) and the runner inside each sandbox
(``hyperloom.measure.tests.fake_runner``, which reports the numbers the planner wrapper below puts into the point), and
the framework package the patch is applied to (a small tree, because no SGLang is installed on CI). MB1 has the real
runner and framework; the GPU variant (``test_integration_gpu.py``) runs them.
"""

from __future__ import annotations

import stat
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import TestRequest
from hyperloom.contracts.recipe import RecipeDelta, apply
from hyperloom.measure.compile import BlobStore, LocalBlobStore, PackageBase
from hyperloom.measure.points import AcceptedMeasurement, default_canonical
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config
from hyperloom_multibranch.contracts.judgement import Verdict
from hyperloom_multibranch.contracts.notebook.vocab import INTEGRATION_BRANCH, Conclusion
from hyperloom_multibranch.rulegate.decision import Action
from hyperloom_multibranch.toolbox.planner import HostedPlanner
from hyperloom_multibranch.toolbox.tests.integration_inputs import (
    IMAGE,
    PATCHED_FILE,
    baseline_recipe,
    comment_patch_delta,
    probes,
    set_workload_environment,
)
from hyperloom_multibranch.toolbox.tests.integration_rig import Variant, build_stack
from hyperloom_multibranch.toolbox.tests.integration_round import run_round

CARDS = (0, 1)


@dataclass(frozen=True)
class Reports:
    """What the fake runner reports for one test: output throughput, an accuracy score if asked, seconds it takes."""

    throughput: float
    accuracy: float | None = None
    seconds: float = 0.0


# Per (branch, purpose), in the order the tests are submitted. Baselines 1000 and 1004: the bar is the floor.
# s1's patch gains 6% and holds on the retest; s2's change gains 4% and loses it on the retest.
SCRIPT = {
    (INTEGRATION_BRANCH, "baseline"): [Reports(1000.0, 0.80), Reports(1004.0, 0.81)],
    ("s1", "explore"): [Reports(1060.0, 0.80, seconds=3.0), Reports(1001.0, 0.80, seconds=6.0)],
    ("s1", "confirm"): [Reports(1055.0, 0.80)],
    ("s2", "explore"): [Reports(1040.0, 0.80, seconds=3.0), Reports(1000.0, 0.80)],
    ("s2", "confirm"): [Reports(995.0, 0.80)],
    ("s3", "explore"): [Reports(1000.0, 0.80), Reports(1000.0, 0.80)],
}


class FakeRunnerPlanner:
    """The real planner; the points it returns also carry what the fake runner reports for the test."""

    def __init__(self, inner: HostedPlanner) -> None:
        self._inner = inner
        self.planned: list[tuple[TestRequest, AcceptedMeasurement]] = []
        self._script = defaultdict(list, {key: list(value) for key, value in SCRIPT.items()})

    @property
    def blobs(self) -> BlobStore:
        return self._inner.blobs

    def plan(self, request: TestRequest, parent: object) -> AcceptedMeasurement:
        accepted = self._inner.plan(request, parent)  # type: ignore[arg-type]
        self.planned.append((request, accepted))
        reports = self._script[(request.branch_id, request.purpose.value)].pop(0)
        assert (reports.accuracy is not None) == request.accuracy, "the script and the request disagree on accuracy"
        lines = [f"fake_throughput: {reports.throughput}", f"fake_delay: {reports.seconds}"]
        if reports.accuracy is not None:
            lines.append(f"fake_accuracy: {reports.accuracy}")
        for point in accepted.points:
            with point.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        return accepted


def _fake_runner_command(directory: Path) -> str:
    """The "interpreter" the runner template starts: it ignores the runner's arguments and starts the fake runner."""
    script = directory / "fake-python"
    script.write_text(f"#!/bin/sh\nexec {sys.executable} -m hyperloom.measure.tests.fake_runner\n", encoding="utf-8")
    script.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return str(script)


def _variant(root: Path, planners: list[FakeRunnerPlanner]) -> tuple[Variant, RecipeDelta]:
    blobs = LocalBlobStore(root / "session" / "hosted" / "blobs")
    framework = root / "framework" / "sglang"
    framework.mkdir(parents=True)
    original = "print('serve')\n"
    (framework / PATCHED_FILE).write_text(original, encoding="utf-8")
    package = PackageBase("sglang", framework, "sglang", "0.0")
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

    def wrap(inner: HostedPlanner) -> FakeRunnerPlanner:
        planners.append(FakeRunnerPlanner(inner))
        return planners[-1]

    variant = Variant(
        CARDS,
        spec,
        canonical,
        IMAGE,
        wrap_planner=wrap,
        python=_fake_runner_command(root),
        locator=lambda layout: package,
        poll_interval_sec=0.1,
        timeout_sec=120,
    )
    return variant, comment_patch_delta(root / "patch-repo", blobs, original)


async def test_a_round_of_two_branches_through_every_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    set_workload_environment(monkeypatch, tmp_path)
    planners: list[FakeRunnerPlanner] = []
    variant, patch = _variant(tmp_path, planners)
    async with build_stack(tmp_path, variant, baseline_recipe()) as stack:
        out = await run_round(stack, probes(patch))
        t1, t2 = out.first["s1"], out.first["s2"]
        # s1: +5.8% on the first measurement waits for the retest; both clear the bar, the accuracy gate passes: commit
        assert out.judgements["s1-first"].verdict is Verdict.NEEDS_CONFIRM
        assert out.judgements[t1].verdict is Verdict.IMPROVED and out.actions[t1] is Action.COMMIT
        head = await stack.repo.head("s1")
        assert head != stack.baseline_commit
        assert await stack.repo.read_recipe(head) == apply(baseline_recipe(), patch)
        assert (await stack.gate_notebook.record(t1)).conclusion is Conclusion.COMMITTED
        # s2: +3.8% that the retest does not confirm: recorded, nothing committed
        assert out.judgements[t2].verdict is Verdict.FLAT and out.actions[t2] is Action.RECORD_ONLY
        assert await stack.repo.head("s2") == stack.baseline_commit
        assert (await stack.gate_notebook.record(t2)).conclusion is Conclusion.RECORDED_ONLY
        # the explore test that ran beside the retest is flat and only recorded; the integration branch never moved
        assert out.actions[out.explore2] is Action.RECORD_ONLY
        assert await stack.repo.head(INTEGRATION_BRANCH) == stack.baseline_commit
        # the confirm-retest ran beside s1's explore test although K is 1, and carried its accuracy value
        assert out.max_running_s1 == 2
        assert (await stack.gate_notebook.record(out.retest["s1"])).result.accuracy[0].score == 0.80  # type: ignore[union-attr]
        # the patch was compiled for the tests that carry it (the first test of s1 and its retest) and for no other
        (planner,) = planners
        carrying = [(r.branch_id, r.purpose.value) for r, accepted in planner.planned if accepted.environment.roots]
        assert carrying == [("s1", "explore"), ("s1", "confirm")]
        (root,) = next(accepted for r, accepted in planner.planned if r.branch_id == "s1").environment.roots
        assert root.import_name == "sglang" and [(e.path, e.op) for e in root.entries] == [(PATCHED_FILE, "upsert")]
