# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The round of ``test_integration_cpu.py`` on a GPU machine, with the real runner, the real framework and a real model.

Faked: the specialists' decisions only (a script instead of a model). The runner in every sandbox is the shipped one,
SGLang serves the model, the patch is applied to the SGLang installed here and the accuracy rounds are real evals, so
the numbers are measured and the verdicts are whatever the rule gate makes of them. Because a comment-only patch and a
changed request cap rarely gain more than the bar, this variant cannot promise a commit; it checks that whatever the
rule gate decided is what the repository and the blackboard say, and that the round's mechanics (fresh sandboxes, verified
bundles, the retest beside a full branch, the withdrawals) hold with real measurements.

Needs SGLang, a model directory and an InferenceX git checkout: ``HL_W15_MODEL``, ``HL_W15_INFERENCEX`` and
``HL_W15_OUT`` (a directory the test makes its own subdirectory in), and optionally ``HL_W15_CARDS`` (default ``6,7``).
Anywhere else the module is skipped.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from hyperloom.measure.compile import COMPONENT_LAYOUTS, locate_installed_package
from hyperloom.measure.points import default_canonical
from hyperloom.measure.tests.hosted_fixtures import spec_context
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config
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

MODEL = os.environ.get("HL_W15_MODEL", "")
INFERENCEX = os.environ.get("HL_W15_INFERENCEX", "")
OUT = os.environ.get("HL_W15_OUT", "")
CARDS = tuple(int(card) for card in os.environ.get("HL_W15_CARDS", "6,7").split(","))
USES_REAL_GPUS = True  # keep the host's amd-smi and rocm-smi: see hyperloom.measure.tests.gpu_free_host
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None or not Path(MODEL).is_dir() or not Path(INFERENCEX, ".git").exists(),
    reason="needs SGLang, HL_W15_MODEL and HL_W15_INFERENCEX (a git checkout)",
)
EVAL_LIMIT = "60"
SERVER_READY_SEC = "900"


def _probe(label: str, **fields: object) -> None:
    print(f"PROBE {label} " + json.dumps(fields, default=str), flush=True)


async def test_a_round_of_two_branches_with_real_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(OUT) / f"w15-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    set_workload_environment(monkeypatch, root / "user-data", MAGPIE_EVAL_LIMIT=EVAL_LIMIT)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=INFERENCEX, capture_output=True, text=True, check=True
    ).stdout.strip()
    spec = spec_context(
        root,
        image_ref="local",
        image_digest=IMAGE,
        gpu_type="MI355X",
        inferencex_commit=commit,
        benchmark_backend="bypass",
    )
    canonical = default_canonical(
        root / "canonical", primary_concurrency=4, config_path=default_baseline_config(), model_path=MODEL
    )
    variant = Variant(
        CARDS,
        spec,
        canonical,
        IMAGE,
        runner_envs={"INFERENCEX_REPO": INFERENCEX, "INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC": SERVER_READY_SEC},
        timeout_sec=2400,
        poll_interval_sec=2.0,
    )
    original = (locate_installed_package(COMPONENT_LAYOUTS["sglang"]).path / PATCHED_FILE).read_text()
    patch = comment_patch_delta(root / "patch-repo", spec.blobs, original)  # type: ignore[arg-type]
    async with build_stack(root, variant, baseline_recipe()) as stack:
        out = await run_round(stack, probes(patch))
        _probe(
            "round",
            baseline_output_throughput=out.baseline,
            verdicts={
                key: (j.verdict.value, round(j.gain_pct, 2), j.accuracy.value) for key, j in out.judgements.items()
            },
            actions={key: action.value for key, action in out.actions.items()},
            max_running_s1=out.max_running_s1,
        )
        assert out.max_running_s1 == 2, "the retest must run beside the explore test of the same branch"
