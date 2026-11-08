# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A base launch record as another session freezes one: a vLLM AgentX base with a golden layer."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from inference_suite import INFERENCEX_CLIENT_COMMIT
from inference_suite.contracts import (
    BaseLaunch,
    ClientLaunch,
    GoldenAcceptance,
    GoldenCurve,
    ImageFacts,
    LaunchModel,
    LaunchPoint,
    LaunchQuery,
    LaunchSource,
    RunnerOwned,
    SourceKind,
    Topology,
    VersionBasis,
    Workload,
)

MODEL_ID = "amd/Qwen3-8B"


def base_record(**changes: object) -> BaseLaunch:
    """A vLLM AgentX base at TP2 and conc 8, its golden layer last; ``changes`` replace its fields."""

    golden = GoldenAcceptance(
        value=3.0,
        method="match-expected",
        token_mode="real-draft-token",
        spec_method="MTP",
        draft_length=3,
        curve=GoldenCurve(repo="https://github.com/org/curves", commit="e" * 40, path="al.json", sha256="f" * 64),
        args=("--speculative-config", '{"method": "mtp", "num_speculative_tokens": 3}'),
        env={"VLLM_SIMULATE_ACC": "3"},
    )
    base = BaseLaunch(
        query=LaunchQuery(
            model_id=MODEL_ID, framework="vllm", board="mi355x", workload=Workload.AGENTX, concurrencies=(8,), tp=2
        ),
        source=LaunchSource(kind=SourceKind.USER_FILE, path="launch.sh", files={"launch.sh": "c" * 64}),
        point=LaunchPoint(conc=8, exact=True),
        image=ImageFacts(run=None, recipe=None, framework_version=None, version_basis=VersionBasis.NOT_APPLICABLE),
        framework="vllm",
        entrypoint=("vllm", "serve"),
        model=LaunchModel(MODEL_ID),
        topology=Topology(tp=2),
        args=("--tensor-parallel-size", "2", "--max-num-seqs", "128", "--chat-template", "a b"),
        env={"VLLM_ROCM_USE_AITER": "1"},
        client=ClientLaunch(tool="aiperf", commit=INFERENCEX_CLIENT_COMMIT),
        resolved_unix=1_700_000_000,
        golden_al=golden,
        runner_owned=RunnerOwned(args=("--host", "--port"), env=("HIP_VISIBLE_DEVICES",)),
    )
    return dataclasses.replace(base, **changes) if changes else base


def write_record(path: Path, base: BaseLaunch | None = None, **edits: object) -> Path:
    """``base`` (:func:`base_record` by default) written to ``path`` as a record, its document's ``edits`` applied."""

    document = (base or base_record()).to_document()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**document, **edits}), encoding="utf-8")
    return path
