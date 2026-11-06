# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the operator's launch file: the shell form fixes every flag, the JSON form is a checked base."""

from __future__ import annotations

import hashlib
import json

import pytest

from inference_suite.contracts import (
    BaseLaunch,
    BaseLaunchError,
    ClientLaunch,
    ImageFacts,
    LaunchModel,
    LaunchPoint,
    LaunchQuery,
    LaunchSource,
    SourceKind,
    Topology,
    VersionBasis,
    Workload,
)
from inference_suite.recipe.shell_launch import ShellLaunch
from inference_suite.recipe.user_file import UserFileError, parse_user_file


def _base(workload: Workload = Workload.AGENTX) -> BaseLaunch:
    return BaseLaunch(
        query=LaunchQuery(
            model_id="moonshotai/Kimi-K3",
            framework="sglang",
            board="mi355x",
            workload=workload,
            concurrencies=(32,),
            isl=8192 if workload is Workload.FIXED else None,
            osl=1024 if workload is Workload.FIXED else None,
        ),
        source=LaunchSource(kind=SourceKind.USER_FILE, path="kimi.sh", files={"kimi.sh": "c" * 64}),
        point=LaunchPoint(conc=32, exact=True),
        image=ImageFacts(run=None, recipe=None, framework_version=None, version_basis=VersionBasis.NOT_APPLICABLE),
        framework="sglang",
        entrypoint=("sglang", "serve"),
        model=LaunchModel("moonshotai/Kimi-K3", "/models/kimi"),
        topology=Topology(tp=8),
        args=("--tp", "8", "--kv-cache-dtype", "fp8_e4m3"),
        env={"SGLANG_USE_AITER": "1"},
        client=ClientLaunch(tool="aiperf", commit="d" * 40),
        resolved_unix=1_700_000_000,
        stripped=("--port",),
    )


def test_shell_form_lifts_the_launch_and_records_the_sha256():
    content = b"export SGLANG_USE_AITER=1\nsglang serve --model-path /m --tp 8 --port 30000\n"

    parsed = parse_user_file("recipe.sh", content, framework="sglang")

    assert parsed.sha256 == hashlib.sha256(content).hexdigest()
    assert isinstance(parsed.launch, ShellLaunch)
    assert parsed.launch.args == ("--tp", "8")
    assert parsed.launch.stripped == ("--port",)
    assert parsed.launch.env == {"SGLANG_USE_AITER": "1"}


def test_shell_flag_valued_by_a_variable_is_an_error():
    content = b"vllm serve $MODEL --port $PORT --tensor-parallel-size=$TP --max-model-len $LEN --block-size 64\n"

    with pytest.raises(UserFileError, match="--tensor-parallel-size, --max-model-len"):
        parse_user_file("recipe.sh", content, framework="vllm")


def test_launch_controls_comment_is_an_error():
    content = b'#!/usr/bin/env bash\n# hyperloom-launch-controls: {"args_mode": "replace"}\nsglang serve --tp 8\n'

    with pytest.raises(UserFileError, match="hyperloom-launch-controls"):
        parse_user_file("current_setting.sh", content, framework="sglang")


@pytest.mark.parametrize("workload", ["fixed", "agentx"])
def test_json_base_launch_reproduces_with_the_same_digest(workload):
    base = _base(Workload(workload))
    content = json.dumps(base.to_document(), indent=2).encode()

    parsed = parse_user_file("base_launch.json", content, framework=base.framework)

    assert parsed.launch == base
    assert parsed.launch.digest() == base.digest()
    assert parsed.sha256 == hashlib.sha256(content).hexdigest()


def test_json_base_launch_with_a_tampered_digest_is_refused():
    document = _base().to_document()
    document["args"] = [*document["args"], "--disable-radix-cache"]

    with pytest.raises(BaseLaunchError):
        parse_user_file("base_launch.json", json.dumps(document).encode(), framework="sglang")


def test_json_base_launch_for_another_framework_is_refused():
    content = json.dumps(_base().to_document()).encode()

    with pytest.raises(UserFileError, match="sglang launch"):
        parse_user_file("base_launch.json", content, framework="vllm")


@pytest.mark.parametrize(
    "content",
    [
        b'{"schema": "inference-suite-config/1"}',
        b"{not json",
        b"\xff\xfe binary",
        b"export SGLANG_USE_AITER=1\nvllm serve /m --block-size 64\n",
    ],
)
def test_other_documents_are_refused(content):
    with pytest.raises(UserFileError):
        parse_user_file("launch.json", content, framework="sglang")
