# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from dataclasses import replace

import pytest

from inference_suite.frameworks import Framework
from inference_suite.multinode.record import (
    Frontend,
    FrontendType,
    PdLaunch,
    PdRecordError,
    PdSource,
    PdSourceKind,
    RoleLaunch,
    read_record,
    write_record,
)


def _launch() -> PdLaunch:
    return PdLaunch(
        source=PdSource(PdSourceKind.PD_TABLE, path="models.yaml", entry="M", files={"models.yaml": "ab"}),
        framework=Framework.SGLANG,
        model="/models/M",
        served_model_name="org/M",
        context_length=2048,
        transfer_backend="mori",
        prefill=RoleLaunch(("p0",), 1, 8, args=("--a", "1"), env={"K": "v"}, stripped=("--port",)),
        decode=RoleLaunch(("d0", "d1"), 1, 16, ep=16, dp_attention=True),
        frontend=Frontend(FrontendType.SGLANG_ROUTER, "r0", ("--policy", "random")),
    )


def test_record_round_trips(tmp_path):
    launch = _launch()
    path = tmp_path / "session" / "pd_launch.json"
    digest = write_record(launch, path)
    assert read_record(path) == launch
    assert digest == launch.digest()


def test_digest_ignores_hosts():
    launch = _launch()
    moved = launch.with_hosts(["x0"], ["y0", "y1"], "z0")
    assert moved.digest() == launch.digest()
    assert moved.prefill.hosts == ("x0",)
    assert moved.frontend.host == "z0"


def test_digest_covers_the_launch():
    launch = _launch()
    assert replace(launch, transfer_backend="nixl").digest() != launch.digest()
    assert replace(launch, model="/models/N").digest() != launch.digest()
    assert replace(launch, context_length=None).digest() != launch.digest()
    assert replace(launch, prefill=replace(launch.prefill, args=("--a", "2"))).digest() != launch.digest()
    assert replace(launch, frontend=replace(launch.frontend, args=())).digest() != launch.digest()


def test_edited_record_is_refused(tmp_path):
    path = tmp_path / "pd_launch.json"
    write_record(_launch(), path)
    document = json.loads(path.read_text())
    document["roles"]["decode"]["args"] = ["--b"]
    path.write_text(json.dumps(document))
    with pytest.raises(PdRecordError, match="digest"):
        read_record(path)


def test_other_schema_is_refused(tmp_path):
    path = tmp_path / "pd_launch.json"
    path.write_text(json.dumps({**_launch().to_document(), "schema": "x/2"}))
    with pytest.raises(PdRecordError, match="schema"):
        read_record(path)
