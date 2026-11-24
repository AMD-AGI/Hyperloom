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
    PdMode,
    PdRecordError,
    PdSource,
    PdSourceKind,
    Role,
    RoleLaunch,
    read_record,
    write_record,
)
from inference_suite.multinode.tests.builders import pd_launch


def _launch() -> PdLaunch:
    prefill = RoleLaunch(("p0",), 1, 8, args=("--a", "1"), env={"K": "v"}, stripped=("--port",))
    return pd_launch(
        {Role.PREFILL: prefill, Role.DECODE: RoleLaunch(("d0", "d1"), 1, 16, ep=16, dp_attention=True)},
        source=PdSource(PdSourceKind.PD_TABLE, path="models.yaml", entry="M", files={"models.yaml": "ab"}),
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
    moved = launch.with_hosts({Role.PREFILL: ["x0"], Role.DECODE: ["y0", "y1"]}, "z0")
    assert moved.digest() == launch.digest()
    assert moved.role(Role.PREFILL).hosts == ("x0",)
    assert moved.frontend.host == "z0"


def test_digest_covers_the_launch():
    launch = _launch()
    assert replace(launch, transfer_backend="nixl").digest() != launch.digest()
    assert replace(launch, model="/models/N").digest() != launch.digest()
    assert replace(launch, context_length=None).digest() != launch.digest()
    assert (
        launch.with_roles({Role.PREFILL: replace(launch.role(Role.PREFILL), args=("--a", "2"))}).digest()
        != launch.digest()
    )
    assert replace(launch, frontend=replace(launch.frontend, args=())).digest() != launch.digest()


def test_digest_covers_the_framework(tmp_path):
    launch = _launch()
    assert replace(launch, framework=Framework.ATOM).digest() != launch.digest()
    path = tmp_path / "pd_launch.json"
    write_record(launch, path)
    document = json.loads(path.read_text())
    document["framework"] = Framework.ATOM.value
    path.write_text(json.dumps(document))
    with pytest.raises(PdRecordError, match="digest"):
        read_record(path)


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


def _aggregated() -> PdLaunch:
    return pd_launch(
        {Role.WORKER: RoleLaunch(("w0", "w1"), 1, 16, args=("--a",))},
        source=PdSource(PdSourceKind.BASE_LAUNCH, entry="M"),
        frontend=Frontend(FrontendType.INFERA),
    )


def test_aggregated_record_round_trips(tmp_path):
    launch = _aggregated()
    path = tmp_path / "pd_launch.json"
    write_record(launch, path)
    document = json.loads(path.read_text())
    assert document["mode"] == "aggregated" and set(document["roles"]) == {"worker"}
    assert read_record(path) == launch
    assert launch.with_hosts({Role.WORKER: ["x0", "x1"]}, "").digest() == launch.digest()


def test_a_launch_runs_the_roles_of_its_mode():
    with pytest.raises(PdRecordError, match="aggregated launch runs"):
        replace(_launch(), mode=PdMode.AGGREGATED)
