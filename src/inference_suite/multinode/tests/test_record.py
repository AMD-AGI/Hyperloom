# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import json
from dataclasses import replace

import pytest

from inference_suite.frameworks import Framework
from inference_suite.multinode.record import (
    SCHEMA_V1,
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


def _launch() -> PdLaunch:
    return PdLaunch(
        source=PdSource(PdSourceKind.PD_TABLE, path="models.yaml", entry="M", files={"models.yaml": "ab"}),
        framework=Framework.SGLANG,
        model="/models/M",
        served_model_name="org/M",
        context_length=2048,
        transfer_backend="mori",
        roles={
            Role.PREFILL: RoleLaunch(("p0",), 1, 8, args=("--a", "1"), env={"K": "v"}, stripped=("--port",)),
            Role.DECODE: RoleLaunch(("d0", "d1"), 1, 16, ep=16, dp_attention=True),
        },
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
    assert moved.prefill.hosts == ("x0",)
    assert moved.frontend.host == "z0"


def test_digest_covers_the_launch():
    launch = _launch()
    assert replace(launch, transfer_backend="nixl").digest() != launch.digest()
    assert replace(launch, model="/models/N").digest() != launch.digest()
    assert replace(launch, context_length=None).digest() != launch.digest()
    assert launch.with_roles({Role.PREFILL: replace(launch.prefill, args=("--a", "2"))}).digest() != launch.digest()
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


def _aggregated() -> PdLaunch:
    return PdLaunch(
        source=PdSource(PdSourceKind.BASE_LAUNCH, entry="M"),
        framework=Framework.SGLANG,
        model="/models/M",
        served_model_name="org/M",
        context_length=None,
        transfer_backend="",
        roles={Role.WORKER: RoleLaunch(("w0", "w1"), 1, 16, args=("--a",))},
        frontend=Frontend(FrontendType.INFERA),
        mode=PdMode.AGGREGATED,
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


def test_previous_schema_reads_as_disaggregated(tmp_path):
    launch = _launch()
    document = launch.to_document()
    del document["mode"]
    canonical = {key: document[key] for key in ("model", "served_model_name", "context_length", "transfer_backend")}
    canonical["roles"] = {role.value: launch.role(role).launch_document() for role in launch.served_roles}
    canonical["frontend"] = launch.frontend.launch_document()
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    document.update(schema=SCHEMA_V1, digest=f"sha256:{hashlib.sha256(raw).hexdigest()}")
    path = tmp_path / "pd_launch.json"
    path.write_text(json.dumps(document))
    read = read_record(path)
    assert read == launch and read.mode is PdMode.DISAGGREGATED
    assert document["digest"] != launch.digest(), "the digest covers the mode"
