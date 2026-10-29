# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The runner's main flow with the measurement backend, the tool checkouts and the GPU queries stubbed."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest
import yaml

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure import signing
from hyperloom.measure.bundle import BundleExpectation, verify_bundle
from hyperloom.measure.compile import LocalBlobStore, PackageBase, compile_environment
from hyperloom.measure.hosted_spec import dir_manifest_sha256
from hyperloom.measure.manifest import ResultManifest
from hyperloom.measure.pins import IMAGE_DIGEST_VAR
from hyperloom.measure.points import default_canonical, generate_points
from hyperloom.measure.runner import run as runner
from hyperloom.measure.returncodes import SERVER_BOOT_TIMEOUT_RC
from hyperloom.measure.runner.hardware import EccCounts, GpuFacts
from hyperloom.measure.long_context.plan import LongContextPlan
from hyperloom.measure.short_prompts.plan import ShortPromptPlan
from hyperloom.measure.socket import RoundOutcome, TestCall, TestOutcome
from hyperloom.measure.spec import MeasurementSpec
from hyperloom.measure.runner.tests import aiperf_source
from hyperloom.measure.tests._recipes import IMAGE, recipe_with
from hyperloom.measure.tests.bundle_factory import REQUEST_ID, spec_document
from hyperloom.contracts.measurement import SandboxFacts
from hyperloom.orchestrator.actions.executors._workload_envs import default_baseline_config

SHEET_IMAGE = spec_document()["image"]["digest"]
POINT = {"benchmark": {"model": "/m", "framework": "sglang", "envs": {"TP": 1, "CONC": 4, "RUN_EVAL": "false"}}}


def _sheet(**changes) -> dict:
    document = spec_document()
    text = yaml.safe_dump(POINT)
    document["config"] = [{"yaml": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}]
    document["attachments"] = []
    document["environment"] = {"roots": [], "runtime": None, "blob_root": "/b"}
    document["model_hashes"] = {}
    document["unset_envs"] = ["STALE_PIN"]
    document["procedure"].update(
        {"benchmark_backend": "bypass", "rounds": {"warmup": 1, "measure": 1}, "deadline_unix": 0}
    )
    document.update(changes)
    return document


class _Backend:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls: list[TestCall] = []

    def __call__(self):
        return self

    def run_test(self, call: TestCall) -> TestOutcome:
        self.calls.append(call)
        return self.behaviour(call)


def _ok(call: TestCall) -> TestOutcome:
    (call.output_dir / "server.log").write_text("log")
    rounds = (RoundOutcome(0, "measure", call.output_dir, 0),)
    return TestOutcome(0, "[]", "", rounds, SandboxFacts(sandbox_id="x"), request_id=call.task_id)


@pytest.fixture
def key():
    return signing.generate_private_key()


@pytest.fixture
def env(tmp_path, key, monkeypatch):
    key_file = tmp_path / "key.pem"
    key_file.write_bytes(signing.private_key_pem(key))
    environ = {
        "HYPERLOOM_SIGNING_KEY_FILE": str(key_file),
        "NODE_NAME": "node-1",
        "E2B_SANDBOX_ID": "sbx-1",
        "ROCR_VISIBLE_DEVICES": "0",
        "MASTER_ADDR": "x",
        "PET_RANK": "0",
        "NCCL_IB_SL": "9",
        IMAGE_DIGEST_VAR: SHEET_IMAGE,
    }
    monkeypatch.setattr(runner, "ensure_inferencex", lambda commit, cache: tmp_path / "ix")
    # a machine that answers every question: the sheet's card, no errors, nothing resident
    monkeypatch.setattr(
        runner.hardware, "read_gpu_facts", lambda visible: [GpuFacts(0, "AMD Instinct MI355X", 1400.0, "SPX/NPS1")]
    )
    monkeypatch.setattr(runner.hardware, "ecc_counts", lambda visible: {0: EccCounts(0, 0)})
    monkeypatch.setattr(runner.hardware, "busy_gpus", lambda visible: [])
    return environ


def _go(tmp_path, env, sheet: dict | bytes | None, backend=None, **kwargs) -> int:
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir(exist_ok=True)
    if sheet is not None:
        raw = sheet if isinstance(sheet, bytes) else json.dumps(sheet).encode()
        (in_dir / "spec.json").write_bytes(raw)
        (in_dir / "READY").write_text("")
    return runner.run(in_dir, out_dir, scratch=tmp_path / "scratch", environ=env, spec_wait_sec=0.05, **kwargs)


def _status(tmp_path) -> dict:
    return json.loads((tmp_path / "out" / "status.json").read_text())


def test_no_sheet_ends_with_no_spec_and_exit_zero(tmp_path, env):
    assert _go(tmp_path, env, None) == 0
    assert _status(tmp_path)["error_class"] == "no_spec"


def test_a_sheet_that_does_not_parse_is_spec_rejected(tmp_path, env):
    assert _go(tmp_path, env, b"{not json") == 3
    assert _status(tmp_path)["error_class"] == "spec_rejected"
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_a_missing_signing_key_stops_the_runner_at_start(tmp_path, env):
    del env["HYPERLOOM_SIGNING_KEY_FILE"]

    assert _go(tmp_path, env, _sheet()) == 2
    assert _status(tmp_path)["error_class"] == "runner_start_failed"


def test_a_successful_test_leaves_a_signed_manifest_and_status_last(tmp_path, env, key, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, _sheet()) == 0

    out = tmp_path / "out"
    raw = (out / "manifest.json").read_bytes()
    staging = tmp_path / "staging"  # what the reader assembles: the result files with the manifest and signature
    shutil.copytree(out / "files", staging)
    shutil.copy(out / "manifest.json", staging)
    shutil.copy(out / "manifest.sig", staging)
    verify_bundle(
        staging,
        BundleExpectation(
            public_key_pem=signing.public_key_pem(key),
            request_id=REQUEST_ID,
            spec_fingerprint=ResultManifest.from_bytes(raw).spec_fingerprint,
            spec_sha256=None,
            sandbox_id="sbx-1",
        ),
    )
    assert signing.verify(raw, (out / "manifest.sig").read_text(), signing.public_key_pem(key))
    manifest = ResultManifest.from_bytes(raw)
    assert manifest.request_id == REQUEST_ID and manifest.outcome.returncode == 0
    assert sorted(f.path for f in manifest.files) == ["farm_inventory.json", "rounds.json", "server.log"]
    assert json.loads((out / "files" / "rounds.json").read_text()) == [
        {"point": 0, "phase": "measure", "dir": ".", "returncode": 0}
    ]
    assert manifest.runner.sandbox_id == "sbx-1" and manifest.runner.node_name == "node-1"
    assert manifest.platform_env_changed == ("NCCL_IB_SL",)
    assert _status(tmp_path) == {"state": "succeeded", "error_class": None, "message": ""}


def test_the_call_is_built_from_the_sheet_and_the_environment_is_the_sandboxes(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    _go(tmp_path, env, _sheet())

    (call,) = backend.calls
    assert call.task_id == REQUEST_ID and call.prep == "grid" and call.kind == "test" and not call.accuracy
    assert call.rounds.warmup_rounds == 1
    assert call.unset_envs == ("STALE_PIN",) and dict(call.env_overrides) == {"AITER_LOG_LEVEL": "WARNING"}
    assert call.session_facts.inferencex_path == str(tmp_path / "ix")
    rewritten = yaml.safe_load(call.points[0].read_text())["benchmark"]
    assert rewritten["inferencex_path"] == str(tmp_path / "ix")
    assert rewritten["envs"]["ROCR_VISIBLE_DEVICES"] == "0"
    assert env["NCCL_IB_SL"] == "3" and "MASTER_ADDR" not in env and "PET_RANK" not in env
    assert "HYPERLOOM_SIGNING_KEY_FILE" not in env and env["HYPERLOOM_BENCHMARK_BACKEND"] == "bypass"


def test_a_failed_test_is_a_signed_result_with_its_failure_class(tmp_path, env, monkeypatch):
    def failing(call):
        return TestOutcome(1, "", "CUDA out of memory: HIP out of memory", (), SandboxFacts(sandbox_id="x"))

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(failing))

    assert _go(tmp_path, env, _sheet()) == 0

    outcome = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).outcome
    assert outcome.returncode == 1 and (outcome.failure_class or outcome.error_class)
    assert _status(tmp_path)["state"] == "succeeded"


@pytest.mark.parametrize("code", [errno.EDQUOT, errno.ENOSPC], ids=["edquot", "enospc"])
def test_a_runner_that_ran_out_of_disk_reports_disk_full_not_its_own_error(tmp_path, env, monkeypatch, code):
    def full(call):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(full))

    assert _go(tmp_path, env, _sheet()) == 1

    status = _status(tmp_path)
    assert (status["state"], status["error_class"]) == ("failed", "disk_full")
    assert f"[Errno {code}]" in status["message"]


def test_a_runner_that_failed_for_any_other_os_error_stays_a_runner_error(tmp_path, env, monkeypatch):
    def denied(call):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(denied))

    assert _go(tmp_path, env, _sheet()) == 1
    assert _status(tmp_path)["error_class"] == "runner_error"


def test_a_runner_error_names_where_it_was_raised(tmp_path, env, monkeypatch):
    """A recursion names its looping frame: the status message is all the application gets of a runner that failed."""

    def descend(depth):
        return descend(depth + 1)

    def recursing(call):
        return descend(0)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(recursing))

    assert _go(tmp_path, env, _sheet()) == 1
    message = _status(tmp_path)["message"]
    assert message.startswith("RecursionError: maximum recursion depth exceeded")
    assert "(at test_run.py:" in message and " descend" in message
    assert message.count(" descend") == 1, "a frame repeated by the recursion is named once"


def test_the_cause_the_backend_read_in_the_logs_is_what_the_signed_result_records(tmp_path, env, monkeypatch):
    reason = (
        "serve_flag: launch_server.py: error: ambiguous option: --chunked-prefill could match --chunked-prefill-size"
    )

    def refused_flag(call):
        return TestOutcome(
            1, "[]", "", (), SandboxFacts(sandbox_id="x"), failure_class="candidate", failure_reason=reason
        )

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(refused_flag))

    assert _go(tmp_path, env, _sheet()) == 0

    outcome = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).outcome
    assert (outcome.failure_class, outcome.failure_reason, outcome.error_class) == ("candidate", reason, None)


def test_a_boot_timeout_is_recorded_as_the_candidates_failure_with_its_return_code(tmp_path, env, monkeypatch):
    reason = "server_boot_timeout: the server was not ready within 1800s of the 3600s a round may take"

    def slow_boot(call):
        return TestOutcome(
            SERVER_BOOT_TIMEOUT_RC,
            "[]",
            "",
            (),
            SandboxFacts(sandbox_id="x"),
            failure_class="candidate",
            failure_reason=reason,
        )

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(slow_boot))

    assert _go(tmp_path, env, _sheet()) == 0

    outcome = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).outcome
    assert (outcome.returncode, outcome.failure_class, outcome.failure_reason) == (
        SERVER_BOOT_TIMEOUT_RC,
        "candidate",
        reason,
    )


def test_a_timeout_is_recorded_as_timed_out(tmp_path, env, monkeypatch):
    def slow(call):
        raise subprocess.TimeoutExpired("t", 1)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(slow))

    _go(tmp_path, env, _sheet())

    outcome = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).outcome
    assert outcome.timed_out and outcome.failure_class == "unclear" and outcome.failure_reason == "timeout"


def test_a_call_the_local_backend_refuses_is_spec_rejected(tmp_path, env, monkeypatch):
    def refuse(call):
        raise runner.TestCallRefused("RUN_EVAL contradicts the call")

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(refuse))

    assert _go(tmp_path, env, _sheet()) == 3
    assert "RUN_EVAL contradicts" in _status(tmp_path)["message"]


def test_hardware_that_differs_from_the_sheet_is_a_preflight_mismatch(tmp_path, env, monkeypatch):
    monkeypatch.setattr(
        runner.hardware, "read_gpu_facts", lambda visible: [GpuFacts(0, "AMD Instinct MI300X", 750.0, None)]
    )
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, _sheet()) == 4
    assert _status(tmp_path)["error_class"] == "preflight_mismatch" and backend.calls == []


def test_new_uncorrectable_ecc_errors_are_a_node_fault_and_no_result(tmp_path, env, monkeypatch):
    counts = iter([{0: EccCounts(0, 0)}, {0: EccCounts(0, 2)}])
    monkeypatch.setattr(runner.hardware, "ecc_counts", lambda visible: next(counts))
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 5
    assert _status(tmp_path)["error_class"] == "node_fault"
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_a_symlink_in_the_output_is_refused_and_nothing_is_signed(tmp_path, env, monkeypatch):
    def planted(call):
        (call.output_dir / "link").symlink_to("/etc/passwd")
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(planted))

    assert _go(tmp_path, env, _sheet()) == 1
    assert _status(tmp_path)["error_class"] == "unsafe_output"
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_the_replay_clients_installation_in_an_agentx_workspace_is_not_part_of_the_signed_output(
    tmp_path, env, monkeypatch
):
    def agentx(call):
        workspace = call.output_dir / "variant_00_test" / "benchmark_sglang_1"
        (workspace / ".agentx-runtime" / "venv" / "bin").mkdir(parents=True)
        (workspace / ".agentx-runtime" / "venv" / "bin" / "python").symlink_to("/usr/bin/python3")
        (workspace / "benchmark_report.json").write_text("{}")
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(agentx))

    assert _go(tmp_path, env, _sheet()) == 0

    files = sorted(f.path for f in ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes()).files)
    assert "variant_00_test/benchmark_sglang_1/benchmark_report.json" in files
    assert not any(".agentx-runtime" in path for path in files)
    assert not list((tmp_path / "out" / "files").rglob(".agentx-runtime"))


def test_the_per_request_export_of_an_agentx_round_stays_in_the_signed_output(tmp_path, env, monkeypatch):
    lines = b'{"metadata": {"benchmark_phase": "profiling"}, "metrics": {}}\n'

    def agentx(call):
        workspace = call.output_dir / "variant_00_test" / "benchmark_sglang_1"
        (workspace / ".agentx-runtime" / "venv").mkdir(parents=True)
        (workspace / "aiperf_artifacts").mkdir(parents=True)
        (workspace / "aiperf_artifacts" / "profile_export.jsonl").write_bytes(lines)
        (workspace / "benchmark_report.json").write_text("{}")
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(agentx))

    assert _go(tmp_path, env, _sheet()) == 0

    manifest = ResultManifest.from_bytes((tmp_path / "out" / "manifest.json").read_bytes())
    listed = {f.path: f for f in manifest.files}
    export = listed["variant_00_test/benchmark_sglang_1/aiperf_artifacts/profile_export.jsonl"]
    assert (export.size, export.sha256) == (len(lines), hashlib.sha256(lines).hexdigest())
    assert not any(".agentx-runtime" in path for path in listed), "the installation goes, the export stays"


def test_a_link_where_the_replay_clients_installation_belongs_is_still_refused(tmp_path, env, monkeypatch):
    def planted(call):
        (call.output_dir / ".agentx-runtime").symlink_to("/etc")
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(planted))

    assert _go(tmp_path, env, _sheet()) == 1
    assert _status(tmp_path)["error_class"] == "unsafe_output"


def test_the_sheet_validates_as_the_writer_produces_it():
    MeasurementSpec.from_document(_sheet())


def _repro_sheet(tmp_path, kind: str) -> dict:
    from hyperloom.measure.tests.repro_fixtures import case_attachment

    class _Blobs:  # the sandbox's /in/blobs
        def put(self, data: bytes) -> str:
            import hashlib

            name = hashlib.sha256(data).hexdigest()
            (tmp_path / "in" / "blobs").mkdir(parents=True, exist_ok=True)
            (tmp_path / "in" / "blobs" / name).write_bytes(data)
            return name

    case = case_attachment(_Blobs(), "exit 0\n")
    attachment = {"id": case.id, "kind": "dir", "blob": f"sha256:{case.sha256}", "manifest_sha256": "1" * 64}
    return _sheet(kind=kind, attachments=[attachment])


def test_a_repro_call_carries_its_attachments_already_unpacked_under_the_work_directory(tmp_path, env, monkeypatch):
    seen: list[bool] = []

    def observing(call: TestCall) -> TestOutcome:
        seen.append((tmp_path / "scratch" / "test" / "attach" / "case-1" / "repro.sh").is_file())
        return _ok(call)

    backend = _Backend(observing)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)
    (tmp_path / "in").mkdir()

    _go(tmp_path, env, _repro_sheet(tmp_path, "repro"))

    (call,) = backend.calls
    assert call.kind == "repro" and [(a.id, a.kind) for a in call.attachments] == [("case-1", "dir")]
    assert call.cwd == str(tmp_path / "scratch" / "test") and seen == [True]


def test_only_a_repro_call_is_told_which_attachments_are_its_case(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)
    (tmp_path / "in").mkdir()

    _go(tmp_path, env, _repro_sheet(tmp_path, "test"))

    assert backend.calls[0].attachments == ()


def test_the_signing_key_file_is_gone_before_the_test_runs_and_the_result_is_still_signed(
    tmp_path, env, key, monkeypatch
):
    key_file = tmp_path / "key.pem"
    seen: list[bool] = []

    def observing(call: TestCall) -> TestOutcome:
        seen.append(key_file.exists())
        return _ok(call)

    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(observing))

    assert _go(tmp_path, env, _sheet()) == 0

    assert seen == [False], "what the test runs must not be able to read the key"
    out = tmp_path / "out"
    assert signing.verify(
        (out / "manifest.json").read_bytes(), (out / "manifest.sig").read_text(), signing.public_key_pem(key)
    )


def test_a_signing_key_file_that_cannot_be_removed_stops_the_runner_at_start(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    def refuse(self, missing_ok=False):
        raise PermissionError("read-only")

    monkeypatch.setattr(runner.Path, "unlink", refuse)

    assert _go(tmp_path, env, _sheet()) != 0

    assert _status(tmp_path)["error_class"] == "runner_start_failed" and backend.calls == []


# --- the start-up check of the cards, the image, and the node after the test ---------------------------------------


def test_a_visible_card_that_holds_memory_stops_the_runner_before_it_waits_for_a_sheet(tmp_path, env, monkeypatch):
    seen: list[list[int]] = []
    monkeypatch.setattr(runner.hardware, "busy_gpus", lambda visible: seen.append(visible) or [0])
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, None) == 2

    status = _status(tmp_path)
    assert status["error_class"] == "runner_start_failed" and "GPU 0 already holds memory" in status["message"]
    assert seen and all(visible == [0] for visible in seen), "only the cards this sandbox may use are asked about"
    assert backend.calls == [] and not (tmp_path / "out" / "manifest.json").exists()


def test_a_card_whose_memory_is_released_while_it_is_read_again_lets_the_test_run(
    tmp_path, env, monkeypatch, settle_clock
):
    """The retry of a test whose server crashed starts on the same card while the crashed server is still torn down."""
    looks: list[float] = []

    def releasing(visible):
        looks.append(settle_clock.now)
        return [0] if settle_clock.now < 20.0 else []

    monkeypatch.setattr(runner.hardware, "busy_gpus", releasing)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 0

    assert _status(tmp_path)["state"] == "succeeded"
    assert looks[0] == 0.0 and looks[-1] == 20.0 and len(looks) == 21, "read every second until it was released"


def test_a_card_still_held_after_the_release_wait_fails_the_start_once_the_wait_is_over(
    tmp_path, env, monkeypatch, settle_clock
):
    monkeypatch.setattr(runner.hardware, "busy_gpus", lambda visible: [0])

    assert _go(tmp_path, env, None) == 2

    assert "GPU 0 already holds memory" in _status(tmp_path)["message"]
    assert settle_clock.now == runner.hardware.CARD_RELEASE_WAIT_SEC


def test_a_free_card_lets_the_test_run_and_leaves_no_note(tmp_path, env, monkeypatch):
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 0

    assert _status(tmp_path) == {"state": "succeeded", "error_class": None, "message": ""}


def test_memory_use_that_cannot_be_read_is_not_a_refusal_but_is_written_down(tmp_path, env, monkeypatch):
    monkeypatch.setattr(runner.hardware, "busy_gpus", lambda visible: None)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 0

    assert "memory use could not be read" in _status(tmp_path)["message"]


def test_a_machine_on_another_image_is_a_preflight_mismatch(tmp_path, env, monkeypatch):
    other = "sha256:" + "99" * 32
    env[IMAGE_DIGEST_VAR] = other
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, _sheet()) == 4

    status = _status(tmp_path)
    assert status["error_class"] == "preflight_mismatch" and backend.calls == []
    assert other in status["message"] and SHEET_IMAGE in status["message"]


def test_a_machine_that_does_not_state_its_image_is_not_refused_but_the_run_says_so(tmp_path, env, monkeypatch):
    del env[IMAGE_DIGEST_VAR]
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 0

    assert f"{IMAGE_DIGEST_VAR} is not set" in _status(tmp_path)["message"]


def test_fewer_gpus_visible_after_the_test_are_a_node_fault_and_no_result(tmp_path, env, monkeypatch):
    two = [
        GpuFacts(0, "AMD Instinct MI355X", 1400.0, "SPX/NPS1"),
        GpuFacts(1, "AMD Instinct MI355X", 1400.0, "SPX/NPS1"),
    ]
    reads = iter([two, two[:1]])
    env["ROCR_VISIBLE_DEVICES"] = "0,1"
    monkeypatch.setattr(runner.hardware, "read_gpu_facts", lambda visible: next(reads))
    monkeypatch.setattr(runner.hardware, "busy_gpus", lambda visible: [])
    monkeypatch.setattr(runner.hardware, "ecc_counts", lambda visible: {0: EccCounts(0, 0), 1: EccCounts(0, 0)})
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet(hardware={**_sheet()["hardware"], "gpu_count": 2})) == 5

    assert _status(tmp_path)["error_class"] == "node_fault" and "fewer GPUs" in _status(tmp_path)["message"]
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_gpus_that_answered_before_the_test_and_do_not_after_it_are_a_node_fault(tmp_path, env, monkeypatch):
    reads = iter([[GpuFacts(0, "AMD Instinct MI355X", 1400.0, "SPX/NPS1")], None])
    monkeypatch.setattr(runner.hardware, "read_gpu_facts", lambda visible: next(reads))
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 5

    assert _status(tmp_path)["error_class"] == "node_fault" and "cannot be read after" in _status(tmp_path)["message"]


@pytest.mark.parametrize(
    ("before", "after", "said"),
    [
        (None, None, "before and after the test"),
        ({0: EccCounts(0, 0)}, None, "after the test"),
        (None, {0: EccCounts(0, 0)}, "before the test"),
    ],
)
def test_ecc_counts_that_cannot_be_read_give_no_verdict_and_the_run_says_so(
    tmp_path, env, monkeypatch, before, after, said
):
    counts = iter([before, after])
    monkeypatch.setattr(runner.hardware, "ecc_counts", lambda visible: next(counts))
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 0

    message = _status(tmp_path)["message"]
    assert f"ECC counts could not be read {said}" in message and "no ECC verdict" in message


def test_gpu_facts_that_cannot_be_read_are_not_compared_and_the_run_says_so(tmp_path, env, monkeypatch):
    monkeypatch.setattr(runner.hardware, "read_gpu_facts", lambda visible: None)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, _sheet()) == 0

    assert "could not be read; they were not compared" in _status(tmp_path)["message"]


# --- what the runner refuses in the sheet before it runs anything --------------------------------------------------


def _blob(tmp_path, data: bytes, *, stored_as: bytes | None = None) -> str:
    """Put ``data`` into the sandbox's /in/blobs under the name of ``stored_as`` (default: its own hash)."""
    name = hashlib.sha256(data if stored_as is None else stored_as).hexdigest()
    (tmp_path / "in" / "blobs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "in" / "blobs" / name).write_bytes(data)
    return f"sha256:{name}"


def _refused(tmp_path, env, monkeypatch, sheet: dict, message: str) -> None:
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, sheet) == 3

    status = _status(tmp_path)
    assert status["error_class"] == "spec_rejected" and message in status["message"], status
    assert backend.calls == [] and not (tmp_path / "out" / "manifest.json").exists()


def test_a_config_whose_hash_is_not_its_text_is_spec_rejected(tmp_path, env, monkeypatch):
    sheet = _sheet()
    sheet["config"][0]["sha256"] = "0" * 64

    _refused(tmp_path, env, monkeypatch, sheet, "does not match the YAML text")


def test_an_attachment_blob_that_is_not_its_address_is_spec_rejected(tmp_path, env, monkeypatch):
    address = _blob(tmp_path, b"tampered", stored_as=b"what the sheet names")
    attachment = {"id": "csv1", "kind": "file", "blob": address, "manifest_sha256": "44" * 32}

    _refused(tmp_path, env, monkeypatch, _sheet(attachments=[attachment]), "does not hash to")


def test_a_blob_that_is_not_in_the_sandbox_is_spec_rejected(tmp_path, env, monkeypatch):
    attachment = {"id": "csv1", "kind": "file", "blob": "sha256:" + "33" * 32, "manifest_sha256": "44" * 32}
    (tmp_path / "in").mkdir()

    _refused(tmp_path, env, monkeypatch, _sheet(attachments=[attachment]), "is not in /in/blobs")


def test_a_framework_file_blob_that_is_not_its_address_is_spec_rejected(tmp_path, env, monkeypatch):
    point = yaml.safe_dump(
        {
            "benchmark": {
                **POINT["benchmark"],
                "envs": {**POINT["benchmark"]["envs"], "HYPERLOOM_FRAMEWORK_PYTHON": sys.executable},
            }
        }
    )
    address = _blob(tmp_path, b"MARKER = 1\n", stored_as=b"another file")
    entry = {"path": "__init__.py", "op": "upsert", "blob": address, "mode": "0644"}
    root = {
        "id": "yaml",
        "kind": "package",
        "import_name": "yaml",
        "dist": {"name": "PyYAML", "version": "0"},
        "entries": [entry],
    }
    sheet = _sheet(
        config=[{"yaml": point, "sha256": hashlib.sha256(point.encode()).hexdigest()}],
        environment={"roots": [root], "runtime": None, "blob_root": "/b"},
    )

    _refused(tmp_path, env, monkeypatch, sheet, "does not hash to")


def _snapshot(tmp_path) -> tuple[str, str]:
    """A snapshot directory of two files and the manifest hash a sheet would name for it."""
    root = tmp_path / "snapshot"
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "a.py").write_text("a = 1\n")
    (root / "b.cfg").write_text("b\n")
    entries = [
        (rel, (root / rel).stat().st_size, hashlib.sha256((root / rel).read_bytes()).hexdigest())
        for rel in ("b.cfg", "lib/a.py")
    ]
    return str(root), dir_manifest_sha256(entries)


def _with_snapshot(snapshot: str, manifest: str) -> dict:
    return _sheet(
        environment={"roots": [], "runtime": {"snapshot": snapshot, "manifest_sha256": manifest}, "blob_root": "/b"}
    )


def test_a_runtime_snapshot_that_is_its_manifest_lets_the_test_run(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, _with_snapshot(*_snapshot(tmp_path))) == 0

    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda root: (root / "lib" / "a.py").write_text("a = 2\n"), "does not match its manifest hash"),
        (lambda root: (root / "extra.py").write_text("x\n"), "does not match its manifest hash"),
        (lambda root: (root / "b.cfg").unlink(), "does not match its manifest hash"),
        (lambda root: (root / "link").symlink_to("/etc/passwd"), "holds a symbolic link"),
        (lambda root: shutil.rmtree(root), "is not a directory"),
    ],
    ids=["changed", "added", "removed", "link", "gone"],
)
def test_a_runtime_snapshot_that_is_not_its_manifest_is_spec_rejected(tmp_path, env, monkeypatch, damage, message):
    snapshot, manifest = _snapshot(tmp_path)
    damage(tmp_path / "snapshot")

    _refused(tmp_path, env, monkeypatch, _with_snapshot(snapshot, manifest), message)


def test_a_model_file_that_is_not_its_recorded_hash_is_spec_rejected(tmp_path, env, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden": 1}')
    point = yaml.safe_dump({"benchmark": {**POINT["benchmark"], "model": str(model)}})
    sheet = _sheet(
        config=[{"yaml": point, "sha256": hashlib.sha256(point.encode()).hexdigest()}],
        model_hashes={"config.json": hashlib.sha256(b'{"hidden": 2}').hexdigest()},
    )

    _refused(tmp_path, env, monkeypatch, sheet, "model file config.json differs from the recorded hash")


def test_a_model_file_that_is_its_recorded_hash_lets_the_test_run(tmp_path, env, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden": 1}')
    point = yaml.safe_dump({"benchmark": {**POINT["benchmark"], "model": str(model)}})
    sheet = _sheet(
        config=[{"yaml": point, "sha256": hashlib.sha256(point.encode()).hexdigest()}],
        model_hashes={"config.json": hashlib.sha256(b'{"hidden": 1}').hexdigest()},
    )
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    assert _go(tmp_path, env, sheet) == 0


def test_a_recorded_model_file_the_machine_does_not_have_is_spec_rejected(tmp_path, env, monkeypatch):
    sheet = _sheet(model_hashes={"config.json": "77" * 32})

    _refused(tmp_path, env, monkeypatch, sheet, "model file config.json unreadable")


# --- the measurement surface the runner signs (design 4.20) --------------------------------------------------------


def _surface_run(tmp_path, env, key, name: str, config: dict) -> str | None:
    """Run the sheet for ``config`` in its own directory; the ``surface_sha256`` of the signed manifest."""
    root = tmp_path / name
    root.mkdir()
    key_file = root / "key.pem"
    key_file.write_bytes(signing.private_key_pem(key))
    text = yaml.safe_dump(config)
    sheet = _sheet(config=[{"yaml": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}])

    assert _go(root, {**env, "HYPERLOOM_SIGNING_KEY_FILE": str(key_file)}, sheet) == 0, _status(root)

    return ResultManifest.from_bytes((root / "out" / "manifest.json").read_bytes()).surface_sha256


def _surface_config(tmp_path, **envs) -> dict:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"hidden": 1}')
    scripts = tmp_path / "ix" / "benchmarks"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "run.sh").write_text("run --isl $ISL\n")
    values = {"TP": 1, "CONC": 4, "RUN_EVAL": "false", "ISL": 1024, "OSL": 1024, **envs}
    benchmark = {"model": str(model), "framework": "sglang", "benchmark_script": "run.sh", "envs": values}
    return {"benchmark": benchmark}


def test_the_runner_signs_the_surface_of_the_yaml_it_executes(tmp_path, env, key, monkeypatch):
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    first = _surface_run(tmp_path, env, key, "first", _surface_config(tmp_path))
    again = _surface_run(tmp_path, env, key, "again", _surface_config(tmp_path))
    other_request = _surface_run(tmp_path, env, key, "conc", _surface_config(tmp_path, CONC=64, RUN_EVAL="true"))

    assert first is not None and len(first) == 64
    assert again == first and other_request == first, "the request's own settings are not part of the surface"


def _rewrite(path_in_tmp: str, text: str):
    """A change that rewrites one of the surface's files and keeps the YAML."""

    def change(tmp_path, config):
        (tmp_path / path_in_tmp).write_text(text)
        return config

    return change


@pytest.mark.parametrize(
    "change",
    [
        lambda tmp_path, config: _surface_config(tmp_path, ISL=2048),
        lambda tmp_path, config: _surface_config(tmp_path, MAGPIE_EVAL_LIMIT=5),
        _rewrite("ix/benchmarks/run.sh", "run --other\n"),
        _rewrite("model/config.json", '{"hidden": 2}'),
    ],
    ids=["workload", "evaluation", "benchmark_script_content", "model_file"],
)
def test_a_changed_surface_input_changes_the_signed_hash(tmp_path, env, key, monkeypatch, change):
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))
    config = _surface_config(tmp_path)
    before = _surface_run(tmp_path, env, key, "before", config)

    after = _surface_run(tmp_path, env, key, "after", change(tmp_path, config))

    assert after != before


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    """What acceptance makes of a request: the canonical surface and the YAML of its point, from the real renderer."""
    for name in (
        "CONC",
        "ISL",
        "OSL",
        "RANDOM_RANGE_RATIO",
        "RUN_EVAL",
        "TP",
        "EP",
        "PORT",
        "MAX_MODEL_LEN",
        "PRECISION",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "MAGPIE_EVAL_TASKS",
        "MAGPIE_EVAL_LIMIT",
        "MAGPIE_EVAL_TOKENIZED_REQUESTS",
        "INFERENCE_OPTIMIZER_SERVER_ARGS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path / "userdata"))
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    canonical = default_canonical(
        tmp_path / "canonical", primary_concurrency=32, config_path=default_baseline_config(), model_path=str(model)
    )
    base = PackageBase("sglang", tmp_path, None, None)
    environment = compile_environment(
        recipe_with(),
        RecipeDelta(),
        blobs=LocalBlobStore(tmp_path / "blobs"),
        active_image_digest=IMAGE,
        locator=lambda _l: base,
    )
    (point,) = generate_points(environment, canonical, tmp_path / "points", concurrency=(8,))
    return canonical, point


def test_the_hash_the_runner_signs_is_the_one_acceptance_computed_for_the_same_request(
    tmp_path, env, key, monkeypatch, accepted
):
    canonical, point = accepted
    text = point.read_text()
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))

    signed = _surface_run(tmp_path, env, key, "run", yaml.safe_load(text))

    assert signed == canonical.surface.sha256

    (tmp_path / "model" / "config.json").write_text('{"changed": true}')
    assert _surface_run(tmp_path, env, key, "later", yaml.safe_load(text)) != canonical.surface.sha256


def test_under_magpie_the_signed_script_hash_is_of_the_launcher_the_backend_installs(tmp_path, env, key, monkeypatch):
    """The backend overwrites the checkout's script with Magpie's launcher; the surface is of the file that runs."""
    magpie = tmp_path / "magpie-checkout"
    (magpie / "Magpie" / "scripts" / "benchmark").mkdir(parents=True)
    (magpie / "Magpie" / "scripts" / "benchmark" / "run.sh").write_text("magpie launcher\n")
    monkeypatch.setattr(runner, "ensure_magpie", lambda commit, cache: magpie)
    monkeypatch.setenv("MAGPIE_PATH", str(magpie))
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _Backend(_ok))
    config = _surface_config(tmp_path)

    def signed(name: str, sheet_backend: str) -> str | None:
        root = tmp_path / name
        root.mkdir()
        (root / "key.pem").write_bytes(signing.private_key_pem(key))
        text = yaml.safe_dump(config)
        sheet = _sheet(config=[{"yaml": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}])
        sheet["procedure"]["benchmark_backend"] = sheet_backend
        assert _go(root, {**env, "HYPERLOOM_SIGNING_KEY_FILE": str(root / "key.pem")}, sheet) == 0, _status(root)
        return ResultManifest.from_bytes((root / "out" / "manifest.json").read_bytes()).surface_sha256

    stale_checkout = signed("bypass", "bypass")  # no launcher installed: the script is the checkout's own
    installed = signed("magpie", "magpie")
    (tmp_path / "ix" / "benchmarks" / "run.sh").write_text("magpie launcher\n")
    already_installed = signed("settled", "bypass")

    assert installed != stale_checkout
    assert installed == already_installed


def _agentx_sheet() -> dict:
    text = yaml.safe_dump({"benchmark": {**POINT["benchmark"], "agentx": {"enabled": True}}})
    return _sheet(config=[{"yaml": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}])


def _parser_when_the_test_runs(tmp_path, seen: list[str]):
    """A backend that records the replay client's parser as it is when the test starts."""

    def run(call):
        seen.append((tmp_path / "ix" / "utils" / "aiperf" / aiperf_source.PARSER_FILE).read_text())
        return _ok(call)

    return _Backend(run)


def test_an_agentx_tests_replay_client_is_patched_before_the_test_starts(tmp_path, env, monkeypatch):
    aiperf_source.write_client(tmp_path / "ix")
    seen: list[str] = []
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _parser_when_the_test_runs(tmp_path, seen))

    assert _go(tmp_path, env, _agentx_sheet()) == 0

    assert len(seen) == 1 and "token_counts=record.token_counts or TokenCounts()," in seen[0]
    assert _status(tmp_path) == {"state": "succeeded", "error_class": None, "message": ""}


def test_a_test_that_is_not_an_agentx_one_leaves_the_replay_client_as_it_is(tmp_path, env, monkeypatch):
    aiperf_source.write_client(tmp_path / "ix")
    seen: list[str] = []
    monkeypatch.setattr(runner, "LocalMeasurementBackend", _parser_when_the_test_runs(tmp_path, seen))

    assert _go(tmp_path, env, _sheet()) == 0

    assert seen == [aiperf_source.PARSER]
    assert _status(tmp_path)["message"] == ""


@pytest.mark.parametrize(
    ("parser", "why"),
    [
        (None, "has no utils/aiperf submodule"),
        (aiperf_source.PARSER.replace("return as error record", "moved"), "inference_result_parser.py does not hold"),
    ],
    ids=["no-client", "other-client"],
)
def test_an_agentx_test_whose_client_cannot_be_patched_still_runs_and_the_status_says_it_ran_without(
    tmp_path, env, monkeypatch, parser, why
):
    if parser is not None:
        aiperf_source.write_client(tmp_path / "ix", parser=parser)
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    assert _go(tmp_path, env, _agentx_sheet()) == 0

    assert len(backend.calls) == 1, "the patch only adds a field to a failed request's line: the test runs without it"
    status = _status(tmp_path)
    assert status["state"] == "succeeded"
    assert status["message"].startswith(
        "the replay client was not patched to export the output token count of a failed"
    )
    assert why in status["message"]


SHORT_PROMPTS = {
    "prompt_set": "short-prompts-0123456789abcdef",
    "prompts_sha256": "a" * 64,
    "max_new_tokens": 32,
    "phase_cap_sec": 1200.0,
    "trigger": ["purpose final"],
}


def test_the_short_prompt_plan_of_the_sheet_is_the_one_the_call_carries(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)
    sheet = _sheet()
    sheet["procedure"]["short_prompts"] = SHORT_PROMPTS

    assert _go(tmp_path, env, sheet) == 0

    (call,) = backend.calls
    assert call.short_prompts is not None and call.short_prompts.to_document() == SHORT_PROMPTS


def test_a_sheet_that_asks_for_no_check_gives_a_call_with_none(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    _go(tmp_path, env, _sheet())

    assert backend.calls[0].short_prompts is None


def test_a_prompt_set_the_sandbox_does_not_hold_is_spec_rejected_before_any_server_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_RUNNER_CACHE", str(tmp_path / "empty-cache"))
    call = TestCall(
        points=(tmp_path / "p.yaml",),
        output_dir=tmp_path / "out",
        timeout_sec=60,
        cwd=str(tmp_path),
        magpie_python="",
        task_id="t",
        short_prompts=ShortPromptPlan.parse(SHORT_PROMPTS, "sheet"),
    )

    with pytest.raises(
        runner.RunnerFailure, match="short-prompt set short-prompts-0123456789abcdef cannot be used"
    ) as info:
        runner._execute(call)
    assert info.value.error_class == "spec_rejected"


LONG_CONTEXT = {
    "prompt_set": "long-context-0123456789abcdef",
    "prompts_sha256": "a" * 64,
    "length": 131072,
    "max_new_tokens": 128,
    "concurrency": 4,
    "phase_cap_sec": 2400.0,
    "trigger": ["request long_context_check"],
}


def test_the_long_context_plan_of_the_sheet_is_the_one_the_call_carries(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)
    sheet = _sheet()
    sheet["procedure"]["long_context"] = LONG_CONTEXT

    assert _go(tmp_path, env, sheet) == 0

    (call,) = backend.calls
    assert call.long_context is not None and call.long_context.to_document() == LONG_CONTEXT
    assert call.short_prompts is None


def test_a_sheet_that_asks_for_no_long_context_check_gives_a_call_with_none(tmp_path, env, monkeypatch):
    backend = _Backend(_ok)
    monkeypatch.setattr(runner, "LocalMeasurementBackend", backend)

    _go(tmp_path, env, _sheet())

    assert backend.calls[0].long_context is None


def test_a_long_context_set_the_sandbox_does_not_hold_is_spec_rejected_before_any_server_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_RUNNER_CACHE", str(tmp_path / "empty-cache"))
    call = TestCall(
        points=(tmp_path / "p.yaml",),
        output_dir=tmp_path / "out",
        timeout_sec=60,
        cwd=str(tmp_path),
        magpie_python="",
        task_id="t",
        long_context=LongContextPlan.parse(LONG_CONTEXT, "sheet"),
    )

    with pytest.raises(
        runner.RunnerFailure, match="long-context set long-context-0123456789abcdef cannot be used"
    ) as info:
        runner._execute(call)
    assert info.value.error_class == "spec_rejected"
