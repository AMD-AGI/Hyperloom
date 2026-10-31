# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted backend against the local sandbox provider and a fake runner (design 12.1 test_hosted_backend)."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import AccuracyValue
from hyperloom.measure.failures import SANDBOX_ACQUIRE_TIMEOUT
from hyperloom.measure.hosted import STOPPED_RETURNCODE, HostedStartError
from hyperloom.measure.hosted_log import STATE_FINISHED, STATE_SENT, STATE_SUBMITTED
from hyperloom.measure.returncodes import (
    HOSTED_INFRA_FAILURE_RC,
    HOSTED_RESULT_REJECTED_RC,
    HOSTED_RUNNER_UNAVAILABLE_RC,
    HOSTED_SPEC_REJECTED_RC,
)
from hyperloom.measure.spec import MeasurementSpec
from hyperloom.measure.tests.hosted_fixtures import (
    HOSTNAME,
    Rig,
    make_call,
    make_rig,
    point_yaml,
    runner_template,
)
from hyperloom.sandbox.socket import SandboxCreateFailed, SandboxGone


@pytest.fixture
def rig(tmp_path: Path):
    made: list[Rig] = []

    def build(**kwargs) -> Rig:
        built = make_rig(tmp_path, **kwargs)
        made.append(built)
        return built

    yield build
    for built in made:
        built.backend.close()
        built.provider.close()


def _run(rig_: Rig, tmp: Path, task: str = "req-1", **fake):
    call = make_call(tmp, task, point=point_yaml(tmp, f"{task}.yaml", **fake))
    return call, rig_.backend.run_test(call)


def _spec_of(rig_: Rig, request_id: str, attempt: int = 1) -> MeasurementSpec:
    return MeasurementSpec.from_bytes(rig_.backend.journal.load_spec(request_id, attempt))


# -- the accepted path ---------------------------------------------------------------------------------------------


def test_a_signed_result_is_verified_archived_and_landed(rig, tmp_path) -> None:
    r = rig()
    call, outcome = _run(r, tmp_path, fake_throughput=4321.5, fake_completed=77)
    assert outcome.returncode == 0
    assert outcome.failure_class is None and outcome.failure_reason is None
    report = call.output_dir / "benchmark_fake_20261014_000000" / "benchmark_report.json"
    assert json.loads(report.read_text())["throughput"]["output_throughput"] == 4321.5
    assert (call.output_dir / "server.log").read_bytes() == b"server up\nserver down\n"
    assert outcome.sandbox.node_name == HOSTNAME and outcome.sandbox.backend == "hosted"
    assert outcome.sandbox.gpu_model == "FakeGPU" and outcome.sandbox.gpu_ids
    record = r.backend.journal.latest("req-1")
    assert record.state == STATE_FINISHED and record.extra["verified"] is True
    stored = Path(record.extra["archive"])
    assert (stored / "manifest.json").is_file() and (stored / "manifest.sig").is_file()
    assert stored.parent.name == "req-1" and stored.name == record.extra["manifest_sha256"]
    assert [x.phase for x in outcome.rounds] == ["measure"]
    assert r.provider.find({}, role="measurement") == [], "the sandbox is closed after the result is read"


def test_the_journal_records_the_sandbox_before_the_sheet_and_the_end_after(rig, tmp_path) -> None:
    r = rig()
    _run(r, tmp_path)
    states = [(x.state, x.attempt) for x in r.backend.journal.records()]
    assert states == [(STATE_SUBMITTED, 1), (STATE_SENT, 1), (STATE_FINISHED, 1)]
    assert r.backend.journal.records()[0].sandbox_id.startswith("lsb-")


def test_rounds_come_from_the_runners_rounds_file_when_it_lists_one(rig, tmp_path) -> None:
    r = rig()
    _, outcome = _run(r, tmp_path, fake_rounds="true")
    assert [(x.point, x.phase, x.returncode) for x in outcome.rounds] == [(0, "measure", 0)]
    assert outcome.rounds[0].output_dir.name == "benchmark_fake_20261014_000000"


def test_without_a_rounds_file_a_report_in_the_warmup_directory_is_not_a_measuring_round(rig, tmp_path) -> None:
    r = rig()
    _, outcome = _run(r, tmp_path, fake_warmup="true")
    assert [(x.phase, x.output_dir.name) for x in outcome.rounds] == [("measure", "benchmark_fake_20261014_000000")]
    assert "warmup_round" not in outcome.rounds[0].output_dir.parts


def test_the_runner_reports_progress_and_on_output_is_called(rig, tmp_path) -> None:
    r = rig()
    heard: list[int] = []
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_delay=0.3), on_output=lambda: heard.append(1))
    assert r.backend.run_test(call).returncode == 0
    assert heard, "the runner's progress file reached the silence detector"


def test_a_running_tests_live_output_is_what_its_runner_reports_and_ends_with_the_test(rig, tmp_path) -> None:
    r = rig()
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_mode="hang", fake_log_line="decode batch 8"))
    assert r.backend.live_output("req-1") is None, "a test that is not running has none"
    result: list = []
    worker = threading.Thread(target=lambda: result.append(r.backend.run_test(call)))
    worker.start()
    try:
        deadline = time.monotonic() + 20
        while r.backend.live_output("req-1") is None and time.monotonic() < deadline:
            time.sleep(0.02)
        reported = r.backend.live_output("req-1")
        assert reported is not None and reported[1] == "decode batch 8" and 0.0 <= reported[0] < 30.0
        assert r.backend.live_output("req-2") is None, "each test has its own"
    finally:
        r.backend.withdraw("req-1")
        worker.join(30)
    assert not worker.is_alive() and result[0].returncode == STOPPED_RETURNCODE
    assert r.backend.live_output("req-1") is None, "what a test said is not kept after it ended"


def test_a_runner_that_reports_no_log_leaves_the_live_output_empty(rig, tmp_path) -> None:
    r = rig()
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_mode="hang"))
    heard: list[int] = []
    call = replace(call, on_output=lambda: heard.append(1))
    worker = threading.Thread(target=lambda: r.backend.run_test(call))
    worker.start()
    try:
        deadline = time.monotonic() + 20
        while not heard and time.monotonic() < deadline:
            time.sleep(0.02)
        assert heard, "the runner's progress was read"
        time.sleep(0.3)  # several more polls of the same file
        assert r.backend.live_output("req-1") is None
    finally:
        r.backend.withdraw("req-1")
        worker.join(30)


def test_events_announce_start_and_finish(rig, tmp_path) -> None:
    r = rig()
    seen: list[tuple[str, dict]] = []

    class Sink:
        def emit(self, type, data):
            seen.append((type, dict(data)))

    backend = r.another_backend()
    backend._events = Sink()
    backend.run_test(make_call(tmp_path))
    kinds = [kind for kind, _ in seen]
    assert kinds == ["hyperloom.measurement_started", "hyperloom.measurement_finished"]
    assert seen[1][1]["accepted"] is True and seen[0][1]["request_id"] == "req-1"


def test_a_result_dir_outside_the_output_dir_is_copied_back_to_its_place(rig, tmp_path) -> None:
    r = rig()
    external = tmp_path / "elsewhere" / "results"
    call = make_call(tmp_path, result_dir=str(external))
    spec_name = "result_dir_results"
    outcome = r.backend.run_test(call)
    assert outcome.returncode == 0
    assert _spec_of(r, "req-1").outputs.result_dir == spec_name
    assert not external.exists(), "the fake runner wrote no files below the result dir, so nothing to copy"


# -- failures the runner reports ----------------------------------------------------------------------------------


def test_a_failed_test_keeps_the_runners_classification(rig, tmp_path) -> None:
    r = rig()
    _, outcome = _run(r, tmp_path, fake_mode="fail_test")
    assert outcome.returncode == 1
    assert (outcome.failure_class.value, outcome.failure_reason) == ("candidate", "import_error")
    assert r.backend.journal.latest("req-1").extra["verified"] is True, "a failing test is still a verified result"


def test_an_unclear_failure_is_passed_on_as_unclear(rig, tmp_path) -> None:
    r = rig()
    _, outcome = _run(r, tmp_path, fake_mode="fail_unknown")
    assert outcome.returncode == 1
    assert (outcome.failure_class.value, outcome.failure_reason) == ("unclear", "no_known_signature")


def test_a_runner_that_reports_a_timeout_raises_after_the_files_landed(rig, tmp_path) -> None:
    r = rig()
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_mode="timed_out"))
    with pytest.raises(subprocess.TimeoutExpired):
        r.backend.run_test(call)
    assert (call.output_dir / "server.log").exists()
    assert r.backend.journal.latest("req-1").state == STATE_FINISHED


def test_a_rejected_sheet_is_not_a_candidate_failure(rig, tmp_path) -> None:
    r = rig()
    call, outcome = _run(r, tmp_path, fake_mode="spec_rejected")
    assert outcome.returncode == HOSTED_SPEC_REJECTED_RC
    assert outcome.failure_class is None
    assert outcome.stderr.startswith("spec_rejected")
    assert (call.output_dir / "hosted-server.log").read_bytes() == b"partial log\n", "the runner's log is kept"


def test_a_node_fault_is_an_infrastructure_failure(rig, tmp_path) -> None:
    r = rig()
    _, outcome = _run(r, tmp_path, fake_mode="node_fault")
    assert outcome.returncode == HOSTED_INFRA_FAILURE_RC
    assert (outcome.failure_class.value, outcome.failure_reason) == ("infra", "node_fault")
    assert outcome.sandbox.node_name == HOSTNAME


# -- results that must be refused ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "reject"),
    [
        ("tamper", "file_hash_mismatch"),
        ("wrong_key", "bad_signature"),
        ("other_sandbox", "sandbox_mismatch"),
        ("other_request", "request_id_mismatch"),
        ("other_spec", "spec_fingerprint_mismatch"),
    ],
)
def test_a_result_that_does_not_verify_is_rejected_and_lands_nowhere(rig, tmp_path, mode, reject) -> None:
    r = rig()
    call, outcome = _run(r, tmp_path, fake_mode=mode)
    assert outcome.returncode == HOSTED_RESULT_REJECTED_RC
    assert outcome.failure_class is None and outcome.failure_reason == "result_rejected"
    assert f"({reject})" in outcome.stderr, outcome.stderr
    assert not call.output_dir.exists() or not any(call.output_dir.iterdir()), "nothing was written to the output dir"
    assert not (tmp_path / "archive").exists() or not any((tmp_path / "archive").iterdir()), "nothing was archived"
    record = r.backend.journal.latest("req-1")
    assert record.extra["verdict"] == "result_rejected" and record.extra["reject"] == reject
    assert "verified" not in record.extra
    assert r.provider.find({}, role="measurement") == []


def test_nothing_is_downloaded_from_a_manifest_whose_signature_fails(rig, tmp_path) -> None:
    r = rig()
    copied: list[str] = []
    real_create = r.provider.create

    def spying_create(*args, **kwargs):
        sandbox = real_create(*args, **kwargs)
        original = sandbox.copy_out
        sandbox.copy_out = lambda path, dest: (copied.append(path), original(path, dest))[1]
        return sandbox

    r.provider.create = spying_create
    _, outcome = _run(r, tmp_path, fake_mode="wrong_key")
    assert outcome.returncode == HOSTED_RESULT_REJECTED_RC
    assert copied == [], "the file list of an unsigned manifest is never used to fetch files"


def test_a_file_the_manifest_does_not_list_is_never_fetched_or_landed(rig, tmp_path) -> None:
    r = rig()
    call, outcome = _run(r, tmp_path, fake_mode="extra_file")
    assert outcome.returncode == 0
    assert not (call.output_dir / "extra.txt").exists()
    assert not list(Path(r.record_archive("req-1")).rglob("extra.txt"))


def test_a_signed_manifest_for_other_sheet_bytes_is_rejected(rig, tmp_path, monkeypatch) -> None:
    """The fingerprint matches (the version salt is not part of the content), but the sheet bytes differ."""
    r = rig()
    import hyperloom.measure.hosted as hosted

    monkeypatch.setattr(hosted, "spec_sha256", lambda raw: "0" * 64)
    _, outcome = _run(r, tmp_path)
    assert outcome.returncode == HOSTED_RESULT_REJECTED_RC
    assert "(spec_hash_mismatch)" in outcome.stderr


def test_a_pinned_runner_version_that_the_runner_does_not_have_is_rejected(rig, tmp_path) -> None:
    r = rig(runner_version="pinned-9")
    _, outcome = _run(r, tmp_path)
    assert outcome.returncode == HOSTED_RESULT_REJECTED_RC
    assert "(spec_fingerprint_mismatch)" in outcome.stderr


def test_a_missing_manifest_is_a_read_failure_not_an_accepted_result(rig, tmp_path) -> None:
    r = rig()
    call, outcome = _run(r, tmp_path, fake_mode="no_manifest")
    assert outcome.returncode == HOSTED_INFRA_FAILURE_RC
    assert (outcome.failure_class.value, outcome.failure_reason) == ("infra", "result_read_failed")
    assert not call.output_dir.exists() or not any(call.output_dir.iterdir())


def test_an_archive_that_cannot_be_written_stops_the_result_before_it_lands(rig, tmp_path) -> None:
    from hyperloom.measure.archive import ArchiveError

    class Broken:
        def put_bundle(self, request_id, manifest_sha256, staging_dir):
            raise ArchiveError("disk full")

    r = rig(archive=Broken())
    call, outcome = _run(r, tmp_path)
    assert outcome.returncode == HOSTED_INFRA_FAILURE_RC
    assert (outcome.failure_class.value, outcome.failure_reason) == ("infra", "archive_failed")
    assert not call.output_dir.exists() or not any(call.output_dir.iterdir())


def test_no_signing_key_means_no_measurement(rig, tmp_path) -> None:
    r = rig(templates=[runner_template(signing=False)])
    call, outcome = _run(r, tmp_path)
    assert outcome.returncode == HOSTED_RESULT_REJECTED_RC
    assert outcome.failure_reason == "no_signing_key"
    assert r.backend.journal.records() == [], "nothing was submitted to a sandbox that cannot sign"
    assert r.provider.find({}, role="measurement") == [], "and the sandbox was closed"


def test_allow_unsigned_is_refused(rig, tmp_path) -> None:
    with pytest.raises(ValueError, match="allow_unsigned is not supported"):
        make_rig(tmp_path, allow_unsigned=True)


# -- what stops a test --------------------------------------------------------------------------------------------


def test_a_test_that_runs_past_its_timeout_raises_and_closes_the_sandbox(rig, tmp_path) -> None:
    r = rig()
    call = make_call(tmp_path, timeout_sec=1, point=point_yaml(tmp_path, fake_mode="hang"))
    with pytest.raises(subprocess.TimeoutExpired):
        r.backend.run_test(call)
    assert r.provider.find({}, role="measurement") == []
    assert r.backend.journal.latest("req-1").extra["verdict"] == "timeout"


def test_a_stopping_run_closes_the_sandbox_and_returns_the_stop_code(rig, tmp_path) -> None:
    flag = threading.Event()
    r = rig(should_stop=flag.is_set)
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_mode="hang"))
    threading.Timer(0.5, flag.set).start()
    outcome = r.backend.run_test(call)
    assert outcome.returncode == STOPPED_RETURNCODE and outcome.failure_class is None
    assert r.provider.find({}, role="measurement") == []


def test_a_sandbox_that_vanishes_is_a_lost_runner(rig, tmp_path) -> None:
    r = rig()
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_mode="hang"))
    result: list = []
    worker = threading.Thread(target=lambda: result.append(r.backend.run_test(call)))
    worker.start()
    deadline = time.monotonic() + 10
    while not r.provider.find({}, role="measurement") and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)
    r.provider.connect(r.provider.find({}, role="measurement")[0].sandbox_id, role="measurement").kill()
    worker.join(30)
    assert not worker.is_alive()
    outcome = result[0]
    assert outcome.returncode == HOSTED_INFRA_FAILURE_RC
    assert (outcome.failure_class.value, outcome.failure_reason) == ("infra", "runner_lost")


def test_no_sandbox_within_the_start_timeout_is_an_infrastructure_failure(rig, tmp_path) -> None:
    r = rig(gpus=1, runner_start_timeout_sec=0.3)
    holder = r.provider.create("runner-1", role="measurement", timeout_sec=60, metadata={})
    try:
        _, outcome = _run(r, tmp_path)
    finally:
        holder.kill()
    assert outcome.returncode == HOSTED_RUNNER_UNAVAILABLE_RC
    assert (outcome.failure_class.value, outcome.failure_reason) == ("infra", "runner_start_timeout")


def test_a_sandbox_that_failed_before_it_ran_is_an_infrastructure_failure_with_the_platforms_message(
    rig, tmp_path, monkeypatch
) -> None:
    r = rig(gpus=1)
    keys: list[str] = []
    message = "500: sandbox_create_failed: workload sbx-f7f56db77ec7244ae34d9886d2ed9dde phase Failed"

    def failing(template, *, role, timeout_sec, metadata, options=None):
        keys.append(metadata["agent-sandbox.io/idempotency-key"])
        raise SandboxCreateFailed(message)

    monkeypatch.setattr(r.provider, "create", failing)
    _, outcome = _run(r, tmp_path)
    assert outcome.returncode == HOSTED_INFRA_FAILURE_RC
    assert (outcome.failure_class.value, outcome.failure_reason) == ("infra", "sandbox_create_failed")
    assert outcome.stderr == f"sandbox_create_failed: {message}"
    assert keys == ["run-1:req-1:1"], "one create, its key not replayed"
    assert r.backend._pool.active == 0
    [line] = r.backend.journal.records()
    assert (line.attempt, line.state, line.sandbox_id) == (1, "finished", None), "no sandbox ran"
    assert line.extra["detail"] == outcome.stderr, "the platform's message outlives the process"
    assert r.backend.journal.next_attempt("req-1") == 2, "the toolbox's one retry is a new attempt with a new key"


def test_a_card_count_without_a_runner_template_is_a_refused_sheet(rig, tmp_path) -> None:
    r = rig()
    call = make_call(tmp_path, point=point_yaml(tmp_path, tp=3))
    outcome = r.backend.run_test(call)
    assert outcome.returncode == HOSTED_SPEC_REJECTED_RC and outcome.failure_reason == "no_runner_template"
    assert r.backend.journal.records() == []


# -- resume ---------------------------------------------------------------------------------------------------------


class _Crash(BaseException):
    """What a killed application process looks like to the code under test: it unwinds without cleaning up."""


def _crash_mid_wait(rig_: Rig, call):
    backend = rig_.another_backend()
    backend._pool.release = lambda runner: None  # a dead process closes nothing

    def die(attempt):
        raise _Crash

    backend._wait = die
    with pytest.raises(_Crash):
        backend.run_test(call)
    return backend


def test_a_restarted_application_reattaches_instead_of_measuring_twice(rig, tmp_path) -> None:
    r = rig()
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_delay=0.5, fake_throughput=1234.0))
    _crash_mid_wait(r, call)
    submitted = r.backend.journal.latest("req-1")
    assert submitted.state == STATE_SENT
    created: list[str] = []
    real_create = r.provider.create
    r.provider.create = lambda *a, **k: created.append("x") or real_create(*a, **k)
    resumed = r.another_backend().run_test(call)
    assert resumed.returncode == 0 and created == [], "the sandbox that was already measuring was re-used"
    assert resumed.sandbox.sandbox_id == submitted.sandbox_id
    record = r.backend.journal.latest("req-1")
    assert (record.state, record.attempt) == (STATE_FINISHED, 1) and record.extra["verified"] is True
    report = call.output_dir / "benchmark_fake_20261014_000000" / "benchmark_report.json"
    assert json.loads(report.read_text())["throughput"]["output_throughput"] == 1234.0


def test_a_restart_that_finds_no_slot_for_its_sandbox_ends_the_attempt_with_the_reason(rig, tmp_path) -> None:
    """The re-attach waits for a slot no longer than the test's timeout; the sandbox it could not hold is closed."""
    r = rig()
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_mode="hang"), timeout_sec=1)
    _crash_mid_wait(r, call)
    submitted = r.backend.journal.latest("req-1")
    other = r.provider.create(
        runner_template().name, role="measurement", timeout_sec=60, metadata={"hyperloom.run-id": r.config.run_id}
    )  # a sandbox of the run that nobody holds: it takes the only slot
    started = time.monotonic()
    outcome = r.another_backend(max_parallel_tests=1).run_test(call)
    assert time.monotonic() - started < 10
    assert outcome.returncode == HOSTED_INFRA_FAILURE_RC
    assert outcome.failure_reason == SANDBOX_ACQUIRE_TIMEOUT and "a sandbox slot" in outcome.stderr
    record = r.backend.journal.latest("req-1")
    assert (record.state, record.attempt, record.sandbox_id) == (STATE_FINISHED, 1, submitted.sandbox_id)
    assert record.extra["verdict"] == SANDBOX_ACQUIRE_TIMEOUT
    running = {info.sandbox_id for info in r.provider.find({}, role="measurement")}
    assert submitted.sandbox_id not in running and other.sandbox_id in running


def test_a_restart_before_the_sheet_was_sent_sends_it_to_the_sandbox_that_waits(rig, tmp_path) -> None:
    """The application died after journalling the sandbox and before ``READY`` was written: nothing would ever start."""
    r = rig()
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_throughput=777.0))
    dying = r.another_backend()
    dying._pool.release = lambda runner: None

    def die(handle, spec):
        raise _Crash

    dying._upload_blobs = die
    with pytest.raises(_Crash):
        dying.run_test(call)
    assert r.backend.journal.latest("req-1").state == STATE_SUBMITTED
    created: list[str] = []
    real_create = r.provider.create
    r.provider.create = lambda *a, **k: created.append("x") or real_create(*a, **k)
    resumed = r.another_backend().run_test(call)
    assert resumed.returncode == 0 and created == [], "the waiting sandbox was used, not a second one"
    assert [x.state for x in r.backend.journal.records()] == [STATE_SUBMITTED, STATE_SENT, STATE_FINISHED]
    report = call.output_dir / "benchmark_fake_20261014_000000" / "benchmark_report.json"
    assert json.loads(report.read_text())["throughput"]["output_throughput"] == 777.0


def test_a_restart_after_the_sandbox_is_gone_measures_again_as_the_next_attempt(rig, tmp_path) -> None:
    r = rig()
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_delay=0.2))
    _crash_mid_wait(r, call)
    first = r.backend.journal.latest("req-1")
    r.provider.connect(first.sandbox_id, role="measurement").kill()
    resumed = r.another_backend().run_test(call)
    assert resumed.returncode == 0
    assert resumed.sandbox.sandbox_id != first.sandbox_id
    attempts = [(x.attempt, x.state) for x in r.backend.journal.records()]
    assert attempts == [
        (1, STATE_SUBMITTED),
        (1, STATE_SENT),
        (1, STATE_FINISHED),
        (2, STATE_SUBMITTED),
        (2, STATE_SENT),
        (2, STATE_FINISHED),
    ]
    assert r.backend.journal.records()[2].extra["verdict"] == "runner_lost"
    assert r.backend.journal.latest("req-1").sandbox_id == resumed.sandbox.sandbox_id


def test_a_second_call_for_a_finished_request_is_the_next_attempt(rig, tmp_path) -> None:
    r = rig()
    _run(r, tmp_path, "again", fake_mode="node_fault")
    call = make_call(tmp_path, "again", point=point_yaml(tmp_path, "again2.yaml"))
    assert r.backend.run_test(call).returncode == 0
    keys = [x.attempt for x in r.backend.journal.records() if x.state == STATE_SUBMITTED]
    assert keys == [1, 2], "the tool layer's infrastructure retry is a new sandbox with the next idempotency key"


def test_a_resumed_landing_that_already_happened_is_not_a_conflict(rig, tmp_path) -> None:
    r = rig()
    call = make_call(tmp_path, point=point_yaml(tmp_path, fake_delay=0.3))
    crashed = r.another_backend()
    crashed._pool.release = lambda runner: None
    real_decided = crashed._decided

    def die_after_landing(*args):
        raise _Crash

    crashed._decided = die_after_landing
    with pytest.raises(_Crash):
        crashed.run_test(call)
    assert (call.output_dir / "server.log").exists(), "the files landed before the crash"
    assert real_decided is not None
    resumed = r.another_backend().run_test(call)
    assert resumed.returncode == 0 and resumed.failure_class is None


# -- the sheet ------------------------------------------------------------------------------------------------------


def test_the_sheet_carries_only_whitelisted_variables_and_never_a_credential(rig, tmp_path) -> None:
    environ = {"HF_TOKEN": "secret", "HYPERLOOM_AGENTX": "1", "PATH": "/x", "INFERENCE_OPTIMIZER_SERVER_ARGS": "--a"}
    r = rig(spec={"process_environ": environ, "platform_env": {"NCCL_IB_SL": "3", "MASTER_ADDR": "m", "HOME": "/h"}})
    call = make_call(tmp_path, env_overrides={"AITER_LOG_LEVEL": "WARNING"}, unset_envs=("FOO",))
    assert r.backend.run_test(call).returncode == 0
    spec = _spec_of(r, "req-1")
    assert dict(spec.process_env) == {
        "HYPERLOOM_AGENTX": "1",
        "INFERENCE_OPTIMIZER_SERVER_ARGS": "--a",
        "AITER_LOG_LEVEL": "WARNING",
    }
    assert dict(spec.platform_env) == {"NCCL_IB_SL": "3"}
    assert spec.unset_envs == ("FOO",)
    assert "secret" not in r.backend.journal.load_spec("req-1", 1).decode()


def test_the_sheet_carries_the_pods_rccl_channel_count_but_not_its_interface_names(rig, tmp_path) -> None:
    pod = {
        "NCCL_MIN_NCHANNELS": "112",
        "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_IB_HCA": "rdma0",
        "GLOO_SOCKET_IFNAME": "eth0",
    }
    r = rig(spec={"platform_env": pod})
    assert r.backend.run_test(make_call(tmp_path)).returncode == 0
    assert dict(_spec_of(r, "req-1").platform_env) == {"NCCL_MIN_NCHANNELS": "112"}


def test_an_env_override_the_old_flow_refuses_refuses_the_sheet(rig, tmp_path) -> None:
    r = rig()
    outcome = r.backend.run_test(make_call(tmp_path, env_overrides={"HF_TOKEN": "x"}))
    assert outcome.returncode == HOSTED_SPEC_REJECTED_RC and outcome.failure_reason == "env_override_refused"
    assert r.backend.journal.records() == []


def test_shared_storage_files_become_attachments_with_placeholders(rig, tmp_path) -> None:
    shared = tmp_path / "shared"
    (shared / "overlay").mkdir(parents=True)
    (shared / "overlay" / "tuned.csv").write_text("m,n,k\n1,2,3\n")
    (shared / "table.csv").write_text("a,b\n")
    model = shared / "model"
    model.mkdir()
    r = rig(spec={"shared_roots": (shared,), "model_dir": model})
    point = tmp_path / "p.yaml"
    point.write_text(
        "benchmark:\n  model: {model}\n  envs:\n    TP: '1'\n    RUN_EVAL: 'false'\n"
        "    AITER_CONFIG_GEMM: {table}\n    PYTHONPATH: {overlay}:/usr/lib\n".format(
            model=model, table=shared / "table.csv", overlay=shared / "overlay"
        )
    )
    assert r.backend.run_test(make_call(tmp_path, point=point)).returncode == 0
    spec = _spec_of(r, "req-1")
    kinds = sorted(a.kind for a in spec.attachments)
    assert kinds == ["dir", "file"]
    by_kind = {a.kind: a for a in spec.attachments}
    assert f"${{HL_ATTACH:{by_kind['file'].id}}}" in spec.config[0].yaml
    assert f"${{HL_ATTACH:{by_kind['dir'].id}}}:/usr/lib" in spec.config[0].yaml
    assert str(shared / "table.csv") not in spec.config[0].yaml and str(shared / "overlay") not in spec.config[0].yaml
    assert f"model: {model}" in spec.config[0].yaml, "the model directory stays a path: it is mounted, not attached"
    blob = by_kind["file"].blob.removeprefix("sha256:")
    assert r.config.spec.blobs.get(blob) == b"a,b\n"
    assert by_kind["file"].manifest_sha256 == blob


def test_a_placeholder_nothing_provides_refuses_the_sheet(rig, tmp_path) -> None:
    r = rig()
    point = tmp_path / "p.yaml"
    point.write_text("benchmark:\n  envs:\n    TP: '1'\n    RUN_EVAL: 'false'\n    X: ${HL_ATTACH:k-missing}\n")
    outcome = r.backend.run_test(make_call(tmp_path, point=point))
    assert outcome.returncode == HOSTED_SPEC_REJECTED_RC and outcome.failure_reason == "placeholder_unknown"


def test_a_sweep_arm_sends_every_point_in_order(rig, tmp_path) -> None:
    r = rig()
    points = []
    for level in (64, 32):
        path = tmp_path / f"conc{level}.yaml"
        path.write_text(f"benchmark:\n  envs:\n    TP: '1'\n    RUN_EVAL: 'false'\n    CONC: '{level}'\n")
        points.append(path)
    outcome = r.backend.run_test(make_call(tmp_path, points=tuple(points), prep="conc_sweep"))
    assert outcome.returncode == 0
    spec = _spec_of(r, "req-1")
    assert ["CONC: '64'" in c.yaml for c in spec.config] == [True, False]
    assert ["CONC: '32'" in c.yaml for c in spec.config] == [False, True]
    assert spec.procedure.prep == "conc_sweep"


def test_points_that_differ_in_more_than_the_concurrency_refuse_the_sheet(rig, tmp_path) -> None:
    r = rig()
    points = (point_yaml(tmp_path, "a.yaml"), point_yaml(tmp_path, "b.yaml", marker="second"))
    outcome = r.backend.run_test(make_call(tmp_path, points=points, prep="conc_sweep"))
    assert outcome.returncode == HOSTED_SPEC_REJECTED_RC and outcome.failure_reason == "sheet_invalid"
    assert "differ in more than the concurrency" in outcome.stderr


def test_a_call_without_a_point_is_refused(rig, tmp_path) -> None:
    r = rig()
    outcome = r.backend.run_test(make_call(tmp_path, points=()))
    assert outcome.returncode == HOSTED_SPEC_REJECTED_RC and outcome.failure_reason == "no_points"


def test_attachment_and_overlay_blobs_reach_the_sandbox_before_the_sheet(rig, tmp_path) -> None:
    from hyperloom.measure.hosted_spec import SpecContext  # noqa: F401

    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "table.csv").write_text("a,b\n")
    r = rig(spec={"shared_roots": (shared,)})
    point = tmp_path / "p.yaml"
    point.write_text(f"benchmark:\n  envs:\n    TP: '1'\n    RUN_EVAL: 'false'\n    CSV: {shared / 'table.csv'}\n")
    writes: list[str] = []
    real_create = r.provider.create

    def spying_create(*args, **kwargs):
        sandbox = real_create(*args, **kwargs)
        original = sandbox.write_file
        sandbox.write_file = lambda path, data: (writes.append(path), original(path, data))[1]
        return sandbox

    r.provider.create = spying_create
    assert r.backend.run_test(make_call(tmp_path, point=point)).returncode == 0
    attachment = _spec_of(r, "req-1").attachments[0]
    sha = attachment.blob.removeprefix("sha256:")
    assert writes == [f"/in/blobs/{sha}", "/in/spec.json", "/in/READY"], "blobs, then the sheet, then the marker"


# -- start checks ---------------------------------------------------------------------------------------------------


def test_start_refuses_a_multi_node_activity_and_ray_execution(tmp_path, monkeypatch) -> None:
    with pytest.raises(HostedStartError, match="single-node"):
        make_rig(tmp_path, nodes=2)
    monkeypatch.setenv("INFERENCE_OPTIMIZER_RAY_EXEC", "1")
    with pytest.raises(HostedStartError, match="RAY_EXEC"):
        make_rig(tmp_path)


def test_start_refuses_without_runner_templates(tmp_path) -> None:
    with pytest.raises(HostedStartError, match="runner template"):
        make_rig(tmp_path, runner_templates={})


def test_the_backend_is_not_usable_before_start(tmp_path) -> None:
    from hyperloom.measure.hosted import HostedMeasurementBackend

    rig_ = make_rig(tmp_path)
    fresh = HostedMeasurementBackend(rig_.provider, rig_.config)
    with pytest.raises(HostedStartError, match="not started"):
        fresh.run_test(make_call(tmp_path))
    rig_.provider.close()


def test_a_sandbox_gone_error_from_the_provider_surfaces_as_lost(rig, tmp_path) -> None:
    r = rig()
    real = r.provider.create

    def create_then_end(*args, **kwargs):
        sandbox = real(*args, **kwargs)
        sandbox.kill()
        return sandbox

    r.provider.create = create_then_end
    outcome = r.backend.run_test(make_call(tmp_path))
    assert outcome.returncode == HOSTED_INFRA_FAILURE_RC and outcome.failure_reason == "runner_lost"
    assert isinstance(SandboxGone("x"), Exception)


def test_the_scores_of_an_accuracy_round_reach_the_outcome_from_the_signed_rounds_file(rig, tmp_path) -> None:
    r = rig()
    _, outcome = _run(r, tmp_path, fake_accuracy="0.64")
    assert [x.phase for x in outcome.rounds] == ["measure", "accuracy"]
    assert outcome.rounds[0].accuracy == ()
    assert outcome.rounds[1].accuracy == (AccuracyValue("gsm8k", "exact_match,flexible-extract", 0.64),)


def test_the_version_of_a_score_reaches_the_outcome_from_the_signed_rounds_file(rig, tmp_path) -> None:
    r = rig()
    _, outcome = _run(r, tmp_path, fake_accuracy="0.64", fake_accuracy_version="lm_eval=0.4.13;gsm8k=3")
    assert outcome.rounds[1].accuracy == (
        AccuracyValue("gsm8k", "exact_match,flexible-extract", 0.64, "lm_eval=0.4.13;gsm8k=3"),
    )


def test_an_accuracy_round_without_scores_reaches_the_outcome_without_scores(rig, tmp_path) -> None:
    r = rig()
    _, outcome = _run(r, tmp_path, fake_accuracy="null")
    assert [(x.phase, x.accuracy) for x in outcome.rounds] == [("measure", ()), ("accuracy", ())]


def test_the_warnings_of_a_round_reach_the_outcome_from_the_signed_rounds_file(rig, tmp_path) -> None:
    r = rig()
    _, outcome = _run(r, tmp_path, fake_warnings='["warmup_round_seed:1", "other", "warmup_round_seed:1"]')
    assert [x.nonfatal_warnings for x in outcome.rounds] == [("warmup_round_seed:1", "other", "warmup_round_seed:1")]
