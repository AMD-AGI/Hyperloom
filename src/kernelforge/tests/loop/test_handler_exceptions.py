"""Handlers that persist, probe or clean up catch the failures those steps expect and nothing else."""

from __future__ import annotations

import asyncio
import pathlib
from types import SimpleNamespace

import pytest

import kernelforge.knowledge as knowledge_package
import kernelforge.loop.aiter_cache as aiter_cache
import kernelforge.loop.archive as archive_module
import kernelforge.loop.knowledge_integration as integ
import kernelforge.loop.run_state as run_state_module
import kernelforge.loop.runner as runner_module
import kernelforge.loop.task_preflight as task_preflight
from kernelforge.tests.test_insession_gate_protection import _gate
from kernelforge.tests.test_loop_runner import _make_loop


def _raiser(failure: Exception):
    def fail(*_args, **_kwargs):
        raise failure

    return fail


def _failure_id(failure: Exception) -> str:
    return type(failure).__name__


EXPECTED_WRITE_FAILURES = [OSError("disk"), TypeError("type"), ValueError("value")]
UNEXPECTED = RuntimeError("defect")


@pytest.mark.parametrize("failure", EXPECTED_WRITE_FAILURES, ids=_failure_id)
def test_archive_marks_itself_degraded_on_an_expected_write_failure(tmp_path, monkeypatch, failure):
    archive = archive_module.CandidateArchive(str(tmp_path))
    monkeypatch.setattr(archive, "_write_text", _raiser(failure))

    path = archive.record(archive_module.CandidateRecord(iteration=1, kernel_source="x = 1\n"))

    assert path is None
    assert archive.persistence_errors == [f"record iteration 1: {failure}"]


def test_archive_does_not_hide_an_unexpected_write_failure(tmp_path, monkeypatch):
    archive = archive_module.CandidateArchive(str(tmp_path))
    monkeypatch.setattr(archive, "_write_text", _raiser(UNEXPECTED))

    with pytest.raises(RuntimeError, match="^defect$"):
        archive.record(archive_module.CandidateRecord(iteration=1, kernel_source="x = 1\n"))


@pytest.mark.parametrize(
    "failure",
    [OSError("walk failed"), RuntimeError("Symlink loop"), ValueError("embedded null byte")],
    ids=_failure_id,
)
def test_gate_reports_an_expected_inventory_scan_failure_as_an_error(tmp_path, monkeypatch, failure):
    gate, _workspace = _gate(tmp_path)
    monkeypatch.setattr(gate, "_iter_snapshot_paths", _raiser(failure))

    states, errors = gate._snapshot_protected_states()

    assert states == {}
    assert errors == [f"protected inventory scan failed: {type(failure).__name__}: {failure}"]


def test_gate_does_not_hide_an_unexpected_inventory_scan_failure(tmp_path, monkeypatch):
    gate, _workspace = _gate(tmp_path)
    monkeypatch.setattr(gate, "_iter_snapshot_paths", _raiser(KeyError("defect")))

    with pytest.raises(KeyError, match="defect"):
        gate._snapshot_protected_states()


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        (lambda: integ._bench_once("driver.py"), None),
        (lambda: integ._correctness_once("driver.py", 30.0), False),
    ],
    ids=["bench", "correctness"],
)
@pytest.mark.parametrize("failure", [OSError("spawn failed"), ValueError("bad timing")], ids=_failure_id)
def test_warm_start_probe_turns_an_expected_failure_into_a_failed_probe(monkeypatch, probe, expected, failure):
    def fail_run(coroutine):
        coroutine.close()
        raise failure

    monkeypatch.setattr(integ.asyncio, "run", fail_run)

    assert probe() is expected


@pytest.mark.parametrize(
    "probe",
    [
        lambda: integ._bench_once("driver.py"),
        lambda: integ._correctness_once("driver.py", 30.0),
    ],
    ids=["bench", "correctness"],
)
def test_warm_start_probe_does_not_hide_an_unexpected_failure(monkeypatch, probe):
    def fail_run(coroutine):
        coroutine.close()
        raise UNEXPECTED

    monkeypatch.setattr(integ.asyncio, "run", fail_run)

    with pytest.raises(RuntimeError, match="^defect$"):
        probe()


def _state_init(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(pathlib.Path, "mkdir", _raiser(failure))
    return run_state_module.LoopStateStore(str(tmp_path))


def _state_save(tmp_path, monkeypatch, failure):
    store = run_state_module.LoopStateStore(str(tmp_path))
    monkeypatch.setattr(run_state_module, "atomic_write_text", _raiser(failure))
    store.save(run_state_module.RunState())
    return store


def _state_append(tmp_path, monkeypatch, failure):
    store = run_state_module.LoopStateStore(str(tmp_path))
    monkeypatch.setattr(run_state_module.os, "fsync", _raiser(failure))
    store.append_event({"type": "note", "iter": 1})
    return store


def _state_read(tmp_path, monkeypatch, failure):
    store = run_state_module.LoopStateStore(str(tmp_path))
    store.events_path.write_text("{}\n")
    monkeypatch.setattr(pathlib.Path, "read_text", _raiser(failure))
    store.read_events()
    return store


STATE_SITES = [_state_init, _state_save, _state_append, _state_read]


@pytest.mark.parametrize("site", STATE_SITES, ids=lambda site: site.__name__)
def test_state_store_does_not_hide_an_unexpected_failure(tmp_path, monkeypatch, site):
    with pytest.raises(RuntimeError, match="^defect$"):
        site(tmp_path, monkeypatch, UNEXPECTED)


@pytest.mark.parametrize(
    ("site", "failure", "operation"),
    [
        (_state_init, OSError("disk"), "create root"),
        (_state_save, OSError("disk"), "save"),
        (_state_save, TypeError("type"), "save"),
        (_state_save, ValueError("value"), "save"),
        (_state_append, OSError("disk"), "append"),
        (_state_append, TypeError("type"), "append"),
        (_state_append, ValueError("value"), "append"),
        (_state_read, OSError("disk"), "read"),
        (_state_read, ValueError("value"), "read"),
    ],
    ids=lambda value: value.__name__ if callable(value) else type(value).__name__,
)
def test_state_store_degrades_on_an_expected_failure(tmp_path, monkeypatch, site, failure, operation):
    store = site(tmp_path, monkeypatch, failure)

    paths = {
        "create root": "",
        "save": f" {store.state_path}",
        "append": f" {store.events_path}",
        "read": f" {store.events_path}",
    }
    assert store.degraded is True
    assert store.persistence_errors == [f"{operation}{paths[operation]}: {failure}"]


def test_state_store_skips_a_malformed_event_line_and_an_undecodable_file(tmp_path):
    store = run_state_module.LoopStateStore(str(tmp_path))
    store.events_path.write_text('not json\n{"type": "note", "iter": 1}\n')

    assert store.read_events() == [{"type": "note", "iter": 1}]
    assert store.degraded is False

    store.events_path.write_bytes(b"\xff\xfe\n")

    assert store.read_events() == []
    assert store.degraded is True


def _loop_usage(tracker_failure):
    return SimpleNamespace(
        _usage=SimpleNamespace(totals=lambda: {"calls": 1}),
        llm_usage={},
        experiment=SimpleNamespace(experiment_id="experiment"),
        tracker=SimpleNamespace(set_llm_usage=_raiser(tracker_failure)),
    )


@pytest.mark.parametrize("failure", EXPECTED_WRITE_FAILURES, ids=_failure_id)
def test_llm_usage_checkpoint_skips_an_expected_persistence_failure(failure):
    loop = _loop_usage(failure)

    runner_module.IterationLoop._checkpoint_llm_usage(loop)

    assert loop.llm_usage == {"calls": 1}


def test_llm_usage_checkpoint_does_not_hide_an_unexpected_failure():
    with pytest.raises(RuntimeError, match="^defect$"):
        runner_module.IterationLoop._checkpoint_llm_usage(_loop_usage(UNEXPECTED))


def _loop_history(publish_failure):
    return SimpleNamespace(
        state_store=SimpleNamespace(read_events=list),
        best_publisher=SimpleNamespace(publish_history=_raiser(publish_failure)),
        persistence_degraded=False,
        persistence_errors=[],
    )


@pytest.mark.parametrize("failure", [OSError("disk"), ValueError("value")], ids=_failure_id)
def test_history_publication_flags_an_expected_failure_as_degraded(failure):
    loop = _loop_history(failure)

    runner_module.IterationLoop._publish_optimization_history(loop)

    assert loop.persistence_degraded is True
    assert loop.persistence_errors == [f"publish optimization history: {failure}"]


def test_history_publication_does_not_hide_an_unexpected_failure():
    with pytest.raises(RuntimeError, match="^defect$"):
        runner_module.IterationLoop._publish_optimization_history(_loop_history(UNEXPECTED))


def test_scope_sources_record_each_unreadable_file_and_keep_reading(tmp_path):
    readable = tmp_path / "readable.py"
    readable.write_text("x = 1\n")
    undecodable = tmp_path / "undecodable.py"
    undecodable.write_bytes(b"\xff\xfe")
    declared = [str(tmp_path / "missing.py"), str(undecodable), str(readable)]
    loop = SimpleNamespace(
        _target_source_files=lambda: declared,
        ic=SimpleNamespace(workspace_dir=str(tmp_path)),
    )

    assert runner_module.IterationLoop._kernel_source_for_scope(loop) == [None, None, "x = 1\n"]


def test_scope_sources_do_not_hide_an_unexpected_read_failure(tmp_path, monkeypatch):
    loop = SimpleNamespace(
        _target_source_files=lambda: [str(tmp_path / "kernel.py")],
        ic=SimpleNamespace(workspace_dir=str(tmp_path)),
    )
    monkeypatch.setattr(pathlib.Path, "read_text", _raiser(UNEXPECTED))

    with pytest.raises(RuntimeError, match="^defect$"):
        runner_module.IterationLoop._kernel_source_for_scope(loop)


def _run_with_cleanup_failure(monkeypatch, failure):
    async def run_impl(*_args, **_kwargs):
        return ["result"]

    monkeypatch.setattr(aiter_cache, "cleanup_current_owned_aiter_locks", _raiser(failure))
    loop = SimpleNamespace(_run_impl=run_impl)
    return asyncio.run(runner_module.IterationLoop.run(loop))


@pytest.mark.parametrize("failure", [OSError("disk"), ValueError("embedded null byte")], ids=_failure_id)
def test_run_returns_its_results_when_the_lock_cleanup_fails_as_expected(monkeypatch, failure):
    assert _run_with_cleanup_failure(monkeypatch, failure) == ["result"]


def test_run_does_not_hide_an_unexpected_lock_cleanup_failure(monkeypatch):
    with pytest.raises(RuntimeError, match="^defect$"):
        _run_with_cleanup_failure(monkeypatch, UNEXPECTED)


def _integrity_violation_summary(restore_failure=None, driver_failure=None):
    session_sink = {
        "integrity_violation": True,
        "integrity_reason": "scripts changed",
        "integrity_restore": _raiser(restore_failure) if restore_failure else (lambda: None),
    }
    loop = SimpleNamespace(
        run_state=None,
        _working_tree_diff=lambda: "diff",
        _read_kernel_source=lambda: "source",
        _git_discard_worktree=lambda: None,
        _validate_driver_integrity=_raiser(driver_failure) if driver_failure else (lambda _state: ""),
    )
    frame = SimpleNamespace(iteration=1, session_sink=session_sink)
    attempt = asyncio.run(
        runner_module.IterationLoop._collect_session_candidate(
            loop,
            frame,
            agent_error=None,
            rationale="",
            plan=SimpleNamespace(path="", executable=False),
            on_iteration=None,
        )
    )
    return attempt.unmeasured_result.validation_summary


REVERT_SUMMARY = (
    "REVERT (protected integrity violation): canonical correctness and benchmark were skipped "
    "before executing the measurement surface. scripts changed"
)


@pytest.mark.parametrize("failure", [RuntimeError("inconsistent"), OSError("disk")], ids=_failure_id)
def test_integrity_restore_records_an_expected_failure(failure):
    summary = _integrity_violation_summary(restore_failure=failure)

    assert summary == (
        f"{REVERT_SUMMARY} Restoration errors: protected snapshot restore failed: {type(failure).__name__}: {failure}"
    )


@pytest.mark.parametrize("failure", [ValueError("digest mismatch"), OSError("unreadable")], ids=_failure_id)
def test_driver_integrity_check_records_an_expected_failure(failure):
    summary = _integrity_violation_summary(driver_failure=failure)

    assert summary == f"{REVERT_SUMMARY} Restoration errors: {failure}"


@pytest.mark.parametrize("stage", ["restore_failure", "driver_failure"])
def test_integrity_handling_does_not_hide_an_unexpected_failure(stage):
    failure = KeyError("defect")
    with pytest.raises(KeyError, match="defect"):
        _integrity_violation_summary(**{stage: failure})


def test_graph_probe_reports_a_spawn_failure_as_a_failed_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(task_preflight.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(task_preflight.asyncio, "create_subprocess_exec", _raiser(OSError("no such interpreter")))

    replays, reason = asyncio.run(task_preflight._count_graph_replays("driver.py", 1, 1))

    assert (replays, reason) == (task_preflight.PROBE_FAILED, "OSError: no such interpreter")


def test_graph_probe_does_not_hide_an_unexpected_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(task_preflight.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(task_preflight.asyncio, "create_subprocess_exec", _raiser(UNEXPECTED))

    with pytest.raises(RuntimeError, match="^defect$"):
        asyncio.run(task_preflight._count_graph_replays("driver.py", 1, 1))


def _orchestration_context(tmp_path, monkeypatch, failure):
    loop, workspace = _make_loop(tmp_path, monkeypatch)
    loop.run_state = run_state_module.RunState(head_commit=loop._git("rev-parse", "HEAD").splitlines()[0])
    loop.config = SimpleNamespace(
        experiments_dir=workspace / "forge_experiments",
        gpu_target="gfx942",
        local_knowledge_dir=str(tmp_path / "knowledge"),
    )
    monkeypatch.setattr(knowledge_package, "build_forge_knowledge", _raiser(failure))
    return loop._build_orchestration_context()


def test_orchestration_context_goes_without_the_knowledge_block_on_an_unreadable_wiki(tmp_path, monkeypatch):
    context = _orchestration_context(tmp_path, monkeypatch, OSError("wiki unreadable"))

    assert context.knowledge_index == ""


def test_orchestration_context_does_not_hide_an_unexpected_knowledge_failure(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="^defect$"):
        _orchestration_context(tmp_path, monkeypatch, UNEXPECTED)
