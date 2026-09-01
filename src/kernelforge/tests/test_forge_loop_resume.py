"""CLI and history tests for explicit forge-loop workspace resume."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

import kernelforge.cli as cli_module
import kernelforge.loop.recovery as recovery_module
import kernelforge.orchestrator.agent as agent_module
import kernelforge.orchestrator.analysis as analysis_module
import kernelforge.orchestrator.orchestration as orchestration_module
import kernelforge.orchestrator.supervisor as supervisor_module
from kernelforge.cli import main
from kernelforge.config import Config
from kernelforge.loop.campaign_config import (
    CampaignConfigStore,
    create_campaign_config,
)
from kernelforge.loop.experience import ExperienceLedger
from kernelforge.loop.run_state import (
    SESSION_PAUSED,
    LoopStateStore,
    RunState,
    WorkspaceLockError,
)
from kernelforge.loop.runner import IterationConfig, IterationLoop


# The forge-loop CLI activates per-workspace aiter cache isolation, which writes
# AITER_ROOT_DIR / AITER_JIT_DIR (and friends) directly into os.environ so child
# tuner processes inherit the redirected build dirs. That is deliberate runtime
# behavior, but it is a process-global mutation monkeypatch does not undo, so it
# leaks into later tests (e.g. resolve_aiter_root picks up a dangling workspace
# path). Snapshot and restore those keys around every test in this module.
_AITER_CACHE_ENV_KEYS = (
    "AITER_ROOT_DIR",
    "AITER_JIT_DIR",
    "FORGE_AITER_CACHE_ROOT",
    "FORGE_AITER_CACHE_OWNER_PID",
    "AITER_REBUILD",
)


@pytest.fixture(autouse=True)
def _isolate_aiter_cache_env():
    import os

    saved = {key: os.environ.get(key) for key in _AITER_CACHE_ENV_KEYS}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_forge_loop_exposes_the_superset_of_resume_and_orchestration_options():
    # The resume CLI is a SUPERSET: it keeps our resume/campaign options and
    # also exposes main's orchestration options (Hyperloom shells out to this
    # command and passes them). Both families must be present.
    runner = CliRunner()

    help_result = runner.invoke(main, ["forge-loop", "--help"])
    assert help_result.exit_code == 0
    for exposed in (
        "--no-profiling",
        "--profiling",
        "--experiments-dir",
        "--git-branch",
        "--result-json",
        "--permission-mode",
        "--gpu-target",
        "--gpu-type",
        "--task-type",
        "--model",
        "--kernel-backend",
        "--supervisor-backend",
        "--profile-timeout-sec",
        "--snr-threshold",
        "--target-functions",
        "--experience-kb",
        "--no-experience-kb",
        "--return-after-read-kb",
        "--resume",
    ):
        assert exposed in help_result.output
    assert "--shapes-json" not in help_result.output
    assert "--workload-key" not in help_result.output
    assert "--max-iters" not in help_result.output


def _install_cli_fakes(monkeypatch, tmp_path):
    captured = {
        "loops": [],
        "warmstarts": [],
        "experiments": {},
        "checkpoints": {},
        "kb_writes": [],
    }
    monkeypatch.setenv("GPU_TARGET", "gfx942")
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "triton")

    def fake_from_env(**overrides):
        captured.setdefault("config_overrides", []).append(dict(overrides))
        return Config(
            project_root=tmp_path / "project",
            workspace=overrides.get("workspace", ""),
            gpu_target=overrides.get("gpu_target", "gfx942"),
            gpu_type=overrides.get("gpu_type", "mi355x"),
            agent_model=overrides.get("agent_model", "test-model"),
        )

    class FakeTracker:
        def __init__(self, experiments_dir):
            self.dir = Path(experiments_dir)

        def set_kb_experience(self, experiment_id, payload):
            captured["kb_experience"] = (experiment_id, payload)

        # The real tracker raises FileNotFoundError for an unknown ID, which is
        # what forces the CLI to materialize the caller-owned recovery record
        # before the first KEEP can checkpoint onto it.
        def get(self, experiment_id):
            try:
                return captured["experiments"][experiment_id]
            except KeyError:
                raise FileNotFoundError(f"Experiment not found: {experiment_id}") from None

        def create(self, task_id="", experiment_id=None, **kwargs):
            record = SimpleNamespace(experiment_id=experiment_id, checkpoint={})
            captured["experiments"][experiment_id] = record
            return record

        def set_checkpoint(self, experiment_id, checkpoint):
            captured["checkpoints"][experiment_id] = checkpoint

    class FakeLoop:
        def __init__(self, iter_config, tracker, config, resume=False):
            self.ic = iter_config
            self.tracker = tracker
            self.config = config
            self.resume = resume
            self.best_wall_ms = 0.8
            self.experiment = SimpleNamespace(
                experiment_id="segment-2" if resume else "segment-1",
                segment_index=2 if resume else 1,
            )
            self.run_state = SimpleNamespace(
                campaign_id="campaign-1",
                session_index=2 if resume else 1,
                next_iteration=9 if resume else 5,
                best=SimpleNamespace(
                    iteration=8 if resume else 4,
                    commit_hash="best-commit",
                ),
            )
            captured["loops"].append(self)

        def validate_resume_preflight(self):
            captured["resume_preflight_validated"] = True

        def _checkpoint_llm_usage(self):
            captured["final_usage_checkpointed"] = True

        async def run(self, **_kwargs):
            captured["run_kwargs"] = _kwargs
            try:
                with LoopStateStore(str(self.ic.workspace_dir)).workspace_lock():
                    pass
            except WorkspaceLockError:
                captured["lock_held_during_run"] = True
            return []

    def fake_warmstart(**kwargs):
        captured["warmstarts"].append(kwargs)
        try:
            with LoopStateStore(str(kwargs["workspace_dir"])).workspace_lock():
                pass
        except WorkspaceLockError:
            captured["lock_held_during_warmstart"] = True
        return {
            "candidate": False,
            "read_reason": "not_configured",
            "read_error": "",
        }

    def fake_write(**kwargs):
        captured["kb_writes"].append(kwargs)
        try:
            with LoopStateStore(str(kwargs["workspace_dir"])).workspace_lock():
                pass
        except WorkspaceLockError:
            captured["lock_held_during_finalization"] = True
        return {"written": False, "reason": "test"}

    monkeypatch.setattr(cli_module.Config, "from_env", staticmethod(fake_from_env))
    monkeypatch.setattr(cli_module, "kb_warmstart", fake_warmstart)
    monkeypatch.setattr(
        cli_module,
        "write_experience_to_kb",
        fake_write,
    )
    monkeypatch.setattr("kernelforge.loop.runner.IterationLoop", FakeLoop)
    monkeypatch.setattr("kernelforge.tracker.ExperimentTracker", FakeTracker)

    def fake_make_agent_fn(**kwargs):
        captured["agent_fn_kwargs"] = kwargs
        return None

    monkeypatch.setattr(agent_module, "make_agent_fn", fake_make_agent_fn)
    analysis_service = object()

    def fake_make_analysis_agent_service(**kwargs):
        captured["analysis_service_kwargs"] = kwargs
        return analysis_service

    monkeypatch.setattr(
        analysis_module,
        "make_analysis_agent_service",
        fake_make_analysis_agent_service,
    )
    captured["analysis_service"] = analysis_service
    orchestration_service = object()

    def fake_make_orchestration_service(**kwargs):
        captured["orchestration_service_kwargs"] = kwargs
        return orchestration_service

    monkeypatch.setattr(
        orchestration_module,
        "make_orchestration_service",
        fake_make_orchestration_service,
    )
    captured["orchestration_service"] = orchestration_service
    monkeypatch.setattr(supervisor_module, "make_supervisor_fn", lambda **_kwargs: None)
    return captured


def _initialize_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("def kernel():\n    return 1\n")
    driver.write_text("pass\n")
    subprocess.run(
        ["git", "init", "-b", "feature/test-cli"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "KernelForge Tests"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return workspace, kernel, driver


def _invoke_forge_loop(
    tmp_path,
    extra_args,
    *,
    existing_state=False,
    existing_config=False,
    include_campaign_inputs=True,
):
    workspace, kernel, driver = _initialize_workspace(tmp_path)
    if existing_config:
        config = create_campaign_config(
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            source_files=[],
            program_md_file=None,
        )
        CampaignConfigStore(str(workspace)).save(config)
    if existing_state or existing_config or "--resume" in extra_args:
        campaign_root = workspace / "forge_experiments"
        campaign_root.mkdir(exist_ok=True)
        (campaign_root / "run_state.json").write_text("{}")
    command = [
        "forge-loop",
        "--workspace",
        str(workspace),
        "--max-hours",
        "1",
        "--no-profiling",
        # These tests fake the loop and exercise campaign/resume orchestration,
        # not the real measurement-driver task preparer (covered separately).
        "--no-prepare-task",
    ]
    if include_campaign_inputs:
        command += [
            "--kernel",
            str(kernel),
            "--driver",
            str(driver),
        ]
    result = CliRunner().invoke(
        main,
        [*command, *extra_args],
    )
    return result, workspace


def _result_payload(output: str) -> dict:
    prefix = "__FORGE_RESULT__"
    start = output.index(prefix) + len(prefix)
    end = output.index(prefix, start)
    return json.loads(output[start:end])


def test_forge_loop_gpu_type_override_reaches_config(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, workspace = _invoke_forge_loop(
        tmp_path,
        ["--gpu-type", "MI300X"],
    )

    assert result.exit_code == 0
    assert captured["config_overrides"][-1]["gpu_type"] == "mi300x"
    assert CampaignConfigStore(str(workspace)).load().gpu_type == "mi300x"


def test_forge_loop_defaults_gpu_type(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(tmp_path, [])

    assert result.exit_code == 0
    assert captured["config_overrides"][-1]["gpu_type"] == "mi355x"


def test_validated_warm_start_publishes_recovery_before_iteration(tmp_path):
    workspace, kernel, _driver = _initialize_workspace(tmp_path)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "-u"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "kb warm-start: apply prior/solution"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    checkpoints = {}

    class Tracker:
        @staticmethod
        def set_checkpoint(experiment_id, checkpoint):
            checkpoints[experiment_id] = checkpoint

    result_json = tmp_path / "forge_cli_result.json"
    result = recovery_module.publish_warm_start_recovery(
        workspace_dir=str(workspace),
        base_commit=base_commit,
        warm={
            "applied": True,
            "pristine_ms": 10.0,
            "keep_baseline_ms": 8.0,
            "mean_case_speedup": 1.25,
            "solution_slug": "prior/solution",
        },
        caller_experiment_id="hyperloom",
        experience_id="experience",
        tracker=Tracker(),
        result_json=str(result_json),
    )

    root = workspace / "forge_experiments"
    manifest = json.loads((root / "best_result.json").read_text())
    sidecar = json.loads(result_json.read_text())
    assert result is not None
    assert manifest["iteration"] == 0
    assert manifest["baseline_wall_ms"] == 10.0
    assert manifest["best_wall_ms"] == 8.0
    assert manifest["search_start_mean_case_speedup"] == 1.25
    assert sidecar["warm_start"] is True
    assert sidecar["search_start_mean_case_speedup"] == 1.25
    assert sidecar["best_commit"] == manifest["commit_hash"]
    assert checkpoints["hyperloom"]["decision"] == "WARM_START"
    assert checkpoints["hyperloom"]["search_start_mean_case_speedup"] == 1.25


def test_return_after_read_kb_skips_iteration_for_validated_improvement(
    tmp_path,
    monkeypatch,
):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    def applied_warmstart(**kwargs):
        workspace = Path(kwargs["workspace_dir"])
        kernel = Path(kwargs["kernel"])
        kernel.write_text("def kernel():\n    return 2\n")
        subprocess.run(["git", "add", "-u"], cwd=workspace, check=True)
        subprocess.run(
            ["git", "commit", "-m", "kb warm-start: apply prior/solution"],
            cwd=workspace,
            check=True,
            capture_output=True,
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return {
            "candidate": True,
            "applied": True,
            "applied_commit": head,
            "applied_rank": 1,
            "pristine_ms": 10.0,
            "keep_baseline_ms": 8.0,
            "mean_case_speedup": 1.25,
            "solution_slug": "prior/solution",
            "speedup": 1.25,
            "program_md_addition": "APPLIED WARM START",
            "reference_program_md_addition": "REFERENCE ONLY",
        }

    monkeypatch.setattr(cli_module, "kb_warmstart", applied_warmstart)
    result_json = tmp_path / "return-after-kb.json"

    result, workspace = _invoke_forge_loop(
        tmp_path,
        [
            "--return-after-read-KB",
            "--result-json",
            str(result_json),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "run_kwargs" not in captured
    assert "agent_fn_kwargs" not in captured
    payload = _result_payload(result.output)
    assert payload == json.loads(result_json.read_text())
    assert payload["returned_after_read_kb"] is True
    assert payload["warm_start"] is True
    assert payload["iteration_count"] == 0
    assert payload["baseline_ms"] == 10.0
    assert payload["best_ms"] == 8.0
    assert payload["total_speedup"] == 1.25
    assert payload["incremental_improved"] is False
    assert payload["kb_experience"]["read"]["applied"] is True
    assert (workspace / "forge_experiments" / "best_result.json").is_file()
    assert "returning before iteration 1" in result.output


def test_return_after_read_kb_continues_without_applied_improvement(
    tmp_path,
    monkeypatch,
):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        ["--return-after-read-kb"],
    )

    assert result.exit_code == 0, result.output
    assert "run_kwargs" in captured
    assert "returned_after_read_kb" not in _result_payload(result.output)


def test_a_warm_start_rejected_by_the_task_suite_is_not_returned(
    tmp_path,
    monkeypatch,
):
    """A candidate the task's own suite failed cannot be the campaign's answer.

    ``kb_warmstart`` runs that suite before it adopts anything, so a candidate
    that clears SNR and breaks the task's tolerance comes back unapplied; the
    run must then search rather than publish it. The suite's own verdict is
    covered end to end in tests/test_kb_warmstart_end_to_end.py.
    """
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    def rejected_warmstart(**_kwargs):
        return {
            "candidate": True,
            "applied": False,
            "reference_reason": "canonical_correctness_failed",
            "pristine_ms": 10.0,
            "keep_baseline_ms": 10.0,
            "solution_slug": "prior/solution",
            "speedup": 1.25,
            "program_md_addition": "REFERENCE ONLY",
            "reference_program_md_addition": "REFERENCE ONLY",
        }

    monkeypatch.setattr(cli_module, "kb_warmstart", rejected_warmstart)
    result_json = tmp_path / "rejected-warm-start.json"

    result, workspace = _invoke_forge_loop(
        tmp_path,
        [
            "--return-after-read-KB",
            "--result-json",
            str(result_json),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _result_payload(result.output)
    assert "returned_after_read_kb" not in payload
    assert "returning before iteration 1" not in result.output
    assert not (workspace / "forge_experiments" / "best_result.json").exists()
    # The run searched instead of answering with the rejected kernel.
    assert "run_kwargs" in captured


@pytest.mark.parametrize(
    "failure_stage",
    ["derived-best-view", "checkpoint", "result_json"],
)
def test_warm_start_post_commit_point_failures_are_degraded(
    tmp_path,
    monkeypatch,
    failure_stage,
):
    workspace, kernel, _driver = _initialize_workspace(tmp_path)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "-u"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "kb warm-start: apply prior/solution"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )

    class Tracker:
        @staticmethod
        def set_checkpoint(_experiment_id, _checkpoint):
            if failure_stage == "checkpoint":
                raise OSError("checkpoint unavailable")

    if failure_stage == "derived-best-view":
        original_publish = recovery_module.BestResultPublisher.publish

        def fail_after_commit_point(self, **kwargs):
            original_publish(self, **kwargs)
            raise OSError("report unavailable")

        monkeypatch.setattr(
            recovery_module.BestResultPublisher,
            "publish",
            fail_after_commit_point,
        )
    if failure_stage == "result_json":

        def fail_result_write(_path, _payload):
            raise OSError("result unavailable")

        monkeypatch.setattr(
            recovery_module,
            "atomic_write_json",
            fail_result_write,
        )
    result_json = tmp_path / "forge_cli_result.json"
    result = recovery_module.publish_warm_start_recovery(
        workspace_dir=str(workspace),
        base_commit=base_commit,
        warm={
            "applied": True,
            "pristine_ms": 10.0,
            "keep_baseline_ms": 8.0,
            "mean_case_speedup": 1.25,
            "solution_slug": "prior/solution",
        },
        caller_experiment_id="hyperloom",
        experience_id="experience",
        tracker=Tracker(),
        result_json=str(result_json),
    )

    assert result is not None
    assert result["persistence_degraded"] is True
    assert failure_stage.replace("_", "-") in result["persistence_errors"][0]
    assert (workspace / "forge_experiments" / "best_result.json").is_file()
    if failure_stage != "result_json":
        sidecar = json.loads(result_json.read_text())
        assert sidecar["persistence_degraded"] is True
    else:
        assert not result_json.exists()


def test_warm_start_publication_failure_rolls_back_and_continues_reference_only(
    tmp_path,
    monkeypatch,
):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    def applied_warmstart(**kwargs):
        workspace = Path(kwargs["workspace_dir"])
        kernel = Path(kwargs["kernel"])
        kernel.write_text("def kernel():\n    return 2\n")
        subprocess.run(["git", "add", "-u"], cwd=workspace, check=True)
        subprocess.run(
            ["git", "commit", "-m", "kb warm-start: apply prior/solution"],
            cwd=workspace,
            check=True,
            capture_output=True,
        )
        return {
            "candidate": True,
            "applied": True,
            "pristine_ms": 10.0,
            "keep_baseline_ms": 8.0,
            "mean_case_speedup": 1.25,
            "solution_slug": "prior/solution",
            "program_md_addition": "APPLIED WARM START",
            "reference_program_md_addition": "REFERENCE ONLY",
        }

    monkeypatch.setattr(cli_module, "kb_warmstart", applied_warmstart)

    def fail_publication(**_kwargs):
        raise OSError("manifest failed")

    monkeypatch.setattr(
        cli_module,
        "publish_warm_start_recovery",
        fail_publication,
    )
    result, workspace = _invoke_forge_loop(tmp_path, [])

    assert result.exit_code == 0, result.output
    campaign = CampaignConfigStore(str(workspace)).load()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == campaign.base_commit
    assert (workspace / "kernel.py").read_text() == "def kernel():\n    return 1\n"
    loop = captured["loops"][0]
    assert loop.ic.baseline_wall_ms == 10.0
    assert loop.ic.publication_baseline_wall_ms is None
    assert "REFERENCE ONLY" in loop.ic.program_md
    assert "APPLIED WARM START" not in loop.ic.program_md
    assert "continuing reference-only" in result.output


def test_warm_start_rollback_failure_exits_cleanly_with_code_two(
    tmp_path,
    monkeypatch,
):
    _install_cli_fakes(monkeypatch, tmp_path)

    def fail_rollback(**_kwargs):
        raise cli_module.WarmStartRollbackError("restore failed")

    monkeypatch.setattr(cli_module, "kb_warmstart", fail_rollback)
    result, _workspace = _invoke_forge_loop(tmp_path, [])

    assert result.exit_code == 2
    assert "warm-start rollback failed" in result.output
    assert "workspace may be inconsistent" in result.output


def _fresh_command(workspace, kernel, driver):
    return [
        "forge-loop",
        "--workspace",
        str(workspace),
        "--kernel",
        str(kernel),
        "--driver",
        str(driver),
        "--max-hours",
        "1",
        "--no-profiling",
        # See _invoke_forge_loop: these tests fake the loop and do not exercise
        # the real task preparer.
        "--no-prepare-task",
    ]


def _driver_integrity_resume(tmp_path, monkeypatch):
    workspace, kernel, driver = _initialize_workspace(tmp_path)
    (workspace / ".gitignore").write_text("driver.py\n")
    subprocess.run(
        ["git", "rm", "--cached", "driver.py"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", ".gitignore"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "ignore ephemeral driver"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    monkeypatch.setenv("GPU_TARGET", "gfx942")
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "triton")
    campaign = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
    )
    CampaignConfigStore(str(workspace)).save(campaign)
    iter_config = IterationConfig(
        kernel_file=str(kernel),
        driver_script=str(driver),
        snr_threshold=campaign.snr_threshold,
        git_branch=campaign.git_branch,
        workspace_dir=str(workspace),
        canonical_driver_sha256=campaign.driver_sha256,
    )
    loop = IterationLoop(
        iter_config,
        SimpleNamespace(),
        config=SimpleNamespace(),
        evolver=SimpleNamespace(),
        resume=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = RunState(
        session_status=SESSION_PAUSED,
        kernel_path="kernel.py",
        task_fingerprint=loop._task_fingerprint(),
        git_branch=campaign.git_branch,
        head_commit=head,
        baseline_case_times={"case": 1.0},
    )
    store = LoopStateStore(str(workspace))
    store.save(state)
    return loop, store, campaign, driver


def test_forge_loop_defaults_to_fresh_and_workspace_campaign_root(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, workspace = _invoke_forge_loop(
        tmp_path,
        [],
    )

    assert result.exit_code == 0, result.output
    loop = captured["loops"][0]
    assert loop.resume is False
    assert not hasattr(loop.ic, "max_iterations")
    assert loop.ic.max_time_hours == 1.0
    assert loop.ic.git_branch == "feature/test-cli"
    assert loop.tracker.dir == workspace / "forge_experiments"
    assert len(captured["warmstarts"]) == 1
    assert captured["run_kwargs"]["workspace_lock_held"] is True
    assert captured["run_kwargs"]["usage"] is not None
    assert captured["run_kwargs"]["orchestration_service"] is captured["orchestration_service"]
    assert captured["run_kwargs"]["analysis_service"] is captured["analysis_service"]
    assert captured["analysis_service_kwargs"]["profiling_enabled"] is False
    assert captured["agent_fn_kwargs"]["profiling_enabled"] is False
    assert captured["orchestration_service_kwargs"]["enable_plan_critic"] is False
    assert captured["lock_held_during_warmstart"] is True
    assert captured["lock_held_during_run"] is True
    assert captured["lock_held_during_finalization"] is True
    assert (workspace / "forge_experiments" / "campaign_config.json").is_file()
    campaign = CampaignConfigStore(str(workspace)).load()
    assert campaign.framework == "unknown"
    assert campaign.driver_sha256 == hashlib.sha256((workspace / campaign.driver_path).read_bytes()).hexdigest()
    assert not (workspace / "forge_experiments" / "result.json").exists()
    payload = _result_payload(result.output)
    assert payload["campaign_id"] == "campaign-1"
    assert payload["session_index"] == 1
    assert payload["segment_index"] == 1
    assert payload["next_iteration"] == 5
    assert payload["best_iteration"] == 4
    assert payload["best_commit"] == "best-commit"
    assert payload["optimization_report"].endswith("optimization_report.md")
    assert payload["optimization_history"].endswith("optimization_history.md")
    assert payload["best_manifest"].endswith("best/manifest.json")
    assert payload["kb_experience"]["read"]["read_reason"] == "not_configured"
    assert payload["kb_experience"]["read"]["read_error"] == ""
    assert captured["kb_experience"][1]["read"] == payload["kb_experience"]["read"]


def test_short_forge_budget_keeps_profiling_disabled(
    tmp_path,
    monkeypatch,
):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        ["--profiling"],
    )

    assert result.exit_code == 0, result.output
    assert captured["run_kwargs"]["analysis_service"] is captured["analysis_service"]
    assert captured["analysis_service_kwargs"]["profiling_enabled"] is False
    assert captured["analysis_service_kwargs"]["timeout_sec"] == 7200
    assert captured["agent_fn_kwargs"]["profiling_enabled"] is False
    assert captured["orchestration_service_kwargs"]["enable_plan_critic"] is False


def test_long_forge_budget_enables_analysis_profiling(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        ["--profiling", "--max-hours", "2.0001"],
    )

    assert result.exit_code == 0, result.output
    assert captured["analysis_service_kwargs"]["profiling_enabled"] is True
    assert captured["agent_fn_kwargs"]["profiling_enabled"] is True
    assert captured["orchestration_service_kwargs"]["enable_plan_critic"] is True


def test_multi_lane_long_horizon_reports_the_critic_it_enabled(
    tmp_path,
    monkeypatch,
):
    """What the banner claims has to be what the round buys.

    A wide round is reviewed like any other, so a banner that still called the
    critic off for multi-lane rounds described a version of the service that no
    longer exists -- and it is the default width that reads it.
    """
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        ["--profiling", "--max-hours", "2.0001", "--lanes", "2"],
    )

    assert result.exit_code == 0, result.output
    assert captured["analysis_service_kwargs"]["profiling_enabled"] is True
    assert captured["agent_fn_kwargs"]["profiling_enabled"] is True
    assert captured["orchestration_service_kwargs"]["enable_plan_critic"] is True
    assert "Plan Critic: enabled (long-horizon, same backend/model)" in result.output


def test_forge_loop_can_disable_all_experience_kb_io(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        ["--no-experience-kb"],
    )

    assert result.exit_code == 0, result.output
    assert captured["warmstarts"] == []
    assert captured["kb_writes"] == []
    payload = _result_payload(result.output)
    assert payload["kb_experience"]["read"]["read_reason"] == "disabled"
    assert payload["kb_experience"]["write"] == {
        "written": False,
        "reason": "disabled",
    }


def test_forge_loop_rejects_return_after_read_when_experience_kb_is_disabled(
    tmp_path,
):
    workspace, kernel, driver = _initialize_workspace(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "forge-loop",
            "--workspace",
            str(workspace),
            "--kernel",
            str(kernel),
            "--driver",
            str(driver),
            "--no-experience-kb",
            "--return-after-read-kb",
        ],
    )

    assert result.exit_code != 0
    assert "cannot be used with --no-experience-kb" in result.output


def test_forge_loop_falls_back_from_unsupported_kernel_backend(tmp_path, monkeypatch):
    _install_cli_fakes(monkeypatch, tmp_path)

    result, workspace = _invoke_forge_loop(
        tmp_path,
        ["--kernel-backend", "tilelang"],
    )

    assert result.exit_code == 0, result.output
    assert "Unknown kernel backend 'tilelang'" in result.output
    assert "falling back to 'flydsl'" in result.output
    assert CampaignConfigStore(str(workspace)).load().kernel_backend == "flydsl"


def test_forge_loop_falls_back_from_unknown_environment_kernel_backend(
    tmp_path,
    monkeypatch,
):
    _install_cli_fakes(monkeypatch, tmp_path)
    monkeypatch.setenv("FORGE_KERNEL_BACKEND", "tilelang")

    result, workspace = _invoke_forge_loop(tmp_path, [])

    assert result.exit_code == 0, result.output
    assert "Unknown kernel backend 'tilelang'" in result.output
    assert "falling back to 'flydsl'" in result.output
    assert CampaignConfigStore(str(workspace)).load().kernel_backend == "flydsl"


def test_forge_loop_persists_deadline_read_status(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    result_json = tmp_path / "forge-result.json"

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        [
            "--deadline-unix",
            str(cli_module.time.time() + 650),
            "--result-json",
            str(result_json),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["warmstarts"] == []
    assert "warm-start skipped: absolute deadline reserve" in result.output
    read = _result_payload(result.output)["kb_experience"]["read"]
    assert read["read_reason"] == "deadline"
    assert read["read_error"] == ""
    assert json.loads(result_json.read_text())["kb_experience"]["read"] == read
    assert captured["kb_experience"][1]["read"] == read


def test_forge_loop_persists_sanitized_read_error(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    result_json = tmp_path / "forge-result.json"
    monkeypatch.setattr(
        cli_module,
        "kb_warmstart",
        lambda **_kwargs: {
            "candidate": False,
            "read_reason": "read_error",
            "read_error": "TimeoutError: service unavailable",
        },
    )

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        ["--result-json", str(result_json)],
    )

    assert result.exit_code == 0, result.output
    read = _result_payload(result.output)["kb_experience"]["read"]
    assert read["read_reason"] == "read_error"
    assert read["read_error"] == "TimeoutError: service unavailable"
    assert json.loads(result_json.read_text())["kb_experience"]["read"] == read
    assert captured["kb_experience"][1]["read"] == read


def test_fresh_cli_roundtrip_infers_direct_framework_before_signature(
    tmp_path,
    monkeypatch,
):
    _install_cli_fakes(monkeypatch, tmp_path)
    workspace, _kernel, driver = _initialize_workspace(tmp_path)
    kernel = workspace / "vllm" / "ops" / "direct.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("import triton\n\n@triton.jit\ndef direct_kernel(x):\n    return x\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add direct kernel"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )

    result = CliRunner().invoke(main, _fresh_command(workspace, kernel, driver))

    assert result.exit_code == 0, result.output
    campaign = CampaignConfigStore(str(workspace)).load()
    assert campaign.framework == "vllm"
    assert campaign.implementation_identity["source_paths"] == ["vllm/ops/direct.py"]


def test_fresh_cli_roundtrip_uses_cross_package_defining_owner(
    tmp_path,
    monkeypatch,
):
    _install_cli_fakes(monkeypatch, tmp_path)
    workspace, _kernel, driver = _initialize_workspace(tmp_path)
    anchor = workspace / "vllm" / "attention" / "entry.py"
    defining = workspace / "aiter" / "ops" / "attention.py"
    anchor.parent.mkdir(parents=True)
    defining.parent.mkdir(parents=True)
    anchor.write_text("def attention_entry(x):\n    return unified_attention_kernel(x)\n")
    defining.write_text("import triton\n\n@triton.jit\ndef unified_attention_kernel(x):\n    return x\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add cross-package kernel"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )

    result = CliRunner().invoke(
        main,
        [
            *_fresh_command(workspace, anchor, driver),
            "--source-files",
            str(defining),
            "--target-functions",
            "unified_attention_kernel",
        ],
    )

    assert result.exit_code == 0, result.output
    campaign = CampaignConfigStore(str(workspace)).load()
    assert campaign.framework == "aiter"
    assert "aiter/ops/attention.py" in campaign.implementation_identity["source_paths"]


def test_fresh_cli_roundtrip_honors_explicit_framework_override(
    tmp_path,
    monkeypatch,
):
    _install_cli_fakes(monkeypatch, tmp_path)
    workspace, _kernel, driver = _initialize_workspace(tmp_path)
    kernel = workspace / "aiter" / "ops" / "explicit.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("import triton\n\n@triton.jit\ndef explicit_kernel(x):\n    return x\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add explicit kernel"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )

    result = CliRunner().invoke(
        main,
        [*_fresh_command(workspace, kernel, driver), "--framework", "vllm"],
    )

    assert result.exit_code == 0, result.output
    campaign = CampaignConfigStore(str(workspace)).load()
    assert campaign.framework == "vllm"


def test_resume_reuses_persisted_framework_without_inference(
    tmp_path,
    monkeypatch,
):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    workspace, kernel, driver = _initialize_workspace(tmp_path)
    campaign = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
        framework="vllm",
    )
    store = CampaignConfigStore(str(workspace))
    store.save(campaign)
    (store.root / "run_state.json").write_text("{}")

    def fail_inference(**_kwargs):
        raise AssertionError("resume must not infer framework")

    monkeypatch.setattr(
        "kernelforge.loop.campaign_config.infer_source_owner_framework",
        fail_inference,
    )
    result = CliRunner().invoke(
        main,
        [
            "forge-loop",
            "--workspace",
            str(workspace),
            "--resume",
            "--max-hours",
            "1",
            "--no-profiling",
            "--no-prepare-task",
        ],
    )

    assert result.exit_code == 0, result.output
    assert store.load().framework == "vllm"
    assert captured["kb_writes"][-1]["framework"] == "vllm"


def test_keep_callback_snapshots_result_and_kb_before_iteration_callback(
    tmp_path,
    monkeypatch,
):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    result_json = tmp_path / "forge_cli_result.json"
    result, _workspace = _invoke_forge_loop(
        tmp_path,
        [
            "--result-json",
            str(result_json),
            "--experiment-id",
            "hyperloom",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "on_iteration" not in captured["run_kwargs"]

    # Discard finalization effects and invoke the exact callback that runner calls
    # synchronously after a durable KEEP, before post-KEEP profiling.
    result_json.unlink()
    captured["kb_writes"].clear()
    captured["loops"][0].experiment = None
    callback = captured["run_kwargs"]["on_best_committed"]
    callback(
        SimpleNamespace(
            kept=True,
            commit_hash="best-commit",
            wall_ms=0.8,
            mean_case_speedup=1.25,
            iteration=4,
            validation_passed=True,
            validation_summary="passed",
            snr_db=40.0,
        )
    )

    snapshot = json.loads(result_json.read_text())
    assert snapshot["best_commit"] == "best-commit"
    assert snapshot["best_ms"] == 0.8
    assert snapshot["search_start_mean_case_speedup"] == 1.0
    assert not captured["kb_writes"]
    remote_callback = captured["run_kwargs"]["on_best_ready"]
    remote_callback(SimpleNamespace(kept=True))
    assert captured["kb_writes"]
    assert captured["kb_writes"][-1]["llm_summary"] is False
    checkpoint = captured["checkpoints"]["hyperloom"]
    assert checkpoint["best_commit"] == "best-commit"
    assert checkpoint["search_start_mean_case_speedup"] == 1.0


def test_forge_loop_materializes_the_caller_owned_recovery_record(
    tmp_path,
    monkeypatch,
):
    """--experiment-id must exist as a record before the first KEEP.

    An external orchestrator that enforces its own wall clock hard-kills this
    process and then salvages the last validated best from
    <experiments-dir>/<experiment-id>.json. Campaign segments deliberately get
    fresh internal IDs (so the resume parent/child chain stays intact), so the
    caller-owned record is a separate channel the CLI must create up front --
    otherwise set_checkpoint raises FileNotFoundError and the salvage silently
    finds nothing.
    """
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        ["--experiment-id", "hyperloom"],
    )

    assert result.exit_code == 0, result.output
    assert "hyperloom" in captured["experiments"]
    # The internal segment identity stays distinct from the caller-owned one.
    assert _result_payload(result.output)["experiment_id"] == "segment-1"


def test_forge_loop_resume_routes_new_budget_without_warmstart(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, workspace = _invoke_forge_loop(
        tmp_path,
        ["--resume", "--max-hours", "2"],
        existing_config=True,
        include_campaign_inputs=False,
    )

    assert result.exit_code == 0, result.output
    loop = captured["loops"][0]
    assert loop.resume is True
    assert not hasattr(loop.ic, "max_iterations")
    assert loop.ic.max_time_hours == 2.0
    assert loop.tracker.dir == workspace / "forge_experiments"
    assert captured["warmstarts"] == []
    payload = _result_payload(result.output)
    assert payload["experiment_id"] == "segment-2"
    assert payload["session_index"] == 2
    assert payload["segment_index"] == 2
    assert payload["next_iteration"] == 9
    assert payload["kb_experience"]["read"]["read_reason"] == "resume"
    assert payload["kb_experience"]["read"]["read_error"] == ""
    assert captured["kb_experience"][1]["read"] == payload["kb_experience"]["read"]


def test_forge_loop_resume_adds_existing_kb_reference_pointer(
    tmp_path,
    monkeypatch,
):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    workspace, kernel, driver = _initialize_workspace(tmp_path)
    campaign = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=None,
    )
    store = CampaignConfigStore(str(workspace))
    store.save(campaign)
    (store.root / "run_state.json").write_text("{}")
    references = store.root / "kb_references"
    reference = references / "sets" / "generation-a" / "reference_01.md"
    reference.parent.mkdir(parents=True)
    reference.write_text("historical solution\n")
    (references / "index.md").write_text(
        "# KernelForge KB references\n\n"
        "- Rank 1: `sets/generation-a/reference_01.md` | "
        "solution `solution/fast` | "
        "speedup 3x | status `applied`\n"
    )

    result = CliRunner().invoke(
        main,
        [
            "forge-loop",
            "--workspace",
            str(workspace),
            "--resume",
            "--max-hours",
            "2",
            "--no-profiling",
            "--no-prepare-task",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["warmstarts"] == []
    program_md = captured["loops"][0].ic.program_md
    assert "forge_experiments/kb_references/index.md" in program_md
    assert "Rank 1 solution `solution/fast` is already applied" in program_md


def test_config_only_failure_retries_fresh_and_releases_setup_lock(
    tmp_path,
    monkeypatch,
):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    workspace, kernel, driver = _initialize_workspace(tmp_path)
    command = _fresh_command(workspace, kernel, driver)
    monkeypatch.setenv("FORGE_PROFILE_TIMEOUT_SEC", "invalid")

    failed = CliRunner().invoke(main, command)

    assert failed.exit_code != 0
    assert "FORGE_PROFILE_TIMEOUT_SEC must be an integer" in failed.output
    store = CampaignConfigStore(str(workspace))
    pending = store.load()
    assert not (store.root / "run_state.json").exists()
    with LoopStateStore(str(workspace)).workspace_lock():
        pass

    monkeypatch.setenv("FORGE_PROFILE_TIMEOUT_SEC", "1800")
    retried = CliRunner().invoke(main, command)

    assert retried.exit_code == 0, retried.output
    assert store.load() == pending
    assert len(captured["loops"]) == 1


def test_config_only_retry_rejects_mismatching_inputs(tmp_path, monkeypatch):
    _install_cli_fakes(monkeypatch, tmp_path)
    workspace, kernel, driver = _initialize_workspace(tmp_path)
    monkeypatch.setenv("FORGE_PROFILE_TIMEOUT_SEC", "invalid")
    initial = CliRunner().invoke(
        main,
        _fresh_command(workspace, kernel, driver),
    )
    store = CampaignConfigStore(str(workspace))
    pending = store.load()
    monkeypatch.setenv("FORGE_PROFILE_TIMEOUT_SEC", "1800")

    mismatched = CliRunner().invoke(
        main,
        _fresh_command(workspace, kernel, driver) + ["--snr-threshold", "31"],
    )

    assert initial.exit_code != 0
    assert mismatched.exit_code != 0
    assert "pending campaign configuration does not match" in mismatched.output
    assert store.load() == pending


def test_config_only_retry_rejects_unrelated_clean_head_movement(
    tmp_path,
    monkeypatch,
):
    _install_cli_fakes(monkeypatch, tmp_path)
    workspace, kernel, driver = _initialize_workspace(tmp_path)
    command = _fresh_command(workspace, kernel, driver)
    monkeypatch.setenv("FORGE_PROFILE_TIMEOUT_SEC", "invalid")
    initial = CliRunner().invoke(main, command)
    store = CampaignConfigStore(str(workspace))
    pending = store.load()
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "unrelated clean movement"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    monkeypatch.setenv("FORGE_PROFILE_TIMEOUT_SEC", "1800")

    retried = CliRunner().invoke(main, command)

    assert initial.exit_code != 0
    assert retried.exit_code != 0
    assert "pending campaign HEAD mismatch" in retried.output
    assert store.load() == pending


def test_config_only_retry_accepts_single_kb_warm_start_child(
    tmp_path,
    monkeypatch,
):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    workspace, kernel, driver = _initialize_workspace(tmp_path)
    command = _fresh_command(workspace, kernel, driver)
    monkeypatch.setenv("FORGE_PROFILE_TIMEOUT_SEC", "invalid")
    initial = CliRunner().invoke(main, command)
    store = CampaignConfigStore(str(workspace))
    pending = store.load()
    kernel.write_text("def kernel():\n    return 2\n")
    subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "kb warm-start: apply verified-solution"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    monkeypatch.setenv("FORGE_PROFILE_TIMEOUT_SEC", "1800")

    retried = CliRunner().invoke(main, command)

    assert initial.exit_code != 0
    assert retried.exit_code == 0, retried.output
    assert store.load() == pending
    assert len(captured["loops"]) == 1


def test_resume_accepts_canonical_driver_digest(tmp_path, monkeypatch):
    loop, _store, campaign, driver = _driver_integrity_resume(
        tmp_path,
        monkeypatch,
    )

    state = loop.validate_resume_preflight()
    loop.run_state = state

    assert campaign.driver_sha256 == hashlib.sha256(driver.read_bytes()).hexdigest()


def test_resume_rejects_unknown_driver_mutation(tmp_path, monkeypatch):
    loop, _store, _campaign, driver = _driver_integrity_resume(
        tmp_path,
        monkeypatch,
    )
    driver.write_text("pass\n# partial unknown mutation\n")

    with pytest.raises(ValueError, match="driver integrity"):
        loop.validate_resume_preflight()


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_resume_rejects_missing_or_changed_program_context(
    tmp_path,
    monkeypatch,
    mutation,
):
    _install_cli_fakes(monkeypatch, tmp_path)
    workspace, kernel, driver = _initialize_workspace(tmp_path)
    source_program = tmp_path / "program.md"
    source_program.write_text("# Immutable task context\n")
    campaign = create_campaign_config(
        workspace_dir=str(workspace),
        kernel=str(kernel),
        driver=str(driver),
        source_files=[],
        program_md_file=str(source_program),
    )
    store = CampaignConfigStore(str(workspace))
    store.save(campaign, program_md=source_program.read_text())
    (store.root / "run_state.json").write_text("{}")
    if mutation == "missing":
        store.program_path.unlink()
    else:
        store.program_path.write_text("# Mutated task context\n")

    result = CliRunner().invoke(
        main,
        [
            "forge-loop",
            "--workspace",
            str(workspace),
            "--resume",
            "--max-hours",
            "1",
            "--no-profiling",
        ],
    )

    assert result.exit_code != 0
    assert "campaign program context" in result.output


def test_resume_rejects_repeated_campaign_inputs(tmp_path, monkeypatch):
    _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        ["--resume"],
        existing_config=True,
        include_campaign_inputs=True,
    )

    assert result.exit_code != 0
    assert "already has immutable configuration" in result.output


def test_forge_loop_rejects_an_unknown_option(tmp_path, monkeypatch):
    """An option this version does not declare is fatal, not silently dropped.

    forge-loop used to tolerate unknown options: it dropped them, warned on
    stderr and recorded them on the result, so that a consumer shipping from a
    separate repository could stay ahead of the installed producer. Vendoring
    put producer and consumer in one tree and one wheel, so that skew can no
    longer happen -- and the tolerance only ever silently absorbed typos, which
    is how seven shipped examples ran an inferred kernel backend for a while.

    ``--max-hour`` is used deliberately: it is a typo of ``--max-hours``, which
    tolerance could not distinguish from an option that did not exist yet. It
    must now cost an exit code before any work starts, not an hour of GPU time
    spent on the ``--max-hours`` default.
    """
    _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        ["--resume", "--max-hour", "2"],
        existing_config=True,
        include_campaign_inputs=False,
    )

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_forge_loop_still_rejects_an_invalid_value_on_a_declared_option(
    tmp_path,
    monkeypatch,
):
    """Tolerating unknown options must not loosen the options that do exist."""
    _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        ["--resume", "--max-hours", "0.1"],
        existing_config=True,
        include_campaign_inputs=False,
    )

    assert result.exit_code != 0
    assert "--max-hours" in result.output


def test_fresh_campaign_contains_no_shape_metadata(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    workspace, kernel, driver = _initialize_workspace(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            "forge-loop",
            "--workspace",
            str(workspace),
            "--kernel",
            str(kernel),
            "--driver",
            str(driver),
            "--max-hours",
            "1",
            "--no-profiling",
            "--no-prepare-task",
        ],
    )

    assert result.exit_code == 0, result.output
    campaign = CampaignConfigStore(str(workspace)).load()
    assert "shapes" not in campaign.to_dict()
    assert "workload_key" not in campaign.to_dict()
    assert captured["loops"]
    assert captured["agent_fn_kwargs"]["validation_timeout_sec"] == 1800
    assert captured["agent_fn_kwargs"]["bench_timeout_sec"] == 300


def test_a_retired_iteration_cap_is_rejected(tmp_path, monkeypatch):
    """A retired option now costs an exit code instead of being absorbed.

    It used to be accepted and ignored so that a caller still passing it would
    not lose the run. Nothing passes it: this repo is the only consumer, and its
    own argv tests assert ``--max-iters`` is never sent (see
    ``test_forge_collective`` and ``test_forge_long_horizon_cli``). What the
    tolerance actually bought was a run that silently ignored what the caller
    asked for, so it is refused at parse time instead.
    """
    _install_cli_fakes(monkeypatch, tmp_path)
    workspace, kernel, driver = _initialize_workspace(tmp_path)

    result = CliRunner().invoke(
        main,
        [
            *_fresh_command(workspace, kernel, driver),
            "--max-iters",
            "1",
        ],
    )

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_fresh_cli_rejects_existing_state_before_git_or_warmstart(tmp_path, monkeypatch):
    captured = _install_cli_fakes(monkeypatch, tmp_path)

    result, _workspace = _invoke_forge_loop(
        tmp_path,
        [],
        existing_state=True,
    )

    assert result.exit_code != 0
    assert "pass --resume" in result.output
    assert captured["warmstarts"] == []
    assert captured["loops"] == []


def test_experience_ledger_reloads_structured_history(tmp_path):
    ledger = ExperienceLedger(str(tmp_path))
    ledger.record_iteration(
        iteration=7,
        outcome="BUILD_FAILED",
        diff_summary="changed vector width",
        error_text="Invalid cast! copy atom width mismatch",
    )

    reloaded = ExperienceLedger(str(tmp_path))

    root = tmp_path / "forge_experiments"
    assert reloaded.entries[0].iteration == 7
    assert "Invalid cast" in reloaded.entries[0].error_sig
    assert any("copy-atom width" in item for item in reloaded.constraints)
    assert "iter 7: BUILD_FAILED" in reloaded.render_for_prompt()
    assert (root / "experience.jsonl").is_file()
    assert (root / "forge_experience.md").is_file()
    assert not (tmp_path / "forge_experience.md").exists()


@pytest.mark.parametrize(
    "bad_line",
    [
        "{malformed-json",
        '{"iteration":' + "9" * 5000 + ',"outcome":"INVALID"}',
        json.dumps(["not", "an", "entry"]),
        json.dumps({"iteration": 2}),
        json.dumps({"iteration": True, "outcome": "INVALID"}),
        json.dumps({"iteration": 2, "outcome": 42}),
        json.dumps({"iteration": 2, "outcome": "INVALID", "error_sig": []}),
    ],
    ids=[
        "malformed-json",
        "oversized-integer",
        "non-object",
        "missing-required-field",
        "invalid-iteration-type",
        "invalid-outcome-type",
        "invalid-optional-field-type",
    ],
)
def test_experience_ledger_skips_bad_rows_without_losing_valid_suffix(
    tmp_path,
    bad_line,
):
    root = tmp_path / "forge_experiments"
    root.mkdir()
    valid_entries = [
        {
            "iteration": 1,
            "outcome": "KEEP",
            "diff_summary": "first valid change",
            "error_sig": "",
        },
        {
            "iteration": 3,
            "outcome": "KEEP",
            "diff_summary": "later valid change",
            "error_sig": "",
        },
    ]
    jsonl_path = root / "experience.jsonl"
    jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(valid_entries[0]),
                bad_line,
                json.dumps(valid_entries[1]),
            ]
        )
        + "\n"
    )

    ledger = ExperienceLedger(str(tmp_path))
    assert [entry.iteration for entry in ledger.entries] == [1, 3]

    ledger.record_iteration(iteration=4, outcome="KEEP")

    persisted = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    assert [entry["iteration"] for entry in persisted] == [1, 3, 4]
    reloaded = ExperienceLedger(str(tmp_path))
    assert [entry.iteration for entry in reloaded.entries] == [1, 3, 4]


def test_fresh_campaign_captures_post_prep_driver_digest_and_base(tmp_path, monkeypatch):
    """Review #1: a fresh campaign with task preparation enabled must persist the
    driver digest and pristine base_commit captured AFTER prep runs.

    Prep may repair the driver (new content -> new sha256) and commit its
    scaffolding (new HEAD). If the campaign froze the pre-prep snapshot, the
    driver-integrity gate would reject the repaired driver and the prep commit
    would leak into the solution diff. The deferred save must instead anchor the
    repaired driver's digest and the post-prep HEAD.
    """
    captured = _install_cli_fakes(monkeypatch, tmp_path)
    workspace, kernel, driver = _initialize_workspace(tmp_path)

    pre_prep_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pre_prep_digest = hashlib.sha256(driver.read_bytes()).hexdigest()

    repaired_body = "pass\n# repaired by prep\n"

    def fake_preflight(**_kwargs):
        return SimpleNamespace(ok=False, profile_ok=False, summary=lambda: "needs prep")

    def fake_prepare(**kwargs):
        # Simulate the prep agent: repair the driver and commit scaffolding into
        # pristine, advancing HEAD and changing the driver digest.
        Path(kwargs["driver"]).write_text(repaired_body)
        (workspace / "task_helper.py").write_text("# scaffolding\n")
        subprocess.run(
            ["git", "add", "-A", "--", ".", ":(exclude)forge_experiments"],
            cwd=workspace,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "prep"],
            cwd=workspace,
            check=True,
            capture_output=True,
        )
        return SimpleNamespace(
            ok=True,
            attempts=1,
            wrote_files=["driver.py", "task_helper.py"],
            created_files=["task_helper.py"],
            rolled_back=False,
            final_preflight=SimpleNamespace(profile_ok=True),
            message="prepared",
            audit_dir="",
        )

    monkeypatch.setattr("kernelforge.loop.task_preparer.preflight_task", fake_preflight)
    monkeypatch.setattr("kernelforge.loop.task_preparer.prepare_task_sync", fake_prepare)

    result = CliRunner().invoke(
        main,
        [
            "forge-loop",
            "--workspace",
            str(workspace),
            "--kernel",
            str(kernel),
            "--driver",
            str(driver),
            "--max-hours",
            "1",
            "--no-profiling",
            "--prepare-task",
        ],
    )
    assert result.exit_code == 0, result.output

    post_prep_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    post_prep_digest = hashlib.sha256(driver.read_bytes()).hexdigest()

    # Sanity: prep actually changed both the driver and HEAD.
    assert post_prep_head != pre_prep_head
    assert post_prep_digest != pre_prep_digest

    campaign = CampaignConfigStore(str(workspace)).load()
    # The persisted campaign must reflect the POST-prep state, not the stale
    # pre-prep snapshot.
    assert campaign.driver_sha256 == post_prep_digest
    assert campaign.base_commit == post_prep_head

    # And the loop must validate against the same repaired driver / base.
    loop = captured["loops"][0]
    assert loop.ic.canonical_driver_sha256 == post_prep_digest
    assert loop.ic.campaign_base_commit == post_prep_head
