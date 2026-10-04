"""Regression tests for the prep -> baseline handover."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelforge.loop import task_preparer

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


# A driver that loads its case table from the invocation specification at RUNTIME -- exactly what the recorded agent
# wrote, and what the contract asks for ("case definitions come from the task's real harness/config").
_AUTHORED_DRIVER = '''\
"""Measurement driver whose case table comes from the task specification."""
import argparse
import json
from pathlib import Path

_SPEC = Path(__file__).resolve().parent / {spec_rel!r}


def _case_ids():
    payload = json.loads(_SPEC.read_text(encoding="utf-8"))
    selectors = payload["tests"]["driver_contract"]["case_selectors"]
    return [selector["CASE_ID"] for selector in selectors]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args, _ = parser.parse_known_args()
    case_ids = _case_ids()
    if args.profile_run:
        return 0
    if args.bench_mode:
        for index, case_id in enumerate(case_ids, start=1):
            print(f"case_ms: {{case_id}} {{float(index):.6f}}")
        print("mean_ms: 1.500000")
        return 0
    print("SNR: 62.13 dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

_SPEC_PAYLOAD = {
    "schema_version": 1,
    "kernel": {"name": "aiter_hipb_mm"},
    "invocation": {"launcher_locator": "aiter/ops/gemm.py: hipb_mm"},
    "tests": {
        "driver_contract": {
            "case_selectors": [
                {"CASE_ID": "case_001", "M": 3118, "N": 5120, "K": 34816},
                {"CASE_ID": "case_002", "M": 3118, "N": 17408, "K": 5120},
            ],
        },
    },
}


def _init_repo(root: Path) -> None:
    task_preparer._git(root, "init", "-q")
    task_preparer._git(root, "config", "user.email", "t@t")
    task_preparer._git(root, "config", "user.name", "t")
    task_preparer._git(root, "add", "-A")
    task_preparer._git(root, "commit", "-q", "--allow-empty", "-m", "task baseline")


def test_stage_preparation_changes_excludes_preexisting_untracked_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tracked = workspace / "tracked.py"
    tracked.write_text("old\n", encoding="utf-8")
    _init_repo(workspace)
    preexisting = workspace / "runtime.cache"
    preexisting.write_text("cache\n", encoding="utf-8")
    pre_untracked = task_preparer._git_untracked(workspace)

    tracked.write_text("new\n", encoding="utf-8")
    (workspace / "driver.py").write_text("driver\n", encoding="utf-8")
    code, output = task_preparer._stage_preparation_changes(workspace, pre_untracked)

    assert code == 0, output
    staged_code, staged = task_preparer._git(workspace, "diff", "--cached", "--name-only")
    assert staged_code == 0
    assert staged.splitlines() == ["driver.py", "tracked.py"]
    assert task_preparer._git_untracked(workspace) == {"runtime.cache"}


def _runtime_spec_path(prompt: str) -> str:
    """The path the prompt advertises as the specification's runtime location."""
    match = re.search(r"`\./([^`]+)` is DURABLE", prompt) or re.search(r"Read on `\./([^`]+)`", prompt)
    assert match, prompt
    return match.group(1)


def _run_driver(driver: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(driver), *args],
        capture_output=True,
        text=True,
    )


def _prepare_with_spec_reading_driver(
    tmp_path, monkeypatch, gitignore: str = "", extra_patch=None, **prepare_kwargs
) -> dict:
    """Run a full preparation whose agent reads the spec at runtime."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("def hipb_mm(a, b):\n    return a @ b\n", encoding="utf-8")
    driver = workspace / ".forge_driver_b0xdwz7g.py"
    if gitignore:
        (workspace / ".gitignore").write_text(gitignore, encoding="utf-8")
    _init_repo(workspace)

    source_spec = tmp_path / "invocation_spec_aiter_hipb_mm.json"
    source_spec.write_text(json.dumps(_SPEC_PAYLOAD), encoding="utf-8")
    captured: dict = {}

    def fake_materialize_reference(target_workspace):
        """Stand in for the shipped examples bundle (kept small and local)."""
        ref_dir = Path(target_workspace) / task_preparer.REFERENCE_SUBDIR
        example = ref_dir / "example-forge-loop"
        example.mkdir(parents=True, exist_ok=True)
        (ref_dir / "README.md").write_text("contract\n", encoding="utf-8")
        (example / "driver.py").write_text("# reference driver\n", encoding="utf-8")
        return ref_dir

    async def fake_agent(**kwargs):
        prompt = kwargs["prompt"]
        captured["prompt"] = prompt
        spec_rel = _runtime_spec_path(prompt)
        captured["spec_rel"] = spec_rel
        driver.write_text(
            _AUTHORED_DRIVER.format(spec_rel=spec_rel),
            encoding="utf-8",
        )
        return "prepared"

    async def executing_preflight(driver_script, *_args, **_kwargs):
        """Validate the driver the way the loop does: by running it."""
        bench = _run_driver(Path(driver_script), "--bench-mode", "--warmup", "1", "--iters", "1")
        captured["preflight_bench"] = bench
        case_ids = re.findall(r"case_ms:\s*(\S+)", bench.stdout)
        ok = bench.returncode == 0 and len(case_ids) == 2
        return task_preparer.PreflightResult(
            ok=ok,
            correctness_ok=ok,
            bench_ok=ok,
            graph_ok=True,
            profile_ok=True,
            reasons=[] if ok else [f"bench exited {bench.returncode}"],
            details={"bench": {"case_count": len(case_ids)}},
        )

    monkeypatch.setattr(task_preparer, "_materialize_reference", fake_materialize_reference)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", fake_agent)
    monkeypatch.setattr(task_preparer, "_preflight_async", executing_preflight)
    monkeypatch.setattr(task_preparer, "PREPARE_MAX_ATTEMPTS", 1)
    if extra_patch is not None:
        extra_patch(monkeypatch)

    prepare_kwargs.setdefault("expected_case_ids", ["case_001", "case_002"])
    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(
                model="test-model",
                experiments_dir=tmp_path / "experiments",
            ),
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            program_md="# Task",
            target_functions=["hipb_mm"],
            source_files=[str(kernel)],
            invocation_spec_file=str(source_spec),
            read_only_files=[str(source_spec)],
            **prepare_kwargs,
        )
    )
    captured["result"] = result
    captured["workspace"] = workspace
    captured["driver"] = driver
    captured["source_spec"] = source_spec
    return captured


def test_prepared_driver_still_runs_after_preparation(tmp_path, monkeypatch):
    """The runtime inputs preparation advertises must survive preparation."""
    prepared = _prepare_with_spec_reading_driver(tmp_path, monkeypatch)
    result = prepared["result"]

    assert result.ok is True, result.message
    bench = _run_driver(prepared["driver"], "--bench-mode", "--warmup", "1", "--iters", "1")
    assert bench.returncode == 0, bench.stdout + bench.stderr
    assert re.findall(r"case_ms:\s*(\S+)", bench.stdout) == ["case_001", "case_002"]


def test_prepared_runtime_inputs_are_committed_as_pristine(tmp_path, monkeypatch):
    """Everything the driver reads at runtime belongs to the pristine commit."""
    prepared = _prepare_with_spec_reading_driver(tmp_path, monkeypatch)
    workspace = prepared["workspace"]
    spec_rel = prepared["spec_rel"]

    code, tracked = task_preparer._git(workspace, "ls-files")
    assert code == 0, tracked
    tracked_paths = {line.strip() for line in tracked.splitlines() if line.strip()}
    assert prepared["driver"].name in tracked_paths
    assert spec_rel in tracked_paths


def test_authoring_scaffolding_is_gone_before_preflight_validates(tmp_path, monkeypatch):
    """Preflight must judge the driver without the authoring-only bundle."""
    prepared = _prepare_with_spec_reading_driver(tmp_path, monkeypatch)
    workspace = prepared["workspace"]

    assert not (workspace / task_preparer.REFERENCE_SUBDIR).exists()
    assert not prepared["spec_rel"].startswith(task_preparer.REFERENCE_SUBDIR)


def test_prompt_marks_the_reference_bundle_as_temporary(tmp_path, monkeypatch):
    """The prompt must not advertise deleted scaffolding as a runtime input."""
    prepared = _prepare_with_spec_reading_driver(tmp_path, monkeypatch)
    prompt = prepared["prompt"]

    assert task_preparer.REFERENCE_SUBDIR in prompt
    assert "never read it at runtime" in prompt
    assert "DURABLE" in prompt


def test_undurable_spec_fails_loudly_instead_of_committing_a_broken_driver(tmp_path, monkeypatch):
    """A runtime input that cannot join the pristine commit must abort prep."""
    prepared = _prepare_with_spec_reading_driver(tmp_path, monkeypatch, gitignore="invocation_spec_*.json\n")
    result = prepared["result"]
    workspace = prepared["workspace"]

    assert result.ok is False
    assert result.rolled_back is True
    assert "would not be durable" in result.message
    assert "git ignore rules" in result.message
    code, log = task_preparer._git(workspace, "log", "--oneline")
    assert code == 0, log
    assert len([line for line in log.splitlines() if line.strip()]) == 1


def test_surviving_scaffolding_aborts_preparation_instead_of_being_committed(tmp_path, monkeypatch):
    """A removal that only half worked breaks the invariant in both directions."""

    def leave_the_bundle_behind(mp):
        real_rmtree = task_preparer._safe_rmtree

        def keep_reference_bundle(path):
            if path is not None and path.name == task_preparer.REFERENCE_SUBDIR:
                return
            real_rmtree(path)

        mp.setattr(task_preparer, "_safe_rmtree", keep_reference_bundle)

    prepared = _prepare_with_spec_reading_driver(tmp_path, monkeypatch, extra_patch=leave_the_bundle_behind)
    result = prepared["result"]
    workspace = prepared["workspace"]

    assert result.ok is False
    assert "could not retire the authoring reference bundle" in result.message
    assert "pristine commit" in result.message
    code, log = task_preparer._git(workspace, "log", "--oneline")
    assert code == 0, log
    assert len([line for line in log.splitlines() if line.strip()]) == 1
    code, tracked = task_preparer._git(workspace, "ls-files")
    assert code == 0, tracked
    assert task_preparer.REFERENCE_SUBDIR not in tracked


def test_a_declared_suite_still_gates_preflight_when_the_spec_cannot_be_staged(tmp_path, monkeypatch, caplog):
    """The caller's list is the one list, so it survives a materialization failure."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("def hipb_mm(a, b):\n    return a @ b\n", encoding="utf-8")
    driver = workspace / "driver.py"
    _init_repo(workspace)
    source_spec = tmp_path / "invocation_spec_aiter_hipb_mm.json"
    source_spec.write_text(json.dumps(_SPEC_PAYLOAD), encoding="utf-8")
    captured: dict = {}

    async def fake_agent(**_kwargs):
        driver.write_text("# prepared driver\n", encoding="utf-8")
        return "prepared"

    async def recording_preflight(*_args, **kwargs):
        captured["expected_case_ids"] = kwargs.get("expected_case_ids")
        return task_preparer.PreflightResult(
            ok=False,
            correctness_ok=False,
            bench_ok=False,
            reasons=["bench produced no timing"],
        )

    monkeypatch.setattr(
        task_preparer,
        "_materialize_invocation_spec",
        lambda *_args, **_kwargs: (None, ""),
    )
    monkeypatch.setattr(task_preparer, "_materialize_reference", lambda _workspace: None)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", fake_agent)
    monkeypatch.setattr(task_preparer, "_preflight_async", recording_preflight)
    monkeypatch.setattr(task_preparer, "PREPARE_MAX_ATTEMPTS", 1)

    with caplog.at_level("WARNING", logger="kernelforge.loop.task_preparer"):
        asyncio.run(
            task_preparer.prepare_task(
                config=SimpleNamespace(
                    model="test-model",
                    experiments_dir=tmp_path / "experiments",
                ),
                workspace_dir=str(workspace),
                kernel=str(kernel),
                driver=str(driver),
                program_md="# Task",
                target_functions=["hipb_mm"],
                source_files=[str(kernel)],
                invocation_spec_file=str(source_spec),
                expected_case_ids=["case_001", "case_002"],
            )
        )

    assert captured["expected_case_ids"] == ["case_001", "case_002"]
    assert "could not materialize the invocation specification" in caplog.text


def test_a_failed_rematerialization_stops_advertising_the_absent_bundle(tmp_path, monkeypatch, caplog):
    """Attempt 2 must not be told to Read a contract that is no longer there."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("def hipb_mm(a, b):\n    return a @ b\n", encoding="utf-8")
    driver = workspace / "driver.py"
    _init_repo(workspace)
    prompts: list[str] = []
    materializations: list[int] = []

    def flaky_materialize_reference(target_workspace):
        materializations.append(1)
        if len(materializations) > 1:
            return None
        ref_dir = Path(target_workspace) / task_preparer.REFERENCE_SUBDIR
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "README.md").write_text("contract\n", encoding="utf-8")
        return ref_dir

    async def fake_agent(**kwargs):
        prompts.append(kwargs["prompt"])
        driver.write_text(f"# attempt {len(prompts)}\n", encoding="utf-8")
        return "prepared"

    async def failing_preflight(*_args, **_kwargs):
        return task_preparer.PreflightResult(
            ok=False,
            correctness_ok=False,
            bench_ok=False,
            reasons=["bench produced no timing"],
        )

    monkeypatch.setattr(task_preparer, "_materialize_reference", flaky_materialize_reference)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", fake_agent)
    monkeypatch.setattr(task_preparer, "_preflight_async", failing_preflight)
    monkeypatch.setattr(task_preparer, "PREPARE_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(task_preparer, "PREPARE_MIN_RETRY_SEC", 0)

    with caplog.at_level("WARNING", logger="kernelforge.loop.task_preparer"):
        asyncio.run(
            task_preparer.prepare_task(
                config=SimpleNamespace(
                    model="test-model",
                    experiments_dir=tmp_path / "experiments",
                ),
                workspace_dir=str(workspace),
                kernel=str(kernel),
                driver=str(driver),
                program_md="# Task",
                target_functions=["hipb_mm"],
                source_files=[str(kernel)],
            )
        )

    assert len(prompts) == 2, prompts
    assert f"{task_preparer.REFERENCE_SUBDIR}/README.md" in prompts[0]
    assert f"{task_preparer.REFERENCE_SUBDIR}/README.md" not in prompts[1]
    assert "No reference files were available" in prompts[1]
    assert "could not re-materialize the authoring reference bundle" in caplog.text


def test_git_indexed_separates_not_indexed_from_could_not_determine(tmp_path):
    """\"Not staged\" and \"never asked\" send the operator to different places."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tracked = workspace / "driver.py"
    tracked.write_text("# driver\n", encoding="utf-8")
    _init_repo(workspace)
    task_preparer._git(workspace, "add", "driver.py")
    untracked = workspace / "invocation_spec_gemm.json"
    untracked.write_text("{}\n", encoding="utf-8")

    assert task_preparer._git_indexed(workspace, tracked) is True
    assert task_preparer._git_indexed(workspace, untracked) is False

    outside = tmp_path / "elsewhere" / "spec.json"
    outside.parent.mkdir()
    outside.write_text("{}\n", encoding="utf-8")
    assert task_preparer._git_indexed(workspace, outside) is None

    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    stray = bare / "driver.py"
    stray.write_text("# driver\n", encoding="utf-8")
    assert task_preparer._git_indexed(bare, stray) is None


def test_a_protected_source_that_cannot_be_read_stops_preparation(tmp_path, monkeypatch):
    """The snapshot is the rollback's only record of a source that git does not track."""
    kernel = tmp_path / "kernel.py"
    kernel.write_text("def kernel(x):\n    return x\n", encoding="utf-8")

    def refuse(self, *args, **kwargs):
        raise PermissionError(f"cannot read {self}")

    monkeypatch.setattr(Path, "read_bytes", refuse)

    with pytest.raises(PermissionError):
        task_preparer._snapshot([kernel])


def test_rollback_restores_originals_and_deletes_only_what_was_absent(tmp_path):
    """Only a path the snapshot found absent may be deleted; everything else is rewritten."""
    kernel = tmp_path / "kernel.py"
    kernel.write_text("original\n", encoding="utf-8")
    created_later = tmp_path / "helper.py"

    snapshot = task_preparer._snapshot([kernel, created_later])
    kernel.write_text("agent edit\n", encoding="utf-8")
    created_later.write_text("agent invention\n", encoding="utf-8")

    assert task_preparer._restore(snapshot) == set()
    assert kernel.read_text(encoding="utf-8") == "original\n"
    assert not created_later.exists()


def test_a_path_that_became_a_directory_is_reported_not_skipped(tmp_path):
    """An absent path the agent turned into a tree is still a path the rollback did not undo."""
    became_a_tree = tmp_path / "helper.py"

    snapshot = task_preparer._snapshot([became_a_tree])
    (became_a_tree / "nested").mkdir(parents=True)

    assert task_preparer._restore(snapshot) == {became_a_tree}


def test_a_source_the_rollback_cannot_put_back_fails_the_preparation(tmp_path, monkeypatch):
    """The other protected sources still come back, and the one that did not is named in the result."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _init_repo(workspace)
    # Untracked, so the byte snapshot is the rollback's only record of them.
    kernel = workspace / "kernel.py"
    kernel.write_text("def hipb_mm(a, b):\n    return a @ b\n", encoding="utf-8")
    helper = workspace / "helper.py"
    helper.write_text("SCALE = 1\n", encoding="utf-8")
    driver = workspace / "driver.py"
    protected = {kernel, helper}
    originals = {p: p.read_text(encoding="utf-8") for p in protected}

    real_write_bytes = Path.write_bytes
    doomed: list[Path] = []

    def refuse_the_first_source_the_rollback_reaches(self, data):
        if self in protected and not doomed:
            doomed.append(self)
        if doomed and self == doomed[0]:
            raise PermissionError(f"cannot write {self}")
        return real_write_bytes(self, data)

    async def agent_that_edits_the_protected_sources(**_kwargs):
        driver.write_text("print('ok')\n", encoding="utf-8")
        kernel.write_text("def hipb_mm(a, b):\n    return a @ b * 2\n", encoding="utf-8")
        helper.write_text("SCALE = 999\n", encoding="utf-8")
        return "prepared"

    async def passing_preflight(*_args, **_kwargs):
        return task_preparer.PreflightResult(
            ok=True,
            correctness_ok=True,
            bench_ok=True,
            graph_ok=True,
            profile_ok=True,
        )

    monkeypatch.setattr(task_preparer, "_materialize_reference", lambda _w: None)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent_that_edits_the_protected_sources)
    monkeypatch.setattr(task_preparer, "_preflight_async", passing_preflight)
    monkeypatch.setattr(task_preparer, "PREPARE_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(Path, "write_bytes", refuse_the_first_source_the_rollback_reaches)

    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(model="test-model", experiments_dir=tmp_path / "experiments"),
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            program_md="# Task",
            target_functions=["hipb_mm"],
            source_files=[str(kernel), str(helper)],
        )
    )

    unreachable = doomed[0]
    survivor = next(iter(protected - {unreachable}))
    assert survivor.read_text(encoding="utf-8") == originals[survivor]
    assert unreachable.read_text(encoding="utf-8") != originals[unreachable]
    # The driver conformed, so this preparation would otherwise have been signed off.
    assert result.final_preflight.ok is True
    assert result.ok is False
    assert result.rolled_back is False
    assert unreachable.as_posix() in result.message
    assert survivor.as_posix() not in result.message


def test_a_source_that_comes_back_on_a_retry_is_not_still_reported_as_lost(tmp_path, monkeypatch):
    """Every rollback attempts the whole snapshot, so the latest one is the answer, not the union of all of them."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    original = "def hipb_mm(a, b):\n    return a @ b\n"
    kernel.write_text(original, encoding="utf-8")
    _init_repo(workspace)
    driver = workspace / "driver.py"

    real_write_bytes = Path.write_bytes
    refused: list[Path] = []

    def refuse_only_the_first_rollback(self, data):
        if self == kernel and not refused:
            refused.append(self)
            raise PermissionError(f"cannot write {self}")
        return real_write_bytes(self, data)

    async def agent_that_edits_the_kernel(**_kwargs):
        driver.write_text("print('ok')\n", encoding="utf-8")
        kernel.write_text("def hipb_mm(a, b):\n    return 0\n", encoding="utf-8")
        return "prepared"

    verdicts = [False, True]

    async def preflight_that_passes_on_the_retry(*_args, **_kwargs):
        ok = verdicts.pop(0)
        return task_preparer.PreflightResult(
            ok=ok,
            correctness_ok=ok,
            bench_ok=ok,
            graph_ok=ok,
            profile_ok=ok,
        )

    monkeypatch.setattr(task_preparer, "_materialize_reference", lambda _w: None)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent_that_edits_the_kernel)
    monkeypatch.setattr(task_preparer, "_preflight_async", preflight_that_passes_on_the_retry)
    monkeypatch.setattr(task_preparer, "PREPARE_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(Path, "write_bytes", refuse_only_the_first_rollback)

    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(model="test-model", experiments_dir=tmp_path / "experiments"),
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            program_md="# Task",
            target_functions=["hipb_mm"],
            source_files=[str(kernel)],
        )
    )

    assert refused == [kernel]
    assert kernel.read_text(encoding="utf-8") == original
    assert result.ok is True, result.message
    assert "could not restore" not in (result.message or "")


def test_preparation_gives_back_the_source_the_caller_had_not_the_one_head_holds(tmp_path, monkeypatch):
    """The snapshot records the caller's working tree, and that is what a protected source is restored to."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kernel = workspace / "kernel.py"
    kernel.write_text("def hipb_mm(a, b):\n    return a @ b\n", encoding="utf-8")
    _init_repo(workspace)
    uncommitted = "def hipb_mm(a, b):\n    return (a @ b).contiguous()\n"
    kernel.write_text(uncommitted, encoding="utf-8")
    driver = workspace / "driver.py"

    async def agent_that_edits_the_kernel(**_kwargs):
        driver.write_text("print('ok')\n", encoding="utf-8")
        kernel.write_text("def hipb_mm(a, b):\n    return 0\n", encoding="utf-8")
        return "prepared"

    async def passing_preflight(*_args, **_kwargs):
        return task_preparer.PreflightResult(
            ok=True,
            correctness_ok=True,
            bench_ok=True,
            graph_ok=True,
            profile_ok=True,
        )

    monkeypatch.setattr(task_preparer, "_materialize_reference", lambda _w: None)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", agent_that_edits_the_kernel)
    monkeypatch.setattr(task_preparer, "_preflight_async", passing_preflight)
    monkeypatch.setattr(task_preparer, "PREPARE_MAX_ATTEMPTS", 1)

    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(model="test-model", experiments_dir=tmp_path / "experiments"),
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            program_md="# Task",
            target_functions=["hipb_mm"],
            source_files=[str(kernel)],
        )
    )

    assert result.ok is True
    assert kernel.read_text(encoding="utf-8") == uncommitted


def test_external_bundle_reuses_its_own_spec_beside_the_driver(tmp_path, monkeypatch):
    """An external bundle already ships the spec next to the driver."""
    output_dir = tmp_path / "forge_attempt"
    workspace = output_dir / "workspace"
    workspace.mkdir(parents=True)
    kernel = workspace / "kernel.py"
    kernel.write_text("def hipb_mm(a, b):\n    return a @ b\n", encoding="utf-8")
    driver = output_dir / "driver.py"
    driver.write_text("BROKEN_DRIVER\n", encoding="utf-8")
    source_spec = output_dir / "invocation_spec_aiter_hipb_mm.json"
    source_spec.write_text(json.dumps(_SPEC_PAYLOAD), encoding="utf-8")
    original_spec_bytes = source_spec.read_bytes()

    def fake_materialize_reference(target_workspace):
        ref_dir = Path(target_workspace) / task_preparer.REFERENCE_SUBDIR
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "README.md").write_text("contract\n", encoding="utf-8")
        return ref_dir

    async def fake_agent(**kwargs):
        staged = Path(kwargs["workspace"])
        spec_rel = _runtime_spec_path(kwargs["prompt"])
        assert spec_rel == source_spec.name
        assert (staged / spec_rel).is_file()
        (staged / "driver.py").write_text(
            _AUTHORED_DRIVER.format(spec_rel=spec_rel),
            encoding="utf-8",
        )
        return "prepared"

    async def executing_preflight(driver_script, *_args, **_kwargs):
        bench = _run_driver(Path(driver_script), "--bench-mode")
        case_ids = re.findall(r"case_ms:\s*(\S+)", bench.stdout)
        ok = bench.returncode == 0 and len(case_ids) == 2
        return task_preparer.PreflightResult(
            ok=ok,
            correctness_ok=ok,
            bench_ok=ok,
            graph_ok=True,
            profile_ok=True,
            reasons=[] if ok else [f"bench exited {bench.returncode}"],
        )

    monkeypatch.setattr(task_preparer, "_materialize_reference", fake_materialize_reference)
    monkeypatch.setattr(task_preparer, "_run_prepare_agent", fake_agent)
    monkeypatch.setattr(task_preparer, "_preflight_async", executing_preflight)
    monkeypatch.setattr(task_preparer, "PREPARE_MAX_ATTEMPTS", 1)

    result = asyncio.run(
        task_preparer.prepare_task(
            config=SimpleNamespace(
                model="test-model",
                experiments_dir=tmp_path / "experiments",
            ),
            workspace_dir=str(workspace),
            kernel=str(kernel),
            driver=str(driver),
            program_md="# Task",
            target_functions=["hipb_mm"],
            source_files=[str(kernel)],
            invocation_spec_file=str(source_spec),
            read_only_files=[str(source_spec)],
        )
    )

    assert result.ok is True, result.message
    assert source_spec.read_bytes() == original_spec_bytes
    bench = _run_driver(driver, "--bench-mode")
    assert bench.returncode == 0, bench.stdout + bench.stderr
    assert re.findall(r"case_ms:\s*(\S+)", bench.stdout) == ["case_001", "case_002"]
