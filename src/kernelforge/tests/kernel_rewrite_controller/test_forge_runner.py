# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from kernelforge.kernel_rewrite_controller import parse_task_payload
from kernelforge.kernel_rewrite_controller.forge_runner import (
    ForgeLoopInvocation,
    build_forge_loop_invocation,
    run_forge_loop,
)
from kernelforge.kernel_rewrite_controller.worktree import OperatorWorktree
from kernelforge.knowledge.kernel_identity import (
    KernelRecipeIdentity,
    kernel_recipe_canonical_id,
)


def _task_and_worktree(tmp_path: Path):
    identity_mapping = {
        "producer": "forge-loop",
        "kernel_name": "fused_moe",
        "framework": "sglang",
        "framework_version": "0.5.0",
        "backend": "triton",
        "gpu": "mi355x",
    }
    operator_id = kernel_recipe_canonical_id(KernelRecipeIdentity.from_mapping(identity_mapping))
    task_dir = tmp_path / "tasks" / operator_id
    task_dir.mkdir(parents=True)
    driver = task_dir / "driver.py"
    driver.write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = tmp_path / "workspace"
    kernel = workspace / "sglang" / "kernels" / "fused_moe.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("VALUE = 1\n", encoding="utf-8")
    task = parse_task_payload(
        {
            "schema_version": 1,
            "identity": identity_mapping,
            "base_commit": "a" * 40,
            "repo_root": str(repo),
            "kernel_path": "sglang/kernels/fused_moe.py",
            "operator_name": "fused_moe",
            "driver_path": "driver.py",
            "source_files": ["sglang/kernels/fused_moe.py"],
            "target_functions": ["fused_moe"],
            "shape_cases": [],
            "priority": 0,
            "reason": "",
            "evidence": [],
        },
        task_dir=task_dir,
    )
    worktree = OperatorWorktree(
        repo_root=repo,
        workspace=workspace,
        branch="forge/controller/test",
        base_commit="a" * 40,
        kernel_path=kernel,
        source_files=(kernel,),
    )
    return task, task_dir, driver, worktree


def test_invocation_maps_task_to_named_kernel_forge_loop(tmp_path: Path) -> None:
    task, task_dir, driver, worktree = _task_and_worktree(tmp_path)
    deadline = time.time() + 3600

    invocation = build_forge_loop_invocation(
        task,
        task_dir=task_dir,
        worktree=worktree,
        deadline_unix=deadline,
    )

    command = list(invocation.command)
    assert command[:4] == [sys.executable, "-m", "kernelforge.cli", "forge-loop"]
    assert command[command.index("--kernel") + 1] == str(worktree.kernel_path)
    assert command[command.index("--driver") + 1] == str(driver)
    assert command[command.index("--workspace") + 1] == str(worktree.workspace)
    assert command[command.index("--operator-name") + 1] == task.operator_name
    assert command[command.index("--framework") + 1] == task.identity.framework
    assert command[command.index("--gpu-type") + 1] == task.identity.gpu
    assert command[command.index("--kernel-backend") + 1] == task.identity.backend
    assert "--auto" not in command
    assert "--nomination-input" not in command


def test_runner_prefers_the_result_json_written_by_the_child(tmp_path: Path) -> None:
    result_json = tmp_path / "result.json"
    payload = {"improved": True, "best_commit": "b" * 40}
    script = "import json, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(json.dumps(" + repr(payload) + "))"
    invocation = ForgeLoopInvocation(
        command=(sys.executable, "-c", script, str(result_json)),
        workspace=tmp_path,
        result_json=result_json,
        deadline_unix=time.time() + 10,
    )

    outcome = run_forge_loop(invocation)

    assert outcome.returncode == 0
    assert outcome.timed_out is False
    assert outcome.result == payload
    assert outcome.improved is True


def test_runner_imports_kernelforge_from_this_repository_src(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("PYTHONPATH", raising=False)
    invocation = ForgeLoopInvocation(
        command=(
            sys.executable,
            "-c",
            "import pathlib, kernelforge; print(pathlib.Path(kernelforge.__file__).resolve())",
        ),
        workspace=tmp_path,
        result_json=tmp_path / "missing.json",
        deadline_unix=time.time() + 10,
    )

    outcome = run_forge_loop(invocation)

    expected = Path(__file__).resolve().parents[2] / "__init__.py"
    assert outcome.returncode == 0
    assert Path(outcome.stdout.strip()) == expected


def test_runner_recovers_a_sentinel_result_when_no_result_file_exists(tmp_path: Path) -> None:
    payload = {"improved": False, "best_commit": ""}
    script = f"import json; print('__FORGE_RESULT__' + json.dumps({payload!r}) + '__FORGE_RESULT__')"
    invocation = ForgeLoopInvocation(
        command=(sys.executable, "-c", script),
        workspace=tmp_path,
        result_json=tmp_path / "missing.json",
        deadline_unix=time.time() + 10,
    )

    outcome = run_forge_loop(invocation)

    assert outcome.returncode == 0
    assert outcome.result == json.loads(json.dumps(payload))
    assert outcome.improved is False


def test_runner_terminates_the_child_process_group_at_deadline(tmp_path: Path) -> None:
    invocation = ForgeLoopInvocation(
        command=(sys.executable, "-c", "import time; time.sleep(60)"),
        workspace=tmp_path,
        result_json=tmp_path / "missing.json",
        deadline_unix=time.time() + 0.1,
    )

    started = time.monotonic()
    outcome = run_forge_loop(invocation)

    assert outcome.timed_out is True
    assert time.monotonic() - started < 10


def test_runner_polls_for_durable_checkpoints_while_child_is_running(tmp_path: Path) -> None:
    result_json = tmp_path / "result.json"
    payload = {"improved": True, "best_commit": "b" * 40}
    script = (
        "import json, pathlib, sys, time; "
        f"pathlib.Path(sys.argv[1]).write_text(json.dumps({payload!r})); "
        "time.sleep(60)"
    )
    callbacks: list[float] = []
    invocation = ForgeLoopInvocation(
        command=(sys.executable, "-c", script, str(result_json)),
        workspace=tmp_path,
        result_json=result_json,
        deadline_unix=time.time() + 1.5,
    )

    outcome = run_forge_loop(
        invocation,
        on_checkpoint=lambda: callbacks.append(time.monotonic()),
    )

    assert outcome.timed_out is True
    assert outcome.result == payload
    assert callbacks
