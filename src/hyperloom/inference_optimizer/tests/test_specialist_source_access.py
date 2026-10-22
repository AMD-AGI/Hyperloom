# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A specialist can open every framework tree its task named, including when none can seed a worktree."""

from __future__ import annotations


from hyperloom.orchestrator.specialists.subprocess_ import (
    SpecialistSubprocessConfig,
    SpecialistSubprocessDispatcher,
)


def _dispatcher(roots: tuple[str, ...]) -> SpecialistSubprocessDispatcher:
    return SpecialistSubprocessDispatcher(SpecialistSubprocessConfig(framework_source_roots=roots))


def test_a_pip_installed_framework_is_still_reachable_without_a_worktree(tmp_path):
    """dist-packages has no checkout to seed ``git worktree add``, which left the source invisible."""
    vllm = tmp_path / "dist-packages" / "vllm"
    aiter = tmp_path / "dist-packages" / "aiter"
    for root in (vllm, aiter):
        root.mkdir(parents=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    dirs = _dispatcher((str(vllm), str(aiter)))._accessible_dirs(workspace, None)

    assert dirs == [str(workspace), str(vllm), str(aiter)]


def test_the_roots_a_worktree_did_not_cover_are_added_too(tmp_path):
    """Only one root can seed the worktree; sglang sessions name five and authored patches on one."""
    sglang = tmp_path / "sglang"
    aiter = tmp_path / "aiter"
    for root in (sglang, aiter):
        root.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()

    dirs = _dispatcher((str(sglang), str(aiter)))._accessible_dirs(workspace, worktree)

    assert dirs == [str(worktree), str(workspace), str(sglang), str(aiter)]


def test_a_root_that_is_not_there_is_skipped(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    dirs = _dispatcher((str(tmp_path / "absent"), ""))._accessible_dirs(workspace, None)

    assert dirs == [str(workspace)]


def test_a_root_repeated_as_the_workspace_is_not_added_twice(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()

    dirs = _dispatcher((str(workspace),))._accessible_dirs(workspace, None)

    assert dirs == [str(workspace)]


def test_a_task_naming_no_roots_is_unchanged(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    worktree = tmp_path / "wt"
    worktree.mkdir()

    assert _dispatcher(())._accessible_dirs(workspace, worktree) == [str(worktree), str(workspace)]


def test_every_accessible_dir_reaches_the_claude_argv(tmp_path):
    framework = tmp_path / "fw"
    framework.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    dispatcher = _dispatcher((str(framework),))

    cmd = dispatcher._build_claude_cmd(
        system_prompt_file=workspace / "system_prompt.md",
        system_prompt="SYS",
        workspace=workspace,
        worktree=None,
        disallowed_tools=frozenset(),
    )

    add_dirs = [cmd[i + 1] for i, value in enumerate(cmd[:-1]) if value == "--add-dir"]
    assert add_dirs == dispatcher._accessible_dirs(workspace, None)
    assert str(framework) in add_dirs


def test_overlapping_roots_are_both_passed_through(tmp_path):
    """A session names nested roots (``/sgl-workspace/aiter`` and ``/sgl-workspace/aiter/aiter``); a redundant grant is harmless."""
    parent = tmp_path / "aiter"
    child = parent / "aiter"
    child.mkdir(parents=True)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    dirs = _dispatcher((str(parent), str(child)))._accessible_dirs(workspace, None)

    assert dirs == [str(workspace), str(parent), str(child)]
