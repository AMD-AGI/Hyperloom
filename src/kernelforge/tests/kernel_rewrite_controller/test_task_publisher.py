# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
from pathlib import Path

from kernelforge.kernel_rewrite_controller.paths import ControllerLayout
from kernelforge.kernel_rewrite_controller.task_publisher import (
    pending_rejections,
    publish_complete_staged_tasks,
    publish_staged_task,
)
from kernelforge.tests.kernel_rewrite_controller.conftest import _git


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def _staged(layout: ControllerLayout, repo: Path, name: str = "draft") -> Path:
    staged = layout.agent_staging_root / name
    staged.mkdir(parents=True)
    (staged / "driver.py").write_text("print('SNR: 100 dB')\n", encoding="utf-8")
    (staged / "task.json").write_text(
        json.dumps(
            {
                "identity": {
                    "producer": "forge-loop",
                    "framework": "standalone",
                    "framework_version": "unknown",
                    "backend": "triton",
                    "gpu": "mi355x",
                },
                "base_commit": "",
                "repo_root": str(repo),
                "kernel_path": "kernel.py",
                "operator_name": "kernel",
                "driver_path": "ignored.py",
                "source_files": ["kernel.py"],
                "target_functions": ["kernel"],
                "shape_cases": [],
                "priority": 0,
                "reason": "offline replay",
                "evidence": [],
            }
        ),
        encoding="utf-8",
    )
    return staged


def _newest_staged_mtime(staged: Path) -> float:
    return max(path.stat().st_mtime for path in (staged, *staged.rglob("*")))


def test_publish_pins_live_head_and_moves_complete_task_atomically(tmp_path: Path) -> None:
    repo, head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)

    result = publish_staged_task(layout, staged)

    assert result.published is True
    assert not staged.exists()
    payload = json.loads((layout.task_dir(result.operator_id) / "task.json").read_text(encoding="utf-8"))
    assert payload["base_commit"] == head
    assert payload["driver_path"] == "driver.py"


def test_publish_normalizes_harmless_agent_identity_variations(tmp_path: Path) -> None:
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)
    task_json = staged / "task.json"
    payload = json.loads(task_json.read_text(encoding="utf-8"))
    payload["identity"].update(
        {
            "producer": " FORGE-LOOP ",
            "framework": " SGLang ",
            "framework_version": " v0.5.17+ROCM ",
            "backend": " TRITON ",
            "gpu": " MI355X ",
        }
    )
    task_json.write_text(json.dumps(payload), encoding="utf-8")

    result = publish_staged_task(layout, staged)

    assert result.published is True
    published = json.loads((layout.task_dir(result.operator_id) / "task.json").read_text(encoding="utf-8"))
    assert published["identity"] == {
        "producer": "forge-loop",
        "kernel_name": "kernel",
        "framework": "sglang",
        # The build the wheel was compiled as is not part of the release a port was
        # written against, and neither is the tag convention the campaign read it under.
        "framework_version": "0.5.17",
        "backend": "triton",
        "gpu": "mi355x",
    }


def test_the_published_identity_states_the_derived_kernel_name(tmp_path: Path) -> None:
    """The file on disk has to name the operator the controller went on to use.

    Nothing downstream re-reads the draft, so a published task.json that still
    showed the agent's spelling would disagree with its own directory name.
    """
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)
    task_json = staged / "task.json"
    payload = json.loads(task_json.read_text(encoding="utf-8"))
    payload["operator_name"] = "backend::Fused.MoE-Kernel"
    task_json.write_text(json.dumps(payload), encoding="utf-8")

    result = publish_staged_task(layout, staged)

    assert result.published is True
    assert result.operator_id.split(":")[2] == "fused_moe"
    published = json.loads((layout.task_dir(result.operator_id) / "task.json").read_text(encoding="utf-8"))
    assert published["identity"]["kernel_name"] == "fused_moe"
    assert published["operator_name"] == "backend::Fused.MoE-Kernel"


def test_a_kernel_belonging_to_no_package_is_published_under_one_word_for_that(
    tmp_path: Path,
) -> None:
    """``standalone`` is not an address; forge-loop resolves it away before it becomes one.

    The override is folded when the run builds its identity, so a draft keeping
    its own spelling named ``standalone`` in its directory and its published
    pointer while the run filed its result under ``unknown``.
    """
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)

    result = publish_staged_task(layout, staged)

    assert result.published is True
    assert result.operator_id.split(":")[3] == "unknown"
    published = json.loads((layout.task_dir(result.operator_id) / "task.json").read_text(encoding="utf-8"))
    assert published["identity"]["framework"] == "unknown"


def test_a_framework_alias_is_published_as_the_package_it_names(tmp_path: Path) -> None:
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)
    task_json = staged / "task.json"
    payload = json.loads(task_json.read_text(encoding="utf-8"))
    payload["identity"]["framework"] = "aiter_meta"
    task_json.write_text(json.dumps(payload), encoding="utf-8")

    result = publish_staged_task(layout, staged)

    assert result.published is True
    assert result.operator_id.split(":")[3] == "aiter"


def _packaged_repo(tmp_path: Path, relative: str) -> tuple[Path, str]:
    """A repository whose kernel sits inside a framework package."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    kernel = repo / relative
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "baseline")
    return repo, _git(repo, "rev-parse", "HEAD")


def _staged_in(layout: ControllerLayout, repo: Path, relative: str, framework: str) -> Path:
    staged = _staged(layout, repo)
    task_json = staged / "task.json"
    payload = json.loads(task_json.read_text(encoding="utf-8"))
    payload["identity"]["framework"] = framework
    payload["kernel_path"] = relative
    payload["source_files"] = [relative]
    task_json.write_text(json.dumps(payload), encoding="utf-8")
    return staged


def test_a_framework_none_of_the_paths_sit_under_is_refused(tmp_path: Path) -> None:
    """The declared framework reaches forge-loop as an override, and an override
    short-circuits the path inference -- so nothing downstream would notice the
    dimension naming a package the source does not live in.
    """
    repo, _head = _packaged_repo(tmp_path, "sglang/kernels/fused_moe.py")
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged_in(layout, repo, "sglang/kernels/fused_moe.py", "vllm")

    result = publish_staged_task(layout, staged)

    assert result.published is False
    assert "none of this task's paths sit under" in result.reason
    assert "sglang/kernels/fused_moe.py" in result.reason
    assert staged.is_dir()


def test_a_path_under_two_packages_keeps_whichever_the_task_declared(tmp_path: Path) -> None:
    """Refuse a contradiction, never re-derive.

    The store holds three pages whose source sits at
    ``sglang/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py``. Which package
    owns that kernel is not a question the order of a tuple should answer, and
    the detector answers it with ``aiter`` only because that is where the tuple
    starts -- re-addressing those pages on that basis would invent a move.
    """
    relative = "sglang/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py"
    repo, _head = _packaged_repo(tmp_path, relative)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged_in(layout, repo, relative, "sglang")

    result = publish_staged_task(layout, staged)

    assert result.published is True
    assert result.operator_id.split(":")[3] == "sglang"


def test_a_path_under_no_known_package_witnesses_nothing(tmp_path: Path) -> None:
    """The detector knows three packages and the dimension holds more than three
    values -- the store's 146 pages include 4 under ``torch``, which no path
    names. Silence has to mean it has nothing to say.
    """
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged_in(layout, repo, "kernel.py", "torch")

    result = publish_staged_task(layout, staged)

    assert result.published is True
    assert result.operator_id.split(":")[3] == "torch"


def test_a_framework_that_is_not_even_a_word_is_refused_with_the_sweep_intact(tmp_path: Path) -> None:
    """One draft's damage stops at that draft.

    ``task.json`` is agent-authored, so a dimension can hold any JSON type at
    all. Folding the framework before checking that it is a string would hand a
    list to ``str.strip``, and the ``AttributeError`` that raises is not one the
    publication sweep catches -- it would leave every other draft staged beside
    this one unpublished.
    """
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    broken = _staged(layout, repo, name="broken")
    payload = json.loads((broken / "task.json").read_text(encoding="utf-8"))
    payload["identity"]["framework"] = ["sglang"]
    (broken / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    intact = _staged(layout, repo, name="intact")
    written_at = max(_newest_staged_mtime(broken), _newest_staged_mtime(intact))

    results = {
        result.source_dir.name: result
        for result in publish_complete_staged_tasks(
            layout,
            quiescent_sec=5.0,
            now=lambda: written_at + 5.0,
        )
    }

    assert results["broken"].published is False
    assert results["broken"].reason
    assert results["intact"].published is True


def test_publish_rejects_a_repo_path_below_git_toplevel(tmp_path: Path) -> None:
    """The refusal names the top level, the one thing the agent cannot resolve."""
    repo, _head = _repo(tmp_path)
    nested = repo / "nested"
    nested.mkdir()
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, nested)

    result = publish_staged_task(layout, staged)

    assert result.published is False
    assert "Git top-level" in result.reason
    assert f"use {repo.resolve()}" in result.reason
    assert staged.is_dir()


def test_a_repo_root_outside_git_is_told_what_to_pass(tmp_path: Path) -> None:
    layout = ControllerLayout(tmp_path / "output")
    loose = tmp_path / "loose"
    loose.mkdir()
    staged = _staged(layout, loose)

    result = publish_staged_task(layout, staged)

    assert result.published is False
    assert "Pass the Git top-level of the repository that holds kernel_path" in result.reason


def test_publish_rejects_source_files_outside_the_pinned_repo(
    tmp_path: Path,
) -> None:
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)
    task_json = staged / "task.json"
    payload = json.loads(task_json.read_text(encoding="utf-8"))
    payload["source_files"].append("python/other_repo/source.py")
    task_json.write_text(json.dumps(payload), encoding="utf-8")

    result = publish_staged_task(layout, staged)

    assert result.published is False
    assert "source path is not tracked" in result.reason
    assert "python/other_repo/source.py" in result.reason
    # The usual cause is a path from the repository on the other side of a call
    # chain, which the bare rule reads as an ordinary typo.
    assert str(repo.resolve()) in result.reason
    assert "belongs in evidence rather than source_files" in result.reason


def test_publish_rejects_duplicate_operator_without_deleting_new_draft(tmp_path: Path) -> None:
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    first = _staged(layout, repo, "first")
    duplicate = _staged(layout, repo, "duplicate")
    assert publish_staged_task(layout, first).published is True

    result = publish_staged_task(layout, duplicate)

    assert result.published is False
    assert "is already published" in result.reason
    assert "drop this draft or point it at a different operator" in result.reason
    assert duplicate.is_dir()


def test_a_staged_task_still_being_written_is_left_alone(tmp_path: Path) -> None:
    # The scan runs on a timer beside the live agent, so a directory whose files were touched a moment ago may still
    # be mid-write.
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)

    # Default window against real time: the files were just written, which is what a scan landing in the same poll
    # tick as the agent's write sees.
    results = publish_complete_staged_tasks(layout)

    assert results == ()
    assert staged.is_dir()
    assert (staged / "driver.py").is_file()


def test_a_quiescent_staged_task_is_published(tmp_path: Path) -> None:
    repo, head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)
    written_at = _newest_staged_mtime(staged)

    results = publish_complete_staged_tasks(
        layout,
        quiescent_sec=5.0,
        now=lambda: written_at + 5.0,
    )

    assert [result.published for result in results] == [True]
    assert not staged.exists()
    payload = json.loads((layout.task_dir(results[0].operator_id) / "task.json").read_text(encoding="utf-8"))
    assert payload["base_commit"] == head


def test_a_refused_draft_is_not_revalidated_until_it_changes(tmp_path: Path) -> None:
    """Refusal keeps the draft, and this scan runs on a half-second timer."""
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)
    (staged / "task.json").write_text("{ not json", encoding="utf-8")
    written_at = _newest_staged_mtime(staged)
    refused: dict[str, float] = {}

    first = publish_complete_staged_tasks(
        layout,
        quiescent_sec=0.0,
        now=lambda: written_at + 5.0,
        refused=refused,
    )
    second = publish_complete_staged_tasks(
        layout,
        quiescent_sec=0.0,
        now=lambda: written_at + 6.0,
        refused=refused,
    )

    assert [result.published for result in first] == [False]
    assert second == ()
    assert staged.is_dir()


def test_a_revised_draft_is_offered_again_after_a_refusal(tmp_path: Path) -> None:
    repo, head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)
    broken = staged / "task.json"
    payload = broken.read_text(encoding="utf-8")
    broken.write_text("{ not json", encoding="utf-8")
    refused: dict[str, float] = {}

    publish_complete_staged_tasks(
        layout,
        quiescent_sec=0.0,
        now=lambda: _newest_staged_mtime(staged) + 5.0,
        refused=refused,
    )
    broken.write_text(payload, encoding="utf-8")
    os.utime(broken, (_newest_staged_mtime(staged) + 10.0,) * 2)
    retried = publish_complete_staged_tasks(
        layout,
        quiescent_sec=0.0,
        now=lambda: _newest_staged_mtime(staged) + 15.0,
        refused=refused,
    )

    assert [result.published for result in retried] == [True]
    published = json.loads((layout.task_dir(retried[0].operator_id) / "task.json").read_text(encoding="utf-8"))
    assert published["base_commit"] == head


def test_a_refusal_is_written_beside_the_draft_that_earned_it(tmp_path: Path) -> None:
    """The scan runs out of process, so a file is the only channel back."""
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)
    payload = json.loads((staged / "task.json").read_text(encoding="utf-8"))
    payload["identity"].pop("gpu")
    (staged / "task.json").write_text(json.dumps(payload), encoding="utf-8")

    publish_complete_staged_tasks(
        layout,
        quiescent_sec=0.0,
        now=lambda: _newest_staged_mtime(staged) + 5.0,
    )

    assert pending_rejections(layout.agent_staging_root) == {
        "draft": "invalid staged task: identity is missing fields: gpu"
    }


def test_writing_the_refusal_does_not_make_the_draft_look_revised(tmp_path: Path) -> None:
    """The note lands inside the directory whose mtime decides revalidation."""
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)
    (staged / "task.json").write_text("{ not json", encoding="utf-8")
    refused: dict[str, float] = {}
    clock = _newest_staged_mtime(staged) + 5.0

    first = publish_complete_staged_tasks(layout, quiescent_sec=0.0, now=lambda: clock, refused=refused)
    second = publish_complete_staged_tasks(layout, quiescent_sec=0.0, now=lambda: clock + 1.0, refused=refused)

    assert [result.published for result in first] == [False]
    assert second == ()


def test_a_published_draft_leaves_no_pending_refusal(tmp_path: Path) -> None:
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    staged = _staged(layout, repo)

    assert publish_staged_task(layout, staged).published is True
    assert pending_rejections(layout.agent_staging_root) == {}


def test_publish_rejects_a_symlinked_staging_directory(tmp_path: Path) -> None:
    repo, _head = _repo(tmp_path)
    layout = ControllerLayout(tmp_path / "output")
    real = _staged(layout, repo, "real")
    link = layout.agent_staging_root / "linked"
    link.symlink_to(real, target_is_directory=True)

    result = publish_staged_task(layout, link)

    assert result.published is False
    assert result.reason == "staged task is not a safe directory"
    assert real.is_dir()
