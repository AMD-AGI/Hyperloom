from __future__ import annotations

import subprocess
from pathlib import Path

from kernelforge.config.edit_policy import FORGE_LOOP_OUTPUT_DIRNAME, FORGE_PROTECTED_PATHS
from kernelforge.workflows.forge_loop.campaign.workspace_files import hide_forge_loop_output, tracked_editable_paths


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_protected_status_is_independent_of_source_hints(tmp_path: Path):
    assert FORGE_PROTECTED_PATHS.is_protected("config.yaml", workspace=tmp_path)
    assert FORGE_PROTECTED_PATHS.is_protected("scripts/task_runner.py", workspace=tmp_path)
    assert FORGE_PROTECTED_PATHS.is_protected("tests/test_kernel.py", workspace=tmp_path)
    assert FORGE_PROTECTED_PATHS.is_protected("src/kernel_test.cu", workspace=tmp_path)
    assert not FORGE_PROTECTED_PATHS.is_protected("src/helper.cu", workspace=tmp_path)


def test_tracked_editable_paths_are_all_non_protected_files(tmp_path: Path):
    _git(tmp_path, "init", "-q")
    files = {
        "src/kernel.py": "KERNEL = 1\n",
        "src/helper.py": "HELPER = 1\n",
        "CMakeLists.txt": "project(kernel)\n",
        "config.yaml": "task: protected\n",
        "scripts/task_runner.py": "print('protected')\n",
        "tests/test_kernel.py": "assert True\n",
        "custom_driver.py": "print('driver')\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(tmp_path, "add", ".")

    editable = tracked_editable_paths(
        tmp_path,
        exact_protected_paths=[tmp_path / "custom_driver.py"],
    )

    assert editable == {
        "CMakeLists.txt",
        "src/helper.py",
        "src/kernel.py",
    }


def test_recursive_inventory_uses_the_same_nested_rules(tmp_path: Path):
    nested_glob = tmp_path / "src" / "deep" / "test_oracle.py"
    nested_dir = tmp_path / "pkg" / "deep" / "benchmarks" / "oracle.bin"
    extra = tmp_path / "pkg" / "references" / "golden.data"
    editable = tmp_path / "src" / "deep" / "kernel.py"
    for path in (nested_glob, nested_dir, extra, editable):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)

    inventory = set(
        FORGE_PROTECTED_PATHS.inventory(
            tmp_path,
            extra_globs=["*/references/*.data"],
        )
    )

    assert nested_glob in inventory
    assert nested_dir in inventory
    assert extra in inventory
    assert editable not in inventory
    assert FORGE_PROTECTED_PATHS.is_protected(nested_glob, workspace=tmp_path)
    assert FORGE_PROTECTED_PATHS.is_protected(nested_dir, workspace=tmp_path)
    assert FORGE_PROTECTED_PATHS.is_protected(
        extra,
        workspace=tmp_path,
        extra_globs=["*/references/*.data"],
    )


def test_recursive_inventory_skips_the_loops_own_output_tree(tmp_path: Path):
    """The loop's output tree is skipped at any depth, while a harness beside the kernel is still protected."""
    staged = tmp_path / FORGE_LOOP_OUTPUT_DIRNAME / "analysis" / "work" / "abc" / "tools" / "venv"
    staged.mkdir(parents=True)
    (staged / "conftest.py").write_text("")
    (staged / "test_oracle.py").write_text("")
    nested = tmp_path / "src" / FORGE_LOOP_OUTPUT_DIRNAME / "conftest.py"
    nested.parent.mkdir(parents=True)
    nested.write_text("")
    kernel_harness = tmp_path / "src" / "conftest.py"
    kernel_harness.write_text("")

    inventory = set(FORGE_PROTECTED_PATHS.inventory(tmp_path))

    assert staged / "conftest.py" not in inventory
    assert staged / "test_oracle.py" not in inventory
    assert nested not in inventory
    assert kernel_harness in inventory


def test_recursive_inventory_includes_missing_exact_paths(tmp_path: Path):
    exact = tmp_path / "generated" / "source_oracle.py"

    inventory = FORGE_PROTECTED_PATHS.inventory(
        tmp_path,
        exact_paths=[exact],
    )

    assert exact.resolve() in inventory


def test_hidden_output_tree_is_never_listed_by_git(tmp_path: Path):
    _git(tmp_path, "init", "-q")
    output_root = hide_forge_loop_output(tmp_path)
    log = output_root / "orchestration" / "iter_001" / "optimization_plan.reasoning.md"
    log.parent.mkdir(parents=True)
    log.write_text("Summary: x\n")

    assert hide_forge_loop_output(tmp_path) == output_root == tmp_path / FORGE_LOOP_OUTPUT_DIRNAME
    assert (output_root / ".gitignore").read_text() == "*\n"
    listed = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert listed.stdout == ""
