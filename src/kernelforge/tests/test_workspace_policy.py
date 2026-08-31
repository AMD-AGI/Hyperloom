from __future__ import annotations

import subprocess
from pathlib import Path

from kernelforge.llm.workspace_policy import (
    is_protected_path,
    protected_path_inventory,
    tracked_editable_paths,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_protected_status_is_independent_of_source_hints(tmp_path: Path):
    assert is_protected_path("config.yaml", workspace=tmp_path)
    assert is_protected_path("scripts/task_runner.py", workspace=tmp_path)
    assert is_protected_path("tests/test_kernel.py", workspace=tmp_path)
    assert is_protected_path("src/kernel_test.cu", workspace=tmp_path)
    assert not is_protected_path("src/helper.cu", workspace=tmp_path)


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
        protected_path_inventory(
            tmp_path,
            extra_globs=["*/references/*.data"],
        )
    )

    assert nested_glob in inventory
    assert nested_dir in inventory
    assert extra in inventory
    assert editable not in inventory
    assert is_protected_path(nested_glob, workspace=tmp_path)
    assert is_protected_path(nested_dir, workspace=tmp_path)
    assert is_protected_path(
        extra,
        workspace=tmp_path,
        extra_globs=["*/references/*.data"],
    )


def test_recursive_inventory_includes_missing_exact_paths(tmp_path: Path):
    exact = tmp_path / "generated" / "source_oracle.py"

    inventory = protected_path_inventory(
        tmp_path,
        exact_paths=[exact],
    )

    assert exact.resolve() in inventory
