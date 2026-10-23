# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The start-up hook gives a server process its own directory for aiter's merged tuned tables.

The framework is the stand-in install of ``framework_fixtures``: its ``update_config_files`` writes the merged view to
the literal ``/tmp/aiter_configs/`` as aiter's does, under a name prefix of the test's own so that what lands there
(only when the hook did not work) can be found and removed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

from hyperloom.measure import aiter_merge_hook
from hyperloom.measure.framework.merge import PRIVATE, SHARED, UNCONFIRMED, MergeView
from hyperloom.measure.tests.framework_fixtures import CORE, make_framework, make_installed_aiter

SHARED_DIRECTORY = Path("/tmp/aiter_configs")
MERGED = "gemm_tuned.csv"
IMPORT_AND_MERGE = (
    "from aiter.jit.core import AITER_CONFIGS\n"
    "import os\n"
    "print(os.environ.get('CHAINED', ''), AITER_CONFIGS.AITER_CONFIG_GEMM_FILE)\n"
)


@pytest.fixture
def prefix(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    name = f"hl-test-{uuid.uuid4().hex[:10]}-"
    monkeypatch.setenv("FAKE_AITER_MERGE_PREFIX", name)
    yield name
    for left in SHARED_DIRECTORY.glob(f"{name}*"):
        left.unlink(missing_ok=True)


@pytest.fixture
def install(tmp_path: Path) -> Path:
    """The stand-in aiter's import root: ``aiter.jit.core`` is importable from it."""
    return make_framework(tmp_path).parent.parent


@pytest.fixture
def tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Two tables a variable lists, named by ``AITER_CONFIG_GEMM`` in this process and in its children."""
    made = []
    for name, row in (("one.csv", "1,1,1,1"), ("two.csv", "2,2,2,2")):
        made.append(tmp_path / name)
        made[-1].write_text(f"M,N,K,us\n{row}\n", encoding="utf-8")
    monkeypatch.setenv("AITER_CONFIG_GEMM", os.pathsep.join(map(str, made)))
    return made


def _core(install: Path, source: str | None = None) -> ModuleType:
    """A fresh copy of the stand-in ``aiter/jit/core.py`` as a module (from ``source`` when given)."""
    path = install / "aiter" / "jit" / "core.py"
    module = ModuleType("fake_core")
    exec(compile(source or path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module


def _server(install: Path, view: MergeView | None, *, extra: str = "") -> list[str]:
    """What a Python process prints that imports the stand-in aiter and merges ``AITER_CONFIG_GEMM``, as the test's server.

    An aiter installed on the machine is on its path too, behind the stand-in.
    """
    path = os.pathsep.join(
        part for part in (extra, str(install), str(make_installed_aiter(install.parents[1]))) if part
    )
    environment = {**os.environ, "PYTHONPATH": path, **(view.environment(path) if view else {})}
    done = subprocess.run(
        [sys.executable, "-c", IMPORT_AND_MERGE],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.split()


def test_the_directory_of_one_module_is_rebound_and_a_copy_of_aiter_loaded_elsewhere_is_not(
    tmp_path: Path, install: Path, tables: list[Path], prefix: str
) -> None:
    mine, other = _core(install), _core(install)

    assert aiter_merge_hook.redirect_merge_directory(mine, str(tmp_path / "mine")) is None
    (tmp_path / "mine").mkdir()

    merged_here = mine.AITER_CONFIG().update_config_files(os.pathsep.join(map(str, tables)), "gemm_tuned")
    merged_there = other.AITER_CONFIG().update_config_files(os.pathsep.join(map(str, tables)), "gemm_tuned")
    assert merged_here == f"{tmp_path / 'mine'}/{prefix}{MERGED}"
    assert merged_there == f"{SHARED_DIRECTORY}/{prefix}{MERGED}"
    assert Path(merged_here).read_text() == Path(merged_there).read_text() == "M,N,K,us\n1,1,1,1\n2,2,2,2\n"


@pytest.mark.parametrize(
    ("source", "reason"),
    [
        ("class Other:\n    pass\n", "no AITER_CONFIG.update_config_files"),
        (CORE.replace('Path("/tmp/aiter_configs/")', 'Path("/somewhere/else")'), "names 0 places"),
        (
            CORE.replace("config_path.mkdir", 'Path("/tmp/aiter_configs").exists()\n        config_path.mkdir'),
            "names 2 places",
        ),
    ],
    ids=["no-such-function", "another-path", "two-places"],
)
def test_a_core_that_does_not_name_the_directory_exactly_once_is_left_as_it_is_and_says_why(
    tmp_path: Path, install: Path, source: str, reason: str
) -> None:
    module = _core(install, source)
    before = getattr(getattr(module, "AITER_CONFIG", None), "update_config_files", None)
    code = getattr(before, "__code__", None)

    said = aiter_merge_hook.redirect_merge_directory(module, str(tmp_path))

    assert said is not None and reason in said
    assert getattr(before, "__code__", None) is code


def test_a_server_process_that_imports_aiter_merges_into_the_directory_of_the_test(
    tmp_path: Path, install: Path, tables: list[Path], prefix: str
) -> None:
    view = MergeView.make(tmp_path / "private")
    printed = _server(install, view)
    assert printed[-1] == f"{view.directory}/{prefix}{MERGED}", "the merged view is below the test's directory"
    assert (view.directory / f"{prefix}{MERGED}").read_text() == "M,N,K,us\n1,1,1,1\n2,2,2,2\n"
    assert not (SHARED_DIRECTORY / f"{prefix}{MERGED}").exists(), "nothing reached the shared directory"

    report = view.report()
    assert report["merge_isolation"] == PRIVATE and report["merge_reason"] is None
    assert report["merge_patched_processes"] == 1 and list(report["merge_files"]) == [f"{prefix}{MERGED}"]


def test_looking_a_module_up_without_importing_it_does_not_cost_the_process_its_redirect(
    tmp_path: Path, install: Path, tables: list[Path], prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = MergeView.make(tmp_path / "private")
    probing = "import importlib.util; importlib.util.find_spec('aiter.jit.core'); " + IMPORT_AND_MERGE
    monkeypatch.setattr(sys.modules[__name__], "IMPORT_AND_MERGE", probing)

    printed = _server(install, view)

    assert printed[-1] == f"{view.directory}/{prefix}{MERGED}"


def test_a_process_without_the_directory_variable_is_not_touched_by_the_hook(
    tmp_path: Path, install: Path, tables: list[Path], prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = MergeView.make(tmp_path / "private")
    monkeypatch.delenv(aiter_merge_hook.MERGE_DIR_ENV, raising=False)

    monkeypatch.setenv("PYTHONPATH", str(view.hook))

    printed = _server(install, None)

    assert printed[-1] == f"{SHARED_DIRECTORY}/{prefix}{MERGED}"
    assert not (view.hook / aiter_merge_hook.STARTED_NOTE).exists()


def test_the_hook_runs_the_sitecustomize_it_shadows(
    tmp_path: Path, install: Path, tables: list[Path], prefix: str
) -> None:
    view = MergeView.make(tmp_path / "private")
    shadowed = tmp_path / "shadowed"
    shadowed.mkdir()
    (shadowed / "sitecustomize.py").write_text("import os\nos.environ['CHAINED'] = 'yes'\n", encoding="utf-8")

    printed = _server(install, view, extra=str(shadowed))

    assert printed[0] == "yes" and printed[-1].startswith(str(view.directory))


def test_a_shadowed_sitecustomize_that_fails_does_not_stop_the_process_and_the_report_says_so(
    tmp_path: Path, install: Path, tables: list[Path], prefix: str
) -> None:
    view = MergeView.make(tmp_path / "private")
    shadowed = tmp_path / "shadowed"
    shadowed.mkdir()
    (shadowed / "sitecustomize.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")

    printed = _server(install, view, extra=str(shadowed))

    assert printed[-1].startswith(str(view.directory)), "aiter was still redirected and the process ran"
    assert "boom" in view.report()["merge_chain_failure"]


def test_a_loader_the_hook_cannot_wrap_is_reported_as_a_process_it_could_not_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    view = MergeView.make(tmp_path / "private")
    monkeypatch.setattr(aiter_merge_hook, "_HERE", str(view.hook))
    spec = ModuleSpec("aiter.jit.core", loader=object())

    class Stub:
        def find_spec(self, fullname: str, path: object = None, target: object = None) -> ModuleSpec:
            return spec

    finder = aiter_merge_hook._AiterCoreFinder(str(view.directory))
    monkeypatch.setattr(sys, "meta_path", [finder, Stub()])

    assert finder.find_spec("aiter.jit.core") is spec

    report = view.report()
    assert report["merge_isolation"] == SHARED and "no exec_module" in report["merge_reason"]


def test_a_process_whose_aiter_cannot_be_redirected_is_reported_and_aiter_still_imports(
    tmp_path: Path, install: Path, tables: list[Path], prefix: str
) -> None:
    view = MergeView.make(tmp_path / "private")
    elsewhere = tmp_path / "elsewhere"
    core = install / "aiter" / "jit" / "core.py"
    core.write_text(CORE.replace('Path("/tmp/aiter_configs/")', f"Path({str(elsewhere)!r})"), encoding="utf-8")

    printed = _server(install, view)

    assert printed[-1] == f"{elsewhere}/{prefix}{MERGED}", "aiter imported and merged where it always did"
    report = view.report()
    assert report["merge_isolation"] == SHARED and "names 0 places" in report["merge_reason"]
    assert report["merge_patched_processes"] == 0


def test_the_report_tells_a_hook_that_never_ran_from_one_that_ran_and_found_no_aiter(tmp_path: Path) -> None:
    view = MergeView.make(tmp_path / "private")
    assert view.report()["merge_isolation"] == UNCONFIRMED
    assert view.report()["merge_reason"] == "no process of the test ran the hook"

    (view.hook / aiter_merge_hook.STARTED_NOTE).write_text(json.dumps({"pid": 1}), encoding="utf-8")
    assert view.report()["merge_isolation"] == PRIVATE, "a server that never imported aiter has nothing to share"


def test_the_environment_puts_the_hook_first_once_and_keeps_the_rest_of_the_path_in_order(tmp_path: Path) -> None:
    view = MergeView.make(tmp_path / "private")

    environment = view.environment(f"/a{os.pathsep}{view.hook}{os.pathsep}/b")

    assert environment["PYTHONPATH"].split(os.pathsep) == [str(view.hook), "/a", "/b"]
    assert environment[aiter_merge_hook.MERGE_DIR_ENV] == str(view.directory)
    assert (view.hook / "sitecustomize.py").read_bytes() == Path(aiter_merge_hook.__file__).read_bytes()
