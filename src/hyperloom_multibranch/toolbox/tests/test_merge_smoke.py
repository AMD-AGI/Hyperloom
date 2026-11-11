# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The merge smoke, against a real framework interpreter: a virtual environment with an ``sglang`` package installed,
the test's patch stack compiled by the real compile, and the real toolbox and hosted backend behind the executor."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import FailureClass, TestResult, Validity
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import CompiledEnvironment, LocalBlobStore, PackageBase, compile_environment
from hyperloom.measure.jit.native.needs import EnvEntry, EnvRoot
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.tests._recipes import IMAGE, PREFIX, git_patch, put_patch, recipe_with
from hyperloom.measure.tests.hosted_fixtures import make_rig, point_yaml
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.toolbox.hosted_executor import HostedExecutor
from hyperloom_multibranch.toolbox.merge_smoke import (
    MERGE_SMOKE,
    MERGE_SMOKE_ENV,
    SMOKE_NOT_RUN,
    MergeSmoke,
    SmokeNotRun,
    merge_smoke_from,
)
from hyperloom_multibranch.toolbox.queue import QueuePolicy
from hyperloom_multibranch.toolbox.tests.helpers import FakePlanner, build, req

POLICY = QueuePolicy(per_branch_k=1, global_concurrency=1)
INIT = "from sglang import lang\nfrom sglang.srt import a\n"
OLD_A = "def f():\n    return 1\n"
OLD_B = "y = 1\n"
#: The image's package: ``lang`` is a directory no patch touches, so the farm links it into the image as a whole.
IMAGE_FILES = {
    "__init__.py": INIT,
    "lang/__init__.py": "from sglang.lang import api\n",
    "lang/api.py": "API = 1\n",
    "srt/__init__.py": "",
    "srt/a.py": OLD_A,
    "srt/b.py": OLD_B,
}


@dataclass(frozen=True)
class Framework:
    """A framework interpreter of its own: a virtual environment with the image's ``sglang`` in its site-packages."""

    python: str
    package: Path


def _framework(directory: Path, files: dict[str, str]) -> Framework:
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(directory)], check=True, capture_output=True)
    python = str(directory / "bin" / "python")
    purelib = subprocess.run(
        [python, "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    package = Path(purelib) / "sglang"
    for rel, text in files.items():
        (package / rel).parent.mkdir(parents=True, exist_ok=True)
        (package / rel).write_text(text)
    return Framework(python, package)


@pytest.fixture
def framework(tmp_path: Path) -> Framework:
    return _framework(tmp_path / "framework", IMAGE_FILES)


def _environment(tmp_path: Path, framework: Framework, *stack: dict[str, tuple[str, str]]) -> CompiledEnvironment:
    """The compiled environment of a recipe whose patch stack is ``stack``: each patch maps a package file to its text
    before and after."""
    blobs = LocalBlobStore(tmp_path / "blobs")
    refs = []
    for index, changes in enumerate(stack):
        before = {f"{PREFIX}/{rel}": old for rel, (old, _) in changes.items()}
        after = {f"{PREFIX}/{rel}": new for rel, (_, new) in changes.items()}
        refs.append(put_patch(blobs, f"{index}.patch", git_patch(tmp_path / f"patch-{index}", before, after)))
    base = PackageBase("sglang", framework.package, "sglang", "0.0.1")
    return compile_environment(
        recipe_with(tuple(refs)), RecipeDelta(), blobs=blobs, active_image_digest=IMAGE, locator=lambda _: base
    )


def _check(tmp_path: Path, framework: Framework, *stack: dict[str, tuple[str, str]], **smoke: object) -> str | None:
    environment = _environment(tmp_path, framework, *stack)
    point = point_yaml(tmp_path, "smoke-point.yaml")
    return MergeSmoke(default_python=framework.python, **smoke).check(environment, (point,))  # type: ignore[arg-type]


#: The stack's first patch breaks ``srt/a.py``; the merge's own change, the second, touches ``srt/b.py`` only.
BROKEN_FIRST = {"srt/a.py": (OLD_A, "def f(:\n    return 1\n")}
MERGE_OWN = {"srt/b.py": (OLD_B, "y = 2\n")}


def test_a_syntax_error_anywhere_in_the_stack_is_the_first_error_line(tmp_path: Path, framework: Framework) -> None:
    error = _check(tmp_path, framework, BROKEN_FIRST, MERGE_OWN)
    assert error is not None and error.startswith("sglang/srt/a.py:1: SyntaxError: "), error


def test_an_import_time_error_names_the_module_and_where_it_failed(tmp_path: Path, framework: Framework) -> None:
    error = _check(tmp_path, framework, {"srt/a.py": (OLD_A, "VALUE = undefined_name\n")}, MERGE_OWN)
    assert error == "import sglang: NameError: name 'undefined_name' is not defined (sglang/srt/a.py:1)"


def test_a_good_stack_passes_and_writes_no_bytecode_into_the_image(tmp_path: Path, framework: Framework) -> None:
    assert _check(tmp_path, framework, {"srt/a.py": (OLD_A, "def f():\n    return 2\n")}, MERGE_OWN) is None
    assert not list(framework.package.rglob("__pycache__")), "the image's package stays as it was"


def test_the_import_sees_no_gpu(tmp_path: Path, framework: Framework, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    probe = (
        "import os\n"
        "seen = tuple(os.environ.get(name) for name in"
        " ('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'))\n"
        "if seen != ('', '', ''):\n"
        "    gpu_visible\n"
    )
    assert _check(tmp_path, framework, {"srt/a.py": (OLD_A, probe)}) is None


#: A per-arch table chosen by a device probe at import, which raises where no GPU is seen.
DEVICE_PROBE = {"srt/a.py": (OLD_A, "def get_gfx():\n    raise RuntimeError('no GPU found')\n\nGFX = get_gfx()\n")}


def test_a_device_probe_at_import_that_finds_no_gpu_is_not_judged(tmp_path: Path, framework: Framework) -> None:
    assert _check(tmp_path, framework, DEVICE_PROBE, MERGE_OWN) is None


def test_a_variable_the_host_lacks_read_at_import_is_not_judged(tmp_path: Path, framework: Framework) -> None:
    knob = {"srt/a.py": (OLD_A, "import os\nKNOB = os.environ['HL_NO_SUCH_VARIABLE']\n")}
    assert _check(tmp_path, framework, knob) is None


def _point_with(tmp_path: Path, envs: dict[str, str]) -> Path:
    point = tmp_path / "envs-point.yaml"
    point.write_text("benchmark:\n  envs:\n" + "".join(f"    {name}: '{value}'\n" for name, value in envs.items()))
    return point


def test_the_import_gets_the_point_s_server_environment_with_no_gpu(tmp_path: Path, framework: Framework) -> None:
    """A recipe variable read at import is there; a device mask the point sets is still overridden."""
    probe = (
        "import os\n"
        "if os.environ.get('NEW_KNOB') != '7':\n"
        "    knob_not_passed\n"
        "if os.environ.get('HIP_VISIBLE_DEVICES') != '':\n"
        "    gpu_visible\n"
    )
    environment = _environment(tmp_path, framework, {"srt/a.py": (OLD_A, probe)})
    smoke = MergeSmoke(default_python=framework.python)
    with_knob = _point_with(tmp_path, {"NEW_KNOB": "7", "HIP_VISIBLE_DEVICES": "3"})
    assert smoke.check(environment, (with_knob,)) is None
    without = _point_with(tmp_path, {"HIP_VISIBLE_DEVICES": "3"})
    error = smoke.check(environment, (without,))
    assert error is not None and "NameError: name 'knob_not_passed'" in error, error


def test_a_missing_module_a_patched_file_imports_is_judged(tmp_path: Path, framework: Framework) -> None:
    error = _check(tmp_path, framework, {"srt/a.py": (OLD_A, "import hl_no_such_module\n")})
    assert error == "import sglang: ModuleNotFoundError: No module named 'hl_no_such_module' (sglang/srt/a.py:1)"


def test_an_attribute_error_in_a_patched_file_is_judged(tmp_path: Path, framework: Framework) -> None:
    error = _check(tmp_path, framework, {"srt/a.py": (OLD_A, "import os\nos.no_such_attribute\n")})
    assert error is not None and error.startswith("import sglang: AttributeError: "), error
    assert error.endswith("(sglang/srt/a.py:2)"), error


def test_an_import_error_raised_in_an_unpatched_module_is_not_judged(tmp_path: Path) -> None:
    """The patched file calls the image's own probe at import, which raises ``ImportError`` with no GPU: the raising
    frame is the image's module, not the patch."""
    probe = "def probe():\n    raise ImportError('no device library without a GPU')\n"
    framework = _framework(tmp_path / "framework", {**IMAGE_FILES, "srt/probe.py": probe})
    calls = {"srt/a.py": (OLD_A, "from sglang.srt.probe import probe\nprobe()\n")}
    assert _check(tmp_path, framework, calls) is None


def test_an_import_the_image_fails_too_is_not_judged(tmp_path: Path) -> None:
    """The patched ``b`` raises an error the smoke judges, but the image's own package fails to import here as well
    (after ``b``, in its ``a``), so the failure says nothing about the candidate."""
    init = "from sglang.srt import b\nfrom sglang.srt import a\n"
    files = {**IMAGE_FILES, "__init__.py": init, "srt/a.py": "import os\nos.environ['HL_NO_SUCH_VARIABLE']\n"}
    framework = _framework(tmp_path / "framework", files)
    assert _check(tmp_path, framework, {"srt/b.py": (OLD_B, "y = undefined_name\n")}) is None


def test_a_missing_interpreter_is_not_run(tmp_path: Path, framework: Framework) -> None:
    missing = replace(framework, python=str(tmp_path / "no-such-python"))
    with pytest.raises(SmokeNotRun, match="is not here"):
        _check(tmp_path, missing, MERGE_OWN)


def test_the_interpreter_the_point_names_is_the_one_smoked_with(tmp_path: Path, framework: Framework) -> None:
    environment = _environment(tmp_path, framework, MERGE_OWN)
    point = tmp_path / "named.yaml"
    point.write_text(f"benchmark:\n  envs:\n    HYPERLOOM_FRAMEWORK_PYTHON: '{tmp_path / 'named-python'}'\n")
    with pytest.raises(SmokeNotRun, match="named-python is not here"):
        MergeSmoke(default_python=framework.python).check(environment, (point,))


def test_an_import_past_its_timeout_is_not_run(tmp_path: Path, framework: Framework) -> None:
    slow = {"srt/a.py": (OLD_A, "import time\ntime.sleep(60)\n")}
    with pytest.raises(SmokeNotRun, match="import check timed out after 2 s"):
        _check(tmp_path, framework, slow, timeout_sec=2)


def test_an_environment_without_patches_is_not_checked(tmp_path: Path) -> None:
    smoke = MergeSmoke(default_python=str(tmp_path / "no-such-python"))
    assert smoke.check(None, ()) is None


def test_the_operator_switch() -> None:
    assert merge_smoke_from({}) == MergeSmoke() and merge_smoke_from({MERGE_SMOKE_ENV: " On "}) == MergeSmoke()
    assert merge_smoke_from({MERGE_SMOKE_ENV: "off"}) is None
    with pytest.raises(ValueError, match="expected 'on' or 'off'"):
        merge_smoke_from({MERGE_SMOKE_ENV: "no"})


# --- through the toolbox: the smoke is the first step, before a sandbox is asked for -------------------------------


class _Planner(FakePlanner):
    def __init__(self, point: Path, environment: CompiledEnvironment) -> None:
        super().__init__()
        self.point, self.environment = point, environment

    def plan(self, request, parent) -> AcceptedMeasurement:  # type: ignore[no-untyped-def]
        super().plan(request, parent)
        return AcceptedMeasurement(self.environment, (self.point,))


@dataclass
class _Backend:
    """The real hosted backend, counting the sandboxes asked for. The fake runner lays no farm, so the call it gets
    carries no environment: what is measured here is whether the test reaches it."""

    inner: object
    calls: int = 0
    seen: list[object] = field(default_factory=list)
    #: The point the first sandbox measures in place of the planned one (a fault the next attempt does not meet).
    first: Path | None = None

    name = "hosted"

    @property
    def journal(self):  # type: ignore[no-untyped-def]
        return self.inner.journal  # type: ignore[attr-defined]

    def run_test(self, call):  # type: ignore[no-untyped-def]
        self.calls += 1
        points = (self.first,) if self.first is not None and self.calls == 1 else call.points
        return self.inner.run_test(replace(call, environment=None, points=points))  # type: ignore[attr-defined]


async def _through_toolbox(
    tmp_path: Path, environment: CompiledEnvironment, smoke: MergeSmoke, first: dict[str, str] | None = None
) -> tuple[TestResult, _Backend, object]:
    rig = make_rig(tmp_path)
    try:
        point = point_yaml(tmp_path, "planned.yaml", fake_mode="ok")
        backend = _Backend(rig.backend, first=point_yaml(tmp_path, "first.yaml", **first) if first else None)
        executor = HostedExecutor(backend, tmp_path / "session")  # type: ignore[arg-type]
        executor.smoke = smoke
        tb = build(
            tmp_path,
            executor=executor,  # type: ignore[arg-type]
            planner=_Planner(point, environment),
            policy=POLICY,
            workload_kind=WorkloadKind.AGENTX,
        )
        await tb.toolbox.start()
        request_id = await tb.toolbox.submit_test(req())
        await asyncio.wait_for(tb.toolbox.drain(), 120)
        entry = tb.toolbox._log.get(request_id)
        assert entry is not None and entry.outcome is not None
        journal = rig.backend.journal.latest(request_id)
        return TestResult.from_document(entry.outcome["result"]), backend, journal
    finally:
        rig.backend.close()
        rig.provider.close()


async def test_a_broken_stack_fails_as_the_candidate_and_no_sandbox_is_asked_for(
    tmp_path: Path, framework: Framework
) -> None:
    environment = _environment(tmp_path, framework, BROKEN_FIRST, MERGE_OWN)
    result, backend, journal = await _through_toolbox(
        tmp_path, environment, MergeSmoke(default_python=framework.python)
    )
    assert result.validity is Validity.CANDIDATE_FAILED and result.failure_class is FailureClass.CANDIDATE
    assert (result.failure_reason or "").startswith(f"{MERGE_SMOKE}: sglang/srt/a.py:1: SyntaxError: ")
    assert backend.calls == 0 and journal is None and len(result.attempts) == 1
    assert result.gpu_seconds == 0 and result.sandbox is None


async def test_an_import_error_fails_as_the_candidate(tmp_path: Path, framework: Framework) -> None:
    environment = _environment(tmp_path, framework, {"srt/a.py": (OLD_A, "VALUE = undefined_name\n")})
    result, backend, _ = await _through_toolbox(tmp_path, environment, MergeSmoke(default_python=framework.python))
    assert result.validity is Validity.CANDIDATE_FAILED and backend.calls == 0
    assert result.failure_reason == (
        f"{MERGE_SMOKE}: import sglang: NameError: name 'undefined_name' is not defined (sglang/srt/a.py:1)"
    )


async def test_a_device_probe_with_no_gpu_goes_on_to_its_measurement(tmp_path: Path, framework: Framework) -> None:
    environment = _environment(tmp_path, framework, DEVICE_PROBE)
    result, backend, journal = await _through_toolbox(
        tmp_path, environment, MergeSmoke(default_python=framework.python)
    )
    assert result.validity is Validity.SUCCEEDED and backend.calls == 1 and journal is not None


async def test_a_good_stack_goes_on_to_its_measurement(tmp_path: Path, framework: Framework) -> None:
    environment = _environment(tmp_path, framework, MERGE_OWN)
    result, backend, journal = await _through_toolbox(
        tmp_path, environment, MergeSmoke(default_python=framework.python)
    )
    assert result.validity is Validity.SUCCEEDED and backend.calls == 1 and journal is not None


async def test_a_smoke_that_cannot_run_is_infrastructure_never_the_candidate(
    tmp_path: Path, framework: Framework
) -> None:
    environment = _environment(tmp_path, framework, BROKEN_FIRST)
    smoke = MergeSmoke(default_python=str(tmp_path / "no-such-python"))
    result, backend, journal = await _through_toolbox(tmp_path, environment, smoke)
    assert result.validity is Validity.MEASUREMENT_FAILED and result.failure_class is FailureClass.INFRA
    assert (result.failure_reason or "").startswith(f"{SMOKE_NOT_RUN}: the framework interpreter ")
    assert backend.calls == 0 and journal is None
    assert len(result.attempts) == 2, "a smoke that could not run is tried once more, then the test ends"


async def test_a_smoke_past_its_timeout_is_infrastructure(tmp_path: Path, framework: Framework) -> None:
    environment = _environment(tmp_path, framework, {"srt/a.py": (OLD_A, "import time\ntime.sleep(60)\n")})
    smoke = MergeSmoke(default_python=framework.python, timeout_sec=2)
    result, backend, _ = await _through_toolbox(tmp_path, environment, smoke)
    assert result.failure_class is FailureClass.INFRA and backend.calls == 0
    assert "timed out after 2 s" in (result.failure_reason or "")


@dataclass(frozen=True)
class _CountingSmoke(MergeSmoke):
    checks: list[int] = field(default_factory=list)

    def check(self, environment: object, points: tuple[Path, ...]) -> str | None:  # type: ignore[override]
        self.checks.append(1)
        return super().check(environment, points)


async def test_a_retry_after_a_passed_smoke_is_not_smoked_again(tmp_path: Path, framework: Framework) -> None:
    environment = _environment(tmp_path, framework, MERGE_OWN)
    smoke = _CountingSmoke(default_python=framework.python)
    result, backend, _ = await _through_toolbox(tmp_path, environment, smoke, first={"fake_mode": "node_fault"})
    assert result.validity is Validity.SUCCEEDED and backend.calls == 2 and len(result.attempts) == 2
    assert len(smoke.checks) == 1, "the second sandbox was not held back for a smoke that already passed"


def test_the_smoke_runs_in_a_directory_of_its_own(
    tmp_path: Path, framework: Framework, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    assert _check(tmp_path, framework, MERGE_OWN) is None
    assert os.listdir(scratch) == [], "each check removes what it laid"


def test_an_import_a_native_stack_s_package_may_answer_is_not_judged(tmp_path: Path) -> None:
    """A stack that also patches a native component (a ``runtime`` root, whose package is not laid until its build is
    ready): the patched framework may import what that package adds, which the image's copy lacks, so its import
    failing says nothing about the candidate."""
    framework = _framework(tmp_path / "framework", IMAGE_FILES)
    kernels = Path(framework.package).parent / "kern"
    kernels.mkdir()
    (kernels / "__init__.py").write_text("OLD = 1\n")
    caller = {"__init__.py": (INIT, "from kern import NEW\n" + INIT)}
    environment = _environment(tmp_path, framework, caller)
    blob = LocalBlobStore(environment.blob_root).put(b"OLD = 1\nNEW = 2\n")
    native = EnvEntry("__init__.py", "upsert", blob, "100644")
    runtime = EnvRoot("kern", "kern", "kern", "0.0.1", (native,), kind="runtime")
    with_native = replace(environment, roots=(*environment.roots, runtime))
    point = point_yaml(tmp_path, "smoke-point.yaml")
    assert MergeSmoke(default_python=framework.python).check(with_native, (point,)) is None
    # Without the native stack the same import failure is the candidate's.
    error = MergeSmoke(default_python=framework.python).check(environment, (point,))
    assert error is not None and error.startswith("import sglang: ImportError: cannot import name 'NEW'"), error
