# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a ``kernel_tune`` job's tuner builds stays with the job: its ``AITER_JIT_DIR`` is a directory below the job's
holding links to the image's prebuilt modules, and the image's ``aiter/jit`` is never written. Its temporary directory is
the one it inherits, where Forge looks for the comparison files. Forge is a stand-in that builds a module and writes a
comparison file where aiter would, through the variables it was given."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import pytest_asyncio

from hyperloom.orchestrator.actions.executors import RoundWatch
from hyperloom_multibranch.toolbox.local_hosted import LocalHostedToolbox
from hyperloom_multibranch.toolbox.tests.kernel_tune_rig import TEST_ID, FakeForge, Make, build_rig, record_test
from hyperloom_multibranch.toolbox.tools import JobStatus

PREBUILT = "module_aiter_core.so"
BUILT = "module_gemm_a8w8_blockscale_bpreshuffle_tune.so"


@dataclass
class BuildingForge(FakeForge):
    """A tuner that builds a module where aiter would install it (``AITER_JIT_DIR``, else the ``jit`` of the aiter it is
    pinned to) and writes its comparison files below the temporary directory ``tempfile`` would pick, recording what it
    found there while it ran."""

    __test__ = False
    seen: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, cmd: list[str], *, env: dict[str, str], watch: RoundWatch) -> subprocess.CompletedProcess[str]:
        jit = Path(env.get("AITER_JIT_DIR") or Path(env["AITER_ROOT_DIR"]) / "aiter" / "jit")
        temporary = Path(env.get("TMPDIR") or tempfile.gettempdir())
        self.seen.append({"links": sorted(p.name for p in jit.iterdir()), "tmp_is_dir": temporary.is_dir()})
        (jit / BUILT).write_bytes(b"built by the tuner")
        (temporary / "aiter_compare").mkdir(exist_ok=True)
        (temporary / "aiter_compare" / "shape0.pt").write_bytes(b"x")
        return super().__call__(cmd, env=env, watch=watch)


@pytest_asyncio.fixture
async def make(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[Make, BuildingForge]]:
    made: list[LocalHostedToolbox] = []
    forge = BuildingForge()
    yield build_rig(tmp_path, monkeypatch, made, forge), forge
    for box in made:
        if not box._closed:
            await box.withdraw_all()
            await box.close()


def image_jit(configs: Path) -> Path:
    """The image's ``aiter/jit`` with one prebuilt module in it."""
    jit = configs.parent / "jit"
    (jit / PREBUILT).write_bytes(b"prebuilt by the image")
    return jit


def listing(directory: Path) -> dict[str, tuple[bytes, int]]:
    return {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in directory.iterdir()}


@pytest.mark.parametrize("fail", [False, True], ids=["tuned", "tuner-failed"])
async def test_the_tuner_builds_into_a_jit_directory_of_the_job_and_never_into_the_image(
    make: tuple[Make, BuildingForge], monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    build, forge = make
    forge.fail = fail
    monkeypatch.delenv("AITER_JIT_DIR", raising=False)
    rig = await build(cards=2)
    jit = image_jit(rig.configs)
    before = listing(jit)
    rig.landed()
    await record_test(rig.blackboard)
    job = await rig.submit(test_request_id=TEST_ID)

    assert await rig.finished(job) is (JobStatus.FAILED if fail else JobStatus.SUCCEEDED)
    ((_, env),) = forge.calls
    own = rig.session / "tool-results" / job / "aiter_jit"
    assert listing(jit) == before, "the image's aiter/jit is byte-identical, with no new file"
    assert env.get("AITER_JIT_DIR") == str(own), "below the job's directory"
    assert forge.seen[0]["links"] == [PREBUILT], "the image's prebuilt modules were linked in before the tuner ran"
    assert os.readlink(own / PREBUILT) == str(jit / PREBUILT), "linked, not copied"
    assert (own / BUILT).read_bytes() == b"built by the tuner", "what the tuner built stays with the job"


async def test_the_tuner_keeps_the_temporary_directory_it_inherits_where_forge_reads_the_comparison_files(
    make: tuple[Make, BuildingForge], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build, forge = make
    inherited = tmp_path / "inherited-tmp"
    inherited.mkdir()
    monkeypatch.setenv("TMPDIR", str(inherited))
    rig = await build(cards=2)
    rig.landed()
    await record_test(rig.blackboard)
    job = await rig.submit(test_request_id=TEST_ID)
    assert await rig.finished(job) is JobStatus.SUCCEEDED

    ((_, env),) = forge.calls
    assert env.get("TMPDIR") == str(inherited), (
        "Forge reads the literal /tmp/aiter_compare, never a TMPDIR of the job's"
    )
    assert (inherited / "aiter_compare" / "shape0.pt").is_file()
