# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""#125 stage 2b through acceptance and the build queue, with the stand-in aiter: a stack that touches one of aiter's
prebuilt modules needs a CPU build of it, waits for it as ``pending_build`` without a card, and once it is ready the
sheet names it; a stack whose ``csrc`` does not compile ends as the candidate's ``compile_error`` before any card is
taken. A host that configures no build store compiles as before (2a): the server builds the module on its card."""

from __future__ import annotations

import asyncio
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.contracts.measurement import FailureClass
from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure import farm as farm_module
from hyperloom.measure import signing
from hyperloom.measure.compile import EnvironmentCompiler, LocalBlobStore
from hyperloom.measure.jit.native.record import FAILED, READY
from hyperloom.measure.jit.native.store import BuildStore
from hyperloom.measure.jit.native.trust import Signer, Trust, TrustedKey
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.runner.native import aiter_modules
from hyperloom.measure.runner.native.aiter_modules import AITER_COMPONENT, AiterModules
from hyperloom.measure.runner.native.builder import BuildMachine
from hyperloom.measure.runner.native.isolated import IsolatedBuilds, Isolation
from hyperloom.measure.runner.native.planning import NativePlanner, NativeSite
from hyperloom.measure.runner.native.tests import _aiter_standin as standin
from hyperloom.measure.spec import Environment
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with
from hyperloom_multibranch.toolbox.native_builds import compiled_needs, native_builds, with_native_plan
from hyperloom_multibranch.toolbox.pending_builds import PendingBuilds
from hyperloom_multibranch.toolbox.store_builds import BuildSite, StoreBuildDriver
from hyperloom_multibranch.toolbox.tests.test_native_builds import REQUEST, Hooks

BUILDER = "sha256:" + "c" * 64
PATCHED = "// patched k.cu\n"


class AiterRig:
    """One session over the stand-in aiter image: the compiler with the module plan, the store and the driver."""

    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp, self.blobs = tmp, LocalBlobStore(tmp / "blobs")
        self.package = standin.aiter_image(tmp / "image")
        locate = lambda name, python, cwd: self.package.parent / name  # noqa: E731
        monkeypatch.setattr(aiter_modules, "locate_package", locate)
        monkeypatch.setattr(farm_module, "locate_package", locate)
        key = signing.generate_private_key()
        self.store = BuildStore(tmp / "store", Trust((TrustedKey(signing.public_key_pem(key), 1),), 1))
        machine = BuildMachine(IMAGE, BUILDER, "gfx942", sys.executable)
        planner = NativePlanner(NativeSite(machine, {}, tmp / "sources", tmp / "image-root"), self.blobs)
        history = partial(self.store.ready_records, AITER_COMPONENT, IMAGE)
        self.modules = AiterModules(planner, history=history, environ={})
        runner = IsolatedBuilds(planner, self.blobs, Isolation((), None), {AITER_COMPONENT: standin.FACTORY})
        site = BuildSite(IMAGE, BUILDER, "gfx942", {AITER_COMPONENT: standin.recipe()})
        self.driver = StoreBuildDriver(self.store, Signer(key), site, compiled_needs, runner)
        self.compiler = EnvironmentCompiler(
            self.blobs, IMAGE, lambda layout: standin.package_base(self.package), modules=self.modules
        )

    def compile(self, text: str, *, modules: bool = True):
        patch = git_patch(self.tmp / "up", {standin.KERNEL: standin.IMAGE_KERNEL}, {standin.KERNEL: text})
        recipe = recipe_with((put_patch(self.blobs, "k.patch", patch),), component="aiter")
        compiler = (
            self.compiler
            if modules
            else EnvironmentCompiler(self.blobs, IMAGE, lambda layout: standin.package_base(self.package))
        )
        return compiler.compile(recipe, RecipeDelta())

    async def wait(self, compiled, hooks: Hooks) -> PendingBuilds:
        builds = PendingBuilds(self.driver, hooks.hooks(), max_infra_retries=0, retry_seconds=0.0)
        needs = builds.plan(REQUEST, AcceptedMeasurement(compiled, ()))
        assert builds.wait("r1", needs), "the build is not stored yet: the test waits as pending_build"
        await asyncio.wait_for(hooks.done.wait(), 300)
        return builds


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AiterRig:
    return AiterRig(tmp_path, monkeypatch)


async def test_a_touched_module_waits_for_its_cpu_build_and_the_sheet_names_it_once_ready(rig: AiterRig) -> None:
    compiled = rig.compile(PATCHED)
    (need,) = compiled.builds
    assert (need.root_id, need.inputs.component) == ("aiter", AITER_COMPONENT)

    hooks = Hooks()
    builds = await rig.wait(compiled, hooks)

    assert hooks.ready == ["r1"] and hooks.ended == []
    record = rig.store.read(need.inputs)
    assert record is not None and record.status == READY
    assert [o.path for o in record.outputs] == [f"{standin.MODULE}.so"]
    sheet = builds.resolved(AcceptedMeasurement(compiled, ())).environment.to_document()
    environment = Environment.parse(sheet, "environment")
    assert [(b.root_id, b.key) for b in environment.builds] == [("aiter", record.key)]
    assert {root.id: root.kind for root in environment.roots} == {"aiter": "package", "aiter.meta": "package"}


async def test_a_csrc_that_does_not_compile_is_the_candidates_compile_error_before_any_card(rig: AiterRig) -> None:
    compiled = rig.compile("int a;\n#error W211 broken\n")

    hooks = Hooks()
    await rig.wait(compiled, hooks)

    ((request_id, end),) = hooks.ended
    assert request_id == "r1" and end.failure_class is FailureClass.CANDIDATE
    assert end.reason.startswith("compile_error: aiter ")
    assert "site/aiter_meta/csrc/kernels/k.cu:2:2: error: W211 broken" in end.reason
    record = rig.store.read(compiled.builds[0].inputs)
    assert record is not None and record.status == FAILED, "stored, so every test of the stack gets the same answer"


def test_without_the_module_plan_the_stack_compiles_as_before_and_needs_no_build(rig: AiterRig) -> None:
    compiled = rig.compile(PATCHED, modules=False)

    assert compiled.builds == ()
    assert [root.id for root in compiled.roots] == ["aiter", "aiter.meta"]


def test_a_host_with_no_build_store_keeps_its_acceptance_as_it_was(tmp_path: Path) -> None:
    native = native_builds({}, LocalBlobStore(tmp_path / "blobs"), IMAGE)
    acceptance = SimpleNamespace(planner=object())

    assert native is None and with_native_plan(acceptance, native) is acceptance  # type: ignore[arg-type]
