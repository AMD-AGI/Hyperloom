# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""#125 stage 4b end to end with a stand-in compiler: a stack with a ``.cu`` edit compiles to a ``runtime`` root that waits
as ``pending_build``, is built out of process, signed, named on the sheet, laid in the farm and imported from it."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from hyperloom.contracts.measurement import BuildPayment, FailureClass, Purpose, TestKind, TestRequest, Validity
from hyperloom.contracts.recipe import ComponentPatches, Recipe, RecipeDelta
from hyperloom.measure import signing
from hyperloom.measure.compile import EnvironmentCompiler, LocalBlobStore, PatchApplyError, UnsupportedComponentError
from hyperloom.measure.jit.native.needs import BuildNotReady, BuildRef, RuntimeSourcesConflict
from hyperloom.measure.farm import FarmError, build_farm, built_roots, prepend_pythonpath, verify_farm
from hyperloom.measure.jit.native.config import (
    BUILD_STORE_ENV,
    BUILD_TRUST_ENV,
    BuildStoreConfigError,
    trust_from_file,
)
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.laying import NamedBuild, SheetBuilds, manifest_sha256
from hyperloom.measure.jit.native.store import BuildInfraError, BuildStore
from hyperloom.measure.jit.native.trust import Signer, Trust, TrustedKey
from hyperloom.measure.points import AcceptedMeasurement
from hyperloom.measure.runner.native.aiter_modules import AiterModules
from hyperloom.measure.runner.native.builder import BuildMachine
from hyperloom.measure.runner.native.isolated import STAGED_FILE, IsolatedBuilds, Isolation
from hyperloom.measure.runner.native.isolated import staged_record
from hyperloom.measure.runner.native.planning import NativePlanner, NativeSite
from hyperloom.measure.runner.native.service import Ensured
from hyperloom.measure.spec import Environment
from hyperloom.measure.tests._recipes import COMMIT, IMAGE, git_patch, put_patch
from hyperloom_multibranch.contracts.toolbox import RequestRejected, WorkloadKind
from hyperloom_multibranch.toolbox import native_builds as native_builds_module
from hyperloom_multibranch.toolbox.acceptance import Acceptance, AcceptanceSetup
from hyperloom_multibranch.toolbox.build_preflight import PreflightSite
from hyperloom_multibranch.toolbox.native_builds import (
    BUILD_GFX_ENV,
    BUILD_IMAGE_ROOT_ENV,
    BUILD_SIGNING_KEY_ENV,
    BUILD_SOURCES_ENV,
    compiled_needs,
    native_builds,
)
from hyperloom_multibranch.toolbox.pending_builds import BuildEnd, BuildHooks, PendingBuilds
from hyperloom_multibranch.toolbox.planner import HostedPlanner
from hyperloom_multibranch.toolbox.store_builds import BuildSite, StoreBuildDriver
from hyperloom_multibranch.toolbox.tests import native_support as fake
from hyperloom_multibranch.toolbox.tests.helpers import FakeRepo, build, req
from hyperloom_multibranch.toolbox.tests import test_short_prompts_option as options_tests

BUILDER = "sha256:" + "c" * 64
#: The compile's own process, for the stand-in: a deployment isolates it (``local_isolation``).
SAME_PROCESS_USER = Isolation((), None)


@dataclass
class Hooks:
    ready: list[str] = field(default_factory=list)
    ended: list[tuple[str, BuildEnd]] = field(default_factory=list)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def hooks(self) -> BuildHooks:
        async def ready(request_id: str) -> None:
            self.ready.append(request_id)
            self.done.set()

        async def ended(request_id: str, end: BuildEnd) -> None:
            self.ended.append((request_id, end))
            self.done.set()

        async def notify(notice: object) -> None:
            return None

        return BuildHooks(ready=ready, ended=ended, notify=notify)


class Rig:
    """One session: blobs, the image, the upstream checkout, the store and its key, the compiler and the driver.

    ``kit`` is the stand-in component (:mod:`native_support`'s ``fakepkg`` by default): its ``COMPONENT``, ``LAYOUT``,
    ``FACTORY``, ``ACT_CU`` and ``BASE_CU``, its ``image``, ``upstream`` and ``stand_in_recipe``."""

    def __init__(self, tmp: Path, isolation: Isolation | None = SAME_PROCESS_USER, kit: ModuleType = fake) -> None:
        self.tmp, self.blobs, self.kit, self._isolation = tmp, LocalBlobStore(tmp / "blobs"), kit, isolation
        self.image = kit.image(tmp / "image")
        kit.upstream(tmp / "sources" / kit.COMPONENT / COMMIT)
        self._key = signing.generate_private_key()
        self.trust = Trust((TrustedKey(signing.public_key_pem(self._key), 1),), 1)
        self.store = BuildStore(tmp / "store", self.trust)
        self._wire()
        self._patches = 0

    def _wire(self) -> None:
        """What lives in the service's process: the planner, the compiler and the driver."""
        kit, recipe = self.kit, self.kit.stand_in_recipe()
        machine = BuildMachine(IMAGE, BUILDER, "gfx942", sys.executable)
        site = NativeSite(machine, {kit.COMPONENT: recipe}, self.tmp / "sources", self.image.site)
        self.planner = NativePlanner(site, self.blobs, {kit.COMPONENT: kit.LAYOUT})
        runner = IsolatedBuilds(self.planner, self.blobs, self._isolation, {kit.COMPONENT: kit.FACTORY})
        build_site = BuildSite(IMAGE, BUILDER, "gfx942", {kit.COMPONENT: recipe})
        self.driver = StoreBuildDriver(self.store, Signer(self._key), build_site, compiled_needs, runner)
        self.compiler = EnvironmentCompiler(
            self.blobs, IMAGE, lambda layout: self.image.package(), {kit.COMPONENT: kit.LAYOUT}, self.planner
        )

    def restarted(self) -> Rig:
        """The same host after the service's process restarted: its store, key, sources and blobs, nothing planned."""
        again = copy.copy(self)
        again._wire()
        return again

    def patch(self, after: dict[str, str | None], before: dict[str, str] | None = None):
        self._patches += 1
        data = git_patch(self.tmp / f"patch-{self._patches}", before or {self.kit.ACT_CU: self.kit.BASE_CU}, after)
        return put_patch(self.blobs, f"{self._patches:04d}.patch", data)

    def recipe(self, *patches, **fields) -> Recipe:
        stack = ComponentPatches(self.kit.COMPONENT, "https://example.invalid/fake.git", COMMIT, tuple(patches))
        return Recipe(base_image_digest=IMAGE, components=(stack,), **fields)

    async def wait(self, accepted: AcceptedMeasurement, hooks: Hooks) -> PendingBuilds:
        builds = PendingBuilds(self.driver, hooks.hooks(), max_infra_retries=0, retry_seconds=0.0)
        needs = builds.plan(REQUEST, accepted)
        assert builds.wait("r1", needs), "the build is not stored yet: the test waits as pending_build"
        await asyncio.wait_for(hooks.done.wait(), 120)
        return builds


REQUEST = TestRequest(
    kind=TestKind.TEST, purpose=Purpose.EXPLORE, branch_id="b1", parent_commit="c0", delta=RecipeDelta()
)
EDITED_CU = "__global__ void act() { /* faster */ }\n"
OTHER_IMAGE = "sha256:" + "d" * 64
BASELINE = TestRequest(
    kind=TestKind.TEST, purpose=Purpose.BASELINE, branch_id="b1", parent_commit="c0", delta=RecipeDelta()
)


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path)


def farm_of(rig: Rig, accepted: AcceptedMeasurement):
    """The sheet's environment as a runner reads it, and the farm it lays."""
    environment = Environment.parse(accepted.environment.to_document(), "environment")
    named = {b.root_id: NamedBuild(b.root_id, b.key, b.manifest_sha256) for b in environment.builds}
    farm = build_farm(
        built_roots(environment.roots, SheetBuilds(rig.store, named, IMAGE)),
        rig.tmp / "farm",
        python=rig.image.python,
        read_blob=lambda address: rig.blobs.get(address.removeprefix("sha256:")),
    )
    return environment, farm


async def test_a_cu_edit_waits_for_its_build_and_is_laid_in_the_farm_and_imported_from_it(rig: Rig) -> None:
    stack = rig.recipe(
        rig.patch(
            {fake.ACT_CU: EDITED_CU, fake.PY_FILE: "SCALE = 2\n"},
            {fake.ACT_CU: fake.BASE_CU, fake.PY_FILE: fake.BASE_PY},
        )
    )
    compiled = rig.compiler.compile(stack, RecipeDelta())
    (root,) = compiled.roots
    assert (root.kind, [e.path for e in root.entries]) == ("runtime", ["ops.py"]), "only the package's own files"
    with pytest.raises(BuildNotReady):
        compiled.to_document()  # no sheet names a build before it is ready

    hooks = Hooks()
    builds = await rig.wait(AcceptedMeasurement(compiled, ()), hooks)
    assert hooks.ready == ["r1"] and hooks.ended == []
    accepted = builds.resolved(AcceptedMeasurement(compiled, ()))

    environment, farm = farm_of(rig, accepted)
    record = rig.store.read(compiled.builds[0].inputs)
    assert [(b.key, b.manifest_sha256) for b in environment.builds] == [(record.key, manifest_sha256(record))]
    output = farm.site / fake.COMPONENT / fake.extension().name
    assert output.read_bytes() == fake.compiled(EDITED_CU) and not output.is_symlink(), "copied, not linked"
    assert (farm.site / fake.COMPONENT / "ops.py").read_text() == "SCALE = 2\n"
    envs: dict = {}
    prepend_pythonpath(envs, farm.site)
    verify_farm(farm, envs, python=rig.image.python, cwd=rig.tmp)
    assert [m.name for m in farm.native] == [fake.module_name()]


async def test_a_broken_cu_is_the_candidates_compile_error_naming_the_cu_line(rig: Rig) -> None:
    compiled = rig.compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: "int a;\n#error W191 broken\n"})), RecipeDelta())

    hooks = Hooks()
    await rig.wait(AcceptedMeasurement(compiled, ()), hooks)

    ((request_id, end),) = hooks.ended
    assert request_id == "r1" and end.failure_class is FailureClass.CANDIDATE
    assert end.reason.startswith(f"compile_error: {fake.COMPONENT} ")
    assert "src/act.cu:2:2 (src/act.hip:4:2): error: W191 broken" in end.reason


async def test_a_host_that_cannot_isolate_the_compile_builds_nothing_and_fails_as_the_machine(tmp_path: Path) -> None:
    rig = Rig(tmp_path, isolation=None)
    compiled = rig.compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})), RecipeDelta())

    hooks = Hooks()
    await rig.wait(AcceptedMeasurement(compiled, ()), hooks)

    ((_, end),) = hooks.ended
    assert end.failure_class is FailureClass.INFRA and "cannot isolate a native build" in end.reason
    assert not any((tmp_path / "store").glob("builds/*/*/manifest.json"))


def test_without_a_native_plan_a_cu_edit_is_refused_as_before(rig: Rig) -> None:
    compiler = EnvironmentCompiler(rig.blobs, IMAGE, lambda layout: rig.image.package(), {fake.COMPONENT: fake.LAYOUT})

    with pytest.raises(PatchApplyError, match=r"'native/src/act.cu' is a native or binary file \(\*\.cu\)"):
        compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})), RecipeDelta())


def test_a_python_only_stack_stays_a_package_root_with_no_build(rig: Rig) -> None:
    stack = rig.recipe(rig.patch({fake.PY_FILE: "SCALE = 2\n"}, {fake.PY_FILE: fake.BASE_PY}))

    compiled = rig.compiler.compile(stack, RecipeDelta())

    assert [r.kind for r in compiled.roots] == ["package"] and compiled.builds == ()
    assert "builds" not in compiled.to_document()


@pytest.mark.parametrize(
    "path",
    ["native/CMakeLists.txt", "native/setup.py", "native/cmake/flags.txt"],
    ids=["cmakelists", "setup", "cmake-dir"],
)
def test_a_change_to_the_build_definition_is_refused_at_submission_naming_the_path(rig: Rig, path: str) -> None:
    patch = rig.patch({fake.ACT_CU: EDITED_CU, path: "changed\n"}, {fake.ACT_CU: fake.BASE_CU, path: "original\n"})

    with pytest.raises(PatchApplyError, match=f"{path!r} is .*build definition"):
        rig.compiler.compile(rig.recipe(patch), RecipeDelta())


def test_a_stack_whose_upstream_commit_has_no_checkout_is_refused(rig: Rig) -> None:
    stack = rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU}))
    other = Recipe(
        base_image_digest=IMAGE,
        components=(
            ComponentPatches(fake.COMPONENT, "https://example.invalid/fake.git", "d" * 40, stack.components[0].patches),
        ),
    )

    with pytest.raises(UnsupportedComponentError, match="no upstream checkout"):
        rig.compiler.compile(other, RecipeDelta())


def test_a_runtime_snapshot_beside_a_native_stack_is_refused(rig: Rig) -> None:
    recipe = rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU}), runtime_snapshot=IMAGE)

    with pytest.raises(RuntimeSourcesConflict, match="runtime snapshot .* and native stacks \\(fakepkg\\)"):
        rig.compiler.compile(recipe, RecipeDelta())


class CompilingPlanner:
    """Acceptance's planner over the rig's compiler: no points, the compiled environment only."""

    def __init__(self, rig: Rig) -> None:
        self.rig, self.blobs = rig, rig.blobs

    def plan(self, request: TestRequest, parent: Recipe) -> AcceptedMeasurement:
        return AcceptedMeasurement(self.rig.compiler.compile(parent, request.delta), ())


class GatedDriver:
    """``driver`` whose builds wait for ``gate``: a build still running when the service's process dies."""

    def __init__(self, driver: StoreBuildDriver) -> None:
        self.driver, self.gate = driver, asyncio.Event()

    def __getattr__(self, name: str) -> object:
        return getattr(self.driver, name)

    async def ensure(self, inputs: BuildInputs) -> Ensured:
        await self.gate.wait()
        return await self.driver.ensure(inputs)


async def test_a_test_waiting_for_its_build_when_the_service_restarts_is_planned_again_built_and_measured(
    rig: Rig,
) -> None:
    repo = FakeRepo(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})))
    before = build(
        rig.tmp / "box", repo=repo, active_image=IMAGE, planner=CompilingPlanner(rig), builds=GatedDriver(rig.driver)
    )
    request_id = await before.toolbox.submit_test(req(purpose=Purpose.BASELINE, change=RecipeDelta()))
    assert (await before.toolbox.progress(request_id)).state == "pending_build"  # type: ignore[union-attr]
    await before.toolbox.close()  # the process dies before the build ran: nothing was stored

    again = rig.restarted()
    after = build(rig.tmp / "box", repo=repo, active_image=IMAGE, planner=CompilingPlanner(again), builds=again.driver)
    await after.toolbox.start()
    await asyncio.wait_for(after.toolbox.drain(), 120)

    result = (await after.notebook.record(request_id)).result
    assert result is not None and result.validity is Validity.SUCCEEDED, result
    assert after.executor.started == [request_id]
    assert [charge.payment for charge in result.builds] == [BuildPayment.PAID], "built once, by this process"
    await after.toolbox.close()


async def test_acceptance_refuses_a_runtime_snapshot_beside_a_native_stack_and_takes_one_without(rig: Rig) -> None:
    def acceptance(recipe: Recipe) -> Acceptance:
        setup = AcceptanceSetup(FakeRepo(recipe), CompilingPlanner(rig), IMAGE, WorkloadKind.AGENTX)
        return Acceptance(setup, is_accepted=lambda request_id: False)

    native = rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU}), runtime_snapshot=OTHER_IMAGE)
    with pytest.raises(RequestRejected, match="runtime snapshot .* and native stacks \\(fakepkg\\)") as refused:
        await acceptance(native).prepare(BASELINE)
    assert refused.value.code == "delta_not_applicable"

    python_only = rig.recipe(
        rig.patch({fake.PY_FILE: "SCALE = 2\n"}, {fake.PY_FILE: fake.BASE_PY}), runtime_snapshot=OTHER_IMAGE
    )
    prepared = await acceptance(python_only).prepare(BASELINE)
    assert prepared.environment.environment.builds == (), "a snapshot of another image is no refusal on its own"


async def test_the_farm_refuses_a_build_the_sheet_does_not_name_exactly(rig: Rig) -> None:
    compiled = rig.compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})), RecipeDelta())
    builds = await rig.wait(AcceptedMeasurement(compiled, ()), Hooks())
    accepted = builds.resolved(AcceptedMeasurement(compiled, ()))
    environment = Environment.parse(accepted.environment.to_document(), "environment")
    (ref,) = environment.builds

    def lay(named: NamedBuild, image: str = IMAGE):
        return built_roots(environment.roots, SheetBuilds(rig.store, {named.root_id: named}, image))

    with pytest.raises(FarmError, match="has the manifest"):
        lay(NamedBuild(ref.root_id, ref.key, "0" * 64))
    with pytest.raises(FarmError, match="was built for"):
        lay(NamedBuild(ref.root_id, ref.key, ref.manifest_sha256), "sha256:" + "e" * 64)
    with pytest.raises(FarmError, match="is not in the store"):
        lay(NamedBuild(ref.root_id, "f" * 64, ref.manifest_sha256))
    with pytest.raises(FarmError, match="needs the native build store"):
        built_roots(environment.roots, None)
    with pytest.raises(FarmError, match="needs its build"):
        build_farm(environment.roots, rig.tmp / "none", python=rig.image.python, read_blob=lambda a: b"")


async def test_verify_farm_refuses_a_native_module_loaded_from_elsewhere(rig: Rig) -> None:
    compiled = rig.compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})), RecipeDelta())
    builds = await rig.wait(AcceptedMeasurement(compiled, ()), Hooks())
    _, farm = farm_of(rig, builds.resolved(AcceptedMeasurement(compiled, ())))
    (module,) = farm.native
    (module.path).unlink()  # the package is the farm's, its native module now the image's

    envs: dict = {}
    prepend_pythonpath(envs, farm.site)
    with pytest.raises(FarmError, match=f"native module {module.name} would be loaded from"):
        verify_farm(farm, envs, python=rig.image.python, cwd=rig.tmp)


# --- the build runner -------------------------------------------------------------------------------------------------


async def test_the_build_runners_child_is_given_none_of_this_processs_variables(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYPERLOOM_BUILD_SIGNING_KEY", "/run/secrets/key")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:1")
    compiled = rig.compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})), RecipeDelta())

    await rig.wait(AcceptedMeasurement(compiled, ()), Hooks())

    seen = rig.store.read(compiled.builds[0].inputs).toolchain["variables"].split()
    assert "HYPERLOOM_BUILD_SIGNING_KEY" not in seen and "HTTPS_PROXY" not in seen
    assert set(seen) <= {"PATH", "HOME", "TMPDIR", "LC_ALL", "LC_CTYPE"}


def _staged_from(tmp_path: Path, write) -> Path:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    write(work)
    return work


def test_what_the_child_staged_is_read_as_untrusted_data(tmp_path: Path) -> None:
    secret = tmp_path / "key.pem"
    secret.write_text('{"status": "ready"}')
    linked = _staged_from(tmp_path, lambda work: (work / STAGED_FILE).symlink_to(secret))
    with pytest.raises(BuildInfraError, match="staged nothing readable"):
        staged_record(linked, (1.0, 1.0))
    infra = _staged_from(tmp_path / "b", lambda work: (work / STAGED_FILE).write_text('{"infra": "disk full"}'))
    with pytest.raises(BuildInfraError, match="the build runner failed: disk full"):
        staged_record(infra, (1.0, 1.0))
    extra = '{"status": "ready", "outputs": [], "toolchain": {}, "log": "", "failure": "", "build_seconds": 0}'
    lying = _staged_from(tmp_path / "c", lambda work: (work / STAGED_FILE).write_text(extra))
    with pytest.raises(BuildInfraError, match="outside the schema"):
        staged_record(lying, (1.0, 1.0))


# --- configuration --------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def ready_host(monkeypatch: pytest.MonkeyPatch) -> list[PreflightSite]:
    """The host checks pass here (test_build_preflight has them); what they were asked about is recorded."""
    asked: list[PreflightSite] = []
    monkeypatch.setattr(native_builds_module, "build_preflight", asked.append)
    return asked


def _configured(tmp_path: Path, **overrides: str) -> dict[str, str]:
    key = signing.generate_private_key()
    tmp_path.mkdir(parents=True, exist_ok=True)
    key_file = tmp_path / "service.key"
    key_file.write_bytes(signing.private_key_pem(key))
    key_file.chmod(0o600)
    trust = tmp_path / "trust.json"
    trust.write_text(
        json.dumps({"min_epoch": 1, "keys": [{"public_key_pem": signing.public_key_pem(key), "epoch": 1}]})
    )
    environ = {
        BUILD_STORE_ENV: str(tmp_path / "store"),
        BUILD_TRUST_ENV: str(trust),
        BUILD_SIGNING_KEY_ENV: str(key_file),
        BUILD_SOURCES_ENV: str(tmp_path / "sources"),
        BUILD_IMAGE_ROOT_ENV: str(tmp_path / "image"),
        BUILD_GFX_ENV: "gfx942",
    }
    return {name: value for name, value in {**environ, **overrides}.items() if value}


def test_nothing_configured_means_no_native_builds(tmp_path: Path) -> None:
    assert native_builds({}, LocalBlobStore(tmp_path), IMAGE) is None


def test_a_configured_store_gives_the_driver_the_plan_and_the_runners_the_store(tmp_path: Path) -> None:
    environ = _configured(tmp_path)

    native = native_builds(environ, LocalBlobStore(tmp_path / "blobs"), f"registry.invalid/fw@{IMAGE}")

    assert native is not None
    assert dict(native.runner_envs) == {
        BUILD_STORE_ENV: environ[BUILD_STORE_ENV],
        BUILD_TRUST_ENV: environ[BUILD_TRUST_ENV],
    }
    assert native.plan.site.machine.image_digest == IMAGE and native.plan.site.machine.builder_image_digest == IMAGE
    assert set(native.plan.site.recipes) == {"sgl_kernel", "vllm"}, "every recipe, vLLM's included"
    planner = HostedPlanner(EnvironmentCompiler(LocalBlobStore(tmp_path / "blobs"), IMAGE), None, tmp_path)  # type: ignore[arg-type]
    compiler = planner.with_native(native.plan, native.modules)._compiler.compiler
    assert compiler.native is native.plan and isinstance(compiler.modules, AiterModules), "aiter's modules build on CPU"


def test_a_configured_store_checks_the_host_with_what_it_was_given_and_refuses_with_its_answer(
    tmp_path: Path, ready_host: list[PreflightSite], monkeypatch: pytest.MonkeyPatch
) -> None:
    environ = {**_configured(tmp_path), "HYPERLOOM_BUILD_PYTHON": "/image/bin/python3"}

    native_builds(environ, LocalBlobStore(tmp_path / "blobs"), IMAGE)
    assert ready_host == [
        PreflightSite(
            tmp_path / "store",
            tmp_path / "trust.json",
            tmp_path / "service.key",
            tmp_path / "sources",
            tmp_path / "image",
            "/image/bin/python3",
        )
    ]

    def not_ready(site: PreflightSite) -> None:
        raise BuildStoreConfigError("setpriv is not on PATH")

    monkeypatch.setattr(native_builds_module, "build_preflight", not_ready)
    with pytest.raises(BuildStoreConfigError, match="setpriv is not on PATH"):
        native_builds(environ, LocalBlobStore(tmp_path / "blobs"), IMAGE)


@pytest.mark.parametrize(
    "missing",
    [BUILD_TRUST_ENV, BUILD_STORE_ENV, BUILD_SIGNING_KEY_ENV, BUILD_SOURCES_ENV, BUILD_IMAGE_ROOT_ENV, BUILD_GFX_ENV],
)
def test_a_store_configured_without_what_the_service_needs_is_refused_naming_it(tmp_path: Path, missing: str) -> None:
    with pytest.raises(BuildStoreConfigError, match=missing):
        native_builds(_configured(tmp_path, **{missing: ""}), LocalBlobStore(tmp_path), IMAGE)


def test_a_trust_file_that_lists_no_key_is_refused(tmp_path: Path) -> None:
    (tmp_path / "trust.json").write_text('{"min_epoch": 1, "keys": []}')

    with pytest.raises(BuildStoreConfigError, match="lists no key"):
        trust_from_file(tmp_path / "trust.json")


async def test_a_ready_build_gone_from_the_store_when_the_test_starts_is_not_measured_without_it(rig: Rig) -> None:
    compiled = rig.compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})), RecipeDelta())
    builds = await rig.wait(AcceptedMeasurement(compiled, ()), Hooks())
    entry = rig.store.entry_dir(compiled.builds[0].inputs)
    for name in ("manifest.json", "manifest.sig", "log.txt"):
        (entry / name).unlink()
    entry.rmdir()

    with pytest.raises(BuildInfraError, match="is not ready in the store"):
        builds.resolved(AcceptedMeasurement(compiled, ()))


# --- the toolbox builders -------------------------------------------------------------------------------------------


@pytest.fixture
def no_ambient_store(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (BUILD_STORE_ENV, BUILD_TRUST_ENV, BUILD_SIGNING_KEY_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("no_ambient_store")
async def test_both_builders_with_nothing_configured_drive_no_build_and_tell_the_runners_nothing(
    tmp_path: Path,
) -> None:
    (tmp_path / "hosted").mkdir()
    local, _ = await options_tests._built(tmp_path)
    hosted = options_tests._hosted(tmp_path / "hosted")

    assert local._builds._driver is None and hosted._builds._driver is None
    assert BUILD_STORE_ENV not in options_tests._runner_env(local)
    assert BUILD_STORE_ENV not in hosted.backend._config.create_envs


@pytest.mark.usefixtures("no_ambient_store")
async def test_both_builders_with_a_store_configured_drive_it_and_tell_the_runners_where_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in _configured(tmp_path / "config").items():
        monkeypatch.setenv(name, value)
    (tmp_path / "hosted").mkdir()

    local, _ = await options_tests._built(tmp_path)
    hosted = options_tests._hosted(tmp_path / "hosted")

    assert isinstance(local._builds._driver, StoreBuildDriver) and isinstance(hosted._builds._driver, StoreBuildDriver)
    assert options_tests._runner_env(local)[BUILD_STORE_ENV] == str(tmp_path / "config" / "store")
    assert hosted.backend._config.create_envs[BUILD_TRUST_ENV] == str(tmp_path / "config" / "trust.json")


def test_a_sheet_names_exactly_the_builds_its_environment_needs(rig: Rig) -> None:
    compiled = rig.compiler.compile(rig.recipe(rig.patch({fake.ACT_CU: EDITED_CU})), RecipeDelta())
    (need,) = compiled.builds
    ready = BuildRef(need.root_id, need.inputs.key, "1" * 64)

    assert compiled.with_builds((ready,)).to_document()["builds"] == [ready.to_document()]
    for refs in ((), (BuildRef(need.root_id, "0" * 64, "1" * 64),), (ready, ready)):
        with pytest.raises(BuildNotReady, match="are not the ready ones"):
            compiled.with_builds(refs)


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604], ids=["world", "group", "others"])
def test_a_signing_key_another_user_could_read_is_refused(tmp_path: Path, mode: int) -> None:
    environ = _configured(tmp_path)
    Path(environ[BUILD_SIGNING_KEY_ENV]).chmod(mode)

    with pytest.raises(BuildStoreConfigError, match="no one else can read"):
        native_builds(environ, LocalBlobStore(tmp_path / "blobs"), IMAGE)


def test_a_signing_key_that_is_not_a_regular_file_is_refused(tmp_path: Path) -> None:
    environ = _configured(tmp_path)
    directory = tmp_path / "key-dir"
    directory.mkdir(mode=0o700)

    with pytest.raises(BuildStoreConfigError, match="must name a regular file"):
        native_builds({**environ, BUILD_SIGNING_KEY_ENV: str(directory)}, LocalBlobStore(tmp_path / "blobs"), IMAGE)
