# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The queue's driver over the real native build store: lookups, the estimate from the store's records, and builds."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyperloom.contracts.measurement import Purpose, TestKind, TestRequest
from hyperloom.contracts.recipe import ConfigDelta, RecipeDelta
from hyperloom.measure import signing
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.record import FAILED, READY, RECORD_FILES
from hyperloom.measure.jit.native.store import BuildInfraError, BuildStore
from hyperloom.measure.jit.native.trust import BuildRefused, Refusal, Signer, Trust, TrustedKey
from hyperloom.measure.runner.native.recipes import sgl_kernel_recipe
from hyperloom.measure.runner.native.service import Staged, ensure_build
from hyperloom_multibranch.toolbox.store_builds import BuildSite, StoreBuildDriver

IMAGE = "sha256:" + "a" * 64
OTHER_IMAGE = "sha256:" + "d" * 64
BUILDER = "sha256:" + "c" * 64
RECIPE = sgl_kernel_recipe()
COMPONENT = RECIPE.component
SITE = BuildSite(IMAGE, BUILDER, "gfx942", {COMPONENT: RECIPE})


def _inputs(source: str, image: str = IMAGE) -> BuildInputs:
    return BuildInputs(image, BUILDER, COMPONENT, "gfx942", source * 64, RECIPE.version)


class Runner:
    """Stages a build with the wall time it is told, or fails it; never compiles."""

    def __init__(self, wall: float = 34.0, status: str = READY, infra: bool = False) -> None:
        self.wall, self.status, self.infra, self.calls = wall, status, infra, 0

    def __call__(self, inputs: BuildInputs):
        def build(scratch: Path) -> Staged:
            self.calls += 1
            if self.infra:
                raise BuildInfraError("the runner was evicted")
            failure = "compile_error: kernel.cu:1: error: stop" if self.status == FAILED else ""
            return Staged(self.status, (), {}, self.wall * 3, self.wall, "log", scratch, failure=failure)

        return build


@pytest.fixture
def key():
    return signing.generate_private_key()


@pytest.fixture
def store(tmp_path: Path, key) -> BuildStore:
    return BuildStore(tmp_path / "store", Trust((TrustedKey(signing.public_key_pem(key), 1),), 1))


def _driver(store: BuildStore, key, runner: Runner, changed: dict[str, str] | None = None) -> StoreBuildDriver:
    return StoreBuildDriver(store, Signer(key), SITE, lambda request, env: changed or {}, runner)


def _publish(store: BuildStore, key, inputs: BuildInputs, wall: float, status: str = READY) -> None:
    ensure_build(store, inputs, Runner(wall, status)(inputs), Signer(key))


def test_the_estimate_is_the_median_wall_time_of_ready_builds_of_the_component_on_the_image(
    store: BuildStore, key
) -> None:
    for source, wall in (("1", 30.0), ("2", 34.0), ("3", 90.0)):
        _publish(store, key, _inputs(source), wall)
    _publish(store, key, _inputs("4", OTHER_IMAGE), 5000.0)
    _publish(store, key, _inputs("5"), 9000.0, FAILED)
    assert _driver(store, key, Runner()).eta_seconds(_inputs("9")) == 34.0


def test_without_a_ready_build_on_the_image_the_estimate_is_the_recipe_budget(store: BuildStore, key) -> None:
    driver = _driver(store, key, Runner())
    assert driver.eta_seconds(_inputs("9")) == RECIPE.budget_sec
    _publish(store, key, _inputs("4", OTHER_IMAGE), 20.0)
    assert driver.eta_seconds(_inputs("9")) == RECIPE.budget_sec


def test_a_record_that_fails_its_signature_says_nothing_of_the_estimate(store: BuildStore, key) -> None:
    _publish(store, key, _inputs("1"), 20.0)
    manifest = store.entry_dir(_inputs("1")) / RECORD_FILES[READY][0]
    manifest.write_bytes(manifest.read_bytes().replace(b'"wall_seconds":20.0', b'"wall_seconds":21.0'))
    assert _driver(store, key, Runner()).eta_seconds(_inputs("9")) == RECIPE.budget_sec


async def test_ensure_builds_once_and_then_reuses_and_lookup_reads_the_entry(store: BuildStore, key) -> None:
    runner = Runner(wall=34.0)
    driver = _driver(store, key, runner)
    assert driver.lookup(_inputs("1")) is None
    first = await driver.ensure(_inputs("1"))
    again = await driver.ensure(_inputs("1"))
    assert (first.reused, again.reused, runner.calls) == (False, True, 1)
    assert (first.record.status, first.record.build_seconds, first.record.wall_seconds) == (READY, 102.0, 34.0)
    assert driver.lookup(_inputs("1")) == first.record


async def test_a_machine_failure_of_the_runner_is_raised_and_stores_nothing(store: BuildStore, key) -> None:
    driver = _driver(store, key, Runner(infra=True))
    with pytest.raises(BuildInfraError, match="the runner was evicted"):
        await driver.ensure(_inputs("1"))
    assert driver.lookup(_inputs("1")) is None


def test_needs_key_each_changed_component_by_the_site_and_its_recipe_version(store: BuildStore, key) -> None:
    request = TestRequest(
        kind=TestKind.TEST,
        purpose=Purpose.EXPLORE,
        branch_id="b1",
        parent_commit="c0",
        delta=RecipeDelta(config=ConfigDelta(server_args="--x")),
    )
    assert _driver(store, key, Runner(), {COMPONENT: "1" * 64}).needs(request, None) == (_inputs("1"),)
    with pytest.raises(KeyError, match="vllm"):
        _driver(store, key, Runner(), {"vllm": "1" * 64}).needs(request, None)


def test_lookup_of_an_entry_that_fails_a_check_is_none_and_ensure_refuses_it(store: BuildStore, key) -> None:
    _publish(store, key, _inputs("1"), 20.0)
    manifest = store.entry_dir(_inputs("1")) / RECORD_FILES[READY][0]
    manifest.write_bytes(manifest.read_bytes() + b" ")
    driver = _driver(store, key, Runner())
    assert driver.lookup(_inputs("1")) is None
    with pytest.raises(BuildRefused) as refused:
        asyncio.run(driver.ensure(_inputs("1")))
    assert refused.value.reason is Refusal.BAD_SIGNATURE
