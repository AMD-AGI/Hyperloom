# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native builds for a toolbox, when its host configures them (#125 stage 4b); nothing changes when it does not.

A host turns them on with the build store and its trust, which the runners read too (``HYPERLOOM_BUILD_STORE`` and
``HYPERLOOM_BUILD_TRUST``, :mod:`hyperloom.measure.jit.native.config`). This process is then the build service: it plans
each stack's build at acceptance, drives it while the test waits without a card, and signs what the build runner staged.
It also builds the aiter modules a stack touches before its test (#125 stage 2b,
:mod:`hyperloom.measure.runner.native.aiter_modules`), with the variables of its own environment that aiter's build reads.
So it also needs:

- ``HYPERLOOM_BUILD_SIGNING_KEY``: the service's Ed25519 private key, a file of this user with mode 0600 (the compile
  runs as another user on this host); its public key must be in the trust file;
- ``HYPERLOOM_BUILD_SOURCES``: the upstream checkouts, ``<dir>/<component>/<upstream_ref>`` (``<dir>/sgl_kernel/<ref>``
  is the SGLang repository, ``<dir>/vllm/<ref>`` vLLM's);
- ``HYPERLOOM_BUILD_IMAGE_ROOT``: the image's directory of installed packages (``site-packages``);
- ``HYPERLOOM_BUILD_GFX``: the GPU architecture the outputs are built for;
- ``HYPERLOOM_BUILD_PYTHON`` (``python3`` when unset): the image's interpreter, which runs the compile;
- ``HYPERLOOM_BUILDER_IMAGE`` (the active image when unset): the digest of the image the build runs in.

A store and trust given without any of the four required ones refuse the build of the toolbox, naming the variable, and
so does a host that is not ready for them (:func:`~hyperloom_multibranch.toolbox.build_preflight.build_preflight`;
``python -m hyperloom_multibranch.toolbox.build_setup`` sets one up). The compile runs in
:class:`~hyperloom.measure.runner.native.isolated.IsolatedBuilds`, out of this process; a host that cannot isolate it
fails each build as the machine's, never compiles in-process.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

from hyperloom.contracts.measurement import TestRequest
from hyperloom.measure.compile import BlobStore
from hyperloom.measure.jit.native.config import (
    BUILD_STORE_ENV,
    BUILD_TRUST_ENV,
    BuildStoreConfigError,
    store_from_environment,
)
from hyperloom.measure.jit.native.trust import Signer
from hyperloom.measure.runner.native.aiter_modules import (
    AITER_COMPONENT,
    AITER_RECIPE_FACTORY,
    AiterModules,
    aiter_module_recipe,
)
from hyperloom.measure.runner.native.builder import BuildMachine
from hyperloom.measure.runner.native.isolated import (
    ISOLATION_UID_NETNS,
    IsolatedBuilds,
    Isolation,
    local_isolation,
)
from hyperloom.measure.runner.native.planning import NativePlanner, NativeSite
from hyperloom.measure.runner.native.recipes import RECIPE_FACTORIES, native_recipes
from hyperloom_multibranch.toolbox.build_preflight import PreflightSite, build_preflight, signing_key_problem
from hyperloom_multibranch.toolbox.store_builds import BuildSite, StoreBuildDriver
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

__all__ = [
    "BUILD_GFX_ENV",
    "BUILD_IMAGE_ROOT_ENV",
    "BUILD_PYTHON_ENV",
    "BUILD_SIGNING_KEY_ENV",
    "BUILD_SOURCES_ENV",
    "BUILDER_IMAGE_ENV",
    "NativeBuilds",
    "compiled_needs",
    "native_builds",
    "with_native_plan",
]

BUILD_SIGNING_KEY_ENV = "HYPERLOOM_BUILD_SIGNING_KEY"
BUILD_SOURCES_ENV = "HYPERLOOM_BUILD_SOURCES"
BUILD_IMAGE_ROOT_ENV = "HYPERLOOM_BUILD_IMAGE_ROOT"
BUILD_GFX_ENV = "HYPERLOOM_BUILD_GFX"
BUILD_PYTHON_ENV = "HYPERLOOM_BUILD_PYTHON"
BUILDER_IMAGE_ENV = "HYPERLOOM_BUILDER_IMAGE"
_DEFAULT_PYTHON = "python3"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativeBuilds:
    """A toolbox's native builds: the queue's driver, the compile's plans (of native stacks, and of aiter's touched
    modules), and what its runners are told."""

    driver: StoreBuildDriver
    plan: NativePlanner
    modules: AiterModules
    runner_envs: Mapping[str, str]


def compiled_needs(request: TestRequest, environment: object) -> Mapping[str, str]:
    """The driver's plan: the native digest of each build acceptance compiled (``AcceptedMeasurement.environment``)."""
    compiled = getattr(environment, "environment", None)
    return {need.inputs.component: need.inputs.native_digest for need in getattr(compiled, "builds", ())}


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise BuildStoreConfigError(f"the native build store is configured and {name} is not set")
    return value


def _signer(path: Path) -> Signer:
    """The service's key, from a file only this user can read: the compile runs as another user on this host, and a key
    it could read is a key it could forge records with.

    Raises:
        BuildStoreConfigError: The file is a link, is not this user's, can be read by anyone else, or is not a key.
    """
    problem = signing_key_problem(path, os.geteuid())
    if problem is not None:
        raise BuildStoreConfigError(f"{BUILD_SIGNING_KEY_ENV} {problem}")
    try:
        return Signer.from_file(path)
    except (OSError, ValueError) as exc:
        raise BuildStoreConfigError(f"{BUILD_SIGNING_KEY_ENV} cannot be used: {type(exc).__name__}") from None


def native_builds(
    environ: Mapping[str, str],
    blobs: BlobStore,
    image_digest: str,
    *,
    isolation: Callable[[], Isolation | None] = local_isolation,
) -> NativeBuilds | None:
    """The native builds ``environ`` configures for a session on ``image_digest`` (see the module); ``None`` for none.

    Raises:
        BuildStoreConfigError: The store is half configured, its trust or signing key cannot be read, or a variable the
            service needs is not set.
    """
    store = store_from_environment(environ)
    if store is None:
        return None
    key = Path(_required(environ, BUILD_SIGNING_KEY_ENV))
    signer = _signer(key)
    image = image_digest.rsplit("@", 1)[-1]
    machine = BuildMachine(
        image,
        environ.get(BUILDER_IMAGE_ENV, "").strip() or image,
        _required(environ, BUILD_GFX_ENV),
        environ.get(BUILD_PYTHON_ENV, "").strip() or _DEFAULT_PYTHON,
    )
    sources, image_root = Path(_required(environ, BUILD_SOURCES_ENV)), Path(_required(environ, BUILD_IMAGE_ROOT_ENV))
    runner_envs = {BUILD_STORE_ENV: environ[BUILD_STORE_ENV].strip(), BUILD_TRUST_ENV: environ[BUILD_TRUST_ENV].strip()}
    shared = (Path(runner_envs[BUILD_STORE_ENV]), Path(runner_envs[BUILD_TRUST_ENV]))
    build_preflight(PreflightSite(*shared, key, sources, image_root, machine.python))
    recipes = native_recipes()
    site = NativeSite(machine, recipes, sources, image_root)
    planner, isolated = NativePlanner(site, blobs), isolation()
    modules = AiterModules(planner, history=partial(store.ready_records, AITER_COMPONENT, machine.image_digest))
    runner = IsolatedBuilds(planner, blobs, isolated, {**RECIPE_FACTORIES, AITER_COMPONENT: AITER_RECIPE_FACTORY})
    _log_isolation(isolated)
    built = {**recipes, AITER_COMPONENT: aiter_module_recipe()}
    build_site = BuildSite(machine.image_digest, machine.builder_image_digest, machine.gfx_arch, built)
    driver = StoreBuildDriver(store, signer, build_site, compiled_needs, runner)
    return NativeBuilds(driver, planner, modules, runner_envs)


def _log_isolation(isolated: Isolation | None) -> None:
    """Say once, at the start of the session, how this host isolates a compile (the level each build's record names)."""
    if isolated is None:
        logger.warning("native builds are configured and this host cannot isolate a compile: every build will fail")
    elif isolated.level == ISOLATION_UID_NETNS:
        logger.info(
            "native builds compile as user %s in a network namespace of their own (%s)", isolated.uid, isolated.level
        )
    else:
        logger.warning(
            "native builds compile as user %s with this host's network: it cannot make a network namespace (%s)",
            isolated.uid,
            isolated.level,
        )


def with_native_plan(acceptance: AcceptanceSetup, native: NativeBuilds | None) -> AcceptanceSetup:
    """``acceptance`` whose planner compiles with ``native``'s plans; as it is without one, or with a planner that takes
    no plan (its native stacks stay refused by their layouts, and aiter's touched modules are built on the card)."""
    with_native = getattr(acceptance.planner, "with_native", None)
    if native is None or with_native is None:
        return acceptance
    return replace(acceptance, planner=with_native(native.plan, native.modules))
