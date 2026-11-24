# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Native builds for a toolbox, when its host configures them; nothing changes when it does not.

A host turns them on with the build store and its trust, which the runners read too (``HYPERLOOM_BUILD_STORE`` and
``HYPERLOOM_BUILD_TRUST``, :mod:`hyperloom.measure.jit.native.config`). This process is then the build service: it plans
each stack's build at acceptance, drives it while the test waits without a card, and signs what the build runner staged.
It also builds the aiter modules a stack touches before its test
(:mod:`hyperloom.measure.runner.native.aiter_modules`), with the variables of its own environment that aiter's build
reads. So it also needs:

- ``HYPERLOOM_BUILD_SIGNING_KEY``: the service's Ed25519 private key, a file of this user with mode 0600 (the compile
  must not be able to read it: as another user it cannot, and under bubblewrap the key is not there); its
  public key must be in the trust file;
- ``HYPERLOOM_BUILD_SOURCES``: the upstream checkouts, ``<dir>/<component>/<upstream_ref>`` (``<dir>/sgl_kernel/<ref>``
  is the SGLang repository, ``<dir>/vllm/<ref>`` vLLM's, ``<dir>/llamacpp/<ref>`` a worktree of the framework checkout
  ``FRAMEWORK_REPO_PATH`` at its ``HEAD``, for a component built from the checkout the operator pinned);
- ``HYPERLOOM_BUILD_IMAGE_ROOT``: the image's directory of installed packages (``site-packages``);
- ``HYPERLOOM_BUILD_GFX``: the GPU architecture the outputs are built for;
- ``HYPERLOOM_BUILD_PYTHON`` (``python3`` when unset): the image's interpreter, which runs the compile;
- ``HYPERLOOM_BUILDER_IMAGE`` (the active image when unset): the digest of the image the build runs in;
- ``HYPERLOOM_BUILD_SGLANG_REF`` (unset when the image names its own): the image's sglang commit, a full commit id, for
  an image that names none (``build_setup --sglang-ref`` records it);
- ``HYPERLOOM_BUILD_CPUS`` (any when unset): the CPUs every compile is pinned to, kept apart from the ones the measured
  servers run on (:func:`~hyperloom.measure.runner.native.isolated.build_cpus`).

A store and trust given without any of the four required ones refuse the build of the toolbox, naming the variable, and
so does a host that is not ready for them (:func:`~hyperloom_multibranch.toolbox.build_preflight.build_preflight`;
``python -m hyperloom_multibranch.toolbox.build_setup`` sets one up). The builds themselves are a
:class:`~hyperloom.measure.runner.native.session.NativeBuildSession`;
this module reads its configuration from the environment and hands the queue its driver. The compile runs out of this
process; a host that cannot isolate it fails each build as the machine's, never compiles in-process. The signing key
and the trust file are registered as the app's private paths
(:data:`~hyperloom.common.confinement.PRIVATE_PATHS`), so a compile's bubblewrap sandbox
(:func:`~hyperloom.measure.runner.native.isolated.compile_policy`) does not show them where a bound tree holds them.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from hyperloom.common import confinement
from hyperloom.contracts.measurement import TestRequest
from hyperloom.measure.compile import BlobStore
from hyperloom.measure.jit.native.config import (
    BUILD_STORE_ENV,
    BUILD_TRUST_ENV,
    BuildStoreConfigError,
    store_from_environment,
)
from hyperloom.measure.jit.native.key import BuildInputs
from hyperloom.measure.jit.native.trust import Signer
from hyperloom.measure.runner.native.aiter_modules import AITER_COMPONENT, aiter_module_recipe
from hyperloom.measure.runner.native.builder import BUILD_PYTHON_ENV, BuildMachine
from hyperloom.measure.runner.native.isolated import Isolation, Unisolated, build_cpus, local_isolation
from hyperloom.measure.runner.native.planning import NativeSite
from hyperloom.measure.runner.native.recipes import native_recipes
from hyperloom.measure.runner.native.session import NativeBuildSession, open_session, pinned_checkouts
from hyperloom_multibranch.toolbox.build_preflight import (
    FULL_COMMIT,
    SGL_KERNEL,
    PreflightSite,
    build_preflight,
    signing_key_problem,
)
from hyperloom_multibranch.toolbox.store_builds import BuildSite, StoreBuildDriver
from hyperloom_multibranch.toolbox.toolbox import AcceptanceSetup

__all__ = [
    "BUILD_GFX_ENV",
    "BUILD_IMAGE_ROOT_ENV",
    "BUILD_SGLANG_REF_ENV",
    "BUILD_SIGNING_KEY_ENV",
    "BUILD_SOURCES_ENV",
    "BUILDER_IMAGE_ENV",
    "NativeBuilds",
    "compiled_needs",
    "named_commits",
    "native_builds",
    "with_native_plan",
]

BUILD_SIGNING_KEY_ENV = "HYPERLOOM_BUILD_SIGNING_KEY"
BUILD_SOURCES_ENV = "HYPERLOOM_BUILD_SOURCES"
BUILD_IMAGE_ROOT_ENV = "HYPERLOOM_BUILD_IMAGE_ROOT"
BUILD_GFX_ENV = "HYPERLOOM_BUILD_GFX"
BUILDER_IMAGE_ENV = "HYPERLOOM_BUILDER_IMAGE"
BUILD_SGLANG_REF_ENV = "HYPERLOOM_BUILD_SGLANG_REF"
_DEFAULT_PYTHON = "python3"


@dataclass(frozen=True)
class NativeBuilds:
    """A toolbox's native builds: the queue's driver, the session it builds through (whose planner and module plan
    acceptance compiles with), and what its runners are told."""

    driver: StoreBuildDriver
    session: NativeBuildSession
    runner_envs: Mapping[str, str]


def compiled_needs(request: TestRequest, environment: object) -> tuple[BuildInputs, ...]:
    """The driver's plan: the inputs of each build acceptance compiled (``AcceptedMeasurement.environment``)."""
    compiled = getattr(environment, "environment", None)
    return tuple(need.inputs for need in getattr(compiled, "builds", ()))


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise BuildStoreConfigError(f"the native build store is configured and {name} is not set")
    return value


def named_commits(environ: Mapping[str, str]) -> dict[str, str]:
    """The upstream commit of each component no image names, by component: the one ``environ`` names for sgl_kernel,
    and the ``HEAD`` of the framework checkout it names for a component built from its pinned checkout
    (:func:`~hyperloom.measure.runner.native.session.pinned_checkouts`).

    Raises:
        BuildStoreConfigError: ``HYPERLOOM_BUILD_SGLANG_REF`` is set to something other than a full commit id, or the
            pinned checkout's ``HEAD`` cannot be read or differs from its tracked files.
    """
    ref = environ.get(BUILD_SGLANG_REF_ENV, "").strip()
    if ref and not FULL_COMMIT.fullmatch(ref):
        raise BuildStoreConfigError(f"{BUILD_SGLANG_REF_ENV} must be a full lowercase commit id")
    pinned = {found.component: found.commit for found in pinned_checkouts(native_recipes(), environ)}
    return {SGL_KERNEL: ref, **pinned} if ref else pinned


def _signer(path: Path) -> Signer:
    """The service's key, from a file only this user can read: a key the compile could read is a key it could forge
    records with.

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
    isolation: Callable[[], Isolation | Unisolated] = local_isolation,
) -> NativeBuilds | None:
    """The native builds ``environ`` configures for a session on ``image_digest`` (see the module); ``None`` for none.

    Raises:
        BuildStoreConfigError: The store is half configured, its trust or signing key cannot be read, a variable the
            service needs is not set, the sglang commit it names is not a full commit id, or the build CPUs it names are
            not a CPU list.
    """
    store = store_from_environment(environ)
    if store is None:
        return None
    key = Path(_required(environ, BUILD_SIGNING_KEY_ENV))
    signer = _signer(key)
    confinement.PRIVATE_PATHS.add(key, "the native build service's signing key")
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
    confinement.PRIVATE_PATHS.add(shared[1], "the native build store's trust file")
    preflight = PreflightSite(*shared, key, sources, image_root, machine.python, named_commits(environ))
    commits = {up.component: up.ref for up in build_preflight(preflight)}
    recipes = native_recipes()
    site = NativeSite(machine, recipes, sources, image_root, commits, build_cpus(environ))
    session = open_session(store, signer, site, blobs, isolation)
    built = {**recipes, AITER_COMPONENT: aiter_module_recipe()}
    build_site = BuildSite(built)
    driver = StoreBuildDriver(store, signer, build_site, compiled_needs, session.runner)
    return NativeBuilds(driver, session, runner_envs)


def with_native_plan(acceptance: AcceptanceSetup, native: NativeBuilds | None) -> AcceptanceSetup:
    """``acceptance`` whose planner compiles with ``native``'s plans; as it is without one, or with a planner that takes
    no plan (its native stacks stay refused by their layouts, and aiter's touched modules are built on the card)."""
    with_native = getattr(acceptance.planner, "with_native", None)
    if native is None or with_native is None:
        return acceptance
    return replace(acceptance, planner=with_native(native.session.planner, native.session.modules))
