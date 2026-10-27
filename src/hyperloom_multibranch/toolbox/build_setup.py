# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Set up a host's native builds (#125 stage 4b, C++ edits of sgl-kernel measured) in one command.

Run it once on the host, as the user the toolbox runs as (root: the compile runs as user 65534 through ``setpriv``)::

    python -m hyperloom_multibranch.toolbox.build_setup ROOT --image-root /opt/venv/lib/python3.10/site-packages \\
        --gfx gfx950 [--builder-image sha256:...] [--sglang-ref COMMIT] [--sglang-repo DIR_OR_URL]

Under ``ROOT`` (on storage the runners share) it makes the build store ``store/``, the service's signing key
``signing.pem`` (Ed25519, mode 0600, this user's; an existing one is kept, never replaced), the trust file
``trust.json`` with its public key, the checkout of each upstream the image installs at its commit,
``sources/<component>/<commit>`` (``sgl_kernel`` from the image's ``sglang`` or its checkout ``/sgl-workspace/sglang``,
or ``--sglang-ref``; ``vllm`` from the checkout the image's ``vllm`` was installed from, where it keeps one) and
``build.env``, the ``HYPERLOOM_BUILD_*`` variables. It then reports every path user 65534 cannot reach and every one
another user (such as the specialists') could change, and exits 1 if there is one (or the existing key, trust file or
checkout is not usable). Running it again changes nothing that is in place.

A launcher turns native builds on by sourcing the file (``export`` lines) before it starts the session::

    . ROOT/build.env && hyperloom-multibranch run ...

and the toolbox then checks the host at start (:mod:`~hyperloom_multibranch.toolbox.build_preflight`). The role prompt
should say: to change sgl-kernel's C++/HIP sources (under ``python/sglang/kernels/aot/csrc``), add to the delta a
``sgl_kernel`` component with ``upstream_repo`` https://github.com/sgl-project/sglang.git and ``upstream_ref`` the image's
sglang commit, whose patches are git diffs against that commit; the toolbox compiles it before the test (the test waits
as ``pending_build``, holding no card) and a compile error comes back as the candidate's failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from hyperloom.measure import signing
from hyperloom.measure.jit.native.config import BUILD_STORE_ENV, BUILD_TRUST_ENV, BuildStoreConfigError, trust_from_file
from hyperloom_multibranch.toolbox.build_preflight import (
    IMAGE_CHECKOUTS,
    IMAGE_SGLANG_CHECKOUT,
    PreflightSite,
    git_head,
    image_upstreams,
    reach_problems,
    signing_key_problem,
    tamper_problems,
)
from hyperloom_multibranch.toolbox.native_builds import (
    BUILD_GFX_ENV,
    BUILD_IMAGE_ROOT_ENV,
    BUILD_PYTHON_ENV,
    BUILD_SIGNING_KEY_ENV,
    BUILD_SOURCES_ENV,
    BUILDER_IMAGE_ENV,
)

__all__ = ["BuildSetup", "SetupError", "main", "host_problems", "set_up"]

_KEY_MODE = 0o600
#: What the setup creates is its user's, readable by all and writable by no one else (the key excepted, 0600).
_UMASK = 0o022
_TRUST_EPOCH = 1
_GIT_TIMEOUT_SEC = 1800
_ENV_FILE = "build.env"
#: What each step says of what it found: made now, or already in place and left as it is.
_CREATED, _KEPT = "created", "kept"
_SGL_KERNEL = "sgl_kernel"


class SetupError(Exception):
    """What stops the setup: the image's sglang commit cannot be read, a checkout cannot be made, or an existing key or
    trust file cannot be used (it is never replaced)."""


@dataclass(frozen=True)
class BuildSetup:
    """What to set up: under ``root``, for the image whose packages are ``image_root``, built for ``gfx`` with the image
    interpreter ``python``; ``sglang_ref`` and ``sglang_repo`` override what is read from the image, and
    ``image_checkouts`` are where the image keeps its upstream checkouts (by default where the framework image does)."""

    root: Path
    image_root: Path
    gfx: str
    python: str
    builder_image: str | None = None
    sglang_ref: str | None = None
    sglang_repo: str | None = None
    #: The image's own checkout of each component's upstream, read when its install names no commit.
    image_checkouts: Mapping[str, Path] = MappingProxyType(IMAGE_CHECKOUTS)


def _directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_key(path: Path) -> str:
    if not path.exists():
        pem = signing.private_key_pem(signing.generate_private_key())
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, _KEY_MODE)
        with os.fdopen(descriptor, "wb") as out:
            out.write(pem)
        return _CREATED
    problem = signing_key_problem(path, os.geteuid())
    if problem is not None:
        raise SetupError(f"the existing signing key {path} {problem}; it was left as it is")
    return _KEPT


def _ensure_trust(path: Path, key: Path) -> str:
    public = signing.public_key_pem(signing.load_private_key(key))
    if not path.exists():
        document = {"keys": [{"public_key_pem": public, "epoch": _TRUST_EPOCH}], "min_epoch": _TRUST_EPOCH}
        _write_public(path, json.dumps(document, indent=2) + "\n")
        return _CREATED
    try:
        trusted = trust_from_file(path)
    except BuildStoreConfigError as exc:
        raise SetupError(f"{exc}; it was left as it is") from None
    if all(found.public_key_pem.strip() != public.strip() for found in trusted.keys):
        raise SetupError(f"the trust file {path} does not list the public key of {key}; it was left as it is")
    return _KEPT


def _write_public(path: Path, text: str) -> None:
    staged = path.with_name(f".{path.name}.tmp")
    with staged.open("w", encoding="utf-8") as out:
        out.write(text)
    os.replace(staged, path)


def _git(*args: str) -> None:
    done = subprocess.run(
        ["git", "-c", "safe.directory=*", "-c", "advice.detachedHead=false", *args],
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=False,
    )
    if done.returncode != 0:
        raise SetupError(f"git {' '.join(args)} exited {done.returncode}: {done.stderr.strip()[-500:]}")


@dataclass(frozen=True)
class _Clone:
    """A checkout to make: ``component``'s upstream at ``ref``, cloned from ``repo`` (``None``: nowhere to clone from)."""

    component: str
    ref: str
    repo: str | None


def _clones(setup: BuildSetup) -> list[_Clone]:
    """The checkout of each upstream the image installs (``--sglang-ref`` and ``--sglang-repo`` override sglang's)."""
    own = dict(setup.image_checkouts)
    if setup.sglang_repo and Path(setup.sglang_repo).is_dir():
        own[_SGL_KERNEL] = Path(setup.sglang_repo)
    clones = {
        up.component: _Clone(up.component, up.ref, None if up.checkout is None else str(up.checkout))
        for up in image_upstreams(setup.image_root, own)
    }
    found = clones.get(_SGL_KERNEL)
    ref = setup.sglang_ref or (found.ref if found else None)
    if ref is not None:
        clones[_SGL_KERNEL] = _Clone(_SGL_KERNEL, ref, setup.sglang_repo or (found.repo if found else None))
    if not any(clone.repo for clone in clones.values()):
        raise SetupError(
            f"no upstream commit with a checkout to clone it from can be read from {setup.image_root} or "
            f"{', '.join(str(path) for path in setup.image_checkouts.values()) or 'an image checkout'}: pass "
            "--sglang-ref and --sglang-repo"
        )
    return list(clones.values())


def _ensure_checkout(sources: Path, clone: _Clone) -> str:
    ref, checkout = clone.ref, sources / clone.component / clone.ref
    if not checkout.exists() and clone.repo is None:
        return "not made: the image keeps no checkout to clone it from"
    if not checkout.exists():
        _directory(checkout.parent)
        staged = checkout.with_name(f".{ref}.tmp")
        shutil.rmtree(staged, ignore_errors=True)
        _git("clone", "--quiet", "--no-checkout", clone.repo, str(staged))
        _git("-C", str(staged), "checkout", "--quiet", ref)
        staged.rename(checkout)
        return _CREATED
    head = git_head(checkout)
    if head != ref:
        raise SetupError(f"the checkout {checkout} is at {head}, not at {ref}; it was left as it is")
    return _KEPT


def _environment(setup: BuildSetup) -> dict[str, str]:
    """The variables ``build.env`` exports for ``setup``."""
    root = setup.root.resolve()
    variables = {
        BUILD_STORE_ENV: str(root / "store"),
        BUILD_TRUST_ENV: str(root / "trust.json"),
        BUILD_SIGNING_KEY_ENV: str(root / "signing.pem"),
        BUILD_SOURCES_ENV: str(root / "sources"),
        BUILD_IMAGE_ROOT_ENV: str(setup.image_root),
        BUILD_GFX_ENV: setup.gfx,
        BUILD_PYTHON_ENV: setup.python,
    }
    if setup.builder_image:
        variables[BUILDER_IMAGE_ENV] = setup.builder_image
    return variables


def _env_file(variables: Mapping[str, str]) -> str:
    lines = [f"export {name}={shlex.quote(value)}" for name, value in variables.items()]
    return (
        "# Native builds of the toolbox (python -m hyperloom_multibranch.toolbox.build_setup).\n"
        + "\n".join(lines)
        + "\n"
    )


def set_up(setup: BuildSetup) -> list[str]:
    """Set up ``setup`` (see the module); what it made or kept, one line each.

    Raises:
        SetupError: See :class:`SetupError`.
    """
    previous = os.umask(_UMASK)
    try:
        variables = _environment(setup)
        root = _directory(setup.root.resolve())
        key, trust = Path(variables[BUILD_SIGNING_KEY_ENV]), Path(variables[BUILD_TRUST_ENV])
        done = [f"store {_directory(Path(variables[BUILD_STORE_ENV]))}"]
        done.append(f"signing key {key}: {_ensure_key(key)}")
        done.append(f"trust file {trust}: {_ensure_trust(trust, key)}")
        sources = Path(variables[BUILD_SOURCES_ENV])
        done += [f"{c.component} checkout of {c.ref}: {_ensure_checkout(sources, c)}" for c in _clones(setup)]
        _write_public(root / _ENV_FILE, _env_file(variables))
        done.append(f"environment {root / _ENV_FILE}")
    finally:
        os.umask(previous)
    return done


def host_problems(setup: BuildSetup) -> list[str]:
    """Each path of ``setup`` user 65534 must reach and cannot, and each one another user could change (the toolbox
    refuses to start on the same, see :mod:`~hyperloom_multibranch.toolbox.build_preflight`)."""
    variables = {name: Path(value) for name, value in _environment(setup).items()}
    site = PreflightSite(
        variables[BUILD_STORE_ENV],
        variables[BUILD_TRUST_ENV],
        variables[BUILD_SIGNING_KEY_ENV],
        variables[BUILD_SOURCES_ENV],
        setup.image_root,
        setup.python,
    )
    checkouts = [site.sources / clone.component / clone.ref for clone in _clones(setup)]
    checkouts = [checkout for checkout in checkouts if checkout.is_dir()]
    return [*reach_problems(site, checkouts), *tamper_problems(site, checkouts)]


def _default_python(image_root: Path) -> str:
    """The image's interpreter beside ``<prefix>/lib/pythonX.Y/site-packages``, else ``python3``."""
    beside = image_root.parent.parent.parent / "bin" / "python3"
    return str(beside) if beside.is_file() else "python3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_setup", description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="the directory to set up (on storage the runners share)")
    parser.add_argument("--image-root", type=Path, required=True, help="the image's site-packages")
    parser.add_argument("--gfx", required=True, help="the GPU architecture to build for, such as gfx950")
    parser.add_argument("--python", help="the image's interpreter (default: bin/python3 of the image root's prefix)")
    parser.add_argument("--builder-image", help="the digest of the image the build runs in (default: the session's)")
    parser.add_argument("--sglang-ref", help="the image's sglang commit (default: read from the image)")
    parser.add_argument("--sglang-repo", help=f"where to clone sglang from (default: {IMAGE_SGLANG_CHECKOUT})")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    setup = BuildSetup(
        root=args.root,
        image_root=args.image_root,
        gfx=args.gfx,
        python=args.python or _default_python(args.image_root),
        builder_image=args.builder_image,
        sglang_ref=args.sglang_ref,
        sglang_repo=args.sglang_repo,
    )
    try:
        for line in set_up(setup):
            print(line)
        problems = host_problems(setup)
    except SetupError as exc:
        print(f"build_setup: {exc}", file=sys.stderr)
        return 1
    for problem in problems:
        print(f"build_setup: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
