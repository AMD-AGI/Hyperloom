# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a host's native builds need before the toolbox starts (#125 stage 4b), checked once and refused by name.

:func:`~hyperloom_multibranch.toolbox.native_builds.native_builds` runs :func:`build_preflight` when the build store is
configured, so a host that is set up wrong refuses at start, naming every item, instead of failing each build later:

- the service runs as root and has ``setpriv`` (the compile runs as user 65534);
- user 65534 can reach what the compile runs: this process's interpreter, its environment and source tree (the build
  child imports the builder from it), the image's interpreter (``HYPERLOOM_BUILD_PYTHON``), the image's packages
  (``HYPERLOOM_BUILD_IMAGE_ROOT``) and the upstream checkouts;
- no user but this one (and root) can change the signing key, the trust file, the store or the upstream checkouts, nor
  a directory above them (a sticky one such as ``/tmp`` excepted): another user on the host, such as the specialists',
  could otherwise swap the key or plant a source or an output;
- ``HYPERLOOM_BUILD_SOURCES`` holds, for at least one native component, the checkout of the commit the image installs
  (``<sources>/<component>/<commit>``, at that commit), and no other commit of a component it has a directory for. The
  commit is read from the image's installed upstream (``sglang`` for ``sgl_kernel``, ``vllm`` for ``vllm``: its
  ``direct_url.json``, or the checkout it was installed from) or, for sglang, its checkout ``/sgl-workspace/sglang``;
- the importable TraceLens is the commit this repository pins (``TRACELENS_REF``): read from a git checkout or the
  installed version's ``+g<commit>``; when neither names a commit, the toolbox's TraceLens imports must succeed.

The signing key's mode and owner are checked where it is loaded (:func:`signing_key_problem`). Access is judged from
mode bits as user 65534 with no supplementary groups (``setpriv --clear-groups``); ACLs are not read.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import sysconfig
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from os import geteuid
from pathlib import Path
from shutil import which
from urllib.parse import unquote, urlparse

from hyperloom.measure.jit.native.config import BuildStoreConfigError
from hyperloom.measure.runner.native.isolated import NOBODY, SOURCE_ROOT
from hyperloom.measure.runner.native.recipes import RECIPE_FACTORIES
from hyperloom.orchestrator.trace_analysis import TRACELENS_REF

__all__ = [
    "IMAGE_CHECKOUTS",
    "IMAGE_SGLANG_CHECKOUT",
    "ImageUpstream",
    "PreflightSite",
    "build_preflight",
    "changeable_by_others",
    "git_head",
    "image_checkouts",
    "image_upstreams",
    "reach_problems",
    "signing_key_problem",
    "tamper_problems",
    "tracelens_problem",
    "unreachable",
]

#: Where the sglang images keep the checkout their ``sglang`` is installed from.
IMAGE_SGLANG_CHECKOUT = Path("/sgl-workspace/sglang")
#: The image's own checkout of a component's upstream, where it keeps one.
IMAGE_CHECKOUTS = {"sgl_kernel": IMAGE_SGLANG_CHECKOUT}
_TRACELENS = "TraceLens"
#: The modules of this repository that import TraceLens when the toolbox starts or analyses a trace.
_TRACELENS_USERS = (
    "hyperloom.orchestrator.trace_analysis.tracelens_skill_runner",
    "hyperloom.orchestrator.trace_analysis.tracelens_analysis",
)
_COMMIT = re.compile(r"[0-9a-f]{40}")
_VERSION_COMMIT = re.compile(r"\+g([0-9a-f]{7,40})")
_GIT_TIMEOUT_SEC = 30
_READ, _RUN = stat.S_IROTH, stat.S_IXOTH
_OWNER_SHIFT, _GROUP_SHIFT = 6, 3
#: Any access by the group or others.
_OTHERS_ACCESS = 0o077
#: Write access by the group or others: what lets another user change a file or a directory's entries.
_SHARED_WRITE = stat.S_IWGRP | stat.S_IWOTH

StatFn = Callable[[Path], os.stat_result]


@dataclass(frozen=True)
class PreflightSite:
    """What the build service was configured with: the store, its trust file and signing key, the upstream checkouts, the
    image's packages and its interpreter."""

    store: Path
    trust: Path
    key: Path
    sources: Path
    image_root: Path
    python: str


@dataclass(frozen=True)
class ImageUpstream:
    """The commit of ``component``'s upstream the image installs, and the checkout it was read from (``None`` when the
    install's metadata named the commit)."""

    component: str
    ref: str
    checkout: Path | None


#: The installed distribution whose commit each native component
#: (:data:`~hyperloom.measure.runner.native.recipes.RECIPE_FACTORIES`) is built from.
_DISTRIBUTIONS = {"sgl_kernel": "sglang", "vllm": "vllm"}


def _git(directory: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_head(directory: Path) -> str | None:
    """The commit checked out in ``directory``, or ``None`` when it is not a git checkout git can read."""
    head = _git(directory, "rev-parse", "HEAD")
    return head if head is not None and _COMMIT.fullmatch(head) else None


def _direct_url(distribution: importlib.metadata.Distribution) -> dict:
    try:
        document = json.loads(distribution.read_text("direct_url.json") or "{}")
    except ValueError:
        return {}
    return document if isinstance(document, dict) else {}


def _installed_upstream(image_root: Path, component: str, distribution: str) -> ImageUpstream | None:
    for installed in importlib.metadata.distributions(name=distribution, path=[str(image_root)]):
        direct = _direct_url(installed)
        commit = (direct.get("vcs_info") or {}).get("commit_id")
        if isinstance(commit, str) and _COMMIT.fullmatch(commit):
            return ImageUpstream(component, commit, None)
        url = urlparse(str(direct.get("url", "")))
        if url.scheme == "file":
            installed_from = Path(unquote(url.path))
            head, top = git_head(installed_from), _git(installed_from, "rev-parse", "--show-toplevel")
            if head is not None and top:
                return ImageUpstream(component, head, Path(top))
    return None


def image_upstreams(image_root: Path, checkouts: Mapping[str, Path] = IMAGE_CHECKOUTS) -> list[ImageUpstream]:
    """The upstream commit of each native component the image whose packages are ``image_root`` installs: the commit
    its installed upstream names, or the head of the checkout it was installed from, or else of the component's
    checkout in ``checkouts``. A component none of them names is left out."""
    found = []
    for component in RECIPE_FACTORIES:
        installed = _installed_upstream(image_root, component, _DISTRIBUTIONS[component])
        own = checkouts.get(component)
        head = git_head(own) if installed is None and own is not None and own.is_dir() else None
        if head is not None:
            installed = ImageUpstream(component, head, own)
        if installed is not None:
            found.append(installed)
    return found


def _allows(found: os.stat_result, bits: int) -> bool:
    """Whether user 65534, with no supplementary group, has ``bits`` (as the others' bits) on ``found``."""
    shift = _OWNER_SHIFT if found.st_uid == NOBODY else _GROUP_SHIFT if found.st_gid == NOBODY else 0
    return (found.st_mode >> shift) & bits == bits


def _describe(path: Path, found: os.stat_result) -> str:
    return f"{path} (mode {stat.S_IMODE(found.st_mode):04o}, owner uid {found.st_uid} gid {found.st_gid})"


def unreachable(path: Path, *, run: bool = False, stat_fn: StatFn = os.stat) -> str | None:
    """Why user 65534 cannot reach ``path`` (list and enter a directory, read a file, and run it when ``run``), naming
    the first directory or file that stops it, on the path as given and as resolved; ``None`` when it can."""
    absolute = Path(os.path.abspath(path))
    for candidate in dict.fromkeys((absolute, Path(os.path.realpath(path)))):
        for directory in reversed(candidate.parents):
            problem = _blocked(directory, _RUN, stat_fn, "pass through")
            if problem:
                return problem
    target = Path(os.path.realpath(path))
    try:
        is_dir = stat.S_ISDIR(stat_fn(target).st_mode)
    except OSError:
        return f"{target} does not exist"
    wanted = _READ | _RUN if is_dir or run else _READ
    return _blocked(target, wanted, stat_fn, "read and enter it" if is_dir else "run it" if run else "read it")


def _blocked(path: Path, bits: int, stat_fn: StatFn, what: str) -> str | None:
    try:
        found = stat_fn(path)
    except OSError:
        return f"{path} does not exist"
    return None if _allows(found, bits) else f"{_describe(path, found)} does not let user {NOBODY} {what}"


def signing_key_problem(path: Path, euid: int) -> str | None:
    """Why ``path`` cannot be the service's signing key: not a regular file of user ``euid`` that no one else can read
    (mode 0600); ``None`` when it can."""
    try:
        found = os.lstat(path)
    except OSError as exc:
        return f"cannot be used: {type(exc).__name__}"
    if not stat.S_ISREG(found.st_mode) or found.st_uid != euid or found.st_mode & _OTHERS_ACCESS:
        return (
            f"must name a regular file of this user (uid {euid}) that no one else can read (mode 0600), and "
            f"{_describe(path, found)} is not: the compile runs as another user on this host and must not be able to "
            "read the key"
        )
    return None


def _changeable(path: Path, owner: int, *, above: bool, follow: bool = True) -> str | None:
    """Why a user other than ``owner`` and root could change ``path``; ``None`` when none could, or it is missing. A link
    not followed counts by its owner only (whoever owns it can replace it; its mode means nothing). A sticky directory
    ``above`` what is checked lets others add entries, not replace this one's."""
    try:
        found = os.stat(path) if follow else os.lstat(path)
    except OSError:
        return None
    if found.st_uid not in (0, owner):
        return f"{_describe(path, found)} is another user's"
    if stat.S_ISLNK(found.st_mode):
        return None
    sticky = above and stat.S_ISDIR(found.st_mode) and found.st_mode & stat.S_ISVTX
    if found.st_mode & _SHARED_WRITE and not sticky:
        return f"{_describe(path, found)} can be written by its group or others"
    return None


def _below(top: Path, files: bool) -> Iterator[Path]:
    for directory, names, file_names in os.walk(top):
        yield from (Path(directory, name) for name in names)
        if files:
            yield from (Path(directory, name) for name in file_names)


def changeable_by_others(path: Path, owner: int, *, tree: bool = False, files: bool = True) -> str | None:
    """Why a user other than ``owner`` (and root) could change ``path``: a component of it as given (a link among them
    by its owner, as the path is used again later and a replaced link would redirect it) or as resolved is another
    user's or can be written by its group or others; with ``tree``, the same of each directory below it (and each file,
    with ``files``), links not followed. ``None`` when no one else could."""
    given, resolved = Path(os.path.abspath(path)), Path(os.path.realpath(path))
    checks = [(component, True, False) for component in reversed(given.parents)]
    checks += [(given, False, False), *((component, True, True) for component in reversed(resolved.parents))]
    checks.append((resolved, False, True))
    if tree:
        checks += [(below, False, False) for below in _below(resolved, files)]
    for component, above, follow in checks:
        problem = _changeable(component, owner, above=above, follow=follow)
        if problem:
            return problem
    return None


def _guarded_items(site: PreflightSite, checkouts: Sequence[Path]) -> Iterator[tuple[str, Path, bool, bool]]:
    """What no other user may change: (what it is, its path, whether what is below it counts, files included)."""
    yield "HYPERLOOM_BUILD_SIGNING_KEY", site.key, False, False
    yield "HYPERLOOM_BUILD_TRUST", site.trust, False, False
    yield "HYPERLOOM_BUILD_STORE", site.store, True, False
    for checkout in checkouts:
        yield f"the {checkout.parent.name} checkout", checkout, True, True


def tamper_problems(site: PreflightSite, checkouts: Sequence[Path]) -> Iterator[str]:
    """Each item of ``site`` another user could change (see the module), with the upstream ``checkouts``. The
    service's user is the signing key's owner (where the key is loaded it must be this process's user)."""
    try:
        owner = os.stat(site.key).st_uid
    except OSError:
        owner = geteuid()
    for what, path, tree, files in _guarded_items(site, checkouts):
        problem = changeable_by_others(path, owner, tree=tree, files=files)
        if problem is not None:
            yield f"users other than uid {owner} could change {what} {path}: {problem}"


def _executable(python: str) -> Path | None:
    found = which(python)
    return None if found is None else Path(found)


def _reach_items(site: PreflightSite, checkouts: Sequence[Path]) -> Iterator[tuple[str, Path | None, bool]]:
    """What user 65534 must reach: (what it is, its path or ``None`` when it cannot be found, whether it is run)."""
    yield "the toolbox interpreter", Path(sys.executable), True
    yield "the toolbox environment", Path(sysconfig.get_paths()["purelib"]), False
    yield "the toolbox standard library", Path(sysconfig.get_paths()["stdlib"]), False
    yield "the toolbox source tree", SOURCE_ROOT / "hyperloom" / "measure" / "runner" / "native" / "child.py", False
    yield f"HYPERLOOM_BUILD_PYTHON {site.python!r}", _executable(site.python), True
    yield "HYPERLOOM_BUILD_IMAGE_ROOT", site.image_root, False
    for checkout in checkouts:
        yield f"the {checkout.parent.name} checkout", checkout, False


def reach_problems(site: PreflightSite, checkouts: Sequence[Path]) -> Iterator[str]:
    """Each path user 65534 must reach (see the module) that it cannot, with the upstream ``checkouts``."""
    for what, path, run in _reach_items(site, checkouts):
        if path is None:
            yield f"{what} is not found on PATH"
            continue
        problem = unreachable(path, run=run)
        if problem is not None:
            yield f"user {NOBODY} cannot reach {what} {path}: {problem}"


def image_checkouts(sources: Path, upstreams: Sequence[ImageUpstream]) -> list[Path]:
    """The checkouts in ``sources`` of the commits the image installs."""
    return [sources / up.component / up.ref for up in upstreams if (sources / up.component / up.ref).is_dir()]


def _sources_problems(site: PreflightSite, upstreams: Sequence[ImageUpstream]) -> Iterator[str]:
    """Whether ``HYPERLOOM_BUILD_SOURCES`` holds the image's checkouts (see the module)."""
    if not upstreams:
        names = " or ".join(_DISTRIBUTIONS.values())
        yield (
            f"the image's upstream commits cannot be read: HYPERLOOM_BUILD_IMAGE_ROOT {site.image_root} installs no "
            f"{names} that names one and {IMAGE_SGLANG_CHECKOUT} is not a git checkout"
        )
        return
    for up in upstreams:
        component, checkout = site.sources / up.component, site.sources / up.component / up.ref
        if component.is_dir() and not checkout.is_dir():
            yield (
                f"HYPERLOOM_BUILD_SOURCES has no checkout of the image's {up.component} commit {up.ref}: {checkout} "
                f"does not exist (it has {sorted(p.name for p in component.iterdir()) or 'none'})"
            )
        head = git_head(checkout) if (checkout / ".git").exists() else up.ref
        if checkout.is_dir() and head != up.ref:
            yield f"HYPERLOOM_BUILD_SOURCES checkout {checkout} is at {head}, not at the image's commit {up.ref}"
    if not image_checkouts(site.sources, upstreams):
        wanted = ", ".join(str(site.sources / up.component / up.ref) for up in upstreams)
        yield f"HYPERLOOM_BUILD_SOURCES holds no checkout of what the image installs: wanted one of {wanted}"


def _same_commit(found: str, pin: str) -> bool:
    return pin.startswith(found) or found.startswith(pin)


def _installed_commit(package: str, location: Path) -> str | None:
    """The commit of the installed distribution ``package`` when it is the one at ``location``."""
    try:
        distribution = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return None
    if Path(str(distribution.locate_file(""))).resolve() != location.resolve():
        return None
    commit = (_direct_url(distribution).get("vcs_info") or {}).get("commit_id")
    if isinstance(commit, str) and commit:
        return commit
    match = _VERSION_COMMIT.search(distribution.version or "")
    return match.group(1) if match else None


def tracelens_problem(
    pin: str = TRACELENS_REF, *, package: str = _TRACELENS, users: Sequence[str] = _TRACELENS_USERS
) -> str | None:
    """Why the importable ``package`` is not the pinned TraceLens ``pin`` (see the module); ``None`` when it is."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return f"{package} is not importable; install the pinned commit {pin}"
    location = Path(next(iter(spec.submodule_search_locations))).parent
    found = git_head(location) if (location / ".git").exists() else _installed_commit(package, location)
    if found is not None:
        if _same_commit(found, pin):
            return None
        return f"{package} at {location} is commit {found}, not the pinned {pin}"
    for user in users:
        try:
            importlib.import_module(user)
        except ImportError as exc:
            return f"{package} at {location} names no commit and {user} cannot import it: {exc}; install {pin}"
    return None


def build_preflight(site: PreflightSite) -> None:
    """Check what the host's native builds need (see the module).

    Raises:
        BuildStoreConfigError: Naming every item that is not as it must be.
    """
    problems: list[str] = []
    if geteuid() != 0:
        problems.append(f"the build service runs as uid {geteuid()}, not root: it cannot compile as user {NOBODY}")
    if which("setpriv") is None:
        problems.append(f"setpriv is not on PATH: the build service cannot compile as user {NOBODY}")
    upstreams = image_upstreams(site.image_root)
    checkouts = image_checkouts(site.sources, upstreams)
    problems.extend(_sources_problems(site, upstreams))
    problems.extend(reach_problems(site, checkouts))
    problems.extend(tamper_problems(site, checkouts))
    tracelens_issue = tracelens_problem()
    if tracelens_issue is not None:
        problems.append(tracelens_issue)
    if problems:
        raise BuildStoreConfigError(
            "native builds are configured (HYPERLOOM_BUILD_STORE) and this host is not ready for them: "
            + "; ".join(problems)
        )
