# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a confined agent may write and read, and what it must not see, derived from configuration.

A :class:`Confinement` lists one launch's writable and readable paths, the credential paths to mask, its devices and
the paths only the app may reach; :meth:`Confinement.policy` turns it into a bubblewrap
:class:`~hyperloom_common.bwrap.BwrapPolicy`.

The sandbox starts from an empty root, so a path no bind shows does not exist inside; a mask is needed only for a
credential inside a bound tree. The masked paths come from configuration, never from a list of tool names in code
(:func:`credential_paths`): this process's home, the XDG directories, every environment variable that names a credential
location, Hugging Face's login files, and the files the app registers in :data:`PRIVATE_PATHS` as it creates them. So
nothing broad under a credential location is bound because a program lives there: :func:`toolchain_paths` binds a
program's own directory, and its parent only away from those locations.
"""

from __future__ import annotations

import json
import os
import pwd
import shutil
import subprocess
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hyperloom_common.bwrap import SYSTEM_ROOTS, BwrapPolicy, Scratch, interpreter_roots

from hyperloom.common.env_safety import SENSITIVE_ENV_NAME_MARKERS, is_secret_shaped_env_name

__all__ = [
    "PRIVATE_PATHS",
    "Confinement",
    "PrivatePaths",
    "credential_paths",
    "interpreter_search_path",
    "toolchain_paths",
]

#: Variables that name where a tool keeps its login (configuration names, not paths).
LOGIN_LOCATION_VARIABLES = frozenset(
    {
        "KUBECONFIG",
        "NETRC",
        "DOCKER_CONFIG",
        "GH_CONFIG_DIR",
        "GNUPGHOME",
        "SSH_AUTH_SOCK",
        "CLAUDE_CONFIG_DIR",
        "CODEX_HOME",
        "AWS_CONFIG_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "CLOUDSDK_CONFIG",
        "AZURE_CONFIG_DIR",
        "PIP_CONFIG_FILE",
        "HF_TOKEN_PATH",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
)
#: A variable ending so names a file holding a credential.
CREDENTIAL_FILE_SUFFIXES = ("_KEY_FILE", "_KEY_PATH", "_TOKEN_FILE", "_TOKEN_PATH", "_CREDENTIALS_FILE")
XDG_VARIABLES = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME")
#: Hugging Face's login files under its config directory (``HF_HOME``, else ``$XDG_CACHE_HOME/huggingface``).
HF_LOGIN_FILES = ("token", "stored_tokens")
#: Variables whose value is a CA bundle the child needs to read.
CA_VARIABLES = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS", "GIT_SSL_CAINFO", "CURL_CA_BUNDLE")
#: Variables whose ``:``-separated entries the child searches for programs, libraries and modules.
SEARCH_PATH_VARIABLES = ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH")
#: Variables naming data trees the child reads (models, the benchmark checkout).
DATA_VARIABLES = ("HF_HOME", "HF_HUB_CACHE", "INFERENCEX_PATH", "MODEL_PATH")
#: How deep a credential directory that holds an allowed path is opened up, one level of entries at a time.
HOME_DEPTH = 4
#: The resolver configuration, which a host may keep as a link out of ``/etc`` (systemd-resolved's stub under ``/run``).
RESOLV_CONF = Path("/etc/resolv.conf")
_ETC = Path("/etc")
_ROOT = Path("/")


def _real(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.realpath(path))


def _at_or_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


@dataclass(frozen=True)
class _Bind:
    """A path shown inside (``inside``) and the host path it shows (``real``), writable or not."""

    inside: Path
    real: Path
    writable: bool

    @property
    def link(self) -> bool:
        """Whether this is a symlink the sandbox recreates (its target is shown by the bind of its real path)."""
        return self.inside != self.real and self.inside.is_symlink()


def _binds(paths: Iterable[Path], writable: bool) -> Iterator[_Bind]:
    """``paths`` that exist, each at its resolved path and, where that differs, at the given one too."""
    for path in paths:
        if not path.is_absolute():
            yield _Bind(path, path, writable)  # refused by BwrapPolicy.problems, named
        elif path.exists():
            given, real = Path(os.path.normpath(path)), _real(path)
            yield _Bind(real, real, writable)
            if given != real:
                yield _Bind(given, real, writable)


def _shown_elsewhere(bind: _Bind, kept: Iterable[_Bind]) -> bool:
    """Whether another bind already shows ``bind``: a given path below any bind (through the links there), or a
    read-only one whose nearest bind above is read-only too."""
    above = [other for other in kept if other.inside != bind.inside and _at_or_under(bind.inside, other.inside)]
    if not above:
        return False
    if bind.inside != bind.real:
        return True
    return not bind.writable and not max(above, key=lambda other: len(other.inside.parts)).writable


def _translated(masked: Iterable[Path], binds: Iterable[_Bind]) -> tuple[str, ...]:
    """The masks at every path inside that shows them (a mask outside every bind is left out: nothing is there), each
    once and none below another (a mask hides what is below it, and is read-only)."""
    shown = [bind for bind in binds if not bind.link]
    found: dict[Path, None] = {}
    for mask in masked:
        if not os.path.lexists(mask):
            continue
        real = _real(mask)
        for bind in shown:
            if _at_or_under(real, bind.real):
                found[bind.inside / real.relative_to(bind.real)] = None
    return tuple(str(mask) for mask in found if not any(other in mask.parents for other in found))


@dataclass(frozen=True)
class Confinement:
    """One launch's binds (``writable``, ``readable``), its masks, devices (``/dev/kfd``, ``/dev/dri``), the paths only
    the app may reach (``app_only``: never bound, nor anything above them), the ``/sys`` paths it reads, and whether
    it gets a network of its own."""

    writable: tuple[Path, ...] = ()
    readable: tuple[Path, ...] = ()
    masked: tuple[Path, ...] = ()
    devices: tuple[Path, ...] = ()
    app_only: tuple[Path, ...] = ()
    sys_paths: tuple[Path, ...] = (Path("/sys"),)
    own_network: bool = False

    def policy(self, *, scratch: Scratch | None = None, chdir: Path | None = None) -> BwrapPolicy:
        """The bubblewrap policy: missing paths dropped; each path bound at its resolved path, and at the given one
        too where that differs (a link becomes the same link inside, ``/bin -> usr/bin``); a readable path that is
        also writable bound once, writable; a read-only path a read-only bind above already shows left out; the
        masks at every bound path that shows them. ``scratch`` is bound at ``/tmp`` and ``/dev/shm`` (else each is
        a fresh tmpfs); ``chdir`` is the working directory inside."""
        writable = list(dict.fromkeys(_binds(self.writable, True)))
        taken = {bind.inside for bind in writable}
        readable = [bind for bind in dict.fromkeys(_binds(self.readable, False)) if bind.inside not in taken]
        candidates = [*writable, *readable]
        kept = [bind for bind in candidates if not _shown_elsewhere(bind, candidates)]
        return BwrapPolicy(
            read_only=tuple(str(bind.inside) for bind in kept if bind.link or not bind.writable),
            writable=tuple(str(bind.inside) for bind in kept if bind.writable and not bind.link),
            masked=_translated(self.masked, kept),
            devices=tuple(map(str, self.devices)),
            tmp=scratch.tmp if scratch else None,
            shm=scratch.shm if scratch else None,
            sys_paths=tuple(map(str, self.sys_paths)),
            own_network=self.own_network,
            chdir=None if chdir is None else str(chdir),
            app_only=tuple(map(str, self.app_only)),
        )


class PrivatePaths:
    """Credential files and directories the app creates at run time; each owner registers its path when it makes it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._why: dict[Path, str] = {}

    def add(self, path: Path, why: str) -> None:
        with self._lock:
            self._why.setdefault(Path(path), why)

    def paths(self) -> tuple[Path, ...]:
        with self._lock:
            return tuple(self._why)


#: The process-wide registry :func:`credential_paths` reads.
PRIVATE_PATHS = PrivatePaths()


def _names_a_credential(name: str) -> bool:
    upper = name.upper()
    return (
        upper in LOGIN_LOCATION_VARIABLES
        or upper.endswith(CREDENTIAL_FILE_SUFFIXES)
        or is_secret_shaped_env_name(upper)
        or any(marker in upper for marker in SENSITIVE_ENV_NAME_MARKERS)
    )


def _absolute_entries(value: str) -> Iterator[Path]:
    for entry in value.split(os.pathsep):
        if os.path.isabs(entry) and os.path.exists(entry):
            yield Path(entry)


def _passwd_home() -> Path | None:
    try:
        return Path(pwd.getpwuid(os.geteuid()).pw_dir)
    except KeyError:
        return None


def _app_home(environ: Mapping[str, str]) -> Path | None:
    home = environ.get("HOME")
    return Path(home) if home and os.path.isabs(home) else _passwd_home()


def _hf_home(environ: Mapping[str, str], home: Path | None) -> Path | None:
    if environ.get("HF_HOME"):
        return Path(environ["HF_HOME"])
    if environ.get("XDG_CACHE_HOME"):
        return Path(environ["XDG_CACHE_HOME"], "huggingface")
    return None if home is None else home / ".cache" / "huggingface"


def _sources(environ: Mapping[str, str], home: Path | None, registered: Iterable[Path]) -> Iterator[Path]:
    """Every configured credential location: the home, XDG directories, credential variables, HF's login, registered."""
    if home is not None:
        yield home
    for name in XDG_VARIABLES:
        if environ.get(name) and os.path.isabs(environ[name]):
            yield Path(environ[name])
    for name, value in environ.items():
        if _names_a_credential(name):
            yield from _absolute_entries(value)
    hf_home = _hf_home(environ, home)
    if hf_home is not None:
        yield from (hf_home / login for login in HF_LOGIN_FILES)
    yield from registered


def _opened_up(directory: Path, visible: tuple[Path, ...], depth: int) -> list[Path]:
    """The entries of ``directory`` to mask so that every path of ``visible`` below it stays reachable: each entry that
    is not one of them nor leads to one, opened up the same way ``depth`` levels down."""
    try:
        entries = sorted(_real(entry) for entry in directory.iterdir())
    except OSError:
        return []
    masks: list[Path] = []
    for entry in entries:
        leads = [path for path in visible if _at_or_under(path, entry)]
        if not leads:
            masks.append(entry)
        elif entry not in leads and depth > 1 and entry.is_dir():
            masks.extend(_opened_up(entry, visible, depth - 1))
    return masks


def _masks_for(path: Path, visible: tuple[Path, ...]) -> list[Path]:
    """``path`` itself when nothing of ``visible`` is at or below it; else its entries, opened up around them."""
    real = _real(path)
    if not real.exists() or real == _ROOT:
        return []
    if not any(_at_or_under(shown, real) for shown in visible):
        return [real]
    if real in visible or not real.is_dir():
        return []
    return _opened_up(real, visible, HOME_DEPTH)


def credential_paths(
    environ: Mapping[str, str],
    *,
    app_home: Path | None = None,
    registered: Iterable[Path] = (),
    given: Iterable[Path] = (),
    allowed: Iterable[Path] = (),
) -> tuple[Path, ...]:
    """The paths to mask in a child of this process: its home (``app_home``, else ``HOME``, else its passwd entry), the
    XDG directories, the value of every variable of ``environ`` that names a credential location (``:``-separated
    entries split, only absolute existing paths), Hugging Face's login files, and ``registered``.

    A source that holds an ``allowed`` path (the policy's writable and readable paths) or a ``given`` one (handed to
    the child on purpose, so it stays readable) is not masked whole: its other entries are, down to :data:`HOME_DEPTH`
    levels. ``given`` paths themselves are never masked, nor is ``/``. :meth:`Confinement.policy` keeps only the masks
    a bind shows.
    """
    home = app_home if app_home is not None else _app_home(environ)
    shown = tuple(dict.fromkeys(_real(path) for path in (*allowed, *given)))
    masks: dict[Path, None] = {}
    for source in _sources(environ, home, registered):
        masks.update(dict.fromkeys(_masks_for(source, shown)))
    return tuple(masks)


@lru_cache(maxsize=16)
def _search_path(python: str, environ: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    script = (
        "import json, sys\n"
        "found = list(sys.path)\n"
        "for name, module in list(sys.modules.items()):\n"
        "    if name.startswith('__editable__') and isinstance(getattr(module, 'MAPPING', None), dict):\n"
        "        found.extend(str(p) for p in module.MAPPING.values())\n"
        "print(json.dumps(found))\n"
    )
    try:
        done = subprocess.run(
            [python, "-c", script], env=dict(environ), capture_output=True, text=True, timeout=60, check=True, cwd="/"
        )
        found = json.loads(done.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return ()
    return tuple(str(entry) for entry in found if isinstance(entry, str) and os.path.isabs(entry))


def interpreter_search_path(environ: Mapping[str, str]) -> tuple[Path, ...]:
    """Where the child's ``python3`` (the first on its ``PATH``) imports from: its ``sys.path`` under the child's
    environment, plus the source trees of setuptools-style editable installs, which no ``sys.path`` entry names.
    Computed once per interpreter and environment; empty when there is no such interpreter or it does not answer."""
    python = _which("python3", environ)
    if python is None:
        return ()
    return tuple(Path(entry) for entry in _search_path(python, tuple(sorted(environ.items()))))


def _which(program: str, environ: Mapping[str, str]) -> str | None:
    """``program`` on the child's ``PATH`` (never the working directory); ``None`` when it is not there."""
    search = os.pathsep.join(entry for entry in environ.get("PATH", "").split(os.pathsep) if os.path.isabs(entry))
    return shutil.which(program, path=search) if search else None


def _credential_roots(environ: Mapping[str, str]) -> tuple[Path, ...]:
    """The credential locations of the child's environment and of this process's own (the child's home may be
    re-pointed; the app's is where its logins are), resolved."""
    found = [*_sources(environ, _app_home(environ), ()), *_sources(os.environ, _app_home(os.environ), ())]
    found.append(_passwd_home() or _ROOT)
    return tuple(dict.fromkeys(_real(path) for path in found if path != _ROOT))


def _program_dirs(argv0: str, environ: Mapping[str, str]) -> Iterator[Path]:
    """What running ``argv0`` needs, through the link and resolved: the program's directory (the program alone when
    that directory is or holds a credential location), and its parent unless that is ``/`` or is, holds or lies
    below one. A native installer's ``~/.local/bin/claude`` so binds ``~/.local/bin`` and the version's directory,
    never all of ``~/.local``, where other tools keep their logins."""
    found = argv0 if os.sep in argv0 else _which(argv0, environ)
    if found is None:
        return
    roots = _credential_roots(environ)
    for program in dict.fromkeys((Path(os.path.normpath(found)), _real(found))):
        directory, parent = _real(program.parent), _real(program.parent.parent)
        if any(_at_or_under(root, directory) for root in roots):
            yield program
            continue
        yield program.parent
        if parent != _ROOT and not any(_at_or_under(root, parent) or _at_or_under(parent, root) for root in roots):
            yield program.parent.parent


def _resolver() -> Iterator[Path]:
    """Where ``/etc/resolv.conf`` points, when that is outside ``/etc`` (a bound ``/etc`` alone shows a dangling
    link there)."""
    if RESOLV_CONF.is_symlink():
        target = _real(RESOLV_CONF)
        if target.exists() and not _at_or_under(target, _ETC):
            yield target


def toolchain_paths(environ: Mapping[str, str], argv0: str) -> tuple[Path, ...]:
    """What a child needs to read to run ``argv0`` under ``environ``: the system roots, this interpreter's prefixes,
    the child's interpreter's import path, every entry of its search-path variables, the program's directory (and
    parent, away from credential locations), its CA bundles, the model and benchmark trees it names, and the
    resolver configuration. Not ``/proc`` (the sandbox mounts its own) nor ``/sys`` (:attr:`Confinement.sys_paths`)."""
    found: list[Path] = [Path(root) for root in SYSTEM_ROOTS]
    found.extend(Path(root) for root in interpreter_roots())
    found.extend(interpreter_search_path(environ))
    for name in (*SEARCH_PATH_VARIABLES, *CA_VARIABLES, *DATA_VARIABLES):
        found.extend(_absolute_entries(environ.get(name, "")))
    found.extend(_program_dirs(argv0, environ))
    found.extend(_resolver())
    return tuple(dict.fromkeys(path for path in found if path != _ROOT))
