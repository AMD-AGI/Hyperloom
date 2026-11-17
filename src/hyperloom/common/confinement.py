# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a confined agent may write and read, and what it must not see, derived from configuration.

A :class:`Confinement` lists one launch's writable and readable paths, the credential paths to mask, its devices and
the paths only the app may reach; :meth:`Confinement.policy` turns it into a bubblewrap
:class:`~hyperloom_common.bwrap.BwrapPolicy`.

The sandbox starts from an empty root, so a path no bind shows does not exist inside; a credential inside a bound tree
is hidden there: a directory by a mask, a file (or a login not made yet) by binding the tree entry by entry without it,
since a mount over a file goes with the file's first rename and none covers one made later. The masked paths come from
configuration, never from a list of tool names in code (:func:`credential_paths`): this process's home, the XDG
directories, every environment variable that names a credential location, Hugging Face's login files, and the files
the app registers in :data:`PRIVATE_PATHS` as it creates them. So nothing broad under a credential location is bound
because a program lives there: :func:`toolchain_paths` binds a program's own directory, and its parent only away from
those locations; a search-path entry (or any other path it derives) that is one is not bound, and one that holds one
is bound entry by entry around it.

Nothing is resolved through what an agent writes. A name in a tree the agent writes may be a symlink it left there, to
any host path, or become one before the next launch; so a path reached through such a tree (looked up in a directory
at or below one, at any step of its resolution) is never bound at its target, never run, and never read for what to
bind (:func:`toolchain_paths`, :meth:`Confinement.policy`). Deriving the toolchain runs nothing an agent could have
written: the child's interpreter is asked where it imports from only when it lies outside those trees and is not
reached through one, and with no site processing, ``PYTHON*`` variable or startup file; its ``.pth`` files and
editable finders are read as data (:func:`interpreter_search_path`).
"""

from __future__ import annotations

import ast
import json
import logging
import os
import pwd
import re
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hyperloom_common.bwrap import SYSTEM_ROOTS, BwrapPolicy, BwrapPolicyError, Scratch, interpreter_roots
from hyperloom_common.credential_names import SENSITIVE_ENV_NAME_MARKERS

from hyperloom.common.env_safety import is_secret_shaped_env_name

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
#: Hugging Face's token file in its directory (``HF_HOME``), and the file of stored tokens beside the token file.
HF_TOKEN = "token"
HF_STORED_TOKENS = "stored_tokens"
#: A variable in a path setting, as :func:`os.path.expandvars` reads one.
_VARIABLE = re.compile(r"\$(\w+|\{[^}]*\})")
#: Where tools keep their logins in a home, relative to it: a path a search-path or data variable names that is one of
#: these, or holds one, is never bound whole (:func:`_around_logins`), though no variable names it (``~/.config`` is
#: the XDG configuration directory when ``XDG_CONFIG_HOME`` is unset).
HOME_LOGINS = (
    ".ssh",
    ".gnupg",
    ".netrc",
    ".git-credentials",
    ".config",
    ".claude",
    ".claude.json",
    ".codex",
    ".docker",
    ".aws",
    ".kube",
    ".azure",
    ".pypirc",
    ".npmrc",
    ".password-store",
    ".vault-token",
    ".local/share/keyrings",
)
#: Where tools keep their logins in the XDG configuration directory (``~/.config``, or ``XDG_CONFIG_HOME``), so that
#: a path below it is not bound whole when it is one of these either.
CONFIG_LOGINS = ("gh", "gcloud", "git", "hub", "rclone", "containers")
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
#: How many symlinks resolving one path may follow, as the kernel allows (``ELOOP`` past it, so nothing is bound).
_MAX_LINKS = 40

log = logging.getLogger(__name__)
_SAID: set[Path] = set()
_SAID_LOCK = threading.Lock()


def _real(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.realpath(path))


def _at_or_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _searches(path: str, roots: tuple[Path, ...]) -> bool:
    """Whether resolving the absolute ``path`` as the kernel does (each symlink followed where it is met, ``..`` the
    parent of where the walk is) looks a name up in a directory at or below one of ``roots``. A directory is checked
    before any name in it is followed, so no link in those trees is ever followed; the walk stops where the kernel's
    does, past :data:`_MAX_LINKS` links."""
    pending, current, links = deque(path.split(os.sep)), _ROOT, 0
    while pending:
        name = pending.popleft()
        if name in ("", os.curdir):
            continue
        if name == os.pardir:
            current = current.parent
            continue
        if any(_at_or_under(current, root) for root in roots):
            return True
        step = current / name
        try:
            target = os.readlink(step)
        except OSError:  # not a link, or nothing there: the walk goes on as realpath's does
            current = step
            continue
        links += 1
        if links > _MAX_LINKS:
            return False
        current = _ROOT if os.path.isabs(target) else current
        pending.extendleft(reversed(target.split(os.sep)))
    return False


def _reached(path: str | os.PathLike[str], roots: tuple[Path, ...]) -> bool:
    """Whether resolving ``path`` (as given, and normalized as a bind names it) looks a name up in one of the trees
    ``roots`` (resolved): there the agent decides what the name is, a symlink to any host path included, and can change
    it before the next launch. A relative path is not resolved here (:meth:`BwrapPolicy.problems` names it)."""
    text = os.fspath(path)
    if not roots or not os.path.isabs(text):
        return False
    return any(_searches(variant, roots) for variant in dict.fromkeys((text, os.path.normpath(text))))


def _agents(path: str | os.PathLike[str], roots: tuple[Path, ...]) -> bool:
    """Whether the agent writing ``roots`` (resolved) decides what ``path`` is: it is reached through one of them
    (:func:`_reached`), or resolves to one or below one."""
    return _reached(path, roots) or any(_at_or_under(_real(path), root) for root in roots)


def _resolved_trees(paths: Iterable[str | os.PathLike[str]]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(_real(path) for path in paths if os.path.isabs(path)))


def _agent_trees(writable: Iterable[Path], scratch: Scratch | None) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """The writable paths to bind, and the trees the agent writes or may have written (resolved).

    A writable path below another as given is left out (the one above shows it, writable; resolving it would follow
    what the agent left there), and so is one reached through ``scratch``, through where it points itself, or through
    the tree of another path kept: a link into a tree, whose target the agent decides. Such a target must decide
    nothing: pointed at ``/``, or at the directory that holds the app's link to a tree, it would have every path
    reached through it, the trees the agent wrote included. So which paths are kept is settled from both sides
    until neither moves: ``surely`` (reached through no tree of a path ``maybe`` kept) and ``maybe`` (reached
    through no tree of a path ``surely`` kept). The paths surely kept are bound; the trees are those of every path
    maybe kept, and ``scratch``. The two agree unless paths are reached through each other's trees, which takes a
    link in one of them (one the agent may have left there): then no path in doubt is bound, and each still counts
    as a tree, so one the agent wrote at an earlier launch is never taken for the app's."""
    given = list(
        dict.fromkeys(Path(os.path.normpath(path)) if os.path.isabs(path) else Path(path) for path in writable)
    )
    top = [path for path in given if not any(other in path.parents for other in given if other.is_absolute())]
    scratches = _resolved_trees((scratch.tmp, scratch.shm) if scratch else ())
    candidates = [path for path in top if not _reached(path, (*scratches, *_resolved_trees((path,))))]

    def unreached(kept: list[Path]) -> list[Path]:
        roots = (*_resolved_trees(kept), *scratches)
        return [path for path in candidates if not _reached(path, roots)]

    surely: list[Path] = []
    maybe = candidates
    for _ in range(len(candidates) + 1):  # each round that does not end keeps one more path surely
        maybe = unreached(surely)
        settled = unreached(maybe)
        if settled == surely:
            break
        surely = settled
    return tuple(surely), _resolved_trees((*maybe, *scratches))


def _trees(writable: Iterable[Path]) -> tuple[Path, ...]:
    """The trees an agent writing ``writable`` writes or may have written, resolved (:func:`_agent_trees`)."""
    return _agent_trees(writable, None)[1]


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


def _hidden(masked: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    """``masked`` resolved, split into the directories (masked by a tmpfs) and everything else: a file, a link, or a
    path nothing is at yet (kept out by not binding it, :func:`_narrowed`)."""
    directories: list[Path] = []
    others: list[Path] = []
    for path in dict.fromkeys(_real(mask) for mask in masked):
        (directories if os.path.isdir(path) else others).append(path)
    return directories, others


def _narrowed(binds: list[_Bind], hidden: list[Path], masks: list[Path], trees: tuple[Path, ...]) -> list[_Bind]:
    """``binds`` with each read-only directory that shows a path of ``hidden`` bound entry by entry instead, down the
    way to it, leaving that path out. A mount over a file goes with the file's first rename, and none covers one made
    later (the login a tool writes after the launch), so such a path is kept out by binding only what is beside it: a
    name the host adds later in a directory so bound is not shown. A directory of ``masks`` met on the way is left
    out too, and a path below one needs nothing more. Nothing in a tree the agent writes (``trees``) is listed, and a
    path of ``hidden`` there is left to :meth:`Confinement.policy`."""
    hidden = [path for path in hidden if not _in_any(path, (*trees, *masks))]
    found: list[_Bind] = []
    for bind in binds:
        below = [path for path in hidden if _at_or_under(path, bind.real)]
        found.extend(_entries(bind, below, masks) if below and not (bind.writable or bind.link) else [bind])
    return _kept_once(found, hidden)


def _in_any(path: Path, roots: Iterable[Path]) -> bool:
    return any(_at_or_under(path, root) for root in roots)


def _kept_once(found: list[_Bind], hidden: list[Path]) -> list[_Bind]:
    """``found`` each once, a read-only entry a writable bind takes left out, and a link only where it stays one."""
    taken = {bind.inside for bind in found if bind.writable}
    found = [bind for bind in dict.fromkeys(found) if bind.writable or bind.inside not in taken]
    shown = [bind.inside for bind in found if not bind.link and bind.inside == bind.real]
    return [bind for bind in found if not bind.link or _shows_link(bind, shown, hidden)]


def _shows_link(bind: _Bind, shown: list[Path], hidden: list[Path]) -> bool:
    """Whether a link met while binding entry by entry stays a link inside: its target is shown by another bind,
    and is not hidden (a link whose target nothing shows would be bound at its target)."""
    return any(_at_or_under(bind.real, other) for other in shown if other != bind.inside) and not any(
        _at_or_under(bind.real, path) for path in hidden
    )


def _entries(bind: _Bind, hidden: list[Path], masks: list[Path]) -> list[_Bind]:
    """The read-only ``bind`` of a directory as one bind per entry, without the paths of ``hidden`` (at or below it)
    and the directories of ``masks``, opened up the same way down to each hidden path; a link becomes the same link
    (:func:`_shows_link`). Nothing when the directory cannot be listed."""
    if bind.real in hidden:
        return []
    try:
        names = sorted(os.listdir(bind.real))
    except OSError:
        return []
    found: list[_Bind] = []
    for name in names:
        inside, real = bind.inside / name, bind.real / name
        below = [path for path in hidden if _at_or_under(path, real)]
        if real in masks or real in below:
            continue
        if os.path.islink(real):
            found.append(_Bind(inside, _real(real), writable=False))
        elif below and os.path.isdir(real):
            found.extend(_entries(_Bind(inside, real, writable=False), below, masks))
        elif not below:
            found.append(_Bind(inside, real, writable=False))
    return found


def _shown(
    candidates: list[_Bind], masked: Iterable[Path], trees: tuple[Path, ...]
) -> tuple[list[_Bind], tuple[str, ...]]:
    """The binds to make of ``candidates``, without what ``masked`` hides (:func:`_narrowed`), and the masks at every
    path they show. A file in a tree the agent writes (``trees``) stays a mask: :meth:`BwrapPolicy.problems` refuses
    it, by name."""
    masks, hidden = _hidden(masked)
    kept = _narrowed([bind for bind in candidates if not _shown_elsewhere(bind, candidates)], hidden, masks, trees)
    written = [path for path in hidden if os.path.exists(path) and _in_any(path, trees)]
    return kept, _translated((*masks, *written), kept)


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
        masked directories masked at every bound path that shows them, and a read-only tree that shows any other
        masked path (a file, or a login not made yet) bound entry by entry without it (:func:`_narrowed`); such a path
        in a tree the agent writes stays a mask, which :meth:`BwrapPolicy.problems` refuses. ``scratch`` is bound at
        ``/tmp`` and ``/dev/shm`` (else each is a fresh tmpfs); ``chdir`` is the working directory inside.

        Nothing is resolved through the trees the agent writes (the writable paths bound and ``scratch``), or may have
        written (:func:`_agent_trees`): a writable path below another, or reached through one, is left out (the one
        above shows it, writable), and a readable one is refused, since its target would be the agent's choice and its
        read-only bind could not hold.

        Raises:
            BwrapPolicyError: A readable path is reached through a tree the agent writes.
        """
        own, trees = _agent_trees(self.writable, scratch)
        reached = [path for path in self.readable if _reached(path, trees)]
        if reached:
            raise BwrapPolicyError(
                [f"{path} is reached through a tree the agent writes, which decides what it is" for path in reached]
            )
        writable = list(dict.fromkeys(_binds(own, True)))
        taken = {bind.inside for bind in writable}
        readable = [bind for bind in dict.fromkeys(_binds(self.readable, False)) if bind.inside not in taken]
        kept, masked = _shown([*writable, *readable], self.masked, trees)
        return BwrapPolicy(
            read_only=tuple(str(bind.inside) for bind in kept if bind.link or not bind.writable),
            writable=tuple(str(bind.inside) for bind in kept if bind.writable and not bind.link),
            masked=masked,
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


def _expanded(value: str, environ: Mapping[str, str], home: Path | None) -> str:
    """``value`` as ``huggingface_hub`` reads a path setting (:func:`os.path.expanduser`, then
    :func:`os.path.expandvars`), from ``environ`` and its ``home``: a variable it does not set is kept as written."""
    if home is not None and (value == "~" or value.startswith("~/")):
        value = str(home) + value[1:]
    return _VARIABLE.sub(lambda found: environ.get(found.group(1).strip("{}"), found.group(0)), value)


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
    yield from _hf_logins(environ, home)
    yield from registered


def _hf_logins(environ: Mapping[str, str], home: Path | None) -> tuple[Path, ...]:
    """Where Hugging Face keeps a login, made yet or not (``hf auth login`` writes it whenever it is run), as
    ``huggingface_hub`` resolves it: the token file (``HF_TOKEN_PATH``, else ``$HF_HOME/token``; ``HF_HOME`` defaults
    to ``$XDG_CACHE_HOME/huggingface``, that to ``~/.cache``) and the stored tokens beside it; and ``HF_HOME``'s pair
    too, where a login made without ``HF_TOKEN_PATH`` set goes."""

    def setting(name: str, default: str | None) -> str | None:
        value = environ.get(name)
        return _expanded(value, environ, home) if value else default

    cache = setting("XDG_CACHE_HOME", None if home is None else str(home / ".cache"))
    hf_home = setting("HF_HOME", None if cache is None else os.path.join(cache, "huggingface"))
    tokens = [setting("HF_TOKEN_PATH", None), None if hf_home is None else os.path.join(hf_home, HF_TOKEN)]
    found = [
        login
        for token in tokens
        if token and os.path.isabs(token)
        for login in (Path(token), Path(os.path.dirname(token), HF_STORED_TOKENS))
    ]
    return tuple(dict.fromkeys(found))


def _first_missing(path: Path) -> Path | None:
    """The first path missing on the way to ``path`` (resolved, not there): where a tool making it starts, in a
    directory that is there; ``None`` when what is there on the way is not a directory, so nothing can be made."""
    while not os.path.exists(path.parent):
        path = path.parent
    return path if os.path.isdir(path.parent) else None


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


def _masks_for(path: Path, visible: tuple[Path, ...], later: frozenset[Path]) -> list[Path]:
    """``path`` itself when nothing of ``visible`` is at or below it; else its entries, opened up around them. A path
    of ``later`` (a login file a tool writes when it logs in) counts while nothing is there yet: the first path
    missing on the way to it, which the tool makes, is hidden (:func:`_first_missing`). None for any other path this
    process cannot look at (below a directory it cannot search): no bind of its shows it either."""
    real = _real(path)
    if real == _ROOT:
        return []
    if not os.path.exists(real):
        missing = _first_missing(real) if path in later else None
        return [] if missing is None else [missing]
    if not any(_at_or_under(shown, real) for shown in visible):
        return [real]
    if real in visible or not real.is_dir():
        return []
    return _opened_up(real, visible, HOME_DEPTH)


def credential_paths(
    environ: Mapping[str, str],
    *,
    given: Iterable[Path] = (),
    allowed: Iterable[Path] = (),
    writable: Iterable[Path] = (),
) -> tuple[Path, ...]:
    """The paths to mask in a child of this process, from this process's configuration and the child's: the home
    (``HOME``, else the passwd entry's; and this process's passwd entry's either way), the XDG directories, the value
    of every variable that names a credential location (``:``-separated entries split, only absolute existing paths)
    and where Hugging Face keeps a login (:func:`_hf_logins`; one not made yet hidden at the first path missing on the
    way to it), each read from this process's environment and from ``environ`` (the child's), and the files
    registered in :data:`PRIVATE_PATHS`. A location of the child's in a tree it writes (``writable``, its scratch too
    when that is on the host: a home or tool directory re-pointed there) is its own, and is not masked; one in a tree
    only in doubt (:func:`_agent_trees`, not bound) is.

    A source that holds an ``allowed`` path (the policy's writable and readable paths) or a ``given`` one (handed to
    the child on purpose, so it stays readable) is not masked whole: its other entries are, down to :data:`HOME_DEPTH`
    levels. ``given`` paths themselves are never masked, nor is ``/``. :meth:`Confinement.policy` keeps only the masks
    a bind shows. An ``allowed`` or ``given`` path reached through a tree the child writes, or may have written,
    counts for nothing here: the agent decides where it points (a link to a credential in a readable tree, left for
    the next launch), and the tree shows it already.
    """
    app = os.environ
    own, agent = _agent_trees(writable, None)
    written = _resolved_trees(own)
    sources = [*_sources(app, _app_home(app), PRIVATE_PATHS.paths()), _passwd_home()]
    sources += [path for path in _sources(environ, _app_home(environ), ()) if not _agents(path, written)]
    shown = tuple(dict.fromkeys(_real(path) for path in (*allowed, *given) if not _reached(path, agent)))
    later = frozenset((*_hf_logins(app, _app_home(app)), *_hf_logins(environ, _app_home(environ))))
    masks: dict[Path, None] = {}
    for source in dict.fromkeys(source for source in sources if source is not None):
        masks.update(dict.fromkeys(_masks_for(source, shown, later)))
    # A login not made yet below another location masked whole is hidden with it.
    return tuple(mask for mask in masks if os.path.exists(mask) or not any(other in mask.parents for other in masks))


#: Run by the child's interpreter with no site processing and no ``PYTHON*`` variable (``-I -S``), so no ``.pth``
#: line, ``sitecustomize`` or startup file runs: its import path without site directories, and its site directories
#: (a virtual environment's too, which only site processing would add), whose ``.pth`` files are then read as data.
_SEARCH_PATH = r"""
import json, os, site, sys
bin_dir = os.path.dirname(os.path.abspath(sys.executable))
prefixes = [sys.prefix, sys.exec_prefix]
if any(os.path.isfile(os.path.join(d, "pyvenv.cfg")) for d in (bin_dir, os.path.dirname(bin_dir))):
    prefixes.insert(0, os.path.dirname(bin_dir))
print(json.dumps({"path": sys.path, "site": site.getsitepackages(prefixes)}))
"""
#: The module a setuptools editable install's ``.pth`` file imports, whose ``MAPPING`` names the source trees.
_EDITABLE_FINDERS = "__editable___*_finder.py"


@lru_cache(maxsize=16)
def _search_path(python: str, library_path: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``python``'s import path and site directories (:data:`_SEARCH_PATH`); empty when it does not answer."""
    env = {} if library_path is None else {"LD_LIBRARY_PATH": library_path}
    command = [python, "-I", "-S", "-c", _SEARCH_PATH]
    try:
        done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60, check=True, cwd="/")
        found = json.loads(done.stdout)
        path, sites = found["path"], found["site"]
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError):
        return (), ()
    return _absolute_strings(path), _absolute_strings(sites)


def _absolute_strings(values: object) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(entry for entry in values if isinstance(entry, str) and os.path.isabs(entry))


def _pth_entries(site_dir: Path) -> Iterator[Path]:
    """The directories the ``.pth`` files of ``site_dir`` add to the import path, read as site processing reads them
    (a line that is not blank, a comment or an ``import`` statement is a path, relative to ``site_dir``), never run."""
    for pth in sorted(site_dir.glob("*.pth")):
        try:
            lines = pth.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            if line.strip() and not line.startswith(("#", "import ", "import\t")):
                entry = Path(os.path.normpath(site_dir / line.rstrip()))
                if entry.exists():
                    yield entry


def _mapping(node: ast.stmt) -> object:
    """The literal assigned to ``MAPPING`` by ``node``; ``None`` for any other statement."""
    target = node.target if isinstance(node, ast.AnnAssign) else None
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    value = getattr(node, "value", None)
    if not (isinstance(target, ast.Name) and target.id == "MAPPING" and value is not None):
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError):
        return None


def _editable_sources(site_dir: Path) -> Iterator[Path]:
    """The source trees setuptools editable installs in ``site_dir`` map, read from their finders' ``MAPPING``
    literal, never imported."""
    for finder in sorted(site_dir.glob(_EDITABLE_FINDERS)):
        try:
            tree = ast.parse(finder.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for node in tree.body:
            mapping = _mapping(node)
            if isinstance(mapping, dict):
                yield from (Path(path) for path in mapping.values() if isinstance(path, str) and os.path.isabs(path))


def interpreter_search_path(environ: Mapping[str, str], *, writable: Iterable[Path]) -> tuple[Path, ...]:
    """Where the child's ``python3`` imports from: its import path and site directories, the directories their
    ``.pth`` files add, and the source trees of setuptools-style editable installs, which no import path entry names.
    Not ``PYTHONPATH`` (:func:`toolchain_paths` adds it) nor the user site directory (an agent's home is its own).

    Nothing an agent could have written runs here, as the app: ``writable`` names the trees the agent writes, and
    the interpreter is the first ``python3`` on the child's ``PATH`` that neither lies in them nor is reached through
    them (an interpreter run by such a path reads the agent's ``pyvenv.cfg``, and through it imports the agent's
    standard library), run with no site processing, ``PYTHON*`` variable or startup file and with only the
    ``LD_LIBRARY_PATH`` entries the agent does not decide; a site directory it decides is not read. The interpreter is
    asked once per path and library path; empty when there is no such interpreter or it does not answer."""
    agent = _trees(writable)
    python = _which("python3", environ, outside=agent)
    if python is None:
        return ()
    libraries = [
        str(entry) for entry in _absolute_entries(environ.get("LD_LIBRARY_PATH", "")) if not _agents(entry, agent)
    ]
    path, sites = _search_path(python, os.pathsep.join(libraries) or None)
    found: dict[Path, None] = dict.fromkeys(Path(entry) for entry in (*path, *sites))
    for site_dir in map(Path, sites):
        if site_dir.is_dir() and not _agents(site_dir, agent):
            found.update(dict.fromkeys(_pth_entries(site_dir)))
            found.update(dict.fromkeys(_editable_sources(site_dir)))
    return tuple(found)


def _which(program: str, environ: Mapping[str, str], *, outside: tuple[Path, ...] = ()) -> str | None:
    """``program`` on the child's ``PATH`` (never the working directory): the first that neither resolves into one of
    ``outside`` (resolved paths) nor is reached through one, by the path that was checked; ``None`` when there is
    none."""
    for entry in environ.get("PATH", "").split(os.pathsep):
        found = shutil.which(program, path=entry) if os.path.isabs(entry) else None
        if found is not None and not _agents(found, outside):
            return found
    return None


def _homes(environ: Mapping[str, str]) -> list[Path]:
    """The child's home, this process's, and this process's passwd entry's, as given."""
    return [home for home in dict.fromkeys((_app_home(environ), _app_home(os.environ), _passwd_home())) if home]


def _credential_roots(environ: Mapping[str, str]) -> tuple[Path, ...]:
    """The credential locations of the child's environment and of this process's own (the child's home may be
    re-pointed; the app's is where its logins are), and the well-known login locations in each home
    (:data:`HOME_LOGINS`) and configuration directory (:data:`CONFIG_LOGINS`), there or not, resolved."""
    found = [*_sources(environ, _app_home(environ), ()), *_sources(os.environ, _app_home(os.environ), ())]
    homes = _homes(environ)
    configs = [home / ".config" for home in homes]
    configs += [
        Path(env["XDG_CONFIG_HOME"]) for env in (environ, os.environ) if env.get("XDG_CONFIG_HOME", "")[:1] == "/"
    ]
    found.extend(home / login for home in homes for login in HOME_LOGINS)
    found.extend(config / login for config in configs for login in CONFIG_LOGINS)
    found.append(_passwd_home() or _ROOT)
    return tuple(dict.fromkeys(_real(path) for path in found if path != _ROOT))


def _program_dirs(argv0: str, environ: Mapping[str, str], agent: tuple[Path, ...]) -> Iterator[Path]:
    """What running ``argv0`` needs, through the link and resolved: the program's directory (the program alone when
    that directory is or holds a credential location), and its parent unless that is ``/`` or is, holds or lies
    below one. A native installer's ``~/.local/bin/claude`` so binds ``~/.local/bin`` and the version's directory,
    never all of ``~/.local``, where other tools keep their logins. Nothing when the program the child runs is one
    the agent decides (in or reached through ``agent``, the trees it writes, resolved): the tree shows it, and where
    it points is the agent's choice; nor for a relative path, which names a program relative to the child's working
    directory, not this process's."""
    if os.sep in argv0 and not os.path.isabs(argv0):
        return
    found = argv0 if os.sep in argv0 else _which(argv0, environ)
    if found is None or _agents(found, agent):
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


def _say_once(path: Path, message: str) -> None:
    with _SAID_LOCK:
        if path in _SAID:
            return
        _SAID.add(path)
    log.warning(message)


def _entries_around(path: Path, roots: tuple[Path, ...], depth: int) -> Iterator[Path]:
    """The entries of the directory ``path`` (which holds a credential location of ``roots``, resolved) that neither
    are nor hold one, and those that hold one opened up the same way ``depth`` levels down; an entry is judged by
    where it resolves, so a link to a credential location is left out too. What lies deeper, or cannot be listed, is
    left out: nothing of it is bound."""
    try:
        names = sorted(os.listdir(path))
    except OSError:
        return
    for name in names:
        entry = path / name
        real = _real(entry)
        if not any(_at_or_under(root, real) for root in roots):
            yield entry
        elif real not in roots and depth > 1 and real.is_dir():
            yield from _entries_around(entry, roots, depth - 1)


def _a_login_file(path: Path, real: Path, logins: tuple[Path, ...]) -> bool:
    """Whether ``path`` (resolved: ``real``) is not a directory and lies below a login location of ``logins``
    (resolved), or is a link that does."""
    if real.is_dir():
        return False
    where = (real, _real(path.parent) / path.name)
    return any(_at_or_under(at, login) and at != login for at in where for login in logins)


def _around_logins(
    paths: Iterable[Path], roots: tuple[Path, ...], logins: tuple[Path, ...], programs: frozenset[Path]
) -> Iterator[Path]:
    """``paths`` without the credential locations ``roots`` (resolved): one that is one (a ``PATH`` entry that is the
    home) is not bound, and said once, since bound whole it would show every login in it, and a mask is only ever
    below a bound path; one that holds one is bound entry by entry around it (:func:`_entries_around`), so no login
    in it is shown, nor one made there after the launch. Below a login location (``logins``: ``roots`` but the
    homes) only a directory is bound (a tool's program folder): a file there, or a link to one or from there, that a
    variable names (``MODEL_PATH=~/.ssh/id_rsa``) is a login, so it is not bound, and said once; the program the
    child runs (``programs``) is."""
    for path in paths:
        real = _real(path)
        if real in roots:
            _say_once(
                real,
                f"confinement: {path} is not shown to the agent: it is a credential location (a home, or a tool's "
                "login), and bound whole it would show every login in it; programs there are not found inside",
            )
        elif path not in programs and _a_login_file(path, real, logins):
            _say_once(
                real,
                f"confinement: {path} is not shown to the agent: it is a file in a tool's login location, which "
                "holds its login; only directories there (a tool's programs) are bound",
            )
        elif any(real in root.parents for root in roots):
            yield from _entries_around(path, roots, HOME_DEPTH)
        else:
            yield path


def toolchain_paths(environ: Mapping[str, str], argv0: str, *, writable: Iterable[Path]) -> tuple[Path, ...]:
    """What a child needs to read to run ``argv0`` under ``environ``: the system roots, this interpreter's prefixes,
    the child's interpreter's import path (:func:`interpreter_search_path`), every entry of its search-path variables,
    the program's directory (and parent, away from credential locations; for an absolute path or a name on ``PATH``),
    its CA bundles, the model and benchmark trees it names, and the resolver configuration. Not ``/proc`` (the
    sandbox mounts its own), ``/sys`` (:attr:`Confinement.sys_paths`), nor what lies in the trees the child writes
    (``writable``; its scratch too, when that is on the host) or is reached through them: those show it already, a
    read-only bind there would take the child's own files from it, and where a link there points is the agent's
    choice. Nor, beyond the system roots, a credential location of the child's or this process's (a ``PATH`` entry
    that is the home), what holds one only entry by entry around it, and below a tool's login location only
    directories, never a file a variable names there (:func:`_around_logins`)."""
    agent = _trees(writable)
    derived: list[Path] = [Path(root) for root in interpreter_roots()]
    derived.extend(interpreter_search_path(environ, writable=agent))
    for name in (*SEARCH_PATH_VARIABLES, *CA_VARIABLES, *DATA_VARIABLES):
        derived.extend(_absolute_entries(environ.get(name, "")))
    programs = tuple(_program_dirs(argv0, environ, agent))
    derived.extend(programs)
    derived.extend(_resolver())
    kept = [path for path in derived if path != _ROOT and not _agents(path, agent)]
    found = [Path(root) for root in SYSTEM_ROOTS]
    roots = _credential_roots(environ)
    homes = {_real(home) for home in _homes(environ)}
    logins = tuple(root for root in roots if root not in homes)
    found.extend(_around_logins(dict.fromkeys(kept), roots, logins, frozenset(programs)))
    return tuple(dict.fromkeys(path for path in found if path != _ROOT and not _agents(path, agent)))
