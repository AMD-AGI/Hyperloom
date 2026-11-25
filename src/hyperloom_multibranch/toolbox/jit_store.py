# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared compiled-code cache as a toolbox's host publishes to it: which directory, and whether it may.

The runners load what the cache holds as compiled code, and the snapshots prove only their own content, so only the
host's account may write it: :func:`private_jit_store` refuses a directory another account owns or could change, and a
link there. The local toolbox and the hosted one both build their store with :func:`session_jit_store`, which also
prunes it (:mod:`hyperloom.measure.jit.prune`) as the session starts.

A store the check refuses, or one that cannot be made or checked (a name below a file, a directory this account cannot
enter or write), does not stop the session: both toolboxes are built inside it, where a hosted run that raises is
started again into the same refusal. The session runs without publishing instead, logs one warning naming the store,
why, and what that session does without it, and records the same in :data:`JIT_STORE_NOTE` of the session directory.
What it does without it depends on the path (the ``without`` text each toolbox passes):

* hosted (:data:`REFUSED_HOSTED`): the host publishes nothing to it. Hosted runners never read the store (every hosted
  test builds cold, and one that patches aiter's ``csrc`` or ``hsa`` is refused by its runner), so it should lie on the
  host's own disk, where no sandbox can reach it;
* local (:data:`REFUSED_LOCAL`): the runners are not told the store, so no test has an aiter JIT directory to build
  touched modules in, and candidates that patch aiter's ``csrc`` or ``hsa`` are refused for the whole session.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from hyperloom.measure.jit.prune import prune_logged
from hyperloom.measure.jit.store import DirJitStore
from hyperloom_common.fs_utils import is_private_dir
from hyperloom_common.io import SHARED, atomic_write_json
from hyperloom_multibranch.toolbox.build_preflight import changeable_by_others

__all__ = [
    "JIT_STORE_ENV",
    "JIT_STORE_NOTE",
    "REFUSED_HOSTED",
    "REFUSED_LOCAL",
    "JitStoreNotPrivate",
    "named_jit_store",
    "private_jit_store",
    "session_jit_store",
]

#: The host environment variable that names the shared compiled-code cache when the toolbox's options do not.
JIT_STORE_ENV = "HYPERLOOM_JIT_STORE"
#: The file of the session directory that says the compiled-code cache was refused, and why.
JIT_STORE_NOTE = "jit_store.json"

#: What a hosted session does without a refused store, said in its warning and note.
REFUSED_HOSTED = (
    "this host publishes nothing to it; hosted runners do not read it in any case: every test builds cold, and a "
    "candidate that patches aiter's csrc or hsa is refused by its runner"
)
#: What a local session does without a refused store, said in its warning and note.
REFUSED_LOCAL = (
    "the runners are not told it and nothing is published, so no test can build aiter's modules: candidates that patch "
    "aiter's csrc or hsa are refused for the whole session"
)

log = logging.getLogger(__name__)


class JitStoreNotPrivate(ValueError):
    """The compiled-code cache is not a directory of this account's that only it can write, or another account could
    change a directory above it."""


def named_jit_store(option: Path | None, environ: Mapping[str, str]) -> Path | None:
    """The absolute directory ``option`` names, else ``HYPERLOOM_JIT_STORE`` of ``environ``; ``None`` when neither does.

    The runner starts in a directory of its own, so a relative path is made absolute here, once. A link is not resolved:
    :func:`private_jit_store` judges what stands under the name, and a link there is refused, never followed.
    """
    named = option or environ.get(JIT_STORE_ENV, "").strip()
    return Path(named).expanduser().absolute() if named else None


def private_jit_store(root: Path) -> Path:
    """``root``, made (mode 0700) when nothing stands under its name.

    Raises:
        JitStoreNotPrivate: ``root`` is not a directory of this account's (a link to one is refused too), or its group or
            others can write it: the runners load what it holds as compiled code, and the snapshots prove only their own
            content. Another account can plant the default name, which stops the build here but feeds it nothing. Nor
            may a directory above it be another account's or writable by its group or others unless it is sticky (as a
            host's temporary directory is): whoever can change one could move the store aside and put one of their own
            in its place (:func:`~.build_preflight.changeable_by_others`, the build store's check).
    """
    with contextlib.suppress(FileExistsError):  # what already stands there, a link included, is judged below
        root.mkdir(mode=0o700, parents=True)
    if not is_private_dir(root):
        raise JitStoreNotPrivate(
            f"the compiled-code cache {root} is not a directory of this account's that only it can write (nor may it "
            f"be a link to one); name one that is with {JIT_STORE_ENV}"
        )
    problem = changeable_by_others(root, os.getuid())
    if problem is not None:
        raise JitStoreNotPrivate(
            f"users other than this account could change the compiled-code cache {root}: {problem}; name one below "
            f"directories only this account or root can change with {JIT_STORE_ENV}"
        )
    return root


def session_jit_store(root: Path, session_dir: Path, without: str) -> DirJitStore | None:
    """The store at ``root`` a session publishes to, checked (:func:`private_jit_store`) and pruned of what no test has
    used for long; a prune that fails is logged and the session goes on.

    ``None`` when the check refuses it or it cannot be made or checked (an :class:`OSError`): the session runs without
    publishing, and says so once in the log and in :data:`JIT_STORE_NOTE` of ``session_dir``, with ``without``, what
    this session does without it (:data:`REFUSED_HOSTED`, :data:`REFUSED_LOCAL`; see the module). Nothing was pruned
    then.
    """
    try:
        store = DirJitStore(private_jit_store(root))
    except JitStoreNotPrivate as exc:
        reason = str(exc)
    except OSError as exc:
        reason = (
            f"the compiled-code cache {root} cannot be made or checked ({type(exc).__name__}: {exc.strerror or exc})"
        )
    else:
        prune_logged(store)
        return store
    log.warning("jit cache: the session runs without the compiled-code cache %s (%s): %s", root, without, reason)
    _note_refused(root, session_dir, reason, without)
    return None


def _note_refused(root: Path, session_dir: Path, reason: str, without: str) -> None:
    note = {"store": str(root), "published": False, "reason": reason, "without": without}
    try:
        atomic_write_json(session_dir / JIT_STORE_NOTE, note, options=SHARED)
    except OSError as exc:
        log.warning("jit cache: %s was not written (%s)", session_dir / JIT_STORE_NOTE, type(exc).__name__)
