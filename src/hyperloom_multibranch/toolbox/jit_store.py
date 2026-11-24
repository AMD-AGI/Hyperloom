# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared compiled-code cache as a toolbox's host publishes to it: which directory, and whether it may.

The runners load what the cache holds as compiled code, and the snapshots prove only their own content, so only the
host's account may write it: :func:`private_jit_store` refuses a directory another account owns or could change, and a
link there. The local toolbox and the hosted one both build their store with :func:`session_jit_store`, which also
prunes it (:mod:`hyperloom.measure.jit.prune`) as the session starts.

A store the check refuses does not stop the session: both toolboxes are built inside it, where a hosted run that raises
is started again into the same refusal. The session runs without the cache instead (nothing published, every test
builds cold, which measures the same), logs one warning naming the store and why, and records it in
:data:`JIT_STORE_NOTE` of the session directory.
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
    "JitStoreNotPrivate",
    "named_jit_store",
    "private_jit_store",
    "session_jit_store",
]

#: The host environment variable that names the shared compiled-code cache when the toolbox's options do not.
JIT_STORE_ENV = "HYPERLOOM_JIT_STORE"
#: The file of the session directory that says the compiled-code cache was refused, and why.
JIT_STORE_NOTE = "jit_store.json"

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


def session_jit_store(root: Path, session_dir: Path) -> DirJitStore | None:
    """The store at ``root`` a session publishes to, checked (:func:`private_jit_store`) and pruned of what no test has
    used for long; a prune that fails is logged and the session goes on.

    ``None`` when the check refuses it: the session runs without the cache, and says so once in the log and in
    :data:`JIT_STORE_NOTE` of ``session_dir`` (see the module). Nothing was pruned then.
    """
    try:
        store = DirJitStore(private_jit_store(root))
    except JitStoreNotPrivate as exc:
        log.warning(
            "jit cache: the session runs without the compiled-code cache %s, nothing is published: %s", root, exc
        )
        _note_refused(root, session_dir, str(exc))
        return None
    prune_logged(store)
    return store


def _note_refused(root: Path, session_dir: Path, reason: str) -> None:
    try:
        atomic_write_json(
            session_dir / JIT_STORE_NOTE, {"store": str(root), "published": False, "reason": reason}, options=SHARED
        )
    except OSError as exc:
        log.warning("jit cache: %s was not written (%s)", session_dir / JIT_STORE_NOTE, type(exc).__name__)
