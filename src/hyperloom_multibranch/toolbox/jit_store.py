# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared compiled-code cache as a toolbox's host publishes to it: which directory, and whether it may.

The runners load what the cache holds as compiled code, and the snapshots prove only their own content, so only the
host's account may write it: :func:`private_jit_store` refuses a directory another account owns or could change, and a
link there. The local toolbox and the hosted one both build their store with :func:`session_jit_store`, which also
prunes it (:mod:`hyperloom.measure.jit.prune`) as the session starts.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Mapping
from pathlib import Path

from hyperloom.measure.jit.prune import prune_logged
from hyperloom.measure.jit.store import DirJitStore
from hyperloom_common.fs_utils import is_private_dir
from hyperloom_multibranch.toolbox.build_preflight import changeable_by_others

__all__ = [
    "JIT_STORE_ENV",
    "JitStoreNotPrivate",
    "named_jit_store",
    "private_jit_store",
    "session_jit_store",
]

#: The host environment variable that names the shared compiled-code cache when the toolbox's options do not.
JIT_STORE_ENV = "HYPERLOOM_JIT_STORE"


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


def session_jit_store(root: Path) -> DirJitStore:
    """The store at ``root`` a session publishes to, checked (:func:`private_jit_store`) and pruned of what no test has
    used for long; a prune that fails is logged and the session goes on.

    Raises:
        JitStoreNotPrivate: As :func:`private_jit_store`; nothing was pruned.
    """
    store = DirJitStore(private_jit_store(root))
    prune_logged(store)
    return store
