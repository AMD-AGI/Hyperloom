# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's rule pack as the sandboxes it starts read it: one staged copy per session, named by its digest.

A local sandbox starts from an emptied environment, so ``HYPERLOOM_RULE_PACK`` does not reach the runner on its own: a
session that names a pack would measure with the reference one, and the digest its manifest records would not describe
what the runner used. The toolbox therefore copies the pack it read to the session directory and names the copy in the
environment of every sandbox template, the way ``framework_base`` snapshots the tuned tables. The copy is the bytes whose
digest the session recorded, so a pack file edited while the session runs changes nothing for the tests that follow.

The reference pack ships with the package, where every sandbox of the same install finds it; a session that reads it
sets nothing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from hyperloom.common.io import WriteOptions, atomic_write_bytes
from hyperloom.common.rule_pack import DEFAULT_PACK_PATH, PACK_ENV, RulePackError, default_pack

__all__ = ["rule_pack_envs"]

_DIGEST_PREFIX = "sha256:"
_SUFFIX = ".yaml"


def rule_pack_envs(directory: Path) -> dict[str, str]:
    """The variable that makes every sandbox of the session read the pack this process read.

    Stages the pack file into ``directory`` and names the copy; empty when the session reads the reference pack.

    Raises:
        RulePackError: When the file cannot be read, or holds other bytes than the process read: the digest the session
            records would then not describe what its sandboxes read.
    """
    pack = default_pack()
    if pack.path == DEFAULT_PACK_PATH:
        return {}
    try:
        raw = pack.path.read_bytes()
    except OSError as exc:
        raise RulePackError(f"rule pack {pack.path}: cannot be read to stage it for the sandboxes: {exc}") from exc
    digest = _DIGEST_PREFIX + hashlib.sha256(raw).hexdigest()
    if digest != pack.digest:
        raise RulePackError(
            f"rule pack {pack.path}: changed since this process read it ({pack.digest}, now {digest}); "
            "restart the session so that its sandboxes and its manifest read the same pack"
        )
    staged = Path(directory) / (digest.removeprefix(_DIGEST_PREFIX) + _SUFFIX)
    if not staged.is_file() or staged.read_bytes() != raw:
        atomic_write_bytes(staged, raw, options=WriteOptions(make_parents=True))
    return {PACK_ENV: str(staged.resolve())}
