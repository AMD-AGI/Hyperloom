# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session that names a rule pack other than the reference one, for the tests of what its sandboxes read."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from hyperloom.common.outside_facts.rule_pack import (
    DEFAULT_PACK_PATH,
    PACK_ENV,
    RulePackIdentity,
    default_pack,
    load_pack,
)

OTHER_VERSION = "2-test"


def other_pack(directory: Path, *, version: str = OTHER_VERSION) -> Path:
    """A pack that holds the reference pack's facts under another version, so that only its identity differs."""
    text, changed = re.subn(
        r"(?m)^version:.*$", f'version: "{version}"', DEFAULT_PACK_PATH.read_text(encoding="utf-8"), 1
    )
    assert changed == 1, "the reference pack states its version on a line of its own"
    path = Path(directory) / "other-pack.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def identity_of(path: Path) -> RulePackIdentity:
    pack = load_pack(path)
    return {"version": pack.version, "digest": pack.digest}


@pytest.fixture
def reading_pack(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Path | None], None]]:
    """``read(path)`` makes this process read the pack at ``path``, as a session that named it does; ``None`` makes it
    read the reference pack. The pack the process had read before is forgotten again at the end of the test."""

    def read(path: Path | None) -> None:
        if path is None:
            monkeypatch.delenv(PACK_ENV, raising=False)
        else:
            monkeypatch.setenv(PACK_ENV, str(path))
        default_pack.cache_clear()

    yield read
    default_pack.cache_clear()
