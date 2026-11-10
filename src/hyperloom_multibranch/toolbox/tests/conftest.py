# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Every test here runs the same on a GPU host and on a CPU host, busy or idle.

The pool asks the kernel's list who holds a card before it hands it out, and a released lease waits until that list
shows no memory on its cards. A test that read the host's own list would fail wherever other work is on the cards, so
every test here reads an idle machine of its own (a test that lists holders injects its own reader on top). A module
that measures on real cards sets ``USES_REAL_GPUS = True`` and reads the host's list.
"""

from pathlib import Path

import pytest

from hyperloom.common import confinement
from hyperloom.measure.tests.gpu_free_host import (  # noqa: F401  (autouse fixtures)
    USES_REAL_GPUS_FLAG,
    idle_card_list,
    machine_without_readable_gpus,
)
from hyperloom.measure.tests.hosted_fixtures import freeze_session_base, staged_tool_cache
from hyperloom_multibranch.toolbox.local_hosted import JIT_STORE_ENV


@pytest.fixture(autouse=True)
def own_jit_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A compiled-code cache of each test's own: a toolbox keeps one on this host when none is named, and a test must
    neither read what another published there nor leave its own behind. The variable, not ``tempfile``, is what moves:
    an unprivileged native build must still reach the temporary trees it is given."""
    monkeypatch.setenv(JIT_STORE_ENV, str(tmp_path / "jit-store"))


@pytest.fixture(autouse=True)
def measurable_session(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test call carries the session's frozen base and the application's staging, so ``<tmp>/session`` has a
    frozen base and the tool cache holds a staging manifest, as after a fresh launch. A module that measures on real
    cards keeps the host's staging and freezes its sessions' real base
    (:func:`~hyperloom_multibranch.toolbox.tests.gpu_session.freeze_gpu_session_base`)."""
    if getattr(request.module, USES_REAL_GPUS_FLAG, False):
        return
    freeze_session_base(tmp_path / "session")
    staged_tool_cache(tmp_path, monkeypatch)


@pytest.fixture(autouse=True)
def own_private_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """The app's private paths of each test's own: a native build service registers its key and trust file there, and
    the next test's sandboxes must not mask them."""
    monkeypatch.setattr(confinement, "PRIVATE_PATHS", confinement.PrivatePaths())
