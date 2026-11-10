# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A restart that cannot read the earlier session leaves it and starts a new one; a resume of a credential session
with no ledger of its own is refused. The first is about closing a session that cannot be closed, the second about
resuming the same session, and neither swallows the other: what the second raises is never taken for the first's
"cannot read what the earlier session left"."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hyperloom.inference_optimizer.session.optimization_journal import JournalNotReadable
from hyperloom.inference_optimizer.session.resume_guard import ResumeBlocked
from hyperloom.orchestrator.bus.storage.connection import DatabaseNotPrivate
from hyperloom_multibranch.scheduling.hosted_run import close_before_starting_again


def _left(root):
    return SimpleNamespace(layout=SimpleNamespace(root=root), ended=SimpleNamespace(value="ended"))


@pytest.mark.parametrize(
    "refusal",
    [
        ResumeBlocked("this session holds a credential and has run before ... Set HYPERLOOM_LEDGER_DIR ..."),
        DatabaseNotPrivate("/tmp/hyperloom-ledger-1000 (owner 0:0 0755) is not a directory of this account's"),
        JournalNotReadable("reports/optimization_journal.json is there but cannot be read"),
    ],
    ids=["resume-blocked", "database-not-private", "journal-not-readable"],
)
def test_a_refusal_to_resume_a_credential_session_is_not_taken_for_an_earlier_session_that_cannot_be_read(
    tmp_path, refusal
):
    def close() -> int:
        raise refusal

    with pytest.raises(type(refusal)):
        close_before_starting_again(close, _left(tmp_path / "earlier"))


def test_an_earlier_session_with_a_file_this_account_cannot_read_is_still_left_and_a_new_one_started(tmp_path):
    root = tmp_path / "earlier"

    def close() -> int:
        raise PermissionError(13, "Permission denied", str(root / "state.json"))

    assert close_before_starting_again(close, _left(root)) is None
