# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A session file is shared by the accounts of the run's group unless the document holds a credential."""

from __future__ import annotations

import pytest

from hyperloom.common.session_files import shared_options
from hyperloom_common.io import WriteOptions


@pytest.mark.parametrize(
    ("document", "shared"),
    [
        ({"tput": 12.5, "server_args": ["--tp", "8"]}, True),
        ({"server_args": ["--tp", "8", "--api-key", "x"]}, False),
        ({"server_args": ["--tp", "8", "--api-key=x"]}, False),
        ({"extra_envs": {"HF_TOKEN": "x"}}, False),
        ({"authored": ["a"], "last_failure_signature": "x"}, True),
        ({"log": "ServerArgs(api_key='sk-abcdefgh')"}, False),
    ],
    ids=["knobs", "flag", "flag-equals", "env-name", "field-names-are-not-credentials", "value"],
)
def test_the_options_are_shared_unless_the_document_holds_a_credential(document, shared):
    assert shared_options(document) == WriteOptions(make_parents=True, shared=shared, share_parents=True)
