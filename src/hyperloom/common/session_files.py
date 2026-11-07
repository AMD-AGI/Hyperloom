# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Who may read what a leg writes into a session: the accounts of the run's group, unless it holds a credential.

The next leg of a run may be another account of the group on shared storage, so a session file is written ``664``
(:data:`~hyperloom_common.shared_storage.SHARED_FILE_MODE`). A document that holds a credential (see
:func:`~hyperloom.common.env_safety.holds_credentials`) is written owner-only instead, and another account's resume
fails reading it rather than being handed the credential; the directories it is made in stay ``2775`` either way.
"""

from __future__ import annotations

from hyperloom.common.env_safety import holds_credentials
from hyperloom_common.io import WriteOptions


def shared_options(document: object) -> WriteOptions:
    """The write options of ``document``: parents ``2775``, and the file ``664`` or ``600`` when it holds a credential."""
    return WriteOptions(make_parents=True, shared=not holds_credentials(document), share_parents=True)
